"""Third-party library knowledge. One HookSpec row per patched method; HOOKS is the only list:
chromadb and pinecone stores, OpenAI and Anthropic clients, and the LangChain and LlamaIndex
retriever base classes.

Patching is class-level (`Owner.method = wrapper`), so instances created before or after
init() are both captured. capture.wrap_call builds each wrapper: it binds the call with
inspect.signature so codecs and adapters read and override arguments by name, delegates to
capture.intercept_*, and awaits when the target is a coroutine function (the async clients).

Idempotent install: every wrapper carries `__contexteval_original__`. If the attribute on
the owner already has it, the spec reports "already" and nothing is rewrapped. The method is
read from `owner.__dict__`, never getattr, so a subclass is not wrapped with its parent's
method (architecture graft 6).

install() and init() live here, not in _capture, so the import graph is a line:
types <- _text <- _intervention <- _capture <- _hooks <- __init__.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Union

from . import _capture as capture
from ._capture import InstallReport, LLMAdapter, Mode, Restore, RetrievalCodec
from ._intervention import InterventionNotApplicable
from ._store import Store, find_project
from .types import Chunk

Status = Literal["installed", "already", "not_installed", "incompatible", "failed"]

INJECTED_ID = "contexteval:injected"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Field access over both Mapping rows and SDK model objects."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --- retrieval codecs -------------------------------------------------------


@dataclass(frozen=True)
class ChromaCodec(RetrievalCodec[Any]):
    """chromadb Collection.query(query_texts=[...], n_results=k) -> dict of per-query lists.

    v1 intervenes only on single-query calls. Multi-query calls record their first row and
    raise InterventionNotApplicable if an intervention is active. Embedding-only queries
    record query=None.
    """

    source: str = "chromadb"
    top_k_param: str | None = "n_results"

    def query_text(self, args: inspect.BoundArguments) -> str | None:
        texts = args.arguments.get("query_texts")
        if isinstance(texts, str):
            return texts
        if isinstance(texts, (list, tuple)) and len(texts) == 1 and isinstance(texts[0], str):
            return texts[0]
        return None

    def decode(self, result: Any) -> tuple[Chunk, ...]:
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [None])[0] or [None] * len(ids)
        distances = (result.get("distances") or [None])[0] or [None] * len(ids)
        return tuple(
            Chunk(
                text=doc if doc is not None else "",
                id=str(cid),
                score=-float(dist) if dist is not None else None,
                source_index=i,
            )
            for i, (cid, doc, dist) in enumerate(zip(ids, documents, distances))
        )

    @staticmethod
    def _injected(key: str, chunk: Chunk) -> Any:
        if key == "ids":
            return chunk.id or INJECTED_ID
        if key == "documents":
            return chunk.text
        if key == "distances":
            return 0.0  # an injected chunk is the best possible match
        return None  # metadatas, embeddings, uris, data

    def encode(self, result: Any, chunks: tuple[Chunk, ...]) -> Any:
        if len(result.get("ids") or []) != 1:
            raise InterventionNotApplicable("chromadb: multi-query calls cannot be intervened on in v1")
        out = dict(result)
        for key, value in result.items():
            # "included" is a flat list of field names, not a per-query list.
            if key == "included" or not isinstance(value, list) or len(value) != 1 or value[0] is None:
                continue
            row = value[0]
            new_row = []
            for c in chunks:
                if c.source_index is None:
                    new_row.append(self._injected(key, c))
                elif key == "documents":
                    new_row.append(c.text)
                else:
                    new_row.append(row[c.source_index])
            out[key] = [new_row]
        return out


class _SyntheticMatch:
    """Stand-in pinecone match for an injected chunk when the native class cannot be built."""

    def __init__(self, id: str, score: float, metadata: dict[str, Any]) -> None:
        self.id = id
        self.score = score
        self.values: list[float] = []
        self.metadata = metadata

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class PineconeCodec(RetrievalCodec[Any]):
    """pinecone Index.query(vector=..., top_k=k, include_metadata=True) -> matches.

    Queries are vectors, so query_text is None and transforms fall back to the run question.
    Chunk text comes from match.metadata[text_key], where the key is process state read at
    call time (review X7): the CLI installs this codec before the app module can name the key.
    """

    source: str = "pinecone"
    top_k_param: str | None = "top_k"

    @property
    def text_key(self) -> str:
        return capture.pinecone_text_key()

    def query_text(self, args: inspect.BoundArguments) -> str | None:
        return None

    def decode(self, result: Any) -> tuple[Chunk, ...]:
        chunks = []
        for i, m in enumerate(_get(result, "matches") or []):
            metadata = _get(m, "metadata") or {}
            score = _get(m, "score")
            chunks.append(
                Chunk(
                    text=str(metadata.get(self.text_key, "")),
                    id=str(_get(m, "id")),
                    score=float(score) if score is not None else None,
                    source_index=i,
                )
            )
        return tuple(chunks)

    def _match(self, template: Any, chunk: Chunk) -> Any:
        fields = {"id": chunk.id or INJECTED_ID, "score": 1.0, "values": [], "metadata": {self.text_key: chunk.text}}
        if isinstance(template, Mapping):
            return fields
        if template is not None:
            try:
                return type(template)(**fields)
            except TypeError:
                pass
        return _SyntheticMatch(fields["id"], fields["score"], fields["metadata"])

    def _with_text(self, match: Any, chunk_text: str) -> Any:
        metadata = {**(_get(match, "metadata") or {}), self.text_key: chunk_text}
        if isinstance(match, Mapping):
            return {**match, "metadata": metadata}
        clone = copy.copy(match)
        clone.metadata = metadata
        return clone

    def encode(self, result: Any, chunks: tuple[Chunk, ...]) -> Any:
        matches = list(_get(result, "matches") or [])
        template = matches[0] if matches else None
        new_matches = []
        for c in chunks:
            if c.source_index is None:
                new_matches.append(self._match(template, c))
                continue
            m = matches[c.source_index]
            text_now = str((_get(m, "metadata") or {}).get(self.text_key, ""))
            new_matches.append(m if text_now == c.text else self._with_text(m, c.text))
        if isinstance(result, Mapping):
            return {**result, "matches": new_matches}
        out = copy.copy(result)
        out.matches = new_matches
        return out


# --- framework retrievers (LangChain, LlamaIndex) ----------------------------
#
# A framework retriever's top_k is state on the retriever object, so these codecs override
# RetrievalCodec.override_top_k: the knob is set for one call and put back when the call
# returns. Their LLM wrappers (ChatOpenAI, llama_index OpenAI, ...) call the official SDKs,
# which the adapters below already hook, so no LLM rows are needed for them.

_knob_lock = threading.Lock()
# (id(instance), attribute) -> (the user's value, calls currently overriding it)
_knobs: dict[tuple[int, str], tuple[Any, int]] = {}


def _override_attr(instance: Any, attr: str, value: Any) -> Restore:
    """Set `instance.attr` for one call and return what puts the user's value back.

    Calls that overlap on one instance (an app that gathers two retrievals for one question)
    all want the same value, because one run has one intervention: the first one in saves the
    user's value and the last one out restores it, so no call restores another call's override.
    Two runs wanting different values on one shared instance are not supported; the runner
    executes one question at a time (plan item 7)."""
    key = (id(instance), attr)
    with _knob_lock:
        saved, depth = _knobs.get(key, (getattr(instance, attr), 0))
        _knobs[key] = (saved, depth + 1)
        setattr(instance, attr, value)

    def restore() -> None:
        with _knob_lock:
            saved, depth = _knobs[key]
            if depth > 1:
                _knobs[key] = (saved, depth - 1)
            else:
                del _knobs[key]
                setattr(instance, attr, saved)

    return restore


@dataclass(frozen=True)
class LangChainRetrieverCodec(RetrievalCodec[Any]):
    """langchain_core BaseRetriever.invoke(input, config=None, **kwargs) -> list[Document].

    Documents carry page_content, metadata, and an optional id; no score. The top_k knob is
    the retriever's own: `search_kwargs["k"]` on a VectorStoreRetriever (the dict is replaced
    for the call, never edited in place) or an int `k` field. VectorStoreRetriever merges the
    call's kwargs over `search_kwargs`, so a per-call `k` is overridden too. A retriever with
    no knob leaves top_k to the hooked call underneath it (a base retriever, a chromadb
    collection) and refuses only when none takes it. langchain-chroma reaches the hooked
    chromadb `Collection.query` underneath; that inner call records nothing (capture's
    outermost rule), so one invoke is one event.
    """

    source: str = "langchain"
    top_k_param: str | None = "k"

    def query_text(self, args: inspect.BoundArguments) -> str | None:
        query = args.arguments.get("input")
        return query if isinstance(query, str) else None

    def override_top_k(self, args: inspect.BoundArguments, top_k: int) -> Restore | None:
        instance = args.arguments.get("self")
        kwargs = args.arguments.get("kwargs")
        if isinstance(kwargs, dict) and "k" in kwargs:
            kwargs["k"] = top_k
        search_kwargs = getattr(instance, "search_kwargs", None)
        if isinstance(search_kwargs, Mapping):
            return _override_attr(instance, "search_kwargs", {**search_kwargs, "k": top_k})
        if isinstance(getattr(instance, "k", None), int):
            return _override_attr(instance, "k", top_k)
        raise InterventionNotApplicable(f"langchain: top_k={top_k} requested but {type(instance).__name__} has no search_kwargs or k")

    def decode(self, result: Any) -> tuple[Chunk, ...]:
        return tuple(
            Chunk(text=str(doc.page_content), id=str(doc.id) if doc.id is not None else None, score=None, source_index=i)
            for i, doc in enumerate(result)
        )

    def encode(self, result: Any, chunks: tuple[Chunk, ...]) -> Any:
        from langchain_core.documents import Document  # the hook fired, so the framework is importable

        out = []
        for c in chunks:
            if c.source_index is None:
                out.append(Document(page_content=c.text, metadata={}, id=c.id or INJECTED_ID))
                continue
            doc = result[c.source_index]
            out.append(doc if doc.page_content == c.text else doc.model_copy(update={"page_content": c.text}))
        return out


@dataclass(frozen=True)
class LlamaIndexRetrieverCodec(RetrievalCodec[Any]):
    """llama_index.core BaseRetriever.retrieve(str_or_query_bundle) -> list[NodeWithScore].

    The query is a string or a QueryBundle (query_str). Chunk text is the node's content without
    metadata, so it matches what the node holds, not what a prompt template adds. The top_k
    knob is the retriever's `similarity_top_k` (a property on VectorIndexRetriever); a retriever
    without one leaves top_k to a hooked call underneath. An injected chunk is a NodeWithScore
    over a TextNode at score 1.0, the best similarity, so a sort by score keeps it first.
    """

    source: str = "llamaindex"
    top_k_param: str | None = "similarity_top_k"

    def query_text(self, args: inspect.BoundArguments) -> str | None:
        query = args.arguments.get("str_or_query_bundle")
        if isinstance(query, str):
            return query
        query_str = getattr(query, "query_str", None)
        return query_str if isinstance(query_str, str) else None

    def override_top_k(self, args: inspect.BoundArguments, top_k: int) -> Restore | None:
        instance = args.arguments.get("self")
        if not isinstance(getattr(instance, "similarity_top_k", None), int):
            raise InterventionNotApplicable(f"llamaindex: top_k={top_k} requested but {type(instance).__name__} has no similarity_top_k")
        return _override_attr(instance, "similarity_top_k", top_k)

    def decode(self, result: Any) -> tuple[Chunk, ...]:
        return tuple(
            Chunk(text=str(scored.node.get_content()), id=str(scored.node.node_id), score=float(scored.score) if scored.score is not None else None, source_index=i)
            for i, scored in enumerate(result)
        )

    def encode(self, result: Any, chunks: tuple[Chunk, ...]) -> Any:
        from llama_index.core.schema import NodeWithScore, TextNode

        out = []
        for c in chunks:
            if c.source_index is None:
                out.append(NodeWithScore(node=TextNode(text=c.text, id_=c.id or INJECTED_ID), score=1.0))
                continue
            scored = result[c.source_index]
            if scored.node.get_content() == c.text:
                out.append(scored)
                continue
            node = scored.node
            rewritten = node.model_copy(update={"text": c.text}) if hasattr(node, "text") else TextNode(text=c.text, id_=node.node_id, metadata=dict(node.metadata))
            out.append(NodeWithScore(node=rewritten, score=scored.score))
        return out


# --- LLM adapters -----------------------------------------------------------


def _content_text(content: Any) -> str:
    """Message content as text: a string, or a list of parts whose text parts are joined."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            part_text = _get(part, "text")
            if isinstance(part_text, str):
                parts.append(part_text)
        return "\n".join(parts)
    return str(content)


def _messages_text(messages: Any) -> str:
    lines = []
    for m in messages or ():
        role = _get(m, "role", "message")
        lines.append(f"{role}: {_content_text(_get(m, 'content'))}")
    return "\n".join(lines)


def _usage(response: Any, input_key: str, output_key: str) -> tuple[int | None, int | None]:
    usage = _get(response, "usage")
    if usage is None:
        return None, None
    return _get(usage, input_key), _get(usage, output_key)


@dataclass(frozen=True)
class OpenAIChatAdapter:
    provider: Literal["openai"] = "openai"

    def prompt_text(self, args: inspect.BoundArguments) -> str:
        return _messages_text(args.arguments.get("messages"))

    def model(self, args: inspect.BoundArguments) -> str:
        return str(args.arguments.get("model", ""))

    def parse(self, response: Any) -> tuple[str | None, int | None, int | None]:
        choices = _get(response, "choices")
        if not choices:
            return None, None, None  # a Stream has no choices; v1 does not read streams
        answer = _get(_get(choices[0], "message"), "content")
        input_tokens, output_tokens = _usage(response, "prompt_tokens", "completion_tokens")
        return answer, input_tokens, output_tokens


@dataclass(frozen=True)
class OpenAIResponsesAdapter:
    provider: Literal["openai"] = "openai"

    def prompt_text(self, args: inspect.BoundArguments) -> str:
        lines = []
        instructions = args.arguments.get("instructions")
        if isinstance(instructions, str) and instructions:
            lines.append(f"instructions: {instructions}")
        user_input = args.arguments.get("input")
        if isinstance(user_input, str):
            lines.append(f"user: {user_input}")
        elif isinstance(user_input, (list, tuple)):
            lines.append(_messages_text(user_input))
        return "\n".join(lines)

    def model(self, args: inspect.BoundArguments) -> str:
        return str(args.arguments.get("model", ""))

    def parse(self, response: Any) -> tuple[str | None, int | None, int | None]:
        answer = _get(response, "output_text")
        if not isinstance(answer, str):
            return None, None, None
        input_tokens, output_tokens = _usage(response, "input_tokens", "output_tokens")
        return answer, input_tokens, output_tokens


@dataclass(frozen=True)
class AnthropicMessagesAdapter:
    provider: Literal["anthropic"] = "anthropic"

    def prompt_text(self, args: inspect.BoundArguments) -> str:
        lines = []
        system = args.arguments.get("system")
        if system is not None and not _is_not_given(system):
            lines.append(f"system: {_content_text(system)}")
        lines.append(_messages_text(args.arguments.get("messages")))
        return "\n".join(lines)

    def model(self, args: inspect.BoundArguments) -> str:
        return str(args.arguments.get("model", ""))

    def parse(self, response: Any) -> tuple[str | None, int | None, int | None]:
        content = _get(response, "content")
        if not isinstance(content, (list, tuple)):
            return None, None, None  # a stream manager has no content list
        answer = "".join(_get(block, "text", "") for block in content if _get(block, "type") == "text")
        input_tokens, output_tokens = _usage(response, "input_tokens", "output_tokens")
        return answer, input_tokens, output_tokens


def _is_not_given(value: Any) -> bool:
    """The SDKs' NOT_GIVEN sentinel is passed explicitly when a caller forwards every kwarg."""
    return type(value).__name__ in ("NotGiven", "Omit")


# --- the registry -----------------------------------------------------------


Handler = Union[RetrievalCodec[Any], LLMAdapter]


@dataclass(frozen=True)
class HookSpec:
    library: str  # import name, the InstallReport key
    targets: tuple[str, ...]  # "module:Owner.method" candidates, first importable wins (version drift)
    handler: Handler


HOOKS: tuple[HookSpec, ...] = (
    HookSpec("chromadb", ("chromadb.api.models.Collection:Collection.query",), ChromaCodec()),
    HookSpec("chromadb", ("chromadb.api.models.AsyncCollection:AsyncCollection.query",), ChromaCodec()),
    HookSpec("pinecone", ("pinecone.db_data.index:Index.query", "pinecone.data.index:Index.query"), PineconeCodec()),
    HookSpec("openai", ("openai.resources.chat.completions:Completions.create",), OpenAIChatAdapter()),
    HookSpec("openai", ("openai.resources.chat.completions:AsyncCompletions.create",), OpenAIChatAdapter()),
    HookSpec("openai", ("openai.resources.responses:Responses.create",), OpenAIResponsesAdapter()),
    HookSpec("openai", ("openai.resources.responses:AsyncResponses.create",), OpenAIResponsesAdapter()),
    HookSpec("anthropic", ("anthropic.resources.messages:Messages.create",), AnthropicMessagesAdapter()),
    HookSpec("anthropic", ("anthropic.resources.messages:AsyncMessages.create",), AnthropicMessagesAdapter()),
    HookSpec("langchain_core", ("langchain_core.retrievers:BaseRetriever.invoke",), LangChainRetrieverCodec()),
    HookSpec("langchain_core", ("langchain_core.retrievers:BaseRetriever.ainvoke",), LangChainRetrieverCodec()),
    HookSpec("llama_index", ("llama_index.core.base.base_retriever:BaseRetriever.retrieve",), LlamaIndexRetrieverCodec()),
    HookSpec("llama_index", ("llama_index.core.base.base_retriever:BaseRetriever.aretrieve",), LlamaIndexRetrieverCodec()),
)

# Worse statuses win when a library has several specs.
_SEVERITY: dict[Status, int] = {"already": 0, "installed": 1, "not_installed": 2, "incompatible": 3, "failed": 4}

# (owner, method name, original) for every patched method, in install order.
_patched: list[tuple[type, str, Callable[..., Any]]] = []


def _resolve(target: str) -> tuple[type, str] | None:
    module_name, _, attr = target.partition(":")
    owner_name, _, method = attr.partition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    owner = getattr(module, owner_name, None)
    return (owner, method) if isinstance(owner, type) else None


def _install_spec(spec: HookSpec) -> Status:
    """Never raises: a library whose import has a failing side effect, or whose target cannot be
    wrapped, is reported as failed and left alone, and the other libraries still get hooked."""
    try:
        for target in spec.targets:
            resolved = _resolve(target)
            if resolved is not None:
                return patch_method(resolved[0], resolved[1], spec.handler)
        return "incompatible" if importlib.util.find_spec(spec.library) is not None else "not_installed"
    except Exception as e:
        capture._log_once(f"install:{spec.library}", f"contexteval could not hook {spec.library}; it runs unhooked", e)
        return "failed"


def install_all(specs: tuple[HookSpec, ...] = HOOKS) -> InstallReport:
    """Patch every spec whose library imports. A library's status is its worst spec status."""
    statuses: dict[str, Status] = {}
    for spec in specs:
        status = _install_spec(spec)
        previous = statuses.get(spec.library)
        if previous is None or _SEVERITY[status] > _SEVERITY[previous]:
            statuses[spec.library] = status
    return InstallReport(statuses)


def patch_method(owner: type, name: str, handler: Handler) -> Status:
    """Replace owner.name with a wrapper routing to capture.intercept_retrieval or intercept_llm.

    Also the test seam: pinecone is tested by patching a stand-in class with PineconeCodec,
    so no pinecone package or key is needed.
    """
    current = owner.__dict__.get(name)
    if current is None or not callable(current):
        return "incompatible"
    if hasattr(current, "__contexteval_original__"):
        return "already"
    try:
        wrapper = capture.wrap_call(current, handler)
    except Exception as e:
        capture._log_once(f"install:{owner.__name__}.{name}", f"contexteval could not wrap {owner.__name__}.{name}; it runs unhooked", e)
        return "failed"
    setattr(owner, name, wrapper)
    _patched.append((owner, name, current))
    return "installed"


def uninstall_all() -> None:
    """Restore every `__contexteval_original__`. Used by test fixtures only."""
    while _patched:
        owner, name, original = _patched.pop()
        setattr(owner, name, original)


# --- entry points -----------------------------------------------------------


def install(mode: Literal["cli", "production"], project_dir: Path | None = None, pinecone_text_key: str | None = None) -> InstallReport:
    """Idempotent. Patches every library in HOOKS that is importable, then sets the process mode
    (which only escalates from OFF, see capture.set_mode). A pinecone text key given here
    applies to calls from now on, whether or not the hooks were already in place."""
    report = install_all(HOOKS)
    capture.set_pinecone_text_key(pinecone_text_key)
    if mode == "production":
        root = project_dir or Path.cwd()
        capture.set_mode(Mode.PRODUCTION, lambda: capture.ProductionWriter(root))
    else:
        capture.set_mode(Mode.CLI)
    return report


def init(project_dir: str | Path | None = None, *, pinecone_text_key: str | None = None) -> InstallReport:
    """Public one-liner. Finds or creates .contexteval/ (walking up from cwd) and installs hooks
    in production mode. `pinecone_text_key` names the metadata field holding chunk text
    (architecture graft 7); the default "text" stands until some caller names another.

    Never raises into the app. A project that cannot hold the database (unwritable path, a
    schema from another build) is logged once and reported in InstallReport.error; nothing is
    hooked and the app runs as if init() had not been called. Under `contexteval run` the CLI
    already installed, so the database is not touched here.
    """
    try:
        root = Path(project_dir) if project_dir is not None else (find_project(Path.cwd()) or Path.cwd())
        if capture.mode() is Mode.OFF:
            Store.open(root).close()
        return install("production", root, pinecone_text_key)
    except Exception as e:
        capture._log_once("init", "contexteval.init() failed; production capture is off", e)
        return InstallReport({}, error=f"{type(e).__name__}: {e}")
