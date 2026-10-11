"""Third-party library knowledge. One HookSpec row per patched method; HOOKS is the only list.

Patching is class-level (`Owner.method = wrapper`), so instances created before or after
init() are both captured. Each wrapper binds the call with inspect.signature so codecs and
adapters read and override arguments by name, then delegates to capture.intercept_*.

Idempotent install: every wrapper carries `__contexteval_original__`. If the attribute on
the owner already has it, the spec reports "already" and nothing is rewrapped. The method is
read from `owner.__dict__`, never getattr, so a subclass is not wrapped with its parent's
method (architecture graft 6).

install() and init() live here, not in _capture, so the import graph is a line:
types <- _text <- _intervention <- _capture <- _hooks <- __init__.
"""

from __future__ import annotations

import copy
import functools
import importlib
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Union

from . import _capture as capture
from ._capture import InstallReport, LLMAdapter, Mode, RetrievalCodec
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
class ChromaCodec:
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
class PineconeCodec:
    """pinecone Index.query(vector=..., top_k=k, include_metadata=True) -> matches.

    Queries are vectors, so query_text is None and transforms fall back to the run question.
    Chunk text comes from match.metadata[text_key].
    """

    text_key: str = "text"
    source: str = "pinecone"
    top_k_param: str | None = "top_k"

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
    HookSpec("pinecone", ("pinecone.db_data.index:Index.query", "pinecone.data.index:Index.query"), PineconeCodec()),
    HookSpec("openai", ("openai.resources.chat.completions:Completions.create",), OpenAIChatAdapter()),
    HookSpec("openai", ("openai.resources.responses:Responses.create",), OpenAIResponsesAdapter()),
    HookSpec("anthropic", ("anthropic.resources.messages:Messages.create",), AnthropicMessagesAdapter()),
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
        wrapper = _wrap(current, handler)
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


def _wrap(original: Callable[..., Any], handler: Handler) -> Callable[..., Any]:
    sig = inspect.signature(original)
    intercept: Callable[..., Any] = capture.intercept_retrieval if isinstance(handler, RetrievalCodec) else capture.intercept_llm

    @functools.wraps(original)
    def wrapper(*a: Any, **kw: Any) -> Any:
        try:
            bound = sig.bind(*a, **kw)
        except TypeError:
            return original(*a, **kw)  # let the library raise its own argument error
        return intercept(handler, lambda b: original(*b.args, **b.kwargs), bound)

    wrapper.__contexteval_original__ = original  # type: ignore[attr-defined]
    return wrapper


# --- entry points -----------------------------------------------------------


def install(mode: Literal["cli", "production"], project_dir: Path | None = None, pinecone_text_key: str = "text") -> InstallReport:
    """Idempotent. Patches every library in HOOKS that is importable, then sets the process mode
    (which only escalates from OFF, see capture.set_mode)."""
    specs = tuple(
        HookSpec(s.library, s.targets, PineconeCodec(text_key=pinecone_text_key)) if s.library == "pinecone" else s
        for s in HOOKS
    )
    report = install_all(specs)
    if mode == "production":
        root = project_dir or Path.cwd()
        capture.set_mode(Mode.PRODUCTION, lambda: capture.ProductionWriter(root))
    else:
        capture.set_mode(Mode.CLI)
    return report


def init(project_dir: str | Path | None = None, *, pinecone_text_key: str = "text") -> InstallReport:
    """Public one-liner. Finds or creates .contexteval/ (walking up from cwd) and installs hooks
    in production mode. `pinecone_text_key` names the metadata field holding chunk text
    (architecture graft 7).

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
