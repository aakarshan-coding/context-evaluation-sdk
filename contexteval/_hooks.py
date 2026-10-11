"""Third-party library knowledge. One HookSpec row per patched method; HOOKS is the only list.

Patching is class-level (`Owner.method = wrapper`), so instances created before or after
init() are both captured. Each wrapper binds the call with inspect.signature so codecs and
adapters read and override arguments by name, then delegates to capture.intercept_*.

Idempotent install: every wrapper carries `__contexteval_original__`. If the attribute on
the owner already has it, the spec reports "already" and nothing is rewrapped.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Literal, Union

from ._capture import InstallReport, LLMAdapter, RetrievalCodec
from .types import Chunk

Status = Literal["installed", "already", "not_installed", "incompatible"]


# --- retrieval codecs -------------------------------------------------------


@dataclass(frozen=True)
class ChromaCodec:
    """chromadb Collection.query(query_texts=[...], n_results=k) -> dict of per-query lists.

    v1 intervenes only on single-query calls. Multi-query calls are recorded per row and
    passed through unchanged. Embedding-only queries record query=None.
    """

    source: str = "chromadb"
    top_k_param: str | None = "n_results"

    def query_text(self, args: inspect.BoundArguments) -> str | None:
        raise NotImplementedError

    def decode(self, result: Any) -> tuple[Chunk, ...]:
        raise NotImplementedError  # documents[0], ids[0], distances[0] -> Chunk(score=-distance)

    def encode(self, result: Any, chunks: tuple[Chunk, ...]) -> Any:
        raise NotImplementedError  # rebuild every per-query list (ids, documents, metadatas, distances)


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
        raise NotImplementedError

    def encode(self, result: Any, chunks: tuple[Chunk, ...]) -> Any:
        raise NotImplementedError


# --- LLM adapters -----------------------------------------------------------


@dataclass(frozen=True)
class OpenAIChatAdapter:
    provider: Literal["openai"] = "openai"

    def prompt_text(self, args: inspect.BoundArguments) -> str:
        raise NotImplementedError  # "role: content" per message; content parts joined by text

    def model(self, args: inspect.BoundArguments) -> str:
        raise NotImplementedError

    def parse(self, response: Any) -> tuple[str | None, int | None, int | None]:
        raise NotImplementedError  # choices[0].message.content, usage.prompt_tokens, usage.completion_tokens


@dataclass(frozen=True)
class OpenAIResponsesAdapter:
    provider: Literal["openai"] = "openai"

    def prompt_text(self, args: inspect.BoundArguments) -> str:
        raise NotImplementedError  # instructions + input (str or item list)

    def model(self, args: inspect.BoundArguments) -> str:
        raise NotImplementedError

    def parse(self, response: Any) -> tuple[str | None, int | None, int | None]:
        raise NotImplementedError  # response.output_text, usage.input_tokens, usage.output_tokens


@dataclass(frozen=True)
class AnthropicMessagesAdapter:
    provider: Literal["anthropic"] = "anthropic"

    def prompt_text(self, args: inspect.BoundArguments) -> str:
        raise NotImplementedError  # system (str or blocks) + messages

    def model(self, args: inspect.BoundArguments) -> str:
        raise NotImplementedError

    def parse(self, response: Any) -> tuple[str | None, int | None, int | None]:
        raise NotImplementedError  # text blocks joined, usage.input_tokens, usage.output_tokens


# --- the registry -----------------------------------------------------------


@dataclass(frozen=True)
class HookSpec:
    library: str  # import name, the InstallReport key
    targets: tuple[str, ...]  # "module:Owner.method" candidates, first importable wins (version drift)
    handler: Union[RetrievalCodec[Any], LLMAdapter]


HOOKS: tuple[HookSpec, ...] = (
    HookSpec("chromadb", ("chromadb.api.models.Collection:Collection.query",), ChromaCodec()),
    HookSpec("pinecone", ("pinecone.db_data.index:Index.query", "pinecone.data.index:Index.query"), PineconeCodec()),
    HookSpec("openai", ("openai.resources.chat.completions:Completions.create",), OpenAIChatAdapter()),
    HookSpec("openai", ("openai.resources.responses:Responses.create",), OpenAIResponsesAdapter()),
    HookSpec("anthropic", ("anthropic.resources.messages:Messages.create",), AnthropicMessagesAdapter()),
)


def install_all(specs: tuple[HookSpec, ...] = HOOKS) -> InstallReport:
    """Patch every spec whose library imports. A library's status is its worst spec status."""
    raise NotImplementedError


def patch_method(owner: type, name: str, handler: Union[RetrievalCodec[Any], LLMAdapter]) -> Status:
    """Replace owner.name with a wrapper routing to capture.intercept_retrieval or intercept_llm.

    Also the test seam: pinecone is tested by patching a stand-in class with PineconeCodec,
    so no pinecone package or key is needed.
    """
    raise NotImplementedError


def uninstall_all() -> None:
    """Restore every `__contexteval_original__`. Used by test fixtures only."""
    raise NotImplementedError


def _wrap(original: Callable[..., Any], handler: Union[RetrievalCodec[Any], LLMAdapter]) -> Callable[..., Any]:
    raise NotImplementedError  # bind args with signature(original); dispatch on handler kind
