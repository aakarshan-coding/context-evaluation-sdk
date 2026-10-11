"""Capture core: the active-run context variable, the two intercept functions every hook
calls, the @retriever decorator, and init().

Library knowledge (which class, which kwarg, which response shape) lives in _hooks.py.
This module knows nothing about chromadb, pinecone, openai, or anthropic.

Who sees what:
- Inside runner.execute: an ActiveRun is set. Hooks read its intervention and append events.
- No active run, mode == "production" (user called init() in their app): hooks pass the call
  through unchanged and buffer events per context; each LLM call flushes the buffer to the
  store as one ungraded Trace.
- No active run, mode == "cli" or "off": hooks pass through and record nothing.
- Inside suppressed() (the SDK's own provider calls): hooks pass through and record nothing.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, Literal, Protocol, TypeVar, overload

from ._intervention import NO_INTERVENTION, Intervention
from .types import Chunk, LLMCall, RetrievalEvent

R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])


# --- per-run state ----------------------------------------------------------


@dataclass
class ActiveRun:
    """Mutable, owned by exactly one runner.execute call. Never shared across runs."""

    question: str
    intervention: Intervention
    retrievals: list[RetrievalEvent] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)


_active: ContextVar[ActiveRun | None] = ContextVar("contexteval_active", default=None)
_ambient: ContextVar[ActiveRun | None] = ContextVar("contexteval_ambient", default=None)
_suppressed: ContextVar[bool] = ContextVar("contexteval_suppressed", default=False)


@contextmanager
def active_run(question: str, intervention: Intervention) -> Iterator[ActiveRun]:
    """Set the run for the duration of one answer() call; reset on exit even on error."""
    raise NotImplementedError  # token = _active.set(ActiveRun(...)); yield; _active.reset(token)


@contextmanager
def suppressed() -> Iterator[None]:
    """Wrap the SDK's own LLM calls (test generation, grading, judge) so hooks skip them."""
    raise NotImplementedError


# --- process state (single owner: this module) ------------------------------


class Mode(Enum):
    OFF = "off"  # hooks not installed
    CLI = "cli"  # installed by the CLI; record only inside active runs
    PRODUCTION = "production"  # installed by init(); also record outside runs


@dataclass
class _Process:
    mode: Mode = Mode.OFF
    sink: Callable[[ActiveRun], None] | None = None  # writes an ambient buffer as a Trace


_process = _Process()


@dataclass(frozen=True)
class InstallReport:
    """library name -> status. Printed by `contexteval run` and returned by init()."""

    statuses: dict[str, Literal["installed", "already", "not_installed", "incompatible"]]


def install(mode: Literal["cli", "production"], project_dir: Path | None = None) -> InstallReport:
    """Idempotent. Patches every library in _hooks.HOOKS that is importable.

    Mode only ever escalates OFF -> CLI or OFF -> PRODUCTION. If the CLI installed first, a
    later init() inside the user's module keeps CLI mode, so `contexteval run` never writes
    production traces.
    """
    raise NotImplementedError  # _hooks.install_all()  (lazy import: _hooks imports this module); set _process once


def init(project_dir: str | Path | None = None) -> InstallReport:
    """Public one-liner. Finds or creates .contexteval/ (walking up from cwd) and installs hooks
    in production mode.
    """
    raise NotImplementedError


# --- the two intercepts every hook calls ------------------------------------


class RetrievalCodec(Protocol[R]):
    """Translates one retrieval API's native call and result to and from Chunks."""

    source: str
    top_k_param: str | None  # name of the argument to override, None if not overridable

    def query_text(self, args: inspect.BoundArguments) -> str | None: ...

    def decode(self, result: R) -> tuple[Chunk, ...]: ...

    def encode(self, result: R, chunks: tuple[Chunk, ...]) -> R:
        """Rebuild a native result: reuse rows by source_index, synthesize rows for injected chunks."""
        ...


class LLMAdapter(Protocol):
    provider: Literal["openai", "anthropic"]

    def prompt_text(self, args: inspect.BoundArguments) -> str: ...

    def model(self, args: inspect.BoundArguments) -> str: ...

    def parse(self, response: Any) -> tuple[str | None, int | None, int | None]:
        """(answer text, input tokens, output tokens). Answer None for streams."""
        ...


def intercept_retrieval(codec: RetrievalCodec[R], call: Callable[[inspect.BoundArguments], R], args: inspect.BoundArguments) -> R:
    """Shared body of every retrieval hook and the @retriever decorator.

    Recording or intervention errors never break the user's call: on an internal error,
    log once and fall back to `call(args)` unchanged. Errors raised by `call` propagate.
    """
    # TODO:
    #   if _suppressed.get(): return call(args)
    #   run = _active.get() or (_ambient_buffer() if _process.mode is PRODUCTION else None)
    #   if run is None: return call(args)
    #   iv = run.intervention
    #   if iv.top_k is not None and codec.top_k_param: args.arguments[codec.top_k_param] = iv.top_k
    #   t0; raw = call(args); latency
    #   retrieved = codec.decode(raw)
    #   delivered = intervention.apply(iv, retrieved, codec.query_text(args) or run.question)
    #   run.retrievals.append(RetrievalEvent(codec.source, query, snapshot(args), retrieved, delivered, latency))
    #   return raw if iv.is_noop else codec.encode(raw, delivered)
    raise NotImplementedError


def intercept_llm(adapter: LLMAdapter, call: Callable[[inspect.BoundArguments], Any], args: inspect.BoundArguments) -> Any:
    """Record only. v1 never changes the prompt (non-retrieval prompt text is read-only)."""
    # TODO: in production mode with no active run, append to the ambient buffer, then
    # flush it through _process.sink and clear it: one LLM call closes one production trace.
    raise NotImplementedError


# --- decorator for custom retrieval (pgvector, SQL, internal APIs) ----------


@dataclass(frozen=True)
class _DecoratorCodec(Generic[R]):
    """Accepts results that are a list of str, of Mappings, or of objects with a text attribute."""

    source: str
    top_k_param: str | None
    query_param: str | None  # None means the first positional argument
    text_field: str  # key or attribute holding chunk text when items are not str
    id_field: str | None
    score_field: str | None

    def query_text(self, args: inspect.BoundArguments) -> str | None:
        raise NotImplementedError

    def decode(self, result: R) -> tuple[Chunk, ...]:
        raise NotImplementedError

    def encode(self, result: R, chunks: tuple[Chunk, ...]) -> R:
        # Injected chunk: str items -> the text; Mapping items -> {text_field: text};
        # objects -> copy.copy(result[0]) with text_field set. Returns the same list type.
        raise NotImplementedError


@overload
def retriever(fn: F) -> F: ...
@overload
def retriever(
    *,
    top_k: str | None = "top_k",
    query: str | None = None,
    text: str = "text",
    id: str | None = "id",
    score: str | None = "score",
) -> Callable[[F], F]: ...
def retriever(fn: Any = None, *, top_k: str | None = "top_k", query: str | None = None, text: str = "text", id: str | None = "id", score: str | None = "score") -> Any:
    """Mark a user search function as a retrieval source.

        @contexteval.retriever                 # top_k override uses a parameter named `top_k`
        @contexteval.retriever(top_k="k")      # the user's parameter is called `k`
        @contexteval.retriever(top_k=None)     # not overridable; top_k experiments warn

    Validated at decoration time (the boundary): an explicitly named top_k or query that is
    not in the signature raises TypeError at import. The default "top_k" silently becomes
    None when absent.
    """
    raise NotImplementedError  # build _DecoratorCodec; wrap with functools.wraps -> intercept_retrieval
