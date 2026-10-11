"""Capture core: the active-run context variable, the two intercept functions every hook
calls, the @retriever decorator, trace finalization, and the process mode.

Library knowledge (which class, which kwarg, which response shape) lives in _hooks.py, as do
install() and init(), so this module imports nothing that imports it back.
This module knows nothing about chromadb, pinecone, openai, or anthropic.

Who sees what:
- Inside runner.execute: an ActiveRun is set. Hooks read its intervention and append events.
- No active run, mode == PRODUCTION (user called init() in their app): hooks pass the call
  through unchanged and buffer events per context; each LLM call flushes the buffer to the
  store as one ungraded Trace.
- No active run, mode == CLI or OFF: hooks pass through and record nothing.
- Inside suppressed() (the SDK's own provider calls): hooks pass through and record nothing.

Error policy at the hook (architecture graft 3): recording errors are swallowed and logged
once; intervention errors propagate as InterventionNotApplicable; the user's retrieval runs
exactly once either way, and its own exceptions propagate untouched.
"""

from __future__ import annotations

import copy
import functools
import inspect
import logging
import threading
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator, Literal, Mapping, Protocol, Sequence, TypeVar, overload, runtime_checkable

from . import _text as text
from ._intervention import BASELINE, NO_INTERVENTION, Intervention, InterventionNotApplicable
from ._intervention import apply as apply_intervention
from ._llm import cost_usd
from ._store import Store
from .types import Answered, Chunk, LLMCall, Metrics, Outcome, RetrievalEvent, TestOutcome, Trace

R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])

log = logging.getLogger("contexteval")


# --- per-run state ----------------------------------------------------------


@dataclass
class ActiveRun:
    """Mutable, owned by exactly one runner.execute call. Never shared across runs."""

    question: str
    intervention: Intervention
    retrievals: list[RetrievalEvent] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)
    # Set when a hook refused the intervention. The app's own `except Exception` may swallow the
    # raised error, so runner.execute re-raises this after answer() returns (graft 3).
    not_applicable: InterventionNotApplicable | None = None

    def record(self, event: RetrievalEvent) -> None:
        self.retrievals.append(event)


AMBIENT_MAX_EVENTS = 32  # retrievals kept per context between LLM calls
AMBIENT_MAX_AGE_S = 60.0  # a retrieval older than this at LLM time belongs to an earlier request


@dataclass
class _Ambient:
    """Production buffer for one context: the retrievals since the last hooked LLM call.

    Production has no runner to mark request boundaries, so the buffer is bounded (an app whose
    LLM is not hooked never drains it) and each event is stamped, so a request that never
    reached an LLM cannot hand its retrievals to the next request on the same worker thread.
    """

    question = ""
    intervention = NO_INTERVENTION
    events: deque[tuple[float, RetrievalEvent]] = field(default_factory=lambda: deque(maxlen=AMBIENT_MAX_EVENTS))

    def record(self, event: RetrievalEvent) -> None:
        self.events.append((perf_counter(), event))

    def close(self, call: LLMCall, now: float) -> ActiveRun:
        """Pair the LLM call with the retrievals it consumed. Events older than AMBIENT_MAX_AGE_S
        are stale. When any recent event shows in the prompt (its query or a delivered chunk),
        the recent events that do not are another request's; when none shows, the app put no
        retrieved text in the prompt and every recent event stays, so a filtered-out context is
        still visible."""
        recent = [e for t, e in self.events if now - t <= AMBIENT_MAX_AGE_S]
        consumed = [e for e in recent if _in_prompt(e, call.prompt)]
        kept = consumed or recent
        question = next((e.query for e in kept if e.query), "")
        return ActiveRun(question=question, intervention=NO_INTERVENTION, retrievals=kept, llm_calls=[call])


def _in_prompt(event: RetrievalEvent, prompt: str) -> bool:
    if event.query and text.chunk_in_prompt(event.query, prompt):
        return True
    return bool(text.in_prompt_pairs((event,), prompt))


def _refuse(run: ActiveRun | _Ambient, error: InterventionNotApplicable) -> InterventionNotApplicable:
    if isinstance(run, ActiveRun):
        run.not_applicable = error
    return error


_active: ContextVar[ActiveRun | None] = ContextVar("contexteval_active", default=None)
_ambient: ContextVar[_Ambient | None] = ContextVar("contexteval_ambient", default=None)
_suppressed: ContextVar[bool] = ContextVar("contexteval_suppressed", default=False)


@contextmanager
def active_run(question: str, intervention: Intervention = NO_INTERVENTION) -> Iterator[ActiveRun]:
    """Set the run for the duration of one answer() call; reset on exit even on error.

    The run is also held in a process-wide slot, because a ContextVar does not reach
    ThreadPoolExecutor workers the app may retrieve on. One run at a time is the runner's
    invariant (plan item 7), so the slot never holds a second run."""
    run = ActiveRun(question=question, intervention=intervention)
    token = _active.set(run)
    _process.current = run
    try:
        yield run
    finally:
        _process.current = None
        _active.reset(token)


@contextmanager
def suppressed() -> Iterator[None]:
    """Wrap the SDK's own LLM calls (test generation, grading, judge) so hooks skip them."""
    token = _suppressed.set(True)
    try:
        yield
    finally:
        _suppressed.reset(token)


def finalize_trace(run: ActiveRun, outcome: Outcome, latency_ms: float, test: TestOutcome | None = None) -> Trace:
    """Freeze a run into a Trace. Token counts come from provider usage when the response had
    it, else from the chars/4 estimate; cost prices those same counts and is None unless every
    call's model is priced."""
    counts = [
        (
            c.input_tokens if c.input_tokens is not None else text.estimate_tokens(c.prompt),
            c.output_tokens if c.output_tokens is not None else text.estimate_tokens(c.answer or ""),
        )
        for c in run.llm_calls
    ]
    input_tokens = sum(i for i, _ in counts)
    output_tokens = sum(o for _, o in counts)
    costs = [cost_usd(c.model, i, o) for c, (i, o) in zip(run.llm_calls, counts)]
    priced = [c for c in costs if c is not None]
    cost = sum(priced) if priced and len(priced) == len(costs) else None
    retrievals = tuple(run.retrievals)
    llm_calls = tuple(run.llm_calls)
    prompt = "\n".join(c.prompt for c in llm_calls)
    return Trace(
        question=run.question,
        outcome=outcome,
        retrievals=retrievals,
        llm_calls=llm_calls,
        metrics=Metrics(input_tokens=input_tokens, output_tokens=output_tokens, latency_ms=latency_ms, cost_usd=cost),
        test=test,
        in_prompt=text.in_prompt_pairs(retrievals, prompt),
    )


# --- process state (single owner: this module) ------------------------------


class Mode(Enum):
    OFF = "off"  # hooks not installed
    CLI = "cli"  # installed by the CLI; record only inside active runs
    PRODUCTION = "production"  # installed by init(); also record outside runs


Sink = Callable[[ActiveRun], None]


@dataclass
class _Process:
    mode: Mode = Mode.OFF
    sink: Sink | None = None  # writes an ambient buffer as a Trace
    current: ActiveRun | None = None  # the one runner.execute in progress, visible to every thread
    logged: set[str] = field(default_factory=set)  # keys of warnings already emitted


_process = _Process()


@dataclass(frozen=True)
class InstallReport:
    """library name -> status. Printed by `contexteval run` and returned by init()."""

    statuses: dict[str, Literal["installed", "already", "not_installed", "incompatible"]]


def set_mode(mode: Mode, make_sink: Callable[[], Sink] | None = None) -> None:
    """Mode only ever escalates OFF -> CLI or OFF -> PRODUCTION. If the CLI installed first, a
    later init() inside the user's module keeps CLI mode, so `contexteval run` never writes
    production traces. The sink is built only when the mode actually changes, so that no-op
    init() opens no database."""
    if _process.mode is Mode.OFF:
        _process.mode = mode
        _process.sink = make_sink() if make_sink is not None else None


def reset_process() -> None:
    """Test seam: back to OFF with no buffered events and no sink."""
    _process.mode = Mode.OFF
    _process.sink = None
    _process.current = None
    _process.logged.clear()
    _ambient.set(None)


def production_sink(project_dir: Path) -> Sink:
    """One production run per process, created on the first flushed trace, so an app that never
    reaches an LLM call leaves no empty run behind."""
    store = Store.open(project_dir)
    lock = threading.Lock()
    run_id: list[Any] = []

    def sink(run: ActiveRun) -> None:
        with lock:
            if not run_id:
                run_id.append(store.new_run("production", None, BASELINE))
            store.add_trace(run_id[0], _ambient_trace(run))

    return sink


def _ambient_trace(run: ActiveRun) -> Trace:
    """Production traces have no runner: the answer is the LLM call's text and latency is the
    sum of the hooked calls. The question was chosen when the buffer closed."""
    answer = run.llm_calls[-1].answer if run.llm_calls else None
    latency = sum(e.latency_ms for e in run.retrievals) + sum(c.latency_ms for c in run.llm_calls)
    return finalize_trace(run, Answered(answer or ""), latency)


def _log_once(key: str, message: str, error: BaseException) -> None:
    if key not in _process.logged:
        _process.logged.add(key)
        log.warning("%s: %s: %s", message, type(error).__name__, error)


def _current_run() -> ActiveRun | _Ambient | None:
    """Where this hook records: the active run, the production buffer, or nowhere (pass through)."""
    run = _active.get() or _process.current
    if run is not None:
        return run
    if _process.mode is not Mode.PRODUCTION:
        return None
    buffer = _ambient.get()
    if buffer is None:
        buffer = _Ambient()
        _ambient.set(buffer)
    return buffer


# --- the two intercepts every hook calls ------------------------------------


@runtime_checkable
class RetrievalCodec(Protocol[R]):
    """Translates one retrieval API's native call and result to and from Chunks."""

    source: str
    top_k_param: str | None  # name of the argument to override, None if not overridable

    def query_text(self, args: inspect.BoundArguments) -> str | None: ...

    def decode(self, result: R) -> tuple[Chunk, ...]: ...

    def encode(self, result: R, chunks: tuple[Chunk, ...]) -> R:
        """Rebuild a native result: reuse rows by source_index, synthesize rows for injected chunks."""
        ...


@runtime_checkable
class LLMAdapter(Protocol):
    provider: Literal["openai", "anthropic"]

    def prompt_text(self, args: inspect.BoundArguments) -> str: ...

    def model(self, args: inspect.BoundArguments) -> str: ...

    def parse(self, response: Any) -> tuple[str | None, int | None, int | None]:
        """(answer text, input tokens, output tokens). Answer None for streams."""
        ...


_SNAPSHOT_MAX_ITEMS = 20


def _json_safe(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth < 3 and isinstance(value, Mapping):
        return {str(k): _json_safe(v, depth + 1) for k, v in value.items()}
    if depth < 3 and isinstance(value, (list, tuple)):
        if len(value) > _SNAPSHOT_MAX_ITEMS:
            return f"<{type(value).__name__} of {len(value)}>"
        return [_json_safe(v, depth + 1) for v in value]
    return f"<{type(value).__name__}>"


def _snapshot(args: inspect.BoundArguments) -> dict[str, Any]:
    return {name: _json_safe(value) for name, value in args.arguments.items() if name not in ("self", "cls")}


def intercept_retrieval(codec: RetrievalCodec[R], call: Callable[[inspect.BoundArguments], R], args: inspect.BoundArguments) -> R:
    """Shared body of every retrieval hook and the @retriever decorator."""
    if _suppressed.get():
        return call(args)
    run = _current_run()
    if run is None:
        return call(args)
    iv = run.intervention

    if iv.top_k is not None:
        if codec.top_k_param is None:
            raise _refuse(run, InterventionNotApplicable(f"{codec.source}: top_k={iv.top_k} requested but this site has no top_k parameter"))
        args.arguments[codec.top_k_param] = iv.top_k

    t0 = perf_counter()
    raw = call(args)
    latency_ms = (perf_counter() - t0) * 1000

    try:
        retrieved = codec.decode(raw)
    except Exception as e:
        if iv.is_noop:
            _log_once(f"decode:{codec.source}", f"contexteval could not read {codec.source} result; not recorded", e)
            return raw
        raise _refuse(run, InterventionNotApplicable(f"{codec.source}: cannot decode result to apply the intervention")) from e

    try:
        query = codec.query_text(args)
    except Exception as e:
        _log_once(f"query:{codec.source}", f"contexteval could not read the {codec.source} query text", e)
        query = None

    if iv.is_noop:
        delivered, out = retrieved, raw
    else:
        try:
            delivered = apply_intervention(iv, retrieved, query if query is not None else run.question)
            out = codec.encode(raw, delivered)
        except InterventionNotApplicable as e:
            raise _refuse(run, e)
        except Exception as e:
            raise _refuse(run, InterventionNotApplicable(f"{codec.source}: cannot rebuild the result after the intervention")) from e

    try:
        run.record(
            RetrievalEvent(
                source=codec.source,
                query=query,
                params=_snapshot(args),
                retrieved=retrieved,
                delivered=delivered,
                latency_ms=latency_ms,
            )
        )
    except Exception as e:
        _log_once(f"record:{codec.source}", f"contexteval could not record the {codec.source} call", e)
    return out


def intercept_llm(adapter: LLMAdapter, call: Callable[[inspect.BoundArguments], Any], args: inspect.BoundArguments) -> Any:
    """Record only. v1 never changes the prompt (non-retrieval prompt text is read-only).
    In production mode with no active run, the LLM call closes the ambient buffer: one call,
    one stored trace. The buffer leaves the context before the call, so a call that raises
    cannot hand its retrievals to the next one."""
    if _suppressed.get():
        return call(args)
    run = _current_run()
    if run is None:
        return call(args)
    if isinstance(run, _Ambient):
        _ambient.set(None)

    try:
        prompt, model = adapter.prompt_text(args), adapter.model(args)
    except Exception as e:
        _log_once(f"prompt:{adapter.provider}", f"contexteval could not read the {adapter.provider} prompt", e)
        prompt, model = "", ""

    t0 = perf_counter()
    response = call(args)
    latency_ms = (perf_counter() - t0) * 1000

    try:
        answer, input_tokens, output_tokens = adapter.parse(response)
    except Exception as e:
        _log_once(f"parse:{adapter.provider}", f"contexteval could not read the {adapter.provider} response", e)
        answer, input_tokens, output_tokens = None, None, None
    llm_call = LLMCall(
        provider=adapter.provider,
        model=model,
        prompt=prompt,
        answer=answer,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )

    if isinstance(run, ActiveRun):
        run.llm_calls.append(llm_call)
        return response
    sink = _process.sink
    if sink is not None:
        try:
            sink(run.close(llm_call, t0))
        except Exception as e:
            _log_once("sink", "contexteval could not store a production trace", e)
    return response


# --- decorator for custom retrieval (pgvector, SQL, internal APIs) ----------


TextOf = Callable[[Any], str]
Rebuild = Callable[[str], Any]


@dataclass(frozen=True)
class _DecoratorCodec:
    """Accepts results that are a sequence of str, of Mappings, or of arbitrary objects.

    Chunk text comes from `text`: a key or attribute name, or a callable over the item. Injected
    chunks are built by `rebuild(text)` when given; otherwise str items become the text and
    Mapping items become {text: text}. Other item types need `rebuild` (architecture graft 4).
    """

    source: str
    top_k_param: str | None
    query_param: str | None
    text: str | TextOf
    id_field: str | None
    score_field: str | None
    rebuild: Rebuild | None

    def _text_of(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        if callable(self.text):
            return str(self.text(item))
        if isinstance(item, Mapping):
            return str(item[self.text])
        return str(getattr(item, self.text))

    @staticmethod
    def _field(item: Any, name: str | None) -> Any:
        if name is None or isinstance(item, str):
            return None
        if isinstance(item, Mapping):
            return item.get(name)
        return getattr(item, name, None)

    def query_text(self, args: inspect.BoundArguments) -> str | None:
        if self.query_param is None:
            return None
        value = args.arguments.get(self.query_param)
        return value if isinstance(value, str) else None

    def decode(self, result: Any) -> tuple[Chunk, ...]:
        # A generator would be exhausted here and reach the user empty; a str or Mapping would
        # decode as characters or keys. Refusing keeps the user's result untouched.
        if isinstance(result, (str, bytes, Mapping)) or not isinstance(result, Sequence):
            raise TypeError(f"{self.source} returned {type(result).__name__}; @retriever expects a list or tuple of items")
        chunks = []
        for index, item in enumerate(result):
            cid = self._field(item, self.id_field)
            score = self._field(item, self.score_field)
            chunks.append(
                Chunk(
                    text=self._text_of(item),
                    id=str(cid) if cid is not None else None,
                    score=float(score) if score is not None else None,
                    source_index=index,
                )
            )
        return tuple(chunks)

    def _build(self, chunk_text: str, items: list[Any]) -> Any:
        if self.rebuild is not None:
            return self.rebuild(chunk_text)
        if items and isinstance(items[0], str):
            return chunk_text
        if items and isinstance(items[0], Mapping) and isinstance(self.text, str):
            return {self.text: chunk_text}
        kind = type(items[0]).__name__ if items else "an empty result"
        raise InterventionNotApplicable(f"{self.source}: cannot build an item for injected text from {kind}; pass rebuild= to @retriever")

    def _with_text(self, item: Any, chunk_text: str) -> Any:
        if isinstance(item, str):
            return chunk_text
        if isinstance(item, Mapping) and isinstance(self.text, str):
            return {**item, self.text: chunk_text}
        if isinstance(self.text, str):
            clone = copy.copy(item)
            setattr(clone, self.text, chunk_text)
            return clone
        raise InterventionNotApplicable(f"{self.source}: cannot rewrite item text through a callable text=; pass rebuild= to @retriever")

    def encode(self, result: Any, chunks: tuple[Chunk, ...]) -> Any:
        items = list(result)
        out = []
        for c in chunks:
            if c.source_index is None:
                out.append(self._build(c.text, items))
                continue
            item = items[c.source_index]
            out.append(item if self._text_of(item) == c.text else self._with_text(item, c.text))
        return tuple(out) if isinstance(result, tuple) else out


@overload
def retriever(fn: F) -> F: ...
@overload
def retriever(
    *,
    top_k: str | None = "top_k",
    query: str | None = None,
    text: str | TextOf = "text",
    id: str | None = "id",
    score: str | None = "score",
    rebuild: Rebuild | None = None,
) -> Callable[[F], F]: ...
def retriever(
    fn: Any = None,
    *,
    top_k: str | None = "top_k",
    query: str | None = None,
    text: str | TextOf = "text",
    id: str | None = "id",
    score: str | None = "score",
    rebuild: Rebuild | None = None,
) -> Any:
    """Mark a user search function as a retrieval source.

        @contexteval.retriever                 # top_k override uses a parameter named `top_k`
        @contexteval.retriever(top_k="k")      # the user's parameter is called `k`
        @contexteval.retriever(top_k=None)     # not overridable; a top_k intervention raises

    Validated at decoration time (the boundary): an explicitly named top_k or query that is
    not in the signature raises TypeError at import. The default "top_k" silently becomes
    None when absent. `query` defaults to the first parameter other than `self` or `cls`.
    """

    def decorate(func: F) -> F:
        sig = inspect.signature(func)
        names = [n for n in sig.parameters if n not in ("self", "cls")]
        top_k_param = top_k
        if top_k is not None and top_k not in sig.parameters:
            if top_k != "top_k":
                raise TypeError(f"@contexteval.retriever: {func.__name__}() has no parameter {top_k!r} to use as top_k")
            top_k_param = None
        if query is not None and query not in sig.parameters:
            raise TypeError(f"@contexteval.retriever: {func.__name__}() has no parameter {query!r} to use as the query")
        codec = _DecoratorCodec(
            source=f"retriever:{func.__name__}",
            top_k_param=top_k_param,
            query_param=query if query is not None else (names[0] if names else None),
            text=text,
            id_field=id,
            score_field=score,
            rebuild=rebuild,
        )

        @functools.wraps(func)
        def wrapper(*a: Any, **kw: Any) -> Any:
            bound = sig.bind(*a, **kw)
            return intercept_retrieval(codec, lambda b: func(*b.args, **b.kwargs), bound)

        wrapper.__contexteval_original__ = func  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate(fn) if fn is not None else decorate
