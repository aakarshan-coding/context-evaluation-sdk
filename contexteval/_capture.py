"""Capture core: the active-run context variable, the two intercept functions every hook
calls (each in a sync and an async form over one shared body), the wrapper builder the hook
table and @retriever share, trace finalization, and the process mode.

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
- Inside another hooked retrieval in the same context (a LangChain retriever over a hooked
  chromadb collection, a @retriever function that queries one): the inner call records nothing.
  The outermost hooked retrieval owns the event and the transforms, so one user-level
  retrieval is one RetrievalEvent and the intervention applies once. top_k goes to the
  outermost layer that can set it: the outer layer first, else the first hooked call
  underneath with a knob (a custom retriever over chromadb, a wrapper over a base retriever).
  Only when the outer call returns with no layer having set it is the intervention refused.

Error policy at the hook (architecture graft 3): recording errors are swallowed and logged
once; intervention errors propagate as InterventionNotApplicable; the user's retrieval runs
exactly once either way, and its own exceptions propagate untouched.
"""

from __future__ import annotations

import atexit
import copy
import functools
import inspect
import logging
import os
import queue
import threading
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable, Iterator, Literal, Mapping, Protocol, Sequence, TypeVar, overload, runtime_checkable

from . import _text as text
from ._intervention import BASELINE, NO_INTERVENTION, Intervention, InterventionNotApplicable
from ._intervention import apply as apply_intervention
from ._llm import cost_usd
from ._store import Store
from .types import Answered, Chunk, LLMCall, Metrics, Outcome, RetrievalEvent, RunId, TestOutcome, Trace

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


@dataclass(frozen=True)
class PendingTrace:
    """A closed production buffer with the LLM call that closed it. Pairing the two reads the
    whole prompt once per event, so it runs on the writer thread, not the user's request."""

    buffer: _Ambient
    call: LLMCall
    at: float

    def close(self) -> ActiveRun:
        return self.buffer.close(self.call, self.at)


def _refuse(run: ActiveRun | _Ambient, error: InterventionNotApplicable) -> InterventionNotApplicable:
    if isinstance(run, ActiveRun):
        run.not_applicable = error
    return error


@dataclass
class _Outermost:
    """The outermost hooked retrieval in progress in this context. It owns the event and the
    transforms. top_k is owned by the outermost layer that can apply it: the run's value stays
    pending until this layer or a hooked call underneath sets its own knob, and the event
    records the knob that was set. When the call returns with the value still pending, the
    outermost layer refuses with its own reason."""

    top_k: int | None  # the run's request; None when the intervention has none
    applied: tuple[str, int] | None = None  # (knob name, value) set by the layer that took it
    refusal: InterventionNotApplicable | None = None  # the outermost layer's reason it could not

    @property
    def pending(self) -> int | None:
        return self.top_k if self.applied is None else None


_active: ContextVar[ActiveRun | None] = ContextVar("contexteval_active", default=None)
_ambient: ContextVar[_Ambient | None] = ContextVar("contexteval_ambient", default=None)
_suppressed: ContextVar[bool] = ContextVar("contexteval_suppressed", default=False)
# Set while a hooked retrieval is in progress in this context. A hooked call underneath records
# nothing and only takes a pending top_k. A ContextVar, so an asyncio task's inner calls see its
# own state, and langchain's executor hand-off copies it.
_retrieving: ContextVar[_Outermost | None] = ContextVar("contexteval_retrieving", default=None)


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


Sink = Callable[[PendingTrace], None]  # receives each closed production buffer


@dataclass
class _Process:
    mode: Mode = Mode.OFF
    sink: Sink | None = None  # the ProductionWriter in production mode; tests inject lists
    current: ActiveRun | None = None  # the one runner.execute in progress, visible to every thread
    pinecone_text_key: str = "text"  # read by the pinecone codec at call time, so a later init() applies
    logged: set[str] = field(default_factory=set)  # keys of warnings already emitted


_process = _Process()


@dataclass(frozen=True)
class InstallReport:
    """library name -> status, returned by init(). `failed` means hooking that library raised;
    it was logged once and the library runs unhooked. `error` is set when init() itself could
    not finish (no project database); then nothing was installed and hooks pass through."""

    statuses: dict[str, Literal["installed", "already", "not_installed", "incompatible", "failed"]]
    error: str | None = None


def mode() -> Mode:
    return _process.mode


def pinecone_text_key() -> str:
    return _process.pinecone_text_key


def set_pinecone_text_key(key: str | None) -> None:
    """Unlike the mode, the key is set by whoever names it last: the CLI's install runs before
    the app module imports, so a user init(pinecone_text_key=) there wins. None leaves it."""
    if key is not None:
        _process.pinecone_text_key = key


def set_mode(mode: Mode, make_sink: Callable[[], Sink] | None = None) -> None:
    """Mode only ever escalates OFF -> CLI or OFF -> PRODUCTION. If the CLI installed first, a
    later init() inside the user's module keeps CLI mode, so `contexteval run` never writes
    production traces. The sink is built only when the mode actually changes, so that no-op
    init() opens no database."""
    if _process.mode is Mode.OFF:
        _process.mode = mode
        _process.sink = make_sink() if make_sink is not None else None


def reset_process() -> None:
    """Test seam: back to OFF with no buffered events, no sink, and the writer stopped."""
    if isinstance(_process.sink, ProductionWriter):
        _process.sink.close()
    _process.mode = Mode.OFF
    _process.sink = None
    _process.current = None
    _process.pinecone_text_key = "text"
    _process.logged.clear()
    _ambient.set(None)
    _retrieving.set(None)
    forget_knobs()


WRITER_QUEUE_MAX = 256  # closed buffers waiting for the writer; beyond this they are dropped and counted
WRITER_FLUSH_S = 5.0
ATEXIT_FLUSH_S = 2.0  # how long interpreter exit waits for pending production traces

_getpid = os.getpid


class _Flush:
    def __init__(self) -> None:
        self.done = threading.Event()


class ProductionWriter:
    """The production sink. The hot path enqueues a closed buffer and returns; one daemon thread
    finalizes and writes it.

    The thread owns the only connection, opened on the first write and dropped when the pid
    changes, because a forked child inherits the parent's connection, queue, and no thread.
    One production run per process, created on the first stored trace, so an app that never
    reaches an LLM call leaves no empty run behind. A full queue drops and counts.
    """

    def __init__(self, project_dir: Path, maxsize: int = WRITER_QUEUE_MAX) -> None:
        self._project_dir = project_dir
        self._maxsize = maxsize
        self._queue: queue.Queue[PendingTrace | _Flush | None] = queue.Queue(maxsize)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pid: int | None = None
        self._store: Store | None = None
        self._run_id: RunId | None = None
        self.dropped = 0
        self.closed = False
        atexit.register(self.flush, ATEXIT_FLUSH_S)

    def __call__(self, pending: PendingTrace) -> None:
        self._ensure_thread()
        try:
            self._queue.put_nowait(pending)
        except queue.Full:
            self.dropped += 1
            _log_once("writer:full", "contexteval dropped a production trace; the writer queue is full")

    def _ensure_thread(self) -> None:
        pid = _getpid()
        with self._lock:
            thread = self._thread
            if thread is not None and thread.is_alive() and self._pid == pid:
                return
            if self._pid != pid:
                self._store, self._run_id = None, None  # inherited from the parent: never use, never close
                self._queue = queue.Queue(self._maxsize)
            self._pid = pid
            self._thread = threading.Thread(target=self._drain, args=(self._queue,), name="contexteval-writer", daemon=True)
            self._thread.start()

    def _drain(self, pending: queue.Queue[PendingTrace | _Flush | None]) -> None:
        while True:
            item = pending.get()
            if item is None:
                break
            if isinstance(item, _Flush):
                item.done.set()
                continue
            try:
                self._write(item)
            except Exception as e:
                _log_once("writer", "contexteval could not store a production trace", e)
        store, self._store = self._store, None
        if store is not None:
            store.close()

    def _write(self, pending: PendingTrace) -> None:
        if self._store is None:
            self._store = Store.open(self._project_dir)
            self._run_id = None
        if self._run_id is None:
            self._run_id = self._store.new_run("production", None, BASELINE)
        self._store.add_trace(self._run_id, _ambient_trace(pending.close()))

    def flush(self, timeout: float = WRITER_FLUSH_S) -> bool:
        """Wait until every buffer enqueued so far is written. False when that took longer than
        `timeout` (the writer keeps going) or the queue is full."""
        thread = self._thread
        if self.closed or thread is None or not thread.is_alive():
            return self._queue.empty()
        marker = _Flush()
        try:
            self._queue.put(marker, timeout=timeout)
        except queue.Full:
            return False
        return marker.done.wait(timeout)

    def close(self, timeout: float = WRITER_FLUSH_S) -> None:
        """Flush, stop the thread, and close the connection."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            self.flush(timeout)
            try:
                self._queue.put(None, timeout=timeout)
            except queue.Full:
                pass
            thread.join(timeout)
        self.closed = True


def flush_production(timeout: float = WRITER_FLUSH_S) -> bool:
    """Block until the production writer has stored every trace closed so far. True when nothing
    is pending (including outside production mode)."""
    sink = _process.sink
    return sink.flush(timeout) if isinstance(sink, ProductionWriter) else True


def production_dropped() -> int:
    """Traces the production writer dropped because its queue was full."""
    sink = _process.sink
    return sink.dropped if isinstance(sink, ProductionWriter) else 0


def _ambient_trace(run: ActiveRun) -> Trace:
    """Production traces have no runner: the answer is the LLM call's text and latency is the
    sum of the hooked calls. The question was chosen when the buffer closed."""
    answer = run.llm_calls[-1].answer if run.llm_calls else None
    latency = sum(e.latency_ms for e in run.retrievals) + sum(c.latency_ms for c in run.llm_calls)
    return finalize_trace(run, Answered(answer or ""), latency)


def _log_once(key: str, message: str, error: BaseException | None = None) -> None:
    if key in _process.logged:
        return
    _process.logged.add(key)
    if error is None:
        log.warning("%s", message)
    else:
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


Restore = Callable[[], None]

_knob_lock = threading.Lock()
# (id(instance), attribute) -> (the user's value, calls currently overriding it)
_knobs: dict[tuple[int, str], tuple[Any, int]] = {}


def override_attr(instance: Any, attr: str, value: Any) -> Restore:
    """Set `instance.attr` for one call and return what puts the user's value back.

    Calls that overlap on one instance (an app that gathers two retrievals for one question)
    all want the same value, because one run has one intervention: the first one in saves the
    user's value and the last one out restores it, so no call restores another call's override.
    Two runs wanting different values on one shared instance are not supported; the runner
    executes one question at a time (plan item 7). The entry is recorded only once setattr
    succeeded, so a setter that rejects the value leaves nothing for a later restore to skip."""
    key = (id(instance), attr)
    with _knob_lock:
        saved, depth = _knobs.get(key, (getattr(instance, attr), 0))
        setattr(instance, attr, value)
        _knobs[key] = (saved, depth + 1)

    def restore() -> None:
        with _knob_lock:
            saved, depth = _knobs[key]
            if depth > 1:
                _knobs[key] = (saved, depth - 1)
            else:
                del _knobs[key]
                setattr(instance, attr, saved)

    return restore


def forget_knobs() -> None:
    """Test seam, with reset_process and _hooks.uninstall_all: drop every saved knob value."""
    with _knob_lock:
        _knobs.clear()


@runtime_checkable
class RetrievalCodec(Protocol[R]):
    """Translates one retrieval API's native call and result to and from Chunks. Codecs subclass
    it explicitly, so the argument form of override_top_k is implemented once here."""

    source: str
    top_k_param: str | None  # name of this site's top_k knob, as the event records it; None when it has none

    def query_text(self, args: inspect.BoundArguments) -> str | None: ...

    def decode(self, result: R) -> tuple[Chunk, ...]: ...

    def encode(self, result: R, chunks: tuple[Chunk, ...]) -> R:
        """Rebuild a native result: reuse rows by source_index, synthesize rows for injected chunks."""
        ...

    def override_top_k(self, args: inspect.BoundArguments, top_k: int) -> Restore | None:
        """Put the run's top_k where this one call reads it. The default sets the argument named
        top_k_param. A codec whose knob is state on the retriever object (a framework retriever's
        `search_kwargs` or `similarity_top_k`) overrides this, sets the knob for the call, and
        returns what puts the user's value back. Raises InterventionNotApplicable when this site
        cannot apply the value; a hooked call underneath may still."""
        if self.top_k_param is None:
            raise InterventionNotApplicable(f"{self.source}: top_k={top_k} requested but this site has no top_k parameter")
        args.arguments[self.top_k_param] = top_k
        return None


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


# Each intercept is one decision before the library call and one after it, split so the sync
# and async wrappers share every rule (decode, intervention, record, error policy) and differ
# only in whether they await the call. `None` from a begin() means pass through untouched.


def _run_restore(codec: RetrievalCodec[Any], restore: Restore | None) -> None:
    if restore is None:
        return
    try:
        restore()
    except Exception as e:
        _log_once(f"restore:{codec.source}", f"contexteval could not restore the {codec.source} top_k knob", e)


@dataclass
class _InnerOverride:
    """A hooked call underneath the outermost retrieval that took the pending top_k onto its own
    knob. It puts the user's value back when its call returns and records nothing."""

    codec: RetrievalCodec[Any]
    restore: Restore

    def release(self) -> None:
        _run_restore(self.codec, self.restore)

    def finish(self, raw: R) -> R:
        return raw


@dataclass
class _PendingRetrieval:
    codec: RetrievalCodec[Any]
    run: ActiveRun | _Ambient
    args: inspect.BoundArguments
    params: dict[str, Any]  # the event's snapshot after this layer's override; the applied knob is added by name at finish
    token: Token[_Outermost | None]  # the _retrieving state this call set; release() resets it
    restore: Restore | None  # puts this layer's instance knob back; None for an argument knob or no override
    t0: float
    outer: _Outermost

    def release(self) -> None:
        """Runs once the library call returned or raised, before finish: inner hooked calls record
        again, and the retriever's own knob holds the user's value before anything else reads it."""
        _retrieving.reset(self.token)
        _run_restore(self.codec, self.restore)

    def finish(self, raw: R) -> R:
        latency_ms = (perf_counter() - self.t0) * 1000
        codec, run, args, iv = self.codec, self.run, self.args, self.run.intervention

        if self.outer.refusal is not None and self.outer.applied is None:
            raise _refuse(run, self.outer.refusal)
        if self.outer.applied is not None:
            name, value = self.outer.applied
            self.params[name] = value

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
                    params=self.params,
                    retrieved=retrieved,
                    delivered=delivered,
                    latency_ms=latency_ms,
                )
            )
        except Exception as e:
            _log_once(f"record:{codec.source}", f"contexteval could not record the {codec.source} call", e)
        return out


def _take_top_k(codec: RetrievalCodec[Any], args: inspect.BoundArguments, outer: _Outermost) -> Restore | None:
    """Offer the pending top_k to this layer. When it applies, `outer.applied` names the knob and
    the restore (if any) comes back for release(). When it cannot, the value stays pending for a
    hooked call underneath, and the outermost layer's own reason is kept for the refusal."""
    top_k = outer.pending
    if top_k is None:
        return None
    try:
        restore = codec.override_top_k(args, top_k)
    except InterventionNotApplicable as e:
        outer.refusal = outer.refusal or e
        return None
    except Exception as e:
        refusal = InterventionNotApplicable(f"{codec.source}: cannot set top_k={top_k} on this retriever")
        refusal.__cause__ = e
        outer.refusal = outer.refusal or refusal
        return None
    outer.applied = (codec.top_k_param or "top_k", top_k)
    return restore


def _begin_retrieval(codec: RetrievalCodec[Any], args: inspect.BoundArguments) -> _PendingRetrieval | _InnerOverride | None:
    if _suppressed.get():
        return None
    outer = _retrieving.get()
    if outer is not None:
        restore = _take_top_k(codec, args, outer)
        return _InnerOverride(codec, restore) if restore is not None else None
    run = _current_run()
    if run is None:
        return None
    outer = _Outermost(top_k=run.intervention.top_k)
    restore = _take_top_k(codec, args, outer)
    params = _snapshot(args)
    return _PendingRetrieval(codec, run, args, params, _retrieving.set(outer), restore, perf_counter(), outer)


def intercept_retrieval(codec: RetrievalCodec[R], call: Callable[[inspect.BoundArguments], R], args: inspect.BoundArguments) -> R:
    """Shared body of every sync retrieval hook and the @retriever decorator."""
    pending = _begin_retrieval(codec, args)
    if pending is None:
        return call(args)
    try:
        raw = call(args)
    finally:
        pending.release()
    return pending.finish(raw)


async def intercept_retrieval_async(codec: RetrievalCodec[R], call: Callable[[inspect.BoundArguments], Awaitable[R]], args: inspect.BoundArguments) -> R:
    """The same intercept for a coroutine target: the call is awaited, nothing else differs."""
    pending = _begin_retrieval(codec, args)
    if pending is None:
        return await call(args)
    try:
        raw = await call(args)
    finally:
        pending.release()
    return pending.finish(raw)


@dataclass
class _PendingLLM:
    adapter: LLMAdapter
    run: ActiveRun | _Ambient
    prompt: str
    model: str
    t0: float

    def finish(self, response: Any) -> Any:
        latency_ms = (perf_counter() - self.t0) * 1000
        adapter, run = self.adapter, self.run
        try:
            answer, input_tokens, output_tokens = adapter.parse(response)
        except Exception as e:
            _log_once(f"parse:{adapter.provider}", f"contexteval could not read the {adapter.provider} response", e)
            answer, input_tokens, output_tokens = None, None, None
        llm_call = LLMCall(
            provider=adapter.provider,
            model=self.model,
            prompt=self.prompt,
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
                sink(PendingTrace(run, llm_call, self.t0))
            except Exception as e:
                _log_once("sink", "contexteval could not store a production trace", e)
        return response


def _begin_llm(adapter: LLMAdapter, args: inspect.BoundArguments) -> _PendingLLM | None:
    if _suppressed.get():
        return None
    run = _current_run()
    if run is None:
        return None
    if isinstance(run, _Ambient):
        _ambient.set(None)
    try:
        prompt, model = adapter.prompt_text(args), adapter.model(args)
    except Exception as e:
        _log_once(f"prompt:{adapter.provider}", f"contexteval could not read the {adapter.provider} prompt", e)
        prompt, model = "", ""
    return _PendingLLM(adapter, run, prompt, model, perf_counter())


def intercept_llm(adapter: LLMAdapter, call: Callable[[inspect.BoundArguments], Any], args: inspect.BoundArguments) -> Any:
    """Record only. v1 never changes the prompt (non-retrieval prompt text is read-only).
    In production mode with no active run, the LLM call closes the ambient buffer: one call,
    one stored trace. The buffer leaves the context before the call, so a call that raises
    cannot hand its retrievals to the next one."""
    pending = _begin_llm(adapter, args)
    response = call(args)
    return response if pending is None else pending.finish(response)


async def intercept_llm_async(adapter: LLMAdapter, call: Callable[[inspect.BoundArguments], Awaitable[Any]], args: inspect.BoundArguments) -> Any:
    pending = _begin_llm(adapter, args)
    response = await call(args)
    return response if pending is None else pending.finish(response)


# --- the one wrapper shape for hooked methods and @retriever -----------------


def is_coroutine_function(fn: Callable[..., Any]) -> bool:
    """Looks through functools.wraps layers: the openai and anthropic SDKs decorate their async
    `create` methods, so the class attribute itself is a plain function returning a coroutine."""
    try:
        return inspect.iscoroutinefunction(inspect.unwrap(fn))
    except ValueError:
        return inspect.iscoroutinefunction(fn)


def wrap_call(original: Callable[..., Any], handler: RetrievalCodec[Any] | LLMAdapter) -> Callable[..., Any]:
    """Bind the call by name, hand it to the intercept that matches the handler, and keep the
    target's sync or async nature: a coroutine function gets an `async def` wrapper, so it is
    still awaitable and still reports as a coroutine function. Arguments the signature rejects
    go straight to the original, so the library raises its own error."""
    sig = inspect.signature(original)
    retrieval = isinstance(handler, RetrievalCodec)
    wrapper: Callable[..., Any]
    if is_coroutine_function(original):
        intercept_async = intercept_retrieval_async if retrieval else intercept_llm_async

        @functools.wraps(original)
        async def wrapper(*a: Any, **kw: Any) -> Any:
            try:
                bound = sig.bind(*a, **kw)
            except TypeError:
                return await original(*a, **kw)
            return await intercept_async(handler, lambda b: original(*b.args, **b.kwargs), bound)  # type: ignore[arg-type]

    else:
        intercept = intercept_retrieval if retrieval else intercept_llm

        @functools.wraps(original)
        def wrapper(*a: Any, **kw: Any) -> Any:
            try:
                bound = sig.bind(*a, **kw)
            except TypeError:
                return original(*a, **kw)
            return intercept(handler, lambda b: original(*b.args, **b.kwargs), bound)  # type: ignore[arg-type]

    wrapper.__contexteval_original__ = original  # type: ignore[attr-defined]
    return wrapper


# --- decorator for custom retrieval (pgvector, SQL, internal APIs) ----------


TextOf = Callable[[Any], str]
Rebuild = Callable[[str], Any]


@dataclass(frozen=True)
class _DecoratorCodec(RetrievalCodec[Any]):
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
    """Mark a user search function as a retrieval source. `async def` functions are supported
    and stay awaitable.

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
        return wrap_call(func, codec)  # type: ignore[return-value]

    return decorate(fn) if fn is not None else decorate
