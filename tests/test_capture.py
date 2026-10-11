"""Capture core: trace finalization, the production buffer and writer, and the active-run slot.
Driven through the public hook surface (intercept_*, @retriever, execute), not internals."""

from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from contexteval import _capture as capture
from contexteval import _hooks, retriever
from contexteval._intervention import NO_INTERVENTION, InjectEvidence, Intervention, InterventionNotApplicable
from contexteval._llm import FakeProvider, cost_usd
from contexteval._runner import execute, make_grader, run_suite
from contexteval._store import Store
from contexteval.types import Answered, CaseId, Evidence, LLMCall, TestCase, Trace


@dataclass
class Doc:
    text: str


class FakeLLM:
    """Stand-in for openai Completions, hooked with the real OpenAIChatAdapter."""

    def create(self, *, messages: list[dict[str, str]], model: str = "gpt-4.1-mini") -> Any:
        if "fail" in messages[-1]["content"]:
            raise RuntimeError("upstream 500")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ans"))], usage=None)


class AsyncFakeLLM:
    """Stand-in for openai AsyncCompletions, hooked with the real OpenAIChatAdapter."""

    async def create(self, *, messages: list[dict[str, str]], model: str = "gpt-4.1-mini") -> Any:
        await asyncio.sleep(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ans"))], usage=None)


@retriever
def search(query: str, top_k: int = 3) -> list[dict[str, str]]:
    return [{"text": f"doc {i} for {query}"} for i in range(top_k)]


@retriever
async def async_search(query: str, top_k: int = 3) -> list[dict[str, str]]:
    await asyncio.sleep(0)
    return [{"text": f"doc {i} for {query}"} for i in range(top_k)]


@pytest.fixture
def llm() -> Iterator[FakeLLM]:
    assert _hooks.patch_method(FakeLLM, "create", _hooks.OpenAIChatAdapter()) == "installed"
    try:
        yield FakeLLM()
    finally:
        _hooks.uninstall_all()
        capture.reset_process()


@pytest.fixture
def async_llm() -> Iterator[AsyncFakeLLM]:
    assert _hooks.patch_method(AsyncFakeLLM, "create", _hooks.OpenAIChatAdapter()) == "installed"
    try:
        yield AsyncFakeLLM()
    finally:
        _hooks.uninstall_all()
        capture.reset_process()


@pytest.fixture
def flushed() -> Iterator[list[capture.ActiveRun]]:
    """Production mode whose sink appends each closed buffer instead of writing it."""
    runs: list[capture.ActiveRun] = []
    capture.set_mode(capture.Mode.PRODUCTION, lambda: lambda pending: runs.append(pending.close()))
    try:
        yield runs
    finally:
        capture.reset_process()


def _ask(llm: FakeLLM, question: str, context: str = "") -> Any:
    return llm.create(messages=[{"role": "user", "content": f"{context}\n\nQ: {question}"}])


@pytest.fixture
def production(project: Path, llm: FakeLLM) -> FakeLLM:
    _hooks.install("production", project)
    return llm


def _locked(project: Path) -> sqlite3.Connection:
    locker = sqlite3.connect(str(project / ".contexteval" / "contexteval.db"), isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    return locker


def _production_traces(project: Path) -> list[Trace]:
    store = Store.open(project)
    try:
        runs = [r for r in store.runs() if r.run.kind == "production"]
        return [s.trace for r in runs for s in store.traces(r.run.id)]
    finally:
        store.close()


def test_production_llm_call_returns_fast_while_the_db_is_write_locked(project: Path, production: FakeLLM) -> None:
    """Review X3. Defect: the sink committed to SQLite on the caller's thread, so a locked database
    stalled the user's LLM call for the busy timeout (measured 5.56 s). The hot path now only
    enqueues; the writer thread stores the trace once the lock is gone."""
    locker = _locked(project)
    try:
        hits = search("locked")
        t0 = perf_counter()
        _ask(production, "locked", "\n".join(h["text"] for h in hits))
        elapsed = perf_counter() - t0
    finally:
        locker.execute("ROLLBACK")
        locker.close()
    assert elapsed < 0.1, f"LLM call took {elapsed * 1000:.1f} ms with the database locked"
    assert capture.flush_production(timeout=10)
    (trace,) = _production_traces(project)
    assert trace.question == "locked" and len(trace.retrievals) == 1 and trace.outcome == Answered("ans")


def test_writer_drops_and_counts_when_the_queue_is_full(project: Path, production: FakeLLM, caplog: pytest.LogCaptureFixture) -> None:
    """Review X3. Defect: an unbounded queue (memory grows while the database is unavailable) or
    a bounded one that blocks the caller. Every submitted trace is either stored or counted as
    dropped, and the drop is logged once."""
    submitted = capture.WRITER_QUEUE_MAX + 20
    locker = _locked(project)
    try:
        for i in range(submitted):
            _ask(production, f"burst {i}")
    finally:
        locker.execute("ROLLBACK")
        locker.close()
    assert capture.flush_production(timeout=30)
    stored = len(_production_traces(project))
    dropped = capture.production_dropped()
    assert dropped > 0 and stored + dropped == submitted
    assert caplog.text.count("writer queue is full") == 1


def test_writer_reopens_its_connection_in_a_forked_child(project: Path, production: FakeLLM, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review X3. Defect: the connection opened at init() is inherited across fork() and shared by
    two processes. A pid change must discard the inherited connection and thread and open a
    fresh one, which shows up as a second production run."""
    _ask(production, "in the parent")
    assert capture.flush_production(timeout=10)
    monkeypatch.setattr(capture, "_getpid", lambda: -1)
    _ask(production, "in the child")
    assert capture.flush_production(timeout=10)
    store = Store.open(project)
    try:
        runs = [r for r in store.runs() if r.run.kind == "production"]
        assert len(runs) == 2 and [r.traces for r in runs] == [1, 1]
    finally:
        store.close()


def test_production_buffer_does_not_bleed_across_requests(llm: FakeLLM, flushed: list[capture.ActiveRun]) -> None:
    """Review X2. Defect: a request that retrieves and never reaches an LLM (cache hit) leaves its
    event in the buffer, and the next request's trace inherits it with the wrong question. A
    retrieval the prompt shows no sign of, while another one is in the prompt, belongs to an
    earlier request; when nothing retrieved is in the prompt, the recent retrievals still count."""
    search("request A, answered from cache")
    hits = search("request B")
    _ask(llm, "request B", "\n".join(h["text"] for h in hits))
    assert len(flushed) == 1
    assert flushed[0].question == "request B"
    assert [e.query for e in flushed[0].retrievals] == ["request B"]

    search("request C, every chunk filtered out")
    _ask(llm, "request C")
    assert flushed[1].question == "request C, every chunk filtered out"
    assert len(flushed[1].retrievals) == 1


def test_production_buffer_is_cleared_when_the_llm_call_fails(llm: FakeLLM, flushed: list[capture.ActiveRun]) -> None:
    """Review X2. Defect: an LLM call that raises leaves the buffered retrievals for the next
    call, which then reports another request's context."""
    search("request D")
    with pytest.raises(RuntimeError, match="upstream 500"):
        _ask(llm, "request D fail")
    _ask(llm, "request E, no retrieval")
    assert len(flushed) == 1 and flushed[0].retrievals == []


def test_production_buffer_is_bounded(llm: FakeLLM, flushed: list[capture.ActiveRun]) -> None:
    """Review X2. Defect: an app whose LLM is not hooked buffers every retrieval forever."""
    for i in range(capture.AMBIENT_MAX_EVENTS * 4):
        search(f"request {i}")
    _ask(llm, "finally")
    assert len(flushed[0].retrievals) == capture.AMBIENT_MAX_EVENTS
    assert flushed[0].retrievals[-1].query == f"request {capture.AMBIENT_MAX_EVENTS * 4 - 1}"


def test_production_buffer_drops_events_older_than_the_window(llm: FakeLLM, flushed: list[capture.ActiveRun], monkeypatch: pytest.MonkeyPatch) -> None:
    """Review X2. Defect: a retrieval from minutes ago is attributed to the LLM call that finally
    happens on the same worker thread."""
    now = [1000.0]
    monkeypatch.setattr(capture, "perf_counter", lambda: now[0])
    search("long ago")
    now[0] += capture.AMBIENT_MAX_AGE_S + 1
    search("just now")
    _ask(llm, "just now")
    assert [e.query for e in flushed[0].retrievals] == ["just now"]


def test_intervention_not_applicable_survives_the_apps_own_except(hooks: None) -> None:
    """Review X5. Defect: the app's `except Exception` fallback swallows InterventionNotApplicable,
    execute returns a Trace, and the experiment row silently reports baseline behavior (graft 3
    forbids this). The error must reach execute's caller however the exception travelled."""

    @retriever(top_k=None)
    def fixed_search(query: str) -> list[str]:
        return ["a"]

    @retriever
    def broken_encode(query: str, top_k: int = 3) -> list[Doc]:
        return [Doc("alpha")]

    def app(question: str) -> str:
        try:
            hits = fixed_search(question)
        except Exception:
            hits = ["fallback"]
        return " ".join(hits)

    def app_rebuild(question: str) -> str:
        try:
            return str(broken_encode(question))
        except Exception:
            return "fallback"

    with pytest.raises(InterventionNotApplicable, match="no top_k parameter"):
        execute(app, "q", Intervention(top_k=10))
    with pytest.raises(InterventionNotApplicable, match="rebuild="):
        execute(app_rebuild, "q", Intervention(transforms=(InjectEvidence(Evidence("gold")),)))
    assert isinstance(execute(app, "q").outcome, Answered)


def test_worker_thread_retrievals_join_the_active_run(hooks: None) -> None:
    """Review X8. Defect: ContextVar does not reach ThreadPoolExecutor workers, so an app that
    retrieves on a worker thread records nothing and a top_k experiment leaves its retrieval
    untouched. The CLI runs one question at a time, so the run is also held process-wide."""

    @retriever
    def search(query: str, top_k: int = 3) -> list[dict[str, str]]:
        return [{"text": f"doc {i} for {query}"} for i in range(top_k)]

    def app(question: str) -> str:
        with ThreadPoolExecutor(1) as pool:
            rows = pool.submit(search, question).result()
        return f"{len(rows)} rows"

    trace = execute(app, "q1", Intervention(top_k=1))
    assert trace.outcome == Answered("1 rows")
    assert len(trace.retrievals) == 1 and trace.retrievals[0].params["top_k"] == 1
    assert execute(app, "q2").outcome == Answered("3 rows")
    assert search("outside any run") == [{"text": f"doc {i} for outside any run"} for i in range(3)]


def test_concurrent_asyncio_tasks_get_separate_production_traces(async_llm: AsyncFakeLLM, flushed: list[capture.ActiveRun]) -> None:
    """v1.1 B. Defect: the production buffer is shared by every task on the thread, so task A's
    LLM call closes a buffer holding B's retrieval too, and B's own trace then has no
    retrieval. Each asyncio task must get its own buffer."""

    async def request(name: str) -> None:
        hits = await async_search(name)
        await asyncio.sleep(0)  # let the other task retrieve before this one reaches the LLM
        context = "\n".join(h["text"] for h in hits)
        await async_llm.create(messages=[{"role": "user", "content": f"{context}\n\nQ: {name}"}])

    async def main() -> None:
        await asyncio.gather(request("request A"), request("request B"))

    asyncio.run(main())
    assert sorted(r.question for r in flushed) == ["request A", "request B"]
    for run in flushed:
        assert [e.query for e in run.retrievals] == [run.question]
        assert len(run.llm_calls) == 1 and f"doc 0 for {run.question}" in run.llm_calls[0].prompt


def test_intervention_not_applicable_survives_an_async_apps_own_except(hooks: None) -> None:
    """v1.1 B, review X5 for async apps. Defect: the refusal raised inside the awaited retriever
    is swallowed by the app's `except Exception`, and execute returns a Trace for an experiment
    row that silently ran the baseline."""

    @retriever(top_k=None)
    async def fixed_search(query: str) -> list[str]:
        return ["a"]

    async def app(question: str) -> str:
        try:
            hits = await fixed_search(question)
        except Exception:
            hits = ["fallback"]
        return " ".join(hits)

    with pytest.raises(InterventionNotApplicable, match="no top_k parameter"):
        execute(app, "q", Intervention(top_k=10))
    assert execute(app, "q").outcome == Answered("a")


def test_async_answer_runs_through_the_suite_with_in_prompt(store: Store, hooks: None, async_llm: AsyncFakeLLM) -> None:
    """v1.1 B. Defect: execute stores str(coroutine) as the answer, grades it, and the trace has
    no retrieval and no LLM call. The second case pins one event loop per run: the app's
    loop-bound clients (an AsyncOpenAI connection pool) must stay valid from one question to
    the next, which a fresh asyncio.run per question breaks."""
    loops: list[asyncio.AbstractEventLoop] = []

    async def app(question: str) -> str:
        loops.append(asyncio.get_running_loop())
        hits = await async_search(question)
        r = await async_llm.create(messages=[{"role": "user", "content": f"{hits[0]['text']}\n\nQ: {question}"}])
        return r.choices[0].message.content

    cases = [TestCase(CaseId(f"c{i}"), f"question {i}", "ans", None, "tests.jsonl") for i in (1, 2)]
    store.upsert_cases("tests", cases)
    summary = run_suite(store, "tests:app", app, cases, make_grader(FakeProvider()))
    assert (summary.cases, summary.passed, summary.crashed) == (2, 2, 0)
    traces = [s.trace for s in store.traces(summary.run)]
    assert [t.question for t in traces] == ["question 1", "question 2"]
    for t in traces:
        assert t.outcome == Answered("ans") and t.test is not None and t.test.passed
        assert [e.query for e in t.retrievals] == [t.question] and len(t.llm_calls) == 1
        assert t.in_prompt == ((0, 0),)
    assert len(loops) == 2 and loops[0] is loops[1]


def test_cost_prices_the_same_tokens_the_metrics_report() -> None:
    """Review X6. Defect: with no provider usage the token totals use the chars/4 estimate but the
    cost prices zero tokens, so a priced model reports $0.0000. A dated model id
    (gpt-4.1-mini-2025-04-14) must also find its price."""
    run = capture.ActiveRun("q", NO_INTERVENTION)
    run.llm_calls.append(LLMCall("openai", "gpt-4.1-mini-2025-04-14", "x" * 4000, "y" * 400, None, None, 1.0))
    trace = capture.finalize_trace(run, Answered("y"), 1.0)
    assert (trace.metrics.input_tokens, trace.metrics.output_tokens) == (1000, 100)
    assert trace.metrics.cost_usd == pytest.approx((1000 * 0.40 + 100 * 1.60) / 1_000_000)
    assert cost_usd("claude-haiku-4-5-20251001", 1000, 0) == pytest.approx(1000 * 1.00 / 1_000_000)
    assert cost_usd("gpt-9-2030-01-01", 1, 1) is None
