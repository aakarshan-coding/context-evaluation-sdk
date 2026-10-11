"""Capture core: trace finalization, the production buffer and writer, and the active-run slot.
Driven through the public hook surface (intercept_*, @retriever, execute), not internals."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from contexteval import _capture as capture
from contexteval import _hooks, retriever
from contexteval._intervention import NO_INTERVENTION, InjectEvidence, Intervention, InterventionNotApplicable
from contexteval._llm import cost_usd
from contexteval._runner import execute
from contexteval.types import Answered, Evidence, LLMCall


@dataclass
class Doc:
    text: str


class FakeLLM:
    """Stand-in for openai Completions, hooked with the real OpenAIChatAdapter."""

    def create(self, *, messages: list[dict[str, str]], model: str = "gpt-4.1-mini") -> Any:
        if "fail" in messages[-1]["content"]:
            raise RuntimeError("upstream 500")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ans"))], usage=None)


@retriever
def search(query: str, top_k: int = 3) -> list[dict[str, str]]:
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
def flushed() -> Iterator[list[capture.ActiveRun]]:
    """Production mode whose sink appends each closed buffer instead of writing it."""
    runs: list[capture.ActiveRun] = []
    capture.set_mode(capture.Mode.PRODUCTION, lambda: runs.append)
    try:
        yield runs
    finally:
        capture.reset_process()


def _ask(llm: FakeLLM, question: str, context: str = "") -> Any:
    return llm.create(messages=[{"role": "user", "content": f"{context}\n\nQ: {question}"}])


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
