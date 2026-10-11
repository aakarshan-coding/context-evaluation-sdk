"""Capture core: trace finalization, the production buffer and writer, and the active-run slot.
Driven through the public hook surface (intercept_*, @retriever, execute), not internals."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from contexteval import _capture as capture
from contexteval import retriever
from contexteval._intervention import NO_INTERVENTION, InjectEvidence, Intervention, InterventionNotApplicable
from contexteval._llm import cost_usd
from contexteval._runner import execute
from contexteval.types import Answered, Evidence, LLMCall


@dataclass
class Doc:
    text: str


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
