"""Capture core: trace finalization, the production buffer and writer, and the active-run slot.
Driven through the public hook surface (intercept_*, @retriever, execute), not internals."""

from __future__ import annotations

import pytest

from contexteval import _capture as capture
from contexteval._intervention import NO_INTERVENTION
from contexteval._llm import cost_usd
from contexteval.types import Answered, LLMCall


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
