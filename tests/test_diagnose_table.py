"""The decision table without an app: diagnose_case with a scripted Replay.

Each row is (location, passing steps) -> expected variant. The defect this catches is a
reordered ladder or a missing flaky short-circuit, which the acceptance test only sees
indirectly.
"""

from __future__ import annotations

import pytest

from contexteval._diagnose import GoldLocation
from contexteval.types import ReplayStep

pytestmark = pytest.mark.skip(reason="P3: diagnose_case is not implemented yet")

ROWS = [
    (GoldLocation.IN_PROMPT, {ReplayStep.BASELINE_RERUN}, "Flaky"),
    (GoldLocation.NOT_RETRIEVED, {ReplayStep.BASELINE_RERUN}, "Flaky"),
    (GoldLocation.IN_PROMPT, {ReplayStep.GOLD_TO_TOP, ReplayStep.ONLY_GOLD}, "Ignored"),
    (GoldLocation.IN_PROMPT, {ReplayStep.ONLY_GOLD}, "Irrelevant"),
    (GoldLocation.IN_PROMPT, set(), "NotContext"),
    (GoldLocation.NOT_RETRIEVED, {ReplayStep.INJECT_EVIDENCE}, "Missing(not_retrieved, confirmed)"),
    (GoldLocation.DROPPED, set(), "Missing(dropped, unconfirmed)"),
]


@pytest.mark.parametrize("location,passing,expected", ROWS)
def test_decision_table(location, passing, expected) -> None:
    """Also asserts the exact sequence of steps requested (laziness: no step after the decider)."""
    raise NotImplementedError
