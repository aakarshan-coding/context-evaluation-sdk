"""The decision table without an app: diagnose_case with a scripted Replay.

Each row is (location, passing steps) -> expected variant and the exact replay sequence. The
defect this catches is a reordered ladder, a missing flaky short-circuit, or a replay issued
after the deciding step, which the acceptance test only sees indirectly.
"""

from __future__ import annotations

import pytest

from contexteval._diagnose import LADDER, MAX_REPLAYS, GoldLocation, GoldMap, diagnose_case
from contexteval.types import Flaky, Ignored, Irrelevant, Missing, NotContext, ReplayOutcome, ReplayStep

GOLD_AT = ((0, 2),)
OTHER = ((0, 0), (0, 1), (0, 3))

RERUN, INJECT, TO_TOP, ONLY = ReplayStep.BASELINE_RERUN, ReplayStep.INJECT_EVIDENCE, ReplayStep.GOLD_TO_TOP, ReplayStep.ONLY_GOLD


def _replays(steps: list[ReplayStep], passing: set[ReplayStep]) -> tuple[ReplayOutcome, ...]:
    return tuple(ReplayOutcome(s, s in passing, "scripted") for s in steps)


# (location, passing steps, expected steps requested in order, expected diagnosis given those outcomes)
ROWS = [
    (GoldLocation.IN_PROMPT, {RERUN}, [RERUN], lambda r: Flaky(r)),
    (GoldLocation.NOT_RETRIEVED, {RERUN, INJECT}, [RERUN], lambda r: Flaky(r)),
    (GoldLocation.DROPPED, {RERUN}, [RERUN], lambda r: Flaky(r)),
    (GoldLocation.IN_PROMPT, {TO_TOP, ONLY}, [RERUN, TO_TOP], lambda r: Ignored(r, gold_at=GOLD_AT)),
    (GoldLocation.IN_PROMPT, {ONLY}, [RERUN, TO_TOP, ONLY], lambda r: Irrelevant(r, removed=OTHER)),
    (GoldLocation.IN_PROMPT, set(), [RERUN, TO_TOP, ONLY], lambda r: NotContext(r)),
    (GoldLocation.NOT_RETRIEVED, {INJECT}, [RERUN, INJECT], lambda r: Missing("not_retrieved", True, r)),
    (GoldLocation.NOT_RETRIEVED, set(), [RERUN, INJECT], lambda r: Missing("not_retrieved", False, r)),
    (GoldLocation.DROPPED, {INJECT}, [RERUN, INJECT], lambda r: Missing("dropped", True, r)),
    (GoldLocation.DROPPED, {TO_TOP, ONLY}, [RERUN, INJECT], lambda r: Missing("dropped", False, r)),
]

IDS = [f"{loc.value}-pass[{'+'.join(sorted(s.value for s in passing)) or 'none'}]" for loc, passing, _, _ in ROWS]


@pytest.mark.parametrize("location,passing,steps,expected", ROWS, ids=IDS)
def test_decision_table(location, passing, steps, expected) -> None:
    """One row of plan items 3 to 5. Asserts the variant, its stored replays, the chunk refs it
    carries, and the exact sequence of steps requested (laziness: no step after the decider;
    row 2 proves the flaky rerun short-circuits a missing case before any inject replay)."""
    requested: list[ReplayStep] = []

    def replay(step: ReplayStep) -> ReplayOutcome:
        requested.append(step)
        return ReplayOutcome(step, step in passing, "scripted")

    gold = GoldMap(location, GOLD_AT if location is GoldLocation.IN_PROMPT else (), OTHER if location is GoldLocation.IN_PROMPT else ())
    diagnosis = diagnose_case(gold, replay)
    assert requested == steps
    assert diagnosis == expected(_replays(steps, passing))


def test_ladder_prefers_the_least_invasive_fix() -> None:
    """Defect: the ladder rows swapped, so a case fixed by both reordering and pruning is
    labeled irrelevant (prune) instead of ignored (reorder, plan item 5's first row)."""
    assert [step for step, _ in LADDER] == [TO_TOP, ONLY]
    assert MAX_REPLAYS == 3
