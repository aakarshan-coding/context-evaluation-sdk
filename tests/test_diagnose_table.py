"""The decision table without an app: diagnose_case with a scripted Replay.

Each row is (location, passing steps) -> expected variant and the exact replay sequence. The
defect this catches is a reordered ladder, a missing flaky short-circuit, or a replay issued
after the deciding step, which the acceptance test only sees indirectly.
"""

from __future__ import annotations

import pytest

from contexteval._diagnose import LADDER, MAX_REPLAYS, GoldLocation, GoldMap, diagnose_case
from contexteval.types import Flaky, Ignored, Irrelevant, Missing, NotContext, LADDER_STEPS, ReplayOutcome, ReplayStep, explain

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
    assert [step for step, _ in LADDER] == [TO_TOP, ONLY] == list(LADDER_STEPS)  # explain() reads LADDER_STEPS
    assert MAX_REPLAYS == 3


@pytest.mark.parametrize(
    "unchanged,passing,steps,expected",
    [
        ({TO_TOP}, {TO_TOP, ONLY}, [RERUN, ONLY], lambda r: Irrelevant(r, removed=OTHER)),
        ({TO_TOP}, {TO_TOP}, [RERUN, ONLY], lambda r: NotContext(r)),
        ({TO_TOP, ONLY}, {TO_TOP, ONLY}, [RERUN], lambda r: NotContext(r)),
    ],
    ids=["gold-already-first", "gold-already-first-chance-pass", "gold-is-the-only-chunk"],
)
def test_ladder_skips_a_step_that_would_not_change_the_context(unchanged, passing, steps, expected) -> None:
    """Review Y3. When gold already sits first, GoldToTop replays the baseline unchanged, so a pass
    is chance (the flaky rerun just failed), not proof the order mattered. Defect: the step runs
    anyway and a chance pass becomes `ignored` (or, with gold the only chunk, OnlyGold's chance
    pass becomes `irrelevant`). The step must not be requested, and its absence from the stored
    replays is the record of the skip."""
    requested: list[ReplayStep] = []

    def replay(step: ReplayStep) -> ReplayOutcome:
        requested.append(step)
        return ReplayOutcome(step, step in passing, "scripted")

    diagnosis = diagnose_case(GoldMap(GoldLocation.IN_PROMPT, GOLD_AT, OTHER, frozenset(unchanged)), replay)
    assert requested == steps
    assert diagnosis == expected(_replays(steps, passing))


def test_skipped_steps_are_named_in_the_explain_line() -> None:
    """Review Y3. Defect: a diagnosis whose ladder skipped a step still says "neither moving gold to
    the top nor keeping only gold fixed it", claiming a replay that never ran."""
    rerun = (ReplayOutcome(RERUN, False, "x"),)
    only = (ReplayOutcome(ONLY, False, "x"),)
    assert explain(NotContext(rerun)) == (
        "gold was already the only context at the top, so moving it to the top and keeping only gold were skipped"
    )
    assert explain(NotContext(rerun + only)) == (
        "keeping only gold did not fix it; moving gold to the top was skipped because gold was already at the top"
    )
    assert explain(Irrelevant(rerun + (ReplayOutcome(ONLY, True, "x"),), removed=OTHER)) == (
        "keeping only the gold chunks fixed it; removed chunks 0, 1, 3 of retrieval 0; "
        "moving gold to the top was skipped because gold was already at the top"
    )
    full = rerun + (ReplayOutcome(TO_TOP, False, "x"), ReplayOutcome(ONLY, False, "x"))
    assert explain(NotContext(full)) == "neither moving gold to the top nor keeping only gold fixed it"
