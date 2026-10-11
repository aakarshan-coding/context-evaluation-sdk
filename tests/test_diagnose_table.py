"""The decision table without an app: diagnose_case with a scripted Replay.

Each row is (location, passing steps) -> expected variant and the exact replay sequence. The
defect this catches is a reordered ladder, a missing flaky short-circuit, or a replay issued
after the deciding step, which the acceptance test only sees indirectly.
"""

from __future__ import annotations

import pytest

from contexteval._diagnose import DEFAULT_SAMPLES, LADDER, MAX_STEPS, GoldLocation, GoldMap, Replay, diagnose_case
from contexteval.types import (
    LADDER_STEPS,
    Flaky,
    Ignored,
    Irrelevant,
    Missing,
    NotContext,
    ReplayOutcome,
    ReplayStep,
    StepTally,
    explain,
    tallies,
)

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
    """One row of plan items 3 to 5, at `--samples 1` (v1 behavior: one replay per step decides).
    Asserts the variant, its stored replays, the chunk refs it carries, and the exact sequence of
    steps requested (laziness: no step after the decider; row 2 proves the flaky rerun
    short-circuits a missing case before any inject replay)."""
    requested: list[ReplayStep] = []

    def replay(step: ReplayStep) -> ReplayOutcome:
        requested.append(step)
        return ReplayOutcome(step, step in passing, "scripted")

    gold = GoldMap(location, GOLD_AT if location is GoldLocation.IN_PROMPT else (), OTHER if location is GoldLocation.IN_PROMPT else ())
    diagnosis = diagnose_case(gold, replay, samples=1)
    assert requested == steps
    assert diagnosis == expected(_replays(steps, passing))


def test_ladder_prefers_the_least_invasive_fix() -> None:
    """Defect: the ladder rows swapped, so a case fixed by both reordering and pruning is
    labeled irrelevant (prune) instead of ignored (reorder, plan item 5's first row)."""
    assert [step for step, _ in LADDER] == [TO_TOP, ONLY] == list(LADDER_STEPS)  # explain() reads LADDER_STEPS
    assert MAX_STEPS == 3


# --- majority sampling (v1.1 A) -------------------------------------------------

IN_PROMPT = GoldMap(GoldLocation.IN_PROMPT, GOLD_AT, OTHER)


def _scripted(script: dict[ReplayStep, list[bool]]) -> tuple[list[ReplayStep], Replay]:
    """A Replay that answers each step from its script in order (a step not in the script fails)
    and records every step requested. A script that runs out raises, which catches a step sampled
    more often than the test allows."""
    requested: list[ReplayStep] = []
    queues = {step: iter(outcomes) for step, outcomes in script.items()}

    def replay(step: ReplayStep) -> ReplayOutcome:
        requested.append(step)
        passed = next(queues[step]) if step in queues else False
        return ReplayOutcome(step, passed, "scripted")

    return requested, replay


def test_default_samples_is_three_and_a_lucky_pass_no_longer_decides() -> None:
    """Audit trace 535: gold to top passed once and the case was labeled ignored, while five
    more replays passed it 2 of 5. Defect: a single replay per step decides the label, so one
    lucky pass at app temperature becomes a label. With the default 3 samples the step is
    sampled until a majority is decided (1 of 3 here), and the ladder moves on to only gold."""
    assert DEFAULT_SAMPLES == 3
    requested, replay = _scripted({TO_TOP: [True, False, False], ONLY: [True, True]})
    diagnosis = diagnose_case(IN_PROMPT, replay)
    assert requested == [RERUN, RERUN, TO_TOP, TO_TOP, TO_TOP, ONLY, ONLY]
    assert isinstance(diagnosis, Irrelevant)
    assert tallies(diagnosis.replays) == (StepTally(RERUN, 0, 2), StepTally(TO_TOP, 1, 3), StepTally(ONLY, 2, 2))
    assert explain(diagnosis) == "keeping only the gold chunks fixed it (2/2); removed chunks 1, 2, 4 of retrieval 1"


def test_samples_one_reproduces_v1_where_the_first_replay_decides() -> None:
    """`--samples 1` must behave as v1 did, so a user who wants the cheap mode gets exactly it.
    Defect: the sampler always draws a second sample, or treats 1 as 3. The same lucky script
    as above now yields ignored after one replay per step."""
    requested, replay = _scripted({TO_TOP: [True, False, False], ONLY: [True, True]})
    diagnosis = diagnose_case(IN_PROMPT, replay, samples=1)
    assert requested == [RERUN, TO_TOP]
    assert diagnosis == Ignored((ReplayOutcome(RERUN, False, "scripted"), ReplayOutcome(TO_TOP, True, "scripted")), gold_at=GOLD_AT)


def test_a_step_stops_sampling_once_two_of_three_agree() -> None:
    """Defect: every step always draws all N samples, tripling the app cost even when the first
    two samples already decide. Two failing reruns end the rerun, two passing gold-to-top replays
    end the ladder, and the explain line reports 2/2, not 2/3."""
    requested, replay = _scripted({TO_TOP: [True, True, True]})
    diagnosis = diagnose_case(IN_PROMPT, replay)
    assert requested == [RERUN, RERUN, TO_TOP, TO_TOP]
    assert isinstance(diagnosis, Ignored)
    assert explain(diagnosis) == "moving the gold chunk to the top fixed it (2/2); gold was at chunk 3 of retrieval 1"


@pytest.mark.parametrize(
    "samples,reruns,flaky,drawn",
    [
        (3, [True, False, False], False, 3),
        (3, [True, False, True], True, 3),
        (3, [True, True, False], True, 2),
        (5, [False, True, True, False, True], True, 5),
        (5, [True, False, False, True, False], False, 5),
    ],
    ids=["1-of-3-is-not-flaky", "2-of-3-is-flaky", "2-of-2-stops-early", "3-of-5-is-flaky", "2-of-5-is-not-flaky"],
)
def test_flaky_needs_a_majority_of_baseline_reruns(samples: int, reruns: list[bool], flaky: bool, drawn: int) -> None:
    """Audit trace 461: one rerun passed and the case was labeled flaky, while five more reruns
    all failed. Defect: the first passing rerun decides flaky. A case is flaky when a majority of
    up to N reruns pass; otherwise the ladder runs and the rerun samples stay in the record."""
    requested, replay = _scripted({RERUN: reruns, TO_TOP: [True] * samples})
    diagnosis = diagnose_case(IN_PROMPT, replay, samples=samples)
    assert requested[:drawn] == [RERUN] * drawn
    assert isinstance(diagnosis, Flaky) == flaky
    assert tallies(diagnosis.replays)[0] == StepTally(RERUN, sum(reruns[:drawn]), drawn)
    if flaky:
        assert requested == [RERUN] * drawn
        assert explain(diagnosis) == f"the unchanged baseline passed on rerun ({sum(reruns[:drawn])}/{drawn})"
    else:
        assert requested.count(TO_TOP) == samples // 2 + 1


def test_missing_confirmation_takes_the_majority_of_inject_replays() -> None:
    """Defect: `confirmed` is read from the first inject replay, so a model that answers the
    injected evidence correctly 1 time in 3 reports an unconfirmed missing label as confirmed."""
    requested, replay = _scripted({INJECT: [True, False, False]})
    diagnosis = diagnose_case(GoldMap(GoldLocation.NOT_RETRIEVED, (), ()), replay)
    assert requested == [RERUN, RERUN, INJECT, INJECT, INJECT]
    assert diagnosis == Missing("not_retrieved", False, diagnosis.replays)
    assert explain(diagnosis) == "the gold chunk was never retrieved; injecting the evidence did not fix it (1/3)"


def test_not_context_names_the_count_of_every_step_it_tried() -> None:
    """Defect: a not_context line hides how close each step came, so a 1/3 on only gold reads
    the same as 0/2."""
    requested, replay = _scripted({ONLY: [False, True, False]})
    diagnosis = diagnose_case(IN_PROMPT, replay)
    assert requested == [RERUN, RERUN, TO_TOP, TO_TOP, ONLY, ONLY, ONLY]
    assert explain(diagnosis) == "neither moving gold to the top (0/2) nor keeping only gold (1/3) fixed it"


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

    diagnosis = diagnose_case(GoldMap(GoldLocation.IN_PROMPT, GOLD_AT, OTHER, frozenset(unchanged)), replay, samples=1)
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
        "keeping only gold (0/1) did not fix it; moving gold to the top was skipped because gold was already at the top"
    )
    assert explain(Irrelevant(rerun + (ReplayOutcome(ONLY, True, "x"),), removed=OTHER)) == (
        "keeping only the gold chunks fixed it (1/1); removed chunks 1, 2, 4 of retrieval 1; "
        "moving gold to the top was skipped because gold was already at the top"
    )
    full = rerun + (ReplayOutcome(TO_TOP, False, "x"), ReplayOutcome(ONLY, False, "x"))
    assert explain(NotContext(full)) == "neither moving gold to the top (0/1) nor keeping only gold (0/1) fixed it"


def test_positions_are_numbered_from_one_in_user_facing_text() -> None:
    """Audit finding: "gold was at chunk 5" meant the sixth chunk, because the line printed the
    stored 0-based index. Defect: the explain line (and so the CLI and dashboard) prints stored
    indexes. Stored refs stay 0-based; only the text counts from 1."""
    d = Ignored((ReplayOutcome(RERUN, False, "x"), ReplayOutcome(TO_TOP, True, "x")), gold_at=((0, 0), (1, 5)))
    assert d.gold_at == ((0, 0), (1, 5))
    assert explain(d) == "moving the gold chunk to the top fixed it (1/1); gold was at chunk 1 of retrieval 1; chunk 6 of retrieval 2"
