"""The diagnoser: ground-truth location, flaky rerun, replay ladder, judge fallback.

The decision logic is two tables and one pure function. diagnose_case only sequences calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Mapping

from ._intervention import GoldToTop, InjectEvidence, Intervention, OnlyGold
from ._llm import Provider
from ._store import Store
from .types import (
    Diagnosis,
    Evidence,
    Flaky,
    Ignored,
    Irrelevant,
    Judged,
    Missing,
    NotContext,
    ReplayOutcome,
    ReplayStep,
    RunId,
    TestCase,
    Trace,
)
from ._runner import App, Grader


class GoldLocation(Enum):
    NOT_RETRIEVED = "not_retrieved"  # no delivered chunk in any retrieval event contains gold
    DROPPED = "dropped"  # a delivered chunk contains gold, the prompt does not
    IN_PROMPT = "in_prompt"  # the prompt contains gold


def locate_gold(evidence: Evidence, trace: Trace) -> GoldLocation:
    """Pure. Checks the prompt first: gold in the prompt is IN_PROMPT whatever the chunks say."""
    raise NotImplementedError


# --- tables -----------------------------------------------------------------

# How each replay step changes context. One row per step; the same Intervention type
# experiments use (plan: "shared by diagnosis replay and experiments").
REPLAY_INTERVENTION: Mapping[ReplayStep, Callable[[Evidence], Intervention]] = {
    ReplayStep.BASELINE_RERUN: lambda e: Intervention(),
    ReplayStep.INJECT_EVIDENCE: lambda e: Intervention(transforms=(InjectEvidence(e),)),
    ReplayStep.GOLD_TO_TOP: lambda e: Intervention(transforms=(GoldToTop(e),)),
    ReplayStep.ONLY_GOLD: lambda e: Intervention(transforms=(OnlyGold(e),)),
}

# Plan item 5, in priority order: the least invasive fix wins. First passing step decides.
LADDER: tuple[tuple[ReplayStep, type[Ignored] | type[Irrelevant]], ...] = (
    (ReplayStep.GOLD_TO_TOP, Ignored),
    (ReplayStep.ONLY_GOLD, Irrelevant),
)

MISSING_CAUSE: Mapping[GoldLocation, Literal["not_retrieved", "dropped"]] = {
    GoldLocation.NOT_RETRIEVED: "not_retrieved",
    GoldLocation.DROPPED: "dropped",
}


Replay = Callable[[ReplayStep], ReplayOutcome]  # closure: execute + grade under REPLAY_INTERVENTION[step]


def diagnose_case(location: GoldLocation, replay: Replay) -> Diagnosis:
    """Pure given `replay`. Tests pass a scripted replay, so the table is tested without an app.

    Order (plan items 3, 4, 5):
      rerun = replay(BASELINE_RERUN); if rerun.passed -> Flaky((rerun,))
      if location in MISSING_CAUSE:
          inj = replay(INJECT_EVIDENCE) -> Missing(cause, confirmed=inj.passed, (rerun, inj))
      for step, variant in LADDER: o = replay(step); if o.passed -> variant(all outcomes so far)
      -> NotContext(all outcomes)
    Replays are lazy: a case stops at the first deciding step, so cost is 2 to 4 app calls.
    """
    raise NotImplementedError


def judge(trace: Trace, provider: Provider) -> Judged:
    """No ground truth: send question, delivered chunks, prompt, answer. Parse JSON
    {"verdict": ..., "rationale": ...}; an unparseable reply raises (boundary), not a guess.
    """
    raise NotImplementedError


@dataclass(frozen=True)
class DiagnoseReport:
    diagnosed: int
    skipped_existing: int
    skipped_crashed: int  # crashed traces are listed as failures but not diagnosed


def diagnose_run(store: Store, run: RunId, app: App | None, grade: Grader, provider: Provider, *, force: bool = False) -> DiagnoseReport:
    """For each failed trace in `run` (all traces for production runs):
    - skip if already diagnosed and not force (idempotent reruns resume where a crash stopped)
    - no case / no evidence -> judge
    - else locate_gold, build the Replay closure over (app, case, grade), diagnose_case
    - save_diagnosis per trace as it completes
    `app` is None only for production runs, which are judge-only.
    """
    raise NotImplementedError


def make_replay(app: App, case: TestCase, evidence: Evidence, grade: Grader) -> Replay:
    """`evidence` is passed separately (not read from case) so the type proves it exists."""
    raise NotImplementedError  # step -> runner.run_case(app, case, grade, REPLAY_INTERVENTION[step](evidence))
