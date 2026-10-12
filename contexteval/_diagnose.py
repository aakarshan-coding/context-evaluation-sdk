"""The diagnoser: ground-truth location, flaky rerun, replay ladder, judge fallback.

The decision logic is two tables and one pure function. diagnose_case only sequences calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Mapping, Sequence

from . import _text as text
from ._intervention import NO_INTERVENTION, GoldToTop, InjectEvidence, Intervention, OnlyGold, apply
from ._llm import Estimate, LLMRequest, Provider, Task, estimate
from ._runner import App, Grader, estimate_grading, run_case
from ._store import Store, StoredTrace
from .types import (
    Answered,
    Chunk,
    ChunkRef,
    Crashed,
    Diagnosis,
    Evidence,
    Flaky,
    Ignored,
    Irrelevant,
    Judged,
    JudgeVerdict,
    Missing,
    NotContext,
    ReplayOutcome,
    Replays,
    ReplayStep,
    RunId,
    TestCase,
    Trace,
)


class GoldLocation(Enum):
    NOT_RETRIEVED = "not_retrieved"  # no retrieved or delivered chunk is gold
    DROPPED = "dropped"  # a gold chunk was retrieved or delivered, but none reached the prompt
    IN_PROMPT = "in_prompt"  # a gold delivered chunk is in the prompt


@dataclass(frozen=True)
class GoldMap:
    """Where the gold sat in a baseline trace. The chunk refs are evidence for the variants
    (architecture graft 1): Ignored reports `gold`, Irrelevant reports `other`."""

    location: GoldLocation
    gold: tuple[ChunkRef, ...]  # in-prompt delivered chunks holding the evidence
    other: tuple[ChunkRef, ...]  # in-prompt delivered chunks without it
    # Ladder steps whose transform leaves every retrieval's delivered chunks as they are. Such a
    # replay is a second baseline rerun, so a pass would be chance; diagnose_case skips it.
    unchanged: frozenset[ReplayStep] = frozenset()
    # The gold was found only by joining two adjacent chunks (text.gold_split): the chunker cut
    # the evidence in two. Carried onto the diagnosis as `split_gold`.
    split: bool = False


def _gold_refs(per_retrieval: Sequence[tuple[Chunk, ...]], evidence: Evidence) -> set[ChunkRef]:
    return {
        (r, c)
        for r, chunks in enumerate(per_retrieval)
        for c, is_gold in enumerate(text.gold_flags([chunk.text for chunk in chunks], evidence))
        if is_gold
    }


def _split(per_retrieval: Sequence[tuple[Chunk, ...]], evidence: Evidence) -> bool:
    """Every retrieval that holds the gold holds it only by a join, and at least one does."""
    verdicts = [
        text.gold_split(texts, evidence)
        for texts in ([chunk.text for chunk in chunks] for chunks in per_retrieval)
        if any(text.gold_flags(texts, evidence))
    ]
    return bool(verdicts) and all(verdicts)


def locate_gold(evidence: Evidence, trace: Trace) -> GoldMap:
    """Pure. Gold is decided per chunk by text.gold_flags, the rule the replay transforms use, and
    never from the prompt text, which also holds the question.

    IN_PROMPT when a gold delivered chunk is in the prompt. DROPPED when gold was delivered but the
    app left it out, or retrieved but cut by the run's setting before user code saw it. `split`
    is read from the chunk list the gold was found in.
    """
    delivered = [e.delivered for e in trace.retrievals]
    delivered_gold = _gold_refs(delivered, evidence)
    gold = tuple(ref for ref in trace.in_prompt if ref in delivered_gold)
    if gold:
        other = tuple(ref for ref in trace.in_prompt if ref not in delivered_gold)
        unchanged = frozenset(
            step
            for step, _ in LADDER
            if all(apply(REPLAY_INTERVENTION[step](evidence), e.delivered, e.query) == e.delivered for e in trace.retrievals)
        )
        return GoldMap(GoldLocation.IN_PROMPT, gold, other, unchanged, split=_split(delivered, evidence))
    if delivered_gold:
        return GoldMap(GoldLocation.DROPPED, (), (), split=_split(delivered, evidence))
    retrieved = [e.retrieved for e in trace.retrievals]
    if _gold_refs(retrieved, evidence):
        return GoldMap(GoldLocation.DROPPED, (), (), split=_split(retrieved, evidence))
    return GoldMap(GoldLocation.NOT_RETRIEVED, (), ())


# --- tables -----------------------------------------------------------------

# How each replay step changes context. One row per step; the same Intervention type
# experiments use (plan: "shared by diagnosis replay and experiments").
REPLAY_INTERVENTION: Mapping[ReplayStep, Callable[[Evidence], Intervention]] = {
    ReplayStep.BASELINE_RERUN: lambda e: Intervention(),
    ReplayStep.INJECT_EVIDENCE: lambda e: Intervention(transforms=(InjectEvidence(e),)),
    ReplayStep.GOLD_TO_TOP: lambda e: Intervention(transforms=(GoldToTop(e),)),
    ReplayStep.ONLY_GOLD: lambda e: Intervention(transforms=(OnlyGold(e),)),
}

Builder = Callable[[Replays, GoldMap], Diagnosis]

# Plan item 5, in priority order: the least invasive fix wins. First passing step decides.
LADDER: tuple[tuple[ReplayStep, Builder], ...] = (
    (ReplayStep.GOLD_TO_TOP, lambda replays, gold: Ignored(replays, gold_at=gold.gold, split_gold=gold.split)),
    (ReplayStep.ONLY_GOLD, lambda replays, gold: Irrelevant(replays, removed=gold.other, split_gold=gold.split)),
)

MISSING_CAUSE: Mapping[GoldLocation, Literal["not_retrieved", "dropped"]] = {
    GoldLocation.NOT_RETRIEVED: "not_retrieved",
    GoldLocation.DROPPED: "dropped",
}

MAX_STEPS = 1 + len(LADDER)  # the rerun plus the full ladder; a missing case stops at 2

# Samples per step. The real-bot audit (examples/realbot/AUDIT.md) found one replay at app
# temperature mislabeled 2 of 5 in-prompt failures; 3 is the smallest odd count with a majority.
DEFAULT_SAMPLES = 3


Replay = Callable[[ReplayStep], ReplayOutcome]  # closure: execute + grade under REPLAY_INTERVENTION[step]


def _majority(step: ReplayStep, replay: Replay, samples: int) -> tuple[list[ReplayOutcome], bool]:
    """Sample `step` until a majority of `samples` is decided, at most `samples` times. `samples`
    is odd (the CLI checks), so a tie cannot happen and the result is the step's verdict."""
    need = samples // 2 + 1
    outcomes: list[ReplayOutcome] = []
    passed = failed = 0
    while passed < need and failed < need:
        outcome = replay(step)
        outcomes.append(outcome)
        if outcome.passed:
            passed += 1
        else:
            failed += 1
    return outcomes, passed >= need


def diagnose_case(gold: GoldMap, replay: Replay, samples: int = DEFAULT_SAMPLES) -> Diagnosis:
    """Pure given `replay`. Tests pass a scripted replay, so the table is tested without an app.

    Order (plan items 3, 4, 5): the flaky rerun first for every case, then the inject
    confirmation for a missing case, then the ladder. Each step draws up to `samples` replays
    and stops once a majority agrees; a step passes when a majority of its samples pass. Steps
    are lazy: a case stops at the first deciding step, so cost is 1 to MAX_STEPS * samples app
    calls. A ladder step in gold.unchanged is not replayed: it would rerun the baseline that
    just failed.
    """
    outcomes, flaky = _majority(ReplayStep.BASELINE_RERUN, replay, samples)
    if flaky:
        return Flaky(tuple(outcomes))
    cause = MISSING_CAUSE.get(gold.location)
    if cause is not None:
        injected, confirmed = _majority(ReplayStep.INJECT_EVIDENCE, replay, samples)
        outcomes += injected
        return Missing(cause, confirmed=confirmed, replays=tuple(outcomes), split_gold=gold.split)
    for step, build in LADDER:
        if step in gold.unchanged:
            continue  # recorded by its absence from the replays; types.explain names it
        drawn, passed = _majority(step, replay, samples)
        outcomes += drawn
        if passed:
            return build(tuple(outcomes), gold)
    return NotContext(tuple(outcomes), split_gold=gold.split)


def make_replay(app: App, case: TestCase, evidence: Evidence, grade: Grader, base: Intervention = NO_INTERVENTION) -> Replay:
    """`evidence` is passed separately (not read from case) so the type proves it exists.
    `base` is the run's own intervention (an experiment setting), so a rerun reproduces the
    conditions the failure was seen under; the replay transform is applied after it."""

    def replay(step: ReplayStep) -> ReplayOutcome:
        extra = REPLAY_INTERVENTION[step](evidence)
        intervention = Intervention(top_k=base.top_k, transforms=(*base.transforms, *extra.transforms))
        trace = run_case(app, case, grade, intervention)
        assert trace.test is not None
        answer = trace.outcome.text if isinstance(trace.outcome, Answered) else None
        return ReplayOutcome(step, trace.test.passed, answer)

    return replay


# --- judge --------------------------------------------------------------------


class JudgeReplyError(Exception):
    """The judge model's reply was not a JSON verdict. Raised at the boundary, never guessed."""


_VERDICTS = ", ".join(v.value for v in JudgeVerdict)
_JUDGE_SYSTEM = (
    "You diagnose why a retrieval-augmented answer failed. You see the question, the chunks the "
    "retriever returned, the prompt sent to the model, and the model's answer. Reply with one JSON "
    'object {"verdict": V, "rationale": R} and nothing else. V is one of: '
    + _VERDICTS
    + ". missing: the chunks lack the information the question needs. irrelevant: the needed "
    "information is present but off-topic chunks derailed the answer. ignored: the information is in "
    "the prompt and the answer contradicts or omits it. not_context: the context is fine and the answer "
    "is still wrong. no_failure: the answer is correct and grounded. R is one sentence."
)
_JUDGE_OUTPUT_TOKENS = 200
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _judge_request(trace: Trace) -> LLMRequest:
    chunks = [chunk.text for event in trace.retrievals for chunk in event.delivered]
    numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(chunks)) or "(none)"
    answer = trace.outcome.text if isinstance(trace.outcome, Answered) else ""
    return LLMRequest(
        task=Task.JUDGE,
        system=_JUDGE_SYSTEM,
        user=f"Question: {trace.question}\n\nRetrieved chunks:\n{numbered}\n\nPrompt sent to the model:\n{trace.prompt_text}\n\nAnswer: {answer}",
        inputs={"question": trace.question, "chunks": "\n".join(chunks), "prompt": trace.prompt_text, "answer": answer},
        max_tokens=_JUDGE_OUTPUT_TOKENS,
    )


def parse_verdict(reply: str) -> Judged:
    """Boundary: model text -> Judged. Tolerates a code fence or prose around the object.
    Anything without a known verdict raises JudgeReplyError."""
    found = _JSON_OBJECT.search(reply)
    try:
        data = json.loads(found.group()) if found else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        raise JudgeReplyError(f"judge reply is not a JSON object: {reply[:200]!r}")
    try:
        verdict = JudgeVerdict(str(data.get("verdict", "")).strip().lower())
    except ValueError:
        raise JudgeReplyError(f"judge verdict {data.get('verdict')!r} is not one of {_VERDICTS}") from None
    return Judged(verdict, str(data.get("rationale", "")).strip())


def judge(trace: Trace, provider: Provider) -> Judged:
    """No ground truth: send question, delivered chunks, prompt, answer. Parse JSON
    {"verdict": ..., "rationale": ...}; an unparseable reply raises (boundary), not a guess.
    """
    return parse_verdict(provider.complete(_judge_request(trace)).text)


# --- a run ----------------------------------------------------------------------


# How diagnose_run treats one pending trace. Skipped paths are counted, never diagnosed:
# a crash has no answer to explain, and an empty production answer (a streamed response the
# hooks could not read) would have the judge label text that was never captured.
PendingPath = Literal["replay", "judge", "crashed", "no_answer"]


@dataclass(frozen=True)
class Pending:
    """One trace still to diagnose. `case` is None for production traces; a case without
    evidence also takes the judge path."""

    stored: StoredTrace
    case: TestCase | None

    @property
    def path(self) -> PendingPath:
        outcome = self.stored.trace.outcome
        if isinstance(outcome, Crashed):
            return "crashed"
        if self.case is not None and self.case.evidence is not None:
            return "replay"
        return "judge" if outcome.text.strip() else "no_answer"


def pending(store: Store, run: RunId, *, force: bool = False) -> list[Pending]:
    """The failed traces of `run` (every trace for a production run) without a diagnosis, or
    all of them with `force`. One list feeds the estimate and the loop, so they agree."""
    production = store.run(run).kind == "production"
    out = []
    for stored in store.traces(run, failed_only=not production):
        if stored.diagnosis is not None and not force:
            continue
        test = stored.trace.test
        out.append(Pending(stored, store.case(test.case_id) if test else None))
    return out


def estimate_diagnosis(provider: Provider, todo: Sequence[Pending], samples: int = DEFAULT_SAMPLES) -> Estimate:
    """SDK calls only (the app's own calls are reported separately): one judge call per
    judge-path trace, and worst case one grade per replay for ground-truth cases, where the
    worst case is every step drawing all `samples`."""
    judged = [_judge_request(p.stored.trace) for p in todo if p.path == "judge"]
    cases = [p.case for p in todo if p.path == "replay" and p.case is not None]
    total = estimate(provider.model, [r.system + r.user for r in judged], _JUDGE_OUTPUT_TOKENS)
    return total + estimate_grading(provider, cases * MAX_STEPS * samples)


@dataclass(frozen=True)
class DiagnoseReport:
    diagnosed: int
    skipped_existing: int
    skipped_crashed: int  # crashed traces are listed as failures but not diagnosed
    skipped_no_answer: int  # judge-path traces with an empty answer (streamed responses)


OnDiagnosis = Callable[[StoredTrace, Diagnosis], None]


def diagnose_run(
    store: Store,
    run: RunId,
    app: App | None,
    grade: Grader,
    provider: Provider,
    *,
    force: bool = False,
    on_diagnosis: OnDiagnosis = lambda stored, diagnosis: None,
    samples: int = DEFAULT_SAMPLES,
) -> DiagnoseReport:
    """For each failed trace in `run` (all traces for production runs):
    - skip if already diagnosed and not force (idempotent reruns resume where a crash stopped)
    - skip and count crashed traces and judge-path traces with no answer text
    - no case / no evidence -> judge
    - else locate_gold, build the Replay closure over (app, case, grade), diagnose_case
    - save_diagnosis per trace as it completes
    `app` is None only for production runs, which are judge-only. `samples` is the replays per
    step diagnose_case may draw.
    """
    info = store.run(run)
    todo = pending(store, run, force=force)
    candidates = store.traces(run, failed_only=info.kind != "production")
    skipped = {"crashed": 0, "no_answer": 0}
    diagnosed = 0
    for p in todo:
        path = p.path
        if path == "crashed" or path == "no_answer":
            skipped[path] += 1
            continue
        if p.case is None or p.case.evidence is None:
            diagnosis: Diagnosis = judge(p.stored.trace, provider)
        else:
            if app is None:
                raise ValueError(f"run {run} has ground-truth cases; diagnosing it needs the app")
            gold = locate_gold(p.case.evidence, p.stored.trace)
            replay = make_replay(app, p.case, p.case.evidence, grade, info.setting.intervention())
            diagnosis = diagnose_case(gold, replay, samples)
        store.save_diagnosis(p.stored.id, diagnosis)
        on_diagnosis(p.stored, diagnosis)
        diagnosed += 1
    return DiagnoseReport(diagnosed, len(candidates) - len(todo), skipped["crashed"], skipped["no_answer"])
