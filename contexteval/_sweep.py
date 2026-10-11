"""The diagnosis-guided sweep (v2): labels in, one plain recommendation out.

The policy is a table from failure kind to candidate settings. The search is greedy over
rounds on a tune split and confirms the winner on a holdout split. Every run goes through
_runner.run_suite under one experiment, so a sweep is an experiment the dashboard already
shows, marked `sweep` and carrying its report.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Iterable, Literal, Mapping, Sequence, Union

from ._intervention import BASELINE, Order, Setting
from ._runner import App, Grader, run_suite
from ._store import RunInfo, Store
from .types import (
    CaseId,
    Diagnosis,
    ExperimentId,
    ExperimentRow,
    Flaky,
    Ignored,
    Irrelevant,
    Judged,
    JudgeVerdict,
    Missing,
    NotContext,
    RunId,
    TestCase,
    never,
)


class FailureKind(Enum):
    """A label, split by cause where the cause decides the fix. The value is what the report prints."""

    NOT_RETRIEVED = "missing/not_retrieved"
    DROPPED = "missing/dropped"
    IRRELEVANT = "irrelevant"
    IGNORED = "ignored"
    NOT_CONTEXT = "not_context"
    FLAKY = "flaky"


def kind_of(d: Diagnosis) -> FailureKind | None:
    """None for a judge verdict of no failure. A judged `missing` has no cause, so it takes the
    retrieval candidates: a larger top_k is the only setting that can help it."""
    match d:
        case Missing(cause="not_retrieved"):
            return FailureKind.NOT_RETRIEVED
        case Missing():
            return FailureKind.DROPPED
        case Irrelevant():
            return FailureKind.IRRELEVANT
        case Ignored():
            return FailureKind.IGNORED
        case NotContext():
            return FailureKind.NOT_CONTEXT
        case Flaky():
            return FailureKind.FLAKY
        case Judged(verdict=JudgeVerdict.NO_FAILURE):
            return None
        case Judged(verdict=v):
            return {
                JudgeVerdict.MISSING: FailureKind.NOT_RETRIEVED,
                JudgeVerdict.IRRELEVANT: FailureKind.IRRELEVANT,
                JudgeVerdict.IGNORED: FailureKind.IGNORED,
                JudgeVerdict.NOT_CONTEXT: FailureKind.NOT_CONTEXT,
            }[v]
        case _:
            never(d)


# --- policy -----------------------------------------------------------------------


@dataclass(frozen=True)
class Scaled:
    """An integer knob set relative to k, the most chunks one retrieval returned in the baseline
    run: top_k=2k widens retrieval, max_chunks=k/2 narrows the prompt."""

    knob: Literal["top_k", "max_chunks"]
    factor: float

    def resolve(self, k: int) -> Setting | None:
        """None when the value would change nothing: a max_chunks at or above k, or top_k == k."""
        value = max(1, round(k * self.factor))
        if (self.knob == "max_chunks" and value >= k) or (self.knob == "top_k" and value == k):
            return None
        return Setting(**{self.knob: value})  # type: ignore[arg-type]


Candidate = Union[Setting, Scaled]


@dataclass(frozen=True)
class Fixable:
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class Unfixable:
    reason: str  # shown in the report; no setting is tried for this kind


# The label-to-candidates policy (design-notes "diagnosis-guided search"). Table order inside a
# kind is the tie-break between candidates of equal weight.
POLICY: Mapping[FailureKind, Fixable | Unfixable] = {
    FailureKind.NOT_RETRIEVED: Fixable((Scaled("top_k", 2), Scaled("top_k", 3))),
    FailureKind.DROPPED: Unfixable(
        "retrieval found the gold chunk and your own code removed it before the prompt; "
        "no retrieval setting changes that filter"
    ),
    FailureKind.IRRELEVANT: Fixable((Setting(rerank="lexical"), Scaled("max_chunks", 0.5), Setting(compress=2))),
    FailureKind.IGNORED: Fixable((Setting(order=Order.ENDS), Setting(order=Order.REVERSE), Setting(compress=2))),
    FailureKind.NOT_CONTEXT: Unfixable(
        "the context held the answer and the model still got it wrong; look at the prompt, the model, or the expected answer"
    ),
    FailureKind.FLAKY: Unfixable("the unchanged app passed on rerun; a setting cannot make the model deterministic"),
}


def candidates_for(kind: FailureKind, k: int) -> list[Setting]:
    rule = POLICY[kind]
    if isinstance(rule, Unfixable):
        return []
    settings = []
    for candidate in rule.candidates:
        setting = candidate.resolve(k) if isinstance(candidate, Scaled) else candidate
        if setting is not None and setting != BASELINE and setting not in settings:
            settings.append(setting)
    return settings


@dataclass(frozen=True)
class Ranked:
    setting: Setting
    weight: int  # failures whose label proposes this setting
    kinds: tuple[FailureKind, ...]


def rank_candidates(counts: Mapping[FailureKind, int], k: int) -> list[Ranked]:
    """Heaviest first; a setting two labels propose carries both counts. Ties keep policy order."""
    weight: dict[Setting, int] = {}
    kinds: dict[Setting, list[FailureKind]] = {}
    for kind in FailureKind:
        n = counts.get(kind, 0)
        if n == 0:
            continue
        for setting in candidates_for(kind, k):
            weight[setting] = weight.get(setting, 0) + n
            kinds.setdefault(setting, []).append(kind)
    ranked = [Ranked(s, weight[s], tuple(kinds[s])) for s in weight]
    return sorted(ranked, key=lambda r: -r.weight)


def knobs(setting: Setting) -> frozenset[str]:
    """The knobs a setting changes from the baseline."""
    return frozenset(f.name for f in fields(setting) if getattr(setting, f.name) != f.default)


def merge(a: Setting, b: Setting) -> Setting:
    """a with b's changed knobs applied. Callers combine settings with disjoint knobs only."""
    values = {f.name: getattr(a, f.name) for f in fields(a)}
    values.update({name: getattr(b, name) for name in knobs(b)})
    return Setting(**values)  # type: ignore[arg-type]


# --- split ------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    seed: int
    holdout_fraction: float
    tune: tuple[CaseId, ...]
    holdout: tuple[CaseId, ...]


def split_cases(cases: Sequence[TestCase], failed: Iterable[CaseId], holdout: float, seed: int) -> Split:
    """Seeded and stratified: the baseline's failures and passes are each shuffled and split, so
    both sides hold failures the candidates can fix. Holdout counts are rounded per stratum, so
    a tiny set can end with an empty holdout; the caller warns about that."""
    failed = set(failed)
    rng = random.Random(seed)
    tune: list[CaseId] = []
    held: list[CaseId] = []
    for stratum in ({c.id for c in cases if c.id in failed}, {c.id for c in cases if c.id not in failed}):
        ids = sorted(stratum)
        rng.shuffle(ids)
        n = round(holdout * len(ids))
        held += ids[:n]
        tune += ids[n:]
    return Split(seed, holdout, tuple(sorted(tune)), tuple(sorted(held)))


# --- plan -------------------------------------------------------------------------


DEFAULT_ROUNDS = 2
DEFAULT_MAX_SETTINGS = 6
DEFAULT_HOLDOUT = 0.3


@dataclass(frozen=True)
class Plan:
    """What the sweep will run, decided before any app call so the cost prompt and the work agree."""

    source_run: RunId
    baseline_k: int  # the most chunks one retrieval returned in the source run
    counts: dict[FailureKind, int]  # labeled failures per kind
    unlabeled: int  # failures the source run has no diagnosis for
    candidates: list[Ranked]  # every candidate, heaviest first; a round takes max_settings of them
    split: Split
    rounds: int
    max_settings: int

    @property
    def unfixable(self) -> list[tuple[FailureKind, int, str]]:
        out = []
        for kind, n in self.counts.items():
            rule = POLICY[kind]
            if isinstance(rule, Unfixable) and n:
                out.append((kind, n, rule.reason))
        return out

    @property
    def round_size(self) -> int:
        return min(len(self.candidates), self.max_settings)

    @property
    def worst_case_runs(self) -> list[int]:
        """Case count of every run the sweep may make: the baseline on tune, every round full on
        tune, and the baseline plus the winner on holdout."""
        tune, held = len(self.split.tune), len(self.split.holdout)
        return [tune] * (1 + self.round_size * self.rounds) + ([held, held] if held and self.candidates else [])

    @property
    def worst_case_app_calls(self) -> int:
        return sum(self.worst_case_runs)


class SweepNotPossible(Exception):
    """The source run cannot seed a sweep. The CLI prints it as a one-line user error."""


def plan_sweep(store: Store, run: RunInfo, *, rounds: int, max_settings: int, holdout: float, seed: int) -> tuple[Plan, dict[CaseId, TestCase]]:
    """Read the source run's labels and retrievals and decide the candidates and the split. The
    cases are the ones the source run ran, so the sweep measures the set it was diagnosed on."""
    traces = store.traces(run.id)
    cases: dict[CaseId, TestCase] = {}
    failed: set[CaseId] = set()
    counts: dict[FailureKind, int] = {}
    unlabeled = 0
    for stored in traces:
        test = stored.trace.test
        case = store.case(test.case_id) if test else None
        if test is None or case is None:
            continue
        cases[case.id] = case
        if test.passed:
            continue
        failed.add(case.id)
        if stored.diagnosis is None:
            unlabeled += 1
            continue
        kind = kind_of(stored.diagnosis)
        if kind is not None:
            counts[kind] = counts.get(kind, 0) + 1
    if not cases:
        raise SweepNotPossible(f"run {run.id} has no graded cases")
    if failed and not counts and unlabeled:
        raise SweepNotPossible(f"run {run.id} has {unlabeled} failures and none is labeled; run `contexteval diagnose` first")
    k = max((len(e.retrieved) for s in traces for e in s.trace.retrievals), default=0)
    if failed and k == 0:
        raise SweepNotPossible(f"run {run.id} captured no retrieval, so no setting can reach the app's context")
    plan = Plan(
        source_run=run.id,
        baseline_k=k,
        counts=counts,
        unlabeled=unlabeled,
        candidates=rank_candidates(counts, k),
        split=split_cases(list(cases.values()), failed, holdout, seed),
        rounds=rounds,
        max_settings=max_settings,
    )
    return plan, cases


# --- search -----------------------------------------------------------------------


def _rank_key(row: ExperimentRow) -> tuple[float, float, float]:
    """Best first: accuracy, then fewer input tokens, then lower latency."""
    return (-row.accuracy, row.mean_input_tokens, row.mean_latency_ms)


def beats(candidate: ExperimentRow, baseline: ExperimentRow) -> bool:
    return candidate.passed > baseline.passed


@dataclass(frozen=True)
class SweepReport:
    """What a sweep did and found. Stored on the experiment; the dashboard and the CLI read it."""

    source_run: RunId
    baseline_k: int
    counts: dict[FailureKind, int]
    split: Split
    baseline_run: RunId  # the baseline on the tune cases
    rounds: tuple[tuple[RunId, ...], ...]  # candidate runs on the tune cases, per round
    holdout_runs: tuple[RunId, RunId] | None  # (baseline, winner) on the holdout cases
    recommended: RunId | None  # the winner's tune run, when it is recommended
    reason: str  # why nothing is recommended; empty when something is

    def to_json(self) -> dict[str, Any]:
        return {
            "source_run": self.source_run,
            "baseline_k": self.baseline_k,
            "counts": {k.value: n for k, n in self.counts.items()},
            "split": {
                "seed": self.split.seed,
                "holdout": self.split.holdout_fraction,
                "tune": list(self.split.tune),
                "holdout_cases": list(self.split.holdout),
            },
            "baseline_run": self.baseline_run,
            "rounds": [list(r) for r in self.rounds],
            "holdout_runs": list(self.holdout_runs) if self.holdout_runs else None,
            "recommended": self.recommended,
            "reason": self.reason,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SweepReport:
        split = data["split"]
        held = data["holdout_runs"]
        return cls(
            source_run=RunId(data["source_run"]),
            baseline_k=data["baseline_k"],
            counts={FailureKind(k): n for k, n in data["counts"].items()},
            split=Split(split["seed"], split["holdout"], tuple(CaseId(c) for c in split["tune"]), tuple(CaseId(c) for c in split["holdout_cases"])),
            baseline_run=RunId(data["baseline_run"]),
            rounds=tuple(tuple(RunId(r) for r in round_) for round_ in data["rounds"]),
            holdout_runs=(RunId(held[0]), RunId(held[1])) if held else None,
            recommended=RunId(data["recommended"]) if data["recommended"] is not None else None,
            reason=data["reason"],
        )


def run_sweep(store: Store, app_spec: str, app: App, cases: Mapping[CaseId, TestCase], grade: Grader, plan: Plan) -> tuple[ExperimentId, SweepReport]:
    """Greedy rounds on the tune cases, then the holdout check. Round 1 runs the heaviest
    candidates. Each later round combines the best setting so far with every round-1 candidate
    that changes other knobs and did not lose to the baseline, and stops when the best setting is
    the baseline or every combination has run. Every run is run_suite under one sweep experiment,
    so a failure halfway leaves the finished rows stored.
    """
    experiment = store.new_experiment("sweep")
    tune = [cases[c] for c in plan.split.tune]
    held = [cases[c] for c in plan.split.holdout]
    runs: dict[Setting, RunId] = {}

    def run(setting: Setting, on: Sequence[TestCase]) -> RunId:
        return run_suite(store, app_spec, app, on, grade, setting, experiment).run

    def rows() -> dict[RunId, ExperimentRow]:
        return {row.run_id: row for row in store.experiment_table(experiment)}

    def best_on_tune(table: Mapping[RunId, ExperimentRow]) -> Setting:
        return min(runs, key=lambda s: _rank_key(table[runs[s]]))

    runs[BASELINE] = run(BASELINE, tune)
    first_round = [r.setting for r in plan.candidates[: plan.max_settings]]
    rounds: list[tuple[RunId, ...]] = []
    for number in range(plan.rounds):
        if number == 0:
            todo = list(first_round)
        else:
            table = rows()
            best = best_on_tune(table)
            if best == BASELINE:
                break
            baseline_passed = table[runs[BASELINE]].passed
            todo = []
            for partner in first_round:
                if not knobs(partner).isdisjoint(knobs(best)) or table[runs[partner]].passed < baseline_passed:
                    continue
                combined = merge(best, partner)
                if combined not in runs and combined not in todo:
                    todo.append(combined)
            todo = todo[: plan.max_settings]
        if not todo:
            break
        for setting in todo:
            runs[setting] = run(setting, tune)
        rounds.append(tuple(runs[s] for s in todo))

    table = rows()
    baseline_row = table[runs[BASELINE]]
    winner = best_on_tune(table)
    holdout_runs = None
    recommended = None
    if winner == BASELINE or not beats(table[runs[winner]], baseline_row):
        contenders = [s for s in runs if s != BASELINE]
        if contenders:
            runner_up = min(contenders, key=lambda s: _rank_key(table[runs[s]]))
            reason = (
                f"no candidate beat the baseline on the tune cases "
                f"(the best, {runner_up.name}, passed {table[runs[runner_up]].accuracy:.0%} against {baseline_row.accuracy:.0%})"
            )
        else:
            reason = "no candidate was run"
    elif held:
        holdout_runs = (run(BASELINE, held), run(winner, held))
        table = rows()
        hb, hw = table[holdout_runs[0]], table[holdout_runs[1]]
        if hw.passed >= hb.passed:
            recommended, reason = runs[winner], ""
        else:
            reason = (
                f"{winner.name} won on the tune cases ({baseline_row.accuracy:.0%} -> {table[runs[winner]].accuracy:.0%}) "
                f"but lost on the holdout cases ({hb.accuracy:.0%} -> {hw.accuracy:.0%}), so it fits the tune cases rather than the app"
            )
    else:
        recommended, reason = runs[winner], ""
    report = SweepReport(
        source_run=plan.source_run,
        baseline_k=plan.baseline_k,
        counts=dict(plan.counts),
        split=plan.split,
        baseline_run=runs[BASELINE],
        rounds=tuple(rounds),
        holdout_runs=holdout_runs,
        recommended=recommended,
        reason=reason,
    )
    store.save_experiment_report(experiment, report.to_json())
    return experiment, report


# --- the recommendation -------------------------------------------------------------


def _token_clause(before: float, after: float) -> str:
    if before <= 0:
        return "Input token use is unknown"
    change = (after - before) / before
    if abs(change) < 0.01:
        return "Uses about the same input tokens"
    return f"Uses {abs(change):.0%} {'more' if change > 0 else 'fewer'} input tokens"


def recommendation_line(report: SweepReport, table: Sequence[ExperimentRow]) -> str:
    """One plain sentence or two: the change to make, the accuracy before and after on tune and
    holdout, and what it costs; or why to keep the current settings."""
    if report.recommended is None:
        return f"Keep the current settings. {report.reason}."
    rows = {row.run_id: row for row in table}
    before, after = rows[report.baseline_run], rows[report.recommended]
    if report.holdout_runs:
        hb, hw = rows[report.holdout_runs[0]], rows[report.holdout_runs[1]]
        holdout = f" (holdout {hb.accuracy:.0%} -> {hw.accuracy:.0%})"
    else:
        holdout = " (no holdout cases, so this is unconfirmed)"
    return (
        f"Set {after.setting_name}. Accuracy {before.accuracy:.0%} -> {after.accuracy:.0%}{holdout}. "
        f"{_token_clause(before.mean_input_tokens, after.mean_input_tokens)}; "
        f"mean latency {before.mean_latency_ms:.0f} ms -> {after.mean_latency_ms:.0f} ms."
    )
