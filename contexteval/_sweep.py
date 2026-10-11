"""The diagnosis-guided sweep (v2): labels in, one plain recommendation out.

The policy is a table from failure kind to candidate settings. The search is greedy over
rounds on a tune split and confirms the winner on a holdout split. Every run is made by the
`Runner` the caller hands in (the CLI passes `_runner.run_suite`) under one experiment, so a
sweep is an experiment the dashboard already shows, marked `sweep` and carrying its report, and
this module never imports the runner or an app: the dashboard reads reports without them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, Union

from ._intervention import BASELINE, InterventionNotApplicable, Order, Setting
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
NEAR_TIE_PERCENT = 5  # the default near-tie window, as a share of the tune cases


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
    tolerance: int | None  # tune cases a cheaper setting may trail the most accurate one by; None for NEAR_TIE_PERCENT of them
    tokens_known: bool  # the source run recorded input tokens; without them no setting is cheaper than another

    @property
    def requested_window(self) -> int:
        if self.tolerance is not None:
            return self.tolerance
        return len(self.split.tune) * NEAR_TIE_PERCENT // 100

    @property
    def window(self) -> int:
        """The near-tie window the search uses: the requested one, or 0 when input tokens are
        unknown, because the window exists to trade accuracy for tokens."""
        return self.requested_window if self.tokens_known else 0

    @property
    def window_note(self) -> str | None:
        """The near-tie window in words, or None when none was asked for and accuracy alone decides."""
        if not self.requested_window:
            return None
        if not self.tokens_known:
            return f"near-tie window: 0 (run {self.source_run} recorded no input tokens, so accuracy alone decides)"
        cases = "case" if self.window == 1 else "cases"
        source = "--tolerance" if self.tolerance is not None else f"{NEAR_TIE_PERCENT}% of {len(self.split.tune)}"
        return f"near-tie window: {self.window} {cases} ({source})"

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
    def worst_case_cases(self) -> list[tuple[CaseId, ...]]:
        """The cases of every run the sweep may make: the baseline on tune, every round full on
        tune, and the baseline plus the winner on holdout. The cost estimate grades this list and
        the app line counts it, so the two cannot disagree."""
        tune, held = self.split.tune, self.split.holdout
        return [tune] * (1 + self.round_size * self.rounds) + ([held, held] if held and self.candidates else [])

    @property
    def worst_case_app_calls(self) -> int:
        return sum(len(run) for run in self.worst_case_cases)


class SweepNotPossible(Exception):
    """No run can seed a sweep, or the chosen one cannot. The CLI prints it as a one-line user error."""


def source_run(store: Store, run: RunId | None) -> RunInfo:
    """The run the sweep reads its labels from: `run` when given, which must be a baseline run;
    else the newest baseline run with a diagnosis, so a `contexteval run` made after `diagnose`
    does not hide the labels; else the newest baseline run, whose failures plan_sweep reports as
    unlabeled."""
    if run is not None:
        try:
            info = store.run(run)
        except KeyError:
            raise SweepNotPossible(f"no run {run}; `contexteval ui` lists runs") from None
        if info.kind != "baseline":
            raise SweepNotPossible(f"run {run} is {'an' if info.kind == 'experiment' else 'a'} {info.kind} run; the sweep starts from a baseline run")
        return info
    newest: RunInfo | None = None
    for summary in store.runs():  # newest first
        if summary.run.kind != "baseline":
            continue
        if newest is None:
            newest = summary.run
        if any(stored.diagnosis is not None for stored in store.traces(summary.run.id)):
            return summary.run
    if newest is None:
        raise SweepNotPossible("no baseline run yet; run `contexteval run` and `contexteval diagnose` first")
    return newest


def plan_sweep(store: Store, run: RunInfo, *, rounds: int, max_settings: int, holdout: float, seed: int, tolerance: int | None) -> tuple[Plan, dict[CaseId, TestCase]]:
    """Read the source run's graded cases, split them, then count the tune cases' labels and
    decide the candidates. The holdout cases' labels never reach the search, so the holdout stays
    unseen by it. The cases are the ones the source run ran, so the sweep measures the set it was
    diagnosed on."""
    traces = store.traces(run.id)
    cases: dict[CaseId, TestCase] = {}
    labels: dict[CaseId, Diagnosis | None] = {}  # the failed cases and their diagnosis
    for stored in traces:
        test = stored.trace.test
        case = store.case(test.case_id) if test else None
        if test is None or case is None:
            continue
        cases[case.id] = case
        if not test.passed:
            labels[case.id] = stored.diagnosis
    if not cases:
        raise SweepNotPossible(f"run {run.id} has no graded cases")
    split = split_cases(list(cases.values()), labels, holdout, seed)
    counts: dict[FailureKind, int] = {}
    unlabeled = 0
    for case_id in split.tune:
        if case_id not in labels:
            continue
        diagnosis = labels[case_id]
        if diagnosis is None:
            unlabeled += 1
            continue
        kind = kind_of(diagnosis)
        if kind is not None:
            counts[kind] = counts.get(kind, 0) + 1
    if unlabeled and not counts:
        raise SweepNotPossible(f"run {run.id} has {unlabeled} failures among the tune cases and none is labeled; run `contexteval diagnose` first")
    k = max((len(e.retrieved) for s in traces for e in s.trace.retrievals), default=0)
    if labels and k == 0:
        raise SweepNotPossible(f"run {run.id} captured no retrieval, so no setting can reach the app's context")
    plan = Plan(
        source_run=run.id,
        baseline_k=k,
        counts=counts,
        unlabeled=unlabeled,
        candidates=rank_candidates(counts, k),
        split=split,
        rounds=rounds,
        max_settings=max_settings,
        tolerance=tolerance,
        tokens_known=any(s.trace.metrics.input_tokens > 0 for s in traces),
    )
    return plan, cases


# --- search -----------------------------------------------------------------------


def choose(rows: Mapping[Setting, ExperimentRow], baseline_passed: int, window: int) -> Setting:
    """The best setting on the same cases. Every setting within `window` passes of the most
    passing one, and passing more than the baseline, is a contender; the cheapest contender wins:
    fewest input tokens, then most passes, then lowest latency. When nothing passes more than the
    baseline, only the most passing settings contend. Table order breaks exact ties."""
    top = max(row.passed for row in rows.values())
    floor = top if top <= baseline_passed else max(top - window, baseline_passed + 1)
    contenders = [s for s, row in rows.items() if row.passed >= floor]
    return min(contenders, key=lambda s: (rows[s].mean_input_tokens, -rows[s].passed, rows[s].mean_latency_ms))


def winner(table: Mapping[Setting, ExperimentRow], window: int) -> Setting | None:
    """The setting the sweep carries forward from the tune rows, BASELINE among them: `choose`
    over them, or None when no setting passed more cases than the baseline. Each round's base and
    the final pick both use it, so a setting that only tied the baseline is never built on."""
    baseline_passed = table[BASELINE].passed
    if max(row.passed for row in table.values()) <= baseline_passed:
        return None
    return choose(table, baseline_passed, window)


@dataclass(frozen=True)
class Decision:
    """The sweep's verdict: the setting to recommend and the more accurate, costlier one it
    passed over; or neither, and the reason."""

    recommended: Setting | None
    passed_over: Setting | None
    reason: str


def decide(table: Mapping[Setting, ExperimentRow], window: int, holdout: Mapping[Setting, ExperimentRow] | None) -> Decision:
    """Pure over the tune rows (BASELINE among them) and the holdout rows of the baseline and the
    winner, or None when no case was held out. The winner is recommended when it passed at least
    as many holdout cases as the baseline; with no holdout it is recommended unconfirmed, which
    the recommendation line says."""
    baseline = table[BASELINE]
    best = winner(table, window)
    if best is None:
        candidates = {s: row for s, row in table.items() if s != BASELINE}
        if not candidates:
            return Decision(None, None, "no candidate ran")
        runner_up = choose(candidates, baseline.passed, 0)
        return Decision(
            None,
            None,
            f"no candidate beat the baseline on the tune cases (the best, {runner_up.name}, passed {candidates[runner_up].accuracy:.0%} against {baseline.accuracy:.0%})",
        )
    if holdout is not None:
        hb, hw = holdout[BASELINE], holdout[best]
        if hw.passed < hb.passed:
            return Decision(
                None,
                None,
                f"{best.name} won on the tune cases ({baseline.accuracy:.0%} -> {table[best].accuracy:.0%}) "
                f"but lost on the holdout cases ({hb.accuracy:.0%} -> {hw.accuracy:.0%}), so it fits the tune cases rather than the app",
            )
    most_accurate = choose(table, baseline.passed, 0)
    return Decision(best, most_accurate if most_accurate != best else None, "")


@dataclass(frozen=True)
class SweepReport:
    """What a sweep did and found. Stored on the experiment; the dashboard and the CLI read it."""

    source_run: RunId
    baseline_k: int
    counts: dict[FailureKind, int]
    split: Split
    baseline_run: RunId | None  # the baseline on the tune cases; None when the sweep stopped before it finished
    rounds: tuple[tuple[RunId, ...], ...]  # candidate runs on the tune cases, per round
    holdout_runs: tuple[RunId, RunId] | None  # (baseline, winner) on the holdout cases
    recommended: RunId | None  # the winner's tune run, when it is recommended
    reason: str  # why nothing is recommended; empty when something is
    passed_over: RunId | None = None  # the more accurate, costlier tune run a recommendation passed over
    not_applicable: dict[str, str] = field(default_factory=dict)  # setting name -> why the app's retrieval could not take it

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
            "passed_over": self.passed_over,
            "not_applicable": dict(self.not_applicable),
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
            baseline_run=RunId(data["baseline_run"]) if data["baseline_run"] is not None else None,
            rounds=tuple(tuple(RunId(r) for r in round_) for round_ in data["rounds"]),
            holdout_runs=(RunId(held[0]), RunId(held[1])) if held else None,
            recommended=RunId(data["recommended"]) if data["recommended"] is not None else None,
            reason=data["reason"],
            passed_over=RunId(data["passed_over"]) if data.get("passed_over") is not None else None,
            not_applicable=dict(data.get("not_applicable", {})),
        )


Runner = Callable[[ExperimentId, Setting, Sequence[TestCase]], RunId]
"""Runs the cases under the setting as one stored run of the experiment and returns its id. The
CLI passes `_runner.run_suite`; tests pass a scripted one. May raise InterventionNotApplicable."""


def run_sweep(store: Store, cases: Mapping[CaseId, TestCase], plan: Plan, run_cases: Runner) -> tuple[ExperimentId, SweepReport]:
    """Greedy rounds on the tune cases, then the holdout check. Round 1 runs the heaviest
    candidates. Each later round combines the winner so far with every round-1 candidate that
    changes other knobs and did not lose to the baseline, and stops when no setting beats the
    baseline or every combination has run. A candidate the app's retrieval cannot take
    (InterventionNotApplicable) is skipped with its reason, and so is every later single-knob
    candidate on that knob. Every run is `run_cases` under one sweep experiment, and the report
    is saved whatever happens: a sweep that stops early (a provider error, Ctrl-C) leaves its
    finished runs stored and a report that says why.
    """
    experiment = store.new_experiment("sweep")
    tune = [cases[c] for c in plan.split.tune]
    held = [cases[c] for c in plan.split.holdout]
    runs: dict[Setting, RunId] = {}
    rounds: list[list[RunId]] = []
    holdout_runs: tuple[RunId, RunId] | None = None
    not_applicable: dict[str, str] = {}
    refused: dict[str, Setting] = {}  # knob -> the single-knob setting the app refused it on

    def rows() -> dict[RunId, ExperimentRow]:
        return {row.run_id: row for row in store.experiment_table(experiment)}

    def tune_rows() -> dict[Setting, ExperimentRow]:
        by_run = rows()
        return {s: by_run[r] for s, r in runs.items()}

    def attempt(setting: Setting) -> None:
        shared = knobs(setting) & refused.keys()
        if shared:
            knob = min(shared)
            not_applicable[setting.name] = f"sets {knob}, which {refused[knob].name} showed the app's retrieval cannot take"
            return
        try:
            runs[setting] = run_cases(experiment, setting, tune)
        except InterventionNotApplicable as e:
            not_applicable[setting.name] = str(e)
            if len(knobs(setting)) == 1:
                refused[next(iter(knobs(setting)))] = setting
            return
        rounds[-1].append(runs[setting])

    def search() -> Decision:
        runs[BASELINE] = run_cases(experiment, BASELINE, tune)
        first_round = [r.setting for r in plan.candidates[: plan.max_settings]]
        for number in range(plan.rounds):
            if number == 0:
                todo = list(first_round)
            else:
                table = tune_rows()
                best = winner(table, plan.window)
                if best is None:
                    break
                baseline_passed = table[BASELINE].passed
                todo = []
                for partner in first_round:
                    if partner not in runs or not knobs(partner).isdisjoint(knobs(best)) or table[partner].passed < baseline_passed:
                        continue
                    combined = merge(best, partner)
                    if combined not in runs and combined not in todo:
                        todo.append(combined)
                todo = todo[: plan.max_settings]
            if not todo:
                break
            rounds.append([])
            for setting in todo:
                attempt(setting)
        table = tune_rows()
        best = winner(table, plan.window)
        holdout_rows = None
        if best is not None and held:
            nonlocal holdout_runs
            holdout_runs = (run_cases(experiment, BASELINE, held), run_cases(experiment, best, held))
            by_run = rows()
            holdout_rows = {BASELINE: by_run[holdout_runs[0]], best: by_run[holdout_runs[1]]}
        decision = decide(table, plan.window, holdout_rows)
        if len(runs) == 1 and not_applicable:
            return Decision(None, None, "no candidate could run; " + "; ".join(f"{name}: {why}" for name, why in not_applicable.items()))
        return decision

    try:
        decision = search()
    except BaseException as e:
        decision = Decision(None, None, f"the sweep stopped early: {type(e).__name__}: {e}")
        raise
    finally:
        report = SweepReport(
            source_run=plan.source_run,
            baseline_k=plan.baseline_k,
            counts=dict(plan.counts),
            split=plan.split,
            baseline_run=runs.get(BASELINE),
            rounds=tuple(tuple(r) for r in rounds if r),
            holdout_runs=holdout_runs,
            recommended=runs[decision.recommended] if decision.recommended is not None else None,
            reason=decision.reason,
            passed_over=runs[decision.passed_over] if decision.passed_over is not None else None,
            not_applicable=not_applicable,
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


def _passed_over_clause(chosen: ExperimentRow, other: ExperimentRow) -> str:
    more = other.passed - chosen.passed
    cases = "case" if more == 1 else "cases"
    if other.mean_input_tokens != chosen.mean_input_tokens and chosen.mean_input_tokens > 0:
        cost = f"uses {(other.mean_input_tokens - chosen.mean_input_tokens) / chosen.mean_input_tokens:.0%} more input tokens"
    elif other.mean_input_tokens != chosen.mean_input_tokens:
        cost = "uses more input tokens"
    else:
        cost = f"is slower ({other.mean_latency_ms:.0f} ms against {chosen.mean_latency_ms:.0f} ms)"
    return f" ({other.setting_name} passed {more} more tune {cases} but {cost})"


def recommendation_line(report: SweepReport, table: Sequence[ExperimentRow]) -> str:
    """One plain sentence or two: the change to make, the accuracy before and after on tune and
    holdout, and what it costs; or why to keep the current settings."""
    if report.recommended is None or report.baseline_run is None:
        return f"Keep the current settings. {report.reason}."
    rows = {row.run_id: row for row in table}
    before, after = rows[report.baseline_run], rows[report.recommended]
    if report.holdout_runs:
        hb, hw = rows[report.holdout_runs[0]], rows[report.holdout_runs[1]]
        holdout = f" (holdout {hb.accuracy:.0%} -> {hw.accuracy:.0%})"
    else:
        holdout = " (no holdout cases, so this is unconfirmed)"
    passed_over = _passed_over_clause(after, rows[report.passed_over]) if report.passed_over is not None else ""
    return (
        f"Set {after.setting_name}{passed_over}. Accuracy {before.accuracy:.0%} -> {after.accuracy:.0%}{holdout}. "
        f"{_token_clause(before.mean_input_tokens, after.mean_input_tokens)}; "
        f"mean latency {before.mean_latency_ms:.0f} ms -> {after.mean_latency_ms:.0f} ms."
    )
