"""v2 auto-sweep: the label-to-candidates policy, the seeded split, the greedy search, and the
recommendation, through `_cli.main` where the user would see it."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

import pytest

from contexteval import _cli, _sweep
from contexteval._intervention import BASELINE, GoldToTop, InjectEvidence, InterventionNotApplicable, OnlyGold, Order, Setting
from contexteval._store import Store
from contexteval._sweep import FailureKind, Fixable, Signal, Unfixable
from contexteval.types import (
    Answered,
    CaseId,
    Chunk,
    Chunking,
    Evidence,
    ExperimentId,
    ExperimentRow,
    Flaky,
    Ignored,
    Irrelevant,
    Judged,
    JudgeVerdict,
    Label,
    Metrics,
    Missing,
    NotContext,
    ReplayOutcome,
    ReplayStep,
    RetrievalEvent,
    RunId,
    TestCase,
    TestOutcome,
    Trace,
)

GOLD_TRANSFORMS = (InjectEvidence, GoldToTop, OnlyGold)
RERUN = (ReplayOutcome(ReplayStep.BASELINE_RERUN, False, "wrong"),)
K = 4
BASE = _sweep.Baseline(k=K, chunk_chars=400)
CAUSES = (*FailureKind, *Signal)


def _knobs(setting: Setting) -> dict[str, object]:
    return {name: value for name, value in setting.to_json().items() if value != BASELINE.to_json()[name]}


# --- policy -----------------------------------------------------------------------


def test_policy_maps_each_label_to_its_candidates_and_the_unfixable_to_none() -> None:
    """Defect: a label whose candidates do not address it (a top_k change for an irrelevant case,
    a reorder for a never-retrieved chunk), a fixable label with no candidates, or dropped,
    not_context, and flaky offered a setting when no setting can change them. v2 chunking: the
    real-bot audit's cause (a chunk cut from its heading reads as ignored or irrelevant) asks for
    title=on under both labels, and a split gold asks for a larger chunk_size and an overlap."""
    assert set(_sweep.POLICY) == set(CAUSES)
    resolved = {cause: [_knobs(s) for s in _sweep.candidates_for(cause, BASE)] for cause in CAUSES}

    assert resolved[FailureKind.NOT_RETRIEVED] and all(set(c) == {"top_k"} and c["top_k"] > K for c in resolved[FailureKind.NOT_RETRIEVED])  # type: ignore[operator]
    irrelevant = resolved[FailureKind.IRRELEVANT]
    assert {"rerank": "lexical"} in irrelevant
    assert any(set(c) == {"max_chunks"} and c["max_chunks"] < K for c in irrelevant)  # type: ignore[operator]
    assert any(set(c) == {"compress"} for c in irrelevant)
    assert {"title": True} in irrelevant
    ignored = resolved[FailureKind.IGNORED]
    assert {"order": "ends"} in ignored and {"order": "reverse"} in ignored and any(set(c) == {"compress"} for c in ignored)
    assert {"title": True} in ignored
    assert resolved[Signal.SPLIT_GOLD] == [{"chunk_size": 800}, {"overlap": 100}]

    for kind in (FailureKind.DROPPED, FailureKind.NOT_CONTEXT, FailureKind.FLAKY):
        rule = _sweep.POLICY[kind]
        assert isinstance(rule, Unfixable) and rule.reason and resolved[kind] == []
    assert "your own code" in _sweep.POLICY[FailureKind.DROPPED].reason  # type: ignore[union-attr]
    for cause in (FailureKind.NOT_RETRIEVED, FailureKind.IRRELEVANT, FailureKind.IGNORED, Signal.SPLIT_GOLD):
        assert isinstance(_sweep.POLICY[cause], Fixable)
    assert {cause for cause in CAUSES if _sweep.points_at_chunking(cause, BASE)} == {FailureKind.IRRELEVANT, FailureKind.IGNORED, Signal.SPLIT_GOLD}


def test_relative_candidates_follow_the_observed_baseline_and_drop_no_ops() -> None:
    """Defect: top_k candidates fixed at 10 whatever the app retrieves (a no-op for an app at
    k=12), a max_chunks candidate at or above k, which changes nothing and wastes a run, a
    chunk_size fixed at 800 for an app whose chunks are already 1000 characters, or a chunking
    candidate proposed when the source run delivered no chunk to measure."""
    assert [_knobs(s)["top_k"] for s in _sweep.candidates_for(FailureKind.NOT_RETRIEVED, _sweep.Baseline(12, 400))] == [24, 36]
    assert [_knobs(s)["top_k"] for s in _sweep.candidates_for(FailureKind.NOT_RETRIEVED, _sweep.Baseline(1, 400))] == [2, 3]
    assert all(_knobs(s).get("max_chunks", 0) < 12 for s in _sweep.candidates_for(FailureKind.IRRELEVANT, _sweep.Baseline(12, 400)))
    assert not any("max_chunks" in _knobs(s) for s in _sweep.candidates_for(FailureKind.IRRELEVANT, _sweep.Baseline(1, 400)))
    assert [_knobs(s) for s in _sweep.candidates_for(Signal.SPLIT_GOLD, _sweep.Baseline(4, 1000))] == [{"chunk_size": 2000}, {"overlap": 250}]
    assert _sweep.candidates_for(Signal.SPLIT_GOLD, _sweep.Baseline(4, 0)) == []


def test_every_diagnosis_has_a_failure_kind() -> None:
    """Defect: a judged label or a Missing cause the policy does not know, so a labeled failure
    silently contributes nothing to the search."""
    assert _sweep.kind_of(Missing("not_retrieved", True, RERUN)) is FailureKind.NOT_RETRIEVED
    assert _sweep.kind_of(Missing("dropped", True, RERUN)) is FailureKind.DROPPED
    assert _sweep.kind_of(Irrelevant(RERUN, ((0, 1),))) is FailureKind.IRRELEVANT
    assert _sweep.kind_of(Ignored(RERUN, ((0, 2),))) is FailureKind.IGNORED
    assert _sweep.kind_of(NotContext(RERUN)) is FailureKind.NOT_CONTEXT
    assert _sweep.kind_of(Flaky(RERUN)) is FailureKind.FLAKY
    for verdict in JudgeVerdict:
        kind = _sweep.kind_of(Judged(verdict, "why"))
        if verdict is JudgeVerdict.NO_FAILURE:
            assert kind is None
        else:
            assert kind is not None and kind.value.startswith(verdict.value)
    assert {kind.value.split("/")[0] for kind in FailureKind} == {label.value for label in Label}


def test_candidates_are_ranked_by_how_many_failures_carry_their_label() -> None:
    """Defect: candidates in table order regardless of the labels seen, so one ignored case
    ranks a reorder above the top_k change three missing cases call for, and --max-settings
    then cuts the wrong candidate; or a candidate shared by two labels counted once."""
    counts = {FailureKind.NOT_RETRIEVED: 3, FailureKind.IGNORED: 1, FailureKind.IRRELEVANT: 1, FailureKind.DROPPED: 2}
    ranked = _sweep.rank_candidates(counts, BASE)
    names = [r.setting.name for r in ranked]
    assert names[:2] == ["top_k=8", "top_k=12"] and [r.weight for r in ranked[:2]] == [3, 3]
    compress = next(r for r in ranked if r.setting.name.startswith("compress="))
    assert compress.weight == 2 and set(compress.kinds) == {FailureKind.IGNORED, FailureKind.IRRELEVANT}
    assert names.index(compress.setting.name) == 2
    title = next(r for r in ranked if r.setting.name == "title=on")
    assert title.weight == 2 and set(title.kinds) == {FailureKind.IGNORED, FailureKind.IRRELEVANT} and names.index("title=on") == 3
    assert len(names) == len(set(names))
    assert all(r.weight >= 1 for r in ranked) and ranked == sorted(ranked, key=lambda r: -r.weight)
    assert _sweep.rank_candidates({FailureKind.DROPPED: 2, FailureKind.FLAKY: 1}, BASE) == []
    split = _sweep.rank_candidates({FailureKind.NOT_CONTEXT: 2, Signal.SPLIT_GOLD: 2}, BASE)
    assert [(r.setting.name, r.weight, r.kinds) for r in split] == [("chunk_size=800", 2, (Signal.SPLIT_GOLD,)), ("overlap=100", 2, (Signal.SPLIT_GOLD,))]
    # A signal is direct evidence, so at equal weight its candidates outrank a label's and survive --max-settings.
    tied = [r.setting.name for r in _sweep.rank_candidates({FailureKind.NOT_RETRIEVED: 1, Signal.SPLIT_GOLD: 1}, BASE)]
    assert tied == ["chunk_size=800", "overlap=100", "top_k=8", "top_k=12"]


def test_no_candidate_setting_can_carry_gold() -> None:
    """Plan item 15 for the sweep. Defect: a candidate built from a Diagnosis or a TestCase that
    reaches the app with evidence in it, so a sweep row wins by peeking at the answer."""
    counts = {cause: 1 for cause in CAUSES}
    for ranked in _sweep.rank_candidates(counts, BASE):
        transforms = ranked.setting.intervention().transforms
        assert not any(isinstance(t, GOLD_TRANSFORMS) for t in transforms), ranked.setting.name
    assert all(isinstance(c, (Setting, _sweep.Scaled)) for rule in _sweep.POLICY.values() if isinstance(rule, Fixable) for c in rule.candidates)


def test_merge_combines_disjoint_knobs_and_names_overlaps() -> None:
    """Defect: a round-2 combination that silently lets one knob overwrite the other's value, or
    that reports two settings as combinable when they set the same knob."""
    a, b = Setting(top_k=8), Setting(order=Order.ENDS, compress=2)
    assert _sweep.merge(a, b) == Setting(top_k=8, order=Order.ENDS, compress=2)
    assert _sweep.knobs(a).isdisjoint(_sweep.knobs(b))
    assert not _sweep.knobs(Setting(top_k=8)).isdisjoint(_sweep.knobs(Setting(top_k=12)))
    assert _sweep.knobs(BASELINE) == frozenset()


# --- split ------------------------------------------------------------------------


def _cases(n: int) -> list[TestCase]:
    return [TestCase(CaseId(f"c{i:02d}"), f"q{i}?", f"a{i}", Evidence(f"e{i}"), "t") for i in range(n)]


def test_split_is_deterministic_by_seed_and_keeps_failures_on_both_sides() -> None:
    """Defect: a split that changes between two runs with the same seed (so a user cannot
    reproduce a recommendation), one that ignores the seed, or one that puts every failure on one
    side so the holdout check is vacuous."""
    cases = _cases(20)
    failed = {c.id for c in cases[:6]}
    first = _sweep.split_cases(cases, failed, holdout=0.3, seed=7)
    again = _sweep.split_cases(cases, failed, holdout=0.3, seed=7)
    other = _sweep.split_cases(cases, failed, holdout=0.3, seed=8)
    assert first == again and first.holdout != other.holdout
    assert set(first.tune) | set(first.holdout) == {c.id for c in cases} and not set(first.tune) & set(first.holdout)
    assert len(first.holdout) == 6  # round(0.3 * 6) failures + round(0.3 * 14) passes
    assert len(failed & set(first.holdout)) == 2 and len(failed & set(first.tune)) == 4
    assert first.tune == tuple(sorted(first.tune)) and first.holdout == tuple(sorted(first.holdout))


def _baseline_trace(case: TestCase, passed: bool) -> Trace:
    chunks = tuple(Chunk(f"chunk {i} for {case.question}", f"doc{i}", 0.5, i) for i in range(K))
    return Trace(
        question=case.question,
        outcome=Answered("right" if passed else "wrong"),
        retrievals=(RetrievalEvent("retriever:search", case.question, {"limit": K}, chunks, chunks, 1.0),),
        llm_calls=(),
        metrics=Metrics(0, 0, 1.0, None),
        test=TestOutcome(case.id, passed, "substring"),
        in_prompt=(),
    )


def test_holdout_labels_do_not_choose_the_candidates(store: Store) -> None:
    """Review Q2. Defect: the labels are counted over every case before the split, so a failure
    that lands in the holdout still proposes candidates, and the holdout is no longer unseen by
    the search. Only the tune cases' labels count, and so do only their unlabeled failures."""
    cases = _cases(8)
    store.upsert_cases("t", cases)
    failed = {c.id for c in cases[:6]}
    split = _sweep.split_cases(cases, failed, holdout=0.5, seed=0)
    held_failures = [c for c in cases if c.id in failed and c.id in split.holdout]
    assert len(held_failures) == 3
    run = store.new_run("baseline", "app:answer", BASELINE)
    for case in cases:
        trace_id = store.add_trace(run, _baseline_trace(case, case.id not in failed))
        if case.id in failed and case.id in split.tune:
            store.save_diagnosis(trace_id, Missing("not_retrieved", True, RERUN))
        elif case in held_failures[:2]:
            store.save_diagnosis(trace_id, Ignored(RERUN, ((0, 2),)))
    plan, _ = _sweep.plan_sweep(store, store.run(run), rounds=2, max_settings=6, holdout=0.5, seed=0, tolerance=None, builder=True)
    assert plan.split == split
    assert plan.counts == {FailureKind.NOT_RETRIEVED: 3} and plan.unlabeled == 0
    assert [r.setting for r in plan.candidates] == [Setting(top_k=8), Setting(top_k=12)]


def test_split_gold_proposes_chunking_candidates_only_when_the_app_has_a_builder(store: Store) -> None:
    """v2 chunking, the sweep side of the split signal. Defects: a split-gold failure proposes
    nothing because its label (not_context) is unfixable; chunk_size and overlap scaled from
    nothing instead of the baseline's longest chunk; a chunking candidate run on an app with no
    build(chunking), which would fail on the first row; or, without a builder, the plan losing
    count of how many failures pointed at chunking, so the CLI cannot say what adding build()
    would buy."""
    cases = _cases(4)
    store.upsert_cases("t", cases)
    run = store.new_run("baseline", "app:answer", BASELINE)
    for case in cases:
        trace_id = store.add_trace(run, _baseline_trace(case, passed=case.id > CaseId("c01")))
        if case.id == CaseId("c00"):
            store.save_diagnosis(trace_id, NotContext(RERUN, split_gold=True))
        elif case.id == CaseId("c01"):
            store.save_diagnosis(trace_id, Missing("not_retrieved", True, RERUN))
    longest = max(len(c.text) for s in store.traces(run) for e in s.trace.retrievals for c in e.delivered)
    chunking = [Setting(chunk_size=2 * longest), Setting(overlap=round(longest * 0.25))]

    with_builder, _ = _sweep.plan_sweep(store, store.run(run), rounds=2, max_settings=6, holdout=0, seed=0, tolerance=None, builder=True)
    assert with_builder.baseline == _sweep.Baseline(K, longest) and with_builder.split_gold == 1 and with_builder.chunking_pointers == 1
    assert with_builder.counts == {FailureKind.NOT_CONTEXT: 1, FailureKind.NOT_RETRIEVED: 1}
    assert [r.setting for r in with_builder.candidates] == [*chunking, Setting(top_k=8), Setting(top_k=12)] and with_builder.withheld == []
    assert [(k, n) for k, n, _ in with_builder.unfixable] == [(FailureKind.NOT_CONTEXT, 1)]

    without, _ = _sweep.plan_sweep(store, store.run(run), rounds=2, max_settings=6, holdout=0, seed=0, tolerance=None, builder=False)
    assert [r.setting for r in without.candidates] == [Setting(top_k=8), Setting(top_k=12)]
    assert [r.setting for r in without.withheld] == chunking and without.chunking_pointers == 1
    assert without.worst_case_app_calls < with_builder.worst_case_app_calls


def test_split_with_no_holdout_keeps_every_case_for_tuning() -> None:
    """Defect: --holdout 0 still withholds a case, or a set too small to hold out loses a case
    silently instead of reporting an empty holdout."""
    cases = _cases(3)
    none = _sweep.split_cases(cases, {cases[0].id}, holdout=0.0, seed=1)
    assert none.holdout == () and set(none.tune) == {c.id for c in cases}
    tiny = _sweep.split_cases(cases[:1], {cases[0].id}, holdout=0.3, seed=1)
    assert tiny.holdout == () and tiny.tune == (cases[0].id,)


# --- the sweep through the CLI -------------------------------------------------------

EASY = [f"easy question {i}?" for i in range(4)]
HARD = [f"hard question {i}?" for i in range(6)]
ALL = EASY + HARD
APP_SOURCE = '''
"""Scripted app for the sweep tests. Retrieves `limit` chunks that never hold the evidence, so
every hard case is missing/not_retrieved. With exactly 4 chunks, a question is answered right
only when pass_at_4.json lists it; with any other count, easy questions are right and hard ones wrong."""
import json
from pathlib import Path

import contexteval

HARD = %r


@contexteval.retriever(top_k="limit")
def search(query: str, limit: int = 2) -> list[str]:
    return [f"chunk {i} about {query}" for i in range(limit)]


def answer(question: str) -> str:
    hits = search(question)
    if len(hits) == 4:
        allowed = json.loads(Path("pass_at_4.json").read_text(encoding="utf-8"))
        return "right" if question in allowed else "wrong"
    return "wrong" if question in HARD else "right"
'''


def _write_cases(project: Path, questions: list[str]) -> None:
    lines = [json.dumps({"question": q, "expected_answer": "right", "evidence": f"the fact behind {q}"}) for q in questions]
    (project / "cases.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scripted_project(project: Path, module: str, pass_at_4: list[str], questions: list[str] = ALL) -> str:
    """init, import the cases, run, diagnose. Returns the app spec. `module` is unique per test
    because the CLI imports it from cwd and Python caches it for the process."""
    (project / f"{module}.py").write_text(APP_SOURCE % HARD, encoding="utf-8")
    (project / "pass_at_4.json").write_text(json.dumps(pass_at_4), encoding="utf-8")
    _write_cases(project, questions)
    assert _cli.main(["init", "--provider", "fake"]) == 0
    assert _cli.main(["build-tests", "--import", "cases.jsonl"]) == 0
    assert _cli.main(["run", f"{module}:answer", "--yes"]) == 0
    assert _cli.main(["diagnose", "--yes"]) == 0
    return f"{module}:answer"


def _sweep_runs(store: Store) -> tuple[_sweep.SweepReport, dict[int, tuple[Setting, int]]]:
    """The latest sweep's report and, per run id, its setting and trace count."""
    info = store.experiments()[0]
    assert info.kind == "sweep" and info.report is not None
    report = _sweep.SweepReport.from_json(info.report)
    runs = {r.run.id: (r.run.setting, r.traces) for r in store.runs() if r.experiment_id == info.id}
    return report, runs


# --- the decision, with no app ----------------------------------------------------------

_ROW_IDS = itertools.count(100)


def _row(setting: Setting, passed: int, tokens: float = 0.0, latency: float = 1.0, cases: int = 8) -> ExperimentRow:
    return ExperimentRow(RunId(next(_ROW_IDS)), setting.name, cases, passed, tokens, latency, None)


def _plan(counts: dict[FailureKind, int], tune: Sequence[CaseId], holdout: Sequence[CaseId] = ()) -> _sweep.Plan:
    return _sweep.Plan(
        source_run=RunId(1),
        baseline=BASE,
        counts=counts,
        split_gold=0,
        chunking_pointers=0,
        unlabeled=0,
        ranked=_sweep.rank_candidates(counts, BASE),
        builder=False,  # these plans drive a scripted Runner with no app, so no index can be rebuilt
        split=_sweep.Split(0, 0.3 if holdout else 0.0, tuple(tune), tuple(holdout)),
        rounds=2,
        max_settings=6,
        tolerance=None,
        tokens_known=True,
    )


def _scripted_runner(
    store: Store,
    passes: Callable[[Setting, CaseId], bool],
    tokens: Callable[[Setting], int] = lambda setting: 0,
    refuse: Callable[[Setting], str | None] = lambda setting: None,
    calls: list[Setting] | None = None,
) -> _sweep.Runner:
    """A Runner with no app. Stores one trace per case, passing when `passes` says so and
    carrying `tokens` input tokens; raises InterventionNotApplicable before any trace when
    `refuse` gives a reason. Every attempted setting is appended to `calls`."""

    def run(experiment: ExperimentId, setting: Setting, cases: Sequence[TestCase]) -> RunId:
        if calls is not None:
            calls.append(setting)
        reason = refuse(setting)
        if reason is not None:
            raise InterventionNotApplicable(reason)
        run_id = store.new_run("experiment", "app:answer", setting, experiment)
        for case in cases:
            ok = passes(setting, case.id)
            trace = Trace(
                question=case.question,
                outcome=Answered("right" if ok else "wrong"),
                retrievals=(),
                llm_calls=(),
                metrics=Metrics(tokens(setting), 0, 1.0, None),
                test=TestOutcome(case.id, ok, "substring"),
                in_prompt=(),
            )
            store.add_trace(run_id, trace)
        return run_id

    return run


def test_decide_is_a_pure_function_over_the_tune_and_holdout_rows() -> None:
    """Review Q9. Defect: the recommendation rule lives in run_sweep's control flow (a `beats`
    check, a `choose` call, the holdout comparison, the passed-over lookup) and cannot be tested
    without runs; or one of its outcomes is wrong: a cheaper near-tie not taken under a window or
    taken under window 0, a holdout loss recommended, a tie with the baseline recommended, or the
    setting a recommendation passed over left out."""
    Decision = _sweep.Decision
    a, b, c = Setting(top_k=8), Setting(top_k=12), Setting(rerank="lexical")
    table = {BASELINE: _row(BASELINE, 3, 100), a: _row(a, 5, 200), b: _row(b, 5, 300), c: _row(c, 4, 100)}
    assert _sweep.decide(table, 0, None) == Decision(a, None, "")
    assert _sweep.decide(table, 1, None) == Decision(c, a, "")
    confirmed = {BASELINE: _row(BASELINE, 2, cases=3), a: _row(a, 3, cases=3)}
    assert _sweep.decide(table, 0, confirmed) == Decision(a, None, "")
    lost = {BASELINE: _row(BASELINE, 2, cases=3), a: _row(a, 1, cases=3)}
    assert _sweep.decide(table, 0, lost) == Decision(
        None, None, "top_k=8 won on the tune cases (38% -> 62%) but lost on the holdout cases (67% -> 33%), so it fits the tune cases rather than the app"
    )
    tied = {BASELINE: _row(BASELINE, 3, 100), a: _row(a, 3, 50), c: _row(c, 2, 100)}
    assert _sweep.decide(tied, 2, None) == Decision(None, None, "no candidate beat the baseline on the tune cases (the best, top_k=8, passed 38% against 38%)")
    assert _sweep.decide({BASELINE: _row(BASELINE, 3)}, 0, None) == Decision(None, None, "no candidate ran")
    assert not hasattr(_sweep, "beats")


def test_round_two_never_builds_on_a_setting_that_only_tied_the_baseline(store: Store) -> None:
    """Review Q3. Defect: round 2 takes its base from the cheapest setting at the top pass count,
    so a candidate that passed exactly the baseline's cases at fewer tokens is combined with every
    other knob for a second round the user pays for, while the final pick says nothing beat the
    baseline. Both use one rule: no setting beats the baseline, the search stops."""
    cases = _cases(8)
    store.upsert_cases("t", cases)
    counts = {FailureKind.NOT_RETRIEVED: 1, FailureKind.IGNORED: 1}
    plan = _plan(counts, [c.id for c in cases])
    assert [r.setting.name for r in plan.candidates] == ["top_k=8", "top_k=12", "order=ends", "order=reverse", "compress=2"]
    calls: list[Setting] = []
    run = _scripted_runner(store, lambda setting, case_id: case_id < CaseId("c04"), lambda setting: 50 if setting == Setting(top_k=8) else 100, calls=calls)
    experiment, report = _sweep.run_sweep(store, {c.id: c for c in cases}, plan, run)
    assert calls == [BASELINE, *(r.setting for r in plan.candidates)]
    assert len(report.rounds) == 1 and report.recommended is None and report.holdout_runs is None
    assert report.reason == "no candidate beat the baseline on the tune cases (the best, top_k=8, passed 50% against 50%)"
    assert _sweep.winner({s: _row(s, 4, t) for s, t in [(BASELINE, 100), (Setting(top_k=8), 50)]}, 2) is None


def test_a_candidate_the_app_cannot_take_is_skipped_with_its_knob_and_the_sweep_goes_on(store: Store) -> None:
    """Review Q1. Defect: InterventionNotApplicable from one candidate (a top_k change on a
    retriever with no top_k knob) aborts the whole sweep with no report, although the other knobs
    could still be tried. The candidate is skipped with its reason, later candidates on the same
    knob are skipped without a run, the report records each, and the sweep still recommends."""
    cases = _cases(8)
    store.upsert_cases("t", cases)
    counts = {FailureKind.NOT_RETRIEVED: 1, FailureKind.IGNORED: 1}
    plan = _plan(counts, [c.id for c in cases])
    ends = Setting(order=Order.ENDS)
    calls: list[Setting] = []
    run = _scripted_runner(
        store,
        lambda setting, case_id: case_id < (CaseId("c05") if setting.order is Order.ENDS else CaseId("c04")),
        refuse=lambda setting: f"search: top_k={setting.top_k} requested but this site has no top_k parameter" if setting.top_k else None,
        calls=calls,
    )
    experiment, report = _sweep.run_sweep(store, {c.id: c for c in cases}, plan, run)
    assert calls == [BASELINE, Setting(top_k=8), ends, Setting(order=Order.REVERSE), Setting(compress=2), Setting(order=Order.ENDS, compress=2)]
    assert report.not_applicable == {
        "top_k=8": "search: top_k=8 requested but this site has no top_k parameter",
        "top_k=12": "sets top_k, which top_k=8 showed the app's retrieval cannot take",
    }
    settings = {r.run.id: r.run.setting for r in store.runs()}
    assert report.recommended is not None and settings[report.recommended] == ends and report.reason == ""
    assert [[settings[r] for r in round_] for round_ in report.rounds] == [[ends, Setting(order=Order.REVERSE), Setting(compress=2)], [Setting(order=Order.ENDS, compress=2)]]
    assert _sweep.SweepReport.from_json(store.experiment(experiment).report or {}) == report


def test_sweep_over_an_app_with_no_top_k_knob_ends_with_a_line_and_a_report(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Review Q1, the reviewers' probe through the CLI. Defect: `@contexteval.retriever` with no
    top_k parameter makes `contexteval sweep` exit 1 on the first candidate with no report stored,
    so the dashboard shows a sweep with rows and no outcome. It exits 0, prints each candidate it
    could not run, and ends with a recommendation line that says why nothing is recommended."""
    (project / "sweep_app_noknob.py").write_text(
        "import contexteval\nHARD = %r\n\n@contexteval.retriever\ndef search(query: str) -> list[str]:\n    return [f'chunk {i} about {query}' for i in range(2)]\n\n"
        "def answer(question: str) -> str:\n    search(question)\n    return 'wrong' if question in HARD else 'right'\n" % HARD,
        encoding="utf-8",
    )
    _write_cases(project, ALL)
    assert _cli.main(["init", "--provider", "fake"]) == 0
    assert _cli.main(["build-tests", "--import", "cases.jsonl"]) == 0
    assert _cli.main(["run", "sweep_app_noknob:answer", "--yes"]) == 0
    assert _cli.main(["diagnose", "--yes"]) == 0
    capsys.readouterr()
    assert _cli.main(["sweep", "--holdout", "0", "--yes"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    assert "not applicable: top_k=4: " in out and "has no top_k parameter" in out
    assert "not applicable: top_k=6: sets top_k, which top_k=4 showed the app's retrieval cannot take" in out
    assert _final(out).startswith("Keep the current settings. no candidate could run; top_k=4: ") and "top_k=6" in _final(out)
    store = Store.open(project)
    try:
        report, runs = _sweep_runs(store)
        assert report.recommended is None and report.rounds == () and set(report.not_applicable) == {"top_k=4", "top_k=6"}
        assert report.baseline_run is not None and runs[report.baseline_run] == (BASELINE, len(ALL))
    finally:
        store.close()


def test_report_is_saved_when_the_sweep_stops_early(store: Store) -> None:
    """Review Q1. Defect: a provider error or Ctrl-C halfway through the rounds leaves the sweep
    experiment with rows and no report, so neither the CLI nor the dashboard can say what
    happened. The report is saved whatever stops the sweep, with the finished runs and the reason,
    even when the baseline run itself did not finish."""
    cases = _cases(4)
    store.upsert_cases("t", cases)
    plan = _plan({FailureKind.NOT_RETRIEVED: 1}, [c.id for c in cases])
    assert [r.setting for r in plan.candidates] == [Setting(top_k=8), Setting(top_k=12)]

    def failing(victim: Setting) -> _sweep.Runner:
        inner = _scripted_runner(store, lambda setting, case_id: True)

        def run(experiment: ExperimentId, setting: Setting, on: Sequence[TestCase]) -> RunId:
            if setting == victim:
                raise RuntimeError("provider down")
            return inner(experiment, setting, on)

        return run

    with pytest.raises(RuntimeError, match="provider down"):
        _sweep.run_sweep(store, {c.id: c for c in cases}, plan, failing(Setting(top_k=12)))
    with pytest.raises(RuntimeError, match="provider down"):
        _sweep.run_sweep(store, {c.id: c for c in cases}, plan, failing(BASELINE))
    later, earlier = store.experiments()
    settings = {r.run.id: r.run.setting for r in store.runs()}
    assert earlier.report is not None and later.report is not None
    halfway = _sweep.SweepReport.from_json(earlier.report)
    assert halfway.reason == "the sweep stopped early: RuntimeError: provider down" and halfway.recommended is None
    assert halfway.baseline_run is not None and settings[halfway.baseline_run] == BASELINE
    assert [[settings[r] for r in round_] for round_ in halfway.rounds] == [[Setting(top_k=8)]]
    at_start = _sweep.SweepReport.from_json(later.report)
    assert at_start.baseline_run is None and at_start.rounds == () and at_start.reason == halfway.reason
    assert _sweep.recommendation_line(at_start, []) == "Keep the current settings. the sweep stopped early: RuntimeError: provider down."


def test_near_tie_prefers_more_passes_before_lower_latency() -> None:
    """Review Q4. Defect: the near-tie key ranks latency before passes, so among contenders with
    equal input tokens a setting that passes two fewer cases wins on 2 ms of latency."""
    ends, rev = Setting(order=Order.ENDS), Setting(order=Order.REVERSE)
    rows = {ends: _row(ends, 34, 500.0, 101.0, 40), rev: _row(rev, 32, 500.0, 99.0, 40)}
    assert _sweep.choose(rows, 30, 2) == ends
    rows[rev] = _row(rev, 32, 400.0, 99.0, 40)
    assert _sweep.choose(rows, 30, 2) == rev


@pytest.mark.xfail(strict=True, reason="the near-tie key ranks a one-token difference above the number of changed knobs")
def test_near_tie_prefers_fewer_changed_knobs_over_a_combination_whose_extra_knob_adds_nothing() -> None:
    """Defect: among settings that pass the same cases at input tokens within noise of each other,
    the key ranks one token and then latency, so the sweep recommends `order=reverse title=on`
    over `title=on` alone on the recorded real-bot rows, though the reorder adds nothing and the
    combination was only faster by chance. Fewer changed knobs win a token tie; a setting well
    above the cheapest (`top_k=8 title=on` at 779) still loses on tokens."""
    title = Setting(title=True)
    rev, ends = Setting(order=Order.REVERSE, title=True), Setting(order=Order.ENDS, title=True)
    wide = Setting(top_k=8, title=True)
    rows = {
        BASELINE: _row(BASELINE, 32, 399.0, 917.0, 35),
        title: _row(title, 35, 435.0, 895.0, 35),
        rev: _row(rev, 35, 434.0, 868.0, 35),
        ends: _row(ends, 35, 434.0, 884.0, 35),
        wide: _row(wide, 35, 779.0, 900.0, 35),
    }
    assert _sweep.winner(rows, 1) == title
    assert _sweep.decide(rows, 1, None) == _sweep.Decision(title, None, "")
    rows[rev] = _row(rev, 35, 300.0, 868.0, 35)
    assert _sweep.winner(rows, 1) == rev


def test_window_is_zero_when_the_source_run_recorded_no_input_tokens(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Review Q4. Defect: a near-tie window on an app whose input tokens are unknown (no hooked
    LLM call, so every row reads 0 tokens): the sweep prints a window it cannot apply and credits
    --tolerance with a choice tokens never made. The window is 0 and the plan says why."""
    _scripted_project(project, "sweep_app_no_tokens", EASY + HARD[:1])
    capsys.readouterr()
    assert _cli.main(["sweep", "--holdout", "0", "--tolerance", "1", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "near-tie window: 0 (run 1 recorded no input tokens, so accuracy alone decides)" in out
    assert "near-tie window: 1" not in out
    store = Store.open(project)
    try:
        report, runs = _sweep_runs(store)
        assert report.recommended is not None and runs[report.recommended][0] == Setting(top_k=4) and report.passed_over is None
    finally:
        store.close()


def test_demo_sweep_tries_the_chunking_the_split_case_asks_for_and_names_the_dropped_case_as_unfixable(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Through `_cli.main` on the planted demo, which offers build(chunking). Defects: the planted
    not_retrieved case does not lead to a top_k candidate; the split case does not lead to a
    chunk_size candidate scaled from the longest baseline chunk, or that candidate is cut by the
    default --max-settings behind the label candidates of equal weight, or it runs on the demo's
    own index so the split case still fails under it; a build the app refuses (an overlap on the
    demo's one-paragraph chunks) stops the sweep instead of being listed as not applicable; the
    dropped case is offered a setting instead of the reason none can help; --holdout 0 is refused
    or not warned about; or the sweep is stored as a manual experiment. On 9 tune cases the
    default near-tie window is 0."""
    pytest.importorskip("chromadb")
    from contexteval.demo import planted

    assert _cli.main(["init", "--provider", "fake"]) == 0
    assert _cli.main(["build-tests", "--import", str(planted.write_jsonl(project / planted.IMPORT_FILE))]) == 0
    assert _cli.main(["run", "contexteval.demo.app:answer", "--yes"]) == 0
    assert _cli.main(["diagnose", "--yes"]) == 0
    capsys.readouterr()
    assert _cli.main(["sweep", "--holdout", "0", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "no setting can fix 1 missing/dropped failure: retrieval found the gold chunk and your own code removed it" in out
    assert "no setting can fix 2 not_context failures" in out and "no setting can fix 1 flaky failure" in out
    assert "; 1 with the evidence split across two adjacent chunks; the baseline retrieved up to 4 chunks of up to " in out
    assert "warning: no holdout cases" in out and "chunking:" not in out
    assert "near-tie window" not in out
    store = Store.open(project)
    try:
        report, runs = _sweep_runs(store)
        source = store.traces(report.source_run)
        longest = max(len(c.text) for s in source for e in s.trace.retrievals for c in e.delivered)
        candidates = out.split("candidates, heaviest first")[1].splitlines()[0]
        assert f"chunk_size={2 * longest} (1)" in candidates and f"overlap={round(longest / 4)} (1)" in candidates
        assert "title=on (2)" in candidates and "top_k=8 (1)" in candidates and "top_k=12 (1)" in candidates
        assert f"not applicable: overlap={round(longest / 4)}: build(overlap={round(longest / 4)}) raised ValueError: overlap" in out
        final = out.rstrip().splitlines()[-1]
        assert final.startswith("Set ") and f"chunk_size={2 * longest}" in final and "(no holdout cases, so this is unconfirmed)" in final

        assert report.recommended is not None and report.holdout_runs is None and report.split.holdout == ()
        recommended, traces = runs[report.recommended]
        assert recommended.chunk_size == 2 * longest and traces == len(planted.PLANTED)
        assert report.counts[FailureKind.NOT_RETRIEVED] == 1 and report.counts[FailureKind.DROPPED] == 1 and report.counts[FailureKind.NOT_CONTEXT] == 2
        for setting, _ in runs.values():
            assert not any(isinstance(t, GOLD_TRANSFORMS) for t in setting.intervention().transforms)
        (split,) = [p for p in planted.PLANTED if p.split]
        passed = {s.trace.test.case_id: s.trace.test.passed for s in store.traces(report.recommended) if s.trace.test}
        baseline = {s.trace.test.case_id: s.trace.test.passed for s in store.traces(report.baseline_run) if s.trace.test}
        assert passed[split.case.id] is True and baseline[split.case.id] is False
        assert sum(passed.values()) > sum(baseline.values())
    finally:
        store.close()


def test_a_setting_that_wins_on_tune_but_loses_on_holdout_is_not_recommended(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Overfitting guard. Defect: the sweep recommends the tune winner without the holdout check,
    or recommends it when the holdout rows show it losing. Under top_k=4 the scripted app passes
    exactly the tune cases, which the test learns from the same seeded split the sweep uses, and
    fails every holdout case, including the ones the baseline passed."""
    _scripted_project(project, "sweep_app_overfit", [])
    store = Store.open(project)
    try:
        cases = store.cases()
    finally:
        store.close()
    hard_ids = {c.id for c in cases if c.question in HARD}
    split = _sweep.split_cases(cases, hard_ids, holdout=0.5, seed=3)
    tune = [c.question for c in cases if c.id in set(split.tune)]
    assert len(tune) == 5 and len(set(tune) & set(HARD)) == 3
    (project / "pass_at_4.json").write_text(json.dumps(tune), encoding="utf-8")
    capsys.readouterr()
    assert _cli.main(["sweep", "--holdout", "0.5", "--seed", "3", "--yes"]) == 0
    out = capsys.readouterr().out
    final = out.rstrip().splitlines()[-1]
    assert final.startswith("Keep the current settings. top_k=4 won on the tune cases (40% -> 100%) but lost on the holdout cases (40% -> 0%)")
    assert "holdout, 5 cases" in out

    store = Store.open(project)
    try:
        report, runs = _sweep_runs(store)
        assert report.recommended is None and report.holdout_runs is not None
        assert [runs[r][0] for r in report.holdout_runs] == [BASELINE, Setting(top_k=4)]
        assert all(runs[r][1] == 5 for r in report.holdout_runs)
        assert set(report.split.tune) == set(split.tune) and set(report.split.holdout) == set(split.holdout)
    finally:
        store.close()


def test_recommendation_states_the_change_and_the_numbers_in_plain_words(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """The user-facing contract for the final line. Defect: a recommendation that names the run
    id or the row instead of the knob to set, or that leaves out the holdout accuracy or the
    cost. The scripted app passes every case under top_k=4, on both splits."""
    _scripted_project(project, "sweep_app_apply", ALL)
    capsys.readouterr()
    assert _cli.main(["sweep", "--holdout", "0.5", "--seed", "1", "--yes"]) == 0
    out = capsys.readouterr().out
    final = out.rstrip().splitlines()[-1]
    assert final.startswith("Set top_k=4. Accuracy 40% -> 100% (holdout 40% -> 100%). Input token use is unknown; mean latency ")
    assert final.endswith(" ms.")
    assert "round 1, 5 tune cases" in out and "holdout, 5 cases" in out
    assert "round 2" not in out  # both candidates set top_k, so there is nothing to combine
    store = Store.open(project)
    try:
        report, runs = _sweep_runs(store)
        assert report.recommended is not None and runs[report.recommended][0] == Setting(top_k=4)
        assert [runs[r][0] for r in report.rounds[0]] == [Setting(top_k=4), Setting(top_k=6)] and len(report.rounds) == 1
    finally:
        store.close()


def test_max_settings_caps_the_runs_per_round(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: --max-settings ignored, so a sweep the user capped to stay inside a budget runs
    every candidate anyway."""
    _scripted_project(project, "sweep_app_cap", ALL)
    capsys.readouterr()
    assert _cli.main(["sweep", "--max-settings", "1", "--holdout", "0", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "candidates, heaviest first (1 of 2" in out
    store = Store.open(project)
    try:
        report, runs = _sweep_runs(store)
        assert [runs[r][0] for r in report.rounds[0]] == [Setting(top_k=4)] and len(report.rounds) == 1
        assert len(runs) == 2  # the baseline and the one candidate
    finally:
        store.close()


def test_estimate_counts_the_worst_case(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: a cost estimate below what the sweep can run (a round-2 combination or the holdout
    runs left out), so the user confirms a smaller bill than the one they may get."""
    spec = _scripted_project(project, "sweep_app_estimate", ALL)
    store = Store.open(project)
    try:
        source = store.latest_run("baseline")
        assert source is not None
        plan, _ = _sweep.plan_sweep(store, source, rounds=3, max_settings=5, holdout=0.3, seed=0, tolerance=1, builder=False)
        assert len(plan.candidates) == 2 and plan.round_size == 2
        assert len(plan.split.tune) == 7 and len(plan.split.holdout) == 3
        assert [len(run) for run in plan.worst_case_cases] == [7] * (1 + 2 * 3) + [3, 3]
        assert plan.worst_case_cases[0] == plan.split.tune and plan.worst_case_cases[-1] == plan.split.holdout
        assert plan.worst_case_app_calls == 55
        capsys.readouterr()
        assert _cli.main(["sweep", "--rounds", "3", "--max-settings", "5", "--yes"]) == 0
        out = capsys.readouterr().out
        app_line, estimate_line = [ln for ln in out.splitlines() if ln.startswith(("app:", "estimate:"))]
        assert app_line == f"app: up to 55 calls to {spec} (7 tune cases x (1 baseline + 2 settings x 3 rounds) + 3 holdout cases x 2, worst case)"
        assert estimate_line.startswith("estimate: 55 LLM calls")
        assert out.index(app_line) < out.index("experiment ")
        _, runs = _sweep_runs(store)
        actual = sum(traces for _, traces in runs.values())
        assert actual == 7 * 3 + 3 * 2 and actual <= plan.worst_case_app_calls
    finally:
        store.close()


def test_sweep_user_errors_and_nothing_to_do(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Boundary: no baseline run, failures that were never diagnosed, a bad --holdout, and a run
    that passed every case are one-line messages, never a traceback, and never an app call."""
    assert _cli.main(["init", "--provider", "fake"]) == 0
    assert _cli.main(["sweep", "some_app:answer", "--yes"]) == 1
    (project / "sweep_app_errors.py").write_text(APP_SOURCE % HARD, encoding="utf-8")
    (project / "pass_at_4.json").write_text("[]", encoding="utf-8")
    _write_cases(project, EASY + HARD[:1])
    assert _cli.main(["build-tests", "--import", "cases.jsonl"]) == 0
    assert _cli.main(["run", "sweep_app_errors:answer", "--yes"]) == 0
    assert _cli.main(["sweep", "--yes"]) == 1
    err = capsys.readouterr().err
    assert "no baseline run yet" in err and "none is labeled; run `contexteval diagnose` first" in err
    with pytest.raises(SystemExit):
        _cli.main(["sweep", "--holdout", "1.0", "--yes"])
    with pytest.raises(SystemExit):
        _cli.main(["sweep", "--rounds", "0", "--yes"])
    (project / "pass_at_4.json").write_text(json.dumps(ALL), encoding="utf-8")
    assert _cli.main(["diagnose", "--yes"]) == 0
    capsys.readouterr()
    assert _cli.main(["sweep", "--holdout", "0", "--yes"]) == 0
    assert "Set top_k=4." in capsys.readouterr().out
    store = Store.open(project)
    try:
        assert len([e for e in store.experiments() if e.kind == "sweep"]) == 1
    finally:
        store.close()


def test_a_holdout_that_rounds_to_zero_cases_is_refused_before_any_app_call(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Review Q5. Defect: on a set so small that the requested holdout rounds to zero cases, the
    sweep runs anyway and recommends a winner marked "unconfirmed" as if the user had opted out
    with --holdout 0. A requested holdout that holds no case is a user error before any app call,
    and the message names --holdout 0 as the opt-in."""
    _scripted_project(project, "sweep_app_tiny", ALL, questions=EASY[:1] + HARD[:1])
    capsys.readouterr()
    assert _cli.main(["sweep", "--yes"]) == 1
    out, err = capsys.readouterr()
    assert err == "error: --holdout 0.3 rounds to 0 of the 2 cases, so no winner could be confirmed; raise it, or pass --holdout 0 to tune on every case and read the result as unconfirmed\n"
    assert "experiment" not in out
    store = Store.open(project)
    try:
        assert store.experiments() == []
    finally:
        store.close()
    assert _cli.main(["sweep", "--holdout", "0", "--yes"]) == 0
    assert "(no holdout cases, so this is unconfirmed)" in _final(capsys.readouterr().out)


def test_sweep_reads_the_newest_diagnosed_baseline_and_names_it(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Review Q6. Defect: the sweep reads the newest baseline run, so a `contexteval run` made
    after `diagnose` hides every label behind "none is labeled". The source is the newest baseline
    run with a diagnosis, named in the output with the newer undiagnosed run, and --run ID picks
    a baseline run by hand; a run of another kind or an unknown id is a one-line error."""
    _scripted_project(project, "sweep_app_source", ALL)
    assert _cli.main(["run", "sweep_app_source:answer", "--yes"]) == 0
    assert _cli.main(["experiment", "top_k=4", "--yes"]) == 0
    capsys.readouterr()
    assert _cli.main(["sweep", "--holdout", "0", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "source: run 1 (the newest baseline run with diagnoses; run 2 is newer but has no diagnosis)" in out
    store = Store.open(project)
    try:
        report, _ = _sweep_runs(store)
        assert report.source_run == 1
        experiment_run = next(r.run.id for r in store.runs() if r.run.kind == "experiment")
    finally:
        store.close()
    assert _cli.main(["sweep", "--run", "1", "--holdout", "0", "--yes"]) == 0
    assert "source: run 1 (--run)\n" in capsys.readouterr().out
    assert _cli.main(["sweep", "--run", "2", "--holdout", "0", "--yes"]) == 1
    assert "run 2 has 6 failures among the tune cases and none is labeled" in capsys.readouterr().err
    assert _cli.main(["sweep", "--run", str(experiment_run), "--yes"]) == 1
    assert f"run {experiment_run} is an experiment run; the sweep starts from a baseline run" in capsys.readouterr().err
    assert _cli.main(["sweep", "--run", "99", "--yes"]) == 1
    assert "no run 99" in capsys.readouterr().err


def test_sweep_with_no_failures_or_only_unfixable_ones_runs_nothing(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: a sweep that starts an experiment and calls the app when the baseline passed every
    case, or when every label is one no setting can fix."""
    _scripted_project(project, "sweep_app_clean", [], EASY)
    capsys.readouterr()
    assert _cli.main(["sweep", "--yes"]) == 0
    assert "nothing to sweep" in capsys.readouterr().out
    store = Store.open(project)
    try:
        source = store.latest_run("baseline")
        assert source is not None
        (trace,) = store.traces(source.id)[:1]
        store.save_diagnosis(trace.id, Flaky((ReplayOutcome(ReplayStep.BASELINE_RERUN, True, "right"),)))
        store._conn.execute("UPDATE traces SET passed = 0 WHERE id = ?", (trace.id,))
        store._conn.commit()
    finally:
        store.close()
    assert _cli.main(["sweep", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "no setting can fix 1 flaky failure" in out and out.rstrip().endswith("Keep the current settings. No setting addresses the labels seen, for the reasons above.")
    store = Store.open(project)
    try:
        assert store.experiments() == []
    finally:
        store.close()


SPLIT_EVIDENCE = "alpha beta gamma delta epsilon zeta eta theta"
SPLIT_HALVES = ["alpha beta gamma delta", "epsilon zeta eta theta"]  # 22 characters each
SPLIT_QUESTION = "the split question?"
SPLIT_APP_SOURCE = '''
"""Scripted app whose chunker cut one case's evidence in two. Its own chunks are the two halves;
build(chunking), when present, merges them for a chunk_size above the app's own 22 characters or
for any overlap, so a chunking candidate fixes the split case. HARD questions are never answered,
so their evidence is never retrieved."""
import contexteval

EVIDENCE = %r
HALVES = %r
HARD = %r
SPLIT_QUESTION = %r
BUILDS = []


def _app(chunks):
    @contexteval.retriever(top_k="limit")
    def search(query: str, limit: int = 3) -> list[str]:
        return chunks[:limit]

    def answer(question: str) -> str:
        hits = search(question)
        if question in HARD:
            return "wrong"
        if question == SPLIT_QUESTION:
            return "right" if any(EVIDENCE in h for h in hits) else "wrong"
        return "right"

    return answer


answer = _app([*HALVES, "noise about nothing"])
%s
'''
SPLIT_APP_BUILD = '''
def build(chunking):
    BUILDS.append(chunking)
    merged = (chunking.chunk_size or 0) > 22 or (chunking.overlap or 0) > 0
    return _app([EVIDENCE, "noise about nothing"] if merged else [*HALVES, "noise about nothing"])
'''


def _split_project(project: Path, module: str, build: bool) -> str:
    """init, import the split case plus two easy and one hard case, run, diagnose. Returns the spec."""
    source = SPLIT_APP_SOURCE % (SPLIT_EVIDENCE, SPLIT_HALVES, HARD, SPLIT_QUESTION, SPLIT_APP_BUILD if build else "")
    (project / f"{module}.py").write_text(source, encoding="utf-8")
    lines = [
        json.dumps({"question": SPLIT_QUESTION, "expected_answer": "right", "evidence": SPLIT_EVIDENCE}),
        *(json.dumps({"question": q, "expected_answer": "right", "evidence": f"the fact behind {q}"}) for q in EASY[:2] + HARD[:1]),
    ]
    (project / "cases.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _cli.main(["init", "--provider", "fake"]) == 0
    assert _cli.main(["build-tests", "--import", "cases.jsonl"]) == 0
    assert _cli.main(["run", f"{module}:answer", "--yes"]) == 0
    assert _cli.main(["diagnose", "--yes"]) == 0
    return f"{module}:answer"


def test_sweep_with_a_builder_tries_the_chunking_the_split_signal_asks_for_and_recommends_it(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """v2 chunking through `_cli.main`. The split case is labeled missing/dropped with split gold
    (no LLM call, so the halves never reach a prompt; the inject replay confirms). Defects: the
    split signal not counted, chunk_size and overlap not proposed or not scaled from the 22
    character halves, a chunking row run on the app's own index (the split case would still
    fail), the index rebuilt per row instead of once per distinct chunking, or a final line that
    does not name a chunking knob when only chunking fixed a case."""
    spec = _split_project(project, "split_app_build", build=True)
    capsys.readouterr()
    assert _cli.main(["sweep", "--holdout", "0", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "sweep: 2 labeled failures among the tune cases (missing/not_retrieved 1, missing/dropped 1); 1 with the evidence split across two adjacent chunks; the baseline retrieved up to 3 chunks of up to 22 characters" in out
    assert "candidates, heaviest first (4 of 4; the number is how many failures ask for it): chunk_size=44 (1), overlap=6 (1), top_k=6 (1), top_k=9 (1)" in out
    assert "chunking:" not in out
    final = _final(out)
    # The app records no input tokens, so every row that passes 3 of 4 ties and latency picks
    # among them; a round-2 combination with a top_k may win, but a chunking knob is always in it.
    assert final.startswith("Set ") and ("chunk_size=44" in final or "overlap=6" in final), final
    builds = sys.modules["split_app_build"].BUILDS  # type: ignore[attr-defined]
    assert builds[:2] == [Chunking(chunk_size=44), Chunking(overlap=6)] and len(builds) == len(set(builds))
    store = Store.open(project)
    try:
        report, runs = _sweep_runs(store)
        assert report.recommended is not None and not runs[report.recommended][0].chunking.is_default
        passed = {s.trace.question: s.trace.test.passed for s in store.traces(report.recommended) if s.trace.test}
        assert passed[SPLIT_QUESTION] is True and passed[HARD[0]] is False
        assert spec == store.run(report.recommended).app
    finally:
        store.close()


def test_sweep_without_a_builder_says_what_chunking_would_buy_and_goes_on_with_retrieval(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Defects: a sweep on an app with no build(chunking) tries a chunking candidate and dies on
    its first row, or silently drops the chunking candidates so the user never learns that one
    failure pointed at chunking and what to add; or it stops instead of running the retrieval
    candidates the other labels asked for."""
    spec = _split_project(project, "split_app_nobuild", build=False)
    capsys.readouterr()
    assert _cli.main(["sweep", "--holdout", "0", "--yes"]) == 0
    out = capsys.readouterr().out
    assert (
        f"chunking: 1 failure points at chunking, but {spec} has no build(chunking), so chunk_size=44, overlap=6 cannot be tried; "
        "add `def build(chunking: contexteval.Chunking)` beside answer that rebuilds the index with contexteval.chunk_text"
    ) in out
    assert "candidates, heaviest first (2 of 2; the number is how many failures ask for it): top_k=6 (1), top_k=9 (1)" in out
    assert "round 1, 4 tune cases" in out and "chunk_size" not in out.split("candidates, heaviest first")[1].split("\n")[0]
    assert _final(out).startswith("Keep the current settings. no candidate beat the baseline on the tune cases")
    store = Store.open(project)
    try:
        report, runs = _sweep_runs(store)
        assert sorted(s.name for s, _ in runs.values()) == ["baseline", "top_k=6", "top_k=9"]
    finally:
        store.close()


PRICED_APP_SOURCE = '''
"""Scripted app whose input tokens grow with the chunks it retrieves: each answer sends the
chunks to the demo's fake LLM, so the capture hooks count the prompt. passing.json maps a chunk
count to the questions answered right at that count; any other count passes the easy ones."""
import json
from pathlib import Path

import contexteval
from contexteval.demo.app import FAKE_MODEL, _client

HARD = %r


@contexteval.retriever(top_k="limit")
def search(query: str, limit: int = 4) -> list[str]:
    return [f"chunk {i} " + "padding words that cost input tokens " * 20 for i in range(limit)]


def answer(question: str) -> str:
    hits = search(question)
    _client().chat.completions.create(model=FAKE_MODEL, messages=[{"role": "user", "content": "\\n".join(hits) + question}])
    allowed = json.loads(Path("passing.json").read_text(encoding="utf-8")).get(str(len(hits)))
    if allowed is None:
        return "wrong" if question in HARD else "right"
    return "right" if question in allowed else "wrong"
'''


def _priced_sweep(
    project: Path, capsys: pytest.CaptureFixture[str], module: str, passing: dict[int, list[str]], *flags: str, questions: list[str] = ALL
) -> tuple[str, _sweep.SweepReport, dict[int, tuple[Setting, int]]]:
    """A sweep over the priced app with --holdout 0, so every case is a tune case. The baseline
    retrieves 4 chunks and fails the 6 hard cases, so the candidates are top_k=8 and top_k=12.
    Returns the printed output, the report, and the runs."""
    (project / f"{module}.py").write_text(PRICED_APP_SOURCE % HARD, encoding="utf-8")
    (project / "passing.json").write_text(json.dumps({str(k): v for k, v in passing.items()}), encoding="utf-8")
    _write_cases(project, questions)
    assert _cli.main(["init", "--provider", "fake"]) == 0
    assert _cli.main(["build-tests", "--import", "cases.jsonl"]) == 0
    assert _cli.main(["run", f"{module}:answer", "--yes"]) == 0
    assert _cli.main(["diagnose", "--yes"]) == 0
    capsys.readouterr()
    assert _cli.main(["sweep", "--holdout", "0", *flags, "--yes"]) == 0
    out = capsys.readouterr().out
    store = Store.open(project)
    try:
        report, runs = _sweep_runs(store)
    finally:
        store.close()
    assert [runs[r][0] for r in report.rounds[0]] == [Setting(top_k=8), Setting(top_k=12)]
    return out, report, runs


def _final(out: str) -> str:
    return out.rstrip().splitlines()[-1]


EASY_29 = EASY + [f"easy question {i}?" for i in range(4, 29)]
ALL_35 = EASY_29 + HARD
TOP8_ONE_BEHIND = {8: EASY + HARD[:3], 12: EASY + HARD[:4]}
TOP8_ONE_BEHIND_OF_35 = {8: EASY_29 + HARD[:3], 12: EASY_29 + HARD[:4]}


def test_default_window_on_35_cases_picks_the_cheaper_setting_one_case_behind(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: a default window that stays 0 on a larger set, so the sweep recommends top_k=12 for
    one more passing case of 35 at half again the input tokens; or a recommendation that hides
    the setting it passed over or the window that allowed it. 5% of 35 is one case."""
    out, report, runs = _priced_sweep(project, capsys, "sweep_app_near_tie", TOP8_ONE_BEHIND_OF_35, questions=ALL_35)
    assert "near-tie window: 1 case (5% of 35)" in out
    assert report.recommended is not None and runs[report.recommended][0] == Setting(top_k=8)
    assert report.passed_over is not None and runs[report.passed_over][0] == Setting(top_k=12)
    assert _final(out).startswith("Set top_k=8 (top_k=12 passed 1 more tune case but uses 50% more input tokens). Accuracy 83% -> 91% "), _final(out)


def test_default_window_on_10_cases_is_accuracy_first(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: a fixed window of one case, 10 points of 10, so the sweep trades a real accuracy
    gain for tokens on a small set. 5% of 10 rounds down to 0."""
    out, report, runs = _priced_sweep(project, capsys, "sweep_app_small_default", TOP8_ONE_BEHIND)
    assert "near-tie window" not in out
    assert report.recommended is not None and runs[report.recommended][0] == Setting(top_k=12)
    assert report.passed_over is None
    assert _final(out).startswith("Set top_k=12. Accuracy 40% -> 80% "), _final(out)


def test_tolerance_zero_overrides_the_default_window(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: --tolerance ignored on a set large enough for a default window, so a user who asks
    for accuracy first still gets the cheaper setting that passes fewer cases."""
    out, report, runs = _priced_sweep(project, capsys, "sweep_app_tolerance_zero", TOP8_ONE_BEHIND_OF_35, "--tolerance", "0", questions=ALL_35)
    assert "near-tie window" not in out
    assert report.recommended is not None and runs[report.recommended][0] == Setting(top_k=12)
    assert report.passed_over is None
    assert _final(out).startswith("Set top_k=12. Accuracy 83% -> 94% "), _final(out)


def test_explicit_tolerance_widens_the_window_on_a_small_set(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: --tolerance 1 replaced by the 0 default on 10 cases, so a user who accepts one case
    of noise still pays for top_k=12; or the window note credits the percentage."""
    out, report, runs = _priced_sweep(project, capsys, "sweep_app_tolerance_one", TOP8_ONE_BEHIND, "--tolerance", "1")
    assert "near-tie window: 1 case (--tolerance)" in out
    assert report.recommended is not None and runs[report.recommended][0] == Setting(top_k=8)
    assert report.passed_over is not None and runs[report.passed_over][0] == Setting(top_k=12)
    assert _final(out).startswith("Set top_k=8 (top_k=12 passed 1 more tune case but uses 50% more input tokens). Accuracy 40% -> 70% "), _final(out)


def test_a_near_tie_that_does_not_beat_the_baseline_is_never_chosen(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: the tolerance window reaching down to the baseline's pass count, so top_k=8, which
    passes no more cases than the baseline, wins on tokens over top_k=12, which passes one more.
    The sweep then recommends a setting that fixes nothing, or keeps the current settings while
    a real improvement sits in the table."""
    out, report, runs = _priced_sweep(project, capsys, "sweep_app_no_beat", {8: EASY, 12: EASY + HARD[:1]}, "--tolerance", "3")
    assert report.recommended is not None and runs[report.recommended][0] == Setting(top_k=12)
    assert report.passed_over is None
    assert _final(out).startswith("Set top_k=12. Accuracy 40% -> 50% "), _final(out)


def test_sweep_rejects_a_negative_tolerance(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Boundary. Defect: --tolerance -1, which leaves no setting a contender, reaches the search
    instead of failing as a usage error before any app call."""
    assert _cli.main(["init", "--provider", "fake"]) == 0
    with pytest.raises(SystemExit):
        _cli.main(["sweep", "app:answer", "--tolerance", "-1", "--yes"])
    assert "--tolerance must be a whole number of cases, 0 or more" in capsys.readouterr().err


def test_sweep_report_round_trips_through_json() -> None:
    """Defect: a stored report that loses the holdout runs, the recommendation, or the split, so
    the dashboard cannot mark the recommended row of an older sweep."""
    report = _sweep.SweepReport(
        source_run=RunId(1),
        baseline_k=4,
        counts={FailureKind.NOT_RETRIEVED: 2, FailureKind.FLAKY: 1},
        split=_sweep.Split(5, 0.3, (CaseId("a"), CaseId("b")), (CaseId("c"),)),
        baseline_run=RunId(2),
        rounds=((RunId(3), RunId(4)), (RunId(5),)),
        holdout_runs=(RunId(6), RunId(7)),
        recommended=RunId(5),
        reason="",
        passed_over=RunId(4),
        not_applicable={"top_k=12": "no top_k parameter"},
    )
    assert _sweep.SweepReport.from_json(json.loads(json.dumps(report.to_json()))) == report
    older = {k: v for k, v in report.to_json().items() if k not in ("passed_over", "not_applicable")}
    assert _sweep.SweepReport.from_json(older) == _sweep.SweepReport.from_json({**older, "passed_over": None, "not_applicable": {}})
    kept = _sweep.SweepReport(report.source_run, 4, {}, report.split, report.baseline_run, (), None, None, "nothing beat the baseline")
    assert _sweep.SweepReport.from_json(kept.to_json()) == kept
    stopped = _sweep.SweepReport(report.source_run, 4, {}, report.split, None, (), None, None, "the sweep stopped early: KeyboardInterrupt: ")
    assert _sweep.SweepReport.from_json(json.loads(json.dumps(stopped.to_json()))) == stopped
