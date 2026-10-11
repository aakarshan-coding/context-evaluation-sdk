"""v2 auto-sweep: the label-to-candidates policy, the seeded split, the greedy search, and the
recommendation, through `_cli.main` where the user would see it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contexteval import _cli, _sweep
from contexteval._intervention import BASELINE, GoldToTop, InjectEvidence, OnlyGold, Order, Setting
from contexteval._store import Store
from contexteval._sweep import FailureKind, Fixable, Unfixable
from contexteval.types import (
    CaseId,
    Evidence,
    Flaky,
    Ignored,
    Irrelevant,
    Judged,
    JudgeVerdict,
    Label,
    Missing,
    NotContext,
    ReplayOutcome,
    ReplayStep,
    RunId,
    TestCase,
)

GOLD_TRANSFORMS = (InjectEvidence, GoldToTop, OnlyGold)
RERUN = (ReplayOutcome(ReplayStep.BASELINE_RERUN, False, "wrong"),)
K = 4


def _knobs(setting: Setting) -> dict[str, object]:
    return {name: value for name, value in setting.to_json().items() if value != BASELINE.to_json()[name]}


# --- policy -----------------------------------------------------------------------


def test_policy_maps_each_label_to_its_candidates_and_the_unfixable_to_none() -> None:
    """Defect: a label whose candidates do not address it (a top_k change for an irrelevant case,
    a reorder for a never-retrieved chunk), a fixable label with no candidates, or dropped,
    not_context, and flaky offered a setting when no setting can change them."""
    assert set(_sweep.POLICY) == set(FailureKind)
    resolved = {kind: [_knobs(s) for s in _sweep.candidates_for(kind, K)] for kind in FailureKind}

    assert resolved[FailureKind.NOT_RETRIEVED] and all(set(c) == {"top_k"} and c["top_k"] > K for c in resolved[FailureKind.NOT_RETRIEVED])  # type: ignore[operator]
    irrelevant = resolved[FailureKind.IRRELEVANT]
    assert {"rerank": "lexical"} in irrelevant
    assert any(set(c) == {"max_chunks"} and c["max_chunks"] < K for c in irrelevant)  # type: ignore[operator]
    assert any(set(c) == {"compress"} for c in irrelevant)
    ignored = resolved[FailureKind.IGNORED]
    assert {"order": "ends"} in ignored and {"order": "reverse"} in ignored and any(set(c) == {"compress"} for c in ignored)

    for kind in (FailureKind.DROPPED, FailureKind.NOT_CONTEXT, FailureKind.FLAKY):
        rule = _sweep.POLICY[kind]
        assert isinstance(rule, Unfixable) and rule.reason and resolved[kind] == []
    assert "your own code" in _sweep.POLICY[FailureKind.DROPPED].reason  # type: ignore[union-attr]
    for kind in (FailureKind.NOT_RETRIEVED, FailureKind.IRRELEVANT, FailureKind.IGNORED):
        assert isinstance(_sweep.POLICY[kind], Fixable)


def test_relative_candidates_follow_the_observed_baseline_k_and_drop_no_ops() -> None:
    """Defect: top_k candidates fixed at 10 whatever the app retrieves (a no-op for an app at
    k=12), or a max_chunks candidate at or above k, which changes nothing and wastes a run."""
    assert [_knobs(s)["top_k"] for s in _sweep.candidates_for(FailureKind.NOT_RETRIEVED, 12)] == [24, 36]
    assert [_knobs(s)["top_k"] for s in _sweep.candidates_for(FailureKind.NOT_RETRIEVED, 1)] == [2, 3]
    assert all(_knobs(s).get("max_chunks", 0) < 12 for s in _sweep.candidates_for(FailureKind.IRRELEVANT, 12))
    assert not any("max_chunks" in _knobs(s) for s in _sweep.candidates_for(FailureKind.IRRELEVANT, 1))


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
    ranked = _sweep.rank_candidates(counts, K)
    names = [r.setting.name for r in ranked]
    assert names[:2] == ["top_k=8", "top_k=12"] and [r.weight for r in ranked[:2]] == [3, 3]
    compress = next(r for r in ranked if r.setting.name.startswith("compress="))
    assert compress.weight == 2 and set(compress.kinds) == {FailureKind.IGNORED, FailureKind.IRRELEVANT}
    assert names.index(compress.setting.name) == 2
    assert len(names) == len(set(names))
    assert all(r.weight >= 1 for r in ranked) and ranked == sorted(ranked, key=lambda r: -r.weight)
    assert _sweep.rank_candidates({FailureKind.DROPPED: 2, FailureKind.FLAKY: 1}, K) == []


def test_no_candidate_setting_can_carry_gold() -> None:
    """Plan item 15 for the sweep. Defect: a candidate built from a Diagnosis or a TestCase that
    reaches the app with evidence in it, so a sweep row wins by peeking at the answer."""
    counts = {kind: 1 for kind in FailureKind}
    for ranked in _sweep.rank_candidates(counts, K):
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


def test_demo_sweep_recommends_a_larger_top_k_and_names_the_dropped_case_as_unfixable(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Through `_cli.main` on the planted demo. Defects: the planted not_retrieved case does not
    lead to a top_k candidate, the recommended setting does not carry the larger top_k that fixes
    it, the dropped case is offered a setting instead of the reason none can help, --holdout 0 is
    refused or not warned about, or the sweep is stored as a manual experiment."""
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
    assert "no setting can fix 1 not_context failure" in out and "no setting can fix 1 flaky failure" in out
    assert "warning: no holdout cases" in out
    assert "candidates, heaviest first" in out and "top_k=8 (1)" in out and "top_k=12 (1)" in out
    final = out.rstrip().splitlines()[-1]
    assert final.startswith("Set ") and "top_k=8" in final and "(no holdout cases, so this is unconfirmed)" in final

    store = Store.open(project)
    try:
        report, runs = _sweep_runs(store)
        assert report.recommended is not None and report.holdout_runs is None and report.split.holdout == ()
        recommended, traces = runs[report.recommended]
        assert recommended.top_k == 8 and traces == len(planted.PLANTED)
        assert report.counts[FailureKind.NOT_RETRIEVED] == 1 and report.counts[FailureKind.DROPPED] == 1
        for setting, _ in runs.values():
            assert not any(isinstance(t, GOLD_TRANSFORMS) for t in setting.intervention().transforms)
        (not_retrieved,) = [p for p in planted.PLANTED if p.cause == "not_retrieved"]
        passed = {s.trace.test.case_id: s.trace.test.passed for s in store.traces(report.recommended) if s.trace.test}
        baseline = {s.trace.test.case_id: s.trace.test.passed for s in store.traces(report.baseline_run) if s.trace.test}
        assert passed[not_retrieved.case.id] is True and baseline[not_retrieved.case.id] is False
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
        plan, _ = _sweep.plan_sweep(store, source, rounds=3, max_settings=5, holdout=0.3, seed=0)
        assert len(plan.candidates) == 2 and plan.round_size == 2
        assert len(plan.split.tune) == 7 and len(plan.split.holdout) == 3
        assert plan.worst_case_runs == [7] * (1 + 2 * 3) + [3, 3]
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
    )
    assert _sweep.SweepReport.from_json(json.loads(json.dumps(report.to_json()))) == report
    kept = _sweep.SweepReport(report.source_run, 4, {}, report.split, report.baseline_run, (), None, None, "nothing beat the baseline")
    assert _sweep.SweepReport.from_json(kept.to_json()) == kept
