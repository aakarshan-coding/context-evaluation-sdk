"""P4: the grid parser, the experiment runner, gold isolation, and DoD item 3 through `_cli.main`."""

from __future__ import annotations

import typing
from pathlib import Path
from statistics import mean

import pytest

from contexteval import _cli, _runner
from contexteval import _text as text
from contexteval._intervention import BASELINE, GoldToTop, InjectEvidence, OnlyGold, Order, Setting
from contexteval._store import Store
from contexteval.types import CaseId, Evidence, TestCase, TestOutcome, Trace

GOLD_TRANSFORMS = (InjectEvidence, GoldToTop, OnlyGold)


# --- parse_grid ---------------------------------------------------------------


def test_parse_grid_is_the_cartesian_product_in_token_order() -> None:
    """Defect: a grid that zips instead of crossing its axes, drops a value, or parses `ends` as
    a string the intervention compiler does not recognise."""
    grid = _cli.parse_grid(["top_k=4,10", "order=as_is,ends", "rerank=lexical"])
    assert grid == [
        Setting(top_k=4, rerank="lexical"),
        Setting(top_k=4, rerank="lexical", order=Order.ENDS),
        Setting(top_k=10, rerank="lexical"),
        Setting(top_k=10, rerank="lexical", order=Order.ENDS),
    ]
    assert _cli.parse_grid(["max_chunks=2", "compress=1"]) == [Setting(max_chunks=2, compress=1)]


def test_parse_grid_knobs_come_from_setting_fields() -> None:
    """Defect: a knob added to Setting but missing from a hand-kept list in the CLI, so the
    grid rejects it as unknown."""
    hints = typing.get_type_hints(Setting)
    assert set(_cli._knob_parsers()) == set(hints)


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        (["temperature=0,1"], "unknown knob 'temperature'"),
        (["top_k=three"], "bad value 'three' for top_k"),
        (["rerank=cross-encoder"], "bad value 'cross-encoder' for rerank"),
        (["order=middle"], "bad value 'middle' for order"),
        (["top_k=0"], "top_k must be a positive integer"),
        (["top_k"], "must look like knob=v1,v2"),
        (["top_k=4", "top_k=10"], "given twice"),
    ],
)
def test_parse_grid_names_the_bad_key_or_value(tokens: list[str], message: str) -> None:
    """Boundary: a typo in the grid is a UserError naming the offending knob or value, never a
    traceback and never a silently ignored axis."""
    with pytest.raises(_cli.UserError, match=message):
        _cli.parse_grid(tokens)


# --- gold isolation -------------------------------------------------------------


def test_experiments_cannot_use_gold() -> None:
    """Plan item 15. Defects: a Setting field that can carry evidence, or a knob value whose
    compiled Intervention contains a gold transform, which would let an experiment row see the
    answer. The full grid crosses every value of every enumerable knob, read from Setting itself."""
    hints = typing.get_type_hints(Setting)
    assert all(Evidence not in (hint, *typing.get_args(hint)) for hint in hints.values())
    tokens = []
    for name, hint in hints.items():
        if typing.get_origin(hint) is typing.Literal:
            tokens.append(f"{name}={','.join(typing.get_args(hint))}")
        elif isinstance(hint, type) and issubclass(hint, Order):
            tokens.append(f"{name}={','.join(o.value for o in Order)}")
        else:
            tokens.append(f"{name}=1,3")
    grid = _cli.parse_grid(tokens)
    assert len(grid) == 2 * 2 * 3 * 2 * 2
    for setting in [BASELINE, *grid]:
        assert not any(isinstance(t, GOLD_TRANSFORMS) for t in setting.intervention().transforms), setting.name


# --- run_experiment -------------------------------------------------------------


def test_run_experiment_prepends_one_baseline_and_stores_each_setting(store: Store) -> None:
    """Defects: no baseline row, a duplicate baseline when the grid already holds it, a run
    stored without its setting, or a run outside the experiment."""
    case = TestCase(CaseId("c1"), "q?", "a", None, "test")
    store.upsert_cases("t", [case])
    seen: list[str] = []

    def app(question: str) -> str:
        seen.append(question)
        return "a"

    def grade(c: TestCase, trace: Trace) -> TestOutcome:
        return TestOutcome(c.id, True, "substring")

    grid = [Setting(top_k=10), BASELINE, Setting(top_k=10), Setting(order=Order.ENDS)]
    eid = _runner.run_experiment(store, "app:answer", app, [case], grade, grid)
    runs = [r for r in reversed(store.runs()) if r.experiment_id == eid]
    assert [r.run.setting for r in runs] == [BASELINE, Setting(top_k=10), Setting(order=Order.ENDS)]
    assert {r.run.kind for r in runs} == {"experiment"} and all(r.traces == 1 for r in runs)
    assert seen == ["q?"] * 3
    assert [row.setting_name for row in store.experiment_table(eid)] == ["baseline", "top_k=10", "order=ends"]


def test_experiment_command_user_errors(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Boundary: a bad grid, a missing app, or no cases is a one-line error with exit 1, checked
    before any app import or cost prompt."""
    assert _cli.main(["init", "--provider", "fake"]) == 0
    assert _cli.main(["experiment", "top_k=x", "--yes"]) == 1
    assert _cli.main(["experiment", "top_k=10", "--yes"]) == 1
    assert _cli.main(["experiment", "my_app:answer", "top_k=10", "--yes"]) == 1
    assert _cli.main(["experiment", "my_app:answer", "--yes"]) == 1
    err = capsys.readouterr().err
    assert "bad value 'x' for top_k" in err and "no app given" in err and "no test cases" in err and "at least one" in err


# --- DoD item 3 ------------------------------------------------------------------


def test_missing_case_passes_with_larger_top_k(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """DoD item 3 through `_cli.main`: init, import the planted cases, run, `experiment top_k=4,10`.

    Defects caught: the top_k override not reaching the store call (the not_retrieved case would
    still fail at top_k=10); a table missing the baseline or a setting row; accuracy, token, or
    latency columns that disagree with the stored traces; the experiment reading gold (the
    dropped case would then pass too). The flaky planted case alternates answers, so it is left
    out of the per-case expectations.
    """
    pytest.importorskip("chromadb")
    from contexteval.demo import app as demo_app
    from contexteval.demo import planted
    from contexteval.types import Label

    assert _cli.main(["init", "--provider", "fake"]) == 0
    jsonl = planted.write_jsonl(project / planted.IMPORT_FILE)
    assert _cli.main(["build-tests", "--import", str(jsonl)]) == 0
    assert _cli.main(["run", "contexteval.demo.app:answer", "--yes"]) == 0
    capsys.readouterr()
    assert _cli.main(["experiment", "top_k=4,10", "--yes"]) == 0  # app from config
    out = capsys.readouterr().out
    assert "8 cases x 3 settings" in out and "estimate: 24 LLM calls" in out

    store = Store.open(project)
    try:
        (eid, _), *_ = store.experiments()
        runs = [r for r in reversed(store.runs()) if r.experiment_id == eid]
        traces = {r.run.setting.name: store.traces(r.run.id) for r in runs}
        table = store.experiment_table(eid)
    finally:
        store.close()

    assert [row.setting_name for row in table] == ["baseline", "top_k=4", "top_k=10"] == list(traces)
    for row in table:
        stored = traces[row.setting_name]
        assert row.cases == len(stored) == len(planted.PLANTED)
        assert row.passed == sum(s.trace.test.passed for s in stored)  # type: ignore[union-attr]
        assert row.mean_input_tokens == pytest.approx(mean(s.trace.metrics.input_tokens for s in stored))
        assert row.mean_latency_ms == pytest.approx(mean(s.trace.metrics.latency_ms for s in stored))
        line = next(ln for ln in out.splitlines() if ln.split()[:1] == [row.setting_name])
        assert f"{row.accuracy:.0%}" in line and f"{row.passed}/{row.cases}" in line
        assert f"{row.mean_input_tokens:.0f}" in line and f"{row.mean_latency_ms:.1f}" in line
    by_name = {row.setting_name: row for row in table}
    assert by_name["top_k=10"].mean_input_tokens > by_name["baseline"].mean_input_tokens

    def outcomes(name: str) -> dict[CaseId, bool]:
        return {s.trace.test.case_id: s.trace.test.passed for s in traces[name] if s.trace.test}

    steady = [p for p in planted.PLANTED if p.expected is not Label.FLAKY]
    assert {p.case.id: outcomes("baseline")[p.case.id] for p in steady} == {p.case.id: p.passes_baseline for p in steady}
    assert {p.case.id: outcomes("top_k=4")[p.case.id] for p in steady} == {p.case.id: p.passes_baseline for p in steady}
    assert {p.case.id: outcomes("top_k=10")[p.case.id] for p in steady} == {
        p.case.id: p.passes_baseline or p.cause == "not_retrieved" for p in steady
    }

    (missing,) = [p for p in planted.PLANTED if p.cause == "not_retrieved"]
    evidence = missing.case.evidence
    assert evidence is not None

    def trace_of(name: str) -> Trace:
        return next(s.trace for s in traces[name] if s.trace.test and s.trace.test.case_id == missing.case.id)

    base, wide = trace_of("baseline"), trace_of("top_k=10")
    assert base.retrievals[0].params["n_results"] == demo_app.TOP_K
    assert wide.retrievals[0].params["n_results"] == 10
    assert not any(text.contains_evidence(c.text, evidence) for c in base.retrievals[0].delivered)
    assert any(text.contains_evidence(c.text, evidence) for c in wide.retrievals[0].delivered)
    assert not text.contains_evidence(base.prompt_text, evidence)
    assert text.contains_evidence(wide.prompt_text, evidence)
    assert base.test is not None and not base.test.passed
    assert wide.test is not None and wide.test.passed
