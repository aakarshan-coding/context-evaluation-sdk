"""P4: the grid parser, the experiment runner, gold isolation, and DoD item 3 through `_cli.main`."""

from __future__ import annotations

import json
import sys
import typing
from pathlib import Path
from statistics import mean

import pytest

from contexteval import _cli, _runner
from contexteval import _text as text
from contexteval._intervention import BASELINE, GoldToTop, InjectEvidence, OnlyGold, Order, Setting
from contexteval._store import Store
from contexteval.types import CaseId, Chunking, Evidence, Judged, TestCase, TestOutcome, Trace

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


def test_parse_grid_mixes_chunking_knobs_with_retrieval_knobs() -> None:
    """v2 chunking. Defect: chunk_size, overlap, or title rejected as an unknown knob; `on`/`off`
    parsed as a string or an int; a setting name that prints `title=True`; or a chunking knob that
    leaks into the retrieval intervention, so a chunk_size row also changes what reaches the prompt."""
    grid = _cli.parse_grid(["top_k=4,10", "title=on,off", "chunk_size=300"])
    assert grid == [
        Setting(top_k=4, title=True, chunk_size=300),
        Setting(top_k=4, title=False, chunk_size=300),
        Setting(top_k=10, title=True, chunk_size=300),
        Setting(top_k=10, title=False, chunk_size=300),
    ]
    assert grid[0].name == "top_k=4 chunk_size=300 title=on" and grid[1].name == "top_k=4 chunk_size=300 title=off"
    assert _cli.parse_grid(["overlap=0,100"]) == [Setting(overlap=0), Setting(overlap=100)]
    assert grid[0].chunking == Chunking(chunk_size=300, title=True) and Setting(top_k=8).chunking.is_default
    assert grid[0].intervention() == Setting(top_k=4).intervention()
    assert Setting.from_json(grid[0].to_json()) == grid[0]
    older = {"top_k": 10, "rerank": "none", "order": "as_is", "max_chunks": None, "compress": None}
    assert Setting.from_json(older) == Setting(top_k=10)


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        (["title=yes"], "bad value 'yes' for title; expected on|off"),
        (["chunk_size=0"], "chunk_size must be a positive integer"),
        (["overlap=-1"], "overlap must be 0 or more"),
        (["chunk_size=100", "overlap=100"], "overlap .* must be smaller than chunk_size"),
    ],
)
def test_parse_grid_rejects_chunking_values_that_cannot_chunk(tokens: list[str], message: str) -> None:
    """Boundary: a bad chunking value is a one-line UserError before any build, never a traceback
    from inside the user's build()."""
    with pytest.raises(_cli.UserError, match=message):
        _cli.parse_grid(tokens)


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
    assert all(Evidence not in (hint, *typing.get_args(hint)) for hint in typing.get_type_hints(Chunking).values())
    ints = {"chunk_size": "4,8", "overlap": "0,2"}  # jointly valid: overlap stays below chunk_size
    tokens = []
    for name, hint in hints.items():
        if typing.get_origin(hint) is typing.Literal:
            tokens.append(f"{name}={','.join(typing.get_args(hint))}")
        elif isinstance(hint, type) and issubclass(hint, Order):
            tokens.append(f"{name}={','.join(o.value for o in Order)}")
        elif bool in typing.get_args(hint):
            tokens.append(f"{name}=on,off")
        else:
            tokens.append(f"{name}={ints.get(name, '1,3')}")
    grid = _cli.parse_grid(tokens)
    assert len(grid) == 2 * 2 * 3 * 2 * 2 * 2 * 2 * 2
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
    eid = _runner.run_experiment(store, _runner.LoadedApp("app:answer", app), [case], grade, grid)
    runs = [r for r in reversed(store.runs()) if r.experiment_id == eid]
    assert [r.run.setting for r in runs] == [BASELINE, Setting(top_k=10), Setting(order=Order.ENDS)]
    assert {r.run.kind for r in runs} == {"experiment"} and all(r.traces == 1 for r in runs)
    assert seen == ["q?"] * 3
    assert [row.setting_name for row in store.experiment_table(eid)] == ["baseline", "top_k=10", "order=ends"]


BUILD_APP_SOURCE = '''
"""Scripted app with the optional builder. Its own index answers "own index"; a built index
answers with the chunking it was built for, so a trace shows which index answered it."""
import sys

import contexteval

BUILDS = []  # every chunking build() was asked for, in order


@contexteval.retriever(top_k="limit")
def search(query: str, limit: int = 2) -> list[str]:
    return [f"chunk {i} about {query}" for i in range(limit)]


def answer(question: str) -> str:
    search(question)
    return "own index"


def build(chunking: contexteval.Chunking):
    BUILDS.append(chunking)

    def built(question: str) -> str:
        search(question)
        return f"index {chunking.name}"

    return built
'''


def _builder_project(project: Path, module: str, expected: str) -> None:
    (project / f"{module}.py").write_text(BUILD_APP_SOURCE, encoding="utf-8")
    cases = project / "cases.jsonl"
    cases.write_text(json.dumps({"question": "which index?", "expected_answer": expected, "evidence": "index facts"}) + "\n", encoding="utf-8")
    assert _cli.main(["init", "--provider", "fake"]) == 0
    assert _cli.main(["build-tests", "--import", str(cases)]) == 0


def test_experiment_builds_each_distinct_chunking_once_and_runs_every_row_on_its_index(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """v2 chunking through `_cli.main`. Defects: a chunking row run on the app's own index (the
    answer would read "own index"), a build per row or per case instead of one per distinct
    Chunking, a baseline that calls build(), or retrieval knobs that stop combining with chunking
    knobs (top_k must still reach the retrieval under a chunk_size row)."""
    _builder_project(project, "build_app_grid", "index chunk_size=800")
    assert _cli.main(["run", "build_app_grid:answer", "--yes"]) == 0
    capsys.readouterr()
    assert _cli.main(["experiment", "chunk_size=300,800", "top_k=2,4", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "1 cases x 5 settings" in out and "2 index builds (chunk_size=300, chunk_size=800)" in out
    assert sys.modules["build_app_grid"].BUILDS == [Chunking(chunk_size=300), Chunking(chunk_size=800)]  # type: ignore[attr-defined]
    store = Store.open(project)
    try:
        eid = store.experiments()[0].id
        runs = [r for r in reversed(store.runs()) if r.experiment_id == eid]
        answers = {r.run.setting.name: [s.trace for s in store.traces(r.run.id)][0] for r in runs}
    finally:
        store.close()
    assert {name: t.outcome.text for name, t in answers.items()} == {  # type: ignore[union-attr]
        "baseline": "own index",
        "top_k=2 chunk_size=300": "index chunk_size=300",
        "top_k=4 chunk_size=300": "index chunk_size=300",
        "top_k=2 chunk_size=800": "index chunk_size=800",
        "top_k=4 chunk_size=800": "index chunk_size=800",
    }
    assert {name: len(t.retrievals[0].delivered) for name, t in answers.items()} == {"baseline": 2, "top_k=2 chunk_size=300": 2, "top_k=4 chunk_size=300": 4, "top_k=2 chunk_size=800": 2, "top_k=4 chunk_size=800": 4}
    assert {name: t.test.passed for name, t in answers.items()}["top_k=4 chunk_size=800"] is True  # type: ignore[union-attr]
    assert {name: t.test.passed for name, t in answers.items()}["baseline"] is False  # type: ignore[union-attr]


def test_experiment_without_a_builder_explains_the_contract_before_any_call(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: `contexteval experiment title=on` on an app with no build() runs the baseline, prints
    the cost prompt, and then fails on the first chunking row with a traceback; or a retrieval-only
    grid on the same app is refused too. The error is one line before any app call, names what to
    add and where, and the retrieval knobs still work."""
    (project / "plain_app.py").write_text(
        "import contexteval\n\n@contexteval.retriever(top_k='limit')\ndef search(q, limit=2):\n    return ['a', 'b'][:limit]\n\ndef answer(q):\n    search(q)\n    return 'a'\n",
        encoding="utf-8",
    )
    cases = project / "cases.jsonl"
    cases.write_text(json.dumps({"question": "q?", "expected_answer": "a"}) + "\n", encoding="utf-8")
    assert _cli.main(["init", "--provider", "fake"]) == 0
    assert _cli.main(["build-tests", "--import", str(cases)]) == 0
    assert _cli.main(["experiment", "plain_app:answer", "title=on", "top_k=4", "--yes"]) == 1
    out, err = capsys.readouterr()
    assert err.startswith("error: title=on needs the app to rebuild its index, and plain_app:answer has no plain_app:build")
    assert "def build(chunking: contexteval.Chunking)" in err and "chunk_text" in err and "estimate:" not in out
    store = Store.open(project)
    try:
        assert store.experiments() == [] and store.runs() == []
    finally:
        store.close()
    assert _cli.main(["experiment", "plain_app:answer", "top_k=1", "--yes"]) == 0


def test_diagnose_replays_an_experiment_run_against_the_index_it_was_built_on(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: `diagnose --run ID` on a chunk_size=800 experiment run replays the failure through
    the app's own index, so the replay answers (and the label) describe a different app than the
    one that failed. The replays carry the built index's answer."""
    _builder_project(project, "build_app_diag", "index chunk_size=300")
    assert _cli.main(["run", "build_app_diag:answer", "--yes"]) == 0
    assert _cli.main(["experiment", "chunk_size=800", "--yes"]) == 0
    store = Store.open(project)
    try:
        built_run = next(r.run.id for r in store.runs() if r.run.setting == Setting(chunk_size=800))
    finally:
        store.close()
    capsys.readouterr()
    assert _cli.main(["diagnose", "--run", str(built_run), "--yes"]) == 0
    assert "missing      which index?" in capsys.readouterr().out
    assert sys.modules["build_app_diag"].BUILDS == [Chunking(chunk_size=800)] * 2  # type: ignore[attr-defined]
    store = Store.open(project)
    try:
        (stored,) = store.traces(built_run)
    finally:
        store.close()
    assert stored.diagnosis is not None and not isinstance(stored.diagnosis, Judged)
    assert {r.answer for r in stored.diagnosis.replays} == {"index chunk_size=800"}  # type: ignore[union-attr]


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
    dropped case would then pass too). The flaky planted case failed only its first call (the
    `run`), so it passes in every experiment row.
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
    n = len(planted.PLANTED)
    assert f"{n} cases x 3 settings" in out and f"estimate: {3 * n} LLM calls" in out

    store = Store.open(project)
    try:
        eid = store.experiments()[0].id
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

    after_first_call = {p.case.id: p.passes_baseline or p.expected is Label.FLAKY for p in planted.PLANTED}
    assert outcomes("baseline") == outcomes("top_k=4") == after_first_call
    assert outcomes("top_k=10") == {p.case.id: after_first_call[p.case.id] or p.cause == "not_retrieved" for p in planted.PLANTED}
    assert by_name["top_k=10"].passed == by_name["baseline"].passed + 1

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
