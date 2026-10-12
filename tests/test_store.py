"""Store round-trips for every stored type, and the schema invariants that mirror the types."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from contexteval._intervention import BASELINE, Order, Setting
from contexteval._store import PROJECT_DIR_NAME, SCHEMA_VERSION, Store, find_project
from contexteval.types import (
    Answered,
    CaseId,
    Chunk,
    Crashed,
    Evidence,
    Flaky,
    Ignored,
    Irrelevant,
    Judged,
    JudgeVerdict,
    LLMCall,
    Metrics,
    Missing,
    NotContext,
    ReplayOutcome,
    ReplayStep,
    RetrievalEvent,
    StepTally,
    TestCase,
    TestOutcome,
    Trace,
    explain,
    label_of,
    tallies,
)

CASE = TestCase(CaseId("abc123"), "When does the refund window close?", "thirty days", Evidence("closes thirty days after"), "docs/policy.md")
NO_EVIDENCE = TestCase(CaseId("def456"), "Who wrote it?", "nobody", None, "tests.jsonl")


def _trace(*, crashed: bool = False, test: TestOutcome | None = None, cost: float | None = 0.001) -> Trace:
    chunks = (Chunk("closes thirty days after delivery", "p1", -0.2, 0), Chunk("unrelated", "p2", -0.9, 1))
    event = RetrievalEvent("chromadb", "refund?", {"n_results": 2, "query_texts": ["refund?"]}, chunks, chunks[:1], 3.5)
    call = LLMCall("openai", "gpt-4o-mini", "user: closes thirty days after delivery\n\nQ: refund?", "thirty days", 20, 4, 120.0)
    return Trace(
        question="refund?",
        outcome=Crashed("ValueError: boom") if crashed else Answered("thirty days"),
        retrievals=(event,),
        llm_calls=() if crashed else (call,),
        metrics=Metrics(20, 4, 130.0, cost),
        test=test,
        in_prompt=((0, 0),),
    )


def test_open_is_idempotent_and_creates_project_dir(tmp_path: Path) -> None:
    """Defect: a second open re-running the schema fails, or the store writes outside .contexteval/."""
    s1 = Store.open(tmp_path)
    s1.close()
    s2 = Store.open(tmp_path)
    assert (tmp_path / PROJECT_DIR_NAME / "contexteval.db").is_file()
    assert s2._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    s2.close()
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project(nested) == tmp_path
    assert find_project(Path(tmp_path.anchor)) in (None, Path(tmp_path.anchor))


def test_case_round_trip_and_upsert_counts_new_ids_only(store: Store) -> None:
    """Defect: a rebuilt test set duplicates cases, or the count reports replaced rows as new."""
    assert store.upsert_cases("synthetic", [CASE, NO_EVIDENCE]) == 2
    assert store.upsert_cases("synthetic", [CASE]) == 0
    assert store.cases() == [CASE, NO_EVIDENCE]
    assert store.cases("synthetic") == [CASE, NO_EVIDENCE]
    assert store.cases("other") == []
    assert store.case(CASE.id) == CASE
    assert store.case(CaseId("nope")) is None
    assert store.case(NO_EVIDENCE.id).evidence is None  # type: ignore[union-attr]


def test_run_round_trip_and_experiment_check(store: Store) -> None:
    """Defect: the run's Setting lost on reload (every trace would read as baseline), or an
    experiment run accepted without an experiment id."""
    setting = Setting(top_k=10, order=Order.ENDS)
    rid = store.new_run("baseline", "app:answer", setting)
    info = store.run(rid)
    assert (info.kind, info.app, info.setting) == ("baseline", "app:answer", setting)
    assert store.latest_run("baseline") == info
    assert store.latest_run("production") is None
    with pytest.raises(sqlite3.IntegrityError):
        store.new_run("experiment", "app:answer", setting, experiment=None)
    with pytest.raises(KeyError):
        store.run(rid + 99)  # type: ignore[operator]


def test_trace_round_trip_graded_and_crashed(store: Store) -> None:
    """Defect: retrieved/delivered collapsed into one list, in_prompt dropped, token usage lost, or
    a crashed trace reloaded as Answered('')."""
    store.upsert_cases("synthetic", [CASE])
    rid = store.new_run("baseline", "app:answer", BASELINE)
    graded = _trace(test=TestOutcome(CASE.id, True, "substring"))
    crashed = _trace(crashed=True, test=TestOutcome(CASE.id, False, "crash"))
    ungraded = _trace()
    ids = [store.add_trace(rid, t) for t in (graded, crashed, ungraded)]
    stored = store.traces(rid)
    assert [s.id for s in stored] == ids
    assert [s.trace for s in stored] == [graded, crashed, ungraded]
    assert stored[0].setting == BASELINE and stored[0].diagnosis is None
    assert store.trace(ids[1]).trace.outcome == Crashed("ValueError: boom")  # type: ignore[union-attr]
    assert [s.id for s in store.traces(rid, failed_only=True)] == [ids[1]]
    assert store.trace(ids[-1] + 50) is None  # type: ignore[operator]


@pytest.mark.parametrize("case_id,passed", [(None, 1), ("abc123", None)], ids=["passed-without-case", "case-without-passed"])
def test_trace_schema_rejects_graded_without_case(store: Store, case_id: str | None, passed: int | None) -> None:
    """Defect: a passed flag stored for a production trace, which the UI would then count, or a
    case trace with no pass flag. Review Y8: a raw insert, because add_trace cannot build either
    row and a missing case id trips the foreign key before the CHECK is ever reached."""
    store.upsert_cases("synthetic", [CASE])
    rid = store.new_run("production", None, BASELINE)
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        store._conn.execute(
            "INSERT INTO traces (run_id, case_id, question, answer, passed, retrievals, llm_calls, in_prompt,"
            " input_tokens, output_tokens, latency_ms) VALUES (?, ?, 'q', 'a', ?, '[]', '[]', '[]', 0, 0, 0.0)",
            (rid, case_id, passed),
        )


REPLAYS = (ReplayOutcome(ReplayStep.BASELINE_RERUN, False, "wrong"), ReplayOutcome(ReplayStep.GOLD_TO_TOP, True, "right"))
DIAGNOSES = [
    Missing("not_retrieved", True, REPLAYS[:1] + (ReplayOutcome(ReplayStep.INJECT_EVIDENCE, True, "right"),)),
    Missing("dropped", False, REPLAYS[:1]),
    Irrelevant(REPLAYS, removed=((0, 1), (0, 2))),
    Ignored(REPLAYS, gold_at=((0, 2),)),
    Ignored(REPLAYS, gold_at=((0, 1), (0, 2)), split_gold=True),
    Missing("dropped", True, REPLAYS[:1], split_gold=True),
    NotContext(REPLAYS),
    NotContext(REPLAYS, split_gold=True),
    Flaky((ReplayOutcome(ReplayStep.BASELINE_RERUN, True, "right"),)),
    Judged(JudgeVerdict.NO_FAILURE, "answer agrees with context"),
    Judged(JudgeVerdict.IGNORED, "the gold chunk was present and unused"),
]


@pytest.mark.parametrize("diagnosis", DIAGNOSES, ids=lambda d: f"{type(d).__name__}{'-split' if getattr(d, 'split_gold', False) else ''}")
def test_diagnosis_round_trip_every_variant(store: Store, diagnosis) -> None:
    """Defect: a variant's evidence fields (graft 1: removed, gold_at; Missing.cause/confirmed;
    the judge verdict; v2's split_gold) not stored, so the UI cannot say what fixed it. A row
    written before split_gold existed reads as not split (the v1 test below)."""
    rid = store.new_run("baseline", "app:answer", BASELINE)
    tid = store.add_trace(rid, _trace())
    store.save_diagnosis(tid, diagnosis)
    loaded = store.trace(tid)
    assert loaded is not None and loaded.diagnosis == diagnosis
    label, source = store._conn.execute("SELECT label, source FROM diagnoses WHERE trace_id = ?", (tid,)).fetchone()
    assert label == label_of(diagnosis).value
    assert source == ("judge" if isinstance(diagnosis, Judged) else "ground_truth")


def test_save_diagnosis_twice_replaces_not_duplicates(store: Store) -> None:
    """Idempotence (graft 2): a second diagnose run on the same trace leaves one row."""
    rid = store.new_run("baseline", "app:answer", BASELINE)
    tid = store.add_trace(rid, _trace())
    store.save_diagnosis(tid, NotContext(REPLAYS))
    store.save_diagnosis(tid, Flaky(REPLAYS[:1]))
    assert store._conn.execute("SELECT COUNT(*) FROM diagnoses").fetchone()[0] == 1
    assert store.trace(tid).diagnosis == Flaky(REPLAYS[:1])  # type: ignore[union-attr]


def test_experiment_table_aggregates_per_run(store: Store) -> None:
    """Defect: accuracy counted over all runs, or a total cost reported when a trace was unpriced."""
    store.upsert_cases("synthetic", [CASE])
    eid = store.new_experiment()
    base = store.new_run("experiment", "app:answer", BASELINE, experiment=eid)
    big = store.new_run("experiment", "app:answer", Setting(top_k=10), experiment=eid)
    store.add_trace(base, _trace(test=TestOutcome(CASE.id, False, "llm")))
    store.add_trace(base, _trace(test=TestOutcome(CASE.id, True, "substring")))
    store.add_trace(big, _trace(test=TestOutcome(CASE.id, True, "substring"), cost=None))
    rows = store.experiment_table(eid)
    assert [(r.run_id, r.setting_name, r.cases, r.passed, r.accuracy) for r in rows] == [(base, "baseline", 2, 1, 0.5), (big, "top_k=10", 1, 1, 1.0)]
    assert rows[0].mean_input_tokens == 20 and rows[0].mean_latency_ms == 130.0
    assert rows[0].total_cost_usd == pytest.approx(0.002)
    assert rows[1].total_cost_usd is None
    (info,) = store.experiments()
    assert (info.id, info.kind, info.report) == (eid, "manual", None) and store.experiment(eid) == info
    assert store.experiment_table(eid + 1) == []  # type: ignore[operator]
    with pytest.raises(KeyError):
        store.experiment(eid + 1)  # type: ignore[operator]


def test_sweep_experiment_keeps_its_kind_and_report(store: Store) -> None:
    """Defect: a sweep is stored as a plain experiment, so the dashboard and the CLI cannot tell
    it from a hand-run grid or find its recommendation; or the report does not round-trip."""
    manual = store.new_experiment()
    sweep = store.new_experiment("sweep")
    report = {"recommended_run": 7, "rounds": [[2, 3]], "labels": {"missing/not_retrieved": 2}}
    store.save_experiment_report(sweep, report)
    assert [(e.id, e.kind, e.report) for e in store.experiments()] == [(sweep, "sweep", report), (manual, "manual", None)]
    store.save_experiment_report(sweep, {"recommended_run": None})
    assert store.experiment(sweep).report == {"recommended_run": None}
    with pytest.raises(sqlite3.IntegrityError):
        store.new_experiment("guess")  # type: ignore[arg-type]


def test_v1_database_migrates_and_a_newer_one_is_refused(tmp_path: Path) -> None:
    """Defect: a project created by 0.2.0 (schema 1, experiments without kind or report) fails to
    open after upgrading, or its experiments come back without a kind; a database written by a
    newer build must still be refused instead of read with the wrong columns."""
    directory = tmp_path / PROJECT_DIR_NAME
    directory.mkdir()
    conn = sqlite3.connect(str(directory / "contexteval.db"))
    conn.executescript(
        "CREATE TABLE experiments (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL);"
        "INSERT INTO experiments (created_at) VALUES ('2026-10-11T00:00:00+00:00');"
        "PRAGMA user_version=1;"
    )
    conn.close()
    store = Store.open(tmp_path)
    try:
        (old,) = store.experiments()
        assert (old.kind, old.report, old.created_at) == ("manual", None, "2026-10-11T00:00:00+00:00")
        assert store.new_experiment("sweep") == old.id + 1
        assert store._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        store.close()
    newer = sqlite3.connect(str(directory / "contexteval.db"))
    newer.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
    newer.commit()
    newer.close()
    with pytest.raises(RuntimeError, match=f"schema version {SCHEMA_VERSION + 1}"):
        Store.open(tmp_path)


_V1_EXPERIMENTS = (
    "CREATE TABLE experiments (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL);"
    "INSERT INTO experiments (created_at) VALUES ('2026-10-11T00:00:00+00:00');"
)
_ADD_KIND = "ALTER TABLE experiments ADD COLUMN kind TEXT NOT NULL DEFAULT 'manual' CHECK (kind IN ('manual','sweep'));"
_ADD_REPORT = "ALTER TABLE experiments ADD COLUMN report TEXT;"


@pytest.mark.parametrize("applied", [_ADD_KIND, _ADD_KIND + _ADD_REPORT], ids=["first-step-only", "every-step-but-the-version"])
def test_migration_rerun_after_a_partial_step_opens_cleanly(tmp_path: Path, applied: str) -> None:
    """Review P3. Defect: executescript committed each ALTER on its own, so a crash between the
    steps and the version bump left a database that failed every later open with "duplicate
    column name". Each step now checks the table before it alters, and the version bump
    commits with the steps, so a rerun converges."""
    directory = tmp_path / PROJECT_DIR_NAME
    directory.mkdir()
    conn = sqlite3.connect(str(directory / "contexteval.db"))
    conn.executescript(_V1_EXPERIMENTS + applied + "PRAGMA user_version=1;")
    conn.close()
    store = Store.open(tmp_path)
    try:
        (old,) = store.experiments()
        assert (old.kind, old.report) == ("manual", None)
        sweep = store.new_experiment("sweep")
        store.save_experiment_report(sweep, {"recommended_run": None})
        assert store.experiment(sweep).report == {"recommended_run": None}
        assert store._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        store.close()
    Store.open(tmp_path).close()


def test_migration_steps_and_version_bump_commit_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review P3. Defect: a step that fails after an earlier one succeeded leaves the earlier
    column committed on its own, or the version bumped past steps that never ran. A failing
    step rolls the whole version step back, and the next open, with the step repaired, applies it."""
    from contexteval import _store

    directory = tmp_path / PROJECT_DIR_NAME
    directory.mkdir()
    conn = sqlite3.connect(str(directory / "contexteval.db"))
    conn.executescript(_V1_EXPERIMENTS + "PRAGMA user_version=1;")
    conn.close()
    good = _store._MIGRATIONS[1]
    broken = _store._AddColumn("experiments", "report", "NOT A COLUMN TYPE (")
    monkeypatch.setitem(_store._MIGRATIONS, 1, (good[0], broken))
    with pytest.raises(sqlite3.OperationalError):
        Store.open(tmp_path)
    conn = sqlite3.connect(str(directory / "contexteval.db"))
    columns = {row[1] for row in conn.execute("PRAGMA table_info(experiments)")}
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert ("kind" in columns, version) == (False, 1)
    monkeypatch.setitem(_store._MIGRATIONS, 1, good)
    store = Store.open(tmp_path)
    try:
        assert store.experiments()[0].kind == "manual"
        assert store._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        store.close()


def test_mean_app_tokens_reads_latest_baseline(store: Store) -> None:
    """Defect: the estimate reads an older baseline, or crashes when the latest run has no traces."""
    assert store.mean_app_tokens() is None
    old = store.new_run("baseline", "app:answer", BASELINE)
    store.add_trace(old, _trace())
    assert store.mean_app_tokens() == (20, 4)
    store.new_run("baseline", "app:answer", BASELINE)
    assert store.mean_app_tokens() is None


def test_runs_lists_every_kind_newest_first_with_counts(store: Store) -> None:
    """Defect: the dashboard's run picker omits experiment or production runs, or miscounts
    ungraded production traces as failures."""
    store.upsert_cases("synthetic", [CASE])
    base = store.new_run("baseline", "app:answer", BASELINE)
    store.add_trace(base, _trace(test=TestOutcome(CASE.id, True, "substring")))
    store.add_trace(base, _trace(test=TestOutcome(CASE.id, False, "llm")))
    store.add_trace(base, _trace(crashed=True, test=TestOutcome(CASE.id, False, "crash")))
    eid = store.new_experiment()
    exp = store.new_run("experiment", "app:answer", Setting(top_k=10), experiment=eid)
    prod = store.new_run("production", None, BASELINE)
    store.add_trace(prod, _trace())

    summaries = store.runs()
    assert [s.run.id for s in summaries] == [prod, exp, base]
    assert [(s.run.kind, s.experiment_id, s.traces, s.passed, s.failed) for s in summaries] == [
        ("production", None, 1, 0, 0),
        ("experiment", eid, 0, 0, 0),
        ("baseline", None, 3, 1, 2),
    ]
    assert summaries[1].run.setting == Setting(top_k=10)


def test_v1_single_sample_diagnoses_still_load_and_read_as_one_of_one(store: Store) -> None:
    """v1.1 A. v1 stored one replay per step, in the same JSON shape v1.1 writes for a sample.
    Defect: the loader expects a per-step count field or a nested samples list and rejects or
    misreads a v1 row. A v1 row loads unchanged and tallies as 1 of 1 per step."""
    rid = store.new_run("baseline", "app:answer", BASELINE)
    tid = store.add_trace(rid, _trace())
    v1_body = json.dumps(
        {
            "kind": "ignored",
            "replays": [
                {"step": "baseline_rerun", "passed": False, "answer": "wrong"},
                {"step": "gold_to_top", "passed": True, "answer": "right"},
            ],
            "gold_at": [[0, 2]],
        }
    )
    store._conn.execute(
        "INSERT INTO diagnoses (trace_id, label, source, body, created_at) VALUES (?, ?, ?, ?, ?)",
        (tid, "ignored", "ground_truth", v1_body, "2026-01-01T00:00:00"),
    )
    store._conn.commit()
    loaded = store.trace(tid)
    assert loaded is not None and loaded.diagnosis == Ignored(REPLAYS, gold_at=((0, 2),))
    assert tallies(loaded.diagnosis.replays) == (StepTally(ReplayStep.BASELINE_RERUN, 0, 1), StepTally(ReplayStep.GOLD_TO_TOP, 1, 1))
    assert explain(loaded.diagnosis) == "moving the gold chunk to the top fixed it (1/1); gold was at chunk 3 of retrieval 1"
