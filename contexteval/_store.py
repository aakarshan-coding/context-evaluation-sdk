"""SQLite store at <project>/.contexteval/contexteval.db (plan item 10).

The schema and JSON encoding are private to this module. Callers get domain types back.
A run is an immutable event (one execution of a case list under one setting); traces belong
to runs; diagnoses belong to traces; experiments group runs. Experiment tables are derived
by SQL aggregation, never stored.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from ._intervention import Setting
from .types import (
    Answered,
    CaseId,
    Chunk,
    Crashed,
    Diagnosis,
    Evidence,
    ExperimentId,
    ExperimentRow,
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
    RunId,
    TestCase,
    TestOutcome,
    Trace,
    TraceId,
    label_of,
    never,
    source_of,
)

PROJECT_DIR_NAME = ".contexteval"
DB_FILE = "contexteval.db"
SCHEMA_VERSION = 2

RunKind = Literal["baseline", "experiment", "production"]
ExperimentKind = Literal["manual", "sweep"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS test_cases (
  id              TEXT PRIMARY KEY,          -- CaseId, content hash: rebuilds upsert
  test_set        TEXT NOT NULL,             -- "synthetic" | import file stem
  question        TEXT NOT NULL,
  expected_answer TEXT NOT NULL,
  evidence        TEXT,                      -- NULL -> judge path
  source          TEXT NOT NULL,
  created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
  id         INTEGER PRIMARY KEY,
  created_at TEXT NOT NULL,
  kind       TEXT NOT NULL DEFAULT 'manual' CHECK (kind IN ('manual','sweep')),
  report     TEXT                            -- sweep report JSON, owned by _sweep; NULL for manual
);
CREATE TABLE IF NOT EXISTS runs (
  id            INTEGER PRIMARY KEY,
  kind          TEXT NOT NULL CHECK (kind IN ('baseline','experiment','production')),
  app           TEXT,                        -- "module:function"; NULL for production
  setting       TEXT NOT NULL,               -- Setting JSON; the trace's active setting
  experiment_id INTEGER REFERENCES experiments(id),
  created_at    TEXT NOT NULL,
  CHECK ((kind = 'experiment') = (experiment_id IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS traces (
  id            INTEGER PRIMARY KEY,
  run_id        INTEGER NOT NULL REFERENCES runs(id),
  case_id       TEXT REFERENCES test_cases(id),
  question      TEXT NOT NULL,
  answer        TEXT,                        -- exactly one of answer / error is set
  error         TEXT,
  passed        INTEGER,                     -- NULL iff case_id IS NULL
  grade_method  TEXT,
  retrievals    TEXT NOT NULL,               -- JSON list of RetrievalEvent
  llm_calls     TEXT NOT NULL,               -- JSON list of LLMCall
  in_prompt     TEXT NOT NULL,               -- JSON list of [retrieval index, chunk index]
  input_tokens  INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  latency_ms    REAL NOT NULL,
  cost_usd      REAL,
  CHECK ((answer IS NULL) <> (error IS NULL)),
  CHECK ((case_id IS NULL) = (passed IS NULL))
);
CREATE TABLE IF NOT EXISTS diagnoses (
  trace_id   INTEGER PRIMARY KEY REFERENCES traces(id),   -- one diagnosis per trace; upsert
  label      TEXT NOT NULL,
  source     TEXT NOT NULL CHECK (source IN ('ground_truth','judge')),
  body       TEXT NOT NULL,                  -- JSON of the Diagnosis variant
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS traces_run ON traces(run_id);
"""

@dataclass(frozen=True)
class _AddColumn:
    """One migration step. It checks the table first, so a step that ran before a crash is a
    no-op on the rerun instead of "duplicate column name" on every later open."""

    table: str
    column: str
    definition: str

    def apply(self, conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({self.table})")}
        if self.column not in columns:
            conn.execute(f"ALTER TABLE {self.table} ADD COLUMN {self.column} {self.definition}")


# Schema version -> the steps that bring it to the next version, applied by _migrate in one
# transaction with the version bump, so an old project opens in place instead of being refused.
_MIGRATIONS: dict[int, tuple[_AddColumn, ...]] = {
    1: (
        _AddColumn("experiments", "kind", "TEXT NOT NULL DEFAULT 'manual' CHECK (kind IN ('manual','sweep'))"),
        _AddColumn("experiments", "report", "TEXT"),
    ),
}


def _migrate(conn: sqlite3.Connection, version: int) -> int:
    """Bring the database from `version` to the next one and return the version it holds after.
    The steps and the version bump commit together, and the version is re-read under the write
    lock, so a process that finds another one already past this step applies nothing."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        if current == version:
            for step in _MIGRATIONS[version]:
                step.apply(conn)
            current = version + 1
            conn.execute(f"PRAGMA user_version={current}")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return current


def find_project(start: Path) -> Path | None:
    """The nearest directory at or above `start` holding .contexteval/."""
    for candidate in (start, *start.parents):
        if (candidate / PROJECT_DIR_NAME).is_dir():
            return candidate
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- JSON encodings (private) -------------------------------------------------


def _chunk_json(c: Chunk) -> dict[str, Any]:
    return {"text": c.text, "id": c.id, "score": c.score, "source_index": c.source_index}


def _chunk_from(d: dict[str, Any]) -> Chunk:
    return Chunk(text=d["text"], id=d["id"], score=d["score"], source_index=d["source_index"])


def _event_json(e: RetrievalEvent) -> dict[str, Any]:
    return {
        "source": e.source,
        "query": e.query,
        "params": dict(e.params),
        "retrieved": [_chunk_json(c) for c in e.retrieved],
        "delivered": [_chunk_json(c) for c in e.delivered],
        "latency_ms": e.latency_ms,
    }


def _event_from(d: dict[str, Any]) -> RetrievalEvent:
    return RetrievalEvent(
        source=d["source"],
        query=d["query"],
        params=d["params"],
        retrieved=tuple(_chunk_from(c) for c in d["retrieved"]),
        delivered=tuple(_chunk_from(c) for c in d["delivered"]),
        latency_ms=d["latency_ms"],
    )


def _llm_json(c: LLMCall) -> dict[str, Any]:
    return {
        "provider": c.provider,
        "model": c.model,
        "prompt": c.prompt,
        "answer": c.answer,
        "input_tokens": c.input_tokens,
        "output_tokens": c.output_tokens,
        "latency_ms": c.latency_ms,
    }


def _llm_from(d: dict[str, Any]) -> LLMCall:
    return LLMCall(**d)


def _replays_json(replays: tuple[ReplayOutcome, ...]) -> list[dict[str, Any]]:
    return [{"step": r.step.value, "passed": r.passed, "answer": r.answer} for r in replays]


def _replays_from(items: list[dict[str, Any]]) -> tuple[ReplayOutcome, ...]:
    return tuple(ReplayOutcome(step=ReplayStep(r["step"]), passed=r["passed"], answer=r["answer"]) for r in items)


def _refs_from(items: list[list[int]]) -> tuple[tuple[int, int], ...]:
    return tuple((r, c) for r, c in items)


def _diagnosis_json(d: Diagnosis) -> dict[str, Any]:
    match d:
        case Missing(cause=cause, confirmed=confirmed, replays=replays):
            return {"kind": "missing", "cause": cause, "confirmed": confirmed, "replays": _replays_json(replays)}
        case Irrelevant(replays=replays, removed=removed):
            return {"kind": "irrelevant", "replays": _replays_json(replays), "removed": [list(p) for p in removed]}
        case Ignored(replays=replays, gold_at=gold_at):
            return {"kind": "ignored", "replays": _replays_json(replays), "gold_at": [list(p) for p in gold_at]}
        case NotContext(replays=replays):
            return {"kind": "not_context", "replays": _replays_json(replays)}
        case Flaky(replays=replays):
            return {"kind": "flaky", "replays": _replays_json(replays)}
        case Judged(verdict=verdict, rationale=rationale):
            return {"kind": "judged", "verdict": verdict.value, "rationale": rationale}
        case _:
            never(d)


def _diagnosis_from(d: dict[str, Any]) -> Diagnosis:
    kind = d["kind"]
    if kind == "missing":
        return Missing(cause=d["cause"], confirmed=d["confirmed"], replays=_replays_from(d["replays"]))
    if kind == "irrelevant":
        return Irrelevant(replays=_replays_from(d["replays"]), removed=_refs_from(d["removed"]))
    if kind == "ignored":
        return Ignored(replays=_replays_from(d["replays"]), gold_at=_refs_from(d["gold_at"]))
    if kind == "not_context":
        return NotContext(replays=_replays_from(d["replays"]))
    if kind == "flaky":
        return Flaky(replays=_replays_from(d["replays"]))
    if kind == "judged":
        return Judged(verdict=JudgeVerdict(d["verdict"]), rationale=d["rationale"])
    raise ValueError(f"unknown diagnosis kind in store: {kind!r}")


# --- domain views ---------------------------------------------------------------


@dataclass(frozen=True)
class StoredTrace:
    id: TraceId
    run_id: RunId
    setting: Setting  # joined from runs; the trace's active setting
    trace: Trace
    diagnosis: Diagnosis | None


@dataclass(frozen=True)
class RunInfo:
    id: RunId
    kind: RunKind
    app: str | None
    setting: Setting
    created_at: str


@dataclass(frozen=True)
class RunSummary:
    run: RunInfo
    experiment_id: ExperimentId | None
    traces: int
    passed: int
    failed: int  # graded traces that failed; ungraded (production) traces count in neither


@dataclass(frozen=True)
class ExperimentInfo:
    id: ExperimentId
    kind: ExperimentKind
    created_at: str
    report: dict[str, Any] | None  # the sweep's stored report (its JSON shape belongs to _sweep)


_TRACE_COLUMNS = """
    t.id, t.run_id, r.setting, t.case_id, t.question, t.answer, t.error, t.passed, t.grade_method,
    t.retrievals, t.llm_calls, t.in_prompt, t.input_tokens, t.output_tokens, t.latency_ms, t.cost_usd,
    d.body
"""
_TRACE_FROM = """
    FROM traces t
    JOIN runs r ON r.id = t.run_id
    LEFT JOIN diagnoses d ON d.trace_id = t.id
"""


class Store:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def open(cls, project_dir: Path) -> Store:
        """Create .contexteval/ and the schema if missing. Idempotent; checks PRAGMA user_version."""
        directory = project_dir / PROJECT_DIR_NAME
        directory.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the UI serves requests from handler threads over one store, and
        # the production writer closes its connection from its own thread. Writes are short and
        # serialized by sqlite's own locking.
        conn = sqlite3.connect(str(directory / DB_FILE), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION, *_MIGRATIONS):
            conn.close()
            raise RuntimeError(f"{directory / DB_FILE}: schema version {version}, this build reads {SCHEMA_VERSION}")
        while 0 < version < SCHEMA_VERSION:
            version = _migrate(conn, version)
        with conn:
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        return cls(conn)

    def close(self) -> None:
        self._conn.close()

    # test cases
    def upsert_cases(self, test_set: str, cases: Iterable[TestCase]) -> int:
        """INSERT OR REPLACE by CaseId. Returns the number of new ids."""
        cases = list(cases)
        existing = {row[0] for row in self._conn.execute("SELECT id FROM test_cases")}
        now = _now()
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO test_cases VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.id, test_set, c.question, c.expected_answer, c.evidence.text if c.evidence else None, c.source, now)
                    for c in cases
                ],
            )
        return len({c.id for c in cases} - existing)

    def cases(self, test_set: str | None = None) -> list[TestCase]:
        sql = "SELECT id, question, expected_answer, evidence, source FROM test_cases"
        params: tuple[Any, ...] = ()
        if test_set is not None:
            sql += " WHERE test_set = ?"
            params = (test_set,)
        return [self._case(row) for row in self._conn.execute(sql + " ORDER BY created_at, id", params)]

    def case(self, case_id: CaseId) -> TestCase | None:
        row = self._conn.execute(
            "SELECT id, question, expected_answer, evidence, source FROM test_cases WHERE id = ?", (case_id,)
        ).fetchone()
        return self._case(row) if row else None

    @staticmethod
    def _case(row: tuple[Any, ...]) -> TestCase:
        cid, question, expected, evidence, source = row
        return TestCase(
            id=CaseId(cid),
            question=question,
            expected_answer=expected,
            evidence=Evidence(evidence) if evidence is not None else None,
            source=source,
        )

    # runs and traces
    def new_run(self, kind: RunKind, app: str | None, setting: Setting, experiment: ExperimentId | None = None) -> RunId:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO runs (kind, app, setting, experiment_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (kind, app, json.dumps(setting.to_json()), experiment, _now()),
            )
        return RunId(cur.lastrowid)

    def add_trace(self, run: RunId, trace: Trace) -> TraceId:
        outcome = trace.outcome
        test = trace.test
        with self._conn:
            cur = self._conn.execute(
                """INSERT INTO traces (run_id, case_id, question, answer, error, passed, grade_method,
                   retrievals, llm_calls, in_prompt, input_tokens, output_tokens, latency_ms, cost_usd)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run,
                    test.case_id if test else None,
                    trace.question,
                    outcome.text if isinstance(outcome, Answered) else None,
                    outcome.error if isinstance(outcome, Crashed) else None,
                    int(test.passed) if test else None,
                    test.method if test else None,
                    json.dumps([_event_json(e) for e in trace.retrievals]),
                    json.dumps([_llm_json(c) for c in trace.llm_calls]),
                    json.dumps([list(p) for p in trace.in_prompt]),
                    trace.metrics.input_tokens,
                    trace.metrics.output_tokens,
                    trace.metrics.latency_ms,
                    trace.metrics.cost_usd,
                ),
            )
        return TraceId(cur.lastrowid)

    def run(self, run: RunId) -> RunInfo:
        row = self._conn.execute("SELECT id, kind, app, setting, created_at FROM runs WHERE id = ?", (run,)).fetchone()
        if row is None:
            raise KeyError(f"no run {run}")
        return self._run(row)

    def latest_run(self, kind: Literal["baseline", "production"]) -> RunInfo | None:
        row = self._conn.execute(
            "SELECT id, kind, app, setting, created_at FROM runs WHERE kind = ? ORDER BY id DESC LIMIT 1", (kind,)
        ).fetchone()
        return self._run(row) if row else None

    def runs(self) -> list[RunSummary]:
        """Every run with its trace counts, newest first."""
        rows = self._conn.execute(
            """SELECT r.id, r.kind, r.app, r.setting, r.created_at, r.experiment_id,
                      COUNT(t.id), COALESCE(SUM(t.passed = 1), 0), COALESCE(SUM(t.passed = 0), 0)
               FROM runs r LEFT JOIN traces t ON t.run_id = r.id
               GROUP BY r.id ORDER BY r.id DESC"""
        ).fetchall()
        return [
            RunSummary(
                run=self._run(row[:5]),
                experiment_id=ExperimentId(row[5]) if row[5] is not None else None,
                traces=row[6],
                passed=row[7],
                failed=row[8],
            )
            for row in rows
        ]

    @staticmethod
    def _run(row: tuple[Any, ...]) -> RunInfo:
        rid, kind, app, setting, created_at = row
        return RunInfo(id=RunId(rid), kind=kind, app=app, setting=Setting.from_json(json.loads(setting)), created_at=created_at)

    def traces(self, run: RunId, *, failed_only: bool = False) -> list[StoredTrace]:
        sql = f"SELECT {_TRACE_COLUMNS} {_TRACE_FROM} WHERE t.run_id = ?"
        if failed_only:
            sql += " AND t.passed = 0"
        return [self._stored(row) for row in self._conn.execute(sql + " ORDER BY t.id", (run,))]

    def trace(self, trace_id: TraceId) -> StoredTrace | None:
        row = self._conn.execute(f"SELECT {_TRACE_COLUMNS} {_TRACE_FROM} WHERE t.id = ?", (trace_id,)).fetchone()
        return self._stored(row) if row else None

    @staticmethod
    def _stored(row: tuple[Any, ...]) -> StoredTrace:
        (
            tid, run_id, setting, case_id, question, answer, error, passed, grade_method,
            retrievals, llm_calls, in_prompt, input_tokens, output_tokens, latency_ms, cost_usd, body,
        ) = row
        trace = Trace(
            question=question,
            outcome=Answered(answer) if error is None else Crashed(error),
            retrievals=tuple(_event_from(e) for e in json.loads(retrievals)),
            llm_calls=tuple(_llm_from(c) for c in json.loads(llm_calls)),
            metrics=Metrics(input_tokens=input_tokens, output_tokens=output_tokens, latency_ms=latency_ms, cost_usd=cost_usd),
            test=TestOutcome(case_id=CaseId(case_id), passed=bool(passed), method=grade_method) if case_id is not None else None,
            in_prompt=_refs_from(json.loads(in_prompt)),
        )
        return StoredTrace(
            id=TraceId(tid),
            run_id=RunId(run_id),
            setting=Setting.from_json(json.loads(setting)),
            trace=trace,
            diagnosis=_diagnosis_from(json.loads(body)) if body is not None else None,
        )

    # diagnoses
    def save_diagnosis(self, trace_id: TraceId, diagnosis: Diagnosis) -> None:
        """Upsert. Re-diagnosing a trace replaces its row."""
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO diagnoses (trace_id, label, source, body, created_at) VALUES (?, ?, ?, ?, ?)",
                (trace_id, label_of(diagnosis).value, source_of(diagnosis), json.dumps(_diagnosis_json(diagnosis)), _now()),
            )

    # experiments
    def new_experiment(self, kind: ExperimentKind = "manual") -> ExperimentId:
        with self._conn:
            cur = self._conn.execute("INSERT INTO experiments (created_at, kind) VALUES (?, ?)", (_now(), kind))
        return ExperimentId(cur.lastrowid)

    def save_experiment_report(self, experiment: ExperimentId, report: Mapping[str, Any]) -> None:
        """Replaces the report, so a sweep can store its outcome once every run is in."""
        with self._conn:
            self._conn.execute("UPDATE experiments SET report = ? WHERE id = ?", (json.dumps(report), experiment))

    def experiment_table(self, experiment: ExperimentId) -> list[ExperimentRow]:
        """GROUP BY run over traces: count, sum(passed), avg(input_tokens), avg(latency), sum(cost)."""
        rows = self._conn.execute(
            """SELECT r.id, r.setting, COUNT(t.id), COALESCE(SUM(t.passed), 0), COALESCE(AVG(t.input_tokens), 0),
                      COALESCE(AVG(t.latency_ms), 0), SUM(t.cost_usd), COUNT(t.cost_usd)
               FROM runs r LEFT JOIN traces t ON t.run_id = r.id
               WHERE r.experiment_id = ?
               GROUP BY r.id ORDER BY r.id""",
            (experiment,),
        ).fetchall()
        table = []
        for run_id, setting, cases, passed, mean_in, mean_lat, cost, priced in rows:
            table.append(
                ExperimentRow(
                    run_id=RunId(run_id),
                    setting_name=Setting.from_json(json.loads(setting)).name,
                    cases=cases,
                    passed=passed,
                    mean_input_tokens=mean_in,
                    mean_latency_ms=mean_lat,
                    # A total is honest only when every trace in the run was priced.
                    total_cost_usd=cost if cases and priced == cases else None,
                )
            )
        return table

    def experiments(self) -> list[ExperimentInfo]:
        """Newest first."""
        return [self._experiment(row) for row in self._conn.execute("SELECT id, kind, created_at, report FROM experiments ORDER BY id DESC")]

    def experiment(self, experiment: ExperimentId) -> ExperimentInfo:
        row = self._conn.execute("SELECT id, kind, created_at, report FROM experiments WHERE id = ?", (experiment,)).fetchone()
        if row is None:
            raise KeyError(f"no experiment {experiment}")
        return self._experiment(row)

    @staticmethod
    def _experiment(row: tuple[Any, ...]) -> ExperimentInfo:
        eid, kind, created_at, report = row
        return ExperimentInfo(id=ExperimentId(eid), kind=kind, created_at=created_at, report=json.loads(report) if report is not None else None)

    def mean_app_tokens(self) -> tuple[int, int] | None:
        """Mean (input, output) app tokens per trace from the latest baseline. Feeds cost estimates."""
        latest = self.latest_run("baseline")
        if latest is None:
            return None
        mean_in, mean_out = self._conn.execute(
            "SELECT AVG(input_tokens), AVG(output_tokens) FROM traces WHERE run_id = ?", (latest.id,)
        ).fetchone()
        if mean_in is None:
            return None
        return round(mean_in), round(mean_out)
