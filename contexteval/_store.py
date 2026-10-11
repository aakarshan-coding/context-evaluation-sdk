"""SQLite store at <project>/.contexteval/contexteval.db (plan item 10).

The schema and JSON encoding are private to this module. Callers get domain types back.
A run is an immutable event (one execution of a case list under one setting); traces belong
to runs; diagnoses belong to traces; experiments group runs. Experiment tables are derived
by SQL aggregation, never stored.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from ._intervention import Setting
from .types import (
    CaseId,
    Diagnosis,
    ExperimentId,
    ExperimentRow,
    RunId,
    TestCase,
    Trace,
    TraceId,
)

SCHEMA_VERSION = 1

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
  created_at TEXT NOT NULL
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
  in_prompt     TEXT NOT NULL,               -- JSON, denormalized from Trace.in_prompt for the UI
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
class StoredTrace:
    id: TraceId
    run_id: RunId
    setting: Setting  # joined from runs; the trace's active setting
    trace: Trace
    diagnosis: Diagnosis | None


@dataclass(frozen=True)
class RunInfo:
    id: RunId
    kind: Literal["baseline", "experiment", "production"]
    app: str | None
    setting: Setting
    created_at: str


class Store:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def open(cls, project_dir: Path) -> Store:
        """Create .contexteval/ and the schema if missing. Idempotent; checks PRAGMA user_version."""
        raise NotImplementedError

    # test cases
    def upsert_cases(self, test_set: str, cases: Iterable[TestCase]) -> int:
        """INSERT OR REPLACE by CaseId. Returns the number of new ids."""
        raise NotImplementedError

    def cases(self, test_set: str | None = None) -> list[TestCase]:
        raise NotImplementedError

    def case(self, case_id: CaseId) -> TestCase | None:
        raise NotImplementedError

    # runs and traces
    def new_run(self, kind: Literal["baseline", "experiment", "production"], app: str | None, setting: Setting, experiment: ExperimentId | None = None) -> RunId:
        raise NotImplementedError

    def add_trace(self, run: RunId, trace: Trace) -> TraceId:
        raise NotImplementedError

    def run(self, run: RunId) -> RunInfo:
        raise NotImplementedError

    def latest_run(self, kind: Literal["baseline", "production"]) -> RunInfo | None:
        raise NotImplementedError

    def traces(self, run: RunId, *, failed_only: bool = False) -> list[StoredTrace]:
        raise NotImplementedError

    def trace(self, trace_id: TraceId) -> StoredTrace | None:
        raise NotImplementedError

    # diagnoses
    def save_diagnosis(self, trace_id: TraceId, diagnosis: Diagnosis) -> None:
        """Upsert. Re-diagnosing a trace replaces its row."""
        raise NotImplementedError

    # experiments
    def new_experiment(self) -> ExperimentId:
        raise NotImplementedError

    def experiment_table(self, experiment: ExperimentId) -> list[ExperimentRow]:
        """GROUP BY run over traces: count, sum(passed), avg(input_tokens), avg(latency), sum(cost)."""
        raise NotImplementedError

    def experiments(self) -> list[tuple[ExperimentId, str]]:
        """(id, created_at), newest first."""
        raise NotImplementedError

    def mean_app_tokens(self) -> tuple[int, int] | None:
        """Mean (input, output) app tokens per trace from the latest baseline. Feeds cost estimates."""
        raise NotImplementedError
