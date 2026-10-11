"""Execute the user's answer(question) in-process, one question at a time (plan item 7).

`execute` is the single path for baseline runs, diagnosis replays, and experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from . import _capture as capture
from ._intervention import BASELINE, NO_INTERVENTION, Intervention, Setting
from ._llm import LLMRequest, Provider, Task
from ._store import Store
from .types import ExperimentId, RunId, TestCase, TestOutcome, Trace

App = Callable[[str], str]
Grader = Callable[[TestCase, Trace], TestOutcome]


def load_app(spec: str) -> App:
    """Boundary: "module:function" -> callable. Prepends cwd to sys.path.

    Installs hooks in CLI mode *before* importing, so module-level clients are patched
    and a user init() call in the module stays a no-op.
    Raises SystemExit with a readable message on a bad spec, import error, or non-callable.
    """
    raise NotImplementedError


def execute(app: App, question: str, intervention: Intervention = NO_INTERVENTION) -> Trace:
    """Run one question with `intervention` active. Never raises for app errors: an exception
    becomes Crashed. The returned Trace is ungraded (test=None).
    """
    # TODO:
    #   with capture.active_run(question, intervention) as run:
    #       t0 = perf_counter()
    #       try: outcome = Answered(app(question))
    #       except Exception as e: outcome = Crashed(f"{type(e).__name__}: {e}")
    #   metrics: tokens from LLMCall usage, else text.estimate_tokens(prompt / answer)
    #   cost from llm.cost_usd per call model
    raise NotImplementedError


def make_grader(provider: Provider) -> Grader:
    """Crash -> fail("crash"); text.answer_matches -> pass("substring");
    else provider GRADE request at temperature 0 -> "PASS"/"FAIL" (plan item 2).
    """
    raise NotImplementedError


def run_case(app: App, case: TestCase, grade: Grader, intervention: Intervention = NO_INTERVENTION) -> Trace:
    """execute + grade. Returns the trace with `test` filled."""
    raise NotImplementedError


@dataclass(frozen=True)
class RunSummary:
    run: RunId
    cases: int
    passed: int
    crashed: int


def run_suite(store: Store, app_spec: str, app: App, cases: Sequence[TestCase], grade: Grader, setting: Setting = BASELINE, experiment: ExperimentId | None = None) -> RunSummary:
    """One stored run: new_run, then run_case + add_trace per case. Committed per trace, so a
    crash halfway leaves a valid partial run; rerunning creates a fresh run (runs are events).
    """
    raise NotImplementedError


def run_experiment(store: Store, app_spec: str, app: App, cases: Sequence[TestCase], grade: Grader, grid: Sequence[Setting]) -> ExperimentId:
    """One run per setting under a new experiment. BASELINE is prepended when the grid lacks it,
    so every table has a comparison row.
    """
    raise NotImplementedError
