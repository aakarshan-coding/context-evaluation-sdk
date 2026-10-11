"""Execute the user's answer(question) in-process, one question at a time (plan item 7).

`execute` is the single path for baseline runs, diagnosis replays, and experiments.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

from . import _capture as capture
from . import _hooks
from . import _text as text
from ._intervention import BASELINE, NO_INTERVENTION, Intervention, InterventionNotApplicable, Setting
from ._llm import Estimate, LLMRequest, Provider, Task, estimate
from ._store import Store
from .types import Answered, Crashed, ExperimentId, Outcome, RunId, TestCase, TestOutcome, Trace

App = Callable[[str], str]
Grader = Callable[[TestCase, Trace], TestOutcome]


class AppLoadError(Exception):
    """The app spec did not resolve to a callable. The CLI prints it as a one-line user error."""


def load_app(spec: str) -> App:
    """Boundary: "module:function" -> callable. Prepends cwd to sys.path.

    Installs hooks in CLI mode *before* importing, so module-level clients are patched
    and a user init() call in the module stays a no-op.
    Raises AppLoadError with a readable message on a bad spec, import error, or non-callable.
    """
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise AppLoadError(f"app must look like module:function, got {spec!r}")
    _hooks.install("cli")
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise AppLoadError(f"cannot import {module_name!r}: {e}") from None
    fn = module
    for part in attr.split("."):
        fn = getattr(fn, part, None)
        if fn is None:
            raise AppLoadError(f"{module_name!r} has no attribute {attr!r}")
    if not callable(fn):
        raise AppLoadError(f"{spec} is not callable")
    return fn  # type: ignore[return-value]


def execute(app: App, question: str, intervention: Intervention = NO_INTERVENTION) -> Trace:
    """Run one question with `intervention` active. Never raises for app errors: an exception
    becomes Crashed. InterventionNotApplicable propagates, even when the app's own except
    swallowed it on the way out of the hook. The returned Trace is ungraded.
    """
    with capture.active_run(question, intervention) as run:
        t0 = perf_counter()
        outcome: Outcome
        try:
            outcome = Answered(str(app(question)))
        except InterventionNotApplicable:
            raise  # an experiment that cannot apply must not pass as a crashed baseline
        except Exception as e:
            outcome = Crashed(f"{type(e).__name__}: {e}")
        if run.not_applicable is not None:
            raise run.not_applicable
        latency_ms = (perf_counter() - t0) * 1000
    return capture.finalize_trace(run, outcome, latency_ms)


_GRADE_SYSTEM = (
    "You grade answers. Reply PASS if the answer states the same fact as the expected answer, "
    "even if worded differently. Reply FAIL otherwise. Reply with one word."
)


def make_grader(provider: Provider) -> Grader:
    """Crash -> fail("crash"); text.answer_matches -> pass("substring");
    else provider GRADE request at temperature 0 -> "PASS"/"FAIL" (plan item 2).
    """

    def grade(case: TestCase, trace: Trace) -> TestOutcome:
        outcome = trace.outcome
        if isinstance(outcome, Crashed):
            return TestOutcome(case.id, False, "crash")
        if text.answer_matches(case.expected_answer, outcome.text):
            return TestOutcome(case.id, True, "substring")
        reply = provider.complete(
            LLMRequest(
                task=Task.GRADE,
                system=_GRADE_SYSTEM,
                user=f"Question: {case.question}\nExpected answer: {case.expected_answer}\nAnswer: {outcome.text}",
                inputs={"question": case.question, "expected_answer": case.expected_answer, "answer": outcome.text},
                max_tokens=5,
            )
        )
        return TestOutcome(case.id, reply.text.strip().upper().startswith("PASS"), "llm")

    return grade


def estimate_grading(provider: Provider, cases: Sequence[TestCase]) -> Estimate:
    """Worst case: every answer misses the substring fast path and needs one GRADE call.
    The answer is unknown before the run, so the expected answer stands in for its length."""
    prompts = [_GRADE_SYSTEM + c.question + c.expected_answer * 2 for c in cases]
    return estimate(provider.model, prompts, 2)


def run_case(app: App, case: TestCase, grade: Grader, intervention: Intervention = NO_INTERVENTION) -> Trace:
    """execute + grade. Returns the trace with `test` filled."""
    trace = execute(app, case.question, intervention)
    return dataclasses.replace(trace, test=grade(case, trace))


@dataclass(frozen=True)
class RunSummary:
    run: RunId
    cases: int
    passed: int
    crashed: int


def run_suite(
    store: Store,
    app_spec: str,
    app: App,
    cases: Sequence[TestCase],
    grade: Grader,
    setting: Setting = BASELINE,
    experiment: ExperimentId | None = None,
) -> RunSummary:
    """One stored run: new_run, then run_case + add_trace per case. Committed per trace, so a
    crash halfway leaves a valid partial run; rerunning creates a fresh run (runs are events).
    """
    run = store.new_run("experiment" if experiment is not None else "baseline", app_spec, setting, experiment)
    intervention = setting.intervention()
    passed = crashed = 0
    for case in cases:
        trace = run_case(app, case, grade, intervention)
        store.add_trace(run, trace)
        passed += bool(trace.test and trace.test.passed)
        crashed += isinstance(trace.outcome, Crashed)
    return RunSummary(run=run, cases=len(cases), passed=passed, crashed=crashed)


def experiment_settings(grid: Sequence[Setting]) -> list[Setting]:
    """BASELINE first, then the grid in order with duplicates and BASELINE itself removed, so
    every table has exactly one comparison row. The cost estimate counts the same list."""
    settings = [BASELINE]
    for setting in grid:
        if setting not in settings:
            settings.append(setting)
    return settings


def run_experiment(store: Store, app_spec: str, app: App, cases: Sequence[TestCase], grade: Grader, grid: Sequence[Setting]) -> ExperimentId:
    """One stored run per setting of experiment_settings(grid), all under one new experiment.
    Each run goes through run_suite, the same path as `contexteval run`, with the setting stored
    on the run. InterventionNotApplicable propagates and leaves the runs done so far stored.
    """
    experiment = store.new_experiment()
    for setting in experiment_settings(grid):
        run_suite(store, app_spec, app, cases, grade, setting, experiment)
    return experiment
