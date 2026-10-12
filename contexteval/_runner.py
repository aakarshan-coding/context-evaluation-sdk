"""Execute the user's answer(question) in-process, one question at a time (plan item 7).

`execute` is the single path for baseline runs, diagnosis replays, and experiments. An
`async def answer` runs to completion on one event loop that lives for the process (one run),
not a fresh loop per question: the app's loop-bound clients (an AsyncOpenAI connection pool,
an async database pool) stay valid from one question to the next, as they do in production,
and no caller of execute needs to carry a loop.
"""

from __future__ import annotations

import asyncio
import atexit
import dataclasses
import importlib
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable, Iterable, Sequence, Union

from . import _capture as capture
from . import _hooks
from . import _text as text
from ._intervention import BASELINE, NO_INTERVENTION, Intervention, InterventionNotApplicable, Setting
from ._llm import Estimate, LLMRequest, Provider, Task, estimate
from ._store import Store
from .types import Answered, Chunking, Crashed, ExperimentId, Outcome, RunId, TestCase, TestOutcome, Trace

App = Callable[[str], Union[str, Awaitable[str]]]
Builder = Callable[[Chunking], App]
"""The optional second entry point, `build(chunking) -> answer`, found by name beside `answer`.
It rebuilds the app's index with that chunking and returns the answer function to run on it."""
Grader = Callable[[TestCase, Trace], TestOutcome]

_loop: asyncio.AbstractEventLoop | None = None


def _await(result: Awaitable[Any]) -> Any:
    """Run an async answer() on the process's one loop, created on first use."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        atexit.register(_loop.close)
    return _loop.run_until_complete(result)


class AppLoadError(Exception):
    """The app spec did not resolve to a callable. The CLI prints it as a one-line user error."""


def _build_attr(answer_attr: str) -> str:
    """Where the builder lives for an answer attribute: `build` beside `answer`, `bot.build` beside `bot.answer`."""
    head, dot, _ = answer_attr.rpartition(".")
    return f"{head}{dot}build"


class NoBuilder(Exception):
    """A setting changes the chunking and the app offers no `build(chunking)`. Chunking happens when
    the app builds its index, before any hook sees a chunk, so contexteval cannot apply it alone.
    The CLI prints this as a one-line user error that says what to add."""

    def __init__(self, spec: str, chunking: Chunking) -> None:
        module_name, _, attr = spec.partition(":")
        self.spec, self.chunking = spec, chunking
        super().__init__(
            f"{chunking.name} needs the app to rebuild its index, and {spec} has no {module_name}:{_build_attr(attr)}; "
            f"add `def build(chunking: contexteval.Chunking)` beside `{attr}` that splits each document with "
            "contexteval.chunk_text(text, chunking.with_defaults(chunk_size=...), title=...), indexes the chunks, "
            "and returns the answer function to run on them"
        )


@dataclass
class LoadedApp:
    """The app module's entry points: `answer`, and `build` when the module offers it. One
    instance lives for one CLI command, so `for_chunking` builds each distinct Chunking once and
    every case and setting that shares it reuses that index."""

    spec: str
    answer: App
    build: Builder | None = None
    _built: dict[Chunking, App] = field(default_factory=dict, repr=False)

    def for_chunking(self, chunking: Chunking) -> App:
        """The answer function for `chunking`: `answer` itself for the app's own chunking (build is
        never called, so `contexteval run` is unchanged for an app that offers it), else the
        cached result of `build(chunking)`. A build that raises or returns no function is
        InterventionNotApplicable, so a sweep skips that candidate and goes on."""
        if chunking.is_default:
            return self.answer
        if self.build is None:
            raise NoBuilder(self.spec, chunking)
        if chunking not in self._built:
            try:
                built = self.build(chunking)
            except Exception as e:
                raise InterventionNotApplicable(f"build({chunking.name}) raised {type(e).__name__}: {e}") from e
            if not callable(built):
                raise InterventionNotApplicable(f"build({chunking.name}) returned {type(built).__name__}, not an answer function")
            self._built[chunking] = built
        return self._built[chunking]

    def ensure_buildable(self, settings: Iterable[Setting]) -> None:
        """Raise NoBuilder for the first setting that needs a build this app cannot make, before
        any app call or cost prompt."""
        for setting in settings:
            if not setting.chunking.is_default and self.build is None:
                raise NoBuilder(self.spec, setting.chunking)


def load_app(spec: str, *, pinecone_text_key: str | None = None) -> LoadedApp:
    """Boundary: "module:function" -> the module's entry points. Prepends cwd to sys.path.

    Installs hooks in CLI mode *before* importing, so module-level clients are patched
    and a user init() call in the module stays a no-op. `pinecone_text_key` is the project's
    configured key for pinecone chunk text; an init(pinecone_text_key=) in the module overrides it.
    The builder is `build` beside the answer attribute, when present. Raises AppLoadError with a
    readable message on a bad spec, import error, non-callable answer, or non-callable build.
    """
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise AppLoadError(f"app must look like module:function, got {spec!r}")
    _hooks.install("cli", pinecone_text_key=pinecone_text_key)
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise AppLoadError(f"cannot import {module_name!r}: {e}") from None

    def resolve(path: str) -> object | None:
        obj: object = module
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    fn = resolve(attr)
    if fn is None:
        raise AppLoadError(f"{module_name!r} has no attribute {attr!r}")
    if not callable(fn):
        raise AppLoadError(f"{spec} is not callable")
    build = resolve(_build_attr(attr))
    if build is not None and not callable(build):
        raise AppLoadError(f"{module_name}:{_build_attr(attr)} is not callable; build(chunking) must be a function that returns the answer function")
    return LoadedApp(spec, fn, build)  # type: ignore[arg-type]


def execute(app: App, question: str, intervention: Intervention = NO_INTERVENTION) -> Trace:
    """Run one question with `intervention` active. Never raises for app errors: an exception
    becomes Crashed. InterventionNotApplicable propagates, even when the app's own except
    swallowed it on the way out of the hook. The returned Trace is ungraded. An awaitable
    result (an `async def answer`) is run to completion first.
    """
    with capture.active_run(question, intervention) as run:
        t0 = perf_counter()
        outcome: Outcome
        try:
            result = app(question)
            if inspect.isawaitable(result):
                result = _await(result)
            outcome = Answered(str(result))
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
    app: LoadedApp,
    cases: Sequence[TestCase],
    grade: Grader,
    setting: Setting = BASELINE,
    experiment: ExperimentId | None = None,
) -> RunSummary:
    """One stored run: new_run, then run_case + add_trace per case. Committed per trace, so a
    crash halfway leaves a valid partial run; rerunning creates a fresh run (runs are events).
    The setting's chunking picks the index (`app.for_chunking`) and its retrieval knobs the
    intervention, decided here once for every run of every command.
    """
    answer = app.for_chunking(setting.chunking)
    run = store.new_run("experiment" if experiment is not None else "baseline", app.spec, setting, experiment)
    intervention = setting.intervention()
    passed = crashed = 0
    for case in cases:
        trace = run_case(answer, case, grade, intervention)
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


def distinct_chunkings(settings: Iterable[Setting]) -> list[Chunking]:
    """The index builds a list of settings needs, in first-seen order: one per distinct
    non-default Chunking. The experiment command prints it beside the setting count."""
    out: list[Chunking] = []
    for setting in settings:
        if not setting.chunking.is_default and setting.chunking not in out:
            out.append(setting.chunking)
    return out


def run_experiment(store: Store, app: LoadedApp, cases: Sequence[TestCase], grade: Grader, grid: Sequence[Setting]) -> ExperimentId:
    """One stored run per setting of experiment_settings(grid), all under one new experiment.
    Each run goes through run_suite, the same path as `contexteval run`, with the setting stored
    on the run. InterventionNotApplicable propagates and leaves the runs done so far stored.
    """
    experiment = store.new_experiment()
    for setting in experiment_settings(grid):
        run_suite(store, app, cases, grade, setting, experiment)
    return experiment
