"""`contexteval` command (argparse, plan item 12). Thin shell: parse, confirm cost, call one function.

    contexteval init [--provider openai|anthropic|fake] [--model M] [--app module:function]
    contexteval build-tests DOCS_DIR [--per-passage N] [--yes]
    contexteval build-tests --import FILE
    contexteval run [module:function] [--test-set NAME] [--yes]
    contexteval diagnose [--run ID] [--force] [--yes]
    contexteval experiment [module:function] KNOB=v1,v2 ... [--yes]
    contexteval sweep [module:function] [--rounds N] [--max-settings N] [--holdout F] [--seed S] [--yes]
    contexteval ui [--port 8765] [--no-open]

Every command finds the project by walking up from cwd to a .contexteval/ directory, except
`init`, which creates it. Exit codes: 0 ok, 1 user error (message, no traceback), 2 aborted at
the cost prompt.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import typing
from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from . import _diagnose, _runner, _sweep, _testset
from ._intervention import InterventionNotApplicable, Setting
from ._llm import Estimate, Provider, ProviderError, make_provider
from ._store import PROJECT_DIR_NAME, Store, StoredTrace, find_project
from .types import Diagnosis, ExperimentRow, RunId, TestCase, explain, label_of

CONFIG_FILE = "config.json"
PROVIDERS = ("openai", "anthropic", "fake")


@dataclass(frozen=True)
class Config:
    """.contexteval/config.json. JSON because tomllib is 3.11+ and core has no deps."""

    provider: str = "openai"
    model: str | None = None
    app: str | None = None  # default for run / experiment when no positional is given
    pinecone_text_key: str | None = None  # metadata key holding chunk text in pinecone matches


class UserError(Exception):
    """Printed as one line and exit code 1."""


def load_config(project: Path) -> Config:
    path = project / PROJECT_DIR_NAME / CONFIG_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise UserError(f"{path} not found; run `contexteval init`") from None
    except json.JSONDecodeError as e:
        raise UserError(f"{path} is not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise UserError(f"{path} must hold a JSON object")
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise UserError(f"{path}: unknown keys {unknown}; valid keys are {sorted(known)}")
    for key, value in data.items():
        if value is not None and not isinstance(value, str):
            raise UserError(f"{path}: {key} must be a string or null")
    if data.get("provider") not in PROVIDERS:
        raise UserError(f"{path}: provider must be one of {list(PROVIDERS)}")
    return Config(**data)


def save_config(project: Path, config: Config) -> Path:
    path = project / PROJECT_DIR_NAME / CONFIG_FILE
    path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    return path


def require_project() -> Path:
    project = find_project(Path.cwd())
    if project is None:
        raise UserError(f"no {PROJECT_DIR_NAME}/ found here or above; run `contexteval init`")
    return project


KnobParser = Callable[[str], object]


def _knob_parsers() -> dict[str, tuple[KnobParser, str]]:
    """Knob name -> (parse one value, valid values for the error), derived from Setting's field
    types so a new knob needs no edit here."""
    parsers: dict[str, tuple[KnobParser, str]] = {}
    for name, hint in typing.get_type_hints(Setting).items():
        args = typing.get_args(hint)
        if isinstance(hint, type) and issubclass(hint, Enum):
            parsers[name] = (hint, "|".join(m.value for m in hint))
        elif typing.get_origin(hint) is typing.Literal:
            parsers[name] = (_one_of(args), "|".join(args))
        elif int in args or hint is int:
            parsers[name] = (int, "a positive integer")
        else:
            raise TypeError(f"Setting.{name}: no grid parser for {hint!r}")
    return parsers


def _one_of(allowed: tuple[str, ...]) -> KnobParser:
    def parse(value: str) -> str:
        if value not in allowed:
            raise ValueError(value)
        return value

    return parse


def parse_grid(tokens: Sequence[str]) -> list[Setting]:
    """["top_k=3,10", "rerank=none,lexical"] -> cartesian product of Settings, first knob slowest.

    The only place strings become Setting values. Unknown knob or bad value -> UserError naming
    it, with the valid knobs or values taken from Setting's fields.
    """
    parsers = _knob_parsers()
    axes: dict[str, list[object]] = {}
    for token in tokens:
        knob, sep, raw = token.partition("=")
        if not sep or not raw:
            raise UserError(f"grid entry {token!r} must look like knob=v1,v2")
        if knob not in parsers:
            raise UserError(f"unknown knob {knob!r}; valid knobs are {', '.join(parsers)}")
        if knob in axes:
            raise UserError(f"knob {knob!r} given twice; list its values once, as {knob}=v1,v2")
        parse, valid = parsers[knob]
        values = []
        for value in raw.split(","):
            try:
                values.append(parse(value))
            except ValueError:
                raise UserError(f"bad value {value!r} for {knob}; expected {valid}") from None
        axes[knob] = values
    settings = []
    for combo in itertools.product(*axes.values()):
        try:
            settings.append(Setting(**dict(zip(axes, combo))))  # type: ignore[arg-type]
        except ValueError as e:
            raise UserError(str(e)) from None
    return settings


def confirm(estimate: Estimate, yes: bool) -> None:
    """Print calls, tokens, and $ estimate; prompt y/N unless --yes. Raises SystemExit(2) on no.

    A closed or non-interactive stdin (EOF) counts as no, so a script without --yes never hangs.
    """
    usd = "unknown cost (model not in the price table)" if estimate.usd is None else f"about ${estimate.usd:.4f}"
    print(
        f"estimate: {estimate.calls} LLM calls, ~{estimate.input_tokens} input and "
        f"~{estimate.output_tokens} output tokens, {usd}"
    )
    if yes:
        return
    try:
        reply = input("continue? [y/N] ")
    except (EOFError, OSError):
        reply = ""
    if reply.strip().lower() not in ("y", "yes"):
        print("aborted")
        raise SystemExit(2)


def _provider(config: Config) -> Provider:
    try:
        return make_provider(config.provider, config.model)
    except ValueError as e:
        raise UserError(str(e)) from None


# --- commands ---------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    project = Path.cwd()
    Store.open(project).close()
    path = save_config(project, Config(provider=args.provider, model=args.model, app=args.app))
    print(f"initialized {path.parent}")
    return 0


def cmd_build_tests(args: argparse.Namespace) -> int:
    if (args.docs_dir is None) == (args.import_file is None):
        raise UserError("give exactly one of DOCS_DIR or --import FILE")
    project = require_project()
    config = load_config(project)
    store = Store.open(project)
    try:
        if args.import_file is not None:
            path = Path(args.import_file)
            try:
                cases = _testset.import_tests(path)
            except ValueError as e:
                raise UserError(str(e)) from None
            new = store.upsert_cases(path.stem, cases)
            print(f"imported {len(cases)} cases ({new} new) into test set {path.stem!r}")
            return 0

        root = Path(args.docs_dir)
        if not root.is_dir():
            raise UserError(f"{root} is not a directory")
        docs = _testset.read_documents(root)
        if not docs:
            raise UserError(f"no .md or .txt files under {root}")
        provider = _provider(config)
        confirm(_testset.estimate_build(docs, provider, args.per_passage), args.yes)
        saved = new = 0

        def save(cases: tuple[TestCase, ...]) -> None:
            nonlocal saved, new
            new += store.upsert_cases(_testset.SYNTHETIC_TEST_SET, cases)
            saved += len(cases)

        try:
            report = _testset.build_tests(docs, provider, args.per_passage, on_cases=save)
        except ProviderError as e:
            what = "1 case from earlier passages is" if saved == 1 else f"{saved} cases from earlier passages are"
            raise UserError(f"{e} ({what} saved)") from None
        print(f"built {len(report.cases)} cases ({new} new) from {len(docs)} documents; rejected {len(report.rejected)}")
        for reason in sorted({r.reason for r in report.rejected}):
            count = sum(r.reason == reason for r in report.rejected)
            print(f"  rejected {count}: {reason}")
        return 0
    finally:
        store.close()


def cmd_run(args: argparse.Namespace) -> int:
    project = require_project()
    config = load_config(project)
    spec = args.app or config.app
    if spec is None:
        raise UserError("no app given; pass MODULE:FUNCTION once and it is remembered")
    store = Store.open(project)
    try:
        cases = store.cases(args.test_set)
        if not cases:
            where = f"test set {args.test_set!r}" if args.test_set else "this project"
            raise UserError(f"no test cases in {where}; run `contexteval build-tests` first")
        provider = _provider(config)
        app = _runner.load_app(spec, pinecone_text_key=config.pinecone_text_key)
        if spec != config.app:
            save_config(project, replace(config, app=spec))
        app_tokens = store.mean_app_tokens()
        if app_tokens is not None:
            print(f"app: {len(cases)} calls to {spec}, ~{app_tokens[0]} input tokens each in the last run")
        confirm(_runner.estimate_grading(provider, cases), args.yes)
        try:
            summary = _runner.run_suite(store, spec, app, cases, _runner.make_grader(provider))
        except ProviderError as e:
            raise UserError(str(e)) from None
        failed = summary.cases - summary.passed - summary.crashed
        print(f"run {summary.run}: {summary.cases} cases, {summary.passed} passed, {failed} failed, {summary.crashed} crashed")
        return 0
    finally:
        store.close()


def _print_diagnosis(stored: StoredTrace, diagnosis: Diagnosis) -> None:
    print(f"  {label_of(diagnosis).value:<12} {stored.trace.question}")
    print(f"               {explain(diagnosis)}")


def cmd_diagnose(args: argparse.Namespace) -> int:
    project = require_project()
    config = load_config(project)
    store = Store.open(project)
    try:
        if args.run is not None:
            try:
                run = store.run(RunId(args.run))
            except KeyError:
                raise UserError(f"no run {args.run}; `contexteval ui` lists runs") from None
        else:
            latest = store.latest_run("baseline")
            if latest is None:
                raise UserError("no baseline run yet; run `contexteval run` first")
            run = latest
        todo = _diagnose.pending(store, run.id, force=args.force)
        if not todo:
            print(f"run {run.id}: nothing to diagnose; every failure is labeled (use --force to redo)")
            return 0
        provider = _provider(config)
        app = None
        if run.kind != "production":
            spec = run.app or config.app
            if spec is None:
                raise UserError(f"run {run.id} has no app recorded; pass one with `contexteval init --app`")
            app = _runner.load_app(spec, pinecone_text_key=config.pinecone_text_key)
            ground_truth = sum(p.path == "replay" for p in todo)
            print(
                f"app: up to {ground_truth * _diagnose.MAX_STEPS * args.samples} replay calls to {spec} "
                f"({ground_truth} failures with evidence, {_diagnose.MAX_STEPS} steps x {args.samples} samples each, worst case)"
            )
        confirm(_diagnose.estimate_diagnosis(provider, todo, args.samples), args.yes)
        print(f"run {run.id}:")
        try:
            report = _diagnose.diagnose_run(
                store,
                run.id,
                app,
                _runner.make_grader(provider),
                provider,
                force=args.force,
                on_diagnosis=_print_diagnosis,
                samples=args.samples,
            )
        except (ProviderError, _diagnose.JudgeReplyError) as e:
            raise UserError(f"{e} (diagnoses so far are saved; rerun to resume)") from None
        print(
            f"diagnosed {report.diagnosed}, skipped {report.skipped_existing} already labeled, "
            f"skipped {report.skipped_crashed} crashed, skipped {report.skipped_no_answer} with no answer text"
        )
        return 0
    finally:
        store.close()


def cmd_experiment(args: argparse.Namespace) -> int:
    tokens = list(args.tokens)
    spec_arg = tokens.pop(0) if ":" in tokens[0] and "=" not in tokens[0] else None
    if not tokens:
        raise UserError("give at least one knob=v1,v2 grid entry")
    grid = parse_grid(tokens)
    project = require_project()
    config = load_config(project)
    spec = spec_arg or config.app
    if spec is None:
        raise UserError("no app given; pass MODULE:FUNCTION or run `contexteval run MODULE:FUNCTION` first")
    store = Store.open(project)
    try:
        cases = store.cases()
        if not cases:
            raise UserError("no test cases in this project; run `contexteval build-tests` first")
        settings = _runner.experiment_settings(grid)
        provider = _provider(config)
        app = _runner.load_app(spec, pinecone_text_key=config.pinecone_text_key)
        print(f"experiment: {len(cases)} cases x {len(settings)} settings ({', '.join(s.name for s in settings)})")
        app_tokens = store.mean_app_tokens()
        if app_tokens is not None:
            print(f"app: {len(cases) * len(settings)} calls to {spec}, ~{app_tokens[0]} input tokens each in the last run")
        confirm(_runner.estimate_grading(provider, list(cases) * len(settings)), args.yes)
        try:
            experiment = _runner.run_experiment(store, spec, app, cases, _runner.make_grader(provider), grid)
        except ProviderError as e:
            raise UserError(str(e)) from None
        except InterventionNotApplicable as e:
            raise UserError(f"the app's retrieval cannot take this setting: {e}") from None
        print(f"experiment {experiment}")
        print(format_experiment_table(store.experiment_table(experiment)))
        return 0
    finally:
        store.close()


def _fraction(value: str) -> float:
    """Boundary check for --holdout: a share of the cases, 0 (none held out) up to but not 1."""
    try:
        f = float(value)
    except ValueError:
        f = -1.0
    if not 0 <= f < 1:
        raise argparse.ArgumentTypeError(f"--holdout must be a fraction from 0 up to but not including 1, got {value!r}")
    return f


def _positive(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return n


def _print_plan(plan: _sweep.Plan, spec: str) -> None:
    labeled = ", ".join(f"{kind.value} {n}" for kind, n in plan.counts.items())
    print(f"sweep: run {plan.source_run}, {sum(plan.counts.values())} labeled failures ({labeled}); the baseline retrieved up to {plan.baseline_k} chunks")
    if plan.unlabeled:
        print(f"warning: {plan.unlabeled} failures have no label and do not guide the search; run `contexteval diagnose` to include them")
    for kind, n, reason in plan.unfixable:
        noun = "failure" if n == 1 else "failures"
        print(f"  no setting can fix {n} {kind.value} {noun}: {reason}")
    if not plan.candidates:
        return
    shown = plan.candidates[: plan.max_settings]
    names = ", ".join(f"{r.setting.name} ({r.weight})" for r in shown)
    print(f"candidates, heaviest first ({len(shown)} of {len(plan.candidates)}; the number is how many failures ask for it): {names}")
    split = plan.split
    print(f"split: {len(split.tune)} tune cases, {len(split.holdout)} holdout cases (seed {split.seed})")
    if not split.holdout:
        print("warning: no holdout cases, so the recommendation is not confirmed on cases the search did not see")
    tune, held = len(split.tune), len(split.holdout)
    print(
        f"app: up to {plan.worst_case_app_calls} calls to {spec} "
        f"({tune} tune cases x (1 baseline + {plan.round_size} settings x {plan.rounds} rounds) + {held} holdout cases x 2, worst case)"
    )


def cmd_sweep(args: argparse.Namespace) -> int:
    project = require_project()
    config = load_config(project)
    spec = args.app or config.app
    if spec is None:
        raise UserError("no app given; pass MODULE:FUNCTION or run `contexteval run MODULE:FUNCTION` first")
    store = Store.open(project)
    try:
        source = store.latest_run("baseline")
        if source is None:
            raise UserError("no baseline run yet; run `contexteval run` and `contexteval diagnose` first")
        try:
            plan, cases = _sweep.plan_sweep(store, source, rounds=args.rounds, max_settings=args.max_settings, holdout=args.holdout, seed=args.seed)
        except _sweep.SweepNotPossible as e:
            raise UserError(str(e)) from None
        _print_plan(plan, spec)
        if not plan.counts:
            print(f"run {source.id} has no labeled failure to fix; nothing to sweep")
            return 0
        if not plan.candidates:
            print("Keep the current settings. No setting addresses the labels seen, for the reasons above.")
            return 0
        provider = _provider(config)
        app = _runner.load_app(spec, pinecone_text_key=config.pinecone_text_key)
        graded = [cases[c] for c in plan.split.tune] * (1 + plan.round_size * plan.rounds) + [cases[c] for c in plan.split.holdout] * 2
        confirm(_runner.estimate_grading(provider, graded), args.yes)
        try:
            experiment, report = _sweep.run_sweep(store, spec, app, cases, _runner.make_grader(provider), plan)
        except ProviderError as e:
            raise UserError(f"{e} (the sweep's finished runs are stored as an experiment)") from None
        except InterventionNotApplicable as e:
            raise UserError(f"the app's retrieval cannot take a candidate setting: {e}") from None
        table = {row.run_id: row for row in store.experiment_table(experiment)}
        print(f"experiment {experiment} (sweep)")
        for number, round_runs in enumerate(report.rounds, 1):
            print(f"round {number}, {len(plan.split.tune)} tune cases")
            print(format_experiment_table([table[report.baseline_run], *(table[r] for r in round_runs)]))
        if report.holdout_runs:
            print(f"holdout, {len(plan.split.holdout)} cases")
            print(format_experiment_table([table[r] for r in report.holdout_runs]))
        print(_sweep.recommendation_line(report, list(table.values())))
        return 0
    finally:
        store.close()


def format_experiment_table(rows: Sequence[ExperimentRow]) -> str:
    header = ("setting", "accuracy", "passed", "mean input tokens", "mean latency ms", "total cost")
    body = [
        (
            r.setting_name,
            f"{r.accuracy:.0%}",
            f"{r.passed}/{r.cases}",
            f"{r.mean_input_tokens:.0f}",
            f"{r.mean_latency_ms:.1f}",
            "unknown" if r.total_cost_usd is None else f"${r.total_cost_usd:.4f}",
        )
        for r in rows
    ]
    widths = [max(len(row[i]) for row in (header, *body)) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in (header, *body))


def cmd_ui(args: argparse.Namespace) -> int:
    from . import _ui

    store = Store.open(require_project())
    try:
        _ui.serve(store, port=args.port, open_browser=not args.no_open)
    except OSError as e:
        raise UserError(f"cannot serve on port {args.port}: {e}") from None
    finally:
        store.close()
    return 0


def _args_init(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=PROVIDERS, default="openai")
    p.add_argument("--model", default=None)
    p.add_argument("--app", default=None, metavar="MODULE:FUNCTION")


def _args_build_tests(p: argparse.ArgumentParser) -> None:
    p.add_argument("docs_dir", nargs="?", metavar="DOCS_DIR")
    p.add_argument("--import", dest="import_file", metavar="FILE")
    p.add_argument("--per-passage", type=int, default=2)
    p.add_argument("--yes", action="store_true")


def _args_run(p: argparse.ArgumentParser) -> None:
    p.add_argument("app", nargs="?", metavar="MODULE:FUNCTION")
    p.add_argument("--test-set", default=None)
    p.add_argument("--yes", action="store_true")


def _odd_samples(value: str) -> int:
    """Boundary check for --samples: a majority needs an odd count, and at least one sample."""
    try:
        n = int(value)
    except ValueError:
        n = 0
    if n < 1 or n % 2 == 0:
        raise argparse.ArgumentTypeError(f"--samples must be a positive odd integer, got {value!r}")
    return n


def _args_diagnose(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", type=int, default=None, metavar="ID")
    p.add_argument("--force", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument(
        "--samples",
        type=_odd_samples,
        default=_diagnose.DEFAULT_SAMPLES,
        metavar="N",
        help=f"replays per step, odd; a step passes when a majority pass (default {_diagnose.DEFAULT_SAMPLES}, 1 decides from one replay)",
    )


def _args_experiment(p: argparse.ArgumentParser) -> None:
    p.add_argument("tokens", nargs="+", metavar="[MODULE:FUNCTION] KNOB=v1,v2")
    p.add_argument("--yes", action="store_true")


def _args_sweep(p: argparse.ArgumentParser) -> None:
    p.add_argument("app", nargs="?", metavar="MODULE:FUNCTION")
    p.add_argument("--rounds", type=_positive, default=_sweep.DEFAULT_ROUNDS, metavar="N", help=f"search rounds; later rounds combine winners (default {_sweep.DEFAULT_ROUNDS})")
    p.add_argument("--max-settings", type=_positive, default=_sweep.DEFAULT_MAX_SETTINGS, metavar="N", help=f"settings tried per round (default {_sweep.DEFAULT_MAX_SETTINGS})")
    p.add_argument("--holdout", type=_fraction, default=_sweep.DEFAULT_HOLDOUT, metavar="F", help=f"share of cases kept to confirm the winner (default {_sweep.DEFAULT_HOLDOUT}; 0 for a tiny set)")
    p.add_argument("--seed", type=int, default=0, metavar="S", help="seed of the tune/holdout split (default 0)")
    p.add_argument("--yes", action="store_true")


def _args_ui(p: argparse.ArgumentParser) -> None:
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true")


@dataclass(frozen=True)
class Command:
    run: Callable[[argparse.Namespace], int]
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]


COMMANDS: dict[str, Command] = {
    "init": Command(cmd_init, "create .contexteval/ with config and database", _args_init),
    "build-tests": Command(cmd_build_tests, "generate test cases from documents, or import a file", _args_build_tests),
    "run": Command(cmd_run, "run every test case through your answer() and grade it", _args_run),
    "diagnose": Command(cmd_diagnose, "label each failure of a run", _args_diagnose),
    "experiment": Command(cmd_experiment, "run a settings grid and compare", _args_experiment),
    "sweep": Command(cmd_sweep, "pick settings from the latest diagnosis, try them, and recommend one", _args_sweep),
    "ui": Command(cmd_ui, "serve the local dashboard", _args_ui),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contexteval", description="Find out why your RAG agent's context failed.")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    for name, command in COMMANDS.items():
        command.add_arguments(sub.add_parser(name, help=command.help))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point ([project.scripts] contexteval = "contexteval._cli:main")."""
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command].run(args)
    except (UserError, _runner.AppLoadError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
