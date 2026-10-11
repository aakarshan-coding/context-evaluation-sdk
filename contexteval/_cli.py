"""`contexteval` command (argparse, plan item 12). Thin shell: parse, confirm cost, call one function.

    contexteval init [--provider openai|anthropic|fake] [--model M] [--app module:function]
    contexteval build-tests DOCS_DIR [--per-passage N] [--yes]
    contexteval build-tests --import FILE
    contexteval run [module:function] [--test-set NAME] [--yes]
    contexteval diagnose [--run ID] [--force] [--yes]
    contexteval experiment KNOB=v1,v2 ... [--app module:function] [--yes]
    contexteval ui [--port 8765] [--no-open]

Every command finds the project by walking up from cwd to a .contexteval/ directory, except
`init`, which creates it. Exit codes: 0 ok, 1 user error (message, no traceback), 2 aborted at
the cost prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Sequence

from ._intervention import Setting
from ._llm import Estimate
from ._store import PROJECT_DIR_NAME, Store, find_project

CONFIG_FILE = "config.json"
PROVIDERS = ("openai", "anthropic", "fake")


@dataclass(frozen=True)
class Config:
    """.contexteval/config.json. JSON because tomllib is 3.11+ and core has no deps."""

    provider: str = "openai"
    model: str | None = None
    app: str | None = None  # default for run / experiment when no positional is given


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


def parse_grid(tokens: Sequence[str]) -> list[Setting]:
    """["top_k=3,10", "rerank=none,lexical"] -> cartesian product of Settings.

    Unknown knob or bad value -> UserError listing the valid knobs and values, which come
    from Setting's fields so the list is never hand-synced.
    """
    raise NotImplementedError


def confirm(estimate: Estimate, yes: bool) -> None:
    """Print calls, tokens, and $ estimate; prompt y/N unless --yes. Raises SystemExit(2) on no."""
    raise NotImplementedError


# --- commands ---------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    project = Path.cwd()
    Store.open(project).close()
    path = save_config(project, Config(provider=args.provider, model=args.model, app=args.app))
    print(f"initialized {path.parent}")
    return 0


def cmd_build_tests(args: argparse.Namespace) -> int:
    raise NotImplementedError


def cmd_run(args: argparse.Namespace) -> int:
    raise NotImplementedError


def cmd_diagnose(args: argparse.Namespace) -> int:
    raise NotImplementedError


def cmd_experiment(args: argparse.Namespace) -> int:
    raise NotImplementedError


def cmd_ui(args: argparse.Namespace) -> int:
    raise NotImplementedError


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


def _args_diagnose(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", type=int, default=None, metavar="ID")
    p.add_argument("--force", action="store_true")
    p.add_argument("--yes", action="store_true")


def _args_experiment(p: argparse.ArgumentParser) -> None:
    p.add_argument("grid", nargs="+", metavar="KNOB=v1,v2")
    p.add_argument("--app", default=None, metavar="MODULE:FUNCTION")
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
    except UserError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
