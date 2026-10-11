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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ._intervention import Setting
from ._llm import Estimate


@dataclass(frozen=True)
class Config:
    """.contexteval/config.json. JSON because tomllib is 3.11+ and core has no deps."""

    provider: str = "openai"
    model: str | None = None
    app: str | None = None  # default for run / experiment when no positional is given


def load_config(project: Path) -> Config:
    raise NotImplementedError  # boundary: unknown keys or wrong types -> UserError


class UserError(Exception):
    """Printed as one line and exit code 1."""


def parse_grid(tokens: Sequence[str]) -> list[Setting]:
    """["top_k=3,10", "rerank=none,lexical"] -> cartesian product of Settings.

    Unknown knob or bad value -> UserError listing the valid knobs and values, which come
    from Setting's fields so the list is never hand-synced.
    """
    raise NotImplementedError


def confirm(estimate: Estimate, yes: bool) -> None:
    """Print calls, tokens, and $ estimate; prompt y/N unless --yes. Raises SystemExit(2) on no."""
    raise NotImplementedError


def cmd_init(args: argparse.Namespace) -> int: raise NotImplementedError
def cmd_build_tests(args: argparse.Namespace) -> int: raise NotImplementedError
def cmd_run(args: argparse.Namespace) -> int: raise NotImplementedError
def cmd_diagnose(args: argparse.Namespace) -> int: raise NotImplementedError
def cmd_experiment(args: argparse.Namespace) -> int: raise NotImplementedError
def cmd_ui(args: argparse.Namespace) -> int: raise NotImplementedError


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "init": cmd_init,
    "build-tests": cmd_build_tests,
    "run": cmd_run,
    "diagnose": cmd_diagnose,
    "experiment": cmd_experiment,
    "ui": cmd_ui,
}


def build_parser() -> argparse.ArgumentParser:
    raise NotImplementedError  # one subparser per COMMANDS key


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point ([project.scripts] contexteval = "contexteval._cli:main")."""
    raise NotImplementedError
