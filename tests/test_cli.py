"""P0: the entry point parses, lists every command, and `init` creates the project."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from contexteval import _cli
from contexteval._store import PROJECT_DIR_NAME

REPO_ROOT = Path(__file__).resolve().parent.parent
COMMANDS = ["init", "build-tests", "run", "diagnose", "experiment", "ui"]


def test_python_m_help_lists_all_six_commands() -> None:
    """Defect: a command registered in COMMANDS but missing from the parser, or `python -m`
    not wired to the same main."""
    out = subprocess.run([sys.executable, "-m", "contexteval", "--help"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    for name in COMMANDS:
        assert name in out.stdout


def test_main_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: --help treated as an error exit, which breaks `contexteval --help` in docs."""
    with pytest.raises(SystemExit) as exc:
        _cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert all(name in out for name in COMMANDS)
    assert set(_cli.COMMANDS) == set(COMMANDS)


def test_no_command_is_a_usage_error() -> None:
    """Defect: bare `contexteval` crashing with a KeyError instead of usage."""
    with pytest.raises(SystemExit) as exc:
        _cli.main([])
    assert exc.value.code == 2


def test_init_creates_config_and_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: init writes the config somewhere other than where every other command reads it, or
    a second init fails on the existing database."""
    monkeypatch.chdir(tmp_path)
    assert _cli.main(["init", "--provider", "anthropic", "--app", "my_app:answer"]) == 0
    project = tmp_path / PROJECT_DIR_NAME
    assert (project / "contexteval.db").is_file()
    assert json.loads((project / "config.json").read_text()) == {"provider": "anthropic", "model": None, "app": "my_app:answer"}
    assert _cli.load_config(tmp_path) == _cli.Config(provider="anthropic", model=None, app="my_app:answer")
    assert _cli.main(["init"]) == 0
    assert _cli.load_config(tmp_path) == _cli.Config()
    assert "initialized" in capsys.readouterr().out


def test_load_config_rejects_unknown_keys_and_bad_provider(tmp_path: Path) -> None:
    """Boundary: a typo in config.json is a one-line UserError, not a traceback deep in a command."""
    project = tmp_path / PROJECT_DIR_NAME
    project.mkdir()
    (project / "config.json").write_text(json.dumps({"provider": "openai", "modle": "x"}))
    with pytest.raises(_cli.UserError, match="unknown keys"):
        _cli.load_config(tmp_path)
    (project / "config.json").write_text(json.dumps({"provider": "gemini"}))
    with pytest.raises(_cli.UserError, match="provider"):
        _cli.load_config(tmp_path)
    with pytest.raises(_cli.UserError, match="not found"):
        _cli.load_config(tmp_path / "elsewhere")


def test_require_project_without_init_is_user_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: a command run outside a project crashes instead of telling the user to init."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(_cli.UserError):
        _cli.require_project()
