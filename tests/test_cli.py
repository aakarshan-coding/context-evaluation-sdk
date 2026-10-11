"""P0: the entry point parses, lists every command, and `init` creates the project.
P2: build-tests, run, and the cost prompt."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from contexteval import _cli
from contexteval._llm import Estimate
from contexteval._store import PROJECT_DIR_NAME, Store

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
    assert json.loads((project / "config.json").read_text()) == {"provider": "anthropic", "model": None, "app": "my_app:answer", "pinecone_text_key": None}
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


# --- P2: build-tests, run, confirm --------------------------------------------


def test_confirm_treats_eof_as_no(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: the cost prompt blocks or proceeds when stdin is closed (CI, piped scripts)."""
    est = Estimate(calls=3, input_tokens=100, output_tokens=10, usd=0.25)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with pytest.raises(SystemExit) as exc:
        _cli.confirm(est, yes=False)
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "3 LLM calls" in out and "$0.2500" in out and "aborted" in out

    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    _cli.confirm(est, yes=False)

    class Exploding(io.StringIO):
        def readline(self, *args: object) -> str:
            raise AssertionError("--yes must not read stdin")

    monkeypatch.setattr(sys, "stdin", Exploding())
    _cli.confirm(est, yes=True)


def test_cli_run_demo_end_to_end(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """P2 acceptance through `_cli.main`. Defects: run not wired to the stored cases, the app
    spec not remembered in config, or a planted case graded against PLANTED's baseline."""
    pytest.importorskip("chromadb")
    from contexteval.demo import planted

    assert _cli.main(["init", "--provider", "fake"]) == 0
    jsonl = planted.write_jsonl(project / planted.IMPORT_FILE)
    assert _cli.main(["build-tests", "--import", str(jsonl)]) == 0
    assert _cli.main(["run", "contexteval.demo.app:answer", "--yes"]) == 0
    assert _cli.load_config(project).app == "contexteval.demo.app:answer"

    store = Store.open(project)
    try:
        latest = store.latest_run("baseline")
        assert latest is not None and latest.app == "contexteval.demo.app:answer"
        traces = store.traces(latest.id)
    finally:
        store.close()
    graded = {s.trace.test.case_id: s.trace.test.passed for s in traces if s.trace.test}
    assert len(traces) == len(graded) == len(planted.PLANTED)
    assert graded == {p.case.id: p.passes_baseline for p in planted.PLANTED}
    healthy = sum(p.passes_baseline for p in planted.PLANTED)
    assert f"{len(planted.PLANTED)} cases, {healthy} passed" in capsys.readouterr().out

    assert _cli.main(["run", "--yes"]) == 0  # app remembered from config


def test_cli_build_tests_from_docs(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: build-tests stores nothing, duplicates on rebuild, or runs without its estimate."""
    from contexteval.demo.app import CORPUS_DIR

    assert _cli.main(["init", "--provider", "fake"]) == 0
    assert _cli.main(["build-tests", str(CORPUS_DIR), "--yes"]) == 0
    assert _cli.main(["build-tests", str(CORPUS_DIR), "--yes"]) == 0
    out = capsys.readouterr().out
    assert out.count("estimate:") == 2 and "(0 new)" in out
    store = Store.open(project)
    try:
        assert len(store.cases("synthetic")) == 8
    finally:
        store.close()


def test_cli_run_and_build_tests_user_errors(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Boundary: missing app, empty test set, and bad arguments are one-line errors, exit 1."""
    assert _cli.main(["init", "--provider", "fake"]) == 0
    assert _cli.main(["run", "--yes"]) == 1
    assert _cli.main(["run", "x:y", "--yes"]) == 1
    assert _cli.main(["build-tests"]) == 1
    assert _cli.main(["build-tests", "--import", str(project / "missing.jsonl")]) == 1
    err = capsys.readouterr().err
    assert "no app given" in err and "no test cases" in err and "exactly one of" in err and "missing.jsonl" in err


def test_run_remembers_the_app_only_after_it_loads(project: Path) -> None:
    """Review Y5. Defect: `run` writes the app spec to config.json before importing it, so one typo
    replaces the remembered app and every later `run` and `experiment` without a spec fails."""
    assert _cli.main(["init", "--provider", "fake", "--app", "contexteval.demo.app:answer"]) == 0
    tests = project / "t.jsonl"
    tests.write_text(json.dumps({"question": "q?", "expected_answer": "a"}) + "\n", encoding="utf-8")
    assert _cli.main(["build-tests", "--import", str(tests)]) == 0
    try:
        code = _cli.main(["run", "no_such_module_for_contexteval:answer", "--yes"])
    except SystemExit as e:
        code = e.code
    assert code not in (0, None)
    assert _cli.load_config(project).app == "contexteval.demo.app:answer"


def test_run_passes_the_configured_pinecone_text_key_to_load_app(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review X7. Defect: config.json has no pinecone_text_key, so under `run` every pinecone
    chunk is decoded from the default "text" key and comes back empty."""
    assert _cli.main(["init", "--provider", "fake"]) == 0
    _cli.save_config(project, _cli.Config(provider="fake", pinecone_text_key="body"))
    tests = project / "t.jsonl"
    tests.write_text(json.dumps({"question": "q?", "expected_answer": "a"}) + "\n", encoding="utf-8")
    assert _cli.main(["build-tests", "--import", str(tests)]) == 0
    seen = {}

    def fake_load_app(spec, pinecone_text_key=None):  # type: ignore[no-untyped-def]
        seen["key"] = pinecone_text_key
        return lambda question: "a"

    monkeypatch.setattr(_cli._runner, "load_app", fake_load_app)
    assert _cli.main(["run", "my_app:answer", "--yes"]) == 0
    assert seen == {"key": "body"}


def test_build_tests_keeps_cases_from_passages_before_a_provider_error(project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Review Y9. Defect: build-tests stores cases only after every passage succeeds, so a provider
    error on the last document throws away the cases the user already paid for."""
    from contexteval._llm import FakeProvider, ProviderError, Task, _fake_generate

    calls = []

    def generate(req):  # type: ignore[no-untyped-def]
        calls.append(req)
        if len(calls) == 2:
            raise ProviderError("openai: RateLimitError: slow down")
        return _fake_generate(req)

    monkeypatch.setattr(_cli, "make_provider", lambda name, model: FakeProvider(handlers={Task.GENERATE_CASES: generate}))
    assert _cli.main(["init", "--provider", "fake"]) == 0
    docs = project / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("The Ridge 2 tent weighs 1.8 kilograms in total.\n", encoding="utf-8")
    (docs / "b.md").write_text("Its poles are aluminum and the fly is silnylon.\n", encoding="utf-8")
    assert _cli.main(["build-tests", str(docs), "--per-passage", "1", "--yes"]) == 1
    assert "1 case from earlier passages is saved" in capsys.readouterr().err
    store = Store.open(project)
    try:
        assert [c.source for c in store.cases()] == [(docs / "a.md").as_posix()]
    finally:
        store.close()
