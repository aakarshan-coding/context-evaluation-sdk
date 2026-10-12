"""P2: grading, execution, and stored runs, against the planted demo app."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

chromadb = pytest.importorskip("chromadb")
openai = pytest.importorskip("openai")

from contexteval import _capture as capture  # noqa: E402
from contexteval._intervention import Intervention, InterventionNotApplicable, Setting  # noqa: E402
from contexteval._llm import AnthropicProvider, FakeProvider, LLMRequest, OpenAIProvider, Task  # noqa: E402
from contexteval._runner import App, AppLoadError, LoadedApp, NoBuilder, execute, load_app, make_grader, run_suite  # noqa: E402
from contexteval._store import Store  # noqa: E402
from contexteval.demo import app as demo_app  # noqa: E402
from contexteval.demo import planted  # noqa: E402
from contexteval.types import Answered, Chunking, Crashed, Metrics, Trace  # noqa: E402

CASE = planted.PLANTED[-1].case


def _trace(outcome: Answered | Crashed) -> Trace:
    return Trace(question=CASE.question, outcome=outcome, retrievals=(), llm_calls=(), metrics=Metrics(0, 0, 0.0, None), test=None, in_prompt=())


def test_grader_substring_fast_path_skips_the_llm() -> None:
    """Defect: the fast path compares raw strings (case, punctuation), or calls the LLM anyway,
    which costs money on every passing case."""
    fake = FakeProvider()
    outcome = make_grader(fake)(CASE, _trace(Answered(f"Yes: ORDERS over {CASE.expected_answer.upper()}!")))
    assert (outcome.case_id, outcome.passed, outcome.method) == (CASE.id, True, "substring")
    assert fake.requests == []


def test_grader_llm_path_uses_structured_inputs() -> None:
    """Defect: a non-substring answer is failed without asking the grader, or the grader's
    verdict is ignored, or the request lacks the answer it must judge."""
    verdicts = iter(["PASS", "fail"])
    fake = FakeProvider(handlers={Task.GRADE: lambda req: next(verdicts)})
    grade = make_grader(fake)
    first = grade(CASE, _trace(Answered("Seventy-five bucks or more.")))
    second = grade(CASE, _trace(Answered("No idea.")))
    assert (first.passed, first.method) == (True, "llm")
    assert (second.passed, second.method) == (False, "llm")
    assert [r.inputs["answer"] for r in fake.requests] == ["Seventy-five bucks or more.", "No idea."]
    assert fake.requests[0].inputs["expected_answer"] == CASE.expected_answer


def test_grader_crash_fails_without_llm() -> None:
    """Defect: a crashed trace sent to the LLM grader, or graded as anything but a crash fail."""
    fake = FakeProvider(handlers={Task.GRADE: lambda req: "PASS"})
    outcome = make_grader(fake)(CASE, _trace(Crashed("RuntimeError: boom")))
    assert (outcome.passed, outcome.method) == (False, "crash") and fake.requests == []


def test_baseline_run_stores_one_graded_trace_per_planted_case(project: Path, store: Store, hooks: None) -> None:
    """P2 acceptance. Defects: a planted case whose baseline pass/fail differs from PLANTED (corpus,
    rule, or app drift), traces stored without their retrieval or LLM call, or ungraded traces."""
    store.upsert_cases("planted", planted.cases())
    summary = run_suite(store, LoadedApp("contexteval.demo.app:answer", demo_app.answer), planted.cases(), make_grader(FakeProvider()))
    stored = store.traces(summary.run)
    assert [s.trace.question for s in stored] == [p.case.question for p in planted.PLANTED]
    for p, s in zip(planted.PLANTED, stored):
        t = s.trace
        assert t.test is not None and t.test.case_id == p.case.id
        assert t.test.passed is p.passes_baseline, p.case.question
        assert len(t.retrievals) == 1 and len(t.retrievals[0].retrieved) == demo_app.TOP_K
        assert len(t.llm_calls) == 1 and t.llm_calls[0].model == demo_app.FAKE_MODEL
        assert isinstance(t.outcome, Answered)
    assert (summary.cases, summary.passed, summary.crashed) == (len(planted.PLANTED), sum(p.passes_baseline for p in planted.PLANTED), 0)
    assert store.run(summary.run).app == "contexteval.demo.app:answer"


def test_crash_in_answer_is_stored_as_crashed(project: Path, store: Store) -> None:
    """Defect: an exception in the user's answer() aborts the run or is stored as an answer."""

    def broken(question: str) -> str:
        raise RuntimeError(f"no index for {question!r}")

    store.upsert_cases("planted", [CASE])
    summary = run_suite(store, LoadedApp("tests:broken", broken), [CASE], make_grader(FakeProvider()))
    (s,) = store.traces(summary.run)
    assert s.trace.outcome == Crashed(f"RuntimeError: no index for {CASE.question!r}")
    assert s.trace.test is not None and (s.trace.test.passed, s.trace.test.method) == (False, "crash")
    assert summary.crashed == 1


def test_intervention_not_applicable_propagates(hooks: None) -> None:
    """Defect: an experiment knob that cannot apply is swallowed as a crashed case, so a grid
    silently degrades to baseline runs."""

    def app(question: str) -> str:
        raise InterventionNotApplicable("no top_k here")

    with pytest.raises(InterventionNotApplicable):
        execute(app, "q", Intervention(top_k=10))


def _openai_client() -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "PASS"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        })

    return openai.OpenAI(api_key="fake", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _anthropic_client() -> Any:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")

    def handler(request: Any) -> Any:
        body = json.loads(request.content)
        assert body["temperature"] == 0
        return httpx2.Response(200, json={
            "id": "m", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": "PASS"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 1},
        })

    return anthropic.Anthropic(api_key="fake", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


def test_sdk_provider_calls_are_not_recorded(hooks: None) -> None:
    """Defect: the grader's own LLM call lands in the user's trace (or, in production mode,
    becomes a production trace). The direct client call in the same run proves the hook is live."""
    req = LLMRequest(Task.GRADE, system="Grade.", user="Q", inputs={})
    oa, an = _openai_client(), _anthropic_client()
    with capture.active_run("q") as run:
        assert OpenAIProvider(client=oa).complete(req).text == "PASS"
        assert AnthropicProvider(client=an).complete(req).text == "PASS"
        assert run.llm_calls == []
        oa.chat.completions.create(model="gpt-4.1-mini", messages=[{"role": "user", "content": "direct"}])
    assert [c.prompt for c in run.llm_calls] == ["user: direct"]


def test_openai_provider_sends_temperature_zero() -> None:
    """Defect: the SDK's grader runs at the API default temperature, so grades are not repeatable."""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "gpt-4.1-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "FAIL"}, "finish_reason": "stop"}],
        })

    client = openai.OpenAI(api_key="fake", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    resp = OpenAIProvider(client=client).complete(LLMRequest(Task.GRADE, system="s", user="u", inputs={}))
    assert resp.text == "FAIL"
    assert seen[0]["temperature"] == 0 and seen[0]["model"] == "gpt-4.1-mini"


def test_load_app_installs_hooks_before_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hooks: None) -> None:
    """Defect: the user module is imported from the wrong directory, or a bad spec raises
    SystemExit from library code (review X10) instead of an error the CLI reports as one line."""
    (tmp_path / "my_rag_app.py").write_text("def answer(q):\n    return q.upper()\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "my_rag_app", raising=False)
    assert load_app("my_rag_app:answer").answer("hi") == "HI"
    assert capture._process.mode is capture.Mode.CLI
    for bad in ("my_rag_app", "no_such_module_xyz:answer", "my_rag_app:missing", "my_rag_app:__doc__"):
        with pytest.raises(AppLoadError):
            load_app(bad)


def test_load_app_finds_build_next_to_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hooks: None) -> None:
    """v2 chunking, the builder contract: `build(chunking) -> answer` is found by name beside the
    answer function, also under a dotted path. Defects: a module with no build loaded as if it
    had one (so a chunking experiment crashes on None), a `build` that is not callable accepted,
    or the builder looked up at module level when answer lives on an object."""
    (tmp_path / "with_build.py").write_text(
        "def answer(q):\n    return 'own'\n\ndef build(chunking):\n    return lambda q: chunking.name\n", encoding="utf-8"
    )
    (tmp_path / "without_build.py").write_text("def answer(q):\n    return q\n", encoding="utf-8")
    (tmp_path / "bad_build.py").write_text("def answer(q):\n    return q\nbuild = 3\n", encoding="utf-8")
    (tmp_path / "dotted.py").write_text(
        "class Bot:\n    def answer(self, q):\n        return 'own'\n    def build(self, chunking):\n        return lambda q: 'built'\nbot = Bot()\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("with_build", "without_build", "bad_build", "dotted"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    loaded = load_app("with_build:answer")
    assert loaded.spec == "with_build:answer" and loaded.answer("hi") == "own" and loaded.build is not None
    assert loaded.for_chunking(Chunking(chunk_size=300))("hi") == "chunk_size=300"
    assert load_app("without_build:answer").build is None
    with pytest.raises(AppLoadError, match="bad_build:build is not callable"):
        load_app("bad_build:answer")
    dotted = load_app("dotted:bot.answer")
    assert dotted.build is not None and dotted.for_chunking(Chunking(title=True))("hi") == "built"


def test_for_chunking_builds_once_per_distinct_chunking_and_never_for_the_apps_own() -> None:
    """Defect: the app rebuilt for every case or every setting that shares a chunking (a sweep's
    round 2 would re-embed the corpus for `top_k=8 title=on` after `title=on`), or built for the
    default chunking, which would change `contexteval run` for every app that offers build()."""
    builds: list[Chunking] = []

    def build(chunking: Chunking) -> App:
        builds.append(chunking)
        return lambda q: chunking.name

    loaded = LoadedApp("app:answer", lambda q: "own", build)
    assert loaded.for_chunking(Chunking()) is loaded.answer and builds == []
    first = loaded.for_chunking(Chunking(chunk_size=300))
    assert loaded.for_chunking(Chunking(chunk_size=300)) is first
    loaded.for_chunking(Chunking(title=True))
    loaded.for_chunking(Chunking(chunk_size=300))
    assert builds == [Chunking(chunk_size=300), Chunking(title=True)]


def test_for_chunking_without_a_builder_states_the_contract() -> None:
    """Defect: a chunking setting on an app with no build() fails with an AttributeError or a
    TypeError on None, instead of one line that says what to add and where."""
    loaded = LoadedApp("my_app:answer", lambda q: "own")
    with pytest.raises(NoBuilder) as e:
        loaded.for_chunking(Chunking(title=True))
    message = str(e.value)
    assert "title=on" in message and "my_app:build" in message and "def build(chunking" in message and "chunk_text" in message
    with pytest.raises(NoBuilder, match="my_app:bot.build"):
        LoadedApp("my_app:bot.answer", lambda q: "own").for_chunking(Chunking(chunk_size=300))
    with pytest.raises(NoBuilder):
        loaded.ensure_buildable([Setting(top_k=8), Setting(top_k=8, overlap=50)])
    loaded.ensure_buildable([Setting(top_k=8)])


def test_a_build_that_raises_or_returns_no_function_is_not_applicable() -> None:
    """Defect: a build() that cannot make the requested chunking (overlap past the app's own size)
    takes down the whole sweep, instead of marking that one candidate not applicable so the search
    goes on with the rest, as a refused top_k does."""

    def build(chunking: Chunking) -> App:
        if chunking.overlap:
            raise ValueError("overlap too large for this corpus")
        return "not a function"  # type: ignore[return-value]

    loaded = LoadedApp("app:answer", lambda q: "own", build)
    with pytest.raises(InterventionNotApplicable, match=r"build\(overlap=500\) raised ValueError: overlap too large"):
        loaded.for_chunking(Chunking(overlap=500))
    with pytest.raises(InterventionNotApplicable, match=r"build\(title=on\) returned str, not an answer function"):
        loaded.for_chunking(Chunking(title=True))


def test_load_app_passes_the_pinecone_text_key_to_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hooks: None) -> None:
    """Review X7. Defect: the CLI has no way to hand the project's pinecone text key to the hooks
    it installs before importing the app."""
    from contexteval._hooks import PineconeCodec

    (tmp_path / "my_rag_app.py").write_text("def answer(q):\n    return q\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "my_rag_app", raising=False)
    load_app("my_rag_app:answer", pinecone_text_key="body")
    assert PineconeCodec().text_key == "body"


def test_cli_reports_a_bad_app_spec_as_a_user_error(project: Path, store: Store, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Review X10. Defect: `contexteval run bad` escapes main() as SystemExit instead of exit
    code 1 with a one-line error on stderr."""
    from contexteval._cli import Config, main, save_config

    save_config(project, Config(provider="fake"))
    store.upsert_cases("planted", [CASE])
    assert main(["run", "no_such_module_xyz:answer", "--yes"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "no_such_module_xyz" in err
