"""Dashboard over real HTTP: a seeded store, the server on port 0 in a daemon thread, and urllib
with a timeout on every request. Plan DoD item 4 and architecture graft 2."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

import pytest

from contexteval import _cli, _ui
from contexteval._intervention import BASELINE, Order, Setting
from contexteval._store import Store
from contexteval.types import (
    Answered,
    CaseId,
    Chunk,
    Evidence,
    ExperimentId,
    Flaky,
    Ignored,
    Irrelevant,
    Judged,
    JudgeVerdict,
    LLMCall,
    Metrics,
    Missing,
    NotContext,
    ReplayOutcome,
    ReplayStep,
    RetrievalEvent,
    RunId,
    TestCase,
    TestOutcome,
    Trace,
    TraceId,
)

TIMEOUT = 10
GOLD = "The refund window closes thirty days after delivery."
SCRIPT = "<script>alert('x')</script> & <b>bold</b>"
OTHER = "Shipping takes five business days."
REFUND = TestCase(CaseId("c-refund"), "When does the refund window close?", "thirty days", Evidence(GOLD), "docs/policy.md")
SHIPPING = TestCase(CaseId("c-ship"), "How long is shipping?", "five business days", Evidence(OTHER), "docs/ship.md")
IMPORTED = TestCase(CaseId("c-import"), "Who signs refunds?", "the manager", None, "tests.jsonl")


def _chunk(text: str, i: int | None, score: float | None = -0.1) -> Chunk:
    return Chunk(text, None if i is None else f"doc{i}", score, i)


def _trace(case: TestCase | None, passed: bool | None, answer: str, delivered_texts: tuple[str, ...], in_prompt: tuple[tuple[int, int], ...]) -> Trace:
    retrieved = (_chunk(OTHER, 0), _chunk(SCRIPT, 1), _chunk(GOLD, 2))
    delivered = tuple(_chunk(t, retrieved_index(t)) for t in delivered_texts)
    prompt = "\n\n".join(delivered_texts[r] for _, r in in_prompt) + f"\n\nQ: {case.question if case else 'q'}"
    return Trace(
        question=case.question if case else "production question",
        outcome=Answered(answer),
        retrievals=(RetrievalEvent("chromadb", "refund window", {"n_results": 3}, retrieved, delivered, 4.0),),
        llm_calls=(LLMCall("openai", "gpt-4o-mini", prompt, answer, 40, 5, 200.0),),
        metrics=Metrics(40, 5, 210.0, 0.0002),
        test=TestOutcome(case.id, passed, "llm") if case is not None and passed is not None else None,
        in_prompt=in_prompt,
    )


def retrieved_index(text: str) -> int:
    return (OTHER, SCRIPT, GOLD).index(text)


def _replay(step: ReplayStep, passed: bool, answer: str | None = "x") -> ReplayOutcome:
    return ReplayOutcome(step, passed, answer)


IGNORED = Ignored(
    replays=(
        _replay(ReplayStep.BASELINE_RERUN, False, "sixty days"),
        _replay(ReplayStep.BASELINE_RERUN, False, "sixty days"),
        _replay(ReplayStep.GOLD_TO_TOP, True, "thirty days"),
        _replay(ReplayStep.GOLD_TO_TOP, False, "sixty days"),
        _replay(ReplayStep.GOLD_TO_TOP, True, "thirty days"),
    ),
    gold_at=((0, 2),),
)
MISSING = Missing("not_retrieved", True, (_replay(ReplayStep.BASELINE_RERUN, False), _replay(ReplayStep.INJECT_EVIDENCE, True)))
IRRELEVANT = Irrelevant(
    replays=(_replay(ReplayStep.BASELINE_RERUN, False), _replay(ReplayStep.GOLD_TO_TOP, False), _replay(ReplayStep.ONLY_GOLD, True)),
    removed=((0, 0), (0, 1)),
)
NOT_CONTEXT = NotContext((_replay(ReplayStep.BASELINE_RERUN, False, None),))
FLAKY = Flaky((_replay(ReplayStep.BASELINE_RERUN, True, "thirty days"),))
JUDGED = Judged(JudgeVerdict.IGNORED, "the prompt held the policy but the answer skipped it")


@dataclass(frozen=True)
class Seed:
    base: RunId
    production: RunId
    experiment: ExperimentId
    ignored: TraceId
    missing: TraceId
    irrelevant: TraceId
    not_context: TraceId
    flaky: TraceId
    passing: TraceId
    imported: TraceId
    judged: TraceId


def _seed(store: Store) -> Seed:
    store.upsert_cases("synthetic", [REFUND, SHIPPING])
    store.upsert_cases("tests", [IMPORTED])
    base = store.new_run("baseline", "app:answer", BASELINE)
    all_three = (OTHER, SCRIPT, GOLD)
    ignored = store.add_trace(base, _trace(REFUND, False, "sixty days", all_three, ((0, 0), (0, 2))))
    missing = store.add_trace(base, _trace(REFUND, False, "unknown", (OTHER, SCRIPT), ((0, 0), (0, 1))))
    irrelevant = store.add_trace(base, _trace(REFUND, False, "five days", all_three, ((0, 0), (0, 1), (0, 2))))
    not_context = store.add_trace(base, _trace(REFUND, False, "no idea", all_three, ((0, 0), (0, 1), (0, 2))))
    flaky = store.add_trace(base, _trace(REFUND, False, "maybe", all_three, ((0, 0), (0, 1), (0, 2))))
    passing = store.add_trace(base, _trace(SHIPPING, True, "five business days", all_three, ((0, 0),)))
    imported = store.add_trace(base, _trace(IMPORTED, False, "the clerk", all_three, ((0, 0),)))
    for tid, d in [(ignored, IGNORED), (missing, MISSING), (irrelevant, IRRELEVANT), (not_context, NOT_CONTEXT), (flaky, FLAKY), (imported, JUDGED)]:
        store.save_diagnosis(tid, d)

    production = store.new_run("production", None, BASELINE)
    judged = store.add_trace(production, _trace(None, None, "thirty days", all_three, ((0, 2),)))
    store.save_diagnosis(judged, Judged(JudgeVerdict.NO_FAILURE, "answer matches the policy"))

    experiment = store.new_experiment()
    for setting, outcomes in [(BASELINE, (False, True)), (Setting(top_k=10), (True, True))]:
        run = store.new_run("experiment", "app:answer", setting, experiment=experiment)
        for case, passed in zip((REFUND, SHIPPING), outcomes):
            store.add_trace(run, _trace(case, passed, "a", all_three, ((0, 0),)))
    return Seed(base, production, experiment, ignored, missing, irrelevant, not_context, flaky, passing, imported, judged)


@dataclass(frozen=True)
class Served:
    url: str
    seed: Seed


@pytest.fixture
def served(project: Path) -> Iterator[Served]:
    store = Store.open(project)
    seed = _seed(store)
    server = _ui.make_server(store, port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield Served(f"http://127.0.0.1:{server.server_address[1]}", seed)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(TIMEOUT)
        store.close()


def _get(url: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            return r.status, r.headers["Content-Type"], r.read()
    except urllib.error.HTTPError as e:
        with e:
            return e.code, e.headers["Content-Type"], e.read()


def _json(url: str) -> Any:
    status, content_type, body = _get(url)
    assert (status, content_type) == (200, "application/json"), body
    return json.loads(body)


def test_runs_endpoint_lists_seeded_runs_with_counts(served: Served) -> None:
    """Defect: the run picker is empty or miscounts, e.g. drops production or experiment runs,
    or counts the ungraded production trace as a failure."""
    runs = _json(served.url + "/api/runs")["runs"]
    by_id = {r["id"]: r for r in runs}
    assert [r["kind"] for r in runs] == ["experiment", "experiment", "production", "baseline"]
    base = by_id[served.seed.base]
    assert (base["app"], base["setting"], base["traces"], base["passed"], base["failed"]) == ("app:answer", "baseline", 7, 1, 6)
    prod = by_id[served.seed.production]
    assert (prod["app"], prod["traces"], prod["passed"], prod["failed"]) == (None, 1, 0, 0)
    assert [r["setting"] for r in runs[:2]] == ["top_k=10", "baseline"]
    assert {r["experiment_id"] for r in runs[:2]} == {served.seed.experiment}


def test_run_traces_carry_pass_fail_label_and_explain_line(served: Served) -> None:
    """Defect: the failure list shows the wrong label for a trace, loses the evidence line,
    or marks a passing trace as failed."""
    s = served.seed
    body = _json(f"{served.url}/api/runs/{s.base}/traces")
    rows = {t["id"]: t for t in body["traces"]}
    got = {tid: (rows[tid]["passed"], rows[tid]["diagnosis"] and rows[tid]["diagnosis"]["label"]) for tid in rows}
    assert got == {
        s.ignored: (False, "ignored"),
        s.missing: (False, "missing"),
        s.irrelevant: (False, "irrelevant"),
        s.not_context: (False, "not_context"),
        s.flaky: (False, "flaky"),
        s.passing: (True, None),
        s.imported: (False, "ignored"),
    }
    assert rows[s.ignored]["diagnosis"]["explain"] == "moving the gold chunk to the top fixed it (2/3); gold was at chunk 3 of retrieval 1"
    assert rows[s.irrelevant]["diagnosis"]["explain"] == "keeping only the gold chunks fixed it (1/1); removed chunks 1, 2 of retrieval 1"
    assert rows[s.missing]["diagnosis"]["explain"] == "the gold chunk was never retrieved; injecting the evidence fixed it (1/1)"
    assert rows[s.imported]["diagnosis"]["source"] == "judge"
    assert rows[s.imported]["diagnosis"]["source"] != rows[s.ignored]["diagnosis"]["source"]
    assert rows[s.ignored]["answer"] == "sixty days"
    prod = _json(f"{served.url}/api/runs/{s.production}/traces")["traces"]
    assert [(t["id"], t["passed"], t["diagnosis"]["label"]) for t in prod] == [(s.judged, None, "no_failure")]


def test_trace_detail_marks_in_prompt_gold_and_diagnosis_chunks(served: Served) -> None:
    """Defect: the detail view confuses retrieved with delivered chunks, loses in-prompt or gold
    marks, drops the expected answer, prompt, or metrics, or cannot point at the chunks the
    diagnosis names."""
    t = _json(f"{served.url}/api/traces/{served.seed.missing}")
    assert (t["question"], t["answer"], t["expected_answer"], t["evidence"]) == (REFUND.question, "unknown", "thirty days", GOLD)
    (event,) = t["retrievals"]
    assert [c["text"] for c in event["retrieved"]] == [OTHER, SCRIPT, GOLD]
    assert [c["gold"] for c in event["retrieved"]] == [False, False, True]
    assert [(c["text"], c["in_prompt"], c["gold"]) for c in event["delivered"]] == [(OTHER, True, False), (SCRIPT, True, False)]
    (call,) = t["llm_calls"]
    assert call["prompt"] == f"{OTHER}\n\n{SCRIPT}\n\nQ: {REFUND.question}"
    assert call["gold"] is False
    assert t["metrics"] == {"input_tokens": 40, "output_tokens": 5, "latency_ms": 210.0, "cost_usd": 0.0002}
    assert [(r["step"], r["passed"]) for r in t["diagnosis"]["replays"]] == [("baseline_rerun", False), ("inject_evidence", True)]

    ignored = _json(f"{served.url}/api/traces/{served.seed.ignored}")
    assert ignored["diagnosis"]["chunks"] == [[0, 2]]
    assert ignored["diagnosis"]["steps"] == [
        {"step": "baseline_rerun", "passed": 0, "total": 2, "majority": False},
        {"step": "gold_to_top", "passed": 2, "total": 3, "majority": True},
    ]
    assert [r["passed"] for r in ignored["diagnosis"]["replays"]] == [False, False, True, False, True]
    assert [(c["in_prompt"], c["gold"]) for c in ignored["retrievals"][0]["delivered"]] == [(True, False), (False, False), (True, True)]
    assert ignored["llm_calls"][0]["gold"] is True
    irrelevant = _json(f"{served.url}/api/traces/{served.seed.irrelevant}")
    assert irrelevant["diagnosis"]["chunks"] == [[0, 0], [0, 1]]


def test_trace_without_evidence_has_no_gold_marks(served: Served) -> None:
    """Defect: a case with no evidence shows every chunk as "not gold", which reads as a finding
    rather than as missing ground truth."""
    t = _json(f"{served.url}/api/traces/{served.seed.imported}")
    assert t["evidence"] is None
    assert t["expected_answer"] == "the manager"
    assert {c["gold"] for c in t["retrievals"][0]["delivered"]} == {None}
    assert t["llm_calls"][0]["gold"] is None
    assert t["diagnosis"]["explain"] == f"judge: ignored; {JUDGED.rationale}"


def test_experiments_endpoint_serves_experiment_table_rows(served: Served) -> None:
    """Defect: the comparison table shows the wrong setting, accuracy, or cost per row."""
    (exp,) = _json(served.url + "/api/experiments")["experiments"]
    assert exp["id"] == served.seed.experiment
    assert [(r["setting"], r["cases"], r["passed"], r["accuracy"]) for r in exp["rows"]] == [
        ("baseline", 2, 1, 0.5),
        ("top_k=10", 2, 2, 1.0),
    ]
    assert exp["rows"][1]["mean_input_tokens"] == 40
    assert exp["rows"][1]["total_cost_usd"] == pytest.approx(0.0004)
    assert _json(f"{served.url}/api/experiments/{served.seed.experiment}") == exp
    assert exp["kind"] == "manual" and exp["recommendation"] is None
    assert [(r["split"], r["recommended"]) for r in exp["rows"]] == [(None, False), (None, False)]


def test_experiments_endpoint_marks_a_sweeps_recommended_row_and_its_splits(served: Served) -> None:
    """v2 sweep. Defect: a sweep listed as a plain experiment, so the page marks its most accurate
    tune row (which may be the one that lost on holdout) instead of the recommended row, shows no
    recommendation, or cannot tell its tune rows from its holdout rows."""
    from contexteval._sweep import FailureKind, Split, SweepReport

    store = Store.open(Path.cwd())
    try:
        sweep = store.new_experiment("sweep")
        runs = {}
        for name, setting, outcomes in [
            ("base", BASELINE, (False, True)),
            ("wide", Setting(top_k=10), (True, True)),
            ("ends", Setting(order=Order.ENDS), (True, True)),
            ("hold_base", BASELINE, (True,)),
            ("hold_wide", Setting(top_k=10), (True,)),
        ]:
            run = store.new_run("experiment", "app:answer", setting, experiment=sweep)
            for case, passed in zip((REFUND, SHIPPING), outcomes):
                store.add_trace(run, _trace(case, passed, "a", (OTHER, SCRIPT, GOLD), ((0, 0),)))
            runs[name] = run
        report = SweepReport(
            source_run=served.seed.base,
            baseline_k=4,
            counts={FailureKind.NOT_RETRIEVED: 1},
            split=Split(0, 0.3, (REFUND.id, SHIPPING.id), (REFUND.id,)),
            baseline_run=runs["base"],
            rounds=((runs["wide"], runs["ends"]),),
            holdout_runs=(runs["hold_base"], runs["hold_wide"]),
            recommended=runs["wide"],
            reason="",
        )
        store.save_experiment_report(sweep, report.to_json())
    finally:
        store.close()
    exp = _json(f"{served.url}/api/experiments/{sweep}")
    assert exp["kind"] == "sweep"
    assert exp["recommendation"] == "Set top_k=10. Accuracy 50% -> 100% (holdout 100% -> 100%). Uses about the same input tokens; mean latency 210 ms -> 210 ms."
    assert [(r["setting"], r["split"], r["recommended"]) for r in exp["rows"]] == [
        ("baseline", "tune", False),
        ("top_k=10", "tune", True),
        ("order=ends", "tune", False),
        ("baseline", "holdout", False),
        ("top_k=10", "holdout", False),
    ]
    listed = _json(served.url + "/api/experiments")["experiments"]
    assert [(e["id"], e["kind"]) for e in listed] == [(sweep, "sweep"), (served.seed.experiment, "manual")]


def test_experiments_endpoint_survives_a_bad_or_missing_sweep_report(served: Served) -> None:
    """Review Q8. Defect: one sweep whose stored report does not parse (written by another build,
    or cut short) makes /api/experiments a 500, so the page shows no experiment at all. Each
    experiment renders on its own; a sweep with an unreadable or missing report shows its rows
    and no recommendation, and says the report could not be read."""
    store = Store.open(Path.cwd())
    try:
        garbled = store.new_experiment("sweep")
        store.add_trace(store.new_run("experiment", "app:answer", BASELINE, experiment=garbled), _trace(REFUND, False, "a", (OTHER,), ((0, 0),)))
        store.save_experiment_report(garbled, {"recommended": 1, "rounds": "not a list"})
        unfinished = store.new_experiment("sweep")
        store.add_trace(store.new_run("experiment", "app:answer", BASELINE, experiment=unfinished), _trace(REFUND, True, "a", (OTHER,), ((0, 0),)))
    finally:
        store.close()
    listed = {e["id"]: e for e in _json(served.url + "/api/experiments")["experiments"]}
    assert set(listed) == {garbled, unfinished, served.seed.experiment}
    assert listed[garbled]["recommendation"] is None and listed[garbled]["report_error"]
    assert [(r["setting"], r["passed"], r["split"], r["recommended"]) for r in listed[garbled]["rows"]] == [("baseline", 0, None, False)]
    assert listed[unfinished]["recommendation"] is None and listed[unfinished]["report_error"] is None
    assert [(r["setting"], r["passed"]) for r in listed[unfinished]["rows"]] == [("baseline", 1)]
    assert [r["setting"] for r in listed[served.seed.experiment]["rows"]] == ["baseline", "top_k=10"]
    assert _json(f"{served.url}/api/experiments/{garbled}")["report_error"] == listed[garbled]["report_error"]


def test_ui_does_not_import_the_runner() -> None:
    """Review Q8. Defect: importing the dashboard imports the sweep, which imports the runner,
    which loads app machinery the dashboard never uses; a runner import error then takes the
    dashboard down with it."""
    probe = "import sys, contexteval._ui; print(sorted(m for m in sys.modules if m.startswith('contexteval._') and m.endswith('_runner')))"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


@pytest.mark.parametrize("path", ["/api/runs/999/traces", "/api/traces/999", "/api/experiments/999", "/api/nope", "/api/traces/abc"])
def test_unknown_ids_and_routes_are_json_404(served: Served, path: str) -> None:
    """Defect: an unknown id returns 200 with empty data, a 500, or an HTML error page the
    page's fetch cannot parse."""
    status, content_type, body = _get(served.url + path)
    assert (status, content_type) == (404, "application/json")
    assert "error" in json.loads(body)


def test_script_in_chunk_round_trips_as_data(served: Served) -> None:
    """Defect: chunk text is escaped, stripped, or served as HTML, so the page shows mangled
    text or the browser could execute it."""
    status, content_type, body = _get(f"{served.url}/api/traces/{served.seed.ignored}")
    assert (status, content_type) == (200, "application/json")
    texts = [c["text"] for c in json.loads(body)["retrievals"][0]["delivered"]]
    assert texts[1] == SCRIPT


def test_index_serves_the_page(served: Served) -> None:
    """Defect: GET / is 404 or serves something other than the packaged ui.html."""
    status, content_type, body = _get(served.url + "/")
    assert (status, content_type) == (200, "text/html; charset=utf-8")
    assert body == resources.files("contexteval").joinpath("ui.html").read_bytes()
    for anchor in ['id="run-picker"', 'id="failures"', 'id="detail"', 'id="experiments"', "/api/runs", "/api/experiments"]:
        assert anchor in body.decode("utf-8")


def _page() -> str:
    return resources.files("contexteval").joinpath("ui.html").read_text(encoding="utf-8")


def test_page_loads_no_external_resources() -> None:
    """Defect: the page pulls a script, font, or stylesheet from the network, so it breaks
    offline and leaks that the dashboard was opened."""
    page = _page()
    assert re.findall(r"(?:https?:)?//[\w-]+(?:\.[\w-]+)+", page) == []
    assert re.findall(r"<(?:link|iframe|img)\b|\bsrc\s*=|@import|url\(", page, re.IGNORECASE) == []


def test_page_never_builds_markup_from_strings() -> None:
    """Defect: a database string reaches an HTML parser (innerHTML and friends), so a chunk
    holding <script> or an onerror attribute runs in the dashboard."""
    page = _page()
    assert re.findall(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|createContextualFragment", page) == []


def test_foreign_host_header_is_refused(served: Served) -> None:
    """Defect: a web page that rebinds its own hostname to 127.0.0.1 (DNS rebinding) can read
    every stored prompt and answer through the dashboard."""
    request = urllib.request.Request(served.url + "/api/runs", headers={"Host": "attacker.example:80"})
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=TIMEOUT)
    with caught.value as response:
        assert response.code == 403
        assert b"runs" not in response.read()
    port = served.url.rsplit(":", 1)[1]
    local = urllib.request.Request(served.url + "/api/runs", headers={"Host": f"localhost:{port}"})
    with urllib.request.urlopen(local, timeout=TIMEOUT) as response:
        assert response.status == 200


def test_concurrent_requests_all_succeed(served: Served) -> None:
    """Defect: two request threads share the sqlite connection unguarded, so some requests fail
    or return another request's rows."""
    url = f"{served.url}/api/traces/{served.seed.ignored}"
    expected = _json(url)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: _json(url), range(40)))
    assert all(r == expected for r in results)


def test_second_server_on_a_busy_port_fails(served: Served) -> None:
    """Defect: on Windows SO_REUSEADDR lets a second dashboard bind the same port, so requests
    land on either server at random."""
    port = int(served.url.rsplit(":", 1)[1])
    store = Store.open(Path.cwd())
    try:
        with pytest.raises(OSError):
            _ui.make_server(store, port=port)
    finally:
        store.close()


@dataclass
class UiCommand:
    """`contexteval ui` running on a daemon thread, with every server it built kept so the test
    can always shut it down."""

    thread: threading.Thread
    servers: list[Any]
    result: list[int]

    def stop(self) -> None:
        for server in self.servers:
            server.shutdown()
        self.thread.join(TIMEOUT)


def _start_ui(monkeypatch: pytest.MonkeyPatch, *argv: str) -> UiCommand:
    servers: list[Any] = []
    real_make_server = _ui.make_server

    def recording_make_server(*args: Any, **kwargs: Any) -> Any:
        servers.append(real_make_server(*args, **kwargs))
        return servers[-1]

    monkeypatch.setattr(_ui, "make_server", recording_make_server)
    result: list[int] = []
    thread = threading.Thread(target=lambda: result.append(_cli.main(["ui", *argv])), daemon=True)
    thread.start()
    return UiCommand(thread, servers, result)


def test_ui_command_serves_the_project_store(project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: `contexteval ui` opens the wrong store, ignores --port, or opens a browser under
    --no-open."""
    seeded = Store.open(project)
    run = seeded.new_run("baseline", "app:answer", BASELINE)
    seeded.close()
    opened: list[str] = []
    monkeypatch.setattr(_ui.webbrowser, "open", opened.append)
    ui = _start_ui(monkeypatch, "--port", "0", "--no-open")
    try:
        for _ in range(200):
            if ui.servers:
                break
            threading.Event().wait(0.05)
        (server,) = ui.servers
        url = f"http://127.0.0.1:{server.server_address[1]}"
        assert [r["id"] for r in _json(url + "/api/runs")["runs"]] == [run]
    finally:
        ui.stop()
    assert ui.result == [0]
    assert opened == []
    assert f"{url}/" in capsys.readouterr().out


def test_ui_command_reports_busy_port(served: Served, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Defect: a busy port crashes `contexteval ui` with a traceback instead of one error line,
    or the command binds the busy port anyway and serves beside the first dashboard."""
    port = served.url.rsplit(":", 1)[1]
    ui = _start_ui(monkeypatch, "--port", port, "--no-open")
    ui.thread.join(TIMEOUT)
    bound_anyway = ui.thread.is_alive()
    ui.stop()
    assert not bound_anyway, "contexteval ui started serving on a port another server holds"
    assert ui.result == [1]
    assert f"cannot serve on port {port}" in capsys.readouterr().err


def test_gold_marks_follow_the_shared_gold_rule(store: Store) -> None:
    """Review Y2. The dashboard must mark gold the way the diagnoser decides it. Defects: the two
    halves of evidence split across adjacent chunks shown as "not gold" while the diagnosis moves
    them as gold; or a prompt marked "contains gold" because the question repeats the evidence."""
    evidence = "The refund window closes thirty days after delivery for every online order."
    halves = ("Refunds. The refund window closes thirty", "days after delivery for every online order. Store credit is issued instead.")
    case = TestCase(CaseId("c-split"), "Does the refund window close thirty days after delivery for every online order?", "yes", Evidence(evidence), "d.md")
    store.upsert_cases("synthetic", [case])
    run = store.new_run("baseline", "app:answer", BASELINE)
    chunks = (_chunk(OTHER, 0), _chunk(halves[0], 1), _chunk(halves[1], 2))
    split = Trace(
        question=case.question,
        outcome=Answered("no"),
        retrievals=(RetrievalEvent("chromadb", case.question, {}, chunks, chunks, 1.0),),
        llm_calls=(
            LLMCall("openai", "m", f"{OTHER}\n{halves[0]}\n{halves[1]}\nQ: {case.question}", "no", 1, 1, 1.0),
            LLMCall("openai", "m", f"{OTHER}\nQ: {case.question}", "no", 1, 1, 1.0),
        ),
        metrics=Metrics(1, 1, 1.0, None),
        test=TestOutcome(case.id, False, "llm"),
        in_prompt=((0, 0), (0, 1), (0, 2)),
    )
    detail = _ui.api_trace(store, store.add_trace(run, split))
    assert [c["gold"] for c in detail["retrievals"][0]["delivered"]] == [False, True, True]
    assert [c["gold"] for c in detail["retrievals"][0]["retrieved"]] == [False, True, True]
    assert [c["gold"] for c in detail["llm_calls"]] == [True, False]


def test_handler_error_is_a_json_500(served: Served, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review Y6. Defect: an exception inside a handler (a locked or corrupt database) escapes
    do_GET, the server drops the connection, and the page shows a network error with no reason."""

    def broken(self: Store) -> None:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(Store, "runs", broken)
    status, content_type, body = _get(served.url + "/api/runs")
    assert (status, content_type) == (500, "application/json")
    assert "database is locked" in json.loads(body)["error"]
    assert _json(served.url + "/api/experiments")["experiments"]


# Runs the page's own script in node against the live server, with just enough DOM to build
# elements, then prints the badge text of each row in the failure list.
_PAGE_HARNESS = r"""
const fs = require("fs");
const [htmlPath, base, runId, traceId] = process.argv.slice(2);
const script = fs.readFileSync(htmlPath, "utf8").match(/<script>([\s\S]*)<\/script>/)[1];
class Node {}
class El extends Node {
  constructor(tag) { super(); this.tag = tag; this.children = []; this.dataset = {}; this.textContent = ""; this.checked = true; this.value = ""; }
  append(...c) { this.children.push(...c); }
  replaceChildren() { this.children = []; }
  setAttribute(k, v) { if (k.startsWith("data-")) this.dataset[k.slice(5)] = v; }
  addEventListener() {}
  scrollIntoView() {}
  get parentElement() { return new El("div"); }
}
class Text extends Node { constructor(t) { super(); this.textContent = t; this.children = []; } }
const ids = {};
globalThis.Node = Node;
globalThis.document = {
  createElement: (t) => new El(t), createTextNode: (t) => new Text(t),
  getElementById: (id) => (ids[id] ??= new El(id)), querySelectorAll: () => [],
};
globalThis.location = { hash: "#run=" + runId };
globalThis.history = { replaceState() {} };
globalThis.matchMedia = () => ({ matches: false });
const realFetch = fetch;
globalThis.fetch = (path, opts) => realFetch(base + path, opts);
(0, eval)(script + "\nglobalThis.selectTrace = selectTrace;");
const started = Date.now();
const text = (n) => (n.children && n.children.length ? n.children.map(text).join("") : n.textContent);
const find = (n, cls) => (n.className === cls ? [n] : (n.children || []).flatMap((c) => find(c, cls)));
const timer = setInterval(async () => {
  const rows = (ids.failures || { children: [] }).children;
  if (!rows.length && Date.now() - started < 5000) return;
  clearInterval(timer);
  if (!traceId) {
    console.log(JSON.stringify(rows.map((li) => li.children[0].children[0].children[0].textContent)));
    return;
  }
  await selectTrace(Number(traceId));
  const [list] = find(ids.detail, "replays");
  console.log(JSON.stringify(list.children.map((li) => ({
    head: text({ children: li.children.filter((c) => c.tag !== "ul") }),
    samples: li.children.filter((c) => c.tag === "ul").flatMap((ul) => ul.children.map(text)),
  }))));
}, 50);
"""


def _run_page(served: Served, tmp_path: Path, *args: str) -> Any:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    harness = tmp_path / "harness.js"
    harness.write_text(_PAGE_HARNESS, encoding="utf-8")
    page = resources.files("contexteval").joinpath("ui.html")
    out = subprocess.run(
        [node, str(harness), str(page), served.url, *args],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_detail_shows_each_step_with_its_pass_count_and_every_sample(served: Served, tmp_path: Path) -> None:
    """v1.1 A. Defect: the detail view lists one line per sample with no per-step verdict, so a
    2-of-3 step reads as three contradicting lines, or it shows only the majority and hides the
    dissenting answers. Each step shows its majority badge, name, count, and every sample."""
    s = served.seed
    assert _run_page(served, tmp_path, str(s.base), str(s.ignored)) == [
        {"head": "fail baseline rerun (0/2)", "samples": ["fail: sixty days", "fail: sixty days"]},
        {"head": "pass gold to top (2/3)", "samples": ["pass: thirty days", "fail: sixty days", "pass: thirty days"]},
    ]
    assert _run_page(served, tmp_path, str(s.base), str(s.not_context)) == [
        {"head": "fail baseline rerun (0/1)", "samples": ["fail (crashed)"]},
    ]


def test_list_shows_the_judge_label_for_judged_production_traces(served: Served, tmp_path: Path) -> None:
    """Review Y6. Defect: the failure list shows "ungraded" for a production trace the judge has
    labeled, because the list only shows a diagnosis badge for graded failures."""
    assert _run_page(served, tmp_path, str(served.seed.production)) == ["judge: no failure"]
