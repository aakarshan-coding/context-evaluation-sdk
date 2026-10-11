"""Local dashboard: stdlib http.server, JSON endpoints, one static page (plan item 11).

Routes are a table of (path pattern, handler). A handler is a function of the Store and the
path's ids that returns a plain dict. The page (ui.html, package data) fetches /api/* and renders
the run picker, failure list, trace detail, and experiment table. Read-only.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Callable

from . import _text as text
from ._store import ExperimentInfo, RunSummary, Store, StoredTrace
from .types import (
    Answered,
    Chunk,
    Crashed,
    Diagnosis,
    Evidence,
    ExperimentId,
    ExperimentRow,
    Ignored,
    Irrelevant,
    Judged,
    RunId,
    Trace,
    TraceId,
    explain,
    label_of,
    never,
    source_of,
    tallies,
)

Json = dict[str, Any]


class NotFound(Exception):
    pass


# --- JSON projections ---------------------------------------------------------


def _run_json(s: RunSummary) -> Json:
    return {
        "id": s.run.id,
        "kind": s.run.kind,
        "app": s.run.app,
        "setting": s.run.setting.name,
        "created_at": s.run.created_at,
        "experiment_id": s.experiment_id,
        "traces": s.traces,
        "passed": s.passed,
        "failed": s.failed,
    }


def _diagnosis_json(d: Diagnosis | None) -> Json | None:
    """`chunks` names the delivered chunks the evidence is about, so the page can mark them."""
    if d is None:
        return None
    replays = () if isinstance(d, Judged) else d.replays
    match d:
        case Irrelevant(removed=refs) | Ignored(gold_at=refs):
            chunks = [list(ref) for ref in refs]
        case _:
            chunks = []
    return {
        "label": label_of(d).value,
        "source": source_of(d),
        "explain": explain(d),
        "chunks": chunks,
        "steps": [{"step": t.step.value, "passed": t.passed, "total": t.total, "majority": t.majority} for t in tallies(replays)],
        "replays": [{"step": r.step.value, "passed": r.passed, "answer": r.answer} for r in replays],
    }


def _outcome_json(st: StoredTrace) -> Json:
    match st.trace.outcome:
        case Answered(text=answer):
            return {"answer": answer, "error": None}
        case Crashed(error=error):
            return {"answer": None, "error": error}
        case other:
            never(other)


def _trace_summary_json(st: StoredTrace) -> Json:
    test = st.trace.test
    return {
        "id": st.id,
        "question": st.trace.question,
        **_outcome_json(st),
        "passed": test.passed if test else None,
        "diagnosis": _diagnosis_json(st.diagnosis),
    }


def _gold(chunks: tuple[Chunk, ...], evidence: Evidence | None) -> list[bool | None]:
    """The diagnoser's gold rule per chunk. None, not False, when the case has no evidence: the
    page then shows no gold mark at all."""
    if evidence is None:
        return [None] * len(chunks)
    return list(text.gold_flags([c.text for c in chunks], evidence))


def _chunk_json(c: Chunk, index: int, gold: bool | None) -> Json:
    return {"index": index, "text": c.text, "id": c.id, "score": c.score, "gold": gold}


def _prompt_has_gold(prompt: str, trace: Trace, delivered_gold: list[list[bool | None]]) -> bool:
    """A prompt holds gold when a gold delivered chunk is in it, as locate_gold decides; the
    question text in the prompt never counts."""
    return any(
        bool(is_gold) and text.chunk_in_prompt(chunk.text, prompt)
        for event, flags in zip(trace.retrievals, delivered_gold)
        for chunk, is_gold in zip(event.delivered, flags)
    )


def _trace_detail_json(st: StoredTrace, store: Store) -> Json:
    trace, test = st.trace, st.trace.test
    case = store.case(test.case_id) if test else None
    evidence = case.evidence if case else None
    in_prompt = set(trace.in_prompt)
    delivered_gold = [_gold(e.delivered, evidence) for e in trace.retrievals]
    return {
        **_trace_summary_json(st),
        "run_id": st.run_id,
        "setting": st.setting.name,
        "expected_answer": case.expected_answer if case else None,
        "evidence": evidence.text if evidence else None,
        "grade_method": test.method if test else None,
        "retrievals": [
            {
                "source": e.source,
                "query": e.query,
                "params": dict(e.params),
                "latency_ms": e.latency_ms,
                "retrieved": [_chunk_json(c, i, g) for i, (c, g) in enumerate(zip(e.retrieved, _gold(e.retrieved, evidence)))],
                "delivered": [
                    {**_chunk_json(c, i, g), "source_index": c.source_index, "in_prompt": (r, i) in in_prompt}
                    for i, (c, g) in enumerate(zip(e.delivered, delivered_gold[r]))
                ],
            }
            for r, e in enumerate(trace.retrievals)
        ],
        "llm_calls": [
            {
                "provider": c.provider,
                "model": c.model,
                "prompt": c.prompt,
                "answer": c.answer,
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
                "latency_ms": c.latency_ms,
                "gold": _prompt_has_gold(c.prompt, trace, delivered_gold) if evidence else None,
            }
            for c in trace.llm_calls
        ],
        "metrics": {
            "input_tokens": trace.metrics.input_tokens,
            "output_tokens": trace.metrics.output_tokens,
            "latency_ms": trace.metrics.latency_ms,
            "cost_usd": trace.metrics.cost_usd,
        },
    }


def _row_json(row: ExperimentRow) -> Json:
    return {
        "setting": row.setting_name,
        "cases": row.cases,
        "passed": row.passed,
        "accuracy": row.accuracy,
        "mean_input_tokens": row.mean_input_tokens,
        "mean_latency_ms": row.mean_latency_ms,
        "total_cost_usd": row.total_cost_usd,
    }


def _experiment_json(store: Store, info: ExperimentInfo) -> Json:
    return {"id": info.id, "created_at": info.created_at, "rows": [_row_json(r) for r in store.experiment_table(info.id)]}


# --- handlers ----------------------------------------------------------------


def api_runs(store: Store) -> Json:
    return {"runs": [_run_json(s) for s in store.runs()]}


def api_run_traces(store: Store, run_id: int) -> Json:
    try:
        store.run(RunId(run_id))
    except KeyError:
        raise NotFound(f"no run {run_id}") from None
    return {"run_id": run_id, "traces": [_trace_summary_json(st) for st in store.traces(RunId(run_id))]}


def api_trace(store: Store, trace_id: int) -> Json:
    st = store.trace(TraceId(trace_id))
    if st is None:
        raise NotFound(f"no trace {trace_id}")
    return _trace_detail_json(st, store)


def api_experiments(store: Store) -> Json:
    return {"experiments": [_experiment_json(store, info) for info in store.experiments()]}


def api_experiment(store: Store, experiment_id: int) -> Json:
    try:
        info = store.experiment(ExperimentId(experiment_id))
    except KeyError:
        raise NotFound(f"no experiment {experiment_id}") from None
    return _experiment_json(store, info)


# Each pattern's groups are integer ids, passed to the handler in order.
ROUTES: tuple[tuple[re.Pattern[str], Callable[..., Json]], ...] = (
    (re.compile(r"/api/runs"), api_runs),
    (re.compile(r"/api/runs/(\d+)/traces"), api_run_traces),
    (re.compile(r"/api/traces/(\d+)"), api_trace),
    (re.compile(r"/api/experiments"), api_experiments),
    (re.compile(r"/api/experiments/(\d+)"), api_experiment),
)


def dispatch(store: Store, path: str) -> tuple[int, Json]:
    for pattern, handler in ROUTES:
        match = pattern.fullmatch(path)
        if match:
            try:
                return 200, handler(store, *(int(g) for g in match.groups()))
            except NotFound as e:
                return 404, {"error": str(e)}
            except Exception as e:  # boundary: the page shows the reason instead of a dropped connection
                return 500, {"error": f"{type(e).__name__}: {e}"}
    return 404, {"error": f"no route {path}"}


# --- server ------------------------------------------------------------------


class _Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already listening, so a
    # second `contexteval ui` would silently share 8765 with the first. Elsewhere it only skips
    # TIME_WAIT, which is what a restart wants.
    allow_reuse_address = sys.platform != "win32"


def make_server(store: Store, port: int = 8765) -> ThreadingHTTPServer:
    """Bound but not yet serving. Port 0 picks a free port (see server.server_address)."""
    page = resources.files("contexteval").joinpath("ui.html").read_bytes()
    # ThreadingHTTPServer runs each request on its own thread, and one sqlite3 connection
    # must not run two statements at once, so requests take turns on the store.
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            # A page on another site can rebind its own hostname to 127.0.0.1 and read the
            # dashboard; the browser still sends that hostname, so only our own names pass.
            port = self.server.server_address[1]
            if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                self._send(403, "application/json", b'{"error": "unexpected Host header"}')
                return
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, "text/html; charset=utf-8", page)
                return
            with lock:
                status, body = dispatch(store, path)
            self._send(status, "application/json", json.dumps(body).encode("utf-8"))

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return _Server(("127.0.0.1", port), Handler)


def serve(store: Store, port: int = 8765, open_browser: bool = True) -> None:
    """Block until Ctrl-C."""
    server = make_server(store, port=port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"contexteval dashboard at {url} (Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
