"""Hooks against real chromadb and the real openai / anthropic SDKs over mock HTTP transports,
pinecone through a stand-in class, plus install idempotence and the mode rules."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest

chromadb = pytest.importorskip("chromadb")
openai = pytest.importorskip("openai")

from contexteval import _capture as capture  # noqa: E402
from contexteval import _hooks  # noqa: E402
from contexteval._intervention import BASELINE, InjectEvidence, Intervention, InterventionNotApplicable, KeepFirst, LexicalRerank  # noqa: E402
from contexteval._store import SCHEMA_VERSION, Store  # noqa: E402
from contexteval.demo.app import HashEmbedding  # noqa: E402
from contexteval.types import Answered, Evidence  # noqa: E402

DOCS = {
    "d1": "The refund window closes thirty days after delivery.",
    "d2": "Refunds are paid to the original payment method.",
    "d3": "Shipping takes three to five business days.",
    "d4": "Our office cat is named Pickles.",
    "d5": "Gift cards cannot be refunded or exchanged.",
}
QUESTION = "When does the refund window close?"
GOLD = Evidence("Refund requests need the order number.")
ANSWER = "thirty days"


# --- app-side doubles -----------------------------------------------------------


@pytest.fixture
def chroma() -> Any:
    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    col = client.create_collection(f"docs_{uuid.uuid4().hex}", embedding_function=HashEmbedding())
    col.add(ids=list(DOCS), documents=list(DOCS.values()), metadatas=[{"src": k} for k in DOCS])
    return col


def _openai_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if request.url.path.endswith("/chat/completions"):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": ANSWER}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
            },
        )
    if request.url.path.endswith("/responses"):
        return httpx.Response(
            200,
            json={
                "id": "resp-1", "object": "response", "created_at": 0, "model": body["model"], "status": "completed",
                "output": [{"type": "message", "id": "msg-1", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": ANSWER, "annotations": []}]}],
                "usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11,
                          "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
                "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            },
        )
    return httpx.Response(404, json={"error": "unexpected path"})


@pytest.fixture
def openai_client() -> Any:
    return openai.OpenAI(api_key="fake", http_client=httpx.Client(transport=httpx.MockTransport(_openai_handler)))


@pytest.fixture
def async_openai_client() -> Any:
    return openai.AsyncOpenAI(api_key="fake", http_client=httpx.AsyncClient(transport=httpx.MockTransport(_openai_handler)))


def _anthropic_response(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "msg-1", "type": "message", "role": "assistant", "model": body["model"],
        "content": [{"type": "text", "text": ANSWER}], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 3},
    }


class _FakeAsyncChromaClient:
    """The HTTP half of chromadb's AsyncClient. AsyncCollection.query validates, embeds, and
    reshapes around this one coroutine, so the real collection class runs in the test."""

    def __init__(self) -> None:
        self.n_results: list[int] = []

    async def _query(self, **kw: Any) -> Any:
        n = kw["n_results"]
        self.n_results.append(n)
        rows = list(DOCS.items())[:n]
        return {
            "ids": [[k for k, _ in rows]], "documents": [[v for _, v in rows]], "distances": [[0.1 * i for i in range(len(rows))]],
            "metadatas": [[{"src": k} for k, _ in rows]], "embeddings": None, "uris": None, "data": None,
            "included": ["metadatas", "documents", "distances"],
        }


@pytest.fixture
def async_chroma() -> tuple[Any, _FakeAsyncChromaClient]:
    from chromadb.api.models.AsyncCollection import AsyncCollection
    from chromadb.types import Collection as CollectionModel

    client = _FakeAsyncChromaClient()
    model = CollectionModel(id=uuid.uuid4(), name="docs", configuration_json={}, serialized_schema=None, metadata=None, dimension=None, tenant="t", database="d")
    return AsyncCollection(client, model, embedding_function=HashEmbedding()), client


def _app(col: Any, client: Any, question: str, *, n_results: int = 3, use: int = 2) -> tuple[str, Any]:
    """A user's answer(): retrieve, put only the first `use` chunks into the prompt, ask the LLM."""
    res = col.query(query_texts=[question], n_results=n_results)
    context = "\n".join(res["documents"][0][:use])
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Answer from context."}, {"role": "user", "content": f"Context:\n{context}\n\nQ: {question}"}],
    )
    return r.choices[0].message.content, res


@dataclass
class Match:
    id: str
    score: float
    metadata: dict[str, Any]
    values: list[float] = field(default_factory=list)


@dataclass
class QueryResponse:
    matches: list[Match]
    namespace: str = ""


class FakeIndex:
    """Stand-in for pinecone Index: same query signature, same response shape."""

    def __init__(self, matches: list[Match]) -> None:
        self.matches = matches
        self.calls: list[int] = []
        self.last: QueryResponse | None = None

    def query(self, vector: list[float] | None = None, top_k: int = 10, include_metadata: bool = False, namespace: str = "") -> QueryResponse:
        self.calls.append(top_k)
        self.last = QueryResponse(matches=self.matches[:top_k])
        return self.last


@pytest.fixture
def pinecone_index() -> Iterator[FakeIndex]:
    assert _hooks.patch_method(FakeIndex, "query", _hooks.PineconeCodec()) == "installed"
    try:
        yield FakeIndex([
            Match("m1", 0.9, {"text": "dogs bark"}),
            Match("m2", 0.8, {"text": "refund policy details"}),
            Match("m3", 0.7, {"text": "cats purr"}),
        ])
    finally:
        _hooks.uninstall_all()
        capture.reset_process()


@pytest.fixture
def production(project: Path) -> Iterator[capture.InstallReport]:
    report = _hooks.install("production", project)
    try:
        yield report
    finally:
        _hooks.uninstall_all()
        capture.reset_process()


# --- P1 acceptance --------------------------------------------------------------


def test_p1_acceptance_stored_trace_has_retrieval_llm_and_in_prompt(chroma: Any, openai_client: Any, hooks: None, store: Store) -> None:
    """P1 acceptance. The collection and client exist before install, so this also proves
    class-level patching reaches pre-existing instances. Defects: retrieval not captured,
    prompt or token usage missing from the LLM call, or a chunk the app never put into the
    prompt reported as in_prompt."""
    with capture.active_run(QUESTION) as run:
        answer, res = _app(chroma, openai_client, QUESTION)
    assert answer == ANSWER
    trace = capture.finalize_trace(run, Answered(answer), latency_ms=12.5)
    tid = store.add_trace(store.new_run("baseline", "tests:app", BASELINE), trace)
    loaded = store.trace(tid)
    assert loaded is not None and loaded.trace == trace
    stored = loaded.trace

    assert len(stored.retrievals) == 1
    ev = stored.retrievals[0]
    assert ev.source == "chromadb" and ev.query == QUESTION and ev.params["n_results"] == 3
    assert [c.text for c in ev.retrieved] == res["documents"][0]
    assert [c.id for c in ev.retrieved] == res["ids"][0]
    assert all(c.score is not None for c in ev.retrieved)
    assert ev.delivered == ev.retrieved

    assert len(stored.llm_calls) == 1
    call = stored.llm_calls[0]
    assert (call.provider, call.model, call.answer) == ("openai", "gpt-4o-mini", ANSWER)
    assert (call.input_tokens, call.output_tokens) == (42, 7)
    assert "system: Answer from context." in call.prompt and QUESTION in call.prompt
    assert res["documents"][0][0] in call.prompt and res["documents"][0][2] not in call.prompt

    assert stored.in_prompt == ((0, 0), (0, 1))
    assert stored.outcome == Answered(ANSWER) and stored.test is None
    assert (stored.metrics.input_tokens, stored.metrics.output_tokens) == (42, 7)
    assert stored.metrics.cost_usd == pytest.approx((42 * 0.15 + 7 * 0.60) / 1_000_000)  # gpt-4o-mini rates


# --- interventions through the chromadb hook -----------------------------------


def test_top_k_override_reaches_chromadb_n_results(hooks: None, chroma: Any) -> None:
    """Defect: the override lands in the snapshot but not in the call, so the user still gets 4 rows."""
    with capture.active_run(QUESTION, Intervention(top_k=2)) as run:
        res = chroma.query(query_texts=[QUESTION], n_results=4)
    assert len(res["documents"][0]) == 2 and len(res["ids"][0]) == 2
    assert run.retrievals[0].params["n_results"] == 2
    assert len(run.retrievals[0].retrieved) == 2


def test_keep_first_changes_what_user_code_receives(hooks: None, chroma: Any) -> None:
    """Defect: the transform is recorded in `delivered` but the native result is returned
    unchanged, so the experiment measures nothing. Also: per-query lists rebuilt out of step
    (metadatas no longer aligned with documents)."""
    with capture.active_run(QUESTION, Intervention(transforms=(KeepFirst(1),))) as run:
        res = chroma.query(query_texts=[QUESTION], n_results=3)
    ev = run.retrievals[0]
    assert len(ev.retrieved) == 3 and len(ev.delivered) == 1
    assert res["documents"][0] == [ev.retrieved[0].text]
    assert res["ids"][0] == [ev.retrieved[0].id]
    assert res["metadatas"][0] == [{"src": ev.retrieved[0].id}]
    assert len(res["distances"][0]) == 1
    assert res["included"] == ["metadatas", "documents", "distances"]


def test_inject_evidence_prepends_a_native_row(hooks: None, chroma: Any) -> None:
    """Defect: the injected chunk appears in documents but not in ids / metadatas / distances,
    so user code that zips the lists drops or misaligns it."""
    with capture.active_run(QUESTION, Intervention(transforms=(InjectEvidence(GOLD),))) as run:
        res = chroma.query(query_texts=[QUESTION], n_results=2)
    assert res["documents"][0][0] == GOLD.text
    assert res["ids"][0][0] == "contexteval:evidence"
    assert res["distances"][0][0] == 0.0 and res["metadatas"][0][0] is None
    assert len(res["documents"][0]) == len(res["ids"][0]) == len(res["metadatas"][0]) == len(res["distances"][0]) == 3
    ev = run.retrievals[0]
    assert len(ev.retrieved) == 2 and ev.delivered[0].source_index is None
    assert res["documents"][0][1:] == [c.text for c in ev.retrieved]


def test_embedding_only_query_records_query_none(hooks: None, chroma: Any) -> None:
    """Defect: a vector-only query crashes the hook or records a bogus query string."""
    with capture.active_run(QUESTION) as run:
        chroma.query(query_embeddings=HashEmbedding()([QUESTION]), n_results=2)
    ev = run.retrievals[0]
    assert ev.query is None and len(ev.retrieved) == 2
    # The params snapshot must be storable and must not carry the raw vector.
    assert len(json.dumps(ev.params)) < 200 and ev.params["n_results"] == 2


# --- other providers ---------------------------------------------------------------


def test_openai_responses_api_recorded(hooks: None, openai_client: Any) -> None:
    """Defect: only chat.completions is hooked, so an app on the Responses API records no LLM call."""
    with capture.active_run("q") as run:
        r = openai_client.responses.create(model="gpt-4o-mini", instructions="Be terse.", input="What is the refund window?")
    assert r.output_text == ANSWER
    call = run.llm_calls[0]
    assert (call.provider, call.model, call.answer) == ("openai", "gpt-4o-mini", ANSWER)
    assert call.prompt == "instructions: Be terse.\nuser: What is the refund window?"
    assert (call.input_tokens, call.output_tokens) == (9, 2)


def test_anthropic_messages_recorded(hooks: None) -> None:
    """Defect: system prompt (string or blocks) missing from the recorded prompt, or usage read
    with the openai field names."""
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")  # anthropic >= 1.x ships its own httpx fork

    def handler(request: Any) -> Any:
        return httpx2.Response(200, json=_anthropic_response(json.loads(request.content)))

    client = anthropic.Anthropic(api_key="fake", http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    with capture.active_run("q") as run:
        r = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=16,
            system="Be terse.",
            messages=[{"role": "user", "content": [{"type": "text", "text": "What is the refund window?"}]}],
        )
    assert r.content[0].text == ANSWER
    call = run.llm_calls[0]
    assert (call.provider, call.model, call.answer) == ("anthropic", "claude-haiku-4-5", ANSWER)
    assert call.prompt == "system: Be terse.\nuser: What is the refund window?"
    assert (call.input_tokens, call.output_tokens) == (11, 3)


def test_async_openai_chat_and_responses_recorded(hooks: None, async_openai_client: Any) -> None:
    """v1.1 B. Defect: AsyncOpenAI is not hooked, so an async app records no LLM call and no
    production trace ever closes; or the wrapper hands the adapter the un-awaited coroutine, so
    prompt and usage are empty and user code gets a coroutine where it expects a response."""

    async def main() -> tuple[Any, Any]:
        chat = await async_openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": QUESTION}])
        resp = await async_openai_client.responses.create(model="gpt-4o-mini", instructions="Be terse.", input=QUESTION)
        return chat, resp

    with capture.active_run(QUESTION) as run:
        chat, resp = asyncio.run(main())
    assert chat.choices[0].message.content == ANSWER and resp.output_text == ANSWER
    first, second = run.llm_calls
    assert (first.provider, first.model, first.answer, first.prompt) == ("openai", "gpt-4o-mini", ANSWER, f"user: {QUESTION}")
    assert (first.input_tokens, first.output_tokens) == (42, 7)
    assert (second.provider, second.model, second.answer) == ("openai", "gpt-4o-mini", ANSWER)
    assert second.prompt == f"instructions: Be terse.\nuser: {QUESTION}"
    assert (second.input_tokens, second.output_tokens) == (9, 2)


def test_async_anthropic_messages_recorded(hooks: None) -> None:
    """v1.1 B. Defect: AsyncAnthropic is not hooked, or its class-level `create` (a sync function
    that returns a coroutine) is wrapped as sync and the adapter parses the coroutine."""
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")

    def handler(request: Any) -> Any:
        return httpx2.Response(200, json=_anthropic_response(json.loads(request.content)))

    client = anthropic.AsyncAnthropic(api_key="fake", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))

    async def main() -> Any:
        return await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=16,
            system="Be terse.",
            messages=[{"role": "user", "content": [{"type": "text", "text": "What is the refund window?"}]}],
        )

    with capture.active_run("q") as run:
        r = asyncio.run(main())
    assert r.content[0].text == ANSWER
    call = run.llm_calls[0]
    assert (call.provider, call.model, call.answer) == ("anthropic", "claude-haiku-4-5", ANSWER)
    assert call.prompt == "system: Be terse.\nuser: What is the refund window?"
    assert (call.input_tokens, call.output_tokens) == (11, 3)


def test_async_chroma_query_is_hooked_with_top_k_and_transform(hooks: None, async_chroma: tuple[Any, _FakeAsyncChromaClient]) -> None:
    """v1.1 B. Defect: only the sync Collection is hooked, so an app on chromadb's AsyncClient
    records no retrieval and a top_k experiment leaves n_results untouched. chromadb's async
    client is HTTP-only, so the real AsyncCollection runs over a stand-in for its HTTP half."""
    col, client = async_chroma
    with capture.active_run(QUESTION, Intervention(top_k=2, transforms=(KeepFirst(1),))) as run:
        res = asyncio.run(col.query(query_texts=[QUESTION], n_results=4))
    assert client.n_results == [2]
    assert res["documents"][0] == [DOCS["d1"]] and res["ids"][0] == ["d1"] and res["metadatas"][0] == [{"src": "d1"}]
    ev = run.retrievals[0]
    assert ev.source == "chromadb" and ev.query == QUESTION and ev.params["n_results"] == 2
    assert [c.text for c in ev.retrieved] == [DOCS["d1"], DOCS["d2"]] and len(ev.delivered) == 1


def test_suppressed_covers_async_sdk_calls(hooks: None, async_openai_client: Any) -> None:
    """v1.1 B. Defect: suppression is honored on the sync path only, so an SDK provider built on
    an async client would record its own grader calls as the app's."""

    async def ask() -> Any:
        return await async_openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    with capture.active_run(QUESTION) as run:
        with capture.suppressed():
            assert asyncio.run(ask()).choices[0].message.content == ANSWER
        assert run.llm_calls == []
        asyncio.run(ask())
        assert len(run.llm_calls) == 1


def test_pinecone_stand_in_top_k_question_fallback_and_native_rows(pinecone_index: FakeIndex) -> None:
    """Pinecone queries are vectors, so transforms fall back to the run question. Defects: the
    override not reaching top_k, rerank applied with a None query (no-op), or injected rows
    that are not the index's own match type."""
    iv = Intervention(top_k=2, transforms=(LexicalRerank(), InjectEvidence(GOLD)))
    with capture.active_run("refund policy", iv) as run:
        res = pinecone_index.query(vector=[0.1, 0.2], top_k=5, include_metadata=True)
    assert pinecone_index.calls == [2]
    ev = run.retrievals[0]
    assert ev.source == "pinecone" and ev.query is None and ev.params["top_k"] == 2
    assert [c.text for c in ev.retrieved] == ["dogs bark", "refund policy details"]
    assert [c.text for c in ev.delivered] == [GOLD.text, "refund policy details", "dogs bark"]
    assert isinstance(res, QueryResponse)
    assert [m.metadata["text"] for m in res.matches] == [GOLD.text, "refund policy details", "dogs bark"]
    assert isinstance(res.matches[0], Match) and res.matches[0].id == "contexteval:evidence"
    assert res.matches[1] is pinecone_index.matches[1] and res.matches[2] is pinecone_index.matches[0]


def test_no_intervention_returns_the_native_result_object(pinecone_index: FakeIndex) -> None:
    """Defect: a no-op intervention still re-encodes, handing user code a copy (identity, type,
    and any attribute the codec does not know about are lost)."""
    with capture.active_run("q") as run:
        res = pinecone_index.query(vector=[0.1], top_k=2)
    assert res is pinecone_index.last
    assert len(run.retrievals) == 1 and run.retrievals[0].delivered == run.retrievals[0].retrieved


# --- modes and suppression -------------------------------------------------------------


def test_suppressed_calls_are_not_recorded(hooks: None, chroma: Any, openai_client: Any) -> None:
    """Defect: the SDK's own grader and judge calls show up as the app's LLM calls."""
    with capture.active_run(QUESTION) as run:
        with capture.suppressed():
            _app(chroma, openai_client, QUESTION)
        assert run.retrievals == [] and run.llm_calls == []
        chroma.query(query_texts=[QUESTION], n_results=1)
        assert len(run.retrievals) == 1


def test_cli_mode_records_only_inside_runs(hooks: None, store: Store, chroma: Any, openai_client: Any) -> None:
    """Defect: `contexteval run` writes production traces for calls made at import time."""
    _app(chroma, openai_client, QUESTION)
    assert capture._ambient.get() is None
    assert store.latest_run("production") is None
    with capture.active_run(QUESTION) as run:
        _app(chroma, openai_client, QUESTION)
    assert len(run.retrievals) == 1 and len(run.llm_calls) == 1


def test_production_mode_one_llm_call_closes_one_stored_trace(production: capture.InstallReport, project: Path, chroma: Any, openai_client: Any) -> None:
    """Defect: production events buffered forever (no trace), or every hooked call stored as its
    own trace so retrievals and the LLM call never meet."""
    assert capture._process.mode is capture.Mode.PRODUCTION
    answer, _ = _app(chroma, openai_client, QUESTION)
    assert capture.flush_production()
    store = Store.open(project)
    try:
        run = store.latest_run("production")
        assert run is not None and run.app is None
        traces = store.traces(run.id)
        assert len(traces) == 1
        t = traces[0].trace
        assert t.question == QUESTION and t.outcome == Answered(answer) and t.test is None
        assert len(t.retrievals) == 1 and len(t.llm_calls) == 1
        assert t.in_prompt == ((0, 0), (0, 1))
        openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        assert capture.flush_production()
        traces = store.traces(run.id)
        assert len(traces) == 2 and traces[1].trace.retrievals == ()
        assert capture._ambient.get() is None
    finally:
        store.close()


def test_init_after_cli_install_keeps_cli_mode(hooks: None, project: Path, chroma: Any, openai_client: Any) -> None:
    """Defect: a user's own contexteval.init() inside my_app escalates `contexteval run` to
    production mode, so a test run also writes production traces."""
    import contexteval

    report = contexteval.init(project)
    assert report.statuses["chromadb"] == "already" and report.statuses["openai"] == "already"
    assert capture._process.mode is capture.Mode.CLI
    _app(chroma, openai_client, QUESTION)
    store = Store.open(project)
    try:
        assert store.latest_run("production") is None
    finally:
        store.close()


class _BodyIndex:
    """Pinecone stand-in whose chunk text lives under metadata["body"], not "text"."""

    def query(self, vector: list[float] | None = None, top_k: int = 3, include_metadata: bool = True) -> dict[str, Any]:
        return {"matches": [{"id": "1", "score": 0.9, "metadata": {"body": "the gold text"}}]}


def test_pinecone_text_key_from_a_later_init_applies_under_the_cli(hooks: None, project: Path) -> None:
    """Review X7. Defect: the CLI installs first with the default key, and the user's
    init(pinecone_text_key="body") inside the app module is discarded as "already", so every
    pinecone chunk is recorded with empty text. The codec must read the key at call time."""
    import contexteval

    assert _hooks.patch_method(_BodyIndex, "query", _hooks.PineconeCodec()) == "installed"
    contexteval.init(project, pinecone_text_key="body")
    assert capture._process.mode is capture.Mode.CLI
    with capture.active_run("q") as run:
        _BodyIndex().query(vector=[0.1])
    assert [c.text for c in run.retrievals[0].retrieved] == ["the gold text"]


def test_plain_init_keeps_a_text_key_the_cli_already_set(hooks: None, project: Path) -> None:
    """Review X7. Defect: a user's plain init() in the app module resets the key the CLI took
    from its configuration back to the default."""
    import contexteval

    _hooks.install("cli", pinecone_text_key="body")
    contexteval.init(project)
    assert _hooks.PineconeCodec().text_key == "body"


# --- init() never raises (review X1) -----------------------------------------------------


@pytest.fixture
def clean_hooks() -> Iterator[None]:
    yield
    _hooks.uninstall_all()
    capture.reset_process()


def _assert_init_failed_safely(report: capture.InstallReport, expected: str) -> None:
    assert report.statuses == {} and report.error is not None and expected in report.error
    assert capture._process.mode is capture.Mode.OFF

    @capture.retriever
    def search(query: str, top_k: int = 3) -> list[str]:
        return ["a"]

    assert search("q") == ["a"] and capture._ambient.get() is None


def test_init_reports_a_schema_mismatch_instead_of_raising(tmp_path: Path, clean_hooks: None) -> None:
    """Review X1. Defect: a database written by another build raised RuntimeError out of init()
    into the user's app at startup."""
    import contexteval

    (tmp_path / ".contexteval").mkdir()
    conn = sqlite3.connect(str(tmp_path / ".contexteval" / "contexteval.db"))
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()
    _assert_init_failed_safely(contexteval.init(tmp_path), f"schema version {SCHEMA_VERSION + 1}")


def test_init_reports_an_unwritable_project_dir_instead_of_raising(tmp_path: Path, clean_hooks: None) -> None:
    """Review X1. Defect: a project path that cannot be created raised OSError out of init()."""
    import contexteval

    (tmp_path / "afile").write_text("x", encoding="utf-8")
    _assert_init_failed_safely(contexteval.init(tmp_path / "afile" / "sub"), "Error")


def test_init_survives_a_library_whose_import_raises(tmp_path: Path, project: Path, monkeypatch: pytest.MonkeyPatch, clean_hooks: None, caplog: pytest.LogCaptureFixture) -> None:
    """Review X1. Defect: a non-ImportError side effect while importing a hooked library (a
    broken optional dependency) propagated out of init(); every other library must still be
    hooked and the broken one reported as failed."""
    import contexteval

    (tmp_path / "boomlib.py").write_text("raise RuntimeError('lib import side effect failed')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "boomlib", raising=False)
    monkeypatch.setattr(_hooks, "HOOKS", (*_hooks.HOOKS, _hooks.HookSpec("boomlib", ("boomlib:Index.query",), _hooks.ChromaCodec())))
    report = contexteval.init(project)
    assert report.error is None
    assert report.statuses["boomlib"] == "failed" and report.statuses["chromadb"] == "installed"
    assert capture._process.mode is capture.Mode.PRODUCTION
    assert caplog.text.count("lib import side effect failed") == 1


def test_patch_method_reports_a_method_without_a_readable_signature(clean_hooks: None) -> None:
    """Review X1. Defect: inspect.signature raising on an odd target escaped install() and init()."""

    class Odd:
        def query(self, q: str) -> list[str]:
            return [q]

    Odd.query.__signature__ = 42  # type: ignore[attr-defined]
    original = Odd.__dict__["query"]
    assert _hooks.patch_method(Odd, "query", _hooks.ChromaCodec()) == "failed"
    assert Odd.__dict__["query"] is original


# --- install mechanics ------------------------------------------------------------------


def test_install_twice_is_noop_and_uninstall_restores_originals() -> None:
    """Defect: a second install wraps the wrapper (double recording), or uninstall leaves the
    wrapper in place so the next test's hooks report 'already' and never record."""
    from chromadb.api.models.AsyncCollection import AsyncCollection
    from chromadb.api.models.Collection import Collection
    from openai.resources.chat.completions import AsyncCompletions, Completions
    from openai.resources.responses import AsyncResponses, Responses

    owners = {Collection: "query", Completions: "create", Responses: "create", AsyncCollection: "query", AsyncCompletions: "create", AsyncResponses: "create"}
    originals = {owner: owner.__dict__[name] for owner, name in owners.items()}
    try:
        first = _hooks.install("cli")
        assert first.statuses["chromadb"] == "installed" and first.statuses["openai"] == "installed"
        expected_pinecone = "installed" if importlib.util.find_spec("pinecone") else "not_installed"
        assert first.statuses["pinecone"] == expected_pinecone
        wrapped = Collection.__dict__["query"]
        assert wrapped is not originals[Collection]
        assert wrapped.__contexteval_original__ is originals[Collection]
        for owner in (AsyncCollection, AsyncCompletions, AsyncResponses):
            name = owners[owner]
            assert owner.__dict__[name].__contexteval_original__ is originals[owner]
            assert inspect.iscoroutinefunction(owner.__dict__[name]), f"{owner.__name__}.{name} wrapper must stay awaitable"
        second = _hooks.install("cli")
        assert second.statuses["chromadb"] == "already" and second.statuses["openai"] == "already"
        assert Collection.__dict__["query"] is wrapped
    finally:
        _hooks.uninstall_all()
        capture.reset_process()
    for owner, name in owners.items():
        assert owner.__dict__[name] is originals[owner]
    assert capture._process.mode is capture.Mode.OFF


def test_patch_method_reads_owner_dict_not_inherited(hooks: None) -> None:
    """Graft 6: a subclass without its own method is 'incompatible', not wrapped with the parent's."""

    class Base:
        def query(self) -> list[str]:
            return []

    class Child(Base):
        pass

    assert _hooks.patch_method(Child, "query", _hooks.ChromaCodec()) == "incompatible"
    assert "query" not in Child.__dict__


class _BrokenCodec:
    source = "broken"
    top_k_param = None

    def query_text(self, args: Any) -> str | None:
        return None

    def decode(self, result: Any) -> Any:
        raise RuntimeError("cannot read this result")

    def encode(self, result: Any, chunks: Any) -> Any:
        raise RuntimeError("cannot build this result")


class _Service:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, q: str) -> list[str]:
        self.calls += 1
        return ["a"]


def test_recording_error_is_swallowed_once_and_retrieval_runs_exactly_once(caplog: pytest.LogCaptureFixture) -> None:
    """Graft 3. Defects: a decode failure breaks the user's call, retries it (two retrievals),
    or logs on every call. With an intervention active the same failure must be loud, because
    silently returning the raw result would turn an experiment into a baseline."""
    assert _hooks.patch_method(_Service, "search", _BrokenCodec()) == "installed"
    svc = _Service()
    try:
        with capture.active_run("q") as run:
            assert svc.search("q") == ["a"]
            assert svc.search("q") == ["a"]
        assert svc.calls == 2 and run.retrievals == []
        assert caplog.text.count("could not read broken result") == 1
        with capture.active_run("q", Intervention(transforms=(KeepFirst(1),))):
            with pytest.raises(InterventionNotApplicable):
                svc.search("q")
        assert svc.calls == 3
    finally:
        _hooks.uninstall_all()
        capture.reset_process()


class _Boom:
    def search(self, q: str) -> list[str]:
        raise ValueError("store is down")


def test_user_exception_propagates_and_records_nothing() -> None:
    """Defect: the hook wraps or swallows the user's own retrieval error."""
    assert _hooks.patch_method(_Boom, "search", _BrokenCodec()) == "installed"
    try:
        with capture.active_run("q") as run:
            with pytest.raises(ValueError, match="store is down"):
                _Boom().search("q")
        assert run.retrievals == []
    finally:
        _hooks.uninstall_all()
        capture.reset_process()
