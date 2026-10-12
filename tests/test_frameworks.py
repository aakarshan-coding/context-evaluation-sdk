"""v2 framework plugins: LangChain and LlamaIndex retrievers through the hook table, against real
langchain-chroma over real chromadb and a real llama-index VectorStoreIndex, both with a
deterministic embedding. Each docstring names the defect the test catches."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping

import pytest

chromadb = pytest.importorskip("chromadb")
pytest.importorskip("langchain_core")
pytest.importorskip("langchain_chroma")
pytest.importorskip("llama_index.core")

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.callbacks import CallbackManagerForRetrieverRun  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.retrievers import BaseRetriever as LCBaseRetriever  # noqa: E402
from llama_index.core import VectorStoreIndex  # noqa: E402
from llama_index.core.base.base_retriever import BaseRetriever as LIBaseRetriever  # noqa: E402
from llama_index.core.embeddings import BaseEmbedding  # noqa: E402
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode  # noqa: E402

from contexteval import _capture as capture  # noqa: E402
from contexteval import _cli, _hooks, retriever  # noqa: E402
from contexteval._intervention import InjectEvidence, Intervention, InterventionNotApplicable, KeepFirst, Setting  # noqa: E402
from contexteval._store import Store  # noqa: E402
from contexteval.demo.app import HashEmbedding  # noqa: E402
from contexteval.types import Evidence, Label, Missing, label_of  # noqa: E402

DOCS = {
    "d1": "The refund window closes thirty days after delivery.",
    "d2": "Refunds are paid to the original payment method.",
    "d3": "Shipping takes three to five business days.",
    "d4": "Our office cat is named Pickles.",
    "d5": "Gift cards cannot be refunded or exchanged.",
}
QUESTION = "When does the refund window close?"
GOLD = Evidence("Refund requests need the order number.")
FRAMEWORK_SPECS = tuple(s for s in _hooks.HOOKS if s.library in ("langchain_core", "llama_index"))


# --- framework doubles over deterministic embeddings ---------------------------------------


class HashEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return HashEmbedding()(texts)

    def embed_query(self, text: str) -> list[float]:
        return HashEmbedding()([text])[0]


class HashEmbedModel(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> list[float]:
        return HashEmbedding()([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return HashEmbedding()([text])[0]


@pytest.fixture
def lc_retriever() -> Any:
    """A langchain-chroma VectorStoreRetriever (k=2) over a real chromadb collection."""
    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    store = Chroma(client=client, collection_name=f"lc_{uuid.uuid4().hex}", embedding_function=HashEmbeddings())
    store.add_texts(list(DOCS.values()), ids=list(DOCS), metadatas=[{"src": k} for k in DOCS])
    return store.as_retriever(search_kwargs={"k": 2})


@pytest.fixture
def li_retriever() -> Any:
    """A llama-index VectorIndexRetriever (similarity_top_k=2) over an in-memory index."""
    nodes = [TextNode(text=t, id_=k) for k, t in DOCS.items()]
    return VectorStoreIndex(nodes=nodes, embed_model=HashEmbedModel()).as_retriever(similarity_top_k=2)


@pytest.fixture
def chroma_col() -> Any:
    """A raw chromadb collection holding DOCS, so a hooked `Collection.query` sits under a retriever."""
    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    col = client.create_collection(f"docs_{uuid.uuid4().hex}", embedding_function=HashEmbedding())
    col.add(ids=list(DOCS), documents=list(DOCS.values()))
    return col


class FixedLC(LCBaseRetriever):
    """A LangChain retriever with no top_k knob of any kind."""

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [Document(page_content=DOCS["d1"], id="d1")]


class ChromaLC(LCBaseRetriever):
    """The common custom-retriever shape: a BaseRetriever subclass that queries chromadb itself. Its
    `n` is not a knob contexteval knows; the hooked `Collection.query` underneath is."""

    col: Any
    n: int = 2

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        rows = self.col.query(query_texts=[query], n_results=self.n)
        return [Document(page_content=t, id=i) for t, i in zip(rows["documents"][0], rows["ids"][0])]


class WrapperLC(LCBaseRetriever):
    """A wrapper in the shape of ContextualCompressionRetriever or MultiQueryRetriever: no knob of
    its own, delegating to a `base_retriever` whose `invoke` is hooked too."""

    base_retriever: LCBaseRetriever

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.base_retriever.invoke(query, config={"callbacks": run_manager.get_child()})[::-1]


class BoomLC(LCBaseRetriever):
    search_kwargs: dict[str, Any] = {"k": 1}

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        raise ValueError("index is down")


class FixedLI(LIBaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return [NodeWithScore(node=TextNode(text=DOCS["d1"], id_="d1"), score=0.5)]


class CappedLI(LIBaseRetriever):
    """A LlamaIndex retriever whose similarity_top_k setter rejects values over 10."""

    def __init__(self) -> None:
        super().__init__()
        self._top_k = 2

    @property
    def similarity_top_k(self) -> int:
        return self._top_k

    @similarity_top_k.setter
    def similarity_top_k(self, value: int) -> None:
        if value > 10:
            raise ValueError("similarity_top_k must be at most 10")
        self._top_k = value

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return [NodeWithScore(node=TextNode(text=t, id_=k), score=0.5) for k, t in list(DOCS.items())[: self._top_k]]


# --- capture ------------------------------------------------------------------------------


def test_langchain_retriever_is_captured_with_chunks_and_query(hooks: None, lc_retriever: Any) -> None:
    """Defect: a LangChain app records no retrieval (only the SDK call), so every failure reads as
    not_retrieved; or the event's query is the RunnableConfig instead of the input string."""
    with capture.active_run(QUESTION) as run:
        docs = lc_retriever.invoke(QUESTION)
    assert [type(d) for d in docs] == [Document, Document]
    (ev,) = run.retrievals
    assert ev.source == "langchain" and ev.query == QUESTION and ev.params == {"input": QUESTION}
    assert [c.text for c in ev.retrieved] == [d.page_content for d in docs]
    assert [c.id for c in ev.retrieved] == [d.id for d in docs] == ["d1", "d2"]
    assert [c.source_index for c in ev.retrieved] == [0, 1] and ev.delivered == ev.retrieved


def test_llamaindex_retriever_is_captured_with_chunks_query_and_scores(hooks: None, li_retriever: Any) -> None:
    """Defect: a LlamaIndex app records no retrieval; or a QueryBundle query is recorded as its
    repr instead of its query_str; or node scores are dropped."""
    with capture.active_run(QUESTION) as run:
        nodes = li_retriever.retrieve(QUESTION)
        li_retriever.retrieve(QueryBundle(query_str="second question"))
    first, second = run.retrievals
    assert first.source == "llamaindex" and first.query == QUESTION and first.params == {"str_or_query_bundle": QUESTION}
    assert [c.text for c in first.retrieved] == [n.node.get_content() for n in nodes]
    assert [c.id for c in first.retrieved] == ["d1", "d2"]
    assert [c.score for c in first.retrieved] == [pytest.approx(n.score) for n in nodes]
    assert second.query == "second question"


# --- top_k override reaches the framework's own knob ----------------------------------------


def test_top_k_override_reaches_search_kwargs_and_is_restored(hooks: None, lc_retriever: Any) -> None:
    """Defect: the override is recorded but never reaches `search_kwargs["k"]`, so the user still
    gets 2 documents; or the retriever is left with k=4 after the call (the user's own dict
    mutated), so the next baseline question silently runs with the experiment's k."""
    users_kwargs = lc_retriever.search_kwargs
    with capture.active_run(QUESTION, Intervention(top_k=4)) as run:
        docs = lc_retriever.invoke(QUESTION)
    assert len(docs) == 4
    (ev,) = run.retrievals
    assert ev.params == {"input": QUESTION, "k": 4} and len(ev.retrieved) == 4
    assert lc_retriever.search_kwargs is users_kwargs and users_kwargs == {"k": 2}
    with capture.active_run(QUESTION) as run:
        assert len(lc_retriever.invoke(QUESTION)) == 2


def test_top_k_override_reaches_similarity_top_k_and_is_restored(hooks: None, li_retriever: Any) -> None:
    """Defect: the override does not reach `similarity_top_k`, or leaves it at the experiment's
    value after the call."""
    with capture.active_run(QUESTION, Intervention(top_k=4)) as run:
        nodes = li_retriever.retrieve(QUESTION)
    assert len(nodes) == 4 and li_retriever.similarity_top_k == 2
    assert run.retrievals[0].params == {"str_or_query_bundle": QUESTION, "similarity_top_k": 4}
    assert len(asyncio.run(li_retriever.aretrieve(QUESTION))) == 2


def test_knob_is_restored_when_the_retriever_raises(hooks: None) -> None:
    """Defect: the override is restored only on the success path, so a retriever that raises
    under an experiment keeps the experiment's k for the rest of the process."""
    boom = BoomLC()
    with capture.active_run(QUESTION, Intervention(top_k=5)):
        with pytest.raises(ValueError, match="index is down"):
            boom.invoke(QUESTION)
    assert boom.search_kwargs == {"k": 1}
    assert capture._knobs == {}


def test_custom_langchain_retriever_over_hooked_chroma_takes_top_k_from_the_inner_call(hooks: None, chroma_col: Any) -> None:
    """Review P1. Defect: since `BaseRetriever.invoke` is hooked, a custom retriever with no knob
    of its own owns the event and refuses top_k, although the hooked chromadb call it makes
    underneath could apply it (and did under 0.2.0). top_k ownership passes to the outermost
    hooked layer that can apply it; the outermost still records the one event, with the knob
    that was actually set."""
    retriever = ChromaLC(col=chroma_col)
    with capture.active_run(QUESTION, Intervention(top_k=4)) as run:
        docs = retriever.invoke(QUESTION)
    assert len(docs) == 4 and run.not_applicable is None
    (ev,) = run.retrievals
    assert ev.source == "langchain" and len(ev.retrieved) == 4 and ev.params["n_results"] == 4
    with capture.active_run(QUESTION) as run:
        assert len(retriever.invoke(QUESTION)) == 2
    (ev,) = run.retrievals
    assert "n_results" not in ev.params


def test_wrapper_retriever_hands_top_k_to_its_base_retriever(hooks: None, lc_retriever: Any) -> None:
    """Review P1. Defect: a wrapper retriever (compression, multi-query) holding a `base_retriever`
    refuses top_k because the wrapper has no `search_kwargs`, while the base retriever's hooked
    `invoke` underneath has the knob. The base applies and restores it; the wrapper records one
    event with `k` as applied."""
    users_kwargs = lc_retriever.search_kwargs
    wrapper = WrapperLC(base_retriever=lc_retriever)
    with capture.active_run(QUESTION, Intervention(top_k=4)) as run:
        docs = wrapper.invoke(QUESTION)
    assert len(docs) == 4 and run.not_applicable is None
    (ev,) = run.retrievals
    assert ev.source == "langchain" and len(ev.retrieved) == 4 and ev.params["k"] == 4
    assert [c.text for c in ev.retrieved] == [d.page_content for d in docs]
    assert lc_retriever.search_kwargs is users_kwargs and users_kwargs == {"k": 2}
    assert capture._knobs == {}


def test_per_call_k_is_overridden_and_the_event_records_the_applied_value(hooks: None, lc_retriever: Any) -> None:
    """Review P2. Defect: `VectorStoreRetriever` merges call kwargs over `search_kwargs`, so an app
    calling `invoke(q, k=1)` under top_k=4 still retrieves 1 document while the event claims
    `k: 4`. The codec sees the bound call arguments and overrides the per-call k as well."""
    with capture.active_run(QUESTION, Intervention(top_k=4)) as run:
        docs = lc_retriever.invoke(QUESTION, k=1)
    assert len(docs) == 4 and run.not_applicable is None
    (ev,) = run.retrievals
    assert len(ev.retrieved) == 4 and ev.params["k"] == 4
    with capture.active_run(QUESTION) as run:
        assert len(lc_retriever.invoke(QUESTION, k=1)) == 1
    assert run.retrievals[0].params["kwargs"] == {"k": 1}


def test_decorated_function_without_a_knob_takes_top_k_from_the_hooked_store_inside(hooks: None, chroma_col: Any) -> None:
    """Review P1, the same handoff for `@retriever(top_k=None)`: the function has no parameter to
    set, but the hooked chromadb call inside has `n_results`. Defect: refused before the call."""

    @retriever(top_k=None)
    def search(query: str) -> list[str]:
        return list(chroma_col.query(query_texts=[query], n_results=2)["documents"][0])

    with capture.active_run(QUESTION, Intervention(top_k=3)) as run:
        hits = search(QUESTION)
    assert len(hits) == 3
    assert [(e.source, len(e.retrieved), e.params["n_results"]) for e in run.retrievals] == [("retriever:search", 3, 3)]


class _UnreadableConfig(Mapping[str, Any]):
    """A config whose items() raises, so the params snapshot cannot read it."""

    def __getitem__(self, key: str) -> Any:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> Any:
        raise RuntimeError("no items for you")


def test_snapshot_failure_leaves_no_override_behind(hooks: None, lc_retriever: Any, caplog: pytest.LogCaptureFixture) -> None:
    """Review P5. Defect: the params snapshot ran after the override and outside the recording
    error guard, so an argument it could not read raised out of the hook before the call, and
    the retriever kept the experiment's k for the rest of the process. The snapshot falls back
    to {} and the call runs; here langchain itself then rejects the config, and the knob is
    restored on that path too."""
    users_kwargs = lc_retriever.search_kwargs
    with capture.active_run(QUESTION, Intervention(top_k=4)):
        with pytest.raises(RuntimeError, match="no items for you"):
            lc_retriever.invoke(QUESTION, config=_UnreadableConfig())
    assert lc_retriever.search_kwargs is users_kwargs and users_kwargs == {"k": 2}
    assert capture._knobs == {}
    assert caplog.text.count("could not snapshot") == 1
    with capture.active_run(QUESTION) as run:
        assert len(lc_retriever.invoke(QUESTION)) == 2
    assert run.retrievals[0].params == {"input": QUESTION}


def test_failed_knob_setattr_leaves_no_entry_and_a_later_restore_restores(hooks: None) -> None:
    """Review P4. Defect: the knob entry was recorded before `setattr` ran, so a setter that
    rejects the experiment's value leaked a depth-1 entry; the next override on that retriever
    saw depth 2 and its restore only decremented, leaving the user's retriever at the
    experiment's value for the rest of the process."""
    retriever = CappedLI()
    with capture.active_run(QUESTION, Intervention(top_k=20)) as run:
        with pytest.raises(InterventionNotApplicable, match="llamaindex: cannot set top_k=20"):
            retriever.retrieve(QUESTION)
    assert run.not_applicable is not None and retriever.similarity_top_k == 2
    assert capture._knobs == {}
    with capture.active_run(QUESTION, Intervention(top_k=5)) as run:
        assert len(retriever.retrieve(QUESTION)) == 5
    assert run.retrievals[0].params["similarity_top_k"] == 5
    assert retriever.similarity_top_k == 2 and capture._knobs == {}


def test_retriever_without_a_top_k_knob_refuses_a_top_k_intervention(hooks: None) -> None:
    """Graft 3. Defect: a top_k experiment on a framework retriever with no knob silently runs the
    baseline and the experiment table shows a fake no-change row. The refusal comes once the
    call has finished with no layer, inner or outer, having applied the value."""
    with capture.active_run(QUESTION, Intervention(top_k=5)) as run:
        with pytest.raises(InterventionNotApplicable, match="langchain: top_k=5 .* no search_kwargs or k"):
            FixedLC().invoke(QUESTION)
        with pytest.raises(InterventionNotApplicable, match="llamaindex: top_k=5 .* no similarity_top_k"):
            FixedLI().retrieve(QUESTION)
    assert run.not_applicable is not None
    with capture.active_run(QUESTION) as run:
        assert FixedLC().invoke(QUESTION)[0].page_content == DOCS["d1"]
        assert FixedLI().retrieve(QUESTION)[0].node.get_content() == DOCS["d1"]
    assert [e.source for e in run.retrievals] == ["langchain", "llamaindex"]


def test_concurrent_overrides_on_one_retriever_restore_the_users_value(hooks: None, lc_retriever: Any) -> None:
    """Defect: two concurrent retrievals on one retriever each save the value they see, so the
    second saves the first's override and restores it last; the user's retriever ends up with
    the experiment's k for good. Under one run both calls want the same value, and the last
    one out must restore the value the user set."""
    users_kwargs = lc_retriever.search_kwargs

    async def main() -> list[list[Document]]:
        return await asyncio.gather(lc_retriever.ainvoke(QUESTION), lc_retriever.ainvoke("How are refunds paid?"))

    with capture.active_run(QUESTION, Intervention(top_k=3)) as run:
        results = asyncio.run(main())
    assert [len(docs) for docs in results] == [3, 3]
    assert len(run.retrievals) == 2 and all(len(e.retrieved) == 3 for e in run.retrievals)
    assert lc_retriever.search_kwargs is users_kwargs and users_kwargs == {"k": 2}
    assert capture._knobs == {}


# --- transforms return the framework's own types --------------------------------------------


def test_keep_first_and_inject_return_langchain_documents(hooks: None, lc_retriever: Any) -> None:
    """Defect: the intervention is recorded in `delivered` but the app still gets the untouched
    list; or the injected chunk comes back as a Chunk or dict and `doc.page_content` raises."""
    with capture.active_run(QUESTION, Intervention(transforms=(KeepFirst(1), InjectEvidence(GOLD)))) as run:
        docs = lc_retriever.invoke(QUESTION)
    (ev,) = run.retrievals
    assert len(ev.retrieved) == 2 and [c.text for c in ev.delivered] == [GOLD.text, DOCS["d1"]]
    assert [type(d) for d in docs] == [Document, Document]
    assert (docs[0].page_content, docs[0].id, docs[0].metadata) == (GOLD.text, "contexteval:evidence", {})
    assert (docs[1].page_content, docs[1].id, docs[1].metadata) == (DOCS["d1"], "d1", {"src": "d1"})


def test_keep_first_and_inject_return_llamaindex_nodes_with_score(hooks: None, li_retriever: Any) -> None:
    """Defect: the injected chunk is not a NodeWithScore over a TextNode, so `n.node.get_content()`
    or `n.score` fails in the user's prompt builder; or the kept node is rebuilt instead of the
    retriever's own object."""
    with capture.active_run(QUESTION, Intervention(transforms=(KeepFirst(1), InjectEvidence(GOLD)))) as run:
        nodes = li_retriever.retrieve(QUESTION)
    (ev,) = run.retrievals
    assert len(ev.retrieved) == 2 and [c.text for c in ev.delivered] == [GOLD.text, DOCS["d1"]]
    assert [type(n) for n in nodes] == [NodeWithScore, NodeWithScore] and isinstance(nodes[0].node, TextNode)
    assert (nodes[0].node.get_content(), nodes[0].node.node_id, nodes[0].score) == (GOLD.text, "contexteval:evidence", 1.0)
    assert nodes[1].node.node_id == "d1" and nodes[1].node.get_content() == DOCS["d1"]


# --- no double counting -------------------------------------------------------------------


def _assert_one_event_once_applied(run: capture.ActiveRun, docs: list[Document]) -> None:
    (ev,) = run.retrievals
    assert ev.source == "langchain"
    assert len(ev.retrieved) == 2, "the inner chromadb call must run with the overridden k and record nothing itself"
    assert [c.text for c in ev.delivered] == [GOLD.text, DOCS["d1"]]
    assert [d.page_content for d in docs] == [GOLD.text, DOCS["d1"]]


def test_langchain_over_chroma_records_one_event_and_applies_the_intervention_once(hooks: None, lc_retriever: Any) -> None:
    """Defect: langchain-chroma calls the hooked `Collection.query`, so one user retrieval records
    two events (langchain and chromadb) and the intervention runs twice: the inner KeepFirst(1)
    leaves one chunk for the outer one, and InjectEvidence puts the gold in twice. The
    outermost hooked retrieval owns the event; inner hooked calls pass through."""
    iv = Intervention(top_k=2, transforms=(KeepFirst(1), InjectEvidence(GOLD)))
    with capture.active_run(QUESTION, iv) as run:
        docs = lc_retriever.invoke(QUESTION)
    _assert_one_event_once_applied(run, docs)


def test_langchain_over_chroma_records_one_event_async(hooks: None, lc_retriever: Any) -> None:
    """Same defect for `ainvoke`: langchain runs the sync chroma query on an executor thread, so
    the pass-through must travel with the task's context, not sit on the calling thread."""
    iv = Intervention(top_k=2, transforms=(KeepFirst(1), InjectEvidence(GOLD)))
    with capture.active_run(QUESTION, iv) as run:
        docs = asyncio.run(lc_retriever.ainvoke(QUESTION))
    _assert_one_event_once_applied(run, docs)


def test_decorated_search_over_a_hooked_store_records_only_the_outer_event(hooks: None) -> None:
    """The same rule for the decorator: a user function marked @retriever that queries chromadb
    inside used to record two events for one search. Outside the function the store records
    itself as before."""
    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    col = client.create_collection(f"docs_{uuid.uuid4().hex}", embedding_function=HashEmbedding())
    col.add(ids=list(DOCS), documents=list(DOCS.values()))

    @retriever
    def search(query: str, top_k: int = 2) -> list[str]:
        return list(col.query(query_texts=[query], n_results=top_k)["documents"][0])

    with capture.active_run(QUESTION, Intervention(top_k=3)) as run:
        hits = search(QUESTION)
        col.query(query_texts=[QUESTION], n_results=1)
    assert len(hits) == 3
    assert [(e.source, len(e.retrieved)) for e in run.retrievals] == [("retriever:search", 3), ("chromadb", 3)]


# --- install ------------------------------------------------------------------------------


def test_framework_hooks_install_once_and_uninstall_restores() -> None:
    """Defect: a second install wraps the wrapper, or uninstall leaves the retriever classes
    patched for the next test; the async wrappers must still be coroutine functions."""
    owners = {LCBaseRetriever: ("invoke", "ainvoke"), LIBaseRetriever: ("retrieve", "aretrieve")}
    originals = {(o, n): o.__dict__[n] for o, names in owners.items() for n in names}
    try:
        first = _hooks.install("cli")
        assert first.statuses["langchain_core"] == "installed" and first.statuses["llama_index"] == "installed"
        for (owner, name), original in originals.items():
            wrapper = owner.__dict__[name]
            assert wrapper is not original and wrapper.__contexteval_original__ is original
            assert inspect.iscoroutinefunction(wrapper) == name.startswith("a")
        second = _hooks.install("cli")
        assert second.statuses["langchain_core"] == "already" and second.statuses["llama_index"] == "already"
    finally:
        _hooks.uninstall_all()
        capture.reset_process()
    for (owner, name), original in originals.items():
        assert owner.__dict__[name] is original


_INIT_PROBE = """
import json, sys
import contexteval
from contexteval import _hooks
mode, project, preload = sys.argv[1], sys.argv[2], sys.argv[3]
if preload:
    __import__(preload)
report = _hooks.install("cli") if mode == "cli" else contexteval.init(project)
frameworks = sorted(m for m in ("langchain_core", "llama_index") if m in sys.modules)
print(json.dumps({"statuses": report.statuses, "error": report.error, "imported": frameworks}))
"""


def _install_in_a_fresh_process(mode: str, project: Path, preload: str = "") -> dict[str, Any]:
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
    run = subprocess.run([sys.executable, "-c", _INIT_PROBE, mode, str(project), preload], capture_output=True, text=True, timeout=120, cwd=project, env=env)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


def test_production_init_hooks_a_framework_only_when_the_app_imported_it(tmp_path: Path) -> None:
    """Review P6. Defect: init() imported LangChain and LlamaIndex on every call (0.7 s and 1.2 s
    here, 7.7 s and 2.8 s on the reviewer's machine), paid by every app that uses neither. In
    production a framework is hooked only when the app has imported it by init() time; the
    CLI installs before the app module loads, so it still hooks everything it can import."""
    cold = _install_in_a_fresh_process("production", tmp_path)
    assert cold["error"] is None and cold["statuses"]["chromadb"] == "installed"
    assert (cold["statuses"]["langchain_core"], cold["statuses"]["llama_index"]) == ("not_imported", "not_imported")
    assert cold["imported"] == []
    warm = _install_in_a_fresh_process("production", tmp_path, preload="langchain_core.retrievers")
    assert (warm["statuses"]["langchain_core"], warm["statuses"]["llama_index"]) == ("installed", "not_imported")
    cli = _install_in_a_fresh_process("cli", tmp_path)
    assert (cli["statuses"]["langchain_core"], cli["statuses"]["llama_index"]) == ("installed", "installed")
    assert cli["imported"] == ["langchain_core", "llama_index"]


def test_not_installed_when_a_framework_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defect: a missing optional framework raises ImportError out of install() or is reported as
    incompatible or failed; the other libraries must still install."""
    for name in ("langchain_core", "langchain_core.retrievers", "llama_index", "llama_index.core", "llama_index.core.base.base_retriever"):
        monkeypatch.setitem(sys.modules, name, None)
    try:
        report = _hooks.install("cli")
    finally:
        _hooks.uninstall_all()
        capture.reset_process()
    assert report.statuses["langchain_core"] == "not_installed" and report.statuses["llama_index"] == "not_installed"
    assert report.statuses["chromadb"] == "installed" and report.error is None


# --- end to end through the CLI --------------------------------------------------------------


def test_langchain_rag_app_is_run_and_diagnosed_through_the_cli(project: Path, hooks: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """DoD for the plugin: `contexteval run my_app:answer` on an unchanged LangChain app (chroma
    retriever plus a chat model over the openai SDK) captures, diagnoses, and experiments like a
    raw chromadb app. Defects: the planted missing case labeled anything but missing /
    not_retrieved / confirmed (the inject replay must hand the app a Document), or the top_k=5
    experiment not retrieving the gold (the override must reach search_kwargs through the CLI)."""
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    import lc_rag_app

    tests = project / "cases.jsonl"
    tests.write_text("".join(json.dumps(c) + "\n" for c in lc_rag_app.CASES), encoding="utf-8")
    assert _cli.main(["init", "--provider", "fake"]) == 0
    assert _cli.main(["build-tests", "--import", str(tests)]) == 0
    assert _cli.main(["run", "lc_rag_app:answer", "--yes"]) == 0
    assert "2 cases, 1 passed, 1 failed, 0 crashed" in capsys.readouterr().out

    assert _cli.main(["diagnose", "--yes"]) == 0
    out = capsys.readouterr().out
    assert f"missing      {lc_rag_app.MISSING_QUESTION}" in out
    assert "the gold chunk was never retrieved; injecting the evidence fixed it (2/2)" in out

    assert _cli.main(["experiment", "top_k=5", "--yes"]) == 0
    capsys.readouterr()

    store = Store.open(project)
    try:
        baseline = store.latest_run("baseline")
        assert baseline is not None
        traces = {s.trace.question: s for s in store.traces(baseline.id)}
        missing = traces[lc_rag_app.MISSING_QUESTION]
        assert isinstance(missing.diagnosis, Missing) and label_of(missing.diagnosis) is Label.MISSING
        assert (missing.diagnosis.cause, missing.diagnosis.confirmed) == ("not_retrieved", True)
        (ev,) = missing.trace.retrievals
        assert ev.source == "langchain" and len(ev.retrieved) == lc_rag_app.TOP_K
        assert missing.trace.llm_calls[0].provider == "openai" and missing.trace.outcome.text == lc_rag_app.DONT_KNOW  # type: ignore[union-attr]
        (experiment,) = [r for r in store.runs() if r.run.setting == Setting(top_k=5)]
        assert (experiment.passed, experiment.failed) == (2, 0)
        for s in store.traces(experiment.run.id):
            assert s.trace.retrievals[0].params["k"] == 5 and len(s.trace.retrievals[0].retrieved) == 5
    finally:
        store.close()
    assert lc_rag_app.retriever.search_kwargs == {"k": lc_rag_app.TOP_K}
