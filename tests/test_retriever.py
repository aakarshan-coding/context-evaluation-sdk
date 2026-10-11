"""@contexteval.retriever: decoration-time validation, top_k override, and the three item shapes."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from contexteval import InterventionNotApplicable, retriever
from contexteval import _capture as capture
from contexteval._intervention import Compress, InjectEvidence, Intervention, KeepFirst
from contexteval.types import Chunk, Evidence

GOLD = Evidence("Refund requests need the order number.")


def test_custom_top_k_name_is_overridden_and_call_recorded() -> None:
    """Defect: the override targets a parameter literally named top_k, so `limit` keeps its default."""
    seen: list[int] = []

    @retriever(top_k="limit")
    def search(query: str, limit: int = 5) -> list[str]:
        seen.append(limit)
        return [f"doc{i}" for i in range(limit)]

    with capture.active_run("q", Intervention(top_k=2)) as run:
        out = search("refund")
    assert seen == [2] and out == ["doc0", "doc1"]
    ev = run.retrievals[0]
    assert ev.source == "retriever:search" and ev.query == "refund"
    assert ev.params == {"query": "refund", "limit": 2}
    assert [c.text for c in ev.retrieved] == ["doc0", "doc1"] and ev.delivered == ev.retrieved


def test_absent_default_top_k_is_none_and_a_top_k_intervention_raises() -> None:
    """Graft 3. Defect: a top_k experiment on a site without top_k silently runs the baseline."""

    @retriever
    def search(query: str) -> list[str]:
        return ["a"]

    @retriever(top_k=None)
    def search_fixed(query: str, top_k: int = 3) -> list[str]:
        return ["a"] * top_k

    for fn in (search, search_fixed):
        with capture.active_run("q", Intervention(top_k=2)):
            with pytest.raises(InterventionNotApplicable, match="no top_k parameter"):
                fn("q")
    with capture.active_run("q") as run:
        assert search("q") == ["a"] and search_fixed("q") == ["a", "a", "a"]
    assert len(run.retrievals) == 2


def test_unknown_explicit_names_raise_at_decoration() -> None:
    """Boundary: a misspelled top_k or query name fails at import, not on the first experiment."""
    with pytest.raises(TypeError, match="no parameter 'k'"):

        @retriever(top_k="k")
        def search(query: str) -> list[str]:
            return []

    with pytest.raises(TypeError, match="no parameter 'question'"):

        @retriever(query="question")
        def search2(q: str) -> list[str]:
            return []


def test_explicit_query_name_is_used_for_the_event() -> None:
    """Defect: the query always read from the first parameter even when told otherwise."""

    @retriever(query="question")
    def search(client: object, question: str, top_k: int = 2) -> list[str]:
        return ["x"]

    with capture.active_run("q") as run:
        search(object(), "where is my order")
    assert run.retrievals[0].query == "where is my order"
    assert run.retrievals[0].params["client"] == "<object>"


def test_str_items_inject_as_plain_text() -> None:
    """Defect: injected evidence wrapped in a dict for a function that returns strings."""

    @retriever
    def search(query: str, top_k: int = 3) -> list[str]:
        return ["alpha", "beta"]

    with capture.active_run("q", Intervention(transforms=(InjectEvidence(GOLD),))) as run:
        out = search("q")
    assert out == [GOLD.text, "alpha", "beta"]
    assert run.retrievals[0].delivered[0].source_index is None


def test_mapping_items_read_named_fields_and_inject_a_mapping() -> None:
    """Defect: text / id / score read from fixed keys instead of the configured ones, or the
    injected item built as a str inside a list of dicts."""

    @retriever(text="body", id="doc_id", score="sim")
    def search(query: str, top_k: int = 3) -> list[dict]:
        return [{"doc_id": 7, "body": "alpha. beta.", "sim": 0.9}, {"doc_id": 8, "body": "gamma", "sim": 0.5}]

    with capture.active_run("q", Intervention(transforms=(InjectEvidence(GOLD),))) as run:
        out = search("q")
    assert out[0] == {"body": GOLD.text}
    assert out[1:] == [{"doc_id": 7, "body": "alpha. beta.", "sim": 0.9}, {"doc_id": 8, "body": "gamma", "sim": 0.5}]
    assert run.retrievals[0].retrieved[0] == Chunk("alpha. beta.", "7", 0.9, 0)


def test_compress_rewrites_mapping_text_and_keeps_other_keys() -> None:
    """Defect: a changed chunk text is recorded but the item handed to user code keeps the full text."""

    @retriever(text="body")
    def search(query: str, top_k: int = 3) -> list[dict]:
        return [{"body": "Cats sleep. Refund window closes.", "src": "policy.md"}]

    with capture.active_run("refund", Intervention(transforms=(Compress(1),))):
        out = search("refund")
    assert out == [{"body": "Refund window closes.", "src": "policy.md"}]


@dataclass
class Doc:
    text: str
    id: str = "d"


def test_object_items_need_rebuild_for_injection() -> None:
    """Graft 4. Defect: the hook guesses an object constructor (or copies item 0) and hands user
    code an item of the wrong shape; with rebuild= the user decides."""

    @retriever
    def search(query: str, top_k: int = 3) -> list[Doc]:
        return [Doc("alpha")]

    with capture.active_run("q", Intervention(transforms=(InjectEvidence(GOLD),))):
        with pytest.raises(InterventionNotApplicable, match="rebuild="):
            search("q")

    @retriever(rebuild=lambda t: Doc(t, id="injected"))
    def search_rebuild(query: str, top_k: int = 3) -> list[Doc]:
        return [Doc("alpha")]

    with capture.active_run("q", Intervention(transforms=(InjectEvidence(GOLD),))) as run:
        out = search_rebuild("q")
    assert out == [Doc(GOLD.text, "injected"), Doc("alpha")]
    assert [c.id for c in run.retrievals[0].retrieved] == ["d"]


def test_callable_text_accessor() -> None:
    """Graft 4. Defect: text= accepts only a field name, so nested or computed text needs a wrapper."""

    @retriever(text=lambda d: d["payload"]["content"], id=None, score=None, rebuild=lambda t: {"payload": {"content": t}})
    def search(query: str, top_k: int = 3) -> list[dict]:
        return [{"payload": {"content": "nested text"}}]

    with capture.active_run("q", Intervention(transforms=(InjectEvidence(GOLD),))) as run:
        out = search("q")
    assert run.retrievals[0].retrieved == (Chunk("nested text", None, None, 0),)
    assert out == [{"payload": {"content": GOLD.text}}, {"payload": {"content": "nested text"}}]


def test_no_intervention_returns_the_functions_own_list_and_tuples_keep_type() -> None:
    """Defect: the decorator always rebuilds the list (identity lost), or turns a tuple into a list."""
    own: list[list[str]] = []

    @retriever
    def search(query: str, top_k: int = 3) -> list[str]:
        own.append(["a", "b"])
        return own[-1]

    @retriever
    def search_tuple(query: str, top_k: int = 3) -> tuple[str, ...]:
        return ("a", "b", "c")

    with capture.active_run("q"):
        assert search("q") is own[-1]
    with capture.active_run("q", Intervention(transforms=(KeepFirst(2),))):
        assert search_tuple("q") == ("a", "b")


def test_generator_result_reaches_the_user_unconsumed(caplog: pytest.LogCaptureFixture) -> None:
    """Review X4. Defect: decode iterates a generator result, so user code receives an exhausted
    iterator and an empty context, with no intervention and in production too. A one-shot result
    is refused (logged once, passed through untouched); under an intervention it is loud."""

    @retriever
    def search(query: str, top_k: int = 3):  # type: ignore[no-untyped-def]
        for i in range(top_k):
            yield f"doc{i}"

    with capture.active_run("q") as run:
        assert list(search("q")) == ["doc0", "doc1", "doc2"]
        assert list(search("q")) == ["doc0", "doc1", "doc2"]
    assert run.retrievals == []
    assert caplog.text.count("could not read retriever:search result") == 1
    with capture.active_run("q", Intervention(transforms=(KeepFirst(1),))):
        with pytest.raises(InterventionNotApplicable):
            search("q")


def test_str_and_mapping_results_are_not_decoded_as_chunks() -> None:
    """Review X4. Defect: a str result is recorded as one chunk per character and a dict result
    as one chunk per key. Neither is a sequence of items, so neither is recorded."""

    @retriever
    def as_text(query: str, top_k: int = 3) -> str:
        return "one answer"

    @retriever
    def as_mapping(query: str, top_k: int = 3) -> dict:
        return {"text": "one answer", "ids": [1, 2]}

    with capture.active_run("q") as run:
        assert as_text("q") == "one answer"
        assert as_mapping("q") == {"text": "one answer", "ids": [1, 2]}
    assert run.retrievals == []


def test_method_retriever_skips_self_for_the_query() -> None:
    """Review X9. Defect: `query` defaults to the first parameter, which is `self` on a method,
    so the event's query is "<Searcher>" and transforms that need the query see None."""

    class Searcher:
        @retriever
        def search(self, question: str, top_k: int = 2) -> list[str]:
            return ["a", "b"]

        @classmethod
        @retriever
        def search_cls(cls, question: str, top_k: int = 2) -> list[str]:
            return ["c"]

    with capture.active_run("q") as run:
        Searcher().search("where is my order")
        Searcher.search_cls("second question")
    assert [e.query for e in run.retrievals] == ["where is my order", "second question"]
    assert "self" not in run.retrievals[0].params and "cls" not in run.retrievals[1].params


def test_outside_any_run_in_off_mode_is_passthrough() -> None:
    """Defect: the decorator records into a buffer nobody flushes when the CLI is not running."""

    @retriever
    def search(query: str, top_k: int = 3) -> list[str]:
        return ["a"]

    assert capture._process.mode is capture.Mode.OFF
    assert search("q") == ["a"]
    assert capture._ambient.get() is None
    assert search.__wrapped__("q") == ["a"]  # type: ignore[attr-defined]
