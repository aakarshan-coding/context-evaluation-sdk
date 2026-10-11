"""Transforms are pure over Chunks, and Setting compiles to a fixed Intervention pipeline."""

from __future__ import annotations

from dataclasses import fields

import pytest

from contexteval._intervention import (
    BASELINE,
    EVIDENCE_CHUNK_ID,
    Compress,
    GoldToTop,
    InjectEvidence,
    Intervention,
    KeepFirst,
    LexicalRerank,
    OnlyGold,
    Order,
    Reorder,
    Setting,
    apply,
    apply_transform,
)
from contexteval.types import Chunk, Evidence

GOLD = Evidence("the refund window closes thirty days after delivery")


def _chunks(*texts: str) -> tuple[Chunk, ...]:
    return tuple(Chunk(text=t, id=f"c{i}", score=float(-i), source_index=i) for i, t in enumerate(texts))


def _texts(chunks: tuple[Chunk, ...]) -> list[str]:
    return [c.text for c in chunks]


def test_keep_first_truncates() -> None:
    """Defect: max_chunks keeps the last n or off by one."""
    assert _texts(apply_transform(KeepFirst(2), _chunks("a", "b", "c"), "q")) == ["a", "b"]


def test_reorder_reverse_and_ends() -> None:
    """Defect: `ends` puts the best chunk in the middle, defeating the lost-in-the-middle experiment."""
    chunks = _chunks("r0", "r1", "r2", "r3", "r4")
    assert _texts(apply_transform(Reorder(Order.REVERSE), chunks, "q")) == ["r4", "r3", "r2", "r1", "r0"]
    assert _texts(apply_transform(Reorder(Order.ENDS), chunks, "q")) == ["r0", "r2", "r4", "r3", "r1"]
    assert _texts(apply_transform(Reorder(Order.AS_IS), chunks, "q")) == ["r0", "r1", "r2", "r3", "r4"]


def test_lexical_rerank_sorts_by_query_overlap_and_is_stable() -> None:
    """Defect: ascending sort, or a query-less call that crashes instead of passing chunks through."""
    chunks = _chunks("cats purr", "refund window closes", "dogs", "the refund")
    assert _texts(apply_transform(LexicalRerank(), chunks, "refund window")) == [
        "refund window closes",
        "the refund",
        "cats purr",
        "dogs",
    ]
    assert apply_transform(LexicalRerank(), chunks, None) == chunks


def test_compress_keeps_source_index_and_top_sentences() -> None:
    """Defect: compress rebuilds chunks without source_index, so the codec cannot map them back to
    native rows and treats them as injected."""
    chunks = _chunks("Cats sleep. The refund window closes. Dogs bark.")
    out = apply_transform(Compress(1), chunks, "refund")
    assert out[0].text == "The refund window closes."
    assert out[0].source_index == 0
    assert apply_transform(Compress(1), chunks, None) == chunks


def test_inject_evidence_prepends_synthetic_chunk() -> None:
    """Defect: evidence appended instead of first, or given a source_index that aliases a real row."""
    out = apply_transform(InjectEvidence(GOLD), _chunks("a", "b"), "q")
    assert _texts(out) == [GOLD.text, "a", "b"]
    assert out[0].source_index is None
    assert out[0].id == EVIDENCE_CHUNK_ID


def test_gold_to_top_and_only_gold_partition_by_evidence() -> None:
    """Plan item 5. Defect: non-gold chunks reordered among themselves, or gold matched by exact
    text only so a chunk wrapping the evidence in extra words is treated as noise."""
    chunks = _chunks("noise one", f"Policy: {GOLD.text}.", "noise two", "noise three")
    assert _texts(apply_transform(GoldToTop(GOLD), chunks, "q")) == [
        f"Policy: {GOLD.text}.",
        "noise one",
        "noise two",
        "noise three",
    ]
    assert _texts(apply_transform(OnlyGold(GOLD), chunks, "q")) == [f"Policy: {GOLD.text}."]


def test_apply_folds_transforms_left_to_right() -> None:
    """Defect: transforms applied in reverse so KeepFirst runs before the rerank it should follow."""
    iv = Intervention(transforms=(LexicalRerank(), KeepFirst(1)))
    out = apply(iv, _chunks("dogs", "refund window"), "refund")
    assert _texts(out) == ["refund window"]
    assert Intervention().is_noop and not iv.is_noop and not Intervention(top_k=3).is_noop


def test_setting_compiles_in_fixed_order() -> None:
    """Defect: a Setting that compiles compress before max_chunks (or drops a knob) changes what
    an experiment row measures without changing its name."""
    s = Setting(top_k=10, rerank="lexical", order=Order.ENDS, max_chunks=3, compress=2)
    iv = s.intervention()
    assert iv.top_k == 10
    assert iv.transforms == (LexicalRerank(), Reorder(Order.ENDS), KeepFirst(3), Compress(2))
    assert BASELINE.intervention().is_noop


def test_setting_name_lists_non_default_knobs_only() -> None:
    """Defect: a baseline row named with every default knob, or a knob missing from the name."""
    assert BASELINE.name == "baseline"
    assert Setting(top_k=10, rerank="lexical").name == "top_k=10 rerank=lexical"
    assert Setting(order=Order.ENDS, compress=1).name == "order=ends compress=1"


def test_setting_json_round_trip_and_boundary_rejections() -> None:
    """Defect: order stored as an Enum repr the loader cannot read back, or a stored row with a
    knob this version does not know silently loading as baseline."""
    s = Setting(top_k=4, order=Order.REVERSE, max_chunks=2)
    assert Setting.from_json(s.to_json()) == s
    with pytest.raises(ValueError):
        Setting.from_json({"gold": "leak"})
    with pytest.raises(ValueError):
        Setting(top_k=0)
    with pytest.raises(ValueError):
        Setting(rerank="cross-encoder")  # type: ignore[arg-type]


def test_setting_has_no_field_that_can_name_evidence() -> None:
    """Plan item 15: experiments never see gold. Defect: a Setting field typed Evidence would let a
    grid row inject the answer."""
    assert all("Evidence" not in str(f.type) for f in fields(Setting))
