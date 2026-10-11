"""The one way to change context: override params before retrieval, transform chunks after.

Diagnosis replays and experiments both build an Intervention and hand it to
runner.execute. Hooks apply it to every retrieval call in the run (plan item 8).

Gold-aware transforms exist only for diagnosis. A Setting has no field that can
produce one, so an experiment cannot leak the answer (plan item 15) by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Union

from .types import Chunk, Evidence, never


# --- transforms (pure: chunks in, chunks out) -------------------------------


@dataclass(frozen=True)
class LexicalRerank:
    pass


class Order(Enum):
    AS_IS = "as_is"
    REVERSE = "reverse"
    ENDS = "ends"  # best chunks at both ends, weakest in the middle (lost-in-the-middle)


@dataclass(frozen=True)
class Reorder:
    order: Order


@dataclass(frozen=True)
class KeepFirst:
    n: int


@dataclass(frozen=True)
class Compress:
    sentences: int


@dataclass(frozen=True)
class InjectEvidence:
    evidence: Evidence  # prepended as a chunk with source_index=None


@dataclass(frozen=True)
class GoldToTop:
    evidence: Evidence  # gold chunks first, others kept in order


@dataclass(frozen=True)
class OnlyGold:
    evidence: Evidence  # drop every chunk that does not contain the evidence


Transform = Union[LexicalRerank, Reorder, KeepFirst, Compress, InjectEvidence, GoldToTop, OnlyGold]


def apply_transform(t: Transform, chunks: tuple[Chunk, ...], query: str | None) -> tuple[Chunk, ...]:
    """`query` falls back to the active run's question when the store call had none (pinecone)."""
    match t:
        case LexicalRerank():
            raise NotImplementedError  # stable sort by text.lexical_score(query, c.text) desc
        case Reorder(order=o):
            raise NotImplementedError
        case KeepFirst(n=n):
            return chunks[:n]
        case Compress(sentences=n):
            raise NotImplementedError  # replace text with text.top_sentences, keep source_index
        case InjectEvidence(evidence=e):
            return (Chunk(text=e.text, id="contexteval:evidence", score=None, source_index=None), *chunks)
        case GoldToTop(evidence=e):
            raise NotImplementedError  # partition by text.contains_evidence, gold first
        case OnlyGold(evidence=e):
            raise NotImplementedError
        case _:
            never(t)


@dataclass(frozen=True)
class Intervention:
    top_k: int | None = None  # overrides the hook's top_k parameter before the call
    transforms: tuple[Transform, ...] = ()  # applied left to right after the call

    @property
    def is_noop(self) -> bool:
        return self.top_k is None and not self.transforms


NO_INTERVENTION = Intervention()


def apply(iv: Intervention, chunks: tuple[Chunk, ...], query: str | None) -> tuple[Chunk, ...]:
    raise NotImplementedError  # fold apply_transform over iv.transforms


# --- experiment settings ----------------------------------------------------


@dataclass(frozen=True)
class Setting:
    """One cell of an experiment grid. Knobs from plan item 15 and nothing else."""

    top_k: int | None = None
    rerank: Literal["none", "lexical"] = "none"
    order: Order = Order.AS_IS
    max_chunks: int | None = None
    compress: int | None = None  # sentences kept per chunk

    def __post_init__(self) -> None:
        # Settings are built from CLI tokens or stored JSON. Reject non-positive ints here
        # so nothing downstream re-checks.
        for knob in ("top_k", "max_chunks", "compress"):
            value = getattr(self, knob)
            if value is not None and value < 1:
                raise ValueError(f"{knob} must be a positive integer, got {value}")

    @property
    def name(self) -> str:
        """Non-default knobs as "top_k=10 rerank=lexical", or "baseline"."""
        raise NotImplementedError

    def intervention(self) -> Intervention:
        """Fixed pipeline order: rerank, order, max_chunks, compress. Decided here once."""
        raise NotImplementedError

    def to_json(self) -> dict[str, object]:
        raise NotImplementedError

    @classmethod
    def from_json(cls, data: dict[str, object]) -> Setting:
        raise NotImplementedError


BASELINE = Setting()
