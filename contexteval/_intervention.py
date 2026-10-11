"""The one way to change context: override params before retrieval, transform chunks after.

Diagnosis replays and experiments both build an Intervention and hand it to
runner.execute. Hooks apply it to every retrieval call in the run (plan item 8).

Gold-aware transforms exist only for diagnosis. A Setting has no field that can
produce one, so an experiment cannot leak the answer (plan item 15) by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Literal, Union

from . import _text as text
from .types import Chunk, Evidence, never


class InterventionNotApplicable(Exception):
    """The active intervention cannot be applied at this retrieval site, for example a top_k
    override on a function with no top_k parameter. Propagates to the caller so an experiment
    never silently degrades into a baseline run (architecture graft 3).
    """


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

EVIDENCE_CHUNK_ID = "contexteval:evidence"


def _ends(chunks: tuple[Chunk, ...]) -> tuple[Chunk, ...]:
    """Ranks 0, 2, 4, ... ascending, then 5, 3, 1 descending: the two best chunks sit at the
    two ends of the prompt and the weakest in the middle."""
    return (*chunks[0::2], *reversed(chunks[1::2]))


def apply_transform(t: Transform, chunks: tuple[Chunk, ...], query: str | None) -> tuple[Chunk, ...]:
    """`query` falls back to the active run's question when the store call had none (pinecone).
    Query-dependent transforms leave the chunks alone when no query is known at all."""
    match t:
        case LexicalRerank():
            if query is None:
                return chunks
            return tuple(sorted(chunks, key=lambda c: -text.lexical_score(query, c.text)))
        case Reorder(order=Order.AS_IS):
            return chunks
        case Reorder(order=Order.REVERSE):
            return tuple(reversed(chunks))
        case Reorder(order=Order.ENDS):
            return _ends(chunks)
        case KeepFirst(n=n):
            return chunks[:n]
        case Compress(sentences=n):
            if query is None:
                return chunks
            return tuple(
                Chunk(text=text.top_sentences(c.text, query, n), id=c.id, score=c.score, source_index=c.source_index)
                for c in chunks
            )
        case InjectEvidence(evidence=e):
            return (Chunk(text=e.text, id=EVIDENCE_CHUNK_ID, score=None, source_index=None), *chunks)
        case GoldToTop(evidence=e):
            gold = tuple(c for c in chunks if text.contains_evidence(c.text, e))
            rest = tuple(c for c in chunks if not text.contains_evidence(c.text, e))
            return (*gold, *rest)
        case OnlyGold(evidence=e):
            return tuple(c for c in chunks if text.contains_evidence(c.text, e))
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
    for t in iv.transforms:
        chunks = apply_transform(t, chunks, query)
    return chunks


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
        if self.rerank not in ("none", "lexical"):
            raise ValueError(f"rerank must be 'none' or 'lexical', got {self.rerank!r}")

    @property
    def name(self) -> str:
        """Non-default knobs as "top_k=10 rerank=lexical", or "baseline"."""
        parts = []
        for f in fields(self):
            value = getattr(self, f.name)
            if value != f.default:
                parts.append(f"{f.name}={value.value if isinstance(value, Enum) else value}")
        return " ".join(parts) or "baseline"

    def intervention(self) -> Intervention:
        """Fixed pipeline order: rerank, order, max_chunks, compress. Decided here once."""
        transforms: list[Transform] = []
        if self.rerank == "lexical":
            transforms.append(LexicalRerank())
        if self.order is not Order.AS_IS:
            transforms.append(Reorder(self.order))
        if self.max_chunks is not None:
            transforms.append(KeepFirst(self.max_chunks))
        if self.compress is not None:
            transforms.append(Compress(self.compress))
        return Intervention(top_k=self.top_k, transforms=tuple(transforms))

    def to_json(self) -> dict[str, object]:
        return {
            "top_k": self.top_k,
            "rerank": self.rerank,
            "order": self.order.value,
            "max_chunks": self.max_chunks,
            "compress": self.compress,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> Setting:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown setting keys: {sorted(unknown)}")
        kwargs = dict(data)
        if "order" in kwargs:
            kwargs["order"] = Order(kwargs["order"])
        return cls(**kwargs)  # type: ignore[arg-type]


BASELINE = Setting()
