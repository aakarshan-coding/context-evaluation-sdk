"""Pure text rules. The one place that decides what "contains", "matches", and "overlaps" mean."""

from __future__ import annotations

from .types import Evidence

EVIDENCE_TOKEN_THRESHOLD = 0.8  # plan item 1
CHARS_PER_TOKEN = 4  # plan item 14


def normalize(text: str) -> str:
    """Casefold, strip punctuation, collapse whitespace."""
    raise NotImplementedError


def tokens(text: str) -> list[str]:
    raise NotImplementedError  # normalize(text).split()


def contains_evidence(text: str, evidence: Evidence) -> bool:
    """True when `text` holds the normalized evidence, or >= 80% of its tokens.

    The token rule covers evidence split across a chunk boundary.
    """
    # TODO: substring on normalized forms first; else
    #   ev = tokens(evidence.text); have = set(tokens(text))
    #   sum(t in have for t in ev) / len(ev) >= EVIDENCE_TOKEN_THRESHOLD
    raise NotImplementedError


def chunk_in_prompt(chunk_text: str, prompt: str) -> bool:
    raise NotImplementedError  # normalized substring


def answer_matches(expected: str, answer: str) -> bool:
    """Grader fast path (plan item 2)."""
    raise NotImplementedError  # normalize(expected) in normalize(answer)


def lexical_score(query: str, text: str) -> float:
    """Term overlap used by rerank=lexical and compress. No model download."""
    raise NotImplementedError


def top_sentences(text: str, query: str, n: int) -> str:
    """Keep the n sentences with highest lexical_score, in original order."""
    raise NotImplementedError


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)
