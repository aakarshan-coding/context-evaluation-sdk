"""Pure text rules. The one place that decides what "contains", "matches", and "overlaps" mean."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from .types import ChunkRef, Evidence, RetrievalEvent

EVIDENCE_TOKEN_THRESHOLD = 0.8  # plan item 1
EVIDENCE_WINDOW_FACTOR = 1.5  # architecture graft 5: token rule holds inside a window this many evidence-lengths wide
CHARS_PER_TOKEN = 4  # plan item 14

_NON_WORD = re.compile(r"[^\w\s]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def normalize(text: str) -> str:
    """Casefold, strip punctuation, collapse whitespace."""
    return " ".join(_NON_WORD.sub(" ", text.casefold()).split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def _max_window_overlap(have: Sequence[str], need: Counter[str], window: int) -> int:
    """Largest multiset overlap between `need` and any `window` consecutive tokens of `have`."""
    in_window: Counter[str] = Counter()
    overlap = best = 0
    for i, tok in enumerate(have):
        if in_window[tok] < need.get(tok, 0):
            overlap += 1
        in_window[tok] += 1
        if i >= window:
            gone = have[i - window]
            in_window[gone] -= 1
            if in_window[gone] < need.get(gone, 0):
                overlap -= 1
        best = max(best, overlap)
    return best


def contains_evidence(text: str, evidence: Evidence) -> bool:
    """True when `text` holds the normalized evidence, or >= 80% of its tokens inside a window
    about 1.5 times the evidence length.

    The token rule covers evidence split across a chunk boundary. The window stops a long text
    from matching on scattered common words.
    """
    ev = tokens(evidence.text)
    if not ev:
        return False
    text_norm = normalize(text)
    if " ".join(ev) in text_norm:
        return True
    window = math.ceil(len(ev) * EVIDENCE_WINDOW_FACTOR)
    overlap = _max_window_overlap(text_norm.split(), Counter(ev), window)
    return overlap / len(ev) >= EVIDENCE_TOKEN_THRESHOLD


def chunk_in_prompt(chunk_text: str, prompt: str) -> bool:
    chunk = normalize(chunk_text)
    return bool(chunk) and chunk in normalize(prompt)


def in_prompt_pairs(retrievals: Sequence[RetrievalEvent], prompt: str) -> tuple[ChunkRef, ...]:
    """(retrieval index, delivered chunk index) for every delivered chunk found in the prompt."""
    prompt_norm = normalize(prompt)
    pairs = []
    for r, event in enumerate(retrievals):
        for c, chunk in enumerate(event.delivered):
            chunk_norm = normalize(chunk.text)
            if chunk_norm and chunk_norm in prompt_norm:
                pairs.append((r, c))
    return tuple(pairs)


def answer_matches(expected: str, answer: str) -> bool:
    """Grader fast path (plan item 2)."""
    exp = normalize(expected)
    return bool(exp) and exp in normalize(answer)


def lexical_score(query: str, text: str) -> float:
    """Term overlap used by rerank=lexical and compress. No model download."""
    q = set(tokens(query))
    if not q:
        return 0.0
    return len(q & set(tokens(text))) / len(q)


def top_sentences(text: str, query: str, n: int) -> str:
    """Keep the n sentences with highest lexical_score, in original order."""
    sentences = [s for s in _SENTENCE_END.split(text.strip()) if s]
    if len(sentences) <= n:
        return text
    ranked = sorted(range(len(sentences)), key=lambda i: (-lexical_score(query, sentences[i]), i))
    keep = sorted(ranked[:n])
    return " ".join(sentences[i] for i in keep)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)
