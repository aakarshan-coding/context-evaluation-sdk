"""Pure text rules: what "contains evidence", "in prompt", and "overlaps" mean (plan item 1)."""

from __future__ import annotations

from contexteval import _text as text
from contexteval.types import Chunk, Evidence, RetrievalEvent

EVIDENCE = Evidence("The refund window closes thirty days after the delivery date for every order")  # 13 tokens


def _event(*texts: str) -> RetrievalEvent:
    chunks = tuple(Chunk(text=t, id=str(i), score=None, source_index=i) for i, t in enumerate(texts))
    return RetrievalEvent(source="test", query="q", params={}, retrieved=chunks, delivered=chunks, latency_ms=0.0)


def test_normalize_casefolds_strips_punctuation_and_collapses_whitespace() -> None:
    """Defect: a comparison that keeps case or punctuation misses a verbatim quote wrapped differently."""
    assert text.normalize("  The Refund,\n window;  CLOSES.  ") == "the refund window closes"


def test_exact_evidence_survives_case_and_punctuation_changes() -> None:
    """Defect: raw substring matching rejects the same sentence rendered with different punctuation."""
    chunk = "Policy: the refund window closes thirty days after the delivery date, for every order!"
    assert text.contains_evidence(chunk, EVIDENCE)


def test_evidence_split_across_chunk_boundary_matches_by_tokens() -> None:
    """Plan item 1: a chunk holding 11 of 13 evidence tokens (the rest fell into the next chunk)
    still counts as gold. Defect: an exact-substring-only rule labels that case as missing."""
    first_chunk = "Section 4. The refund window closes thirty days after the delivery date for"
    assert text.contains_evidence(first_chunk, EVIDENCE)


def test_below_threshold_overlap_is_not_evidence() -> None:
    """Defect: a threshold applied as > 0 instead of >= 80% accepts a chunk sharing a few words."""
    partial = "The refund window closes"  # 4 of 13 tokens
    assert not text.contains_evidence(partial, EVIDENCE)


def test_windowed_rule_rejects_scattered_tokens_in_long_text() -> None:
    """Architecture graft 5. Every evidence token appears somewhere in a long prompt, but never
    close together. Defect: the unwindowed 80% rule (set intersection over the whole text) says
    the prompt contains gold, so a real missing failure is mislabeled."""
    filler = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit", "sed", "do"]
    words: list[str] = []
    for i, tok in enumerate(text.tokens(EVIDENCE.text)):
        words.extend(filler[(i + j) % len(filler)] for j in range(25))
        words.append(tok)
    scattered = " ".join(words)
    assert all(tok in scattered.split() for tok in text.tokens(EVIDENCE.text))
    assert not text.contains_evidence(scattered, EVIDENCE)


def test_windowed_rule_accepts_dense_match_inside_long_text() -> None:
    """Defect: a window that only looks at the start of the text misses evidence near the end."""
    prompt = " ".join(["noise"] * 500) + " the refund window closes thirty days after the delivery date for every"
    assert text.contains_evidence(prompt, EVIDENCE)


def test_empty_evidence_never_matches() -> None:
    """Defect: an empty evidence string is a substring of everything and would mark every chunk gold."""
    assert not text.contains_evidence("anything", Evidence(""))
    assert not text.contains_evidence("anything", Evidence("..."))


def test_gold_split_is_true_only_when_the_evidence_is_found_by_joining_alone() -> None:
    """v2 chunking, the split signal. Defect: a split reported when a chunk also holds the evidence
    whole (the join was not needed), when the halves are not adjacent (no gold at all), or when
    nothing holds it; or no split reported for the one case that is one: adjacent halves and no
    whole chunk. The flags and the signal come from one rule, so they agree on which chunks are gold."""
    first = "Section 4. The refund window closes thirty days after"
    second = "the delivery date for every order. Section 5 covers exchanges."
    assert not text.contains_evidence(first, EVIDENCE) and not text.contains_evidence(second, EVIDENCE)
    assert text.gold_split([first, second], EVIDENCE) is True
    assert text.gold_flags([first, second], EVIDENCE) == (True, True)
    assert text.gold_split(["noise", first, second, "noise"], EVIDENCE) is True
    assert text.gold_split([first, second, EVIDENCE.text], EVIDENCE) is False
    assert text.gold_split([first, "noise", second], EVIDENCE) is False
    assert text.gold_split(["noise"], EVIDENCE) is False and text.gold_split([], EVIDENCE) is False


def test_in_prompt_pairs_lists_only_delivered_chunks_found_in_prompt() -> None:
    """Defect: pairs computed over `retrieved` instead of `delivered`, or a chunk the app never put
    into the prompt reported as in_prompt."""
    events = (_event("alpha beta", "gamma delta"), _event("epsilon zeta", "eta theta"))
    prompt = "Context:\nALPHA, beta.\nEta theta!\nQuestion: ?"
    assert text.in_prompt_pairs(events, prompt) == ((0, 0), (1, 1))
    assert text.chunk_in_prompt("gamma delta", prompt) is False
    assert text.chunk_in_prompt("", prompt) is False


def test_answer_matches_is_normalized_substring() -> None:
    """Grader fast path (plan item 2). Defect: exact comparison fails on 'Paris.' vs 'paris'."""
    assert text.answer_matches("Paris", "The capital is paris.")
    assert not text.answer_matches("Paris", "I do not know.")
    assert not text.answer_matches("", "anything")


def test_lexical_score_is_query_term_coverage() -> None:
    """Defect: scoring by text length or by raw count lets a long chunk outrank an exact hit."""
    assert text.lexical_score("refund window", "the refund window closes") == 1.0
    assert text.lexical_score("refund window", "the window is open") == 0.5
    assert text.lexical_score("", "anything") == 0.0


def test_top_sentences_keeps_best_n_in_original_order() -> None:
    """Defect: returning sentences sorted by score scrambles the passage; or n larger than the
    sentence count drops text."""
    passage = "Cats sleep a lot. The refund window closes in thirty days. Dogs bark. Refunds need a receipt."
    assert text.top_sentences(passage, "refund window receipt", 2) == (
        "The refund window closes in thirty days. Refunds need a receipt."
    )
    assert text.top_sentences(passage, "refund", 10) == passage


def test_estimate_tokens_is_chars_over_four_with_floor_one() -> None:
    """Plan item 14. Defect: an empty prompt estimated at zero tokens divides later cost math by zero."""
    assert text.estimate_tokens("") == 1
    assert text.estimate_tokens("x" * 400) == 100


def test_answer_matches_needs_whole_tokens() -> None:
    """Review Y1. Defect: the fast path matches inside words, so a short expected answer passes
    on an unrelated reply ("4" in "14 days", "No" in "I do not know") and the LLM grader never
    runs."""
    assert not text.answer_matches("4", "It takes 14 days.")
    assert not text.answer_matches("No", "I do not know.")
    assert not text.answer_matches("yes", "Your eyes see it.")
    assert text.answer_matches("14 days", "It takes 14 days.")
    assert text.answer_matches("No", "No, it cannot.")


def test_exact_evidence_needs_whole_tokens() -> None:
    """Review Y1. Defect: the exact branch of contains_evidence matches across word boundaries,
    so "the cat" is found in "bathe catalog" and a chunk without the evidence counts as gold."""
    assert not text.contains_evidence("bathe catalog", Evidence("the cat"))
    assert not text.contains_evidence("we concatenate strings", Evidence("cat"))
    assert not text.contains_evidence("this is greenhouse", Evidence("is green"))
    assert text.contains_evidence("feed the cat daily", Evidence("the cat"))


def test_in_prompt_needs_whole_tokens() -> None:
    """Review Y1. Defect: a chunk "Yes" is reported in a prompt that only says "Yesterday", so
    in_prompt marks chunks the app never sent."""
    assert text.in_prompt_pairs((_event("Yes", "is sunny"),), "Yesterday is sunny") == ((0, 1),)
    assert text.chunk_in_prompt("Yes", "Yesterday is sunny") is False
    assert text.chunk_in_prompt("Yes", "Yes. Yesterday is sunny") is True
