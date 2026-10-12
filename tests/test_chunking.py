"""v2 chunking: the `Chunking` value type and the public `chunk_text` helper apps call inside
`build(chunking)`. Each test's docstring names the defect it catches."""

from __future__ import annotations

import pytest

import contexteval
from contexteval.types import Chunking

P1 = "Alpha paragraph with forty characters in it."  # 45 chars
P2 = "Beta paragraph also of about forty chars."  # 41 chars
P3 = "Gamma paragraph rounds out the small document."  # 46 chars
DOC = f"# Title line\n\n{P1}\n\n{P2}\n\n\n{P3}\n"
RESOLVED = Chunking(chunk_size=100, overlap=0, title=False)


def test_chunk_text_packs_whole_paragraphs_up_to_the_limit_and_never_cuts_inside_one() -> None:
    """Defect: a chunker that cuts inside a paragraph (so a sentence can end up in two chunks), one
    that ignores the limit and packs everything into one chunk, or one that puts every paragraph
    in its own chunk whatever the limit. Blank lines of any width separate paragraphs, and a
    paragraph longer than the limit is its own chunk, as the real bot's chunker did."""
    assert contexteval.chunk_text(DOC, RESOLVED) == [f"# Title line\n\n{P1}", f"{P2}\n\n{P3}"]
    assert contexteval.chunk_text(DOC, Chunking(chunk_size=1000, overlap=0, title=False)) == [f"# Title line\n\n{P1}\n\n{P2}\n\n{P3}"]
    assert contexteval.chunk_text(DOC, Chunking(chunk_size=1, overlap=0, title=False)) == ["# Title line", P1, P2, P3]
    assert contexteval.chunk_text("   \n\n  ", RESOLVED) == []


def test_chunk_text_overlap_carries_the_end_of_the_previous_chunk_in_whole_words() -> None:
    """Defect: no overlap at all, so evidence that straddles a chunk boundary is in no chunk; a tail
    cut in the middle of a word; a tail that does not count toward the limit, so chunks grow past
    it; or a chunk made of the carried tail alone when the next paragraph does not fit beside it."""
    chunks = contexteval.chunk_text(DOC, Chunking(chunk_size=100, overlap=20, title=False))
    assert chunks[0] == f"# Title line\n\n{P1}"
    assert chunks[1].startswith("forty characters in it.\n\n" + P2), chunks[1]
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[1].split("\n\n")[0] in P1 and not chunks[1].startswith("y characters")
    assert len(chunks) == 3 and chunks[2].endswith(P3) and chunks[2] != chunks[1].split("\n\n")[-1]
    assert all(c.split("\n\n")[-1] in (P1, P2, P3, "# Title line") for c in chunks)
    whole = contexteval.chunk_text(f"{P1}\n\n{P2}", Chunking(chunk_size=50, overlap=49, title=False))
    assert whole == [P1, f"{P1}\n\n{P2}"]


def test_chunk_text_title_prefixes_every_chunk_without_changing_the_packing() -> None:
    """The real-bot audit's root cause: a spec list cut from its heading carries no product name.
    Defect: the title on the first chunk only, a title that counts toward the limit and shifts the
    packing (so a title experiment also changes chunk boundaries), or a prefix when the app gave
    no title."""
    on = Chunking(chunk_size=100, overlap=0, title=True)
    with_title = contexteval.chunk_text(DOC, on, title="Trailhawk 29")
    without = contexteval.chunk_text(DOC, RESOLVED, title="Trailhawk 29")
    assert with_title == [f"Trailhawk 29\n\n{c}" for c in without]
    assert contexteval.chunk_text(DOC, on) == without
    assert contexteval.chunk_text(DOC, RESOLVED, title="Trailhawk 29") == without


def test_chunk_text_refuses_a_chunking_with_an_unset_knob() -> None:
    """Defect: chunk_text fills an unset knob with a built-in size, so `title=on` on an app whose
    own chunks are 1000 characters silently also moves it to 500, and the experiment measures two
    changes as one. The error names with_defaults as the fix."""
    for partial in (Chunking(), Chunking(title=True), Chunking(chunk_size=300, overlap=0), Chunking(chunk_size=300, title=False)):
        with pytest.raises(ValueError, match="with_defaults"):
            contexteval.chunk_text(DOC, partial)


def test_chunking_with_defaults_fills_only_the_unset_knobs() -> None:
    """Defect: with_defaults overwrites a knob the experiment set, so every row rebuilds at the
    app's own size; or leaves a knob unset, so chunk_text refuses it."""
    assert Chunking().with_defaults(chunk_size=500) == Chunking(500, 0, False)
    assert Chunking(title=True).with_defaults(chunk_size=500, overlap=50) == Chunking(500, 50, True)
    assert Chunking(chunk_size=300).with_defaults(chunk_size=500, overlap=50, title=True) == Chunking(300, 50, True)
    assert Chunking().is_default and not Chunking(overlap=0).is_default
    assert Chunking(chunk_size=800, overlap=100, title=True).name == "chunk_size=800 overlap=100 title=on"
    assert Chunking(title=False).name == "title=off" and Chunking().name == "the app's own chunking"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_size": 0}, "chunk_size must be a positive integer"),
        ({"overlap": -1}, "overlap must be 0 or more"),
        ({"chunk_size": 100, "overlap": 100}, "overlap .* must be smaller than chunk_size"),
    ],
)
def test_chunking_rejects_values_that_cannot_chunk(kwargs: dict[str, int], message: str) -> None:
    """Boundary: a zero size would loop or put every character in its own chunk, and an overlap at
    or past the size would carry the whole chunk forward. Defect: the value reaches chunk_text."""
    with pytest.raises(ValueError, match=message):
        Chunking(**kwargs)
    assert Chunking(chunk_size=100, overlap=99).overlap == 99


def test_chunking_and_chunk_text_are_public() -> None:
    """Defect: the builder contract makes users import an underscore module."""
    assert contexteval.Chunking is Chunking
    assert "Chunking" in contexteval.__all__ and "chunk_text" in contexteval.__all__
