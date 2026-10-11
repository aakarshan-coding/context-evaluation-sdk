"""Definition-of-done checks 2 and 3, run through the real CLI entry point (`_cli.main`)."""

from __future__ import annotations

import pytest

chromadb = pytest.importorskip("chromadb")  # demo extra; installed in the dev extra
openai = pytest.importorskip("openai")


def test_every_planted_case_gets_its_label(project, store, hooks) -> None:
    """build-tests --import planted.jsonl, run demo, diagnose --yes; then for each Planted p:
    label_of(diagnosis) == p.expected, and Missing.cause / .confirmed match p.cause / p.confirmed.
    """
    raise NotImplementedError


def test_missing_case_passes_with_larger_top_k(project, store, hooks) -> None:
    """experiment top_k=3,10 --yes; the not_retrieved planted case fails in the top_k=3 run and
    passes in the top_k=10 run; table rows have accuracy, mean tokens, mean latency.
    """
    raise NotImplementedError
