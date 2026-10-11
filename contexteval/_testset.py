"""Test set builder (plan item 13) and user test import. Both are boundaries that produce TestCase."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._llm import Estimate, Provider
from .types import CaseId, TestCase


@dataclass(frozen=True)
class Document:
    path: str
    text: str


@dataclass(frozen=True)
class Rejection:
    document: str
    reason: str  # "evidence not verbatim", "invalid JSON", "missing field: answer"


@dataclass(frozen=True)
class BuildReport:
    cases: tuple[TestCase, ...]
    rejected: tuple[Rejection, ...]


def read_documents(root: Path) -> list[Document]:
    """Every .md and .txt under root, sorted by path for deterministic output."""
    raise NotImplementedError


def passages(doc: Document, max_chars: int = 2000) -> list[str]:
    """Paragraph-aligned windows sent to the generator. Our own split, unrelated to the user's chunking."""
    raise NotImplementedError


def estimate_build(docs: list[Document], provider: Provider, per_passage: int) -> Estimate:
    raise NotImplementedError


def build_tests(docs: list[Document], provider: Provider, per_passage: int = 2) -> BuildReport:
    """One GENERATE_CASES request per passage asking for a JSON list of
    {question, answer, evidence}. Each item goes through parse_generated.
    """
    raise NotImplementedError


def parse_generated(item: object, doc: Document) -> TestCase | Rejection:
    """Reject unless item has non-empty str fields and `evidence` is a verbatim substring of
    doc.text (exact match on the raw text, not normalized). This is where Evidence is proven.
    """
    raise NotImplementedError


def import_tests(path: Path) -> list[TestCase]:
    """JSONL or CSV with question, expected_answer, optional evidence. Raises ValueError naming
    the line on any malformed row; nothing is half-imported.
    """
    raise NotImplementedError


def case_id(question: str, evidence: str | None) -> CaseId:
    raise NotImplementedError  # sha1(normalize(question) + "\0" + normalize(evidence or ""))[:16]
