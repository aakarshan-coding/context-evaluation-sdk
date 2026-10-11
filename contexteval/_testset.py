"""Test set builder (plan item 13) and user test import. Both are boundaries that produce TestCase."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import _text as text
from ._llm import Estimate, LLMRequest, Provider, Task, estimate
from .types import CaseId, Evidence, TestCase

SYNTHETIC_TEST_SET = "synthetic"
_OUTPUT_TOKENS_PER_CASE = 120

_SYSTEM = (
    "You write evaluation cases for a retrieval-augmented question answering system. "
    "Reply with only a JSON list. Each item has string fields question, answer, evidence. "
    "evidence must be copied character for character from the passage, one or two sentences "
    "that contain the answer. The answer must be short and appear in the evidence."
)


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
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".md", ".txt"))
    return [Document(path=p.as_posix(), text=p.read_text(encoding="utf-8")) for p in paths]


_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")


def passages(doc: Document, max_chars: int = 2000) -> list[str]:
    """Paragraph-aligned windows sent to the generator. Our own split, unrelated to the user's
    chunking. Each window is a slice of the document text, so quoted evidence stays verbatim."""
    bounds = [0, *(m.end() for m in _PARAGRAPH_BREAK.finditer(doc.text)), len(doc.text)]
    windows = []
    start = 0
    for prev, end in zip(bounds, bounds[1:]):
        if end - start > max_chars and prev > start:
            windows.append(doc.text[start:prev])
            start = prev
    windows.append(doc.text[start:])
    return [w.strip() for w in windows if w.strip()]


def _request(doc: Document, passage: str, per_passage: int) -> LLMRequest:
    return LLMRequest(
        task=Task.GENERATE_CASES,
        system=_SYSTEM,
        user=f"Write {per_passage} cases from this passage of {doc.path}.\n\nPassage:\n{passage}",
        inputs={"document": doc.path, "passage": passage, "count": str(per_passage)},
        max_tokens=_OUTPUT_TOKENS_PER_CASE * per_passage + 200,
    )


def estimate_build(docs: list[Document], provider: Provider, per_passage: int) -> Estimate:
    prompts = [r.system + r.user for d in docs for r in (_request(d, p, per_passage) for p in passages(d))]
    return estimate(provider.model, prompts, _OUTPUT_TOKENS_PER_CASE * per_passage)


OnCases = Callable[[tuple[TestCase, ...]], None]


def build_tests(
    docs: list[Document], provider: Provider, per_passage: int = 2, on_cases: OnCases = lambda cases: None
) -> BuildReport:
    """One GENERATE_CASES request per passage asking for a JSON list of
    {question, answer, evidence}. Each item goes through parse_generated.

    `on_cases` gets each passage's new cases as soon as they are parsed, so a caller that stores
    them keeps every paid case when a later request raises ProviderError.
    """
    cases: dict[CaseId, TestCase] = {}
    rejected: list[Rejection] = []
    for doc in docs:
        for passage in passages(doc):
            reply = provider.complete(_request(doc, passage, per_passage)).text
            try:
                items = json.loads(_strip_fence(reply))
            except json.JSONDecodeError:
                rejected.append(Rejection(doc.path, "invalid JSON"))
                continue
            if not isinstance(items, list):
                rejected.append(Rejection(doc.path, "reply is not a JSON list"))
                continue
            new: list[TestCase] = []
            for item in items:
                result = parse_generated(item, doc)
                if isinstance(result, Rejection):
                    rejected.append(result)
                elif result.id not in cases:
                    cases[result.id] = result
                    new.append(result)
            on_cases(tuple(new))
    return BuildReport(cases=tuple(cases.values()), rejected=tuple(rejected))


def _strip_fence(reply: str) -> str:
    """Models often wrap JSON in a ```json fence despite instructions."""
    s = reply.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        s = s.rsplit("```", 1)[0]
    return s


def parse_generated(item: object, doc: Document) -> TestCase | Rejection:
    """Reject unless item has non-empty str fields and `evidence` is a verbatim substring of
    doc.text (exact match on the raw text, not normalized). This is where Evidence is proven.
    """
    if not isinstance(item, dict):
        return Rejection(doc.path, "item is not a JSON object")
    for key in ("question", "answer", "evidence"):
        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            return Rejection(doc.path, f"missing field: {key}")
    question, answer, evidence = item["question"].strip(), item["answer"].strip(), item["evidence"].strip()
    if evidence not in doc.text:
        return Rejection(doc.path, "evidence not verbatim")
    return TestCase(id=case_id(question, evidence), question=question, expected_answer=answer, evidence=Evidence(evidence), source=doc.path)


def import_tests(path: Path) -> list[TestCase]:
    """JSONL or CSV with question, expected_answer, optional evidence. Raises ValueError naming
    the line on any malformed row; nothing is half-imported.
    """
    try:
        raw = path.read_text(encoding="utf-8-sig")  # Excel writes "CSV UTF-8" with a BOM
    except OSError as e:
        raise ValueError(f"{path}: {e.strerror or e}") from None
    if path.suffix.lower() == ".csv":
        rows = [(i + 2, row) for i, row in enumerate(csv.DictReader(raw.splitlines()))]
    else:
        rows = []
        for lineno, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append((lineno, json.loads(line)))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e.msg}") from None
    cases = []
    for lineno, row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{lineno}: expected an object with question and expected_answer")
        for key in ("question", "expected_answer"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{path}:{lineno}: missing field {key!r}")
        evidence = row.get("evidence")
        if evidence is not None and not isinstance(evidence, str):
            raise ValueError(f"{path}:{lineno}: evidence must be a string")
        evidence = evidence.strip() if evidence else None
        question = row["question"].strip()
        cases.append(
            TestCase(
                id=case_id(question, evidence),
                question=question,
                expected_answer=row["expected_answer"].strip(),
                evidence=Evidence(evidence) if evidence else None,
                source=path.as_posix(),
            )
        )
    return cases


def case_id(question: str, evidence: str | None) -> CaseId:
    digest = hashlib.sha1((text.normalize(question) + "\0" + text.normalize(evidence or "")).encode("utf-8"))
    return CaseId(digest.hexdigest()[:16])
