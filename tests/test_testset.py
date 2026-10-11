"""P2: the test set builder proves evidence verbatim, rebuilds upsert, and imports are all-or-nothing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contexteval import _testset
from contexteval._llm import FakeProvider, LLMRequest, Task
from contexteval._store import Store
from contexteval._testset import Document, build_tests, case_id, import_tests, passages, read_documents
from contexteval.demo import planted
from contexteval.demo.app import CORPUS_DIR

DOC_TEXT = "# Tents\n\nThe Ridge 2 tent weighs 1.8 kilograms.\n\nIts poles are aluminum and the fly is silnylon.\n"


def _docs(tmp_path: Path) -> list[Document]:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "tents.md").write_text(DOC_TEXT, encoding="utf-8")
    (tmp_path / "docs" / "ignored.pdf").write_text("not a doc", encoding="utf-8")
    return read_documents(tmp_path / "docs")


def test_build_keeps_verbatim_evidence_and_rejects_fabricated(tmp_path: Path) -> None:
    """Defect: a case whose evidence the model paraphrased or invented is stored, so later gold
    checks look for text that is in no document. Also: a non-JSON reply crashing the build."""
    replies = iter(
        [
            json.dumps(
                [
                    {"question": "How heavy is the Ridge 2?", "answer": "1.8 kilograms", "evidence": "The Ridge 2 tent weighs 1.8 kilograms."},
                    {"question": "What are the poles made of?", "answer": "aluminum", "evidence": "The poles are made of aluminum."},
                    {"question": "What is the fly made of?", "evidence": "the fly is silnylon"},
                ]
            ),
            "Sure! Here are your cases.",
        ]
    )
    fake = FakeProvider(handlers={Task.GENERATE_CASES: lambda req: next(replies)})
    docs = _docs(tmp_path)
    report = build_tests(docs, fake, per_passage=3)
    report_bad_json = build_tests(docs, fake, per_passage=3)

    assert [c.question for c in report.cases] == ["How heavy is the Ridge 2?"]
    case = report.cases[0]
    assert case.evidence is not None and case.evidence.text in DOC_TEXT
    assert case.id == case_id(case.question, case.evidence.text)
    assert sorted(r.reason for r in report.rejected) == ["evidence not verbatim", "missing field: answer"]
    assert report_bad_json.cases == () and [r.reason for r in report_bad_json.rejected] == ["invalid JSON"]
    assert [d.path.rsplit("/", 1)[-1] for d in docs] == ["tents.md"]


def test_fake_provider_build_over_demo_corpus_is_all_verbatim() -> None:
    """Defect: the builder dispatches the fake on prompt text, or passages are not slices of the
    document, so the default generator's evidence fails the verbatim check."""
    fake = FakeProvider()
    docs = read_documents(CORPUS_DIR)
    report = build_tests(docs, fake, per_passage=2)
    by_path = {d.path: d.text for d in docs}
    assert len(report.cases) == 2 * len(docs) and report.rejected == ()
    assert all(c.evidence is not None and c.evidence.text in by_path[c.source] for c in report.cases)
    assert all(r.task is Task.GENERATE_CASES and r.inputs["passage"] in by_path[r.inputs["document"]] for r in fake.requests)


def test_passages_split_on_paragraphs_within_budget() -> None:
    """Defect: a passage cut mid-paragraph, or one that is not a verbatim slice of the document."""
    doc = Document("d.md", "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(10)))
    windows = passages(doc, max_chars=400)
    assert len(windows) > 1 and all(w in doc.text for w in windows)
    assert all(w.startswith("Paragraph") for w in windows)
    assert sum(w.count("Paragraph") for w in windows) == 10


def test_rebuild_upserts_without_duplicates(tmp_path: Path, store: Store) -> None:
    """Defect: ids that change between builds (random or time-based) so every rebuild doubles
    the test set."""
    docs = read_documents(CORPUS_DIR)
    first = build_tests(docs, FakeProvider())
    assert store.upsert_cases(_testset.SYNTHETIC_TEST_SET, first.cases) == len(first.cases)
    second = build_tests(docs, FakeProvider())
    assert store.upsert_cases(_testset.SYNTHETIC_TEST_SET, second.cases) == 0
    assert len(store.cases()) == len(first.cases)


def test_import_jsonl(tmp_path: Path) -> None:
    """Defect: imported ids differ from built ids for the same content, evidence-less rows get
    an empty Evidence instead of None, or a malformed row half-imports the file."""
    path = tmp_path / "mine.jsonl"
    path.write_text(
        json.dumps({"question": "Q1?", "expected_answer": "A1", "evidence": "E one."}) + "\n\n"
        + json.dumps({"question": "Q2?", "expected_answer": "A2"}) + "\n",
        encoding="utf-8",
    )
    cases = import_tests(path)
    assert [(c.question, c.expected_answer) for c in cases] == [("Q1?", "A1"), ("Q2?", "A2")]
    assert cases[0].evidence is not None and cases[0].evidence.text == "E one." and cases[1].evidence is None
    assert cases[0].id == case_id("Q1?", "E one.")

    path.write_text(path.read_text(encoding="utf-8") + '{"question": "Q3?"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"mine\.jsonl:4: missing field 'expected_answer'"):
        import_tests(path)
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":1: invalid JSON"):
        import_tests(path)


def test_planted_jsonl_round_trips_to_planted_ids(tmp_path: Path) -> None:
    """Defect: `build-tests --import planted.jsonl` yields ids that do not match PLANTED, so the
    acceptance tests cannot find each planted case's stored trace."""
    cases = import_tests(planted.write_jsonl(tmp_path / planted.IMPORT_FILE))
    assert [c.id for c in cases] == [c.id for c in planted.cases()]
    assert [c.evidence for c in cases] == [c.evidence for c in planted.cases()]


def test_fake_provider_generate_reads_inputs_not_prompt() -> None:
    """Defect: the fake parses the rendered prompt, so changing prompt wording breaks tests."""
    req = LLMRequest(Task.GENERATE_CASES, system="", user="unrelated", inputs={"passage": "Alpha beta gamma delta epsilon zeta.", "count": "1"})
    items = json.loads(FakeProvider().complete(req).text)
    assert items == [{"question": "According to the document, what completes: Alpha beta gamma ...?", "answer": "delta epsilon zeta", "evidence": "Alpha beta gamma delta epsilon zeta."}]


def test_import_csv_with_a_byte_order_mark(tmp_path: Path) -> None:
    """Review Y9. Excel saves "CSV UTF-8" with a byte order mark. Defect: the first header reads
    "\ufeffquestion", so every row fails with "missing field 'question'"."""
    path = tmp_path / "tests.csv"
    path.write_bytes("question,expected_answer\nHow heavy is the tent?,1.8 kilograms\n".encode("utf-8-sig"))
    (case,) = import_tests(path)
    assert (case.question, case.expected_answer) == ("How heavy is the tent?", "1.8 kilograms")
