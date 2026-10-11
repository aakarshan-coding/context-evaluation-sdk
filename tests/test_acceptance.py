"""Definition-of-done checks 2 and 3, run through the real CLI entry point (`_cli.main`)."""

from __future__ import annotations

from pathlib import Path

import pytest

chromadb = pytest.importorskip("chromadb")  # demo extra; installed in the dev extra
openai = pytest.importorskip("openai")

from contexteval import _cli  # noqa: E402
from contexteval._store import Store  # noqa: E402
from contexteval.demo import fake_llm, planted  # noqa: E402
from contexteval.types import Missing, label_of  # noqa: E402


def _latest_diagnoses(project: Path) -> dict[str, object]:
    """case id -> Diagnosis | None for every trace of the latest baseline run."""
    store = Store.open(project)
    try:
        latest = store.latest_run("baseline")
        assert latest is not None
        traces = store.traces(latest.id)
    finally:
        store.close()
    return {s.trace.test.case_id: s.diagnosis for s in traces if s.trace.test}


def test_every_planted_case_gets_its_label(project: Path, hooks: None, capsys: pytest.CaptureFixture[str]) -> None:
    """DoD item 2 through the CLI: init, import planted.jsonl, run the demo, diagnose. Defects:
    any planted failure labeled differently from PLANTED (a wrong ladder order, a flaky rerun
    that does not run first, a missing cause read from the wrong side of the app's filter, or
    confirmed not reflecting the inject replay); a healthy case diagnosed at all; and a second
    `diagnose` that is not a no-op (graft 2): it must leave every diagnosis as it was and make
    no app call, which the flaky counter would record."""
    assert _cli.main(["init", "--provider", "fake"]) == 0
    jsonl = planted.write_jsonl(project / planted.IMPORT_FILE)
    assert _cli.main(["build-tests", "--import", str(jsonl)]) == 0
    assert _cli.main(["run", "contexteval.demo.app:answer", "--yes"]) == 0
    capsys.readouterr()

    assert _cli.main(["diagnose", "--yes"]) == 0
    out = capsys.readouterr().out
    failures = [p for p in planted.PLANTED if p.expected is not None]
    assert f"diagnosed {len(failures)}, skipped 0 already labeled, skipped 0 crashed" in out

    diagnoses = _latest_diagnoses(project)
    assert set(diagnoses) == {p.case.id for p in planted.PLANTED}
    labels = {p.case.question: label_of(diagnoses[p.case.id]) for p in planted.PLANTED if diagnoses[p.case.id] is not None}
    assert labels == {p.case.question: p.expected for p in failures}
    for p in failures:
        d = diagnoses[p.case.id]
        assert f"{p.expected.value:<12} {p.case.question}" in out  # type: ignore[union-attr]
        if p.cause is not None:
            assert isinstance(d, Missing)
            assert (d.cause, d.confirmed) == (p.cause, p.confirmed)
        else:
            assert not isinstance(d, Missing)

    state_file = fake_llm.default_state_file()
    counters_before = state_file.read_text(encoding="utf-8")
    assert _cli.main(["diagnose", "--yes"]) == 0
    assert "nothing to diagnose" in capsys.readouterr().out
    assert _latest_diagnoses(project) == diagnoses
    assert state_file.read_text(encoding="utf-8") == counters_before
