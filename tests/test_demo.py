"""P2: the planted demo is the backbone for the P3 and P4 acceptance tests. These tests pin its
corpus ranks, check that each fake rule plus the replay ladder yields the planted label, and
check that the flaky counter survives separate processes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

chromadb = pytest.importorskip("chromadb")
openai = pytest.importorskip("openai")

from contexteval import _cli  # noqa: E402
from contexteval import _text as text  # noqa: E402
from contexteval._intervention import NO_INTERVENTION, GoldToTop, InjectEvidence, Intervention, OnlyGold  # noqa: E402
from contexteval._llm import FakeProvider  # noqa: E402
from contexteval._runner import make_grader, run_case  # noqa: E402
from contexteval._store import Store  # noqa: E402
from contexteval.demo import app as demo_app  # noqa: E402
from contexteval.demo import fake_llm, planted  # noqa: E402
from contexteval.demo.fake_llm import NeedsEvidenceFirst, NeedsNoDistractor  # noqa: E402
from contexteval.types import Label  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
K = demo_app.TOP_K


def _ranks(p: planted.Planted) -> tuple[list[int], list[str]]:
    """1-based ranks of chunks holding the evidence, and every chunk in rank order."""
    col = demo_app.collection()
    docs = col.query(query_texts=[p.case.question], n_results=col.count())["documents"][0]
    assert p.case.evidence is not None
    return [i + 1 for i, d in enumerate(docs) if text.contains_evidence(d, p.case.evidence)], docs


@pytest.mark.parametrize("p", planted.PLANTED, ids=lambda p: p.expected.value + (f"-{p.cause}" if p.cause else "") if p.expected else "healthy")
def test_planted_gold_ranks(p: planted.Planted) -> None:
    """Defect: a corpus or question edit moves a gold chunk, so a planted case stops exercising
    its failure (for example the not_retrieved gold entering the top 4, or leaving the top 10,
    which breaks the P4 top_k=10 experiment)."""
    gold, docs = _ranks(p)
    assert len(gold) == 1, f"exactly one chunk should hold the evidence, found ranks {gold}"
    (rank,) = gold
    if p.cause == "not_retrieved":
        assert K < rank <= 10
    elif p.cause == "dropped":
        assert rank <= K and demo_app.ARCHIVED_MARKER in docs[rank - 1]
        assert demo_app.ARCHIVED_MARKER not in p.case.evidence.text  # type: ignore[union-attr]
    elif isinstance(p.rule, NeedsEvidenceFirst):
        assert 1 < rank <= K
    elif isinstance(p.rule, NeedsNoDistractor):
        assert rank <= K and any(p.rule.distractor_marker in d for d in docs[:K])
    else:
        assert rank <= K


def _passes(p: planted.Planted, iv: Intervention) -> bool:
    trace = run_case(demo_app.answer, p.case, make_grader(FakeProvider()), iv)
    assert trace.test is not None
    return trace.test.passed


# Replay outcomes each label requires, in plan item 5's ladder order.
EXPECTED_REPLAYS: dict[Label, dict[str, bool]] = {
    Label.MISSING: {"baseline": False, "inject": True},
    Label.IGNORED: {"baseline": False, "gold_to_top": True},
    Label.IRRELEVANT: {"baseline": False, "gold_to_top": False, "only_gold": True},
    Label.NOT_CONTEXT: {"baseline": False, "gold_to_top": False, "only_gold": False},
}


@pytest.mark.parametrize("p", [p for p in planted.PLANTED if p.expected is not Label.FLAKY], ids=lambda p: p.case.question)
def test_rules_plus_ladder_yield_the_planted_label(p: planted.Planted, project: Path, hooks: None) -> None:
    """Defect: a fake rule whose correctness condition does not match the replay that should fix
    it, so the diagnoser (P3) would assign a different label than PLANTED says."""
    assert p.case.evidence is not None
    replays = {
        "baseline": NO_INTERVENTION,
        "inject": Intervention(transforms=(InjectEvidence(p.case.evidence),)),
        "gold_to_top": Intervention(transforms=(GoldToTop(p.case.evidence),)),
        "only_gold": Intervention(transforms=(OnlyGold(p.case.evidence),)),
    }
    if p.expected is None:
        assert _passes(p, NO_INTERVENTION)
        return
    outcomes = {step: _passes(p, replays[step]) for step in EXPECTED_REPLAYS[p.expected]}
    assert outcomes == EXPECTED_REPLAYS[p.expected]


def test_not_retrieved_case_passes_at_top_k_10(project: Path, hooks: None) -> None:
    """Defect: the planted missing case cannot be fixed by a larger top_k, which P4's acceptance
    (DoD item 3) depends on."""
    (p,) = [p for p in planted.PLANTED if p.cause == "not_retrieved"]
    assert not _passes(p, NO_INTERVENTION)
    assert _passes(p, Intervention(top_k=10))


def test_flipflop_alternates_and_persists_in_state_file(tmp_path: Path) -> None:
    """Defect: the flaky counter kept in memory, or shared across questions."""
    (p,) = [p for p in planted.PLANTED if p.expected is Label.FLAKY]
    state = tmp_path / "state.json"
    prompt = demo_app.build_prompt(p.case.question, ["Trailhead Gear is headquartered in Boulder, Colorado."])
    other = demo_app.build_prompt("unscripted question?", ["Some chunk."])
    answers = [fake_llm.respond(prompt, state) for _ in range(3)]
    assert answers == [fake_llm.WRONG, p.rule.answer, fake_llm.WRONG]  # type: ignore[union-attr]
    assert fake_llm.respond(other, state) == "Some chunk."
    assert '"Where is Trailhead Gear headquartered?": 3' in state.read_text(encoding="utf-8")


def test_flaky_counter_persists_across_separate_run_processes(project: Path) -> None:
    """Defect: the flaky counter lives in process memory, so `run` and `diagnose` (separate
    processes) both see call 1 and the planted flaky case is mislabeled. Two real
    `contexteval run` processes must see fail, then pass."""
    assert _cli.main(["init", "--provider", "fake"]) == 0
    jsonl = planted.write_jsonl(project / planted.IMPORT_FILE)
    assert _cli.main(["build-tests", "--import", str(jsonl)]) == 0
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "CONTEXTEVAL_DEMO_LLM": "fake"}
    for _ in range(2):
        subprocess.run(
            [sys.executable, "-m", "contexteval", "run", "contexteval.demo.app:answer", "--yes"],
            cwd=project, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=240, check=True,
        )
    (flaky,) = [p for p in planted.PLANTED if p.expected is Label.FLAKY]
    store = Store.open(project)
    try:
        latest = store.latest_run("baseline")
        assert latest is not None
        runs = [store.traces(latest.id - 1), store.traces(latest.id)]
    finally:
        store.close()
    outcome = [{s.trace.question: s.trace.test.passed for s in run if s.trace.test} for run in runs]
    assert (outcome[0][flaky.case.question], outcome[1][flaky.case.question]) == (False, True)
    del outcome[0][flaky.case.question], outcome[1][flaky.case.question]
    assert outcome[0] == outcome[1] == {p.case.question: p.passes_baseline for p in planted.PLANTED if p is not flaky}

