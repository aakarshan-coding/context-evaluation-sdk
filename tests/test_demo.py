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
from contexteval._diagnose import GoldLocation, locate_gold  # noqa: E402
from contexteval._intervention import NO_INTERVENTION, GoldToTop, InjectEvidence, Intervention, OnlyGold  # noqa: E402
from contexteval._llm import FakeProvider  # noqa: E402
from contexteval._runner import make_grader, run_case  # noqa: E402
from contexteval._store import Store  # noqa: E402
from contexteval.demo import app as demo_app  # noqa: E402
from contexteval.demo import fake_llm, planted  # noqa: E402
from contexteval.demo.fake_llm import NeedsEvidenceFirst, NeedsNoDistractor  # noqa: E402
from contexteval.types import Chunking, Crashed, Label  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
K = demo_app.TOP_K


def _ranks(p: planted.Planted) -> tuple[list[int], list[str]]:
    """1-based ranks of chunks holding the evidence, and every chunk in rank order."""
    col = demo_app.collection()
    docs = col.query(query_texts=[p.case.question], n_results=col.count())["documents"][0]
    assert p.case.evidence is not None
    return [i + 1 for i, d in enumerate(docs) if text.contains_evidence(d, p.case.evidence)], docs


@pytest.mark.parametrize("p", planted.PLANTED, ids=lambda p: (p.expected.value + (f"-{p.cause}" if p.cause else "") + ("-split" if p.split else "")) if p.expected else "healthy")
def test_planted_gold_ranks(p: planted.Planted, project: Path) -> None:
    """Defect: a corpus or question edit moves a gold chunk, so a planted case stops exercising
    its failure (for example the not_retrieved gold entering the top 4, or leaving the top 10,
    which breaks the P4 top_k=10 experiment; or the split case's halves no longer both in the
    top 4 and next to each other, so the join rule never fires)."""
    gold, docs = _ranks(p)
    if p.split:
        assert p.case.evidence is not None
        assert gold == [], f"no single chunk should hold the split evidence, found ranks {gold}"
        halves = [i + 1 for i, d in enumerate(docs) if text.lexical_score(d, p.case.evidence.text) > 0.9]
        assert len(halves) == 2 and halves[1] == halves[0] + 1 and halves[1] <= K, halves
        assert text.gold_flags(docs[:K], p.case.evidence) == tuple(i + 1 in halves for i in range(K))
        assert text.gold_split(docs[:K], p.case.evidence)
        return
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


def test_demo_build_keeps_its_own_chunking_by_default_and_a_larger_chunk_size_fixes_the_split_case(project: Path, hooks: None) -> None:
    """v2 chunking on the demo. Defects: the default chunking no longer one paragraph per chunk
    (every planted rank would shift), build() returning an app on the app's own index, the split
    case labeled without the split signal, or no chunking that packs the two fitting paragraphs
    together and so makes the split case pass. The chunk_size tried is what the sweep proposes:
    twice the longest chunk at baseline. The other planted labels are unchanged at baseline."""
    (p,) = [p for p in planted.PLANTED if p.split]
    assert p.case.evidence is not None
    for path in sorted(demo_app.CORPUS_DIR.glob("*.md")):
        paragraphs = [x.strip() for x in path.read_text(encoding="utf-8").split("\n\n") if x.strip() and not x.startswith("#")]
        assert [demo_app.corpus_chunks()[f"{path.name}#{n}"] for n in range(len(paragraphs))] == paragraphs
    assert not _passes(p, NO_INTERVENTION)
    trace = run_case(demo_app.answer, p.case, make_grader(FakeProvider()))
    gold = locate_gold(p.case.evidence, trace)
    assert gold.location is GoldLocation.IN_PROMPT and gold.split is True and len(gold.gold) == 2
    longest = max(len(c) for c in demo_app.corpus_chunks().values())
    packed = Chunking(chunk_size=2 * longest)
    assert demo_app.corpus_chunks(packed)["fitting.md#0"].startswith(p.case.evidence.text.split(". ")[0])
    assert text.contains_evidence(demo_app.corpus_chunks(packed)["fitting.md#0"], p.case.evidence)
    built = demo_app.build(packed)
    assert run_case(built, p.case, make_grader(FakeProvider())).test.passed is True  # type: ignore[union-attr]
    titled = demo_app.corpus_chunks(Chunking(title=True))
    assert titled["fitting.md#0"] == "Boot fitting\n\n" + demo_app.corpus_chunks()["fitting.md#0"]


def test_not_retrieved_case_passes_at_top_k_10(project: Path, hooks: None) -> None:
    """Defect: the planted missing case cannot be fixed by a larger top_k, which P4's acceptance
    (DoD item 3) depends on."""
    (p,) = [p for p in planted.PLANTED if p.cause == "not_retrieved"]
    assert not _passes(p, NO_INTERVENTION)
    assert _passes(p, Intervention(top_k=10))


def test_flaky_rule_fails_only_the_first_call_and_persists_in_state_file(tmp_path: Path) -> None:
    """Defects: the flaky counter kept in memory, or shared across questions; a later call
    failing again, which makes experiment rows after the first run noisy."""
    (p,) = [p for p in planted.PLANTED if p.expected is Label.FLAKY]
    state = tmp_path / "state.json"
    prompt = demo_app.build_prompt(p.case.question, ["Trailhead Gear is headquartered in Boulder, Colorado."])
    other = demo_app.build_prompt("unscripted question?", ["Some chunk."])
    answers = [fake_llm.respond(prompt, state) for _ in range(4)]
    assert answers == [fake_llm.WRONG] + [p.rule.answer] * 3  # type: ignore[union-attr]
    assert fake_llm.respond(other, state) == "Some chunk."
    assert '"Where is Trailhead Gear headquartered?": 4' in state.read_text(encoding="utf-8")


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



def _spawn(code: str, *args: str, cwd: Path | None = None) -> subprocess.Popen[str]:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "CONTEXTEVAL_DEMO_LLM": "fake"}
    return subprocess.Popen(
        [sys.executable, "-c", code, *args], cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def test_flaky_counter_survives_concurrent_processes(tmp_path: Path) -> None:
    """Defect: an unlocked read-modify-write of the state file. Before the lock, 4 processes x
    200 bumps ended near 700, with PermissionError from os.replace on Windows."""
    state = tmp_path / "state.json"
    code = "import sys; from pathlib import Path; from contexteval.demo import fake_llm\nfor _ in range(100): fake_llm._bump(Path(sys.argv[1]), 'q')"
    procs = [_spawn(code, str(state)) for _ in range(4)]
    errors = [p.communicate(timeout=240)[1] for p in procs]
    assert [p.returncode for p in procs] == [0] * 4, errors
    assert '"q": 400' in state.read_text(encoding="utf-8")


def test_demo_collection_opens_concurrently_on_a_fresh_directory(tmp_path: Path) -> None:
    """Defect: every process sets up and writes the shared chroma directory. Before the fix, 8
    processes opening a fresh directory at once failed about one round in four with "table
    collections already exists" or "database is locked"."""
    code = (
        "import sys; from contexteval.demo import app\n"
        "print(len(app._collection_at(sys.argv[1]).query(query_texts=['warranty'], n_results=app.TOP_K)['documents'][0]))"
    )
    for round_ in range(2):
        path = str(tmp_path / f"chroma{round_}")
        procs = [_spawn(code, path) for _ in range(8)]
        outs = [p.communicate(timeout=240) for p in procs]
        assert [(p.returncode, out.strip()) for p, (out, _) in zip(procs, outs)] == [(0, str(K))] * 8, [err[-500:] for _, err in outs]


def test_concurrent_run_processes_share_one_project(project: Path) -> None:
    """Defects: concurrent `contexteval run` processes on one fresh project crash cases on the
    shared chroma directory or lose a flaky-counter update. Each run must finish with no crash,
    the counter must count every run, and only one run sees the flaky case's first call."""
    assert _cli.main(["init", "--provider", "fake"]) == 0
    jsonl = planted.write_jsonl(project / planted.IMPORT_FILE)
    assert _cli.main(["build-tests", "--import", str(jsonl)]) == 0
    code = "from contexteval._cli import main; raise SystemExit(main(['run', 'contexteval.demo.app:answer', '--yes']))"
    procs = [_spawn(code, cwd=project) for _ in range(3)]
    outs = [p.communicate(timeout=240) for p in procs]
    assert [p.returncode for p in procs] == [0] * 3, [err[-500:] for _, err in outs]
    (flaky,) = [p for p in planted.PLANTED if p.expected is Label.FLAKY]
    store = Store.open(project)
    try:
        runs = [store.traces(r.run.id) for r in store.runs()]
    finally:
        store.close()
    assert len(runs) == 3
    assert not [s.trace.outcome for run in runs for s in run if isinstance(s.trace.outcome, Crashed)]
    flaky_passed = sorted(s.trace.test.passed for run in runs for s in run if s.trace.question == flaky.case.question)  # type: ignore[union-attr]
    assert flaky_passed == [False, True, True]
    assert f'"{flaky.case.question}": 3' in fake_llm.default_state_file().read_text(encoding="utf-8")
