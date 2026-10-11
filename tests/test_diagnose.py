"""P3 units around the table: locate_gold on hand-built traces, the judge's reply boundary, and
diagnose_run over a store (judge path, production runs, crashed traces, idempotent reruns)."""

from __future__ import annotations

import json

import pytest

from contexteval import _diagnose
from contexteval import _text as text
from contexteval._diagnose import GoldLocation, JudgeReplyError, diagnose_run, judge, locate_gold, parse_verdict
from contexteval._intervention import BASELINE, GoldToTop, OnlyGold, apply_transform
from contexteval._llm import FakeProvider, LLMRequest, Task
from contexteval._runner import make_grader
from contexteval._store import Store
from contexteval.types import (
    Answered,
    CaseId,
    Chunk,
    Crashed,
    Evidence,
    Judged,
    JudgeVerdict,
    LLMCall,
    Metrics,
    RetrievalEvent,
    TestCase,
    TestOutcome,
    Trace,
    label_of,
)

GOLD = Evidence("Phone support is open from 7 am to 9 pm Mountain Time, every day of the year.")
OFF_TOPIC = "Orders over 75 dollars ship free with standard shipping."
CASE = TestCase(CaseId("c1"), "When is phone support open?", "7 am to 9 pm", GOLD, "planted")
NO_EVIDENCE = TestCase(CaseId("c2"), "Who founded it?", "two climbing guides", None, "imported.jsonl")


def _trace(delivered: list[str], in_prompt: list[str], *, answer: str = "I don't know.", test: TestOutcome | None = None, crashed: bool = False) -> Trace:
    chunks = tuple(Chunk(t, f"id{i}", -0.1 * i, i) for i, t in enumerate(delivered))
    event = RetrievalEvent("chromadb", CASE.question, {"n_results": len(chunks)}, chunks, chunks, 2.0)
    prompt = "Context:\n" + "\n".join(in_prompt) + f"\n\nQuestion: {CASE.question}"
    call = LLMCall("openai", "gpt-4o-mini", prompt, answer, 50, 5, 30.0)
    return Trace(
        question=CASE.question,
        outcome=Crashed("RuntimeError: boom") if crashed else Answered(answer),
        retrievals=(event,),
        llm_calls=() if crashed else (call,),
        metrics=Metrics(50, 5, 32.0, None),
        test=test,
        in_prompt=tuple((0, i) for i, t in enumerate(delivered) if t in in_prompt),
    )


def test_locate_gold_in_prompt_reports_gold_and_other_chunk_positions() -> None:
    """Defect: IN_PROMPT decided from the chunks instead of the prompt, or the chunk refs that
    Ignored.gold_at / Irrelevant.removed display (graft 1) pointing at the wrong positions."""
    delivered = [OFF_TOPIC, "Returns take 45 days.", GOLD.text]
    gold = locate_gold(GOLD, _trace(delivered, delivered))
    assert gold.location is GoldLocation.IN_PROMPT
    assert gold.gold == ((0, 2),)
    assert gold.other == ((0, 0), (0, 1))


def test_locate_gold_dropped_versus_not_retrieved() -> None:
    """Defect: a gold chunk the app filtered out before the prompt counted as not_retrieved,
    which would send the user to tune top_k instead of their filter."""
    assert locate_gold(GOLD, _trace([OFF_TOPIC, GOLD.text + " [ARCHIVED]"], [OFF_TOPIC])).location is GoldLocation.DROPPED
    assert locate_gold(GOLD, _trace([OFF_TOPIC], [OFF_TOPIC])).location is GoldLocation.NOT_RETRIEVED


def _event_trace(question: str, retrieved: list[str], delivered: list[str], prompt_chunks: list[str]) -> Trace:
    """A trace whose prompt is the given chunks plus the question, with in_prompt computed by the
    real capture rule (as finalize_trace does)."""
    def chunks(texts: list[str]) -> tuple[Chunk, ...]:
        return tuple(Chunk(t, f"id{i}", None, i) for i, t in enumerate(texts))

    event = RetrievalEvent("chromadb", question, {}, chunks(retrieved), chunks(delivered), 1.0)
    prompt = "Context:\n" + "\n".join(prompt_chunks) + f"\n\nQuestion: {question}"
    return Trace(
        question=question,
        outcome=Answered("I don't know."),
        retrievals=(event,),
        llm_calls=(LLMCall("openai", "gpt-4o-mini", prompt, "I don't know.", 1, 1, 1.0),),
        metrics=Metrics(1, 1, 1.0, None),
        test=None,
        in_prompt=text.in_prompt_pairs((event,), prompt),
    )


def test_locate_gold_never_counts_the_question_as_gold() -> None:
    """Review Y2a. A generated question repeats most of the evidence words, and no chunk holds the
    evidence. Defect: gold decided over the whole prompt, which includes the question, reports
    IN_PROMPT with no gold chunk, so the ladder replays a case whose gold was never retrieved."""
    question = "According to the document, what completes: Phone support is open from 7 am to 9 pm Mountain Time, every day of ...?"
    trace = _event_trace(question, [OFF_TOPIC], [OFF_TOPIC], [OFF_TOPIC])
    gold = locate_gold(GOLD, trace)
    assert gold.location is GoldLocation.NOT_RETRIEVED
    assert gold.gold == ()


SPLIT_A = "Hours. Phone support is open from 7 am to 9 pm"
SPLIT_B = "Mountain Time, every day of the year. Email replies take two days."


def test_locate_gold_joins_adjacent_chunks_that_split_the_evidence() -> None:
    """Review Y2b. The user's chunker cut the evidence about in half, so neither chunk alone holds
    80% of it. Defect: gold=() while the prompt holds the evidence, so GoldToTop is a no-op,
    OnlyGold drops every chunk, and the case is mislabeled not_context. Both halves are gold, and
    the replay transforms move and keep exactly those two chunks, in order."""
    assert not text.contains_evidence(SPLIT_A, GOLD) and not text.contains_evidence(SPLIT_B, GOLD)
    delivered = [OFF_TOPIC, SPLIT_A, SPLIT_B]
    gold = locate_gold(GOLD, _event_trace(CASE.question, delivered, delivered, delivered))
    assert gold.location is GoldLocation.IN_PROMPT
    assert gold.gold == ((0, 1), (0, 2))
    assert gold.other == ((0, 0),)

    chunks = tuple(Chunk(t, str(i), None, i) for i, t in enumerate(delivered))
    assert [c.text for c in apply_transform(GoldToTop(GOLD), chunks, "q")] == [SPLIT_A, SPLIT_B, OFF_TOPIC]
    assert [c.text for c in apply_transform(OnlyGold(GOLD), chunks, "q")] == [SPLIT_A, SPLIT_B]


def test_locate_gold_does_not_join_chunks_that_are_not_adjacent() -> None:
    """Defect: halves joined across an unrelated chunk, so two partial matches anywhere in the
    list count as gold."""
    delivered = [SPLIT_A, OFF_TOPIC, SPLIT_B]
    assert locate_gold(GOLD, _event_trace(CASE.question, delivered, delivered, delivered)).location is GoldLocation.NOT_RETRIEVED


def test_locate_gold_reports_dropped_when_a_setting_removed_retrieved_gold() -> None:
    """Review Y2c. On an experiment run (max_chunks=1) the store returned the gold chunk but the
    setting cut it before user code saw it. Defect: only `delivered` is read, so the case reports
    not_retrieved and sends the user to raise top_k when their own setting dropped the gold."""
    trace = _event_trace(CASE.question, [OFF_TOPIC, GOLD.text], [OFF_TOPIC], [OFF_TOPIC])
    assert locate_gold(GOLD, trace).location is GoldLocation.DROPPED


@pytest.mark.parametrize(
    "reply",
    [
        '{"verdict": "ignored", "rationale": "The answer contradicts chunk 2."}',
        '```json\n{"verdict": "IGNORED", "rationale": "The answer contradicts chunk 2."}\n```',
        'Sure. {"verdict": "ignored", "rationale": "The answer contradicts chunk 2."} Hope that helps.',
    ],
)
def test_parse_verdict_accepts_a_verdict_in_fences_or_prose(reply: str) -> None:
    """Defect: a model that wraps its JSON in a code fence or a sentence fails every judge call."""
    assert parse_verdict(reply) == Judged(JudgeVerdict.IGNORED, "The answer contradicts chunk 2.")


@pytest.mark.parametrize("reply", ["I think it was ignored.", '{"verdict": "maybe"}', "[]", ""])
def test_parse_verdict_rejects_anything_without_a_known_verdict(reply: str) -> None:
    """Defect: an unparseable reply silently becomes some default label (a guess stored as a verdict)."""
    with pytest.raises(JudgeReplyError):
        parse_verdict(reply)


def test_judge_sends_the_trace_and_returns_the_verdict() -> None:
    """Defect: the judge prompt omits the chunks, the prompt, or the answer, so the model judges
    blind; or the verdict is not taken from the reply."""
    provider = FakeProvider(handlers={Task.JUDGE: lambda req: json.dumps({"verdict": "no_failure", "rationale": "grounded"})})
    trace = _trace([OFF_TOPIC, GOLD.text], [OFF_TOPIC, GOLD.text], answer="7 am to 9 pm")
    assert judge(trace, provider) == Judged(JudgeVerdict.NO_FAILURE, "grounded")
    (req,) = provider.requests
    assert req.task is Task.JUDGE
    assert req.inputs["question"] == CASE.question
    assert GOLD.text in req.inputs["chunks"] and OFF_TOPIC in req.inputs["chunks"]
    assert req.inputs["prompt"] == trace.prompt_text and req.inputs["answer"] == "7 am to 9 pm"
    assert GOLD.text in req.user and "7 am to 9 pm" in req.user


def test_diagnose_run_judges_a_production_run_and_skips_it_the_second_time(store: Store) -> None:
    """Defect: production traces (no case, so no `passed`) filtered out as not-failed and never
    judged; a no_failure verdict rejected; or a second diagnose re-spending judge calls (graft 2)."""
    run = store.new_run("production", None, BASELINE)
    store.add_trace(run, _trace([GOLD.text], [GOLD.text], answer="7 am to 9 pm"))
    store.add_trace(run, _trace([OFF_TOPIC], [OFF_TOPIC]))
    verdicts = iter(["no_failure", "missing"])
    provider = FakeProvider(handlers={Task.JUDGE: lambda req: json.dumps({"verdict": next(verdicts), "rationale": "r"})})

    report = diagnose_run(store, run, None, make_grader(provider), provider)
    assert (report.diagnosed, report.skipped_existing, report.skipped_crashed) == (2, 0, 0)
    labels = [(label_of(s.diagnosis), s.diagnosis.rationale) for s in store.traces(run) if s.diagnosis]
    assert labels == [(JudgeVerdict.NO_FAILURE, "r"), (JudgeVerdict.MISSING, "r")]
    assert len(provider.requests) == 2

    again = diagnose_run(store, run, None, make_grader(provider), provider)
    assert (again.diagnosed, again.skipped_existing) == (0, 2)
    assert len(provider.requests) == 2


def test_diagnose_run_force_rejudges(store: Store) -> None:
    """Defect: --force ignored, so a bad verdict can never be replaced."""
    run = store.new_run("production", None, BASELINE)
    store.add_trace(run, _trace([OFF_TOPIC], [OFF_TOPIC]))
    provider = FakeProvider(handlers={Task.JUDGE: lambda req: json.dumps({"verdict": "missing", "rationale": "first"})})
    diagnose_run(store, run, None, make_grader(provider), provider)
    provider.handlers[Task.JUDGE] = lambda req: json.dumps({"verdict": "not_context", "rationale": "second"})
    report = diagnose_run(store, run, None, make_grader(provider), provider, force=True)
    assert report.diagnosed == 1
    (stored,) = store.traces(run)
    assert stored.diagnosis == Judged(JudgeVerdict.NOT_CONTEXT, "second")


def test_diagnose_run_judges_cases_without_evidence_and_skips_crashes(store: Store) -> None:
    """Defect: an imported case with no evidence sent down the replay path (there is nothing to
    inject), a crashed trace replayed or judged, or a passing trace diagnosed at all."""
    store.upsert_cases("imported", [CASE, NO_EVIDENCE])
    run = store.new_run("baseline", "app:answer", BASELINE)
    store.add_trace(run, _trace([OFF_TOPIC], [OFF_TOPIC], test=TestOutcome(NO_EVIDENCE.id, False, "llm")))
    store.add_trace(run, _trace([OFF_TOPIC], [OFF_TOPIC], test=TestOutcome(CASE.id, False, "crash"), crashed=True))
    store.add_trace(run, _trace([GOLD.text], [GOLD.text], answer="7 am to 9 pm", test=TestOutcome(CASE.id, True, "substring")))
    provider = FakeProvider(handlers={Task.JUDGE: lambda req: json.dumps({"verdict": "missing", "rationale": "nothing relevant"})})

    def app(question: str) -> str:
        raise AssertionError("no ground-truth case here, so the app must not run")

    report = diagnose_run(store, run, app, make_grader(provider), provider)
    assert (report.diagnosed, report.skipped_existing, report.skipped_crashed) == (1, 0, 1)
    by_case = {s.trace.test.case_id: s.diagnosis for s in store.traces(run) if s.trace.test}
    assert by_case[NO_EVIDENCE.id] == Judged(JudgeVerdict.MISSING, "nothing relevant")
    assert by_case[CASE.id] is None
    assert [r.task for r in provider.requests] == [Task.JUDGE]


def test_diagnose_run_needs_the_app_for_ground_truth(store: Store) -> None:
    """Defect: a ground-truth case with app=None silently judged, hiding a CLI wiring bug."""
    store.upsert_cases("imported", [CASE])
    run = store.new_run("baseline", "app:answer", BASELINE)
    store.add_trace(run, _trace([OFF_TOPIC], [OFF_TOPIC], test=TestOutcome(CASE.id, False, "llm")))
    provider = FakeProvider()
    with pytest.raises(ValueError, match="needs the app"):
        diagnose_run(store, run, None, make_grader(provider), provider)


def test_estimate_counts_judge_calls_and_worst_case_grades(store: Store) -> None:
    """Defect: the diagnose estimate omits the judge calls or prices replay grading at zero, so the
    cost prompt under-reports."""
    store.upsert_cases("imported", [CASE, NO_EVIDENCE])
    run = store.new_run("baseline", "app:answer", BASELINE)
    store.add_trace(run, _trace([OFF_TOPIC], [OFF_TOPIC], test=TestOutcome(NO_EVIDENCE.id, False, "llm")))
    store.add_trace(run, _trace([OFF_TOPIC], [OFF_TOPIC], test=TestOutcome(CASE.id, False, "llm")))
    store.add_trace(run, _trace([OFF_TOPIC], [OFF_TOPIC], test=TestOutcome(CASE.id, False, "crash"), crashed=True))
    todo = _diagnose.pending(store, run)
    assert len(todo) == 3
    est = _diagnose.estimate_diagnosis(FakeProvider(), todo)
    assert est.calls == 1 + _diagnose.MAX_REPLAYS
    assert est.usd == 0.0
