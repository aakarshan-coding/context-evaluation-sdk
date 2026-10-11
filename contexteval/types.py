"""Core domain types. Pure data, no I/O. Every other module speaks these.

Invariants carried by construction:
- A Trace either answered or crashed, never both (Outcome union).
- A Trace is graded iff it came from a TestCase (one optional field, `test`).
- A Diagnosis is one variant per label; its label and source are derived, never stored twice.
- A trace's active Setting lives on its run, not on the trace (single source of truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping, NewType, NoReturn, Union

CaseId = NewType("CaseId", str)  # sha1(normalized question + evidence), stable across rebuilds
TraceId = NewType("TraceId", int)
RunId = NewType("RunId", int)
ExperimentId = NewType("ExperimentId", int)

# (retrieval index, delivered chunk index) inside one Trace.
ChunkRef = tuple[int, int]


def never(value: NoReturn) -> NoReturn:
    """Exhaustiveness guard for match statements (typing.assert_never is 3.11+)."""
    raise AssertionError(f"unhandled variant: {value!r}")


# --- test cases -------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """A passage quoted verbatim from a source document (plan item 1).

    Built only by testset.parse_generated (which proves it is a substring of the
    document) or testset.import_tests (user-supplied, trusted as given).
    """

    text: str


@dataclass(frozen=True)
class TestCase:
    __test__ = False  # keep pytest from collecting this class

    id: CaseId
    question: str
    expected_answer: str
    evidence: Evidence | None  # None only for imported cases without evidence -> judge path
    source: str  # document path (synthetic) or import file path (imported)


# --- captured context -------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    text: str
    id: str | None
    score: float | None
    # Index into the native result this chunk was decoded from. None means the
    # chunk was injected by an intervention, so the codec must build a native row.
    source_index: int | None


@dataclass(frozen=True)
class RetrievalEvent:
    source: str  # "chromadb" | "pinecone" | "retriever:<function name>"
    query: str | None  # None when the store was queried by vector only
    params: Mapping[str, Any]  # JSON-safe snapshot of the call's params after overrides
    retrieved: tuple[Chunk, ...]  # what the store returned
    delivered: tuple[Chunk, ...]  # what user code received after the intervention
    latency_ms: float


@dataclass(frozen=True)
class LLMCall:
    provider: Literal["openai", "anthropic"]
    model: str
    prompt: str  # flattened text of system + messages, as sent
    answer: str | None  # None for streamed responses (sync-only parsing in v1)
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float


@dataclass(frozen=True)
class Answered:
    text: str


@dataclass(frozen=True)
class Crashed:
    error: str  # "ExceptionType: message"


Outcome = Union[Answered, Crashed]


@dataclass(frozen=True)
class Metrics:
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float | None  # None when a model is missing from the price table


@dataclass(frozen=True)
class TestOutcome:
    __test__ = False

    case_id: CaseId
    passed: bool
    method: Literal["substring", "llm", "crash"]


@dataclass(frozen=True)
class Trace:
    """One execution of the user's answer(question)."""

    question: str
    outcome: Outcome
    retrievals: tuple[RetrievalEvent, ...]
    llm_calls: tuple[LLMCall, ...]
    metrics: Metrics
    test: TestOutcome | None  # None for production traces and ungraded replays
    # Delivered chunks whose text appears in prompt_text. Computed once by the capture layer
    # when the trace is finalized (_capture.finalize_trace), so this module never imports _text.
    in_prompt: tuple[ChunkRef, ...]

    @property
    def prompt_text(self) -> str:
        """All prompts sent to an LLM during this run, joined. Gold checks read this."""
        return "\n".join(c.prompt for c in self.llm_calls)


# --- diagnosis --------------------------------------------------------------


class Label(Enum):
    MISSING = "missing"
    IRRELEVANT = "irrelevant"
    IGNORED = "ignored"
    NOT_CONTEXT = "not_context"
    FLAKY = "flaky"


class JudgeVerdict(Enum):
    MISSING = "missing"
    IRRELEVANT = "irrelevant"
    IGNORED = "ignored"
    NOT_CONTEXT = "not_context"
    NO_FAILURE = "no_failure"  # unlabeled production traffic may simply be fine


class ReplayStep(Enum):
    BASELINE_RERUN = "baseline_rerun"
    INJECT_EVIDENCE = "inject_evidence"
    GOLD_TO_TOP = "gold_to_top"
    ONLY_GOLD = "only_gold"


@dataclass(frozen=True)
class ReplayOutcome:
    step: ReplayStep
    passed: bool
    answer: str | None  # None when the replay crashed


Replays = tuple[ReplayOutcome, ...]


@dataclass(frozen=True)
class Missing:
    cause: Literal["not_retrieved", "dropped"]
    confirmed: bool  # injecting the evidence as the first chunk made it pass
    replays: Replays


@dataclass(frozen=True)
class Irrelevant:
    replays: Replays
    removed: tuple[ChunkRef, ...]  # the non-gold chunks whose removal made it pass


@dataclass(frozen=True)
class Ignored:
    replays: Replays
    gold_at: tuple[ChunkRef, ...]  # where the gold sat before it was moved to the top


@dataclass(frozen=True)
class NotContext:
    replays: Replays


@dataclass(frozen=True)
class Flaky:
    replays: Replays  # exactly the baseline rerun that passed


@dataclass(frozen=True)
class Judged:
    verdict: JudgeVerdict
    rationale: str


Diagnosis = Union[Missing, Irrelevant, Ignored, NotContext, Flaky, Judged]


def label_of(d: Diagnosis) -> Label | JudgeVerdict:
    match d:
        case Missing():
            return Label.MISSING
        case Irrelevant():
            return Label.IRRELEVANT
        case Ignored():
            return Label.IGNORED
        case NotContext():
            return Label.NOT_CONTEXT
        case Flaky():
            return Label.FLAKY
        case Judged(verdict=v):
            return v
        case _:
            never(d)


def source_of(d: Diagnosis) -> Literal["ground_truth", "judge"]:
    return "judge" if isinstance(d, Judged) else "ground_truth"


def _positions(refs: tuple[ChunkRef, ...]) -> str:
    """(0, 2), (0, 3), (1, 0) -> "chunks 2, 3 of retrieval 0; chunk 0 of retrieval 1"."""
    by_retrieval: dict[int, list[int]] = {}
    for r, c in refs:
        by_retrieval.setdefault(r, []).append(c)
    parts = []
    for r, cs in sorted(by_retrieval.items()):
        noun = "chunk" if len(cs) == 1 else "chunks"
        parts.append(f"{noun} {', '.join(str(c) for c in sorted(cs))} of retrieval {r}")
    return "; ".join(parts) if parts else "no chunks"


def explain(d: Diagnosis) -> str:
    """Human evidence line, e.g. "keeping only gold chunks fixed it"."""
    match d:
        case Missing(cause="not_retrieved", confirmed=ok):
            fix = "fixed it" if ok else "did not fix it"
            return f"the gold chunk was never retrieved; injecting the evidence {fix}"
        case Missing(confirmed=ok):
            fix = "fixed it" if ok else "did not fix it"
            return f"the gold chunk was retrieved but dropped before the prompt; injecting the evidence {fix}"
        case Irrelevant(removed=removed):
            return f"keeping only the gold chunks fixed it; removed {_positions(removed)}"
        case Ignored(gold_at=at):
            return f"moving the gold chunk to the top fixed it; gold was at {_positions(at)}"
        case NotContext():
            return "neither moving gold to the top nor keeping only gold fixed it"
        case Flaky():
            return "the unchanged baseline passed on rerun"
        case Judged(verdict=v, rationale=why):
            return f"judge: {v.value}; {why}"
        case _:
            never(d)


# --- experiments ------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentRow:
    """One grid row. Always derived from stored traces by Store.experiment_table."""

    setting_name: str
    cases: int
    passed: int
    mean_input_tokens: float
    mean_latency_ms: float
    total_cost_usd: float | None

    @property
    def accuracy(self) -> float:
        return self.passed / self.cases if self.cases else 0.0
