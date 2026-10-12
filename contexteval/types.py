"""Core domain types. Pure data, no I/O. Every other module speaks these.

Invariants carried by construction:
- A Trace either answered or crashed, never both (Outcome union).
- A Trace is graded iff it came from a TestCase (one optional field, `test`).
- A Diagnosis is one variant per label; its label and source are derived, never stored twice.
- A trace's active Setting lives on its run, not on the trace (single source of truth).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
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
    source: str  # "chromadb" | "pinecone" | "langchain" | "llamaindex" | "retriever:<function name>"
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


# Every sample drawn, in the order drawn. A step appears once per sample, so a v1 diagnosis
# (one sample per step) is a v1.1 diagnosis whose every step was decided by one sample.
Replays = tuple[ReplayOutcome, ...]


@dataclass(frozen=True)
class StepTally:
    """How one step's samples went. The step passed when a strict majority of its samples
    passed. The diagnoser stops sampling once a majority of N is sure, so a 2-of-2 tally is a
    unanimous step at N=3, not a tie."""

    step: ReplayStep
    passed: int
    total: int

    @property
    def majority(self) -> bool:
        return self.passed > self.total - self.passed

    @property
    def count(self) -> str:
        return f"{self.passed}/{self.total}"


def tallies(replays: Replays) -> tuple[StepTally, ...]:
    """Per-step pass counts, in the order the steps were first sampled."""
    order: list[ReplayStep] = []
    passed: dict[ReplayStep, int] = {}
    total: dict[ReplayStep, int] = {}
    for r in replays:
        if r.step not in total:
            order.append(r.step)
            passed[r.step] = total[r.step] = 0
        total[r.step] += 1
        passed[r.step] += r.passed
    return tuple(StepTally(s, passed[s], total[s]) for s in order)


# split_gold on the four variants below: the evidence was found only by joining two adjacent
# chunks (text.gold_split), so the app's chunker cut it in two. It is the one signal that points
# at chunking rather than retrieval, and the sweep reads it. Flaky has no gold to speak of.


@dataclass(frozen=True)
class Missing:
    cause: Literal["not_retrieved", "dropped"]
    confirmed: bool  # injecting the evidence as the first chunk made it pass
    replays: Replays
    split_gold: bool = False  # possible for `dropped` only: both halves were retrieved and then cut


@dataclass(frozen=True)
class Irrelevant:
    replays: Replays
    removed: tuple[ChunkRef, ...]  # the non-gold chunks whose removal made it pass
    split_gold: bool = False


@dataclass(frozen=True)
class Ignored:
    replays: Replays
    gold_at: tuple[ChunkRef, ...]  # where the gold sat before it was moved to the top
    split_gold: bool = False


@dataclass(frozen=True)
class NotContext:
    replays: Replays
    split_gold: bool = False


@dataclass(frozen=True)
class Flaky:
    replays: Replays  # only baseline reruns, a majority of which passed


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
    """Stored refs are 0-based; people count from 1 (the real-bot audit read "chunk 5" as the
    fifth). (0, 2), (0, 3), (1, 0) -> "chunks 3, 4 of retrieval 1; chunk 1 of retrieval 2"."""
    by_retrieval: dict[int, list[int]] = {}
    for r, c in refs:
        by_retrieval.setdefault(r, []).append(c)
    parts = []
    for r, cs in sorted(by_retrieval.items()):
        noun = "chunk" if len(cs) == 1 else "chunks"
        parts.append(f"{noun} {', '.join(str(c + 1) for c in sorted(cs))} of retrieval {r + 1}")
    return "; ".join(parts) if parts else "no chunks"


LADDER_STEPS = (ReplayStep.GOLD_TO_TOP, ReplayStep.ONLY_GOLD)
# Ladder step -> (what it does, why the diagnoser skips it when it would change nothing).
_LADDER_TEXT = {
    ReplayStep.GOLD_TO_TOP: ("moving gold to the top", "gold was already at the top"),
    ReplayStep.ONLY_GOLD: ("keeping only gold", "gold was the only context"),
}


def skipped_steps(d: Diagnosis) -> tuple[ReplayStep, ...]:
    """Ladder steps the diagnoser skipped because they would not change the context. The ladder
    runs every step before the deciding one, so a missing step means a skip."""
    match d:
        case Irrelevant(replays=replays):
            before = LADDER_STEPS[:1]
        case NotContext(replays=replays):
            before = LADDER_STEPS
        case _:
            return ()
    ran = {r.step for r in replays}
    return tuple(s for s in before if s not in ran)


def _skip_clause(steps: tuple[ReplayStep, ...]) -> str:
    return "; ".join(f"{_LADDER_TEXT[s][0]} was skipped because {_LADDER_TEXT[s][1]}" for s in steps)


def _count(d: Diagnosis, step: ReplayStep) -> str:
    """The step's count in parentheses, as in "(2/3)": samples that passed over samples drawn."""
    assert not isinstance(d, Judged)
    (tally,) = [t for t in tallies(d.replays) if t.step is step]
    return f"({tally.count})"


SPLIT_GOLD_TEXT = "the evidence was split across two adjacent chunks"


def split_gold_of(d: Diagnosis) -> bool:
    """Whether the diagnosis carries the split signal; Flaky and Judged never do."""
    return isinstance(d, (Missing, Irrelevant, Ignored, NotContext)) and d.split_gold


def explain(d: Diagnosis) -> str:
    """Human evidence line, e.g. "keeping only the gold chunks fixed it (2/3)". Each step the
    diagnoser decided on carries its pass count, so a 2/3 reads as less sure than a 2/2. A split
    gold ends the line with SPLIT_GOLD_TEXT."""
    line = _explain_steps(d)
    return f"{line}; {SPLIT_GOLD_TEXT}" if split_gold_of(d) else line


def _explain_steps(d: Diagnosis) -> str:
    match d:
        case Missing(cause="not_retrieved", confirmed=ok):
            fix = "fixed it" if ok else "did not fix it"
            return f"the gold chunk was never retrieved; injecting the evidence {fix} {_count(d, ReplayStep.INJECT_EVIDENCE)}"
        case Missing(confirmed=ok):
            fix = "fixed it" if ok else "did not fix it"
            return f"the gold chunk was retrieved but dropped before the prompt; injecting the evidence {fix} {_count(d, ReplayStep.INJECT_EVIDENCE)}"
        case Irrelevant(removed=removed):
            skipped = skipped_steps(d)
            line = f"keeping only the gold chunks fixed it {_count(d, ReplayStep.ONLY_GOLD)}; removed {_positions(removed)}"
            return f"{line}; {_skip_clause(skipped)}" if skipped else line
        case Ignored(gold_at=at):
            return f"moving the gold chunk to the top fixed it {_count(d, ReplayStep.GOLD_TO_TOP)}; gold was at {_positions(at)}"
        case NotContext():
            skipped = skipped_steps(d)
            tried = [f"{_LADDER_TEXT[s][0]} {_count(d, s)}" for s in LADDER_STEPS if s not in skipped]
            if not skipped:
                return f"neither {tried[0]} nor {tried[1]} fixed it"
            if not tried:
                return "gold was already the only context at the top, so moving it to the top and keeping only gold were skipped"
            return f"{' and '.join(tried)} did not fix it; {_skip_clause(skipped)}"
        case Flaky():
            return f"the unchanged baseline passed on rerun {_count(d, ReplayStep.BASELINE_RERUN)}"
        case Judged(verdict=v, rationale=why):
            return f"judge: {v.value}; {why}"
        case _:
            never(d)


# --- experiments ------------------------------------------------------------


def knob_text(value: object) -> str:
    """How a knob value prints in a setting name: `on`/`off` for a flag, an Enum's value, else str."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


@dataclass(frozen=True)
class Chunking:
    """How an app splits its documents when it builds its index. These are index-time knobs, a
    separate family from the retrieval-time knobs on `Setting`: contexteval cannot apply them
    itself, so an app that wants to try them offers `build(chunking) -> answer` next to
    `answer`. None means the app's own value, so `Chunking()` is the app's own chunking.
    """

    chunk_size: int | None = None  # characters per chunk; whole paragraphs are packed up to it
    overlap: int | None = None  # characters carried from the end of one chunk into the next
    title: bool | None = None  # prefix every chunk with its document's title

    def __post_init__(self) -> None:
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError(f"chunk_size must be a positive integer, got {self.chunk_size}")
        if self.overlap is not None and self.overlap < 0:
            raise ValueError(f"overlap must be 0 or more, got {self.overlap}")
        if self.chunk_size is not None and self.overlap is not None and self.overlap >= self.chunk_size:
            raise ValueError(f"overlap ({self.overlap}) must be smaller than chunk_size ({self.chunk_size})")

    @property
    def is_default(self) -> bool:
        return self == Chunking()

    @property
    def is_complete(self) -> bool:
        return self.chunk_size is not None and self.overlap is not None and self.title is not None

    def with_defaults(self, *, chunk_size: int, overlap: int = 0, title: bool = False) -> Chunking:
        """Fill every unset knob with the app's own value, so chunk_text can run. A knob the
        experiment set is kept."""
        return Chunking(
            chunk_size=self.chunk_size if self.chunk_size is not None else chunk_size,
            overlap=self.overlap if self.overlap is not None else overlap,
            title=self.title if self.title is not None else title,
        )

    @property
    def name(self) -> str:
        """The set knobs as "chunk_size=800 title=on", or "the app's own chunking"."""
        parts = [f"{f.name}={knob_text(getattr(self, f.name))}" for f in fields(self) if getattr(self, f.name) is not None]
        return " ".join(parts) or "the app's own chunking"


@dataclass(frozen=True)
class ExperimentRow:
    """One grid row. Always derived from stored traces by Store.experiment_table."""

    run_id: RunId  # the run the row aggregates; a sweep names its recommended row by it
    setting_name: str
    cases: int
    passed: int
    mean_input_tokens: float
    mean_latency_ms: float
    total_cost_usd: float | None

    @property
    def accuracy(self) -> float:
        return self.passed / self.cases if self.cases else 0.0
