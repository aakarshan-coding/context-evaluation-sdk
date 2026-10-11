"""Scriptable fake app LLM. Its answer is a pure function of the prompt it sees (plus one
persisted counter for the flaky rule), so every planted label is deterministic.

Each rule states which context condition makes the model answer correctly. The diagnoser's
replays change exactly those conditions, so the expected label follows from the rule:

    rule               correct iff                         baseline   expected label
    NeedsEvidence      evidence anywhere in the chunks     gold absent missing (not_retrieved | dropped)
    NeedsEvidenceFirst evidence in the first chunk         gold 2nd+   ignored
    NeedsNoDistractor  evidence present, no marker chunk   distractor  irrelevant
    AlwaysWrong        never                               any         not_context
    FlipFlop           odd call count fails, even passes   1st call    flaky

Unscripted questions (synthetic cases from build-tests) use NeedsEvidence with the whole
context echoed back, so the grader's substring fast path passes iff the answer was in context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from ..types import Evidence


@dataclass(frozen=True)
class NeedsEvidence:
    evidence: Evidence
    answer: str


@dataclass(frozen=True)
class NeedsEvidenceFirst:
    evidence: Evidence
    answer: str


@dataclass(frozen=True)
class NeedsNoDistractor:
    evidence: Evidence
    answer: str
    distractor_marker: str


@dataclass(frozen=True)
class AlwaysWrong:
    pass


@dataclass(frozen=True)
class FlipFlop:
    answer: str


Rule = Union[NeedsEvidence, NeedsEvidenceFirst, NeedsNoDistractor, AlwaysWrong, FlipFlop]

WRONG = "I don't know."


def respond(prompt: str, state_file: Path) -> str:
    """parse_prompt -> look up planted.script()[question] -> evaluate rule (match, exhaustive).

    FlipFlop increments a per-question counter in `state_file` (JSON). The counter must persist
    because `run` and `diagnose` are separate processes: run is call 1 (fail), the diagnoser's
    baseline rerun is call 2 (pass). Re-running the workflow keeps the odd/even alternation.
    """
    raise NotImplementedError


def transport(state_file: Path | None = None) -> Any:
    """httpx.MockTransport answering POST /chat/completions with an OpenAI-shaped JSON body
    (content from respond, usage from text.estimate_tokens). Default state file:
    .contexteval/demo_fake_state.json in the cwd project.
    """
    raise NotImplementedError
