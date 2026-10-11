"""The one list of planted cases. Demo test cases, the fake LLM's script, and the acceptance
tests' expected labels are all derived from PLANTED; nothing is listed twice.

    python -m contexteval.demo.planted [PATH]   # writes planted.jsonl for `build-tests --import`
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .._testset import case_id
from ..types import Evidence, Label, TestCase
from .fake_llm import AlwaysWrong, FlipFlop, NeedsEvidence, NeedsEvidenceFirst, NeedsNoDistractor, Rule

IMPORT_FILE = "planted.jsonl"
SOURCE = "contexteval.demo.planted"


@dataclass(frozen=True)
class Planted:
    case: TestCase
    rule: Rule
    expected: Label | None  # None: a healthy case that passes at baseline
    cause: Literal["not_retrieved", "dropped"] | None = None  # set iff expected is MISSING
    confirmed: bool | None = None  # expected Missing.confirmed

    @property
    def passes_baseline(self) -> bool:
        return self.expected is None


def _case(question: str, answer: str, evidence: str) -> TestCase:
    return TestCase(id=case_id(question, evidence), question=question, expected_answer=answer, evidence=Evidence(evidence), source=SOURCE)


def _planted(question: str, answer: str, evidence: str, rule_of: type, reply: str, **kw: object) -> tuple[TestCase, Rule]:
    case = _case(question, answer, evidence)
    rule = rule_of(evidence=case.evidence, answer=reply, **kw)
    return case, rule


_NOT_RETRIEVED = _planted(
    "How long is the warranty on a sleeping bag?",
    "ten year",
    "Down-filled sleepers carry a ten year guarantee against baffle and seam failure.",
    NeedsEvidence,
    "Sleeping bags have a ten year guarantee against seam failure.",
)
# The evidence leaves out the [ARCHIVED] marker, so the injected chunk survives the app's filter.
_DROPPED = _planted(
    "What is store credit for a returned tent worth?",
    "110 percent",
    "Store credit for a returned tent is issued as a gift card worth 110 percent of the tent's purchase price.",
    NeedsEvidence,
    "A gift card worth 110 percent of the purchase price.",
)
_IGNORED = _planted(
    "How long do warranty repairs take?",
    "three weeks",
    "Repairs under warranty usually take three weeks, and we pay shipping both ways.",
    NeedsEvidenceFirst,
    "Warranty repairs usually take three weeks.",
)
_IRRELEVANT = _planted(
    "Can fuel canisters be shipped by air?",
    "cannot be shipped by air",
    "Fuel canisters cannot be shipped by air and are sold in our stores only.",
    NeedsNoDistractor,
    "No. Fuel canisters cannot be shipped by air; buy them in a store.",
    distractor_marker="[FORUM]",
)
_NOT_CONTEXT = _case(
    "When is phone support open?",
    "7 am to 9 pm",
    "Phone support is open from 7 am to 9 pm Mountain Time, every day of the year.",
)
_FLAKY = _case(
    "Where is Trailhead Gear headquartered?",
    "Boulder, Colorado",
    "Trailhead Gear is headquartered in Boulder, Colorado, and was founded in 1987 by two climbing guides.",
)
_HEALTHY_RETURNS = _planted(
    "Within how many days of delivery can unused items be returned?",
    "45 days",
    "Unused items in their original packaging can be returned within 45 days of delivery for a full refund.",
    NeedsEvidence,
    "You have 45 days from delivery.",
)
_HEALTHY_SHIPPING = _planted(
    "What order total qualifies for free shipping?",
    "75 dollars",
    "Orders over 75 dollars ship free with standard shipping.",
    NeedsEvidence,
    "Orders over 75 dollars ship free.",
)

PLANTED: tuple[Planted, ...] = (
    Planted(*_NOT_RETRIEVED, Label.MISSING, cause="not_retrieved", confirmed=True),
    Planted(*_DROPPED, Label.MISSING, cause="dropped", confirmed=True),
    Planted(*_IGNORED, Label.IGNORED),
    Planted(*_IRRELEVANT, Label.IRRELEVANT),
    Planted(_NOT_CONTEXT, AlwaysWrong(), Label.NOT_CONTEXT),
    Planted(_FLAKY, FlipFlop(answer="Trailhead Gear is based in Boulder, Colorado."), Label.FLAKY),
    Planted(*_HEALTHY_RETURNS, None),
    Planted(*_HEALTHY_SHIPPING, None),
)


def cases() -> tuple[TestCase, ...]:
    return tuple(p.case for p in PLANTED)


def script() -> dict[str, Rule]:
    return {p.case.question: p.rule for p in PLANTED}


def write_jsonl(path: Path) -> Path:
    """The planted cases in `build-tests --import` format."""
    lines = [
        json.dumps({"question": c.question, "expected_answer": c.expected_answer, "evidence": c.evidence.text if c.evidence else None})
        for c in cases()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_jsonl(Path(sys.argv[1] if len(sys.argv) > 1 else IMPORT_FILE)))
