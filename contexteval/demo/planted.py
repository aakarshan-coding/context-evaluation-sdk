"""The one list of planted failures. Demo test cases, the fake LLM's script, and the
acceptance test's expected labels are all derived from PLANTED; nothing is listed twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..types import Label, TestCase
from .fake_llm import Rule


@dataclass(frozen=True)
class Planted:
    case: TestCase
    rule: Rule
    expected: Label
    cause: Literal["not_retrieved", "dropped"] | None = None  # set iff expected is MISSING
    confirmed: bool | None = None  # expected Missing.confirmed


PLANTED: tuple[Planted, ...] = ()
# Filled at implementation time against corpus/*.md. One entry per row:
#   missing/not_retrieved  NeedsEvidence, gold chunk ranks 5th          confirmed=True
#   missing/dropped        NeedsEvidence, gold chunk carries [ARCHIVED] confirmed=True
#   ignored                NeedsEvidenceFirst, gold ranks 2nd
#   irrelevant             NeedsNoDistractor, gold ranks 1st, distractor ranks 2nd
#   not_context            AlwaysWrong, gold ranks 1st
#   flaky                  FlipFlop, gold ranks 1st
# The dropped case's evidence must exclude the [ARCHIVED] marker, or the injected chunk is
# filtered again and confirmed flips to False.


def cases() -> tuple[TestCase, ...]:
    return tuple(p.case for p in PLANTED)


def script() -> dict[str, Rule]:
    return {p.case.question: p.rule for p in PLANTED}


IMPORT_FILE = "planted.jsonl"  # written by `python -m contexteval.demo.planted` for `build-tests --import`
