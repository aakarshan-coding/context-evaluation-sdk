"""Scriptable fake app LLM. Its answer is a pure function of the prompt it sees (plus one
persisted counter for the flaky rule), so every planted label is deterministic.

Each rule states the context condition under which the model answers correctly. The
diagnoser's replays change exactly those conditions, so each expected label follows from
the rule plus plan item 5's ladder (flaky rerun first, then inject for missing, then gold to
top before only gold):

    rule               correct iff                    baseline prompt         rerun  inject  to-top  only-gold  label
    NeedsEvidence      evidence in any chunk          gold not retrieved      fail   pass    -       -          missing/not_retrieved, confirmed
    NeedsEvidence      evidence in any chunk          gold dropped [ARCHIVED] fail   pass    -       -          missing/dropped, confirmed
    NeedsEvidenceFirst evidence in the FIRST chunk    gold 2nd to 4th         fail   -       pass    -          ignored
    NeedsNoDistractor  evidence present, no marker    gold + marker chunk     fail   -       fail    pass       irrelevant
    AlwaysWrong        never                          gold present            fail   -       fail    fail       not_context
    FailsFirstCall     every call after the first     gold present, call 1    pass   -       -       -          flaky
    NeedsEvidence      evidence in any chunk          gold present            (passes at baseline: healthy)

Unscripted questions (synthetic cases from build-tests) echo the whole context back, so the
grader's substring fast path passes iff the answer was in context.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from .. import _text as text
from ..types import Evidence, never
from ._lock import file_lock

STATE_FILE_NAME = "demo_fake_state.json"


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
class FailsFirstCall:
    answer: str


Rule = Union[NeedsEvidence, NeedsEvidenceFirst, NeedsNoDistractor, AlwaysWrong, FailsFirstCall]

WRONG = "I don't know."


def _bump(state_file: Path, question: str) -> int:
    """Increment and persist the per-question call counter; returns the new count.

    The lock makes the read-modify-write one step across concurrent CLI processes. Without it,
    updates are lost and os.replace fails on Windows while another process has the file open.
    """
    with file_lock(state_file.with_suffix(".lock")):
        try:
            counts = json.loads(state_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            counts = {}
        counts[question] = counts.get(question, 0) + 1
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(counts, indent=2), encoding="utf-8")
        os.replace(tmp, state_file)
    return counts[question]


def evaluate(rule: Rule, question: str, chunks: list[str], state_file: Path) -> str:
    def has_gold(chunk: str, evidence: Evidence) -> bool:
        return text.contains_evidence(chunk, evidence)

    match rule:
        case NeedsEvidence(evidence=e, answer=a):
            return a if any(has_gold(c, e) for c in chunks) else WRONG
        case NeedsEvidenceFirst(evidence=e, answer=a):
            return a if chunks and has_gold(chunks[0], e) else WRONG
        case NeedsNoDistractor(evidence=e, answer=a, distractor_marker=m):
            ok = any(has_gold(c, e) for c in chunks) and not any(m in c for c in chunks)
            return a if ok else WRONG
        case AlwaysWrong():
            return WRONG
        case FailsFirstCall(answer=a):
            return a if _bump(state_file, question) > 1 else WRONG
        case _:
            never(rule)


def respond(prompt: str, state_file: Path) -> str:
    """parse_prompt -> look up planted.script()[question] -> evaluate the rule.

    FailsFirstCall increments a per-question counter in `state_file` (JSON). The counter must
    persist because `run` and `diagnose` are separate processes: run is call 1 (fail), the
    diagnoser's baseline rerun is call 2 (pass). Every later call passes, so experiment rows
    after the first run are stable.
    """
    from .app import parse_prompt
    from .planted import script

    question, chunks = parse_prompt(prompt)
    rule = script().get(question)
    if rule is None:
        return "\n".join(chunks) or WRONG
    return evaluate(rule, question, chunks, state_file)


def default_state_file() -> Path:
    """<project>/.contexteval/demo_fake_state.json for the project around the cwd."""
    from .app import data_dir

    return data_dir() / STATE_FILE_NAME


def transport(state_file: Path | None = None) -> Any:
    """httpx.MockTransport answering POST /chat/completions with an OpenAI-shaped JSON body
    (content from respond, usage from text.estimate_tokens). With no state_file, the default
    is resolved per request, so the counter follows the cwd's project.
    """
    import httpx

    def handle(request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/chat/completions"):
            return httpx.Response(404, json={"error": {"message": f"fake LLM has no {request.url.path}"}})
        body = json.loads(request.content)
        user = [m for m in body["messages"] if m.get("role") == "user"]
        prompt = user[-1]["content"] if user else ""
        content = respond(prompt, state_file or default_state_file())
        prompt_tokens, completion_tokens = text.estimate_tokens(prompt), text.estimate_tokens(content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
            },
        )

    return httpx.MockTransport(handle)
