"""The SDK's own LLM: test generation, grading, judge (plan item 9). Not the user's app LLM.

Every real provider call runs inside capture.suppressed(), so the SDK never records itself.
Temperature is fixed at 0 inside each implementation; callers cannot set it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol

from . import _text as text


class Task(Enum):
    GENERATE_CASES = "generate_cases"
    GRADE = "grade"
    JUDGE = "judge"


@dataclass(frozen=True)
class LLMRequest:
    task: Task
    system: str
    user: str  # rendered from `inputs` by the calling module
    inputs: Mapping[str, str]  # structured source of the prompt; the fake dispatches on these
    max_tokens: int = 1024


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class Provider(Protocol):
    name: str
    model: str

    def complete(self, req: LLMRequest) -> LLMResponse:
        """Raises ProviderError on transport or auth failure. Never returns partial text."""
        ...


class ProviderError(Exception):
    pass


@dataclass
class OpenAIProvider:
    model: str = "gpt-4.1-mini"
    name: str = "openai"
    client: Any = field(default=None, repr=False)  # openai.OpenAI, built on first use

    def complete(self, req: LLMRequest) -> LLMResponse:
        # Lazy: _capture imports this module, and core must import without openai installed.
        from . import _capture

        try:
            import openai
        except ImportError:
            raise ProviderError("the openai provider needs `pip install contexteval[openai]`") from None
        try:
            if self.client is None:
                self.client = openai.OpenAI()
            with _capture.suppressed():
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=req.max_tokens,
                    messages=[{"role": "system", "content": req.system}, {"role": "user", "content": req.user}],
                )
        except openai.OpenAIError as e:
            raise ProviderError(f"openai: {type(e).__name__}: {e}") from e
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else text.estimate_tokens(req.system + req.user),
            output_tokens=usage.completion_tokens if usage else 0,
        )


# Current-generation Claude models reject non-default sampling parameters with a 400, so
# temperature 0 goes only to models that accept it. anthropic 1.x dropped `temperature` from
# messages.create's signature, so it travels in extra_body.
_ANTHROPIC_NO_TEMPERATURE = ("claude-haiku-5", "claude-sonnet-5", "claude-opus-5", "claude-opus-4-7", "claude-opus-4-8", "claude-fable")


@dataclass
class AnthropicProvider:
    model: str = "claude-haiku-4-5"
    name: str = "anthropic"
    client: Any = field(default=None, repr=False)

    def complete(self, req: LLMRequest) -> LLMResponse:
        from . import _capture

        try:
            import anthropic
        except ImportError:
            raise ProviderError("the anthropic provider needs `pip install contexteval[anthropic]`") from None
        sampling: dict[str, Any] = {} if self.model.startswith(_ANTHROPIC_NO_TEMPERATURE) else {"extra_body": {"temperature": 0}}
        try:
            if self.client is None:
                self.client = anthropic.Anthropic()
            with _capture.suppressed():
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=req.max_tokens,
                    system=req.system,
                    messages=[{"role": "user", "content": req.user}],
                    **sampling,
                )
        except anthropic.AnthropicError as e:
            raise ProviderError(f"anthropic: {type(e).__name__}: {e}") from e
        answer = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return LLMResponse(text=answer, input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)


_SENTENCE = re.compile(r"[^.!?\n]+[.!?]")


def _fake_generate(req: LLMRequest) -> str:
    """One case per sentence of the passage, up to inputs["count"]. Evidence is the sentence as
    it appears in the passage and the answer is its last three words, so every case is verbatim."""
    items = []
    for match in _SENTENCE.finditer(req.inputs["passage"]):
        sentence = match.group().strip()
        words = sentence.rstrip(".!?").split()
        if len(words) < 5:
            continue
        items.append(
            {
                "question": f"According to the document, what completes: {' '.join(words[:-3])} ...?",
                "answer": " ".join(words[-3:]),
                "evidence": sentence,
            }
        )
        if len(items) >= int(req.inputs.get("count", "1")):
            break
    return json.dumps(items)


_FAKE_DEFAULTS: dict[Task, Callable[[LLMRequest], str]] = {
    Task.GENERATE_CASES: _fake_generate,
    Task.GRADE: lambda req: "FAIL",
    Task.JUDGE: lambda req: json.dumps({"verdict": "no_failure", "rationale": "fake"}),
}


@dataclass
class FakeProvider:
    """Deterministic provider for tests and the offline demo. Records every request.

    Handlers see the structured request (task and inputs) and never parse prompt text.
    Defaults:
    - GENERATE_CASES: one case per sentence of inputs["passage"]; evidence is the sentence,
      answer its last three words. Always valid JSON.
    - GRADE: "FAIL" (so tests prove the substring fast path did the passing).
    - JUDGE: {"verdict": "no_failure", "rationale": "fake"}.
    """

    handlers: dict[Task, Callable[[LLMRequest], str]] = field(default_factory=dict)
    requests: list[LLMRequest] = field(default_factory=list)
    model: str = "fake"
    name: str = "fake"

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        out = self.handlers.get(req.task, _FAKE_DEFAULTS[req.task])(req)
        return LLMResponse(text=out, input_tokens=text.estimate_tokens(req.system + req.user), output_tokens=text.estimate_tokens(out))


def make_provider(name: str, model: str | None) -> Provider:
    """Boundary: config/CLI string -> Provider. Raises ValueError naming the valid choices."""
    if name == "openai":
        return OpenAIProvider(model=model) if model else OpenAIProvider()
    if name == "anthropic":
        return AnthropicProvider(model=model) if model else AnthropicProvider()
    if name == "fake":
        return FakeProvider()
    raise ValueError(f"unknown provider {name!r}; choose one of openai, anthropic, fake")


# --- cost ---------------------------------------------------------------------


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


# USD per million tokens, standard uncached rates as of 2026-10.
PRICES: Mapping[str, Price] = {
    "fake": Price(0.0, 0.0),
    "gpt-4.1-mini": Price(0.40, 1.60),
    "gpt-4.1-nano": Price(0.10, 0.40),
    "gpt-4o-mini": Price(0.15, 0.60),
    "claude-haiku-4-5": Price(1.00, 5.00),
    "claude-haiku-5-5": Price(0.10, 0.50),
    "claude-sonnet-4-6": Price(3.00, 15.00),
    "claude-sonnet-5-5": Price(2.00, 10.00),
}


# Providers pin snapshots with a date suffix: gpt-4.1-mini-2025-04-14, claude-haiku-4-5-20251001.
_DATE_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})$")


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICES.get(model) or PRICES.get(_DATE_SUFFIX.sub("", model))
    if price is None:
        return None
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000


@dataclass(frozen=True)
class Estimate:
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float | None

    def __add__(self, other: Estimate) -> Estimate:
        usd = None if self.usd is None or other.usd is None else self.usd + other.usd
        return Estimate(self.calls + other.calls, self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens, usd)


def estimate(model: str, prompts: Iterable[str], output_tokens_each: int) -> Estimate:
    """chars / 4 per prompt (plan item 14)."""
    prompts = list(prompts)
    input_tokens = sum(text.estimate_tokens(p) for p in prompts)
    output_tokens = output_tokens_each * len(prompts)
    return Estimate(len(prompts), input_tokens, output_tokens, cost_usd(model, input_tokens, output_tokens))
