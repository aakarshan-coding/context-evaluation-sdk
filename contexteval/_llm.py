"""The SDK's own LLM: test generation, grading, judge (plan item 9). Not the user's app LLM.

Every real provider call runs inside capture.suppressed(), so the SDK never records itself.
Temperature is fixed at 0 inside each implementation; callers cannot set it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Mapping, Protocol


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
    model: str = "gpt-4o-mini"
    name: str = "openai"

    def complete(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError  # lazy `import openai`; chat.completions.create(temperature=0)


@dataclass
class AnthropicProvider:
    model: str = "claude-haiku-4-5"
    name: str = "anthropic"

    def complete(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError  # lazy `import anthropic`; messages.create(temperature=0)


@dataclass
class FakeProvider:
    """Deterministic provider for tests and the offline demo. Records every request.

    Default handlers:
    - GENERATE_CASES: evidence = first sentence of inputs["passage"], answer = its last
      noun-ish token run, question templated from it. Always valid JSON.
    - GRADE: "FAIL" (so tests prove the substring fast path did the passing).
    - JUDGE: {"verdict": "no_failure", "rationale": "fake"} unless overridden.
    """

    handlers: dict[Task, Callable[[LLMRequest], str]] = field(default_factory=dict)
    requests: list[LLMRequest] = field(default_factory=list)
    model: str = "fake"
    name: str = "fake"

    def complete(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError


def make_provider(name: str, model: str | None) -> Provider:
    """Boundary: config/CLI string -> Provider. Raises ValueError naming the valid choices."""
    raise NotImplementedError


# --- cost ---------------------------------------------------------------------


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


PRICES: Mapping[str, Price] = {
    # filled at implementation time from provider pricing pages; "fake" costs 0
    "fake": Price(0.0, 0.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    raise NotImplementedError  # None when model not in PRICES


@dataclass(frozen=True)
class Estimate:
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float | None

    def __add__(self, other: Estimate) -> Estimate:
        raise NotImplementedError


def estimate(model: str, prompts: Iterable[str], output_tokens_each: int) -> Estimate:
    """chars / 4 per prompt (plan item 14)."""
    raise NotImplementedError
