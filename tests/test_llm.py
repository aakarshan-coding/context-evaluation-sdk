"""The SDK's own providers, as their requests reach the API (mock HTTP transport, no network)."""

from __future__ import annotations

import json
from typing import Any

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

from contexteval._llm import LLMRequest, OpenAIProvider, Task  # noqa: E402


def _sent(model: str) -> dict[str, Any]:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "PASS"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        })

    client = openai.OpenAI(api_key="fake", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    req = LLMRequest(Task.GRADE, system="s", user="u", inputs={}, max_tokens=64)
    assert OpenAIProvider(model=model, client=client).complete(req).text == "PASS"
    (body,) = seen
    return body


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "o3", "o4-mini"])
def test_openai_reasoning_models_get_max_completion_tokens_and_no_temperature(model: str) -> None:
    """Review Y7. Defect: the provider sends `temperature` and `max_tokens` to gpt-5 and o-series
    models, which the API rejects with a 400, so every grade and judge call fails."""
    body = _sent(model)
    assert "temperature" not in body and "max_tokens" not in body
    assert body["max_completion_tokens"] == 64


@pytest.mark.parametrize("model", ["gpt-4.1-mini", "gpt-4o-mini", "omni-custom"])
def test_openai_chat_models_keep_temperature_zero_and_max_tokens(model: str) -> None:
    """Defect: the reasoning-model branch catches ordinary chat models (any name starting with
    "o"), so their grades lose temperature 0 and stop being repeatable."""
    body = _sent(model)
    assert body["temperature"] == 0 and body["max_tokens"] == 64
    assert "max_completion_tokens" not in body
