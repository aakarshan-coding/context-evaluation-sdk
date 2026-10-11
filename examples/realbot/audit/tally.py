"""Import first: appends every OpenAI request's usage to tally.jsonl under COMMAND."""
import json
import sys
from pathlib import Path
from openai._base_client import SyncAPIClient

TALLY = Path.cwd() / "tally.jsonl"
COMMAND = [sys.argv[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]]
_original = SyncAPIClient.request


def _role(options) -> str:
    body = options.json_data if isinstance(options.json_data, dict) else {}
    if "input" in body and "messages" not in body:
        return "embed"
    first = (body.get("messages") or [{}])[0]
    content = first.get("content") if isinstance(first, dict) else ""
    return "app" if isinstance(content, str) and content.startswith("You are the customer support assistant") else "sdk"


def request(self, cast_to, options, *args, **kwargs):
    result = _original(self, cast_to, options, *args, **kwargs)
    usage = getattr(result, "usage", None)
    row = {"command": COMMAND[0], "role": _role(options), "model": getattr(result, "model", None),
           "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0, "completion_tokens": getattr(usage, "completion_tokens", 0) or 0}
    with TALLY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return result


SyncAPIClient.request = request
