"""Sum ./tally.jsonl (written by tally.py) into calls and dollars per command and role."""

import json
from collections import defaultdict
from pathlib import Path

PRICES = {  # USD per 1M tokens (input, output)
    "gpt-4.1-mini": (0.40, 1.60),
    "text-embedding-3-small": (0.02, 0.0),
}


def price(model: str) -> tuple[float, float]:
    for name, p in PRICES.items():
        if model and model.startswith(name):
            return p
    raise KeyError(model)


rows = [json.loads(line) for line in (Path.cwd() / "tally.jsonl").read_text().splitlines() if line]
groups: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0])
for r in rows:
    pin, pout = price(r["model"])
    cost = (r["prompt_tokens"] * pin + r["completion_tokens"] * pout) / 1e6
    g = groups[(r["command"], r["role"])]
    g[0] += 1
    g[1] += cost
total = 0.0
for (command, role), (calls, cost) in sorted(groups.items()):
    print(f"{command:12} {role:6} calls={calls:4d} ${cost:.4f}")
    total += cost
print(f"TOTAL calls={len(rows)} ${total:.4f}")
