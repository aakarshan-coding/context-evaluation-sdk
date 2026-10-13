"""Run the real commands the video shows and save their real output to captured/<scene>.json.

Usage: python capture.py demo|realbot WORKDIR

`demo` installs contexteval from PyPI into a throwaway venv, then runs the fake-provider demo.
`realbot` runs examples/realbot against gpt-4.1-mini (needs OPENAI_API_KEY). Every command that
prints an estimate is first run without --yes, which prints the estimate and aborts; the run stops
if the summed estimates or the tallied real spend pass the budget.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "captured"
BUDGET_USD = 0.40
PRICES = {"gpt-4.1-mini": (0.40, 1.60), "text-embedding-3-small": (0.02, 0.0)}  # USD per 1M tokens


def env() -> dict[str, str]:
    e = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8", PYTHONUTF8="1", NO_COLOR="1")
    e.pop("CONTEXTEVAL_DEMO_LLM", None)
    return e


def execute(argv: list[str], cwd: Path) -> tuple[str, float]:
    start = time.monotonic()
    proc = subprocess.run(argv, cwd=cwd, env=env(), stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, encoding="utf-8", timeout=1100)
    out = (proc.stdout + proc.stderr).rstrip("\n")
    if proc.returncode != 0:
        raise SystemExit(f"{argv} exited {proc.returncode}:\n{out}")
    return out, time.monotonic() - start


def tallied_spend(cwd: Path) -> float:
    tally = cwd / "tally.jsonl"
    if not tally.exists():
        return 0.0
    total = 0.0
    for line in tally.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        pin, pout = next(p for name, p in PRICES.items() if (row["model"] or "").startswith(name))
        total += (row["prompt_tokens"] * pin + row["completion_tokens"] * pout) / 1e6
    return total


def capture_demo(work: Path) -> dict:
    venv = work / "pipvenv"
    project = work / "demo-project"
    for d in (venv, project):
        shutil.rmtree(d, ignore_errors=True)
    project.mkdir(parents=True)
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=280)
    vpy = venv / "Scripts" / "python.exe"
    steps = []
    out, secs = execute([str(vpy), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "contexteval"], project)
    steps.append({"cmd": "pip install contexteval", "out": out, "seconds": secs})
    shutil.rmtree(venv, ignore_errors=True)
    for shown in [
        "contexteval init --provider fake",
        "python -m contexteval.demo.planted",
        "contexteval build-tests --import planted.jsonl",
        "contexteval run contexteval.demo.app:answer --yes",
        "contexteval diagnose --yes",
        "contexteval sweep --holdout 0 --yes",
    ]:
        args = shown.split()
        argv = [sys.executable, "-m", *args[2:]] if args[0] == "python" else [sys.executable, "-m", "contexteval", *args[1:]]
        out, secs = execute(argv, project)
        steps.append({"cmd": shown, "out": out, "seconds": secs})
    return {"project": str(project), "steps": steps}


def capture_realbot(work: Path) -> dict:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    project = work / "realbot-project"
    shutil.rmtree(project, ignore_errors=True)
    project.mkdir(parents=True)
    shutil.copytree(REPO / "examples/realbot/corpus", project / "corpus")
    shutil.copy(REPO / "examples/realbot/customer_questions.jsonl", project)
    ce = str(REPO / "examples/realbot/audit/ce.py")
    estimated = 0.0
    steps = []
    for shown in [
        "contexteval init --provider openai --model gpt-4.1-mini",
        "contexteval build-tests --import customer_questions.jsonl",
        "contexteval build-tests corpus --per-passage 1 --yes",
        "contexteval run examples.realbot.bot:answer --yes",
        "contexteval diagnose --yes",
        "contexteval sweep --yes",
    ]:
        args = shown.split()[1:]
        if "--yes" in args:
            proc = subprocess.run([sys.executable, ce, *[a for a in args if a != "--yes"]], cwd=project, env=env(),
                                  stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", timeout=600)
            match = re.search(r"about \$([0-9.]+)", proc.stdout)
            if match is None:
                raise SystemExit(f"no estimate for {shown}:\n{proc.stdout}{proc.stderr}")
            estimated += float(match.group(1))
            print(f"{shown}: estimate ${match.group(1)}, cumulative ${estimated:.4f}, spent ${tallied_spend(project):.4f}", flush=True)
            if estimated > BUDGET_USD or tallied_spend(project) > BUDGET_USD:
                raise SystemExit("budget exceeded; stopping")
        out, secs = execute([sys.executable, ce, *args], project)
        steps.append({"cmd": shown, "out": out, "seconds": secs})
        print(f"{shown}: done in {secs:.0f}s, spent ${tallied_spend(project):.4f}", flush=True)
        if tallied_spend(project) > BUDGET_USD:
            raise SystemExit("budget exceeded; stopping")
    return {"project": str(project), "steps": steps, "estimated_usd": round(estimated, 4),
            "spent_usd": round(tallied_spend(project), 4)}


def main() -> None:
    scene, work = sys.argv[1], Path(sys.argv[2]).resolve()
    result = {"demo": capture_demo, "realbot": capture_realbot}[scene](work)
    OUT.mkdir(exist_ok=True)
    project = result.pop("project")
    text = json.dumps(result, indent=1, ensure_ascii=False).replace(project.replace("\\", "\\\\"), ".")
    (OUT / f"{scene}.json").write_text(text, encoding="utf-8")
    print(f"wrote {OUT / f'{scene}.json'}")


if __name__ == "__main__":
    main()
