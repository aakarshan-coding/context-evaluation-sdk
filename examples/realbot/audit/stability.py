"""Repeat diagnosis replays N times per stored failure to see whether a single replay's verdict is stable.

python stability.py N TRACE_ID...   (run inside the project)
"""
import sys
from pathlib import Path

import tally  # noqa: F401
from contexteval._cli import _provider, load_config
from contexteval._diagnose import locate_gold, make_replay
from contexteval._runner import load_app, make_grader
from contexteval._store import Store, find_project
from contexteval.types import ReplayStep

n = int(sys.argv[1])
project = find_project(Path.cwd())
store = Store.open(project)
config = load_config(project)
grade = make_grader(_provider(config))
app = load_app("examples.realbot.bot:answer")
for tid in map(int, sys.argv[2:]):
    st = store.trace(tid)
    case = store.case(st.trace.test.case_id)
    gold = locate_gold(case.evidence, st.trace)
    replay = make_replay(app, case, case.evidence, grade, store.run(st.run_id).setting.intervention())
    steps = [ReplayStep.BASELINE_RERUN] + ([ReplayStep.INJECT_EVIDENCE] if gold.location.value != "in_prompt" else [ReplayStep.GOLD_TO_TOP, ReplayStep.ONLY_GOLD])
    cells = []
    for step in steps:
        outs = [replay(step) for _ in range(n)]
        cells.append(f"{step.value} {sum(o.passed for o in outs)}/{n}")
        for o in outs:
            if not o.passed:
                print(f"    trace {tid} {step.value} FAIL: {o.answer[:160]!r}")
    print(f"trace {tid} run {st.run_id} [{st.setting.name}] {case.question}\n  " + "  ".join(cells), flush=True)
