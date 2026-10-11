"""Print stored traces for hand audit.

python dump.py RUN_ID [failed|passed|all] [--full]
Run from inside the project directory.
"""

import sys
from pathlib import Path

from contexteval import _text as text
from contexteval._store import Store, find_project
from contexteval.types import Answered, explain

run_id = int(sys.argv[1])
which = sys.argv[2] if len(sys.argv) > 2 else "failed"
full = "--full" in sys.argv
store = Store.open(find_project(Path.cwd()))
for st in store.traces(run_id):
    t = st.trace
    if which == "failed" and t.test.passed or which == "passed" and not t.test.passed:
        continue
    case = store.case(t.test.case_id)
    print("=" * 100)
    print(f"trace {st.id}  passed={t.test.passed} ({t.test.method})  Q: {t.question}")
    print(f"  expected: {case.expected_answer}")
    print(f"  evidence: {case.evidence.text if case.evidence else None}   [{case.source.rsplit('/', 1)[-1]}]")
    ans = t.outcome.text if isinstance(t.outcome, Answered) else t.outcome
    print(f"  answer:   {ans}")
    for r, ev in enumerate(t.retrievals):
        flags = text.gold_flags([c.text for c in ev.delivered], case.evidence) if case.evidence else ()
        print(f"  retrieval {r} query={ev.query!r} params={dict(ev.params)}")
        for c, chunk in enumerate(ev.delivered):
            mark = "GOLD" if flags and flags[c] else "    "
            inp = "P" if (r, c) in t.in_prompt else "-"
            body = chunk.text if full else chunk.text[:110].replace("\n", " ")
            print(f"    [{c}] {mark} {inp} {chunk.id} score={chunk.score:.3f}  {body}")
    if st.diagnosis is not None:
        print(f"  DIAGNOSIS: {explain(st.diagnosis)}")
        for rep in getattr(st.diagnosis, "replays", ()):
            print(f"    replay {rep.step.value}: passed={rep.passed}  {rep.answer!r}")
