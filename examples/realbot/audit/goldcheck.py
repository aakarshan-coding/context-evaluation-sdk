"""For every graded trace, where locate_gold puts the evidence, split by pass/fail. Lists passes whose gold is not in the prompt."""
import sys
from collections import Counter
from pathlib import Path
from contexteval._diagnose import locate_gold
from contexteval._store import Store, find_project
store = Store.open(find_project(Path.cwd()))
for run in map(int, sys.argv[1:]):
    counts = Counter()
    for st in store.traces(run):
        case = store.case(st.trace.test.case_id)
        loc = locate_gold(case.evidence, st.trace).location.value
        counts[(st.trace.test.passed, loc)] += 1
        if st.trace.test.passed and loc != "in_prompt":
            print(f"  run {run} trace {st.id} PASS with gold {loc}: {st.trace.question}")
    print(f"run {run}:", dict(counts))
