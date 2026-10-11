import sys
from pathlib import Path
from contexteval._store import Store, find_project
store = Store.open(find_project(Path.cwd()))
for st in store.traces(int(sys.argv[1])):
    t = st.trace
    if not t.test.passed:
        continue
    case = store.case(t.test.case_id)
    print(f"[{st.id}] {t.test.method:9} Q: {t.question}\n    EXP: {case.expected_answer}\n    ANS: {t.outcome.text[:300]!r}")
