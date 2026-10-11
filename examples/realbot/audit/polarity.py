"""Across all runs: passes whose answer opens with the opposite yes/no of the expected answer."""
import sqlite3
from contexteval import _text as text
c = sqlite3.connect(".contexteval/contexteval.db")
rows = c.execute("select t.id, t.run_id, t.grade_method, c.question, c.expected_answer, t.answer from traces t join test_cases c on c.id = t.case_id where t.passed = 1").fetchall()
n = 0
for tid, run, method, q, exp, ans in rows:
    e = text.tokens(exp)[:1]
    a = text.tokens(ans or "")[:1]
    if e and a and e[0] in ("yes", "no") and a[0] in ("yes", "no") and e != a:
        n += 1
        print(f"trace {tid} run {run} {method}: Q={q!r} EXP={exp!r}\n   ANS={ans[:220]!r}")
print("opposite-polarity passes:", n, "of", len(rows))
