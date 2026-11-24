"""Print the next answers to check by hand, skipping those already in checks.csv."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
audit = pd.read_pickle(HERE / "audit.pkl")
done = set(pd.read_csv(HERE / "checks.csv").key) if (HERE / "checks.csv").exists() else set()
todo = audit[(audit.rule == "check") & audit.judge.notna() & ~audit.key.isin(done)].drop_duplicates("key")
print("left", len(todo))
for _, r in todo.head(int(sys.argv[1])).iterrows():
    answer = " ".join(str(r.answer).split())
    print(f"#{r.key} J{int(r.judge) if r.judge == r.judge else '?'} Q: {r.question[:150]}\n  E: {str(r.expected)[:200]}\n  A: {answer[:420]}")
