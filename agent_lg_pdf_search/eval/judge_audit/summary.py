"""Judge errors per stage, from the rules in build.py and the hand checks in checks.csv."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
audit = pd.read_pickle(HERE / "audit.pkl")
checks = pd.read_csv(HERE / "checks.csv").drop_duplicates("key", keep="last")
audit = audit.merge(checks, on="key", how="left")
audit.loc[audit.rule == "ok", "verdict"] = "ok"
audit["fn"] = (audit.verdict == "err") & (audit.judge == 0)
audit["fp"] = (audit.verdict == "err") & (audit.judge == 1)
# The fair score flips every wrong verdict.
audit["fair"] = audit.judge.where(audit.verdict != "err", 1 - audit.judge)
by = audit.groupby("stage").agg(
    rows=("n", "size"), unchecked=("verdict", lambda v: v.isna().sum()),
    judge=("judge", "mean"), fair=("fair", "mean"), fn=("fn", "sum"), fp=("fp", "sum"),
)
by["judge"] = (by.judge * 100).round(1)
by["fair"] = (by.fair * 100).round(1)
by["errors"] = by.fn + by.fp
print(by.to_string())
print("total errors", int(by.errors.sum()), "of", int(by.rows.sum()), "=", round(by.errors.sum() / by.rows.sum() * 100, 1), "%")
audit[audit.verdict == "err"][["stage", "n", "source", "question", "judge", "note"]].to_csv(HERE / "errors.csv", index=False, encoding="utf-8")
