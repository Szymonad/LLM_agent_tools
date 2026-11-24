"""Collect every judged answer of stages 00-05 into one audit table.

Rows the rules below settle are marked; the rest are checked by hand
and their check is written to checks.csv (row id, ok/err, note).
"""
import glob
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
R = HERE.parent / "results"
STAGES = ["00_baseline_v3", "01_router", "02_docling_ocr", "03_vlm_image_captions", "04_doc_name_embed", "05_no_thinking"]
NOT_COVERED = re.compile(r"do(es)? not (cover|contain|mention|provide|specify|include)|not (mentioned|provided|available|specified|found)|cannot be determined|no information", re.I)


def numbers(text):
    return {n.replace(",", "").rstrip("0").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", str(text))}


def rule(row):
    answer = "" if pd.isna(row.answer) else str(row.answer)
    if not answer.strip() or answer.startswith("ERROR"):
        return "ok" if row.judge == 0 else "check"
    if str(row.expected).strip().lower() == "not answerable":
        return "ok" if (row.judge == 1) == bool(NOT_COVERED.search(answer)) else "check"
    # Saying the documents lack an answer the reference has is wrong.
    if row.judge == 0 and NOT_COVERED.search(answer[:200]):
        return "ok"
    expected = numbers(row.expected)
    if row.judge == 1 and expected and len(str(row.expected)) < 20 and expected <= numbers(answer):
        return "ok"
    # A bare number expected and none of it in the answer is wrong, words aside.
    if row.judge == 0 and re.fullmatch(r"[\d.,%$ -]+", str(row.expected).strip()) and not expected <= numbers(answer):
        return "ok"
    return "check"


rows = []
for stage in STAGES:
    df = pd.read_excel(sorted(glob.glob(str(R / stage / "*/answers.xlsx")))[-1])
    df.insert(0, "stage", stage)
    df.insert(1, "n", range(len(df)))
    rows.append(df[["stage", "n", "source", "question", "expected", "answer", "judge"]])
audit = pd.concat(rows, ignore_index=True)
audit["rule"] = audit.apply(rule, axis=1)
# The same answer to the same question is checked once.
audit["key"] = audit.groupby(["question", audit.answer.astype(str), "judge"]).ngroup()
audit.to_pickle(HERE / "audit.pkl")
print(audit.rule.value_counts().to_dict(), "unique to check:", audit[audit.rule == "check"].key.nunique())
