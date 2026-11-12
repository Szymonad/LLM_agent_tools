"""Grades the agent's answers with the model as a judge.

Run from the repo root:

    python -m agent_lg_pdf_search.eval.judge [--file NAME]
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from agent_lg_pdf_search.agent import MODEL
from agent_lg_pdf_search.eval.evaluate import RESULTS

JUDGE = """You grade an answer given by an assistant to a question about documents.
You get the question, the reference answer and the assistant's answer.

Decide whether the assistant's answer agrees with the reference answer.
- The content counts, not the form. A sentence instead of a bare number is fine.
  A rounding that suits the question is fine.
- When the reference answer is a list, every element of it is required.
- An answer saying that the documents do not contain the information is wrong,
  unless the reference answer is "Not answerable". Then only such an answer is
  correct.

Write only 1 (correct) or 0 (wrong)."""


def grade(question, expected, answer):
    """Ask the model whether the answer agrees with the reference answer.

    An empty answer or one starting with ``ERROR`` (a failed run in
    ``answers.py``) is wrong without calling the model. The model thinks
    before it answers and the reasoning is not part of ``content``, so only
    the last 0 or 1 in the reply is used.

    Args:
        question (str): Evaluation question.
        expected (str): Reference answer.
        answer (str): The agent's answer, may be empty.

    Returns:
        int | None: 1 if correct, 0 if wrong, ``None`` if the reply holds
        neither.
    """
    if pd.isna(answer) or not str(answer).strip() or str(answer).startswith("ERROR"):
        return 0
    reply = MODEL.invoke(
        [
            SystemMessage(JUDGE),
            HumanMessage(f"Question:\n{question}\n\nReference answer:\n{expected}\n\nAssistant's answer:\n{answer}"),
        ]
    )
    marks = re.findall(r"[01]", reply.content.strip())
    if not marks:
        return None
    return int(marks[-1])


def run(path):
    """Grade every answer in the file.

    One line per row is printed on stdout as it finishes. A failed model
    call does not stop the run: the row gets ``None`` and the error type is
    printed.

    Args:
        path (pathlib.Path): Excel file written by ``answers.save``.

    Returns:
        pandas.DataFrame: The file's table with the column ``judge`` added
        (1, 0 or ``None``).
    """
    table = pd.read_excel(path)
    grades = []
    for number, row in enumerate(table.itertuples(), start=1):
        try:
            mark = grade(row.question, row.expected, row.answer)
        except Exception as error:
            mark = None
            print(f"{number:3} ERROR {type(error).__name__}")
        grades.append(mark)
        print(f"{number:3}/{len(table)} {mark} | {row.question[:60]}")
    table["judge"] = pd.Series(grades, index=table.index, dtype="Int64")
    return table


def agreement(table):
    """Compare the judge with the human grades.

    Only rows with both ``correct`` and ``judge`` filled in are counted.

    Args:
        table (pandas.DataFrame): Table returned by ``run``.

    Returns:
        dict: Keys ``rows`` (number of compared rows), and when it is not
        zero also ``agreed`` (share of equal grades), ``judge_yes_human_no``
        and ``judge_no_human_yes``.
    """
    both = table.dropna(subset=["correct", "judge"])
    if both.empty:
        return {"rows": 0}
    human = both["correct"].astype(int)
    judge = both["judge"].astype(int)
    return {
        "rows": len(both),
        "agreed": (human == judge).mean(),
        "judge_yes_human_no": int(((judge == 1) & (human == 0)).sum()),
        "judge_no_human_yes": int(((judge == 0) & (human == 1)).sum()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grade the agent's answers with the model.")
    parser.add_argument("--file", default="answers.xlsx", help="name of the answers file in results/")
    args = parser.parse_args()
    table = run(RESULTS / args.file)
    out = RESULTS / f"{Path(args.file).stem}_judged.xlsx"
    table.to_excel(out, index=False)
    print("saved", out)
    print("judge accuracy", table["judge"].dropna().astype(int).mean())
    if "source" in table.columns and table["source"].notna().any():
        for source, part in table.groupby("source"):
            print("judge accuracy", source, part["judge"].dropna().astype(int).mean())
    print("agreement", agreement(table))
