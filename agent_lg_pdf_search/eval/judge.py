"""Grades the agent's answers with the model as a judge.

Run from the repo root:

    python -m agent_lg_pdf_search.eval.judge --stage NAME [--run FOLDER]

The grades go into the ``judge`` column of the run's ``answers.xlsx`` (the
newest run of the stage unless ``--run`` is given), the judge model into its
``run.json``, and ``results.db`` is rebuilt.
"""
import argparse
import json
import re
from datetime import datetime

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from agent_lg_pdf_search.agent import MODEL
from agent_lg_pdf_search.config import SERVER
from agent_lg_pdf_search.eval import to_sqlite
from agent_lg_pdf_search.eval.answers import model_file
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


def run_folder(stage, run=None):
    """Folder of a run: the given one, or the newest of the stage.

    Args:
        stage (str): Name of the stage folder in ``results/``.
        run (str | None): Name of the run folder, ``None`` for the newest.

    Returns:
        pathlib.Path: Folder holding ``answers.xlsx``.

    Raises:
        SystemExit: The folder or its ``answers.xlsx`` does not exist.
    """
    if run:
        folder = RESULTS / stage / run
    else:
        runs = sorted(path.parent for path in (RESULTS / stage).glob("*/answers.xlsx"))
        if not runs:
            raise SystemExit(f"no run with answers.xlsx in {RESULTS / stage}")
        folder = runs[-1]
    if not (folder / "answers.xlsx").exists():
        raise SystemExit(f"no answers.xlsx in {folder}")
    return folder


def note_judge(folder):
    """Record in the run's ``run.json`` which model graded it and when.

    Runs graded by different judges are not comparable, so this is kept
    next to the run. A missing ``run.json`` is created.

    Args:
        folder (pathlib.Path): Folder of the run.
    """
    path = folder / "run.json"
    description = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    description["judge_model"] = model_file(SERVER)
    description["judged"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(description, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grade the agent's answers with the model.")
    parser.add_argument("--stage", required=True, help="stage folder in results/, e.g. 00_baseline_pypdf")
    parser.add_argument("--run", help="run folder in the stage, the newest if left out")
    args = parser.parse_args()
    folder = run_folder(args.stage, args.run)
    path = folder / "answers.xlsx"
    try:
        path.open("r+b").close()
    except PermissionError:
        raise SystemExit(f"{path} is open in another program, close it first")
    table = run(path)
    # Grading takes minutes; read the file again so that grades typed into
    # ``correct`` meanwhile are kept, and replace only the judge column.
    current = pd.read_excel(path)
    if len(current) != len(table):
        raise SystemExit(f"{path} changed its number of rows while grading, nothing saved")
    current["judge"] = table["judge"].values
    table = current
    table.to_excel(path, index=False)
    note_judge(folder)
    print("saved", path)
    print("judge accuracy", table["judge"].dropna().astype(int).mean())
    if "source" in table.columns and table["source"].notna().any():
        for source, part in table.groupby("source"):
            print("judge accuracy", source, part["judge"].dropna().astype(int).mean())
    print("agreement", agreement(table))
    to_sqlite.build()
    print("rebuilt", to_sqlite.DATABASE)
