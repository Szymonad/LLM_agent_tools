"""Copies the runs from results/ into one SQLite file for Datasette.

``answers.py`` and ``judge.py`` call ``build`` themselves; run it by hand only
after editing an ``answers.xlsx`` (e.g. the ``correct`` column):

    python -m agent_lg_pdf_search.eval.to_sqlite
    datasette agent_lg_pdf_search/eval/results/results.db --setting truncate_cells_html 200

A run is a folder ``results/<stage>/<run>/`` holding ``answers.xlsx`` and
usually ``run.json``. Stage folders starting with ``_`` (e.g. ``_old``) are
skipped. Datasette listens on 127.0.0.1 only; do not add ``--host 0.0.0.0``.
"""
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
DATABASE = RESULTS / "results.db"
#: Columns the comparison needs; a run without them gets them empty.
NEEDED = ["source", "correct", "judge", "page_hit", "seconds"]


def find_runs(results=RESULTS):
    """Folders of all runs, sorted by stage and then by run.

    Args:
        results (pathlib.Path): The ``results`` folder.

    Returns:
        list[pathlib.Path]: Folders holding ``answers.xlsx``.
    """
    return sorted(path.parent for path in results.glob("*/*/answers.xlsx") if not path.parts[-3].startswith("_"))


def read_run(folder):
    """Read one run: its answers and its description.

    Args:
        folder (pathlib.Path): Folder from ``find_runs``.

    Returns:
        tuple[pandas.DataFrame, dict]: The answers with ``stage``, ``run``
        and ``number`` (position of the question, from 1) put in front,
        and the contents of ``run.json`` (empty if the file is missing)
        with ``stage`` and ``run`` set.
    """
    stage, run = folder.parent.name, folder.name
    table = pd.read_excel(folder / "answers.xlsx")
    for column in NEEDED:
        if column not in table.columns:
            table[column] = None
    table.insert(0, "number", range(1, len(table) + 1))
    table.insert(0, "run", run)
    table.insert(0, "stage", stage)
    path = folder / "run.json"
    description = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    description.update(stage=stage, run=run)
    return table, description


def compare_view(sources):
    """SQL of the view ``stage_compare``: one row per run, metrics in percent.

    Args:
        sources (list[str]): Kinds of questions found in the ``source``
            column; each gets its own judge column.

    Returns:
        str: ``CREATE VIEW`` statement.
    """
    per_source = "".join(
        f",\n  ROUND(100.0 * AVG(CASE WHEN a.source = '{source.replace(chr(39), chr(39) * 2)}' THEN a.judge END), 1)"
        f" AS judge_{re.sub(r'[^0-9a-zA-Z]+', '_', source).strip('_').lower()}"
        for source in sources
    )
    return f"""CREATE VIEW stage_compare AS
SELECT
  a.stage,
  a.run,
  COUNT(*) AS questions,
  ROUND(100.0 * AVG(a.judge), 1) AS judge_pct,
  ROUND(100.0 * AVG(a.correct), 1) AS human_pct,
  ROUND(100.0 * AVG(a.page_hit), 1) AS page_hit_pct{per_source},
  ROUND(AVG(a.seconds), 1) AS avg_seconds,
  r.questions_file,
  SUBSTR(r.questions_sha256, 1, 8) AS questions_sha,
  r.chat_model,
  r.judge_model,
  r.git_commit
FROM answers AS a
LEFT JOIN runs AS r ON r.stage = a.stage AND r.run = a.run
GROUP BY a.stage, a.run
ORDER BY a.stage, a.run"""


def build(results=RESULTS, database=DATABASE):
    """Write every run in ``results`` into ``database``.

    Tables: ``answers`` (one row per question per run), ``runs`` (one row
    per run, from ``run.json``) and the view ``stage_compare``. The old
    tables are dropped inside the existing file instead of deleting it, so
    a running Datasette keeps working.

    Args:
        results (pathlib.Path): The ``results`` folder.
        database (pathlib.Path): SQLite file to write.

    Returns:
        dict: Number of rows per table.
    """
    tables, descriptions = [], []
    for folder in find_runs(results):
        table, description = read_run(folder)
        tables.append(table)
        descriptions.append(description)
    answers = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["stage", "run", "number", *NEEDED])
    answers["page_hit"] = answers["page_hit"].map({True: 1, False: 0, 1: 1, 0: 0}).astype("Int64")
    keys = {"questions_file", "questions_sha256", "chat_model", "judge_model", "commit"}.union(*map(set, descriptions))
    runs = pd.DataFrame(descriptions, columns=["stage", "run", *sorted(keys - {"stage", "run"})])
    runs = runs.rename(columns={"commit": "git_commit"})
    sources = sorted(answers["source"].dropna().unique())
    with sqlite3.connect(database) as connection:
        old = connection.execute("SELECT type, name FROM sqlite_master WHERE type IN ('view', 'table') AND name NOT LIKE 'sqlite_%' ORDER BY type DESC").fetchall()
        for kind, name in old:
            connection.execute(f'DROP {kind.upper()} "{name}"')
        answers.to_sql("answers", connection, index=False)
        runs.to_sql("runs", connection, index=False)
        connection.execute(compare_view(sources))
    return {"answers": len(answers), "runs": len(runs)}


if __name__ == "__main__":
    for name, rows in build().items():
        print(f"{rows:5} {name}")
    print("saved", DATABASE)
