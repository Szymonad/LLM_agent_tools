"""Copies the answer files from results/ into one SQLite file for Datasette.

Run from the repo root:

    python -m agent_lg_pdf_search.eval.to_sqlite
    datasette agent_lg_pdf_search/eval/results/results.db --setting truncate_cells_html 200

The database is rebuilt from scratch on every run, so run it again after a new
``answers.py`` or ``judge.py`` run. Datasette listens on 127.0.0.1 only; do not
add ``--host 0.0.0.0``.
"""
import sqlite3
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
DATABASE = RESULTS / "results.db"


def build(results=RESULTS, database=DATABASE):
    """Write every ``*.xlsx`` in ``results`` into ``database``.

    Each file becomes a table named after the file. All files together also go
    into the table ``all_answers`` with the column ``run`` (the file name), so
    runs can be compared question by question.

    Args:
        results (pathlib.Path): Folder with the Excel files.
        database (pathlib.Path): SQLite file to write, replaced if it exists.

    Returns:
        dict: Number of rows per table.
    """
    files = sorted(path for path in results.glob("*.xlsx") if not path.name.startswith(".~lock"))
    database.unlink(missing_ok=True)
    counts = {}
    tables = []
    with sqlite3.connect(database) as connection:
        for path in files:
            table = pd.read_excel(path)
            table.to_sql(path.stem, connection, index=False)
            counts[path.stem] = len(table)
            tables.append(table.assign(run=path.stem))
        if tables:
            combined = pd.concat(tables, ignore_index=True)
            combined.to_sql("all_answers", connection, index=False)
            counts["all_answers"] = len(combined)
    return counts


if __name__ == "__main__":
    for name, rows in build().items():
        print(f"{rows:5} {name}")
    print("saved", DATABASE)
