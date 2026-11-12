"""Builds ``questions_v2.json`` from the raw MMLongBench-Doc questions.

Keeps the questions about the PDFs in ``pdf/`` that are answerable from plain
text, plus the unanswerable ones. Run from the repo root:

    python -m agent_lg_pdf_search.eval.build_questions
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "mmlongbench_raw.json"
PDF_DIR = HERE / "pdf"
OUT = HERE / "questions_v2.json"

PURE_TEXT = "['Pure-text (Plain-text)']"
NOT_ANSWERABLE = "Not answerable"


def build():
    """Select and convert the questions.

    Returns:
        list[dict]: In the order of the raw file. Items have the keys of
        ``questions.json`` (``question``, ``doc``, ``pages``, ``answer``,
        ``type``) and ``source``: ``pure-text`` or ``not-answerable``.
        ``pages`` is empty for the unanswerable questions.
    """
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    files = {path.name for path in PDF_DIR.glob("*.pdf")}
    questions = []
    for entry in raw:
        if entry["doc_id"] not in files:
            continue
        if entry["answer"] == NOT_ANSWERABLE:
            source, pages = "not-answerable", []
        elif entry["evidence_sources"] == PURE_TEXT:
            source, pages = "pure-text", ast.literal_eval(entry["evidence_pages"])
        else:
            continue
        questions.append(
            {
                "question": entry["question"],
                "doc": entry["doc_id"],
                "pages": pages,
                "answer": entry["answer"],
                "type": entry["doc_type"],
                "source": source,
            }
        )
    return questions


if __name__ == "__main__":
    questions = build()
    OUT.write_text(json.dumps(questions, indent=1, ensure_ascii=False), encoding="utf-8")
    for source in ("pure-text", "not-answerable"):
        print(source, sum(item["source"] == source for item in questions))
    print("documents", len({item["doc"] for item in questions}))
    print("saved", OUT)
