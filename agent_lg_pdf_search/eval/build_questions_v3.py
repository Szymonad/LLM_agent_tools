"""Builds ``questions_v3.json``: every kind of question the agent must handle.

Run once from the repo root; the file is then frozen for the whole project:

    python -m agent_lg_pdf_search.eval.build_questions_v3

Kinds (the ``source`` field):

- ``pure-text``, ``table``, ``chart``, ``figure``, ``layout``: MMLongBench-Doc
  questions whose evidence is of that single kind. Questions with mixed
  evidence are left out, so every kind can be scored on its own.
- ``not-answerable``: MMLongBench-Doc questions with no answer in the PDFs.
- ``scan``: the answerable questions about ``SCAN_DOCS``. Those PDFs are
  rendered to page images without a text layer (``scan_<name>.pdf`` in
  ``pdf/``) and the originals are moved to ``pdf_originals/``, so the index
  holds only the scan and no text copy competes with it. ``content`` keeps
  the kind of the evidence.
- ``general``: questions from ``general_questions.json``, unrelated to the
  documents, written before the router.

``route`` is what a router should decide: ``general`` or ``documents``.
"""
import ast
import json
import shutil
from pathlib import Path

import pypdfium2

HERE = Path(__file__).resolve().parent
RAW = HERE / "mmlongbench_raw.json"
GENERAL = HERE / "general_questions.json"
PDF_DIR = HERE / "pdf"
ORIGINALS = HERE / "pdf_originals"
OUT = HERE / "questions_v3.json"

#: Short text PDFs with many answerable questions and no layout questions (there are only 7 of those).
SCAN_DOCS = [
    "a5879805d70c854ea4361e43a84e3bb2.pdf",
    "afe620b9beac86c1027b96d31d396407.pdf",
    "c31e6580d0175ab3f9d99d1ff0bfa000.pdf",
    "f8d3a162ab9507e021d83dd109118b60.pdf",
]
SCAN_DPI = 150
SCAN_JPEG_QUALITY = 80

KINDS = {
    "['Pure-text (Plain-text)']": "pure-text",
    "['Table']": "table",
    "['Chart']": "chart",
    "['Figure']": "figure",
    "['Generalized-text (Layout)']": "layout",
}
NOT_ANSWERABLE = "Not answerable"


def scan_name(doc):
    return f"scan_{doc}"


def make_scan(source, target):
    """Write ``target``: every page of ``source`` as a JPEG, with no text layer."""
    document = pypdfium2.PdfDocument(source)
    try:
        images = [page.render(scale=SCAN_DPI / 72).to_pil().convert("RGB") for page in document]
    finally:
        document.close()
    images[0].save(target, "PDF", save_all=True, append_images=images[1:], resolution=SCAN_DPI, quality=SCAN_JPEG_QUALITY)


def make_scans():
    """Replace every PDF in ``SCAN_DOCS`` with its scan. Safe to run again."""
    ORIGINALS.mkdir(exist_ok=True)
    for doc in SCAN_DOCS:
        original = ORIGINALS / doc
        if (PDF_DIR / doc).exists():
            shutil.move(PDF_DIR / doc, original)
        target = PDF_DIR / scan_name(doc)
        if not target.exists():
            make_scan(original, target)
            print("scan", target.name)


def build():
    """Select and convert the questions.

    Returns:
        list[dict]: MMLongBench-Doc questions in the order of the raw file,
        then the general ones. Keys: ``question``, ``doc``, ``pages``,
        ``answer``, ``type``, ``source``, ``route``, and ``content`` for
        ``scan``. ``doc`` is empty and ``pages`` too for ``general``.
    """
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    files = {path.name for path in PDF_DIR.glob("*.pdf")} | set(SCAN_DOCS)
    questions = []
    for entry in raw:
        doc = entry["doc_id"]
        if doc not in files:
            continue
        scanned = doc in SCAN_DOCS
        extra = {}
        if entry["answer"] == NOT_ANSWERABLE:
            source, pages = "not-answerable", []
        else:
            pages = ast.literal_eval(entry["evidence_pages"])
            if not pages:
                continue
            if scanned:
                source = "scan"
                extra["content"] = ", ".join(ast.literal_eval(entry["evidence_sources"]))
            elif entry["evidence_sources"] in KINDS:
                source = KINDS[entry["evidence_sources"]]
            else:
                continue
        questions.append(
            {
                "question": entry["question"],
                "doc": scan_name(doc) if scanned else doc,
                "pages": pages,
                "answer": entry["answer"],
                "type": entry["doc_type"],
                "source": source,
                "route": "documents",
                **extra,
            }
        )
    for item in json.loads(GENERAL.read_text(encoding="utf-8")):
        questions.append(
            {"question": item["question"], "doc": "", "pages": [], "answer": item["answer"], "type": "", "source": "general", "route": "general"}
        )
    return questions


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"{OUT} exists and is frozen; delete it by hand to build it again")
    make_scans()
    questions = build()
    OUT.write_text(json.dumps(questions, indent=1, ensure_ascii=False), encoding="utf-8")
    for source in ("pure-text", "table", "chart", "figure", "layout", "scan", "not-answerable", "general"):
        print(source, sum(item["source"] == source for item in questions))
    print("total", len(questions))
    print("saved", OUT)
