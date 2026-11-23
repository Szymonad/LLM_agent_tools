"""Stage 2 test: Docling speed per page and OCR quality (Polish + English) on simulated scans."""
import difflib
import json
import re
import sys
import time
from pathlib import Path

import pypdfium2
from pypdf import PdfReader

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions, RapidOcrOptions, OcrMode
from docling.document_converter import DocumentConverter, PdfFormatOption

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(exist_ok=True)
DEVICE = sys.argv[3] if len(sys.argv) > 3 else "cpu"

SAMPLES = [
    (REPO / "agent_lg_pdf_search/pdf/CPIK W06.pdf", "pl"),
    (REPO / "agent_lg_pdf_search/pdf/struktury-json.pdf", "pl"),
    (REPO / "agent_lg_pdf_search/eval/pdf_originals/f8d3a162ab9507e021d83dd109118b60.pdf", "en"),
]


def make_scan(source, target, dpi=150):
    document = pypdfium2.PdfDocument(source)
    images = [page.render(scale=dpi / 72).to_pil().convert("RGB") for page in document]
    document.close()
    images[0].save(target, "PDF", save_all=True, append_images=images[1:], resolution=dpi, quality=80)


def words(text):
    return re.findall(r"\w+", text.lower())


def similarity(reference, text):
    return difflib.SequenceMatcher(None, words(reference), words(text), autojunk=False).ratio()


def diacritics(text):
    return len(re.findall(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]", text))


def converter(ocr):
    options = PdfPipelineOptions(do_ocr=ocr is not None, do_table_structure=True)
    options.accelerator_options = AcceleratorOptions(device=DEVICE, num_threads=8)
    if ocr is not None:
        options.ocr_options = ocr
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def page_texts(result):
    document = result.document
    return {number: document.export_to_markdown(page_no=number) for number in sorted(document.pages)}


def run(name, conv, path):
    started = time.perf_counter()
    result = conv.convert(path)
    seconds = time.perf_counter() - started
    return page_texts(result), seconds


report = []
engines = {
    "easyocr_pl_en": EasyOcrOptions(lang=["pl", "en"], mode=OcrMode.FULL_PAGE),
    "rapidocr_default": RapidOcrOptions(mode=OcrMode.FULL_PAGE),
}
converters = {name: converter(options) for name, options in engines.items()}
text_converter = converter(None)
for source, language in SAMPLES:
    reference = {number: page.extract_text() for number, page in enumerate(PdfReader(source).pages, start=1)}
    scan = OUT / f"scan_{source.name}"
    if not scan.exists():
        make_scan(source, scan)
    pages, seconds = run("text", text_converter, source)
    row = {"file": source.name, "lang": language, "pages": len(reference), "text_layer_s_per_page": round(seconds / len(reference), 2),
           "text_layer_similarity": round(similarity(" ".join(reference.values()), " ".join(pages.values())), 3)}
    for name, conv in converters.items():
        pages, seconds = run(name, conv, scan)
        joined = " ".join(pages.values())
        row[name] = {
            "s_per_page": round(seconds / len(reference), 2),
            "similarity": round(similarity(" ".join(reference.values()), joined), 3),
            "diacritics": diacritics(joined),
            "reference_diacritics": diacritics(" ".join(reference.values())),
        }
        (OUT / f"{source.stem}.{name}.md").write_text("\n\n---\n\n".join(pages.values()), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False), flush=True)
    report.append(row)
(OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
