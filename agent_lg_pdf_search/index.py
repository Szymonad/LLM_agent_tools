from langchain_core.documents import Document
from pypdf import PdfReader

from config import PDF_PATH


def read_pages():
    """One Document per PDF page, with its 1-based page number in metadata."""
    reader = PdfReader(PDF_PATH)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text.strip():
            print(f"page {number}: no text, skipped")
            continue
        pages.append(Document(page_content=text, metadata={"page": number}))
    return pages


if __name__ == "__main__":
    for page in read_pages():
        print(page.metadata["page"], len(page.page_content), repr(page.page_content[:120]))
