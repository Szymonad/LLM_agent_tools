import time
from functools import lru_cache

import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_TOKENS, EMBED_SERVER, HTTP_TIMEOUT, PDF_PATH


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

@lru_cache(maxsize=None)
def count_tokens(text):
    """Token count as the embedding server sees it; cached because the splitter measures the same pieces many times."""
    response = requests.post(f"{EMBED_SERVER}/tokenize", json={"content": text}, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return len(response.json()["tokens"])


def split(pages):
    """Whole PDF cut as one text, so the overlap also crosses page boundaries; each chunk lists its pages."""
    text = "\n".join(page.page_content for page in pages)
    # Where each page starts in text, to map a chunk back to the pages it came from.
    starts = []
    position = 0
    for page in pages:
        starts.append(position)
        position += len(page.page_content) + 1  # +1 for the "\n" that joins the pages

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS, chunk_overlap=CHUNK_OVERLAP, length_function=count_tokens
    )
    chunks = []
    start = -1
    for piece in splitter.split_text(text):
        # Own search instead of add_start_index: it mixes characters with our token overlap
        # and put 26 of 60 chunks at the wrong position.
        start = text.find(piece, start + 1)
        end = start + len(piece)
        numbers = [
            page.metadata["page"]
            for page, page_start in zip(pages, starts)
            if page_start < end and page_start + len(page.page_content) > start
        ]
        chunks.append(Document(page_content=piece, metadata={"pages": numbers}))
    return chunks


if __name__ == "__main__":
    start = time.perf_counter()
    chunks = split(read_pages())
    print(f"read_pages and split: {time.perf_counter() - start:.1f} s")
    # enumerate numbers the pairs from 1: chunk 1 -> 2, chunk 2 -> 3, ...
    for number, (previous, current) in enumerate(zip(chunks, chunks[1:]), start=1):
        # Pages are split separately, so there is no overlap across them.
        if previous.metadata["page"] != current.metadata["page"]:
            print(f"chunk {number} -> {number + 1}: 0 shared tokens (different pages)")
            continue
        a, b = previous.page_content, current.page_content
        # Longest end of a that is also the start of b; trying the longest first, so the first hit wins.
        shared = ""
        for size in range(min(len(a), len(b)), 0, -1):
            if a.endswith(b[:size]):
                shared = b[:size]
                break
        print(f"chunk {number} -> {number + 1}: {count_tokens(shared)} shared tokens")