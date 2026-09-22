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
    """Pages cut into chunks the embedding server accepts; each chunk keeps its page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS, chunk_overlap=CHUNK_OVERLAP, length_function=count_tokens
    )
    return splitter.split_documents(pages)


if __name__ == "__main__":
    chunks = split(read_pages())
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