import time
from functools import lru_cache

import requests
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import (
    CHUNK_OVERLAP,
    CHUNK_TOKENS,
    EMBED_SERVER,
    HTTP_TIMEOUT,
    PASSAGE_PREFIX,
    PDF_PATH,
    QUERY_PREFIX,
)


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


def join_pages(pages):
    """All pages as one text, plus where each page lies in it: (page number, start, end)."""
    text = ""
    spans = []
    for page in pages:
        spans.append((page.metadata["page"], len(text), len(text) + len(page.page_content)))
        text += page.page_content + "\n"
    return text, spans


def split(pages):
    """Whole PDF cut as one text, so the overlap also crosses page boundaries; each chunk lists its pages."""
    text, spans = join_pages(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS, chunk_overlap=CHUNK_OVERLAP, length_function=count_tokens
    )
    chunks = []
    start = 0
    for chunk_text in splitter.split_text(text):
        start = text.find(chunk_text, start)
        end = start + len(chunk_text)
        print(start, end)
        numbers = [number for number, page_start, page_end in spans if page_start < end and page_end > start]
        chunks.append(Document(page_content=chunk_text, metadata={"pages": numbers}))
    return chunks


class PrefixedEmbeddings(OpenAIEmbeddings):
    """EmbeddingGemma on llama-server, with the prefix it expects on each passage and query."""

    def embed_documents(self, texts, chunk_size=None, **kwargs):
        return super().embed_documents([PASSAGE_PREFIX + text for text in texts], chunk_size, **kwargs)

    def embed_query(self, text, **kwargs):
        # The parent's embed_query goes through embed_documents above and would add PASSAGE_PREFIX as well.
        return super().embed_documents([QUERY_PREFIX + text], **kwargs)[0]


EMBEDDINGS = PrefixedEmbeddings(
    base_url=f"{EMBED_SERVER}/v1",
    # llama-server runs without --api-key and serves one model, so neither value is checked.
    api_key="unused",
    model="embeddinggemma",
    # Otherwise LangChain tokenizes with OpenAI's tiktoken and sends token ids that mean nothing to Gemma.
    check_embedding_ctx_length=False,
    timeout=HTTP_TIMEOUT,
)


if __name__ == "__main__":
    chunks = split(read_pages())
    start = time.perf_counter()
    vectors = EMBEDDINGS.embed_documents([chunk.page_content for chunk in chunks])
    print(f"embed: {len(vectors)} vectors x {len(vectors[0])} numbers, {time.perf_counter() - start:.1f} s")
    # The server returns vectors of length 1, which is what lets search use a plain dot product.
    print("length of the first vector:", round(sum(x * x for x in vectors[0]) ** 0.5, 4))