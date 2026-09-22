import time
from functools import lru_cache

import requests
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import (
    CHUNK_OVERLAP,
    CHUNK_TOKENS,
    EMBED_SERVER,
    HTTP_TIMEOUT,
    INDEX_PATH,
    TOP_K,
    PASSAGE_PREFIX,
    PDF_DIR,
    QUERY_PREFIX,
)


def read_pages(path):
    """One Document per PDF page, with the file name and its 1-based page number in metadata."""
    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text.strip():
            print(f"{path.name} page {number}: no text, skipped")
            continue
        pages.append(Document(page_content=text, metadata={"source": path.name, "page": number}))
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
    if not pages:
        return []
    source = pages[0].metadata["source"]
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
        chunks.append(Document(page_content=chunk_text, metadata={"source": source, "pages": numbers}))
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


def build():
    """Reads every PDF in PDF_DIR, cuts them into chunks, embeds them and writes one index to disk."""
    chunks = []
    for path in sorted(PDF_DIR.glob("*.pdf")):
        # Each file is split on its own, so a chunk never joins the end of one PDF with the start of another.
        pages = read_pages(path)
        file_chunks = split(pages)
        print(f"{path.name}: {len(pages)} pages, {len(file_chunks)} chunks")
        chunks += file_chunks
    store = InMemoryVectorStore(EMBEDDINGS)
    # add_documents is what calls EMBEDDINGS.embed_documents under the hood.
    store.add_documents(chunks)
    store.dump(str(INDEX_PATH))
    return store


@lru_cache(maxsize=1)
def load_store():
    """The index built earlier, read once per run; the agent needs no PDF and no splitting."""
    return InMemoryVectorStore.load(str(INDEX_PATH), EMBEDDINGS)


def search(query, k=TOP_K):
    """The k chunks closest to the question, best first."""
    return load_store().similarity_search(query, k=k)


if __name__ == "__main__":
    if not INDEX_PATH.exists():
        build()
    question = "jakie są wnioski wynikające z tego bania i pomiarów?"
    for document, score in load_store().similarity_search_with_score(question, k=TOP_K):
        print(f"{score:.3f} pages {document.metadata['pages']}: {document.page_content[:80]!r}")

