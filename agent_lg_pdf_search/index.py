"""Build and query the vector index of PDF chunks.

Reading PDFs, splitting them into token-sized chunks, embedding the chunks
with EmbeddingGemma on llama-server and searching the saved index.
"""

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
    """Read a PDF into one Document per page.

    Pages without extractable text are skipped and reported on stdout.

    Args:
        path (pathlib.Path): PDF file to read.

    Returns:
        list[Document]: Pages in file order. Metadata holds ``source``
        (file name) and ``page`` (1-based page number).
    """
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
    """Count the tokens of a text as the embedding server sees it.

    Results are cached, because the splitter measures the same pieces many
    times.

    Args:
        text (str): Text to tokenize.

    Returns:
        int: Number of tokens.

    Raises:
        requests.HTTPError: If the server answers with an error status.
        requests.ConnectionError: If the server cannot be reached.
        requests.Timeout: If the server does not answer within
            ``HTTP_TIMEOUT``.
    """
    response = requests.post(f"{EMBED_SERVER}/tokenize", json={"content": text}, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return len(response.json()["tokens"])


def join_pages(pages):
    """Join pages into one text and record where each page lies in it.

    Args:
        pages (list[Document]): Pages as returned by ``read_pages``.

    Returns:
        tuple[str, list[tuple[int, int, int]]]: The joined text and one
        ``(page number, start, end)`` span per page, as character offsets
        into that text.
    """
    text = ""
    spans = []
    for page in pages:
        spans.append((page.metadata["page"], len(text), len(text) + len(page.page_content)))
        text += page.page_content + "\n"
    return text, spans


def split(pages):
    """Split the pages of one PDF into overlapping chunks.

    The whole PDF is cut as one text, so the overlap also crosses page
    boundaries.

    Args:
        pages (list[Document]): Pages of a single PDF, as returned by
            ``read_pages``.

    Returns:
        list[Document]: Chunks in text order. Metadata holds ``source``
        (file name) and ``pages`` (numbers of the pages the chunk covers).
        Empty if ``pages`` is empty.
    """
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
    """Embed texts with EmbeddingGemma on llama-server.

    EmbeddingGemma expects a different prefix on passages and on queries,
    so each method adds its own before sending the text.
    """

    def embed_documents(self, texts, chunk_size=None, **kwargs):
        """Embed passages, each with ``PASSAGE_PREFIX`` in front.

        Args:
            texts (list[str]): Passages to embed.
            chunk_size (int | None): Number of texts sent per request.
                ``None`` keeps the default of ``OpenAIEmbeddings``.
            **kwargs: Passed on to ``OpenAIEmbeddings.embed_documents``.

        Returns:
            list[list[float]]: One vector per passage, in input order.
        """
        return super().embed_documents([PASSAGE_PREFIX + text for text in texts], chunk_size, **kwargs)

    def embed_query(self, text, **kwargs):
        """Embed a search query with ``QUERY_PREFIX`` in front.

        Args:
            text (str): Query to embed.
            **kwargs: Passed on to ``OpenAIEmbeddings.embed_documents``.

        Returns:
            list[float]: Vector of the query.
        """
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
    """Build the index from every PDF in ``PDF_DIR`` and write it to disk.

    Each file is read, split into chunks and embedded. All chunks go into
    one index, saved at ``INDEX_PATH``. A summary line per file is printed
    on stdout.

    Returns:
        InMemoryVectorStore: The index that was written.
    """
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
    """Load the index written by ``build``.

    The result is cached, so the file at ``INDEX_PATH`` is read once per
    run. No PDF is read and nothing is split or embedded here.

    Returns:
        InMemoryVectorStore: The loaded index.
    """
    return InMemoryVectorStore.load(str(INDEX_PATH), EMBEDDINGS)


def search(query, k=TOP_K):
    """Find the chunks closest to a query.

    Args:
        query (str): Question to search for.
        k (int): Number of chunks to return.

    Returns:
        list[Document]: Up to ``k`` chunks, best match first.
    """
    return load_store().similarity_search(query, k=k)


def search_scored(query, k=TOP_K):
    """Find the chunks closest to a query, with their similarity.

    Args:
        query (str): Question to search for.
        k (int): Number of chunks to return.

    Returns:
        list[tuple[Document, float]]: Up to ``k`` pairs of chunk and
        cosine similarity, best match first.
    """
    return load_store().similarity_search_with_score(query, k=k)


# if __name__ == "__main__":
#     if not INDEX_PATH.exists():
#         build()
#     question = "what is EmbeddingGemma model"
#     for document, score in load_store().similarity_search_with_score(question, k=TOP_K):
#         print(f"{score:.3f} pdf {document.metadata['source']} pages {document.metadata['pages']}: {document.page_content[:50]!r}")

