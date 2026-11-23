"""Build and query the vector index of PDF chunks.

Reading PDFs with Docling (text layer, OCR of scans and pictures, tables as
Markdown), splitting them into token-sized chunks, embedding the chunks with
EmbeddingGemma on llama-server and searching the saved index.
"""

import re
import time
from functools import lru_cache

import requests
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdfium2

from agent_lg_pdf_search.config import (
    CHUNK_OVERLAP,
    CHUNK_TOKENS,
    DOCLING_THREADS,
    EMBED_DIM,
    EMBED_SERVER,
    HTTP_TIMEOUT,
    INDEX_PATH,
    MIN_PICTURE_AREA,
    TOP_K,
    PARSER,
    PASSAGE_PREFIX,
    PDF_DIR,
    QUERY_PREFIX,
    SCAN_TEXT_CHARS,
)

PLACEHOLDER = "<!-- image -->"
#: Start of every picture description in the text, e.g. ``[Rysunek, str. 12]``.
PICTURE_MARK = "[Rysunek, str."

@lru_cache(maxsize=2)
def converter(pictures=False):
    """Docling converter, built once: loading its models takes seconds.

    OCR runs on the bitmaps of a page (the whole page for a scan) with
    RapidOCR, which reads Polish diacritics; tables come out as Markdown.
    With ``pictures`` the images of the pictures are kept, for ``captions``;
    without, reading is lighter (no page renders at twice the size kept in memory).
    Docling is imported here: it pulls in torch, which the agent answering
    questions from a ready index does not need.
    """
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=True, do_table_structure=True, ocr_options=RapidOcrOptions())
    options.accelerator_options = AcceleratorOptions(device="cpu", num_threads=DOCLING_THREADS)
    if pictures:
        options.generate_picture_images = True
        # Twice the PDF resolution, so the vision model can read small chart labels.
        options.images_scale = 2.0
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def text_layer_sizes(path):
    """Characters (without white space) in the text layer of every page, 1-based."""
    document = pypdfium2.PdfDocument(path)
    try:
        return {
            number: len("".join(page.get_textpage().get_text_range().split()))
            for number, page in enumerate(document, start=1)
        }
    finally:
        document.close()


def large_pictures(document, number):
    """Pictures of a page in reading order, with ``None`` for the small ones."""
    from docling_core.types.doc import PictureItem

    page = document.pages[number].size
    pictures = []
    for item, level in document.iterate_items(page_no=number):
        if not isinstance(item, PictureItem):
            continue
        box = item.prov[0].bbox
        share = abs(box.width * box.height) / (page.width * page.height)
        pictures.append(item if share >= MIN_PICTURE_AREA else None)
    return pictures


def describe_pictures(text, document, name, number, cache):
    """Put a description of every large picture of a page where it stands.

    Args:
        text (str): Markdown of the page with ``PLACEHOLDER`` for each picture.
        document (DoclingDocument): The converted PDF.
        name (str): File name, for the message when the pictures do not match.
        number (int): Page number.
        cache (captions.CaptionCache): Descriptions, and the model for new ones.

    Returns:
        tuple[str, list[str]]: The text with each placeholder replaced by
        ``[Rysunek, str. N] description``, or removed for a small picture or
        an empty description; and the descriptions put in. If the
        placeholders do not match the pictures one to one, they are removed
        and the descriptions go to the end of the page.
    """
    pictures = large_pictures(document, number)
    descriptions = []
    for picture in pictures:
        caption = cache.caption(picture.get_image(document)) if picture is not None else ""
        descriptions.append(f"{PICTURE_MARK} {number}] {caption}" if caption else "")
    parts = text.split(PLACEHOLDER)
    if len(parts) != len(descriptions) + 1:
        print(f"{name} page {number}: {len(parts) - 1} placeholders, {len(descriptions)} pictures, descriptions put at the end")
        text = "\n\n".join([text.replace(PLACEHOLDER, ""), *filter(None, descriptions)])
    else:
        text = "".join(part + description for part, description in zip(parts, [*descriptions, ""]))
    return text, [description for description in descriptions if description]


def read_pages(path, cache=None):
    """Read a PDF into one Document per page, as Markdown in reading order.

    Pages that give no text even after OCR are skipped and reported on
    stdout.

    Args:
        path (pathlib.Path): PDF file to read.
        cache (captions.CaptionCache | None): Picture descriptions. ``None``
            drops the pictures, so reading needs no vision server.

    Returns:
        list[Document]: Pages in file order. Metadata holds ``source``
        (file name), ``page`` (1-based page number), ``origin``: ``ocr``
        for a page with no text layer, otherwise ``text``, and
        ``pictures``: ``(start, end)`` of every description in the text.
    """
    result = converter(pictures=cache is not None).convert(path)
    if result.status.name != "SUCCESS":
        print(f"{path.name}: {result.status.name}, {[error.error_message for error in result.errors]}")
    document = result.document
    sizes = text_layer_sizes(path)
    missing = sorted(set(sizes) - set(document.pages))
    if missing:
        print(f"{path.name} pages {missing}: not converted, skipped")
    scans = {number for number, size in sizes.items() if size < SCAN_TEXT_CHARS}
    pages = []
    for number in sorted(document.pages):
        # Unescaped, so "R&D" and "first_last" stay as in the PDF; compact tables, so padding does not fill the chunks.
        text = document.export_to_markdown(
            page_no=number, escape_underscores=False, escape_html=False, image_placeholder=PLACEHOLDER, compact_tables=True
        )
        descriptions = []
        if cache is not None:
            text, descriptions = describe_pictures(text, document, path.name, number, cache)
        # A removed placeholder leaves its blank lines behind.
        text = re.sub(r"\n{3,}", "\n\n", text.replace(PLACEHOLDER, "")).strip()
        if not text:
            print(f"{path.name} page {number}: no text, skipped")
            continue
        origin = "ocr" if number in scans else "text"
        spans, start = [], 0
        for description in descriptions:
            start = text.find(description, start)
            spans.append((start, start + len(description)))
            start += len(description)
        metadata = {"source": path.name, "page": number, "origin": origin, "pictures": spans}
        pages.append(Document(page_content=text, metadata=metadata))
    return pages

SESSION = requests.Session()

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
    response = SESSION.post(f"{EMBED_SERVER}/tokenize", json={"content": text}, timeout=HTTP_TIMEOUT)
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
        (file name), ``pages`` (numbers of the pages the chunk covers) and
        ``origin``: ``ocr`` if any of those pages is a scan, else ``image``
        if the chunk holds a picture description, else ``text``.
        Empty if ``pages`` is empty.
    """
    if not pages:
        return []
    source = pages[0].metadata["source"]
    text, spans = join_pages(pages)
    scans = {page.metadata["page"] for page in pages if page.metadata["origin"] == "ocr"}
    # Where the picture descriptions lie in the joined text; a long one spans several chunks.
    pictures = [
        (page_start + start, page_start + end)
        for page, (number, page_start, page_end) in zip(pages, spans)
        for start, end in page.metadata.get("pictures", [])
    ]

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
        if scans.intersection(numbers):
            origin = "ocr"
        elif any(picture_start < end and picture_end > start for picture_start, picture_end in pictures):
            origin = "image"
        else:
            origin = "text"
        chunks.append(Document(page_content=chunk_text, metadata={"source": source, "pages": numbers, "origin": origin}))
    return chunks


class PrefixedEmbeddings(OpenAIEmbeddings):
    """Embed texts with EmbeddingGemma on llama-server.

    EmbeddingGemma expects a different prefix on passages and on queries,
    so each method adds its own before sending the text. The server always
    returns 768 numbers; each method keeps the first ``EMBED_DIM`` of
    them, which EmbeddingGemma supports for 512, 256 and 128.
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
        vectors = super().embed_documents([PASSAGE_PREFIX + text for text in texts], chunk_size, **kwargs)
        return [vector[:EMBED_DIM] for vector in vectors]

    def embed_query(self, text, **kwargs):
        """Embed a search query with ``QUERY_PREFIX`` in front.

        Args:
            text (str): Query to embed.
            **kwargs: Passed on to ``OpenAIEmbeddings.embed_documents``.

        Returns:
            list[float]: Vector of the query.
        """
        # The parent's embed_query goes through embed_documents above and would add PASSAGE_PREFIX as well.
        return super().embed_documents([QUERY_PREFIX + text], **kwargs)[0][:EMBED_DIM]


EMBEDDINGS = PrefixedEmbeddings(
    base_url=f"{EMBED_SERVER}/v1",
    # llama-server runs without --api-key and serves one model, so neither value is checked.
    api_key="unused",
    model="embeddinggemma",
    # Otherwise LangChain tokenizes with OpenAI's tiktoken and sends token ids that mean nothing to Gemma.
    check_embedding_ctx_length=False,
    chunk_size=64,
    timeout=HTTP_TIMEOUT,
)


def build():
    """Build the index from every PDF in ``PDF_DIR`` and write it to disk.

    Each file is read, split into chunks and embedded. All chunks go into
    one index, saved at ``INDEX_PATH``. Pictures get descriptions from the
    vision server, kept in ``captions.json`` next to the index. A summary
    line per file is printed on stdout.

    Returns:
        InMemoryVectorStore: The index that was written.
    """
    from agent_lg_pdf_search import captions

    cache = captions.CaptionCache(INDEX_PATH.with_name("captions.json"))
    chunks = []
    try:
        for path in sorted(PDF_DIR.glob("*.pdf")):
            # Each file is split on its own, so a chunk never joins the end of one PDF with the start of another.
            pages = read_pages(path, cache)
            file_chunks = split(pages)
            print(f"{path.name}: {len(pages)} pages, {len(file_chunks)} chunks, {cache.new} new descriptions so far")
            chunks += file_chunks
    finally:
        # Descriptions cost GPU time; keep them even when the build stops halfway.
        cache.save()
    store = InMemoryVectorStore(EMBEDDINGS)
    # add_documents is what calls EMBEDDINGS.embed_documents under the hood.
    store.add_documents(chunks)
    store.dump(str(INDEX_PATH))
    parser_file().write_text(PARSER, encoding="utf-8")
    return store


def parser_file():
    """File next to the index naming the parser it was built with."""
    return INDEX_PATH.with_name(INDEX_PATH.stem + ".parser.txt")


def index_parser():
    """Parser the index at ``INDEX_PATH`` was built with; ``pypdf`` for an index older than this file."""
    path = parser_file()
    return path.read_text(encoding="utf-8").strip() if path.exists() else "pypdf"


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


if __name__ == "__main__":
    if not INDEX_PATH.exists():
        build()
    question = "what is EmbeddingGemma model"
    for document, score in load_store().similarity_search_with_score(question, k=TOP_K):
        print(f"{score:.3f} pdf {document.metadata['source']} pages {document.metadata['pages']}: {document.page_content[:50]!r}")

