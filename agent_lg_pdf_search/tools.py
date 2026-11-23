"""Contract between the model and the PDFs: system prompts, the chunks formatted for them and the tools."""

from functools import lru_cache

from langchain_core.tools import tool

from agent_lg_pdf_search import index

MAX_TOOL_CHARS = 4000
MAX_FOUND_PAGES = 20
SNIPPET_MARGIN = 40

BEHAVIOUR = """You answer questions about PDF documents. Answer in the language of the question.

Use only the numbered fragments below. They are the only thing you know about these documents.
If they do not contain the answer, say that the documents do not cover it; never fill the gap
from your own knowledge.

After every fact write the number of the fragment it came from, in square brackets, like [2].
Never write anything else inside square brackets."""

ROUTER = """Do not answer the last question. The user has a collection of PDF documents: reports,
papers, manuals, guidebooks, presentations and notes. Decide whether the last question needs them.

Write general only when the question clearly needs no document: greetings and small talk,
translation, arithmetic, writing code, or textbook common knowledge such as capitals,
authors of famous books, dates of famous historical events and basic science.
Write documents for everything else: any question that mentions a document, report, page,
figure, table or chart; any question about statistics, numbers, money, organisations,
companies, products, places, contact details or rules; anything that could be in the documents.
When in doubt, write documents.

Write only one word: documents or general."""

GENERAL_BEHAVIOUR = """You are a helpful assistant. Answer in the language of the question,
briefly, from your own knowledge. Do not mention documents."""

#: Put in front of every answer that does not come from the documents.
GENERAL_MARK = "(z wiedzy ogólnej, nie z dokumentów)"

REPHRASE = """Rewrite the last user question as a standalone English search query. Never answer it.

Use the conversation to work out what the question is about, but add nothing that is not in it.
Write only the query."""

TOOL_BEHAVIOUR = """You answer questions about PDF documents. Answer in the language of the question.

Use only the results of the tools. They are the only thing you know about these documents.
Start with search_documents. When the question points at a page, read it with get_page.
When the question asks to count occurrences or needs an exact string such as an email or a number,
use find_text. Do not guess file names; list_documents shows them.

End the answer with the source: file name and page. If the tools do not give the answer, say that
the documents do not cover it; never fill the gap from your own knowledge."""


def format_chunks(chunks):
    """Numbered fragments for the prompt; the model cites numbers, never file names."""
    return "\n\n".join(f"[{number}]\n{chunk.page_content}" for number, chunk in enumerate(chunks, start=1))


def sources_of(chunks):
    """The real source of each fragment, in the order the model sees them."""
    return [
        f"{chunk.metadata['source']} p. {', '.join(str(page) for page in chunk.metadata['pages'])}"
        for chunk in chunks
    ]


def expand_citations(answer, sources):
    """Puts the file name and pages of fragment 2 in place of the [2] written by the model."""
    for number, source in enumerate(sources, start=1):
        answer = answer.replace(f"[{number}]", f"[{source}]")
    return answer


def pdf_names():
    """File names of the PDFs in the current ``index.PDF_DIR``, sorted."""
    return sorted(path.name for path in index.PDF_DIR.glob("*.pdf"))


@lru_cache(maxsize=None)
def pages_of(document):
    """Text of every page of a PDF, cached per file name.

    Pages without text are missing from the result.

    Args:
        document (str): File name from ``pdf_names``.

    Returns:
        dict[int, str]: Page number (1-based) to the text of the page.
    """
    return {page.metadata["page"]: page.page_content for page in index.read_pages(index.PDF_DIR / document)}


def unknown_document(document):
    """Message for the model when the file name matches no PDF."""
    return f"No document named '{document}'. Available documents: {', '.join(pdf_names())}"


@tool(response_format="content_and_artifact")
def search_documents(query: str):
    """Find the passages of the documents closest in meaning to the query.

    Args:
        query: Short English search query.
    """
    hits = index.search_scored(query)
    content = "\n\n".join(
        f"[{number}] {chunk.metadata['source']} p. {', '.join(str(page) for page in chunk.metadata['pages'])}\n"
        f"{chunk.page_content}"
        for number, (chunk, score) in enumerate(hits, start=1)
    )
    sources = [{"source": chunk.metadata["source"], "pages": chunk.metadata["pages"]} for chunk, score in hits]
    return content, sources


@tool(response_format="content_and_artifact")
def get_page(document: str, page: int):
    """Read the text of one page of a document.

    Args:
        document: File name of the PDF, e.g. report.pdf.
        page: Page number, starting from 1.
    """
    if document not in pdf_names():
        return unknown_document(document), []
    pages = pages_of(document)
    if page not in pages:
        return f"No text on page {page} of {document}. Pages with text: {min(pages)}-{max(pages)}", []
    return pages[page][:MAX_TOOL_CHARS], [{"source": document, "pages": [page]}]


@tool(response_format="content_and_artifact")
def find_text(document: str, text: str):
    """Count the occurrences of an exact text in a document, ignoring case, and list the pages with it.

    Args:
        document: File name of the PDF, e.g. report.pdf.
        text: Text to look for, matched literally.
    """
    if document not in pdf_names():
        return unknown_document(document), []
    needle = text.lower()
    found = []
    for number, page_text in pages_of(document).items():
        lowered = page_text.lower()
        count = lowered.count(needle)
        if count:
            start = lowered.index(needle)
            snippet = page_text[max(0, start - SNIPPET_MARGIN) : start + len(needle) + SNIPPET_MARGIN]
            found.append((number, count, " ".join(snippet.split())))
    if not found:
        return f"'{text}' does not occur in {document}.", []
    total = sum(count for number, count, snippet in found)
    lines = [f"'{text}' occurs {total} times in {document}, on {len(found)} pages."]
    lines += [f"p. {number}: {count}x, ...{snippet}..." for number, count, snippet in found[:MAX_FOUND_PAGES]]
    if len(found) > MAX_FOUND_PAGES:
        lines.append(f"Only the first {MAX_FOUND_PAGES} pages are listed.")
    sources = [{"source": document, "pages": [number]} for number, count, snippet in found[:MAX_FOUND_PAGES]]
    return "\n".join(lines), sources


@tool(response_format="content_and_artifact")
def list_documents():
    """List the file names of all available documents."""
    return "\n".join(pdf_names()), []


# from langchain_core.documents import Document

# def main():
#     chunks = [
#     Document(page_content='EmbeddingGemma (768d) 578 308M 16 69.7 65.1',
#              metadata={'source': 'EmbeddingGemma_technical_report.pdf', 'pages': [10]}),
#     Document(page_content='Gaussian filters are standardized in ISO 16610-21.',
#              metadata={'source': 'Selected_Filtration_Methods_of_ISO-16610.pdf', 'pages': [27, 28]}),
#     Document(page_content='Mieszanie liniowe obrazow (tzw. mieszanie alfa).',
#              metadata={'source': 'CPIK W06.pdf', 'pages': [2]}),
# ]
#     a = format_chunks(chunks)
#     b = sources_of(chunks)
#     print(b)

# if __name__ == "__main__":
#     main()