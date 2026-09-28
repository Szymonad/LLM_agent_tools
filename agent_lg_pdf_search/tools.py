"""Contract between the model and the PDFs: system prompt and the chunks formatted for it."""

BEHAVIOUR = """You answer questions about PDF documents. Answer in the language of the question.

Use only the numbered fragments below. They are the only thing you know about these documents.
If they do not contain the answer, say that the documents do not cover it; never fill the gap
from your own knowledge.

After every fact write the number of the fragment it came from, in square brackets, like [2].
Never write anything else inside square brackets."""

REPHRASE = """You rewrite questions into search queries. You never answer them.

Given the conversation, rewrite the last user question so that it can be understood on its own,
filling in names the question leaves out. Always write the query in English, even when the question
is in another language, and name the subject instead of writing "it" or "this".
Keep it under 15 words. Write only the query."""


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


from langchain_core.documents import Document

def main():
    chunks = [
    Document(page_content='EmbeddingGemma (768d) 578 308M 16 69.7 65.1',
             metadata={'source': 'EmbeddingGemma_technical_report.pdf', 'pages': [10]}),
    Document(page_content='Gaussian filters are standardized in ISO 16610-21.',
             metadata={'source': 'Selected_Filtration_Methods_of_ISO-16610.pdf', 'pages': [27, 28]}),
    Document(page_content='Mieszanie liniowe obrazow (tzw. mieszanie alfa).',
             metadata={'source': 'CPIK W06.pdf', 'pages': [2]}),
]
    a = format_chunks(chunks)
    b = sources_of(chunks)
    print(b)

if __name__ == "__main__":
    main()