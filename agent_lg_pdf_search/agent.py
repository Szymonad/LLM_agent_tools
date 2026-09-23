"""Terminal agent that answers questions about the PDFs in the index."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import HTTP_TIMEOUT, SERVER
from index import search
from tools import BEHAVIOUR, format_chunks

MODEL = ChatOpenAI(
    base_url=f"{SERVER}/v1",
    # llama-server runs without --api-key and serves one model, so neither value is checked.
    api_key="unused",
    model="qwen3",
    # The answer must come from the fragments, so there is nothing to be creative about.
    temperature=0,
    timeout=HTTP_TIMEOUT,
)


def ask(question):
    """One turn: find the chunks, hand them to the model, return its answer."""
    chunks = search(question)
    messages = [
        SystemMessage(BEHAVIOUR),
        HumanMessage(f"{format_chunks(chunks)}\n\nQuestion: {question}"),
    ]
    return MODEL.invoke(messages).content


if __name__ == "__main__":
    print(ask("how many parameters does EmbeddingGemma have?"))
