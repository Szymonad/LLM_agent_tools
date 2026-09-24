"""Terminal agent that answers questions about the PDFs in the index."""
import logging
from pathlib import Path

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from config import HTTP_TIMEOUT, SERVER
from index import search
from tools import BEHAVIOUR, expand_citations, format_chunks, sources_of

logging.basicConfig(
    # Next to this file, not in whatever folder the terminal happens to be in.
    filename=Path(__file__).with_name("agent.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("agent_lg_pdf_search")

MODEL = ChatOpenAI(
    base_url=f"{SERVER}",
    # llama-server runs without --api-key and serves one model, so neither value is checked.
    api_key="unused",
    model="qwen3",
    # The answer must come from the fragments, so there is nothing to be creative about.
    temperature=0,
    timeout=HTTP_TIMEOUT,
)


class State(MessagesState):
    """Conversation, the chunks found for the last question and where each of them came from."""

    context: str
    sources: list[str]


def retrieve(state):
    """Searches the index with the last question and puts the chunks in the state."""
    chunks = search(state["messages"][-1].content)
    return {"context": format_chunks(chunks), "sources": sources_of(chunks)}


def answer(state):
    """Asks the model with the chunks from retrieve; the answer goes back into messages."""
    messages = [SystemMessage(BEHAVIOUR), SystemMessage(state["context"]), *state["messages"]]
    reply = MODEL.invoke(messages)
    # The numbers mean different fragments next turn, so the history keeps the real sources.
    reply.content = expand_citations(reply.content, state["sources"])
    return {"messages": [reply]}


def build_graph():
    """The whole agent: every question goes through retrieve and then answer."""
    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    # The checkpointer keeps the conversation between questions, under the thread id below. rest are defaut values (for learning)
    return graph.compile(
    checkpointer=InMemorySaver(),
    cache=None,
    store=None,
    interrupt_before=None,
    interrupt_after=None,
    debug=False,
    name=None,
    transformers=None,
    )



def context_limit():
    """The context window the chat server was started with, read once per run."""
    response = requests.get(f"{SERVER}/props", timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return response.json()["default_generation_settings"]["n_ctx"]


def print_state(state):
    """The state in a readable form: what is in it, of what type and how big."""
    print(f"state: {len(state['messages'])} messages, context {len(state.get('context', ''))} chars")
    for message in state["messages"]:
        text = message.content.replace("\n", " ")
        print(f"  {type(message).__name__:13} str {len(message.content):5} chars | {text[:60]}")
    for number, source in enumerate(state.get("sources", []), start=1):
        print(f"  chunk [{number}]     {source}")
            

THREAD = {"configurable": {"thread_id": "agent_lg_pdf_search"}}


def ask(agent, question):
    """One turn, logged step by step; returns the state after the turn."""
    log.info("question: %s", question)
    for step in agent.stream({"messages": [HumanMessage(question)]}, THREAD, stream_mode="updates"):
        for node, update in step.items():
            if node == "retrieve":
                log.info("chunks: %s", " | ".join(update["sources"]))
            if node == "answer":
                reply = update["messages"][-1]
                usage = reply.usage_metadata or {}
                log.info("answer: %s", reply.content.replace("\n", " "))
                log.info(
                    "tokens: %s in + %s out, finish=%s",
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                    reply.response_metadata.get("finish_reason"),
                )
    return agent.get_state(THREAD).values


def main():
    """Terminal loop; an empty line ends it."""
    agent = build_graph()
    used = 0
    context_lim = context_limit()
    while True:
        question = input(f"\n{used}/{context_lim} > ").strip()
        if not question:
            break
        state = ask(agent, question)
        print(state["messages"][-1].content)
        print('=======================')
        print_state(state)
        print('=======================')
        # What the last turn cost, shown in the prompt before the next question.
        used = (state["messages"][-1].usage_metadata or {}).get("total_tokens", 0)


if __name__ == "__main__":
    main()
