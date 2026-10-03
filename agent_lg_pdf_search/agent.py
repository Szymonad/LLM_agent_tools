"""Terminal agent that answers questions about the PDFs in the index."""
# import operator
# from typing import Annotated

import logging
import time
from pathlib import Path

import requests
from langchain_core.globals import set_debug
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from agent_lg_pdf_search.config import CHUNK_OVERLAP, CHUNK_TOKENS, HTTP_TIMEOUT, SERVER, TOP_K
from agent_lg_pdf_search.index import load_store, search_scored
from agent_lg_pdf_search.tools import BEHAVIOUR, REPHRASE, expand_citations, format_chunks, sources_of

logging.basicConfig(
    # Next to this file, not in whatever folder the terminal happens to be in.
    filename=Path(__file__).with_name("agent.log"),
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("agent_lg_pdf_search")

# set_debug(True)

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
    """State passed between the nodes of the graph.

    ``messages`` is inherited from ``MessagesState`` and holds the whole
    conversation. The other fields describe the last question only and are
    overwritten on every turn.

    Attributes:
        query (str): Standalone search query written by ``rephrase``.
        context (str): Found chunks formatted as one text for the model.
        sources (list[str]): Origin of each chunk, in the order of
            ``context``.
        scores (list[float]): Similarity of each chunk, in the same order.
    """

    query: str
    context: str
    sources: list[str]
    scores: list[float]
    # sources: Annotated[list[str], operator.add]


def rephrase(state):
    """Rewrite the last question into a standalone English search query.

    The documents are mostly English, and a follow-up question often makes
    no sense without the earlier turns. The model is called with thinking
    disabled and only the first line of its reply is used. Token usage is
    written to the log.

    Args:
        state (State): Current state; ``messages`` is read.

    Returns:
        dict: Update with ``query``. If the model returns nothing, the
        last message is used unchanged.
    """
    reply = MODEL.invoke(
        [SystemMessage(BEHAVIOUR), *state["messages"], SystemMessage(REPHRASE)],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    usage = reply.usage_metadata or {}
    cached = usage.get("input_token_details", {}).get("cache_read", 0)
    log.info(
        "REPHRASE: prompt %s (%s cached, %s prefilled), answer %s",
        usage.get("input_tokens"),
        cached,
        usage.get("input_tokens", 0) - cached,
        usage.get("output_tokens"),
    )
    lines = reply.content.strip().splitlines()
    return {"query": lines[0].strip('"') if lines else state["messages"][-1].content}


def retrieve(state):
    """Search the index with the rewritten query.

    Args:
        state (State): Current state; ``query`` is read.

    Returns:
        dict: Update with ``context``, ``sources`` and ``scores`` (rounded
        to three decimals), all in the order returned by the search.
    """
    hits = search_scored(state["query"])
    chunks = [chunk for chunk, score in hits]
    return {
        "context": format_chunks(chunks),
        "sources": sources_of(chunks),
        "scores": [round(score, 3) for chunk, score in hits],
    }


def answer(state):
    """Answer the question from the chunks found by ``retrieve``.

    The chunks are sent as a system message after the conversation.
    Citation markers in the reply are expanded into their sources.

    Args:
        state (State): Current state; ``messages``, ``context`` and
            ``sources`` are read.

    Returns:
        dict: Update with ``messages`` holding the reply, which is appended
        to the conversation.
    """
    messages = [SystemMessage(BEHAVIOUR), *state["messages"], SystemMessage(state["context"])]
    reply = MODEL.invoke(messages)
    reply.content = expand_citations(reply.content, state["sources"])
    return {"messages": [reply]}


def build_graph():
    """Build the agent graph.

    Every question goes through ``rephrase``, ``retrieve`` and ``answer``,
    in that order. An in-memory checkpointer keeps the conversation between
    questions for as long as the process runs.

    Returns:
        CompiledStateGraph: The agent, ready for ``invoke`` or ``stream``.
    """
    graph = StateGraph(State)
    graph.add_node("rephrase", rephrase)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_edge(START, "rephrase")
    graph.add_edge("rephrase", "retrieve")
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
    """Read the context window the chat server was started with.

    Returns:
        int: Context size in tokens (``n_ctx``).

    Raises:
        requests.HTTPError: If the server answers with an error status.
        requests.ConnectionError: If the server cannot be reached.
        requests.Timeout: If the server does not answer within
            ``HTTP_TIMEOUT``.
    """
    response = requests.get(f"{SERVER}/props", timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return response.json()["default_generation_settings"]["n_ctx"]


def print_state(state):
    """Print the state on stdout in a readable form.

    One line per message (type, length, first 60 characters) and one line
    per chunk (number, similarity, source).

    Args:
        state (dict): State values as returned by ``ask``.
    """
    print(f"state: {len(state['messages'])} messages, context {len(state.get('context', ''))} chars")
    for message in state["messages"]:
        text = message.content.replace("\n", " ")
        print(f"  {type(message).__name__:13} str {len(message.content):5} chars | {text[:60]}")
    scores = state.get("scores", [])
    for number, source in enumerate(state.get("sources", []), start=1):
        score = scores[number - 1] if number <= len(scores) else "?"
        print(f"  chunk [{number}]     {score}  {source}")
            

ANSWER_SLOT = 0
REPHRASE_SLOT = 1

THREAD = {"configurable": {"thread_id": "agent_lg_pdf_search"}}


def log_settings():
    """Write the settings of this run to the log.

    One line per run with the model, context size, ``TOP_K``, chunk
    settings and index size, so old log entries stay comparable.
    """
    store = load_store()
    log.info(
        "SETUP:    model=%s n_ctx=%s TOP_K=%s CHUNK_TOKENS=%s overlap=%s index=%s chunks",
        MODEL.model_name,
        context_limit(),
        TOP_K,
        CHUNK_TOKENS,
        CHUNK_OVERLAP,
        len(store.store),
    )


def ask(agent, question):
    """Run one turn of the conversation and log it step by step.

    For each node the log gets its duration and its result: the rewritten
    query, the found chunks, the answer and its token usage.

    Args:
        agent (CompiledStateGraph): Graph returned by ``build_graph``.
        question (str): Question typed by the user.

    Returns:
        dict: State values after the turn.
    """
    log.info("QUESTION: %s", question)
    started = time.perf_counter()
    previous = started
    for step in agent.stream({"messages": [HumanMessage(question)]}, THREAD, stream_mode="updates"):
        for node, update in step.items():
            took = time.perf_counter() - previous
            previous = time.perf_counter()
            log.info("TIME:     %s %.1f s", node, took)
            if node == "rephrase":
                log.info("QUERY:    %s", update["query"])
            if node == "retrieve":
                log.info(
                    "CHUNKS:   %s",
                    " | ".join(f"{score} {source}" for score, source in zip(update["scores"], update["sources"])),
                )
            if node == "answer":
                reply = update["messages"][-1]
                usage = reply.usage_metadata or {}
                cached = usage.get("input_token_details", {}).get("cache_read", 0)
                log.info("ANSWER:   %s", reply.content.replace("\n", " "))
                log.info(
                    "TOKENS:   prompt %s (%s cached, %s prefilled), answer %s, finish=%s",
                    usage.get("input_tokens"),
                    cached,
                    usage.get("input_tokens", 0) - cached,
                    usage.get("output_tokens"),
                    reply.response_metadata.get("finish_reason"),
                )
    log.info("TURN:     %.1f s", time.perf_counter() - started)
    return agent.get_state(THREAD).values


def main():
    """Run the terminal loop.

    The prompt shows the tokens used by the last turn against the context
    limit. A failed turn is logged and printed, and the loop goes on. An
    empty line ends it.
    """
    agent = build_graph()
    used = 0
    context_lim = context_limit()
    log_settings()
    while True:
        question = input(f"\n{used}/{context_lim} > ").strip()
        if not question:
            break
        try:
            state = ask(agent, question)
        except Exception as error:
            log.error("FAILED:   %s: %s", type(error).__name__, error)
            print(f"blad: {type(error).__name__}: {error}")
            continue
        print(state["messages"][-1].content)
        print("================================================================")
        print_state(state)
        print("================================================================")
        used = (state["messages"][-1].usage_metadata or {}).get("total_tokens", 0)


if __name__ == "__main__":
    main()
