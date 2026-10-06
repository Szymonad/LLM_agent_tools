"""Collects the agent's answers to the evaluation questions for review.

Run from the repo root:

    python -m agent_lg_pdf_search.eval.answers
"""
from langchain_core.messages import HumanMessage

from agent_lg_pdf_search.agent import build_graph
from agent_lg_pdf_search.eval.evaluate import load_questions


def ask(agent, item, number):
    """Put one evaluation question through the whole agent.

    The question gets a conversation of its own, so the earlier questions
    do not reach the model as history.

    Args:
        agent (CompiledStateGraph): Graph returned by ``agent.build_graph``.
        item (dict): Question as returned by ``load_questions``.
        number (int): Position of the question, used as the thread id.

    Returns:
        dict: Keys ``question``, ``query`` (the search query the agent
        wrote), ``answer`` and ``expected``.
    """
    thread = {"configurable": {"thread_id": str(number)}}
    state = agent.invoke({"messages": [HumanMessage(item["question"])]}, thread)
    return {
        "question": item["question"],
        "query": state["query"],
        "answer": state["messages"][-1].content,
        "expected": item["answer"],
    }
