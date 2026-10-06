"""Collects the agent's answers to the evaluation questions for review.

Run from the repo root:

    python -m agent_lg_pdf_search.eval.answers
"""
import time

import pandas as pd
from langchain_core.messages import HumanMessage

from agent_lg_pdf_search.agent import build_graph
from agent_lg_pdf_search.eval.evaluate import RESULTS, load_questions


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


def run():
    """Ask the agent every evaluation question.

    One line per question is printed on stdout as it finishes. A question
    that fails does not stop the run: its row gets the error as the answer
    and an empty query.

    Returns:
        list[dict]: One row per question, as returned by ``ask``.
    """
    agent = build_graph()
    questions = load_questions()
    rows = []
    for number, item in enumerate(questions, start=1):
        started = time.perf_counter()
        try:
            row = ask(agent, item, number)
        except Exception as error:
            row = {
                "question": item["question"],
                "query": "",
                "answer": f"ERROR {type(error).__name__}: {error}",
                "expected": item["answer"],
            }
        rows.append(row)
        print(f"{number:3}/{len(questions)} {time.perf_counter() - started:5.1f} s | {item['question'][:60]}")
    return rows


def save(rows):
    """Write the answers to ``results/answers.xlsx``.

    One row per question, with the columns ``question``, ``query``,
    ``answer`` and ``expected``.

    Args:
        rows (list[dict]): Rows as returned by ``run``.

    Returns:
        pathlib.Path: The file that was written.

    Raises:
        PermissionError: If ``answers.xlsx`` is open in Excel.
    """
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "answers.xlsx"
    table = pd.DataFrame(rows, columns=["question", "query", "answer", "expected"])
    table.to_excel(out, index=False)
    return out
