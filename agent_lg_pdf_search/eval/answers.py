"""Collects the agent's answers to the evaluation questions for review.

Run from the repo root:

    python -m agent_lg_pdf_search.eval.answers [--questions FILE] [--out NAME]
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage

from agent_lg_pdf_search import index
from agent_lg_pdf_search.agent import build_graph
from agent_lg_pdf_search.eval.evaluate import QUESTIONS, RESULTS, load_questions


def ask(agent, item, number):
    """Put one evaluation question through the whole agent.

    The question gets a conversation of its own, so the earlier questions
    do not reach the model as history. The agent's state does not hold the
    chunk ids, so the search is repeated with the agent's query; the same
    query always gives the same chunks.

    Args:
        agent (CompiledStateGraph): Graph returned by ``agent.build_graph``.
        item (dict): Question as returned by ``load_questions``.
        number (int): Position of the question, used as the thread id.

    Returns:
        dict: Keys ``question``, ``query`` (the search query the agent
        wrote), ``answer``, ``expected``, ``source`` (kind of the question,
        empty if the question has none), ``input_tokens`` and
        ``output_tokens`` (of the answer step only, ``None`` if the server
        did not report them), ``chunks`` (ids of the chunks the model
        got, best first, separated by commas) and ``page_hit`` (``True``
        if any chunk comes from the expected document and has at least one
        of the expected pages).
    """
    thread = {"configurable": {"thread_id": str(number)}}
    state = agent.invoke({"messages": [HumanMessage(item["question"])]}, thread)
    hits = index.search_scored(state["query"])
    reply = state["messages"][-1]
    usage = reply.usage_metadata or {}
    page_hit = any(
        chunk.metadata["source"] == item["doc"] and set(chunk.metadata["pages"]) & set(item["pages"])
        for chunk, score in hits
    )
    return {
        "question": item["question"],
        "query": state["query"],
        "answer": reply.content,
        "expected": item["answer"],
        "source": item.get("source", ""),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "chunks": ", ".join(chunk.id for chunk, score in hits),
        "page_hit": page_hit,
    }


def run(questions_path):
    """Ask the agent every evaluation question.

    One line per question is printed on stdout as it finishes. A question
    that fails does not stop the run: its row gets the error as the answer
    and an empty query and chunk list, and ``None`` as the token counts and
    ``page_hit``.

    Args:
        questions_path (pathlib.Path): Questions file for ``load_questions``.

    Returns:
        list[dict]: One row per question, as returned by ``ask``, with
        ``seconds`` added: the time the question took.
    """
    agent = build_graph()
    questions = load_questions(questions_path)
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
                "source": item.get("source", ""),
                "input_tokens": None,
                "output_tokens": None,
                "chunks": "",
                "page_hit": None,
            }
        row["seconds"] = round(time.perf_counter() - started, 1)
        rows.append(row)
        print(f"{number:3}/{len(questions)} {row['seconds']:5.1f} s | {item['question'][:60]}")
    return rows


def save(rows, name):
    """Write the answers to a file in ``results/``.

    One row per question, with the columns ``question``, ``query``,
    ``answer``, ``expected``, ``source``, ``correct`` (empty, to be filled
    in by hand), ``page_hit``, ``seconds``, ``input_tokens``,
    ``output_tokens`` and ``chunks``.

    Args:
        rows (list[dict]): Rows as returned by ``run``.
        name (str): File name, e.g. ``answers.xlsx``.

    Returns:
        pathlib.Path: The file that was written.

    Raises:
        PermissionError: If the file is open in Excel.
    """
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / name
    table = pd.DataFrame(rows, columns=["question", "query", "answer", "expected", "source", "correct", "page_hit", "seconds", "input_tokens", "output_tokens", "chunks"])
    table.to_excel(out, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect the agent's answers to the evaluation questions.")
    parser.add_argument("--questions", type=Path, default=QUESTIONS, help="questions JSON file")
    parser.add_argument("--out", default="answers.xlsx", help="name of the file in results/")
    args = parser.parse_args()
    rows = run(args.questions)
    print("saved", save(rows, args.out))
