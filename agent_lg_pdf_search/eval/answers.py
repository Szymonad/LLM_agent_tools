"""Collects the agent's answers to the evaluation questions for review.

Run from the repo root:

    python -m agent_lg_pdf_search.eval.answers --stage NAME [--questions FILE]

Every run gets its own folder, ``results/<stage>/<date_time>/``, with
``answers.xlsx`` and ``run.json`` (the settings the answers were made with),
so earlier runs are never overwritten and the stages can be compared.
``results.db`` is rebuilt at the end.
"""
import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from langchain_core.messages import HumanMessage

from agent_lg_pdf_search import index
from agent_lg_pdf_search.agent import build_graph
from agent_lg_pdf_search.config import (
    CHUNK_OVERLAP,
    CHUNK_TOKENS,
    EMBED_SERVER,
    HTTP_TIMEOUT,
    REPHRASE,
    SERVER,
    TOP_K,
)
from agent_lg_pdf_search.eval import to_sqlite
from agent_lg_pdf_search.eval.evaluate import QUESTIONS, RESULTS, load_questions

REPO = Path(__file__).resolve().parents[2]
#: Stage names become folder names, so only lowercase letters, digits and ``_``.
STAGE_NAME = re.compile(r"^[a-z0-9_]+$")


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
        got, best first, separated by commas), ``page_hit`` (``True``
        if any chunk comes from the expected document and has at least one
        of the expected pages, ``None`` for a question with no expected page),
        ``route`` (``documents`` or ``general``, as the agent decided) and
        ``route_ok`` (whether it matches the question's ``route``, ``None``
        if the question has none).
    """
    thread = {"configurable": {"thread_id": str(number)}}
    state = agent.invoke({"messages": [HumanMessage(item["question"])]}, thread)
    # A question the router sent to general knowledge has no search.
    query = state.get("query", "")
    hits = index.search_scored(query) if query else []
    reply = state["messages"][-1]
    usage = reply.usage_metadata or {}
    # A question with no expected page (``general``, ``not-answerable``) has no page to hit.
    page_hit = None if not item["pages"] else any(
        chunk.metadata["source"] == item["doc"] and set(chunk.metadata["pages"]) & set(item["pages"])
        for chunk, score in hits
    )
    # Without a router every question goes to the documents.
    route = state.get("route", "documents")
    return {
        "question": item["question"],
        "query": query,
        "answer": reply.content,
        "expected": item["answer"],
        "source": item.get("source", ""),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "chunks": ", ".join(chunk.id for chunk, score in hits),
        "page_hit": page_hit,
        "route": route,
        "route_ok": route == item["route"] if "route" in item else None,
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
                "route": "",
                "route_ok": None,
            }
        row["seconds"] = round(time.perf_counter() - started, 1)
        rows.append(row)
        print(f"{number:3}/{len(questions)} {row['seconds']:5.1f} s | {item['question'][:60]}")
    return rows


def new_run_folder(stage, started):
    """Create the folder of a new run: ``results/<stage>/<date_time>/``.

    Two runs started in the same minute get ``-2``, ``-3``... appended, so a
    folder is never reused.

    Args:
        stage (str): Name of the stage, e.g. ``00_baseline_pypdf``.
        started (datetime.datetime): Start of the run, gives the folder name.

    Returns:
        pathlib.Path: The created, empty folder.
    """
    base = RESULTS / stage / started.strftime("%Y-%m-%d_%H%M")
    folder, number = base, 1
    while folder.exists():
        number += 1
        folder = base.with_name(f"{base.name}-{number}")
    folder.mkdir(parents=True)
    return folder


def git_state():
    """Commit the code was at and whether it had uncommitted changes.

    Returns:
        dict: ``commit`` (short hash) and ``dirty`` (bool), both ``None``
        when git cannot be run.
    """
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(changes.strip())}


def model_file(server):
    """File name of the model a llama-server instance has loaded.

    Args:
        server (str): Address of the server, e.g. ``http://127.0.0.1:8081``.

    Returns:
        str | None: Name of the GGUF file, ``None`` if the server does not answer.
    """
    try:
        response = requests.get(f"{server}/props", timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return Path(response.json()["model_path"]).name
    except (requests.RequestException, KeyError, ValueError):
        return None


def describe_run(stage, questions_path, rows, started, finished):
    """Collect what is needed to tell later how a run was made.

    Two runs are comparable only when they used the same questions, so the
    file's SHA-256 is stored next to its name.

    Args:
        stage (str): Name of the stage.
        questions_path (pathlib.Path): Questions file the run used.
        rows (list[dict]): Rows as returned by ``run``.
        started (datetime.datetime): Start of the run.
        finished (datetime.datetime): End of the run.

    Returns:
        dict: Contents of ``run.json``.
    """
    path = Path(questions_path).resolve()
    return {
        "stage": stage,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        **git_state(),
        "questions_file": path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path),
        "questions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "questions": len(rows),
        "chat_model": model_file(SERVER),
        "embedding_model": model_file(EMBED_SERVER),
        "index_file": Path(index.INDEX_PATH).resolve().relative_to(REPO).as_posix(),
        "top_k": TOP_K,
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_dim": index.EMBED_DIM,
        "rephrase": REPHRASE,
    }


def save(rows, folder, description):
    """Write the answers and the run description to the run folder.

    ``answers.xlsx`` has one row per question, with the columns
    ``question``, ``query``, ``answer``, ``expected``, ``source``,
    ``correct`` (empty, to be filled in by hand), ``page_hit``, ``seconds``,
    ``input_tokens``, ``output_tokens``, ``chunks``, ``route`` and
    ``route_ok``.

    Args:
        rows (list[dict]): Rows as returned by ``run``.
        folder (pathlib.Path): Folder made by ``new_run_folder``.
        description (dict): Contents of ``run.json``, from ``describe_run``.

    Returns:
        pathlib.Path: The answers file that was written.
    """
    (folder / "run.json").write_text(json.dumps(description, ensure_ascii=False, indent=1), encoding="utf-8")
    out = folder / "answers.xlsx"
    table = pd.DataFrame(rows, columns=["question", "query", "answer", "expected", "source", "correct", "page_hit", "seconds", "input_tokens", "output_tokens", "chunks", "route", "route_ok"])
    table.to_excel(out, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect the agent's answers to the evaluation questions.")
    parser.add_argument("--stage", required=True, help="stage folder in results/, e.g. 00_baseline_pypdf")
    parser.add_argument("--questions", type=Path, default=QUESTIONS, help="questions JSON file")
    args = parser.parse_args()
    if not STAGE_NAME.match(args.stage):
        parser.error("--stage may hold only lowercase letters, digits and _")
    if not index.INDEX_PATH.exists():
        parser.error(f"no index at {index.INDEX_PATH}, build it first")
    started = datetime.now()
    rows = run(args.questions)
    folder = new_run_folder(args.stage, started)
    print("saved", save(rows, folder, describe_run(args.stage, args.questions, rows, started, datetime.now())))
    to_sqlite.build()
    print("rebuilt", to_sqlite.DATABASE)
