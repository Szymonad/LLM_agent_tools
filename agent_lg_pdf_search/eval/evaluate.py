"""Measures retrieval on the MMLongBench-Doc questions in this folder.

Run from the repo root:

    python -m agent_lg_pdf_search.eval.evaluate --rebuild
    python -m agent_lg_pdf_search.eval.evaluate --rephrase
    python -m agent_lg_pdf_search.eval.evaluate --compare
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from agent_lg_pdf_search import index
from agent_lg_pdf_search.config import CHUNK_OVERLAP, CHUNK_TOKENS, TOP_K

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "questions.json"
RESULTS = HERE / "results"

index.PDF_DIR = HERE / "pdf"
index.INDEX_PATH = HERE / "index.json"


def load_questions():
    """Read the evaluation questions from ``questions.json``.

    Returns:
        list[dict]: One item per question, with ``question`` (text),
        ``doc`` (expected file name) and ``pages`` (expected page numbers).
    """
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def rewrite(question, temp):
    """Rewrite a question into a search query, as the agent does.

    Uses the same model and prompts as ``agent.rephrase``, but with a
    single question instead of a conversation. The agent is imported here,
    so a run without ``--rephrase`` needs no chat server.

    Args:
        question (str): Question from the evaluation set.
        temp (float): Sampling temperature for the model.

    Returns:
        str: First line of the reply, or ``question`` unchanged if the
        model returns nothing.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent_lg_pdf_search.agent import MODEL
    from agent_lg_pdf_search.tools import BEHAVIOUR, REPHRASE

    reply = MODEL.invoke(
        [SystemMessage(BEHAVIOUR), HumanMessage(question), SystemMessage(REPHRASE)],
        temperature=temp,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    lines = reply.content.strip().splitlines()
    return lines[0].strip('"') if lines else question


def check(hits, expected_doc, expected_pages, cutoffs):
    """Check whether the search found the expected file and page.

    Measured at every cut-off in ``cutoffs``: the first N chunks. A page
    counts as found when a chunk from the expected file covers at least
    one of the expected pages. The recall is the share of the expected
    pages that the chunks from the expected file cover.

    Args:
        hits (list[tuple[Document, float]]): Chunks with their similarity,
            best first, as returned by ``index.search_scored``.
        expected_doc (str): File name the answer is in.
        expected_pages (list[int]): Page numbers the answer is on.
        cutoffs (list[int]): Numbers of top chunks to measure at.

    Returns:
        dict[str, bool | float]: For every N in ``cutoffs``, keys
        ``file@N`` and ``page@N`` (bool) and ``recall@N`` (float from 0
        to 1).
    """
    result = {}
    for k in cutoffs:
        taken = hits[:k]
        result[f"file@{k}"] = any(chunk.metadata["source"] == expected_doc for chunk, score in taken)
        result[f"page@{k}"] = any(
            chunk.metadata["source"] == expected_doc and set(chunk.metadata["pages"]) & set(expected_pages)
            for chunk, score in taken
        )
        found_pages = set()
        for chunk, score in taken:
            if chunk.metadata["source"] == expected_doc:
                found_pages.update(chunk.metadata["pages"])
        result[f"recall@{k}"] = round(len(found_pages & set(expected_pages)) / len(expected_pages), 3)
    return result


def run(rebuild, use_rephrase, k, step, label, temp):
    """Measure retrieval on every question and save the result.

    Each question is searched in the index and checked with ``check`` at
    every ``step``-th cut-off starting from 1, and at ``k`` itself.
    The summary and the per-question rows are written to
    ``results/<label>.json`` and the summary is printed on stdout.

    Args:
        rebuild (bool): Build the index first. It is also built when the
            index file does not exist.
        use_rephrase (bool): Rewrite each question with the model before
            searching.
        k (int): Number of chunks to fetch per question. Also the last
            cut-off.
        step (int): Distance between cut-offs: 1 gives 1, 2, 3, ...,
            2 gives 1, 3, 5, ...
        label (str): Name of the run, used as the result file name.
        temp (float): Temperature for the rewrite; ignored without
            ``use_rephrase``.
    """
    if rebuild or not index.INDEX_PATH.exists():
        print("building the index from", index.PDF_DIR)
        started = time.perf_counter()
        index.build()
        index.load_store.cache_clear()
        print(f"index ready in {time.perf_counter() - started:.1f} s")

    cutoffs = list(range(1, k + 1, step))
    if cutoffs[-1] != k:
        cutoffs.append(k)
    keys = [f"{kind}@{n}" for kind in ("file", "page") for n in cutoffs]
    recall_keys = [f"recall@{n}" for n in cutoffs]

    questions = load_questions()
    rows = []
    started = time.perf_counter()
    for number, item in enumerate(questions, start=1):
        query = rewrite(item["question"], temp) if use_rephrase else item["question"]
        hits = index.search_scored(query, k=k)
        row = {
            "question": item["question"],
            "query": query,
            "doc": item["doc"],
            "pages": item["pages"],
            "best_score": round(hits[0][1], 3) if hits else None,
            "found": [
                {"source": chunk.metadata["source"], "pages": chunk.metadata["pages"], "score": round(score, 3)}
                for chunk, score in hits
            ],
        }
        row.update(check(hits, item["doc"], item["pages"], cutoffs))
        rows.append(row)
        print(f"{number:3}/{len(questions)} file@{k}={row[f'file@{k}']} page@{k}={row[f'page@{k}']} | {item['question'][:60]}")
    took = time.perf_counter() - started

    summary = {
        "label": label,
        "rephrase": use_rephrase,
        "temperature": temp if use_rephrase else None,
        "k": k,
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k_config": TOP_K,
        "questions": len(rows),
        "seconds": round(took, 1),
    }
    for key in keys:
        summary[key] = sum(1 for row in rows if row[key])
    for key in recall_keys:
        summary[key] = round(sum(row[key] for row in rows) / len(rows), 3)
    summary["mean_best_score"] = round(sum(row["best_score"] or 0 for row in rows) / len(rows), 3)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{label}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(f"--- {label}: chunk={CHUNK_TOKENS}/{CHUNK_OVERLAP} k={k} rephrase={use_rephrase} ({took:.1f} s)")
    for key in keys:
        print(f"    {key}: {summary[key]:3}/{len(rows)}")
    for key in recall_keys:
        print(f"    {key}: {summary[key]}")
    print(f"    mean best score: {summary['mean_best_score']}")
    print(f"    saved {out}")


def metric_order(name):
    """Give the sort key of a metric name such as ``file@10``.

    Args:
        name (str): Metric name, kind and cut-off joined by ``@``.

    Returns:
        tuple[str, int]: Kind and cut-off, so ``file@10`` sorts after
        ``file@5``.
    """
    kind, cutoff = name.split("@")
    return kind, int(cutoff)


def compare():
    """Print all saved runs and their averages as two tables on stdout.

    The first table has one row per file in ``results``, sorted by file
    name. The second groups the runs with the same settings, so the
    repeats of one temperature, and shows the mean and the standard
    deviation of every metric. A metric that a run did not measure is
    shown as ``<NA>``.
    """
    files = sorted(RESULTS.rglob("*.json"))
    if not files:
        print("no results in", RESULTS)
        return
    summaries = [json.loads(path.read_text(encoding="utf-8"))["summary"] for path in files]
    table = pd.DataFrame(summaries)

    metrics = sorted([column for column in table.columns if "@" in column], key=metric_order)
    counts = [column for column in metrics if not column.startswith("recall")]
    table[counts] = table[counts].astype("Int64")
    settings = ["chunk_tokens", "chunk_overlap", "k", "rephrase", "temperature"]

    print(table[["label"] + settings + metrics + ["mean_best_score", "seconds"]].to_string(index=False))

    groups = table.groupby(settings, dropna=False)
    averages = groups[metrics].agg(["mean", "std"]).round(2)
    averages.insert(0, "runs", groups.size())
    print()
    print(averages.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="rebuild eval/index.json before measuring")
    parser.add_argument("--rephrase", action="store_true", help="rewrite each question with the model first")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--step", type=int, default=1, help="distance between cut-offs, from 1 up to k")
    parser.add_argument("--label", default=None)
    parser.add_argument("--compare", action="store_true", help="only print the saved runs")
    args = parser.parse_args()

    temperatures = [2.0, 3.0, 5.0, 7.0, 9.9] if args.rephrase else [0.0]
    repeats = 5 if args.rephrase else 1

    if args.compare:
        compare()
    else:
        label = args.label or f"chunk{CHUNK_TOKENS}" + ("-rephrase" if args.rephrase else "") + f"_k{args.k}"
        rebuild = args.rebuild
        for temperature in temperatures:
            for repeat in range(1, repeats + 1):
                temp_text = f"{temperature:.1f}".replace(".", "")
                run(rebuild, args.rephrase, args.k, args.step, f"{label}_temp_{temp_text}_{repeat}", temperature)
                rebuild = False
