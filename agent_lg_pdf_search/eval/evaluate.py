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
    one of the expected pages.

    Args:
        hits (list[tuple[Document, float]]): Chunks with their similarity,
            best first, as returned by ``index.search_scored``.
        expected_doc (str): File name the answer is in.
        expected_pages (list[int]): Page numbers the answer is on.
        cutoffs (list[int]): Numbers of top chunks to measure at.

    Returns:
        dict[str, bool]: Keys ``file@N`` and ``page@N`` for every N in
        ``cutoffs``.
    """
    result = {}
    for k in cutoffs:
        taken = hits[:k]
        result[f"file@{k}"] = any(chunk.metadata["source"] == expected_doc for chunk, score in taken)
        result[f"page@{k}"] = any(
            chunk.metadata["source"] == expected_doc and set(chunk.metadata["pages"]) & set(expected_pages)
            for chunk, score in taken
        )
    return result


def run(rebuild, use_rephrase, k, label, temp):
    """Measure retrieval on every question and save the result.

    Each question is searched in the index and checked with ``check``.
    The summary and the per-question rows are written to
    ``results/<label>.json`` and the summary is printed on stdout.

    Args:
        rebuild (bool): Build the index first. It is also built when the
            index file does not exist.
        use_rephrase (bool): Rewrite each question with the model before
            searching.
        k (int): Number of chunks to fetch per question.
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
        row.update(check(hits, item["doc"], item["pages"], [1, 3, 5]))
        rows.append(row)
        print(f"{number:3}/{len(questions)} file@5={row['file@5']} page@5={row['page@5']} | {item['question'][:60]}")
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
    for key in ("file@1", "file@3", "file@5", "page@1", "page@3", "page@5"):
        summary[key] = sum(1 for row in rows if row[key])
    summary["mean_best_score"] = round(sum(row["best_score"] or 0 for row in rows) / len(rows), 3)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{label}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(f"--- {label}: chunk={CHUNK_TOKENS}/{CHUNK_OVERLAP} k={k} rephrase={use_rephrase} ({took:.1f} s)")
    for key in ("file@1", "file@3", "file@5", "page@1", "page@3", "page@5"):
        print(f"    {key}: {summary[key]:3}/{len(rows)}")
    print(f"    mean best score: {summary['mean_best_score']}")
    print(f"    saved {out}")


def compare():
    """Print the summaries of all saved runs as one table on stdout.

    One row per file in ``results``, sorted by file name.
    """
    files = sorted(RESULTS.rglob("*.json"))
    if not files:
        print("no results in", RESULTS)
        return
    keys = ("file@1", "file@3", "file@5", "page@1", "page@3", "page@5", "mean_best_score", "seconds")
    print(f"{'label':34} {'chunk':>7} {'k':>2} {'reph':>5} " + " ".join(f"{key:>10}" for key in keys))
    for path in files:
        s = json.loads(path.read_text(encoding="utf-8"))["summary"]
        chunk = f"{s['chunk_tokens']}/{s['chunk_overlap']}"
        print(
            f"{s['label'][:34]:34} {chunk:>7} {s['k']:>2} {str(s['rephrase']):>5} "
            + " ".join(f"{s[key]:>10}" for key in keys)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="rebuild eval/index.json before measuring")
    parser.add_argument("--rephrase", action="store_true", help="rewrite each question with the model first")
    parser.add_argument("--k", type=int, default=5)
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
                run(rebuild, args.rephrase, args.k, f"{label}_temp_{temp_text}_{repeat}", temperature)
                rebuild = False
