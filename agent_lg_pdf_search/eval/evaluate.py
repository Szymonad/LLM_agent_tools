"""Measures retrieval on the MMLongBench-Doc questions in this folder.

Run from the agent folder:
    python agent_lg_pdf_search/eval/evaluate.py --rebuild
    python agent_lg_pdf_search/eval/evaluate.py --rephrase --label chunk200-rephrase
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import index
from config import CHUNK_OVERLAP, CHUNK_TOKENS, TOP_K

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "questions.json"
RESULTS = HERE / "results"

index.PDF_DIR = HERE / "pdf"
index.INDEX_PATH = HERE / "index.json"


def load_questions():
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def rewrite(question, temp):
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent import MODEL
    from tools import BEHAVIOUR, REPHRASE

    reply = MODEL.invoke(
        [SystemMessage(BEHAVIOUR), HumanMessage(question), SystemMessage(REPHRASE)],
        temperature=temp,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    lines = reply.content.strip().splitlines()
    return lines[0].strip('"') if lines else question


def check(hits, expected_doc, expected_pages):
    """For each cut-off: whether the right file, and the right page, is among the first k chunks."""
    wynik = {}
    for k in (1, 3, 5):
        wziete = hits[:k]
        wynik[f"file@{k}"] = any(chunk.metadata["source"] == expected_doc for chunk, score in wziete)
        wynik[f"page@{k}"] = any(
            chunk.metadata["source"] == expected_doc and set(chunk.metadata["pages"]) & set(expected_pages)
            for chunk, score in wziete
        )
    return wynik


def run(rebuild, use_rephrase, k, label, temp):
    if rebuild or not index.INDEX_PATH.exists():
        print("buduje indeks z", index.PDF_DIR)
        started = time.perf_counter()
        index.build()
        index.load_store.cache_clear()
        print(f"indeks gotowy w {time.perf_counter() - started:.1f} s")

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
        row.update(check(hits, item["doc"], item["pages"]))
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
    print(f"    srednia najlepsza ocena: {summary['mean_best_score']}")
    print(f"    zapisano {out}")


def compare():
    """Prints every saved run next to each other."""
    files = sorted(RESULTS.glob("*.json"))
    if not files:
        print("brak wynikow w", RESULTS)
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
        etykieta = args.label or f"chunk{CHUNK_TOKENS}" + ("-rephrase" if args.rephrase else "") + f"_k{args.k}"
        rebuild = args.rebuild
        for temperature in temperatures:
            for repeat in range(1, repeats + 1):
                temp_text = f"{temperature:.1f}".replace(".", "")
                run(rebuild, args.rephrase, args.k, f"{etykieta}_temp_{temp_text}_{repeat}", temperature)
                rebuild = False
