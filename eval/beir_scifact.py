"""Public retrieval benchmark: BEIR SciFact. Run: python eval/beir_scifact.py

SciFact (Wadden et al., 2020; BEIR, Thakur et al., 2021): 5,183 scientific
abstracts and 300 test claims with expert relevance labels. Scores are
NDCG@10 (the BEIR headline metric), Recall@10 and Recall@100.

Compares three pipelines over the same corpus:
  1. BM25 keyword baseline (rank-bm25, eval only; the app does not use it)
  2. Dense: bge-small-en-v1.5 embeddings in an in-memory Qdrant collection
  3. Dense + re-rank: top-K dense candidates re-scored by a cross-encoder
     (ms-marco-MiniLM-L-6-v2)

Data is fetched from the Hugging Face hub on first run (or read from
SCIFACT_DIR). Embedding the corpus takes a few minutes on a laptop CPU.
Writes eval/beir_scifact_results.md.
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app import embeddings  # noqa: E402

DATA_DIR = Path(os.getenv("SCIFACT_DIR", ROOT / "eval" / "data" / "scifact"))
FILES = {
    "corpus.parquet": "https://huggingface.co/datasets/BeIR/scifact/resolve/main/corpus/corpus-00000-of-00001.parquet",
    "queries.parquet": "https://huggingface.co/datasets/BeIR/scifact/resolve/main/queries/queries-00000-of-00001.parquet",
    "qrels_test.tsv": "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv",
}
RERANK_DEPTH = int(os.getenv("RERANK_DEPTH", "30"))  # same candidate count as the app


def load():
    import pyarrow.parquet as pq

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        if not (DATA_DIR / name).exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, DATA_DIR / name)
    corpus = pq.read_table(DATA_DIR / "corpus.parquet").to_pylist()
    queries = {q["_id"]: q["text"] for q in pq.read_table(DATA_DIR / "queries.parquet").to_pylist()}
    qrels: dict[str, dict[str, int]] = {}
    for line in (DATA_DIR / "qrels_test.tsv").read_text().splitlines()[1:]:
        qid, did, score = line.split("\t")
        qrels.setdefault(qid, {})[did] = int(score)
    return corpus, {qid: queries[qid] for qid in qrels}, qrels


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    relevant = {d for d, g in rel.items() if g > 0}
    return len(relevant & set(ranked[:k])) / len(relevant)


def evaluate(runs: dict[str, list[str]], qrels) -> dict:
    n = len(qrels)
    return {
        "ndcg10": sum(ndcg_at_k(runs[q], qrels[q]) for q in qrels) / n,
        "r10": sum(recall_at_k(runs[q], qrels[q], 10) for q in qrels) / n,
        "r100": sum(recall_at_k(runs[q], qrels[q], 100) for q in qrels) / n,
    }


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def main() -> None:
    corpus, queries, qrels = load()
    ids = [d["_id"] for d in corpus]
    texts = [f"{d['title']}. {d['text']}".strip() for d in corpus]
    qids = list(queries)
    results = {}

    # 1. BM25 baseline
    from rank_bm25 import BM25Okapi

    t0 = time.perf_counter()
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    bm25_runs = {}
    for qid in qids:
        scores = bm25.get_scores(tokenize(queries[qid]))
        bm25_runs[qid] = [ids[i] for i in np.argsort(-scores)[:100]]
    results["BM25 (keyword baseline)"] = (evaluate(bm25_runs, qrels), (time.perf_counter() - t0) * 1000 / len(qids))

    # 2. Dense retrieval with Qdrant
    from app.retrieval import VectorStore

    cache = DATA_DIR / "doc_vectors_bge_small.npy"
    t0 = time.perf_counter()
    if cache.exists():
        doc_vecs = np.load(cache)
    else:
        doc_vecs = embeddings.embed_documents(texts)
        np.save(cache, doc_vecs)
    index_s = time.perf_counter() - t0
    print(f"corpus embedded in {index_s:.0f} s", flush=True)
    store = VectorStore()
    store.add(0, doc_vecs)
    t0 = time.perf_counter()
    q_vecs = embeddings.embed_queries([queries[q] for q in qids])
    dense_runs = {}
    for qid, qv in zip(qids, q_vecs):
        dense_runs[qid] = [ids[i] for i, _ in store.search(qv, limit=100, min_score=-1.0)]
    results["Dense: bge-small + Qdrant"] = (evaluate(dense_runs, qrels), (time.perf_counter() - t0) * 1000 / len(qids))

    # 3. Dense + cross-encoder re-ranking
    from app import reranker

    pos = {d: i for i, d in enumerate(ids)}
    t0 = time.perf_counter()
    rr_runs = {}
    for n, qid in enumerate(qids, start=1):
        if n % 50 == 0:
            print(f"re-ranked {n}/{len(qids)} queries", flush=True)
        cands = dense_runs[qid][:RERANK_DEPTH]
        scores = reranker.score(queries[qid], [texts[pos[d]] for d in cands])
        order = [cands[i] for i in np.argsort(-np.asarray(scores))]
        rr_runs[qid] = order + dense_runs[qid][RERANK_DEPTH:]
    rr_ms = (time.perf_counter() - t0) * 1000 / len(qids)
    results[f"Dense + cross-encoder re-rank (top {RERANK_DEPTH})"] = (evaluate(rr_runs, qrels), rr_ms + results["Dense: bge-small + Qdrant"][1])

    base = results["BM25 (keyword baseline)"][0]["ndcg10"]
    out = [
        "# BEIR SciFact retrieval benchmark",
        "",
        f"Generated by `python eval/beir_scifact.py`. {len(corpus):,} scientific abstracts, {len(qids)} test "
        "queries with expert relevance labels. Deterministic; timings from a 2-core cloud CPU.",
        "",
        "| Pipeline | NDCG@10 | vs BM25 | Recall@10 | Recall@100 | ms / query (CPU) |",
        "|---|---|---|---|---|---|",
    ]
    for name, (m, ms) in results.items():
        gain = (m["ndcg10"] / base - 1) * 100
        out.append(
            f"| {name} | {m['ndcg10']:.3f} | {gain:+.1f}% | {m['r10']:.3f} | {m['r100']:.3f} | {ms:.0f} |"
        )
    out += [
        "",
        f"Re-ranking covers the top {RERANK_DEPTH} vector-search candidates per query, the same depth "
        f"the app uses. Embedding model `{embeddings.MODEL_NAME}`, re-ranker `{reranker.MODEL_NAME}`.",
        "",
    ]
    text = "\n".join(out)
    (ROOT / "eval" / "beir_scifact_results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
