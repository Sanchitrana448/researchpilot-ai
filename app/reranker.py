"""
Cross-encoder re-ranking: the second stage of retrieval.

Vector search compares a query vector with passage vectors computed
separately, which is fast but coarse. A cross-encoder reads the query and
one passage together and scores the pair, which is slower but more
accurate, so it is only run on the top candidates from vector search.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (ONNX export by Xenova), trained
on MS MARCO passage ranking. Scores are relevance logits: higher is better,
and they are not probabilities.

Off by default. On BEIR SciFact (eval/beir_scifact.py) re-ranking the top 30
vector-search results gave no gain over bge-small alone (NDCG@10 0.701 vs
0.706) and cost about 140x the latency, likely because the model was trained
on web search queries rather than scientific claims. Turn it on per request
with {"rerank": true} or for the server with RERANK=1. Set
RERANKER_MODEL_PATH to load from a local folder.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from .embeddings import MODEL_CACHE, THREADS

MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
ENABLED = os.getenv("RERANK", "0") == "1"
# Loading the cross-encoder takes the app from about 360 MB to about 500 MB,
# too close to the 512 MB limit of Render's free tier, so it is not offered
# there unless RERANK_AVAILABLE=1.
AVAILABLE = os.getenv("RERANK_AVAILABLE", "0" if os.getenv("RENDER") else "1") == "1"
BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "16"))


@lru_cache(maxsize=1)
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    local = os.getenv("RERANKER_MODEL_PATH")
    if local:
        return TextCrossEncoder(MODEL_NAME, specific_model_path=local, local_files_only=True, threads=THREADS)
    return TextCrossEncoder(MODEL_NAME, cache_dir=MODEL_CACHE, threads=THREADS)


def score(query: str, passages: List[str]) -> List[float]:
    """Relevance logit for each (query, passage) pair, in input order."""
    if not passages:
        return []
    return [float(s) for s in _model().rerank(query, passages, batch_size=BATCH_SIZE)]


def warm_up() -> None:
    if ENABLED and AVAILABLE:
        score("warm up", ["warm up"])
