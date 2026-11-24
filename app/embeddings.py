"""
Dense text embeddings for retrieval.

Model: BAAI/bge-small-en-v1.5 (33M-parameter BERT encoder, 384 dimensions),
run as quantised ONNX through fastembed, so it needs no GPU and no PyTorch.

Queries get BGE's recommended retrieval instruction; passages are embedded
as-is. Vectors are L2-normalised, so cosine similarity is a dot product.

Model files are downloaded once into MODEL_CACHE (the Docker build does this
ahead of time). Set BGE_MODEL_PATH to load from a local folder instead.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MODEL_CACHE = os.getenv("MODEL_CACHE", str(os.path.join(os.path.dirname(__file__), "..", ".model_cache")))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
# fastembed falls back to one ONNX thread unless told otherwise. On Render
# (which sets RENDER) cpu_count() reports the host, not the small CPU share
# the service gets, so use one thread there.
_default_threads = 1 if os.getenv("RENDER") else (os.cpu_count() or 1)
THREADS = int(os.getenv("ONNX_THREADS", str(_default_threads)))


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    local = os.getenv("BGE_MODEL_PATH")
    if local:
        return TextEmbedding(MODEL_NAME, specific_model_path=local, local_files_only=True, threads=THREADS)
    return TextEmbedding(MODEL_NAME, cache_dir=MODEL_CACHE, threads=THREADS)


def _normalise(vecs) -> np.ndarray:
    arr = np.asarray(list(vecs), dtype=np.float32).reshape(-1, DIM)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def embed_documents(texts: List[str]) -> np.ndarray:
    """Passage vectors, shape (len(texts), DIM), in input order.

    Texts are embedded shortest-first so each batch holds similar lengths and
    little compute is spent on padding, then put back in the original order.
    """
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    vecs = _normalise(_model().embed([texts[i] for i in order], batch_size=BATCH_SIZE))
    out = np.empty_like(vecs)
    out[order] = vecs
    return out


def embed_queries(queries: List[str]) -> np.ndarray:
    """Query vectors with the BGE retrieval instruction prepended."""
    if not queries:
        return np.zeros((0, DIM), dtype=np.float32)
    return _normalise(_model().embed([QUERY_INSTRUCTION + q for q in queries], batch_size=BATCH_SIZE))


def warm_up() -> None:
    embed_queries(["warm up"])
