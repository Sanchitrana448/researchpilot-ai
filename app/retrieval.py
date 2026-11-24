"""
Ingestion, chunking and two-stage retrieval.

Stage 1, vector search: every chunk is embedded once at ingest
(bge-small-en-v1.5, app/embeddings.py) and stored in a Qdrant collection.
A question is embedded the same way and the nearest chunks by cosine
similarity become candidates.

Stage 2, optional re-ranking: a cross-encoder (app/reranker.py) reads the
question together with each candidate and re-orders them. Off by default,
because it did not improve results on the public benchmark.

Refusal: candidates below MIN_SIMILARITY are dropped in stage 1, so an
off-topic question retrieves nothing and the pipeline refuses to answer
instead of citing the least-bad passage. The re-ranker is not used for this:
it scores reworded but correct matches almost as low as off-topic ones.

Measured on the public BEIR SciFact benchmark in eval/beir_scifact.py.
"""
from __future__ import annotations

import logging
import math
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import embeddings, reranker

logger = logging.getLogger("researchpilot.retrieval")

CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
# Cosine floor for "is this on topic at all". bge-small scores unrelated text
# around 0.35-0.47 and genuine matches from about 0.55 up in
# eval/benchmark_data.py, so 0.50 sits in the gap. Chosen on that small set.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.50"))
# Keep only passages within this distance of the best match. Topically close
# documents all clear the absolute floor, so without this a question about
# one subtopic pulls in every document from the same field as "evidence".
RELATIVE_MARGIN = float(os.getenv("RELATIVE_MARGIN", "0.15"))


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    text: str


@dataclass
class Document:
    id: str
    source: str
    text: str
    quality_score: float = 1.0


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    """Sliding-window word chunking with overlap.

    200 words is roughly 260 tokens, inside the 512-token limit of both models,
    so nothing is silently truncated before it is embedded.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def score_source_quality(source: str) -> float:
    """Crude source-quality prior based on the source name.

    Favours named and structured sources over anonymous pastes. This is a
    hardcoded domain list, not a model, and it should be replaced by a real
    domain-authority or recency signal before anyone relies on it.
    """
    s = source.lower()
    if s.startswith("http"):
        if any(d in s for d in [".gov", ".edu", "arxiv.org", "nature.com", "who.int"]):
            return 1.0
        return 0.75
    if s.endswith((".pdf", ".md", ".txt")):
        return 0.85
    return 0.6


class VectorStore:
    """Cosine vector search over chunk embeddings.

    Uses Qdrant's in-process (":memory:") mode: the same client API as a
    Qdrant server, so moving to a hosted collection is a constructor change.
    """

    COLLECTION = "chunks"

    def __init__(self, dim: int = embeddings.DIM):
        from qdrant_client import QdrantClient, models

        self._models = models
        self._client = QdrantClient(":memory:")
        self._client.create_collection(
            self.COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        self.backend = "qdrant (in-memory)"

    def add(self, start_id: int, vectors: np.ndarray) -> None:
        if len(vectors) == 0:
            return
        self._client.upload_collection(
            self.COLLECTION,
            vectors=vectors,
            ids=list(range(start_id, start_id + len(vectors))),
            batch_size=256,
        )

    def search(self, vector: np.ndarray, limit: int, min_score: float) -> List[tuple]:
        """Return [(chunk_position, cosine_similarity)], best first."""
        hits = self._client.query_points(
            self.COLLECTION, query=vector.tolist(), limit=limit, score_threshold=min_score
        ).points
        return [(int(h.id), float(h.score)) for h in hits]


class RerankUnavailable(RuntimeError):
    pass


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Index:
    """In-memory two-stage retrieval index."""

    def __init__(self):
        self.documents: dict[str, Document] = {}
        self.chunks: List[Chunk] = []
        self._vectors = VectorStore()

    def add_document(self, text: str, source: str) -> Document:
        doc_id = str(uuid.uuid4())[:8]
        doc = Document(id=doc_id, source=source, text=text, quality_score=score_source_quality(source))
        self.documents[doc_id] = doc
        seen_hashes = {hash(c.text) for c in self.chunks}
        first_new = len(self.chunks)
        for i, ctext in enumerate(chunk_text(text)):
            h = hash(ctext)
            # Same passage in two documents is one piece of evidence, not two.
            # Indexing it twice would let it corroborate itself downstream.
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            self.chunks.append(Chunk(id=f"{doc_id}-{i}", doc_id=doc_id, source=source, text=ctext))
        # Only new chunks are embedded; existing vectors stay valid.
        new = self.chunks[first_new:]
        self._vectors.add(first_new, embeddings.embed_documents([c.text for c in new]))
        return doc

    def search(self, query: str, top_k: int = 6, rerank: Optional[bool] = None) -> List[dict]:
        """Return the top_k chunks with similarity, re-rank score and final score."""
        if not self.chunks:
            return []
        use_rerank = (reranker.ENABLED and reranker.AVAILABLE) if rerank is None else rerank
        if use_rerank and not reranker.AVAILABLE:
            raise RerankUnavailable("Re-ranking is switched off on this deployment to stay within its memory limit.")
        limit = max(CANDIDATES, top_k * 3) if use_rerank else top_k * 3
        hits = self._vectors.search(embeddings.embed_queries([query])[0], limit, MIN_SIMILARITY)
        if not hits:
            return []
        best = hits[0][1]
        hits = [(i, sim) for i, sim in hits if sim >= best - RELATIVE_MARGIN]

        rerank_scores: dict[int, float] = {}
        if use_rerank:
            scores = reranker.score(query, [self.chunks[i].text for i, _ in hits])
            rerank_scores = {i: s for (i, _), s in zip(hits, scores)}

        results = []
        for idx, sim in hits:
            if use_rerank:
                relevance = _sigmoid(rerank_scores[idx])
            else:
                relevance = sim
            chunk = self.chunks[idx]
            quality = self.documents[chunk.doc_id].quality_score
            results.append(
                {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "text": chunk.text,
                    "similarity": round(sim, 4),
                    "rerank_score": round(rerank_scores[idx], 3) if use_rerank else None,
                    "source_quality": quality,
                    # Relevance dominates; the source prior only breaks near-ties.
                    "combined_score": round(0.85 * relevance + 0.15 * quality, 4),
                    "retrieval_mode": "dense+rerank" if use_rerank else "dense",
                }
            )
        results.sort(key=lambda r: -r["combined_score"])
        return results[:top_k]

    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "embedding_model": embeddings.MODEL_NAME,
            "vector_store": self._vectors.backend,
            "reranker": reranker.MODEL_NAME,
            "rerank_available": reranker.AVAILABLE,
            "rerank_by_default": reranker.ENABLED and reranker.AVAILABLE,
        }
