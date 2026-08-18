"""
Ingestion, chunking and retrieval.

TF-IDF rather than embeddings, so there's no API dependency in the retrieval
path. The cost is that matching is lexical and paraphrases get missed. Moving
to dense vectors means reimplementing _build and search; nothing outside this
module touches the vectorizer.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Sliding-window word chunking with overlap."""
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


class Index:
    """In-memory hybrid (TF-IDF + duplicate-aware) retrieval index."""

    def __init__(self):
        self.documents: dict[str, Document] = {}
        self.chunks: List[Chunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None

    def add_document(self, text: str, source: str) -> Document:
        doc_id = str(uuid.uuid4())[:8]
        doc = Document(id=doc_id, source=source, text=text, quality_score=score_source_quality(source))
        self.documents[doc_id] = doc
        seen_hashes = {hash(c.text) for c in self.chunks}
        for i, ctext in enumerate(chunk_text(text)):
            h = hash(ctext)
            # Same passage in two documents is one piece of evidence, not two.
            # Indexing it twice would let it corroborate itself downstream.
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            self.chunks.append(Chunk(id=f"{doc_id}-{i}", doc_id=doc_id, source=source, text=ctext))
        self._build()
        return doc

    def _build(self):
        if not self.chunks:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])

    def search(self, query: str, top_k: int = 6) -> List[dict]:
        """Return ranked chunks with similarity + source-quality-adjusted score."""
        if not self.chunks or self._vectorizer is None:
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        # Over-fetch on similarity alone, then re-rank the wider pool with the
        # quality prior. Re-ranking only the final k would have nothing to move.
        ranked_idx = np.argsort(-sims)[: top_k * 3]
        results = []
        for idx in ranked_idx:
            sim = float(sims[idx])
            if sim <= 0:
                continue
            chunk = self.chunks[idx]
            quality = self.documents[chunk.doc_id].quality_score
            combined = 0.75 * sim + 0.25 * quality
            results.append(
                {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "text": chunk.text,
                    "similarity": round(sim, 4),
                    "source_quality": quality,
                    "combined_score": round(combined, 4),
                }
            )
        results.sort(key=lambda r: -r["combined_score"])
        return results[:top_k]

    def stats(self) -> dict:
        return {"documents": len(self.documents), "chunks": len(self.chunks)}
