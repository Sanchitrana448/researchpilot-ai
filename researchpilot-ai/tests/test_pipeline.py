import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import Index, chunk_text
from app.agents import run_research


def test_chunking_produces_overlap_windows():
    text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_text(text, chunk_size=500, overlap=80)
    assert len(chunks) >= 2
    assert all(chunks)


def test_index_add_and_search():
    idx = Index()
    idx.add_document(
        "Retrieval-augmented generation (RAG) combines a retriever with a "
        "generator so that language models can ground answers in external "
        "documents instead of relying purely on parametric memory.",
        "source-a.txt",
    )
    idx.add_document(
        "Vector databases store embeddings and support approximate nearest "
        "neighbour search, which is commonly used to power RAG retrieval.",
        "source-b.txt",
    )
    results = idx.search("What is RAG?", top_k=3)
    assert len(results) > 0
    assert results[0]["similarity"] > 0


def test_duplicate_chunks_are_deduplicated():
    idx = Index()
    text = "This exact passage will be repeated across two documents for testing."
    idx.add_document(text, "doc1.txt")
    idx.add_document(text, "doc2.txt")
    assert idx.stats()["chunks"] == 1


def test_research_pipeline_end_to_end_offline():
    idx = Index()
    idx.add_document(
        "Studies show that unit testing reduces production defects. "
        "Teams with high test coverage report fewer regressions.",
        "study1.txt",
    )
    idx.add_document(
        "Some practitioners argue unit testing does not significantly reduce "
        "defects when integration testing is weak.",
        "study2.txt",
    )
    result = run_research(idx, "Does unit testing reduce production defects?", top_k=4)
    assert result.question.startswith("Does unit testing")
    assert result.metrics["llm_mode"] == "offline_extractive"
    assert isinstance(result.evidence, list)
    assert result.metrics["latency_seconds"] >= 0


def test_research_with_no_documents_returns_no_evidence():
    idx = Index()
    result = run_research(idx, "Anything?", top_k=3)
    assert result.evidence == []
    assert "Insufficient evidence" in result.answer
