import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.retrieval import Index
from benchmark_data import PASSAGES, check_overlap


@pytest.fixture(scope="module")
def idx():
    index = Index()
    for pid, text in PASSAGES.items():
        index.add_document(text, f"{pid}.txt")
    return index


def test_benchmark_queries_follow_overlap_rules():
    check_overlap()


def test_vector_search_finds_reworded_question(idx):
    # Shares no content words with p04, so keyword matching cannot find it.
    q = "Why should the exam data stay untouched while I adjust settings?"
    assert idx.search(q, top_k=3, rerank=False)[0]["source"] == "p04.txt"


def test_off_topic_question_retrieves_nothing(idx):
    for q in ("What is the capital of Australia?", "How long should I boil an egg?"):
        assert idx.search(q, rerank=True) == []


def test_rerank_adds_cross_encoder_score(idx):
    top = idx.search("What does a model registry store?", top_k=1, rerank=True)[0]
    assert top["source"] == "p08.txt"
    assert top["rerank_score"] > 0
    assert top["retrieval_mode"] == "dense+rerank"
    plain = idx.search("What does a model registry store?", top_k=1, rerank=False)[0]
    assert plain["rerank_score"] is None and plain["retrieval_mode"] == "dense"


def test_api_reports_models_and_rerank_flag():
    with TestClient(app) as client:
        client.post("/sample")
        body = client.post("/research", json={"question": "How should a RAG system be evaluated?", "rerank": False}).json()
        assert body["metrics"]["retrieval_mode"] == "dense"
        health = client.get("/health").json()
        assert health["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert health["vector_store"].startswith("qdrant")


def test_rerank_request_is_refused_where_unavailable(monkeypatch):
    from app import reranker

    monkeypatch.setattr(reranker, "AVAILABLE", False)
    with TestClient(app) as client:
        client.post("/sample")
        r = client.post("/research", json={"question": "How should a RAG system be evaluated?", "rerank": True})
        assert r.status_code == 400
        assert "memory" in r.json()["detail"]


def test_off_topic_question_is_refused_end_to_end():
    # Regression: offline sub-question templates ("what evidence supports
    # claims about ...") used to match the RAG sample corpus on their own.
    from app.agents import run_research
    from app.sample_data import SAMPLE_DOCS

    index = Index()
    for d in SAMPLE_DOCS:
        index.add_document(d["text"], d["source"])
    for q in ("How do I repot an orchid?", "How long should I boil an egg?"):
        result = run_research(index, q)
        assert result.evidence == []
