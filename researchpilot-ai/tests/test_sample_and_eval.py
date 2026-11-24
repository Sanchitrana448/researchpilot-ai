import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.agents import _detect_contradictions, run_research
from app.main import app
from app.retrieval import Index
from app.sample_data import EVAL_QUESTIONS, SAMPLE_DOCS


def test_sample_endpoint_loads_corpus_and_is_idempotent():
    client = TestClient(app)
    first = client.post("/sample").json()
    second = client.post("/sample").json()
    assert first["documents"] >= len(SAMPLE_DOCS)
    assert second["chunks"] == first["chunks"]


def test_eval_questions_retrieve_expected_source_with_no_invalid_citations():
    idx = Index()
    for d in SAMPLE_DOCS:
        idx.add_document(d["text"], d["source"])
    for question, expected in EVAL_QUESTIONS:
        result = run_research(idx, question)
        assert expected in {e["source"] for e in result.evidence}
        assert result.metrics["invalid_citations"] == 0


def test_critic_ignores_stopword_only_overlap():
    a = {"id": "S1", "text": "The cat sat on a mat and it was in the sun."}
    b = {"id": "S2", "text": "A dog is not in the house and it was at the door."}
    assert _detect_contradictions([a, b]) == []
