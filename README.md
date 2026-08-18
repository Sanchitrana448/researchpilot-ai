# ResearchPilot AI

**Autonomous multi-agent research & decision engine.** Give it a research question and a set of source documents; it decomposes the question, retrieves and ranks evidence, cross-checks sources for contradictions, and produces a cited, uncertainty-aware answer — never fabricating a source.

Runs **fully offline out of the box** (no API key required) using deterministic extractive planning and synthesis. Set `OPENAI_API_KEY` to automatically upgrade to full LLM-based reasoning with zero code changes.

## Why this project exists

Recruiters and hiring panels for AI/ML engineering roles want evidence of three things: agentic system design, retrieval/RAG engineering, and responsible-AI judgment (grounding, citations, honesty about uncertainty). This project demonstrates all three in a single, runnable artifact instead of a toy notebook.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000
```

Or with Docker:

```bash
docker build -t researchpilot .
docker run -p 8000:8000 researchpilot
```

Optional — enable live LLM synthesis:

```bash
export OPENAI_API_KEY=sk-...
uvicorn app.main:app --reload
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | /ingest/text | Add a raw text source to the corpus |
| POST | /ingest/file | Upload a .pdf/.txt/.md source |
| POST | /research | Run the full agent pipeline on a question |
| GET | /research/{id} | Retrieve a past result |
| GET | /health | Liveness + corpus stats |

Full interactive docs at /docs (OpenAPI/Swagger).

## Evaluation

tests/test_pipeline.py covers: chunking correctness, retrieval relevance, duplicate-chunk detection, end-to-end offline pipeline execution, and the zero-evidence edge case. Run with:

```bash
pytest tests/ -v
```

## Tech stack

Python · FastAPI · scikit-learn (TF-IDF retrieval) · pypdf · Pydantic · Docker · pytest. LLM layer is provider-agnostic (OpenAI-compatible Chat Completions).
