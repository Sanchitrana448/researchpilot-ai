# ResearchPilot AI

[![tests](https://github.com/Sanchitrana448/researchpilot-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Sanchitrana448/researchpilot-ai/actions/workflows/ci.yml)

Live: https://researchpilot-ai-yw7y.onrender.com  
*(free tier, so it may take ~50s to wake outside weekday daytime)*

Ask a research question against a set of documents and get back an answer where every claim points to a specific passage. If the evidence is thin or the sources disagree, it says so instead of papering over it.

Works with no API key at all. Without one it falls back to extractive synthesis, which stitches together retrieved passages with citations rather than generating prose. Set `OPENAI_API_KEY` and it switches to LLM synthesis on the next request, no code change.

## The part I actually care about

Most RAG demos will happily cite `[S3]` when there is no S3. The synthesis step here is untrusted by design: after the answer comes back, a validator pulls every `[Sx]` marker out of the text and checks it against the evidence that was actually retrieved. Anything that doesn't match gets flagged in `uncertainty_notes` and counted in `invalid_citations`.

The offline path can't fabricate a citation at all, since it only ever emits text it pulled from the corpus. That's the main reason I kept it as a real mode rather than a stub.

## Pipeline

```
question
  -> planner            decompose into 3-5 sub-questions
  -> retrieval          TF-IDF over chunked, deduplicated corpus
  -> evidence           rank and label passages [S1..Sn]
  -> critic             scan evidence pairs for contradictions
  -> synthesis          cited answer (LLM, or extractive fallback)
  -> citation validator verify every [Sx] maps to real evidence
  -> quality control    uncertainty notes, coverage, latency
```

Each stage is a plain function returning a dict, so you can inspect the whole trace from the API response.

## Running it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://localhost:8000. Docker works too:

```bash
docker build -t researchpilot .
docker run -p 8000:8000 researchpilot
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/ingest/text` | Add raw text to the corpus |
| `POST` | `/ingest/file` | Upload a `.pdf`/`.txt`/`.md` |
| `POST` | `/research` | Run the pipeline on a question |
| `GET` | `/research/{id}` | Fetch a past result |
| `GET` | `/health` | Liveness and corpus stats |

Swagger docs at `/docs`.

## Retrieval details

Two things in `retrieval.py` that aren't obvious from the endpoint list:

**Duplicate chunks are dropped at ingest.** If the same paragraph appears in two uploaded documents, it gets indexed once. Without this, quoting the same passage twice looks like two independent sources agreeing, which inflates apparent confidence for no reason.

**Ranking isn't just cosine similarity.** It over-fetches `top_k * 3` by similarity, then re-ranks with `0.75 * similarity + 0.25 * source_quality`, where source quality is a crude heuristic on the source name (`.gov`/`.edu`/arxiv score higher than an anonymous paste). Swapping that for a real domain-authority or recency model is a drop-in change.

## Tests

```bash
pytest tests/ -v
```

Five tests: chunk windowing and overlap, index add/search, duplicate deduplication, the full offline pipeline end to end, and the zero-document case (which should return no evidence and refuse to answer, not hallucinate).

## Limitations

- TF-IDF only. No dense embeddings, so it matches on lexical overlap and will miss paraphrases. This is the first thing I'd replace.
- The contradiction detector is a negation heuristic: it looks for evidence pairs sharing 6+ terms where one contains a negation word and the other doesn't. It catches blunt "X causes Y" vs "X does not cause Y" pairs and nothing subtler. It's deliberately tuned to be quiet rather than thorough, since a noisy critic is worse than none.
- The index is in-memory. Restart and the corpus is gone.
- Source-quality scoring is a hardcoded domain list, not a real model.

## Stack

Python, FastAPI, scikit-learn for TF-IDF, pypdf, Pydantic, Docker, pytest. The LLM layer targets the OpenAI Chat Completions shape, so any compatible endpoint works.
