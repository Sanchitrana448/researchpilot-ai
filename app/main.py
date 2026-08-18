from __future__ import annotations

import io
import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import llm
from .agents import ResearchResult, run_research
from .retrieval import Index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("researchpilot")

app = FastAPI(
    title="ResearchPilot AI",
    description="Autonomous AI research & decision engine — agentic RAG with citation-grounded synthesis.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

INDEX = Index()
RESULTS: dict[str, ResearchResult] = {}

STATIC_DIR = Path(__file__).parent.parent / "frontend"


class IngestTextRequest(BaseModel):
    text: str
    source: str = "user-provided-text"


class ResearchRequest(BaseModel):
    question: str
    top_k: int = 6


@app.get("/health")
def health():
    return {"status": "ok", "llm_mode": "live" if llm.llm_available() else "offline_extractive", **INDEX.stats()}


@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest):
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty")
    doc = INDEX.add_document(req.text, req.source)
    return {"doc_id": doc.id, "source": doc.source, "quality_score": doc.quality_score, **INDEX.stats()}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    content = await file.read()
    text = ""
    if file.filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise HTTPException(400, f"Failed to parse PDF: {e}")
    else:
        text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(400, "No extractable text found in file")

    doc = INDEX.add_document(text, file.filename)
    return {"doc_id": doc.id, "source": doc.source, "quality_score": doc.quality_score, **INDEX.stats()}


@app.get("/corpus/stats")
def corpus_stats():
    return INDEX.stats()


@app.post("/research")
def research(req: ResearchRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")
    if INDEX.stats()["documents"] == 0:
        raise HTTPException(400, "No documents ingested yet. POST /ingest/text or /ingest/file first.")
    result = run_research(INDEX, req.question, top_k=req.top_k)
    RESULTS[result.id] = result
    return asdict(result)


@app.get("/research/{result_id}")
def get_research(result_id: str):
    result = RESULTS.get(result_id)
    if not result:
        raise HTTPException(404, "Result not found")
    return asdict(result)


@app.get("/research")
def list_research():
    return [{"id": r.id, "question": r.question, "created_at": r.created_at} for r in RESULTS.values()]


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>ResearchPilot AI</h1><p>Frontend not built. See /docs for API.</p>"
