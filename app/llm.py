"""
Pluggable LLM provider.

ResearchPilot AI runs fully offline out of the box (extractive planning +
extractive synthesis, no external calls, no API key required) so recruiters
and reviewers can run the demo with zero setup.

If OPENAI_API_KEY is set, the system automatically upgrades to real LLM-backed
planning and synthesis via the OpenAI Chat Completions API, with no code
changes required.
"""
from __future__ import annotations

import os
import re
from typing import List

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def llm_available() -> bool:
    return bool(OPENAI_API_KEY)


def _call_openai(system: str, user: str, max_tokens: int = 600) -> str:
    import httpx

    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def decompose_question(question: str, n: int = 4) -> List[str]:
    """Break a research question into sub-questions."""
    if llm_available():
        try:
            text = _call_openai(
                "You are a research planning agent. Break the user's research "
                "question into 3-5 crisp, independently-answerable sub-questions. "
                "Return one per line, no numbering, no extra commentary.",
                question,
                max_tokens=250,
            )
            lines = [l.strip("-• \t") for l in text.splitlines() if l.strip()]
            if lines:
                return lines[:5]
        except Exception:
            pass

    # Offline heuristic planner: split on conjunctions / punctuation and
    # generate standard analytical angles as a deterministic fallback.
    parts = re.split(r"\band\b|,|;|\?", question, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    angles = [
        f"What is the current state of: {question.strip('? ')}?",
        f"What evidence supports or contradicts claims about: {question.strip('? ')}?",
        f"What are the key risks, limitations or open questions regarding: {question.strip('? ')}?",
    ]
    sub_qs = parts[:2] + angles
    seen, out = set(), []
    for q in sub_qs:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:n]


def synthesize_report(question: str, evidence_blocks: List[dict]) -> str:
    """Produce a cited synthesis from ranked evidence blocks.

    evidence_blocks: [{"id": "S1", "text": ..., "source": ...}, ...]
    """
    if llm_available():
        try:
            context = "\n\n".join(
                f"[{b['id']}] (source: {b['source']}) {b['text']}" for b in evidence_blocks
            )
            return _call_openai(
                "You are a research synthesis agent. Using ONLY the numbered "
                "evidence blocks provided, write a concise, well-cited answer "
                "to the research question. Cite evidence inline like [S1]. "
                "If evidence is insufficient for a claim, explicitly say so. "
                "Never invent facts not present in the evidence.",
                f"Research question: {question}\n\nEvidence:\n{context}",
                max_tokens=700,
            )
        except Exception:
            pass

    # Offline extractive synthesis: stitch together the top-ranked evidence
    # sentences with explicit citations. Fully deterministic, zero hallucination
    # risk, and transparent about what is/isn't covered.
    if not evidence_blocks:
        return (
            "Insufficient evidence was retrieved to answer this question. "
            "No claims are made. Consider adding more source documents."
        )
    lines = [f"Based on {len(evidence_blocks)} retrieved evidence passage(s):\n"]
    for b in evidence_blocks:
        snippet = b["text"].strip().replace("\n", " ")
        if len(snippet) > 320:
            snippet = snippet[:317] + "..."
        lines.append(f"- {snippet} [{b['id']}]")
    lines.append(
        "\nNote: generated in offline extractive mode (no LLM key configured). "
        "Set OPENAI_API_KEY to enable full natural-language synthesis."
    )
    return "\n".join(lines)
