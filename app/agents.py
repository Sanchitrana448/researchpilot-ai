"""
Research pipeline: planner -> retrieval -> evidence -> critic -> synthesis ->
citation validator -> quality control.

Every stage is a plain function returning a dict rather than a class holding
state, so the whole trace can be serialised straight into the API response and
inspected. Debugging a bad answer means reading the stage outputs, not
attaching a debugger.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import List

from . import llm
from .retrieval import Index

NEGATION_WORDS = {"not", "no", "never", "cannot", "can't", "isn't", "doesn't", "won't", "unlikely", "false"}


@dataclass
class ResearchResult:
    id: str
    question: str
    sub_questions: List[str]
    evidence: List[dict]
    contradictions: List[dict]
    uncertainty_notes: List[str]
    answer: str
    citations: List[str]
    metrics: dict
    created_at: float = field(default_factory=time.time)


def _detect_contradictions(evidence: List[dict]) -> List[dict]:
    """Flag evidence pairs that cover the same ground but disagree on polarity.

    Requires 6+ shared terms before considering a pair at all, then checks
    whether exactly one side contains a negation. Tuned to stay quiet: a critic
    that cries wolf gets ignored, which is worse than not having one. Output is
    a prompt for human review, not a verdict.
    """
    contradictions = []
    for i in range(len(evidence)):
        for j in range(i + 1, len(evidence)):
            a, b = evidence[i], evidence[j]
            a_words = set(re.findall(r"[a-z']+", a["text"].lower()))
            b_words = set(re.findall(r"[a-z']+", b["text"].lower()))
            overlap = a_words & b_words
            if len(overlap) < 6:
                continue
            a_neg = bool(a_words & NEGATION_WORDS)
            b_neg = bool(b_words & NEGATION_WORDS)
            if a_neg != b_neg:
                contradictions.append(
                    {
                        "evidence_a": a["id"],
                        "evidence_b": b["id"],
                        "shared_terms": sorted(overlap)[:8],
                        "note": "One passage contains a negation the other does not on overlapping "
                        "terms -- possible conflicting claims. Review manually.",
                    }
                )
    return contradictions


def run_research(index: Index, question: str, top_k: int = 6) -> ResearchResult:
    t0 = time.time()

    # 1-2. Planner: decompose into sub-questions
    sub_questions = llm.decompose_question(question)

    # 3-4. Retrieval: gather + rank evidence per sub-question, then merge/dedupe
    seen_chunk_ids = set()
    evidence: List[dict] = []
    for sq in sub_questions:
        for r in index.search(sq, top_k=top_k):
            if r["chunk_id"] in seen_chunk_ids:
                continue
            seen_chunk_ids.add(r["chunk_id"])
            evidence.append(r)
    evidence.sort(key=lambda r: -r["combined_score"])
    evidence = evidence[: top_k * 2]

    # 5-6. Extract citation-ready evidence blocks
    labeled_evidence = []
    for i, e in enumerate(evidence, start=1):
        cid = f"S{i}"
        labeled_evidence.append({**e, "id": cid})

    # 7. Critic: cross-check contradictions
    contradictions = _detect_contradictions(labeled_evidence)

    # 8. Uncertainty
    uncertainty_notes = []
    if not labeled_evidence:
        uncertainty_notes.append("No supporting evidence was retrieved for this question.")
    elif len(labeled_evidence) < 2:
        uncertainty_notes.append("Only one evidence passage was found; confidence is low.")
    if contradictions:
        uncertainty_notes.append(
            f"{len(contradictions)} potential contradiction(s) detected between sources -- see 'contradictions'."
        )
    low_quality = [e for e in labeled_evidence if e["source_quality"] < 0.7]
    if low_quality:
        uncertainty_notes.append(
            f"{len(low_quality)} passage(s) came from lower-confidence sources."
        )

    # 9-10. Synthesis: cited narrative answer
    answer = llm.synthesize_report(question, labeled_evidence)

    # 11. Citation validator: every [Sx] cited in answer must exist in evidence
    cited_ids = set(re.findall(r"\[(S\d+)\]", answer))
    valid_ids = {e["id"] for e in labeled_evidence}
    invalid_citations = cited_ids - valid_ids
    if invalid_citations:
        uncertainty_notes.append(
            f"Citation validator flagged unsupported citation ids: {sorted(invalid_citations)}"
        )

    metrics = {
        "sub_questions": len(sub_questions),
        "evidence_retrieved": len(labeled_evidence),
        "contradictions_found": len(contradictions),
        "citations_used": len(cited_ids),
        "citation_coverage": round(len(cited_ids) / max(1, len(labeled_evidence)), 2),
        "invalid_citations": len(invalid_citations),
        "llm_mode": "live" if llm.llm_available() else "offline_extractive",
        "latency_seconds": round(time.time() - t0, 3),
    }

    return ResearchResult(
        id=str(uuid.uuid4())[:10],
        question=question,
        sub_questions=sub_questions,
        evidence=labeled_evidence,
        contradictions=contradictions,
        uncertainty_notes=uncertainty_notes,
        answer=answer,
        citations=sorted(cited_ids),
        metrics=metrics,
    )
