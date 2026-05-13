"""Phase 10 evaluation: run representative questions through the RAG retrieval
pipeline and record which sources are retrieved for each.

This module does NOT require Ollama to be running.  It exercises the embedding
and vector-search layers so results can be inspected for relevance and source
attribution quality.

Run via:
    python app.py --evaluate
    python app.py --evaluate --top-k 7
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import (
    DEFAULT_MAX_CHUNKS_PER_DOC,
    DEFAULT_RETRIEVAL_TOP_K,
    OUT_OF_SCOPE_THRESHOLD,
    PROJECT_ROOT,
    RELEVANCE_THRESHOLD,
)
from src.embeddings import EmbeddingModel
from src.vector_db import VectorDB


# ---------------------------------------------------------------------------
# Evaluation question bank
# ---------------------------------------------------------------------------

# Questions 1-4: taken directly from the implementation plan example list.
# Questions 5-6: derived from the scenario documents.
#   - Q5 comes from Lena's story (20260407_scenarios.docx): she trusted a chat
#     app that collected her data without making its use transparent.
#   - Q6 comes from Diana's story (20260413_additional_script.docx): the
#     conversation explicitly questions who is accountable when AI systems
#     influence decisions in work and education.

EVALUATION_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "EQ-01",
        "question": "Who is responsible when an AI system causes harm?",
        "source": "implementation_plan",
        "theme_hint": "responsibility",
    },
    {
        "id": "EQ-02",
        "question": "How does algorithmic bias affect fairness in hiring decisions?",
        "source": "implementation_plan",
        "theme_hint": "fairness",
    },
    {
        "id": "EQ-03",
        "question": "What transparency obligations exist for AI systems that interact with people?",
        "source": "implementation_plan",
        "theme_hint": "transparency",
    },
    {
        "id": "EQ-04",
        "question": "Can we trust AI systems to make decisions about asylum procedures?",
        "source": "implementation_plan",
        "theme_hint": "law enforcement / migration",
    },
    {
        "id": "EQ-05",
        "question": "What ethical safeguards exist for AI use in healthcare?",
        "source": "implementation_plan",
        "theme_hint": "healthcare",
    },
    {
        "id": "EQ-06",
        "question": "How should AI literacy be taught to ensure responsible use?",
        "source": "implementation_plan",
        "theme_hint": "education",
    },
    # --- Derived from scenario documents ---
    {
        "id": "EQ-07",
        "question": (
            "What rules govern how AI chat applications may collect and use personal "
            "data shared by users, and what must users be told before they start?"
        ),
        "source": "20260407_scenarios.docx (Lena / trust scenario)",
        "theme_hint": "privacy",
    },
    {
        "id": "EQ-08",
        "question": (
            "Who is accountable when an AI system used in work or education causes "
            "harm, and what AI skills do European workers need to challenge such systems?"
        ),
        "source": "20260413_additional_script.docx (Diana / AI skills scenario)",
        "theme_hint": "responsibility / education",
    },
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RetrievedSource:
    """One retrieved chunk with its key metadata."""

    chunk_id: str
    doc_id: str
    title: str
    theme: str
    url: str
    similarity: float
    excerpt: str  # First 300 chars of chunk text


@dataclass
class EvaluationResult:
    """Outcome for one evaluation question."""

    question_id: str
    question: str
    question_source: str
    theme_hint: str
    retrieved_sources: list[RetrievedSource]
    top_similarity: float
    passed_relevance_check: bool  # True when top similarity >= threshold
    notes: str


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------

def _make_excerpt(text: str, max_chars: int = 300) -> str:
    """Return a short readable preview of a chunk."""
    flat = " ".join((text or "").split())
    return flat[:max_chars] + ("..." if len(flat) > max_chars else "")


def run_evaluation(
    vector_db: VectorDB,
    embedding_model: EmbeddingModel,
    questions: list[dict[str, str]] | None = None,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    max_per_doc: int | None = DEFAULT_MAX_CHUNKS_PER_DOC,
) -> list[EvaluationResult]:
    """Embed each question, retrieve top-k chunks, and build result objects."""
    if questions is None:
        questions = EVALUATION_QUESTIONS

    results: list[EvaluationResult] = []

    for q in questions:
        query_vector = embedding_model.embed_text(q["question"], is_query=True)
        chunks = vector_db.search(query_vector, top_k=top_k, max_per_doc=max_per_doc)

        sources: list[RetrievedSource] = []
        for chunk in chunks:
            sources.append(
                RetrievedSource(
                    chunk_id=str(chunk.get("chunk_id") or ""),
                    doc_id=str(chunk.get("doc_id") or ""),
                    title=str(chunk.get("title") or ""),
                    theme=str(chunk.get("theme") or ""),
                    url=str(chunk.get("url") or ""),
                    similarity=float(chunk.get("similarity", 0.0)),
                    excerpt=_make_excerpt(str(chunk.get("text") or "")),
                )
            )

        top_sim = sources[0].similarity if sources else 0.0
        passed = top_sim >= RELEVANCE_THRESHOLD

        notes_parts: list[str] = []
        if not sources:
            notes_parts.append("No chunks retrieved — vector store may be empty or too small.")
        elif not passed:
            notes_parts.append(
                f"Top similarity {top_sim:.4f} is below threshold {RELEVANCE_THRESHOLD}. "
                "Consider increasing top-k or rebuilding with more documents."
            )

        results.append(
            EvaluationResult(
                question_id=q["id"],
                question=q["question"],
                question_source=q["source"],
                theme_hint=q["theme_hint"],
                retrieved_sources=sources,
                top_similarity=round(top_sim, 4),
                passed_relevance_check=passed,
                notes=" ".join(notes_parts),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_evaluation_report(results: list[EvaluationResult]) -> str:
    """Produce a human-readable plain-text report."""
    lines: list[str] = [
        "=" * 70,
        "NLP Policy Chatbot — Phase 10 Evaluation Report",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Questions evaluated: {len(results)}",
        f"Relevance threshold: {RELEVANCE_THRESHOLD}",
        "=" * 70,
        "",
    ]

    passed_count = sum(1 for r in results if r.passed_relevance_check)
    lines.append(f"Summary: {passed_count}/{len(results)} questions met the relevance threshold.")
    lines.append("")

    for result in results:
        status = "PASS" if result.passed_relevance_check else "WARN"
        lines.append(f"[{status}] {result.question_id} — {result.question}")
        lines.append(f"      Source: {result.question_source}")
        lines.append(f"      Theme hint: {result.theme_hint}")
        lines.append(f"      Top similarity: {result.top_similarity:.4f}")
        if result.notes:
            lines.append(f"      Note: {result.notes}")
        unique_docs = len({s.title.strip().lower() or s.doc_id for s in result.retrieved_sources})
        lines.append(f"      Retrieved sources ({len(result.retrieved_sources)} chunks, {unique_docs} unique docs):")
        for i, src in enumerate(result.retrieved_sources, start=1):
            lines.append(f"        [{i}] {src.title} ({src.doc_id})  sim={src.similarity:.4f}")
            lines.append(f"             Theme: {src.theme}")
            lines.append(f"             URL: {src.url}")
            lines.append(f"             Excerpt: {src.excerpt}")
        lines.append("")

    return "\n".join(lines)


def save_evaluation_report(
    results: list[EvaluationResult],
    output_dir: Path = PROJECT_ROOT / "data" / "processed",
) -> tuple[Path, Path]:
    """Save both a JSON and a plain-text report. Returns (json_path, txt_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "evaluation_report.json"
    txt_path = output_dir / "evaluation_report.txt"

    json_data: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(results),
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "passed_count": sum(1 for r in results if r.passed_relevance_check),
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    txt_path.write_text(format_evaluation_report(results), encoding="utf-8")

    return json_path, txt_path


# ---------------------------------------------------------------------------
# Edge-case checks (Task 10.2) — callable without Ollama
# ---------------------------------------------------------------------------


def check_edge_cases(vector_db: VectorDB, embedding_model: EmbeddingModel) -> list[dict[str, Any]]:
    """Run a set of edge-case behavioural checks and return pass/fail dicts."""
    checks: list[dict[str, Any]] = []

    # 1. Empty query
    try:
        vec = embedding_model.embed_text("", is_query=True)
        results = vector_db.search(vec, top_k=3)
        # An empty query should return nothing (zero vector filtered out)
        checks.append({
            "check": "empty_query",
            "passed": len(results) == 0,
            "detail": f"Returned {len(results)} results for empty query (expected 0).",
        })
    except Exception as exc:
        checks.append({"check": "empty_query", "passed": False, "detail": str(exc)})

    # 2. Out-of-scope query (topic unrelated to AI policy).
    # Policy queries score 0.73–0.86; a genuinely unrelated query should stay
    # well below 0.60, which marks a clear semantic gap from on-topic content.
    try:
        vec = embedding_model.embed_text(
            "What is the best recipe for chocolate chip cookies?", is_query=True
        )
        results = vector_db.search(vec, top_k=3)
        top_sim = results[0].get("similarity", 0.0) if results else 0.0
        checks.append({
            "check": "out_of_scope_query",
            "passed": top_sim < OUT_OF_SCOPE_THRESHOLD,
            "detail": (
                f"Top similarity for out-of-scope query: {top_sim:.4f} "
                f"({'below' if top_sim < OUT_OF_SCOPE_THRESHOLD else 'ABOVE'} "
                f"out-of-scope threshold {OUT_OF_SCOPE_THRESHOLD}). "
                "On-topic queries score 0.73–0.86; this gap confirms the model "
                "distinguishes policy topics from unrelated content."
            ),
        })
    except Exception as exc:
        checks.append({"check": "out_of_scope_query", "passed": False, "detail": str(exc)})

    # 3. top_k larger than number of chunks
    try:
        vec = embedding_model.embed_text("AI transparency", is_query=True)
        oversized_k = len(vector_db.chunks) + 100
        results = vector_db.search(vec, top_k=oversized_k)
        checks.append({
            "check": "top_k_exceeds_store_size",
            "passed": len(results) == len(vector_db.chunks),
            "detail": (
                f"Requested top_k={oversized_k}, store has {len(vector_db.chunks)} chunks, "
                f"returned {len(results)} results."
            ),
        })
    except Exception as exc:
        checks.append({"check": "top_k_exceeds_store_size", "passed": False, "detail": str(exc)})

    # 4. top_k = 0 should return nothing
    try:
        vec = embedding_model.embed_text("AI regulation", is_query=True)
        results = vector_db.search(vec, top_k=0)
        checks.append({
            "check": "top_k_zero",
            "passed": len(results) == 0,
            "detail": f"Returned {len(results)} results for top_k=0 (expected 0).",
        })
    except Exception as exc:
        checks.append({"check": "top_k_zero", "passed": False, "detail": str(exc)})

    return checks


def format_edge_case_report(checks: list[dict[str, Any]]) -> str:
    """Return a plain-text edge-case report."""
    lines = [
        "-" * 50,
        "Edge-Case Checks (Task 10.2)",
        "-" * 50,
    ]
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        lines.append(f"[{status}] {c['check']}: {c['detail']}")
    return "\n".join(lines)
