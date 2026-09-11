"""
RAG demo script (Milestone 3).

Runs the real RAG flow (real sentence-transformers embeddings + the
deterministic extractive answer generator — no hosted LLM) against the
existing pilot database (1 Standard, 1 QCO document, 129 clauses).

Demonstrates:
  1. A question the corpus can actually answer -> grounded answer with
     citations pointing at real clauses/pages/URLs.
  2. A question the corpus cannot answer -> a grounded "not enough
     information" response, NOT a fabricated answer.

Usage:
    python scripts/rag_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from app.db.session import SessionLocal  # noqa: E402
from app.services.answer_generation import ExtractiveAnswerGenerationProvider  # noqa: E402
from app.services.embedding_provider import get_default_embedding_provider  # noqa: E402
from app.services.rag import RAGConfig, RAGService  # noqa: E402
from app.services.retrieval import BISRetrievalService, VectorRetrievalBackend  # noqa: E402

IN_SCOPE_QUESTION = "Which appliances need mandatory certification before they can be sold?"
OUT_OF_SCOPE_QUESTION = "What is the maximum speed limit for cars on Indian national highways?"


def _print_result(question: str, result) -> None:
    print(f"\n{'=' * 90}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 90}")
    print(f"grounded={result.grounded}  retrieval_method={result.retrieval_method}  "
          f"context_used={result.context_used}")
    print("\n--- ANSWER ---")
    print(result.answer)
    print("\n--- CITATIONS ---")
    if not result.citations:
        print("  (none)")
    for c in result.citations:
        print(f"  [{c.index}] {c.standard_number or '(no standard)'} "
              f"({c.document_type or 'unknown type'})")
        print(f"      Document : {c.document_title}")
        print(f"      Clause   : {c.clause_number} ({c.clause_type})")
        print(f"      Page     : {c.page_number}")
        print(f"      Source   : {c.source_url}")
        print(f"      Score    : {c.relevance_score:.4f}")


def main() -> int:
    session = SessionLocal()
    try:
        provider = get_default_embedding_provider()
        backend = VectorRetrievalBackend(session, provider)
        retrieval_service = BISRetrievalService(session, backends={"vector": backend})
        rag = RAGService(
            session,
            retrieval_service,
            ExtractiveAnswerGenerationProvider(),
            config=RAGConfig(),
        )

        # --- Diagnostic: show raw scores so the min_score_vector threshold
        # in RAGConfig is justified by evidence, not guesswork. ---
        print("=== Raw retrieval scores (diagnostic, before RAGConfig filtering) ===")
        for question in (IN_SCOPE_QUESTION, OUT_OF_SCOPE_QUESTION):
            raw = retrieval_service.search(question, method="vector", limit=3)
            scores = [f"{r.relevance.score:.4f}" for r in raw]
            print(f"  {question!r}\n    top scores: {scores}")

        result_1 = rag.answer(IN_SCOPE_QUESTION)
        _print_result(IN_SCOPE_QUESTION, result_1)

        result_2 = rag.answer(OUT_OF_SCOPE_QUESTION)
        _print_result(OUT_OF_SCOPE_QUESTION, result_2)

        print(f"\n{'=' * 90}")
        print("VERIFICATION")
        print(f"{'=' * 90}")
        print(f"In-scope question grounded (expected True) : {result_1.grounded}")
        print(f"Out-of-scope question grounded (expected False): {result_2.grounded}")
        if result_2.grounded:
            print("WARNING: out-of-scope question was answered as grounded — "
                  "this means the system may be fabricating relevance. Investigate "
                  "RAGConfig.min_score_vector.")
        else:
            print("Confirmed: system did NOT fabricate an answer for the "
                  "out-of-scope question.")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
