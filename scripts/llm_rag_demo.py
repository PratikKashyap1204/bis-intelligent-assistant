"""
Real LLM RAG demo script (Milestone 4).

Unlike scripts/rag_demo.py (which uses the deterministic
ExtractiveAnswerGenerationProvider), this script makes REAL calls to the
configured LLM provider (OpenAI). It is NEVER run automatically by the
test suite or by any other script.

Safety / cost control:
  - Refuses to run at all if OPENAI_API_KEY is not set in the environment
    (checked BEFORE constructing any client or touching the network).
  - Makes exactly 2 LLM calls total: one in-scope question, one
    out-of-scope question. No loops, no retries, no batch processing.
  - Uses the configured LLM_TIMEOUT_SECONDS (default 20s) — no unbounded
    waits.
  - Does not regenerate embeddings or touch the database beyond read-only
    retrieval queries against the existing pilot data.

Usage:
    export OPENAI_API_KEY=sk-...      # only in your local shell / .env, never committed
    python scripts/llm_rag_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from app.config import settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.services.answer_generation import LLMAnswerGenerationProvider  # noqa: E402
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
        print(f"  [{c.index}] {c.standard_number or '(no standard)'} ({c.document_type or 'unknown type'})")
        print(f"      Document : {c.document_title}")
        print(f"      Clause   : {c.clause_number} ({c.clause_type})")
        print(f"      Page     : {c.page_number}")
        print(f"      Source   : {c.source_url}")
        print(f"      Score    : {c.relevance_score:.4f}")


def main() -> int:
    if not settings.OPENAI_API_KEY:
        print(
            "OPENAI_API_KEY is not set — refusing to run this demo.\n"
            "This script makes real (billed) LLM API calls and will not run without "
            "an explicitly configured key.\n\n"
            "To run it:\n"
            "  1. Add OPENAI_API_KEY=sk-... to your local .env (never commit this file).\n"
            "  2. Optionally set ANSWER_PROVIDER=llm in .env (this script uses the LLM "
            "provider directly regardless of ANSWER_PROVIDER).\n"
            "  3. Re-run: python scripts/llm_rag_demo.py\n\n"
            "No API call was made."
        )
        return 1

    print("=== Real LLM RAG Demo (Milestone 4) ===")
    print(f"Model    : {settings.LLM_MODEL_NAME}")
    print(f"Temp     : {settings.LLM_TEMPERATURE}")
    print(f"Timeout  : {settings.LLM_TIMEOUT_SECONDS}s")
    print("This script makes exactly 2 real LLM API calls (one per question below).")

    session = SessionLocal()
    try:
        embedding_provider = get_default_embedding_provider()
        backend = VectorRetrievalBackend(session, embedding_provider)
        retrieval_service = BISRetrievalService(session, backends={"vector": backend})

        llm_provider = LLMAnswerGenerationProvider(
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        rag = RAGService(session, retrieval_service, llm_provider, config=RAGConfig())

        result_1 = rag.answer(IN_SCOPE_QUESTION)  # LLM call #1 (only if context found)
        _print_result(IN_SCOPE_QUESTION, result_1)

        result_2 = rag.answer(OUT_OF_SCOPE_QUESTION)  # LLM call #2 only if context found;
        # expected to short-circuit with ZERO LLM calls since retrieval should find no
        # sufficiently relevant context for this question (see RAGConfig.min_score_vector).
        _print_result(OUT_OF_SCOPE_QUESTION, result_2)

        print(f"\n{'=' * 90}")
        print("VERIFICATION")
        print(f"{'=' * 90}")
        print(f"In-scope question grounded (expected True) : {result_1.grounded}")
        print(f"Out-of-scope question grounded (expected False): {result_2.grounded}")
        if result_2.grounded:
            print(
                "WARNING: out-of-scope question was answered as grounded — this would mean "
                "either retrieval returned unexpectedly high-scoring matches, or the LLM "
                "fabricated a citation despite the grounding prompt. Investigate before "
                "trusting this provider."
            )
        else:
            print("Confirmed: system did NOT fabricate an answer for the out-of-scope question.")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
