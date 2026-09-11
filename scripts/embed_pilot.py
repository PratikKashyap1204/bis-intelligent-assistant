"""
Embedding pilot script (Milestone 2).

Generates embeddings for all currently-ingested Clause rows using the
real production embedding provider (local sentence-transformers model —
see app/services/embedding_provider.py). This is the ONLY place the real
model is loaded; automated tests use DeterministicHashEmbeddingProvider
instead so the test suite stays fast and offline.

Safe to re-run: unchanged clause content is skipped (no duplicate rows,
no repeated model inference) — see app/services/embedding_service.py.

Usage:
    python scripts/embed_pilot.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from app.db.session import SessionLocal  # noqa: E402
from app.services.embedding_provider import get_default_embedding_provider  # noqa: E402
from app.services.embedding_service import ClauseEmbeddingService  # noqa: E402


def main() -> int:
    print("=== BIS Embedding Pilot ===\n")

    provider = get_default_embedding_provider()
    print(f"Provider  : {provider.model_name}")

    t0 = time.time()
    # Loading the real model happens lazily on first use — trigger it here
    # so the "model load" time is visible separately from "embedding" time.
    dimension = provider.dimension
    print(f"Dimension : {dimension}")
    print(f"Model load time: {time.time() - t0:.1f}s\n")

    session = SessionLocal()
    try:
        service = ClauseEmbeddingService(session, provider)

        t1 = time.time()
        summary = service.embed_all_clauses()
        session.commit()
        elapsed = time.time() - t1

        print("=== Embedding Run Summary ===")
        print(f"Clauses considered : {summary.total_considered}")
        print(f"Created            : {summary.created}")
        print(f"Updated            : {summary.updated}")
        print(f"Skipped (unchanged): {summary.skipped_unchanged}")
        print(f"Skipped (empty)    : {summary.skipped_empty}")
        print(f"Embedding time     : {elapsed:.2f}s")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
