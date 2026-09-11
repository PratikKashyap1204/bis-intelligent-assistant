"""
Ad-hoc semantic retrieval demo against the real pilot data (Milestone 2
Step 9 verification). Not a test — prints results for manual inspection.

Queries are deliberately phrased differently from the stored clause
wording to check whether vector search finds relevant clauses beyond
exact keyword overlap.
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
from app.services.embedding_provider import get_default_embedding_provider  # noqa: E402
from app.services.retrieval import BISRetrievalService, VectorRetrievalBackend  # noqa: E402

QUERIES = [
    "Which appliances need mandatory certification before they can be sold?",
    "Who is in charge of enforcing this rule?",
    "What happens if a company does not follow this order?",
    "Is there a grace period before this rule takes effect?",
]


def main() -> int:
    session = SessionLocal()
    try:
        provider = get_default_embedding_provider()
        backend = VectorRetrievalBackend(session, provider)
        service = BISRetrievalService(session, backends={"vector": backend})

        for query in QUERIES:
            print(f"\n{'=' * 90}\nQUERY: {query!r}\n{'=' * 90}")
            results = service.search(query, method="vector", limit=3)
            if not results:
                print("  (no results)")
                continue
            for i, r in enumerate(results, start=1):
                print(
                    f"  #{i} score={r.relevance.score:.4f} clause={r.clause_number!r} "
                    f"type={r.clause_type} page={r.page_number}"
                )
                text = (r.clause_text or "").strip().replace("\n", " ")
                print(f"     text: {text[:220]}{'...' if len(text) > 220 else ''}")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
