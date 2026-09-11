"""
Milestone 6 offline configuration sweep.

Runs the existing M5 evaluation dataset (unchanged) against candidate
hybrid-fusion and vector-abstention settings. Read-only DB queries; no
LLM/API calls; does not modify retrieval defaults.

Usage:
    python scripts/run_m6_sweep.py [--save]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from app.db.session import SessionLocal  # noqa: E402
from app.services.embedding_provider import get_default_embedding_provider  # noqa: E402
from app.services.evaluation import DEFAULT_K_VALUES, load_eval_dataset, run_evaluation  # noqa: E402
from app.services.retrieval import (  # noqa: E402
    BISRetrievalService,
    HybridRetrievalBackend,
    KeywordRetrievalBackend,
    VectorRetrievalBackend,
)

REPORTS_DIR = PROJECT_ROOT / "data" / "eval" / "reports"


def _metrics(report) -> dict:
    def rnd(d):
        return {str(k): round(v, 3) if v == v else None for k, v in d.items()}

    cats = {}
    for cat, sub in report.by_category().items():
        cats[cat] = {
            "recall": rnd(sub.recall_at_k),
            "precision": rnd(sub.precision_at_k),
            "abstention_rate": sub.abstention_rate,
            "scored": sub.scored_case_count,
            "no_result": sub.no_result_case_count,
        }
    return {
        "recall": rnd(report.recall_at_k),
        "precision": rnd(report.precision_at_k),
        "abstention_rate": report.abstention_rate,
        "by_category": cats,
    }


def _print(label: str, m: dict) -> None:
    r, p = m["recall"], m["precision"]
    print(
        f"{label:42s}  "
        f"R@1={r['1']:.3f} R@3={r['3']:.3f} R@5={r['5']:.3f} R@10={r['10']:.3f}  "
        f"P@1={p['1']:.3f} P@5={p['5']:.3f} P@10={p['10']:.3f}  "
        f"abst={m['abstention_rate']}"
    )
    mc = m["by_category"].get("multi_clause", {})
    if mc.get("scored"):
        print(f"{'':42s}  multi_clause R@5={mc['recall']['5']:.3f} R@10={mc['recall']['10']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    cases = load_eval_dataset()
    provider = get_default_embedding_provider()
    session = SessionLocal()
    rows = []

    try:
        keyword = KeywordRetrievalBackend(session)

        # --- vector abstention sweep (standalone vector backend) ---
        print("=== vector min_similarity sweep (cosine similarity floor) ===")
        for thresh in (None, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
            vector = VectorRetrievalBackend(session, provider, min_similarity=thresh)
            service = BISRetrievalService(
                session, backend=keyword, backends={"keyword": keyword, "vector": vector}
            )
            report = run_evaluation(session, service, cases, method="vector")
            m = _metrics(report)
            label = f"vector min_sim={thresh}"
            _print(label, m)
            rows.append({"config": label, **m})

        # --- hybrid RRF ---
        print("\n=== hybrid RRF sweep (vector min_sim=None) ===")
        vector = VectorRetrievalBackend(session, provider, min_similarity=None)
        for rrf_k in (10, 20, 60):
            hybrid = HybridRetrievalBackend(keyword, vector, rrf_k=rrf_k, fusion="rrf")
            service = BISRetrievalService(
                session,
                backend=keyword,
                backends={"keyword": keyword, "vector": vector, "hybrid": hybrid},
            )
            report = run_evaluation(session, service, cases, method="hybrid")
            m = _metrics(report)
            label = f"hybrid rrf_k={rrf_k}"
            _print(label, m)
            rows.append({"config": label, **m})

        # --- hybrid weighted minmax ---
        print("\n=== hybrid weighted minmax sweep ===")
        for wk, wv in ((0.3, 0.7), (0.5, 0.5), (0.7, 0.3)):
            hybrid = HybridRetrievalBackend(
                keyword, vector, fusion="weighted", keyword_weight=wk, vector_weight=wv
            )
            service = BISRetrievalService(
                session,
                backend=keyword,
                backends={"keyword": keyword, "vector": vector, "hybrid": hybrid},
            )
            report = run_evaluation(session, service, cases, method="hybrid")
            m = _metrics(report)
            label = f"hybrid weighted kw={wk} vec={wv}"
            _print(label, m)
            rows.append({"config": label, **m})

        # --- hybrid RRF + vector abstention (best-looking rrf_k from above, plus 60) ---
        print("\n=== hybrid RRF k=60 + vector min_similarity ===")
        for thresh in (0.25, 0.30, 0.35):
            vector_t = VectorRetrievalBackend(session, provider, min_similarity=thresh)
            hybrid = HybridRetrievalBackend(keyword, vector_t, rrf_k=60, fusion="rrf")
            service = BISRetrievalService(
                session,
                backend=keyword,
                backends={"keyword": keyword, "vector": vector_t, "hybrid": hybrid},
            )
            report = run_evaluation(session, service, cases, method="hybrid")
            m = _metrics(report)
            label = f"hybrid rrf_k=60 + vec_min={thresh}"
            _print(label, m)
            rows.append({"config": label, **m})

        if args.save:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out = REPORTS_DIR / f"m6_sweep_{ts}.json"
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"\nSaved {out}")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
