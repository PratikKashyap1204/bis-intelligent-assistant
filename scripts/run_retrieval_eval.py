"""
Milestone 5: offline retrieval evaluation runner.

Runs data/eval/retrieval_eval_dataset.json against the REAL database
using BOTH existing retrieval backends (keyword, vector) — no LLM, no
external API calls, read-only DB queries only.

Usage:
    python scripts/run_retrieval_eval.py [--save baseline|after]

Saves a JSON report to data/eval/reports/<tag>_<timestamp>.json for
reproducibility (before/after comparisons for any retrieval change).
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
from app.services.evaluation import DEFAULT_K_VALUES, EvaluationReport, load_eval_dataset, run_evaluation  # noqa: E402
from app.services.retrieval import BISRetrievalService, VectorRetrievalBackend  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "data" / "eval" / "reports"


def _report_to_dict(report: EvaluationReport) -> dict:
    return {
        "method": report.method,
        "k_values": list(report.k_values),
        "scored_case_count": report.scored_case_count,
        "no_result_case_count": report.no_result_case_count,
        "recall_at_k": {str(k): v for k, v in report.recall_at_k.items()},
        "precision_at_k": {str(k): v for k, v in report.precision_at_k.items()},
        "abstention_rate": report.abstention_rate,
        "by_category": {
            cat: {
                "scored_case_count": sub.scored_case_count,
                "no_result_case_count": sub.no_result_case_count,
                "recall_at_k": {str(k): v for k, v in sub.recall_at_k.items()},
                "precision_at_k": {str(k): v for k, v in sub.precision_at_k.items()},
                "abstention_rate": sub.abstention_rate,
            }
            for cat, sub in report.by_category().items()
        },
        "cases": [
            {
                "id": cr.case.id,
                "category": cr.case.category,
                "question": cr.case.question,
                "expected_clause_ids": cr.case.expected_clause_ids,
                "expect_no_result": cr.case.expect_no_result,
                "retrieved_clause_ids": cr.retrieved_clause_ids,
                "recall_at_k": {str(k): v for k, v in cr.recall_at_k.items()},
                "precision_at_k": {str(k): v for k, v in cr.precision_at_k.items()},
                "correctly_abstained": cr.correctly_abstained,
            }
            for cr in report.case_results
        ],
    }


def _print_report(report: EvaluationReport) -> None:
    print(f"\n=== {report.method.upper()} retrieval — overall "
          f"({report.scored_case_count} scored cases, {report.no_result_case_count} no-result cases) ===")
    print("Recall@K   :", {k: round(v, 3) for k, v in report.recall_at_k.items()})
    print("Precision@K:", {k: round(v, 3) for k, v in report.precision_at_k.items()})
    if report.abstention_rate is not None:
        print(f"Abstention rate (no-result cases correctly returning nothing): "
              f"{report.abstention_rate:.3f}")

    print(f"\n--- {report.method.upper()} by category ---")
    for cat, sub in sorted(report.by_category().items()):
        if sub.scored_case_count:
            print(f"  {cat:20s} n={sub.scored_case_count:2d}  "
                  f"Recall@5={sub.recall_at_k.get(5, float('nan')):.3f}  "
                  f"Precision@5={sub.precision_at_k.get(5, float('nan')):.3f}")
        else:
            print(f"  {cat:20s} n=0 scored (no-result category); "
                  f"abstention_rate={sub.abstention_rate}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=None, help="Tag for saving a JSON report (e.g. 'baseline').")
    args = parser.parse_args()

    cases = load_eval_dataset()
    print(f"Loaded {len(cases)} evaluation cases.")

    session = SessionLocal()
    try:
        vector_provider = get_default_embedding_provider()
        retrieval_service = BISRetrievalService(
            session, backends={"vector": VectorRetrievalBackend(session, vector_provider)}
        )

        reports = {}
        for method in ("keyword", "vector"):
            report = run_evaluation(
                session, retrieval_service, cases, method=method, k_values=DEFAULT_K_VALUES
            )
            reports[method] = report
            _print_report(report)

        if args.save:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = REPORTS_DIR / f"{args.save}_{timestamp}.json"
            out_path.write_text(
                json.dumps(
                    {m: _report_to_dict(r) for m, r in reports.items()},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"\nSaved report to {out_path}")
    finally:
        session.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
