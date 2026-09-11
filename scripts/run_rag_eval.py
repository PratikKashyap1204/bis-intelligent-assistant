"""
Milestone 7: offline RAG/context evaluation runner.

Runs data/eval/retrieval_eval_dataset.json through RAGService + the
extractive answer provider (never an LLM) for keyword, vector, and
hybrid. Scores the *selected context window* (not the raw retrieval list).

Usage:
    python scripts/run_rag_eval.py [--save tag]
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
from app.services.evaluation import RagEvaluationReport, load_eval_dataset, run_rag_evaluation  # noqa: E402
from app.services.retrieval import build_retrieval_service  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "data" / "eval" / "reports"


def _report_to_dict(report: RagEvaluationReport) -> dict:
    return {
        "method": report.method,
        "scored_case_count": report.scored_case_count,
        "no_result_case_count": report.no_result_case_count,
        "context_recall": report.context_recall,
        "context_precision": report.context_precision,
        "grounded_rate": report.grounded_rate,
        "ungrounded_rate": report.ungrounded_rate,
        "citation_integrity_rate": report.citation_integrity_rate,
        "by_category": {
            cat: {
                "scored_case_count": sub.scored_case_count,
                "no_result_case_count": sub.no_result_case_count,
                "context_recall": sub.context_recall,
                "context_precision": sub.context_precision,
                "grounded_rate": sub.grounded_rate,
                "ungrounded_rate": sub.ungrounded_rate,
                "citation_integrity_rate": sub.citation_integrity_rate,
            }
            for cat, sub in report.by_category().items()
        },
        "cases": [
            {
                "id": cr.case.id,
                "category": cr.case.category,
                "expect_no_result": cr.case.expect_no_result,
                "expected_clause_ids": cr.case.expected_clause_ids,
                "context_clause_ids": cr.context_clause_ids,
                "cited_clause_ids": cr.cited_clause_ids,
                "grounded": cr.grounded,
                "context_used": cr.context_used,
                "context_recall": cr.context_recall,
                "context_precision": cr.context_precision,
                "correctly_ungrounded": cr.correctly_ungrounded,
                "citations_subset_of_context": cr.citations_subset_of_context,
            }
            for cr in report.case_results
        ],
    }


def _print_report(report: RagEvaluationReport) -> None:
    print(
        f"\n=== {report.method.upper()} RAG context — "
        f"{report.scored_case_count} scored, {report.no_result_case_count} no-result ==="
    )
    print(f"Context recall   : {report.context_recall}")
    print(f"Context precision: {report.context_precision}")
    print(f"Grounded rate (in-scope): {report.grounded_rate}")
    print(f"Ungrounded rate (no-result): {report.ungrounded_rate}")
    print(f"Citation integrity: {report.citation_integrity_rate}")
    print(f"--- {report.method.upper()} by category ---")
    for cat, sub in sorted(report.by_category().items()):
        if sub.scored_case_count:
            print(
                f"  {cat:20s} n={sub.scored_case_count:2d}  "
                f"ctxR={sub.context_recall}  ctxP={sub.context_precision}  "
                f"grounded={sub.grounded_rate}"
            )
        else:
            print(
                f"  {cat:20s} n=0 scored; ungrounded_rate={sub.ungrounded_rate}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=None, help="Tag for saving a JSON report.")
    args = parser.parse_args()

    cases = load_eval_dataset()
    print(f"Loaded {len(cases)} evaluation cases.")

    session = SessionLocal()
    try:
        provider = get_default_embedding_provider()
        retrieval_service = build_retrieval_service(session, provider)
        reports = {}
        for method in ("keyword", "vector", "hybrid"):
            report = run_rag_evaluation(session, retrieval_service, cases, method=method)
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
