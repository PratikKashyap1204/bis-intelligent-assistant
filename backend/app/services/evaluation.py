"""
Offline retrieval evaluation framework (Milestone 5).

Loads a hand-curated evaluation dataset (see
``data/eval/retrieval_eval_dataset.json``, built by inspecting real,
already-ingested corpus content — never fabricated), runs it against the
EXISTING ``BISRetrievalService`` (keyword and/or vector backends,
unmodified interface from Milestone 1/2), and computes standard
information-retrieval metrics (Recall@K, Precision@K) using ``Clause.id``
as the sole ground-truth identity.

This module does NOT call any LLM or external API — it is pure, offline,
deterministic comparison of returned ``RetrievalResult.clause_id`` values
against a case's ``expected_clause_ids``.

Handling "no relevant result expected" cases (out_of_scope /
absent_from_corpus, see the dataset's ``expect_no_result`` flag):
    Recall@K is not a meaningful concept when there is nothing to recall
    (0/0) — instead these cases contribute to a separate "correctly
    abstained" rate: did the backend return zero results, or did it
    return results anyway (which, for a keyword backend that found no
    term matches, is usually correctly empty; for a vector backend with
    ``min_similarity=None``, which returns nearest neighbours regardless
    of true relevance). With a cosine-similarity floor
    (``VectorRetrievalBackend.min_similarity``) vector search can return
    [] when every neighbour is below that floor. These cases are EXCLUDED from the
    Recall@K/Precision@K averages (see ``EvaluationReport`` — reported
    separately) rather than silently forced into a misleading number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from app.services.answer_generation import ExtractiveAnswerGenerationProvider
from app.services.rag import RAGConfig, RAGService
from app.services.retrieval import BISRetrievalService

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "eval" / "retrieval_eval_dataset.json"
)

DEFAULT_K_VALUES: Sequence[int] = (1, 3, 5, 10)


# ---------------------------------------------------------------------------
# Dataset loading / validation
# ---------------------------------------------------------------------------


class EvaluationDatasetError(ValueError):
    """Raised when the evaluation dataset file is missing or malformed."""


@dataclass(frozen=True)
class EvalCase:
    """One evaluation case. ``expected_clause_ids`` is the ONLY field used
    to score a retrieval hit — ``expected_document_ids``/
    ``expected_standard_ids`` are supporting provenance for humans, never
    used as a match key (matches the project rule that clause_number/
    document/standard groupings are not a substitute for Clause.id)."""

    id: str
    category: str
    question: str
    expected_clause_ids: List[int]
    expected_document_ids: List[int] = field(default_factory=list)
    expected_standard_ids: List[int] = field(default_factory=list)
    expect_no_result: bool = False
    rationale: str = ""


def load_eval_dataset(path: Optional[Path] = None) -> List[EvalCase]:
    """
    Load and validate the evaluation dataset JSON file.

    Raises EvaluationDatasetError on any structural problem (missing
    file, malformed JSON, missing required fields, or a case whose
    ``expect_no_result`` is inconsistent with a non-empty
    ``expected_clause_ids`` list) rather than silently proceeding with
    bad ground truth.
    """
    dataset_path = path or DEFAULT_DATASET_PATH
    if not dataset_path.exists():
        raise EvaluationDatasetError(f"Evaluation dataset not found: {dataset_path}")

    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationDatasetError(f"Evaluation dataset is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict) or "cases" not in raw:
        raise EvaluationDatasetError("Evaluation dataset must be a JSON object with a 'cases' key.")

    cases_raw = raw["cases"]
    if not isinstance(cases_raw, list) or not cases_raw:
        raise EvaluationDatasetError("Evaluation dataset 'cases' must be a non-empty list.")

    cases: List[EvalCase] = []
    seen_ids = set()
    required_fields = ("id", "category", "question", "expected_clause_ids")

    for i, raw_case in enumerate(cases_raw):
        if not isinstance(raw_case, dict):
            raise EvaluationDatasetError(f"Case at index {i} is not a JSON object.")
        missing = [f for f in required_fields if f not in raw_case]
        if missing:
            raise EvaluationDatasetError(
                f"Case at index {i} is missing required field(s): {missing}"
            )
        if raw_case["id"] in seen_ids:
            raise EvaluationDatasetError(f"Duplicate case id: {raw_case['id']!r}")
        seen_ids.add(raw_case["id"])

        expected_clause_ids = raw_case["expected_clause_ids"]
        if not isinstance(expected_clause_ids, list):
            raise EvaluationDatasetError(
                f"Case {raw_case['id']!r}: expected_clause_ids must be a list."
            )
        expect_no_result = bool(raw_case.get("expect_no_result", False))
        if expect_no_result and expected_clause_ids:
            raise EvaluationDatasetError(
                f"Case {raw_case['id']!r}: expect_no_result=True but "
                f"expected_clause_ids is non-empty — inconsistent ground truth."
            )
        if not expect_no_result and not expected_clause_ids:
            raise EvaluationDatasetError(
                f"Case {raw_case['id']!r}: expect_no_result=False but "
                f"expected_clause_ids is empty — inconsistent ground truth."
            )

        cases.append(
            EvalCase(
                id=raw_case["id"],
                category=raw_case["category"],
                question=raw_case["question"],
                expected_clause_ids=list(expected_clause_ids),
                expected_document_ids=list(raw_case.get("expected_document_ids", [])),
                expected_standard_ids=list(raw_case.get("expected_standard_ids", [])),
                expect_no_result=expect_no_result,
                rationale=raw_case.get("rationale", ""),
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: Sequence[int], expected_ids: Sequence[int], k: int) -> float:
    """
    Fraction of expected_ids found within the top-k retrieved_ids.

    Undefined (raises ValueError) if expected_ids is empty — callers must
    exclude expect_no_result cases from this metric (see EvaluationReport).
    """
    if not expected_ids:
        raise ValueError("recall_at_k is undefined for an empty expected_ids set.")
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for e in expected_ids if e in top_k)
    return hits / len(expected_ids)


def precision_at_k(retrieved_ids: Sequence[int], expected_ids: Sequence[int], k: int) -> float:
    """
    Fraction of the top-k retrieved_ids that are relevant (in expected_ids).

    If fewer than k results were returned, precision is computed over the
    number actually returned (standard IR convention for short result
    lists) — returns 0.0 if nothing was returned at all.
    """
    top_k = list(retrieved_ids[:k])
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in expected_ids)
    return hits / len(top_k)


# ---------------------------------------------------------------------------
# Per-case / aggregate results
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case: EvalCase
    retrieved_clause_ids: List[int]
    recall_at_k: Dict[int, Optional[float]]  # None for expect_no_result cases
    precision_at_k: Dict[int, Optional[float]]
    correctly_abstained: Optional[bool]  # only meaningful for expect_no_result cases


@dataclass
class EvaluationReport:
    """
    Aggregate metrics for one retrieval method run over one dataset.

    ``recall_at_k`` / ``precision_at_k``: averaged ONLY over cases where
    expect_no_result is False (a real relevant set exists) — see module
    docstring for why 0/0 cases are excluded rather than distorting the
    average.

    ``abstention_rate``: over expect_no_result cases only, the fraction
    where the backend returned ZERO results (the "correct" behaviour for
    a question with no true answer). Reported separately, never blended
    into recall/precision.
    """

    method: str
    k_values: Sequence[int]
    case_results: List[CaseResult]
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    abstention_rate: Optional[float]
    scored_case_count: int  # cases with a real expected set (used for recall/precision)
    no_result_case_count: int  # expect_no_result cases (used for abstention_rate)

    def by_category(self) -> Dict[str, "EvaluationReport"]:
        """Split this report into one sub-report per question category."""
        categories = sorted({cr.case.category for cr in self.case_results})
        out: Dict[str, EvaluationReport] = {}
        for cat in categories:
            subset = [cr for cr in self.case_results if cr.case.category == cat]
            out[cat] = _aggregate(self.method, self.k_values, subset)
        return out


def _aggregate(method: str, k_values: Sequence[int], case_results: List[CaseResult]) -> EvaluationReport:
    scored = [cr for cr in case_results if not cr.case.expect_no_result]
    no_result = [cr for cr in case_results if cr.case.expect_no_result]

    recall_at_k: Dict[int, float] = {}
    precision_at_k: Dict[int, float] = {}
    for k in k_values:
        recall_vals = [cr.recall_at_k[k] for cr in scored if cr.recall_at_k[k] is not None]
        precision_vals = [cr.precision_at_k[k] for cr in scored if cr.precision_at_k[k] is not None]
        recall_at_k[k] = (sum(recall_vals) / len(recall_vals)) if recall_vals else float("nan")
        precision_at_k[k] = (
            (sum(precision_vals) / len(precision_vals)) if precision_vals else float("nan")
        )

    abstention_rate: Optional[float] = None
    if no_result:
        abstained = sum(1 for cr in no_result if cr.correctly_abstained)
        abstention_rate = abstained / len(no_result)

    return EvaluationReport(
        method=method,
        k_values=k_values,
        case_results=case_results,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        abstention_rate=abstention_rate,
        scored_case_count=len(scored),
        no_result_case_count=len(no_result),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_evaluation(
    session: Session,
    retrieval_service: BISRetrievalService,
    cases: List[EvalCase],
    *,
    method: str,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    retrieval_limit: Optional[int] = None,
) -> EvaluationReport:
    """
    Run every case in ``cases`` against ``retrieval_service`` using the
    given ``method`` ("keyword", "vector", or "hybrid") and compute Recall@K /
    Precision@K. Pure read-only queries against the existing DB — makes
    no LLM/API calls and mutates nothing.
    """
    # NOTE: BISRetrievalService.search() returns a mix of standard-,
    # document-, and clause-level RetrievalResult rows (only the last has
    # a non-None clause_id — see retrieval.py). If the raw search limit
    # were only max(k_values), standard/document-level hits could occupy
    # top ranks and push real clause candidates entirely out of the
    # window BEFORE this function ever filters to clause-only ids —
    # discovered empirically while evaluating q19 (a cross_document
    # case): the top 3 raw hits were document-title matches with
    # clause_id=None, which silently truncated the real clause hits at
    # rank 10. Requesting a generous raw limit here (independent of the
    # K values actually being scored) avoids that — this is an
    # evaluation-harness fix, not a change to retrieval ranking/weights.
    limit = retrieval_limit or max(50, max(k_values) * 5)
    case_results: List[CaseResult] = []

    for case in cases:
        results = retrieval_service.search(case.question, method=method, limit=limit)
        retrieved_clause_ids = [r.clause_id for r in results if r.clause_id is not None]

        if case.expect_no_result:
            case_results.append(
                CaseResult(
                    case=case,
                    retrieved_clause_ids=retrieved_clause_ids,
                    recall_at_k={k: None for k in k_values},
                    precision_at_k={k: None for k in k_values},
                    correctly_abstained=(len(retrieved_clause_ids) == 0),
                )
            )
            continue

        recall_map = {k: recall_at_k(retrieved_clause_ids, case.expected_clause_ids, k) for k in k_values}
        precision_map = {
            k: precision_at_k(retrieved_clause_ids, case.expected_clause_ids, k) for k in k_values
        }
        case_results.append(
            CaseResult(
                case=case,
                retrieved_clause_ids=retrieved_clause_ids,
                recall_at_k=recall_map,
                precision_at_k=precision_map,
                correctly_abstained=None,
            )
        )

    return _aggregate(method, k_values, case_results)


# ---------------------------------------------------------------------------
# RAG / context evaluation (Milestone 7)
# ---------------------------------------------------------------------------
#
# Retrieval metrics (above) score ranked search lists. The public answer
# path only ever grounds on the *selected* RAG context (dedup + score
# floor + max_context_items, default 5). This harness scores that window
# with the extractive provider — never an LLM — so DEFAULT_METHOD can be
# chosen from evidence rather than retrieval@10 alone.


@dataclass
class RagCaseResult:
    case: EvalCase
    context_clause_ids: List[int]
    cited_clause_ids: List[int]
    grounded: bool
    context_used: int
    retrieval_method: str
    context_recall: Optional[float]
    context_precision: Optional[float]
    correctly_ungrounded: Optional[bool]
    citations_subset_of_context: bool
    evidence_status: str


@dataclass
class RagEvaluationReport:
    """Aggregate RAG/context metrics for one retrieval method."""

    method: str
    case_results: List[RagCaseResult]
    scored_case_count: int
    no_result_case_count: int
    context_recall: Optional[float]
    context_precision: Optional[float]
    grounded_rate: Optional[float]
    ungrounded_rate: Optional[float]
    citation_integrity_rate: float
    supported_rate: Optional[float]
    partially_supported_rate: Optional[float]
    insufficient_rate: Optional[float]
    out_of_scope_rate: Optional[float]

    def by_category(self) -> Dict[str, "RagEvaluationReport"]:
        categories = sorted({cr.case.category for cr in self.case_results})
        return {
            cat: _aggregate_rag(
                self.method, [cr for cr in self.case_results if cr.case.category == cat]
            )
            for cat in categories
        }


def _aggregate_rag(method: str, case_results: List[RagCaseResult]) -> RagEvaluationReport:
    scored = [cr for cr in case_results if not cr.case.expect_no_result]
    no_result = [cr for cr in case_results if cr.case.expect_no_result]

    def _mean(vals: List[float]) -> Optional[float]:
        return (sum(vals) / len(vals)) if vals else None

    return RagEvaluationReport(
        method=method,
        case_results=case_results,
        scored_case_count=len(scored),
        no_result_case_count=len(no_result),
        context_recall=_mean([cr.context_recall for cr in scored if cr.context_recall is not None]),
        context_precision=_mean(
            [cr.context_precision for cr in scored if cr.context_precision is not None]
        ),
        grounded_rate=_mean([1.0 if cr.grounded else 0.0 for cr in scored]) if scored else None,
        ungrounded_rate=(
            _mean([1.0 if cr.correctly_ungrounded else 0.0 for cr in no_result])
            if no_result
            else None
        ),
        citation_integrity_rate=(
            sum(1 for cr in case_results if cr.citations_subset_of_context) / len(case_results)
            if case_results
            else 1.0
        ),
        supported_rate=_mean(
            [1.0 if cr.evidence_status == "supported" else 0.0 for cr in scored]
        )
        if scored
        else None,
        partially_supported_rate=_mean(
            [1.0 if cr.evidence_status == "partially_supported" else 0.0 for cr in scored]
        )
        if scored
        else None,
        insufficient_rate=_mean(
            [1.0 if cr.evidence_status == "insufficient" else 0.0 for cr in no_result]
        )
        if no_result
        else None,
        out_of_scope_rate=_mean(
            [1.0 if cr.evidence_status == "out_of_scope" else 0.0 for cr in no_result]
        )
        if no_result
        else None,
    )


def run_rag_evaluation(
    session: Session,
    retrieval_service: BISRetrievalService,
    cases: List[EvalCase],
    *,
    method: str,
    config: Optional[RAGConfig] = None,
) -> RagEvaluationReport:
    """
    Run extractive RAG on every eval case for ``method``.

    Offline and deterministic: uses ExtractiveAnswerGenerationProvider
    (no LLM, no network). Scores whether gold ``Clause.id`` values appear
    in the selected context window, whether no-result cases stay
    ungrounded, whether citations are a subset of that context, and the
    categorical evidence_status rates (not a numeric confidence score).
    """
    rag = RAGService(
        session,
        retrieval_service,
        ExtractiveAnswerGenerationProvider(),
        config=config or RAGConfig(),
    )
    case_results: List[RagCaseResult] = []

    for case in cases:
        answer = rag.answer(case.question, method=method)
        context_ids = [s.clause_id for s in answer.sources if s.clause_id is not None]
        cited_ids = [c.clause_id for c in answer.citations if c.clause_id is not None]
        context_set = set(context_ids)
        citations_ok = all(
            (cid in context_set) for cid in cited_ids
        ) and all(
            (c.clause_id is None or c.clause_id in context_set) for c in answer.citations
        )

        if case.expect_no_result:
            case_results.append(
                RagCaseResult(
                    case=case,
                    context_clause_ids=context_ids,
                    cited_clause_ids=cited_ids,
                    grounded=answer.grounded,
                    context_used=answer.context_used,
                    retrieval_method=answer.retrieval_method,
                    context_recall=None,
                    context_precision=None,
                    correctly_ungrounded=(not answer.grounded),
                    citations_subset_of_context=citations_ok,
                    evidence_status=answer.evidence_status,
                )
            )
            continue

        gold = case.expected_clause_ids
        hits = sum(1 for g in gold if g in context_set)
        case_results.append(
            RagCaseResult(
                case=case,
                context_clause_ids=context_ids,
                cited_clause_ids=cited_ids,
                grounded=answer.grounded,
                context_used=answer.context_used,
                retrieval_method=answer.retrieval_method,
                context_recall=(hits / len(gold)) if gold else None,
                context_precision=(hits / len(context_ids)) if context_ids else 0.0,
                correctly_ungrounded=None,
                citations_subset_of_context=citations_ok,
                evidence_status=answer.evidence_status,
            )
        )

    return _aggregate_rag(method, case_results)
