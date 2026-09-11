"""
Tests for app.services.evaluation (Milestone 5 offline retrieval
evaluation framework).

Synthetic sample data only (IS 9999 / example.invalid), same pattern as
test_retrieval.py / test_ingestion_integration.py — never touches the
live bis_db pilot rows and makes no LLM/API calls.
"""

from __future__ import annotations

import json
import math

import pytest

from app.services.bis_ingestion import (
    BISIngestionService,
    RawClause,
    RawDocument,
    RawStandard,
)
from app.services.embedding_provider import DeterministicHashEmbeddingProvider
from app.services.embedding_service import ClauseEmbeddingService
from app.services.evaluation import (
    EvalCase,
    EvaluationDatasetError,
    load_eval_dataset,
    precision_at_k,
    recall_at_k,
    run_evaluation,
)
from app.services.retrieval import BISRetrievalService, VectorRetrievalBackend

SAMPLE_STANDARD = RawStandard(
    is_number="IS 8888",
    title="Sample Evaluation Standard",
    year=2050,
    status="ACTIVE",
    scope="Sample scope for evaluation tests.",
    source_url="https://example.invalid/sample-eval-standard",
)

SAMPLE_DOCUMENT = RawDocument(
    title="Sample Evaluation QCO Circular",
    document_type="QCO",
    source_url="https://example.invalid/sample-eval-document.pdf",
    standard_is_number="IS 8888",
    content_hash="deadbeef" * 8,
    language="en",
)

# Two clauses share clause_number "1" — mirrors the real, observed
# collision (clause_number is not globally unique within a document);
# Clause.id (not clause_number) is what evaluation ground truth uses.
SAMPLE_CLAUSES = [
    RawClause(
        clause_number="1",
        title=None,
        content="Certification is mandatory for widgets under this QCO.",
        page_number=1,
        language="en",
        clause_type="CLAUSE",
        sequence_in_document=1,
    ),
    RawClause(
        clause_number="1",
        title=None,
        content="Penalty for contravention of this widget QCO is a fine.",
        page_number=2,
        language="en",
        clause_type="CLAUSE",
        sequence_in_document=2,
    ),
]


def _seed(db_session):
    ingestion = BISIngestionService(db_session)
    standard, _ = ingestion.ingest_standard(SAMPLE_STANDARD)
    document, _, _ = ingestion.ingest_document(
        SAMPLE_DOCUMENT, SAMPLE_CLAUSES, standard=standard
    )
    db_session.commit()
    clauses = document.clauses
    return standard, document, clauses


# ---------------------------------------------------------------------------
# Pure metric unit tests
# ---------------------------------------------------------------------------


def test_recall_at_k_basic():
    assert recall_at_k([1, 2, 3], [1, 2], k=3) == 1.0
    assert recall_at_k([1, 2, 3], [1, 2], k=1) == 0.5
    assert recall_at_k([3, 4, 5], [1, 2], k=5) == 0.0


def test_recall_at_k_raises_for_empty_expected():
    with pytest.raises(ValueError):
        recall_at_k([1, 2], [], k=5)


def test_precision_at_k_basic():
    assert precision_at_k([1, 2, 3], [1, 2], k=3) == pytest.approx(2 / 3)
    assert precision_at_k([1, 2], [1, 2], k=5) == 1.0  # fewer than k returned
    assert precision_at_k([], [1, 2], k=5) == 0.0


def test_precision_at_k_no_relevant_hits():
    assert precision_at_k([9, 8, 7], [1, 2], k=3) == 0.0


# ---------------------------------------------------------------------------
# Dataset loading / validation
# ---------------------------------------------------------------------------


def test_load_real_eval_dataset_succeeds():
    cases = load_eval_dataset()
    assert len(cases) >= 20
    categories = {c.category for c in cases}
    assert {"direct", "paraphrase", "multi_clause", "cross_document",
            "out_of_scope", "absent_from_corpus"} <= categories
    # Every case must use clause_id-based ground truth, never bare numbers
    # that look like clause_number strings.
    for case in cases:
        assert isinstance(case.expected_clause_ids, list)
        assert all(isinstance(cid, int) for cid in case.expected_clause_ids)


def test_load_eval_dataset_missing_file(tmp_path):
    with pytest.raises(EvaluationDatasetError):
        load_eval_dataset(tmp_path / "does_not_exist.json")


def test_load_eval_dataset_rejects_malformed_json(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(EvaluationDatasetError):
        load_eval_dataset(bad_file)


def test_load_eval_dataset_rejects_missing_cases_key(tmp_path):
    f = tmp_path / "dataset.json"
    f.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(EvaluationDatasetError):
        load_eval_dataset(f)


def test_load_eval_dataset_rejects_inconsistent_expect_no_result(tmp_path):
    f = tmp_path / "dataset.json"
    f.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "bad1",
                        "category": "direct",
                        "question": "?",
                        "expected_clause_ids": [1],
                        "expect_no_result": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationDatasetError):
        load_eval_dataset(f)


def test_load_eval_dataset_rejects_empty_expected_without_no_result_flag(tmp_path):
    f = tmp_path / "dataset.json"
    f.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "bad2",
                        "category": "direct",
                        "question": "?",
                        "expected_clause_ids": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationDatasetError):
        load_eval_dataset(f)


def test_load_eval_dataset_rejects_duplicate_case_ids(tmp_path):
    f = tmp_path / "dataset.json"
    case = {"id": "dup", "category": "direct", "question": "?", "expected_clause_ids": [1]}
    f.write_text(json.dumps({"cases": [case, dict(case)]}), encoding="utf-8")
    with pytest.raises(EvaluationDatasetError):
        load_eval_dataset(f)


# ---------------------------------------------------------------------------
# Integration: run_evaluation against a real (seeded) database, using the
# keyword backend (offline, no vector model needed) — mirrors what
# scripts/run_retrieval_eval.py does against the real corpus.
# ---------------------------------------------------------------------------


def test_run_evaluation_keyword_scores_a_direct_hit(db_session):
    _, _, clauses = _seed(db_session)
    certification_clause = clauses[0]
    assert certification_clause.clause_number == "1"

    case = EvalCase(
        id="t1",
        category="direct",
        question="Is certification mandatory for widgets?",
        expected_clause_ids=[certification_clause.id],
    )

    service = BISRetrievalService(db_session)
    report = run_evaluation(db_session, service, [case], method="keyword")

    assert report.scored_case_count == 1
    assert report.no_result_case_count == 0
    assert report.recall_at_k[5] == 1.0
    assert report.recall_at_k[1] == 1.0


def test_run_evaluation_handles_duplicate_clause_numbers_via_clause_id(db_session):
    # Both seeded clauses share clause_number "1" but have distinct
    # clause_id — ground truth must disambiguate via clause_id, exactly
    # the real-world collision this milestone addresses.
    _, _, clauses = _seed(db_session)
    certification_clause, penalty_clause = clauses[0], clauses[1]
    assert certification_clause.clause_number == penalty_clause.clause_number == "1"
    assert certification_clause.id != penalty_clause.id

    case = EvalCase(
        id="t2",
        category="direct",
        question="What is the penalty for contravention of the widget QCO?",
        expected_clause_ids=[penalty_clause.id],
    )
    service = BISRetrievalService(db_session)
    report = run_evaluation(db_session, service, [case], method="keyword")

    # The correct clause_id (penalty_clause.id) must be recoverable even
    # though another real clause shares the same clause_number.
    retrieved = report.case_results[0].retrieved_clause_ids
    assert penalty_clause.id in retrieved[:5]


def test_run_evaluation_expect_no_result_case_tracks_abstention(db_session):
    _seed(db_session)
    case = EvalCase(
        id="t3",
        category="out_of_scope",
        question="What is the airspeed velocity of an unladen swallow?",
        expected_clause_ids=[],
        expect_no_result=True,
    )
    service = BISRetrievalService(db_session)
    report = run_evaluation(db_session, service, [case], method="keyword")

    assert report.scored_case_count == 0
    assert report.no_result_case_count == 1
    # No real terms match this seeded corpus, so the backend correctly
    # returns nothing and the case is scored as correctly abstained.
    assert report.abstention_rate == 1.0
    # expect_no_result cases must NOT be blended into recall/precision:
    # with zero scored cases, the aggregate is NaN (undefined), never a
    # fabricated 0.0 average over nothing.
    for v in report.recall_at_k.values():
        assert math.isnan(v)
    assert report.case_results[0].recall_at_k[5] is None
    assert report.case_results[0].precision_at_k[5] is None


def test_by_category_splits_reports_correctly(db_session):
    _, _, clauses = _seed(db_session)
    cases = [
        EvalCase(
            id="direct1",
            category="direct",
            question="Is certification mandatory for widgets?",
            expected_clause_ids=[clauses[0].id],
        ),
        EvalCase(
            id="oos1",
            category="out_of_scope",
            question="What is the airspeed velocity of an unladen swallow?",
            expected_clause_ids=[],
            expect_no_result=True,
        ),
    ]
    service = BISRetrievalService(db_session)
    report = run_evaluation(db_session, service, cases, method="keyword")
    by_cat = report.by_category()

    assert set(by_cat.keys()) == {"direct", "out_of_scope"}
    assert by_cat["direct"].scored_case_count == 1
    assert by_cat["direct"].no_result_case_count == 0
    assert by_cat["out_of_scope"].scored_case_count == 0
    assert by_cat["out_of_scope"].no_result_case_count == 1


def _vector_service(db_session, clauses):
    """Offline vector backend over DeterministicHashEmbeddingProvider."""
    provider = DeterministicHashEmbeddingProvider()
    ClauseEmbeddingService(db_session, provider).embed_clauses(clauses)
    db_session.commit()
    return BISRetrievalService(
        db_session, backends={"vector": VectorRetrievalBackend(db_session, provider)}
    )


def test_run_evaluation_vector_scores_a_direct_hit(db_session):
    _, _, clauses = _seed(db_session)
    certification_clause = clauses[0]
    case = EvalCase(
        id="v1",
        category="direct",
        question="Is certification mandatory for widgets?",
        expected_clause_ids=[certification_clause.id],
    )
    service = _vector_service(db_session, clauses)
    report = run_evaluation(db_session, service, [case], method="vector")

    assert report.scored_case_count == 1
    assert report.recall_at_k[5] == 1.0
    assert certification_clause.id in report.case_results[0].retrieved_clause_ids[:5]


def test_run_evaluation_vector_does_not_abstain_on_unrelated_query(db_session):
    # Vector nearest-neighbour search always returns *something* even for
    # an unrelated question — this is a known backend limitation, not a
    # bug in the evaluation harness. expect_no_result cases must still
    # be tracked as abstention failures rather than blended into Recall@K.
    _, _, clauses = _seed(db_session)
    case = EvalCase(
        id="v2",
        category="out_of_scope",
        question="What is the airspeed velocity of an unladen swallow?",
        expected_clause_ids=[],
        expect_no_result=True,
    )
    service = _vector_service(db_session, clauses)
    report = run_evaluation(db_session, service, [case], method="vector")

    assert report.scored_case_count == 0
    assert report.no_result_case_count == 1
    assert report.abstention_rate == 0.0
    assert report.case_results[0].retrieved_clause_ids
    for v in report.recall_at_k.values():
        assert math.isnan(v)
