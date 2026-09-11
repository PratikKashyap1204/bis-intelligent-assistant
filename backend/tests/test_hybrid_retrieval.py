"""
Milestone 6 tests: hybrid fusion, vector similarity abstention, and
identity/provenance regressions.

Synthetic sample data only (IS 8888 / example.invalid). Offline and
deterministic — uses DeterministicHashEmbeddingProvider, never the real
sentence-transformers model, and never an LLM/API.
"""

from __future__ import annotations

from app.models.clause import Clause
from app.services.bis_ingestion import (
    BISIngestionService,
    RawClause,
    RawDocument,
    RawStandard,
)
from app.services.embedding_provider import DeterministicHashEmbeddingProvider
from app.services.embedding_service import ClauseEmbeddingService
from app.services.evaluation import EvalCase, run_evaluation
from app.services.answer_generation import ExtractiveAnswerGenerationProvider
from app.services.rag import RAGConfig, RAGService
from app.services.retrieval import (
    BISRetrievalService,
    EVALUATED_VECTOR_MIN_COSINE_SIMILARITY,
    HybridRetrievalBackend,
    KeywordRetrievalBackend,
    RelevanceInfo,
    RetrievalResult,
    SearchFilters,
    VectorRetrievalBackend,
    build_retrieval_service,
    minmax_normalize,
    reciprocal_rank_fusion,
    retrieval_identity_key,
    weighted_minmax_fusion,
)
from sqlalchemy import select


SAMPLE_STANDARD = RawStandard(
    is_number="IS 8888",
    title="Sample Hybrid Evaluation Standard",
    year=2050,
    status="ACTIVE",
    scope="Sample scope for hybrid retrieval tests.",
    source_url="https://example.invalid/hybrid-standard",
)

SAMPLE_DOCUMENT = RawDocument(
    title="Sample Hybrid QCO Circular",
    document_type="QCO",
    source_url="https://example.invalid/hybrid-document.pdf",
    standard_is_number="IS 8888",
    content_hash="feedface" * 8,
    language="en",
)

SAMPLE_CLAUSES = [
    RawClause(
        clause_number="1",
        content="Certification is mandatory for widgets under this QCO.",
        page_number=1,
        language="en",
        clause_type="CLAUSE",
        sequence_in_document=1,
    ),
    RawClause(
        clause_number="1",
        content="Penalty for contravention of this widget QCO is a fine.",
        page_number=2,
        language="en",
        clause_type="CLAUSE",
        sequence_in_document=2,
    ),
    RawClause(
        clause_number="3",
        content="The Bureau is the certifying and enforcing authority for widgets.",
        page_number=3,
        language="en",
        clause_type="CLAUSE",
        sequence_in_document=3,
    ),
]


def _seed(db_session):
    ingestion = BISIngestionService(db_session)
    standard, _ = ingestion.ingest_standard(SAMPLE_STANDARD)
    document, _, _ = ingestion.ingest_document(
        SAMPLE_DOCUMENT, SAMPLE_CLAUSES, standard=standard
    )
    db_session.commit()
    clauses = list(
        db_session.execute(select(Clause).where(Clause.document_id == document.id)).scalars()
    )
    return standard, document, clauses


def _hit(clause_id, score, method, clause_number="1", document_id=1, standard_id=1):
    return RetrievalResult(
        standard_id=standard_id,
        document_id=document_id,
        clause_id=clause_id,
        clause_number=clause_number,
        relevance=RelevanceInfo(score=score, method=method),
    )


# ---------------------------------------------------------------------------
# Pure fusion helpers (no database)
# ---------------------------------------------------------------------------


def test_retrieval_identity_key_uses_clause_id_not_clause_number():
    a = _hit(clause_id=10, score=1.0, method="keyword", clause_number="1")
    b = _hit(clause_id=20, score=1.0, method="keyword", clause_number="1")
    assert retrieval_identity_key(a) != retrieval_identity_key(b)
    assert retrieval_identity_key(a) == ("clause", 10)


def test_minmax_normalize_basic():
    assert minmax_normalize([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]
    assert minmax_normalize([]) == []
    assert minmax_normalize([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]
    assert minmax_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_rrf_prefers_items_ranked_high_in_both_lists():
    kw = [_hit(1, 9.0, "keyword"), _hit(2, 4.0, "keyword"), _hit(3, 1.0, "keyword")]
    vec = [_hit(2, 0.9, "vector"), _hit(1, 0.4, "vector"), _hit(4, 0.2, "vector")]
    fused = reciprocal_rank_fusion([kw, vec], rrf_k=60)
    ids = [h.clause_id for h in fused]
    # clause 1 and 2 appear in both lists; 2 is rank-1 vector + rank-2 keyword.
    assert set(ids[:2]) == {1, 2}
    assert fused[0].relevance.method == "hybrid"
    assert "keyword" in fused[0].relevance.matched_fields or "vector" in fused[0].relevance.matched_fields


def test_rrf_does_not_collapse_duplicate_clause_numbers():
    kw = [_hit(10, 5.0, "keyword", clause_number="1"), _hit(20, 4.0, "keyword", clause_number="1")]
    vec = [_hit(20, 0.8, "vector", clause_number="1")]
    fused = reciprocal_rank_fusion([kw, vec], rrf_k=10)
    ids = [h.clause_id for h in fused]
    assert 10 in ids and 20 in ids
    assert len(ids) == 2


def test_weighted_minmax_fusion_respects_weights():
    kw = [_hit(1, 10.0, "keyword"), _hit(2, 0.0, "keyword")]
    vec = [_hit(2, 1.0, "vector"), _hit(1, 0.0, "vector")]
    fused = weighted_minmax_fusion([(kw, 1.0), (vec, 0.0)])
    assert fused[0].clause_id == 1


def test_rrf_rejects_invalid_k():
    import pytest

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[_hit(1, 1.0, "keyword")]], rrf_k=0)


# ---------------------------------------------------------------------------
# Vector abstention
# ---------------------------------------------------------------------------


def test_vector_empty_query_returns_no_results(db_session):
    _, _, clauses = _seed(db_session)
    provider = DeterministicHashEmbeddingProvider()
    ClauseEmbeddingService(db_session, provider).embed_clauses(clauses)
    backend = VectorRetrievalBackend(db_session, provider)
    assert backend.search("", SearchFilters(), 10) == []
    assert backend.search("   ", SearchFilters(), 10) == []


def test_vector_min_similarity_can_abstain(db_session):
    _, _, clauses = _seed(db_session)
    provider = DeterministicHashEmbeddingProvider()
    ClauseEmbeddingService(db_session, provider).embed_clauses(clauses)
    # Cosine similarity is at most 1.0; a floor above 1.0 must drop every neighbour.
    backend = VectorRetrievalBackend(db_session, provider, min_similarity=1.01)
    results = backend.search("Certification is mandatory for widgets", SearchFilters(), 10)
    assert results == []


def test_vector_without_min_similarity_still_returns_neighbours(db_session):
    _, _, clauses = _seed(db_session)
    provider = DeterministicHashEmbeddingProvider()
    ClauseEmbeddingService(db_session, provider).embed_clauses(clauses)
    backend = VectorRetrievalBackend(db_session, provider, min_similarity=None)
    results = backend.search("zzzz totally unrelated query xyz", SearchFilters(), 10)
    assert results  # Milestone 2/5 default: nearest neighbours, no abstention


def test_vector_abstention_does_not_fabricate_clause_ids(db_session):
    _, _, clauses = _seed(db_session)
    real_ids = {c.id for c in clauses}
    provider = DeterministicHashEmbeddingProvider()
    ClauseEmbeddingService(db_session, provider).embed_clauses(clauses)
    backend = VectorRetrievalBackend(db_session, provider, min_similarity=0.0)
    results = backend.search("widgets", SearchFilters(), 10)
    assert all(r.clause_id in real_ids for r in results)


# ---------------------------------------------------------------------------
# Hybrid backend
# ---------------------------------------------------------------------------


def _hybrid_service(db_session, clauses, **hybrid_kwargs):
    provider = DeterministicHashEmbeddingProvider()
    ClauseEmbeddingService(db_session, provider).embed_clauses(clauses)
    db_session.commit()
    keyword = KeywordRetrievalBackend(db_session)
    vector = VectorRetrievalBackend(db_session, provider)
    hybrid = HybridRetrievalBackend(keyword, vector, **hybrid_kwargs)
    return BISRetrievalService(
        db_session,
        backend=keyword,
        backends={"keyword": keyword, "vector": vector, "hybrid": hybrid},
    ), keyword, vector


def test_hybrid_empty_query_returns_no_results(db_session):
    _, _, clauses = _seed(db_session)
    service, _, _ = _hybrid_service(db_session, clauses)
    assert service.search("", method="hybrid") == []


def test_hybrid_preserves_provenance_and_clause_id(db_session):
    standard, document, clauses = _seed(db_session)
    service, _, _ = _hybrid_service(db_session, clauses)
    results = service.search("mandatory certification widgets", method="hybrid")
    assert results
    for hit in results:
        if hit.clause_id is None:
            continue
        assert hit.standard_id == standard.id
        assert hit.document_id == document.id
        assert hit.clause_id in {c.id for c in clauses}
        assert hit.source_url == document.source_url
        assert hit.relevance.method == "hybrid"


def test_hybrid_does_not_break_keyword_or_vector_methods(db_session):
    _, _, clauses = _seed(db_session)
    service, _, _ = _hybrid_service(db_session, clauses)
    kw = service.search("mandatory certification widgets", method="keyword")
    vec = service.search("mandatory certification widgets", method="vector")
    assert kw and kw[0].relevance.method == "keyword"
    assert vec and vec[0].relevance.method == "vector"


def test_hybrid_default_search_without_method_stays_keyword(db_session):
    _, _, clauses = _seed(db_session)
    service, _, _ = _hybrid_service(db_session, clauses)
    results = service.search("mandatory certification widgets")
    assert results
    assert results[0].relevance.method == "keyword"


def test_hybrid_retrieves_multiple_distinct_clause_ids(db_session):
    _, _, clauses = _seed(db_session)
    cert, penalty, bureau = clauses
    assert cert.clause_number == penalty.clause_number == "1"
    assert cert.id != penalty.id
    service, _, _ = _hybrid_service(db_session, clauses)
    case = EvalCase(
        id="mc",
        category="multi_clause",
        question="What certification requirement and penalty apply to widgets?",
        expected_clause_ids=[cert.id, penalty.id],
    )
    report = run_evaluation(db_session, service, [case], method="hybrid")
    retrieved = report.case_results[0].retrieved_clause_ids
    assert cert.id in retrieved
    assert penalty.id in retrieved
    assert report.recall_at_k[5] == 1.0


def test_hybrid_weighted_fusion_mode_runs(db_session):
    _, _, clauses = _seed(db_session)
    service, _, _ = _hybrid_service(
        db_session, clauses, fusion="weighted", keyword_weight=0.7, vector_weight=0.3
    )
    results = service.search("enforcing authority widgets", method="hybrid")
    assert results
    assert results[0].relevance.method == "hybrid"


def test_hybrid_does_not_invent_results_for_empty_backends():
    fused = reciprocal_rank_fusion([[], []])
    assert fused == []


class _ListBackend:
    """Minimal RetrievalBackend stand-in for fusion-unit tests (no database)."""

    def __init__(self, hits, method="keyword"):
        self._hits = hits
        self.method = method

    def search(self, query, filters, limit):
        return list(self._hits)[:limit]


def test_hybrid_fuses_clauses_then_appends_document_only_keyword_hits():
    doc_hit = RetrievalResult(
        document_id=9,
        document_title="Some QCO",
        relevance=RelevanceInfo(score=100.0, method="keyword"),
    )
    clause_kw = _hit(1, 4.0, "keyword")
    clause_vec = _hit(2, 0.9, "vector")
    hybrid = HybridRetrievalBackend(
        _ListBackend([doc_hit, clause_kw], "keyword"),
        _ListBackend([clause_vec], "vector"),
        rrf_k=60,
    )
    results = hybrid.search("query", SearchFilters(), 10)
    clause_ids = [r.clause_id for r in results if r.clause_id is not None]
    assert set(clause_ids) == {1, 2}
    assert any(r.document_id == 9 and r.clause_id is None for r in results)
    last_clause_idx = max(i for i, r in enumerate(results) if r.clause_id is not None)
    first_doc_idx = next(i for i, r in enumerate(results) if r.clause_id is None)
    assert last_clause_idx < first_doc_idx


def test_evaluated_vector_floor_is_cosine_similarity_not_distance():
    # Documented production floor: cosine similarity (1 - pgvector distance).
    assert EVALUATED_VECTOR_MIN_COSINE_SIMILARITY == 0.30
    assert 0.0 < EVALUATED_VECTOR_MIN_COSINE_SIMILARITY < 1.0


def test_hybrid_respects_is_number_filter(db_session):
    _, _, clauses = _seed(db_session)
    service, _, _ = _hybrid_service(db_session, clauses)
    hits = service.search("widgets", method="hybrid", is_number="IS 8888")
    assert hits
    assert all(h.standard_number == "IS 8888" for h in hits if h.standard_number)
    assert service.search("widgets", method="hybrid", is_number="IS 0000") == []


def test_hybrid_does_not_fabricate_clause_ids(db_session):
    _, _, clauses = _seed(db_session)
    real_ids = {c.id for c in clauses}
    service, _, _ = _hybrid_service(db_session, clauses)
    results = service.search("widgets certification penalty bureau", method="hybrid")
    assert results
    assert all(r.clause_id is None or r.clause_id in real_ids for r in results)


def test_build_retrieval_service_default_method_is_keyword(db_session):
    _, _, clauses = _seed(db_session)
    provider = DeterministicHashEmbeddingProvider()
    ClauseEmbeddingService(db_session, provider).embed_clauses(clauses)
    db_session.commit()
    service = build_retrieval_service(
        db_session, provider, vector_min_similarity=None
    )
    assert set(service.backends) >= {"keyword", "vector", "hybrid"}
    default_hits = service.search("mandatory certification widgets")
    assert default_hits
    assert default_hits[0].relevance.method == "keyword"
    hybrid_hits = service.search("mandatory certification widgets", method="hybrid")
    assert hybrid_hits
    assert hybrid_hits[0].relevance.method == "hybrid"


def test_rag_hybrid_method_stays_grounded_in_retrieved_clauses(db_session):
    _, _, clauses = _seed(db_session)
    service, _, _ = _hybrid_service(db_session, clauses)
    rag = RAGService(
        db_session,
        service,
        ExtractiveAnswerGenerationProvider(),
        config=RAGConfig(min_score_keyword=0.0, min_score_vector=-1.0),
    )
    result = rag.answer("mandatory certification widgets", method="hybrid")
    assert result.retrieval_method == "hybrid"
    retrieved_ids = {c.id for c in clauses}
    for source in result.sources:
        if source.clause_id is not None:
            assert source.clause_id in retrieved_ids
    for citation in result.citations:
        assert citation.standard_number in {None, "IS 8888"}
