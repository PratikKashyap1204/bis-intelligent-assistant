"""
Tests for Milestone 3: answer generation, RAG service, grounding rules,
citation integrity, context limits, and the /api/search/answer endpoint.

All tests use ExtractiveAnswerGenerationProvider and
DeterministicHashEmbeddingProvider — fully offline, no network, no real
model weights, no LLM credentials (see app/services/answer_generation.py
and app/services/embedding_provider.py for why).

Synthetic sample data only (IS 9999 / example.invalid), same convention
as test_embeddings.py / test_retrieval.py.
"""

from __future__ import annotations

from typing import List

from sqlalchemy import select

from app.models.clause import Clause
from app.services.answer_generation import (
    CORPUS_SCOPE_NOTE,
    ContextItem,
    ExtractiveAnswerGenerationProvider,
    GeneratedAnswer,
)
from app.services.bis_ingestion import (
    BISIngestionService,
    RawClause,
    RawDocument,
    RawStandard,
)
from app.services.embedding_provider import DeterministicHashEmbeddingProvider
from app.services.embedding_service import ClauseEmbeddingService
from app.services.rag import RAGConfig, RAGService
from app.services.retrieval import BISRetrievalService, VectorRetrievalBackend

# ---------------------------------------------------------------------------
# Sample pilot-like data: one Standard + one QCO-type document, several
# clauses, so tests can exercise the Standard-vs-QCO distinction and
# multi-clause context selection/dedup/capping.
# ---------------------------------------------------------------------------

SAMPLE_STANDARD = RawStandard(
    is_number="IS 9999",
    title="Household Electric Appliance Safety Requirements",
    year=2099,
    status="ACTIVE",
    scope="Safety requirements for household electric appliances.",
    source_url="https://example.invalid/sample-standard",
)

SAMPLE_QCO_DOCUMENT = RawDocument(
    title="Sample QCO Circular for Household Appliances",
    document_type="QCO",
    source_url="https://example.invalid/sample-qco.pdf",
    standard_is_number="IS 9999",
    content_hash="cafebabe" * 8,
    language="en",
)

SAMPLE_CLAUSES = [
    RawClause(
        clause_number="3",
        title="Certification",
        content=(
            "All manufacturers of electrical appliances covered by this QCO are "
            "required to obtain BIS certification before such appliances may be "
            "sold in the domestic market."
        ),
        page_number=2,
        language="en",
        clause_type="CLAUSE",
    ),
    RawClause(
        clause_number="4.1",
        title="Earthing",
        content="Earthing and grounding requirements for household electrical appliances.",
        page_number=3,
        language="en",
        clause_type="CLAUSE",
    ),
    RawClause(
        clause_number="5",
        title="Marking",
        content="Packaging labels shall be printed clearly in English.",
        page_number=4,
        language="en",
        clause_type="CLAUSE",
    ),
    RawClause(
        clause_number="Annex A",
        title="List of appliances",
        content="Toasters, kettles, and blenders are listed in this annex.",
        page_number=8,
        language="en",
        clause_type="ANNEX",
    ),
]


def _seed(db_session):
    ingestion = BISIngestionService(db_session)
    standard, _ = ingestion.ingest_standard(SAMPLE_STANDARD)
    document, _, _ = ingestion.ingest_document(
        SAMPLE_QCO_DOCUMENT, SAMPLE_CLAUSES, standard=standard
    )
    db_session.commit()
    clauses = list(
        db_session.execute(select(Clause).where(Clause.document_id == document.id)).scalars()
    )
    provider = DeterministicHashEmbeddingProvider()
    ClauseEmbeddingService(db_session, provider).embed_clauses(clauses)
    db_session.commit()
    return standard, document, clauses, provider


def _rag_service(db_session, provider, config=None) -> RAGService:
    backend = VectorRetrievalBackend(db_session, provider)
    retrieval_service = BISRetrievalService(db_session, backends={"vector": backend})
    return RAGService(
        db_session,
        retrieval_service,
        ExtractiveAnswerGenerationProvider(),
        config=config,
    )


# ---------------------------------------------------------------------------
# AnswerGenerationProvider / ExtractiveAnswerGenerationProvider (pure, no DB)
# ---------------------------------------------------------------------------

def test_extractive_provider_returns_insufficient_answer_for_empty_context():
    provider = ExtractiveAnswerGenerationProvider()
    result = provider.generate("Some question", [])

    assert result.grounded is False
    assert result.cited_indices == []
    assert "does not establish an answer" in result.answer_text
    assert CORPUS_SCOPE_NOTE in result.answer_text


def test_extractive_provider_cites_every_context_item_and_quotes_only_given_text():
    item = ContextItem(
        index=1,
        standard_number="IS 9999",
        standard_title="Sample Standard",
        document_title="Sample QCO",
        document_type="QCO",
        clause_number="3",
        clause_type="CLAUSE",
        clause_text="A very specific extractable fact about certification.",
        page_number=2,
        source_url="https://example.invalid/doc.pdf",
        relevance_score=0.55,
        retrieval_method="vector",
    )
    provider = ExtractiveAnswerGenerationProvider()
    result = provider.generate("Does certification apply?", [item])

    assert result.grounded is True
    assert result.cited_indices == [1]
    assert "[1]" in result.answer_text
    assert "A very specific extractable fact about certification." in result.answer_text
    # Never invents facts outside the given text: no fabricated dates/authorities.
    assert "penalty" not in result.answer_text.lower()
    assert CORPUS_SCOPE_NOTE in result.answer_text


def test_extractive_provider_distinguishes_standard_from_qco():
    standard_item = ContextItem(
        index=1,
        standard_number="IS 9999",
        standard_title="Sample Standard Title",
        document_title="Sample Standard Doc",
        document_type="STANDARD",
        clause_number="1",
        clause_type="CLAUSE",
        clause_text="Standard clause text.",
        page_number=1,
        source_url=None,
        relevance_score=0.5,
        retrieval_method="vector",
    )
    qco_item = ContextItem(
        index=2,
        standard_number="IS 9999",
        standard_title="Sample Standard Title",
        document_title="Sample QCO Circular",
        document_type="QCO",
        clause_number="3",
        clause_type="CLAUSE",
        clause_text="QCO clause text.",
        page_number=2,
        source_url=None,
        relevance_score=0.5,
        retrieval_method="vector",
    )
    provider = ExtractiveAnswerGenerationProvider()
    result = provider.generate("q", [standard_item, qco_item])

    assert "Indian Standard IS 9999" in result.answer_text
    assert "QCO circular 'Sample QCO Circular'" in result.answer_text


# ---------------------------------------------------------------------------
# RAGService: retrieval -> context assembly -> generation -> citations
# ---------------------------------------------------------------------------

def test_relevant_query_retrieves_context_and_returns_grounded_answer(db_session):
    standard, document, clauses, provider = _seed(db_session)
    rag = _rag_service(db_session, provider)

    result = rag.answer("What certification requirements exist for appliances?")

    assert result.grounded is True
    assert result.context_used > 0
    assert result.citations
    assert result.retrieval_method == "vector"


def test_rag_service_passes_retrieved_context_to_provider(db_session):
    standard, document, clauses, provider = _seed(db_session)

    captured: List[ContextItem] = []

    class RecordingProvider:
        def generate(self, question, context):
            captured.extend(context)
            return GeneratedAnswer(
                answer_text="stub", grounded=True, cited_indices=[c.index for c in context]
            )

    backend = VectorRetrievalBackend(db_session, provider)
    retrieval_service = BISRetrievalService(db_session, backends={"vector": backend})
    rag = RAGService(db_session, retrieval_service, RecordingProvider())

    rag.answer("Certification requirements for appliances")

    assert captured  # provider actually received non-empty context
    for item in captured:
        assert item.clause_text  # real clause text, not empty/placeholder


def test_answer_contains_citations_matching_database_records(db_session):
    standard, document, clauses, provider = _seed(db_session)
    rag = _rag_service(db_session, provider)

    result = rag.answer("What certification requirements exist for appliances?")

    assert result.citations
    clause_numbers_in_db = {c.clause_number for c in clauses}
    for citation in result.citations:
        # Every citation field traces back to the real seeded records —
        # nothing invented by the (deterministic) provider.
        assert citation.standard_number == standard.is_number
        assert citation.standard_title == standard.title
        assert citation.document_title == document.title
        assert citation.document_type == "QCO"
        assert citation.clause_number in clause_numbers_in_db
        assert citation.clause_id in {c.id for c in clauses}
        matching_clause = next(c for c in clauses if c.id == citation.clause_id)
        assert matching_clause.clause_number == citation.clause_number
        assert citation.source_url == document.source_url
        assert isinstance(citation.relevance_score, float)
        assert f"[{citation.index}]" in result.answer
        assert matching_clause.content[:50] in result.answer


def test_insufficient_context_produces_grounded_not_enough_information(db_session):
    standard, document, clauses, provider = _seed(db_session)
    rag = _rag_service(db_session, provider, config=RAGConfig(min_score_vector=0.999))

    # Threshold set unreachably high -> every retrieved hit is filtered
    # out -> RAGService must fall back to the fixed "insufficient" answer
    # rather than ever passing weak/irrelevant context to the provider.
    result = rag.answer("What certification requirements exist for appliances?")

    assert result.grounded is False
    assert result.context_used == 0
    assert result.citations == []
    assert "does not establish an answer" in result.answer


def test_provider_cannot_invent_citations_beyond_given_context(db_session):
    standard, document, clauses, provider = _seed(db_session)

    class MaliciousProvider:
        """Simulates a buggy/adversarial provider that cites indices that
        were never actually given to it, or don't exist at all."""

        def generate(self, question, context):
            real_indices = [c.index for c in context]
            fabricated = [max(real_indices, default=0) + 999, -1, 0]
            return GeneratedAnswer(
                answer_text="I cite [1], [2], and some fabricated sources.",
                grounded=True,
                cited_indices=real_indices + fabricated,
            )

    backend = VectorRetrievalBackend(db_session, provider)
    retrieval_service = BISRetrievalService(db_session, backends={"vector": backend})
    rag = RAGService(db_session, retrieval_service, MaliciousProvider())

    result = rag.answer("What certification requirements exist for appliances?")

    # Only citations corresponding to REAL context items survive; the
    # fabricated indices are silently dropped, never surfaced as citations.
    assert result.citations
    real_context_count = result.context_used
    assert len(result.citations) == real_context_count
    for citation in result.citations:
        assert citation.clause_number is not None
        assert citation.standard_number == standard.is_number


def test_context_size_is_bounded_by_max_context_items(db_session):
    standard, document, clauses, provider = _seed(db_session)
    config = RAGConfig(max_context_items=2, min_score_vector=-1.0)  # accept all matches
    rag = _rag_service(db_session, provider, config=config)

    result = rag.answer("appliances requirements certification earthing marking")

    assert result.context_used <= 2
    assert len(result.sources) <= 2


def test_context_size_is_bounded_by_total_character_cap(db_session):
    standard, document, clauses, provider = _seed(db_session)
    # Every clause here is well under 800 chars; force a tiny total cap so
    # only the single highest-ranked item can fit.
    config = RAGConfig(
        max_context_items=10,
        max_chars_per_clause=800,
        max_total_context_chars=1,
        min_score_vector=-1.0,
    )
    rag = _rag_service(db_session, provider, config=config)

    result = rag.answer("appliances requirements certification earthing marking")

    assert result.context_used == 1


def test_duplicate_retrieval_results_are_removed(db_session):
    standard, document, clauses, provider = _seed(db_session)
    backend = VectorRetrievalBackend(db_session, provider)
    retrieval_service = BISRetrievalService(db_session, backends={"vector": backend})

    duplicated_results = retrieval_service.search(
        "certification", method="vector", limit=10
    )
    duplicated_results = duplicated_results + duplicated_results  # simulate duplicates

    from app.services.rag import _select_context

    selected = _select_context(duplicated_results, RAGConfig(min_score_vector=-1.0))
    clause_ids = [r.clause_id for r in selected]
    assert len(clause_ids) == len(set(clause_ids))  # no duplicate clause_id in selection


def test_provenance_preserved_through_full_rag_flow(db_session):
    standard, document, clauses, provider = _seed(db_session)
    rag = _rag_service(db_session, provider)

    result = rag.answer("What certification requirements exist for appliances?")

    assert result.sources
    for source in result.sources:
        assert source.standard_id == standard.id
        assert source.standard_number == standard.is_number
        assert source.document_id == document.id
        assert source.document_title == document.title
        assert source.clause_id is not None
        assert source.source_url == document.source_url


def test_mock_provider_allows_fully_offline_rag(db_session):
    """
    The whole RAG flow (retrieval + context assembly + generation +
    citations) must work with zero network access: DeterministicHash
    embeddings + ExtractiveAnswerGenerationProvider only.
    """
    standard, document, clauses, provider = _seed(db_session)
    assert provider.model_name == "test-hash-embedding-v1"  # confirms no real model used

    rag = _rag_service(db_session, provider)
    result = rag.answer("certification requirements for appliances")

    assert result.answer
    assert result.grounded is True


def test_keyword_method_still_works_via_rag_service(db_session):
    """Milestone 1's keyword backend remains usable through RAGService too."""
    standard, document, clauses, provider = _seed(db_session)
    backend = VectorRetrievalBackend(db_session, provider)
    retrieval_service = BISRetrievalService(db_session, backends={"vector": backend})
    rag = RAGService(
        db_session, retrieval_service, ExtractiveAnswerGenerationProvider(),
        config=RAGConfig(min_score_keyword=0.0),
    )

    result = rag.answer("Earthing", method="keyword")

    assert result.retrieval_method == "keyword"
    assert result.grounded is True
    assert any(c.clause_number == "4.1" for c in result.citations)


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

def test_answer_endpoint_returns_grounded_response(db_session, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import search as search_api
    from app.db import session as db_session_module
    from app.main import app

    _seed_result = _seed(db_session)
    _, _, _, provider = _seed_result

    # Force the API to use the deterministic test provider (no real model
    # download/load) and the same db_session used to seed the data.
    monkeypatch.setattr(search_api, "_vector_provider", provider)
    monkeypatch.setattr(search_api, "_get_vector_provider", lambda: provider)

    def override_get_db():
        yield db_session

    app.dependency_overrides[db_session_module.get_db] = override_get_db
    try:
        client = TestClient(app)
        response = client.post(
            "/api/search/answer",
            json={"query": "What certification requirements exist for appliances?"},
        )
    finally:
        app.dependency_overrides.pop(db_session_module.get_db, None)

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["citations"]
    assert body["retrieval_method"] == "vector"
    assert body["context_used"] > 0
    assert body["sources"]
    assert "[1]" in body["answer"]
    for citation in body["citations"]:
        assert citation.get("clause_id") is not None
        assert citation["clause_id"] in {c.id for c in _seed_result[2]}
    for source in body["sources"]:
        if source.get("clause_text"):
            assert source.get("clause_id") is not None


def test_answer_endpoint_rejects_unknown_method(db_session, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import search as search_api
    from app.db import session as db_session_module
    from app.main import app

    _, _, _, provider = _seed(db_session)
    monkeypatch.setattr(search_api, "_vector_provider", provider)
    monkeypatch.setattr(search_api, "_get_vector_provider", lambda: provider)

    def override_get_db():
        yield db_session

    app.dependency_overrides[db_session_module.get_db] = override_get_db
    try:
        client = TestClient(app)
        response = client.post(
            "/api/search/answer",
            json={"query": "What certification requirements exist?", "method": "semantic"},
        )
    finally:
        app.dependency_overrides.pop(db_session_module.get_db, None)

    assert response.status_code == 400
    assert "Unknown retrieval method" in response.json()["detail"]


def test_answer_endpoint_rejects_blank_and_oversized_query(db_session, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import search as search_api
    from app.db import session as db_session_module
    from app.main import app
    from app.schemas.rag import MAX_QUERY_LENGTH

    _, _, _, provider = _seed(db_session)
    monkeypatch.setattr(search_api, "_vector_provider", provider)
    monkeypatch.setattr(search_api, "_get_vector_provider", lambda: provider)

    def override_get_db():
        yield db_session

    app.dependency_overrides[db_session_module.get_db] = override_get_db
    try:
        client = TestClient(app)
        blank = client.post("/api/search/answer", json={"query": "   "})
        huge = client.post("/api/search/answer", json={"query": "x" * (MAX_QUERY_LENGTH + 1)})
    finally:
        app.dependency_overrides.pop(db_session_module.get_db, None)

    assert blank.status_code == 422
    assert huge.status_code == 422


def test_answer_endpoint_insufficient_context_for_unrelated_question(db_session, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import search as search_api
    from app.db import session as db_session_module
    from app.main import app

    _, _, _, provider = _seed(db_session)

    monkeypatch.setattr(search_api, "_vector_provider", provider)
    monkeypatch.setattr(search_api, "_get_vector_provider", lambda: provider)

    def override_get_db():
        yield db_session

    app.dependency_overrides[db_session_module.get_db] = override_get_db
    try:
        client = TestClient(app)
        # Deterministic hash provider gives near-zero similarity to text
        # sharing no words with any seeded clause.
        response = client.post(
            "/api/search/answer",
            json={"query": "zzz nonexistent unrelated qwerty banana spaceship topic"},
        )
    finally:
        app.dependency_overrides.pop(db_session_module.get_db, None)

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["citations"] == []
    assert "does not establish an answer" in body["answer"]

