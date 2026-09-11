"""
Milestone 4 integration tests: RAGService + LLMAnswerGenerationProvider,
and the /api/search/answer endpoint wired to the LLM provider.

Uses the same FakeOpenAIClient pattern as test_llm_answer_generation.py —
fully offline, no real network/LLM call. Retrieval still uses
DeterministicHashEmbeddingProvider (Milestone 2's offline test provider).
"""

from __future__ import annotations

import json

from sqlalchemy import select

from app.models.clause import Clause
from app.services.bis_ingestion import BISIngestionService, RawClause, RawDocument, RawStandard
from app.services.embedding_provider import DeterministicHashEmbeddingProvider
from app.services.embedding_service import ClauseEmbeddingService
from app.services.answer_generation import LLMAnswerGenerationProvider
from app.services.rag import RAGService
from app.services.retrieval import BISRetrievalService, VectorRetrievalBackend
from tests.test_llm_answer_generation import FakeOpenAIClient

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


def test_rag_service_still_works_with_llm_provider(db_session):
    """13. existing 95 (now higher) tests continue passing; RAGService still
    works when the answer_provider is LLM-backed (fake client)."""
    standard, document, clauses, embed_provider = _seed(db_session)

    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {"answer": "Certification is required before sale. [SOURCE_1]", "citation_ids": ["SOURCE_1"]}
        )
    )
    llm_provider = LLMAnswerGenerationProvider(api_key="fake-key", client=fake_client)

    backend = VectorRetrievalBackend(db_session, embed_provider)
    retrieval_service = BISRetrievalService(db_session, backends={"vector": backend})
    rag = RAGService(db_session, retrieval_service, llm_provider)

    result = rag.answer("What certification requirements exist for appliances?")

    assert result.grounded is True
    assert result.citations
    citation = result.citations[0]
    # Citation metadata comes from RAGService/real DB records, not the LLM.
    assert citation.standard_number == standard.is_number
    assert citation.document_title == document.title
    assert citation.clause_number in {c.clause_number for c in clauses}
    assert citation.source_url == document.source_url
    # RAGService does not rewrite the LLM's answer text (it only validates
    # citation IDs into structured Citation objects) — the literal
    # "[SOURCE_1]" marker from the fake LLM response is expected to still
    # appear in the raw answer text as-is.
    assert "[SOURCE_1]" in result.answer


def test_rag_service_with_llm_provider_handles_insufficient_context(db_session):
    standard, document, clauses, embed_provider = _seed(db_session)

    fake_client = FakeOpenAIClient(response_content=json.dumps({"answer": "unused", "citation_ids": []}))
    llm_provider = LLMAnswerGenerationProvider(api_key="fake-key", client=fake_client)

    backend = VectorRetrievalBackend(db_session, embed_provider)
    retrieval_service = BISRetrievalService(db_session, backends={"vector": backend})
    from app.services.rag import RAGConfig

    rag = RAGService(
        db_session, retrieval_service, llm_provider, config=RAGConfig(min_score_vector=0.999)
    )

    result = rag.answer("What certification requirements exist for appliances?")

    assert result.grounded is False
    assert result.citations == []
    assert fake_client.calls == []  # RAGService found no context -> no LLM call at all


def test_api_endpoint_works_with_llm_provider_configured(db_session, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import search as search_api
    from app.db import session as db_session_module
    from app.main import app

    _, _, _, embed_provider = _seed(db_session)

    fake_client = FakeOpenAIClient(
        response_content=json.dumps(
            {"answer": "Certification is required. [SOURCE_1]", "citation_ids": ["SOURCE_1"]}
        )
    )
    llm_provider = LLMAnswerGenerationProvider(api_key="fake-key", client=fake_client)

    monkeypatch.setattr(search_api, "_vector_provider", embed_provider)
    monkeypatch.setattr(search_api, "_get_vector_provider", lambda: embed_provider)
    monkeypatch.setattr(search_api, "_answer_provider", llm_provider)
    monkeypatch.setattr(search_api, "_get_answer_provider", lambda: llm_provider)

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
    assert len(fake_client.calls) == 1


def test_extractive_provider_and_api_endpoint_still_work_unaffected(db_session, monkeypatch):
    """Milestone 3 behavior (extractive provider, default config) must be
    completely unaffected by adding the LLM provider option."""
    from fastapi.testclient import TestClient

    from app.api import search as search_api
    from app.db import session as db_session_module
    from app.main import app

    _, _, _, embed_provider = _seed(db_session)

    monkeypatch.setattr(search_api, "_vector_provider", embed_provider)
    monkeypatch.setattr(search_api, "_get_vector_provider", lambda: embed_provider)
    # Explicitly reset the cached answer provider to the default (extractive).
    monkeypatch.setattr(search_api, "_answer_provider", None)

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
