"""
Tests for Milestone 2: embedding provider, storage, generation service,
and vector retrieval.

All tests use DeterministicHashEmbeddingProvider — a fast, offline,
dependency-free stand-in (see app/services/embedding_provider.py). No
test loads the real sentence-transformers model or touches the network;
that only happens in scripts/embed_pilot.py, run manually.

Synthetic sample data only (IS 9999 / example.invalid), same convention
as the other integration test files.
"""

from __future__ import annotations

from sqlalchemy import select, text

from app.models.clause import Clause
from app.models.embedding import ClauseEmbedding
from app.services.bis_ingestion import (
    BISIngestionService,
    RawClause,
    RawDocument,
    RawStandard,
)
from app.services.embedding_provider import (
    DEFAULT_EMBEDDING_DIMENSION,
    DeterministicHashEmbeddingProvider,
)
from app.services.embedding_service import ClauseEmbeddingService, content_hash_of
from app.services.retrieval import BISRetrievalService, SearchFilters, VectorRetrievalBackend

SAMPLE_STANDARD = RawStandard(
    is_number="IS 9999",
    title="Household Electric Appliance Safety Requirements",
    year=2099,
    status="ACTIVE",
    scope="Safety requirements for household electric appliances.",
    source_url="https://example.invalid/sample-standard",
)

SAMPLE_DOCUMENT = RawDocument(
    title="Sample Household Appliance Circular",
    document_type="OTHER",
    source_url="https://example.invalid/sample-document.pdf",
    standard_is_number="IS 9999",
    content_hash="cafebabe" * 8,
    language="en",
)

SAMPLE_CLAUSES = [
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
        SAMPLE_DOCUMENT, SAMPLE_CLAUSES, standard=standard
    )
    db_session.commit()
    clauses = list(
        db_session.execute(select(Clause).where(Clause.document_id == document.id)).scalars()
    )
    return standard, document, clauses


# ---------------------------------------------------------------------------
# pgvector availability
# ---------------------------------------------------------------------------

def test_pgvector_extension_is_installed(db_session):
    row = db_session.execute(
        text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
    ).first()
    assert row is not None, "vector extension must be enabled for these tests to be meaningful"


# ---------------------------------------------------------------------------
# Embedding provider interface (no database needed)
# ---------------------------------------------------------------------------

def test_deterministic_provider_generate_matches_batch():
    provider = DeterministicHashEmbeddingProvider()
    single = provider.generate("Earthing requirements")
    batch = provider.generate_batch(["Earthing requirements"])
    assert single == batch[0]
    assert len(single) == DEFAULT_EMBEDDING_DIMENSION


def test_deterministic_provider_is_deterministic():
    provider = DeterministicHashEmbeddingProvider()
    a = provider.generate("Earthing requirements for appliances.")
    b = provider.generate("Earthing requirements for appliances.")
    assert a == b


def test_deterministic_provider_similar_text_scores_higher_than_unrelated():
    provider = DeterministicHashEmbeddingProvider()

    def cos(x, y):
        return sum(xi * yi for xi, yi in zip(x, y))

    base = provider.generate("Earthing requirements for household electric appliances.")
    similar = provider.generate("Household electric appliance earthing requirement text.")
    unrelated = provider.generate("Completely unrelated text about tropical fruit farming.")

    assert cos(base, similar) > cos(base, unrelated)


# ---------------------------------------------------------------------------
# Embedding persistence + dedup + re-embedding
# ---------------------------------------------------------------------------

def test_embed_clauses_creates_one_row_per_clause(db_session):
    _, _, clauses = _seed(db_session)
    provider = DeterministicHashEmbeddingProvider()
    service = ClauseEmbeddingService(db_session, provider)

    summary = service.embed_clauses(clauses)
    db_session.commit()

    assert summary.created == len(clauses)
    assert summary.updated == 0
    assert summary.skipped_unchanged == 0

    rows = db_session.execute(select(ClauseEmbedding)).scalars().all()
    assert len(rows) == len(clauses)
    for row in rows:
        assert row.model_name == provider.model_name
        assert row.model_dimension == DEFAULT_EMBEDDING_DIMENSION
        assert len(row.embedding) == DEFAULT_EMBEDDING_DIMENSION


def test_embed_clauses_skips_unchanged_content_on_rerun(db_session):
    _, _, clauses = _seed(db_session)
    provider = DeterministicHashEmbeddingProvider()
    service = ClauseEmbeddingService(db_session, provider)

    first = service.embed_clauses(clauses)
    db_session.commit()
    assert first.created == len(clauses)

    second = service.embed_clauses(clauses)
    db_session.commit()

    assert second.created == 0
    assert second.updated == 0
    assert second.skipped_unchanged == len(clauses)

    rows = db_session.execute(select(ClauseEmbedding)).scalars().all()
    assert len(rows) == len(clauses)  # no duplicates created


def test_embed_clauses_reembeds_when_content_changes(db_session):
    _, _, clauses = _seed(db_session)
    provider = DeterministicHashEmbeddingProvider()
    service = ClauseEmbeddingService(db_session, provider)

    service.embed_clauses(clauses)
    db_session.commit()

    target_clause = clauses[0]
    original_hash = content_hash_of(target_clause.content)

    target_clause.content = "Completely different content after a standard revision."
    db_session.commit()

    summary = service.embed_clauses(clauses)
    db_session.commit()

    assert summary.updated == 1
    assert summary.created == 0
    assert summary.skipped_unchanged == len(clauses) - 1

    updated_row = db_session.execute(
        select(ClauseEmbedding).where(ClauseEmbedding.clause_id == target_clause.id)
    ).scalar_one()
    assert updated_row.content_hash != original_hash
    assert updated_row.embedding_text == target_clause.content

    # still exactly one embedding row per clause — no duplicate inserted
    rows = db_session.execute(
        select(ClauseEmbedding).where(ClauseEmbedding.clause_id == target_clause.id)
    ).scalars().all()
    assert len(rows) == 1


def test_embed_clauses_skips_empty_content(db_session):
    _, document, clauses = _seed(db_session)
    empty_clause = Clause(document_id=document.id, clause_number="6", content="   ")
    db_session.add(empty_clause)
    db_session.commit()

    provider = DeterministicHashEmbeddingProvider()
    service = ClauseEmbeddingService(db_session, provider)
    summary = service.embed_clauses(clauses + [empty_clause])
    db_session.commit()

    assert summary.skipped_empty == 1
    assert summary.created == len(clauses)


def test_embed_all_clauses_filters_by_standard_number(db_session):
    _, _, clauses = _seed(db_session)
    provider = DeterministicHashEmbeddingProvider()
    service = ClauseEmbeddingService(db_session, provider)

    summary = service.embed_all_clauses(standard_is_number="IS 9999")
    db_session.commit()

    assert summary.created == len(clauses)

    none_summary = service.embed_all_clauses(standard_is_number="IS 0000")
    assert none_summary.total_considered == 0


# ---------------------------------------------------------------------------
# Vector retrieval
# ---------------------------------------------------------------------------

def test_vector_retrieval_returns_relevant_records(db_session):
    standard, document, clauses = _seed(db_session)
    provider = DeterministicHashEmbeddingProvider()
    ClauseEmbeddingService(db_session, provider).embed_clauses(clauses)
    db_session.commit()

    backend = VectorRetrievalBackend(db_session, provider)
    service = BISRetrievalService(db_session, backends={"vector": backend})

    # Different wording than any stored clause, but shares real words with
    # the earthing clause ("household", "electric", "requirements").
    results = service.search(
        "What household electric requirements exist?", method="vector"
    )

    assert results
    assert results[0].relevance.method == "vector"
    assert results[0].clause_number == "4.1"  # the earthing clause should rank first
    assert results[0].relevance.score > 0


def test_vector_retrieval_preserves_provenance(db_session):
    standard, document, clauses = _seed(db_session)
    provider = DeterministicHashEmbeddingProvider()
    ClauseEmbeddingService(db_session, provider).embed_clauses(clauses)
    db_session.commit()

    backend = VectorRetrievalBackend(db_session, provider)
    results = backend.search("Earthing requirements for appliances", SearchFilters(), 10)

    assert results
    for hit in results:
        assert hit.standard_id == standard.id
        assert hit.standard_number == standard.is_number
        assert hit.standard_title == standard.title
        assert hit.document_id == document.id
        assert hit.document_title == document.title
        assert hit.clause_id is not None
        assert hit.source_url == document.source_url


def test_vector_retrieval_respects_clause_type_filter(db_session):
    standard, document, clauses = _seed(db_session)
    provider = DeterministicHashEmbeddingProvider()
    ClauseEmbeddingService(db_session, provider).embed_clauses(clauses)
    db_session.commit()

    backend = VectorRetrievalBackend(db_session, provider)
    service = BISRetrievalService(db_session, backends={"vector": backend})

    results = service.search("appliances", method="vector", clause_type="ANNEX")
    assert results
    assert all(r.clause_type == "ANNEX" for r in results)


def test_vector_retrieval_empty_query_returns_no_results(db_session):
    _, _, clauses = _seed(db_session)
    provider = DeterministicHashEmbeddingProvider()
    ClauseEmbeddingService(db_session, provider).embed_clauses(clauses)
    db_session.commit()

    backend = VectorRetrievalBackend(db_session, provider)
    service = BISRetrievalService(db_session, backends={"vector": backend})

    assert service.search("", method="vector") == []
    assert service.search("   ", method="vector") == []


# ---------------------------------------------------------------------------
# Existing keyword retrieval still works + backend swapping
# ---------------------------------------------------------------------------

def test_keyword_retrieval_still_works_after_milestone_2_changes(db_session):
    standard, document, clauses = _seed(db_session)
    service = BISRetrievalService(db_session)  # default keyword backend, unchanged usage

    results = service.search("Earthing")
    assert results
    assert any(r.relevance.method == "keyword" for r in results)


def test_service_can_swap_between_keyword_and_vector_by_method(db_session):
    standard, document, clauses = _seed(db_session)
    provider = DeterministicHashEmbeddingProvider()
    ClauseEmbeddingService(db_session, provider).embed_clauses(clauses)
    db_session.commit()

    vector_backend = VectorRetrievalBackend(db_session, provider)
    service = BISRetrievalService(db_session, backends={"vector": vector_backend})

    keyword_results = service.search("Earthing")  # default backend, no method= given
    vector_results = service.search("Earthing", method="vector")

    assert keyword_results and keyword_results[0].relevance.method == "keyword"
    assert vector_results and vector_results[0].relevance.method == "vector"
