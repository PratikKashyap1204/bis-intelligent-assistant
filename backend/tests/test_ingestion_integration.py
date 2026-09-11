"""
Integration tests for BISIngestionService.

Require a real PostgreSQL database (TEST_DATABASE_URL) because Standard
and Document use JSONB-adjacent/Postgres-specific behaviour elsewhere in
the schema; these tests are skipped automatically when TEST_DATABASE_URL
is not set (see conftest.py's `pg_engine` / `db_session` fixtures).

All sample data below is obviously synthetic ("IS 9999", "example.invalid")
— never real BIS content — per the project rule that automated tests must
not depend on external BIS availability and must not fabricate real
standard data.
"""

from __future__ import annotations

import dataclasses

from app.models.clause import Clause
from app.models.document import Document
from app.models.standard import Standard
from app.services.bis_ingestion import BISIngestionService, RawClause, RawDocument, RawStandard

SAMPLE_STANDARD = RawStandard(
    is_number="IS 9999",
    title="Sample Test Standard for Ingestion Tests",
    year=2099,
    status="ACTIVE",
    source_url="https://example.invalid/sample-standard",
)

SAMPLE_DOCUMENT = RawDocument(
    title="Sample Test Document",
    document_type="OTHER",
    source_url="https://example.invalid/sample-document.pdf",
    standard_is_number="IS 9999",
    content_hash="deadbeef" * 8,
)

SAMPLE_CLAUSES = [
    RawClause(clause_number="1", content="Sample clause one content."),
    RawClause(clause_number="2", content="Sample clause two content."),
]


# ---------------------------------------------------------------------------
# Standard: creation + duplicate detection
# ---------------------------------------------------------------------------

def test_ingest_standard_creates_new_row(db_session):
    service = BISIngestionService(db_session)

    standard, created = service.ingest_standard(SAMPLE_STANDARD)
    db_session.commit()

    assert created is True
    assert standard.id is not None
    count = db_session.query(Standard).filter(Standard.is_number == "IS 9999").count()
    assert count == 1


def test_ingest_standard_twice_updates_instead_of_duplicating(db_session):
    service = BISIngestionService(db_session)

    first, created_first = service.ingest_standard(SAMPLE_STANDARD)
    db_session.commit()

    updated_raw = dataclasses.replace(SAMPLE_STANDARD, title="Updated Title")
    second, created_second = service.ingest_standard(updated_raw)
    db_session.commit()

    assert created_first is True
    assert created_second is False
    assert first.id == second.id

    count = db_session.query(Standard).filter(Standard.is_number == "IS 9999").count()
    assert count == 1

    refreshed = db_session.get(Standard, first.id)
    assert refreshed.title == "Updated Title"


# ---------------------------------------------------------------------------
# Document + clauses: creation + duplicate detection + idempotency
# ---------------------------------------------------------------------------

def test_ingest_document_with_clauses_creates_rows(db_session):
    service = BISIngestionService(db_session)
    standard, _ = service.ingest_standard(SAMPLE_STANDARD)

    document, created, clause_count = service.ingest_document(
        SAMPLE_DOCUMENT, SAMPLE_CLAUSES, standard=standard
    )
    db_session.commit()

    assert created is True
    assert clause_count == 2
    assert document.standard_id == standard.id

    stored = db_session.query(Clause).filter(Clause.document_id == document.id).all()
    assert len(stored) == 2


def test_ingest_document_twice_is_idempotent_not_duplicated(db_session):
    service = BISIngestionService(db_session)
    standard, _ = service.ingest_standard(SAMPLE_STANDARD)

    doc1, created1, count1 = service.ingest_document(
        SAMPLE_DOCUMENT, SAMPLE_CLAUSES, standard=standard
    )
    db_session.commit()

    doc2, created2, count2 = service.ingest_document(
        SAMPLE_DOCUMENT, SAMPLE_CLAUSES, standard=standard
    )
    db_session.commit()

    assert created1 is True
    assert created2 is False
    assert doc1.id == doc2.id

    doc_rows = (
        db_session.query(Document)
        .filter(Document.source_url == SAMPLE_DOCUMENT.source_url)
        .count()
    )
    assert doc_rows == 1

    clause_rows = db_session.query(Clause).filter(Clause.document_id == doc2.id).count()
    assert clause_rows == 2  # NOT 4 — clauses were replaced, not appended


def test_ingest_document_deduplicates_by_content_hash_even_with_new_url(db_session):
    """Same content_hash, different source_url -> still treated as the same document."""
    service = BISIngestionService(db_session)
    standard, _ = service.ingest_standard(SAMPLE_STANDARD)

    doc1, created1, _ = service.ingest_document(SAMPLE_DOCUMENT, SAMPLE_CLAUSES, standard=standard)
    db_session.commit()

    mirrored = dataclasses.replace(
        SAMPLE_DOCUMENT, source_url="https://example.invalid/mirror-copy.pdf"
    )
    doc2, created2, _ = service.ingest_document(mirrored, SAMPLE_CLAUSES, standard=standard)
    db_session.commit()

    assert created1 is True
    assert created2 is False
    assert doc1.id == doc2.id
    # source_url was updated to the latest mirror, but it's still one row
    assert db_session.query(Document).count() == 1


# ---------------------------------------------------------------------------
# Full pilot-style flow, run twice end-to-end
# ---------------------------------------------------------------------------

def test_full_ingestion_flow_run_twice_produces_no_duplicates(db_session):
    """
    Simulates scripts/ingest_pilot.py's flow with sample (non-BIS) data,
    executed twice — the exact scenario required by the project's
    idempotency rule.
    """
    service = BISIngestionService(db_session)
    last_document = None

    for _ in range(2):
        standard, _ = service.ingest_standard(SAMPLE_STANDARD)
        document, _, _ = service.ingest_document(SAMPLE_DOCUMENT, SAMPLE_CLAUSES, standard=standard)
        db_session.commit()
        last_document = document

    assert db_session.query(Standard).filter(Standard.is_number == "IS 9999").count() == 1
    assert (
        db_session.query(Document)
        .filter(Document.source_url == SAMPLE_DOCUMENT.source_url)
        .count()
        == 1
    )
    assert (
        db_session.query(Clause).filter(Clause.document_id == last_document.id).count() == 2
    )
