"""
BIS data ingestion service.

This module defines:
  1. Raw data containers — lightweight dataclasses populated by the
     fetcher/PDF-extraction/clause-parsing modules (or, for now, manually
     transcribed from an official BIS source — see scripts/ingest_pilot.py).
  2. BISIngestionService — persists that data into PostgreSQL via
     SQLAlchemy, with upsert-by-natural-key semantics so re-running
     ingestion is idempotent.

Only ``ingest_standard`` and ``ingest_document`` are implemented, matching
the current pilot scope (one Standard + one Document + its Clauses).
``ingest_laboratory`` and ``ingest_certification_requirement`` remain
interface stubs for a later stage.

Ingestion workflow (current pilot)
-----------------------------------
1. Acquisition   — app/services/fetcher.py downloads a known official URL.
2. Extraction    — app/services/pdf_extractor.py reads text page-by-page.
3. Structuring   — app/services/clause_parser.py turns pages into RawClause.
4. Normalization — app/services/normalizer.py cleans whitespace artifacts.
5. Persistence   — BISIngestionService (this module) upserts into the DB.
6. Provenance    — app/services/provenance.py appends a JSONL log entry.

Future ingestion phases
------------------------
Phase 2: Generalize the fetcher into a rate-limited, repeatable scraper
         for multiple standards (still fetch-by-known-URL, just more of
         them, driven by a catalogue rather than a single script).
Phase 3: ingest_laboratory / ingest_certification_requirement.
Phase 4: Embeddings + RAG — embed Clause.content, store vectors in pgvector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.clause import Clause
from app.models.document import Document
from app.models.standard import Standard


# ---------------------------------------------------------------------------
# Raw data containers
# ---------------------------------------------------------------------------
# Plain Python dataclasses — no SQLAlchemy, no DB imports, no I/O. A fetcher,
# PDF extractor, or clause parser produces instances of these; the service
# below persists them.

@dataclass
class RawStandard:
    """Data for a single BIS standard, transcribed/scraped from an official source."""

    is_number: str
    title: str
    year: int | None = None
    revision: str | None = None
    status: str = "ACTIVE"
    scope: str | None = None
    source_url: str | None = None

    # Extended metadata from BIS's "Know Your Standard" portal — all optional,
    # not every source publishes every field.
    technical_committee: str | None = None
    group: str | None = None
    sub_group: str | None = None
    sub_sub_group: str | None = None
    aspect: str | None = None
    certification_type: str | None = None
    relevant_ministries: str | None = None
    reaffirmed_year: int | None = None
    equivalent_international_standard: str | None = None
    degree_of_equivalence: str | None = None
    harmonized_with: str | None = None
    price: float | None = None
    num_amendments: int | None = None
    title_hi: str | None = None
    scope_hi: str | None = None


@dataclass
class RawDocument:
    """Data for a BIS document (PDF page, HTML, gazette) before DB persistence."""

    title: str
    document_type: str       # Must be a valid DocumentType value
    source_url: str | None = None
    standard_is_number: str | None = None   # Resolved to standard_id on insert
    published_date: date | None = None
    local_file_path: str | None = None
    content_hash: str | None = None
    language: str | None = None
    amendment_number: str | None = None


@dataclass
class RawClause:
    """A single clause/section extracted from a BIS document."""

    clause_number: str
    content: str
    title: str | None = None
    page_number: int | None = None
    language: str | None = None
    clause_type: str | None = None
    # Milestone 5: 1-based position of this clause among all clauses
    # parsed for the SAME document, in document order. Purely additive
    # disambiguation/provenance metadata for when ``clause_number`` is not
    # globally unique within a document (see clause_parser.py) — it does
    # NOT replace Clause.id as the primary stable identity, and it is
    # optional so existing callers that build RawClause manually (tests,
    # older scripts) are unaffected.
    sequence_in_document: int | None = None


@dataclass
class RawLaboratory:
    """Scraped data for a BIS-recognised testing laboratory."""

    name: str
    lab_code: str
    address: str | None = None
    state: str | None = None
    district: str | None = None
    source_url: str | None = None
    validity_date: date | None = None


@dataclass
class RawCertificationRequirement:
    """Scraped data for a product certification requirement."""

    product_name: str        # Resolved to product_id on insert
    standard_is_number: str  # Resolved to standard_id on insert
    scheme: str | None = None
    mandatory: bool = False
    qco: bool = False
    requirements: dict[str, Any] = field(default_factory=dict)
    source_url: str | None = None


# ---------------------------------------------------------------------------
# Ingestion service
# ---------------------------------------------------------------------------

# Fields on RawStandard that map 1:1 onto Standard columns of the same name,
# excluding the natural-key fields (is_number, year) which are handled
# separately during upsert lookup/insert.
_STANDARD_MUTABLE_FIELDS = (
    "title",
    "revision",
    "status",
    "scope",
    "source_url",
    "technical_committee",
    "group",
    "sub_group",
    "sub_sub_group",
    "aspect",
    "certification_type",
    "relevant_ministries",
    "reaffirmed_year",
    "equivalent_international_standard",
    "degree_of_equivalence",
    "harmonized_with",
    "price",
    "num_amendments",
    "title_hi",
    "scope_hi",
)

_DOCUMENT_MUTABLE_FIELDS = (
    "document_type",
    "title",
    "source_url",
    "local_file_path",
    "published_date",
    "language",
    "content_hash",
    "amendment_number",
)


class BISIngestionService:
    """
    Persists RawStandard / RawDocument / RawClause data into PostgreSQL.

    All methods use upsert-by-natural-key semantics (see each docstring)
    so re-running ingestion with the same input is idempotent: it updates
    existing rows in place rather than inserting duplicates.

    Methods use ``session.flush()``, not ``session.commit()`` — the
    caller controls the transaction boundary (see scripts/ingest_pilot.py),
    so a failure partway through an ingestion run can be rolled back as a
    single unit rather than leaving partially-committed data.
    """

    def __init__(self, session: Session):
        self.session = session

    # ------------------------------------------------------------------
    # Standards
    # ------------------------------------------------------------------

    def ingest_standard(self, raw: RawStandard) -> tuple[Standard, bool]:
        """
        Upsert a BIS standard.

        Natural identity: (is_number, year). If a row with the same
        IS number and year already exists, its mutable fields are updated
        in place; otherwise a new row is inserted.

        Returns:
            (standard, created) — the ORM instance and whether it was
            newly inserted (True) or updated (False).
        """
        existing = self.session.execute(
            select(Standard).where(
                Standard.is_number == raw.is_number,
                Standard.year == raw.year,
            )
        ).scalar_one_or_none()

        values = {name: getattr(raw, name) for name in _STANDARD_MUTABLE_FIELDS}

        if existing is not None:
            for name, value in values.items():
                setattr(existing, name, value)
            self.session.flush()
            return existing, False

        standard = Standard(is_number=raw.is_number, year=raw.year, **values)
        self.session.add(standard)
        self.session.flush()  # assign standard.id
        return standard, True

    # ------------------------------------------------------------------
    # Documents (+ clauses)
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        raw: RawDocument,
        clauses: Optional[list[RawClause]] = None,
        standard: Optional[Standard] = None,
    ) -> tuple[Document, bool, int]:
        """
        Upsert a BIS document, optionally replacing its clauses.

        Natural identity: ``content_hash`` if present, otherwise
        ``source_url``. If a matching row exists, its mutable fields are
        updated in place; otherwise a new row is inserted.

        The parent Standard is resolved from the ``standard`` argument if
        given (preferred — avoids a redundant lookup when the caller has
        just ingested it in the same run), otherwise by looking up
        ``raw.standard_is_number`` (most recent year wins if there are
        multiple matches).

        If ``clauses`` is provided, ALL existing clauses for this document
        are replaced with the new set — this keeps re-ingestion idempotent
        (the same document always ends up with exactly the clauses just
        parsed, never an ever-growing duplicated set).

        Returns:
            (document, created, clauses_written)
        """
        resolved_standard_id = standard.id if standard is not None else None
        if resolved_standard_id is None and raw.standard_is_number:
            match = self.session.execute(
                select(Standard)
                .where(Standard.is_number == raw.standard_is_number)
                .order_by(Standard.year.desc())
            ).scalars().first()
            if match is not None:
                resolved_standard_id = match.id

        existing = None
        if raw.content_hash:
            existing = self.session.execute(
                select(Document).where(Document.content_hash == raw.content_hash)
            ).scalar_one_or_none()
        if existing is None and raw.source_url:
            existing = self.session.execute(
                select(Document).where(Document.source_url == raw.source_url)
            ).scalar_one_or_none()

        values = {name: getattr(raw, name) for name in _DOCUMENT_MUTABLE_FIELDS}
        values["standard_id"] = resolved_standard_id

        if existing is not None:
            for name, value in values.items():
                setattr(existing, name, value)
            document = existing
            created = False
        else:
            document = Document(**values)
            self.session.add(document)
            created = True

        self.session.flush()  # assign document.id

        clauses_written = 0
        if clauses is not None:
            # Idempotent replace: delete whatever clauses this document
            # currently has, then write the freshly-parsed set. Re-parsing
            # the same document never appends duplicates.
            self.session.query(Clause).filter(
                Clause.document_id == document.id
            ).delete(synchronize_session=False)

            for raw_clause in clauses:
                self.session.add(
                    Clause(
                        document_id=document.id,
                        clause_number=raw_clause.clause_number,
                        title=raw_clause.title,
                        content=raw_clause.content,
                        page_number=raw_clause.page_number,
                        language=raw_clause.language,
                        clause_type=raw_clause.clause_type,
                        sequence_in_document=raw_clause.sequence_in_document,
                    )
                )
                clauses_written += 1

            self.session.flush()

        return document, created, clauses_written

    # ------------------------------------------------------------------
    # Laboratories — not yet implemented (out of scope for this pilot)
    # ------------------------------------------------------------------

    def ingest_laboratory(self, raw: RawLaboratory) -> None:
        """
        Validate and upsert a BIS laboratory entry.

        Implementation steps (future stage):
        - Deduplicate by lab_code.
        - Update validity_date, address, state, district on re-ingestion.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Certification requirements — not yet implemented (out of scope)
    # ------------------------------------------------------------------

    def ingest_certification_requirement(self, raw: RawCertificationRequirement) -> None:
        """
        Validate and persist a product certification requirement.

        Implementation steps (future stage):
        - Resolve Product by name (exact match or fuzzy).
        - Resolve Standard by is_number.
        - Insert CertificationRequirement row.
        - Optionally create EntityRelationship edge: PRODUCT → GOVERNED_BY → STANDARD.
        """
        raise NotImplementedError
