"""
Milestone 5 corpus expansion — additional real, freely-accessible BIS QCO
Gazette notifications.

Ingests TWO more real, freely downloadable, official Government-of-India
Gazette notifications from bis.gov.in (verified reachable before this
script was written — see MILESTONE_5_REPORT.md). Both are the same kind
of source as the Milestone 1 pilot document: a QCO (Quality Control
Order) *implementing* one or more Indian Standards, NOT the sold/
copyrighted text of the Indian Standard itself.

IMPORTANT — Standard vs. Document distinction (explicit project rule):
    The clauses ingested here come ONLY from the QCO/Gazette PDF text.
    They are attached to a Document with document_type="QCO". The
    Standard rows created below carry ONLY the handful of fields the QCO
    Gazette notification itself states about the standard it references
    (IS number, title, year) — never the standard's own scope/clause
    text, which this project does not have access to and does not
    fabricate. Do not confuse "a QCO that references IS 3513" with
    "the text of IS 3513" — they are different documents in this schema.

Sources (each URL was verified reachable — HTTP 200 — before ingestion):
  1. Resin Treated Compressed Wood Laminates (Quality Control) Order, 2024
     https://www.bis.gov.in/wp-content/uploads/2024/03/Resin-Treated-Compressed-Wood-Laminates-QCO-2024.pdf
     References THREE Indian Standards (IS 3513 Parts 1/2/3:1989) for three
     different use-cases of the same product family. Because
     ``documents.standard_id`` is a single nullable FK (by design — see
     app/models/document.py docstring: "some documents ... are not tied
     to a single IS number"), this Document's standard_id is left NULL
     rather than arbitrarily picking one of the three as "the" standard.
     All three Standard rows are still ingested and are independently
     searchable/citable by IS number.

  2. Self-Contained Drinking Water Cooler (Quality Control) Order, 2024
     https://bis.gov.in/wp-content/uploads/2024/05/Self-Contained-Drinking-Water-Cooler-QCO-2024.pdf
     References ONE Indian Standard (IS 1475 (Part 1):2001), so this
     Document's standard_id IS set, same pattern as the Milestone 1 pilot.

Bilingual Gazette text handling:
    Both PDFs are official bilingual (Hindi + English) Gazette of India
    notifications. Before clause parsing, page text is cleaned with
    app.services.normalizer.strip_devanagari_lines() (drops the Hindi
    rendering of the same legal order) and strip_gazette_boilerplate_lines()
    (drops generic Gazette running-heads/footers) — see that module's
    docstring for the full rationale. This preprocessing is NOT applied to
    the Milestone 1 pilot document (which is English-only and unaffected).

Reliability / "skip rather than force" policy:
    If a source fails to fetch, fails to extract, or yields zero usable
    clauses after cleaning, this script logs the failure, SKIPS that
    source, and continues with the others — it never fabricates a
    Document/Clause to make the count look better. See the printed
    summary and MILESTONE_5_REPORT.md for what happened for each source.

Usage:
    python scripts/ingest_additional_standards.py

Safe to re-run: ingestion is idempotent (upsert by natural key, same as
scripts/ingest_pilot.py). Does NOT touch the existing IS 302 pilot data.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from app.db.session import SessionLocal  # noqa: E402
from app.services.bis_ingestion import (  # noqa: E402
    BISIngestionService,
    RawDocument,
    RawStandard,
)
from app.services.clause_parser import parse_pages_to_clauses  # noqa: E402
from app.services.fetcher import FetchError, fetch_to_raw  # noqa: E402
from app.services.normalizer import (  # noqa: E402
    strip_devanagari_lines,
    strip_gazette_boilerplate_lines,
)
from app.services.pdf_extractor import PdfExtractionError, extract_pages  # noqa: E402
from app.services.provenance import log_event  # noqa: E402

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass
class QCOSource:
    slug: str
    filename: str
    url: str
    document_title: str
    standards: List[RawStandard]
    # IS number of the ONE standard to link via Document.standard_id, or
    # None if this QCO spans multiple standards (left NULL — see module
    # docstring).
    primary_standard_is_number: Optional[str]


# ---------------------------------------------------------------------------
# Source catalogue. Only fields actually stated in each QCO's own Gazette
# table are filled in below (IS number, title of the standard, year).
# Everything else (scope, technical_committee, etc.) is left None —
# consistent with the project rule to never fabricate Standard metadata
# beyond what a real source states.
# ---------------------------------------------------------------------------

SOURCES: List[QCOSource] = [
    QCOSource(
        slug="resin_wood_laminates_qco_2024",
        filename="Resin-Treated-Compressed-Wood-Laminates-QCO-2024.pdf",
        url=(
            "https://www.bis.gov.in/wp-content/uploads/2024/03/"
            "Resin-Treated-Compressed-Wood-Laminates-QCO-2024.pdf"
        ),
        document_title=(
            "Resin Treated Compressed Wood Laminates (Quality Control) Order, "
            "2024 (S.O. 1018(E), Gazette of India, dated 29 February 2024)"
        ),
        standards=[
            RawStandard(
                is_number="IS 3513 (Part 1)",
                title=(
                    "Resin treated compressed wood laminates (compregs) — "
                    "For electrical purposes"
                ),
                year=1989,
                status="ACTIVE",  # per the currently-in-force 2024 QCO's own table
                source_url=None,  # no dedicated standard metadata page was fetched
            ),
            RawStandard(
                is_number="IS 3513 (Part 2)",
                title=(
                    "Resin treated compressed wood laminates (compregs) — "
                    "For chemical purposes"
                ),
                year=1989,
                status="ACTIVE",
                source_url=None,
            ),
            RawStandard(
                is_number="IS 3513 (Part 3)",
                title=(
                    "Resin treated compressed wood laminates (compregs) — "
                    "For general purposes"
                ),
                year=1989,
                status="ACTIVE",
                source_url=None,
            ),
        ],
        primary_standard_is_number=None,
    ),
    QCOSource(
        slug="drinking_water_cooler_qco_2024",
        filename="Self-Contained-Drinking-Water-Cooler-QCO-2024.pdf",
        url=(
            "https://bis.gov.in/wp-content/uploads/2024/05/"
            "Self-Contained-Drinking-Water-Cooler-QCO-2024.pdf"
        ),
        document_title=(
            "Self-Contained Drinking Water Cooler (Quality Control) Order, "
            "2024 (S.O. 2112(E), Gazette of India, dated 24 May 2024)"
        ),
        standards=[
            RawStandard(
                is_number="IS 1475 (Part 1)",
                title="Self-Contained Drinking Water Coolers — Energy Consumption and Performance",
                year=2001,
                status="ACTIVE",
                source_url=None,
            ),
        ],
        primary_standard_is_number="IS 1475 (Part 1)",
    ),
]


def _clean_gazette_pages(pages):
    """Bilingual-Gazette-specific cleanup — see module docstring."""
    return [
        (page_no, strip_gazette_boilerplate_lines(strip_devanagari_lines(text)))
        for page_no, text in pages
    ]


def ingest_one_source(source: QCOSource) -> dict:
    """Fetch/extract/parse/persist one QCOSource. Never raises for a
    recoverable failure — returns a result dict with outcome details so
    main() can print a clear per-source report and skip cleanly."""
    result = {"slug": source.slug, "url": source.url, "outcome": None, "detail": None}

    try:
        fetch_result = fetch_to_raw(
            url=source.url, slug=source.slug, filename=source.filename, raw_dir=DATA_RAW_DIR
        )
    except FetchError as exc:
        log_event(source_url=source.url, entity_type="Document", entity_id=None,
                   outcome="failed", error=str(exc))
        result["outcome"] = "SKIPPED"
        result["detail"] = f"fetch failed: {exc}"
        return result

    try:
        pages = extract_pages(fetch_result.local_path)
    except PdfExtractionError as exc:
        log_event(source_url=source.url, entity_type="Document", entity_id=None,
                   outcome="failed", content_hash=fetch_result.content_hash, error=str(exc))
        result["outcome"] = "SKIPPED"
        result["detail"] = f"PDF extraction failed: {exc}"
        return result

    cleaned_pages = _clean_gazette_pages(pages)
    raw_clauses = parse_pages_to_clauses(cleaned_pages, language="en")

    if not raw_clauses:
        log_event(source_url=source.url, entity_type="Document", entity_id=None,
                   outcome="failed", content_hash=fetch_result.content_hash,
                   error="No clauses parsed after cleaning")
        result["outcome"] = "SKIPPED"
        result["detail"] = "no clauses parsed after cleaning — not forced into the corpus"
        return result

    raw_document = RawDocument(
        title=source.document_title,
        document_type="QCO",
        source_url=source.url,
        standard_is_number=source.primary_standard_is_number,
        local_file_path=str(fetch_result.local_path),
        content_hash=fetch_result.content_hash,
        language="en",
    )

    session = SessionLocal()
    try:
        service = BISIngestionService(session)

        standards_info = []
        for raw_standard in source.standards:
            standard, created = service.ingest_standard(raw_standard)
            standards_info.append(
                (standard.is_number, standard.year, standard.id, created)
            )

        # Resolve the primary standard object (if any) so ingest_document
        # doesn't have to re-look it up — mirrors ingest_pilot.py's pattern.
        from app.models.standard import Standard  # local import, avoids polluting module scope
        from sqlalchemy import select

        primary_standard = None
        if source.primary_standard_is_number:
            primary_standard = session.execute(
                select(Standard)
                .where(Standard.is_number == source.primary_standard_is_number)
                .order_by(Standard.year.desc())
            ).scalars().first()

        document, doc_created, clauses_written = service.ingest_document(
            raw_document, raw_clauses, standard=primary_standard
        )
        session.commit()

        document_id = document.id
        document_title = document.title
    except Exception as exc:
        session.rollback()
        log_event(source_url=source.url, entity_type="Document", entity_id=None,
                   outcome="failed", content_hash=fetch_result.content_hash, error=str(exc))
        result["outcome"] = "FAILED"
        result["detail"] = str(exc)
        return result
    finally:
        session.close()

    for is_number, year, sid, created in standards_info:
        log_event(source_url=source.url, entity_type="Standard", entity_id=sid,
                  outcome="created" if created else "updated")
    log_event(source_url=source.url, entity_type="Document", entity_id=document_id,
              outcome="created" if doc_created else "updated",
              content_hash=fetch_result.content_hash,
              extra={"clauses_written": clauses_written})

    result["outcome"] = "OK"
    result["detail"] = {
        "document_id": document_id,
        "document_title": document_title,
        "document_created": doc_created,
        "clauses_written": clauses_written,
        "standards": standards_info,
    }
    return result


def main() -> int:
    print("=== Milestone 5: Additional BIS QCO Corpus Ingestion ===\n")
    any_failed = False
    for source in SOURCES:
        print(f"--- {source.slug} ---")
        print(f"  URL: {source.url}")
        result = ingest_one_source(source)
        print(f"  Outcome: {result['outcome']}")
        if result["outcome"] == "OK":
            d = result["detail"]
            print(f"  Document  : {'CREATED' if d['document_created'] else 'UPDATED'} "
                  f"(id={d['document_id']}) clauses_written={d['clauses_written']}")
            for is_number, year, sid, created in d["standards"]:
                print(f"  Standard  : {'CREATED' if created else 'UPDATED'} "
                      f"(id={sid}) {is_number}:{year}")
        else:
            print(f"  Detail: {result['detail']}")
            any_failed = any_failed or result["outcome"] == "FAILED"
        print()

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
