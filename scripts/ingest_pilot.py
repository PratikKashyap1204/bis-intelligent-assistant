"""
Pilot ingestion script — IS 302 (Part 1):2024.

Ingests ONE real Indian Standard end-to-end, using two official BIS
sources (no invented data, no paywalled/copyrighted content):

  1. Standard metadata — manually transcribed from BIS's official
     "Know Your Standard" portal:
     https://www.services.bis.gov.in/php/BIS_2.0/bisconnect/knowyourstandards/Indian_standards/isdetails/MzE5MDE=
     (Every field below was read directly off that page. Fields the page
     did not provide — e.g. free-text `scope`, Hindi title — are left
     as None rather than guessed. `status=ACTIVE` is corroborated by
     BIS's e-Sale listing for the same standard.)

  2. Document + clauses — the official BIS circular implementing the
     2024 QCO for this standard (freely published PDF, NOT the paid/
     copyrighted IS 302 standard text itself):
     https://www.bis.gov.in/wp-content/uploads/2024/12/Circular_GcO6_2024-12-16.pdf
     (Ref. CMD-III/16, dated 16 Dec 2024)

Usage:
    python scripts/ingest_pilot.py

Safe to re-run: ingestion is idempotent (upsert by natural key — see
backend/app/services/bis_ingestion.py). Re-running should report
"UPDATED" instead of "CREATED" and the same clause count, never growing.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from app.services.pdf_extractor import PdfExtractionError, extract_pages  # noqa: E402
from app.services.provenance import log_event  # noqa: E402

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
SLUG = "is_302_part1_2024"

# ---------------------------------------------------------------------------
# 1. Official BIS metadata — transcribed directly from "Know Your Standard".
#    Fields not shown on that page are left as None (never guessed).
# ---------------------------------------------------------------------------
STANDARD_METADATA_SOURCE_URL = (
    "https://www.services.bis.gov.in/php/BIS_2.0/bisconnect/knowyourstandards/"
    "Indian_standards/isdetails/MzE5MDE="
)

RAW_STANDARD = RawStandard(
    is_number="IS 302 (Part 1)",
    title=(
        "Household and Similar Electrical Appliances \u2014 Safety Part 1 "
        "General Requirements (Seventh Revision)"
    ),
    year=2024,
    revision="Seventh Revision",
    status="ACTIVE",  # corroborated by BIS e-Sale listing for this standard
    scope=None,  # not published as free text on the metadata page
    source_url=STANDARD_METADATA_SOURCE_URL,
    technical_committee="ETD 32",
    group="Electrical Appliances and Accessories",
    sub_group="Domestic Appliances",
    sub_sub_group="Safety of electrical appliances",
    aspect="Safety Standard",
    certification_type="Mandatory Registration",
    relevant_ministries="Ministry of Commerce and Industry",
    reaffirmed_year=None,  # left blank on the official page
    equivalent_international_standard="IEC 60335-1:2020",
    degree_of_equivalence="Identical under dual numbering",
    harmonized_with="IEC",
    price=None,  # not shown on this metadata source
    num_amendments=0,  # page states "No amendment issued"
    title_hi=None,  # no separate Hindi title text was published
    scope_hi=None,
)

# ---------------------------------------------------------------------------
# 2. Official BIS QCO implementation circular (freely published PDF).
# ---------------------------------------------------------------------------
QCO_CIRCULAR_URL = "https://www.bis.gov.in/wp-content/uploads/2024/12/Circular_GcO6_2024-12-16.pdf"
QCO_CIRCULAR_FILENAME = "Circular_GcO6_2024-12-16.pdf"
QCO_CIRCULAR_TITLE = (
    "Implementation of Safety of Household, Commercial and Similar Electrical "
    "Appliances (Quality Control) Order, 2024 and revised IS 302 (Part 1):2024/"
    "IEC 60335-1:2020 (Circular Ref. CMD-III/16, dated 16 Dec 2024)"
)


def main() -> int:
    print(f"=== BIS Pilot Ingestion: {RAW_STANDARD.is_number}:{RAW_STANDARD.year} ===\n")
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # Fetch the QCO circular PDF (the Standard's own metadata above was
    # transcribed directly from its HTML source — no file to download).
    # ------------------------------------------------------------------
    try:
        fetch_result = fetch_to_raw(
            url=QCO_CIRCULAR_URL,
            slug=SLUG,
            filename=QCO_CIRCULAR_FILENAME,
            raw_dir=DATA_RAW_DIR,
        )
    except FetchError as exc:
        log_event(
            source_url=QCO_CIRCULAR_URL,
            entity_type="Document",
            entity_id=None,
            outcome="failed",
            error=str(exc),
        )
        print(f"FAILED to fetch QCO circular: {exc}")
        return 1

    print(
        f"Fetched document : {fetch_result.local_path}\n"
        f"  size            : {fetch_result.byte_size} bytes\n"
        f"  sha256          : {fetch_result.content_hash}\n"
        f"  from_cache      : {fetch_result.from_cache}"
    )

    # ------------------------------------------------------------------
    # Extract PDF text page-by-page.
    # ------------------------------------------------------------------
    try:
        pages = extract_pages(fetch_result.local_path)
    except PdfExtractionError as exc:
        log_event(
            source_url=QCO_CIRCULAR_URL,
            entity_type="Document",
            entity_id=None,
            outcome="failed",
            content_hash=fetch_result.content_hash,
            error=str(exc),
        )
        print(f"FAILED to extract PDF text: {exc}")
        return 1

    print(f"Extracted        : {len(pages)} page(s)")
    empty_pages = [p for p, text in pages if not text.strip()]
    if empty_pages:
        warnings.append(f"{len(empty_pages)} page(s) had no extractable text: {empty_pages}")

    # ------------------------------------------------------------------
    # Parse pages into clauses.
    # ------------------------------------------------------------------
    raw_clauses = parse_pages_to_clauses(pages, language="en")
    if not raw_clauses:
        warnings.append("No clauses were parsed from the document.")

    raw_document = RawDocument(
        title=QCO_CIRCULAR_TITLE,
        document_type="QCO",
        source_url=QCO_CIRCULAR_URL,
        standard_is_number=RAW_STANDARD.is_number,
        local_file_path=str(fetch_result.local_path),
        content_hash=fetch_result.content_hash,
        language="en",
    )

    # ------------------------------------------------------------------
    # Persist through BISIngestionService inside ONE transaction — a
    # failure anywhere below rolls back the whole run, never leaving
    # partially-committed data.
    # ------------------------------------------------------------------
    session = SessionLocal()
    try:
        service = BISIngestionService(session)
        standard, standard_created = service.ingest_standard(RAW_STANDARD)
        document, document_created, clauses_written = service.ingest_document(
            raw_document, raw_clauses, standard=standard
        )
        session.commit()
        # Capture plain values BEFORE closing the session — session.commit()
        # expires ORM attributes by default, so reading them after the
        # session is closed would raise DetachedInstanceError.
        standard_id = standard.id
        standard_is_number = standard.is_number
        standard_year = standard.year
        document_id = document.id
        document_title = document.title
    except Exception as exc:
        session.rollback()
        log_event(
            source_url=QCO_CIRCULAR_URL,
            entity_type="Document",
            entity_id=None,
            outcome="failed",
            content_hash=fetch_result.content_hash,
            error=str(exc),
        )
        print(f"FAILED to persist ingestion: {exc}")
        raise
    finally:
        session.close()

    # ------------------------------------------------------------------
    # Provenance logging (success).
    # ------------------------------------------------------------------
    log_event(
        source_url=STANDARD_METADATA_SOURCE_URL,
        entity_type="Standard",
        entity_id=standard_id,
        outcome="created" if standard_created else "updated",
    )
    log_event(
        source_url=QCO_CIRCULAR_URL,
        entity_type="Document",
        entity_id=document_id,
        outcome="created" if document_created else "updated",
        content_hash=fetch_result.content_hash,
        extra={"clauses_written": clauses_written},
    )

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    print("\n=== Ingestion Summary ===")
    print(f"Standard    : {'CREATED' if standard_created else 'UPDATED'} "
          f"(id={standard_id}) {standard_is_number}:{standard_year}")
    print(f"Document    : {'CREATED' if document_created else 'UPDATED'} "
          f"(id={document_id}) {document_title[:70]}...")
    print(f"Clauses     : {clauses_written} extracted/written")
    print("Source URLs :")
    print(f"  - Standard metadata : {STANDARD_METADATA_SOURCE_URL}")
    print(f"  - QCO circular PDF  : {QCO_CIRCULAR_URL}")
    print(f"Content hash (PDF) : {fetch_result.content_hash}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nNo warnings.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
