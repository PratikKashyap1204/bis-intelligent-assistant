"""
PDF text extraction, page by page.

Wraps pdfplumber. Given a local PDF path, returns the text of every page
in reading order, preserving 1-indexed page numbers so downstream code
(the clause parser, and eventually citations shown to end users) can
always point back to an exact page.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be opened, or has no usable pages."""


def extract_pages(pdf_path: Path | str) -> list[tuple[int, str]]:
    """
    Extract text from every page of a PDF.

    Returns:
        A list of (page_number, page_text) tuples, 1-indexed. Pages with
        no extractable text (e.g. scanned image pages with no text layer)
        still appear in the result with an empty string — pages are never
        silently dropped, so callers can tell "no text" apart from
        "missing page" and decide how to handle it (e.g. OCR later).

    Raises:
        PdfExtractionError: if the file doesn't exist, isn't a valid PDF,
        or has zero pages.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise PdfExtractionError(f"PDF not found: {pdf_path}")

    pages: list[tuple[int, str]] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                raise PdfExtractionError(f"PDF has zero pages: {pdf_path}")

            for index, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:  # pdfplumber/pdfminer can raise varied per-page errors
                    logger.warning(
                        "Failed to extract text from page %s of %s: %s",
                        index,
                        pdf_path,
                        exc,
                    )
                    text = ""
                if not text.strip():
                    logger.info("Page %s of %s has no extractable text.", index, pdf_path)
                pages.append((index, text))
    except PdfExtractionError:
        raise
    except Exception as exc:
        raise PdfExtractionError(f"Failed to open/parse PDF {pdf_path}: {exc}") from exc

    return pages
