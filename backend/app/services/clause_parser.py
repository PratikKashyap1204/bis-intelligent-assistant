"""
Modular clause/section parser.

Converts extracted PDF page text (the output of
``app.services.pdf_extractor.extract_pages``) into a list of
``RawClause`` objects, ready for ``BISIngestionService.ingest_document``.

Supported structures (initial version)
----------------------------------------
- Numbered clauses/paragraphs: "1", "1.1", "1.1.2", optionally with a
  trailing period (e.g. "1.", "4.1.").
- Annex / Annexure sections: "Annex A", "Annexure - I", "ANNEXURE II".

Anything that doesn't match a recognised heading pattern is treated as a
CONTINUATION of the current clause's content — never as a new clause. If
no heading has been seen yet, leading text is captured under an implicit
"PREAMBLE" pseudo-clause rather than being silently dropped or invented
as a fake numbered clause.

This parser is intentionally conservative, per the project's ingestion
rules: it will not pretend arbitrary paragraphs are clauses when the
structure cannot be reliably determined. Extend the regexes here as more
BIS document styles are ingested — keep each style's logic easy to
isolate and test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.bis_ingestion import RawClause
from app.services.normalizer import normalize_text

# Numbered clause / paragraph heading, e.g. "1 Scope", "1.1 General",
# "4.1.2 Requirements", "1. Some text starts here".
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")

# Annex / Annexure heading, e.g. "Annex A", "Annexure - I", "ANNEXURE II:".
_ANNEX_RE = re.compile(
    r"^Annex(?:ure)?\s*[-\u2013\u2014:]?\s*([IVXLCDM]+|[A-Za-z0-9]+)\s*[-\u2013\u2014:]?\s*(.*)$",
    re.IGNORECASE,
)

PREAMBLE_CLAUSE_NUMBER = "PREAMBLE"

CLAUSE_TYPE_CLAUSE = "CLAUSE"
CLAUSE_TYPE_ANNEX = "ANNEX"
CLAUSE_TYPE_PREAMBLE = "PREAMBLE"


@dataclass
class _OpenClause:
    """Mutable accumulator for the clause currently being built."""

    clause_number: str
    title: str | None
    page_number: int
    clause_type: str
    content_lines: list[str]


def parse_pages_to_clauses(
    pages: list[tuple[int, str]],
    language: str | None = None,
) -> list[RawClause]:
    """
    Parse extracted page text into a flat, ordered list of RawClause objects.

    Args:
        pages: output of ``pdf_extractor.extract_pages`` — a list of
            (page_number, page_text) tuples in reading order.
        language: optional ISO 639-1 code to stamp onto every clause
            (e.g. "en"). Left as None if unknown.

    Returns:
        A list of RawClause, in document order. Content is passed through
        normalize_text() before being stored — whitespace only, meaning
        is preserved.
    """
    results: list[RawClause] = []
    current: _OpenClause | None = None

    def flush_current() -> None:
        if current is None:
            return
        content = normalize_text("\n".join(current.content_lines))
        if not content and not current.title:
            return  # nothing meaningful captured — don't emit an empty clause
        results.append(
            RawClause(
                clause_number=current.clause_number,
                title=current.title,
                content=content,
                page_number=current.page_number,
                language=language,
                clause_type=current.clause_type,
            )
        )

    for page_number, page_text in pages:
        if not page_text:
            continue

        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            annex_match = _ANNEX_RE.match(line)
            numbered_match = None if annex_match else _NUMBERED_RE.match(line)

            if annex_match:
                flush_current()
                label = annex_match.group(1)
                rest = annex_match.group(2).strip()
                current = _OpenClause(
                    clause_number=f"Annexure {label}",
                    title=rest or None,
                    page_number=page_number,
                    clause_type=CLAUSE_TYPE_ANNEX,
                    content_lines=[],
                )
            elif numbered_match:
                flush_current()
                number = numbered_match.group(1)
                rest = numbered_match.group(2).strip()
                current = _OpenClause(
                    clause_number=number,
                    title=None,
                    page_number=page_number,
                    clause_type=CLAUSE_TYPE_CLAUSE,
                    content_lines=[rest] if rest else [],
                )
            else:
                if current is None:
                    current = _OpenClause(
                        clause_number=PREAMBLE_CLAUSE_NUMBER,
                        title=None,
                        page_number=page_number,
                        clause_type=CLAUSE_TYPE_PREAMBLE,
                        content_lines=[],
                    )
                current.content_lines.append(line)

    flush_current()
    return results
