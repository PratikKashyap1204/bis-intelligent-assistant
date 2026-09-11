"""
Text normalization utilities.

Cleans obvious PDF-extraction artifacts — excessive whitespace, repeated
blank lines, trailing spaces — WITHOUT altering the meaning of the text.
This is intentionally conservative: it does not reflow paragraphs, fix
hyphenation, change casing, or touch punctuation.

The raw extracted text/PDF is never overwritten by this module. Callers
(clause_parser.py, fetcher.py) keep the original raw content separately
so normalization can always be re-run or improved without losing source
fidelity.
"""

from __future__ import annotations

import re

_TRAILING_WHITESPACE_RE = re.compile(r"[ \t]+\n")
_MULTIPLE_SPACES_RE = re.compile(r"[ \t]{2,}")
_MULTIPLE_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """
    Collapse excessive whitespace/blank lines and strip trailing spaces.

    Safe to call on already-clean text (idempotent).
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WHITESPACE_RE.sub("\n", text)
    text = _MULTIPLE_SPACES_RE.sub(" ", text)
    text = _MULTIPLE_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Milestone 5: line-level filters for official Gazette of India notifications
# ---------------------------------------------------------------------------
#
# Several real BIS Quality Control Order (QCO) documents are published as
# bilingual Gazette of India notifications: the Hindi text of the order is
# printed first, followed by an English rendering of the *same* legal
# order, interleaved with standard Gazette running heads/footers (registry
# number, "PUBLISHED BY AUTHORITY", page/issue footers, etc.).
#
# Feeding that raw bilingual text straight into clause_parser.py (which
# assumes one declared ``language`` per document) would either mislabel
# Hindi clause text as English, or cause the same legal clause number
# (e.g. "1", "2", "3", "4") to appear twice in one document purely because
# it is printed once in each language — on top of the parser occasionally
# misreading Gazette page furniture (e.g. "1490 GI/2024 (1)") as a new
# numbered clause.
#
# These two filters remove that boilerplate/other-language noise BEFORE
# clause parsing. They are line-level and conservative: they only drop
# lines that are (a) written in the Devanagari script, or (b) match a
# small set of well-known, GENERIC Gazette-of-India formatting patterns
# that recur on every page of any Gazette of India notification (not
# specific to any single ingested document). They never rewrite, merge,
# or reorder the remaining lines, and never invent clause structure.

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Generic Gazette of India running-head / footer / registry patterns.
# Verified against two independently published 2024 BIS QCO Gazette
# notifications (Resin Treated Compressed Wood Laminates QCO 2024 and
# Self-Contained Drinking Water Cooler QCO 2024) — these are standard
# Government-of-India Gazette formatting elements, not tuned to either
# document's specific content.
_GAZETTE_BOILERPLATE_PATTERNS = (
    re.compile(r"^CG-", re.IGNORECASE),                       # digital signature code
    re.compile(r"^xxxGIDExxx$", re.IGNORECASE),                # digital signature marker
    re.compile(r"^EXTRAORDINARY$", re.IGNORECASE),              # Gazette part label
    re.compile(r"^PART\s+[IVXLCDM]+[—–\-]", re.IGNORECASE),     # "PART II—Section 3..."
    re.compile(r"^PUBLISHED BY AUTHORITY$", re.IGNORECASE),
    re.compile(r"^No\.\s*\d+\]", re.IGNORECASE),                # Gazette issue number line
    re.compile(r"^\d+\s*GI\s*/\s*\d{4}\s*\(\d+\)$", re.IGNORECASE),  # page footer, e.g. "1490 GI/2024 (1)"
    re.compile(r"THE GAZETTE OF INDIA", re.IGNORECASE),         # running head, any position
    re.compile(r"^REGD\.?\s*No", re.IGNORECASE),                # registry number line
)


def strip_devanagari_lines(text: str) -> str:
    """
    Drop every line that contains at least one Devanagari character.

    Used only for bilingual (Hindi + English) Gazette notifications where
    the Hindi text is a translation of the same legal order printed
    elsewhere on the page in English — not additional information. Never
    used on the Milestone 1 pilot document (which is English-only).
    """
    if not text:
        return ""
    return "\n".join(
        line for line in text.splitlines() if not _DEVANAGARI_RE.search(line)
    )


def strip_gazette_boilerplate_lines(text: str) -> str:
    """
    Drop lines matching known Gazette-of-India running-head/footer
    patterns (see ``_GAZETTE_BOILERPLATE_PATTERNS``). Content lines
    (including numbered clauses and table rows) are never touched.
    """
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(pattern.search(stripped) for pattern in _GAZETTE_BOILERPLATE_PATTERNS):
            continue
        lines.append(line)
    return "\n".join(lines)
