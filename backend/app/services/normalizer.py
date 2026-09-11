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
