"""
BIS retrieval layer (Milestone 1).

This module searches the ingested Standard / Document / Clause tables and
returns structured hits with provenance. It does NOT generate answers, and
it does not use embeddings.

How it works today
------------------
``BISRetrievalService.search()`` is the public interface. It delegates to a
``RetrievalBackend``. The default backend is ``KeywordRetrievalBackend``,
which matches query tokens against existing text columns with ILIKE and
scores hits in Python.

How embeddings slot in later (Milestone 2)
------------------------------------------
pgvector is NOT available on the current ``postgres:16`` Docker image
(confirmed: only ``plpgsql`` is installed; ``vector`` is not even in
``pg_available_extensions``). We therefore do not add vector columns or
change Docker in this milestone.

When pgvector is added, implement another class with the same
``RetrievalBackend.search(...)`` method (e.g. ``VectorRetrievalBackend``)
and pass it into ``BISRetrievalService(session, backend=...)``. Callers of
``search()`` do not need to change.

Provenance
----------
Clause hits are always loaded via Document → Standard joins, so a clause
result keeps the Standard → Document → Clause chain. Document hits include
their parent Standard when ``documents.standard_id`` is set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, joinedload

from app.models.clause import Clause
from app.models.document import Document
from app.models.standard import Standard

# Common function words. "is" is intentionally NOT in this list so a query
# like "IS 302" still keeps the "IS" token.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "are",
        "was",
        "were",
        "be",
        "been",
        "what",
        "which",
        "how",
        "does",
        "do",
        "with",
        "from",
        "by",
        "this",
        "that",
        "these",
        "those",
        "can",
        "could",
        "should",
        "would",
        "about",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Weights for keyword scoring. Exact identity fields rank above body text.
_WEIGHTS = {
    "is_number": 10.0,
    "standard_title": 5.0,
    "standard_scope": 2.0,
    "document_title": 4.0,
    "clause_number": 8.0,
    "clause_title": 3.0,
    "clause_text": 1.0,
}

DEFAULT_LIMIT = 20


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass
class RelevanceInfo:
    """How this hit was ranked. ``method`` is 'keyword' today; Milestone 2 may use 'vector'."""

    score: float
    method: str
    matched_fields: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """
    One retrieval hit. Clause-level hits fill the whole Standard → Document →
    Clause chain. Standard- or document-only hits leave the unused ids/text
    as None rather than inventing values.
    """

    standard_id: Optional[int] = None
    standard_number: Optional[str] = None
    standard_title: Optional[str] = None
    document_id: Optional[int] = None
    document_title: Optional[str] = None
    clause_id: Optional[int] = None
    clause_number: Optional[str] = None
    clause_type: Optional[str] = None
    clause_text: Optional[str] = None
    page_number: Optional[int] = None
    source_url: Optional[str] = None
    relevance: RelevanceInfo = field(
        default_factory=lambda: RelevanceInfo(score=0.0, method="keyword")
    )


@dataclass
class SearchFilters:
    """Optional metadata filters. Only fields that exist in the current schema."""

    is_number: Optional[str] = None
    year: Optional[int] = None
    document_id: Optional[int] = None
    language: Optional[str] = None
    clause_type: Optional[str] = None


class RetrievalBackend(Protocol):
    """
    Swappable search strategy.

    Milestone 1 ships ``KeywordRetrievalBackend``. Milestone 2 can add a
    vector backend with this same method signature.
    """

    def search(
        self,
        query: str,
        filters: SearchFilters,
        limit: int,
    ) -> list[RetrievalResult]:
        ...


# ---------------------------------------------------------------------------
# Tokenisation / scoring helpers (pure Python, easy to unit-test)
# ---------------------------------------------------------------------------

def tokenize_query(query: str) -> list[str]:
    """Split a query into lowercase alphanumeric tokens, dropping stopwords."""
    if not query or not query.strip():
        return []
    tokens = [t.lower() for t in _TOKEN_RE.findall(query)]
    meaningful = [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]
    return meaningful


def _contains(text: Optional[str], term: str) -> bool:
    if not text:
        return False
    return term in text.lower()


def _ilike(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _score_fields(field_values: dict[str, Optional[str]], terms: list[str]) -> tuple[float, list[str]]:
    score = 0.0
    matched: list[str] = []
    for field_name, value in field_values.items():
        hits = sum(1 for term in terms if _contains(value, term))
        if hits:
            score += _WEIGHTS.get(field_name, 1.0) * hits
            matched.append(field_name)
    return score, matched


# ---------------------------------------------------------------------------
# Keyword backend
# ---------------------------------------------------------------------------

class KeywordRetrievalBackend:
    """ILIKE token matching over Standard, Document, and Clause text columns."""

    method = "keyword"

    def __init__(self, session: Session):
        self.session = session

    def search(
        self,
        query: str,
        filters: SearchFilters,
        limit: int,
    ) -> list[RetrievalResult]:
        terms = tokenize_query(query)
        if not terms:
            return []

        hits: list[RetrievalResult] = []
        hits.extend(self._search_standards(terms, filters))
        hits.extend(self._search_documents(terms, filters))
        hits.extend(self._search_clauses(terms, filters))

        hits.sort(key=lambda h: h.relevance.score, reverse=True)
        return hits[:limit]

    def _standard_filter_clauses(self, filters: SearchFilters):
        clauses = []
        if filters.is_number:
            clauses.append(Standard.is_number.ilike(_ilike(filters.is_number), escape="\\"))
        if filters.year is not None:
            clauses.append(Standard.year == filters.year)
        return clauses

    def _search_standards(self, terms: list[str], filters: SearchFilters) -> list[RetrievalResult]:
        if filters.document_id is not None or filters.clause_type is not None:
            return []  # these filters only apply to document/clause hits

        stmt = select(Standard)
        for clause in self._standard_filter_clauses(filters):
            stmt = stmt.where(clause)
        if filters.language:
            # Standards have no language column; skip them under a language filter.
            return []

        term_match = or_(
            *[Standard.is_number.ilike(_ilike(t), escape="\\") for t in terms],
            *[Standard.title.ilike(_ilike(t), escape="\\") for t in terms],
            *[Standard.scope.ilike(_ilike(t), escape="\\") for t in terms],
        )
        stmt = stmt.where(term_match)

        results: list[RetrievalResult] = []
        for standard in self.session.execute(stmt).scalars():
            score, matched = _score_fields(
                {
                    "is_number": standard.is_number,
                    "standard_title": standard.title,
                    "standard_scope": standard.scope,
                },
                terms,
            )
            if score <= 0:
                continue
            results.append(
                RetrievalResult(
                    standard_id=standard.id,
                    standard_number=standard.is_number,
                    standard_title=standard.title,
                    source_url=standard.source_url,
                    relevance=RelevanceInfo(
                        score=score, method=self.method, matched_fields=matched
                    ),
                )
            )
        return results

    def _search_documents(self, terms: list[str], filters: SearchFilters) -> list[RetrievalResult]:
        if filters.clause_type is not None:
            return []

        stmt = (
            select(Document)
            .outerjoin(Standard, Document.standard_id == Standard.id)
            .options(joinedload(Document.standard))
        )
        for clause in self._standard_filter_clauses(filters):
            stmt = stmt.where(clause)
        if filters.document_id is not None:
            stmt = stmt.where(Document.id == filters.document_id)
        if filters.language:
            stmt = stmt.where(Document.language == filters.language)

        term_match = or_(*[Document.title.ilike(_ilike(t), escape="\\") for t in terms])
        stmt = stmt.where(term_match)

        results: list[RetrievalResult] = []
        for document in self.session.execute(stmt).unique().scalars():
            standard = document.standard
            score, matched = _score_fields({"document_title": document.title}, terms)
            if score <= 0:
                continue
            results.append(
                RetrievalResult(
                    standard_id=standard.id if standard is not None else None,
                    standard_number=standard.is_number if standard is not None else None,
                    standard_title=standard.title if standard is not None else None,
                    document_id=document.id,
                    document_title=document.title,
                    source_url=document.source_url or (
                        standard.source_url if standard is not None else None
                    ),
                    relevance=RelevanceInfo(
                        score=score, method=self.method, matched_fields=matched
                    ),
                )
            )
        return results

    def _search_clauses(self, terms: list[str], filters: SearchFilters) -> list[RetrievalResult]:
        stmt = (
            select(Clause)
            .join(Document, Clause.document_id == Document.id)
            .outerjoin(Standard, Document.standard_id == Standard.id)
            .options(joinedload(Clause.document).joinedload(Document.standard))
        )
        for clause in self._standard_filter_clauses(filters):
            stmt = stmt.where(clause)
        if filters.document_id is not None:
            stmt = stmt.where(Document.id == filters.document_id)
        if filters.language:
            stmt = stmt.where(
                or_(Clause.language == filters.language, Document.language == filters.language)
            )
        if filters.clause_type:
            stmt = stmt.where(Clause.clause_type == filters.clause_type)

        term_match = or_(
            *[Clause.clause_number.ilike(_ilike(t), escape="\\") for t in terms],
            *[Clause.title.ilike(_ilike(t), escape="\\") for t in terms],
            *[Clause.content.ilike(_ilike(t), escape="\\") for t in terms],
        )
        stmt = stmt.where(term_match)

        results: list[RetrievalResult] = []
        for clause_row in self.session.execute(stmt).unique().scalars():
            document = clause_row.document
            standard = document.standard if document is not None else None
            score, matched = _score_fields(
                {
                    "clause_number": clause_row.clause_number,
                    "clause_title": clause_row.title,
                    "clause_text": clause_row.content,
                },
                terms,
            )
            if score <= 0:
                continue
            source_url = None
            if document is not None and document.source_url:
                source_url = document.source_url
            elif standard is not None:
                source_url = standard.source_url
            results.append(
                RetrievalResult(
                    standard_id=standard.id if standard is not None else None,
                    standard_number=standard.is_number if standard is not None else None,
                    standard_title=standard.title if standard is not None else None,
                    document_id=document.id if document is not None else None,
                    document_title=document.title if document is not None else None,
                    clause_id=clause_row.id,
                    clause_number=clause_row.clause_number,
                    clause_type=clause_row.clause_type,
                    clause_text=clause_row.content,
                    page_number=clause_row.page_number,
                    source_url=source_url,
                    relevance=RelevanceInfo(
                        score=score, method=self.method, matched_fields=matched
                    ),
                )
            )
        return results


# ---------------------------------------------------------------------------
# Public service
# ---------------------------------------------------------------------------

class BISRetrievalService:
    """
    Search ingested BIS data.

    ``search()`` is the stable API. Swap ``backend`` to change keyword
    matching for vector search later without rewriting callers.
    """

    def __init__(
        self,
        session: Session,
        backend: Optional[RetrievalBackend] = None,
    ):
        self.session = session
        self.backend: RetrievalBackend = backend or KeywordRetrievalBackend(session)

    def search(
        self,
        query: str,
        *,
        is_number: Optional[str] = None,
        year: Optional[int] = None,
        document_id: Optional[int] = None,
        language: Optional[str] = None,
        clause_type: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[RetrievalResult]:
        """
        Return ranked retrieval hits for ``query``.

        Filters are ANDed with the keyword match and only use columns that
        already exist. Unknown / unset fields on a hit are left as None.
        """
        filters = SearchFilters(
            is_number=is_number,
            year=year,
            document_id=document_id,
            language=language,
            clause_type=clause_type,
        )
        return self.backend.search(query, filters, limit)
