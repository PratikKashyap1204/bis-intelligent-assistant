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
from typing import Dict, Optional, Protocol

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, joinedload

from app.models.clause import Clause
from app.models.document import Document
from app.models.embedding import ClauseEmbedding
from app.models.standard import Standard
from app.services.embedding_provider import EmbeddingProvider

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
        # Milestone 5: added after the retrieval evaluation showed
        # "Who won the FIFA World Cup in 2022?" (an out-of-scope
        # question with zero real relevance) matching clause_text on
        # BOTH real "Penalty for contravention... Any person WHO
        # contravenes..." clauses (ids 392, 400) purely via the token
        # "who" — a common English function word that was simply missing
        # from this list (unlike the other wh-words already above).
        "who",
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

# Milestone 5: terms tokenize_query() must still RETURN (so "IS 302" stays
# ["is", "302"] — see test_tokenize_query_drops_stopwords_keeps_is_number_tokens)
# but that carry ~zero discriminative signal on their own for SCORING.
# The retrieval evaluation showed the bare token "is" alone — present in
# nearly every natural-language English question — matching is_number/
# standard_title on EVERY standard (every is_number literally starts with
# "IS "), producing false-positive top hits (score 15-16) for completely
# unrelated questions (e.g. "What is the scope defined by IS 3513...",
# "What are the certification requirements under the Toys QCO?"). Unlike
# tokenize_query's stopword list (which controls what counts as a query
# term at all — kept permissive on purpose for "IS 302"-style queries),
# this set only controls what may contribute to a match SCORE.
_LOW_SIGNAL_SCORING_TERMS = frozenset({"is"})

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
    # e.g. "STANDARD", "QCO", "AMENDMENT" — see DocumentType. Added in
    # Milestone 3 so RAG citations can distinguish a BIS Standard from a
    # QCO/circular. Optional/defaulted so existing Milestone 1/2 callers
    # that construct RetrievalResult without it are unaffected.
    document_type: Optional[str] = None
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
    # See _LOW_SIGNAL_SCORING_TERMS: these terms are still valid query
    # tokens (e.g. for the SQL candidate-selection WHERE clause) but must
    # not, on their own, cause a field to be counted as "matched" or
    # contribute to the score — a real, higher-signal term still can.
    scoring_terms = [t for t in terms if t not in _LOW_SIGNAL_SCORING_TERMS]
    for field_name, value in field_values.items():
        hits = sum(1 for term in scoring_terms if _contains(value, term))
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
                    document_type=document.document_type,
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
                    document_type=document.document_type if document is not None else None,
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
# Vector backend (Milestone 2)
# ---------------------------------------------------------------------------

class VectorRetrievalBackend:
    """
    Semantic search over ClauseEmbedding rows using pgvector cosine distance.

    Implements the same RetrievalBackend contract as KeywordRetrievalBackend,
    so BISRetrievalService.search() does not need to change to use it.

    Only searches at clause granularity (embeddings are stored per-clause —
    see app/models/embedding.py). Standard/document-only hits are a keyword
    backend concept; vector search always returns clause-level hits with
    full Standard -> Document -> Clause provenance attached.
    """

    method = "vector"

    def __init__(self, session: Session, provider: EmbeddingProvider):
        self.session = session
        self.provider = provider

    def search(
        self,
        query: str,
        filters: SearchFilters,
        limit: int,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            return []

        query_vector = self.provider.generate(query)

        distance = ClauseEmbedding.embedding.cosine_distance(query_vector)
        stmt = (
            select(ClauseEmbedding, distance.label("distance"))
            .join(Clause, ClauseEmbedding.clause_id == Clause.id)
            .join(Document, ClauseEmbedding.document_id == Document.id)
            .outerjoin(Standard, ClauseEmbedding.standard_id == Standard.id)
            .where(ClauseEmbedding.model_name == self.provider.model_name)
            .options(
                joinedload(ClauseEmbedding.clause),
                joinedload(ClauseEmbedding.document),
                joinedload(ClauseEmbedding.standard),
            )
        )

        if filters.is_number:
            stmt = stmt.where(Standard.is_number.ilike(_ilike(filters.is_number), escape="\\"))
        if filters.year is not None:
            stmt = stmt.where(Standard.year == filters.year)
        if filters.document_id is not None:
            stmt = stmt.where(Document.id == filters.document_id)
        if filters.language:
            stmt = stmt.where(
                or_(Clause.language == filters.language, Document.language == filters.language)
            )
        if filters.clause_type:
            stmt = stmt.where(Clause.clause_type == filters.clause_type)

        stmt = stmt.order_by(distance).limit(limit)

        results: list[RetrievalResult] = []
        for row in self.session.execute(stmt).unique().all():
            emb: ClauseEmbedding = row[0]
            cosine_distance = float(row[1])
            similarity = 1.0 - cosine_distance  # pgvector cosine_distance == 1 - cosine_similarity

            clause = emb.clause
            document = emb.document
            standard = emb.standard

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
                    document_type=document.document_type if document is not None else None,
                    clause_id=clause.id if clause is not None else None,
                    clause_number=clause.clause_number if clause is not None else None,
                    clause_type=clause.clause_type if clause is not None else None,
                    clause_text=clause.content if clause is not None else None,
                    page_number=clause.page_number if clause is not None else None,
                    source_url=source_url,
                    relevance=RelevanceInfo(
                        score=similarity, method=self.method, matched_fields=["embedding"]
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

    ``search()`` is the stable API. The default backend is keyword search
    (unchanged from Milestone 1). Pass ``backends={"vector": VectorRetrievalBackend(...)}``
    and call ``search(..., method="vector")`` to use semantic search —
    existing callers that never pass ``method`` are unaffected.
    """

    def __init__(
        self,
        session: Session,
        backend: Optional[RetrievalBackend] = None,
        backends: Optional[Dict[str, RetrievalBackend]] = None,
    ):
        self.session = session
        self.backend: RetrievalBackend = backend or KeywordRetrievalBackend(session)
        self.backends: Dict[str, RetrievalBackend] = dict(backends or {})
        self.backends.setdefault("keyword", self.backend)

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
        method: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """
        Return ranked retrieval hits for ``query``.

        Filters are ANDed with the match and only use columns that already
        exist. Unknown / unset fields on a hit are left as None.

        ``method``: which registered backend to use ("keyword" is always
        registered; "vector" if one was supplied via ``backends``). Leaving
        it unset uses the default backend passed to ``__init__`` — existing
        callers from Milestone 1 do not need to change.
        """
        backend = self.backends[method] if method is not None else self.backend

        filters = SearchFilters(
            is_number=is_number,
            year=year,
            document_id=document_id,
            language=language,
            clause_type=clause_type,
        )
        return backend.search(query, filters, limit)
