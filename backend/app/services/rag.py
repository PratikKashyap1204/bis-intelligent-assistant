"""
RAG (retrieval-augmented answer) service (Milestone 3).

User question -> BISRetrievalService.search() -> relevant BIS clauses ->
deterministic context assembly -> AnswerGenerationProvider -> grounded
answer + citations/provenance.

This module does NOT talk to any LLM or embedding model directly — it
only orchestrates the existing ``BISRetrievalService`` (Milestone 1/2,
unmodified interface) and an ``AnswerGenerationProvider`` (Milestone 3,
see app/services/answer_generation.py). Swapping either one (e.g. a
future real-LLM provider, or a hybrid retrieval backend) does not require
changing this file's public API.

Context selection (deterministic, configurable via RAGConfig)
---------------------------------------------------------------
1. Retrieve up to ``retrieval_limit`` raw results from BISRetrievalService
   (already ranked highest-relevance first by the backend).
2. Drop results below a per-method minimum relevance score (filters out
   near-random vector nearest-neighbours for off-topic questions — vector
   search always returns *something*, even for a completely unrelated
   query, so an absolute "no results" check alone is not enough).
3. Remove duplicate hits (same standard/document/clause).
4. Cap to at most ``max_context_items`` items, and cap each item's quoted
   text to ``max_chars_per_clause`` and the running total to
   ``max_total_context_chars`` — always keeping the highest-ranked items
   first and dropping lower-ranked ones once a cap is hit, never the
   reverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.services.answer_generation import AnswerGenerationProvider, ContextItem
from app.services.retrieval import BISRetrievalService, RetrievalResult

DEFAULT_METHOD = "vector"
# M7 RAG/context eval on the 26-case M5 dataset (extractive provider,
# retrieval_limit=10, max_context_items=5):
#   keyword context_recall=0.158  (standard/document hits occupy the window)
#   vector  context_recall=0.474
#   hybrid  context_recall=0.474
# Ungrounded rate on no-result cases was 0.143 for all three.
# Vector remains the default: keyword is worse at this operating point;
# hybrid is not better than vector and is worse on multi-clause retrieval.


@dataclass
class RAGConfig:
    """Tunable, deterministic context-selection limits. See module docstring."""

    retrieval_limit: int = 10
    max_context_items: int = 5
    max_chars_per_clause: int = 800
    max_total_context_chars: int = 4000

    # Minimum relevance score required to be considered "sufficiently
    # relevant" grounding context, per retrieval method. Keyword scores
    # are unbounded weighted term-match sums (any real match is > 0), so
    # 0.0 effectively means "any real keyword hit counts". Vector scores
    # are cosine similarity in [-1, 1]; nearest-neighbour search always
    # returns *some* result even for an unrelated query, so this cannot
    # be 0.0 — see scripts/rag_demo.py for the empirical measurement this
    # default (0.30) is based on: clearly relevant hits on this corpus
    # scored ~0.5-0.65, while a deliberately unrelated question's best
    # match scored well below this. This is a heuristic, not a true
    # confidence measure — see known limitations in the final report.
    min_score_keyword: float = 0.0
    min_score_vector: float = 0.30


@dataclass
class Citation:
    """
    Structured citation, built ONLY from a real retrieval result — never
    parsed out of generated answer text. See RAGService._build_citations.
    """

    index: int
    standard_number: Optional[str]
    standard_title: Optional[str]
    document_title: Optional[str]
    document_type: Optional[str]
    clause_number: Optional[str]
    clause_type: Optional[str]
    page_number: Optional[int]
    source_url: Optional[str]
    relevance_score: float
    clause_id: Optional[int] = None
    document_id: Optional[int] = None


@dataclass
class RAGAnswer:
    """Return value of RAGService.answer()."""

    answer: str
    citations: List[Citation]
    sources: List[RetrievalResult]
    retrieval_method: str
    grounded: bool
    context_used: int


def _dedup_key(result: RetrievalResult) -> Tuple:
    return (result.standard_id, result.document_id, result.clause_id, result.clause_number)


def _deduplicate(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """Remove exact-duplicate hits, keeping the first (highest-ranked) occurrence."""
    seen = set()
    deduped = []
    for r in results:
        key = _dedup_key(r)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def _passes_threshold(result: RetrievalResult, config: RAGConfig) -> bool:
    if result.relevance.method == "vector":
        return result.relevance.score >= config.min_score_vector
    return result.relevance.score >= config.min_score_keyword


def _select_context(
    results: List[RetrievalResult], config: RAGConfig
) -> List[RetrievalResult]:
    """
    Deterministic context selection: dedup -> threshold -> cap count ->
    cap total length. Input is assumed already sorted highest-relevance
    first (both KeywordRetrievalBackend and VectorRetrievalBackend return
    results in that order).
    """
    deduped = _deduplicate(results)
    relevant = [r for r in deduped if _passes_threshold(r, config)]

    selected: List[RetrievalResult] = []
    total_chars = 0
    for r in relevant:
        if len(selected) >= config.max_context_items:
            break
        text = (r.clause_text or "").strip()
        text_len = min(len(text), config.max_chars_per_clause)
        if selected and total_chars + text_len > config.max_total_context_chars:
            # Stop adding lower-ranked items once the total cap would be
            # exceeded — always keep what's already selected (highest
            # ranked) rather than dropping earlier items to make room.
            break
        selected.append(r)
        total_chars += text_len
    return selected


def _to_context_item(index: int, result: RetrievalResult, config: RAGConfig) -> ContextItem:
    text = (result.clause_text or "").strip()
    if len(text) > config.max_chars_per_clause:
        text = text[: config.max_chars_per_clause].rstrip() + "…"
    return ContextItem(
        index=index,
        standard_number=result.standard_number,
        standard_title=result.standard_title,
        document_title=result.document_title,
        document_type=result.document_type,
        clause_number=result.clause_number,
        clause_type=result.clause_type,
        clause_text=text,
        page_number=result.page_number,
        source_url=result.source_url,
        relevance_score=result.relevance.score,
        retrieval_method=result.relevance.method,
        clause_id=result.clause_id,
        document_id=result.document_id,
    )


def _to_citation(item: ContextItem) -> Citation:
    return Citation(
        index=item.index,
        standard_number=item.standard_number,
        standard_title=item.standard_title,
        document_title=item.document_title,
        document_type=item.document_type,
        clause_number=item.clause_number,
        clause_type=item.clause_type,
        page_number=item.page_number,
        source_url=item.source_url,
        relevance_score=item.relevance_score,
        clause_id=item.clause_id,
        document_id=item.document_id,
    )


class RAGService:
    """
    Orchestrates BISRetrievalService + AnswerGenerationProvider into one
    grounded-answer call. See module docstring for the full flow.
    """

    def __init__(
        self,
        session: Session,
        retrieval_service: BISRetrievalService,
        answer_provider: AnswerGenerationProvider,
        config: Optional[RAGConfig] = None,
    ):
        self.session = session
        self.retrieval_service = retrieval_service
        self.answer_provider = answer_provider
        self.config = config or RAGConfig()

    def answer(
        self,
        question: str,
        *,
        method: Optional[str] = None,
        is_number: Optional[str] = None,
        year: Optional[int] = None,
        document_id: Optional[int] = None,
        language: Optional[str] = None,
        clause_type: Optional[str] = None,
    ) -> RAGAnswer:
        resolved_method = method or DEFAULT_METHOD

        raw_results = self.retrieval_service.search(
            question,
            method=resolved_method,
            limit=self.config.retrieval_limit,
            is_number=is_number,
            year=year,
            document_id=document_id,
            language=language,
            clause_type=clause_type,
        )

        selected = _select_context(raw_results, self.config)
        context_items = [
            _to_context_item(i + 1, result, self.config) for i, result in enumerate(selected)
        ]

        generated = self.answer_provider.generate(question, context_items)
        citations = self._build_citations(generated.cited_indices, context_items)

        return RAGAnswer(
            answer=generated.answer_text,
            citations=citations,
            sources=selected,
            retrieval_method=resolved_method,
            grounded=generated.grounded,
            context_used=len(context_items),
        )

    @staticmethod
    def _build_citations(
        cited_indices: List[int], context_items: List[ContextItem]
    ) -> List[Citation]:
        """
        Build Citation objects strictly from the real ContextItem objects
        that were actually sent to the provider. Any index the provider
        returns that does NOT correspond to a real context item is
        silently dropped — a provider (mock, or a future LLM-backed one)
        cannot fabricate a citation this way; it can only ever point back
        to grounding context that genuinely came from the database.
        """
        by_index: Dict[int, ContextItem] = {item.index: item for item in context_items}
        citations = []
        for idx in cited_indices:
            item = by_index.get(idx)
            if item is None:
                continue  # defensively drop any index not in real context
            citations.append(_to_citation(item))
        return citations
