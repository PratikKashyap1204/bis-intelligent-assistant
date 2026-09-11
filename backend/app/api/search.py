"""
Search/answer API routes (Milestone 3).

POST /api/search/answer — minimal RAG endpoint. Intentionally does NOT
include authentication, streaming, chat history, or anything beyond a
single grounded question -> answer call, per the Milestone 3 brief.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.rag import AnswerRequest, AnswerResponse, CitationRead, SourceRead
from app.services.answer_generation import AnswerGenerationProvider, get_default_answer_provider
from app.services.embedding_provider import EmbeddingProvider, get_default_embedding_provider
from app.services.rag import RAGService
from app.services.retrieval import BISRetrievalService, VectorRetrievalBackend

router = APIRouter(prefix="/api/search", tags=["search"])

# The real embedding provider lazily loads a sentence-transformers model
# (~10s cold start) — cached as a module-level singleton so it is loaded
# at most once per process, not once per request. Stateless/thread-safe
# to reuse: it only wraps read-only model inference calls.
_vector_provider: Optional[EmbeddingProvider] = None

# The answer provider (extractive or LLM, per ANSWER_PROVIDER — see
# app/config.py and app/services/answer_generation.py) is likewise cached
# per-process: for the LLM provider this avoids re-reading configuration
# on every request; for the extractive provider it's stateless anyway.
_answer_provider: Optional[AnswerGenerationProvider] = None


def _get_vector_provider() -> EmbeddingProvider:
    global _vector_provider
    if _vector_provider is None:
        _vector_provider = get_default_embedding_provider()
    return _vector_provider


def _get_answer_provider() -> AnswerGenerationProvider:
    global _answer_provider
    if _answer_provider is None:
        _answer_provider = get_default_answer_provider()
    return _answer_provider


def _build_rag_service(db: Session) -> RAGService:
    vector_provider = _get_vector_provider()
    retrieval_service = BISRetrievalService(
        db, backends={"vector": VectorRetrievalBackend(db, vector_provider)}
    )
    return RAGService(db, retrieval_service, _get_answer_provider())


@router.post("/answer", response_model=AnswerResponse)
def answer_question(request: AnswerRequest, db: Session = Depends(get_db)) -> AnswerResponse:
    rag_service = _build_rag_service(db)
    result = rag_service.answer(
        request.query,
        method=request.method,
        is_number=request.is_number,
        clause_type=request.clause_type,
    )

    return AnswerResponse(
        answer=result.answer,
        citations=[CitationRead(**c.__dict__) for c in result.citations],
        retrieval_method=result.retrieval_method,
        sources=[
            SourceRead(
                standard_number=s.standard_number,
                standard_title=s.standard_title,
                document_title=s.document_title,
                document_type=s.document_type,
                clause_number=s.clause_number,
                clause_type=s.clause_type,
                clause_text=s.clause_text,
                page_number=s.page_number,
                source_url=s.source_url,
                relevance_score=s.relevance.score,
                relevance_method=s.relevance.method,
            )
            for s in result.sources
        ],
        grounded=result.grounded,
        context_used=result.context_used,
    )
