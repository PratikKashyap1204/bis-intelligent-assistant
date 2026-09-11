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
from app.services.answer_generation import ExtractiveAnswerGenerationProvider
from app.services.embedding_provider import EmbeddingProvider, get_default_embedding_provider
from app.services.rag import RAGService
from app.services.retrieval import BISRetrievalService, VectorRetrievalBackend

router = APIRouter(prefix="/api/search", tags=["search"])

# The real embedding provider lazily loads a sentence-transformers model
# (~10s cold start) — cached as a module-level singleton so it is loaded
# at most once per process, not once per request. Stateless/thread-safe
# to reuse: it only wraps read-only model inference calls.
_vector_provider: Optional[EmbeddingProvider] = None


def _get_vector_provider() -> EmbeddingProvider:
    global _vector_provider
    if _vector_provider is None:
        _vector_provider = get_default_embedding_provider()
    return _vector_provider


def _build_rag_service(db: Session) -> RAGService:
    provider = _get_vector_provider()
    retrieval_service = BISRetrievalService(
        db, backends={"vector": VectorRetrievalBackend(db, provider)}
    )
    return RAGService(db, retrieval_service, ExtractiveAnswerGenerationProvider())


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
