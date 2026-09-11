"""Pydantic schemas for the RAG answer-generation API (Milestone 3)."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


MAX_QUERY_LENGTH = 2000


class AnswerRequest(BaseModel):
    """Request body for POST /api/search/answer."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        examples=["Which appliances require mandatory certification before sale?"],
    )
    method: Optional[str] = Field(
        None,
        description=(
            "Retrieval method: 'vector' (default, semantic), 'keyword', "
            "or 'hybrid' (RRF of keyword + vector). Unset uses RAG's default."
        ),
        examples=["vector"],
    )
    is_number: Optional[str] = Field(None, description="Optional filter: BIS IS number.")
    clause_type: Optional[str] = Field(None, description="Optional filter: e.g. 'CLAUSE', 'ANNEX'.")

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class CitationRead(BaseModel):
    """One structured citation, built directly from a retrieval result."""

    index: int
    standard_number: Optional[str] = None
    standard_title: Optional[str] = None
    document_title: Optional[str] = None
    document_type: Optional[str] = None
    clause_number: Optional[str] = None
    clause_type: Optional[str] = None
    page_number: Optional[int] = None
    source_url: Optional[str] = None
    relevance_score: float
    clause_id: Optional[int] = None
    document_id: Optional[int] = None


class SourceRead(BaseModel):
    """One raw retrieval result used (or considered) as context, for transparency."""

    standard_number: Optional[str] = None
    standard_title: Optional[str] = None
    document_title: Optional[str] = None
    document_type: Optional[str] = None
    clause_number: Optional[str] = None
    clause_type: Optional[str] = None
    clause_text: Optional[str] = None
    page_number: Optional[int] = None
    source_url: Optional[str] = None
    relevance_score: float
    relevance_method: str
    clause_id: Optional[int] = None
    document_id: Optional[int] = None


class AnswerResponse(BaseModel):
    """Response body for POST /api/search/answer."""

    answer: str
    citations: List[CitationRead]
    retrieval_method: str
    sources: List[SourceRead]
    grounded: bool
    context_used: int
    evidence_status: str = Field(
        default="insufficient",
        description=(
            "Categorical evidence label: supported, partially_supported, "
            "insufficient, or out_of_scope. This is not a calibrated numeric "
            "confidence score."
        ),
        examples=["supported"],
    )
