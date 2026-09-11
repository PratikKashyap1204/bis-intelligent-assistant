"""Pydantic schemas for the Document model."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocumentType


class DocumentCreate(BaseModel):
    """Fields required to create a new Document record."""

    title: str = Field(..., examples=["IS 302:2008 — Household Electric Irons"])
    document_type: DocumentType
    standard_id: Optional[int] = None
    source_url: Optional[str] = None
    local_file_path: Optional[str] = None
    published_date: Optional[date] = None


class DocumentRead(BaseModel):
    """Full Document representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    document_type: str
    standard_id: Optional[int]
    source_url: Optional[str]
    local_file_path: Optional[str]
    published_date: Optional[date]
    created_at: datetime
