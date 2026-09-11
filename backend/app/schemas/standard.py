"""Pydantic schemas for the Standard model."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.standard import StandardStatus


class StandardCreate(BaseModel):
    """Fields required to create a new Standard record."""

    is_number: str = Field(..., examples=["IS 302", "IS 1998"])
    title: str = Field(..., examples=["Specification for Household Electric Irons"])
    year: Optional[int] = Field(None, examples=[2003])
    revision: Optional[str] = Field(None, examples=["First Revision"])
    status: StandardStatus = StandardStatus.ACTIVE
    scope: Optional[str] = None
    source_url: Optional[str] = None


class StandardUpdate(BaseModel):
    """Fields that can be updated on an existing Standard."""

    title: Optional[str] = None
    year: Optional[int] = None
    revision: Optional[str] = None
    status: Optional[StandardStatus] = None
    scope: Optional[str] = None
    source_url: Optional[str] = None


class StandardRead(BaseModel):
    """Full Standard representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    is_number: str
    title: str
    year: Optional[int]
    revision: Optional[str]
    status: str
    scope: Optional[str]
    source_url: Optional[str]
    created_at: datetime
    updated_at: datetime
