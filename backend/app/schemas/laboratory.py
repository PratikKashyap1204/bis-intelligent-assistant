"""Pydantic schemas for the Laboratory model."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LaboratoryCreate(BaseModel):
    """Fields required to create a new Laboratory record."""

    name: str = Field(..., examples=["National Test House, Kolkata"])
    lab_code: str = Field(..., examples=["NTH/KOL/001"])
    address: Optional[str] = None
    state: Optional[str] = Field(None, examples=["West Bengal"])
    district: Optional[str] = Field(None, examples=["Kolkata"])
    source_url: Optional[str] = None
    validity_date: Optional[date] = None


class LaboratoryRead(BaseModel):
    """Full Laboratory representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    lab_code: str
    address: Optional[str]
    state: Optional[str]
    district: Optional[str]
    source_url: Optional[str]
    validity_date: Optional[date]
