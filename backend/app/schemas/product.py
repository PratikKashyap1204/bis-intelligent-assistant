"""Pydantic schemas for the Product model."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProductCreate(BaseModel):
    """Fields required to create a new Product record."""

    name: str = Field(..., examples=["Electric Iron"])
    category: Optional[str] = Field(None, examples=["Electrical Appliances"])
    description: Optional[str] = None
    attributes: Optional[dict[str, Any]] = Field(
        None,
        examples=[{"voltage": "230V", "wattage": "1000W"}],
    )


class ProductUpdate(BaseModel):
    """Fields that can be updated on an existing Product."""

    name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    attributes: Optional[dict[str, Any]] = None


class ProductRead(BaseModel):
    """Full Product representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    category: Optional[str]
    description: Optional[str]
    attributes: Optional[dict[str, Any]]
    created_at: datetime
