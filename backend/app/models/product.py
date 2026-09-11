"""Product model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import BigInteger, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.certification import CertificationRequirement


class Product(Base):
    """A product that may be governed by one or more BIS standards."""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)

    # JSONB stores product-specific structured attributes:
    # e.g. {"voltage": "230V", "wattage": "1000W", "material": "ABS plastic"}
    # JSONB is binary-stored in PostgreSQL: supports indexing and fast key queries.
    attributes: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # relationships
    certification_requirements: Mapped[List[CertificationRequirement]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
    )
