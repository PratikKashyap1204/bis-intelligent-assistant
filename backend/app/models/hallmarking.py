"""HallmarkingRequirement model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.standard import Standard


class HallmarkingRequirement(Base):
    """
    Hallmarking specification for a precious metal under a BIS standard.

    Each row describes one (metal, fineness) combination governed by a
    particular IS standard. BIS hallmarks are mandatory for jewellery
    under the BIS (Amendment) Act, 2022.
    """

    __tablename__ = "hallmarking_requirements"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    standard_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("standards.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # e.g. "GOLD", "SILVER", "PLATINUM"
    metal: Mapped[str] = mapped_column(String(50), nullable=False)

    # Purity specification: e.g. "999", "958", "916", "875", "750", "585"
    fineness: Mapped[Optional[str]] = mapped_column(String(50))

    # Description of the hallmarking process / steps required
    process: Mapped[Optional[str]] = mapped_column(Text)

    # BIS source page for this requirement
    source_url: Mapped[Optional[str]] = mapped_column(String(2000))

    # relationships
    standard: Mapped[Standard] = relationship(back_populates="hallmarking_requirements")
