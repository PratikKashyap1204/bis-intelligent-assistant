"""TestRequirement model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.standard import Standard


class TestRequirement(Base):
    """A named test requirement defined within a BIS standard."""

    __tablename__ = "test_requirements"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    standard_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("standards.id", ondelete="CASCADE"), nullable=False
    )

    # Clause reference within the standard, e.g. "5.1", "Table 2"
    clause: Mapped[Optional[str]] = mapped_column(String(100))

    # Short name of the test, e.g. "Insulation Resistance Test"
    test_name: Mapped[str] = mapped_column(String(500), nullable=False)

    # Description / criteria for the test
    description: Mapped[Optional[str]] = mapped_column(Text)

    # relationships
    standard: Mapped[Standard] = relationship(back_populates="test_requirements")

    __table_args__ = (
        Index("ix_test_requirements_standard_id", "standard_id"),
    )
