"""Laboratory model."""

from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import BigInteger, Date, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Laboratory(Base):
    """A BIS-recognised testing laboratory."""

    __tablename__ = "laboratories"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(String(500), nullable=False)

    # BIS-assigned lab code, used as the natural key in BIS publications
    lab_code: Mapped[str] = mapped_column(String(100), nullable=False)

    address: Mapped[Optional[str]] = mapped_column(Text)
    state: Mapped[Optional[str]] = mapped_column(String(100))
    district: Mapped[Optional[str]] = mapped_column(String(100))

    # BIS page / gazette where this laboratory is listed
    source_url: Mapped[Optional[str]] = mapped_column(String(2000))

    # Date until which the laboratory's accreditation is valid
    validity_date: Mapped[Optional[date]] = mapped_column(Date)

    __table_args__ = (
        Index("ix_laboratories_lab_code", "lab_code"),
    )
