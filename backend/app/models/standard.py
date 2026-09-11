"""Standard model."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.certification import CertificationRequirement
    from app.models.document import Document
    from app.models.hallmarking import HallmarkingRequirement
    from app.models.test_requirement import TestRequirement


class StandardStatus(str, enum.Enum):
    """Publication status of a BIS Indian Standard."""

    ACTIVE = "ACTIVE"
    WITHDRAWN = "WITHDRAWN"
    UNDER_REVISION = "UNDER_REVISION"
    SUPERSEDED = "SUPERSEDED"


class Standard(Base):
    """A BIS Indian Standard identified by its IS number."""

    __tablename__ = "standards"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # IS number as published by BIS: e.g. "IS 302", "IS 1998", "IS 9000 (Part 1)"
    is_number: Mapped[str] = mapped_column(String(100), nullable=False)

    title: Mapped[str] = mapped_column(String(1000), nullable=False)

    # Year of publication/last revision
    year: Mapped[Optional[int]] = mapped_column(Integer)

    # Human-readable revision label: "First Revision", "Second Revision", etc.
    revision: Mapped[Optional[str]] = mapped_column(String(100))

    # Stored as a plain string; Python-level validation uses StandardStatus enum.
    # Keeping it a String column avoids ALTER TYPE migrations when adding statuses.
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=StandardStatus.ACTIVE.value
    )

    # Official scope / abstract as written in the standard
    scope: Mapped[Optional[str]] = mapped_column(Text)

    # URL on BIS website where this standard is listed / purchasable
    source_url: Mapped[Optional[str]] = mapped_column(String(2000))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Extended metadata from BIS's official "Know Your Standard" portal.
    # Added for the first real ingestion pilot (IS 302 Part 1:2024) —
    # see app/services/bis_ingestion.py. All nullable: not every standard
    # page publishes every one of these fields, and older/simpler records
    # (or future standards from other sources) may not have them either.
    # Cross-references (e.g. "superseded by", "related standard") are
    # intentionally NOT columns here — use EntityRelationship instead so
    # we don't need a schema change every time a new relationship type
    # is needed.
    # ------------------------------------------------------------------
    technical_committee: Mapped[Optional[str]] = mapped_column(String(100))
    group: Mapped[Optional[str]] = mapped_column(String(255))
    sub_group: Mapped[Optional[str]] = mapped_column(String(255))
    sub_sub_group: Mapped[Optional[str]] = mapped_column(String(255))
    aspect: Mapped[Optional[str]] = mapped_column(String(255))
    certification_type: Mapped[Optional[str]] = mapped_column(String(100))
    relevant_ministries: Mapped[Optional[str]] = mapped_column(Text)
    reaffirmed_year: Mapped[Optional[int]] = mapped_column(Integer)
    equivalent_international_standard: Mapped[Optional[str]] = mapped_column(String(255))
    degree_of_equivalence: Mapped[Optional[str]] = mapped_column(String(255))
    harmonized_with: Mapped[Optional[str]] = mapped_column(String(100))
    price: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    num_amendments: Mapped[Optional[int]] = mapped_column(Integer)
    # Hindi text mirrors of title/scope, to support bilingual EN/HI answers.
    title_hi: Mapped[Optional[str]] = mapped_column(String(1000))
    scope_hi: Mapped[Optional[str]] = mapped_column(Text)

    # relationships
    documents: Mapped[List[Document]] = relationship(
        back_populates="standard", cascade="all, delete-orphan"
    )
    test_requirements: Mapped[List[TestRequirement]] = relationship(
        back_populates="standard", cascade="all, delete-orphan"
    )
    certification_requirements: Mapped[List[CertificationRequirement]] = relationship(
        back_populates="standard", cascade="all, delete-orphan"
    )
    hallmarking_requirements: Mapped[List[HallmarkingRequirement]] = relationship(
        back_populates="standard", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_standards_is_number", "is_number"),
        Index("ix_standards_status", "status"),
        # Natural identity for upsert during ingestion: the same IS number
        # published in the same year is the same Standard record.
        UniqueConstraint("is_number", "year", name="uq_standards_is_number_year"),
    )
