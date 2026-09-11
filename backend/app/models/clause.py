"""Clause model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.document import Document


class Clause(Base):
    """A numbered clause extracted from a BIS document."""

    __tablename__ = "clauses"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )

    # e.g. "3", "4.1", "4.1.2", "Annex A"
    clause_number: Mapped[str] = mapped_column(String(50), nullable=False)

    # Short heading of the clause
    title: Mapped[Optional[str]] = mapped_column(String(500))

    # Full text content of the clause
    content: Mapped[Optional[str]] = mapped_column(Text)

    # Page number in the source PDF (useful for citation)
    page_number: Mapped[Optional[int]] = mapped_column(Integer)

    # ISO 639-1 code of the clause's language, e.g. "en", "hi".
    language: Mapped[Optional[str]] = mapped_column(String(10))

    # Coarse classification set by the clause parser, e.g. "CLAUSE",
    # "ANNEX", "PREAMBLE" — helps later retrieval/ranking without needing
    # a schema change for every new document style.
    clause_type: Mapped[Optional[str]] = mapped_column(String(50))

    # ------------------------------------------------------------------
    # Milestone 5: additive disambiguation metadata (NOT an identity
    # column). ``id`` (the primary key above) remains the sole stable
    # identity for a clause. ``clause_number`` is not globally unique
    # within a document — real BIS/QCO documents reuse plain numbers
    # ("1", "2", ...) across independently-numbered lists, clauses, and
    # tables (observed empirically in the Milestone 1 pilot QCO circular:
    # clause_number "1" appears 5 times in that one document). This
    # column records the 1-based position of this clause among all
    # clauses parsed from the SAME document, in document order, purely
    # as extra provenance to disambiguate repeated clause_number values
    # for humans/tooling (e.g. the retrieval evaluation dataset) — it is
    # never used as a join key or primary identity.
    # ------------------------------------------------------------------
    sequence_in_document: Mapped[Optional[int]] = mapped_column(Integer)

    # relationships
    document: Mapped[Document] = relationship(back_populates="clauses")

    __table_args__ = (
        Index("ix_clauses_document_id", "document_id"),
    )
