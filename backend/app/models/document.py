"""Document model."""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import BigInteger, Date, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.clause import Clause
    from app.models.standard import Standard


class DocumentType(str, enum.Enum):
    """Classification of a BIS-related document."""

    STANDARD = "STANDARD"
    PRODUCT_MANUAL = "PRODUCT_MANUAL"
    AMENDMENT = "AMENDMENT"
    QCO = "QCO"
    SCHEME = "SCHEME"
    FAQ = "FAQ"
    OTHER = "OTHER"


class Document(Base):
    """
    A BIS-related document that may or may not belong to a specific standard.

    ``standard_id`` is nullable because some documents (e.g. QCOs that span
    multiple standards, or general FAQs) are not tied to a single IS number.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Nullable FK — some documents are not tied to a single standard
    standard_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("standards.id", ondelete="SET NULL"), index=True
    )

    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(String(1000), nullable=False)

    # Where this document was found / can be downloaded from
    source_url: Mapped[Optional[str]] = mapped_column(String(2000))

    # Local path after downloading (populated by the ingestion pipeline)
    local_file_path: Mapped[Optional[str]] = mapped_column(Text)

    published_date: Mapped[Optional[date]] = mapped_column(Date)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ------------------------------------------------------------------
    # Provenance / dedup metadata (added for the ingestion pipeline)
    # ------------------------------------------------------------------
    # ISO 639-1 code of the document's language, e.g. "en", "hi".
    language: Mapped[Optional[str]] = mapped_column(String(10))
    # SHA-256 hex digest of the raw downloaded file — used to detect that
    # the exact same content was already ingested, even under a different
    # source_url (e.g. mirrored copies).
    content_hash: Mapped[Optional[str]] = mapped_column(String(64))
    # For AMENDMENT-type documents: the amendment number, e.g. "Amendment No. 1".
    amendment_number: Mapped[Optional[str]] = mapped_column(String(50))

    # relationships
    standard: Mapped[Optional[Standard]] = relationship(back_populates="documents")
    clauses: Mapped[List[Clause]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_documents_document_type", "document_type"),
        Index("ix_documents_content_hash", "content_hash"),
    )
