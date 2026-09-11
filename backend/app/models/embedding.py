"""
ClauseEmbedding model (Milestone 2).

One row per (clause, model_name): a pgvector embedding of that clause's
text, plus enough denormalized provenance (document_id, standard_id) to
support retrieval without extra joins, while the foreign keys remain the
source of truth for the Standard -> Document -> Clause chain.

Why clause-level, not document-level:
  Embedding an entire document as one vector would blur together many
  unrelated clauses (scope, annex tables, signatures, etc.) into a single
  vector, destroying the ability to cite a specific clause. Clauses are
  already BIS's own natural "chunk" boundary, so this milestone embeds at
  that granularity rather than introducing a separate re-chunking step.

Why keyed by (clause_id, model_name), not just clause_id:
  Allows a future model upgrade (e.g. switching to a multilingual model)
  to coexist with the old embeddings during a transition, rather than
  forcing an all-or-nothing cutover.

Change detection / re-embedding:
  ``content_hash`` is the SHA-256 of the exact text that was embedded.
  Before generating a new embedding, the embedding service compares this
  hash against the clause's current content — if unchanged, the existing
  row is left alone (no duplicate work, no duplicate row); if changed,
  the row is updated in place (see app/services/embedding_service.py).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.services.embedding_provider import DEFAULT_EMBEDDING_DIMENSION

if TYPE_CHECKING:
    from app.models.clause import Clause
    from app.models.document import Document
    from app.models.standard import Standard


class ClauseEmbedding(Base):
    """A vector embedding of one Clause's text, for a specific embedding model."""

    __tablename__ = "clause_embeddings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    clause_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("clauses.id", ondelete="CASCADE"), nullable=False
    )
    # Denormalized for filtering/joins without always going through Clause.
    # The FK relationship (via Clause.document) remains the source of truth.
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    standard_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("standards.id", ondelete="SET NULL")
    )

    # Which model produced this vector, and at what width — lets multiple
    # model versions coexist and makes a model upgrade explicit/detectable.
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_dimension: Mapped[int] = mapped_column(Integer, nullable=False)

    # SHA-256 of `embedding_text` — used to skip re-embedding unchanged
    # content and to detect when a clause's content has changed.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # The exact text that was embedded (kept for debugging/auditing —
    # normally equal to Clause.content at embedding time).
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)

    embedding: Mapped[list] = mapped_column(Vector(DEFAULT_EMBEDDING_DIMENSION), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # relationships
    clause: Mapped["Clause"] = relationship()
    document: Mapped["Document"] = relationship()
    standard: Mapped[Optional["Standard"]] = relationship()

    __table_args__ = (
        # One embedding per clause per model — re-embedding updates this
        # row in place instead of inserting a duplicate.
        UniqueConstraint("clause_id", "model_name", name="uq_clause_embeddings_clause_model"),
        Index("ix_clause_embeddings_document_id", "document_id"),
        Index("ix_clause_embeddings_standard_id", "standard_id"),
        # No ANN index (ivfflat/hnsw) yet: with 129 rows an exact sequential
        # scan over <=> is both fast and exact. Add one (e.g. ivfflat with
        # lists tuned to corpus size) once the corpus is large enough for
        # approximate search to matter — see Milestone 3 recommendation.
    )
