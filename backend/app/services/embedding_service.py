"""
Clause embedding generation service (Milestone 2).

Turns Clause rows into ClauseEmbedding rows via an EmbeddingProvider.
This is a controlled, on-demand operation over already-ingested clauses —
it never fetches/scrapes anything and never runs automatically; a
developer runs scripts/embed_pilot.py (or calls this service directly)
to (re)build embeddings for the current corpus.

Idempotency / change detection
-------------------------------
For each Clause considered:
  - No existing ClauseEmbedding row for (clause_id, model_name)  -> CREATE
  - Existing row, content_hash unchanged                          -> SKIP
  - Existing row, content_hash different (clause text changed)    -> UPDATE

This means re-running embedding generation on unchanged content does no
model inference and creates no duplicate rows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.clause import Clause
from app.models.document import Document
from app.models.embedding import ClauseEmbedding
from app.services.embedding_provider import EmbeddingProvider


def content_hash_of(text: str) -> str:
    """SHA-256 hex digest of the exact text that will be/was embedded."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class EmbeddingRunSummary:
    """Outcome counts for one embed_clauses() call."""

    created: int = 0
    updated: int = 0
    skipped_unchanged: int = 0
    skipped_empty: int = 0
    total_considered: int = 0
    clause_ids_created: List[int] = field(default_factory=list)
    clause_ids_updated: List[int] = field(default_factory=list)


class ClauseEmbeddingService:
    """Generates and persists ClauseEmbedding rows for Clause content."""

    def __init__(self, session: Session, provider: EmbeddingProvider):
        self.session = session
        self.provider = provider

    def embed_clauses(self, clauses: List[Clause], batch_size: int = 32) -> EmbeddingRunSummary:
        """
        Embed (or skip/update) each clause in ``clauses``.

        Only clauses that are new or whose content changed are actually
        sent to the embedding model — this is checked via content_hash
        BEFORE calling the (potentially slow) provider, so unchanged runs
        do zero model inference.
        """
        summary = EmbeddingRunSummary(total_considered=len(clauses))

        # Pre-load existing embeddings for these clauses in one query
        # (avoids one SELECT per clause).
        clause_ids = [c.id for c in clauses]
        existing_by_clause_id = {}
        if clause_ids:
            existing_rows = self.session.execute(
                select(ClauseEmbedding).where(
                    ClauseEmbedding.clause_id.in_(clause_ids),
                    ClauseEmbedding.model_name == self.provider.model_name,
                )
            ).scalars()
            existing_by_clause_id = {row.clause_id: row for row in existing_rows}

        to_embed: List[Clause] = []
        to_embed_hashes: List[str] = []

        for clause in clauses:
            text = (clause.content or "").strip()
            if not text:
                summary.skipped_empty += 1
                continue

            new_hash = content_hash_of(text)
            existing = existing_by_clause_id.get(clause.id)
            if existing is not None and existing.content_hash == new_hash:
                summary.skipped_unchanged += 1
                continue

            to_embed.append(clause)
            to_embed_hashes.append(new_hash)

        # Batch-generate only for clauses that actually need it.
        for start in range(0, len(to_embed), batch_size):
            batch = to_embed[start : start + batch_size]
            batch_hashes = to_embed_hashes[start : start + batch_size]
            texts = [(c.content or "").strip() for c in batch]
            vectors = self.provider.generate_batch(texts)

            for clause, text, new_hash, vector in zip(batch, texts, batch_hashes, vectors):
                existing = existing_by_clause_id.get(clause.id)
                if existing is not None:
                    existing.content_hash = new_hash
                    existing.embedding_text = text
                    existing.embedding = vector
                    existing.model_dimension = self.provider.dimension
                    summary.updated += 1
                    summary.clause_ids_updated.append(clause.id)
                else:
                    row = ClauseEmbedding(
                        clause_id=clause.id,
                        document_id=clause.document_id,
                        standard_id=(
                            clause.document.standard_id if clause.document is not None else None
                        ),
                        model_name=self.provider.model_name,
                        model_dimension=self.provider.dimension,
                        content_hash=new_hash,
                        embedding_text=text,
                        embedding=vector,
                    )
                    self.session.add(row)
                    summary.created += 1
                    summary.clause_ids_created.append(clause.id)

            self.session.flush()

        return summary

    def embed_all_clauses(
        self,
        standard_is_number: Optional[str] = None,
        batch_size: int = 32,
    ) -> EmbeddingRunSummary:
        """
        Embed every Clause currently in the database (optionally filtered
        to one standard's clauses via its IS number).
        """
        stmt = select(Clause).join(Document, Clause.document_id == Document.id)
        if standard_is_number:
            from app.models.standard import Standard

            stmt = stmt.join(Standard, Document.standard_id == Standard.id).where(
                Standard.is_number == standard_is_number
            )
        clauses = list(self.session.execute(stmt).scalars())
        return self.embed_clauses(clauses, batch_size=batch_size)
