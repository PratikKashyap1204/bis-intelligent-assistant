"""EntityRelationship model.

A generic edge table that can link any two entities (standards, products,
documents, laboratories, etc.) with a named relationship type.

This avoids creating a separate join table for every possible pair of entities
and lets the graph of BIS knowledge evolve without schema migrations.

Example rows:
  (STANDARD, 42, RELATED_STANDARD,          STANDARD,  77)
  (PRODUCT,   5, GOVERNED_BY,               STANDARD,  42)
  (STANDARD, 42, HAS_AMENDMENT,             DOCUMENT,  18)
  (STANDARD, 42, TESTED_BY,                 LABORATORY, 3)
  (PRODUCT,   5, HAS_CERTIFICATION_REQUIREMENT, CERTIFICATION_REQUIREMENT, 9)
"""

from __future__ import annotations

import enum

from sqlalchemy import BigInteger, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RelationshipType(str, enum.Enum):
    """Named edge types in the BIS knowledge graph."""

    RELATED_STANDARD = "RELATED_STANDARD"
    GOVERNED_BY = "GOVERNED_BY"
    HAS_AMENDMENT = "HAS_AMENDMENT"
    TESTED_BY = "TESTED_BY"
    HAS_CERTIFICATION_REQUIREMENT = "HAS_CERTIFICATION_REQUIREMENT"
    SUPERSEDED_BY = "SUPERSEDED_BY"
    PART_OF = "PART_OF"


class EntityRelationship(Base):
    """
    Generic directed edge between two entities in the BIS knowledge graph.

    ``source_entity_type`` / ``target_entity_type`` hold uppercase table
    aliases such as "STANDARD", "PRODUCT", "DOCUMENT", "LABORATORY".

    ``source_entity_id`` / ``target_entity_id`` are the integer PKs in those
    respective tables. No FK constraint is declared here — the generic design
    intentionally avoids coupling to individual tables.
    """

    __tablename__ = "entity_relationships"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    source_entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    source_entity_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    relationship_type: Mapped[str] = mapped_column(String(100), nullable=False)

    target_entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    target_entity_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __table_args__ = (
        # Look up all outgoing edges from a source entity
        Index("ix_entity_rel_source", "source_entity_type", "source_entity_id"),
        # Look up all incoming edges to a target entity
        Index("ix_entity_rel_target", "target_entity_type", "target_entity_id"),
        # Filter edges by relationship type
        Index("ix_entity_rel_type", "relationship_type"),
    )
