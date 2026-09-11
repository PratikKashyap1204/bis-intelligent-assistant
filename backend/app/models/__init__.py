"""
Import every model so that Base.metadata discovers all tables.

This module must be imported before calling Base.metadata.create_all()
or running Alembic migrations. Each import registers the model's table
with the shared DeclarativeBase metadata.
"""

# Order matters only to satisfy Python's module resolution —
# SQLAlchemy resolves FK targets by string name at mapper configuration time,
# so circular imports are not an issue here.

from app.models.product import Product  # noqa: F401
from app.models.standard import Standard, StandardStatus  # noqa: F401
from app.models.document import Document, DocumentType  # noqa: F401
from app.models.clause import Clause  # noqa: F401
from app.models.test_requirement import TestRequirement  # noqa: F401
from app.models.laboratory import Laboratory  # noqa: F401
from app.models.certification import CertificationRequirement  # noqa: F401
from app.models.hallmarking import HallmarkingRequirement  # noqa: F401
from app.models.relationship import EntityRelationship, RelationshipType  # noqa: F401

__all__ = [
    "Product",
    "Standard",
    "StandardStatus",
    "Document",
    "DocumentType",
    "Clause",
    "TestRequirement",
    "Laboratory",
    "CertificationRequirement",
    "HallmarkingRequirement",
    "EntityRelationship",
    "RelationshipType",
]
