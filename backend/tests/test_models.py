"""
Unit tests for models and Pydantic schemas.

These tests import model classes and schemas but do NOT open a database
connection.  They verify:

  - Every required column is declared on the right table
  - Enum members carry the expected string values
  - SQLAlchemy indexes are configured correctly
  - Pydantic schemas validate well-formed data and reject bad data

The PostgreSQL integration test at the bottom is skipped unless
TEST_DATABASE_URL is set in the environment (see conftest.py).
"""

import os

import pytest

from app.models.certification import CertificationRequirement
from app.models.clause import Clause
from app.models.document import Document, DocumentType
from app.models.hallmarking import HallmarkingRequirement
from app.models.laboratory import Laboratory
from app.models.product import Product
from app.models.relationship import EntityRelationship, RelationshipType
from app.models.standard import Standard, StandardStatus
from app.models.test_requirement import TestRequirement
from app.schemas.document import DocumentCreate
from app.schemas.laboratory import LaboratoryCreate
from app.schemas.product import ProductCreate, ProductRead
from app.schemas.standard import StandardCreate, StandardRead


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def col_names(model) -> set[str]:
    """Return the set of column names declared on a model's table."""
    return {c.name for c in model.__table__.columns}


def index_col_names(model) -> set[str]:
    """Return the set of all column names covered by the model's indexes."""
    return {
        col.name
        for idx in model.__table__.indexes
        for col in idx.columns
    }


# ---------------------------------------------------------------------------
# Column presence tests
# ---------------------------------------------------------------------------

def test_product_columns():
    assert {"id", "name", "category", "description", "attributes", "created_at"} <= col_names(Product)


def test_standard_columns():
    assert {
        "id", "is_number", "title", "year", "revision", "status",
        "scope", "source_url", "created_at", "updated_at",
    } <= col_names(Standard)


def test_document_columns():
    assert {
        "id", "standard_id", "document_type", "title",
        "source_url", "local_file_path", "published_date", "created_at",
    } <= col_names(Document)


def test_clause_columns():
    assert {"id", "document_id", "clause_number", "title", "content", "page_number"} <= col_names(Clause)


def test_test_requirement_columns():
    assert {"id", "standard_id", "clause", "test_name", "description"} <= col_names(TestRequirement)


def test_laboratory_columns():
    assert {
        "id", "name", "lab_code", "address", "state",
        "district", "source_url", "validity_date",
    } <= col_names(Laboratory)


def test_certification_requirement_columns():
    assert {
        "id", "product_id", "standard_id", "scheme",
        "mandatory", "qco", "requirements", "source_url",
    } <= col_names(CertificationRequirement)


def test_hallmarking_requirement_columns():
    assert {"id", "standard_id", "metal", "fineness", "process", "source_url"} <= col_names(HallmarkingRequirement)


def test_entity_relationship_columns():
    assert {
        "id", "source_entity_type", "source_entity_id",
        "relationship_type",
        "target_entity_type", "target_entity_id",
    } <= col_names(EntityRelationship)


# ---------------------------------------------------------------------------
# Enum value tests
# ---------------------------------------------------------------------------

def test_document_type_values():
    assert DocumentType.STANDARD.value == "STANDARD"
    assert DocumentType.PRODUCT_MANUAL.value == "PRODUCT_MANUAL"
    assert DocumentType.AMENDMENT.value == "AMENDMENT"
    assert DocumentType.QCO.value == "QCO"
    assert DocumentType.SCHEME.value == "SCHEME"
    assert DocumentType.FAQ.value == "FAQ"
    assert DocumentType.OTHER.value == "OTHER"


def test_standard_status_values():
    assert StandardStatus.ACTIVE.value == "ACTIVE"
    assert StandardStatus.WITHDRAWN.value == "WITHDRAWN"
    assert StandardStatus.UNDER_REVISION.value == "UNDER_REVISION"
    assert StandardStatus.SUPERSEDED.value == "SUPERSEDED"


def test_relationship_type_values():
    assert RelationshipType.GOVERNED_BY.value == "GOVERNED_BY"
    assert RelationshipType.RELATED_STANDARD.value == "RELATED_STANDARD"
    assert RelationshipType.HAS_AMENDMENT.value == "HAS_AMENDMENT"
    assert RelationshipType.TESTED_BY.value == "TESTED_BY"
    assert RelationshipType.HAS_CERTIFICATION_REQUIREMENT.value == "HAS_CERTIFICATION_REQUIREMENT"


# ---------------------------------------------------------------------------
# Index tests
# ---------------------------------------------------------------------------

def test_standard_has_is_number_and_status_indexes():
    cols = index_col_names(Standard)
    assert "is_number" in cols, "Standard should be indexed on is_number"
    assert "status" in cols, "Standard should be indexed on status"


def test_document_has_document_type_index():
    assert "document_type" in index_col_names(Document)


def test_clause_has_document_id_index():
    assert "document_id" in index_col_names(Clause)


def test_test_requirement_has_standard_id_index():
    assert "standard_id" in index_col_names(TestRequirement)


def test_laboratory_has_lab_code_index():
    assert "lab_code" in index_col_names(Laboratory)


# ---------------------------------------------------------------------------
# Pydantic schema tests
# ---------------------------------------------------------------------------

def test_standard_create_minimal():
    s = StandardCreate(is_number="IS 302", title="Spec for Electric Irons")
    assert s.is_number == "IS 302"
    assert s.status == StandardStatus.ACTIVE   # default
    assert s.year is None


def test_standard_create_full():
    s = StandardCreate(
        is_number="IS 1998",
        title="Spec for Portable Tools",
        year=2010,
        revision="Second Revision",
        status=StandardStatus.ACTIVE,
        scope="Covers portable electric tools.",
        source_url="https://www.bis.gov.in/example",
    )
    assert s.year == 2010
    assert s.revision == "Second Revision"


def test_standard_create_requires_is_number():
    with pytest.raises(Exception):
        StandardCreate(title="Missing IS number")  # is_number is required


def test_product_create_minimal():
    p = ProductCreate(name="Electric Iron")
    assert p.name == "Electric Iron"
    assert p.attributes is None


def test_product_create_with_attributes():
    p = ProductCreate(
        name="Electric Iron",
        category="Electrical Appliances",
        attributes={"voltage": "230V", "wattage": "1000W"},
    )
    assert p.attributes["voltage"] == "230V"


def test_document_create_requires_document_type():
    with pytest.raises(Exception):
        DocumentCreate(title="Some doc")  # document_type is required


def test_document_create_valid():
    d = DocumentCreate(title="IS 302 PDF", document_type=DocumentType.STANDARD)
    assert d.document_type == DocumentType.STANDARD
    assert d.standard_id is None  # nullable


def test_laboratory_create_valid():
    lab = LaboratoryCreate(name="National Test House", lab_code="NTH/KOL/001")
    assert lab.lab_code == "NTH/KOL/001"
    assert lab.validity_date is None


# ---------------------------------------------------------------------------
# Integration test: table creation (requires TEST_DATABASE_URL)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set — skipping PostgreSQL integration test",
)
def test_create_all_tables(pg_engine):
    """Verify that all tables are created in a real PostgreSQL database."""
    from sqlalchemy import inspect

    inspector = inspect(pg_engine)
    tables = set(inspector.get_table_names())

    expected = {
        "products",
        "standards",
        "documents",
        "clauses",
        "test_requirements",
        "laboratories",
        "certification_requirements",
        "hallmarking_requirements",
        "entity_relationships",
    }
    assert expected <= tables, f"Missing tables: {expected - tables}"
