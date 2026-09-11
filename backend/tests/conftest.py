"""
Pytest configuration and shared fixtures.

Unit tests in this suite do NOT require a running database — they only
import models and schemas to verify Python-level definitions.

Integration tests (table creation, queries) are gated behind the
``TEST_DATABASE_URL`` environment variable and are skipped automatically
when it is not set.  To run them:

    export TEST_DATABASE_URL="postgresql+psycopg2://bis_user:secret@localhost:5432/bis_test"
    pytest
"""

import os

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


@pytest.fixture(scope="session")
def pg_engine() -> Engine:
    """
    Session-scoped engine pointing at TEST_DATABASE_URL.

    Creates all tables before the session, drops them after.
    Skipped automatically when TEST_DATABASE_URL is not set.
    """
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set — skipping PostgreSQL integration tests")

    import app.models  # noqa: F401 — registers all models with metadata
    from app.db.base import Base

    engine = create_engine(url, echo=False)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(pg_engine: Engine) -> Session:
    """
    Function-scoped session bound to TEST_DATABASE_URL.

    Cleans up (deletes) rows from the ingestion-related tables after each
    test, so tests don't leak state into one another. Skipped
    automatically (via pg_engine) when TEST_DATABASE_URL is not set.
    """
    from app.models.clause import Clause
    from app.models.document import Document
    from app.models.standard import Standard

    TestSessionLocal = sessionmaker(bind=pg_engine)
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.query(Clause).delete()
        session.query(Document).delete()
        session.query(Standard).delete()
        session.commit()
        session.close()
