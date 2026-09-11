"""
Development utility: create all database tables.

Run from the project root:

    cd bis-intelligent-assistant
    source backend/.venv/bin/activate
    python scripts/init_db.py

This script:
  1. Reads DATABASE_URL from the environment (or .env file).
  2. Imports every model module so SQLAlchemy registers the tables.
  3. Calls Base.metadata.create_all() — safe to run repeatedly
     (CREATE TABLE IF NOT EXISTS semantics).

NOTE: This is for development only.
      In production, use Alembic migrations (to be added later).
"""

import sys
from pathlib import Path

# Add backend/ to sys.path so 'app.*' imports work when running from project root.
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Load .env from project root before importing settings
from dotenv import load_dotenv  # noqa: E402
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Import settings (reads DATABASE_URL)
from app.config import settings  # noqa: E402

# Import all models — this registers every table with Base.metadata
import app.models  # noqa: F401, E402

from app.db.base import Base  # noqa: E402
from app.db.session import engine  # noqa: E402


def main() -> None:
    print(f"Target database : {settings.DATABASE_URL}")
    print("Creating tables (CREATE TABLE IF NOT EXISTS)...")

    Base.metadata.create_all(bind=engine)

    tables = sorted(Base.metadata.tables.keys())
    print(f"Done. {len(tables)} table(s) registered:")
    for t in tables:
        print(f"  ✓  {t}")


if __name__ == "__main__":
    main()
