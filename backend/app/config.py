"""Application configuration loaded from environment variables.

All settings are read from the process environment.  In development,
values can be placed in a ``.env`` file at the project root — python-dotenv
loads it automatically before any setting is accessed.

Add new settings here as the project grows.  Never hardcode secrets.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels above backend/app/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Application settings sourced entirely from the environment."""

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    APP_NAME: str = os.getenv("APP_NAME", "BIS Intelligent Assistant")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    # Format: postgresql+psycopg2://user:password@host:port/dbname
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://bis_user:change_me@localhost:5432/bis_db",
    )

    # ------------------------------------------------------------------
    # Embeddings (Milestone 2)
    # ------------------------------------------------------------------
    # Local sentence-transformers model — no API key needed. Overridable so
    # a future multilingual model can be swapped in via configuration alone
    # (e.g. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )


settings = Settings()
