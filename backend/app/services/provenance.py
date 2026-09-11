"""
Ingestion provenance logging — MVP version.

Each ingestion attempt appends one JSON line to
``data/metadata/ingestion_log.jsonl`` recording what was fetched/ingested
and the outcome. This is intentionally a flat append-only file, NOT a
database table — cheap to add now, and easy to graduate into a proper
``IngestionRun`` table later once the ingestion pipeline covers more
than one pilot standard.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# backend/app/services/provenance.py -> parents[3] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_PATH = PROJECT_ROOT / "data" / "metadata" / "ingestion_log.jsonl"


def log_event(
    *,
    source_url: Optional[str],
    entity_type: str,
    entity_id: Optional[int],
    outcome: str,
    content_hash: Optional[str] = None,
    error: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """
    Append one provenance record to the ingestion log.

    Args:
        outcome: one of "created", "updated", "failed" (free-form string,
            not enforced, to keep the MVP simple).
    """
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_url": source_url,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "content_hash": content_hash,
        "outcome": outcome,
        "error": error,
    }
    if extra:
        record["extra"] = extra

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
