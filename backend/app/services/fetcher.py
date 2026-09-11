"""
Controlled, fetch-by-known-URL document acquisition.

This module downloads exactly one developer-provided URL and saves the raw
bytes to disk. It is intentionally NOT a crawler: it does not discover
links, paginate, or explore a site — every call targets one specific,
known official URL supplied by the caller (see scripts/ingest_pilot.py).

Raw files are never modified after being written — see
app/services/normalizer.py for where cleanup happens, which always
operates on a copy, never on the file saved here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx

DEFAULT_TIMEOUT = 30.0  # seconds


class FetchError(Exception):
    """Raised when a document cannot be acquired from its source URL."""


@dataclass
class FetchResult:
    """Metadata about a successfully fetched (or cached) document."""

    url: str
    local_path: Path
    content_hash: str
    content_type: str | None
    byte_size: int
    from_cache: bool


def fetch_to_raw(
    url: str,
    slug: str,
    filename: str,
    raw_dir: Path,
    timeout: float = DEFAULT_TIMEOUT,
) -> FetchResult:
    """
    Download `url` and save it under `raw_dir/slug/filename`.

    If a file already exists at that exact path, it is reused (re-hashed,
    not re-downloaded) rather than re-fetched. This is a simple,
    deterministic caching strategy appropriate for a controlled,
    known-URL pipeline — delete the file manually to force a re-fetch.

    Returns:
        FetchResult with the local path, SHA-256 content hash, and
        whether the file was served from the local cache.

    Raises:
        FetchError: on connection errors, timeouts, non-2xx HTTP status,
        or an empty response body.
    """
    target_dir = raw_dir / slug
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    if target_path.exists():
        content = target_path.read_bytes()
        if not content:
            raise FetchError(f"Cached file is empty, delete and re-fetch: {target_path}")
        return FetchResult(
            url=url,
            local_path=target_path,
            content_hash=hashlib.sha256(content).hexdigest(),
            content_type=None,
            byte_size=len(content),
            from_cache=True,
        )

    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
    except httpx.TimeoutException as exc:
        raise FetchError(f"Timed out fetching {url}: {exc}") from exc
    except httpx.ConnectError as exc:
        raise FetchError(f"Connection error fetching {url}: {exc}") from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"HTTP error fetching {url}: {exc}") from exc

    if response.status_code >= 400:
        raise FetchError(
            f"Unexpected status {response.status_code} fetching {url}: "
            f"{response.text[:200]!r}"
        )

    content = response.content
    if not content:
        raise FetchError(f"Empty response body fetching {url}")

    target_path.write_bytes(content)

    return FetchResult(
        url=url,
        local_path=target_path,
        content_hash=hashlib.sha256(content).hexdigest(),
        content_type=response.headers.get("content-type"),
        byte_size=len(content),
        from_cache=False,
    )
