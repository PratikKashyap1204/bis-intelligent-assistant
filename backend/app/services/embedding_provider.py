"""
Embedding provider abstraction (Milestone 2).

An ``EmbeddingProvider`` turns text into a fixed-length vector. Nothing
else in this codebase should import a specific embedding library directly
— always go through this interface, so the provider can be swapped
(different model, or a hosted API) without touching callers.

Provider selected for this MVP: sentence-transformers, local model
--------------------------------------------------------------------
Model: ``sentence-transformers/all-MiniLM-L6-v2`` (384 dimensions).

Why this, and not a hosted API (OpenAI/Cohere/etc.):
  - Cost: zero — no per-call billing, no rate limits, no API key to manage
    for a student/hackathon project with a small, fixed pilot corpus.
  - Reproducibility: the exact same model runs the same way on any
    machine, offline, once downloaded. A hosted API can silently change
    model versions or availability behind an endpoint.
  - No network dependency at query/embed time (only the first model
    download requires internet); this matters for a live SIH demo.
  - Quality: MiniLM is a well-established, general-purpose sentence
    embedding model with solid results on English technical/legal text
    for an MVP. It is not a specialised legal/standards embedding model —
    see limitations below.
  - Dimensionality (384) keeps pgvector storage and similarity search
    cheap at our current (tiny) corpus size.

Tradeoffs / limitations (documented, not hidden):
  - English-only quality. The pilot corpus is 100% English, so this is
    fine today. ``all-MiniLM-L6-v2`` was NOT trained for Hindi.
    The natural upgrade path for multilingual/Hindi support (mentioned
    as a future requirement) is swapping to a multilingual sentence-
    transformers model such as
    ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``
    (also 384-dim — a drop-in replacement for the vector column). This
    is intentionally NOT done now, to avoid a heavier model when the
    corpus is 100% English (`clauses.language = "en"` for every row) —
    see Milestone 3+ recommendation in the final report.
  - Not fine-tuned on BIS/legal/standards text specifically. It captures
    general semantic similarity, not domain-specific legal nuance.
  - Local inference only — no GPU is assumed here; fine for our current
    129-clause corpus, would need batching/GPU consideration at scale.

Testing
-------
Loading the real model is slow (seconds) and downloads weights on first
use, so automated tests use ``DeterministicHashEmbeddingProvider``
instead (see below) — fast, offline, no ML dependency required to run
the test suite. Only the manual pilot script
(``scripts/embed_pilot.py``) exercises the real model.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Protocol

# Dimension of the production model (sentence-transformers/all-MiniLM-L6-v2).
# Also used as the default for the deterministic test provider so tests
# exercise the same vector width as production.
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class EmbeddingProvider(Protocol):
    """
    Anything that can turn text into fixed-length vectors.

    ``model_name`` and ``dimension`` are used to tag stored embeddings
    (see app/models/embedding.py) so multiple providers/model versions
    can coexist without colliding, and so a model upgrade is detectable.
    """

    model_name: str
    dimension: int

    def generate(self, text: str) -> List[float]:
        ...

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        ...


class SentenceTransformerEmbeddingProvider:
    """
    Production provider: a local sentence-transformers model.

    The model is loaded lazily (on first ``generate``/``generate_batch``
    call), not at import time — importing this module must stay cheap so
    it doesn't slow down every test collection or unrelated code path.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model = None  # lazy-loaded SentenceTransformer instance
        self._dimension: Optional[int] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Imported here (not at module level) so importing this file never
        # requires torch/sentence-transformers unless this provider is
        # actually used.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        assert self._dimension is not None
        return self._dimension

    def generate(self, text: str) -> List[float]:
        return self.generate_batch([text])[0]

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        self._ensure_loaded()
        assert self._model is not None
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,  # cosine similarity == dot product on unit vectors
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]


def get_default_embedding_provider() -> "SentenceTransformerEmbeddingProvider":
    """
    Build the production provider using the configured model name
    (``EMBEDDING_MODEL_NAME`` in .env / app.config.settings).

    Imported lazily inside the function body to avoid a hard import-time
    dependency from this module onto app.config for callers (like tests)
    that only need the deterministic provider.
    """
    from app.config import settings

    return SentenceTransformerEmbeddingProvider(model_name=settings.EMBEDDING_MODEL_NAME)


class DeterministicHashEmbeddingProvider:
    """
    Deterministic, dependency-free stand-in for tests.

    NOT a real semantic embedding model. It maps text to a fixed-length
    vector via feature hashing over word tokens: each token deterministically
    lights up a handful of dimensions (via its hash), so texts sharing
    words produce more similar vectors than texts that share none — enough
    behaviour to exercise ranking/ordering logic in tests without ever
    downloading a model or depending on randomness.

    Uses the SAME dimension as the production model (384) so persisted
    test embeddings exercise the real pgvector column width.
    """

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION, model_name: str = "test-hash-embedding-v1"):
        self.dimension = dimension
        self.model_name = model_name

    def generate(self, text: str) -> List[float]:
        return self.generate_batch([text])[0]

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            # Use a few hashed positions per token so overlap between texts
            # sharing words is reflected in more than one dimension.
            for offset in range(4):
                idx = int.from_bytes(digest[offset * 4 : offset * 4 + 4], "big") % self.dimension
                sign = 1.0 if digest[offset] % 2 == 0 else -1.0
                vector[idx] += sign
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector
