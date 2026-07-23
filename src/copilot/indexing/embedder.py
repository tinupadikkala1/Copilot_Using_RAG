"""Ollama embedding wrapper using nomic-embed-text.

Replaces sentence-transformers from the original plan. Calls the Ollama
/api/embed endpoint which is available in Ollama 0.31+.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import numpy as np

from copilot.config import get_settings

logger = logging.getLogger(__name__)

EMBED_DIM = 768  # nomic-embed-text produces 768-dim vectors


class Embedder:
    """Embed texts using nomic-embed-text via the Ollama API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        settings = get_settings()
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.embedding_model
        self._timeout = timeout
        self._client = httpx.Client(timeout=self._timeout)
        logger.info("Embedder initialised: model=%s url=%s", self._model, self._base_url)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 embeddings of shape (n, 768).

        Args:
            texts: List of text strings to embed.

        Returns:
            Float32 numpy array of shape (len(texts), 768).

        Raises:
            RuntimeError: If the Ollama API call fails.
        """
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)

        payload: dict[str, Any] = {
            "model": self._model,
            "input": texts,
        }

        try:
            resp = self._client.post(f"{self._base_url}/api/embed", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            logger.exception("Ollama embed API call failed")
            raise RuntimeError(f"Embedding failed for model {self._model}: {exc}") from exc

        embeddings = data.get("embeddings")
        if not embeddings or not isinstance(embeddings, list):
            raise RuntimeError(
                f"Unexpected Ollama embed response: missing 'embeddings' key. " f"Response: {data}"
            )

        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.shape[1] != EMBED_DIM:
            logger.warning(
                "Unexpected embedding dim: got %d, expected %d",
                vectors.shape[1],
                EMBED_DIM,
            )

        # L2-normalize so dot product == cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
        vectors = vectors / norms

        return vectors
