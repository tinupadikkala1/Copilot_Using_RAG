"""Top-k semantic retriever over the vector store."""

from __future__ import annotations

import logging

from copilot.indexing.embedder import Embedder
from copilot.indexing.vector_store import VectorStore
from copilot.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever:
    """Semantic retriever that embeds queries and searches the vector store.

    Args:
        store: Vector store instance (e.g. ChromaStore).
        embedder: Embedder instance for encoding queries.
    """

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Embed the query and return the top-k most similar chunks.

        Args:
            query: The user's question or search query.
            k: Number of results to return.

        Returns:
            List of RetrievedChunk objects sorted by descending similarity.

        Raises:
            ValueError: If the query is empty after stripping.
        """
        if not query.strip():
            raise ValueError("query must be non-empty")
        vector = self._embedder.encode([query])[0].tolist()
        results = self._store.query(vector, k)
        logger.debug("Retrieved %d chunks for query", len(results))
        return results

    @staticmethod
    def top_score(results: list[RetrievedChunk]) -> float:
        """Return the highest similarity score from the results.

        Args:
            results: Retrieved chunks from a query.

        Returns:
            The highest score, or 0.0 if there are no results.
        """
        return results[0].score if results else 0.0
