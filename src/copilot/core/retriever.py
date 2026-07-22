"""Top-k semantic retriever over the vector store with hybrid search and MMR diversity."""

from __future__ import annotations

import logging

from copilot.core.bm25 import BM25Retriever
from copilot.indexing.embedder import Embedder
from copilot.indexing.vector_store import VectorStore
from copilot.schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever:
    """Semantic retriever with hybrid search (dense + BM25) and MMR diversity.

    Args:
        store: Vector store instance (e.g. ChromaStore).
        embedder: Embedder instance for encoding queries.
        hybrid_weight: Weight for dense scores when fusing with BM25.
        mmr_lambda: MMR diversity vs relevance trade-off (1.0 = pure relevance).
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        hybrid_weight: float = 0.7,
        mmr_lambda: float = 0.7,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._hybrid_weight = hybrid_weight
        self._mmr_lambda = mmr_lambda
        self._bm25: BM25Retriever | None = None

    def rebuild_bm25(self, chunks: list[Chunk]) -> None:
        """Build or refresh the BM25 index from the full list of chunks.

        Call this after the vector store has been populated.
        """
        self._bm25 = BM25Retriever(chunks)
        logger.info("BM25 index rebuilt with %d chunks", len(chunks))

    def retrieve(self, query: str, k: int = 5, query_vector: list[float] | None = None) -> list[RetrievedChunk]:
        """Embed the query and return the top-k most similar chunks.

        If ``query_vector`` is provided, it is used directly instead of
        encoding the query, saving one Ollama embed call.

        Uses hybrid search (dense + BM25) when BM25 index is available,
        and re-ranks with MMR for diversity.

        Args:
            query: The user's question or search query.
            k: Number of results to return.
            query_vector: Optional pre-computed query embedding vector.

        Returns:
            List of RetrievedChunk objects sorted by descending similarity.

        Raises:
            ValueError: If the query is empty after stripping.
        """
        if not query.strip():
            raise ValueError("query must be non-empty")

        # --- Dense retrieval ---
        if query_vector is not None:
            vector = query_vector
        else:
            vector = self._embedder.encode([query])[0].tolist()

        # Fetch more for reranking diversity.
        dense_pool = self._store.query(vector, k + 5)

        # --- Hybrid fusion with BM25 ---
        if self._bm25 is not None:
            bm25_results = self._bm25.search(query, k + 5)
            candidate_map: dict[str, RetrievedChunk] = {}

            for rc in dense_pool:
                candidate_map[rc.chunk.chunk_id] = rc

            # Merge — use dense scores where both exist, BM25 rank where not.
            bm25_scores = {}
            if bm25_results:
                max_bm25 = max(s for _, s in bm25_results)
                for idx, score in bm25_results:
                    chunk_id = rc_idx_to_id(idx)
                    if chunk_id:
                        bm25_scores[chunk_id] = score / max_bm25 if max_bm25 > 0 else 0.0

            # Normalise dense scores.
            if dense_pool:
                max_dense = max(rc.score for rc in dense_pool)
                if max_dense > 0:
                    for rc in dense_pool:
                        normalised = rc.score / max_dense
                        bm25 = bm25_scores.get(rc.chunk.chunk_id, 0.0)
                        rc.score = self._hybrid_weight * normalised + (1 - self._hybrid_weight) * bm25

            # Sort by hybrid score.
            dense_pool.sort(key=lambda x: -x.score)

        results = dense_pool[:k]

        # --- MMR re-ranking for diversity ---
        if len(results) > 1:
            results = self._mmr_rerank(vector, results, k, self._mmr_lambda)

        logger.debug("Retrieved %d chunks for query (hybrid+MMR)", len(results))
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

    @staticmethod
    def _mmr_rerank(
        query_vec: list[float],
        candidates: list[RetrievedChunk],
        k: int,
        mmr_lambda: float,
    ) -> list[RetrievedChunk]:
        """Maximum Marginal Relevance re-ranking for diversity.

        Selects a subset of ``k`` candidates that balances relevance to the
        query and diversity from already selected items.
        """
        import numpy as np

        if len(candidates) <= k:
            return candidates

        query_arr = np.array(query_vec, dtype=np.float32)
        mat = np.array([rc.chunk._vec for rc in candidates]) if hasattr(candidates[0].chunk, "_vec") else None

        # If we don't have chunk vectors, just return top-k.
        if mat is None or len(mat) != len(candidates):
            return candidates[:k]

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        # Start with the most relevant item.
        sims = mat @ query_arr
        first = int(np.argmax(sims))
        selected_indices.append(first)
        remaining.remove(first)

        for _ in range(1, k):
            if not remaining:
                break
            best_score = -1.0
            best_idx = -1
            for r in remaining:
                rel = float(mat[r] @ query_arr)
                max_div = max(float(mat[r] @ mat[s]) for s in selected_indices) if selected_indices else 0.0
                mmr_score = mmr_lambda * rel - (1 - mmr_lambda) * max_div
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = r
            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        return [candidates[i] for i in selected_indices]


def rc_idx_to_id(chunk_index: int) -> str | None:
    """Placeholder — the BM25 retriever returns indices, which we map back
    to chunk_ids via the vector store's chunk list. Not yet wired.
    """
    return None
