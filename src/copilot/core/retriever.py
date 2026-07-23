"""Top-k semantic retriever over the vector store with optional hybrid search."""

from __future__ import annotations

import logging

from copilot.core.bm25 import BM25Retriever
from copilot.indexing.embedder import Embedder
from copilot.indexing.vector_store import ChromaStore, VectorStore
from copilot.schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever:
    """Semantic retriever with optional BM25 hybrid search.

    Supports both dense-only and dense+BM25 hybrid retrieval.
    BM25 index is populated via ``rebuild_bm25()``.

    Args:
        store: Vector store instance (e.g. ChromaStore).
        embedder: Embedder instance for encoding queries.
        hybrid_weight: Weight for dense scores when fusing with BM25 (0.0 = BM25 only).
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        hybrid_weight: float = 0.5,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._hybrid_weight = hybrid_weight
        self._bm25: BM25Retriever | None = None
        self._bm25_chunk_ids: list[str] = []  # parallel list mapping BM25 index -> chunk_id

    def rebuild_bm25(self, chunks: list[Chunk]) -> None:
        """Build or refresh the BM25 index from a list of chunks.

        Call this after the vector store has been populated (typically
        from ``ChromaStore.get_all_chunks()`` or after ``build_index()``).

        Args:
            chunks: Full list of Chunk objects currently in the vector store.
        """
        if not chunks:
            self._bm25 = None
            self._bm25_chunk_ids = []
            logger.warning("BM25 rebuild skipped: empty chunks list")
            return
        self._bm25 = BM25Retriever(chunks)
        self._bm25_chunk_ids = [c.chunk_id for c in chunks]
        logger.info("BM25 index rebuilt with %d chunks", len(chunks))

    def retrieve(self, query: str, k: int = 5, query_vector: list[float] | None = None) -> list[RetrievedChunk]:
        """Embed the query and return the top-k most similar chunks.

        If ``query_vector`` is provided, it is used directly instead of
        encoding the query, saving one Ollama embed call.

        Uses hybrid search (dense + BM25) when BM25 index is available.
        When BM25 is not available, falls back to pure dense retrieval.

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

        dense_results = self._store.query(vector, k)

        # --- Hybrid fusion with BM25 ---
        if self._bm25 is not None and self._bm25_chunk_ids:
            bm25_results = self._bm25.search(query, k * 2)  # Get more BM25 results for diversity

            if bm25_results:
                max_bm25 = max(s for _, s in bm25_results) if bm25_results else 1.0
                bm25_score_map: dict[str, float] = {}
                bm25_chunk_map: dict[str, int] = {}  # chunk_id -> bm25 index
                for idx, score in bm25_results:
                    if idx < len(self._bm25_chunk_ids):
                        cid = self._bm25_chunk_ids[idx]
                        bm25_score_map[cid] = score / max_bm25 if max_bm25 > 0 else 0.0
                        bm25_chunk_map[cid] = idx

                # Normalise dense scores and fuse.
                max_dense = max((rc.score for rc in dense_results), default=1.0)
                dense_cids: set[str] = set()
                if max_dense > 0:
                    for rc in dense_results:
                        dense_cids.add(rc.chunk.chunk_id)
                        norm_dense = rc.score / max_dense
                        bm25_score = bm25_score_map.get(rc.chunk.chunk_id, 0.0)
                        rc.score = self._hybrid_weight * norm_dense + (1 - self._hybrid_weight) * bm25_score

                # Add BM25-only results (not in dense).
                for cid, norm_score in bm25_score_map.items():
                    if cid not in dense_cids and self._bm25 is not None:
                        idx = bm25_chunk_map[cid]
                        if idx < len(self._bm25._chunks):
                            chunk = self._bm25._chunks[idx]
                            bm25_only_score = (1 - self._hybrid_weight) * norm_score
                            dense_results.append(RetrievedChunk(chunk=chunk, score=bm25_only_score))

                # Re-sort and truncate to k.
                dense_results.sort(key=lambda x: -x.score)
                dense_results = dense_results[:k]

        logger.debug("Retrieved %d chunks for query (hybrid=%s)", len(dense_results), self._bm25 is not None)
        # Ensure results include chunks from multiple source documents.
        dense_results = self._diversify_results(dense_results, k)
        return dense_results

    @staticmethod
    def _diversify_results(results: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        """Ensure results include chunks from multiple source documents.

        Uses a round-robin approach: take the best chunk from each document
        first, then fill remaining slots with the next-best from any document.
        This prevents all results being from a single file.
        """
        if len(results) <= 1:
            return results

        # Group by source document.
        from collections import OrderedDict

        by_doc: OrderedDict[str, list[RetrievedChunk]] = OrderedDict()
        for rc in results:
            doc_id = rc.chunk.doc_id
            if doc_id not in by_doc:
                by_doc[doc_id] = []
            by_doc[doc_id].append(rc)

        # If all from one doc, just return as-is (nothing to diversify).
        if len(by_doc) <= 1:
            return results[:k]

        # Round-robin: take best from each doc first.
        diversified: list[RetrievedChunk] = []
        doc_indices: dict[str, int] = {doc: 0 for doc in by_doc}

        # First pass: one from each document (by score order).
        for doc_id, chunks in by_doc.items():
            if len(diversified) < k:
                diversified.append(chunks[0])
                doc_indices[doc_id] = 1

        # Second pass: fill remaining slots with next-best chunks.
        while len(diversified) < k:
            added = False
            for doc_id, chunks in by_doc.items():
                idx = doc_indices[doc_id]
                if idx < len(chunks) and len(diversified) < k:
                    diversified.append(chunks[idx])
                    doc_indices[doc_id] = idx + 1
                    added = True
            if not added:
                break

        # Sort final results by score (best first).
        diversified.sort(key=lambda x: -x.score)
        return diversified[:k]

    @property
    def embedder(self) -> Embedder:
        """Public access to the internal Embedder (used by HyDE in pipeline)."""
        return self._embedder

    @staticmethod
    def top_score(results: list[RetrievedChunk]) -> float:
        """Return the highest similarity score from the results.

        Args:
            results: Retrieved chunks from a query.

        Returns:
            The highest score, or 0.0 if there are no results.
        """
        return results[0].score if results else 0.0
