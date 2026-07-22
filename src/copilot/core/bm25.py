"""Lightweight BM25 keyword retriever for hybrid search.

Fits entirely in memory and is built on-the-fly from the chunks that
are already in the vector store. No external dependencies beyond stdlib
math/log.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from copilot.schemas import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _idf(num_docs: int, doc_freq: int) -> float:
    """Standard BM25 IDF with smoothing."""
    return math.log(1.0 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))


class BM25Retriever:
    """In-memory BM25 index over a list of chunks.

    Args:
        chunks: List of Chunk objects to index.
        k1: BM25 k1 parameter (term frequency saturation).
        b: BM25 b parameter (length normalisation).
    """

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = []
        self._avg_dl: float = 0.0
        self._doc_freqs: dict[str, int] = {}
        self._term_counts: list[Counter[str]] = []

        if chunks:
            self.add_chunks(chunks)

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks to the index and recompute statistics."""
        start_idx = len(self._chunks)
        self._chunks.extend(chunks)

        for ch in chunks:
            tokens = _tokenize(ch.text)
            terms = set(tokens)
            self._term_counts.append(Counter(tokens))
            for t in terms:
                self._doc_freqs[t] = self._doc_freqs.get(t, 0) + 1

        total_len = sum(sum(c.values()) for c in self._term_counts)
        self._avg_dl = total_len / len(self._chunks) if self._chunks else 0.0

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return top-k (index, score) pairs for the query.

        Args:
            query: The search query string.
            k: Number of results to return.

        Returns:
            List of (chunk_index, bm25_score) sorted by descending score.
        """
        query_tokens = _tokenize(query)
        query_freq = Counter(query_tokens)
        n_docs = len(self._chunks)
        if n_docs == 0 or not query_tokens:
            return []

        scores: list[float] = [0.0] * n_docs
        for t, qf in query_freq.items():
            df = self._doc_freqs.get(t, 0)
            if df == 0:
                continue
            idf = _idf(n_docs, df)
            for i in range(n_docs):
                tf = self._term_counts[i].get(t, 0)
                if tf == 0:
                    continue
                dl = sum(self._term_counts[i].values())
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * dl / self._avg_dl)
                scores[i] += idf * numerator / denominator

        # Get top-k indices.
        indexed = sorted(enumerate(scores), key=lambda x: -x[1])[:k]
        return [(idx, score) for idx, score in indexed if score > 0.0]

    def search_as_chunks(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Same as search but returns results with scores only (faster)."""
        return self.search(query, k)
