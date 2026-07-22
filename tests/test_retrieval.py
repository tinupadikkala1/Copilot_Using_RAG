"""Unit tests for the retrieval pipeline (embedder, vector store, index builder).

These tests use the mock KB (3 small markdown docs) and the real
Ollama nomic-embed-text model. Tests that require Ollama running are
marked with ``@pytest.mark.ollama`` and will be skipped if Ollama is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from copilot.indexing.embedder import EMBED_DIM, Embedder
from copilot.indexing.index_builder import build_index
from copilot.indexing.vector_store import ChromaStore
from copilot.schemas import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ollama_available() -> bool:
    """Check whether Ollama is running and has the embedding model."""
    import httpx

    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = resp.json().get("models", [])
        return any("nomic-embed-text" in m.get("name", "") for m in models)
    except (httpx.HTTPError, ConnectionError):
        return False


ollama_ready = ollama_available()


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------


class TestEmbedder:
    """Tests for the Ollama-based Embedder."""

    def test_embedder_init(self) -> None:
        """Embedder should initialise without errors (lazy — no API call)."""
        emb = Embedder()
        assert emb is not None

    @pytest.mark.skipif(not ollama_ready, reason="Ollama not running or nomic-embed-text missing")
    def test_encode_single_text(self) -> None:
        """A single text should return shape (1, 768)."""
        emb = Embedder()
        vec = emb.encode(["Hello, world!"])
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (1, EMBED_DIM), f"Expected (1, {EMBED_DIM}), got {vec.shape}"
        assert vec.dtype == np.float32

    @pytest.mark.skipif(not ollama_ready, reason="Ollama not running or nomic-embed-text missing")
    def test_encode_multiple_texts(self) -> None:
        """Multiple texts should return (n, 768)."""
        emb = Embedder()
        texts = ["First query.", "Second query.", "Third query."]
        vec = emb.encode(texts)
        assert vec.shape == (3, EMBED_DIM)

    @pytest.mark.skipif(not ollama_ready, reason="Ollama not running or nomic-embed-text missing")
    def test_encode_empty_list(self) -> None:
        """An empty list should return an empty array."""
        emb = Embedder()
        vec = emb.encode([])
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (0, EMBED_DIM)

    @pytest.mark.skipif(not ollama_ready, reason="Ollama not running or nomic-embed-text missing")
    def test_similar_texts_have_higher_similarity(self) -> None:
        """Similar texts should have higher cosine similarity than unrelated texts."""
        emb = Embedder()
        vecs = emb.encode(
            [
                "How do I reset my password?",
                "Password reset instructions",
                "What is the weather like today?",
            ]
        )
        sim_similar = float(vecs[0] @ vecs[1])  # dot product of normalized vectors
        sim_unrelated = float(vecs[0] @ vecs[2])
        assert (
            sim_similar > sim_unrelated
        ), f"Similar ({sim_similar:.3f}) should be closer than unrelated ({sim_unrelated:.3f})"


# ---------------------------------------------------------------------------
# ChromaStore tests
# ---------------------------------------------------------------------------


class TestChromaStore:
    """Tests for the ChromaDB vector store wrapper."""

    def test_empty_store_count(self, tmp_path: Path) -> None:
        """A fresh store should have a count of 0."""
        store = ChromaStore(persist_dir=str(tmp_path / "chroma"))
        assert store.count() == 0

    def test_upsert_and_query(self, tmp_path: Path) -> None:
        """Upserting a chunk and querying should return it."""
        store = ChromaStore(persist_dir=str(tmp_path / "chroma"))
        chunk = Chunk(
            chunk_id="test::0",
            doc_id="test",
            title="Test",
            text="Refunds are issued within 5 business days.",
            ordinal=0,
            content_hash=Chunk.make_hash("Refunds are issued within 5 business days."),
            source_path="test.md",
        )
        # A simple mock vector: dim=768, all zeros with a 1 at index 0.
        vector = [1.0 if i == 0 else 0.0 for i in range(EMBED_DIM)]
        store.upsert([chunk], [vector])
        assert store.count() == 1

        # Query with a similar vector.
        query_vec = [0.9 if i == 0 else 0.0 for i in range(EMBED_DIM)]
        results = store.query(query_vec, k=5)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "test::0"
        assert results[0].score > 0.8

    def test_upsert_multiple_batches(self, tmp_path: Path) -> None:
        """Multiple upserts should accumulate."""
        store = ChromaStore(persist_dir=str(tmp_path / "chroma_batches"))
        chunks = [
            Chunk(
                chunk_id=f"doc::{i}",
                doc_id="doc",
                title=f"Doc {i}",
                text=f"Document number {i} content.",
                ordinal=i,
                content_hash=Chunk.make_hash(f"Document number {i} content."),
                source_path="doc.md",
            )
            for i in range(5)
        ]
        vectors = [[1.0 if j == i else 0.0 for j in range(EMBED_DIM)] for i in range(5)]
        store.upsert(chunks[:3], vectors[:3])
        store.upsert(chunks[3:], vectors[3:])
        assert store.count() == 5

    def test_query_returns_top_k(self, tmp_path: Path) -> None:
        """Query with k=2 should return at most 2 results."""
        store = ChromaStore(persist_dir=str(tmp_path / "chroma_topk"))
        for i in range(10):
            chunk = Chunk(
                chunk_id=f"doc::{i}",
                doc_id="doc",
                title=f"Doc {i}",
                text=f"Content {i}.",
                ordinal=i,
                content_hash=Chunk.make_hash(f"Content {i}."),
                source_path="doc.md",
            )
            vector = [1.0 if j == i else 0.0 for j in range(EMBED_DIM)]
            store.upsert([chunk], [vector])

        query_vec = [1.0 if j == 0 else 0.0 for j in range(EMBED_DIM)]
        results = store.query(query_vec, k=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Index builder integration test (requires Ollama)
# ---------------------------------------------------------------------------


class TestIndexBuilder:
    """Integration tests for the full ETL pipeline."""

    @pytest.mark.skipif(not ollama_ready, reason="Ollama not running or nomic-embed-text missing")
    def test_build_index_from_mock_kb(self, mock_kb_path: Path, tmp_path: Path) -> None:
        """Building an index from the mock KB should succeed and return chunks."""
        embedder = Embedder()
        store = ChromaStore(persist_dir=str(tmp_path / "integration"))
        count = build_index(
            kb_root=mock_kb_path,
            store=store,
            embedder=embedder,
            chunk_size=800,
            overlap=150,
        )
        assert count > 0, "Index builder must produce at least one chunk"
        assert store.count() == count, "Store count should match indexed count"

    @pytest.mark.skipif(not ollama_ready, reason="Ollama not running or nomic-embed-text missing")
    def test_index_is_queryable(self, mock_kb_path: Path, tmp_path: Path) -> None:
        """After building the index, a retrieval query should return results."""
        embedder = Embedder()
        store = ChromaStore(persist_dir=str(tmp_path / "queryable"))
        build_index(mock_kb_path, store, embedder, chunk_size=800, overlap=150)

        # Query about refunds — should return refund_policy chunks.
        query_vec = embedder.encode(["How do I get a refund?"])[0].tolist()
        results = store.query(query_vec, k=3)
        assert len(results) > 0, "Query must return results"
        # At least one result should mention refunds.
        texts = [r.chunk.text.lower() for r in results]
        assert any("refund" in t for t in texts), "Refund query should return refund-related chunks"
