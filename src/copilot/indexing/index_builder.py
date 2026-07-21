"""ETL orchestration: load -> chunk -> dedupe -> embed -> upsert."""

from __future__ import annotations

import logging
from pathlib import Path

from copilot.indexing.embedder import Embedder
from copilot.indexing.vector_store import VectorStore
from copilot.ingestion.chunker import chunk_document, dedupe_chunks
from copilot.ingestion.loaders import load_documents

logger = logging.getLogger(__name__)


def build_index(
    kb_root: Path,
    store: VectorStore,
    embedder: Embedder,
    chunk_size: int = 800,
    overlap: int = 150,
    batch_size: int = 128,
) -> int:
    """Build/refresh the vector index.

    Loads documents from ``kb_root``, chunks them, deduplicates,
    embeds in batches, and upserts into the vector store.

    Args:
        kb_root: Directory tree containing KB documents.
        store: Vector store instance (e.g. ChromaStore).
        embedder: Embedder instance wrapping the Ollama embedding model.
        chunk_size: Maximum characters per chunk.
        overlap: Overlap characters between consecutive chunks.
        batch_size: Number of chunks to embed and upsert per batch.

    Returns:
        Number of chunks indexed.
    """
    docs = load_documents(kb_root)
    all_chunks = [c for d in docs for c in chunk_document(d, chunk_size, overlap)]
    unique = dedupe_chunks(all_chunks)

    if not unique:
        logger.warning("No chunks to index under %s", kb_root)
        return 0

    for i in range(0, len(unique), batch_size):
        batch = unique[i : i + batch_size]
        vectors = embedder.encode([c.text for c in batch]).tolist()
        store.upsert(batch, vectors)

    logger.info("Index build complete: %d chunks", len(unique))
    return len(unique)
