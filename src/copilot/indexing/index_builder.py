"""ETL orchestration: load -> chunk -> dedupe -> embed -> upsert.

Supports file-hash tracking to skip re-indexing unchanged files,
and an optional progress callback for real-time UI updates.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable

from copilot.indexing.embedder import Embedder
from copilot.indexing.vector_store import ChromaStore, VectorStore
from copilot.ingestion.chunker import chunk_document, dedupe_chunks
from copilot.ingestion.loaders import load_documents

logger = logging.getLogger(__name__)

# Name of the manifest file stored inside kb_root that tracks file hashes.
_HASH_MANIFEST = ".file_hashes.json"


# ---------------------------------------------------------------------------
#  File-hash helpers
# ---------------------------------------------------------------------------


def _compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_hash_manifest(kb_root: Path) -> dict[str, str]:
    """Load the file-hash manifest, or return an empty dict."""
    manifest_path = kb_root / _HASH_MANIFEST
    if manifest_path.exists():
        try:
            return dict(json.loads(manifest_path.read_text()))
        except Exception:
            logger.warning("Corrupt hash manifest, rebuilding from scratch")
    return {}


def _save_hash_manifest(kb_root: Path, manifest: dict[str, str]) -> None:
    """Persist the file-hash manifest."""
    manifest_path = kb_root / _HASH_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.debug("Hash manifest saved (%d entries)", len(manifest))


# ---------------------------------------------------------------------------
#  Index builder
# ---------------------------------------------------------------------------


def build_index(
    kb_root: Path,
    store: VectorStore,
    embedder: Embedder,
    chunk_size: int = 800,
    overlap: int = 150,
    batch_size: int = 128,
    *,
    retriever=None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> int:
    """Build/refresh the vector index.

    Loads documents from ``kb_root``, chunks them, deduplicates,
    embeds in batches, and upserts into the vector store.

    If the file-hash manifest shows that **no files have changed** since
    the last successful build, the entire build is skipped and the
    current chunk count is returned immediately.

    Args:
        kb_root: Directory tree containing KB documents.
        store: Vector store instance (e.g. ChromaStore).
        embedder: Embedder instance wrapping the Ollama embedding model.
        chunk_size: Maximum characters per chunk.
        overlap: Overlap characters between consecutive chunks.
        batch_size: Number of chunks to embed and upsert per batch.
        retriever: Optional Retriever instance. If provided, the BM25 index
                   is refreshed after the vector store is updated.
        progress_callback: Optional callable ``fn(current, total, phase)``
                           invoked after each batch is embedded. **current**
                           is the batch index (1-based), **total** is the
                           total number of batches, and **phase** is a
                           short human-readable description.

    Returns:
        Number of chunks indexed (or existing chunks if nothing changed).
    """
    docs = load_documents(kb_root)

    # --- Compute current file hashes ---
    current_hashes: dict[str, str] = {}
    for doc in docs:
        fpath = doc.source_path
        if fpath:
            p = Path(fpath)
            if p.exists():
                current_hashes[p.name] = _compute_file_hash(p)

    # --- Check if anything changed since last build ---
    manifest = _load_hash_manifest(kb_root)
    if current_hashes == manifest and store.count() > 0:
        logger.info(
            "No file changes detected. Skipping rebuild (%d chunks exist).",
            store.count(),
        )
        return store.count()

    # --- Full rebuild ---
    all_chunks = [c for d in docs for c in chunk_document(d, chunk_size, overlap)]
    unique = dedupe_chunks(all_chunks)

    if not unique:
        logger.warning("No chunks to index under %s", kb_root)
        _save_hash_manifest(kb_root, current_hashes)
        return 0

    num_batches = (len(unique) + batch_size - 1) // batch_size

    for i in range(0, len(unique), batch_size):
        batch = unique[i : i + batch_size]
        batch_idx = i // batch_size

        if progress_callback:
            progress_callback(
                batch_idx + 1,
                num_batches,
                f"Embedding batch {batch_idx + 1}/{num_batches} ({len(batch)} chunks)",
            )

        vectors = embedder.encode([c.text for c in batch]).tolist()
        store.upsert(batch, vectors)

    # Refresh BM25 index if a retriever was provided.
    if retriever is not None and hasattr(store, "get_all_chunks"):
        try:
            all_stored = store.get_all_chunks()
            retriever.rebuild_bm25(all_stored)
            logger.info("BM25 index refreshed with %d chunks", len(all_stored))
        except Exception:
            logger.exception("Failed to refresh BM25 index")

    # Save the hash manifest so future builds can skip unchanged files.
    _save_hash_manifest(kb_root, current_hashes)

    if progress_callback:
        progress_callback(num_batches, num_batches, f"Done — {len(unique)} chunks indexed")

    logger.info("Index build complete: %d chunks", len(unique))
    return len(unique)
