"""Deterministic chunking with overlap and content-hash deduplication."""

from __future__ import annotations

import logging
import re

from copilot.schemas import Chunk, RawDocument

logger = logging.getLogger(__name__)

_PARA_SPLIT = re.compile(r"\n\s*\n")


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedy paragraph-aware splitter with fixed-size overlap fallback.

    Args:
        text: Input text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunks.

    Raises:
        ValueError: If chunk_size <= 0 or overlap is out of range.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")

    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= chunk_size:
            buffer = f"{buffer}\n{para}".strip()
        else:
            if buffer:
                chunks.append(buffer)
            # Hard-split paragraphs longer than chunk_size.
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[chunk_size - overlap :]
            buffer = para
    if buffer:
        chunks.append(buffer)

    # Apply overlap between consecutive chunks.
    if overlap and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev[-overlap:]
            overlapped.append(f"{tail}{cur}")
        chunks = overlapped
    return chunks


def chunk_document(doc: RawDocument, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Split a RawDocument into ordered Chunk objects.

    Args:
        doc: The document to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Overlap characters between consecutive chunks.

    Returns:
        A list of Chunk objects with provenance metadata.
    """
    pieces = _split_text(doc.text, chunk_size, overlap)
    chunks: list[Chunk] = []
    for ordinal, piece in enumerate(pieces):
        content_hash = Chunk.make_hash(piece)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{ordinal}",
                doc_id=doc.doc_id,
                title=doc.title,
                text=piece,
                ordinal=ordinal,
                content_hash=content_hash,
                source_path=doc.source_path,
            )
        )
    return chunks


def dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks with duplicate content_hash, keeping the first occurrence.

    Args:
        chunks: List of Chunk objects, possibly containing duplicates.

    Returns:
        A deduplicated list preserving insertion order.
    """
    seen: set[str] = set()
    unique: list[Chunk] = []
    for c in chunks:
        if c.content_hash in seen:
            continue
        seen.add(c.content_hash)
        unique.append(c)
    logger.info("Deduplicated %d -> %d chunks", len(chunks), len(unique))
    return unique
