"""Deterministic chunking with overlap, section-awareness, and deduplication."""

from __future__ import annotations

import logging
import re

from copilot.schemas import Chunk, RawDocument

logger = logging.getLogger(__name__)

_PARA_SPLIT = re.compile(r"\n\s*\n")

# Patterns that indicate section headings (Markdown, plain text, PDF-extracted).
# Simplified - matches common heading patterns.
_HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE),           # Markdown: ## Heading
    re.compile(r"^(\d+(?:\.\d+)*\.?)\s+([A-Z].+)$", re.MULTILINE),  # 1. Title, 1.1 Title
    re.compile(r"^([A-Z][A-Za-z\s]{2,50})\n\s*\n", re.MULTILINE),   # Title line followed by blank line
    re.compile(r"^(.+?)\n-{3,}$", re.MULTILINE),  # Title followed by underline
]


def _split_by_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_title, section_content) pairs.
    
    Uses simple, robust pattern matching that works well with PDF-extracted text.
    """
    # Try multiple pattern strategies and pick the best split
    strategies = [
        (re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE), lambda m: m.group(1)),
        (re.compile(r"^(\d+(?:\.\d+)*\.?)\s+([A-Z].+)$", re.MULTILINE), lambda m: m.group(2)),
        (re.compile(r"^([A-Z][A-Za-z\s]{3,40})\n\s*\n", re.MULTILINE), lambda m: m.group(1)),
    ]
    
    for pattern, title_extractor in strategies:
        matches = list(pattern.finditer(text))
        if len(matches) >= 2:
            # Found multiple sections with this pattern
            sections: list[tuple[str, str]] = []
            
            # Content before first section
            if matches[0].start() > 0:
                preamble = text[:matches[0].start()].strip()
                if preamble:
                    sections.append(("", preamble))
            
            # Each section
            for i, match in enumerate(matches):
                title = title_extractor(match).strip()
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                content = text[start:end].strip()
                if content:
                    sections.append((title, content))
            
            return sections if sections else [("", text)]
    
    # No detectable sections — return entire text as one section
    return [("", text)]


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
            overlapped.append(f"{tail} {cur}")
        chunks = overlapped
    return chunks


def chunk_document(doc: RawDocument, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Split a RawDocument into ordered Chunk objects with section awareness.

    First splits the document by detected sections (headings), then chunks
    each section separately. This ensures chunks don't mix content from
    different sections, and each chunk is prefixed with its section title
    for better retrieval accuracy.

    Args:
        doc: The document to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Overlap characters between consecutive chunks.

    Returns:
        A list of Chunk objects with provenance metadata.
    """
    sections = _split_by_sections(doc.text)

    chunks: list[Chunk] = []
    ordinal = 0

    for section_title, section_content in sections:
        # Prefix each chunk with its section title for context
        prefix = f"[Section: {section_title}]\n" if section_title else ""
        # Adjust chunk size to account for prefix
        effective_chunk_size = chunk_size - len(prefix)
        if effective_chunk_size < 100:
            effective_chunk_size = chunk_size  # Don't prefix if it would make chunks too small
            prefix = ""

        pieces = _split_text(section_content, effective_chunk_size, overlap)

        for piece in pieces:
            # Add section prefix to each chunk text
            chunk_text = f"{prefix}{piece}" if prefix else piece
            content_hash = Chunk.make_hash(chunk_text)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{ordinal}",
                    doc_id=doc.doc_id,
                    title=f"{doc.title} - {section_title}" if section_title else doc.title,
                    text=chunk_text,
                    ordinal=ordinal,
                    content_hash=content_hash,
                    source_path=doc.source_path,
                )
            )
            ordinal += 1

    # Fallback: if section detection produced nothing, use flat chunking
    if not chunks:
        pieces = _split_text(doc.text, chunk_size, overlap)
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
