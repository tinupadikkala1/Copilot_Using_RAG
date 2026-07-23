"""Deterministic chunking with overlap, section-awareness, and deduplication."""

from __future__ import annotations

import logging
import re

from copilot.schemas import Chunk, RawDocument

logger = logging.getLogger(__name__)

_PARA_SPLIT = re.compile(r"\n\s*\n")

# Patterns that indicate section headers (Markdown, plain text, PDF-extracted).
_HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+.+", re.MULTILINE),           # Markdown: ## Heading
    re.compile(r"^[A-Z][A-Za-z\s]{2,50}:?\s*$", re.MULTILINE),  # ALL-CAPS or Title Case lines
    re.compile(r"^\d+\.\s+[A-Z].+", re.MULTILINE),       # Numbered: 1. Section Title
]


def _detect_current_section(text: str, position: int) -> str:
    """Find the nearest section heading before the given position."""
    lines = text[:position].split("\n")
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Check if this looks like a heading
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        # Title-case line that's short (likely a heading)
        if (len(stripped) < 80 and stripped[0].isupper() and
            not stripped.endswith(".") and not stripped.endswith(",")):
            # Check if it looks like a section title
            words = stripped.split()
            if len(words) <= 8 and all(w[0].isupper() or w.lower() in ("and", "or", "the", "of", "in", "to", "for", "a", "an", "with") for w in words if w):
                return stripped
    return ""


def _split_by_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_title, section_content) pairs.

    Detects headings and groups content under them. Content before
    the first heading gets an empty section title.
    """
    # Try to split by markdown headings first
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_re.finditer(text))

    if not matches:
        # Try numbered sections: "1. Title" or "1.1 Title"
        heading_re = re.compile(r"^(\d+(?:\.\d+)*\.?\s+)([A-Z].{2,60})$", re.MULTILINE)
        matches = list(heading_re.finditer(text))

    if not matches:
        # No detectable sections — return entire text as one section
        return [("", text)]

    sections: list[tuple[str, str]] = []

    # Content before first heading
    if matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    # Each heading starts a section that runs until the next heading
    for i, match in enumerate(matches):
        title = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else match.group(0).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append((title, content))

    return sections


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
