"""Shared, validated data contracts used across every layer."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    """Supported source document types."""

    MD = "md"
    HTML = "html"
    PDF = "pdf"
    TXT = "txt"
    CSV = "csv"


class RawDocument(BaseModel):
    """A parsed source document before chunking."""

    doc_id: str
    source_path: str
    source_type: SourceType
    title: str
    text: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """An embeddable unit of text with provenance for citations."""

    chunk_id: str
    doc_id: str
    title: str
    text: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    content_hash: str
    source_path: str

    @field_validator("content_hash")
    @classmethod
    def _hash_not_empty(cls, v: str) -> str:
        if len(v) != 64:
            raise ValueError("content_hash must be a 64-char sha256 hex digest")
        return v

    @staticmethod
    def make_hash(text: str) -> str:
        """Return the sha256 hex digest of normalized text."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class RetrievedChunk(BaseModel):
    """A chunk returned from the vector store with its similarity score."""

    chunk: Chunk
    score: float = Field(ge=0.0, le=1.0)  # cosine similarity normalized to [0,1]


class Citation(BaseModel):
    """An inline citation marker pointing to a source chunk."""

    marker: int = Field(ge=1)  # the [n] shown to the user
    chunk_id: str
    title: str
    source_path: str


class ChatResponse(BaseModel):
    """The complete response returned from the pipeline."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    intent: str
    escalated: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    session_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
