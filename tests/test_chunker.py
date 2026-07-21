"""Unit tests for the chunker module."""

from __future__ import annotations

import pytest

from copilot.ingestion.chunker import _split_text, chunk_document, dedupe_chunks
from copilot.schemas import Chunk, RawDocument, SourceType


class TestSplitText:
    """Tests for the internal _split_text function."""

    def test_chunk_size_and_overlap(self) -> None:
        """Chunks respect the size boundary; overlap is present between consecutive chunks."""
        text = "word " * 500  # ~2500 chars
        chunks = _split_text(text, chunk_size=200, overlap=30)
        # After overlap is applied, chunks may be up to chunk_size + overlap.
        max_allowed = 200 + 30
        assert all(
            len(c) <= max_allowed for c in chunks
        ), f"All chunks must be ≤ chunk_size + overlap ({max_allowed})"
        if len(chunks) > 1:
            # The second chunk should contain the tail of the first (overlap).
            tail = chunks[0][-30:]
            assert tail in chunks[1], "Overlap content should appear in the next chunk"

    def test_split_respects_paragraphs(self) -> None:
        """Paragraph boundaries should be preferred split points."""
        text = "A" * 100 + "\n\n" + "B" * 500 + "\n\n" + "C" * 100
        chunks = _split_text(text, chunk_size=200, overlap=0)
        # The first chunk should be paragraph 1 alone since it fits.
        assert len(chunks[0]) <= 200
        # The long paragraph B should be hard-split.
        assert any("B" * 200 in c for c in chunks)

    def test_raises_on_invalid_params(self) -> None:
        """Invalid chunk_size or overlap should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_size"):
            _split_text("hello", chunk_size=0, overlap=0)

    def test_small_text_fits_single_chunk(self) -> None:
        """Text smaller than chunk_size should produce one chunk."""
        text = "Hello, world!"
        chunks = _split_text(text, chunk_size=800, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestChunkDocument:
    """Tests for the chunk_document function."""

    def test_chunks_have_valid_content_hash(self) -> None:
        """Every chunk should have a 64-character hex content_hash."""
        doc = RawDocument(
            doc_id="test",
            source_path="test.md",
            source_type=SourceType.MD,
            title="Test",
            text="Hello world " * 100,
        )
        chunks = chunk_document(doc, chunk_size=200, overlap=20)
        for c in chunks:
            assert len(c.content_hash) == 64, "content_hash must be 64 hex chars"
            assert c.content_hash == Chunk.make_hash(c.text), "Hash must match text"

    def test_ordinals_are_sequential(self) -> None:
        """Ordinal values should be 0, 1, 2, ..."""
        doc = RawDocument(
            doc_id="test",
            source_path="test.md",
            source_type=SourceType.MD,
            title="Test",
            text="word " * 500,
        )
        chunks = chunk_document(doc, chunk_size=200, overlap=0)
        ordinals = [c.ordinal for c in chunks]
        assert ordinals == list(range(len(chunks)))

    def test_chunk_id_format(self) -> None:
        """chunk_id should follow 'doc_id::ordinal' pattern."""
        doc = RawDocument(
            doc_id="refund_policy",
            source_path="refund.md",
            source_type=SourceType.MD,
            title="Refund Policy",
            text="word " * 500,
        )
        chunks = chunk_document(doc, chunk_size=300, overlap=30)
        for c in chunks:
            expected_prefix = f"{doc.doc_id}::"
            assert c.chunk_id.startswith(
                expected_prefix
            ), f"chunk_id '{c.chunk_id}' should start with '{expected_prefix}'"

    def test_single_doc_fixture(self, single_doc: RawDocument) -> None:
        """The single_doc fixture should chunk deterministically."""
        chunks = chunk_document(single_doc, chunk_size=400, overlap=50)
        assert len(chunks) >= 1, "Document must produce at least one chunk"
        for c in chunks:
            assert len(c.text) <= 400


class TestDedupeChunks:
    """Tests for the dedupe_chunks function."""

    def test_dedupe_removes_identical(self) -> None:
        """Two identical chunks should be deduped to one."""
        doc = RawDocument(
            doc_id="dup_test",
            source_path="dup.md",
            source_type=SourceType.MD,
            title="Duplicate Test",
            text="Hello world " * 100,
        )
        chunks_a = chunk_document(doc, chunk_size=200, overlap=0)
        chunks_b = chunk_document(doc, chunk_size=200, overlap=0)
        combined = chunks_a + chunks_b
        deduped = dedupe_chunks(combined)
        assert len(deduped) <= len(combined)
        # The first occurrence of each hash should be kept.
        hashes = [c.content_hash for c in combined]
        expected = []
        seen = set()
        for h in hashes:
            if h not in seen:
                expected.append(h)
                seen.add(h)
        assert [c.content_hash for c in deduped] == expected

    def test_dedup_preserves_order(self) -> None:
        """Deduplication should preserve the original insertion order."""
        text_a = "Alpha " * 50
        text_b = "Beta " * 50
        # Two different texts produce different hashes — no dedup.
        hash_a = Chunk.make_hash(text_a)
        hash_b = Chunk.make_hash(text_b)
        assert hash_a != hash_b, "Distinct texts must have distinct hashes"

        chunks = [
            Chunk(
                chunk_id="a::0",
                doc_id="a",
                title="A",
                text=text_a,
                ordinal=0,
                content_hash=hash_a,
                source_path="a.md",
            ),
            Chunk(
                chunk_id="b::0",
                doc_id="b",
                title="B",
                text=text_b,
                ordinal=0,
                content_hash=hash_b,
                source_path="b.md",
            ),
            Chunk(
                chunk_id="a::1",
                doc_id="a",
                title="A (dup)",
                text=text_a,
                ordinal=1,
                content_hash=hash_a,
                source_path="a.md",
            ),
        ]
        deduped = dedupe_chunks(chunks)
        assert len(deduped) == 2, "Only two unique chunks should remain"
        assert deduped[0].chunk_id == "a::0", "First occurrence should be kept"
        assert deduped[1].chunk_id == "b::0", "Second unique should be second"

    def test_dedupe_on_mock_kb(self, raw_docs: list[RawDocument]) -> None:
        """Deduplication on the mock KB should not remove any chunks
        (all are distinct documents)."""
        from copilot.ingestion.chunker import chunk_document

        all_chunks = []
        for doc in raw_docs:
            all_chunks.extend(chunk_document(doc, chunk_size=800, overlap=150))
        deduped = dedupe_chunks(all_chunks)
        assert len(deduped) == len(all_chunks), "All mock KB chunks should be unique"


class TestContentHash:
    """Tests for the Chunk.make_hash static method."""

    def test_content_hash_is_sha256(self) -> None:
        """content_hash must be a 64-character hex string (SHA-256)."""
        h = Chunk.make_hash("Hello, world!")
        assert len(h) == 64, "SHA-256 hex digest must be 64 characters"
        int(h, 16)  # Should not raise — must be valid hex

    def test_deterministic_hash(self) -> None:
        """Same text should produce the same hash every time."""
        text = "The quick brown fox jumps over the lazy dog."
        assert Chunk.make_hash(text) == Chunk.make_hash(text)

    def test_hash_empty_string(self) -> None:
        """An empty string should produce a valid hash (of empty content)."""
        h = Chunk.make_hash("")
        assert len(h) == 64


class TestLoaderIntegration:
    """Integration tests for loaders + chunker together."""

    def test_mock_kb_loads_three_docs(self, raw_docs: list) -> None:
        """The mock KB should produce exactly 3 RawDocuments."""
        assert len(raw_docs) == 3, "mock_kb has 3 files"

    def test_mock_kb_all_supported_types(self, raw_docs: list) -> None:
        """All loaded docs should be of type SourceType.MD."""
        from copilot.schemas import SourceType

        for doc in raw_docs:
            assert doc.source_type == SourceType.MD

    def test_chunk_mock_kb_produces_chunks(self, raw_docs: list) -> None:
        """Each document in the mock KB should produce at least one chunk."""
        for doc in raw_docs:
            chunks = chunk_document(doc, chunk_size=800, overlap=150)
            assert len(chunks) >= 1, f"Doc '{doc.doc_id}' must produce chunks"
