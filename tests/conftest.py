"""Shared test fixtures and configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from copilot.ingestion.loaders import load_documents
from copilot.schemas import RawDocument, SourceType


@pytest.fixture(scope="session")
def mock_kb_path() -> Path:
    """Return the path to the mock KB fixture directory."""
    return Path(__file__).parent / "fixtures" / "mock_kb"


@pytest.fixture(scope="session")
def raw_docs(mock_kb_path: Path) -> list[RawDocument]:
    """Load all documents from the mock KB once per session."""
    return load_documents(mock_kb_path)


@pytest.fixture()
def single_doc() -> RawDocument:
    """Return a single RawDocument for isolated chunking tests."""
    return RawDocument(
        doc_id="test_doc",
        source_path="tests/fixtures/mock_kb/test_doc.md",
        source_type=SourceType.MD,
        title="Test Document",
        text=(
            "This is the first paragraph of the test document. "
            "It contains enough text to test chunking behavior.\n\n"
            "This is the second paragraph. It follows a double newline "
            "so the chunker should treat it as a separate paragraph.\n\n"
            "This is the third paragraph, which should be joined with "
            "the second paragraph if they fit within the chunk size. "
            "We want to make sure the paragraph-aware splitter works correctly."
        ),
    )
