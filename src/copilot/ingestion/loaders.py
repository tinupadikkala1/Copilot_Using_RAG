"""Multi-format KB loaders. Each loader returns cleaned plain text."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable

from bs4 import BeautifulSoup
from pypdf import PdfReader

from copilot.schemas import RawDocument, SourceType

logger = logging.getLogger(__name__)


def _read_txt(path: Path) -> str:
    """Read a plain text file."""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_md(path: Path) -> str:
    """Read a Markdown file as plain text."""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_html(path: Path) -> str:
    """Parse an HTML file and extract clean text (strip scripts, styles)."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"Unreadable PDF: {path}") from exc
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_csv(path: Path) -> str:
    """Convert CSV rows into a pipe-delimited text representation."""
    rows: list[str] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh):
            rows.append(" | ".join(cell.strip() for cell in row))
    return "\n".join(rows)


# Mapping of file extensions to their loader functions.
LOADERS: dict[str, Callable[[Path], str]] = {
    ".txt": _read_txt,
    ".md": _read_md,
    ".html": _read_html,
    ".htm": _read_html,
    ".pdf": _read_pdf,
    ".csv": _read_csv,
}

_EXT_TO_TYPE: dict[str, SourceType] = {
    ".txt": SourceType.TXT,
    ".md": SourceType.MD,
    ".html": SourceType.HTML,
    ".htm": SourceType.HTML,
    ".pdf": SourceType.PDF,
    ".csv": SourceType.CSV,
}


def load_documents(root: Path) -> list[RawDocument]:
    """Load every supported file under ``root`` into RawDocument objects.

    Args:
        root: Directory tree to scan recursively for supported files.

    Returns:
        A list of RawDocument objects, one per successfully loaded file.
    """
    docs: list[RawDocument] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            logger.debug("Skipping unsupported file: %s", path)
            continue
        try:
            text = loader(path).strip()
        except ValueError:
            logger.exception("Failed to load %s", path)
            continue
        if not text:
            logger.warning("Empty document skipped: %s", path)
            continue
        docs.append(
            RawDocument(
                doc_id=path.stem,
                source_path=str(path),
                source_type=_EXT_TO_TYPE[path.suffix.lower()],
                title=path.stem.replace("_", " ").title(),
                text=text,
            )
        )
    logger.info("Loaded %d documents from %s", len(docs), root)
    return docs
