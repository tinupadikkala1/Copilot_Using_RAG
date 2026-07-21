#!/usr/bin/env python3
"""CLI entrypoint: build (or refresh) the vector index from a KB directory.

Usage:
    PYTHONPATH=src python scripts/build_index.py [--kb-root data/kb_raw] [--persist-dir data/chroma]

All flags are optional; defaults come from settings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from copilot.indexing.embedder import Embedder
from copilot.indexing.index_builder import build_index
from copilot.indexing.vector_store import ChromaStore
from copilot.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Build/refresh the vector index from KB documents."
    )
    parser.add_argument(
        "--kb-root",
        default="data/kb_raw",
        help="Directory containing KB documents (default: data/kb_raw)",
    )
    parser.add_argument(
        "--persist-dir",
        default="data/chroma",
        help="ChromaDB persistence directory (default: data/chroma)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Characters per chunk (default: 800)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Overlap characters between chunks (default: 150)",
    )
    args = parser.parse_args()

    kb_root = Path(args.kb_root)
    if not kb_root.exists():
        logger.error("KB root does not exist: %s", kb_root)
        raise SystemExit(1)

    embedder = Embedder()
    store = ChromaStore(persist_dir=args.persist_dir)

    count = build_index(
        kb_root=kb_root,
        store=store,
        embedder=embedder,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    logger.info("Indexing complete: %d chunks indexed.", count)


if __name__ == "__main__":
    main()
