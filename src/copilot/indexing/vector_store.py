"""Backend-agnostic vector store. Default: Chroma (persistent, free, local)."""

from __future__ import annotations

import logging
from typing import Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings

from copilot.schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Protocol defining the vector store interface."""

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    def query(self, vector: list[float], k: int) -> list[RetrievedChunk]: ...
    def count(self) -> int: ...


class ChromaStore:
    """Persistent Chroma collection using cosine space.

    Args:
        persist_dir: Directory path for ChromaDB persistence.
        collection: Name of the collection to use.
    """

    def __init__(self, persist_dir: str, collection: str = "kb_index") -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaStore initialised: dir=%s collection=%s count=%d",
            persist_dir,
            collection,
            self.count(),
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or update chunks and their vectors.

        Args:
            chunks: List of Chunk objects to store.
            vectors: Corresponding embedding vectors of shape (n, dim).

        Raises:
            ValueError: If chunks and vectors length differ.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors length mismatch: " f"{len(chunks)} vs {len(vectors)}"
            )
        if not chunks:
            return

        self._col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "ordinal": c.ordinal,
                    "content_hash": c.content_hash,
                    "source_path": c.source_path,
                }
                for c in chunks
            ],
        )
        logger.info("Upserted %d chunks (total=%d)", len(chunks), self.count())

    def query(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        """Return the top-k most similar chunks to the query vector.

        Args:
            vector: Query embedding vector.
            k: Number of results to return.

        Returns:
            List of RetrievedChunk objects sorted by descending similarity.
        """
        res = self._col.query(query_embeddings=[vector], n_results=k)
        out: list[RetrievedChunk] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        for cid, text, meta, dist in zip(ids, docs, metas, dists):
            chunk = Chunk(
                chunk_id=cid,
                doc_id=str(meta["doc_id"]),
                title=str(meta["title"]),
                text=text,
                ordinal=int(meta["ordinal"]),
                content_hash=str(meta["content_hash"]),
                source_path=str(meta["source_path"]),
            )
            # Chroma cosine distance is in [0, 2]; convert to similarity [0, 1].
            score = max(0.0, min(1.0, 1.0 - dist / 2.0))
            out.append(RetrievedChunk(chunk=chunk, score=score))

        return out

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        return self._col.count()
