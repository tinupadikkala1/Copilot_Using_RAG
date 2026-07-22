"""DI wiring — cached singleton instances of pipeline components."""

from __future__ import annotations

import functools

from copilot.config import get_settings
from copilot.core.generation import LLMClient
from copilot.core.intent import IntentClassifier
from copilot.core.pipeline import SupportPipeline
from copilot.core.retriever import Retriever
from copilot.indexing.embedder import Embedder
from copilot.indexing.vector_store import ChromaStore


@functools.lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return a cached Embedder singleton."""
    return Embedder()


@functools.lru_cache(maxsize=1)
def get_vector_store() -> ChromaStore:
    """Return a cached ChromaStore singleton."""
    settings = get_settings()
    return ChromaStore(persist_dir=settings.persist_dir)


@functools.lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Return a cached Retriever singleton, populating its BM25 index
    from any chunks already in the vector store."""
    retriever = Retriever(store=get_vector_store(), embedder=get_embedder())
    # Populate BM25 index from existing chunks (if any).
    store = get_vector_store()
    chunks = store.get_all_chunks()
    if chunks:
        retriever.rebuild_bm25(chunks)
    return retriever


@functools.lru_cache(maxsize=1)
def get_intent_classifier() -> IntentClassifier:
    """Return a cached IntentClassifier singleton."""
    settings = get_settings()
    return IntentClassifier(
        embedder=get_embedder(),
        min_confidence=settings.min_intent_confidence,
    )


@functools.lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Return a cached LLMClient singleton."""
    return LLMClient()


@functools.lru_cache(maxsize=1)
def get_pipeline() -> SupportPipeline:
    """Return a cached SupportPipeline singleton."""
    settings = get_settings()
    return SupportPipeline(
        retriever=get_retriever(),
        intent_clf=get_intent_classifier(),
        llm=get_llm_client(),
        k=settings.retrieval_k,
        min_groundedness=settings.min_groundedness,
    )
