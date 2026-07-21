"""Integration tests for the SupportPipeline.

Tests that require Ollama are marked with ``@pytest.mark.skipif``.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import httpx
import pytest

from copilot.core.generation import LLMClient
from copilot.core.intent import IntentClassifier
from copilot.core.pipeline import SupportPipeline
from copilot.core.retriever import Retriever
from copilot.indexing.embedder import Embedder
from copilot.indexing.index_builder import build_index
from copilot.indexing.vector_store import ChromaStore


def ollama_available() -> bool:
    """Check whether Ollama is running and has the required models."""
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = resp.json().get("models", [])
        names = [m.get("name", "") for m in models]
        has_embed = any("nomic-embed-text" in n for n in names)
        has_llm = any("qwen" in n for n in names)
        return has_embed and has_llm
    except (httpx.HTTPError, ConnectionError):
        return False


ollama_ready = ollama_available()


class TestPipeline:
    """Integration tests for the full pipeline."""

    @pytest.mark.skipif(not ollama_ready, reason="Ollama not ready")
    def test_pipeline_answers_greeting(self, mock_kb_path: Path, tmp_path: Path) -> None:
        """Greeting queries should return the greeting reply without escalation."""
        embedder = Embedder()
        store = ChromaStore(persist_dir=str(tmp_path / "pipeline_greeting"))
        build_index(mock_kb_path, store, embedder, chunk_size=800, overlap=150)

        retriever = Retriever(store, embedder)
        intent_clf = IntentClassifier(embedder)
        llm = LLMClient()
        pipeline = SupportPipeline(retriever, intent_clf, llm)

        resp = pipeline.answer_query("Hello!", session_id=uuid.uuid4().hex)
        assert not resp.escalated
        assert (
            "Hi!" in resp.answer
            or "hello" in resp.answer.lower()
            or "support" in resp.answer.lower()
        )

    @pytest.mark.skipif(not ollama_ready, reason="Ollama not ready")
    def test_pipeline_handles_rag_query(self, mock_kb_path: Path, tmp_path: Path) -> None:
        """A KB-backed query should return a cited answer."""
        embedder = Embedder()
        store = ChromaStore(persist_dir=str(tmp_path / "pipeline_rag"))
        build_index(mock_kb_path, store, embedder, chunk_size=800, overlap=150)

        retriever = Retriever(store, embedder)
        intent_clf = IntentClassifier(embedder)
        llm = LLMClient()
        pipeline = SupportPipeline(retriever, intent_clf, llm)

        resp = pipeline.answer_query(
            "How do I reset my password?",
            session_id=uuid.uuid4().hex,
        )
        assert not resp.escalated
        assert len(resp.answer) > 0
        # Should either have citations or mention reset/password.
        has_citations = len(resp.citations) > 0
        has_keywords = any(
            kw in resp.answer.lower() for kw in ["reset", "password", "forgot", "login"]
        )
        assert (
            has_citations or has_keywords
        ), f"Expected password-related answer, got: {resp.answer[:100]}"

    @pytest.mark.skipif(not ollama_ready, reason="Ollama not ready")
    def test_pipeline_escalates_human_request(self, mock_kb_path: Path, tmp_path: Path) -> None:
        """Requesting a human agent should result in escalation with a ticket."""
        embedder = Embedder()
        store = ChromaStore(persist_dir=str(tmp_path / "pipeline_esc"))
        build_index(mock_kb_path, store, embedder, chunk_size=800, overlap=150)

        retriever = Retriever(store, embedder)
        intent_clf = IntentClassifier(embedder)
        llm = LLMClient()
        pipeline = SupportPipeline(retriever, intent_clf, llm)

        resp = pipeline.answer_query(
            "I want to talk to a real person",
            session_id=uuid.uuid4().hex,
        )
        assert resp.escalated
        assert "ticket" in resp.answer.lower() or "specialist" in resp.answer.lower()
