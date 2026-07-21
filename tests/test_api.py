"""Integration tests for the FastAPI server.

Tests use the TestClient from Starlette/FastAPI so no server needs to be
running. Tests that require Ollama are marked as usual.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from copilot.serving.api import create_app

# Set a test API key for the test session.
os.environ["COPILOT_API_KEY"] = "test-api-key-123"
os.environ["COPILOT_OLLAMA_BASE_URL"] = "http://localhost:11434"


def ollama_available() -> bool:
    """Check whether Ollama is running."""
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except (httpx.HTTPError, ConnectionError):
        return False


@pytest.fixture()
def client() -> TestClient:
    """Return a FastAPI TestClient with fresh app instance."""
    app = create_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------


class TestLanding:
    """Tests for the root landing page."""

    def test_landing_returns_identity(self, client: TestClient) -> None:
        """The root route should return project identity."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "project_topic" in data
        assert "full_name" in data
        assert "registered_email" in data
        assert data["project_topic"] == "Autonomous Customer Support Copilot"

    def test_healthz(self, client: TestClient) -> None:
        """The health check should return ok."""
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    """Tests for the /chat endpoint."""

    def test_chat_requires_api_key(self, client: TestClient) -> None:
        """A request without API key should return 401 or 503."""
        resp = client.post("/chat", json={"message": "Hello"})
        assert resp.status_code in (401, 503)

    def test_chat_with_invalid_key(self, client: TestClient) -> None:
        """An invalid API key should return 401."""
        resp = client.post(
            "/chat",
            headers={"x-api-key": "wrong-key"},
            json={"message": "Hello"},
        )
        assert resp.status_code == 401

    def test_chat_validates_message_length(self, client: TestClient) -> None:
        """An empty message should be rejected."""
        resp = client.post(
            "/chat",
            headers={"x-api-key": "test-api-key-123"},
            json={"message": ""},
        )
        assert resp.status_code == 422

    @pytest.mark.skipif(not ollama_available(), reason="Ollama not running")
    def test_chat_valid_key_returns_response_shape(self, client: TestClient) -> None:
        """A valid request should return a ChatResponse-shaped JSON."""
        resp = client.post(
            "/chat",
            headers={"x-api-key": "test-api-key-123"},
            json={"message": "Hello there"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "intent" in data
        assert "session_id" in data
        assert "escalated" in data


# ---------------------------------------------------------------------------
# Feedback endpoint
# ---------------------------------------------------------------------------


class TestFeedbackEndpoint:
    """Tests for the /feedback endpoint."""

    def test_feedback_requires_api_key(self, client: TestClient) -> None:
        """Feedback without API key should return 401 or 503."""
        resp = client.post(
            "/feedback",
            json={
                "session_id": "test",
                "query": "Hello",
                "answer": "Hi!",
                "rating": "up",
            },
        )
        assert resp.status_code in (401, 503)

    def test_feedback_validates_rating(self, client: TestClient) -> None:
        """Invalid rating values should be rejected."""
        resp = client.post(
            "/feedback",
            headers={"x-api-key": "test-api-key-123"},
            json={
                "session_id": "test",
                "query": "Hello",
                "answer": "Hi!",
                "rating": "invalid",
            },
        )
        assert resp.status_code == 422

    def test_feedback_success(self, client: TestClient) -> None:
        """Valid feedback should return 'recorded'."""
        resp = client.post(
            "/feedback",
            headers={"x-api-key": "test-api-key-123"},
            json={
                "session_id": "test-session",
                "query": "How do I reset?",
                "answer": "Click forgot password.",
                "rating": "up",
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "recorded"}


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    def test_metrics_requires_api_key(self, client: TestClient) -> None:
        """Metrics without API key should return 401 or 503."""
        resp = client.get("/metrics")
        assert resp.status_code in (401, 503)

    def test_metrics_returns_shape(self, client: TestClient) -> None:
        """A valid metrics request should return the expected fields."""
        resp = client.get(
            "/metrics",
            headers={"x-api-key": "test-api-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "deflection_rate" in data
        assert "csat" in data
        assert "p50" in data
        assert "p95" in data


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------


class TestUploadEndpoint:
    """Tests for the /upload and /upload/build endpoints."""

    def test_upload_requires_api_key(self, client: TestClient) -> None:
        """Upload without API key should return 401 or 503."""
        resp = client.post("/upload")
        assert resp.status_code in (401, 503)

    def test_upload_requires_files(self, client: TestClient) -> None:
        """Upload without files should return 422."""
        resp = client.post(
            "/upload",
            headers={"x-api-key": "test-api-key-123"},
        )
        assert resp.status_code == 422

    def test_upload_valid_file(self, client: TestClient, tmp_path) -> None:
        """Upload a valid .md file should succeed."""
        # Patch the KB_RAW path to use tmp_path for test isolation.
        import copilot.serving.api as api

        original_kb = api.KB_RAW
        api.KB_RAW = tmp_path / "kb_raw"
        api.KB_RAW.mkdir(parents=True, exist_ok=True)

        try:
            content = b"# Test\nThis is a test document for uploading."
            resp = client.post(
                "/upload",
                headers={"x-api-key": "test-api-key-123"},
                files={"files": ("test_doc.md", content, "text/markdown")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["total_saved"] == 1
            assert len(data["errors"]) == 0
            assert data["saved"][0]["filename"] == "test_doc.md"

            # Verify file was actually saved.
            saved_path = api.KB_RAW / "test_doc.md"
            assert saved_path.exists()
            assert saved_path.read_bytes() == content
        finally:
            api.KB_RAW = original_kb

    def test_upload_rejects_unsupported_format(self, client: TestClient) -> None:
        """Uploading an unsupported file format should report an error."""
        resp = client.post(
            "/upload",
            headers={"x-api-key": "test-api-key-123"},
            files={"files": ("bad_file.exe", b"fake exe content", "application/octet-stream")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_saved"] == 0
        assert data["total_errors"] == 1
        assert "Unsupported format" in data["errors"][0]["error"]

    def test_upload_empty_file_rejected(self, client: TestClient, tmp_path) -> None:
        """Uploading an empty file should report an error."""
        import copilot.serving.api as api

        original_kb = api.KB_RAW
        api.KB_RAW = tmp_path / "kb_empty"
        api.KB_RAW.mkdir(parents=True, exist_ok=True)
        try:
            resp = client.post(
                "/upload",
                headers={"x-api-key": "test-api-key-123"},
                files={"files": ("empty.md", b"", "text/markdown")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_saved"] == 0
            assert "Empty file" in data["errors"][0]["error"]
        finally:
            api.KB_RAW = original_kb

    def test_upload_multiple_files(self, client: TestClient, tmp_path) -> None:
        """Uploading multiple valid files should save all of them."""
        import copilot.serving.api as api

        original_kb = api.KB_RAW
        api.KB_RAW = tmp_path / "kb_multi"
        api.KB_RAW.mkdir(parents=True, exist_ok=True)
        try:
            resp = client.post(
                "/upload",
                headers={"x-api-key": "test-api-key-123"},
                files=[
                    ("files", ("doc1.md", b"# Doc 1", "text/markdown")),
                    ("files", ("doc2.txt", b"Doc 2 content", "text/plain")),
                    ("files", ("doc3.csv", b"a,b,c\n1,2,3", "text/csv")),
                ],
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_saved"] == 3
            assert data["total_errors"] == 0
        finally:
            api.KB_RAW = original_kb

    def test_build_requires_api_key(self, client: TestClient) -> None:
        """Build index without API key should return 401 or 503."""
        resp = client.post("/upload/build")
        assert resp.status_code in (401, 503)
