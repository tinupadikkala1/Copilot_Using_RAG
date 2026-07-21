"""Integration tests for the FastAPI server.

Tests use the TestClient from Starlette/FastAPI so no server needs to be
running. Tests that require Ollama are marked as usual.
"""

from __future__ import annotations

import os

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
    def test_chat_valid_key_returns_response_shape(
        self, client: TestClient
    ) -> None:
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
