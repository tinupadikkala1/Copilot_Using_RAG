"""Tests for the query router."""

from __future__ import annotations

from copilot.core.router import Route, route


class TestRoute:
    """Tests for the route function."""

    def test_greeting_routes_to_smalltalk(self) -> None:
        """Greeting intents should route to SMALLTALK."""
        r = route("greeting", 0.9, 0.5)
        assert r == Route.SMALLTALK

    def test_human_agent_routes_to_escalate(self) -> None:
        """human_agent intents should route to ESCALATE."""
        r = route("human_agent", 0.9, 0.5)
        assert r == Route.ESCALATE

    def test_unknown_routes_to_escalate(self) -> None:
        """Unknown intents should route to ESCALATE."""
        r = route("unknown", 0.3, 0.5)
        assert r == Route.ESCALATE

    def test_low_confidence_routes_to_escalate(self) -> None:
        """Low-confidence predictions should route to ESCALATE."""
        r = route("billing", 0.2, 0.7, min_intent_conf=0.35)
        assert r == Route.ESCALATE

    def test_low_retrieval_routes_to_escalate(self) -> None:
        """Low retrieval scores should route to ESCALATE."""
        r = route("how_to", 0.8, 0.2, min_retrieval=0.35)
        assert r == Route.ESCALATE

    def test_normal_query_routes_to_rag(self) -> None:
        """A normal high-confidence query should route to RAG."""
        r = route("how_to", 0.8, 0.7)
        assert r == Route.RAG

    def test_sensitive_intent_low_retrieval_escalates(self) -> None:
        """Billing with retrieval_score < 0.6 should escalate."""
        r = route("billing", 0.8, 0.4)
        assert r == Route.ESCALATE

    def test_sensitive_intent_good_retrieval_routes_to_rag(self) -> None:
        """Billing with retrieval_score >= 0.6 should go to RAG."""
        r = route("billing", 0.8, 0.7)
        assert r == Route.RAG
