"""Tests for the escalation logic."""

from __future__ import annotations

from copilot.core.escalation import should_escalate


class TestShouldEscalate:
    """Tests for the escalation decision logic."""

    def test_user_requested_human(self) -> None:
        """If intent is 'human_agent', it should always escalate."""
        decision = should_escalate(
            intent="human_agent",
            intent_confidence=0.9,
            retrieval_score=0.9,
            groundedness=0.9,
        )
        assert decision.escalate
        assert decision.reason == "user_requested_human"

    def test_low_retrieval(self) -> None:
        """Low retrieval score should trigger escalation."""
        decision = should_escalate(
            intent="how_to",
            intent_confidence=0.8,
            retrieval_score=0.2,
            groundedness=0.8,
            min_retrieval=0.35,
        )
        assert decision.escalate
        assert decision.reason == "low_retrieval_confidence"

    def test_low_groundedness(self) -> None:
        """Low groundedness score should trigger escalation."""
        decision = should_escalate(
            intent="how_to",
            intent_confidence=0.8,
            retrieval_score=0.9,
            groundedness=0.3,
            min_groundedness=0.60,
        )
        assert decision.escalate
        assert decision.reason == "low_groundedness"

    def test_low_intent_confidence(self) -> None:
        """Low intent confidence should trigger escalation."""
        decision = should_escalate(
            intent="unknown",
            intent_confidence=0.2,
            retrieval_score=0.5,
            groundedness=0.7,
            min_intent_conf=0.35,
        )
        assert decision.escalate
        assert decision.reason == "low_intent_confidence"

    def test_sensitive_intent_low_groundedness(self) -> None:
        """Billing intent with groundedness under 0.75 should escalate."""
        decision = should_escalate(
            intent="billing",
            intent_confidence=0.8,
            retrieval_score=0.7,
            groundedness=0.6,
        )
        assert decision.escalate
        assert decision.reason == "sensitive_intent_needs_review"

    def test_auto_resolved(self) -> None:
        """Normal high-confidence query should auto-resolve."""
        decision = should_escalate(
            intent="how_to",
            intent_confidence=0.85,
            retrieval_score=0.7,
            groundedness=0.85,
        )
        assert not decision.escalate
        assert decision.reason == "auto_resolved"

    def test_technical_intent_high_confidence_auto_resolve(self) -> None:
        """Technical intent with all signals high should auto-resolve."""
        decision = should_escalate(
            intent="technical",
            intent_confidence=0.9,
            retrieval_score=0.8,
            groundedness=0.9,
        )
        assert not decision.escalate
        assert decision.reason == "auto_resolved"

    def test_edge_case_boundaries(self) -> None:
        """Test exact threshold boundaries."""
        # Exactly at min_retrieval — should not escalate by retrieval alone.
        decision = should_escalate(
            intent="how_to",
            intent_confidence=0.8,
            retrieval_score=0.35,
            groundedness=0.8,
            min_retrieval=0.35,
        )
        assert not decision.escalate
