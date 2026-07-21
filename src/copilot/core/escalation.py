"""Auto-escalation logic and human-queue ticket creation."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from copilot.analytics.db import get_connection
from copilot.core.intent import SENSITIVE_INTENTS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EscalationDecision:
    """The result of an escalation check."""

    escalate: bool
    reason: str


def should_escalate(
    intent: str,
    intent_confidence: float,
    retrieval_score: float,
    groundedness: float,
    *,
    min_retrieval: float = 0.35,
    min_groundedness: float = 0.60,
    min_intent_conf: float = 0.35,
) -> EscalationDecision:
    """Determine whether a query should be escalated to a human agent.

    Escalation triggers:
    - User explicitly requests a human agent.
    - Retrieval confidence is too low.
    - Generated answer is not well-grounded in the context.
    - Intent is unknown or low-confidence.
    - Sensitive intents (billing) with insufficient grounding.

    Args:
        intent: Predicted intent label.
        intent_confidence: Confidence of the intent prediction.
        retrieval_score: Top retrieval similarity score.
        groundedness: Groundedness score of the generated answer.
        min_retrieval: Minimum acceptable retrieval score.
        min_groundedness: Minimum acceptable groundedness score.
        min_intent_conf: Minimum acceptable intent confidence.

    Returns:
        An EscalationDecision with the result and reason.
    """
    if intent == "human_agent":
        return EscalationDecision(True, "user_requested_human")

    if retrieval_score < min_retrieval:
        return EscalationDecision(True, "low_retrieval_confidence")

    if groundedness < min_groundedness:
        return EscalationDecision(True, "low_groundedness")

    if intent == "unknown" or intent_confidence < min_intent_conf:
        return EscalationDecision(True, "low_intent_confidence")

    if intent in SENSITIVE_INTENTS and groundedness < 0.75:
        return EscalationDecision(True, "sensitive_intent_needs_review")

    return EscalationDecision(False, "auto_resolved")


def create_ticket(session_id: str, query: str, reason: str) -> str:
    """Persist an escalation ticket to the human queue.

    Args:
        session_id: The conversation session identifier.
        query: The user's query that triggered escalation.
        reason: The escalation reason.

    Returns:
        The generated ticket ID (e.g. ``"ESC-abc123def456"``).
    """
    ticket_id = f"ESC-{uuid.uuid4().hex[:12]}"
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO escalations (ticket_id, session_id, query, reason, status) "
            "VALUES (?, ?, ?, ?, 'open')",
            (ticket_id, session_id, query, reason),
        )
        conn.commit()
    logger.info("Created escalation %s (reason=%s)", ticket_id, reason)
    return ticket_id
