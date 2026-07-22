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


# Per-intent dynamic thresholds for escalation.
# These override the global defaults for specific intents.
# Tighter thresholds = more escalations (cautious).
# Looser thresholds = more auto-resolutions (confident).
_PER_INTENT_THRESHOLDS: dict[str, dict[str, float]] = {
    "billing": {
        "min_retrieval": 0.40,      # More cautious with money
        "min_groundedness": 0.65,
        "min_intent_conf": 0.40,
    },
    "human_agent": {
        # Human agent requests always escalate, so thresholds don't matter much.
        "min_retrieval": 0.0,
        "min_groundedness": 0.0,
        "min_intent_conf": 0.0,
    },
    "technical": {
        "min_retrieval": 0.30,       # More forgiving — technical issues vary widely
        "min_groundedness": 0.55,
        "min_intent_conf": 0.30,
    },
    "how_to": {
        "min_retrieval": 0.35,
        "min_groundedness": 0.60,
        "min_intent_conf": 0.35,
    },
    "account": {
        "min_retrieval": 0.35,
        "min_groundedness": 0.60,
        "min_intent_conf": 0.35,
    },
    "unknown": {
        "min_retrieval": 0.50,      # Unknown intents need strong retrieval
        "min_groundedness": 0.70,
        "min_intent_conf": 0.0,
    },
}


class EscalationConfig:
    """Holds escalation thresholds, with optional per-intent overrides."""

    def __init__(
        self,
        min_retrieval: float = 0.35,
        min_groundedness: float = 0.60,
        min_intent_conf: float = 0.35,
        per_intent: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.default_min_retrieval = min_retrieval
        self.default_min_groundedness = min_groundedness
        self.default_min_intent_conf = min_intent_conf
        self._per_intent = per_intent or _PER_INTENT_THRESHOLDS

    def for_intent(self, intent: str) -> dict[str, float]:
        """Return the effective thresholds for a given intent."""
        overrides = self._per_intent.get(intent, {})
        return {
            "min_retrieval": overrides.get("min_retrieval", self.default_min_retrieval),
            "min_groundedness": overrides.get("min_groundedness", self.default_min_groundedness),
            "min_intent_conf": overrides.get("min_intent_conf", self.default_min_intent_conf),
        }


# Singleton config — can be updated at runtime.
escalation_config = EscalationConfig()


def should_escalate(
    intent: str,
    intent_confidence: float,
    retrieval_score: float,
    groundedness: float,
    *,
    min_retrieval: float | None = None,
    min_groundedness: float | None = None,
    min_intent_conf: float | None = None,
) -> EscalationDecision:
    """Determine whether a query should be escalated to a human agent.

    Supports both global thresholds and per-intent overrides.
    Per-intent thresholds from ``escalation_config`` are used when the
    explicit keyword arguments are not provided.

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
        min_retrieval: Minimum acceptable retrieval score (uses per-intent if None).
        min_groundedness: Minimum acceptable groundedness score.
        min_intent_conf: Minimum acceptable intent confidence.

    Returns:
        An EscalationDecision with the result and reason.
    """
    # Resolve thresholds — prefer explicit args, fall back to per-intent, then defaults.
    intent_cfg = escalation_config.for_intent(intent)
    effective_min_retrieval = min_retrieval if min_retrieval is not None else intent_cfg["min_retrieval"]
    effective_min_groundedness = min_groundedness if min_groundedness is not None else intent_cfg["min_groundedness"]
    effective_min_intent_conf = min_intent_conf if min_intent_conf is not None else intent_cfg["min_intent_conf"]

    if intent == "human_agent":
        return EscalationDecision(True, "user_requested_human")

    if retrieval_score < effective_min_retrieval:
        return EscalationDecision(True, "low_retrieval_confidence")

    if groundedness < effective_min_groundedness:
        return EscalationDecision(True, "low_groundedness")

    if intent == "unknown" or intent_confidence < effective_min_intent_conf:
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
