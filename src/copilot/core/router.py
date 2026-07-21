"""Query router: choose the handler based on intent and confidence."""

from __future__ import annotations

import logging
from enum import Enum

from copilot.core.intent import SENSITIVE_INTENTS

logger = logging.getLogger(__name__)


class Route(str, Enum):
    """The possible routes for a query."""

    RAG = "rag"  # Answer via retrieval-augmented generation.
    ESCALATE = "escalate"  # Immediately escalate to a human agent.
    SMALLTALK = "smalltalk"  # Handle with a simple greeting / chitchat.


def route(
    intent: str,
    intent_confidence: float,
    retrieval_score: float,
    *,
    min_intent_conf: float = 0.35,
    min_retrieval: float = 0.35,
) -> Route:
    """Decide the route for a query based on intent and retrieval signals.

    Args:
        intent: Predicted intent label (from IntentClassifier).
        intent_confidence: Confidence of the intent prediction [0, 1].
        retrieval_score: Top retrieval similarity score [0, 1].
        min_intent_conf: Minimum confidence to trust the intent label.
        min_retrieval: Minimum retrieval score for a RAG answer.

    Returns:
        The appropriate Route for the query.
    """
    if intent == "greeting":
        logger.debug("Routing: greeting -> SMALLTALK")
        return Route.SMALLTALK

    if intent == "human_agent":
        logger.debug("Routing: human_agent -> ESCALATE")
        return Route.ESCALATE

    if intent == "unknown" or intent_confidence < min_intent_conf:
        logger.debug(
            "Routing: low intent confidence (%.2f) -> ESCALATE",
            intent_confidence,
        )
        return Route.ESCALATE

    if retrieval_score < min_retrieval:
        logger.debug(
            "Routing: low retrieval score (%.2f) -> ESCALATE",
            retrieval_score,
        )
        return Route.ESCALATE

    if intent in SENSITIVE_INTENTS and retrieval_score < 0.6:
        logger.debug("Routing: sensitive intent + low retrieval -> ESCALATE")
        return Route.ESCALATE

    logger.debug("Routing: %s -> RAG", intent)
    return Route.RAG
