"""Feedback capture. Feedback recycles into the offline eval set and few-shot examples."""

from __future__ import annotations

import logging
from typing import Literal

from copilot.analytics.db import get_connection
from copilot.core.intent import INTENT_EXAMPLES

logger = logging.getLogger(__name__)

Rating = Literal["up", "down"]


# Track candidate queries for intent expansion.
# In production, this would be persisted; for now it's an in-memory accumulator.
_feedback_intent_candidates: list[dict] = []


def record_feedback(
    session_id: str,
    query: str,
    answer: str,
    rating: Rating,
    correction: str | None = None,
) -> None:
    """Persist a user rating and optional correction for continuous improvement.

    The feedback loop works as:
    1. 👎 + correction rows are exported by the eval script into a regression set.
    2. High-quality 👍 Q->answer pairs become few-shot exemplars.
    3. Analytics dashboards track week-over-week CSAT drift.
    4. 👎 ratings from known intents add the query as a new seed example
       for retraining the intent classifier (requires manual review).

    Args:
        session_id: The conversation session identifier.
        query: The original user query.
        answer: The answer that was given.
        rating: ``"up"`` for thumbs-up, ``"down"`` for thumbs-down.
        correction: Optional free-text correction from the user.
    """
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO feedback (session_id, query, answer, rating, correction) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, query, answer, rating, correction),
        )
        conn.commit()

    # Track 👎 feedback for potential intent retraining.
    if rating == "down":
        _feedback_intent_candidates.append({
            "query": query,
            "session_id": session_id,
            "correction": correction,
        })
        if len(_feedback_intent_candidates) >= 10:
            logger.info(
                "Accumulated %d negative feedback entries — consider reviewing "
                "and adding representative queries to INTENT_EXAMPLES",
                len(_feedback_intent_candidates),
            )

    logger.info("Recorded feedback session=%s rating=%s", session_id, rating)


def get_feedback_candidates(min_rating: str = "down", limit: int = 50) -> list[dict]:
    """Retrieve feedback entries for offline analysis and intent retraining.

    Args:
        min_rating: Minimum rating filter ("up" or "down").
        limit: Maximum number of entries to return.

    Returns:
        List of feedback dicts with keys: session_id, query, answer, rating, correction.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT session_id, query, answer, rating, correction "
            "FROM feedback WHERE rating = ? ORDER BY id DESC LIMIT ?",
            (min_rating, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_positive_examples(limit: int = 20) -> list[dict]:
    """Get top 👍 Q&A pairs for use as few-shot exemplars.

    Args:
        limit: Maximum number of examples to return.

    Returns:
        List of dicts with keys: query, answer.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT query, answer FROM feedback "
            "WHERE rating='up' AND answer IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"query": r[0], "answer": r[1]} for r in rows]
