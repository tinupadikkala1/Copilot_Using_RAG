"""Feedback capture. Feedback recycles into the offline eval set and few-shot examples."""

from __future__ import annotations

import logging
from typing import Literal

from copilot.analytics.db import get_connection

logger = logging.getLogger(__name__)

Rating = Literal["up", "down"]


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
    logger.info("Recorded feedback session=%s rating=%s", session_id, rating)
