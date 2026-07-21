"""Resolution reports: deflection rate and CSAT proxy."""

from __future__ import annotations

from copilot.analytics.db import get_connection


def deflection_rate() -> float:
    """Fraction of sessions resolved without human escalation.

    A session is considered "deflected" if no turn within it was escalated.
    Returns a value in [0, 1].
    """
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(DISTINCT session_id) FROM turns").fetchone()[0]
        escalated = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM turns WHERE escalated = 1"
        ).fetchone()[0]
    if total == 0:
        return 0.0
    return (total - escalated) / total


def csat() -> float:
    """Customer Satisfaction proxy: 👍 / (👍 + 👎).

    Returns a value in [0, 1], or 0.0 if no feedback has been collected.
    """
    with get_connection() as conn:
        up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='up'").fetchone()[0]
        down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='down'").fetchone()[0]
    denom = up + down
    return up / denom if denom else 0.0


def total_sessions() -> int:
    """Return the total number of unique sessions recorded."""
    with get_connection() as conn:
        result = conn.execute("SELECT COUNT(DISTINCT session_id) FROM turns").fetchone()
        return int(result[0]) if result else 0


def total_escalations() -> int:
    """Return the total number of open escalation tickets."""
    with get_connection() as conn:
        result = conn.execute("SELECT COUNT(*) FROM escalations WHERE status='open'").fetchone()
        return int(result[0]) if result else 0
