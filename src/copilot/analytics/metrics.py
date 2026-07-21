"""Record per-turn metrics and compute latency percentiles."""

from __future__ import annotations

import numpy as np

from copilot.analytics.db import get_connection
from copilot.schemas import ChatResponse


def record_turn(resp: ChatResponse, latency_ms: float) -> None:
    """Persist one completed turn to the analytics database.

    Args:
        resp: The ChatResponse returned to the user.
        latency_ms: End-to-end latency of the pipeline in milliseconds.
    """
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO turns (session_id, intent, escalated, confidence, latency_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                resp.session_id,
                resp.intent,
                int(resp.escalated),
                resp.confidence,
                latency_ms,
            ),
        )
        conn.commit()


def latency_percentiles() -> dict[str, float]:
    """Return the p50 and p95 latency (in ms) across all recorded turns.

    Returns:
        Dict with keys ``"p50"`` and ``"p95"``. Returns 0.0 for both
        if there is no data yet.
    """
    with get_connection() as conn:
        rows = [r[0] for r in conn.execute("SELECT latency_ms FROM turns")]
    if not rows:
        return {"p50": 0.0, "p95": 0.0}
    arr = np.array(rows)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }
