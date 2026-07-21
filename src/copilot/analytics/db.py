"""SQLite persistence for turns, feedback, and escalations."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from copilot.config import get_settings

_settings = get_settings()
DB_PATH = Path(_settings.db_path)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    intent TEXT NOT NULL,
    escalated INTEGER NOT NULL,
    confidence REAL NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating TEXT NOT NULL CHECK (rating IN ('up','down')),
    correction TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def init_db() -> None:
    """Ensure the database file and all required tables exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Context manager yielding a writable SQLite connection.

    Usage::

        with get_connection() as conn:
            conn.execute("SELECT ...")
            conn.commit()

    The connection is automatically closed after the ``with`` block.
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
