"""Per-user session scratchpad (SQLite).

A chronological log of what happened in each turn — user messages, plans,
decisions, tool calls, blocks, summaries. The planner pulls the last few
entries into its prompt so a follow-up turn can reference an earlier one.

No vector store. SMB conversations are short, so semantic recall isn't
needed; a simple "recent N" is enough. RAG retrieval is a separate
component, not this memory.

Domain-agnostic: stores whatever redacted text the nodes pass in. Never store
raw PII here — the intake node redacts before anything reaches memory.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SessionScratchpad:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS scratchpad (
                       user_id TEXT NOT NULL,
                       turn_id TEXT NOT NULL,
                       seq INTEGER NOT NULL,
                       ts TEXT NOT NULL,
                       kind TEXT NOT NULL,
                       content TEXT NOT NULL,
                       PRIMARY KEY (user_id, turn_id, seq)
                   )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scratch_user_ts ON scratchpad(user_id, ts DESC)"
            )

    def append(self, user_id: str, turn_id: str, kind: str, content: str) -> None:
        # seq is per-(user,turn): the SELECT finds the current max seq for this
        # turn (or -1 if none yet) and adds 1, so entries are numbered 0,1,2...
        # Doing it in one statement avoids a read-then-write race.
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO scratchpad (user_id, turn_id, seq, ts, kind, content) "
                "SELECT ?, ?, COALESCE(MAX(seq), -1) + 1, datetime('now'), ?, ? "
                "FROM scratchpad WHERE user_id = ? AND turn_id = ?",
                (user_id, turn_id, kind, content, user_id, turn_id),
            )

    def recent(self, user_id: str, n: int = 5) -> list[dict]:
        # Fetch the newest n (DESC), then reverse so the caller gets them in
        # chronological order — easiest for the planner to read as a story.
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT kind, content, ts FROM scratchpad WHERE user_id = ? "
                "ORDER BY ts DESC, seq DESC LIMIT ?",
                (user_id, n),
            ).fetchall()
        return [dict(r) for r in rows][::-1]  # chronological order

    def get(self, user_id: str, turn_id: str) -> list[dict]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT kind, content, ts FROM scratchpad WHERE user_id = ? AND turn_id = ? "
                "ORDER BY seq",
                (user_id, turn_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def prune(self, user_id: str, older_than_days: int) -> int:
        with sqlite3.connect(self.path) as conn:
            cur = conn.execute(
                "DELETE FROM scratchpad WHERE user_id = ? AND "
                "ts < datetime('now', ?)",
                (user_id, f"-{older_than_days} days"),
            )
            return cur.rowcount

    def clear(self) -> None:
        """Delete all rows. Used between notebook scenarios to keep each trail clean.

        We delete rows rather than the file because Windows holds the SQLite
        # file handle open between connections and won't let us unlink it.
        """
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM scratchpad")