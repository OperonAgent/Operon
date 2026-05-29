"""
Operon Task Registry — lightweight task lifecycle tracker.

Adapted from OpenClaw src/tasks/task-registry.store.sqlite.ts.

Tracks sub-agent tasks, cron runs, and delegated work through their full
lifecycle with notify policies so callers can choose how much status noise
they want.

Statuses: queued → running → succeeded | failed | timed_out | cancelled | lost
Notify policies:
  done_only     — only report on terminal states (succeeded/failed/timed_out)
  state_changes — report every status transition
  silent        — never notify (fire-and-forget)
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from core.sqlite_utils import configure_wal, reconcile_columns, jitter_write

_DB_DIR  = Path.home() / ".operon"
_DB_PATH = _DB_DIR / "task_registry.db"

TASK_STATUSES  = {"queued", "running", "succeeded", "failed",
                  "timed_out", "cancelled", "lost"}
TERMINAL_STATUSES = {"succeeded", "failed", "timed_out", "cancelled", "lost"}
NOTIFY_POLICIES = {"done_only", "state_changes", "silent"}

_DDL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id          TEXT PRIMARY KEY,
    label            TEXT NOT NULL DEFAULT '',
    task             TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'queued',
    notify_policy    TEXT NOT NULL DEFAULT 'done_only',
    owner_key        TEXT NOT NULL DEFAULT '',
    created_at       REAL NOT NULL,
    started_at       REAL,
    ended_at         REAL,
    last_event_at    REAL,
    cleanup_after    REAL,
    error            TEXT,
    progress_summary TEXT,
    terminal_summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_tr_status     ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tr_owner_key  ON tasks(owner_key);
CREATE INDEX IF NOT EXISTS idx_tr_cleanup    ON tasks(cleanup_after);
"""


class TaskRegistry:
    """Per-process in-memory + SQLite task registry."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        path = db_path or _DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db   = str(path)
        self._lock = threading.Lock()
        self._init_db()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            configure_wal(conn)
            conn.executescript(_DDL)
            reconcile_columns(conn, _DDL)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db, check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Public API ─────────────────────────────────────────────────────────────

    def create(
        self,
        task:          str,
        label:         str         = "",
        notify_policy: str         = "done_only",
        owner_key:     str         = "",
        cleanup_after_hours: float = 24.0,
    ) -> str:
        """Create a task in 'queued' status. Returns task_id."""
        if notify_policy not in NOTIFY_POLICIES:
            notify_policy = "done_only"
        task_id     = str(uuid.uuid4())[:12]
        now         = time.time()
        cleanup_at  = now + cleanup_after_hours * 3600
        def _write():
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO tasks (task_id, label, task, status, notify_policy, "
                    "owner_key, created_at, last_event_at, cleanup_after) "
                    "VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?)",
                    (task_id, label, task, notify_policy, owner_key, now, now, cleanup_at),
                )
                conn.commit()
            finally:
                conn.close()
        jitter_write(_write)
        return task_id

    def transition(
        self,
        task_id:  str,
        status:   str,
        error:    Optional[str] = None,
        summary:  Optional[str] = None,
    ) -> bool:
        """Move a task to a new status. Returns True on success."""
        if status not in TASK_STATUSES:
            return False
        now = time.time()
        def _write():
            conn = self._connect()
            try:
                updates = ["status=?", "last_event_at=?"]
                values  = [status, now]
                if status == "running" and error is None:
                    updates.append("started_at=?"); values.append(now)
                if status in TERMINAL_STATUSES:
                    updates.append("ended_at=?"); values.append(now)
                if error is not None:
                    updates.append("error=?"); values.append(error)
                if summary:
                    key = "terminal_summary" if status in TERMINAL_STATUSES else "progress_summary"
                    updates.append(f"{key}=?"); values.append(summary)
                values.append(task_id)
                conn.execute(
                    f"UPDATE tasks SET {', '.join(updates)} WHERE task_id=?", values
                )
                conn.commit()
                return conn.execute(
                    "SELECT changes()"
                ).fetchone()[0] > 0
            finally:
                conn.close()
        return jitter_write(_write) or False

    def get(self, task_id: str) -> Optional[dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list(
        self,
        status:    Optional[str] = None,
        owner_key: Optional[str] = None,
        limit:     int           = 100,
    ) -> list[dict]:
        conn = self._connect()
        try:
            wheres, params = [], []
            if status:
                wheres.append("status=?"); params.append(status)
            if owner_key:
                wheres.append("owner_key=?"); params.append(owner_key)
            where_clause = f"WHERE {' AND '.join(wheres)}" if wheres else ""
            rows = conn.execute(
                f"SELECT * FROM tasks {where_clause} "
                f"ORDER BY created_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def should_notify(self, task_id: str, new_status: str) -> bool:
        """Return True if a status change should be surfaced to the user."""
        task = self.get(task_id)
        if not task:
            return False
        policy = task.get("notify_policy", "done_only")
        if policy == "silent":
            return False
        if policy == "done_only":
            return new_status in TERMINAL_STATUSES
        return True  # state_changes

    def cleanup_expired(self) -> int:
        """Delete tasks whose cleanup_after timestamp has passed."""
        now = time.time()
        def _write():
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM tasks WHERE cleanup_after IS NOT NULL "
                    "AND cleanup_after < ? AND status IN "
                    "('succeeded','cancelled','lost')",
                    (now,),
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()
        return jitter_write(_write) or 0
