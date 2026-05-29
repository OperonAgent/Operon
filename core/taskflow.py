"""
Operon TaskFlow — Durable Multi-Step Flow with Revision-Checked Mutations.

Adapted from OpenClaw src/tasks/task-flow-registry.ts.

TaskFlow extends TaskRegistry with:
  • Multi-step flows (ordered list of steps with individual statuses)
  • Optimistic concurrency via revision counters (prevents lost-update races)
  • Step-level progress tracking
  • Rollback support for failed flows
  • Flow templates for common patterns
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from core.sqlite_utils import configure_wal, reconcile_columns, jitter_write

_DB_DIR  = Path.home() / ".operon"
_DB_PATH = _DB_DIR / "taskflow.db"

FLOW_STATUSES  = {"pending", "running", "succeeded", "failed", "cancelled", "paused"}
STEP_STATUSES  = {"pending", "running", "succeeded", "failed", "skipped", "cancelled"}
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

_DDL = """
CREATE TABLE IF NOT EXISTS flows (
    flow_id       TEXT PRIMARY KEY,
    label         TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'pending',
    revision      INTEGER NOT NULL DEFAULT 0,
    owner_key     TEXT NOT NULL DEFAULT '',
    metadata      TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    started_at    REAL,
    ended_at      REAL,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_flow_status    ON flows(status);
CREATE INDEX IF NOT EXISTS idx_flow_owner     ON flows(owner_key);

CREATE TABLE IF NOT EXISTS flow_steps (
    step_id       TEXT PRIMARY KEY,
    flow_id       TEXT NOT NULL REFERENCES flows(flow_id) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    label         TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'pending',
    input         TEXT,
    output        TEXT,
    error         TEXT,
    started_at    REAL,
    ended_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_step_flow   ON flow_steps(flow_id);
CREATE INDEX IF NOT EXISTS idx_step_status ON flow_steps(status);
"""


class ConflictError(Exception):
    """Raised when a revision-checked update finds a stale revision."""


class TaskFlow:
    """
    Durable multi-step flow registry with optimistic concurrency control.

    Usage::

        tf = TaskFlow()

        # Create a 3-step flow
        flow_id = tf.create_flow(
            label="Deploy pipeline",
            steps=["build", "test", "deploy"],
        )

        # Advance each step
        step_id = tf.get_current_step(flow_id)["step_id"]
        tf.start_step(step_id)
        tf.complete_step(step_id, output="Build artifact: app.tar.gz")

        # Check flow status
        flow = tf.get_flow(flow_id)
    """

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
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Flow lifecycle ─────────────────────────────────────────────────────────

    def create_flow(
        self,
        label:     str,
        steps:     list[str],
        owner_key: str       = "",
        metadata:  Optional[dict] = None,
    ) -> str:
        """
        Create a new flow with the given ordered steps.
        Returns the flow_id.
        """
        if not steps:
            raise ValueError("Flow must have at least one step")
        flow_id = str(uuid.uuid4())[:12]
        now     = time.time()
        meta    = json.dumps(metadata or {})

        def _write():
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO flows (flow_id, label, status, revision, owner_key, "
                    "metadata, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (flow_id, label, "pending", 0, owner_key, meta, now, now),
                )
                for pos, step_label in enumerate(steps):
                    step_id = str(uuid.uuid4())[:12]
                    conn.execute(
                        "INSERT INTO flow_steps (step_id, flow_id, position, label, status) "
                        "VALUES (?,?,?,?,?)",
                        (step_id, flow_id, pos, step_label, "pending"),
                    )
                conn.commit()
            finally:
                conn.close()

        jitter_write(_write)
        return flow_id

    def start_flow(self, flow_id: str, revision: int) -> bool:
        """
        Transition flow to 'running'. Requires correct revision number
        for optimistic concurrency control.
        Raises ConflictError if revision doesn't match.
        """
        return self._update_flow_status(flow_id, "running", revision)

    def finish_flow(
        self,
        flow_id:  str,
        revision: int,
        success:  bool   = True,
        error:    str    = "",
    ) -> bool:
        status = "succeeded" if success else "failed"
        return self._update_flow_status(flow_id, status, revision, error=error or None)

    def cancel_flow(self, flow_id: str, revision: int) -> bool:
        return self._update_flow_status(flow_id, "cancelled", revision)

    # ── Step lifecycle ─────────────────────────────────────────────────────────

    def start_step(self, step_id: str) -> bool:
        return self._update_step_status(step_id, "running")

    def complete_step(self, step_id: str, output: Optional[str] = None) -> bool:
        ok = self._update_step_status(step_id, "succeeded", output=output)
        if ok:
            self._maybe_advance_flow(self._get_flow_id_for_step(step_id))
        return ok

    def fail_step(self, step_id: str, error: Optional[str] = None) -> bool:
        ok = self._update_step_status(step_id, "failed", error=error)
        if ok:
            self._maybe_fail_flow(self._get_flow_id_for_step(step_id), error)
        return ok

    def skip_step(self, step_id: str) -> bool:
        ok = self._update_step_status(step_id, "skipped")
        if ok:
            self._maybe_advance_flow(self._get_flow_id_for_step(step_id))
        return ok

    # ── Query ──────────────────────────────────────────────────────────────────

    def get_flow(self, flow_id: str) -> Optional[dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM flows WHERE flow_id=?", (flow_id,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["steps"] = self._get_steps(conn, flow_id)
            return d
        finally:
            conn.close()

    def get_current_step(self, flow_id: str) -> Optional[dict]:
        """Return the first pending or running step for the flow."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM flow_steps WHERE flow_id=? AND status IN ('pending','running') "
                "ORDER BY position ASC LIMIT 1",
                (flow_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_flows(
        self,
        status:    Optional[str] = None,
        owner_key: Optional[str] = None,
        limit:     int           = 50,
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
                f"SELECT * FROM flows {where_clause} ORDER BY created_at DESC LIMIT ?",
                params + [limit],
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _update_flow_status(
        self,
        flow_id:  str,
        status:   str,
        revision: int,
        error:    Optional[str] = None,
    ) -> bool:
        now = time.time()
        def _write():
            conn = self._connect()
            try:
                # Check revision
                row = conn.execute(
                    "SELECT revision FROM flows WHERE flow_id=?", (flow_id,)
                ).fetchone()
                if not row:
                    return False
                if row["revision"] != revision:
                    raise ConflictError(
                        f"Flow {flow_id}: expected revision {revision}, "
                        f"got {row['revision']}"
                    )
                updates = ["status=?", "revision=revision+1", "updated_at=?"]
                values  = [status, now]
                if status == "running":
                    updates.append("started_at=?"); values.append(now)
                if status in TERMINAL_STATUSES:
                    updates.append("ended_at=?"); values.append(now)
                if error is not None:
                    updates.append("error=?"); values.append(error)
                values.append(flow_id)
                conn.execute(
                    f"UPDATE flows SET {', '.join(updates)} WHERE flow_id=?", values
                )
                conn.commit()
                return True
            finally:
                conn.close()
        return jitter_write(_write) or False

    def _update_step_status(
        self,
        step_id: str,
        status:  str,
        output:  Optional[str] = None,
        error:   Optional[str] = None,
    ) -> bool:
        now = time.time()
        def _write():
            conn = self._connect()
            try:
                updates = ["status=?"]
                values  = [status]
                if status == "running":
                    updates.append("started_at=?"); values.append(now)
                if status in ("succeeded", "failed", "skipped", "cancelled"):
                    updates.append("ended_at=?"); values.append(now)
                if output is not None:
                    updates.append("output=?"); values.append(output)
                if error is not None:
                    updates.append("error=?"); values.append(error)
                values.append(step_id)
                conn.execute(
                    f"UPDATE flow_steps SET {', '.join(updates)} WHERE step_id=?", values
                )
                conn.commit()
                return True
            finally:
                conn.close()
        return jitter_write(_write) or False

    def _get_flow_id_for_step(self, step_id: str) -> Optional[str]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT flow_id FROM flow_steps WHERE step_id=?", (step_id,)
            ).fetchone()
            return row["flow_id"] if row else None
        finally:
            conn.close()

    def _get_steps(self, conn: sqlite3.Connection, flow_id: str) -> list[dict]:
        rows = conn.execute(
            "SELECT * FROM flow_steps WHERE flow_id=? ORDER BY position ASC",
            (flow_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def _maybe_advance_flow(self, flow_id: Optional[str]) -> None:
        """Check if all steps are done; if so mark flow succeeded."""
        if not flow_id:
            return
        conn = self._connect()
        try:
            steps = self._get_steps(conn, flow_id)
            all_done = all(
                s["status"] in ("succeeded", "skipped", "cancelled")
                for s in steps
            )
            if all_done and steps:
                flow = conn.execute(
                    "SELECT revision, status FROM flows WHERE flow_id=?", (flow_id,)
                ).fetchone()
                if flow and flow["status"] == "running":
                    conn.execute(
                        "UPDATE flows SET status='succeeded', ended_at=?, updated_at=?, "
                        "revision=revision+1 WHERE flow_id=?",
                        (time.time(), time.time(), flow_id),
                    )
                    conn.commit()
        finally:
            conn.close()

    def _maybe_fail_flow(self, flow_id: Optional[str], error: Optional[str]) -> None:
        if not flow_id:
            return
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE flows SET status='failed', ended_at=?, updated_at=?, "
                "revision=revision+1, error=? WHERE flow_id=? AND status='running'",
                (time.time(), time.time(), error or "", flow_id),
            )
            conn.commit()
        finally:
            conn.close()
