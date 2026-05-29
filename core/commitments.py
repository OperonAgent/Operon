"""
Operon Commitment Tracker — implicit promise detection and follow-up.

Adapted from OpenClaw src/commitments/types.ts.

When an agent says "I'll check back on that tomorrow" or "I'll remind you
before the meeting", it has made an implicit commitment.  This module extracts,
stores, and surfaces those commitments so they can be delivered as follow-up
messages.

Kinds:
  event_check_in   — "check back after X happens"
  deadline_check   — "remind you before Y deadline"
  care_check_in    — "check how you're doing with Z"
  open_loop        — vague promise to follow up

Sensitivity:
  routine   — scheduling, reminders
  personal  — health, personal goals
  care      — emotional support, wellbeing
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from core.sqlite_utils import configure_wal, reconcile_columns, jitter_write

_DB_DIR  = Path.home() / ".operon"
_DB_PATH = _DB_DIR / "commitments.db"

COMMITMENT_KINDS = {"event_check_in", "deadline_check", "care_check_in", "open_loop"}
COMMITMENT_SENSITIVITIES = {"routine", "personal", "care"}
COMMITMENT_STATUSES = {"pending", "sent", "dismissed", "snoozed", "expired"}

# Phrases that strongly suggest a commitment was made
_COMMITMENT_SIGNALS = [
    r"i(?:'ll| will) (?:check|follow up|remind|get back|ping|message) (?:you|back)",
    r"(?:let me|i(?:'ll| will)) know (?:how|when|if)",
    r"remind(?:er)? (?:for|about|to) (?:you|the)",
    r"check(?:ing)? (?:in|back) (?:after|before|on|tomorrow|next|in a)",
    r"follow(?:ing)? up (?:after|before|on|with)",
    r"(?:i(?:'ll| will)) (?:keep|stay) (?:you )?posted",
    r"i(?:'ll| will) (?:look|circle) back",
    r"don't forget (?:to|about|the)",
]
_COMMITMENT_RE = re.compile(
    "|".join(_COMMITMENT_SIGNALS),
    re.IGNORECASE,
)

_DDL = """
CREATE TABLE IF NOT EXISTS commitments (
    id               TEXT PRIMARY KEY,
    kind             TEXT NOT NULL DEFAULT 'open_loop',
    sensitivity      TEXT NOT NULL DEFAULT 'routine',
    status           TEXT NOT NULL DEFAULT 'pending',
    reason           TEXT NOT NULL DEFAULT '',
    suggested_text   TEXT NOT NULL DEFAULT '',
    dedupe_key       TEXT NOT NULL DEFAULT '',
    confidence       REAL NOT NULL DEFAULT 0.5,
    earliest_ms      REAL NOT NULL,
    latest_ms        REAL NOT NULL,
    timezone         TEXT NOT NULL DEFAULT 'UTC',
    session_key      TEXT NOT NULL DEFAULT '',
    agent_id         TEXT NOT NULL DEFAULT '',
    source_message   TEXT,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    attempts         INTEGER NOT NULL DEFAULT 0,
    last_attempt_at  REAL,
    sent_at          REAL,
    dismissed_at     REAL,
    snoozed_until    REAL,
    expired_at       REAL
);
CREATE INDEX IF NOT EXISTS idx_commit_status  ON commitments(status);
CREATE INDEX IF NOT EXISTS idx_commit_earliest ON commitments(earliest_ms);
CREATE INDEX IF NOT EXISTS idx_commit_session  ON commitments(session_key);
"""


class CommitmentTracker:
    """
    Extract, persist, and surface agent commitments.

    Typical flow::

        tracker = CommitmentTracker()
        new_commits = tracker.extract_from_exchange(user_text, assistant_text)
        pending = tracker.get_due(now_ms=time.time() * 1000)
        for c in pending:
            # deliver c["suggested_text"] to the user
            tracker.mark_sent(c["id"])
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
        return conn

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract_from_exchange(
        self,
        assistant_text: str,
        user_text:      str      = "",
        session_key:    str      = "",
        agent_id:       str      = "",
    ) -> list[dict]:
        """
        Detect commitments in an assistant response.
        Returns list of newly-created commitment dicts.
        """
        if not assistant_text:
            return []
        matches = _COMMITMENT_RE.findall(assistant_text)
        if not matches:
            return []

        now_ms   = time.time() * 1000
        created  = []
        for match in matches[:3]:   # cap at 3 per exchange
            dedupe = _make_dedupe_key(match)
            if self._exists_dedupe(dedupe, session_key):
                continue
            # Estimate due window: default to 24h from now
            earliest = now_ms + 3_600_000      # 1h
            latest   = now_ms + 86_400_000     # 24h
            kind     = _classify_kind(assistant_text)
            commit   = self.add(
                kind          = kind,
                reason        = match[:120],
                earliest_ms   = earliest,
                latest_ms     = latest,
                dedupe_key    = dedupe,
                suggested_text= _make_followup_text(assistant_text),
                session_key   = session_key,
                agent_id      = agent_id,
                source_message= user_text[:200] if user_text else None,
            )
            if commit:
                created.append(commit)
        return created

    def add(
        self,
        kind:           str,
        reason:         str,
        earliest_ms:    float,
        latest_ms:      float,
        dedupe_key:     str      = "",
        suggested_text: str      = "",
        sensitivity:    str      = "routine",
        confidence:     float    = 0.7,
        session_key:    str      = "",
        agent_id:       str      = "",
        source_message: Optional[str] = None,
    ) -> Optional[dict]:
        """Persist a new commitment. Returns the created record or None on dupe."""
        if kind not in COMMITMENT_KINDS:
            kind = "open_loop"
        if sensitivity not in COMMITMENT_SENSITIVITIES:
            sensitivity = "routine"
        cid = str(uuid.uuid4())[:12]
        now = time.time()
        def _write():
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO commitments (id, kind, sensitivity, status, reason, "
                    "suggested_text, dedupe_key, confidence, earliest_ms, latest_ms, "
                    "session_key, agent_id, source_message, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (cid, kind, sensitivity, "pending", reason, suggested_text,
                     dedupe_key, confidence, earliest_ms, latest_ms,
                     session_key, agent_id, source_message, now, now),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                pass   # dedupe_key collision — already exists
            finally:
                conn.close()
        jitter_write(_write)
        return self.get(cid)

    def get(self, commitment_id: str) -> Optional[dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM commitments WHERE id=?", (commitment_id,)
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d.setdefault("commitment_id", d.get("id", ""))
            return d
        finally:
            conn.close()

    def get_due(
        self,
        now_ms:      Optional[float] = None,
        session_key: Optional[str]   = None,
    ) -> list[dict]:
        """Return pending commitments whose earliest_ms has passed."""
        if now_ms is None:
            now_ms = time.time() * 1000
        conn = self._connect()
        try:
            wheres = ["status='pending'", "earliest_ms <= ?"]
            params: list = [now_ms]
            if session_key:
                wheres.append("session_key=?")
                params.append(session_key)
            rows = conn.execute(
                f"SELECT * FROM commitments WHERE {' AND '.join(wheres)} "
                "ORDER BY earliest_ms ASC LIMIT 20",
                params,
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d.setdefault("commitment_id", d.get("id", ""))
                result.append(d)
            return result
        finally:
            conn.close()

    def get_pending(self, session_key: Optional[str] = None) -> list[dict]:
        """Return all pending commitments regardless of due time."""
        conn = self._connect()
        try:
            if session_key:
                rows = conn.execute(
                    "SELECT * FROM commitments WHERE status='pending' AND session_key=? "
                    "ORDER BY earliest_ms ASC",
                    (session_key,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM commitments WHERE status='pending' "
                    "ORDER BY earliest_ms ASC"
                ).fetchall()
            # Add commitment_id alias for callers that use the longer key name
            result = []
            for r in rows:
                d = dict(r)
                d.setdefault("commitment_id", d.get("id", ""))
                result.append(d)
            return result
        finally:
            conn.close()

    def mark_sent(self, commitment_id: str) -> bool:
        return self._set_status(commitment_id, "sent", sent_at=time.time())

    def dismiss(self, commitment_id: str) -> bool:
        return self._set_status(commitment_id, "dismissed", dismissed_at=time.time())

    def snooze(self, commitment_id: str, until_ms: float) -> bool:
        return self._set_status(commitment_id, "snoozed", snoozed_until=until_ms)

    def expire_old(self, max_age_days: float = 30) -> int:
        cutoff = (time.time() - max_age_days * 86400) * 1000
        def _write():
            conn = self._connect()
            try:
                cur = conn.execute(
                    "UPDATE commitments SET status='expired', expired_at=? "
                    "WHERE status='pending' AND latest_ms < ?",
                    (time.time(), cutoff),
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()
        return jitter_write(_write) or 0

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _set_status(self, cid: str, status: str, **extra) -> bool:
        now = time.time()
        def _write():
            conn = self._connect()
            try:
                updates = ["status=?", "updated_at=?"]
                values  = [status, now]
                for k, v in extra.items():
                    updates.append(f"{k}=?"); values.append(v)
                values.append(cid)
                conn.execute(
                    f"UPDATE commitments SET {', '.join(updates)} WHERE id=?", values
                )
                conn.commit()
            finally:
                conn.close()
        jitter_write(_write)
        return True

    def _exists_dedupe(self, dedupe_key: str, session_key: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM commitments WHERE dedupe_key=? AND session_key=? "
                "AND status='pending'",
                (dedupe_key, session_key),
            ).fetchone()
            return row is not None
        finally:
            conn.close()


# ── Module-level helpers ────────────────────────────────────────────────────────

def _make_dedupe_key(signal: str) -> str:
    import hashlib
    return hashlib.sha256(signal.lower().strip().encode()).hexdigest()[:16]


def _classify_kind(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("meeting", "deadline", "due", "appointment", "exam")):
        return "deadline_check"
    if any(w in t for w in ("event", "happen", "launch", "release", "after")):
        return "event_check_in"
    if any(w in t for w in ("feel", "doing", "better", "okay", "recover", "health")):
        return "care_check_in"
    return "open_loop"


def _make_followup_text(assistant_text: str) -> str:
    # Extract first sentence as the basis for a follow-up message
    sentences = re.split(r'[.!?]', assistant_text)
    first = next((s.strip() for s in sentences if len(s.strip()) > 10), "")
    return f"Following up: {first[:100]}" if first else "Just checking in as promised."
