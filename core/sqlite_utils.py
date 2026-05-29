"""
Shared SQLite helpers for all Operon stores.

Ported / adapted from Hermes Agent hermes_state.py:
  - _reconcile_columns: declarative schema migration (no version table needed)
  - configure_wal: WAL mode with jitter checkpoint to avoid convoy pattern
"""

from __future__ import annotations

import json
import random
import sqlite3
import time


def reconcile_columns(conn: sqlite3.Connection, schema_sql: str) -> None:
    """
    Diff `schema_sql` against the live DB and ADD any missing columns.

    Algorithm:
    1. Create an in-memory DB and run schema_sql on it.
    2. For each table in schema_sql, compare PRAGMA table_info against the
       live DB's PRAGMA table_info.
    3. Issue ALTER TABLE ADD COLUMN for any column present in the schema but
       absent in the live DB.

    New columns inherit their DEFAULT from the schema definition.
    Renamed / removed columns are intentionally left alone (non-destructive).
    """
    mem = sqlite3.connect(":memory:")
    try:
        mem.executescript(schema_sql)
        for (table,) in mem.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall():
            schema_cols = {
                row[1]: row  # name → full row
                for row in mem.execute(f"PRAGMA table_info({table})").fetchall()
            }
            try:
                live_cols = {
                    row[1]
                    for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
            except Exception:
                continue  # table doesn't exist yet — CREATE IF NOT EXISTS will handle it
            for col_name, col_row in schema_cols.items():
                if col_name not in live_cols:
                    col_type    = col_row[2] or "TEXT"
                    col_notnull = col_row[3]
                    col_default = col_row[4]
                    default_clause = f" DEFAULT {col_default}" if col_default is not None else ""
                    not_null_clause = " NOT NULL" if col_notnull else ""
                    try:
                        conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN "
                            f"{col_name} {col_type}{not_null_clause}{default_clause}"
                        )
                    except Exception:
                        pass  # e.g. NOT NULL without DEFAULT on a populated table
    finally:
        mem.close()


_wal_warning_issued: set = set()   # deduplicate per (pid, db label)
_write_counters:    dict = {}       # db-path → write count


def configure_wal(conn: sqlite3.Connection, checkpoint_every: int = 50) -> None:
    """
    Enable WAL mode with checkpoint tracking and NFS/SMB fallback.

    - WAL mode dramatically reduces write contention for multi-thread use.
    - Falls back to DELETE journal on NFS/SMB where WAL locking fails,
      logging a deduplicated warning so the log is not spammed.
      (Adapted from Hermes Agent hermes_state.apply_wal_with_fallback.)
    - Every `checkpoint_every` writes calls PRAGMA wal_checkpoint(PASSIVE)
      to prevent unbounded WAL file growth.
    """
    import logging, os
    db_name = conn.execute("PRAGMA database_list").fetchone()
    db_label = db_name[2] if db_name else "unknown"
    try:
        result = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        if result and result[0].lower() != "wal":
            # WAL was rejected (e.g. NFS) — fall back to DELETE journal
            _key = (os.getpid(), db_label)
            if _key not in _wal_warning_issued:
                _wal_warning_issued.add(_key)
                logging.getLogger("operon.sqlite").warning(
                    "WAL mode unavailable for %s (NFS/read-only?); using DELETE journal.",
                    db_label,
                )
            conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass  # Proceed without WAL on any error


def _maybe_checkpoint(conn: sqlite3.Connection, checkpoint_every: int = 50) -> None:
    """
    Issue a passive WAL checkpoint every `checkpoint_every` writes.
    Call this from any write path to prevent WAL file from growing unbounded.
    """
    db_name  = conn.execute("PRAGMA database_list").fetchone()
    db_label = db_name[2] if db_name else "unknown"
    count    = _write_counters.get(db_label, 0) + 1
    _write_counters[db_label] = count
    if count % checkpoint_every == 0:
        try:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except Exception:
            pass


# ── Multimodal content sentinel ───────────────────────────────────────────────
# Adapted from Hermes Agent hermes_state.py.
# Prefix for storing list/dict content in a TEXT column without a separate
# BLOB column.  NUL byte cannot appear in normal text — unambiguous.

_CONTENT_JSON_PREFIX = "\x00json:"


def encode_content(value) -> str:
    """Encode a value for TEXT column storage.  str passes through; others get JSON prefix."""
    if isinstance(value, str):
        return value
    return _CONTENT_JSON_PREFIX + json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def decode_content(raw: str):
    """Decode a value stored with encode_content."""
    if isinstance(raw, str) and raw.startswith(_CONTENT_JSON_PREFIX):
        try:
            return json.loads(raw[len(_CONTENT_JSON_PREFIX):])
        except Exception:
            return raw
    return raw


def jitter_write(fn, max_retries: int = 15, min_ms: int = 20, max_ms: int = 150):
    """
    Execute `fn()` with random-jitter retry on SQLite OperationalError.

    Breaks the "convoy pattern" where all writers back off by the same fixed
    interval and re-collide.  Hermes uses 20-150ms random jitter.
    """
    last_exc = None
    for _ in range(max_retries):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            last_exc = e
            time.sleep(random.uniform(min_ms, max_ms) / 1000)
    raise last_exc
