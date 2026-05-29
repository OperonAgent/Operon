"""
Operon Memory Pipeline — V3 (SQLite + FTS5 full-text search).

Upgrades from V2 keyword scan to proper FTS5 full-text search:
  • Ranked relevance search via SQLite FTS5
  • Tag-based organisation and importance scores
  • Background async extraction (zero input lag)
  • Deduplication on first 60 chars
  • JSON migration on first run
"""

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

MEMORY_DB   = Path.home() / ".operon" / "memory.db"
MEMORY_JSON = Path.home() / ".operon" / "memory.json"

_PREF_PATTERNS = [
    r"(?:i\s+)?(?:always|prefer|like|want|need|use|hate)\s+(.+)",
    r"remember\s+(?:that\s+)?(.+)",
    r"my\s+(?:name|project|workspace|language|editor)\s+is\s+(.+)",
    r"(?:don'?t|never)\s+(.+)",
    r"set\s+(?:my\s+)?(\w[\w\s]+)\s+to\s+(.+)",
]
_FACT_PATTERNS = [
    r"(?:created?|wrote?|saved?|stored?)\s+(?:file\s+)?['\"]?(/[\w./-]+)['\"]?",
    r"(?:path|directory|folder|repo)\s*[=:]\s*['\"]?(/[\w./-]+)['\"]?",
]

# Main table + FTS5 virtual table for ranked search
_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    type         TEXT    NOT NULL DEFAULT 'fact',
    content      TEXT    NOT NULL,
    tags         TEXT    NOT NULL DEFAULT '',
    importance   INTEGER NOT NULL DEFAULT 3,
    created_at   REAL    NOT NULL,
    accessed_at  REAL    NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_type       ON memories(type);
CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
"""

_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(content, tags, content='memories', content_rowid='id');
"""

_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES('delete', old.id, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES('delete', old.id, old.content, old.tags);
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;
"""


def _now() -> float:
    return time.time()


class MemoryPipeline:

    def __init__(self, config):
        self._config = config
        MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)
        self._lock  = threading.Lock()
        self._fts_ok = False
        self._init_db()
        self._migrate_json()

    # ── DB setup ──────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(MEMORY_DB), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        from core.sqlite_utils import configure_wal, reconcile_columns
        with self._conn() as c:
            configure_wal(c)
            c.executescript(_DDL)
            reconcile_columns(c, _DDL)
            # Try to create FTS5 table (may not exist if DB was created before V3)
            try:
                c.executescript(_FTS_DDL)
                c.executescript(_FTS_TRIGGERS)
                self._fts_ok = True
            except Exception:
                self._fts_ok = False

        # Backfill FTS index from existing rows if needed
        if self._fts_ok:
            try:
                with self._conn() as c:
                    count = c.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
                    if count == 0:
                        c.execute(
                            "INSERT INTO memories_fts(rowid, content, tags) "
                            "SELECT id, content, tags FROM memories"
                        )
            except Exception:
                self._fts_ok = False

    def _migrate_json(self) -> None:
        if not MEMORY_JSON.exists():
            return
        try:
            data = json.loads(MEMORY_JSON.read_text())
            if not isinstance(data, list) or not data:
                return
            ts = _now()
            with self._lock:
                with self._conn() as c:
                    if c.execute("SELECT COUNT(*) FROM memories").fetchone()[0] > 0:
                        return
                    for item in data:
                        content = item.get("content", "").strip()
                        if content:
                            c.execute(
                                "INSERT INTO memories(type,content,tags,importance,"
                                "created_at,accessed_at) VALUES(?,?,?,?,?,?)",
                                (item.get("type", "legacy"), content, "", 3, ts, ts),
                            )
            MEMORY_JSON.rename(str(MEMORY_JSON) + ".migrated")
        except Exception:
            pass

    # ── Background extraction ─────────────────────────────────────────────────

    def async_evaluate_and_save(self, exchange: list[dict]) -> None:
        if not exchange:
            return
        combined = " ".join(m.get("content", "") for m in exchange)
        new_mems = self._extract(combined)
        if new_mems:
            ts = _now()
            with self._lock:
                with self._conn() as c:
                    for m in new_mems:
                        exists = c.execute(
                            "SELECT id FROM memories "
                            "WHERE LOWER(SUBSTR(content,1,60))=LOWER(SUBSTR(?,1,60))",
                            (m["content"],),
                        ).fetchone()
                        if not exists:
                            c.execute(
                                "INSERT INTO memories(type,content,tags,importance,"
                                "created_at,accessed_at) VALUES(?,?,?,?,?,?)",
                                (m["type"], m["content"], m.get("tags", ""),
                                 m.get("importance", 3), ts, ts),
                            )

    # ── Extraction helpers ────────────────────────────────────────────────────

    def _extract(self, text: str) -> list[dict]:
        found = []
        for pattern in _PREF_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                content = m.group(0).strip()
                if len(content) > 8:
                    found.append({"type": "preference", "content": content, "importance": 3})
        for pattern in _FACT_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                content = m.group(0).strip()
                if len(content) > 5:
                    found.append({"type": "fact", "content": content, "importance": 2})
        seen, unique = set(), []
        for item in found:
            key = item["content"][:60].lower()
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique[:8]

    # ── Public API ────────────────────────────────────────────────────────────

    def get_all(self) -> list[dict]:
        with self._lock:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT * FROM memories ORDER BY importance DESC, created_at DESC"
                ).fetchall()
                return [dict(r) for r in rows]

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """
        FTS5 ranked full-text search with keyword fallback.
        Returns results sorted by relevance score.
        """
        with self._lock:
            with self._conn() as c:
                # ── FTS5 ranked search ────────────────────────────────────────
                if self._fts_ok:
                    try:
                        rows = c.execute(
                            "SELECT m.*, rank "
                            "FROM memories m "
                            "JOIN memories_fts fts ON m.id = fts.rowid "
                            "WHERE memories_fts MATCH ? "
                            "ORDER BY rank "
                            "LIMIT ?",
                            (query, limit),
                        ).fetchall()
                        if rows:
                            return [dict(r) for r in rows]
                    except Exception:
                        pass   # fall through to keyword

                # ── Keyword fallback ──────────────────────────────────────────
                words   = query.lower().split()
                results = []
                for row in c.execute(
                    "SELECT * FROM memories ORDER BY importance DESC, accessed_at DESC"
                ).fetchall():
                    row_d = dict(row)
                    text  = (row_d["content"] + " " + row_d["tags"]).lower()
                    if any(w in text for w in words):
                        results.append(row_d)
                    if len(results) >= limit:
                        break
                return results

    def add_manual(self, content: str, tags: str = "", importance: int = 3,
                   mem_type: str = "manual") -> None:
        ts = _now()
        with self._lock:
            with self._conn() as c:
                exists = c.execute(
                    "SELECT id FROM memories "
                    "WHERE LOWER(SUBSTR(content,1,60))=LOWER(SUBSTR(?,1,60))",
                    (content,),
                ).fetchone()
                if not exists:
                    c.execute(
                        "INSERT INTO memories(type,content,tags,importance,"
                        "created_at,accessed_at) VALUES(?,?,?,?,?,?)",
                        (mem_type, content, tags, importance, ts, ts),
                    )

    def clear(self) -> None:
        with self._lock:
            with self._conn() as c:
                # Deleting from the content table fires the auto-sync triggers
                # that keep the FTS5 shadow table in sync — no need to touch
                # memories_fts directly (doing so on a content= table raises
                # an error in SQLite FTS5).
                c.execute("DELETE FROM memories")

    def delete_by_id(self, mem_id: int) -> None:
        with self._lock:
            with self._conn() as c:
                c.execute("DELETE FROM memories WHERE id=?", (mem_id,))

    def get_context_string(self) -> str:
        rows = self.get_all()
        top  = rows[:30]
        if not top:
            return ""
        lines = ["LONG-TERM MEMORIES (active context):"]
        for m in top:
            tag_str = f" [{m['tags']}]" if m.get("tags") else ""
            lines.append(f"  [{m['type'].upper()}{tag_str}] {m['content']}")
        return "\n".join(lines)

    @property
    def fts_enabled(self) -> bool:
        return self._fts_ok
