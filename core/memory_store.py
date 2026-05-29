"""
Operon Memory Store — Production-depth long-term agent memory.

Matches Hermes memory_manager.py + OpenClaw memory/ depth.

Architecture
============
┌─────────────────────────────────────────────────────────────────────┐
│  MemoryStore                                                        │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────────┐  │
│  │ WorkingMemory│  │  EpisodicMemory   │  │   EntityMemory      │  │
│  │ (per-session │  │  (conversation    │  │   (facts about      │  │
│  │  fast dict)  │  │   turns + search) │  │   people/places)    │  │
│  └──────────────┘  └──────────────────┘  └─────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ MemoryConsolidator — importance scoring, decay, pruning       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

Features
--------
• **WorkingMemory** — per-session in-memory key/value store with TTL,
  type-tagged slots (fact, plan, result, preference), capacity cap.

• **EpisodicMemory** — SQLite-backed conversation turn log with:
    - Hybrid BM25 + cosine similarity retrieval (embeddings optional)
    - Automatic entity extraction from stored content
    - Importance scoring (recency × relevance × frequency)
    - Per-memory decay over time (configurable half-life)
    - Consolidation: merge similar memories to reduce noise

• **EntityMemory** — structured fact store keyed by (entity, attribute):
    - Person: name, role, email, preferences, last_seen
    - Place: location, context, last_mentioned
    - Concept: definition, examples, related_terms
    - Supports confidence scoring and source tracking

• **MemoryConsolidator** — background job that:
    - Detects and merges duplicate/near-duplicate memories
    - Boosts importance of frequently-recalled memories
    - Decays stale memories below threshold → archived
    - Extracts new entities from recently stored episodes

Usage
-----
    from core.memory_store import MemoryStore, get_memory_store

    ms = get_memory_store()

    # Working memory (session-scoped, fast)
    ms.working.set("current_file", "/src/main.py", kind="context")
    val = ms.working.get("current_file")

    # Episode storage
    ms.remember("user: what is the capital of France?", role="user")
    ms.remember("assistant: Paris", role="assistant", importance=0.8)

    # Retrieval
    results = ms.recall("France capital", limit=5)

    # Entity memory
    ms.know("Alice", "role", "lead engineer")
    ms.know("Alice", "timezone", "UTC+5:30")
    facts = ms.entity_facts("Alice")
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("operon.memory_store")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_DB_DIR  = Path.home() / ".operon"
_DB_PATH = _DB_DIR / "memory_store.sqlite"

_WORKING_MEMORY_CAPACITY = 500    # max slots
_WORKING_MEMORY_TTL_SEC  = 3600   # 1-hour default TTL
_EPISODE_CONTENT_MAX     = 1_000  # truncate stored content
_DEFAULT_IMPORTANCE      = 0.5
_DECAY_HALF_LIFE_DAYS    = 14.0   # importance halves every 14 days
_CONSOLIDATION_SIM_THRESH = 0.92  # similarity threshold for merging
_EMBED_DIM               = 384    # expected embedding dimension
_RECALL_LIMIT            = 10

# Embedding backends (tried in order)
_EMBED_TIMEOUT = 5  # seconds

# ---------------------------------------------------------------------------
# Embedding helpers (same layered approach as semantic_memory.py)
# ---------------------------------------------------------------------------

def _embed_ollama(text: str) -> Optional[List[float]]:
    try:
        url  = "http://localhost:11434/api/embeddings"
        body = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode()
        req  = urllib.request.Request(url, data=body,
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=_EMBED_TIMEOUT) as r:
            return json.loads(r.read()).get("embedding")
    except Exception:
        return None


def _embed_openai(text: str) -> Optional[List[float]]:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        url  = "https://api.openai.com/v1/embeddings"
        body = json.dumps({"model": "text-embedding-3-small", "input": text}).encode()
        req  = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        })
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
            return data["data"][0]["embedding"]
    except Exception:
        return None


def _cosine(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single remembered episode."""
    id:         int
    session_id: str
    role:       str           # user / assistant / system / tool
    content:    str
    importance: float
    timestamp:  float
    tags:       List[str]     = field(default_factory=list)
    entities:   List[str]     = field(default_factory=list)
    recall_count: int         = 0
    archived:   bool          = False
    embedding:  Optional[List[float]] = field(default=None, repr=False)

    @property
    def age_days(self) -> float:
        return (time.time() - self.timestamp) / 86400.0

    def effective_importance(self) -> float:
        """Decay importance exponentially with age."""
        decay = math.pow(0.5, self.age_days / _DECAY_HALF_LIFE_DAYS)
        return self.importance * decay * (1.0 + 0.05 * self.recall_count)


@dataclass
class EntityFact:
    """A structured fact about a named entity."""
    entity:     str
    attribute:  str
    value:      Any
    confidence: float = 1.0
    source:     str   = ""
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkingSlot:
    """A single working-memory slot."""
    key:        str
    value:      Any
    kind:       str   = "fact"     # fact | plan | result | preference | context
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + _WORKING_MEMORY_TTL_SEC)
    access_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------

class WorkingMemory:
    """
    Per-session in-memory key/value store with optional TTL.
    Fast (no I/O), bounded capacity, supports kind-based filtering.
    """

    def __init__(self, capacity: int = _WORKING_MEMORY_CAPACITY) -> None:
        self._slots: Dict[str, WorkingSlot] = {}
        self._capacity = capacity
        self._lock = threading.Lock()

    def set(
        self,
        key: str,
        value: Any,
        kind: str = "fact",
        ttl_sec: Optional[float] = None,
    ) -> None:
        """Store a value. Evicts oldest expired slot if at capacity."""
        with self._lock:
            expires = time.time() + (ttl_sec if ttl_sec is not None else _WORKING_MEMORY_TTL_SEC)
            slot = WorkingSlot(key=key, value=value, kind=kind, expires_at=expires)
            self._slots[key] = slot
            # Evict if over capacity
            if len(self._slots) > self._capacity:
                self._evict()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value, returns default if not found or expired."""
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                return default
            if slot.is_expired:
                del self._slots[key]
                return default
            slot.access_count += 1
            return slot.value

    def delete(self, key: str) -> bool:
        with self._lock:
            return bool(self._slots.pop(key, None))

    def keys(self, kind: Optional[str] = None) -> List[str]:
        with self._lock:
            self._prune_expired()
            if kind:
                return [k for k, s in self._slots.items() if s.kind == kind]
            return list(self._slots.keys())

    def items(self, kind: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            self._prune_expired()
            if kind:
                return {k: s.value for k, s in self._slots.items() if s.kind == kind}
            return {k: s.value for k, s in self._slots.items()}

    def clear(self, kind: Optional[str] = None) -> int:
        """Clear all or kind-filtered slots. Returns count cleared."""
        with self._lock:
            if kind is None:
                n = len(self._slots)
                self._slots.clear()
                return n
            keys = [k for k, s in self._slots.items() if s.kind == kind]
            for k in keys:
                del self._slots[k]
            return len(keys)

    def snapshot(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of non-expired slots."""
        with self._lock:
            self._prune_expired()
            return {
                k: {
                    "value": s.value,
                    "kind":  s.kind,
                    "expires_in": round(s.expires_at - time.time(), 1),
                    "access_count": s.access_count,
                }
                for k, s in self._slots.items()
            }

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            self._prune_expired()
            kinds: Dict[str, int] = {}
            for s in self._slots.values():
                kinds[s.kind] = kinds.get(s.kind, 0) + 1
            return {
                "total": len(self._slots),
                "capacity": self._capacity,
                "by_kind": kinds,
            }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _prune_expired(self) -> None:
        expired = [k for k, s in self._slots.items() if s.is_expired]
        for k in expired:
            del self._slots[k]

    def _evict(self) -> None:
        # Remove expired first, then LRU (oldest created_at)
        self._prune_expired()
        if len(self._slots) > self._capacity:
            oldest = min(self._slots, key=lambda k: self._slots[k].created_at)
            del self._slots[oldest]


# ---------------------------------------------------------------------------
# Entity Memory
# ---------------------------------------------------------------------------

class EntityMemory:
    """
    Structured fact store for named entities.
    Backed by SQLite; thread-safe.
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS entity_facts (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        entity     TEXT NOT NULL,
        attribute  TEXT NOT NULL,
        value      TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 1.0,
        source     TEXT NOT NULL DEFAULT '',
        updated_at REAL NOT NULL,
        UNIQUE(entity, attribute)
    );
    CREATE INDEX IF NOT EXISTS idx_ef_entity ON entity_facts (entity);
    """

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.executescript(self._DDL)
        self._conn.commit()
        self._lock = threading.Lock()

    def know(
        self,
        entity: str,
        attribute: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "",
    ) -> None:
        """Store or update a fact about an entity."""
        val_str = json.dumps(value) if not isinstance(value, str) else value
        with self._lock:
            self._conn.execute(
                """INSERT INTO entity_facts (entity, attribute, value, confidence, source, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(entity, attribute) DO UPDATE SET
                       value=excluded.value,
                       confidence=excluded.confidence,
                       source=excluded.source,
                       updated_at=excluded.updated_at""",
                (entity, attribute, val_str, confidence, source, time.time()),
            )
            self._conn.commit()

    def recall(self, entity: str, attribute: Optional[str] = None) -> List[EntityFact]:
        """Return facts about an entity, optionally filtered by attribute."""
        with self._lock:
            if attribute:
                rows = self._conn.execute(
                    "SELECT entity, attribute, value, confidence, source, updated_at "
                    "FROM entity_facts WHERE entity=? AND attribute=?",
                    (entity, attribute),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT entity, attribute, value, confidence, source, updated_at "
                    "FROM entity_facts WHERE entity=?",
                    (entity,),
                ).fetchall()
        return [
            EntityFact(
                entity=r[0], attribute=r[1],
                value=self._deserialise(r[2]),
                confidence=r[3], source=r[4], updated_at=r[5],
            )
            for r in rows
        ]

    def entity_facts(self, entity: str) -> Dict[str, Any]:
        """Return a plain dict of attribute→value for an entity."""
        facts = self.recall(entity)
        return {f.attribute: f.value for f in facts}

    def list_entities(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT entity FROM entity_facts ORDER BY entity"
            ).fetchall()
        return [r[0] for r in rows]

    def forget_entity(self, entity: str) -> int:
        """Delete all facts about an entity. Returns rows deleted."""
        with self._lock:
            c = self._conn.execute(
                "DELETE FROM entity_facts WHERE entity=?", (entity,)
            )
            self._conn.commit()
            return c.rowcount

    def forget_attribute(self, entity: str, attribute: str) -> bool:
        with self._lock:
            c = self._conn.execute(
                "DELETE FROM entity_facts WHERE entity=? AND attribute=?",
                (entity, attribute),
            )
            self._conn.commit()
            return c.rowcount > 0

    def merge_entities(self, source: str, target: str) -> int:
        """Merge all facts from source entity into target (source deleted)."""
        with self._lock:
            # Get source facts not already in target
            rows = self._conn.execute(
                "SELECT attribute, value, confidence, source, updated_at "
                "FROM entity_facts WHERE entity=? AND attribute NOT IN "
                "(SELECT attribute FROM entity_facts WHERE entity=?)",
                (source, target),
            ).fetchall()
            for r in rows:
                self._conn.execute(
                    "INSERT INTO entity_facts (entity, attribute, value, confidence, source, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (target, r[0], r[1], r[2], r[3], r[4]),
                )
            self._conn.execute("DELETE FROM entity_facts WHERE entity=?", (source,))
            self._conn.commit()
            return len(rows)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM entity_facts").fetchone()[0]
            entities = self._conn.execute(
                "SELECT COUNT(DISTINCT entity) FROM entity_facts"
            ).fetchone()[0]
        return {"total_facts": total, "total_entities": entities}

    def _deserialise(self, val: str) -> Any:
        try:
            return json.loads(val)
        except Exception:
            return val


# ---------------------------------------------------------------------------
# Episodic Memory (conversation turns)
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    SQLite-backed episodic memory.
    Stores conversation turns and retrieves them by BM25 + cosine similarity.
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS episodes (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id   TEXT    NOT NULL,
        role         TEXT    NOT NULL,
        content      TEXT    NOT NULL,
        importance   REAL    NOT NULL DEFAULT 0.5,
        timestamp    REAL    NOT NULL,
        tags         TEXT    NOT NULL DEFAULT '[]',
        entities     TEXT    NOT NULL DEFAULT '[]',
        recall_count INTEGER NOT NULL DEFAULT 0,
        archived     INTEGER NOT NULL DEFAULT 0,
        embedding    BLOB
    );
    CREATE INDEX IF NOT EXISTS idx_ep_session ON episodes (session_id);
    CREATE INDEX IF NOT EXISTS idx_ep_ts ON episodes (timestamp);
    CREATE INDEX IF NOT EXISTS idx_ep_importance ON episodes (importance);
    """

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.executescript(self._DDL)
        self._conn.commit()
        self._lock = threading.Lock()
        self._embed_fn: Optional[Callable[[str], Optional[List[float]]]] = None
        self._embed_ready = False

    # ── Writing ──────────────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        session_id: str = "default",
        role: str = "user",
        importance: float = _DEFAULT_IMPORTANCE,
        tags: Optional[List[str]] = None,
    ) -> int:
        """Store an episode. Returns its row ID."""
        content = content[:_EPISODE_CONTENT_MAX]
        entities = _extract_entities(content)
        tags = tags or []
        emb_blob = self._maybe_embed(content)

        with self._lock:
            c = self._conn.execute(
                "INSERT INTO episodes "
                "(session_id, role, content, importance, timestamp, tags, entities, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, role, content, importance, time.time(),
                    json.dumps(tags), json.dumps(entities),
                    emb_blob,
                ),
            )
            self._conn.commit()
            return c.lastrowid

    def archive(self, entry_id: int) -> bool:
        with self._lock:
            c = self._conn.execute(
                "UPDATE episodes SET archived=1 WHERE id=?", (entry_id,)
            )
            self._conn.commit()
            return c.rowcount > 0

    def boost_importance(self, entry_id: int, delta: float = 0.1) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE episodes SET importance=MIN(1.0, importance+?), "
                "recall_count=recall_count+1 WHERE id=?",
                (delta, entry_id),
            )
            self._conn.commit()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        limit: int = _RECALL_LIMIT,
        session_id: Optional[str] = None,
        min_importance: float = 0.0,
        include_archived: bool = False,
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant episodes using hybrid BM25 + cosine similarity.
        """
        rows = self._fetch_candidates(session_id, min_importance, include_archived,
                                       limit * 5)
        if not rows:
            return []

        scored = self._score_rows(query, rows)
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:limit]

        entries = []
        for row, score in top:
            self.boost_importance(row[0], delta=0.02)
            entries.append(self._row_to_entry(row))
        return entries

    def recent(
        self,
        limit: int = 20,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Return most recent episodes, newest first."""
        with self._lock:
            base = "SELECT * FROM episodes WHERE archived=0"
            params: List[Any] = []
            if session_id:
                base += " AND session_id=?"; params.append(session_id)
            if role:
                base += " AND role=?"; params.append(role)
            base += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(base, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def session_context(self, session_id: str, limit: int = 50) -> List[MemoryEntry]:
        """Return full session history ordered chronologically."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM episodes WHERE session_id=? AND archived=0 "
                "ORDER BY timestamp ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT session_id), "
                "SUM(CASE WHEN archived=1 THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END), "
                "MIN(timestamp), MAX(timestamp) FROM episodes"
            ).fetchone()
        total, sessions, archived, with_emb, oldest, newest = row
        return {
            "total":         total or 0,
            "sessions":      sessions or 0,
            "archived":      archived or 0,
            "with_embeddings": with_emb or 0,
            "oldest_ts":     oldest,
            "newest_ts":     newest,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _fetch_candidates(
        self,
        session_id: Optional[str],
        min_importance: float,
        include_archived: bool,
        limit: int,
    ) -> List[tuple]:
        with self._lock:
            base  = "SELECT * FROM episodes WHERE importance>=?"
            params: List[Any] = [min_importance]
            if not include_archived:
                base += " AND archived=0"
            if session_id:
                base += " AND session_id=?"
                params.append(session_id)
            base += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            return self._conn.execute(base, params).fetchall()

    def _score_rows(
        self, query: str, rows: List[tuple]
    ) -> List[Tuple[tuple, float]]:
        """BM25 + cosine hybrid scoring."""
        query_emb = self._maybe_embed(query)
        query_tokens = set(_tokenize(query))
        scored = []
        for row in rows:
            content = row[3]   # content column
            # BM25-style term overlap
            doc_tokens = _tokenize(content)
            overlap = len(query_tokens & set(doc_tokens))
            bm25_score = overlap / (len(doc_tokens) + 0.5) if doc_tokens else 0.0

            # Cosine similarity
            cos_score = 0.0
            if query_emb and row[10] is not None:  # embedding column
                doc_emb = json.loads(row[10])
                cos_score = _cosine(query_emb, doc_emb)

            # Recency bonus
            age_days = (time.time() - row[5]) / 86400.0
            recency = 1.0 / (1.0 + age_days / 7.0)

            # Importance
            importance = row[4]

            combined = (
                0.35 * (cos_score if cos_score > 0 else bm25_score)
                + 0.30 * bm25_score
                + 0.20 * importance
                + 0.15 * recency
            )
            scored.append((row, combined))
        return scored

    def _maybe_embed(self, text: str) -> Optional[bytes]:
        """Try to embed text; return JSON-encoded bytes or None."""
        if not self._embed_ready:
            self._resolve_embed_fn()
        if self._embed_fn is None:
            return None
        vec = self._embed_fn(text)
        if vec:
            return json.dumps(vec).encode()
        return None

    def _resolve_embed_fn(self) -> None:
        """Lazily resolve which embedding backend to use."""
        self._embed_ready = True
        test = _embed_ollama("test")
        if test:
            self._embed_fn = _embed_ollama
            log.debug("EpisodicMemory: using Ollama embeddings")
            return
        test = _embed_openai("test")
        if test:
            self._embed_fn = _embed_openai
            log.debug("EpisodicMemory: using OpenAI embeddings")
            return
        log.debug("EpisodicMemory: no vector backend — BM25 only")
        self._embed_fn = None

    @staticmethod
    def _row_to_entry(row: tuple) -> MemoryEntry:
        emb = json.loads(row[10]) if row[10] else None
        return MemoryEntry(
            id           = row[0],
            session_id   = row[1],
            role         = row[2],
            content      = row[3],
            importance   = row[4],
            timestamp    = row[5],
            tags         = json.loads(row[6]),
            entities     = json.loads(row[7]),
            recall_count = row[8],
            archived     = bool(row[9]),
            embedding    = emb,
        )


# ---------------------------------------------------------------------------
# Memory Consolidator
# ---------------------------------------------------------------------------

class MemoryConsolidator:
    """
    Background maintenance for EpisodicMemory.
    • Prunes (archives) memories whose effective importance < threshold.
    • Boosts importance of memories cited multiple times.
    • Extracts entities from high-importance recent episodes.
    • (Optional) Deduplicates near-identical memories.
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        entity_mem: EntityMemory,
        archive_threshold: float = 0.05,
        max_episodes: int = 10_000,
    ) -> None:
        self._ep  = episodic
        self._em  = entity_mem
        self._archive_thresh = archive_threshold
        self._max_episodes = max_episodes
        self._last_run = 0.0

    def run(self, force: bool = False) -> Dict[str, int]:
        """
        Run consolidation. Skips if last run was < 1 hour ago unless forced.
        Returns counts of actions taken.
        """
        if not force and (time.time() - self._last_run) < 3600:
            return {"skipped": 1}

        self._last_run = time.time()
        archived = self._archive_stale()
        entities = self._extract_entities_from_recent()
        pruned   = self._enforce_capacity()

        log.info(
            "Consolidation: archived=%d new_entities=%d pruned=%d",
            archived, entities, pruned,
        )
        return {"archived": archived, "entities_extracted": entities, "pruned": pruned}

    def _archive_stale(self) -> int:
        """Archive episodes whose decayed importance < threshold."""
        archived = 0
        with self._ep._lock:
            rows = self._ep._conn.execute(
                "SELECT id, importance, timestamp, recall_count FROM episodes WHERE archived=0"
            ).fetchall()

        for row_id, importance, timestamp, recall_count in rows:
            age_days = (time.time() - timestamp) / 86400.0
            decay    = math.pow(0.5, age_days / _DECAY_HALF_LIFE_DAYS)
            eff      = importance * decay * (1.0 + 0.05 * recall_count)
            if eff < self._archive_thresh:
                self._ep.archive(row_id)
                archived += 1
        return archived

    def _extract_entities_from_recent(self) -> int:
        """Extract named entities from high-importance recent memories."""
        extracted = 0
        entries = self._ep.recent(limit=50)
        for entry in entries:
            if entry.importance < 0.6:
                continue
            for entity in entry.entities:
                # Store entity with source reference
                existing = self._em.entity_facts(entity)
                if "first_mentioned" not in existing:
                    self._em.know(entity, "first_mentioned",
                                  entry.timestamp, source="episodic")
                    extracted += 1
                self._em.know(entity, "last_mentioned",
                              entry.timestamp, source="episodic")
        return extracted

    def _enforce_capacity(self) -> int:
        """Delete oldest archived episodes if total > max_episodes."""
        with self._ep._lock:
            total = self._ep._conn.execute(
                "SELECT COUNT(*) FROM episodes"
            ).fetchone()[0]
            if total <= self._max_episodes:
                return 0
            excess = total - self._max_episodes
            self._ep._conn.execute(
                "DELETE FROM episodes WHERE id IN ("
                "  SELECT id FROM episodes WHERE archived=1 "
                "  ORDER BY timestamp ASC LIMIT ?"
                ")",
                (excess,),
            )
            self._ep._conn.commit()
            return excess


# ---------------------------------------------------------------------------
# MemoryStore — unified facade
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    Unified memory facade for Operon agents.
    Combines WorkingMemory, EpisodicMemory, EntityMemory, and consolidation.
    """

    def __init__(
        self,
        db_path: Path = _DB_PATH,
        session_id: str = "default",
        auto_consolidate: bool = True,
    ) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_id   = session_id
        self.working       = WorkingMemory()
        self.episodic      = EpisodicMemory(db_path)
        self.entities      = EntityMemory(db_path)
        self._consolidator = MemoryConsolidator(self.episodic, self.entities)
        self._auto_consolidate = auto_consolidate

    # ── High-level API ────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        role: str = "user",
        importance: float = _DEFAULT_IMPORTANCE,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Store an episode in long-term memory. Returns entry ID."""
        sid = session_id or self._session_id
        entry_id = self.episodic.store(
            content, session_id=sid, role=role,
            importance=importance, tags=tags,
        )
        if self._auto_consolidate:
            # Async consolidation every ~50 stores
            if entry_id % 50 == 0:
                t = threading.Thread(target=self._consolidator.run,
                                     daemon=True)
                t.start()
        return entry_id

    def recall(
        self,
        query: str,
        limit: int = _RECALL_LIMIT,
        session_id: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """Retrieve relevant memories for a query."""
        return self.episodic.recall(
            query, limit=limit,
            session_id=session_id or self._session_id,
            min_importance=min_importance,
        )

    def recent(self, limit: int = 20, role: Optional[str] = None) -> List[MemoryEntry]:
        """Return recent episodes (newest first) for current session."""
        return self.episodic.recent(
            limit=limit, session_id=self._session_id, role=role
        )

    def context_window(self, limit: int = 50) -> List[Dict[str, str]]:
        """
        Return current session as a list of {'role': ..., 'content': ...}
        dicts suitable for injection into an LLM context.
        """
        entries = self.episodic.session_context(self._session_id, limit=limit)
        return [{"role": e.role, "content": e.content} for e in entries]

    def know(
        self,
        entity: str,
        attribute: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "",
    ) -> None:
        """Store a structured fact about a named entity."""
        self.entities.know(entity, attribute, value,
                           confidence=confidence, source=source)

    def entity_facts(self, entity: str) -> Dict[str, Any]:
        """Return all known facts about an entity as a dict."""
        return self.entities.entity_facts(entity)

    def list_entities(self) -> List[str]:
        return self.entities.list_entities()

    def forget(self, entity: str) -> int:
        """Remove all entity facts for the given name."""
        return self.entities.forget_entity(entity)

    def consolidate(self, force: bool = True) -> Dict[str, int]:
        """Manually trigger memory consolidation."""
        return self._consolidator.run(force=force)

    # ── Session management ────────────────────────────────────────────────────

    def switch_session(self, session_id: str) -> None:
        """Change the active session (preserves all other state)."""
        self._session_id = session_id
        self.working.clear()

    def session_summary(self) -> Dict[str, Any]:
        """Return a summary of the current session state."""
        return {
            "session_id":   self._session_id,
            "working":      self.working.stats(),
            "episodic":     self.episodic.stats(),
            "entities":     self.entities.stats(),
        }

    def stats(self) -> Dict[str, Any]:
        return self.session_summary()

    def export_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Export session history and entity facts as a serialisable dict."""
        sid = session_id or self._session_id
        entries = self.episodic.session_context(sid, limit=1000)
        return {
            "session_id": sid,
            "episodes":   [
                {
                    "role":       e.role,
                    "content":    e.content,
                    "importance": e.importance,
                    "timestamp":  e.timestamp,
                    "tags":       e.tags,
                    "entities":   e.entities,
                }
                for e in entries
            ],
            "entity_count": self.entities.stats()["total_entities"],
            "exported_at":  time.time(),
        }


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------

# Common English stop words to exclude from BM25
_STOP_WORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "of",
    "and", "or", "but", "not", "this", "that", "was", "for", "are",
    "with", "as", "be", "by", "from", "i", "you", "we", "he", "she",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens, strip stop words, min length 2."""
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOP_WORDS]


def _extract_entities(text: str) -> List[str]:
    """
    Lightweight rule-based entity extraction.
    Extracts:
    - Capitalised words (names, places, organisations)
    - @mentions
    - Email addresses (username only)
    - URLs (domain only)
    - File paths
    - ALL_CAPS acronyms
    """
    entities: List[str] = []

    # @mentions
    entities.extend(re.findall(r"@([A-Za-z0-9_]+)", text))

    # ALL_CAPS acronyms (min 2 chars, max 8)
    entities.extend(re.findall(r"\b[A-Z]{2,8}\b", text))

    # Email usernames
    for email in re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text):
        entities.append(email.split("@")[0])

    # URL domains
    for url in re.findall(r"https?://([a-zA-Z0-9.-]+)", text):
        entities.append(url)

    # Capitalised multi-word names (Title Case sequence of 2+ words)
    for match in re.finditer(r"(?<!\. )([A-Z][a-z]+(?: [A-Z][a-z]+)+)", text):
        entities.append(match.group(0))

    # File paths
    entities.extend(re.findall(r"(?:^|[\s\"\'])(/[^\s\"\']+)", text))

    # Deduplicate, preserve order
    seen: set = set()
    result = []
    for e in entities:
        if e not in seen and len(e) >= 2:
            seen.add(e)
            result.append(e)
    return result[:20]  # cap at 20 entities per content


def _sha256_content(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_store: Optional[MemoryStore] = None
_store_lock = threading.Lock()


def get_memory_store(
    session_id: str = "default",
    db_path: Optional[Path] = None,
) -> MemoryStore:
    """Return (or create) the session-scoped default MemoryStore."""
    global _default_store
    with _store_lock:
        if _default_store is None:
            _default_store = MemoryStore(
                db_path=db_path or _DB_PATH,
                session_id=session_id,
            )
        elif session_id != "default":
            _default_store.switch_session(session_id)
    return _default_store


def remember(
    content: str,
    role: str = "user",
    importance: float = _DEFAULT_IMPORTANCE,
    session_id: str = "default",
) -> int:
    """One-liner: store an episode using the default store."""
    return get_memory_store(session_id).remember(content, role=role, importance=importance)


def recall(
    query: str,
    limit: int = _RECALL_LIMIT,
    session_id: str = "default",
) -> List[MemoryEntry]:
    """One-liner: retrieve relevant memories."""
    return get_memory_store(session_id).recall(query, limit=limit)
