"""
core/vector_memory.py — LanceDB Semantic Vector Memory Backend

Provides embedding-based long-term memory storage and retrieval using
LanceDB (local, zero-server vector database) + sentence-transformers
for local embeddings (no API key required).

Architecture:
  VectorMemoryEntry   — dataclass for a stored memory record
  EmbeddingEngine     — wraps sentence-transformers, caches model
  VectorStore         — LanceDB table management (create/upsert/search)
  VectorMemory        — high-level API: remember/recall/forget/summarize
  SemanticDeduplicator— detects near-duplicate entries before insertion
  MemoryConsolidator  — periodically merges related memories to reduce noise

Usage:
    vm = VectorMemory()
    vm.remember("User prefers dark mode and hates verbose output")
    vm.remember("Python project: operon AI terminal cockpit")

    results = vm.recall("what does the user prefer?", top_k=5)
    for r in results:
        print(r.text, r.score)

    facts = vm.get_context_for(query="python preferences", limit=3)
    # → list of strings ready to inject into system prompt
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("operon.vector_memory")

# ── Storage paths ─────────────────────────────────────────────────────────────
_DEFAULT_DB_PATH  = Path.home() / ".operon" / "vector_memory"
_DEFAULT_TABLE    = "memories"
_DEFAULT_MODEL    = "all-MiniLM-L6-v2"   # fast, 384-dim, runs on CPU, ~80MB
_EMBED_DIM        = 384
_DEDUP_THRESHOLD  = 0.92   # cosine similarity above this = duplicate
_MAX_MEMORIES     = 50_000
_CONTEXT_MAX_CHARS = 4_000  # max chars returned in get_context_for()

# ── Optional imports — detected at module load, loaded lazily on first use ────
# We probe availability cheaply (importlib.util.find_spec is ~0ms) so we don't
# pay the 3-5s sentence_transformers / PyTorch startup penalty until the first
# actual embedding is requested.
import importlib.util as _importlib_util
import sys as _sys_mod


def _probe(name: str) -> bool:
    """Check if a module is available without importing it.
    Handles already-mocked sys.modules (e.g. in tests) gracefully."""
    if name in _sys_mod.modules:
        return True
    try:
        return _importlib_util.find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


_LANCE = _probe("lancedb")
_ST    = _probe("sentence_transformers")

# The actual modules are stored here once loaded
_lancedb_mod: "Any | None"   = None
_st_mod:      "Any | None"   = None


def _ensure_lancedb():
    """Lazily import lancedb on first use."""
    global _lancedb_mod, _LANCE
    if _lancedb_mod is None:
        try:
            import lancedb as _lb
            _lancedb_mod = _lb
        except ImportError:
            _LANCE = False
    return _lancedb_mod


def _ensure_sentence_transformers():
    """Lazily import SentenceTransformer on first use."""
    global _st_mod, _ST
    if _st_mod is None:
        try:
            from sentence_transformers import SentenceTransformer as _STC
            _st_mod = _STC
        except ImportError:
            _ST = False
    return _st_mod


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class VectorMemoryEntry:
    """A single memory record."""
    text:       str
    source:     str   = "user"     # user | agent | file | entity | task | summary
    category:   str   = "general"  # general | person | project | code | preference | fact
    tags:       List[str] = field(default_factory=list)
    session_id: str   = ""
    created_at: str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 0
    score:      float = 0.0    # populated on retrieval
    entry_id:   str   = ""     # SHA-256 of text[:64]

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = hashlib.sha256(self.text[:64].encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        return {
            "entry_id":    self.entry_id,
            "text":        self.text,
            "source":      self.source,
            "category":    self.category,
            "tags":        json.dumps(self.tags),
            "session_id":  self.session_id,
            "created_at":  self.created_at,
            "updated_at":  self.updated_at,
            "access_count": self.access_count,
        }

    @staticmethod
    def from_dict(d: Dict, score: float = 0.0) -> "VectorMemoryEntry":
        tags = d.get("tags", "[]")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        return VectorMemoryEntry(
            text         = d.get("text", ""),
            source       = d.get("source", "user"),
            category     = d.get("category", "general"),
            tags         = tags,
            session_id   = d.get("session_id", ""),
            created_at   = d.get("created_at", ""),
            updated_at   = d.get("updated_at", ""),
            access_count = d.get("access_count", 0),
            entry_id     = d.get("entry_id", ""),
            score        = score,
        )


# ── Embedding engine ──────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Wraps sentence-transformers with lazy loading and caching.
    Falls back to a simple TF-IDF-style bag-of-words if ST unavailable.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Optional[Any] = None
        self._available = _ST

    def _load(self) -> None:
        if self._model is None and self._available:
            try:
                SentenceTransformer = _ensure_sentence_transformers()
                if SentenceTransformer is None:
                    self._available = False
                    return
                self._model = SentenceTransformer(self._model_name)
                log.info("VectorMemory: loaded embedding model %s", self._model_name)
            except Exception as e:
                log.warning("VectorMemory: could not load ST model: %s", e)
                self._available = False

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts. Returns list of float vectors."""
        if not texts:
            return []
        self._load()
        if self._model is not None:
            try:
                vecs = self._model.encode(texts, normalize_embeddings=True)
                return [v.tolist() for v in vecs]
            except Exception as e:
                log.warning("Embedding error: %s", e)
        # Fallback: deterministic hash-based pseudo-embedding (poor quality but won't crash)
        return [self._hash_embed(t) for t in texts]

    def embed_one(self, text: str) -> List[float]:
        result = self.embed([text])
        return result[0] if result else [0.0] * _EMBED_DIM

    @staticmethod
    def _hash_embed(text: str, dim: int = _EMBED_DIM) -> List[float]:
        """Deterministic pseudo-embedding via hashing — fallback only."""
        import struct
        h = hashlib.sha256(text.encode()).digest()
        # Repeat digest to fill dim floats
        raw = (h * (dim * 4 // len(h) + 1))[: dim * 4]
        vals = list(struct.unpack(f"{dim}f", raw))
        norm = (sum(v * v for v in vals) ** 0.5) or 1.0
        return [v / norm for v in vals]

    @property
    def available(self) -> bool:
        return self._available or True   # fallback always available

    @property
    def dim(self) -> int:
        return _EMBED_DIM


# ── LanceDB vector store ──────────────────────────────────────────────────────

class VectorStore:
    """
    Manages a LanceDB table for vector memory.
    Falls back to in-memory list storage if LanceDB is unavailable.
    """

    def __init__(
        self,
        db_path:    Path  = _DEFAULT_DB_PATH,
        table_name: str   = _DEFAULT_TABLE,
        engine:     Optional[EmbeddingEngine] = None,
    ) -> None:
        self._db_path    = db_path
        self._table_name = table_name
        self._engine     = engine or EmbeddingEngine()
        self._db: Optional[Any]    = None
        self._table: Optional[Any] = None
        self._fallback: List[Dict] = []   # in-memory fallback
        self._use_lance = _LANCE

    def _ensure_open(self) -> None:
        if self._table is not None:
            return
        if not self._use_lance:
            return
        try:
            self._db_path.mkdir(parents=True, exist_ok=True)
            import lancedb
            import pyarrow as pa
            self._db = lancedb.connect(str(self._db_path))
            if self._table_name in self._db.table_names():
                self._table = self._db.open_table(self._table_name)
            else:
                schema = pa.schema([
                    pa.field("entry_id",    pa.utf8()),
                    pa.field("text",        pa.utf8()),
                    pa.field("source",      pa.utf8()),
                    pa.field("category",    pa.utf8()),
                    pa.field("tags",        pa.utf8()),
                    pa.field("session_id",  pa.utf8()),
                    pa.field("created_at",  pa.utf8()),
                    pa.field("updated_at",  pa.utf8()),
                    pa.field("access_count", pa.int32()),
                    pa.field("vector",      pa.list_(pa.float32(), _EMBED_DIM)),
                ])
                self._table = self._db.create_table(self._table_name, schema=schema)
                log.info("VectorStore: created table '%s'", self._table_name)
        except Exception as e:
            log.warning("VectorStore: LanceDB init failed (%s), using in-memory fallback", e)
            self._use_lance = False

    def upsert(self, entry: VectorMemoryEntry) -> None:
        """Insert or update an entry (matched by entry_id)."""
        self._ensure_open()
        vec = self._engine.embed_one(entry.text)
        row = entry.to_dict()
        row["vector"] = vec

        if self._table is not None:
            try:
                # Delete existing if same entry_id, then add
                self._table.delete(f"entry_id = '{entry.entry_id}'")
                import pandas as pd
                self._table.add(pd.DataFrame([row]))
                return
            except Exception as e:
                log.warning("VectorStore upsert error: %s", e)

        # Fallback — update in-memory list
        self._fallback = [r for r in self._fallback if r.get("entry_id") != entry.entry_id]
        self._fallback.append(row)

    def search(
        self,
        query:      str,
        top_k:      int   = 10,
        category:   Optional[str] = None,
        source:     Optional[str] = None,
        min_score:  float = 0.0,
    ) -> List[VectorMemoryEntry]:
        """Semantic search — returns top-k most relevant memories."""
        self._ensure_open()
        query_vec = self._engine.embed_one(query)

        if self._table is not None:
            try:
                q = self._table.search(query_vec).limit(top_k * 2)
                if category:
                    q = q.where(f"category = '{category}'")
                if source:
                    q = q.where(f"source = '{source}'")
                rows = q.to_list()
                results = []
                for row in rows:
                    score = float(1.0 - row.get("_distance", 1.0))
                    if score >= min_score:
                        e = VectorMemoryEntry.from_dict(row, score=score)
                        results.append(e)
                return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
            except Exception as e:
                log.warning("VectorStore search error: %s", e)

        # Fallback — cosine similarity over in-memory list
        return self._fallback_search(query_vec, top_k, category, source, min_score)

    def _fallback_search(
        self,
        query_vec:  List[float],
        top_k:      int,
        category:   Optional[str],
        source:     Optional[str],
        min_score:  float,
    ) -> List[VectorMemoryEntry]:
        scored = []
        for row in self._fallback:
            if category and row.get("category") != category:
                continue
            if source and row.get("source") != source:
                continue
            vec = row.get("vector", [])
            if vec:
                score = _cosine(query_vec, vec)
                if score >= min_score:
                    scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [VectorMemoryEntry.from_dict(r, score=s) for s, r in scored[:top_k]]

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        self._ensure_open()
        if self._table is not None:
            try:
                self._table.delete(f"entry_id = '{entry_id}'")
                return True
            except Exception:
                pass
        before = len(self._fallback)
        self._fallback = [r for r in self._fallback if r.get("entry_id") != entry_id]
        return len(self._fallback) < before

    def count(self) -> int:
        """Return total number of stored memories."""
        self._ensure_open()
        if self._table is not None:
            try:
                return self._table.count_rows()
            except Exception:
                pass
        return len(self._fallback)

    def all_texts(self, limit: int = 1000) -> List[str]:
        """Return all stored memory texts (for consolidation)."""
        self._ensure_open()
        if self._table is not None:
            try:
                rows = self._table.to_pandas().head(limit)
                return rows["text"].tolist()
            except Exception:
                pass
        return [r["text"] for r in self._fallback[:limit]]


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ── Semantic deduplicator ─────────────────────────────────────────────────────

class SemanticDeduplicator:
    """
    Before inserting, check if a near-identical entry already exists.
    Uses vector similarity — no exact-string comparison.
    """

    def __init__(self, store: VectorStore, threshold: float = _DEDUP_THRESHOLD) -> None:
        self._store     = store
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def is_duplicate(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Returns (is_dup, existing_entry_id_or_None).
        """
        results = self._store.search(text, top_k=1, min_score=self._threshold)
        if results and results[0].score >= self._threshold:
            return True, results[0].entry_id
        return False, None


# ── Memory consolidator ───────────────────────────────────────────────────────

class MemoryConsolidator:
    """
    Periodically groups related memories and merges them into summaries,
    reducing noise and keeping the store compact.

    Triggered automatically when count > max_memories * 0.8.
    """

    def __init__(self, store: VectorStore, max_memories: int = _MAX_MEMORIES) -> None:
        self._store       = store
        self._max         = max_memories
        self._last_run    = 0.0

    def should_consolidate(self) -> bool:
        if time.time() - self._last_run < 3600:  # max once per hour
            return False
        return self._store.count() > self._max * 0.8

    def consolidate(self) -> int:
        """
        Simple consolidation: find redundant entries (similarity > 0.95)
        and keep only the most recent one.
        Returns number of entries removed.
        """
        if not self.should_consolidate():
            return 0

        texts  = self._store.all_texts(limit=500)
        engine = self._store._engine
        if not texts:
            return 0

        vecs     = engine.embed(texts)
        to_delete: set = set()

        for i in range(len(texts)):
            if i in to_delete:
                continue
            for j in range(i + 1, len(texts)):
                if j in to_delete:
                    continue
                if _cosine(vecs[i], vecs[j]) > 0.95:
                    to_delete.add(j)

        removed = 0
        for idx in to_delete:
            entry_id = hashlib.sha256(texts[idx][:64].encode()).hexdigest()[:16]
            if self._store.delete(entry_id):
                removed += 1

        self._last_run = time.time()
        log.info("MemoryConsolidator: removed %d redundant entries", removed)
        return removed


# ── High-level VectorMemory API ───────────────────────────────────────────────

class VectorMemory:
    """
    Primary API for Operon's vector-based long-term memory.

    Usage:
        vm = VectorMemory()
        vm.remember("User hates verbose output", category="preference")
        results = vm.recall("what are user preferences?")
        context = vm.get_context_for("coding task", limit=5)
    """

    def __init__(
        self,
        db_path:       Path = _DEFAULT_DB_PATH,
        model_name:    str  = _DEFAULT_MODEL,
        dedup:         bool = True,
        auto_consolidate: bool = True,
    ) -> None:
        self._engine      = EmbeddingEngine(model_name)
        self._store       = VectorStore(db_path, engine=self._engine)
        self._deduper     = SemanticDeduplicator(self._store) if dedup else None
        self._consolidator = MemoryConsolidator(self._store) if auto_consolidate else None
        self._session_id  = ""

    def set_session(self, session_id: str) -> None:
        self._session_id = session_id

    # ── Write ─────────────────────────────────────────────────────────────────

    def remember(
        self,
        text:     str,
        source:   str = "agent",
        category: str = "general",
        tags:     Optional[List[str]] = None,
        force:    bool = False,
    ) -> Tuple[bool, str]:
        """
        Store a memory. Returns (stored, reason).

        Skips duplicates unless force=True.
        """
        text = text.strip()
        if not text:
            return False, "empty text"

        if not force and self._deduper:
            is_dup, dup_id = self._deduper.is_duplicate(text)
            if is_dup:
                return False, f"duplicate of {dup_id}"

        entry = VectorMemoryEntry(
            text       = text,
            source     = source,
            category   = category,
            tags       = tags or [],
            session_id = self._session_id,
        )
        self._store.upsert(entry)

        if self._consolidator and self._consolidator.should_consolidate():
            self._consolidator.consolidate()

        return True, entry.entry_id

    def remember_many(
        self,
        items: "List[Dict | VectorMemoryEntry]",
    ) -> "List[Tuple[bool, str]]":
        """Batch remember. Each item: dict {text, source?, category?, tags?} or VectorMemoryEntry."""
        results = []
        for item in items:
            if isinstance(item, VectorMemoryEntry):
                results.append(self.remember(
                    text=item.text, source=item.source,
                    category=item.category, tags=item.tags,
                ))
            else:
                results.append(self.remember(
                    text     = item.get("text", ""),
                    source   = item.get("source", "agent"),
                    category = item.get("category", "general"),
                    tags     = item.get("tags", []),
                ))
        return results

    # ── Read ──────────────────────────────────────────────────────────────────

    def recall(
        self,
        query:    str,
        top_k:    int  = 10,
        category: Optional[str] = None,
        source:   Optional[str] = None,
        min_score: float = 0.1,
    ) -> List[VectorMemoryEntry]:
        """Semantic search. Returns entries sorted by relevance."""
        return self._store.search(
            query     = query,
            top_k     = top_k,
            category  = category,
            source    = source,
            min_score = min_score,
        )

    def get_context_for(
        self,
        query:  str,
        limit:  int  = 8,
        source: Optional[str] = None,
    ) -> List[str]:
        """
        Return a list of memory strings suitable for system prompt injection.
        Total chars capped at _CONTEXT_MAX_CHARS.
        """
        results = self.recall(query, top_k=limit, source=source, min_score=0.15)
        out: List[str] = []
        total = 0
        for r in results:
            if total + len(r.text) > _CONTEXT_MAX_CHARS:
                break
            out.append(r.text)
            total += len(r.text)
        return out

    def build_context_block(self, query: str, limit: int = 8) -> str:
        """Return a formatted block for injection into system prompt."""
        facts = self.get_context_for(query, limit)
        if not facts:
            return ""
        lines = ["[Relevant Memory]"]
        for i, f in enumerate(facts, 1):
            lines.append(f"  {i}. {f}")
        return "\n".join(lines)

    # ── Forget ────────────────────────────────────────────────────────────────

    def forget(self, entry_id: str) -> bool:
        return self._store.delete(entry_id)

    def forget_by_text(self, text: str) -> bool:
        entry_id = hashlib.sha256(text[:64].encode()).hexdigest()[:16]
        return self._store.delete(entry_id)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._store.count()

    def stats(self) -> Dict:
        return {
            "total_memories":   self.count(),
            "embedding_model":  self._engine._model_name,
            "lance_available":  _LANCE,
            "st_available":     _ST,
            "db_path":          str(self._store._db_path),
            "dedup_threshold":  _DEDUP_THRESHOLD,
        }

    def summary(self) -> str:
        s = self.stats()
        return (
            f"VectorMemory: {s['total_memories']} memories  │  "
            f"model={s['embedding_model']}  │  "
            f"LanceDB={'✓' if s['lance_available'] else '✗'}  │  "
            f"path={s['db_path']}"
        )


# ── Convenience module-level functions ────────────────────────────────────────

_default_vm: Optional[VectorMemory] = None


def get_vector_memory() -> VectorMemory:
    """Return (or create) the module-level VectorMemory singleton."""
    global _default_vm
    if _default_vm is None:
        _default_vm = VectorMemory()
    return _default_vm


def remember(text: str, **kwargs) -> Tuple[bool, str]:
    return get_vector_memory().remember(text, **kwargs)


def recall(query: str, top_k: int = 10) -> List[VectorMemoryEntry]:
    return get_vector_memory().recall(query, top_k=top_k)


def get_context(query: str, limit: int = 8) -> str:
    return get_vector_memory().build_context_block(query, limit)


# ── Tool definitions (for main.py _TOOL_DEFINITIONS) ─────────────────────────

_TOOL_DEFINITIONS: List[Dict] = [
    {
        "name": "vector_memory_remember",
        "description": "Store a fact or observation in semantic long-term memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text":     {"type": "string", "description": "The text/fact to remember"},
                "category": {"type": "string", "enum": ["general","person","project","code","preference","fact"], "default": "general"},
                "source":   {"type": "string", "default": "agent"},
                "tags":     {"type": "array", "items": {"type": "string"}, "default": []},
            },
            "required": ["text"],
        },
    },
    {
        "name": "vector_memory_recall",
        "description": "Retrieve the most relevant memories for a query using semantic search.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":    {"type": "string"},
                "top_k":    {"type": "integer", "default": 5},
                "category": {"type": "string"},
                "min_score":{"type": "number", "default": 0.1},
            },
            "required": ["query"],
        },
    },
    {
        "name": "vector_memory_forget",
        "description": "Delete a memory by its text content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    },
]


def _vm_remember(text: str, category: str = "general", source: str = "agent", tags: list = []) -> dict:
    ok, reason = get_vector_memory().remember(text, source=source, category=category, tags=tags)
    return {"success": ok, "reason": reason, "total": get_vector_memory().count()}


def _vm_recall(query: str, top_k: int = 5, category: str = None, min_score: float = 0.1) -> dict:
    results = get_vector_memory().recall(query, top_k=top_k, category=category, min_score=min_score)
    return {
        "success": True,
        "count": len(results),
        "memories": [{"text": r.text, "score": round(r.score, 3), "category": r.category} for r in results],
    }


def _vm_forget(text: str) -> dict:
    ok = get_vector_memory().forget_by_text(text)
    return {"success": ok}


_DISPATCH: Dict[str, Any] = {
    "vector_memory_remember": _vm_remember,
    "vector_memory_recall":   _vm_recall,
    "vector_memory_forget":   _vm_forget,
}
