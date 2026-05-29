"""
core/memory_facade.py — one unified API over Operon's three memory backends.

Operon has three complementary memory systems:
  • MemoryPipeline (FTS5)  — fast keyword recall, always available, local SQLite.
  • VectorMemory (LanceDB)  — semantic similarity recall (optional, lazy).
  • ObsidianMemory (vault)  — human-readable markdown notes (optional).

Historically callers had to know which backend to touch. This facade gives a
single, simple API — remember() / recall() / context() — that fans writes out
to whichever backends are available and merges reads, deduplicating results.

It is intentionally tolerant: any backend that is missing or errors is skipped,
so the facade always works even with only the FTS5 store present.

    facade = MemoryFacade(pipeline=memory)        # vector/obsidian auto-detected
    facade.remember("User prefers dark mode", tags=["pref"])
    hits = facade.recall("what theme does the user like?")
    print(facade.context("theme", limit=5))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

# Backends are all optional; import lazily/defensively.
try:
    from core.memory import MemoryPipeline  # type: ignore
except Exception:  # pragma: no cover
    MemoryPipeline = None  # type: ignore


@dataclass
class MemoryHit:
    """A unified search result from any backend."""
    content: str
    source:  str            # "fts5" | "vector" | "obsidian"
    score:   float = 0.0
    meta:    dict = field(default_factory=dict)


class MemoryFacade:
    """Unified read/write across FTS5, vector, and Obsidian memory."""

    def __init__(
        self,
        pipeline: Any = None,
        vector:   Any = None,
        obsidian: Any = None,
        enable_vector:   bool = True,
        enable_obsidian: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self._vector   = vector
        self._obsidian = obsidian
        self._enable_vector   = enable_vector
        self._enable_obsidian = enable_obsidian

    # ── lazy backend accessors ─────────────────────────────────────────────────

    @property
    def vector(self):
        if self._vector is None and self._enable_vector:
            try:
                from core.vector_memory import get_vector_memory
                self._vector = get_vector_memory()
            except Exception:
                self._vector = False  # mark as unavailable
        return self._vector or None

    @property
    def obsidian(self):
        if self._obsidian is None and self._enable_obsidian:
            try:
                from core.obsidian_memory import get_obsidian_memory
                self._obsidian = get_obsidian_memory()
            except Exception:
                self._obsidian = False
        return self._obsidian or None

    # ── write ───────────────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        tags:    Optional[List[str]] = None,
        importance: int = 3,
        category: str = "general",
        to_obsidian: bool = False,
    ) -> dict:
        """
        Store a memory across all available backends. Returns a dict of which
        backends accepted it: {"fts5": bool, "vector": bool, "obsidian": bool}.
        """
        tags = tags or []
        result = {"fts5": False, "vector": False, "obsidian": False}

        # FTS5 (primary, always-on)
        if self.pipeline is not None:
            try:
                self.pipeline.add_manual(content, tags=",".join(tags),
                                         importance=importance)
                result["fts5"] = True
            except Exception:
                pass

        # Vector (semantic)
        v = self.vector
        if v is not None:
            try:
                stored, _ = v.remember(content, category=category, tags=tags)
                result["vector"] = bool(stored)
            except Exception:
                pass

        # Obsidian (only when explicitly requested — it writes files)
        if to_obsidian:
            o = self.obsidian
            if o is not None:
                try:
                    o.add_fact(category, content, tags=tags)
                    result["obsidian"] = True
                except Exception:
                    pass

        return result

    # ── read ─────────────────────────────────────────────────────────────────────

    def recall(self, query: str, limit: int = 10) -> List[MemoryHit]:
        """
        Search all available backends and return merged, de-duplicated hits.
        Vector hits (semantic) are ranked first when available.
        """
        hits: List[MemoryHit] = []
        seen: set = set()

        def _add(content: str, source: str, score: float, meta: dict):
            key = (content or "").strip().lower()[:160]
            if not key or key in seen:
                return
            seen.add(key)
            hits.append(MemoryHit(content=content, source=source, score=score, meta=meta))

        # Vector first (semantic relevance)
        v = self.vector
        if v is not None:
            try:
                for entry in v.recall(query, top_k=limit):
                    text  = getattr(entry, "text", None) or getattr(entry, "content", "") or str(entry)
                    score = float(getattr(entry, "score", 0.0) or 0.0)
                    _add(text, "vector", score, {"category": getattr(entry, "category", "")})
            except Exception:
                pass

        # FTS5 keyword
        if self.pipeline is not None:
            try:
                for row in self.pipeline.search(query, limit=limit):
                    content = row.get("content", "") if isinstance(row, dict) else str(row)
                    _add(content, "fts5", 0.0, row if isinstance(row, dict) else {})
            except Exception:
                pass

        # Obsidian (if enabled)
        o = self.obsidian
        if o is not None:
            try:
                ctx = o.get_context(query, limit)
                if ctx:
                    _add(str(ctx), "obsidian", 0.0, {})
            except Exception:
                pass

        # Stable sort: vector (by score desc) then the rest in insertion order.
        hits.sort(key=lambda h: (h.source != "vector", -h.score))
        return hits[:limit]

    def context(self, query: str, limit: int = 5) -> str:
        """Return a newline-joined context block suitable for a system prompt."""
        hits = self.recall(query, limit=limit)
        if not hits:
            return ""
        lines = []
        for h in hits:
            tag = {"vector": "≈", "fts5": "•", "obsidian": "▣"}.get(h.source, "•")
            lines.append(f"  {tag} {h.content.strip()[:200]}")
        return "\n".join(lines)

    # ── introspection ─────────────────────────────────────────────────────────────

    def backends_available(self) -> dict:
        return {
            "fts5":     self.pipeline is not None,
            "vector":   self.vector is not None,
            "obsidian": self.obsidian is not None,
        }

    def stats(self) -> dict:
        out = {"backends": self.backends_available()}
        try:
            if self.pipeline is not None:
                out["fts5_count"] = len(self.pipeline.get_all())
        except Exception:
            pass
        try:
            v = self.vector
            if v is not None:
                out["vector_count"] = v.count()
        except Exception:
            pass
        return out


_facade_singleton: Optional[MemoryFacade] = None


def get_memory_facade(pipeline: Any = None, **kwargs) -> MemoryFacade:
    """Process-wide singleton facade. Pass the FTS5 pipeline on first call."""
    global _facade_singleton
    if _facade_singleton is None:
        _facade_singleton = MemoryFacade(pipeline=pipeline, **kwargs)
    elif pipeline is not None and _facade_singleton.pipeline is None:
        _facade_singleton.pipeline = pipeline
    return _facade_singleton
