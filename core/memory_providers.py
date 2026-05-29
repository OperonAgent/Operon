"""
Operon Pluggable Memory Providers.

Inspired by Hermes Agent's 8 external memory provider plugins.
Provides a common interface so different backends can be swapped in
without changing the rest of the codebase.

Priority chain (highest to lowest):
  1. Mem0       — if mem0ai installed + MEM0_API_KEY set
  2. Builtin    — Operon's own SemanticMemory (always available)

Adding a new provider: subclass MemoryProvider and register it in
get_provider_chain().

Configuration
-------------
  # Use Mem0 as primary memory store:
  export MEM0_API_KEY=m0-xxxxxxxxxx

  # Use Mem0 with local Ollama embedding:
  export MEM0_API_KEY=m0-xxxxxxxxxx
  export MEM0_EMBEDDING_MODEL=nomic-embed-text
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

log = logging.getLogger("operon.memory_providers")


# ── Base class ────────────────────────────────────────────────────────────────

class MemoryProvider(ABC):
    """Abstract base class for all memory backends."""

    name: str = "base"

    @abstractmethod
    def save(self, session_id: str, role: str, content: str) -> bool:
        """Persist a single turn. Returns True on success."""

    @abstractmethod
    def recall(self, query: str, top_k: int = 5, user_id: str = "") -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for a query.
        Returns list of {content, score, source, metadata} dicts.
        """

    @abstractmethod
    def available(self) -> bool:
        """Return True if this provider can be used in the current environment."""

    def as_context_block(self, query: str, top_k: int = 5, user_id: str = "") -> str:
        """Return a formatted block ready for system prompt injection."""
        results = self.recall(query, top_k=top_k, user_id=user_id)
        if not results:
            return ""
        lines = [f"[MEMORY — {self.name}]"]
        for r in results:
            score = f"  (score {r.get('score', '?'):.2f})" if isinstance(r.get("score"), float) else ""
            lines.append(f"  {r.get('content', '')[:200]}{score}")
        lines.append("[END MEMORY]")
        return "\n".join(lines)


# ── Builtin provider (wraps SemanticMemory) ───────────────────────────────────

class BuiltinProvider(MemoryProvider):
    """Wraps Operon's existing SemanticMemory. Always available."""

    name = "builtin"

    def __init__(self, semantic_mem=None):
        self._sem = semantic_mem

    def available(self) -> bool:
        return self._sem is not None

    def save(self, session_id: str, role: str, content: str) -> bool:
        if self._sem is None:
            return False
        try:
            self._sem.save(session_id, role, content)
            return True
        except Exception as e:
            log.debug("BuiltinProvider.save error: %s", e)
            return False

    def recall(self, query: str, top_k: int = 5, user_id: str = "") -> List[Dict[str, Any]]:
        if self._sem is None:
            return []
        try:
            results = self._sem.recall(query, top_k=top_k)
            return [
                {
                    "content":  r.get("content", ""),
                    "score":    r.get("similarity", 0.0),
                    "source":   "builtin",
                    "metadata": {"role": r.get("role", ""), "ts": r.get("timestamp", 0)},
                }
                for r in results
            ]
        except Exception as e:
            log.debug("BuiltinProvider.recall error: %s", e)
            return []


# ── Mem0 provider ─────────────────────────────────────────────────────────────

class Mem0Provider(MemoryProvider):
    """
    Uses mem0ai for cross-session user memory with semantic search.
    pip install mem0ai
    export MEM0_API_KEY=m0-xxxxxxxxxx
    """

    name = "mem0"

    def __init__(self):
        self._client = None
        self._user_id = os.environ.get("MEM0_USER_ID", "operon_user")

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from mem0 import MemoryClient
            api_key = os.environ.get("MEM0_API_KEY", "").strip()
            if not api_key:
                return None
            self._client = MemoryClient(api_key=api_key)
            return self._client
        except ImportError:
            return None
        except Exception as e:
            log.debug("Mem0Provider init error: %s", e)
            return None

    def available(self) -> bool:
        return self._get_client() is not None

    def save(self, session_id: str, role: str, content: str) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            client.add(
                [{"role": role, "content": content}],
                user_id=self._user_id,
                metadata={"session_id": session_id},
            )
            return True
        except Exception as e:
            log.debug("Mem0Provider.save error: %s", e)
            return False

    def recall(self, query: str, top_k: int = 5, user_id: str = "") -> List[Dict[str, Any]]:
        client = self._get_client()
        if client is None:
            return []
        uid = user_id or self._user_id
        try:
            results = client.search(query, user_id=uid, limit=top_k)
            return [
                {
                    "content":  r.get("memory", r.get("text", "")),
                    "score":    r.get("score", 0.0),
                    "source":   "mem0",
                    "metadata": r.get("metadata", {}),
                }
                for r in (results if isinstance(results, list) else results.get("results", []))
            ]
        except Exception as e:
            log.debug("Mem0Provider.recall error: %s", e)
            return []


# ── Provider chain ────────────────────────────────────────────────────────────

class MemoryProviderChain:
    """
    Tries providers in priority order. Saves to the first available one;
    recalls from all available ones and merges results by score.
    """

    def __init__(self, providers: List[MemoryProvider]):
        self._providers = [p for p in providers if p.available()]
        log.info(
            "MemoryProviderChain: %d active provider(s): %s",
            len(self._providers),
            [p.name for p in self._providers],
        )

    def save(self, session_id: str, role: str, content: str) -> None:
        for p in self._providers:
            try:
                p.save(session_id, role, content)
            except Exception:
                pass

    def recall(self, query: str, top_k: int = 5, user_id: str = "") -> List[Dict[str, Any]]:
        seen:    set = set()
        merged:  List[Dict[str, Any]] = []
        for p in self._providers:
            try:
                results = p.recall(query, top_k=top_k, user_id=user_id)
                for r in results:
                    key = r.get("content", "")[:60]
                    if key not in seen:
                        seen.add(key)
                        r["provider"] = p.name
                        merged.append(r)
            except Exception:
                pass
        # Sort by score descending, return top_k
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return merged[:top_k]

    def as_context_block(self, query: str, top_k: int = 5, user_id: str = "") -> str:
        results = self.recall(query, top_k=top_k, user_id=user_id)
        if not results:
            return ""
        lines = [f"[MEMORY — {len(results)} relevant memories]"]
        for r in results:
            score_str = f" ({r['score']:.2f})" if isinstance(r.get("score"), float) else ""
            provider  = f" [{r.get('provider', '?')}]"
            lines.append(f"  {r.get('content', '')[:200]}{score_str}{provider}")
        lines.append("[END MEMORY]")
        return "\n".join(lines)

    def status(self) -> List[Dict[str, Any]]:
        return [
            {"name": p.name, "available": p.available()}
            for p in self._providers
        ]

    @property
    def active_providers(self) -> List[str]:
        return [p.name for p in self._providers]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_provider_chain(semantic_mem=None) -> MemoryProviderChain:
    """
    Build the provider chain from available backends.
    Order: Mem0 (if configured) → Builtin (always)
    """
    providers: List[MemoryProvider] = []

    # Mem0 — only if API key is set
    if os.environ.get("MEM0_API_KEY"):
        providers.append(Mem0Provider())

    # Builtin — always last (fallback)
    providers.append(BuiltinProvider(semantic_mem))

    return MemoryProviderChain(providers)
