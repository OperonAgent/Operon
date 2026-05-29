"""
Operon Credential Pool — Multi-key failover for API providers.

Matches Hermes credential_pool.py depth.

Manages a rotating pool of API credentials per provider. When one key hits
a rate-limit, quota exhaustion, or auth error, the pool automatically rotates
to the next available key and retries the call transparently.

Architecture:
  • CredentialPool — one pool per provider (or one shared pool for all).
  • KeySlot — tracks one credential: key, status, usage count, last error.
  • .get(provider)    → returns the best available key for a provider.
  • .rotate(provider) → marks current key as degraded, returns next key.
  • .report_error(provider, key, error) → classifies error, auto-rotates.
  • .report_success(provider, key)      → resets backoff for that key.
  • .status()         → per-provider health dashboard.

Keys can be loaded from:
  - Environment variables: OPENAI_API_KEY_1, OPENAI_API_KEY_2, …
  - Knowledge base / secrets file (JSON)
  - Explicit .add(provider, key) call

Usage:
    from core.credential_pool import CredentialPool

    pool = CredentialPool()
    pool.load_from_env("openai",     prefix="OPENAI_API_KEY")
    pool.load_from_env("anthropic",  prefix="ANTHROPIC_API_KEY")

    key = pool.get("openai")          # best available key
    try:
        result = call_api(key)
        pool.report_success("openai", key)
    except RateLimitError as e:
        pool.report_error("openai", key, e)
        key2 = pool.rotate("openai")  # move to next key
        result = call_api(key2)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("operon.credential_pool")


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class KeyStatus(str, Enum):
    ACTIVE   = "active"    # healthy, available
    DEGRADED = "degraded"  # errors seen but not exhausted
    COOLING  = "cooling"   # in backoff window
    BANNED   = "banned"    # permanent auth failure / revoked

class ErrorKind(str, Enum):
    RATE_LIMIT   = "rate_limit"   # 429 — rotate + backoff
    QUOTA        = "quota"        # 402/quota exhausted — mark banned
    AUTH         = "auth"         # 401/403 — mark banned
    TIMEOUT      = "timeout"      # network — degrade, retry
    SERVER       = "server"       # 5xx — degrade
    UNKNOWN      = "unknown"      # degrade

# Seconds to cool off per error type
_BACKOFF: Dict[ErrorKind, float] = {
    ErrorKind.RATE_LIMIT: 60.0,
    ErrorKind.TIMEOUT:    10.0,
    ErrorKind.SERVER:     30.0,
    ErrorKind.UNKNOWN:    20.0,
}

# Keywords for error classification
_ERROR_PATTERNS: Dict[ErrorKind, List[str]] = {
    ErrorKind.RATE_LIMIT: ["rate limit", "429", "too many requests", "ratelimit"],
    ErrorKind.QUOTA:      ["quota", "billing", "402", "payment", "insufficient_quota"],
    ErrorKind.AUTH:       ["401", "403", "unauthorized", "forbidden", "invalid api key",
                           "invalid_api_key", "authentication"],
    ErrorKind.TIMEOUT:    ["timeout", "timed out", "connection", "network"],
    ErrorKind.SERVER:     ["500", "502", "503", "504", "server error", "internal error"],
}


# ---------------------------------------------------------------------------
# KeySlot
# ---------------------------------------------------------------------------

@dataclass
class KeySlot:
    provider:     str
    key:          str
    label:        str     = ""      # human name, e.g. "project-A-key"
    status:       KeyStatus = KeyStatus.ACTIVE
    use_count:    int     = 0
    error_count:  int     = 0
    last_used:    float   = 0.0
    last_error:   float   = 0.0
    cooldown_until: float = 0.0
    last_error_msg: str   = ""
    last_error_kind: str  = ""

    @property
    def is_available(self) -> bool:
        if self.status == KeyStatus.BANNED:
            return False
        if self.status == KeyStatus.COOLING:
            if time.time() > self.cooldown_until:
                self.status = KeyStatus.ACTIVE  # auto-recover
                return True
            return False
        return True

    @property
    def masked_key(self) -> str:
        if len(self.key) <= 8:
            return "****"
        return self.key[:4] + "…" + self.key[-4:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider":    self.provider,
            "label":       self.label or self.masked_key,
            "status":      self.status.value,
            "use_count":   self.use_count,
            "error_count": self.error_count,
            "available":   self.is_available,
            "last_error":  self.last_error_msg[:80] if self.last_error_msg else "",
        }


# ---------------------------------------------------------------------------
# CredentialPool
# ---------------------------------------------------------------------------

class CredentialPool:
    """
    Thread-safe multi-key pool with automatic rotation and backoff.
    """

    def __init__(self) -> None:
        self._slots:    Dict[str, List[KeySlot]] = {}   # provider → [slots]
        self._current:  Dict[str, int]            = {}   # provider → current index
        self._lock = Lock()

    # ── Loading credentials ───────────────────────────────────────────────────

    def add(self, provider: str, key: str, label: str = "") -> None:
        """Add a single key for a provider."""
        provider = provider.lower().strip()
        with self._lock:
            if provider not in self._slots:
                self._slots[provider]  = []
                self._current[provider] = 0
            # Deduplicate
            existing = {s.key for s in self._slots[provider]}
            if key in existing:
                return
            slot = KeySlot(provider=provider, key=key, label=label)
            self._slots[provider].append(slot)
            log.debug("added key for %s: %s (%d total)",
                      provider, slot.masked_key, len(self._slots[provider]))

    def load_from_env(
        self,
        provider: str,
        prefix: str,
        max_keys: int = 20,
    ) -> int:
        """
        Load keys from environment variables.
        Looks for PREFIX, PREFIX_1, PREFIX_2, …, PREFIX_N.
        Returns count of keys loaded.
        """
        loaded = 0
        # Try bare prefix first
        bare = os.environ.get(prefix, "").strip()
        if bare:
            self.add(provider, bare, label=prefix)
            loaded += 1
        # Try numbered variants
        for i in range(1, max_keys + 1):
            k = os.environ.get(f"{prefix}_{i}", "").strip()
            if k:
                self.add(provider, k, label=f"{prefix}_{i}")
                loaded += 1
        log.info("load_from_env(%s, %s): %d key(s) loaded", provider, prefix, loaded)
        return loaded

    def load_from_file(self, path: str) -> int:
        """
        Load keys from a JSON file.
        Format: {"openai": ["sk-...", "sk-..."], "anthropic": ["sk-ant-..."]}
        or:     {"openai": [{"key": "sk-...", "label": "prod"}, ...], ...}
        Returns total keys loaded.
        """
        total = 0
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for provider, entries in data.items():
                if isinstance(entries, list):
                    for e in entries:
                        if isinstance(e, str):
                            self.add(provider, e)
                            total += 1
                        elif isinstance(e, dict) and "key" in e:
                            self.add(provider, e["key"], label=e.get("label", ""))
                            total += 1
                elif isinstance(entries, str):
                    self.add(provider, entries)
                    total += 1
        except Exception as ex:
            log.warning("load_from_file(%s) failed: %s", path, ex)
        return total

    # ── Getting & rotating keys ───────────────────────────────────────────────

    def get(self, provider: str) -> Optional[str]:
        """
        Return the best available key for provider.
        Tries current slot first; rotates to next available if needed.
        Returns None if no keys are available.
        """
        provider = provider.lower().strip()
        with self._lock:
            slots = self._slots.get(provider)
            if not slots:
                log.warning("get(%s): no keys registered", provider)
                return None

            idx = self._current.get(provider, 0)
            # Try from current position first
            for offset in range(len(slots)):
                i = (idx + offset) % len(slots)
                slot = slots[i]
                if slot.is_available:
                    if offset > 0:
                        # We had to skip — update current
                        self._current[provider] = i
                    slot.use_count += 1
                    slot.last_used  = time.time()
                    return slot.key

            log.error("get(%s): all %d keys unavailable", provider, len(slots))
            return None

    def rotate(self, provider: str) -> Optional[str]:
        """
        Mark the current key as degraded and advance to the next available key.
        Returns the new key or None.
        """
        provider = provider.lower().strip()
        with self._lock:
            slots = self._slots.get(provider)
            if not slots:
                return None

            idx    = self._current.get(provider, 0)
            current_slot = slots[idx]
            if current_slot.status == KeyStatus.ACTIVE:
                current_slot.status = KeyStatus.DEGRADED

            # Find next available
            for offset in range(1, len(slots) + 1):
                i = (idx + offset) % len(slots)
                if slots[i].is_available:
                    self._current[provider] = i
                    slots[i].use_count += 1
                    slots[i].last_used  = time.time()
                    log.info("rotated %s key from slot %d → slot %d", provider, idx, i)
                    return slots[i].key

            log.warning("rotate(%s): no healthy key to rotate to", provider)
            return None

    # ── Feedback ──────────────────────────────────────────────────────────────

    def report_success(self, provider: str, key: str) -> None:
        """Mark a key as healthy after a successful call."""
        provider = provider.lower().strip()
        with self._lock:
            slot = self._find_slot(provider, key)
            if slot:
                slot.status      = KeyStatus.ACTIVE
                slot.error_count = max(0, slot.error_count - 1)

    def report_error(
        self,
        provider: str,
        key: str,
        error: Any,
        kind: Optional[ErrorKind] = None,
    ) -> Tuple[KeyStatus, Optional[str]]:
        """
        Record an error for a key. Returns (new_status, next_key_or_None).
        Automatically rotates if error is a ban or quota type.
        """
        provider = provider.lower().strip()
        error_str = str(error).lower()
        kind = kind or self._classify_error(error_str)

        with self._lock:
            slot = self._find_slot(provider, key)
            if not slot:
                return KeyStatus.ACTIVE, None

            slot.error_count   += 1
            slot.last_error     = time.time()
            slot.last_error_msg = str(error)[:200]
            slot.last_error_kind = kind.value

            if kind in (ErrorKind.AUTH, ErrorKind.QUOTA):
                slot.status = KeyStatus.BANNED
                log.warning("key %s BANNED (%s): %s", slot.masked_key, provider, kind.value)
            elif kind == ErrorKind.RATE_LIMIT:
                slot.status         = KeyStatus.COOLING
                slot.cooldown_until = time.time() + _BACKOFF[ErrorKind.RATE_LIMIT]
                log.info("key %s cooling for %.0fs (%s rate-limited)",
                         slot.masked_key, _BACKOFF[ErrorKind.RATE_LIMIT], provider)
            else:
                slot.status         = KeyStatus.COOLING
                backoff             = _BACKOFF.get(kind, _BACKOFF[ErrorKind.UNKNOWN])
                slot.cooldown_until = time.time() + backoff

        # Rotate to next key
        next_key = self.rotate(provider)
        return slot.status, next_key

    # ── With-retry helper ─────────────────────────────────────────────────────

    def call_with_retry(
        self,
        provider: str,
        fn: Any,          # callable(key) → result
        max_tries: int = 3,
    ) -> Tuple[Any, bool]:
        """
        Call fn(key) with automatic key rotation on failure.
        Returns (result, success_bool).
        """
        for attempt in range(max_tries):
            key = self.get(provider)
            if not key:
                log.error("call_with_retry(%s): no keys left after %d attempts",
                          provider, attempt)
                return None, False
            try:
                result = fn(key)
                self.report_success(provider, key)
                return result, True
            except Exception as exc:
                log.warning("attempt %d/%d for %s failed: %s",
                            attempt + 1, max_tries, provider, exc)
                _, next_key = self.report_error(provider, key, exc)
                if next_key is None and attempt < max_tries - 1:
                    log.warning("no more keys for %s, stopping retries", provider)
                    break
        return None, False

    # ── Status & introspection ────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return per-provider health summary."""
        with self._lock:
            out = {}
            for provider, slots in self._slots.items():
                active   = [s for s in slots if s.status == KeyStatus.ACTIVE]
                cooling  = [s for s in slots if s.status == KeyStatus.COOLING]
                banned   = [s for s in slots if s.status == KeyStatus.BANNED]
                degraded = [s for s in slots if s.status == KeyStatus.DEGRADED]
                out[provider] = {
                    "total":    len(slots),
                    "active":   len(active),
                    "cooling":  len(cooling),
                    "banned":   len(banned),
                    "degraded": len(degraded),
                    "current_slot": self._current.get(provider, 0),
                    "keys":     [s.to_dict() for s in slots],
                }
            return out

    def providers(self) -> List[str]:
        with self._lock:
            return list(self._slots.keys())

    def key_count(self, provider: str) -> int:
        with self._lock:
            return len(self._slots.get(provider.lower(), []))

    def available_count(self, provider: str) -> int:
        with self._lock:
            return sum(1 for s in self._slots.get(provider.lower(), []) if s.is_available)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _find_slot(self, provider: str, key: str) -> Optional[KeySlot]:
        for slot in self._slots.get(provider, []):
            if slot.key == key:
                return slot
        return None

    @staticmethod
    def _classify_error(error_str: str) -> ErrorKind:
        error_lower = error_str.lower()
        for kind, patterns in _ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in error_lower:
                    return kind
        return ErrorKind.UNKNOWN


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_pool: Optional[CredentialPool] = None


def get_pool() -> CredentialPool:
    """Return the session-scoped default credential pool."""
    global _default_pool
    if _default_pool is None:
        _default_pool = CredentialPool()
        # Auto-load from common environment variable patterns
        _auto_load(_default_pool)
    return _default_pool


def _auto_load(pool: CredentialPool) -> None:
    """Auto-load keys from well-known environment variable names."""
    providers = {
        "openai":    ["OPENAI_API_KEY", "OPENAI_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "gemini":    ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "cohere":    ["COHERE_API_KEY"],
        "mistral":   ["MISTRAL_API_KEY"],
        "groq":      ["GROQ_API_KEY"],
        "perplexity":["PERPLEXITY_API_KEY"],
        "together":  ["TOGETHER_API_KEY"],
        "fireworks": ["FIREWORKS_API_KEY"],
    }
    for provider, prefixes in providers.items():
        for prefix in prefixes:
            pool.load_from_env(provider, prefix)
