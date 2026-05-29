"""
Operon Tool Result Storage — Large output persistence.

Matches Hermes tool_result_storage.py depth.

When a tool returns output exceeding the threshold, the full output is written
to a temp file and the LLM receives a preview + file path reference instead.
This prevents context-window overflow while keeping all data accessible.

Defense operates at three levels:
  1. Per-tool output cap (individual tool truncates before returning)
  2. Per-result persistence (this module) — large outputs → temp file
  3. Per-turn aggregate budget — total size cap across all results per turn

Usage:
    from core.tool_result_storage import ToolResultStorage

    storage = ToolResultStorage()
    result  = {"success": True, "output": very_large_string}
    result  = storage.maybe_persist("shell_exec", "call_id_123", result)
    # If output was large, result["output"] is now a preview + file path
    # Full output accessible via result["_full_output_path"]
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
log = logging.getLogger("operon.tool_result_storage")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default threshold: outputs larger than this get persisted to disk
DEFAULT_THRESHOLD_CHARS = 8_000

# Hard maximum chars before we truncate the preview in the LLM context
PREVIEW_CHARS = 2_000

# Maximum aggregate chars across all tool results in one turn
MAX_TURN_BUDGET_CHARS = 120_000

# Storage directory (inside system tmp)
_STORE_DIR = os.path.join(tempfile.gettempdir(), "operon-results")

# Per-tool overrides: some tools always produce large output
_TOOL_THRESHOLDS: Dict[str, int] = {
    "shell_exec":       6_000,
    "git_diff":         4_000,
    "db_query":         10_000,
    "data_analysis":    12_000,
    "dir_list":         5_000,
    "file_read":        10_000,
    "web_scrape":       8_000,
    "duckduckgo_search": 6_000,
    "pdf_extract":      12_000,
}


# ---------------------------------------------------------------------------
# Persisted result record
# ---------------------------------------------------------------------------

@dataclass
class PersistedResult:
    call_id:    str
    tool_name:  str
    file_path:  str
    size_chars: int
    created_at: float = field(default_factory=time.time)
    checksum:   str   = ""

    def to_dict(self) -> Dict:
        return {
            "call_id":    self.call_id,
            "tool_name":  self.tool_name,
            "file_path":  self.file_path,
            "size_chars": self.size_chars,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# ToolResultStorage
# ---------------------------------------------------------------------------

class ToolResultStorage:
    """Persists large tool outputs to disk; replaces LLM context with preview."""

    def __init__(
        self,
        threshold_chars:    int  = DEFAULT_THRESHOLD_CHARS,
        preview_chars:      int  = PREVIEW_CHARS,
        max_turn_budget:    int  = MAX_TURN_BUDGET_CHARS,
        store_dir:          str  = _STORE_DIR,
        cleanup_after_secs: int  = 3_600,   # 1 hour
    ) -> None:
        self._threshold        = threshold_chars
        self._preview_chars    = preview_chars
        self._max_turn_budget  = max_turn_budget
        self._store_dir        = store_dir
        self._cleanup_secs     = cleanup_after_secs
        self._session_results: List[PersistedResult] = []
        self._turn_char_total  = 0

        Path(store_dir).mkdir(parents=True, exist_ok=True)

    # ── Per-result persistence ────────────────────────────────────────────────

    def maybe_persist(
        self,
        tool_name: str,
        call_id:   str,
        result:    Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        If the result output is large, persist it to disk and return
        a truncated preview + file path reference.
        """
        output = self._extract_output(result)
        if output is None:
            return result

        threshold = _TOOL_THRESHOLDS.get(tool_name, self._threshold)
        if len(output) <= threshold:
            return result

        # Persist to disk
        persisted = self._write(tool_name, call_id, output)
        if not persisted:
            return result   # Write failed — return original (truncated by caller)

        # Replace output with preview
        preview = self._make_preview(output, tool_name, persisted)
        result  = dict(result)   # shallow copy
        key     = self._output_key(result)
        result[key]                 = preview
        result["_full_output_path"] = persisted.file_path
        result["_output_truncated"] = True
        result["_full_size_chars"]  = persisted.size_chars

        log.info("Persisted %s output: %d chars → %s",
                 tool_name, persisted.size_chars, persisted.file_path)
        return result

    # ── Per-turn aggregate budget ─────────────────────────────────────────────

    def enforce_turn_budget(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        After all tool results in one turn are collected, if total output
        exceeds max_turn_budget, persist the largest non-yet-persisted ones.
        """
        total = sum(len(self._extract_output(r) or "") for r in results)
        if total <= self._max_turn_budget:
            self._turn_char_total = total
            return results

        # Sort by output size descending, persist until under budget
        indexed = [(i, r, len(self._extract_output(r) or "")) for i, r in enumerate(results)]
        indexed.sort(key=lambda x: x[2], reverse=True)

        out = list(results)
        running = total
        for i, result, size in indexed:
            if running <= self._max_turn_budget:
                break
            if result.get("_output_truncated"):
                continue
            output = self._extract_output(result)
            if not output:
                continue
            call_id  = result.get("call_id", f"turn_{i}")
            tool_nm  = result.get("_tool_name", f"tool_{i}")
            persisted = self._write(tool_nm, call_id, output)
            if persisted:
                r2        = dict(result)
                key       = self._output_key(result)
                r2[key]   = self._make_preview(output, tool_nm, persisted)
                r2["_full_output_path"] = persisted.file_path
                r2["_output_truncated"] = True
                out[i]    = r2
                running  -= (size - len(r2[key]))

        self._turn_char_total = running
        return out

    # ── Read back full output ─────────────────────────────────────────────────

    def read_full(self, file_path: str) -> Optional[str]:
        """Read a previously persisted result back from disk."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except Exception as e:
            log.warning("Failed to read persisted result %s: %s", file_path, e)
            return None

    def read_result(self, result: Dict) -> str:
        """Convenience: returns full output from result dict."""
        path = result.get("_full_output_path")
        if path:
            return self.read_full(path) or result.get("output", "")
        return result.get("output", result.get("stdout", ""))

    # ── Session stats ─────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        total_size = sum(r.size_chars for r in self._session_results)
        return {
            "persisted_results": len(self._session_results),
            "total_chars_saved": total_size,
            "turn_char_total":   self._turn_char_total,
            "store_dir":         self._store_dir,
        }

    def list_persisted(self) -> List[Dict]:
        return [r.to_dict() for r in self._session_results]

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup_old(self) -> int:
        """Delete persisted files older than cleanup_after_secs. Returns count deleted."""
        cutoff  = time.time() - self._cleanup_secs
        deleted = 0
        store   = Path(self._store_dir)
        for p in store.glob("operon_result_*.txt"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    deleted += 1
            except Exception:
                pass
        return deleted

    def cleanup_session(self) -> None:
        """Delete all persisted files from this session."""
        for r in self._session_results:
            try:
                Path(r.file_path).unlink(missing_ok=True)
            except Exception:
                pass
        self._session_results.clear()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _write(
        self, tool_name: str, call_id: str, output: str
    ) -> Optional[PersistedResult]:
        try:
            checksum = hashlib.sha256(output.encode("utf-8", "replace")).hexdigest()[:16]
            fname    = f"operon_result_{call_id[:16]}_{checksum}.txt"
            fpath    = os.path.join(self._store_dir, fname)
            with open(fpath, "w", encoding="utf-8", errors="replace") as fh:
                fh.write(output)
            pr = PersistedResult(
                call_id=call_id, tool_name=tool_name,
                file_path=fpath, size_chars=len(output), checksum=checksum)
            self._session_results.append(pr)
            return pr
        except Exception as e:
            log.warning("Failed to persist result: %s", e)
            return None

    def _make_preview(
        self, output: str, tool_name: str, pr: PersistedResult
    ) -> str:
        preview = output[:self._preview_chars]
        if len(output) > self._preview_chars:
            preview += f"\n\n[...output truncated — {pr.size_chars:,} chars total...]"
        return (
            f"{preview}\n\n"
            f"[FULL OUTPUT SAVED]\n"
            f"  Tool:    {tool_name}\n"
            f"  Size:    {pr.size_chars:,} chars\n"
            f"  Path:    {pr.file_path}\n"
            f"  To read: use file_read tool with path={pr.file_path!r}"
        )

    @staticmethod
    def _extract_output(result: Dict) -> Optional[str]:
        """Find the primary output field in a tool result dict."""
        for key in ("output", "stdout", "content", "text", "body", "data"):
            val = result.get(key)
            if isinstance(val, str) and val:
                return val
        return None

    @staticmethod
    def _output_key(result: Dict) -> str:
        for key in ("output", "stdout", "content", "text", "body", "data"):
            if key in result and isinstance(result[key], str):
                return key
        return "output"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_storage: Optional[ToolResultStorage] = None


def get_storage(
    threshold_chars: int = DEFAULT_THRESHOLD_CHARS,
) -> ToolResultStorage:
    """Return the session-scoped default storage instance."""
    global _default_storage
    if _default_storage is None:
        _default_storage = ToolResultStorage(threshold_chars=threshold_chars)
    return _default_storage


def maybe_persist_result(
    tool_name: str,
    call_id:   str,
    result:    Dict[str, Any],
) -> Dict[str, Any]:
    """Convenience: persist a single result using the default storage."""
    return get_storage().maybe_persist(tool_name, call_id, result)


def enforce_turn_budget(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convenience: enforce turn budget across all results."""
    return get_storage().enforce_turn_budget(results)
