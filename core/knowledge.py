"""
Operon KnowledgeBase — permanent key-value facts store.

Saved to ~/.operon/knowledge.json and injected into EVERY session's
system prompt.  Unlike conversation memory (SQLite FTS5), this is
structured, explicitly keyed, and never auto-expires.

Typical uses:
  user_name, user_email, project_path, preferred_language,
  coding_style, api_base_urls, timezone, any long-lived preference.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

KNOWLEDGE_PATH = Path.home() / ".operon" / "knowledge.json"


class KnowledgeBase:

    def __init__(self):
        KNOWLEDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        try:
            if KNOWLEDGE_PATH.exists():
                raw = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            pass
        return {}

    def _save(self) -> None:
        try:
            KNOWLEDGE_PATH.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        """Store or overwrite a fact."""
        key = key.strip().lower().replace(" ", "_")
        self._data[key] = {
            "value":   str(value),
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self._save()

    def get(self, key: str) -> Optional[str]:
        """Return the stored value string, or None if key not found."""
        key = key.strip().lower().replace(" ", "_")
        entry = self._data.get(key)
        if entry is None:
            return None
        # Support both the new dict format and plain string values (legacy)
        if isinstance(entry, dict):
            return entry.get("value")
        return str(entry)

    def get_all(self) -> dict:
        """Return {key: value_string} for all stored facts."""
        result = {}
        for k, v in self._data.items():
            result[k] = v["value"] if isinstance(v, dict) else str(v)
        return result

    def delete(self, key: str) -> bool:
        key = key.strip().lower().replace(" ", "_")
        if key in self._data:
            del self._data[key]
            self._save()
            return True
        return False

    def clear(self) -> None:
        self._data = {}
        self._save()

    def __len__(self) -> int:
        return len(self._data)

    # ── System prompt injection ───────────────────────────────────────────────

    def as_system_block(self) -> str:
        """
        Return a formatted block for injection into the system prompt.
        Empty string if nothing is stored.
        """
        if not self._data:
            return ""
        lines = [
            "════════════════════════════════════════════════════",
            "PERMANENT KNOWLEDGE  (persists across all sessions)",
            "════════════════════════════════════════════════════",
        ]
        for k, v in self._data.items():
            val   = v["value"]   if isinstance(v, dict) else str(v)
            lines.append(f"  {k}: {val}")
        lines.append(
            "Use knowledge_set to update these facts when you learn new information."
        )
        return "\n".join(lines)
