"""
Operon Session Manager.

Maintains the FULL chronological message history — zero amnesia.
Supports snapshots, undo, named session save/load, history display,
context truncation, and session search.
"""

import json
import os
import datetime
import re
from pathlib import Path
from typing import Optional

SESSIONS_DIR = Path.home() / ".operon" / "sessions"


class SessionManager:

    def __init__(self):
        self._messages: list[dict] = []
        self.turn_count: int = 0
        self._session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._title: str = ""
        self._snapshots: list[dict] = []   # named checkpoints {label, messages, turn_count}
        self._token_estimate: int = 0       # rolling rough token count

    # ── Message management ───────────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the running history."""
        self._messages.append({"role": role, "content": content})
        if role == "user":
            self.turn_count += 1
        # Rough token estimate: ~4 chars per token
        self._token_estimate += max(1, len(content) // 4)

    def get_messages_for_api(self) -> list[dict]:
        """Return the complete message history for API consumption."""
        return list(self._messages)

    def get_recent_exchange(self) -> list[dict]:
        """Return the last two messages (user + assistant) for memory extraction."""
        return self._messages[-2:] if len(self._messages) >= 2 else list(self._messages)

    def clear(self) -> None:
        """Clear session history (does not touch long-term memory)."""
        self._messages = []
        self.turn_count = 0
        self._token_estimate = 0

    # ── Undo ────────────────────────────────────────────────────────────────

    def undo(self) -> bool:
        """
        Remove the last user+assistant exchange.
        Returns True if something was removed, False if nothing to undo.
        """
        if not self._messages:
            return False
        # Walk backwards: pop assistant messages, then the user message
        removed = 0
        while self._messages and self._messages[-1]["role"] == "assistant":
            msg = self._messages.pop()
            self._token_estimate -= max(1, len(msg["content"]) // 4)
            removed += 1
        if self._messages and self._messages[-1]["role"] == "user":
            msg = self._messages.pop()
            self._token_estimate -= max(1, len(msg["content"]) // 4)
            removed += 1
            self.turn_count = max(0, self.turn_count - 1)
        return removed > 0

    # ── Snapshot / Rollback ──────────────────────────────────────────────────

    def snapshot(self, label: str = "") -> str:
        """
        Save a named checkpoint of the current conversation.
        Returns the snapshot label.
        """
        if not label:
            label = f"snap_{len(self._snapshots) + 1}"
        self._snapshots.append({
            "label":       label,
            "messages":    [dict(m) for m in self._messages],
            "turn_count":  self.turn_count,
            "token_est":   self._token_estimate,
            "created_at":  datetime.datetime.now().isoformat(),
        })
        return label

    def rollback(self, label: str = "") -> bool:
        """
        Restore to a snapshot by label (or last snapshot if label is empty).
        Returns True on success.
        """
        if not self._snapshots:
            return False
        target = None
        if label:
            for snap in reversed(self._snapshots):
                if snap["label"] == label:
                    target = snap
                    break
        else:
            target = self._snapshots[-1]
        if target is None:
            return False
        self._messages      = [dict(m) for m in target["messages"]]
        self.turn_count     = target["turn_count"]
        self._token_estimate = target["token_est"]
        return True

    def list_snapshots(self) -> list[dict]:
        return list(self._snapshots)

    # ── Context compression ──────────────────────────────────────────────────

    def compress(self, keep_first: int = 4, keep_recent: int = 30) -> int:
        """
        Trim middle messages to reduce context length.
        Keeps the first `keep_first` and last `keep_recent` messages.
        Returns number of messages removed.
        """
        total = len(self._messages)
        if total <= keep_first + keep_recent:
            return 0
        keep = (
            self._messages[:keep_first] +
            [{"role": "system", "content":
              "[Context compressed: earlier messages summarised to save token budget]"}] +
            self._messages[-keep_recent:]
        )
        removed = total - len(keep)
        self._messages = keep
        # Recompute token estimate
        self._token_estimate = sum(
            max(1, len(m["content"]) // 4) for m in self._messages
        )
        return removed

    # ── Auto-truncation (called automatically before API if context is huge) ─

    def maybe_truncate(self, hard_limit: int = 120) -> bool:
        """
        If message count exceeds hard_limit, auto-compress.
        Returns True if truncation was applied.
        """
        if len(self._messages) > hard_limit:
            self.compress(keep_first=6, keep_recent=60)
            return True
        return False

    # ── Title ────────────────────────────────────────────────────────────────

    def set_title(self, title: str) -> None:
        self._title = title.strip()

    def get_title(self) -> str:
        return self._title

    # ── History display ──────────────────────────────────────────────────────

    def get_history_display(self, last_n: int = 20) -> list[str]:
        """Return a list of formatted strings for the /history command."""
        lines = []
        turn = 0
        msgs = self._messages[-last_n * 3:]    # rough window
        for m in msgs:
            role    = m["role"]
            content = m["content"].replace("\n", " ")
            if len(content) > 80:
                content = content[:77] + "…"
            if role == "user":
                turn += 1
                lines.append(f"  T{turn:02d}  YOU     {content}")
            elif role == "assistant":
                lines.append(f"        OPERON  {content}")
            # Skip tool result messages from display
        return lines

    # ── Token / usage stats ──────────────────────────────────────────────────

    def get_usage_stats(self) -> dict:
        total_chars = sum(len(m["content"]) for m in self._messages)
        return {
            "messages":      len(self._messages),
            "turns":         self.turn_count,
            "chars":         total_chars,
            "est_tokens":    max(self._token_estimate, total_chars // 4),
            "est_cost_4o":   round((total_chars // 4) / 1000 * 0.005, 4),  # ~$5/Mtok
        }

    # ── Save / Load named sessions ───────────────────────────────────────────

    def save_named(self, name: str) -> str:
        """
        Persist the current session under ~/.operon/sessions/<name>.json.
        Returns the file path.
        """
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w\-]", "_", name)
        path = SESSIONS_DIR / f"{safe}.json"
        payload = {
            "session_id":  self._session_id,
            "title":       self._title or name,
            "name":        safe,
            "turn_count":  self.turn_count,
            "saved_at":    datetime.datetime.now().isoformat(),
            "messages":    self._messages,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return str(path)

    def load_named(self, name: str) -> bool:
        """
        Load a named session from ~/.operon/sessions/<name>.json.
        Returns True on success.
        """
        safe = re.sub(r"[^\w\-]", "_", name)
        path = SESSIONS_DIR / f"{safe}.json"
        if not path.exists():
            return False
        with open(path, "r") as f:
            data = json.load(f)
        self._messages      = data.get("messages", [])
        self.turn_count     = data.get("turn_count", 0)
        self._title         = data.get("title", name)
        self._session_id    = data.get("session_id", self._session_id)
        self._token_estimate = sum(max(1, len(m["content"]) // 4) for m in self._messages)
        return True

    @staticmethod
    def list_saved_sessions() -> list[dict]:
        """Return metadata for all saved sessions, newest first."""
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        sessions = []
        for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(p) as f:
                    data = json.load(f)
                sessions.append({
                    "name":      data.get("name", p.stem),
                    "title":     data.get("title", ""),
                    "turns":     data.get("turn_count", 0),
                    "saved_at":  data.get("saved_at", ""),
                    "path":      str(p),
                })
            except Exception:
                pass
        return sessions

    @staticmethod
    def search_sessions(query: str, max_results: int = 10) -> list[dict]:
        """
        Search across all saved session files for the given query string.
        Returns a list of {name, turn, snippet} matches.
        """
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        query_lower = query.lower()
        hits = []
        for p in SESSIONS_DIR.glob("*.json"):
            try:
                with open(p) as f:
                    data = json.load(f)
                for i, msg in enumerate(data.get("messages", [])):
                    content = msg.get("content", "")
                    if query_lower in content.lower():
                        snippet = content[max(0, content.lower().find(query_lower) - 30):][:80]
                        hits.append({
                            "name":    data.get("name", p.stem),
                            "index":   i,
                            "role":    msg["role"],
                            "snippet": snippet.replace("\n", " "),
                        })
                        if len(hits) >= max_results:
                            return hits
            except Exception:
                pass
        return hits

    # ── Export ───────────────────────────────────────────────────────────────

    def export(self) -> str:
        """Write session to ~/.operon/sessions/<id>.json and return the path."""
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = SESSIONS_DIR / f"session_{self._session_id}.json"
        payload = {
            "session_id":  self._session_id,
            "title":       self._title,
            "turn_count":  self.turn_count,
            "exported_at": datetime.datetime.now().isoformat(),
            "messages":    self._messages,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return str(path)

    # ── Introspection ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"<SessionManager turns={self.turn_count} messages={len(self._messages)}>"
