"""
Operon BtW (By-The-Way) Sidebar Messages.

Adapted from OpenClaw src/agents/btw.ts.

BtW messages are out-of-band informational notes that the agent surfaces
to the user alongside (not instead of) its main response.  They are used
for:
  • Flagging something noteworthy that's off the main topic
  • Surfacing warnings without interrupting the flow
  • Providing context hints the user might find useful later

BtW messages are displayed in a sidebar/footer area of the TUI, separate
from the main assistant response.  They expire after a configurable TTL
and are never included in the LLM's message history (no context bloat).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# ── Severity levels ────────────────────────────────────────────────────────────

class BtWLevel(str, Enum):
    INFO    = "info"     # neutral note
    HINT    = "hint"     # helpful suggestion
    WARN    = "warn"     # potential issue worth noting
    NOTICE  = "notice"   # policy / security notice


# ── Message dataclass ──────────────────────────────────────────────────────────

@dataclass
class BtWMessage:
    id:          str
    level:       BtWLevel
    text:        str
    created_at:  float = field(default_factory=time.time)
    expires_at:  Optional[float] = None   # None = never expires
    source:      str = ""                 # which component created this
    dismissed:   bool = False

    @property
    def expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def active(self) -> bool:
        return not self.dismissed and not self.expired

    def dismiss(self) -> None:
        self.dismissed = True

    def to_display(self) -> str:
        prefix = {
            BtWLevel.INFO:   "ℹ",
            BtWLevel.HINT:   "💡",
            BtWLevel.WARN:   "⚠",
            BtWLevel.NOTICE: "🔒",
        }.get(self.level, "•")
        return f"{prefix} {self.text}"


# ── BtW channel ───────────────────────────────────────────────────────────────

class BtWChannel:
    """
    In-memory channel for BtW sidebar messages.

    Usage::

        btw = BtWChannel()
        btw.post("You might want to save that file.", level=BtWLevel.HINT)
        active = btw.get_active()   # rendered in sidebar
    """

    def __init__(self, default_ttl_seconds: float = 300.0) -> None:
        self._messages:  list[BtWMessage] = []
        self._lock       = threading.Lock()
        self._default_ttl = default_ttl_seconds
        self._listeners: list[Callable[[BtWMessage], None]] = []

    # ── Post ──────────────────────────────────────────────────────────────────

    def post(
        self,
        text:        str,
        level:       BtWLevel = BtWLevel.INFO,
        source:      str      = "",
        ttl_seconds: Optional[float] = None,
    ) -> BtWMessage:
        """Create and enqueue a new BtW message. Returns the message."""
        ttl  = ttl_seconds if ttl_seconds is not None else self._default_ttl
        msg  = BtWMessage(
            id          = str(uuid.uuid4())[:8],
            level       = level,
            text        = text,
            source      = source,
            expires_at  = (time.time() + ttl) if ttl > 0 else None,
        )
        with self._lock:
            self._messages.append(msg)
            # Notify listeners
            for fn in self._listeners:
                try:
                    fn(msg)
                except Exception:
                    pass
        return msg

    def hint(self, text: str, **kw) -> BtWMessage:
        return self.post(text, level=BtWLevel.HINT, **kw)

    def warn(self, text: str, **kw) -> BtWMessage:
        return self.post(text, level=BtWLevel.WARN, **kw)

    def notice(self, text: str, **kw) -> BtWMessage:
        return self.post(text, level=BtWLevel.NOTICE, **kw)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_active(self) -> list[BtWMessage]:
        """Return all non-dismissed, non-expired messages."""
        with self._lock:
            return [m for m in self._messages if m.active]

    def get_all(self) -> list[BtWMessage]:
        with self._lock:
            return list(self._messages)

    def get_by_level(self, level: BtWLevel) -> list[BtWMessage]:
        with self._lock:
            return [m for m in self._messages if m.level == level and m.active]

    # ── Dismiss ───────────────────────────────────────────────────────────────

    def dismiss(self, message_id: str) -> bool:
        with self._lock:
            for m in self._messages:
                if m.id == message_id:
                    m.dismiss()
                    return True
        return False

    def dismiss_all(self) -> int:
        with self._lock:
            count = sum(1 for m in self._messages if m.active)
            for m in self._messages:
                m.dismiss()
        return count

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def sweep_expired(self) -> int:
        """Remove expired messages from the internal list. Returns count removed."""
        with self._lock:
            before = len(self._messages)
            self._messages = [m for m in self._messages if not m.expired]
            return before - len(self._messages)

    # ── Listeners ─────────────────────────────────────────────────────────────

    def add_listener(self, fn: Callable[[BtWMessage], None]) -> None:
        """Register a callback that fires whenever a new message is posted."""
        with self._lock:
            self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[BtWMessage], None]) -> None:
        with self._lock:
            self._listeners = [l for l in self._listeners if l is not fn]

    # ── Render ────────────────────────────────────────────────────────────────

    def render_sidebar(self, max_messages: int = 5) -> str:
        """
        Render active BtW messages for display in the TUI sidebar.
        Returns an empty string if there are no active messages.
        """
        active = self.get_active()
        if not active:
            return ""
        shown = active[-max_messages:]   # most recent N
        lines = ["─── By the way ───────────────────────"]
        for m in shown:
            lines.append(m.to_display())
        lines.append("─────────────────────────────────────")
        return "\n".join(lines)

    def render_inline(self) -> str:
        """
        Render active BtW messages as a compact inline block.
        Suitable for terminal output without a dedicated sidebar panel.
        """
        active = self.get_active()
        if not active:
            return ""
        parts = [m.to_display() for m in active]
        return "\n".join(parts)

    def __len__(self) -> int:
        return len(self.get_active())


# ── Module-level default channel ──────────────────────────────────────────────
# Import and use this singleton throughout the codebase.

_default_channel: Optional[BtWChannel] = None


def get_channel() -> BtWChannel:
    """Return the module-level default BtW channel, creating it if needed."""
    global _default_channel
    if _default_channel is None:
        _default_channel = BtWChannel()
    return _default_channel


# Convenience module-level functions

def post(text: str, level: BtWLevel = BtWLevel.INFO, **kw) -> BtWMessage:
    return get_channel().post(text, level=level, **kw)

def hint(text: str, **kw) -> BtWMessage:
    return get_channel().hint(text, **kw)

def warn(text: str, **kw) -> BtWMessage:
    return get_channel().warn(text, **kw)

def notice(text: str, **kw) -> BtWMessage:
    return get_channel().notice(text, **kw)

def get_active() -> list[BtWMessage]:
    return get_channel().get_active()

def render() -> str:
    return get_channel().render_inline()
