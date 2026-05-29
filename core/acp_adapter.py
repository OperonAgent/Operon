"""
Operon ACP (Agent Control Protocol) Adapter.

Adapted from Hermes Agent acp_adapter/ and OpenClaw src/acp/.

ACP is a lightweight inter-agent coordination protocol that allows:
  1. Parent agents to spawn and control child agents
  2. Agents to emit structured events to a shared event ledger
  3. Tool-call permission relay (child asks parent before executing)
  4. Status/progress broadcasts between agents

This adapter provides a minimal Python implementation of the ACP protocol
suitable for single-process multi-agent scenarios as well as socket-based
multi-process setups.

Event types:
  agent_started     — agent began running
  agent_finished    — agent finished (with outcome)
  tool_requested    — agent wants to run a tool
  tool_permitted    — permission granted
  tool_denied       — permission denied
  progress          — intermediate status update
  message           — free-form message between agents
  error             — agent error
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


# ── Event types ────────────────────────────────────────────────────────────────

class ACPEventType(str, Enum):
    AGENT_STARTED    = "agent_started"
    AGENT_FINISHED   = "agent_finished"
    TOOL_REQUESTED   = "tool_requested"
    TOOL_PERMITTED   = "tool_permitted"
    TOOL_DENIED      = "tool_denied"
    PROGRESS         = "progress"
    MESSAGE          = "message"
    ERROR            = "error"


# ── Event dataclass ────────────────────────────────────────────────────────────

@dataclass
class ACPEvent:
    event_id:   str
    event_type: ACPEventType
    agent_id:   str
    timestamp:  float
    payload:    dict  = field(default_factory=dict)
    parent_id:  str   = ""
    session_id: str   = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "ACPEvent":
        return cls(
            event_id   = d["event_id"],
            event_type = ACPEventType(d["event_type"]),
            agent_id   = d["agent_id"],
            timestamp  = d["timestamp"],
            payload    = d.get("payload", {}),
            parent_id  = d.get("parent_id", ""),
            session_id = d.get("session_id", ""),
        )


# ── Event ledger ───────────────────────────────────────────────────────────────

class ACPEventLedger:
    """
    In-memory (+ optional file-backed) event ledger.

    Subscribers can register callbacks for specific event types.
    """

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        max_events:   int            = 1000,
    ) -> None:
        self._events:      list[ACPEvent]                     = []
        self._lock         = threading.Lock()
        self._subscribers: dict[str, list[Callable[[ACPEvent], None]]] = {}
        self._persist_path = persist_path
        self._max_events   = max_events

        if persist_path and persist_path.exists():
            self._load(persist_path)

    # ── Publish ───────────────────────────────────────────────────────────────

    def publish(self, event: ACPEvent) -> None:
        """Publish an event to the ledger and notify subscribers."""
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

        # Notify subscribers (outside lock to avoid deadlock)
        for handler in self._subscribers.get(event.event_type.value, []):
            try:
                handler(event)
            except Exception:
                pass
        for handler in self._subscribers.get("*", []):
            try:
                handler(event)
            except Exception:
                pass

        if self._persist_path:
            self._append_to_file(event)

    # ── Subscribe ─────────────────────────────────────────────────────────────

    def subscribe(
        self,
        event_type: str,   # ACPEventType value or "*" for all
        handler:    Callable[[ACPEvent], None],
    ) -> None:
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(
        self,
        event_type: str,
        handler:    Callable[[ACPEvent], None],
    ) -> None:
        with self._lock:
            handlers = self._subscribers.get(event_type, [])
            self._subscribers[event_type] = [h for h in handlers if h is not handler]

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_events(
        self,
        agent_id:   Optional[str]          = None,
        event_type: Optional[ACPEventType] = None,
        since:      Optional[float]        = None,
        limit:      int                    = 100,
    ) -> list[ACPEvent]:
        with self._lock:
            events = list(self._events)

        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    def get_latest(self, agent_id: str) -> Optional[ACPEvent]:
        with self._lock:
            for e in reversed(self._events):
                if e.agent_id == agent_id:
                    return e
        return None

    # ── Persistence ───────────────────────────────────────────────────────────

    def _append_to_file(self, event: ACPEvent) -> None:
        try:
            with open(str(self._persist_path), "a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")
        except Exception:
            pass

    def _load(self, path: Path) -> None:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self._events.append(ACPEvent.from_dict(json.loads(line)))
        except Exception:
            pass


# ── ACP Agent ─────────────────────────────────────────────────────────────────

class ACPAgent:
    """
    Represents a single agent in the ACP network.
    Provides methods for emitting events and requesting permissions.
    """

    def __init__(
        self,
        agent_id:   str,
        ledger:     ACPEventLedger,
        parent_id:  str = "",
        session_id: str = "",
    ) -> None:
        self.agent_id   = agent_id
        self.ledger     = ledger
        self.parent_id  = parent_id
        self.session_id = session_id
        self._permission_callbacks: dict[str, Callable[[str, dict], bool]] = {}

    # ── Emit helpers ──────────────────────────────────────────────────────────

    def _emit(self, event_type: ACPEventType, payload: dict = {}) -> ACPEvent:
        event = ACPEvent(
            event_id   = str(uuid.uuid4())[:12],
            event_type = event_type,
            agent_id   = self.agent_id,
            timestamp  = time.time(),
            payload    = payload,
            parent_id  = self.parent_id,
            session_id = self.session_id,
        )
        self.ledger.publish(event)
        return event

    def started(self, task: str = "") -> ACPEvent:
        return self._emit(ACPEventType.AGENT_STARTED, {"task": task})

    def finished(self, outcome: str, summary: str = "") -> ACPEvent:
        return self._emit(ACPEventType.AGENT_FINISHED, {
            "outcome": outcome,
            "summary": summary,
        })

    def progress(self, message: str, percent: float = 0.0) -> ACPEvent:
        return self._emit(ACPEventType.PROGRESS, {
            "message": message,
            "percent": percent,
        })

    def send_message(self, to: str, text: str, **extra) -> ACPEvent:
        return self._emit(ACPEventType.MESSAGE, {
            "to":   to,
            "text": text,
            **extra,
        })

    def error(self, message: str, exc: Optional[str] = None) -> ACPEvent:
        return self._emit(ACPEventType.ERROR, {
            "message": message,
            "exception": exc or "",
        })

    # ── Permission relay ──────────────────────────────────────────────────────

    def request_permission(
        self,
        tool_name: str,
        params:    dict,
        timeout_s: float = 5.0,
    ) -> bool:
        """
        Ask for permission to run a tool.

        If a permission callback is registered, calls it synchronously.
        Otherwise defaults to True (permit).

        In a real multi-process setup this would send a request to the
        parent agent and wait for a TOOL_PERMITTED / TOOL_DENIED event.
        """
        self._emit(ACPEventType.TOOL_REQUESTED, {
            "tool":   tool_name,
            "params": params,
        })

        callback = self._permission_callbacks.get(tool_name) or \
                   self._permission_callbacks.get("*")
        if callback:
            try:
                permitted = callback(tool_name, params)
            except Exception:
                permitted = True
        else:
            permitted = True   # default: allow

        evt_type = ACPEventType.TOOL_PERMITTED if permitted else ACPEventType.TOOL_DENIED
        self._emit(evt_type, {"tool": tool_name, "permitted": permitted})
        return permitted

    def set_permission_callback(
        self,
        tool_name: str,
        callback:  Callable[[str, dict], bool],
    ) -> None:
        """
        Register a callback for permission decisions.
        Use tool_name="*" to catch all tools.
        """
        self._permission_callbacks[tool_name] = callback


# ── Module-level default ledger ────────────────────────────────────────────────

_default_ledger: Optional[ACPEventLedger] = None


def get_ledger() -> ACPEventLedger:
    """Return the module-level default event ledger."""
    global _default_ledger
    if _default_ledger is None:
        persist = Path.home() / ".operon" / "acp_events.jsonl"
        _default_ledger = ACPEventLedger(persist_path=persist)
    return _default_ledger


def make_agent(
    agent_id:   Optional[str] = None,
    parent_id:  str           = "",
    session_id: str           = "",
) -> ACPAgent:
    """Create a new ACPAgent connected to the default ledger."""
    aid = agent_id or str(uuid.uuid4())[:12]
    return ACPAgent(aid, get_ledger(), parent_id=parent_id, session_id=session_id)
