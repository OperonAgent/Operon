"""
Operon Goal Tracker — Persistent Long-Horizon Goals.

Inspired by Hermes Agent's /goal command. Goals persist across sessions
and are automatically injected into the system prompt so the agent always
knows what it's working toward.

Usage (slash commands)
----------------------
    /goal set "Launch the new product page by Friday"
    /goal list
    /goal update 1 "Design done, writing copy now"
    /goal complete 1
    /goal delete 1
    /goal clear

Tool usage (agent-callable)
---------------------------
    goal_set(title, description, deadline, priority)
    goal_update(goal_id, progress_note, status)
    goal_list()
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_GOALS_PATH = Path.home() / ".operon" / "goals.json"

_STATUS_ACTIVE    = "active"
_STATUS_COMPLETE  = "complete"
_STATUS_PAUSED    = "paused"
_STATUS_ABANDONED = "abandoned"


# ── Storage helpers ───────────────────────────────────────────────────────────

def _load() -> List[Dict[str, Any]]:
    if _GOALS_PATH.exists():
        try:
            return json.loads(_GOALS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save(goals: List[Dict[str, Any]]) -> None:
    _GOALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GOALS_PATH.write_text(
        json.dumps(goals, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _next_id(goals: List[Dict[str, Any]]) -> int:
    if not goals:
        return 1
    return max(g.get("id", 0) for g in goals) + 1


# ── Public functions (tool-callable) ─────────────────────────────────────────

def goal_set(
    title:       str = "",
    description: str = "",
    deadline:    str = "",
    priority:    str = "medium",
    **_,
) -> dict:
    """
    Create a new persistent goal.

    Args:
        title       — short goal title (required)
        description — detailed description (optional)
        deadline    — target date/time as string e.g. '2026-06-01' (optional)
        priority    — 'high' | 'medium' | 'low' (optional, default 'medium')

    Returns:
        {success, goal_id, title, error}
    """
    if not title:
        return {"success": False, "error": "title is required."}

    goals = _load()
    goal  = {
        "id":          _next_id(goals),
        "title":       title.strip(),
        "description": description.strip(),
        "deadline":    deadline.strip(),
        "priority":    priority.lower() if priority.lower() in ("high", "medium", "low") else "medium",
        "status":      _STATUS_ACTIVE,
        "progress":    [],
        "created_at":  time.time(),
        "updated_at":  time.time(),
    }
    goals.append(goal)
    _save(goals)
    return {
        "success":  True,
        "goal_id":  goal["id"],
        "title":    goal["title"],
        "error":    "",
    }


def goal_update(
    goal_id:       int  = 0,
    progress_note: str  = "",
    status:        str  = "",
    **_,
) -> dict:
    """
    Add a progress note and/or update the status of a goal.

    Args:
        goal_id       — goal ID from goal_list (required)
        progress_note — note describing current progress (optional)
        status        — 'active' | 'paused' | 'complete' | 'abandoned' (optional)

    Returns:
        {success, goal_id, status, error}
    """
    if not goal_id:
        return {"success": False, "error": "goal_id is required."}

    goals   = _load()
    goal    = next((g for g in goals if g["id"] == int(goal_id)), None)
    if goal is None:
        return {"success": False, "error": f"Goal #{goal_id} not found."}

    if progress_note:
        goal["progress"].append({
            "note": progress_note.strip(),
            "time": time.time(),
        })

    if status and status.lower() in (_STATUS_ACTIVE, _STATUS_COMPLETE, _STATUS_PAUSED, _STATUS_ABANDONED):
        goal["status"] = status.lower()

    goal["updated_at"] = time.time()
    _save(goals)
    return {
        "success": True,
        "goal_id": goal["id"],
        "status":  goal["status"],
        "error":   "",
    }


def goal_list(
    status: str = "",
    **_,
) -> dict:
    """
    List all goals, optionally filtered by status.

    Args:
        status — filter by 'active' | 'complete' | 'paused' | 'abandoned' (optional)

    Returns:
        {success, goals: [{id, title, status, priority, deadline, progress_count}], count, error}
    """
    goals = _load()
    if status:
        goals = [g for g in goals if g.get("status") == status.lower()]

    return {
        "success": True,
        "goals": [
            {
                "id":             g["id"],
                "title":          g["title"],
                "description":    g.get("description", ""),
                "status":         g.get("status", _STATUS_ACTIVE),
                "priority":       g.get("priority", "medium"),
                "deadline":       g.get("deadline", ""),
                "progress_count": len(g.get("progress", [])),
                "latest_note":    g["progress"][-1]["note"] if g.get("progress") else "",
            }
            for g in goals
        ],
        "count": len(goals),
        "error": "",
    }


def goal_complete(goal_id: int = 0, **_) -> dict:
    """Mark a goal as complete."""
    return goal_update(goal_id=goal_id, status=_STATUS_COMPLETE)


def goal_delete(goal_id: int = 0, **_) -> dict:
    """
    Permanently delete a goal.

    Args:
        goal_id — goal ID (required)
    """
    if not goal_id:
        return {"success": False, "error": "goal_id is required."}
    goals = _load()
    before = len(goals)
    goals  = [g for g in goals if g["id"] != int(goal_id)]
    if len(goals) == before:
        return {"success": False, "error": f"Goal #{goal_id} not found."}
    _save(goals)
    return {"success": True, "goal_id": goal_id, "error": ""}


# ── System prompt injection ───────────────────────────────────────────────────

def as_system_block() -> str:
    """
    Return a compact block of active goals for injection into the system prompt.
    Returns empty string if no active goals.
    """
    goals = [g for g in _load() if g.get("status") == _STATUS_ACTIVE]
    if not goals:
        return ""

    lines = ["[ACTIVE GOALS]"]
    for g in goals:
        line = f"  #{g['id']} [{g['priority'].upper()}] {g['title']}"
        if g.get("deadline"):
            line += f"  (due: {g['deadline']})"
        if g.get("progress"):
            line += f"\n       Latest: {g['progress'][-1]['note']}"
        lines.append(line)
    lines.append("[END GOALS]")
    return "\n".join(lines)


# ── GoalTracker management class ─────────────────────────────────────────────

class GoalTracker:
    """Thin wrapper for goal management, mirroring Curator / SecretsManager patterns."""

    def set(self, title: str, description: str = "", deadline: str = "",
            priority: str = "medium") -> dict:
        return goal_set(title=title, description=description,
                        deadline=deadline, priority=priority)

    def update(self, goal_id: int, progress_note: str = "", status: str = "") -> dict:
        return goal_update(goal_id=goal_id, progress_note=progress_note, status=status)

    def list_goals(self, status: str = "") -> List[Dict[str, Any]]:
        return goal_list(status=status)["goals"]

    def complete(self, goal_id: int) -> dict:
        return goal_complete(goal_id=goal_id)

    def delete(self, goal_id: int) -> dict:
        return goal_delete(goal_id=goal_id)

    def clear(self) -> int:
        goals = _load()
        count = len(goals)
        _save([])
        return count

    def as_system_block(self) -> str:
        return as_system_block()

    def __len__(self) -> int:
        return len([g for g in _load() if g.get("status") == _STATUS_ACTIVE])
