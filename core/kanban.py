"""
Operon Kanban — Full SQLite-backed task management system.

Matches Hermes' kanban_db.py + kanban.py + kanban_tools.py depth.

Features:
  - SQLite-backed with full CRUD and history/audit trail
  - Sub-task decomposition linked to parent
  - Agent-loop integration (agent can create/update/query tickets)
  - Status workflow: todo → in_progress → blocked → review → done → cancelled
  - Priority levels: critical, high, medium, low
  - Labels / tags per task
  - Assignee support
  - Full history log (every change recorded with actor + timestamp)
  - Board view (ASCII kanban columns)
  - Bulk operations (bulk-assign, bulk-label, bulk-status)
  - Import/export JSON
  - Slash command support (/kanban add/list/show/board/etc.)
  - Search by text, label, assignee, status, priority
  - Due dates with overdue detection
  - Sprint / milestone grouping
  - Dependency graph (task A blocks task B)
  - Statistics and burn-down data

Usage:
    from core.kanban import KanbanDB, Task, TaskStatus, TaskPriority

    db = KanbanDB()
    task = db.create("Fix login bug", priority="high", labels=["auth", "bug"])
    db.start(task.id)
    db.complete(task.id, comment="Fixed in commit abc123")
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import logging
log = logging.getLogger("operon.kanban")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = os.path.expanduser("~/.operon/kanban.db")
_SCHEMA_VERSION = 3

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS tasks (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'todo',
    priority     TEXT NOT NULL DEFAULT 'medium',
    assignee     TEXT NOT NULL DEFAULT '',
    parent_id    TEXT REFERENCES tasks(id) ON DELETE SET NULL,
    sprint       TEXT NOT NULL DEFAULT '',
    milestone    TEXT NOT NULL DEFAULT '',
    due_date     TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    started_at   REAL,
    completed_at REAL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS task_labels (
    task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    label   TEXT NOT NULL,
    PRIMARY KEY (task_id, label)
);

CREATE TABLE IF NOT EXISTS task_deps (
    task_id   TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    blocks_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    PRIMARY KEY (task_id, blocks_id)
);

CREATE TABLE IF NOT EXISTS task_history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id   TEXT NOT NULL,
    ts        REAL NOT NULL,
    actor     TEXT NOT NULL DEFAULT 'agent',
    field     TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    comment   TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sprints (
    name       TEXT PRIMARY KEY,
    goal       TEXT NOT NULL DEFAULT '',
    start_date TEXT,
    end_date   TEXT,
    status     TEXT NOT NULL DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_tasks_status   ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
CREATE INDEX IF NOT EXISTS idx_tasks_parent   ON tasks(parent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_sprint   ON tasks(sprint);
CREATE INDEX IF NOT EXISTS idx_history_task   ON task_history(task_id);
"""

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    TODO        = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED     = "blocked"
    REVIEW      = "review"
    DONE        = "done"
    CANCELLED   = "cancelled"

    @classmethod
    def valid(cls) -> List[str]:
        return [s.value for s in cls]

    @classmethod
    def from_str(cls, s: str) -> "TaskStatus":
        s = s.lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "start": "in_progress", "started": "in_progress", "wip": "in_progress",
            "doing": "in_progress", "complete": "done", "completed": "done",
            "finish": "done", "finished": "done", "close": "done", "closed": "done",
            "cancel": "cancelled", "skip": "cancelled", "block": "blocked", "pr": "review",
        }
        s = aliases.get(s, s)
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"Unknown status: {s!r}. Valid: {cls.valid()}")


class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"

    @classmethod
    def valid(cls) -> List[str]:
        return [p.value for p in cls]

    @classmethod
    def from_str(cls, s: str) -> "TaskPriority":
        s = s.lower()
        aliases = {"crit": "critical", "urgent": "critical", "hi": "high",
                   "med": "medium", "normal": "medium", "lo": "low"}
        s = aliases.get(s, s)
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"Unknown priority: {s!r}. Valid: {cls.valid()}")


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id:           str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title:        str   = ""
    description:  str   = ""
    status:       str   = TaskStatus.TODO.value
    priority:     str   = TaskPriority.MEDIUM.value
    assignee:     str   = ""
    parent_id:    Optional[str] = None
    sprint:       str   = ""
    milestone:    str   = ""
    due_date:     Optional[str] = None
    created_at:   float = field(default_factory=time.time)
    updated_at:   float = field(default_factory=time.time)
    started_at:   Optional[float] = None
    completed_at: Optional[float] = None
    metadata:     Dict[str, Any] = field(default_factory=dict)
    labels:       List[str] = field(default_factory=list)
    subtasks:     List["Task"] = field(default_factory=list)
    blocked_by:   List[str] = field(default_factory=list)
    blocks:       List[str] = field(default_factory=list)

    @property
    def status_icon(self) -> str:
        return {"todo": "○", "in_progress": "◉", "blocked": "⊘",
                "review": "◎", "done": "✓", "cancelled": "✗"}.get(self.status, "?")

    @property
    def priority_icon(self) -> str:
        return {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(self.priority, "⚪")

    @property
    def is_overdue(self) -> bool:
        if not self.due_date or self.status in ("done", "cancelled"):
            return False
        try:
            due = datetime.fromisoformat(self.due_date).replace(tzinfo=timezone.utc)
            return due < datetime.now(timezone.utc)
        except ValueError:
            return False

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    @property
    def lead_time_days(self) -> Optional[float]:
        if self.completed_at:
            return (self.completed_at - self.created_at) / 86400
        return None

    @property
    def cycle_time_days(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) / 86400
        return None

    def to_dict(self, include_subtasks: bool = True) -> Dict:
        d = {k: v for k, v in self.__dict__.items()
             if not k.startswith("_") and k != "subtasks"}
        d["is_overdue"] = self.is_overdue
        d["age_days"]   = round(self.age_days, 1)
        if include_subtasks and self.subtasks:
            d["subtasks"] = [s.to_dict(include_subtasks=False) for s in self.subtasks]
        return d

    def one_line(self) -> str:
        due    = f" [due:{self.due_date}{'⚠' if self.is_overdue else ''}]" if self.due_date else ""
        labels = f" [{','.join(self.labels)}]" if self.labels else ""
        assign = f" @{self.assignee}" if self.assignee else ""
        parent = f" ↳{self.parent_id}" if self.parent_id else ""
        return (f"{self.status_icon} [{self.id}] {self.priority_icon} "
                f"{self.title}{parent}{assign}{labels}{due}")


# ---------------------------------------------------------------------------
# History record
# ---------------------------------------------------------------------------

@dataclass
class HistoryRecord:
    id:        int
    task_id:   str
    ts:        float
    actor:     str
    field:     str
    old_value: Optional[str]
    new_value: Optional[str]
    comment:   str

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "task_id": self.task_id, "ts": self.ts,
            "time": datetime.fromtimestamp(self.ts).strftime("%Y-%m-%d %H:%M"),
            "actor": self.actor, "field": self.field,
            "old_value": self.old_value, "new_value": self.new_value,
            "comment": self.comment,
        }


# ---------------------------------------------------------------------------
# KanbanDB
# ---------------------------------------------------------------------------

class KanbanDB:
    """SQLite-backed kanban board for Operon agents."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        row = self._conn.execute("SELECT version FROM schema_version").fetchone()
        if not row:
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,))
            self._conn.commit()

    @contextmanager
    def _tx(self) -> Generator:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def create(self, title: str, description: str = "", priority: str = "medium",
               labels: Optional[List[str]] = None, assignee: str = "",
               parent_id: Optional[str] = None, sprint: str = "",
               milestone: str = "", due_date: Optional[str] = None,
               metadata: Optional[Dict] = None, actor: str = "agent") -> Task:
        if not title or not title.strip():
            raise ValueError("Task title is required.")
        pri  = TaskPriority.from_str(priority).value
        task = Task(title=title.strip(), description=description, priority=pri,
                    assignee=assignee, parent_id=parent_id, sprint=sprint,
                    milestone=milestone, due_date=due_date, metadata=metadata or {})
        with self._tx():
            self._conn.execute("""
                INSERT INTO tasks (id, title, description, status, priority, assignee,
                    parent_id, sprint, milestone, due_date, created_at, updated_at, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (task.id, task.title, task.description, task.status, task.priority,
                  task.assignee, task.parent_id, task.sprint, task.milestone,
                  task.due_date, task.created_at, task.updated_at,
                  json.dumps(task.metadata)))
            if labels:
                for lbl in labels:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO task_labels (task_id, label) VALUES (?,?)",
                        (task.id, lbl.lower().strip()))
                task.labels = [l.lower().strip() for l in labels]
            self._record_history(task.id, "created", None, task.title,
                                 comment=f"priority={pri}", actor=actor)
        return task

    def get(self, task_id: str) -> Optional[Task]:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE id = ? OR id LIKE ?",
            (task_id, f"{task_id}%")).fetchone()
        return self._row_to_task(row, load_relations=True) if row else None

    def update(self, task_id: str, *, title: Optional[str] = None,
               description: Optional[str] = None, priority: Optional[str] = None,
               assignee: Optional[str] = None, sprint: Optional[str] = None,
               milestone: Optional[str] = None, due_date: Optional[str] = None,
               metadata_update: Optional[Dict] = None, actor: str = "agent",
               comment: str = "") -> Optional[Task]:
        task = self.get(task_id)
        if not task:
            return None
        updates: Dict[str, Any] = {}
        if title       is not None: updates["title"]       = title.strip()
        if description is not None: updates["description"] = description
        if priority    is not None: updates["priority"]    = TaskPriority.from_str(priority).value
        if assignee    is not None: updates["assignee"]    = assignee
        if sprint      is not None: updates["sprint"]      = sprint
        if milestone   is not None: updates["milestone"]   = milestone
        if due_date    is not None: updates["due_date"]    = due_date
        if not updates and not metadata_update:
            return task
        updates["updated_at"] = time.time()
        if metadata_update:
            meta = task.metadata or {}
            meta.update(metadata_update)
            updates["metadata"] = json.dumps(meta)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        with self._tx():
            self._conn.execute(
                f"UPDATE tasks SET {set_clause} WHERE id = ?",
                list(updates.values()) + [task.id])
            for f_name, new_val in updates.items():
                if f_name in ("updated_at", "metadata"):
                    continue
                old_val = getattr(task, f_name, None)
                if str(old_val) != str(new_val):
                    self._record_history(task.id, f_name, str(old_val), str(new_val),
                                         comment=comment, actor=actor)
        return self.get(task.id)

    def set_status(self, task_id: str, status: str, comment: str = "",
                   actor: str = "agent") -> Optional[Task]:
        task = self.get(task_id)
        if not task:
            return None
        new_status = TaskStatus.from_str(status)
        old_status = task.status
        ts    = time.time()
        extra: Dict[str, Any] = {"status": new_status.value, "updated_at": ts}
        if new_status == TaskStatus.IN_PROGRESS and not task.started_at:
            extra["started_at"] = ts
        if new_status in (TaskStatus.DONE, TaskStatus.CANCELLED) and not task.completed_at:
            extra["completed_at"] = ts
        set_clause = ", ".join(f"{k} = ?" for k in extra)
        with self._tx():
            self._conn.execute(
                f"UPDATE tasks SET {set_clause} WHERE id = ?",
                list(extra.values()) + [task.id])
            self._record_history(task.id, "status", old_status, new_status.value,
                                 comment=comment, actor=actor)
        return self.get(task.id)

    def start(self, task_id: str, comment: str = "", actor: str = "agent") -> Optional[Task]:
        return self.set_status(task_id, "in_progress", comment=comment, actor=actor)

    def block(self, task_id: str, reason: str = "", actor: str = "agent") -> Optional[Task]:
        return self.set_status(task_id, "blocked", comment=reason, actor=actor)

    def review(self, task_id: str, comment: str = "", actor: str = "agent") -> Optional[Task]:
        return self.set_status(task_id, "review", comment=comment, actor=actor)

    def complete(self, task_id: str, comment: str = "", actor: str = "agent") -> Optional[Task]:
        return self.set_status(task_id, "done", comment=comment, actor=actor)

    def cancel(self, task_id: str, reason: str = "", actor: str = "agent") -> Optional[Task]:
        return self.set_status(task_id, "cancelled", comment=reason, actor=actor)

    def delete(self, task_id: str) -> bool:
        task = self.get(task_id)
        if not task:
            return False
        with self._tx():
            self._conn.execute(
                "DELETE FROM tasks WHERE id = ? OR parent_id = ?", (task.id, task.id))
        return True

    # ── Labels ───────────────────────────────────────────────────────────────

    def add_labels(self, task_id: str, labels: List[str]) -> bool:
        task = self.get(task_id)
        if not task:
            return False
        with self._tx():
            for lbl in labels:
                self._conn.execute(
                    "INSERT OR IGNORE INTO task_labels (task_id, label) VALUES (?,?)",
                    (task.id, lbl.lower().strip()))
        return True

    def remove_labels(self, task_id: str, labels: List[str]) -> bool:
        task = self.get(task_id)
        if not task:
            return False
        with self._tx():
            for lbl in labels:
                self._conn.execute(
                    "DELETE FROM task_labels WHERE task_id=? AND label=?",
                    (task.id, lbl.lower()))
        return True

    # ── Dependencies ─────────────────────────────────────────────────────────

    def add_dependency(self, task_id: str, blocks_id: str) -> bool:
        with self._tx():
            self._conn.execute(
                "INSERT OR IGNORE INTO task_deps (task_id, blocks_id) VALUES (?,?)",
                (task_id, blocks_id))
        return True

    def remove_dependency(self, task_id: str, blocks_id: str) -> bool:
        with self._tx():
            self._conn.execute(
                "DELETE FROM task_deps WHERE task_id=? AND blocks_id=?",
                (task_id, blocks_id))
        return True

    def can_start(self, task_id: str) -> Tuple[bool, List[str]]:
        rows = self._conn.execute("""
            SELECT t.id FROM task_deps d
            JOIN tasks t ON t.id = d.task_id
            WHERE d.blocks_id = ? AND t.status NOT IN ('done','cancelled')
        """, (task_id,)).fetchall()
        blockers = [r["id"] for r in rows]
        return (len(blockers) == 0, blockers)

    # ── Sub-tasks ─────────────────────────────────────────────────────────────

    def create_subtask(self, parent_id: str, title: str, description: str = "",
                       priority: str = "medium", actor: str = "agent") -> Optional[Task]:
        parent = self.get(parent_id)
        if not parent:
            return None
        return self.create(title=title, description=description,
                           priority=priority, parent_id=parent.id, actor=actor)

    def get_subtasks(self, parent_id: str) -> List[Task]:
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE parent_id = ? ORDER BY created_at",
            (parent_id,)).fetchall()
        return [self._row_to_task(r, load_relations=True) for r in rows]

    def get_tree(self, task_id: str) -> Optional[Task]:
        task = self.get(task_id)
        if task:
            task.subtasks = self.get_subtasks(task.id)
            for sub in task.subtasks:
                sub.subtasks = self.get_subtasks(sub.id)
        return task

    # ── Queries ───────────────────────────────────────────────────────────────

    def list(self, status: Optional[str] = None, priority: Optional[str] = None,
             assignee: Optional[str] = None, sprint: Optional[str] = None,
             label: Optional[str] = None, parent_id: Optional[str] = None,
             overdue: bool = False, search: Optional[str] = None,
             limit: int = 200, include_subtasks: bool = False) -> List[Task]:
        conditions: List[str] = []
        params: List[Any]     = []

        if not include_subtasks:
            conditions.append("parent_id IS NULL")
        if status:
            statuses      = [TaskStatus.from_str(s).value for s in status.split(",")]
            placeholders  = ",".join("?" * len(statuses))
            conditions.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if priority:
            conditions.append("priority = ?")
            params.append(TaskPriority.from_str(priority).value)
        if assignee:
            conditions.append("assignee = ?")
            params.append(assignee)
        if sprint:
            conditions.append("sprint = ?")
            params.append(sprint)
        if parent_id:
            conditions.append("parent_id = ?")
            params.append(parent_id)
        if overdue:
            now_iso = datetime.now(timezone.utc).date().isoformat()
            conditions.append(
                "due_date IS NOT NULL AND due_date < ? AND status NOT IN ('done','cancelled')")
            params.append(now_iso)
        if search:
            conditions.append("(title LIKE ? OR description LIKE ?)")
            params += [f"%{search}%", f"%{search}%"]

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        order_by = """ORDER BY
            CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                          WHEN 'medium' THEN 2 ELSE 3 END,
            created_at DESC"""

        if label:
            sql = f"""
                SELECT DISTINCT t.* FROM tasks t
                JOIN task_labels l ON l.task_id = t.id
                WHERE l.label = ? {('AND ' + ' AND '.join(conditions)) if conditions else ''}
                {order_by} LIMIT ?"""
            rows = self._conn.execute(sql, [label] + params + [limit]).fetchall()
        else:
            sql = f"SELECT * FROM tasks {where} {order_by} LIMIT ?"
            rows = self._conn.execute(sql, params + [limit]).fetchall()

        return [self._row_to_task(r, load_relations=True) for r in rows]

    def list_active(self) -> List[Task]:
        return self.list(status="todo,in_progress,review,blocked")

    def list_done(self, limit: int = 50) -> List[Task]:
        return self.list(status="done", limit=limit)

    def search(self, query: str, limit: int = 20) -> List[Task]:
        return self.list(search=query, limit=limit)

    # ── History ───────────────────────────────────────────────────────────────

    def get_history(self, task_id: str, limit: int = 50) -> List[HistoryRecord]:
        rows = self._conn.execute("""
            SELECT * FROM task_history WHERE task_id = ?
            ORDER BY ts DESC LIMIT ?
        """, (task_id, limit)).fetchall()
        return [HistoryRecord(**dict(r)) for r in rows]

    def add_comment(self, task_id: str, comment: str, actor: str = "user") -> bool:
        task = self.get(task_id)
        if not task:
            return False
        self._record_history(task.id, "comment", None, None,
                             comment=comment, actor=actor)
        return True

    # ── Board ─────────────────────────────────────────────────────────────────

    def board(self, sprint: Optional[str] = None, max_col_width: int = 32) -> str:
        columns = [
            (TaskStatus.TODO,        "TODO"),
            (TaskStatus.IN_PROGRESS, "IN PROGRESS"),
            (TaskStatus.BLOCKED,     "BLOCKED"),
            (TaskStatus.REVIEW,      "REVIEW"),
            (TaskStatus.DONE,        "DONE"),
        ]
        col_data = {s.value: self.list(status=s.value, sprint=sprint, limit=20)
                    for s, _ in columns}

        width   = max_col_width
        divider = "─" * width
        header  = "  ".join(f"┌{divider}┐" for _ in columns)
        labels  = "  ".join(f"│ {lbl:<{width-2}} │" for _, lbl in columns)
        sep     = "  ".join(f"├{divider}┤" for _ in columns)

        lines = [header, labels, sep]
        max_rows = max((len(col_data[s.value]) for s, _ in columns), default=0)

        for i in range(max_rows):
            row_parts = []
            for status_enum, _ in columns:
                tasks = col_data[status_enum.value]
                if i < len(tasks):
                    t    = tasks[i]
                    cell = f"[{t.id}] {t.title}"
                    if len(cell) > width - 4:
                        cell = cell[:width - 7] + "..."
                    row_parts.append(f"│ {t.priority_icon}{cell:<{width-4}} │")
                else:
                    row_parts.append(f"│{' ' * width}│")
            lines.append("  ".join(row_parts))

        footer = "  ".join(f"└{divider}┘" for _ in columns)
        lines.append(footer)
        counts = "  ".join(f"{lbl}({len(col_data[s.value])})" for s, lbl in columns)
        lines.append(f"  {counts}")
        return "\n".join(lines)

    # ── Summary / stats ───────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        rows = self._conn.execute("""
            SELECT status, priority, COUNT(*) as cnt
            FROM tasks WHERE parent_id IS NULL
            GROUP BY status, priority
        """).fetchall()
        by_status: Dict[str, int]   = {}
        by_priority: Dict[str, int] = {}
        total = 0
        for r in rows:
            by_status[r["status"]]     = by_status.get(r["status"], 0)     + r["cnt"]
            by_priority[r["priority"]] = by_priority.get(r["priority"], 0) + r["cnt"]
            total += r["cnt"]
        overdue   = len(self.list(overdue=True))
        completed = self._conn.execute("""
            SELECT AVG((completed_at - created_at) / 86400.0) as avg_lead
            FROM tasks WHERE status = 'done' AND completed_at IS NOT NULL
        """).fetchone()
        avg_lead  = round(completed["avg_lead"] or 0.0, 1)
        return {"total": total, "by_status": by_status, "by_priority": by_priority,
                "overdue": overdue, "avg_lead_time_days": avg_lead}

    # ── Sprint ────────────────────────────────────────────────────────────────

    def create_sprint(self, name: str, goal: str = "",
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> bool:
        with self._tx():
            self._conn.execute(
                "INSERT OR IGNORE INTO sprints (name, goal, start_date, end_date) VALUES (?,?,?,?)",
                (name, goal, start_date, end_date))
        return True

    def assign_sprint(self, task_ids: List[str], sprint: str) -> int:
        return sum(1 for tid in task_ids if self.update(tid, sprint=sprint))

    def sprint_summary(self, sprint: str) -> Dict[str, Any]:
        rows = self._conn.execute("""
            SELECT status, COUNT(*) as cnt FROM tasks
            WHERE sprint = ? AND parent_id IS NULL GROUP BY status
        """, (sprint,)).fetchall()
        by_status = {r["status"]: r["cnt"] for r in rows}
        total     = sum(by_status.values())
        done      = by_status.get("done", 0)
        return {"sprint": sprint, "total": total, "done": done,
                "completion_pct": round(done / max(total, 1) * 100, 1),
                "by_status": by_status}

    # ── Export / Import ───────────────────────────────────────────────────────

    def export_json(self, include_done: bool = True) -> str:
        status_filter = None if include_done else "todo,in_progress,blocked,review"
        tasks = self.list(status=status_filter, limit=10_000)
        return json.dumps({"exported_at": datetime.now().isoformat(),
                           "total": len(tasks),
                           "tasks": [t.to_dict() for t in tasks]},
                          indent=2, default=str)

    def import_json(self, json_str: str) -> Tuple[int, int]:
        data       = json.loads(json_str)
        tasks_data = data.get("tasks", data) if isinstance(data, dict) else data
        imported = skipped = 0
        for td in tasks_data:
            if self.get(td["id"]):
                skipped += 1
                continue
            with self._tx():
                self._conn.execute("""
                    INSERT INTO tasks (id, title, description, status, priority, assignee,
                        parent_id, sprint, milestone, due_date, created_at, updated_at,
                        started_at, completed_at, metadata)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (td["id"], td.get("title",""), td.get("description",""),
                      td.get("status","todo"), td.get("priority","medium"),
                      td.get("assignee",""), td.get("parent_id"),
                      td.get("sprint",""), td.get("milestone",""), td.get("due_date"),
                      td.get("created_at", time.time()), td.get("updated_at", time.time()),
                      td.get("started_at"), td.get("completed_at"),
                      json.dumps(td.get("metadata", {}))))
                for lbl in td.get("labels", []):
                    self._conn.execute(
                        "INSERT OR IGNORE INTO task_labels VALUES (?,?)", (td["id"], lbl))
            imported += 1
        return imported, skipped

    # ── Bulk ops ──────────────────────────────────────────────────────────────

    def bulk_set_status(self, task_ids: List[str], status: str,
                        comment: str = "", actor: str = "agent") -> int:
        return sum(1 for tid in task_ids
                   if self.set_status(tid, status, comment=comment, actor=actor))

    def bulk_assign(self, task_ids: List[str], assignee: str) -> int:
        return sum(1 for tid in task_ids if self.update(tid, assignee=assignee))

    def bulk_label(self, task_ids: List[str], labels: List[str]) -> int:
        return sum(1 for tid in task_ids if self.add_labels(tid, labels))

    # ── Agent-friendly wrappers ───────────────────────────────────────────────

    def agent_create(self, title: str, **kwargs) -> Dict:
        try:
            task = self.create(title, **kwargs)
            return {"success": True, "task": task.to_dict(), "id": task.id,
                    "message": f"Task [{task.id}] created: {task.title}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def agent_update(self, task_id: str, **kwargs) -> Dict:
        status  = kwargs.pop("status", None)
        comment = kwargs.pop("comment", "")
        try:
            if status:
                task = self.set_status(task_id, status, comment=comment)
            else:
                task = self.update(task_id, comment=comment, **kwargs)
            if not task:
                return {"success": False, "error": f"Task {task_id!r} not found"}
            return {"success": True, "task": task.to_dict()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def agent_list(self, **kwargs) -> Dict:
        try:
            tasks = self.list(**kwargs)
            return {"success": True, "count": len(tasks),
                    "tasks": [t.to_dict(include_subtasks=False) for t in tasks]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def agent_board(self, sprint: Optional[str] = None) -> Dict:
        try:
            return {"success": True, "board": self.board(sprint=sprint)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _row_to_task(self, row: sqlite3.Row, load_relations: bool = False) -> Task:
        d        = dict(row)
        meta_raw = d.pop("metadata", "{}")
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = {}
        task = Task(
            id=d["id"], title=d["title"], description=d.get("description", ""),
            status=d.get("status","todo"), priority=d.get("priority","medium"),
            assignee=d.get("assignee",""), parent_id=d.get("parent_id"),
            sprint=d.get("sprint",""), milestone=d.get("milestone",""),
            due_date=d.get("due_date"), created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()), started_at=d.get("started_at"),
            completed_at=d.get("completed_at"), metadata=meta)
        if load_relations:
            task.labels = [r["label"] for r in self._conn.execute(
                "SELECT label FROM task_labels WHERE task_id = ?", (task.id,)).fetchall()]
            task.blocks = [r["blocks_id"] for r in self._conn.execute(
                "SELECT blocks_id FROM task_deps WHERE task_id = ?", (task.id,)).fetchall()]
            task.blocked_by = [r["task_id"] for r in self._conn.execute(
                "SELECT task_id FROM task_deps WHERE blocks_id = ?", (task.id,)).fetchall()]
        return task

    def _record_history(self, task_id: str, field: str,
                        old_val: Optional[str], new_val: Optional[str],
                        comment: str = "", actor: str = "agent") -> None:
        self._conn.execute("""
            INSERT INTO task_history (task_id, ts, actor, field, old_value, new_value, comment)
            VALUES (?,?,?,?,?,?,?)
        """, (task_id, time.time(), actor, field, old_val, new_val, comment))

    def close(self) -> None:
        self._conn.close()

    def __repr__(self) -> str:
        s = self.summary()
        return (f"<KanbanDB total={s['total']} "
                f"todo={s['by_status'].get('todo',0)} "
                f"wip={s['by_status'].get('in_progress',0)} "
                f"done={s['by_status'].get('done',0)}>")


# ---------------------------------------------------------------------------
# Slash-command handler
# ---------------------------------------------------------------------------

def handle_kanban_command(args: List[str], db: Optional[KanbanDB] = None) -> str:
    """Parse and execute /kanban sub-commands."""
    if db is None:
        db = _default_db()
    if not args:
        return _kanban_help()
    cmd  = args[0].lower()
    rest = args[1:]

    if cmd in ("add", "new", "create"):
        if not rest:
            return "Usage: /kanban add \"Title\" [--priority high] [--label bug,auth]"
        title, rest = _pop_title(rest)
        opts     = _parse_opts(rest)
        priority = opts.get("priority", "medium")
        labels   = [l.strip() for l in opts.get("label","").split(",") if l.strip()]
        assignee = opts.get("assign", opts.get("assignee","")).lstrip("@")
        r = db.agent_create(title, priority=priority, labels=labels, assignee=assignee,
                            sprint=opts.get("sprint",""), due_date=opts.get("due"))
        if r["success"]:
            return f"✓ Created [{r['id']}]: {db.get(r['id']).one_line()}"
        return f"✗ {r['error']}"

    elif cmd in ("list", "ls", "todo", "tasks"):
        opts   = _parse_opts(rest)
        tasks  = db.list(status=opts.get("status"), priority=opts.get("priority"),
                         assignee=opts.get("assign"), sprint=opts.get("sprint"),
                         label=opts.get("label"), search=opts.get("search"),
                         limit=int(opts.get("limit","50")))
        if not tasks:
            return "No tasks found."
        return f"Tasks ({len(tasks)}):\n" + "\n".join(f"  {t.one_line()}" for t in tasks)

    elif cmd in ("show","get","info"):
        if not rest: return "Usage: /kanban show <id>"
        task = db.get_tree(rest[0]) if db.get(rest[0]) else None
        if not task:
            return f"Task {rest[0]!r} not found."
        lines = [
            f"[{task.id}] {task.priority_icon} {task.title}",
            f"  Status:   {task.status_icon} {task.status}",
            f"  Priority: {task.priority}",
            f"  Assignee: {task.assignee or '—'}",
            f"  Sprint:   {task.sprint or '—'}",
            f"  Labels:   {', '.join(task.labels) or '—'}",
            f"  Due:      {task.due_date or '—'}{'  ⚠ OVERDUE' if task.is_overdue else ''}",
            f"  Age:      {task.age_days:.1f} days",
        ]
        if task.description:
            lines.append(f"\n  {task.description}")
        if task.subtasks:
            lines.append(f"\nSubtasks ({len(task.subtasks)}):")
            lines.extend(f"  {s.one_line()}" for s in task.subtasks)
        return "\n".join(lines)

    elif cmd == "start":
        if not rest: return "Usage: /kanban start <id>"
        t = db.start(rest[0], comment=" ".join(rest[1:]))
        return f"✓ Started [{t.id}]" if t else f"✗ Not found"

    elif cmd in ("done","complete","finish"):
        if not rest: return "Usage: /kanban done <id> [comment]"
        t = db.complete(rest[0], comment=" ".join(rest[1:]))
        return f"✓ Completed [{t.id}]: {t.title}" if t else f"✗ Not found"

    elif cmd == "block":
        if not rest: return "Usage: /kanban block <id> [reason]"
        t = db.block(rest[0], reason=" ".join(rest[1:]))
        return f"✓ Blocked [{t.id}]" if t else f"✗ Not found"

    elif cmd == "review":
        if not rest: return "Usage: /kanban review <id>"
        t = db.review(rest[0])
        return f"✓ In review [{t.id}]" if t else f"✗ Not found"

    elif cmd in ("cancel","skip"):
        if not rest: return "Usage: /kanban cancel <id> [reason]"
        t = db.cancel(rest[0], reason=" ".join(rest[1:]))
        return f"✓ Cancelled [{t.id}]" if t else f"✗ Not found"

    elif cmd in ("update","edit","set"):
        if not rest: return "Usage: /kanban update <id> [--title ...] [--priority ...]"
        opts = _parse_opts(rest[1:])
        r = db.agent_update(rest[0], title=opts.get("title"), priority=opts.get("priority"),
                            assignee=opts.get("assign"), sprint=opts.get("sprint"),
                            due_date=opts.get("due"), status=opts.get("status"),
                            comment=opts.get("comment",""))
        return f"✓ Updated [{rest[0]}]" if r["success"] else f"✗ {r['error']}"

    elif cmd in ("sub","subtask"):
        if len(rest) < 2: return "Usage: /kanban sub <parent_id> \"subtask title\""
        title, _ = _pop_title(rest[1:])
        sub = db.create_subtask(rest[0], title)
        return f"✓ Subtask [{sub.id}] → [{rest[0]}]: {sub.title}" if sub else f"✗ Parent not found"

    elif cmd == "assign":
        if len(rest) < 2: return "Usage: /kanban assign <id> @user"
        t = db.update(rest[0], assignee=rest[1].lstrip("@"))
        return f"✓ Assigned [{t.id}] to {t.assignee}" if t else f"✗ Not found"

    elif cmd in ("label","tag"):
        if len(rest) < 2: return "Usage: /kanban label <id> tag1,tag2"
        db.add_labels(rest[0], [l.strip() for l in rest[1].split(",")])
        return f"✓ Labels added to [{rest[0]}]"

    elif cmd in ("dep","depends"):
        if len(rest) < 3 or rest[1].lower() != "blocks":
            return "Usage: /kanban dep <id> blocks <other_id>"
        db.add_dependency(rest[0], rest[2])
        return f"✓ [{rest[0]}] blocks [{rest[2]}]"

    elif cmd in ("board","view"):
        opts = _parse_opts(rest)
        return db.board(sprint=opts.get("sprint"))

    elif cmd in ("history","log","audit"):
        if not rest: return "Usage: /kanban history <id>"
        records = db.get_history(rest[0])
        if not records: return f"No history for [{rest[0]}]"
        lines = [f"History for [{rest[0]}]:"]
        for r in records:
            ts = datetime.fromtimestamp(r.ts).strftime("%m-%d %H:%M")
            if r.field == "comment":
                lines.append(f"  {ts} 💬 {r.actor}: {r.comment}")
            else:
                lines.append(f"  {ts} {r.actor}: {r.field} {r.old_value!r}→{r.new_value!r}"
                             + (f" ({r.comment})" if r.comment else ""))
        return "\n".join(lines)

    elif cmd in ("delete","rm","remove"):
        if not rest: return "Usage: /kanban delete <id>"
        ok = db.delete(rest[0])
        return f"✓ Deleted [{rest[0]}]" if ok else f"✗ Not found"

    elif cmd in ("comment","note"):
        if len(rest) < 2: return "Usage: /kanban comment <id> <text>"
        ok = db.add_comment(rest[0], " ".join(rest[1:]), actor="user")
        return f"✓ Comment added to [{rest[0]}]" if ok else f"✗ Not found"

    elif cmd == "sprint":
        if not rest: return "Usage: /kanban sprint create|board|summary <name>"
        sub_cmd = rest[0].lower()
        if sub_cmd == "create" and len(rest) >= 2:
            opts = _parse_opts(rest[2:])
            db.create_sprint(rest[1], goal=opts.get("goal",""),
                             start_date=opts.get("start"), end_date=opts.get("end"))
            return f"✓ Sprint '{rest[1]}' created"
        elif sub_cmd in ("board","view") and len(rest) >= 2:
            return db.board(sprint=rest[1])
        elif sub_cmd in ("summary","stats") and len(rest) >= 2:
            s = db.sprint_summary(rest[1])
            return (f"Sprint '{s['sprint']}': {s['total']} tasks, "
                    f"{s['done']} done ({s['completion_pct']}%)")
        elif sub_cmd == "assign" and len(rest) >= 3:
            n = db.assign_sprint(rest[2:], rest[1])
            return f"✓ {n} tasks → sprint '{rest[1]}'"
        return "Usage: /kanban sprint create|board|summary|assign <name>"

    elif cmd in ("summary","stats","status"):
        s = db.summary()
        status_str   = "  ".join(f"{k}:{v}" for k,v in s["by_status"].items())
        priority_str = "  ".join(f"{k}:{v}" for k,v in s["by_priority"].items())
        return (f"Kanban: {s['total']} tasks\n"
                f"  Status:   {status_str}\n"
                f"  Priority: {priority_str}\n"
                f"  Overdue:  {s['overdue']}\n"
                f"  Avg lead: {s['avg_lead_time_days']}d")

    elif cmd == "export":
        return db.export_json()

    elif cmd in ("search","find","grep"):
        if not rest: return "Usage: /kanban search <query>"
        tasks = db.search(" ".join(rest))
        return "\n".join(f"  {t.one_line()}" for t in tasks) or "No results."

    elif cmd in ("help","?"):
        return _kanban_help()

    else:
        return f"Unknown kanban sub-command: {cmd!r}\n\n{_kanban_help()}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_db_singleton: Optional[KanbanDB] = None

def _default_db() -> KanbanDB:
    global _db_singleton
    if _db_singleton is None:
        _db_singleton = KanbanDB()
    return _db_singleton

def _pop_title(args: List[str]) -> Tuple[str, List[str]]:
    if not args:
        return "", []
    joined = " ".join(args)
    m = re.match(r'^["\'](.+?)["\']', joined)
    if m:
        title = m.group(1)
        rest  = joined[m.end():].strip()
        return title, rest.split() if rest else []
    return args[0], args[1:]

def _parse_opts(args: List[str]) -> Dict[str, str]:
    opts: Dict[str, str] = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            key = a[2:]
            if i + 1 < len(args) and not args[i+1].startswith("--"):
                opts[key] = args[i+1]; i += 2
            else:
                opts[key] = "true"; i += 1
        else:
            i += 1
    return opts

def _kanban_help() -> str:
    return """\
/kanban commands:
  add "title" [--priority critical|high|medium|low] [--label tag1,tag2] [--assign @user] [--sprint name] [--due 2025-12-31]
  list [--status todo|in_progress|blocked|review|done] [--priority high] [--label bug] [--search text]
  show <id>          — full details + subtasks
  start <id>         — → in_progress
  done <id> [note]   — → done
  block <id> [why]   — → blocked
  review <id>        — → review
  cancel <id> [why]  — → cancelled
  update <id> [--title ...] [--priority ...] [--sprint ...] [--due ...]
  sub <parent> "subtask"
  assign <id> @user
  label <id> tag1,tag2
  dep <id> blocks <other_id>
  board [--sprint name]
  history <id>
  comment <id> text
  delete <id>
  sprint create|board|summary|assign <name>
  summary
  search <query>
  export"""


# ---------------------------------------------------------------------------
# Tool-registry-compatible public functions
# ---------------------------------------------------------------------------

def kanban_create(title: str = "", priority: str = "medium", description: str = "",
                  labels: str = "", assignee: str = "", sprint: str = "",
                  due_date: str = "", **_) -> Dict:
    if not title:
        return {"success": False, "error": "title is required"}
    db         = _default_db()
    label_list = [l.strip() for l in labels.split(",") if l.strip()] if labels else []
    return db.agent_create(title, priority=priority, description=description,
                           labels=label_list, assignee=assignee.lstrip("@"),
                           sprint=sprint, due_date=due_date or None)


def kanban_update(task_id: str = "", status: str = "", comment: str = "",
                  priority: str = "", assignee: str = "", **_) -> Dict:
    if not task_id:
        return {"success": False, "error": "task_id is required"}
    db = _default_db()
    return db.agent_update(task_id, status=status or None, comment=comment,
                           priority=priority or None,
                           assignee=assignee.lstrip("@") if assignee else None)


def kanban_list(status: str = "", priority: str = "", label: str = "",
                assignee: str = "", search: str = "", sprint: str = "",
                limit: int = 50, **_) -> Dict:
    db = _default_db()
    return db.agent_list(status=status or None, priority=priority or None,
                         label=label or None, assignee=assignee or None,
                         search=search or None, sprint=sprint or None, limit=limit)


def kanban_board(sprint: str = "", **_) -> Dict:
    return _default_db().agent_board(sprint=sprint or None)


def kanban_summary(**_) -> Dict:
    try:
        return {"success": True, **_default_db().summary()}
    except Exception as e:
        return {"success": False, "error": str(e)}
