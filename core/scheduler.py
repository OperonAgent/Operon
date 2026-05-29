"""
Operon Background Scheduler.

Persists scheduled tasks to ~/.operon/schedule.json.
A daemon thread wakes every 10 seconds, checks due tasks, and fires them
by calling the agent loop through an injected runner callable.
"""

import json
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

SCHEDULE_FILE = Path.home() / ".operon" / "schedule.json"


class ScheduledTask:
    def __init__(self, task_id: str, prompt: str, interval_seconds: int,
                 label: str = "", next_run: float = 0.0, enabled: bool = True,
                 run_count: int = 0, last_run: float = 0.0):
        self.task_id = task_id
        self.prompt = prompt
        self.interval_seconds = interval_seconds
        self.label = label or prompt[:40]
        self.next_run = next_run or time.time() + interval_seconds
        self.enabled = enabled
        self.run_count = run_count
        self.last_run = last_run

    def to_dict(self) -> dict:
        return {
            "task_id":          self.task_id,
            "prompt":           self.prompt,
            "interval_seconds": self.interval_seconds,
            "label":            self.label,
            "next_run":         self.next_run,
            "enabled":          self.enabled,
            "run_count":        self.run_count,
            "last_run":         self.last_run,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduledTask":
        return cls(**d)


class TaskScheduler:

    def __init__(self):
        SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._tasks: dict[str, ScheduledTask] = {}
        self._runner: Optional[Callable[[str], None]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._load()

    def set_runner(self, runner: Callable[[str], None]) -> None:
        """Inject the agent runner. Called with (prompt_string)."""
        self._runner = runner

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add(self, prompt: str, interval_seconds: int, label: str = "") -> str:
        task_id = str(uuid.uuid4())[:8]
        task = ScheduledTask(
            task_id=task_id,
            prompt=prompt,
            interval_seconds=interval_seconds,
            label=label or prompt[:40],
        )
        with self._lock:
            self._tasks[task_id] = task
            self._save()
        return task_id

    def remove(self, task_id: str) -> bool:
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                self._save()
                return True
        return False

    def toggle(self, task_id: str) -> Optional[bool]:
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].enabled = not self._tasks[task_id].enabled
                self._save()
                return self._tasks[task_id].enabled
        return None

    def list_tasks(self) -> list[dict]:
        with self._lock:
            return [t.to_dict() for t in self._tasks.values()]

    def clear(self) -> None:
        with self._lock:
            self._tasks.clear()
            self._save()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        data = [t.to_dict() for t in self._tasks.values()]
        SCHEDULE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not SCHEDULE_FILE.exists():
            return
        try:
            data = json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
            for d in data:
                t = ScheduledTask.from_dict(d)
                self._tasks[t.task_id] = t
        except Exception:
            pass

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            now = time.time()
            due = []
            with self._lock:
                for task in self._tasks.values():
                    if task.enabled and now >= task.next_run:
                        due.append(task)
                        task.last_run = now
                        task.next_run = now + task.interval_seconds
                        task.run_count += 1
                if due:
                    self._save()

            for task in due:
                if self._runner:
                    try:
                        self._runner(task.prompt)
                    except Exception as e:
                        print(
                            f"\n  [Scheduler] Task '{task.label}' raised: "
                            f"{type(e).__name__}: {e}",
                            file=sys.stderr,
                        )

            time.sleep(10)
