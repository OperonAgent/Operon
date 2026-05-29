"""
Operon Multi-Agent Orchestrator.

Enables spawning isolated child agents with scoped toolsets and independent
sessions. Inspired by OpenClaw's Lobster engine.

Architecture
------------
  Orchestrator           — coordinates named worker agents
  AgentSpec              — defines a worker's name, toolset, system_prompt override
  WorkerResult           — result of a completed worker run

Usage
-----
  from core.orchestrator import Orchestrator, AgentSpec

  orch = Orchestrator(base_runner=run_agent_loop_fn)

  result = orch.run_worker(
      AgentSpec(
          name="researcher",
          toolset=["duckduckgo_search", "web_scrape", "file_write"],
          prompt="Research the latest news on quantum computing and write a summary to /tmp/qc.md",
      )
  )

  # Or run multiple workers in parallel
  results = orch.run_parallel([spec1, spec2, spec3])
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentSpec:
    """Specification for a single worker agent run."""
    name:          str
    prompt:        str
    toolset:       List[str] = field(default_factory=list)   # empty = all tools
    system_note:   str = ""    # extra instruction prepended to system prompt
    max_iters:     int = 8
    timeout_s:     int = 120
    depth:         int = 0     # current spawn depth (auto-set by Orchestrator)


@dataclass
class WorkerResult:
    """Output from a completed worker agent."""
    name:         str
    prompt:       str
    response:     str
    success:      bool
    error:        str = ""
    elapsed_s:    float = 0.0
    tool_calls:   int = 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sibling file lock (prevent parallel workers clobbering the same file)
# ---------------------------------------------------------------------------

_LOCK_DIR = Path.home() / ".operon" / "file_locks"


@contextmanager
def file_lock(path: str, timeout_s: float = 30.0) -> Generator[None, None, None]:
    """
    Advisory file lock using fcntl (POSIX) or a lockfile (Windows fallback).
    Use as a context manager to serialize access to a file across sibling workers.

    Example
    -------
        with file_lock("/tmp/shared_output.txt"):
            Path("/tmp/shared_output.txt").write_text(content)
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = path.replace("/", "_").replace("\\", "_").lstrip("_")
    lock_path = _LOCK_DIR / f"{safe_name}.lock"

    # POSIX fcntl approach
    try:
        lock_file = open(lock_path, "w")
        deadline  = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass
    except (AttributeError, OSError):
        # Windows fallback: use a threading lock keyed by path
        if not hasattr(file_lock, "_locks"):
            file_lock._locks = {}
        lock = file_lock._locks.setdefault(path, threading.Lock())
        with lock:
            yield


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Coordinates spawning isolated child agent runs.

    Parameters
    ----------
    base_runner : callable(prompt: str, toolset: list, system_note: str) → str
        Function that runs a single agent turn and returns the text response.
        Typically a closure over run_agent_loop that creates a fresh SessionManager.
    max_spawn_depth : int
        Maximum nesting depth for sub-agent spawning (default 2).
        Prevents infinite recursion when agents spawn their own sub-agents.
    """

    def __init__(self, base_runner: Callable, max_spawn_depth: int = 2) -> None:
        self._runner          = base_runner
        self._history:        List[WorkerResult] = []
        self._max_depth:      int = max_spawn_depth
        self._depth_lock      = threading.Lock()

    # ------------------------------------------------------------------
    # Single worker
    # ------------------------------------------------------------------

    def run_worker(self, spec: AgentSpec) -> WorkerResult:
        """
        Run a single isolated worker agent synchronously.
        Returns a WorkerResult when the agent finishes or times out.
        """
        # Depth guard — prevent infinite sub-agent recursion
        if spec.depth >= self._max_depth:
            result = WorkerResult(
                name=spec.name, prompt=spec.prompt,
                response="", success=False,
                error=(
                    f"Max spawn depth ({self._max_depth}) reached. "
                    "Cannot spawn further sub-agents."
                ),
            )
            self._history.append(result)
            return result

        t0     = time.monotonic()
        result = WorkerResult(
            name=spec.name, prompt=spec.prompt,
            response="", success=False, elapsed_s=0.0,
        )

        # Run in a thread so we can apply the timeout
        container: Dict[str, Any] = {}

        def _run():
            try:
                container["response"] = self._runner(
                    prompt      = spec.prompt,
                    toolset     = spec.toolset,
                    system_note = spec.system_note,
                    max_iters   = spec.max_iters,
                )
            except Exception as e:
                container["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=spec.timeout_s)

        result.elapsed_s = round(time.monotonic() - t0, 2)

        if t.is_alive():
            result.error   = f"Worker '{spec.name}' timed out after {spec.timeout_s}s"
            result.success = False
        elif "error" in container:
            result.error   = container["error"]
            result.success = False
        else:
            result.response = container.get("response", "")
            result.success  = True

        self._history.append(result)
        return result

    # ------------------------------------------------------------------
    # Parallel workers
    # ------------------------------------------------------------------

    def run_parallel(self, specs: List[AgentSpec]) -> List[WorkerResult]:
        """
        Run multiple workers in parallel threads.
        Returns a list of WorkerResults in the same order as specs.
        """
        results: List[Optional[WorkerResult]] = [None] * len(specs)
        threads = []

        def _worker(idx: int, spec: AgentSpec):
            results[idx] = self.run_worker(spec)

        for i, spec in enumerate(specs):
            t = threading.Thread(target=_worker, args=(i, spec), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Pipeline (sequential, output feeds next input)
    # ------------------------------------------------------------------

    def run_pipeline(self, steps: List[AgentSpec]) -> List[WorkerResult]:
        """
        Run agents in a sequential pipeline where each step's output is
        appended to the next step's prompt as context.

        Returns all WorkerResults in order.
        """
        results: List[WorkerResult] = []
        context = ""

        for spec in steps:
            if context:
                enriched_prompt = (
                    f"[PREVIOUS STEP OUTPUT]\n{context}\n[END PREVIOUS STEP]\n\n"
                    f"{spec.prompt}"
                )
                enriched_spec = AgentSpec(
                    name        = spec.name,
                    prompt      = enriched_prompt,
                    toolset     = spec.toolset,
                    system_note = spec.system_note,
                    max_iters   = spec.max_iters,
                    timeout_s   = spec.timeout_s,
                )
            else:
                enriched_spec = spec

            result  = self.run_worker(enriched_spec)
            results.append(result)

            if result.success:
                context = result.response
            else:
                # Stop pipeline on failure
                break

        return results

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(self) -> List[WorkerResult]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def summary(self) -> Dict[str, Any]:
        total     = len(self._history)
        succeeded = sum(1 for r in self._history if r.success)
        return {
            "total_runs":   total,
            "succeeded":    succeeded,
            "failed":       total - succeeded,
            "avg_elapsed":  round(
                sum(r.elapsed_s for r in self._history) / total, 2
            ) if total else 0.0,
        }
