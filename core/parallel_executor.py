"""core/parallel_executor.py — Concurrent tool execution for Operon.

When the LLM emits action.type = "parallel_tools" with a list of calls,
this module fires them all simultaneously using a thread pool and
collects results into a single session message.

Format the LLM should emit:
    {
      "action": {
        "type": "parallel_tools",
        "calls": [
          {"tool_name": "duckduckgo_search", "params": {"query": "..."}, "id": "t1"},
          {"tool_name": "web_scrape",        "params": {"url": "..."},   "id": "t2"}
        ]
      }
    }

Results are injected as individual [TOOL_RESULT: ...] blocks so the model
sees each result labelled by the id it gave in the calls array.
"""

from __future__ import annotations

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool call to be executed."""
    tool_name: str
    params:    Dict[str, Any]
    call_id:   str = ""                    # optional label from the LLM


@dataclass
class ToolResult:
    """The outcome of one tool call."""
    call:       ToolCall
    result:     Any
    success:    bool
    error:      str = ""
    duration_s: float = 0.0


@dataclass
class ParallelResult:
    """Aggregate result from running multiple tools in parallel."""
    results:       List[ToolResult]
    total_time_s:  float = 0.0
    all_succeeded: bool  = True

    @property
    def combined_result_str(self) -> str:
        """Format all results as labelled TOOL_RESULT blocks for the session."""
        parts = []
        for tr in self.results:
            label  = tr.call.call_id or tr.call.tool_name
            status = "✓" if tr.success else "✗"
            body   = json.dumps(tr.result, indent=2, default=str) if tr.result is not None else tr.error
            # Truncate very large individual results
            if len(body) > 4000:
                body = body[:4000] + f"\n[…truncated {len(body)-4000} chars]"
            parts.append(f"[TOOL_RESULT: {label}] {status} ({tr.duration_s:.2f}s)\n{body}")
        return "\n\n".join(parts)

    @property
    def summary_line(self) -> str:
        ok  = sum(1 for r in self.results if r.success)
        bad = len(self.results) - ok
        return (f"Parallel: {len(self.results)} tools "
                f"({ok} ok{f', {bad} failed' if bad else ''}, "
                f"{self.total_time_s:.2f}s total)")


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class ParallelToolExecutor:
    """Thread-pool based parallel tool executor.

    Parameters
    ----------
    max_workers:
        Maximum concurrent threads. Default 4 — safe for most APIs and
        prevents accidental DoS on rate-limited endpoints.
    per_call_timeout:
        Seconds before a single tool call is considered timed-out.
        The future is cancelled (best-effort) and an error result is returned.
    """

    def __init__(
        self,
        max_workers:       int   = 4,
        per_call_timeout:  float = 60.0,
    ) -> None:
        self.max_workers      = max_workers
        self.per_call_timeout = per_call_timeout

    def run(
        self,
        calls:         List[ToolCall],
        execute_fn:    Callable[[str, Dict], Any],
        print_fn:      Callable[[str], None] = print,
    ) -> ParallelResult:
        """Execute *calls* concurrently using *execute_fn(tool_name, params)*.

        Parameters
        ----------
        calls:
            The list of ToolCall objects.
        execute_fn:
            Callable that runs a single tool.  Should match the signature
            ``tool_registry.execute(tool_name, params)`` and return a result dict.
        print_fn:
            Function used to print progress lines (theme-aware in main.py).
        """
        if not calls:
            return ParallelResult(results=[], total_time_s=0.0, all_succeeded=True)

        results:    List[ToolResult] = []
        wall_start: float            = time.perf_counter()

        # Map future → ToolCall so we can annotate results
        future_to_call: Dict[Future, ToolCall] = {}

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(calls))) as pool:
            for call in calls:
                fut = pool.submit(self._run_one, call, execute_fn)
                future_to_call[fut] = call

            for fut in as_completed(future_to_call, timeout=self.per_call_timeout * len(calls)):
                call = future_to_call[fut]
                try:
                    tr = fut.result(timeout=self.per_call_timeout)
                except Exception as exc:
                    tr = ToolResult(
                        call       = call,
                        result     = None,
                        success    = False,
                        error      = f"Executor error: {exc}",
                        duration_s = 0.0,
                    )
                results.append(tr)
                status = "✓" if tr.success else "✗"
                label  = call.call_id or call.tool_name
                print_fn(f"  [∥] {status} {label} ({tr.duration_s:.2f}s)")

        # Sort back into original call order for deterministic session messages
        call_order = {id(c): i for i, c in enumerate(calls)}
        results.sort(key=lambda r: call_order.get(id(r.call), 999))

        total_time = time.perf_counter() - wall_start
        all_ok     = all(r.success for r in results)

        return ParallelResult(
            results       = results,
            total_time_s  = total_time,
            all_succeeded = all_ok,
        )

    @staticmethod
    def _run_one(call: ToolCall, execute_fn: Callable) -> ToolResult:
        """Execute one tool call and return its ToolResult."""
        t0 = time.perf_counter()
        try:
            result  = execute_fn(call.tool_name, call.params)
            success = True
            if isinstance(result, dict):
                success = result.get("success", True)
            error = (result.get("error", "") if isinstance(result, dict) else "")
        except Exception as exc:
            result  = None
            success = False
            error   = str(exc)
        duration = time.perf_counter() - t0
        return ToolResult(
            call       = call,
            result     = result,
            success    = success,
            error      = error,
            duration_s = duration,
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_parallel_calls(action: Dict[str, Any]) -> List[ToolCall]:
    """Extract a list of ToolCall objects from a parallel_tools action dict.

    Accepts two shapes:
    1. ``{"type": "parallel_tools", "calls": [{tool_name, params, id?}, ...]}``
    2. ``{"type": "tool", "parallel": true, "calls": [...]}``
    """
    raw_calls = action.get("calls", [])
    if not isinstance(raw_calls, list):
        return []

    tool_calls = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        name = item.get("tool_name") or item.get("name") or item.get("tool") or ""
        if not name:
            continue
        params  = item.get("params") or item.get("arguments") or item.get("input") or {}
        call_id = item.get("id") or item.get("call_id") or name
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                params = {}
        tool_calls.append(ToolCall(tool_name=name, params=params, call_id=call_id))
    return tool_calls


# ---------------------------------------------------------------------------
# System prompt snippet
# ---------------------------------------------------------------------------

PARALLEL_TOOLS_PROMPT_SNIPPET = """\
## Parallel Tool Execution

When a task requires multiple *independent* pieces of information simultaneously
(e.g. searching two topics at once, scraping two URLs, running two lookups),
you can execute them in PARALLEL by using:

    {
      "action": {
        "type": "parallel_tools",
        "calls": [
          {"tool_name": "duckduckgo_search", "params": {"query": "topic A"}, "id": "search_a"},
          {"tool_name": "duckduckgo_search", "params": {"query": "topic B"}, "id": "search_b"}
        ]
      }
    }

Rules:
- Only use parallel_tools when calls are truly independent (no call depends on another's output).
- Max 4 calls per parallel batch.
- Each call must have a unique "id" so results can be matched.
- If any call fails, the others still succeed — you receive all results.
"""
