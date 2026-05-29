"""
Iteration Budget — thread-safe turn counter with refund() for read-only tools.

Ported from Hermes Agent iteration_budget.py.

Read-only (idempotent) tools like file_read, dir_list, web_search don't actually
make progress — they're just information-gathering.  Counting them against the
iteration budget causes agents to stall on complex tasks.  refund() lets callers
return those spent iterations so the budget only depletes on *acting* turns.
"""

from __future__ import annotations

import threading
from typing import FrozenSet


# Tools that don't change the world — iterations spent on these can be refunded.
REFUNDABLE_TOOLS: FrozenSet[str] = frozenset({
    "file_read", "file_exists", "file_info", "dir_list", "file_search",
    "duckduckgo_search", "web_scrape", "web_fetch",
    "browser_get_url", "browser_snapshot",
    "db_query", "knowledge_get", "knowledge_list",
    "git_status", "git_diff", "git_log",
    "x_search", "clarify",
})


class IterationBudget:
    """
    Per-turn iteration counter.  Replaces a plain `range(max_iters)` loop
    with a smarter counter that can refund iterations for non-acting tool calls.

    Usage::

        budget = IterationBudget(max_iters=12)
        while not budget.is_exhausted:
            # ... run one iteration ...
            if tool_name in REFUNDABLE_TOOLS and tool_succeeded:
                budget.refund()
    """

    def __init__(self, max_iters: int = 12) -> None:
        self._max  = max_iters
        self._used = 0
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def remaining(self) -> int:
        """Iterations left before the budget is exhausted."""
        with self._lock:
            return max(0, self._max - self._used)

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def is_exhausted(self) -> bool:
        return self.remaining == 0

    def consume(self, n: int = 1) -> bool:
        """
        Spend `n` iterations.  Returns True if budget still has iterations left.
        Call at the START of every loop iteration.
        """
        with self._lock:
            self._used += n
            return self._used <= self._max

    def refund(self, n: int = 1) -> None:
        """
        Return `n` previously consumed iterations.  Never goes below 0.
        Call after a successful read-only tool call.
        """
        with self._lock:
            self._used = max(0, self._used - n)

    def refund_if_read_only(self, tool_name: str, tool_succeeded: bool = True) -> bool:
        """
        Convenience: refund one iteration iff `tool_name` is in REFUNDABLE_TOOLS
        and the call succeeded.  Returns True if a refund was applied.
        """
        if tool_succeeded and tool_name in REFUNDABLE_TOOLS:
            self.refund()
            return True
        return False

    def reset(self) -> None:
        with self._lock:
            self._used = 0

    def __repr__(self) -> str:
        return f"IterationBudget(used={self._used}/{self._max})"
