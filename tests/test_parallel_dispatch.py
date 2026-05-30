"""
tests/test_parallel_dispatch.py — prove tool dispatch runs concurrently.

Verifies that ParallelToolExecutor actually executes independent tool calls
in parallel (wall-time << sum of per-call sleeps), which is Operon's answer to
"one slow tool call blocks the whole turn".
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.parallel_executor import (
    ParallelToolExecutor, ToolCall, parse_parallel_calls,
)


def _slow_tool(name: str, params: dict) -> dict:
    time.sleep(params.get("sleep", 0.3))
    return {"success": True, "echo": params.get("x")}


class TestConcurrency:
    def test_runs_in_parallel_not_serial(self):
        # Use a long per-task sleep so thread/scheduler overhead is negligible
        # and the bound is robust on contended CI runners (the previous tight
        # 0.8s bound on 4x0.3s was flaky on loaded 2-core Linux runners).
        n, sleep_s = 4, 1.0
        calls = [ToolCall(tool_name="slow", params={"sleep": sleep_s, "x": i},
                          call_id=f"c{i}") for i in range(n)]
        ex = ParallelToolExecutor(max_workers=n)
        t0 = time.perf_counter()
        result = ex.run(calls, execute_fn=_slow_tool, print_fn=lambda *_: None)
        wall = time.perf_counter() - t0
        # Serial would be n*sleep_s = 4.0s. Concurrent should be ~1s; assert a
        # very generous 2.5s ceiling — proves overlap while tolerating overhead.
        assert wall < (n * sleep_s) * 0.6, f"not concurrent: {wall:.2f}s of {n*sleep_s}s serial"
        assert result.all_succeeded
        assert len(result.results) == 4

    def test_results_in_original_order(self):
        calls = [ToolCall(tool_name="slow",
                          params={"sleep": 0.05 * (4 - i), "x": i},
                          call_id=f"c{i}") for i in range(4)]
        ex = ParallelToolExecutor(max_workers=4)
        result = ex.run(calls, execute_fn=_slow_tool, print_fn=lambda *_: None)
        xs = [r.result["echo"] for r in result.results]
        assert xs == [0, 1, 2, 3]  # deterministic order despite varied durations

    def test_empty_calls(self):
        ex = ParallelToolExecutor(max_workers=4)
        result = ex.run([], execute_fn=_slow_tool, print_fn=lambda *_: None)
        assert result.all_succeeded and not result.results

    def test_one_failure_isolated(self):
        def flaky(name, params):
            if params.get("x") == 2:
                raise RuntimeError("boom")
            return {"success": True}
        calls = [ToolCall(tool_name="f", params={"x": i}, call_id=f"c{i}")
                 for i in range(3)]
        ex = ParallelToolExecutor(max_workers=3)
        result = ex.run(calls, execute_fn=flaky, print_fn=lambda *_: None)
        assert not result.all_succeeded
        assert sum(1 for r in result.results if r.success) == 2


class TestParsing:
    def test_parse_calls_shape(self):
        action = {"type": "parallel_tools", "calls": [
            {"tool_name": "a", "params": {"x": 1}, "id": "1"},
            {"tool_name": "b", "params": {"x": 2}, "id": "2"},
        ]}
        calls = parse_parallel_calls(action)
        assert len(calls) == 2
        assert calls[0].tool_name == "a"

    def test_parse_empty(self):
        assert parse_parallel_calls({"type": "parallel_tools", "calls": []}) == []
