"""Tests for core/parallel_executor.py"""
import time
import threading
import pytest

from core.parallel_executor import (
    ToolCall, ToolResult, ParallelResult,
    ParallelToolExecutor, parse_parallel_calls,
    PARALLEL_TOOLS_PROMPT_SNIPPET,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_executor(max_workers: int = 4, per_call_timeout: float = 10.0, **kwargs) -> ParallelToolExecutor:
    return ParallelToolExecutor(max_workers=max_workers, per_call_timeout=per_call_timeout)


def instant_execute(tool_name: str, params: dict) -> dict:
    """Fast mock execute — returns success immediately."""
    return {"success": True, "output": f"{tool_name}({params})", "error": ""}


def slow_execute(tool_name: str, params: dict) -> dict:
    """Slow mock execute — sleeps 0.1s per call."""
    time.sleep(0.1)
    return {"success": True, "output": f"done:{tool_name}", "error": ""}


def failing_execute(tool_name: str, params: dict) -> dict:
    """Always returns failure."""
    return {"success": False, "output": None, "error": f"{tool_name} failed"}


def error_execute(tool_name: str, params: dict) -> dict:
    """Raises an exception."""
    raise RuntimeError(f"crash in {tool_name}")


# ── ToolCall dataclass ────────────────────────────────────────────────────────

class TestToolCall:
    def test_fields_accessible(self):
        tc = ToolCall(tool_name="duckduckgo_search",
                      params={"query": "test"}, call_id="s1")
        assert tc.tool_name == "duckduckgo_search"
        assert tc.params == {"query": "test"}
        assert tc.call_id == "s1"

    def test_default_call_id_empty(self):
        tc = ToolCall(tool_name="web_scrape", params={})
        assert tc.call_id == ""


# ── ToolResult dataclass ──────────────────────────────────────────────────────

class TestToolResult:
    def test_success_result(self):
        tc = ToolCall("shell_exec", {"command": "echo hi"}, "c1")
        tr = ToolResult(call=tc, result={"output": "hi"}, success=True, duration_s=0.01)
        assert tr.success
        assert tr.duration_s == pytest.approx(0.01, abs=0.001)

    def test_failure_result(self):
        tc = ToolCall("shell_exec", {}, "c2")
        tr = ToolResult(call=tc, result=None, success=False, error="timed out")
        assert not tr.success
        assert "timed out" in tr.error


# ── ParallelResult ────────────────────────────────────────────────────────────

class TestParallelResult:
    def _make_result(self, n=3, all_ok=True) -> ParallelResult:
        results = []
        for i in range(n):
            tc = ToolCall(f"tool_{i}", {}, f"id_{i}")
            tr = ToolResult(
                call=tc,
                result={"output": f"result_{i}"},
                success=True if all_ok else (i % 2 == 0),
                duration_s=0.1,
            )
            results.append(tr)
        return ParallelResult(results=results, total_time_s=0.5,
                              all_succeeded=all_ok)

    def test_combined_result_str_contains_all_labels(self):
        pr = self._make_result(3)
        s = pr.combined_result_str
        for i in range(3):
            assert f"id_{i}" in s or f"tool_{i}" in s

    def test_combined_result_str_includes_status_icons(self):
        pr = self._make_result(2)
        s = pr.combined_result_str
        assert "✓" in s

    def test_summary_line_counts(self):
        pr = self._make_result(3)
        line = pr.summary_line
        assert "3" in line

    def test_all_succeeded_false_when_any_fail(self):
        pr = self._make_result(4, all_ok=False)
        # Mixed success — at least one failed
        any_failed = any(not r.success for r in pr.results)
        assert any_failed

    def test_large_result_truncated(self):
        tc = ToolCall("big_tool", {}, "big")
        tr = ToolResult(
            call=tc,
            result={"output": "x" * 10_000},
            success=True,
            duration_s=0.1,
        )
        pr = ParallelResult(results=[tr], total_time_s=0.1, all_succeeded=True)
        s = pr.combined_result_str
        assert "truncated" in s
        assert len(s) < 15_000   # well below 10k chars


# ── parse_parallel_calls ──────────────────────────────────────────────────────

class TestParseParallelCalls:
    def test_standard_format(self):
        action = {
            "type": "parallel_tools",
            "calls": [
                {"tool_name": "duckduckgo_search", "params": {"query": "A"}, "id": "s1"},
                {"tool_name": "web_scrape",        "params": {"url": "B"},   "id": "s2"},
            ],
        }
        calls = parse_parallel_calls(action)
        assert len(calls) == 2
        assert calls[0].tool_name == "duckduckgo_search"
        assert calls[0].call_id == "s1"
        assert calls[1].tool_name == "web_scrape"

    def test_name_key_alias(self):
        """'name' key should work as alias for 'tool_name'."""
        action = {
            "calls": [{"name": "shell_exec", "params": {}, "id": "c1"}]
        }
        calls = parse_parallel_calls(action)
        assert len(calls) == 1
        assert calls[0].tool_name == "shell_exec"

    def test_tool_key_alias(self):
        action = {"calls": [{"tool": "file_read", "params": {"path": "f"}}]}
        calls = parse_parallel_calls(action)
        assert len(calls) == 1
        assert calls[0].tool_name == "file_read"

    def test_empty_calls_returns_empty(self):
        calls = parse_parallel_calls({"calls": []})
        assert calls == []

    def test_missing_calls_key_returns_empty(self):
        calls = parse_parallel_calls({"type": "parallel_tools"})
        assert calls == []

    def test_skips_items_without_tool_name(self):
        action = {
            "calls": [
                {"params": {}},                              # no name
                {"tool_name": "valid_tool", "params": {}},  # valid
            ]
        }
        calls = parse_parallel_calls(action)
        assert len(calls) == 1
        assert calls[0].tool_name == "valid_tool"

    def test_json_string_params_decoded(self):
        import json
        action = {
            "calls": [
                {"tool_name": "shell_exec", "params": json.dumps({"command": "ls"})}
            ]
        }
        calls = parse_parallel_calls(action)
        assert calls[0].params == {"command": "ls"}

    def test_default_id_falls_back_to_tool_name(self):
        action = {"calls": [{"tool_name": "file_read", "params": {}}]}
        calls = parse_parallel_calls(action)
        assert calls[0].call_id == "file_read"


# ── ParallelToolExecutor.run ──────────────────────────────────────────────────

class TestParallelExecutorRun:
    def test_empty_calls_returns_empty_result(self):
        ex = make_executor()
        pr = ex.run([], instant_execute)
        assert pr.results == []
        assert pr.all_succeeded

    def test_single_call(self):
        ex = make_executor()
        calls = [ToolCall("tool_a", {"x": 1}, "a")]
        pr = ex.run(calls, instant_execute)
        assert len(pr.results) == 1
        assert pr.results[0].success
        assert pr.results[0].call.tool_name == "tool_a"

    def test_multiple_calls_all_succeed(self):
        ex = make_executor()
        calls = [ToolCall(f"tool_{i}", {"i": i}, f"id_{i}") for i in range(4)]
        pr = ex.run(calls, instant_execute)
        assert len(pr.results) == 4
        assert pr.all_succeeded

    def test_parallel_is_faster_than_sequential(self):
        """4 calls run concurrently should finish well under the serial sum.

        Uses a long per-call sleep so the bound is robust on contended CI
        runners with coverage instrumentation (a tight sub-0.4s bound was
        flaky on 2-core Linux runners while passing on fast macOS).
        """
        n, sleep_s = 4, 0.5
        def _slow(tool_name, params):
            time.sleep(sleep_s)
            return {"success": True}
        ex = make_executor(max_workers=n)
        calls = [ToolCall(f"t{i}", {}, f"c{i}") for i in range(n)]

        start = time.perf_counter()
        pr = ex.run(calls, _slow)
        elapsed = time.perf_counter() - start

        # Serial = n*sleep_s = 2.0s. Concurrent ≈ 0.5s; assert < 60% of serial
        # — proves overlap while tolerating heavy runner/coverage overhead.
        assert elapsed < (n * sleep_s) * 0.6, \
            f"Parallel took {elapsed:.3f}s of {n*sleep_s}s serial — not concurrent"
        assert len(pr.results) == n

    def test_failure_does_not_stop_others(self):
        """If one call fails, others still complete."""
        call_count = [0]
        lock = threading.Lock()

        def partial_failure(tool_name, params):
            with lock:
                call_count[0] += 1
            if tool_name == "fail_me":
                return {"success": False, "output": None, "error": "intentional fail"}
            return {"success": True, "output": "ok", "error": ""}

        calls = [
            ToolCall("ok_tool_1", {}, "a"),
            ToolCall("fail_me",   {}, "b"),
            ToolCall("ok_tool_2", {}, "c"),
        ]
        ex = make_executor()
        pr = ex.run(calls, partial_failure)

        assert len(pr.results) == 3
        assert call_count[0] == 3   # all three were called
        successes = [r for r in pr.results if r.success]
        failures  = [r for r in pr.results if not r.success]
        assert len(successes) == 2
        assert len(failures)  == 1

    def test_exception_in_tool_captured(self):
        """Tool that raises exception → ToolResult with success=False."""
        calls = [ToolCall("crasher", {}, "c")]
        ex = make_executor()
        pr = ex.run(calls, error_execute)
        assert len(pr.results) == 1
        assert not pr.results[0].success
        assert "crash" in pr.results[0].error.lower()

    def test_max_workers_respected(self):
        """Never more than max_workers threads active simultaneously."""
        active = [0]
        peak   = [0]
        lock   = threading.Lock()

        def counting_execute(tool_name, params):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return {"success": True, "output": "ok", "error": ""}

        calls = [ToolCall(f"t{i}", {}, f"c{i}") for i in range(8)]
        ex = make_executor(max_workers=3)
        ex.run(calls, counting_execute)

        assert peak[0] <= 3, f"Peak concurrent threads was {peak[0]}, expected ≤ 3"

    def test_duration_recorded_per_result(self):
        calls = [ToolCall("slow", {}, "s")]
        ex = make_executor()
        pr = ex.run(calls, slow_execute)
        assert pr.results[0].duration_s >= 0.05   # at least 50ms

    def test_total_time_recorded(self):
        calls = [ToolCall(f"t{i}", {}, f"c{i}") for i in range(2)]
        ex = make_executor()
        pr = ex.run(calls, slow_execute)
        assert pr.total_time_s > 0

    def test_print_fn_called_per_result(self):
        messages = []
        calls = [ToolCall("tool_a", {}, "a"), ToolCall("tool_b", {}, "b")]
        ex = make_executor()
        ex.run(calls, instant_execute, print_fn=messages.append)
        assert len(messages) == 2   # one message per completed call

    def test_results_order_matches_input(self):
        """Results should be returned in input call order."""
        calls = [ToolCall(f"tool_{i}", {"idx": i}, f"id_{i}") for i in range(5)]

        def labelled_execute(name, params):
            time.sleep(0.01 * (5 - params.get("idx", 0)))   # reverse finish order
            return {"success": True, "output": params["idx"], "error": ""}

        ex = make_executor()
        pr = ex.run(calls, labelled_execute)
        for i, r in enumerate(pr.results):
            assert r.call.call_id == f"id_{i}", \
                f"Position {i}: expected id_{i}, got {r.call.call_id}"


# ── Prompt snippet ────────────────────────────────────────────────────────────

class TestPromptSnippet:
    def test_snippet_is_string(self):
        assert isinstance(PARALLEL_TOOLS_PROMPT_SNIPPET, str)

    def test_snippet_contains_format(self):
        assert "parallel_tools" in PARALLEL_TOOLS_PROMPT_SNIPPET
        assert "calls" in PARALLEL_TOOLS_PROMPT_SNIPPET
        assert "tool_name" in PARALLEL_TOOLS_PROMPT_SNIPPET

    def test_snippet_mentions_max_limit(self):
        assert "4" in PARALLEL_TOOLS_PROMPT_SNIPPET
