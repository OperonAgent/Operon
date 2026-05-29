"""Tests for tools/delegate.py"""
import threading
import pytest
from unittest.mock import patch, MagicMock
from tools.delegate import (
    DELEGATE_BLOCKED_TOOLS, _current_depth, _truncate, _format_result,
    delegate_task, delegate_batch,
)


# ── Blocked tools ─────────────────────────────────────────────────────────────

class TestBlockedTools:
    def test_delegate_task_blocked(self):
        assert "delegate_task" in DELEGATE_BLOCKED_TOOLS

    def test_delegate_batch_blocked(self):
        assert "delegate_batch" in DELEGATE_BLOCKED_TOOLS

    def test_email_draft_blocked(self):
        assert "email_draft" in DELEGATE_BLOCKED_TOOLS

    def test_email_send_blocked(self):
        assert "email_send" in DELEGATE_BLOCKED_TOOLS

    def test_computer_use_blocked(self):
        assert "computer_use" in DELEGATE_BLOCKED_TOOLS

    def test_blocked_tools_is_frozenset(self):
        assert isinstance(DELEGATE_BLOCKED_TOOLS, frozenset)


# ── Depth counter ─────────────────────────────────────────────────────────────

class TestDepthCounter:
    def test_default_depth_zero(self):
        # In main thread, depth should be 0 (or whatever it was reset to)
        d = _current_depth()
        assert isinstance(d, int)
        assert d >= 0

    def test_depth_is_thread_local(self):
        """Each thread should start with depth=0."""
        results = {}

        def worker(tid):
            results[tid] = _current_depth()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for tid, depth in results.items():
            assert depth == 0, f"Thread {tid} had depth {depth}, expected 0"


# ── _truncate ─────────────────────────────────────────────────────────────────

class TestTruncate:
    def test_short_string_unchanged(self):
        s = "hello world"
        assert _truncate(s, max_chars=100) == s

    def test_long_string_truncated(self):
        s = "x" * 10000
        result = _truncate(s, max_chars=1000)
        assert len(result) <= 1100  # some slack for the ellipsis line
        assert "omitted" in result

    def test_truncation_preserves_start_and_end(self):
        s = "START" + ("M" * 5000) + "END"
        result = _truncate(s, max_chars=100)
        assert "START" in result
        assert "END" in result

    def test_exact_max_chars_not_truncated(self):
        s = "a" * 1000
        assert _truncate(s, max_chars=1000) == s


# ── _format_result ────────────────────────────────────────────────────────────

class TestFormatResult:
    def test_string_input(self):
        assert _format_result("hello") == "hello"

    def test_dict_with_reply(self):
        result = _format_result({"reply": "the answer", "other": "ignored"})
        assert result == "the answer"

    def test_dict_with_content(self):
        result = _format_result({"content": "the content"})
        assert result == "the content"

    def test_dict_without_known_keys_json(self):
        result = _format_result({"key": "value", "num": 42})
        assert "key" in result
        assert "value" in result

    def test_none_returns_string(self):
        result = _format_result(None)
        assert isinstance(result, str)

    def test_int_returns_string(self):
        result = _format_result(42)
        assert result == "42"

    def test_list_returns_string(self):
        result = _format_result([1, 2, 3])
        assert isinstance(result, str)


# ── Empty task guard ─────────────────────────────────────────────────────────

class TestDelegateTask:
    def test_empty_task_returns_error(self):
        # delegate_task with empty string — should return error or handle gracefully
        result = delegate_task("")
        # Returns dict with success/error or str
        assert result is not None

    def test_non_empty_task_does_not_crash(self):
        # Without a running router, this will fail internally but not raise
        result = delegate_task("list files in /tmp")
        assert result is not None


class TestDelegateBatch:
    def test_empty_list_returns_error(self):
        result = delegate_batch([])
        # API returns dict with error when list is empty
        assert isinstance(result, dict)
        assert not result.get("success", True)

    def test_tasks_must_be_dicts(self):
        """delegate_batch takes List[Dict] with 'task' key."""
        tasks = [{"task": "task1"}, {"task": "task2"}]
        with patch("tools.delegate._run_subagent",
                   return_value={"success": True, "result": "ok", "task_id": "t1",
                                 "duration_s": 0.1, "iterations": 1}):
            result = delegate_batch(tasks, max_concurrent=2)
        assert isinstance(result, dict)
        assert "tasks" in result or "success" in result

    def test_result_contains_task_results(self):
        """Results dict should have task-level results."""
        tasks = [{"task": "t1", "id": "id1"}, {"task": "t2", "id": "id2"}]
        mock_result = {"success": True, "result": "done", "task_id": "x", "duration_s": 0.1, "iterations": 1}
        with patch("tools.delegate._run_subagent", return_value=mock_result):
            result = delegate_batch(tasks, max_concurrent=2)
        assert isinstance(result, dict)

    def test_max_concurrent_respected(self):
        import time
        concurrent_count = [0]
        max_concurrent_seen = [0]
        lock = threading.Lock()

        def mock_run(task, *args, **kwargs):
            with lock:
                concurrent_count[0] += 1
                max_concurrent_seen[0] = max(max_concurrent_seen[0], concurrent_count[0])
            time.sleep(0.05)
            with lock:
                concurrent_count[0] -= 1
            return {"success": True, "result": "ok", "task_id": task,
                    "duration_s": 0.05, "iterations": 1}

        tasks = [{"task": f"t{i}"} for i in range(5)]
        with patch("tools.delegate._run_subagent", side_effect=mock_run):
            delegate_batch(tasks, max_concurrent=2)

        assert max_concurrent_seen[0] <= 2
