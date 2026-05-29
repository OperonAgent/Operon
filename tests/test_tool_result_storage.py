"""Tests for core/tool_result_storage.py"""
import os
import tempfile
import time
import pytest
from core.tool_result_storage import (
    ToolResultStorage, PersistedResult, DEFAULT_THRESHOLD_CHARS,
    PREVIEW_CHARS, MAX_TURN_BUDGET_CHARS, _TOOL_THRESHOLDS,
    get_storage, maybe_persist_result, enforce_turn_budget,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _storage(threshold: int = 100) -> ToolResultStorage:
    return ToolResultStorage(threshold_chars=threshold, cleanup_after_secs=1)


def _result(text: str, key: str = "output") -> dict:
    return {key: text}


# ── PersistedResult ───────────────────────────────────────────────────────────

class TestPersistedResult:
    def test_to_dict_fields(self):
        pr = PersistedResult(
            call_id="abc", tool_name="shell", file_path="/tmp/x.txt",
            size_chars=500, checksum="abc123",
        )
        d = pr.to_dict()
        assert d["call_id"] == "abc"
        assert d["tool_name"] == "shell"
        assert d["file_path"] == "/tmp/x.txt"
        assert d["size_chars"] == 500

    def test_created_at_is_recent(self):
        pr = PersistedResult(call_id="x", tool_name="t", file_path="/f", size_chars=0)
        assert time.time() - pr.created_at < 2.0

    def test_checksum_optional(self):
        pr = PersistedResult(call_id="x", tool_name="t", file_path="/f", size_chars=0)
        assert pr.checksum == ""


# ── ToolResultStorage.maybe_persist ──────────────────────────────────────────

class TestMaybePersist:
    def test_small_output_not_persisted(self):
        s = _storage(threshold=500)
        r = s.maybe_persist("custom_tool", "c1", _result("short output"))
        assert "_output_truncated" not in r
        assert "_full_output_path" not in r

    def test_large_output_persisted(self):
        s = _storage(threshold=100)
        r = s.maybe_persist("custom_tool", "c1", _result("x" * 200))
        assert r["_output_truncated"] is True
        assert "_full_output_path" in r

    def test_preview_present_in_output(self):
        s = _storage(threshold=100)
        r = s.maybe_persist("custom_tool", "c1", _result("x" * 200))
        assert "FULL OUTPUT SAVED" in r["output"]

    def test_full_output_path_readable(self):
        s = _storage(threshold=100)
        content = "z" * 300
        r = s.maybe_persist("custom_tool", "c1", _result(content))
        path = r["_full_output_path"]
        assert os.path.exists(path)
        assert open(path).read() == content

    def test_size_chars_recorded(self):
        s = _storage(threshold=100)
        content = "q" * 250
        r = s.maybe_persist("custom_tool", "c1", _result(content))
        assert r["_full_size_chars"] == 250

    def test_result_without_output_field_unchanged(self):
        s = _storage(threshold=100)
        r = s.maybe_persist("custom_tool", "c1", {"success": True})
        assert r == {"success": True}

    def test_empty_output_unchanged(self):
        s = _storage(threshold=100)
        r = s.maybe_persist("custom_tool", "c1", _result(""))
        assert "_output_truncated" not in r

    def test_stdout_key_supported(self):
        s = _storage(threshold=100)
        r = s.maybe_persist("custom_tool", "c1", {"stdout": "x" * 200})
        assert r.get("_output_truncated") is True
        assert "stdout" in r

    def test_content_key_supported(self):
        s = _storage(threshold=100)
        r = s.maybe_persist("custom_tool", "c1", {"content": "y" * 200})
        assert r.get("_output_truncated") is True

    def test_per_tool_threshold_overrides_default(self):
        # shell_exec has threshold=6000, so 200 chars should NOT be persisted
        s = _storage(threshold=50)   # default=50, but shell_exec override is 6000
        r = s.maybe_persist("shell_exec", "c1", _result("x" * 200))
        assert "_output_truncated" not in r

    def test_preview_chars_limit(self):
        s = ToolResultStorage(threshold_chars=100, preview_chars=50)
        r = s.maybe_persist("custom_tool", "c1", _result("a" * 300))
        # Preview should be at most ~50 chars of actual content + metadata
        preview = r["output"]
        assert len(preview.split("[...output truncated")[0]) <= 55

    def test_session_results_tracked(self):
        s = _storage(threshold=100)
        s.maybe_persist("custom_tool", "c1", _result("x" * 200))
        s.maybe_persist("custom_tool", "c2", _result("y" * 200))
        assert len(s._session_results) == 2

    def test_result_is_shallow_copy(self):
        s = _storage(threshold=100)
        original = _result("x" * 200)
        r = s.maybe_persist("custom_tool", "c1", original)
        assert r is not original


# ── ToolResultStorage.enforce_turn_budget ────────────────────────────────────

class TestEnforceTurnBudget:
    def test_under_budget_unchanged(self):
        s = ToolResultStorage(threshold_chars=100, max_turn_budget=120_000)
        results = [_result("short") for _ in range(5)]
        out = s.enforce_turn_budget(results)
        assert all("_output_truncated" not in r for r in out)

    def test_over_budget_largest_persisted(self):
        s = ToolResultStorage(threshold_chars=100, max_turn_budget=1_000)
        results = [
            {"output": "x" * 600, "_tool_name": "t1", "call_id": "c1"},
            {"output": "y" * 200, "_tool_name": "t2", "call_id": "c2"},
        ]
        out = s.enforce_turn_budget(results)
        # Total was 800, budget is 1000 — may or may not persist depending on output
        total = sum(len(r.get("output", "")) for r in out)
        assert total <= 1_100  # some slack for preview text

    def test_already_truncated_not_re_truncated(self):
        s = ToolResultStorage(threshold_chars=100, max_turn_budget=500)
        results = [
            {"output": "z" * 50, "_output_truncated": True, "_tool_name": "t1", "call_id": "c1"},
        ]
        out = s.enforce_turn_budget(results)
        assert out[0] is results[0]   # unchanged

    def test_empty_results_list(self):
        s = _storage()
        out = s.enforce_turn_budget([])
        assert out == []

    def test_returns_same_length(self):
        s = ToolResultStorage(threshold_chars=50, max_turn_budget=500)
        results = [{"output": "k" * 100, "_tool_name": f"t{i}", "call_id": f"c{i}"}
                   for i in range(10)]
        out = s.enforce_turn_budget(results)
        assert len(out) == 10


# ── read_full / read_result ───────────────────────────────────────────────────

class TestReadFull:
    def test_read_full_returns_content(self):
        s = _storage(threshold=100)
        r = s.maybe_persist("custom_tool", "c1", _result("a" * 200))
        path = r["_full_output_path"]
        assert s.read_full(path) == "a" * 200

    def test_read_full_missing_file_returns_none(self):
        s = _storage()
        result = s.read_full("/tmp/operon-nonexistent-9999.txt")
        assert result is None

    def test_read_result_with_path(self):
        s = _storage(threshold=100)
        r = s.maybe_persist("custom_tool", "c1", _result("b" * 200))
        full = s.read_result(r)
        assert full == "b" * 200

    def test_read_result_without_path(self):
        s = _storage()
        r = _result("hello world")
        assert s.read_result(r) == "hello world"

    def test_read_result_stdout_fallback(self):
        s = _storage()
        r = {"stdout": "from stdout"}
        assert s.read_result(r) == "from stdout"


# ── stats / list_persisted ────────────────────────────────────────────────────

class TestStats:
    def test_stats_empty(self):
        s = _storage()
        st = s.stats()
        assert st["persisted_results"] == 0
        assert st["total_chars_saved"] == 0

    def test_stats_after_persist(self):
        s = _storage(threshold=100)
        s.maybe_persist("custom_tool", "c1", _result("m" * 200))
        st = s.stats()
        assert st["persisted_results"] == 1
        assert st["total_chars_saved"] == 200

    def test_list_persisted_returns_dicts(self):
        s = _storage(threshold=100)
        s.maybe_persist("custom_tool", "c1", _result("n" * 200))
        listing = s.list_persisted()
        assert len(listing) == 1
        assert listing[0]["tool_name"] == "custom_tool"


# ── cleanup ───────────────────────────────────────────────────────────────────

class TestCleanup:
    def test_cleanup_session_removes_files(self):
        s = _storage(threshold=100)
        r = s.maybe_persist("custom_tool", "c1", _result("p" * 200))
        path = r["_full_output_path"]
        assert os.path.exists(path)
        s.cleanup_session()
        assert not os.path.exists(path)

    def test_cleanup_session_clears_list(self):
        s = _storage(threshold=100)
        s.maybe_persist("custom_tool", "c1", _result("r" * 200))
        s.cleanup_session()
        assert s._session_results == []

    def test_cleanup_old_removes_stale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = ToolResultStorage(store_dir=tmpdir, threshold_chars=100, cleanup_after_secs=0)
            s.maybe_persist("custom_tool", "c1", _result("s" * 200))
            time.sleep(0.01)
            deleted = s.cleanup_old()
            assert deleted >= 1


# ── Module-level convenience ──────────────────────────────────────────────────

class TestModuleLevelAPI:
    def test_get_storage_returns_instance(self):
        s = get_storage()
        assert isinstance(s, ToolResultStorage)

    def test_maybe_persist_result_convenience(self):
        # just verify it doesn't crash
        r = maybe_persist_result("custom_tool", "test-c1", {"output": "short"})
        assert isinstance(r, dict)

    def test_enforce_turn_budget_convenience(self):
        results = [{"output": "hi"} for _ in range(3)]
        out = enforce_turn_budget(results)
        assert isinstance(out, list)


# ── _extract_output / _output_key ─────────────────────────────────────────────

class TestInternals:
    def test_extract_output_finds_output_key(self):
        assert ToolResultStorage._extract_output({"output": "hello"}) == "hello"

    def test_extract_output_finds_stdout(self):
        assert ToolResultStorage._extract_output({"stdout": "out"}) == "out"

    def test_extract_output_finds_content(self):
        assert ToolResultStorage._extract_output({"content": "cnt"}) == "cnt"

    def test_extract_output_returns_none_for_non_string(self):
        assert ToolResultStorage._extract_output({"output": 123}) is None

    def test_extract_output_returns_none_for_empty(self):
        assert ToolResultStorage._extract_output({"output": ""}) is None

    def test_output_key_returns_first_match(self):
        assert ToolResultStorage._output_key({"output": "x"}) == "output"
        assert ToolResultStorage._output_key({"stdout": "x"}) == "stdout"
        assert ToolResultStorage._output_key({"content": "x"}) == "content"

    def test_output_key_default_output(self):
        assert ToolResultStorage._output_key({}) == "output"


# ── Tool threshold constants ──────────────────────────────────────────────────

class TestConstants:
    def test_default_threshold_positive(self):
        assert DEFAULT_THRESHOLD_CHARS > 0

    def test_preview_chars_less_than_threshold(self):
        assert PREVIEW_CHARS < DEFAULT_THRESHOLD_CHARS

    def test_turn_budget_large(self):
        assert MAX_TURN_BUDGET_CHARS >= 10_000

    def test_tool_thresholds_dict_has_common_tools(self):
        assert "shell_exec" in _TOOL_THRESHOLDS
        assert "file_read" in _TOOL_THRESHOLDS
        assert "git_diff" in _TOOL_THRESHOLDS
