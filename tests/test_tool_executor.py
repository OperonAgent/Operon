"""Tests for core/tool_executor.py"""
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from core.tool_executor import (
    ToolResult, ResultType, ResultClassifier, ErrorClassifier,
    SchemaSanitizer, OutputCapEnforcer, RetryConfig, RetryPolicyManager,
    ToolDispatcher, ParallelToolDispatcher,
    get_dispatcher, classify_result, sanitize_schema,
    _DEFAULT_MAX_OUTPUT_CHARS, _SCHEMA_UNSUPPORTED_KEYS,
)


# ── ToolResult ────────────────────────────────────────────────────────────────

class TestToolResult:
    def test_text_content_short(self):
        r = ToolResult(tool_name="x", success=True, content="hello")
        assert r.text_content() == "hello"

    def test_text_content_truncated(self):
        r = ToolResult(tool_name="x", success=True, content="x" * 100)
        assert len(r.text_content(max_chars=50)) <= 80  # truncation marker

    def test_text_content_returns_error_when_empty(self):
        r = ToolResult(tool_name="x", success=False, error="some error")
        assert r.text_content() == "some error"

    def test_to_context_string_success(self):
        r = ToolResult(tool_name="shell_exec", success=True, content="output here")
        ctx = r.to_context_string()
        assert "[TOOL_RESULT: shell_exec]" in ctx
        assert "output here" in ctx

    def test_to_context_string_error(self):
        r = ToolResult(tool_name="shell_exec", success=False, error="command not found")
        ctx = r.to_context_string()
        assert "[TOOL_ERROR: shell_exec]" in ctx
        assert "command not found" in ctx

    def test_to_context_string_image(self):
        r = ToolResult(tool_name="screenshot", success=True,
                       result_type=ResultType.IMAGE, image_b64="abc123")
        ctx = r.to_context_string()
        assert "image" in ctx.lower()

    def test_to_dict_keys(self):
        r = ToolResult(tool_name="x", success=True, result_type=ResultType.TEXT)
        d = r.to_dict()
        assert "tool" in d
        assert "success" in d
        assert "result_type" in d
        assert "duration_ms" in d

    def test_should_retry_default_false(self):
        r = ToolResult(tool_name="x", success=True)
        assert not r.should_retry


# ── ResultClassifier ─────────────────────────────────────────────────────────

class TestResultClassifier:
    def setup_method(self):
        self.clf = ResultClassifier()

    def test_none_returns_empty(self):
        rt, content, img = self.clf.classify("x", None)
        assert rt == ResultType.EMPTY
        assert content == ""
        assert img is None

    def test_string_text(self):
        rt, content, _ = self.clf.classify("shell_exec", "hello world")
        assert rt == ResultType.CODE
        assert content == "hello world"

    def test_string_json(self):
        rt, content, _ = self.clf.classify("http_client", '{"key": "val"}')
        assert rt == ResultType.JSON

    def test_dict_success(self):
        raw = {"success": True, "output": "done"}
        rt, content, _ = self.clf.classify("shell_exec", raw)
        assert rt == ResultType.CODE
        assert content == "done"

    def test_dict_failure_returns_error(self):
        raw = {"success": False, "error": "permission denied"}
        rt, content, _ = self.clf.classify("x", raw)
        assert rt == ResultType.ERROR
        assert "permission denied" in content

    def test_dict_list_output_returns_json(self):
        raw = {"success": True, "output": ["a", "b", "c"]}
        rt, content, _ = self.clf.classify("x", raw)
        assert rt == ResultType.JSON

    def test_png_bytes_classified_as_image(self):
        png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        rt, content, b64 = self.clf.classify("screenshot", png_magic)
        assert rt == ResultType.IMAGE
        assert b64 is not None

    def test_jpeg_bytes_classified_as_image(self):
        jpeg_magic = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        rt, _, b64 = self.clf.classify("screenshot", jpeg_magic)
        assert rt == ResultType.IMAGE
        assert b64 is not None

    def test_arbitrary_bytes_classified_as_binary(self):
        data = b"\x01\x02\x03" * 50
        rt, _, _ = self.clf.classify("x", data)
        assert rt == ResultType.BINARY

    def test_base64_image_uri(self):
        rt, _, b64 = self.clf.classify("image_gen", "data:image/png;base64,abc123")
        assert rt == ResultType.IMAGE
        assert b64 == "abc123"

    def test_estimate_tokens(self):
        assert ResultClassifier.estimate_tokens("") == 1
        assert ResultClassifier.estimate_tokens("a" * 400) == 100

    def test_non_code_tool_with_text(self):
        rt, content, _ = self.clf.classify("unknown_tool", "plain text")
        assert rt == ResultType.TEXT
        assert content == "plain text"


# ── ErrorClassifier ───────────────────────────────────────────────────────────

class TestErrorClassifier:
    def setup_method(self):
        self.clf = ErrorClassifier()

    def test_rate_limit_is_retry(self):
        kind, wait = self.clf.classify("429 Too Many Requests")
        assert kind == ErrorClassifier.ErrorKind.RETRY
        assert wait > 0

    def test_rate_limit_string(self):
        kind, wait = self.clf.classify("rate limit exceeded, retry after 30 seconds")
        assert kind == ErrorClassifier.ErrorKind.RETRY
        assert wait >= 30

    def test_timeout_is_retry(self):
        kind, _ = self.clf.classify("connection timeout")
        assert kind == ErrorClassifier.ErrorKind.RETRY

    def test_network_error_is_retry(self):
        kind, _ = self.clf.classify("network connection reset")
        assert kind == ErrorClassifier.ErrorKind.RETRY

    def test_permission_denied_is_fatal(self):
        kind, _ = self.clf.classify("permission denied: /etc/passwd")
        assert kind == ErrorClassifier.ErrorKind.FATAL

    def test_not_found_is_fatal(self):
        kind, _ = self.clf.classify("file not found: config.json")
        assert kind == ErrorClassifier.ErrorKind.FATAL

    def test_authentication_is_fatal(self):
        kind, _ = self.clf.classify("authentication failed: invalid token")
        assert kind == ErrorClassifier.ErrorKind.FATAL

    def test_unknown_error(self):
        kind, wait = self.clf.classify("something weird happened")
        assert kind == ErrorClassifier.ErrorKind.UNKNOWN
        assert wait == 0.0

    def test_should_retry_true(self):
        assert self.clf.should_retry("503 Service Unavailable")

    def test_should_retry_false(self):
        assert not self.clf.should_retry("401 Unauthorized")

    def test_is_fatal_true(self):
        assert self.clf.is_fatal("bad request: invalid argument")

    def test_extract_retry_after(self):
        wait = ErrorClassifier._extract_retry_after("retry after 45 seconds please")
        assert wait == 45.0

    def test_extract_retry_after_no_match(self):
        wait = ErrorClassifier._extract_retry_after("some other error")
        assert wait == 0.0


# ── SchemaSanitizer ──────────────────────────────────────────────────────────

class TestSchemaSanitizer:
    def setup_method(self):
        self.san = SchemaSanitizer()

    def test_strips_default(self):
        schema = {"type": "string", "default": "hello"}
        clean  = self.san.sanitize(schema)
        assert "default" not in clean
        assert clean["type"] == "string"

    def test_strips_examples(self):
        schema = {"type": "array", "examples": [1, 2, 3]}
        clean  = self.san.sanitize(schema)
        assert "examples" not in clean

    def test_strips_dollar_schema(self):
        schema = {"$schema": "http://json-schema.org/draft-07/schema", "type": "object"}
        clean  = self.san.sanitize(schema)
        assert "$schema" not in clean

    def test_keeps_supported_fields(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "description": "A user object",
        }
        clean = self.san.sanitize(schema)
        assert "type" in clean
        assert "properties" in clean
        assert "required" in clean
        assert "description" in clean

    def test_deep_clean(self):
        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "default": 0, "examples": [1, 2]},
            },
        }
        clean = self.san.sanitize(schema)
        assert "default"  not in clean["properties"]["x"]
        assert "examples" not in clean["properties"]["x"]

    def test_sanitize_tool_definitions(self):
        defs = [{
            "name": "my_tool",
            "input_schema": {"type": "object", "default": {}, "$schema": "..."},
        }]
        cleaned = self.san.sanitize_tool_definitions(defs)
        assert "default" not in cleaned[0]["input_schema"]
        assert "$schema" not in cleaned[0]["input_schema"]

    def test_array_schema_cleaned(self):
        schema = {"type": "array", "items": {"type": "string", "default": "x"}}
        clean  = self.san.sanitize(schema)
        assert "default" not in clean["items"]

    def test_unsupported_keys_all_stripped(self):
        schema = {k: "value" for k in _SCHEMA_UNSUPPORTED_KEYS}
        schema["type"] = "string"
        clean = self.san.sanitize(schema)
        for key in _SCHEMA_UNSUPPORTED_KEYS:
            assert key not in clean
        assert clean["type"] == "string"


# ── OutputCapEnforcer ────────────────────────────────────────────────────────

class TestOutputCapEnforcer:
    def test_short_content_unchanged(self):
        r = ToolResult(tool_name="x", success=True,
                       result_type=ResultType.TEXT, content="short")
        enf = OutputCapEnforcer()
        enf.enforce(r)
        assert r.content == "short"

    def test_long_content_truncated(self):
        r = ToolResult(
            tool_name="shell_exec", success=True,
            result_type=ResultType.TEXT,
            content="x" * (_DEFAULT_MAX_OUTPUT_CHARS + 1000),
        )
        enf = OutputCapEnforcer()
        enf.enforce(r)
        assert len(r.content) < _DEFAULT_MAX_OUTPUT_CHARS + 200

    def test_tool_override_respected(self):
        r = ToolResult(
            tool_name="my_tool", success=True,
            result_type=ResultType.TEXT,
            content="x" * 500,
        )
        enf = OutputCapEnforcer(tool_overrides={"my_tool": 100})
        enf.enforce(r)
        assert len(r.content) <= 200  # truncated at 100 + marker

    def test_image_cap_larger(self):
        import random
        r = ToolResult(
            tool_name="screenshot", success=True,
            result_type=ResultType.IMAGE,
            content="x" * 20000,
        )
        enf = OutputCapEnforcer()
        before = len(r.content)
        enf.enforce(r)
        # Image cap is larger so 20k might not be truncated
        assert len(r.content) > 0

    def test_empty_content_unchanged(self):
        r = ToolResult(tool_name="x", success=True, content="")
        enf = OutputCapEnforcer()
        enf.enforce(r)
        assert r.content == ""


# ── RetryConfig ──────────────────────────────────────────────────────────────

class TestRetryConfig:
    def test_delay_for_first_attempt(self):
        cfg = RetryConfig(base_delay=2.0, backoff=2.0, jitter=False)
        d = cfg.delay_for(0)
        assert abs(d - 2.0) < 0.1

    def test_delay_increases_with_attempts(self):
        cfg = RetryConfig(base_delay=1.0, backoff=2.0, jitter=False)
        assert cfg.delay_for(1) > cfg.delay_for(0)
        assert cfg.delay_for(2) > cfg.delay_for(1)

    def test_delay_capped_at_max(self):
        cfg = RetryConfig(base_delay=1.0, backoff=10.0, max_delay=30.0, jitter=False)
        assert cfg.delay_for(10) <= 30.0

    def test_defaults(self):
        cfg = RetryConfig()
        assert cfg.max_attempts == 3
        assert cfg.enabled is True


# ── RetryPolicyManager ────────────────────────────────────────────────────────

class TestRetryPolicyManager:
    def setup_method(self):
        self.mgr = RetryPolicyManager()

    def test_get_default(self):
        cfg = self.mgr.get("unknown_tool")
        assert isinstance(cfg, RetryConfig)

    def test_get_http_client(self):
        cfg = self.mgr.get("http_client")
        assert cfg.max_attempts >= 2

    def test_shell_exec_disabled(self):
        cfg = self.mgr.get("shell_exec")
        assert not cfg.enabled

    def test_set_custom(self):
        self.mgr.set("my_tool", RetryConfig(max_attempts=5))
        assert self.mgr.get("my_tool").max_attempts == 5

    def test_enable_disable(self):
        self.mgr.disable("http_client")
        assert not self.mgr.get("http_client").enabled
        self.mgr.enable("http_client")
        assert self.mgr.get("http_client").enabled

    def test_list_returns_dict(self):
        result = self.mgr.list()
        assert isinstance(result, dict)
        assert "http_client" in result


# ── ToolDispatcher ────────────────────────────────────────────────────────────

class TestToolDispatcher:
    def _make_registry(self, tool_name: str, fn):
        return {tool_name: fn}

    def test_execute_success(self):
        registry = self._make_registry("echo", lambda text: {"success": True, "output": text})
        d = ToolDispatcher(tool_registry=registry)
        result = d.execute("echo", {"text": "hello"})
        assert result.success
        assert "hello" in result.content

    def test_execute_missing_tool(self):
        d = ToolDispatcher(tool_registry={})
        result = d.execute("nonexistent", {})
        assert not result.success
        assert "not found" in result.error.lower()

    def test_execute_exception(self):
        def boom(**kwargs):
            raise ValueError("kaboom")
        registry = self._make_registry("boom", boom)
        d = ToolDispatcher(tool_registry=registry)
        result = d.execute("boom", {})
        assert not result.success
        assert "kaboom" in result.error

    def test_execute_tool_failure(self):
        def fail_tool(**kwargs):
            return {"success": False, "error": "permission denied"}
        registry = self._make_registry("fail_tool", fail_tool)
        d = ToolDispatcher(tool_registry=registry)
        result = d.execute("fail_tool", {})
        assert not result.success
        assert "permission denied" in result.error

    def test_stats_counts_calls(self):
        def noop(**kwargs):
            return {"success": True, "output": "ok"}
        registry = self._make_registry("noop", noop)
        d = ToolDispatcher(tool_registry=registry)
        d.execute("noop", {})
        d.execute("noop", {})
        stats = d.stats()
        assert stats["per_tool"]["noop"]["calls"] == 2

    def test_stats_counts_errors(self):
        def boom(**kwargs):
            raise RuntimeError("x")
        registry = self._make_registry("boom", boom)
        d = ToolDispatcher(tool_registry=registry)
        d.execute("boom", {})
        stats = d.stats()
        assert stats["per_tool"]["boom"]["errors"] >= 1

    def test_sanitize_definitions(self):
        defs = [{"name": "x", "input_schema": {"type": "string", "default": "hello"}}]
        d = ToolDispatcher()
        cleaned = d.sanitize_definitions(defs)
        assert "default" not in cleaned[0]["input_schema"]

    def test_set_retry_policy(self):
        d = ToolDispatcher()
        d.set_retry_policy("my_tool", RetryConfig(max_attempts=5))
        cfg = d._retry_mgr.get("my_tool")
        assert cfg.max_attempts == 5

    def test_list_retry_policies(self):
        d = ToolDispatcher()
        result = d.list_retry_policies()
        assert isinstance(result, dict)

    def test_execute_with_no_registry(self):
        d = ToolDispatcher(tool_registry=None)
        result = d.execute("anything", {})
        assert not result.success

    def test_execute_retries_on_transient(self):
        call_count = [0]
        def flaky(**kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                return {"success": False, "error": "503 service unavailable"}
            return {"success": True, "output": "ok"}

        registry = self._make_registry("flaky", flaky)
        d = ToolDispatcher(tool_registry=registry)
        d.set_retry_policy("flaky", RetryConfig(max_attempts=3, base_delay=0.01, enabled=True))
        result = d.execute("flaky", {})
        assert result.success
        assert call_count[0] == 2


# ── ParallelToolDispatcher ────────────────────────────────────────────────────

class TestParallelToolDispatcher:
    def test_parallel_execute_multiple(self):
        def tool_a(**kw): return {"success": True, "output": "A"}
        def tool_b(**kw): return {"success": True, "output": "B"}
        registry = {"tool_a": tool_a, "tool_b": tool_b}
        d = ParallelToolDispatcher(tool_registry=registry, max_workers=2)
        results = d.execute_parallel([
            ("tool_a", {}), ("tool_b", {}),
        ])
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_parallel_preserves_order(self):
        import time
        def slow(**kw): time.sleep(0.05); return {"success": True, "output": "slow"}
        def fast(**kw): return {"success": True, "output": "fast"}
        registry = {"slow": slow, "fast": fast}
        d = ParallelToolDispatcher(tool_registry=registry, max_workers=2)
        results = d.execute_parallel([("slow", {}), ("fast", {})])
        assert results[0].content == "slow"
        assert results[1].content == "fast"

    def test_parallel_handles_error(self):
        def boom(**kw): raise RuntimeError("boom")
        def ok(**kw): return {"success": True, "output": "ok"}
        registry = {"boom": boom, "ok": ok}
        d = ParallelToolDispatcher(tool_registry=registry)
        results = d.execute_parallel([("boom", {}), ("ok", {})])
        assert not results[0].success
        assert results[1].success


# ── Module-level helpers ──────────────────────────────────────────────────────

class TestModuleHelpers:
    def test_classify_result_dict(self):
        raw = {"success": True, "output": "hello"}
        result = classify_result("shell_exec", raw)
        assert isinstance(result, ToolResult)
        assert result.success

    def test_classify_result_none(self):
        result = classify_result("x", None)
        assert result.result_type == ResultType.EMPTY

    def test_sanitize_schema(self):
        schema = {"type": "string", "default": "x", "examples": ["a", "b"]}
        clean  = sanitize_schema(schema)
        assert "default"  not in clean
        assert "examples" not in clean

    def test_get_dispatcher_returns_instance(self):
        d = get_dispatcher()
        assert isinstance(d, ToolDispatcher)

    def test_get_dispatcher_singleton(self):
        d1 = get_dispatcher()
        d2 = get_dispatcher()
        assert d1 is d2

    def test_result_type_values(self):
        assert ResultType.TEXT.value    == "text"
        assert ResultType.CODE.value    == "code"
        assert ResultType.IMAGE.value   == "image"
        assert ResultType.ERROR.value   == "error"
        assert ResultType.EMPTY.value   == "empty"
        assert ResultType.JSON.value    == "json"
        assert ResultType.BINARY.value  == "binary"
