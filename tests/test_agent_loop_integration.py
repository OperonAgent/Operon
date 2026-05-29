"""
tests/test_agent_loop_integration.py — integration tests for the agent loop's
core decision unit: model-response parsing/repair + action dispatch.

These are NOT smoke tests — they feed realistic, messy model outputs through
the real parser (ModelRouter.parse_response) and the real parallel-call parser,
asserting the exact action the loop would dispatch.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.router import ModelRouter
from core.parallel_executor import parse_parallel_calls, ParallelToolExecutor, ToolCall

P = ModelRouter.parse_response


# ── Parser: the 7 documented passes ────────────────────────────────────────────

class TestParseClean:
    def test_pure_json_response(self):
        assert P('{"action": "response", "content": "hello"}')["content"] == "hello"

    def test_pure_json_tool(self):
        d = P('{"action": {"type": "tool", "tool_name": "shell_exec", "params": {"cmd": "ls"}}}')
        assert d["action"]["tool_name"] == "shell_exec"

    def test_list_of_objects_picks_first_dict(self):
        d = P('[{"action": "response", "content": "a"}, {"x": 1}]')
        assert d["content"] == "a"


class TestParseMarkdownFence:
    def test_json_fence(self):
        d = P('Here you go:\n```json\n{"action": "response", "content": "x"}\n```')
        assert d["content"] == "x"

    def test_bare_fence(self):
        d = P('```\n{"action": "response", "content": "y"}\n```')
        assert d["content"] == "y"

    def test_prose_around_fence(self):
        d = P('Sure! ```json\n{"k": 1}\n``` hope that helps')
        assert d["k"] == 1


class TestParseRepair:
    def test_trailing_comma(self):
        d = P('{"a": 1, "b": 2,}')
        assert d == {"a": 1, "b": 2}

    def test_python_booleans(self):
        d = P('{"ok": True, "bad": False, "none": None}')
        assert d["ok"] is True and d["bad"] is False and d["none"] is None

    def test_embedded_in_prose(self):
        d = P('The answer is {"action": "response", "content": "42"} as computed.')
        assert d["content"] == "42"

    def test_first_of_multiple_objects(self):
        d = P('{"action": "response", "content": "first"}{"action": "response", "content": "second"}')
        assert d["content"] == "first"


class TestParseEdgeCases:
    def test_empty_returns_none(self):
        assert P("") is None

    def test_pure_prose_returns_none(self):
        # No JSON at all → None (loop treats as plain response)
        assert P("just some text with no json") is None

    def test_bare_scalar_returns_none(self):
        assert P("42") is None

    def test_whitespace_only(self):
        assert P("   \n  ") is None


# ── Action-shape normalisation (mirrors run_agent_loop) ────────────────────────

def _normalise_action(parsed: dict) -> dict:
    """Mirror of the loop's action-normalisation, for assertion in tests."""
    action = parsed.get("action", {})
    if isinstance(action, str):
        if action == "response":
            return {"type": "response", "content": parsed.get("content", "")}
        return {"type": "tool", "tool_name": action, "params": parsed.get("params", {})}
    return action if isinstance(action, dict) else {"type": "response", "content": ""}


class TestActionNormalisation:
    def test_string_response_action(self):
        a = _normalise_action(P('{"action": "response", "content": "hi"}'))
        assert a["type"] == "response" and a["content"] == "hi"

    def test_string_tool_action(self):
        a = _normalise_action(P('{"action": "shell_exec", "params": {"cmd": "ls"}}'))
        assert a["type"] == "tool" and a["tool_name"] == "shell_exec"

    def test_dict_tool_action(self):
        a = _normalise_action(P('{"action": {"type": "tool", "tool_name": "x", "params": {}}}'))
        assert a["type"] == "tool" and a["tool_name"] == "x"


# ── parallel_tools end-to-end through the executor ─────────────────────────────

class TestParallelToolsFlow:
    def test_parse_then_execute(self):
        action = {
            "type": "parallel_tools",
            "calls": [
                {"tool_name": "echo", "params": {"v": 1}, "id": "a"},
                {"tool_name": "echo", "params": {"v": 2}, "id": "b"},
            ],
        }
        calls = parse_parallel_calls(action)
        assert len(calls) == 2

        def execfn(name, params):
            return {"success": True, "doubled": params["v"] * 2}

        result = ParallelToolExecutor(max_workers=2).run(
            calls, execute_fn=execfn, print_fn=lambda *_: None)
        assert result.all_succeeded
        doubled = sorted(r.result["doubled"] for r in result.results)
        assert doubled == [2, 4]

    def test_full_model_output_to_parallel_dispatch(self):
        raw = ('```json\n{"action": {"type": "parallel_tools", "calls": ['
               '{"tool_name": "t", "params": {"x": 5}, "id": "1"}]}}\n```')
        parsed = P(raw)
        action = parsed["action"]
        assert action["type"] == "parallel_tools"
        calls = parse_parallel_calls(action)
        assert calls[0].params["x"] == 5


# ── Tool dispatch via a registry-like execute_fn ───────────────────────────────

class TestToolDispatch:
    def test_dispatch_routes_to_named_tool(self):
        registry = {"adder": lambda **kw: {"success": True, "sum": kw["a"] + kw["b"]}}
        parsed = P('{"action": {"type": "tool", "tool_name": "adder", "params": {"a": 2, "b": 3}}}')
        a = parsed["action"]
        out = registry[a["tool_name"]](**a["params"])
        assert out["sum"] == 5

    def test_unknown_tool_detectable(self):
        registry = {"known": lambda **kw: {}}
        parsed = P('{"action": {"type": "tool", "tool_name": "ghost", "params": {}}}')
        assert parsed["action"]["tool_name"] not in registry
