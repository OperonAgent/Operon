"""tests/test_cmd_handlers.py — modular command dispatch layer."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cmd_handlers import CommandContext, DISPATCH, dispatch, command


def _ctx(cmd, **over):
    parts = cmd.split()
    theme = MagicMock()
    theme.box = lambda l, **k: "BOX"
    theme.info = theme.success = theme.error = theme.warning = lambda x: str(x)
    session = MagicMock()
    session.get_history_display.return_value = ["t1"]
    session.get_usage_stats.return_value = {
        "turns": 1, "messages": 2, "chars": 50, "est_tokens": 12, "est_cost_4o": 0.01}
    session.__len__ = lambda s: 3
    session.compress.return_value = 5
    session.undo.return_value = True
    tr = MagicMock(); tr.tools = {"a": 1, "b": 2}
    ct = MagicMock(); ct._calls = []
    base = dict(command=cmd, parts=parts, cmd=parts[0], args=parts[1:],
                theme=theme, session=session, tool_registry=tr, cost_tracker=ct)
    base.update(over)
    return CommandContext(**base)


class TestRegistry:
    def test_known_commands_registered(self):
        for c in ("/clear", "/undo", "/history", "/compress", "/tools", "/usage", "/cost"):
            assert c in DISPATCH

    def test_dispatch_unknown_returns_false(self):
        assert dispatch(_ctx("/nonexistent")) is False

    def test_dispatch_known_returns_true(self):
        with patch("builtins.print"):
            assert dispatch(_ctx("/clear")) is True

    def test_command_decorator_registers(self):
        @command("/__test_cmd__")
        def _h(ctx): ctx.theme.info("hi")
        assert "/__test_cmd__" in DISPATCH
        del DISPATCH["/__test_cmd__"]


class TestSessionCommands:
    def test_clear_calls_session(self):
        ctx = _ctx("/clear")
        with patch("builtins.print"):
            dispatch(ctx)
        ctx.session.clear.assert_called_once()

    def test_undo(self):
        ctx = _ctx("/undo")
        with patch("builtins.print"):
            dispatch(ctx)
        ctx.session.undo.assert_called_once()

    def test_history_default_n(self):
        ctx = _ctx("/history")
        with patch("builtins.print"):
            dispatch(ctx)
        ctx.session.get_history_display.assert_called_with(last_n=20)

    def test_history_custom_n(self):
        ctx = _ctx("/history 7")
        with patch("builtins.print"):
            dispatch(ctx)
        ctx.session.get_history_display.assert_called_with(last_n=7)

    def test_compress(self):
        ctx = _ctx("/compress")
        with patch("builtins.print"):
            dispatch(ctx)
        ctx.session.compress.assert_called_once()


class TestInfoCommands:
    def test_tools_lists(self):
        ctx = _ctx("/tools")
        with patch("builtins.print") as p:
            dispatch(ctx)
        assert p.called

    def test_usage_no_crash(self):
        ctx = _ctx("/usage")
        with patch("builtins.print"):
            dispatch(ctx)
        ctx.session.get_usage_stats.assert_called_once()

    def test_usage_handles_exception(self):
        ctx = _ctx("/usage")
        ctx.session.get_usage_stats.side_effect = RuntimeError("x")
        with patch("builtins.print"):
            dispatch(ctx)  # must not raise

    def test_cost_no_calls(self):
        ctx = _ctx("/cost")
        with patch("builtins.print"):
            dispatch(ctx)

    def test_cost_with_calls(self):
        ctx = _ctx("/cost")
        ctx.cost_tracker._calls = [1]
        ctx.cost_tracker.session_report.return_value = ["report"]
        with patch("builtins.print"):
            dispatch(ctx)
        ctx.cost_tracker.session_report.assert_called_once()


class TestConfigCommands:
    def test_registered(self):
        for c in ("/models", "/config", "/soul"):
            assert c in DISPATCH

    def test_models(self):
        cfg = MagicMock()
        cfg.get.side_effect = lambda k, d=None: (
            {"a": {"provider": "openai"}} if k == "model_profiles" else "a")
        ctx = _ctx("/models", config=cfg)
        with patch("builtins.print") as p:
            dispatch(ctx)
        assert p.called

    def test_config(self):
        cfg = MagicMock(); cfg.get_safe_display.return_value = {"model": "gpt-4o"}
        ctx = _ctx("/config", config=cfg)
        with patch("builtins.print"):
            dispatch(ctx)
        cfg.get_safe_display.assert_called_once()

    def test_soul_none(self):
        ctx = _ctx("/soul", soul=None)
        with patch("builtins.print"):
            dispatch(ctx)  # must not raise

    def test_soul_read(self):
        soul = MagicMock(); soul.read.return_value = "I am Operon."
        ctx = _ctx("/soul", soul=soul)
        with patch("builtins.print"):
            dispatch(ctx)
        soul.read.assert_called_once()


class TestContextDataclass:
    def test_fields_present(self):
        c = _ctx("/clear")
        assert c.cmd == "/clear"
        assert c.args == []
        assert isinstance(c.extras, dict)
