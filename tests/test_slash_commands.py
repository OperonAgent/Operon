"""
tests/test_slash_commands.py — Smoke-tests for every handle_command branch.

Each test calls handle_command() with a mocked session/config/theme and asserts
it returns (or raises SystemExit) without error.  This catches NameError,
UnboundLocalError, KeyError, and similar regressions introduced when the
large handle_command function is edited.
"""

from __future__ import annotations

import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# ── Make sure repo root is on sys.path ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
from core.config import ConfigManager


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return ConfigManager()


@pytest.fixture
def session():
    m = MagicMock()
    m.history = []
    m.session_id = "test-session-001"
    m.get_usage_stats.return_value = {
        "turns": 5,
        "messages": 10,
        "chars": 5000,
        "est_tokens": 1250,
        "est_cost_4o": 0.0025,
    }
    return m


@pytest.fixture
def memory():
    m = MagicMock()
    m.search.return_value = []
    m.get_all.return_value = []
    return m


@pytest.fixture
def theme():
    m = MagicMock()
    m.box     = lambda lines, **kw: "\n".join(str(x) for x in lines)
    m.info    = lambda x: str(x)
    m.success = lambda x: str(x)
    m.error   = lambda x: str(x)
    m.warning = lambda x: str(x)
    return m


def _run(cmd: str, cfg, session, memory, theme):
    """Execute handle_command under suppressed stdout; re-raise non-SystemExit."""
    with patch("builtins.print"):
        main.handle_command(
            cmd, cfg, session, memory, theme,
            tool_registry=MagicMock(),
            skills=MagicMock(),
        )


# ── Core info commands ────────────────────────────────────────────────────────

class TestCoreCommands:
    def test_help(self, cfg, session, memory, theme):
        _run("/help", cfg, session, memory, theme)

    def test_clear(self, cfg, session, memory, theme):
        _run("/clear", cfg, session, memory, theme)

    def test_tools(self, cfg, session, memory, theme):
        _run("/tools", cfg, session, memory, theme)

    def test_usage(self, cfg, session, memory, theme):
        _run("/usage", cfg, session, memory, theme)

    def test_cost(self, cfg, session, memory, theme):
        _run("/cost", cfg, session, memory, theme)

    def test_history(self, cfg, session, memory, theme):
        _run("/history", cfg, session, memory, theme)

    def test_memory(self, cfg, session, memory, theme):
        _run("/memory", cfg, session, memory, theme)

    def test_skills(self, cfg, session, memory, theme):
        _run("/skills", cfg, session, memory, theme)

    def test_model(self, cfg, session, memory, theme):
        _run("/model", cfg, session, memory, theme)

    def test_models(self, cfg, session, memory, theme):
        _run("/models", cfg, session, memory, theme)

    def test_config(self, cfg, session, memory, theme):
        _run("/config", cfg, session, memory, theme)

    def test_context(self, cfg, session, memory, theme):
        _run("/context", cfg, session, memory, theme)

    def test_session(self, cfg, session, memory, theme):
        _run("/session", cfg, session, memory, theme)

    def test_stats(self, cfg, session, memory, theme):
        _run("/stats", cfg, session, memory, theme)

    def test_approve(self, cfg, session, memory, theme):
        _run("/approve", cfg, session, memory, theme)

    def test_reflect(self, cfg, session, memory, theme):
        _run("/reflect", cfg, session, memory, theme)

    def test_doctor(self, cfg, session, memory, theme):
        # Uses importlib internally — must not crash even with mocked print
        _run("/doctor", cfg, session, memory, theme)

    def test_router(self, cfg, session, memory, theme):
        _run("/router", cfg, session, memory, theme)

    def test_soul(self, cfg, session, memory, theme):
        _run("/soul", cfg, session, memory, theme)

    def test_toolsets(self, cfg, session, memory, theme):
        _run("/toolsets", cfg, session, memory, theme)

    def test_local(self, cfg, session, memory, theme):
        _run("/local", cfg, session, memory, theme)


# ── Kanban ────────────────────────────────────────────────────────────────────

class TestKanbanCommands:
    def test_kanban_bare(self, cfg, session, memory, theme):
        _run("/kanban", cfg, session, memory, theme)

    def test_kanban_list(self, cfg, session, memory, theme):
        # Previously crashed with KeyError: 'total' (agent_list returns 'count')
        _run("/kanban list", cfg, session, memory, theme)

    def test_kanban_board(self, cfg, session, memory, theme):
        _run("/kanban board", cfg, session, memory, theme)

    def test_kanban_export(self, cfg, session, memory, theme):
        # Previously crashed with UnboundLocalError: 'os' — bare `import os` shadowed
        _run("/kanban export", cfg, session, memory, theme)

    def test_kanban_help(self, cfg, session, memory, theme):
        _run("/kanban help", cfg, session, memory, theme)

    def test_kanban_add(self, cfg, session, memory, theme):
        _run("/kanban add Test task title", cfg, session, memory, theme)

    def test_kanban_list_with_status(self, cfg, session, memory, theme):
        _run("/kanban list todo", cfg, session, memory, theme)


# ── Checkpoint ────────────────────────────────────────────────────────────────

class TestCheckpointCommands:
    def test_checkpoint_bare(self, cfg, session, memory, theme):
        # Previously crashed with UnboundLocalError: 'os' referenced before assignment
        _run("/checkpoint", cfg, session, memory, theme)

    def test_checkpoint_status(self, cfg, session, memory, theme):
        _run("/checkpoint status", cfg, session, memory, theme)

    def test_checkpoint_list(self, cfg, session, memory, theme):
        _run("/checkpoint list", cfg, session, memory, theme)

    def test_checkpoint_help(self, cfg, session, memory, theme):
        _run("/checkpoint help", cfg, session, memory, theme)

    def test_checkpoint_create(self, cfg, session, memory, theme):
        _run("/checkpoint create test snapshot", cfg, session, memory, theme)


# ── Vector / Obsidian / Synth / Desktop ──────────────────────────────────────

class TestSubCommandHandlers:
    def test_vector_bare(self, cfg, session, memory, theme):
        # Previously crashed with NameError: name 'args' is not defined
        _run("/vector", cfg, session, memory, theme)

    def test_vector_status(self, cfg, session, memory, theme):
        _run("/vector status", cfg, session, memory, theme)

    def test_vector_recall(self, cfg, session, memory, theme):
        _run("/vector recall test query", cfg, session, memory, theme)

    def test_desktop_bare(self, cfg, session, memory, theme):
        # Previously crashed with NameError: name 'args' is not defined
        _run("/desktop", cfg, session, memory, theme)

    def test_desktop_status(self, cfg, session, memory, theme):
        _run("/desktop status", cfg, session, memory, theme)

    def test_synth_bare(self, cfg, session, memory, theme):
        # Previously crashed with NameError: name 'args' is not defined
        _run("/synth", cfg, session, memory, theme)

    def test_synth_status(self, cfg, session, memory, theme):
        _run("/synth status", cfg, session, memory, theme)

    def test_obsidian_bare(self, cfg, session, memory, theme):
        _run("/obsidian", cfg, session, memory, theme)

    def test_obsidian_status(self, cfg, session, memory, theme):
        _run("/obsidian status", cfg, session, memory, theme)


# ── Service commands ──────────────────────────────────────────────────────────

class TestServiceCommands:
    def test_dashboard(self, cfg, session, memory, theme):
        _run("/dashboard", cfg, session, memory, theme)

    def test_mcp(self, cfg, session, memory, theme):
        _run("/mcp", cfg, session, memory, theme)

    def test_webhook(self, cfg, session, memory, theme):
        _run("/webhook", cfg, session, memory, theme)

    def test_heartbeat(self, cfg, session, memory, theme):
        _run("/heartbeat", cfg, session, memory, theme)

    def test_gateway(self, cfg, session, memory, theme):
        _run("/gateway", cfg, session, memory, theme)

    def test_serve(self, cfg, session, memory, theme):
        _run("/serve", cfg, session, memory, theme)

    def test_rag(self, cfg, session, memory, theme):
        _run("/rag", cfg, session, memory, theme)

    def test_secrets(self, cfg, session, memory, theme):
        _run("/secrets", cfg, session, memory, theme)

    def test_pool(self, cfg, session, memory, theme):
        _run("/pool", cfg, session, memory, theme)

    def test_curator(self, cfg, session, memory, theme):
        _run("/curator", cfg, session, memory, theme)


# ── Task / planning commands ──────────────────────────────────────────────────

class TestTaskCommands:
    def test_goal(self, cfg, session, memory, theme):
        _run("/goal", cfg, session, memory, theme)

    def test_macro(self, cfg, session, memory, theme):
        _run("/macro", cfg, session, memory, theme)

    def test_mesh(self, cfg, session, memory, theme):
        _run("/mesh", cfg, session, memory, theme)

    def test_tasks(self, cfg, session, memory, theme):
        _run("/tasks", cfg, session, memory, theme)

    def test_schedule(self, cfg, session, memory, theme):
        _run("/schedule", cfg, session, memory, theme)

    def test_status(self, cfg, session, memory, theme):
        _run("/status", cfg, session, memory, theme)


# ── Advanced agent commands ───────────────────────────────────────────────────

class TestAgentCommands:
    def test_voice(self, cfg, session, memory, theme):
        _run("/voice", cfg, session, memory, theme)

    def test_swe(self, cfg, session, memory, theme):
        _run("/swe", cfg, session, memory, theme)

    def test_plugin(self, cfg, session, memory, theme):
        _run("/plugin", cfg, session, memory, theme)

    def test_plugins(self, cfg, session, memory, theme):
        _run("/plugins", cfg, session, memory, theme)

    def test_conv(self, cfg, session, memory, theme):
        _run("/conv", cfg, session, memory, theme)


# ── /usage defensive coverage (was MagicMock.__format__ crash) ───────────────

class TestUsageDefensive:
    def test_usage_with_proper_stats(self, cfg, session, memory, theme):
        session.get_usage_stats.return_value = {
            "turns": 10, "messages": 20,
            "chars": 100_000, "est_tokens": 25_000, "est_cost_4o": 0.5,
        }
        _run("/usage", cfg, session, memory, theme)

    def test_usage_with_zero_stats(self, cfg, session, memory, theme):
        session.get_usage_stats.return_value = {
            "turns": 0, "messages": 0,
            "chars": 0, "est_tokens": 0, "est_cost_4o": 0.0,
        }
        _run("/usage", cfg, session, memory, theme)

    def test_usage_missing_keys(self, cfg, session, memory, theme):
        # Partial response should not crash
        session.get_usage_stats.return_value = {}
        _run("/usage", cfg, session, memory, theme)

    def test_usage_exception(self, cfg, session, memory, theme):
        session.get_usage_stats.side_effect = RuntimeError("DB error")
        _run("/usage", cfg, session, memory, theme)


# ── Exit is a SystemExit not a crash ─────────────────────────────────────────

class TestExitCommand:
    def test_exit_raises_systemexit(self, cfg, session, memory, theme):
        with patch("builtins.print"), pytest.raises(SystemExit):
            main.handle_command("/exit", cfg, session, memory, theme)

    def test_quit_raises_systemexit(self, cfg, session, memory, theme):
        with patch("builtins.print"), pytest.raises(SystemExit):
            main.handle_command("/quit", cfg, session, memory, theme)
