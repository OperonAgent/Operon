"""tests/test_spawn_agent.py — the spawn_agent factory meta-tool."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools.registry as r


@pytest.fixture(autouse=True)
def _reset_factory():
    saved = r._agent_factory
    r._agent_factory = None
    yield
    r._agent_factory = saved


class TestRegistration:
    def test_registered_in_dispatch_and_defs(self):
        assert "spawn_agent" in r._DISPATCH
        assert any(d["name"] == "spawn_agent" for d in r._TOOL_DEFINITIONS)

    def test_blocked_for_sub_agents(self):
        # workers must not recursively spawn
        assert "spawn_agent" in r.DELEGATE_BLOCKED_TOOLS

    def test_in_agent_toolset(self):
        assert "spawn_agent" in r.TOOLSETS["agent"]


class TestGracefulPaths:
    def test_requires_objective(self):
        out = r.spawn_agent(persona="engineer", objective="")
        assert out["success"] is False and "objective" in out["error"]

    def test_unwired_factory(self):
        out = r.spawn_agent(persona="engineer", objective="do x")
        assert out["success"] is False and "factory" in out["error"].lower()

    def test_missing_persona_defaults(self):
        r.set_agent_factory(lambda p, o, t: {"success": True, "persona": p,
                                             "output": "ok", "error": ""})
        out = r.spawn_agent(objective="do x")
        assert out["success"] is True
        assert out["persona"] == "generalist"


class TestToolAllocation:
    def test_comma_string_normalised(self):
        captured = {}
        r.set_agent_factory(lambda p, o, t: captured.update(tools=t) or
                            {"success": True, "persona": p, "output": "", "error": ""})
        r.spawn_agent("researcher", "find", allocated_tools="web_search, file_read")
        assert captured["tools"] == ["web_search", "file_read"]

    def test_blocked_tools_stripped_from_allocation(self):
        captured = {}
        r.set_agent_factory(lambda p, o, t: captured.update(tools=t) or
                            {"success": True, "persona": p, "output": "", "error": ""})
        r.spawn_agent("engineer", "x",
                      allocated_tools=["file_read", "spawn_agent", "sub_agent"])
        assert "spawn_agent" not in captured["tools"]
        assert "sub_agent" not in captured["tools"]
        assert "file_read" in captured["tools"]

    def test_factory_exception_caught(self):
        def boom(p, o, t):
            raise RuntimeError("kaboom")
        r.set_agent_factory(boom)
        out = r.spawn_agent("engineer", "x")
        assert out["success"] is False and "kaboom" in out["error"]
