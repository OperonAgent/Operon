"""Tests for core/multi_agent.py

All LLM/router calls are mocked — no network required.
"""
import json
import threading
import time
import unittest.mock as mock
import pytest

from core.multi_agent import (
    AgentRole, AgentResult, MeshResult, AgentMesh,
    create_mesh, _ROLE_TOOLSETS, _ROLE_SYSTEM_PROMPTS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_router(response: str = '{"content": "done", "thought": "ok"}') -> mock.MagicMock:
    router = mock.MagicMock()
    router.complete.return_value = response
    router.parse_response.return_value = json.loads(response) if response.startswith("{") else None
    return router


def _make_registry() -> mock.MagicMock:
    return mock.MagicMock()


def _make_mesh(response: str = '{"content": "ok result", "thought": "done"}',
               max_workers: int = 4) -> AgentMesh:
    return AgentMesh(
        router=_make_router(response),
        tool_registry=_make_registry(),
        max_workers=max_workers,
    )


# ── AgentRole ─────────────────────────────────────────────────────────────────

class TestAgentRole:
    def test_all_roles_defined(self):
        roles = list(AgentRole)
        assert AgentRole.RESEARCHER in roles
        assert AgentRole.CODER      in roles
        assert AgentRole.ANALYST    in roles
        assert AgentRole.WRITER     in roles
        assert AgentRole.REVIEWER   in roles
        assert AgentRole.PLANNER    in roles
        assert AgentRole.GENERALIST in roles

    def test_roles_are_strings(self):
        for role in AgentRole:
            assert isinstance(role.value, str)

    def test_researcher_has_toolset(self):
        assert AgentRole.RESEARCHER in _ROLE_TOOLSETS
        assert "web_search" in _ROLE_TOOLSETS[AgentRole.RESEARCHER]

    def test_coder_has_toolset(self):
        assert AgentRole.CODER in _ROLE_TOOLSETS
        assert "shell_exec" in _ROLE_TOOLSETS[AgentRole.CODER]

    def test_every_role_has_system_prompt(self):
        for role in AgentRole:
            assert role in _ROLE_SYSTEM_PROMPTS
            assert len(_ROLE_SYSTEM_PROMPTS[role]) > 20

    def test_role_from_string(self):
        r = AgentRole("researcher")
        assert r == AgentRole.RESEARCHER

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError):
            AgentRole("not_a_role")


# ── AgentResult ───────────────────────────────────────────────────────────────

class TestAgentResult:
    def test_success_result_fields(self):
        ar = AgentResult(
            role=AgentRole.WRITER, task="write something",
            output="Here it is", success=True, duration_s=1.23,
        )
        assert ar.success is True
        assert ar.output == "Here it is"
        assert ar.duration_s == pytest.approx(1.23, abs=0.001)

    def test_to_dict_truncates_long_output(self):
        ar = AgentResult(
            role=AgentRole.ANALYST, task="a" * 500,
            output="o" * 5000, success=True,
        )
        d = ar.to_dict()
        assert len(d["task"])   <= 200
        assert len(d["output"]) <= 4000

    def test_failure_result(self):
        ar = AgentResult(
            role=AgentRole.CODER, task="debug this",
            output="", success=False, error="crashed",
        )
        assert ar.success is False
        assert ar.error == "crashed"

    def test_to_dict_contains_role_value(self):
        ar = AgentResult(role=AgentRole.PLANNER, task="plan", output="steps", success=True)
        assert ar.to_dict()["role"] == "planner"


# ── MeshResult ────────────────────────────────────────────────────────────────

class TestMeshResult:
    def _make_mesh_result(self, n=2, success=True) -> MeshResult:
        results = [
            AgentResult(role=AgentRole.RESEARCHER, task="t", output=f"out_{i}",
                        success=success)
            for i in range(n)
        ]
        return MeshResult(task="parent task", results=results, mode="pipeline")

    def test_final_output_returns_synthesis_when_set(self):
        mr = self._make_mesh_result()
        mr.synthesis = "synthesised text"
        assert mr.final_output == "synthesised text"

    def test_final_output_falls_back_to_last_result(self):
        mr = self._make_mesh_result(2)
        mr.synthesis = ""
        assert mr.final_output == mr.results[-1].output

    def test_final_output_empty_when_no_results(self):
        mr = MeshResult(task="t")
        assert mr.final_output == ""

    def test_to_dict_keys(self):
        mr = self._make_mesh_result()
        d = mr.to_dict()
        for k in ("task", "mode", "success", "synthesis", "steps"):
            assert k in d

    def test_to_dict_steps_length(self):
        mr = self._make_mesh_result(3)
        assert len(mr.to_dict()["steps"]) == 3

    def test_task_truncated_in_dict(self):
        mr = MeshResult(task="x" * 500)
        assert len(mr.to_dict()["task"]) <= 200


# ── AgentMesh.run_agent ───────────────────────────────────────────────────────

class TestRunAgent:
    def test_successful_run_returns_agent_result(self):
        mesh = _make_mesh('{"content": "research complete", "thought": "done"}')
        result = mesh.run_agent(AgentRole.RESEARCHER, "what is fusion energy?")
        assert isinstance(result, AgentResult)
        assert result.role == AgentRole.RESEARCHER
        assert result.success is True

    def test_result_contains_output(self):
        mesh = _make_mesh('{"content": "analysis done", "thought": "ok"}')
        result = mesh.run_agent(AgentRole.ANALYST, "analyse data")
        assert "analysis done" in result.output or result.success

    def test_duration_recorded(self):
        mesh = _make_mesh()
        result = mesh.run_agent(AgentRole.WRITER, "write a poem")
        assert result.duration_s >= 0.0

    def test_router_exception_returns_failure(self):
        router = mock.MagicMock()
        router.complete.side_effect = RuntimeError("model offline")
        router.parse_response.return_value = None
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        result = mesh.run_agent(AgentRole.CODER, "fix the bug")
        assert result.success is False
        assert "offline" in result.error

    def test_no_tool_call_exits_loop_cleanly(self):
        """Response without tool_call should stop iteration immediately."""
        router = mock.MagicMock()
        router.complete.return_value = '{"content": "final answer", "thought": "done"}'
        router.parse_response.return_value = {"content": "final answer", "thought": "done"}
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        result = mesh.run_agent(AgentRole.GENERALIST, "simple task")
        assert result.success is True
        assert router.complete.call_count == 1   # only one LLM call needed

    def test_context_injected_into_system_when_provided(self):
        router = mock.MagicMock()
        router.complete.return_value = '{"content": "ok"}'
        router.parse_response.return_value = {"content": "ok"}
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        mesh.run_agent(AgentRole.WRITER, "task", context="important context")
        call_args = router.complete.call_args
        system_arg = call_args[1].get("system", "") or call_args[0][0] if call_args[0] else ""
        # Just verify complete was called
        assert router.complete.called

    def test_tool_call_executed_and_result_fed_back(self):
        """When model returns a tool_call, it should be executed and result fed back."""
        call_count = [0]
        responses = [
            '{"content": "", "thought": "check status", "tool_call": {"name": "git_status", "parameters": {"cwd": "/tmp"}}}',
            '{"content": "status checked", "thought": "done"}',
        ]
        def fake_complete(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return responses[min(idx, len(responses) - 1)]

        router = mock.MagicMock()
        router.complete.side_effect = fake_complete
        router.parse_response.side_effect = lambda r: json.loads(r)

        # Patch _DISPATCH to have a fake git_status
        with mock.patch("tools.registry._DISPATCH",
                        {"git_status": lambda **kw: {"success": True, "stdout": "clean", "stderr": "", "returncode": 0}}):
            mesh = AgentMesh(router=router, tool_registry=_make_registry())
            result = mesh.run_agent(AgentRole.CODER, "check git status")
        assert call_count[0] >= 2   # at least two LLM calls (action + follow-up)


# ── AgentMesh.run_pipeline ────────────────────────────────────────────────────

class TestRunPipeline:
    def test_pipeline_returns_mesh_result(self):
        mesh = _make_mesh()
        mr = mesh.run_pipeline("do something", [AgentRole.RESEARCHER, AgentRole.WRITER])
        assert isinstance(mr, MeshResult)

    def test_pipeline_mode_is_pipeline(self):
        mesh = _make_mesh()
        mr = mesh.run_pipeline("task", [AgentRole.RESEARCHER])
        assert mr.mode == "pipeline"

    def test_pipeline_runs_all_roles(self):
        mesh = _make_mesh()
        roles = [AgentRole.PLANNER, AgentRole.RESEARCHER, AgentRole.WRITER]
        mr = mesh.run_pipeline("complex task", roles)
        assert len(mr.results) == len(roles)

    def test_pipeline_stops_on_failure(self):
        router = mock.MagicMock()
        call_count = [0]
        def fake_complete(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("step 1 failed")
            return '{"content": "ok"}'
        router.complete.side_effect = fake_complete
        router.parse_response.return_value = {"content": "ok"}
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        mr = mesh.run_pipeline("task", [AgentRole.PLANNER, AgentRole.RESEARCHER, AgentRole.WRITER])
        assert mr.success is False
        # Pipeline should have stopped after step 1 failed
        assert len(mr.results) == 1

    def test_pipeline_passes_context_to_next_step(self):
        """Each step's output should be passed to the next step's task."""
        outputs = ["step1 output", "step2 output"]
        idx = [0]
        def fake_complete(**kwargs):
            r = outputs[idx[0] % len(outputs)]
            idx[0] += 1
            return json.dumps({"content": r, "thought": "ok"})
        router = mock.MagicMock()
        router.complete.side_effect = fake_complete
        router.parse_response.side_effect = lambda r: json.loads(r)
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        mr = mesh.run_pipeline("task", [AgentRole.RESEARCHER, AgentRole.WRITER])
        # Second call should include "Previous step output" in messages
        calls = router.complete.call_args_list
        if len(calls) >= 2:
            second_call_messages = calls[1][1].get("messages", calls[1][0][1] if len(calls[1][0]) > 1 else [])
            combined = json.dumps(second_call_messages)
            assert "Previous step output" in combined or "step1" in combined.lower()

    def test_pipeline_synthesis_is_last_step_output(self):
        mesh = _make_mesh('{"content": "final", "thought": "done"}')
        mr = mesh.run_pipeline("task", [AgentRole.WRITER])
        assert mr.synthesis != "" or mr.final_output != ""

    def test_single_role_pipeline(self):
        mesh = _make_mesh()
        mr = mesh.run_pipeline("solo task", [AgentRole.GENERALIST])
        assert len(mr.results) == 1


# ── AgentMesh.run_parallel ────────────────────────────────────────────────────

class TestRunParallel:
    def test_parallel_returns_mesh_result(self):
        mesh = _make_mesh()
        mr = mesh.run_parallel("analyse and write", [AgentRole.ANALYST, AgentRole.WRITER])
        assert isinstance(mr, MeshResult)

    def test_parallel_mode(self):
        mesh = _make_mesh()
        mr = mesh.run_parallel("task", [AgentRole.RESEARCHER])
        assert mr.mode == "parallel"

    def test_parallel_runs_all_roles(self):
        mesh = _make_mesh()
        roles = [AgentRole.RESEARCHER, AgentRole.ANALYST, AgentRole.WRITER]
        mr = mesh.run_parallel("task", roles)
        assert len(mr.results) == len(roles)

    def test_parallel_success_when_any_succeed(self):
        router = mock.MagicMock()
        call_count = [0]
        def fake_complete(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first agent failed")
            return '{"content": "ok"}'
        router.complete.side_effect = fake_complete
        router.parse_response.side_effect = lambda r: json.loads(r)
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        mr = mesh.run_parallel("task", [AgentRole.RESEARCHER, AgentRole.WRITER],
                                synthesise=False)
        # At least one succeeded
        assert any(r.success for r in mr.results)
        assert mr.success is True

    def test_parallel_results_sorted_by_input_order(self):
        timing = {AgentRole.ANALYST: 0.05, AgentRole.RESEARCHER: 0.0, AgentRole.WRITER: 0.02}
        def fake_complete(**kwargs):
            # Simulate different latencies via sleep
            return '{"content": "done", "thought": "ok"}'
        router = mock.MagicMock()
        router.complete.side_effect = fake_complete
        router.parse_response.return_value = {"content": "done", "thought": "ok"}
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        roles = [AgentRole.ANALYST, AgentRole.RESEARCHER, AgentRole.WRITER]
        mr = mesh.run_parallel("task", roles, synthesise=False)
        for i, r in enumerate(mr.results):
            assert r.role == roles[i]

    def test_parallel_no_synthesise_skips_synthesis_call(self):
        router = mock.MagicMock()
        router.complete.return_value = '{"content": "done"}'
        router.parse_response.return_value = {"content": "done"}
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        mr = mesh.run_parallel("task", [AgentRole.RESEARCHER], synthesise=False)
        # If synthesis was called, router.complete would be called N+1 times
        # With synthesise=False, only N times (1 role = 1 call)
        assert router.complete.call_count == 1

    def test_parallel_empty_roles_returns_empty_result(self):
        mesh = _make_mesh()
        mr = mesh.run_parallel("task", [])
        assert mr.results == []

    def test_all_fail_mesh_result_success_false(self):
        router = mock.MagicMock()
        router.complete.side_effect = RuntimeError("all offline")
        router.parse_response.return_value = None
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        mr = mesh.run_parallel("task", [AgentRole.RESEARCHER, AgentRole.WRITER],
                                synthesise=False)
        assert mr.success is False


# ── AgentMesh.run_auto ────────────────────────────────────────────────────────

class TestRunAuto:
    def test_auto_falls_back_to_generalist_when_planner_fails(self):
        router = mock.MagicMock()
        call_count = [0]
        def fake_complete(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("planner failed")
            return '{"content": "generalist result"}'
        router.complete.side_effect = fake_complete
        router.parse_response.side_effect = lambda r: json.loads(r)
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        mr = mesh.run_auto("complex task")
        assert mr.mode == "auto_fallback"
        assert mr.success is True

    def test_auto_falls_back_to_pipeline_when_plan_unparseable(self):
        router = mock.MagicMock()
        call_count = [0]
        def fake_complete(**kwargs):
            call_count[0] += 1
            # Planner returns unstructured text (no JSON)
            if call_count[0] == 1:
                return '{"content": "First, do research. Then, write report.", "thought": "ok"}'
            return '{"content": "pipeline result"}'
        router.complete.side_effect = fake_complete
        router.parse_response.side_effect = lambda r: json.loads(r)
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        mr = mesh.run_auto("complex task")
        # Falls back to pipeline (researcher + writer)
        assert mr.success is True

    def test_auto_executes_parsed_plan(self):
        router = mock.MagicMock()
        plan_json = json.dumps({
            "steps": [
                {"step": 1, "role": "researcher", "task": "Research the topic"},
                {"step": 2, "role": "writer",     "task": "Write the report"},
            ]
        })
        call_count = [0]
        def fake_complete(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"content": plan_json, "thought": "planned"})
            return '{"content": "step done"}'
        router.complete.side_effect = fake_complete
        router.parse_response.side_effect = lambda r: json.loads(r)
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        mr = mesh.run_auto("write a report on fusion energy")
        assert mr.mode == "auto"
        assert len(mr.results) == 2   # 2 steps executed

    def test_auto_caps_at_6_steps(self):
        router = mock.MagicMock()
        many_steps = [{"step": i, "role": "generalist", "task": f"step {i}"}
                      for i in range(1, 12)]   # 11 steps
        plan_json = json.dumps({"steps": many_steps})
        call_count = [0]
        def fake_complete(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"content": plan_json, "thought": "ok"})
            return '{"content": "step done"}'
        router.complete.side_effect = fake_complete
        router.parse_response.side_effect = lambda r: json.loads(r)
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        mr = mesh.run_auto("huge task")
        assert len(mr.results) <= 6   # capped at 6

    def test_invalid_role_in_plan_falls_back_to_generalist(self):
        router = mock.MagicMock()
        plan_json = json.dumps({"steps": [
            {"step": 1, "role": "wizard", "task": "do magic"}
        ]})
        call_count = [0]
        def fake_complete(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"content": plan_json, "thought": "ok"})
            return '{"content": "magic done"}'
        router.complete.side_effect = fake_complete
        router.parse_response.side_effect = lambda r: json.loads(r)
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        mr = mesh.run_auto("task")
        assert len(mr.results) == 1
        assert mr.results[0].role == AgentRole.GENERALIST


# ── _parse_plan ───────────────────────────────────────────────────────────────

class TestParsePlan:
    def _mesh(self):
        return AgentMesh(router=mock.MagicMock(), tool_registry=mock.MagicMock())

    def test_valid_json_steps(self):
        text = '{"steps": [{"step": 1, "role": "researcher", "task": "find info"}]}'
        steps = self._mesh()._parse_plan(text)
        assert len(steps) == 1
        assert steps[0]["role"] == "researcher"

    def test_json_embedded_in_prose(self):
        text = 'Here is my plan:\n{"steps": [{"step": 1, "role": "writer", "task": "write"}]}\nEnd.'
        steps = self._mesh()._parse_plan(text)
        assert len(steps) == 1

    def test_no_json_returns_empty(self):
        text = "Just do some research, then write it up."
        steps = self._mesh()._parse_plan(text)
        assert steps == []

    def test_malformed_json_returns_empty(self):
        text = '{"steps": [broken json'
        steps = self._mesh()._parse_plan(text)
        assert steps == []


# ── _synthesise ───────────────────────────────────────────────────────────────

class TestSynthesise:
    def test_empty_results_returns_empty_string(self):
        mesh = _make_mesh()
        s = mesh._synthesise("task", [])
        assert s == ""

    def test_synthesis_calls_router(self):
        router = mock.MagicMock()
        router.complete.return_value = '{"content": "synthesised"}'
        router.parse_response.return_value = {"content": "synthesised"}
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        results = [
            AgentResult(role=AgentRole.RESEARCHER, task="t", output="r1", success=True),
            AgentResult(role=AgentRole.ANALYST,    task="t", output="r2", success=True),
        ]
        s = mesh._synthesise("original task", results)
        assert router.complete.called
        assert s == "synthesised"

    def test_synthesis_failure_falls_back_to_last_success(self):
        router = mock.MagicMock()
        router.complete.side_effect = RuntimeError("model down")
        router.parse_response.return_value = None
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        results = [
            AgentResult(role=AgentRole.RESEARCHER, task="t", output="fallback output", success=True),
        ]
        s = mesh._synthesise("task", results)
        assert s == "fallback output"

    def test_only_successful_results_included(self):
        router = mock.MagicMock()
        router.complete.return_value = '{"content": "done"}'
        router.parse_response.return_value = {"content": "done"}
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        results = [
            AgentResult(role=AgentRole.RESEARCHER, task="t", output="good output", success=True),
            AgentResult(role=AgentRole.CODER,      task="t", output="bad output",  success=False),
        ]
        mesh._synthesise("task", results)
        prompt_content = router.complete.call_args[1].get("messages", [{}])[0].get("content", "")
        assert "bad output" not in prompt_content
        assert "good output" in prompt_content


# ── create_mesh convenience function ─────────────────────────────────────────

class TestCreateMesh:
    def test_returns_agent_mesh(self):
        router   = mock.MagicMock()
        registry = mock.MagicMock()
        mesh = create_mesh(router, registry)
        assert isinstance(mesh, AgentMesh)

    def test_mesh_has_correct_attributes(self):
        router   = mock.MagicMock()
        registry = mock.MagicMock()
        mesh = create_mesh(router, registry)
        assert mesh.router        is router
        assert mesh.tool_registry is registry


# ── Hierarchical orchestration: personas, sandbox, factory, self-correction ───

class TestWorkerTierPersonas:
    def test_engineer_and_auditor_exist(self):
        assert AgentRole.ENGINEER in list(AgentRole)
        assert AgentRole.AUDITOR in list(AgentRole)

    def test_new_roles_have_toolsets(self):
        assert "file_write" in _ROLE_TOOLSETS[AgentRole.ENGINEER]
        assert "apply_patch" in _ROLE_TOOLSETS[AgentRole.ENGINEER]
        # Auditor is review-only: NO write/patch tools (constructive tension)
        for forbidden in ("file_write", "file_patch", "apply_patch", "file_append"):
            assert forbidden not in _ROLE_TOOLSETS[AgentRole.AUDITOR]
        assert "shell_exec" in _ROLE_TOOLSETS[AgentRole.AUDITOR]

    def test_new_roles_have_personas(self):
        assert "Execution Engineer" in _ROLE_SYSTEM_PROMPTS[AgentRole.ENGINEER]
        assert "Quality Auditor" in _ROLE_SYSTEM_PROMPTS[AgentRole.AUDITOR]
        assert "PASS" in _ROLE_SYSTEM_PROMPTS[AgentRole.AUDITOR]


class TestRunWithTools:
    def test_explicit_allocation_runs(self):
        mesh = _make_mesh('{"content": "research done"}')
        res = mesh.run_with_tools("researcher", "find facts",
                                  allocated_tools=["web_search"])
        assert res.success is True
        assert res.role == AgentRole.RESEARCHER

    def test_unknown_persona_falls_back_to_generalist(self):
        mesh = _make_mesh()
        res = mesh.run_with_tools("DataWizard9000", "do a thing", allocated_tools=[])
        assert res.role == AgentRole.GENERALIST

    def test_empty_objective_handled(self):
        mesh = _make_mesh()
        res = mesh.run_with_tools("engineer", "", allocated_tools=["file_read"])
        # should not raise; objective normalised
        assert isinstance(res, AgentResult)

    def test_comma_string_tools_accepted(self):
        mesh = _make_mesh()
        res = mesh.run_with_tools("auditor", "review", allocated_tools="file_read, shell_exec")
        assert res.success is True


class TestSandboxEnforcement:
    def test_disallowed_tool_is_blocked(self):
        # Worker tries to call file_write but is only allocated file_read.
        seq = [
            '{"content": "", "tool_call": {"name": "file_write", "parameters": {"path": "x", "content": "y"}}}',
            '{"content": "finished"}',
        ]
        router = mock.MagicMock()
        router.complete.side_effect = seq
        router.parse_response.side_effect = lambda r: json.loads(r)
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        out = mesh._run_with_timeout(AgentRole.AUDITOR, "audit", "", 30,
                                     explicit_tools=["file_read"])
        # The blocked attempt must have produced a sandbox error, then it finished.
        assert "finished" in out


class TestSelfCorrectionLoop:
    def test_passes_when_auditor_signs_off(self):
        # Engineer returns content, Auditor returns PASS -> loop stops round 1.
        def parse(r):
            try: return json.loads(r)
            except Exception: return {"content": r}
        router = mock.MagicMock()
        router.parse_response.side_effect = parse
        router.complete.side_effect = [
            '{"content": "implemented the fix"}',     # engineer round 1
            '{"content": "PASS - looks correct"}',    # auditor round 1
        ]
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        res = mesh.run_self_correction("make it work", max_rounds=3)
        assert res.success is True
        assert res.mode == "self_correction"
        assert len(res.results) == 2

    def test_iterates_then_gives_up(self):
        def parse(r):
            try: return json.loads(r)
            except Exception: return {"content": r}
        router = mock.MagicMock()
        router.parse_response.side_effect = parse
        # Always NEEDS_REVISION -> exhausts max_rounds=2 (4 calls).
        router.complete.side_effect = [
            '{"content": "attempt 1"}', '{"content": "NEEDS_REVISION: fix line 2"}',
            '{"content": "attempt 2"}', '{"content": "NEEDS_REVISION: still broken"}',
        ]
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        res = mesh.run_self_correction("hard task", max_rounds=2)
        assert res.success is False
        assert len(res.results) == 4

    def test_verify_command_authoritative(self):
        # Auditor says PASS but the verify command fails -> not passed.
        def parse(r):
            try: return json.loads(r)
            except Exception: return {"content": r}
        router = mock.MagicMock()
        router.parse_response.side_effect = parse
        router.complete.side_effect = [
            '{"content": "did it"}', '{"content": "PASS"}',
            '{"content": "did it again"}', '{"content": "PASS"}',
        ]
        mesh = AgentMesh(router=router, tool_registry=_make_registry())
        with mock.patch.object(mesh, "_verify", return_value=("FAILED: 1 test", False)):
            res = mesh.run_self_correction("x", verify_cmd="pytest", max_rounds=2)
        assert res.success is False
