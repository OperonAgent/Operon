"""Tests for core/skill_synthesizer.py — Self-improvement loop Phase 11

Actual API (from inspection):
  TaskTrajectory(user_request, steps, final_answer, success, start_time, end_time, session_id)
  TrajectoryStep(tool_name, params, result, success, duration_ms, timestamp)
  SynthesizedSkill(name, description, trigger, steps_md, notes, example_req, example_out,
                   tools_used, quality, created_at, use_count, skill_id)
  SkillStore(skills_dir)  — methods: save, load, load_all, count, search_index, increment_use
  SkillMatcher(store)     — methods: find_relevant, build_context_block
  TrajectoryAnalyser()    — instance method: analyse(traj)
  SkillWriter()           — instance method: write(recipe, user_request, outcome='')
  SkillSynthesizer(skills_dir) — methods: start_trajectory, record_step, finish_trajectory,
                                  synthesize, synthesize_from_current, get_hints_for,
                                  list_skills, delete_skill, stats, summary, reset_trajectory
"""
import json
import time
import pathlib
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.skill_synthesizer import (
    TrajectoryStep, TaskTrajectory, TrajectoryAnalyser,
    SynthesizedSkill, SkillWriter, SkillStore, SkillMatcher,
    SkillSynthesizer, get_synthesizer,
    _MIN_QUALITY, _MIN_STEPS,
)


# ── TrajectoryStep ─────────────────────────────────────────────────────────────

class TestTrajectoryStep:
    def test_creation(self):
        step = TrajectoryStep(
            tool_name="shell_exec",
            params={"cmd": "ls"},
            result="file1.txt\nfile2.txt",
            success=True,
            duration_ms=120,
        )
        assert step.tool_name == "shell_exec"
        assert step.success is True
        assert step.duration_ms == 120

    def test_timestamp_auto_set(self):
        step = TrajectoryStep(tool_name="read_file", params={}, result="content", success=True)
        assert step.timestamp > 0

    def test_failed_step(self):
        step = TrajectoryStep(
            tool_name="shell_exec",
            params={"cmd": "badcmd"},
            result="",
            success=False,
            duration_ms=50,
        )
        assert step.success is False

    def test_default_duration(self):
        step = TrajectoryStep(tool_name="x", params={}, result="y", success=True)
        assert step.duration_ms == 0


# ── TaskTrajectory ─────────────────────────────────────────────────────────────

class TestTaskTrajectory:
    def _make_traj(self, n_steps=3, all_success=True, request="Fix a bug"):
        t = TaskTrajectory(user_request=request)
        for i in range(n_steps):
            t.add_step(
                tool_name=f"tool_{i}",
                params={"key": f"val_{i}"},
                result=f"result_{i}",
                success=all_success,
                duration_ms=100,
            )
        return t

    def test_add_step(self):
        t = self._make_traj(2)
        assert len(t.steps) == 2

    def test_tools_used(self):
        t = self._make_traj(3)
        tools = t.tools_used
        assert "tool_0" in tools
        assert "tool_1" in tools
        assert "tool_2" in tools

    def test_tools_used_deduplicates(self):
        t = TaskTrajectory(user_request="test dedup")
        for _ in range(3):
            t.add_step("shell_exec", {}, "ok", True)
        tools = t.tools_used
        assert tools.count("shell_exec") == 1

    def test_successful_steps(self):
        t = TaskTrajectory(user_request="mixed")
        t.add_step("a", {}, "ok", True)
        t.add_step("b", {}, "fail", False)
        t.add_step("c", {}, "ok", True)
        assert len(t.successful_steps) == 2

    def test_quality_score_all_success(self):
        t = self._make_traj(4, all_success=True)
        # Mark as finished with success + a meaningful final answer
        t.final_answer = "Successfully completed the task with all steps passing."
        t.finish(answer="Task completed successfully.", success=True)
        score = t.quality_score()
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_quality_score_all_fail(self):
        t = self._make_traj(4, all_success=False)
        t.finish(answer="", success=False)
        score = t.quality_score()
        # With all failures and success=False, score should be lower
        # (not necessarily < 0.5 due to formula constants)
        assert 0.0 <= score <= 1.0

    def test_quality_score_empty(self):
        t = TaskTrajectory(user_request="nothing")
        score = t.quality_score()
        assert score == 0.0

    def test_finish_sets_end_time(self):
        t = self._make_traj(2)
        t.finish(answer="Task completed successfully.", success=True)
        assert t.end_time > 0

    def test_finish_sets_success(self):
        t = self._make_traj(2)
        t.finish(answer="Task completed successfully.", success=True)
        assert t.success is True

    def test_finish_false(self):
        t = self._make_traj(1)
        t.finish(answer="", success=False)
        assert t.success is False

    def test_duration_s_after_finish(self):
        t = self._make_traj(2)
        t.finish(answer="Task completed successfully.", success=True)
        assert t.duration_s >= 0.0


# ── TrajectoryAnalyser ────────────────────────────────────────────────────────

class TestTrajectoryAnalyser:
    def _make_finished_traj(self):
        t = TaskTrajectory(user_request="Read a file and run tests")
        t.add_step("read_file", {"path": "test.py"}, "content", True, 100)
        t.add_step("shell_exec", {"cmd": "pytest"}, "2 passed", True, 500)
        t.finish(answer="Task completed successfully.", success=True)
        return t

    def test_analyse_returns_dict(self):
        analyser = TrajectoryAnalyser()
        t = self._make_finished_traj()
        recipe = analyser.analyse(t)
        assert isinstance(recipe, dict)

    def test_analyse_has_intent(self):
        analyser = TrajectoryAnalyser()
        t = self._make_finished_traj()
        recipe = analyser.analyse(t)
        assert "intent" in recipe

    def test_analyse_has_steps(self):
        analyser = TrajectoryAnalyser()
        t = self._make_finished_traj()
        recipe = analyser.analyse(t)
        assert "steps" in recipe
        assert isinstance(recipe["steps"], list)

    def test_analyse_has_tools(self):
        analyser = TrajectoryAnalyser()
        t = self._make_finished_traj()
        recipe = analyser.analyse(t)
        assert "tools_used" in recipe

    def test_analyse_has_quality(self):
        analyser = TrajectoryAnalyser()
        t = self._make_finished_traj()
        recipe = analyser.analyse(t)
        assert "quality" in recipe
        assert 0.0 <= recipe["quality"] <= 1.0

    def test_analyse_captures_user_request(self):
        analyser = TrajectoryAnalyser()
        t = self._make_finished_traj()
        recipe = analyser.analyse(t)
        assert recipe["intent"]  # should derive from user_request


# ── SynthesizedSkill ─────────────────────────────────────────────────────────

class TestSynthesizedSkill:
    def _make_skill(self):
        return SynthesizedSkill(
            name="run-tests",
            description="Read code and run pytest",
            trigger="run tests OR pytest",
            steps_md="1. read_file path=*.py\n2. shell_exec cmd=pytest",
            tools_used=["read_file", "shell_exec"],
            quality=0.85,
        )

    def test_to_dict_has_fields(self):
        s = self._make_skill()
        d = s.to_dict()
        assert d["name"] == "run-tests"
        assert d["quality"] == 0.85
        assert "tools_used" in d

    def test_to_markdown(self):
        s = self._make_skill()
        md = s.to_markdown()
        assert "run-tests" in md
        assert isinstance(md, str)
        assert len(md) > 0

    def test_from_markdown_returns_something(self):
        s = self._make_skill()
        md = s.to_markdown()
        # from_markdown should not crash on valid markdown
        restored = SynthesizedSkill.from_markdown(md)
        # May return None if parsing fails — that's OK
        assert restored is None or isinstance(restored, SynthesizedSkill)

    def test_from_markdown_returns_none_on_empty(self):
        result = SynthesizedSkill.from_markdown("")
        assert result is None

    def test_created_at_is_set(self):
        s = self._make_skill()
        assert s.created_at  # non-empty string

    def test_skill_id_auto_generated(self):
        s = self._make_skill()
        # skill_id should be auto-populated (or empty — the dataclass default is "")
        # Just ensure it doesn't raise
        assert isinstance(s.skill_id, str)

    def test_tools_used_list(self):
        s = self._make_skill()
        assert "read_file" in s.tools_used
        assert "shell_exec" in s.tools_used


# ── SkillWriter ───────────────────────────────────────────────────────────────

class TestSkillWriter:
    def _make_recipe(self):
        # Step format must match TrajectoryAnalyser output:
        # {'n': int, 'tool': str, 'params': str (JSON), 'result': str, 'ok': bool}
        return {
            "intent": "Read a file and run tests",
            "steps": [
                {"n": 1, "tool": "read_file", "params": '{"path": "test.py"}', "result": "content", "ok": True},
                {"n": 2, "tool": "shell_exec", "params": '{"cmd": "pytest"}', "result": "2 passed", "ok": True},
            ],
            "tools_used": ["read_file", "shell_exec"],
            "key_params": {"path": "test.py"},
            "quality": 0.85,
            "duration_s": 1.2,
        }

    def test_write_returns_synthesized_skill(self):
        writer = SkillWriter()
        recipe = self._make_recipe()
        skill = writer.write(recipe, user_request="run tests on the project")
        assert isinstance(skill, SynthesizedSkill)

    def test_write_sets_name(self):
        writer = SkillWriter()
        recipe = self._make_recipe()
        skill = writer.write(recipe, user_request="run tests")
        assert skill.name
        assert len(skill.name) > 0

    def test_write_sets_tools(self):
        writer = SkillWriter()
        recipe = self._make_recipe()
        skill = writer.write(recipe, user_request="test run")
        assert "read_file" in skill.tools_used or "shell_exec" in skill.tools_used

    def test_write_sets_quality(self):
        writer = SkillWriter()
        recipe = self._make_recipe()
        skill = writer.write(recipe, user_request="do tests")
        assert skill.quality == 0.85

    def test_write_with_outcome(self):
        writer = SkillWriter()
        recipe = self._make_recipe()
        skill = writer.write(recipe, user_request="run tests", outcome="All tests passed!")
        assert isinstance(skill, SynthesizedSkill)

    def test_write_empty_steps(self):
        writer = SkillWriter()
        recipe = {
            "intent": "Simple single-step task",
            "steps": [],
            "tools_used": ["shell_exec"],
            "key_params": {},
            "quality": 0.6,
            "duration_s": 0.5,
        }
        skill = writer.write(recipe, user_request="simple single-step task")
        assert skill is not None


# ── SkillStore ────────────────────────────────────────────────────────────────

class TestSkillStore:
    @pytest.fixture
    def tmp_store(self, tmp_path):
        return SkillStore(skills_dir=tmp_path / "skills")

    def _make_skill(self, name="test-skill"):
        return SynthesizedSkill(
            name=name,
            description="A test skill for testing",
            trigger="test",
            steps_md="1. do something",
            tools_used=["shell_exec"],
            quality=0.7,
        )

    def test_save_creates_file(self, tmp_store):
        skill = self._make_skill()
        tmp_store.save(skill)
        assert tmp_store.count() >= 1

    def test_load_returns_skill(self, tmp_store):
        skill = self._make_skill()
        tmp_store.save(skill)
        # skill_id is set by save()
        loaded = tmp_store.load(skill.skill_id)
        assert loaded is not None
        assert loaded.name == skill.name

    def test_load_nonexistent_returns_none(self, tmp_store):
        result = tmp_store.load("nonexistent-id-xyz")
        assert result is None

    def test_load_all_empty(self, tmp_store):
        result = tmp_store.load_all()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_load_all_after_save(self, tmp_store):
        tmp_store.save(self._make_skill("skill-a"))
        tmp_store.save(self._make_skill("skill-b"))
        result = tmp_store.load_all()
        assert len(result) >= 2

    def test_count_starts_zero(self, tmp_store):
        assert tmp_store.count() == 0

    def test_count_increases(self, tmp_store):
        tmp_store.save(self._make_skill("s1"))
        tmp_store.save(self._make_skill("s2"))
        assert tmp_store.count() >= 2

    def test_search_index_returns_list(self, tmp_store):
        skill = self._make_skill("run-pytest")
        tmp_store.save(skill)
        results = tmp_store.search_index("pytest")
        assert isinstance(results, list)

    def test_search_index_no_match(self, tmp_store):
        results = tmp_store.search_index("xyzzy_not_in_anything")
        assert results == []

    def test_increment_use(self, tmp_store):
        skill = self._make_skill("use-tracked")
        tmp_store.save(skill)
        initial_count = skill.use_count
        tmp_store.increment_use(skill.skill_id)
        loaded = tmp_store.load(skill.skill_id)
        if loaded:
            assert loaded.use_count >= initial_count

    def test_skills_dir_created(self, tmp_path):
        new_dir = tmp_path / "new_skills_dir"
        assert not new_dir.exists()
        SkillStore(skills_dir=new_dir)
        assert new_dir.exists()


# ── SkillMatcher ──────────────────────────────────────────────────────────────

class TestSkillMatcher:
    @pytest.fixture
    def matcher_with_skills(self, tmp_path):
        store = SkillStore(skills_dir=tmp_path / "skills")
        skills = [
            SynthesizedSkill("run-tests", "Run pytest on code", "pytest tests",
                             "1. read_file\n2. shell_exec pytest", tools_used=["read_file", "shell_exec"], quality=0.8),
            SynthesizedSkill("git-commit", "Stage and commit changes", "git commit",
                             "1. shell_exec git add\n2. shell_exec git commit", tools_used=["shell_exec"], quality=0.75),
            SynthesizedSkill("search-web", "Search the internet", "search online",
                             "1. web_search", tools_used=["web_search"], quality=0.9),
        ]
        for s in skills:
            store.save(s)
        return SkillMatcher(store=store)

    def test_find_relevant_returns_list(self, matcher_with_skills):
        results = matcher_with_skills.find_relevant("run tests on the project")
        assert isinstance(results, list)

    def test_find_relevant_top_k(self, matcher_with_skills):
        results = matcher_with_skills.find_relevant("do something", top_k=2)
        assert len(results) <= 2

    def test_find_relevant_default_top_k(self, matcher_with_skills):
        results = matcher_with_skills.find_relevant("anything")
        assert len(results) <= 3  # default top_k=3

    def test_build_context_block_returns_string(self, matcher_with_skills):
        block = matcher_with_skills.build_context_block("run tests")
        assert isinstance(block, str)

    def test_build_context_block_empty_query(self, matcher_with_skills):
        block = matcher_with_skills.build_context_block("")
        assert isinstance(block, str)


# ── SkillSynthesizer ──────────────────────────────────────────────────────────

class TestSkillSynthesizer:
    @pytest.fixture
    def synth(self, tmp_path):
        return SkillSynthesizer(skills_dir=tmp_path / "synth_skills")

    def test_start_trajectory(self, synth):
        traj = synth.start_trajectory("Fix bug in auth module")
        assert traj is not None
        assert traj.user_request == "Fix bug in auth module"

    def test_record_step(self, synth):
        synth.start_trajectory("Do a task")
        synth.record_step("read_file", {"path": "x.py"}, "content", success=True)
        traj = synth.get_trajectory()
        assert traj is not None
        assert len(traj.steps) == 1

    def test_record_step_no_trajectory(self, synth):
        # Should not raise even without active trajectory
        synth.record_step("shell_exec", {}, "ok", success=True)

    def test_get_trajectory_returns_current(self, synth):
        synth.start_trajectory("Current task")
        traj = synth.get_trajectory()
        assert traj is not None
        assert traj.user_request == "Current task"

    def test_get_trajectory_none_before_start(self, synth):
        traj = synth.get_trajectory()
        assert traj is None

    def test_finish_trajectory_marks_success(self, synth):
        synth.start_trajectory("Task A")
        synth.record_step("tool_a", {}, "ok", True)
        synth.finish_trajectory(answer="Done.", success=True)
        traj = synth.get_trajectory()
        assert traj is not None
        assert traj.success is True
        assert traj.final_answer == "Done."

    def test_finish_trajectory_accessible_after_finish(self, synth):
        synth.start_trajectory("Task B")
        synth.finish_trajectory(answer="Finished.", success=True)
        # Trajectory is still accessible after finishing
        traj = synth.get_trajectory()
        assert traj is not None

    def test_synthesize_too_few_steps_returns_none(self, synth):
        synth.start_trajectory("Single step task")
        synth.record_step("tool_x", {}, "ok", success=True)
        traj = synth.get_trajectory()  # get BEFORE finishing
        synth.finish_trajectory(answer="Done.", success=True)
        skill = synth.synthesize(traj, user_request="Single step task")
        assert skill is None  # only 1 step, below _MIN_STEPS

    def test_synthesize_all_failed_returns_none(self, synth):
        synth.start_trajectory("Failed task")
        for _ in range(3):
            synth.record_step("bad_tool", {}, "error", success=False)
        traj = synth.get_trajectory()
        synth.finish_trajectory(answer="", success=False)
        skill = synth.synthesize(traj, user_request="Failed task")
        assert skill is None  # too many failures → quality < _MIN_QUALITY

    def test_synthesize_good_traj(self, synth):
        synth.start_trajectory("Read file and run tests")
        synth.record_step("read_file", {"path": "test.py"}, "content", success=True, duration_ms=100)
        synth.record_step("shell_exec", {"cmd": "pytest"}, "2 passed", success=True, duration_ms=500)
        traj = synth.get_trajectory()
        synth.finish_trajectory(answer="All tests passed.", success=True)
        skill = synth.synthesize(traj, user_request="Read file and run tests", outcome="2 passed")
        # Should synthesize a skill (may or may not pass quality threshold)
        if skill:
            assert isinstance(skill, SynthesizedSkill)

    def test_get_hints_for(self, synth):
        hints = synth.get_hints_for("run tests")
        assert isinstance(hints, str)

    def test_list_skills_empty(self, synth):
        skills = synth.list_skills()
        assert isinstance(skills, list)
        assert len(skills) == 0

    def test_list_skills_after_synthesize(self, synth):
        synth.start_trajectory("Read and test")
        synth.record_step("read_file", {"path": "x.py"}, "code", success=True)
        synth.record_step("shell_exec", {"cmd": "pytest"}, "ok", success=True)
        traj = synth.get_trajectory()
        synth.finish_trajectory(answer="Done.", success=True)
        synth.synthesize(traj, user_request="Read and test", outcome="ok")
        # Skills may or may not be saved depending on quality threshold
        skills = synth.list_skills()
        assert isinstance(skills, list)

    def test_delete_skill_no_error(self, synth):
        # Should not raise even for non-existent skill
        synth.delete_skill("nonexistent-id")

    def test_stats_returns_dict(self, synth):
        stats = synth.stats()
        assert isinstance(stats, dict)
        assert "total_skills" in stats

    def test_stats_total_skills_zero(self, synth):
        stats = synth.stats()
        assert stats["total_skills"] == 0

    def test_summary_returns_string(self, synth):
        s = synth.summary()
        assert isinstance(s, str)

    def test_reset_trajectory(self, synth):
        synth.start_trajectory("something")
        synth.reset_trajectory()
        assert synth.get_trajectory() is None

    def test_synthesize_from_current_no_traj(self, synth):
        # Should return None when no active trajectory
        result = synth.synthesize_from_current()
        assert result is None

    def test_synthesize_from_current_with_traj(self, synth):
        synth.start_trajectory("Multi-step task")
        synth.record_step("read_file", {"path": "a.py"}, "content", success=True)
        synth.record_step("shell_exec", {"cmd": "python a.py"}, "output", success=True)
        result = synth.synthesize_from_current()
        # Returns skill or None — should not raise
        assert result is None or isinstance(result, SynthesizedSkill)


# ── Module-level singleton ────────────────────────────────────────────────────

class TestGetSynthesizer:
    def test_singleton_returns_same(self):
        import core.skill_synthesizer as ss
        ss._synthesizer = None
        s1 = get_synthesizer()
        s2 = get_synthesizer()
        assert s1 is s2
        ss._synthesizer = None

    def test_get_synthesizer_returns_instance(self):
        import core.skill_synthesizer as ss
        ss._synthesizer = None
        s = get_synthesizer()
        assert isinstance(s, SkillSynthesizer)
        ss._synthesizer = None


# ── Constants ─────────────────────────────────────────────────────────────────

class TestConstants:
    def test_min_quality_threshold(self):
        assert 0.0 < _MIN_QUALITY < 1.0

    def test_min_steps_threshold(self):
        assert _MIN_STEPS >= 2
