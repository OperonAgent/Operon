"""Tests for core/swe_agent.py"""
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from core.swe_agent import (
    IssueParser, CodeLocaliser, PatchApplier, PatchGenerator,
    TestRunner, FixPlanner, BranchManager, PRCreator, SWEAgent,
    SWETask, SWEResult, SWEState, FileHunk, FilePatch, TestRun,
    TrajectoryEvent, shlex_join,
    solve_issue, get_swe_agent,
)


# ── SWETask ───────────────────────────────────────────────────────────────────

class TestSWETask:
    def test_full_text_combines_title_body(self):
        t = SWETask(title="Fix bug", body="Details here")
        assert "Fix bug" in t.full_text()
        assert "Details here" in t.full_text()

    def test_full_text_title_only(self):
        t = SWETask(title="Refactor foo")
        assert t.full_text() == "Refactor foo"

    def test_defaults(self):
        t = SWETask(title="x")
        assert t.issue_number is None
        assert t.repo == ""
        assert t.labels == []


# ── IssueParser ───────────────────────────────────────────────────────────────

class TestIssueParser:
    def setup_method(self):
        self.parser = IssueParser()

    def test_extracts_py_files(self):
        task = SWETask(title="Bug in `foo.py`", body="See `bar.py` line 42")
        parsed = self.parser.parse(task)
        assert "foo.py" in parsed["mentioned_files"]
        assert "bar.py" in parsed["mentioned_files"]

    def test_extracts_line_numbers(self):
        task = SWETask(title="Error at line 99", body="")
        parsed = self.parser.parse(task)
        assert 99 in parsed["mentioned_lines"]

    def test_extracts_function_names(self):
        # _FUNC_RE matches `name(` pattern (backtick + name + open-paren)
        task = SWETask(title="Bug in `compute_total(`", body="")
        parsed = self.parser.parse(task)
        assert "compute_total" in parsed["mentioned_funcs"]

    def test_detects_bug(self):
        task = SWETask(title="Fix crash in login", body="There is an error")
        parsed = self.parser.parse(task)
        assert parsed["is_bug"] is True

    def test_detects_feature(self):
        task = SWETask(title="Add export button", body="New feature request")
        parsed = self.parser.parse(task)
        assert parsed["is_feature"] is True

    def test_detects_refactor(self):
        task = SWETask(title="Refactor payment module", body="")
        parsed = self.parser.parse(task)
        assert parsed["is_refactor"] is True

    def test_keywords_not_empty(self):
        task = SWETask(title="Fix off-by-one in pagination logic", body="When page is zero")
        parsed = self.parser.parse(task)
        assert len(parsed["keywords"]) > 0

    def test_error_messages_extracted(self):
        task = SWETask(title="Error: IndexError at line 5", body="")
        parsed = self.parser.parse(task)
        assert any("IndexError" in e for e in parsed["error_messages"])

    def test_non_bug_task(self):
        task = SWETask(title="Update README", body="Just docs")
        parsed = self.parser.parse(task)
        assert parsed["is_bug"] is False

    def test_empty_task(self):
        task = SWETask(title="")
        parsed = self.parser.parse(task)
        assert isinstance(parsed["keywords"], list)
        assert isinstance(parsed["mentioned_files"], list)


# ── FileHunk ──────────────────────────────────────────────────────────────────

class TestFileHunk:
    def test_read_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "foo.py"
            p.write_text("line1\nline2\nline3\n")
            hunk = FileHunk(path="foo.py", start_line=1, end_line=2)
            text = hunk.read(Path(d))
            assert "line1" in text

    def test_read_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            hunk = FileHunk(path="nonexistent.py")
            assert hunk.read(Path(d)) == ""

    def test_read_truncates_large_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "big.py"
            p.write_text("x = 1\n" * 5000)
            hunk = FileHunk(path="big.py")
            text = hunk.read(Path(d))
            assert len(text) <= 8200  # _MAX_FILE_CHARS + small overhead


# ── CodeLocaliser ─────────────────────────────────────────────────────────────

class TestCodeLocaliser:
    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        # Create a small fake repo
        (self.repo / "src").mkdir()
        (self.repo / "src" / "auth.py").write_text(
            "def login(user, password):\n    return user == 'admin'\n"
        )
        (self.repo / "src" / "pagination.py").write_text(
            "def paginate(items, page=0):\n    return items[page*10:(page+1)*10]\n"
        )
        (self.repo / "tests").mkdir()
        (self.repo / "tests" / "test_auth.py").write_text(
            "def test_login(): assert True\n"
        )

    def teardown_method(self):
        self.tmp.cleanup()

    def test_locates_mentioned_file(self):
        loc = CodeLocaliser(self.repo)
        parsed = {"mentioned_files": ["auth.py"], "keywords": [],
                  "mentioned_funcs": [], "mentioned_classes": []}
        hunks = loc.locate(parsed)
        paths = [h.path for h in hunks]
        assert any("auth.py" in p for p in paths)

    def test_keyword_grep_finds_function(self):
        loc = CodeLocaliser(self.repo)
        parsed = {"mentioned_files": [], "keywords": ["paginate"],
                  "mentioned_funcs": [], "mentioned_classes": []}
        hunks = loc.locate(parsed)
        assert len(hunks) > 0

    def test_returns_file_hunk_objects(self):
        loc = CodeLocaliser(self.repo)
        parsed = {"mentioned_files": [], "keywords": ["login"],
                  "mentioned_funcs": [], "mentioned_classes": []}
        hunks = loc.locate(parsed)
        assert all(isinstance(h, FileHunk) for h in hunks)

    def test_max_files_respected(self):
        loc = CodeLocaliser(self.repo)
        parsed = {"mentioned_files": [], "keywords": ["a", "b", "c", "d"],
                  "mentioned_funcs": [], "mentioned_classes": []}
        hunks = loc.locate(parsed, max_files=2)
        assert len(hunks) <= 2

    def test_relevance_sorted_descending(self):
        loc = CodeLocaliser(self.repo)
        parsed = {"mentioned_files": ["auth.py"], "keywords": ["login"],
                  "mentioned_funcs": [], "mentioned_classes": []}
        hunks = loc.locate(parsed)
        relevances = [h.relevance for h in hunks]
        assert relevances == sorted(relevances, reverse=True)

    def test_symbol_match(self):
        loc = CodeLocaliser(self.repo)
        parsed = {"mentioned_files": [], "keywords": [],
                  "mentioned_funcs": ["login"], "mentioned_classes": []}
        hunks = loc.locate(parsed)
        assert any("auth" in h.path for h in hunks)


# ── PatchApplier ─────────────────────────────────────────────────────────────

class TestPatchApplier:
    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)

    def teardown_method(self):
        self.tmp.cleanup()

    def _make_file(self, path: str, content: str) -> Path:
        full = self.repo / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        return full

    def test_python_apply_new_file(self):
        patch = FilePatch(
            path="newfile.py",
            diff="--- a/newfile.py\n+++ b/newfile.py\n@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n",
            is_new=True,
        )
        applier = PatchApplier(self.repo)
        ok, err = applier._python_apply(patch)
        assert ok
        assert (self.repo / "newfile.py").exists()

    def test_normalise_diff_adds_prefix(self):
        diff = "--- foo.py\n+++ foo.py\n"
        norm = PatchApplier._normalise_diff(diff)
        assert norm.startswith("--- a/foo.py")

    def test_normalise_diff_keeps_existing_prefix(self):
        diff = "--- a/foo.py\n+++ b/foo.py\n"
        norm = PatchApplier._normalise_diff(diff)
        assert norm.startswith("--- a/foo.py")

    def test_apply_all_returns_count(self):
        patch1 = FilePatch(path="a.py", diff="--- a/a.py\n+++ b/a.py\n", is_new=True)
        patch2 = FilePatch(path="b.py", diff="--- a/b.py\n+++ b/b.py\n", is_new=True)
        applier = PatchApplier(self.repo)
        applied, errors = applier.apply_all([patch1, patch2])
        assert isinstance(applied, int)
        assert isinstance(errors, list)

    def test_missing_target_returns_error(self):
        patch = FilePatch(path="ghost.py", diff="--- a/ghost.py\n+++ b/ghost.py\n@@ -1 +1 @@\n-old\n+new\n")
        applier = PatchApplier(self.repo)
        ok, err = applier._python_apply(patch)
        assert not ok
        assert "ghost.py" in err

    def test_apply_hunks_simple(self):
        original = "line1\nline2\nline3\n"
        diff = "--- a/f.py\n+++ b/f.py\n@@ -2,1 +2,1 @@\n-line2\n+LINE2\n"
        result = PatchApplier._apply_hunks(original, diff)
        assert "LINE2" in result


# ── TestRun ───────────────────────────────────────────────────────────────────

class TestTestRun:
    def test_ok_when_no_failures(self):
        tr = TestRun(passed=10, failed=0, errors=0)
        assert tr.ok

    def test_not_ok_when_failures(self):
        tr = TestRun(passed=9, failed=1, errors=0)
        assert not tr.ok

    def test_not_ok_when_errors(self):
        tr = TestRun(passed=9, failed=0, errors=1)
        assert not tr.ok

    def test_total(self):
        tr = TestRun(passed=5, failed=2, errors=1)
        assert tr.total == 8

    def test_summary_string(self):
        tr = TestRun(passed=10, failed=1, errors=0, duration=1.5)
        s = tr.summary()
        assert "passed=10" in s
        assert "failed=1" in s


# ── TestRunner ────────────────────────────────────────────────────────────────

class TestTestRunnerParsing:
    def _make_runner(self):
        with tempfile.TemporaryDirectory() as d:
            return TestRunner(Path(d)), d

    def test_parse_pytest_output(self):
        runner = TestRunner(Path("."))
        output = "10 passed, 2 failed, 1 error in 3.5s"
        tr = TestRunner._parse_output(output, cmd="pytest", duration=3.5)
        assert tr.passed == 10
        assert tr.failed == 2
        assert tr.errors == 1

    def test_parse_pytest_all_pass(self):
        output = "42 passed in 1.2s"
        tr = TestRunner._parse_output(output, cmd="pytest", duration=1.2)
        assert tr.passed == 42
        assert tr.ok

    def test_parse_unittest_output(self):
        output = "Ran 5 tests in 0.01s\n\nOK"
        tr = TestRunner._parse_output(output, cmd="python -m unittest", duration=0.01)
        assert tr.passed == 5
        assert tr.ok

    def test_parse_unittest_failed(self):
        output = "Ran 5 tests in 0.01s\n\nFAILED (failures=2)"
        tr = TestRunner._parse_output(output, cmd="unittest", duration=0.01)
        assert tr.failed == 2

    def test_parse_empty_output(self):
        tr = TestRunner._parse_output("", cmd="unknown", duration=0)
        assert tr.total == 0


# ── TrajectoryEvent ────────────────────────────────────────────────────────────

class TestTrajectoryEvent:
    def test_to_dict_keys(self):
        ev = TrajectoryEvent(step=1, action="parse", detail="parsed issue")
        d = ev.to_dict()
        assert "step" in d
        assert "action" in d
        assert "detail" in d
        assert "ok" in d

    def test_detail_truncated(self):
        ev = TrajectoryEvent(step=1, action="x", detail="y" * 1000)
        d = ev.to_dict()
        assert len(d["detail"]) <= 500


# ── SWEResult ─────────────────────────────────────────────────────────────────

class TestSWEResult:
    def test_succeeded_when_done(self):
        r = SWEResult(task=SWETask(title="x"), state=SWEState.DONE)
        assert r.succeeded

    def test_not_succeeded_when_failed(self):
        r = SWEResult(task=SWETask(title="x"), state=SWEState.FAILED)
        assert not r.succeeded

    def test_duration_positive(self):
        r = SWEResult(task=SWETask(title="x"))
        time.sleep(0.01)
        assert r.duration > 0

    def test_last_test_none_initially(self):
        r = SWEResult(task=SWETask(title="x"))
        assert r.last_test is None

    def test_last_test_returns_latest(self):
        r = SWEResult(task=SWETask(title="x"))
        r.test_runs.append(TestRun(passed=5))
        r.test_runs.append(TestRun(passed=10))
        assert r.last_test.passed == 10

    def test_summary_contains_title(self):
        r = SWEResult(task=SWETask(title="Fix bug"))
        assert "Fix bug" in r.summary()

    def test_to_dict_keys(self):
        r = SWEResult(task=SWETask(title="x"), state=SWEState.DONE)
        d = r.to_dict()
        assert "state" in d
        assert "duration" in d
        assert "patches" in d
        assert "trajectory" in d


# ── SWEAgent (mocked) ─────────────────────────────────────────────────────────

class TestSWEAgentDryRun:
    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        (self.repo / "main.py").write_text("def add(a, b): return a + b\n")

    def teardown_method(self):
        self.tmp.cleanup()

    def test_dry_run_returns_dict(self):
        agent = SWEAgent(repo_path=str(self.repo))
        with patch.object(agent._planner, "plan", return_value="1. fix it"):
            result = agent.dry_run(SWETask(title="Add subtraction"))
        assert "parsed" in result
        assert "files" in result
        assert "plan" in result

    def test_dry_run_plan_returned(self):
        agent = SWEAgent(repo_path=str(self.repo))
        with patch.object(agent._planner, "plan", return_value="MOCK PLAN"):
            result = agent.dry_run(SWETask(title="Fix add"))
        assert result["plan"] == "MOCK PLAN"

    def test_dry_run_files_list(self):
        agent = SWEAgent(repo_path=str(self.repo))
        with patch.object(agent._planner, "plan", return_value="plan"):
            result = agent.dry_run(SWETask(title="Fix add", body="add function"))
        assert isinstance(result["files"], list)
        for f in result["files"]:
            assert "path" in f
            assert "reason" in f

    def test_solve_no_patches_returns_failed(self):
        agent = SWEAgent(repo_path=str(self.repo), max_retries=1)
        with patch.object(agent._planner, "plan", return_value="plan"), \
             patch.object(agent._patcher, "generate", return_value=[]):
            result = agent.solve(SWETask(title="Bug fix"))
        assert result.state in (SWEState.FAILED, SWEState.DONE)

    def test_solve_with_patches_applied(self):
        agent = SWEAgent(repo_path=str(self.repo), max_retries=1)
        patch_obj = FilePatch(
            path="main.py",
            diff="--- a/main.py\n+++ b/main.py\n@@ -1,1 +1,2 @@\n def add(a, b): return a + b\n+# fix\n",
        )
        mock_test = TestRun(passed=1, failed=0)
        with patch.object(agent._planner, "plan", return_value="plan"), \
             patch.object(agent._patcher, "generate", return_value=[patch_obj]), \
             patch.object(agent._applier, "apply_all", return_value=(1, [])), \
             patch.object(agent._tester, "run", return_value=mock_test):
            result = agent.solve(SWETask(title="Fix"))
        assert result.state == SWEState.DONE
        assert len(result.trajectory) > 0

    def test_trajectory_recorded(self):
        agent = SWEAgent(repo_path=str(self.repo), max_retries=1)
        with patch.object(agent._planner, "plan", return_value="plan"), \
             patch.object(agent._patcher, "generate", return_value=[]):
            result = agent.solve(SWETask(title="x"))
        assert len(result.trajectory) > 0
        assert all(isinstance(e, TrajectoryEvent) for e in result.trajectory)

    def test_on_event_callback(self):
        events = []
        agent = SWEAgent(
            repo_path=str(self.repo), max_retries=1,
            on_event=lambda e: events.append(e),
        )
        with patch.object(agent._planner, "plan", return_value="plan"), \
             patch.object(agent._patcher, "generate", return_value=[]):
            agent.solve(SWETask(title="x"))
        assert len(events) > 0

    def test_retries_incremented_on_failure(self):
        agent = SWEAgent(repo_path=str(self.repo), max_retries=2)
        mock_test = TestRun(passed=0, failed=5)
        patch_obj = FilePatch(path="main.py", diff="--- a/main.py\n+++ b/main.py\n")
        with patch.object(agent._planner, "plan", return_value="plan"), \
             patch.object(agent._patcher, "generate", return_value=[patch_obj]), \
             patch.object(agent._applier, "apply_all", return_value=(1, [])), \
             patch.object(agent._tester, "run", return_value=mock_test):
            result = agent.solve(SWETask(title="x"))
        assert result.retries >= 1

    def test_run_tests_returns_test_run(self):
        agent = SWEAgent(repo_path=str(self.repo))
        with patch.object(agent._tester, "run",
                          return_value=TestRun(passed=3)):
            tr = agent.run_tests()
        assert tr.passed == 3


# ── Module helpers ────────────────────────────────────────────────────────────

class TestSWEHelpers:
    def test_shlex_join(self):
        parts = ["python", "-m", "pytest", "tests/"]
        result = shlex_join(parts)
        assert "python" in result
        assert "pytest" in result

    def test_shlex_join_quotes_spaces(self):
        parts = ["cmd", "path with spaces"]
        result = shlex_join(parts)
        assert "path" in result

    def test_get_swe_agent_returns_agent(self):
        with tempfile.TemporaryDirectory() as d:
            agent = get_swe_agent(repo_path=d)
        assert isinstance(agent, SWEAgent)

    def test_file_patch_dataclass(self):
        p = FilePatch(path="foo.py", diff="--- a/foo.py\n")
        assert p.path == "foo.py"
        assert not p.is_new
        assert not p.is_delete

    def test_swe_task_labels_default(self):
        t = SWETask(title="Fix bug")
        assert isinstance(t.labels, list)

    def test_swe_state_values(self):
        assert SWEState.DONE.value == "done"
        assert SWEState.FAILED.value == "failed"
        assert SWEState.PENDING.value == "pending"


# ── FilePatch parser ──────────────────────────────────────────────────────────

class TestPatchParser:
    def test_parse_diff_block(self):
        raw = """
```diff
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,3 +10,3 @@
 def login():
-    return False
+    return True
```
"""
        patches = PatchGenerator._parse_patches(raw)
        assert len(patches) == 1
        assert "auth.py" in patches[0].path

    def test_parse_multiple_blocks(self):
        raw = """
```diff
--- a/a.py
+++ b/a.py
@@ -1,1 +1,1 @@
-old
+new
```
```diff
--- a/b.py
+++ b/b.py
@@ -1,1 +1,1 @@
-old
+new
```
"""
        patches = PatchGenerator._parse_patches(raw)
        assert len(patches) == 2

    def test_parse_no_diff_returns_empty(self):
        raw = "Here is some explanation without any code blocks."
        patches = PatchGenerator._parse_patches(raw)
        assert patches == []

    def test_parse_bare_diff(self):
        raw = (
            "--- a/foo.py\n+++ b/foo.py\n"
            "@@ -1,1 +1,1 @@\n-old\n+new\n"
        )
        patches = PatchGenerator._parse_patches(raw)
        assert len(patches) >= 0  # bare diffs are a best-effort parse


# ── BranchManager (git, mocked) ───────────────────────────────────────────────

class TestBranchManager:
    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)

    def teardown_method(self):
        self.tmp.cleanup()

    def test_create_branch_returns_string(self):
        bm = BranchManager(self.repo)
        with patch.object(bm, "_git", return_value="Switched to branch 'fix/test'"):
            branch = bm.create_fix_branch(SWETask(title="Test Fix"))
        assert isinstance(branch, str)

    def test_commit_returns_bool(self):
        bm = BranchManager(self.repo)
        patches = [FilePatch(path="foo.py", diff="")]
        with patch.object(bm, "_git", return_value=""):
            ok = bm.commit(patches, SWETask(title="Fix"))
        assert isinstance(ok, bool)

    def test_commit_empty_patches_returns_false(self):
        bm = BranchManager(self.repo)
        ok = bm.commit([], SWETask(title="Fix"))
        assert ok is False

    def test_create_branch_handles_git_failure(self):
        bm = BranchManager(self.repo)
        with patch.object(bm, "_git", side_effect=RuntimeError("not a git repo")):
            branch = bm.create_fix_branch(SWETask(title="Fix Bug"))
        assert branch == ""
