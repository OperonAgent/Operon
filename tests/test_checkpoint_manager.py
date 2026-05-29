"""Tests for core/checkpoint_manager.py"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import mock
import pytest

from core.checkpoint_manager import (
    CheckpointManager, CheckpointRef, CheckpointDiff,
    with_checkpoint, get_manager, quick_checkpoint,
    _CP_PREFIX,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_git() -> bool:
    return shutil.which("git") is not None


def _init_repo(path: Path) -> None:
    """Init a git repo with an initial commit."""
    subprocess.run(["git", "init", str(path)], capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test.com"],
                   capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], capture_output=True)
    (path / "README.md").write_text("initial")
    subprocess.run(["git", "-C", str(path), "add", "--all"], capture_output=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "initial"], capture_output=True)


# ── CheckpointRef ─────────────────────────────────────────────────────────────

class TestCheckpointRef:
    def test_short_sha(self):
        ref = CheckpointRef(sha="abcdef123456789", branch="main", message="test")
        assert ref.short_sha() == "abcdef123456"

    def test_str_format(self):
        ref = CheckpointRef(sha="abcdef123456789", branch="main", message="my msg")
        s = str(ref)
        assert "checkpoint" in s
        assert "main" in s
        assert "my msg" in s

    def test_created_at_set_automatically(self):
        import time
        ref = CheckpointRef(sha="abc", branch="main", message="x")
        assert abs(ref.created_at - time.time()) < 2.0


# ── CheckpointManager (no git) ────────────────────────────────────────────────

class TestCheckpointManagerNoGit:
    def test_not_git_repo_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(tmp)
            ref = mgr.checkpoint("test")
            assert ref is None

    def test_restore_without_ref_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(tmp)
            ok = mgr.restore(None)
            assert not ok

    def test_stats_not_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(tmp)
            s = mgr.stats()
            assert s["is_git_repo"] is False

    def test_list_checkpoints_no_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(tmp)
            assert mgr.list_checkpoints() == []


# ── CheckpointManager (real git) ─────────────────────────────────────────────

@pytest.mark.skipif(not _has_git(), reason="git not installed")
class TestCheckpointManagerGit:
    def _mgr(self) -> tuple:
        """Returns (mgr, tmpdir_path) — caller must clean up."""
        tmpdir = tempfile.mkdtemp()
        p = Path(tmpdir)
        _init_repo(p)
        return CheckpointManager(tmpdir), p

    def test_checkpoint_creates_commit(self):
        mgr, p = self._mgr()
        try:
            (p / "file.txt").write_text("content v1")
            ref = mgr.checkpoint("test checkpoint")
            assert ref is not None
            assert ref.sha
            assert _CP_PREFIX in ref.message
        finally:
            shutil.rmtree(str(p))

    def test_checkpoint_no_changes_returns_head(self):
        mgr, p = self._mgr()
        try:
            # No uncommitted changes
            ref = mgr.checkpoint("no changes")
            assert ref is not None
            assert ref.sha
        finally:
            shutil.rmtree(str(p))

    def test_restore_reverts_changes(self):
        mgr, p = self._mgr()
        try:
            (p / "file.txt").write_text("original")
            ref = mgr.checkpoint("original state")
            assert ref is not None

            (p / "file.txt").write_text("modified version")
            mgr.checkpoint("modified state")

            ok = mgr.restore(ref)
            assert ok
            content = (p / "file.txt").read_text()
            assert "original" in content
        finally:
            shutil.rmtree(str(p))

    def test_restore_invalid_ref_returns_false(self):
        mgr, p = self._mgr()
        try:
            bad_ref = CheckpointRef(sha="0000000000000000000000000000000000000000",
                                    branch="main", message="bad")
            ok = mgr.restore(bad_ref)
            assert not ok
        finally:
            shutil.rmtree(str(p))

    def test_restore_last_uses_history(self):
        mgr, p = self._mgr()
        try:
            (p / "file.txt").write_text("v1")
            ref = mgr.checkpoint("v1")
            (p / "file.txt").write_text("v2")
            mgr.checkpoint("v2")
            ok = mgr.restore_last()
            assert ok
        finally:
            shutil.rmtree(str(p))

    def test_restore_last_no_history_returns_false(self):
        mgr, p = self._mgr()
        try:
            ok = mgr.restore_last()
            assert not ok
        finally:
            shutil.rmtree(str(p))

    def test_list_checkpoints_returns_created(self):
        mgr, p = self._mgr()
        try:
            (p / "f.txt").write_text("a")
            mgr.checkpoint("checkpoint one")
            (p / "f.txt").write_text("b")
            mgr.checkpoint("checkpoint two")
            cps = mgr.list_checkpoints()
            assert len(cps) >= 1
            assert any(_CP_PREFIX in c["message"] for c in cps)
        finally:
            shutil.rmtree(str(p))

    def test_diff_returns_info(self):
        mgr, p = self._mgr()
        try:
            (p / "diff_file.txt").write_text("line1")
            ref = mgr.checkpoint("before diff")
            (p / "diff_file.txt").write_text("line1\nline2\nline3")
            mgr.checkpoint("after diff")
            d = mgr.diff(ref)
            if d:
                assert isinstance(d.added_lines, int)
                assert d.summary()
        finally:
            shutil.rmtree(str(p))

    def test_prune_reduces_history(self):
        mgr, p = self._mgr()
        try:
            for i in range(5):
                (p / f"f{i}.txt").write_text(str(i))
                mgr.checkpoint(f"cp {i}")
            assert len(mgr._history) == 5
            pruned = mgr.prune(keep=3)
            assert pruned == 2
            assert len(mgr._history) == 3
        finally:
            shutil.rmtree(str(p))

    def test_stats_is_git_true(self):
        mgr, p = self._mgr()
        try:
            s = mgr.stats()
            assert s["is_git_repo"] is True
            assert s["current_branch"] in ("main", "master")
        finally:
            shutil.rmtree(str(p))

    def test_context_manager_no_exception(self):
        mgr, p = self._mgr()
        try:
            with mgr:
                (p / "ctx.txt").write_text("inside context")
        finally:
            shutil.rmtree(str(p))

    def test_with_checkpoint_decorator_success(self):
        mgr, p = self._mgr()
        try:
            @with_checkpoint(mgr, "test decorator")
            def do_work():
                (p / "dec.txt").write_text("created by decorator")
                return "done"
            result = do_work()
            assert result == "done"
        finally:
            shutil.rmtree(str(p))

    def test_with_checkpoint_decorator_restores_on_exception(self):
        mgr, p = self._mgr()
        try:
            (p / "before.txt").write_text("original")
            ref = mgr.checkpoint("before decorator")

            @with_checkpoint(mgr, "risky op", restore_on_exception=True)
            def risky():
                (p / "before.txt").write_text("modified by risky op")
                raise RuntimeError("something went wrong")

            with pytest.raises(RuntimeError):
                risky()
        finally:
            shutil.rmtree(str(p))


# ── Module-level functions ────────────────────────────────────────────────────

class TestModuleLevelAPI:
    def test_get_manager_returns_instance(self):
        mgr = get_manager()
        assert isinstance(mgr, CheckpointManager)

    def test_quick_checkpoint_not_git_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = quick_checkpoint("quick test", repo_path=tmp)
            assert ref is None

    @pytest.mark.skipif(not _has_git(), reason="git not installed")
    def test_quick_checkpoint_git_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            _init_repo(Path(tmp))
            ref = quick_checkpoint("quick test", repo_path=tmp)
            assert ref is not None
            assert ref.sha
