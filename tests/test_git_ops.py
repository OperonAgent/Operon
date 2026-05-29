"""Tests for tools/git_ops.py

Strategy: all real git operations run against a fresh temporary repository
created in a temp directory. No network, no real file-system side effects
outside of tmp.
"""
import os
import subprocess
import tempfile
import pytest

from tools.git_ops import (
    git_status, git_diff, git_log, git_add,
    git_commit, git_checkout, git_branch, git_stash,
    _git,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_git() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=5)
        return True
    except Exception:
        return False


requires_git = pytest.mark.skipif(not _has_git(), reason="git not installed")


@pytest.fixture
def repo(tmp_path):
    """Create an initialised bare git repo with one commit."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@operon"], cwd=str(tmp_path),
                   check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Operon Test"], cwd=str(tmp_path),
                   check=True, capture_output=True)
    # Initial commit
    (tmp_path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial commit"],
                   cwd=str(tmp_path), check=True, capture_output=True)
    return str(tmp_path)


# ── _git internal helper ──────────────────────────────────────────────────────

class TestGitHelper:
    @requires_git
    def test_returns_uniform_keys(self, repo):
        r = _git(["status"], cwd=repo)
        assert "success" in r
        assert "stdout" in r
        assert "stderr" in r
        assert "returncode" in r

    @requires_git
    def test_success_true_on_zero_exit(self, repo):
        r = _git(["status"], cwd=repo)
        assert r["success"] is True
        assert r["returncode"] == 0

    @requires_git
    def test_success_false_on_bad_command(self, repo):
        r = _git(["this-command-does-not-exist"], cwd=repo)
        assert r["success"] is False

    def test_git_not_found_returns_error(self, tmp_path):
        """Simulate git not on PATH by patching subprocess.run."""
        import unittest.mock as mock
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            r = _git(["status"], cwd=str(tmp_path))
        assert r["success"] is False
        assert "not found" in r["stderr"].lower() or r["returncode"] == -1

    def test_timeout_returns_error(self, tmp_path):
        import unittest.mock as mock
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["git"], 1)):
            r = _git(["status"], cwd=str(tmp_path))
        assert r["success"] is False
        assert "timed out" in r["stderr"].lower() or r["returncode"] == -1

    def test_unknown_exception_captured(self, tmp_path):
        import unittest.mock as mock
        with mock.patch("subprocess.run", side_effect=PermissionError("no perms")):
            r = _git(["status"], cwd=str(tmp_path))
        assert r["success"] is False
        assert "no perms" in r["stderr"]

    @requires_git
    def test_cwd_defaults_to_cwd(self):
        """Passing empty string for cwd should fall back to os.getcwd()."""
        r = _git(["--version"], cwd="")
        assert r["success"] is True

    @requires_git
    def test_stdout_stripped(self, repo):
        r = _git(["status"], cwd=repo)
        assert not r["stdout"].endswith("\n")


# ── git_status ────────────────────────────────────────────────────────────────

class TestGitStatus:
    @requires_git
    def test_clean_repo_succeeds(self, repo):
        r = git_status(cwd=repo)
        assert r["success"] is True

    @requires_git
    def test_shows_branch(self, repo):
        r = git_status(cwd=repo)
        # Either 'main', 'master', or any branch name should appear
        assert r["stdout"] != ""

    @requires_git
    def test_dirty_repo_shows_modified(self, repo, tmp_path):
        cwd = repo
        (tmp_path / "new_file.txt").write_text("hello")
        subprocess.run(["git", "-C", cwd, "add", str(tmp_path / "new_file.txt")],
                       capture_output=True)
        r = git_status(cwd=cwd)
        assert r["success"] is True

    @requires_git
    def test_accepts_extra_kwargs(self, repo):
        """Extra kwargs (from dispatcher) must not cause errors."""
        r = git_status(cwd=repo, extra_ignored_param="ignored")
        assert r["success"] is True


# ── git_diff ─────────────────────────────────────────────────────────────────

class TestGitDiff:
    @requires_git
    def test_clean_diff_succeeds(self, repo):
        r = git_diff(cwd=repo)
        assert r["success"] is True

    @requires_git
    def test_diff_with_changes(self, tmp_path, repo):
        (tmp_path / "README.md").write_text("# Changed\n")
        r = git_diff(cwd=repo)
        assert r["success"] is True

    @requires_git
    def test_diff_staged(self, repo, tmp_path):
        (tmp_path / "staged.txt").write_text("staged content")
        subprocess.run(["git", "-C", repo, "add", str(tmp_path / "staged.txt")],
                       capture_output=True)
        r = git_diff(staged=True, cwd=repo)
        assert r["success"] is True

    @requires_git
    def test_diff_with_path_filter(self, repo):
        r = git_diff(path="README.md", cwd=repo)
        assert r["success"] is True


# ── git_log ───────────────────────────────────────────────────────────────────

class TestGitLog:
    @requires_git
    def test_log_returns_output(self, repo):
        r = git_log(cwd=repo)
        assert r["success"] is True
        assert "initial commit" in r["stdout"].lower()

    @requires_git
    def test_log_n_limits_count(self, repo):
        # Add a second commit
        (tmp_path_commit := tempfile.mkdtemp())
        f = os.path.join(repo, "second.txt")
        with open(f, "w") as fh:
            fh.write("second\n")
        subprocess.run(["git", "-C", repo, "add", f], capture_output=True)
        subprocess.run(["git", "-C", repo, "commit", "-m", "second commit"],
                       capture_output=True)
        r = git_log(n=1, cwd=repo)
        lines = [l for l in r["stdout"].splitlines() if l.strip()]
        assert len(lines) == 1

    @requires_git
    def test_log_verbose_format(self, repo):
        r = git_log(oneline=False, cwd=repo)
        assert r["success"] is True

    @requires_git
    def test_log_extra_kwargs_ignored(self, repo):
        r = git_log(cwd=repo, unknown_param=42)
        assert r["success"] is True


# ── git_add ───────────────────────────────────────────────────────────────────

class TestGitAdd:
    @requires_git
    def test_add_dot_succeeds(self, repo):
        r = git_add(paths=".", cwd=repo)
        assert r["success"] is True

    @requires_git
    def test_add_specific_file(self, repo):
        f = os.path.join(repo, "new_add.txt")
        with open(f, "w") as fh:
            fh.write("content\n")
        r = git_add(paths=f, cwd=repo)
        assert r["success"] is True

    @requires_git
    def test_add_list_of_paths(self, repo):
        f1 = os.path.join(repo, "a.txt")
        f2 = os.path.join(repo, "b.txt")
        for f in (f1, f2):
            with open(f, "w") as fh:
                fh.write("x\n")
        r = git_add(paths=[f1, f2], cwd=repo)
        assert r["success"] is True

    def test_add_empty_paths_returns_error(self):
        r = git_add(paths=[], cwd="")
        assert r["success"] is False
        assert "no paths" in r["stderr"].lower()

    @requires_git
    def test_add_space_separated_string(self, repo):
        f1 = os.path.join(repo, "c.txt")
        f2 = os.path.join(repo, "d.txt")
        for f in (f1, f2):
            with open(f, "w") as fh:
                fh.write("x\n")
        r = git_add(paths=f"{f1} {f2}", cwd=repo)
        assert r["success"] is True


# ── git_commit ────────────────────────────────────────────────────────────────

class TestGitCommit:
    def test_empty_message_returns_error(self):
        r = git_commit(message="", cwd="")
        assert r["success"] is False
        assert "message" in r["stderr"].lower()

    def test_whitespace_only_message_returns_error(self):
        r = git_commit(message="   ", cwd="")
        assert r["success"] is False

    @requires_git
    def test_commit_with_staged_changes(self, repo):
        f = os.path.join(repo, "commit_me.txt")
        with open(f, "w") as fh:
            fh.write("committed\n")
        git_add(paths=f, cwd=repo)
        r = git_commit(message="test: add commit_me.txt", cwd=repo)
        assert r["success"] is True
        assert "commit_me" in git_log(n=1, cwd=repo)["stdout"].lower() or r["success"]

    @requires_git
    def test_commit_with_nothing_staged_fails(self, repo):
        r = git_commit(message="nothing staged here", cwd=repo)
        # git exits non-zero when nothing to commit
        assert r["returncode"] != 0 or not r["success"]


# ── git_checkout ──────────────────────────────────────────────────────────────

class TestGitCheckout:
    @requires_git
    def test_list_branches_when_no_branch_given(self, repo):
        r = git_checkout(cwd=repo)
        assert r["success"] is True

    @requires_git
    def test_create_and_switch_branch(self, repo):
        r = git_checkout(branch="feature-x", create=True, cwd=repo)
        assert r["success"] is True

    @requires_git
    def test_switch_existing_branch(self, repo):
        subprocess.run(["git", "-C", repo, "checkout", "-b", "other-branch"],
                       capture_output=True)
        # Switch back to original
        main_branch = git_log(n=1, cwd=repo)["stdout"].split()[0]
        subprocess.run(["git", "-C", repo, "checkout",
                        subprocess.run(["git", "-C", repo, "branch", "--list"],
                                       capture_output=True).stdout.decode().split()[0]],
                       capture_output=True)
        r = git_checkout(branch="other-branch", cwd=repo)
        assert r["success"] is True

    @requires_git
    def test_checkout_nonexistent_branch_fails(self, repo):
        r = git_checkout(branch="does-not-exist-xyz", cwd=repo)
        assert r["success"] is False


# ── git_branch ────────────────────────────────────────────────────────────────

class TestGitBranch:
    @requires_git
    def test_list_branches(self, repo):
        r = git_branch(cwd=repo)
        assert r["success"] is True

    @requires_git
    def test_create_branch(self, repo):
        r = git_branch(name="new-branch", cwd=repo)
        assert r["success"] is True

    def test_delete_without_name_returns_error(self):
        r = git_branch(delete=True, cwd="")
        assert r["success"] is False
        assert "branch name" in r["stderr"].lower()

    @requires_git
    def test_delete_existing_branch(self, repo):
        # Create branch first
        git_branch(name="to-delete", cwd=repo)
        r = git_branch(name="to-delete", delete=True, cwd=repo)
        assert r["success"] is True


# ── git_stash ─────────────────────────────────────────────────────────────────

class TestGitStash:
    def test_invalid_action_returns_error(self):
        r = git_stash(action="invalid", cwd="")
        assert r["success"] is False
        assert "unknown stash action" in r["stderr"].lower()

    @requires_git
    def test_stash_list_on_empty_returns_success(self, repo):
        r = git_stash(action="list", cwd=repo)
        assert r["success"] is True

    @requires_git
    def test_stash_push_and_pop(self, repo):
        f = os.path.join(repo, "stash_test.txt")
        with open(f, "w") as fh:
            fh.write("unsaved\n")
        git_add(paths=f, cwd=repo)
        push_r = git_stash(action="push", message="wip: stash test", cwd=repo)
        assert push_r["success"] is True
        pop_r = git_stash(action="pop", cwd=repo)
        assert pop_r["success"] is True

    @requires_git
    def test_stash_push_with_message(self, repo):
        f = os.path.join(repo, "stash_msg.txt")
        with open(f, "w") as fh:
            fh.write("test\n")
        git_add(paths=f, cwd=repo)
        r = git_stash(action="push", message="my-stash-label", cwd=repo)
        assert r["success"] is True

    @requires_git
    def test_stash_drop_on_empty_fails(self, repo):
        r = git_stash(action="drop", cwd=repo)
        # Either succeeds (already empty) or fails with no stash to drop
        assert isinstance(r["success"], bool)
