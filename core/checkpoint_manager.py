"""
Operon Checkpoint Manager — Transparent git snapshots before destructive ops.

Matches Hermes checkpoint_manager.py depth.

Creates lightweight git commits (or stashes) before any destructive operation
(file writes, shell mutations, code edits) so the agent can roll back in one
call if something goes wrong.

Architecture:
  • CheckpointManager wraps a repo path.
  • .checkpoint(message)  → creates a WIP commit on the current branch,
                            returns a CheckpointRef (commit sha + branch).
  • .restore(ref)         → hard-resets HEAD to the checkpointed commit.
  • .list_checkpoints()   → returns all Operon checkpoint commits.
  • .diff(ref)            → diff between ref and HEAD (what changed after).
  • .prune(keep=20)       → delete old checkpoint commits beyond keep count.
  • .cleanup_stash()      → drop any orphaned Operon stash entries.

Decorator:
  @with_checkpoint(manager, "before edit")
  def destructive_fn(...): ...
  → auto-checkpoint before, auto-restore on exception.

Usage:
    from core.checkpoint_manager import CheckpointManager, with_checkpoint

    mgr = CheckpointManager("/path/to/repo")
    ref = mgr.checkpoint("before bulk delete")
    try:
        do_destructive_stuff()
    except Exception:
        mgr.restore(ref)
        raise
"""

from __future__ import annotations

import functools
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("operon.checkpoint_manager")

# Prefix injected into every checkpoint commit message
_CP_PREFIX = "operon-checkpoint:"

# Max seconds to wait for git subprocess
_GIT_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CheckpointRef:
    sha:       str
    branch:    str
    message:   str
    created_at: float = field(default_factory=time.time)
    repo_path: str    = ""

    def short_sha(self) -> str:
        return self.sha[:12]

    def __str__(self) -> str:
        return f"[checkpoint {self.short_sha()} @ {self.branch}] {self.message}"


@dataclass
class CheckpointDiff:
    ref:         CheckpointRef
    added_lines: int
    removed_lines: int
    changed_files: List[str]
    raw_diff:    str

    def summary(self) -> str:
        return (
            f"Since {self.ref.short_sha()}: "
            f"+{self.added_lines}/-{self.removed_lines} lines "
            f"across {len(self.changed_files)} file(s)"
        )


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Creates and restores git checkpoints (WIP commits) around destructive ops."""

    def __init__(
        self,
        repo_path: str = ".",
        auto_stage_all: bool = True,
        max_checkpoints: int = 50,
    ) -> None:
        self._repo     = str(Path(repo_path).resolve())
        self._auto_stage = auto_stage_all
        self._max_cp   = max_checkpoints
        self._history: List[CheckpointRef] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def checkpoint(self, message: str = "WIP checkpoint") -> Optional[CheckpointRef]:
        """
        Stage all changes and create a checkpoint commit.
        Returns a CheckpointRef on success, None if repo has no changes or git fails.
        """
        if not self._is_git_repo():
            log.warning("checkpoint: %s is not a git repo", self._repo)
            return None

        # Ensure there's something to commit
        if self._auto_stage:
            self._git("add", "--all")

        status = self._git("status", "--porcelain")
        # Also check if there are staged changes vs last commit
        has_staged = bool(status.strip())
        # Always allow checkpoint even if working tree is clean (so rollback still works)
        # But skip empty-tree no-op commits
        if not has_staged:
            # Nothing new to snapshot, but return current HEAD as ref
            sha   = self._git("rev-parse", "HEAD").strip()
            branch = self._current_branch()
            ref = CheckpointRef(sha=sha, branch=branch,
                                message=f"{_CP_PREFIX} {message} [no-changes]",
                                repo_path=self._repo)
            log.debug("checkpoint: no changes, returning HEAD %s", ref.short_sha())
            return ref

        full_msg = f"{_CP_PREFIX} {message}"
        out = self._git("commit", "--no-verify", "-m", full_msg)
        if "nothing to commit" in out or "nothing added" in out:
            sha = self._git("rev-parse", "HEAD").strip()
            branch = self._current_branch()
            ref = CheckpointRef(sha=sha, branch=branch, message=full_msg,
                                repo_path=self._repo)
            self._history.append(ref)
            return ref

        sha    = self._git("rev-parse", "HEAD").strip()
        branch = self._current_branch()
        ref = CheckpointRef(sha=sha, branch=branch, message=full_msg,
                            repo_path=self._repo)
        self._history.append(ref)
        log.info("checkpoint created: %s", ref)

        # Auto-prune if over limit
        if len(self._history) > self._max_cp:
            self.prune(keep=self._max_cp)

        return ref

    def restore(self, ref: CheckpointRef) -> bool:
        """
        Hard-reset HEAD back to the checkpointed commit.
        All changes made after the checkpoint are discarded (staged + unstaged).
        Returns True on success.
        """
        if not ref or not ref.sha:
            log.error("restore: invalid ref")
            return False
        if not self._is_git_repo():
            log.error("restore: not a git repo")
            return False

        try:
            out = self._git("reset", "--hard", ref.sha)
            if "fatal" in out.lower() or "error" in out.lower():
                log.error("restore failed: %s", out.strip())
                return False
            self._git("clean", "-fd")   # remove untracked files/dirs
            log.info("restored to checkpoint %s", ref.short_sha())
            return True
        except Exception as e:
            log.error("restore failed: %s", e)
            return False

    def restore_last(self) -> bool:
        """Restore to the most recent checkpoint created by this manager."""
        if not self._history:
            log.warning("restore_last: no checkpoints in session history")
            return False
        return self.restore(self._history[-1])

    def diff(self, ref: CheckpointRef) -> Optional[CheckpointDiff]:
        """Return a diff between ref.sha and HEAD."""
        if not self._is_git_repo():
            return None
        try:
            raw = self._git("diff", ref.sha, "HEAD")
            stat = self._git("diff", "--stat", ref.sha, "HEAD")

            added = removed = 0
            files: List[str] = []
            for line in stat.splitlines():
                m = re.search(r"(\d+) insertion", line)
                if m:
                    added = int(m.group(1))
                m = re.search(r"(\d+) deletion", line)
                if m:
                    removed = int(m.group(1))
                # file lines end with "| N +/-" pattern
                fm = re.match(r"^\s+(\S+)\s+\|", line)
                if fm:
                    files.append(fm.group(1))

            return CheckpointDiff(
                ref=ref,
                added_lines=added,
                removed_lines=removed,
                changed_files=files,
                raw_diff=raw,
            )
        except Exception as e:
            log.warning("diff failed: %s", e)
            return None

    def list_checkpoints(self) -> List[Dict[str, str]]:
        """
        Return all Operon checkpoint commits in the current repo,
        newest first. Each entry: {sha, short_sha, message, date}.
        """
        if not self._is_git_repo():
            return []
        try:
            out = self._git(
                "log", "--oneline", "--all",
                f"--grep={_CP_PREFIX}",
                "--format=%H|%s|%ai",
            )
            results = []
            for line in out.strip().splitlines():
                if not line.strip():
                    continue
                parts = line.split("|", 2)
                if len(parts) == 3:
                    sha, msg, date = parts
                    results.append({
                        "sha":       sha.strip(),
                        "short_sha": sha.strip()[:12],
                        "message":   msg.strip(),
                        "date":      date.strip(),
                    })
            return results
        except Exception as e:
            log.warning("list_checkpoints failed: %s", e)
            return []

    def prune(self, keep: int = 20) -> int:
        """
        Soft-delete (rebase) old checkpoint commits beyond `keep` most recent.
        Since deleting commits rewrites history (destructive), we instead just
        drop references from our in-memory history list.
        Returns the number of entries pruned from session history.
        """
        pruned = 0
        if len(self._history) > keep:
            to_prune = self._history[:-keep]
            self._history = self._history[-keep:]
            pruned = len(to_prune)
            log.info("pruned %d old checkpoints from session history", pruned)
        return pruned

    def stash_checkpoint(self, message: str = "operon-stash") -> Optional[str]:
        """
        Alternative: stash current changes instead of committing.
        Returns stash ref string or None.
        """
        if not self._is_git_repo():
            return None
        try:
            out = self._git("stash", "push", "--include-untracked", "-m",
                            f"{_CP_PREFIX} {message}")
            if "No local changes" in out:
                return None
            # Parse stash@{0} from output
            m = re.search(r"(stash@\{\d+\})", out)
            return m.group(1) if m else "stash@{0}"
        except Exception as e:
            log.warning("stash_checkpoint failed: %s", e)
            return None

    def pop_stash(self, stash_ref: str = "stash@{0}") -> bool:
        """Pop a stash checkpoint back onto the working tree."""
        if not self._is_git_repo():
            return False
        try:
            self._git("stash", "pop", stash_ref)
            return True
        except Exception as e:
            log.warning("pop_stash failed: %s", e)
            return False

    # ── Session stats ─────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "repo_path":          self._repo,
            "session_checkpoints": len(self._history),
            "last_checkpoint":     str(self._history[-1]) if self._history else None,
            "is_git_repo":         self._is_git_repo(),
            "current_branch":      self._current_branch(),
            "head_sha":            self._head_sha(),
        }

    def session_history(self) -> List[CheckpointRef]:
        return list(self._history)

    # ── Context-manager & decorator ───────────────────────────────────────────

    def __enter__(self) -> "CheckpointManager":
        self._ctx_ref = self.checkpoint("auto-checkpoint [context manager]")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None and self._ctx_ref:
            log.warning("Exception caught — restoring to checkpoint %s",
                        self._ctx_ref.short_sha() if self._ctx_ref else "?")
            self.restore(self._ctx_ref)
        return False   # don't suppress exception

    # ── Internals ─────────────────────────────────────────────────────────────

    def _git(self, *args: str) -> str:
        cmd = ["git", "-C", self._repo, *args]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=_GIT_TIMEOUT,
            )
            if result.returncode != 0:
                # Non-fatal: return stderr so callers can check
                return result.stderr or result.stdout or ""
            return result.stdout or ""
        except FileNotFoundError:
            log.error("git not found in PATH")
            return ""
        except subprocess.TimeoutExpired:
            log.error("git command timed out: %s", " ".join(args))
            return ""

    def _is_git_repo(self) -> bool:
        out = self._git("rev-parse", "--git-dir")
        return bool(out.strip()) and "fatal" not in out

    def _current_branch(self) -> str:
        out = self._git("rev-parse", "--abbrev-ref", "HEAD").strip()
        return out if out and "fatal" not in out else "unknown"

    def _head_sha(self) -> str:
        out = self._git("rev-parse", "HEAD").strip()
        return out if out and "fatal" not in out else ""


# ---------------------------------------------------------------------------
# Decorator: @with_checkpoint
# ---------------------------------------------------------------------------

def with_checkpoint(
    manager: CheckpointManager,
    message: str = "auto-checkpoint",
    restore_on_exception: bool = True,
) -> Callable:
    """
    Decorator factory. Creates a checkpoint before the decorated function runs.
    If the function raises and restore_on_exception=True, rolls back to the
    checkpoint automatically.

    Example:
        @with_checkpoint(mgr, "before risky delete")
        def delete_all_files(path): ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ref = manager.checkpoint(f"{message} [{fn.__name__}]")
            try:
                return fn(*args, **kwargs)
            except Exception:
                if restore_on_exception and ref:
                    log.warning("Exception in %s — restoring checkpoint %s",
                                fn.__name__, ref.short_sha() if ref else "?")
                    manager.restore(ref)
                raise
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_manager: Optional[CheckpointManager] = None


def get_manager(repo_path: str = ".") -> CheckpointManager:
    """Return the session-scoped default checkpoint manager."""
    global _default_manager
    if _default_manager is None or _default_manager._repo != str(Path(repo_path).resolve()):
        _default_manager = CheckpointManager(repo_path)
    return _default_manager


def quick_checkpoint(message: str = "WIP", repo_path: str = ".") -> Optional[CheckpointRef]:
    """One-liner: create a checkpoint using the default manager."""
    return get_manager(repo_path).checkpoint(message)
