"""
Operon Git Operations Tool.

Wraps common `git` CLI commands via subprocess so Operon agents can inspect
and manipulate Git repositories without shell injection risks.

Every public function accepts an optional `cwd` keyword argument (the
repository root to operate in) and arbitrary extra keyword arguments (`**_`)
so callers can pass tool-dispatch payloads directly without pre-filtering.

All functions return a uniform dict::

    {
        "success":    bool,   # True when returncode == 0
        "stdout":     str,    # decoded, trailing-whitespace-stripped output
        "stderr":     str,    # decoded stderr
        "returncode": int,    # raw process exit code (-1 on internal error)
    }
"""

import os
import subprocess
from typing import List, Union


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _git(args: List[str], cwd: str, timeout: int = 15) -> dict:
    """
    Run ``git <args>`` inside *cwd* and return a uniform result dict.

    Handles three categories of failure gracefully:
    - ``FileNotFoundError``   – git binary not found on PATH
    - ``subprocess.TimeoutExpired`` – command exceeded *timeout* seconds
    - Any other ``Exception`` – unexpected OS / permission errors
    """
    work_dir = cwd or os.getcwd()

    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        stdout = result.stdout.decode("utf-8", errors="replace").rstrip()
        stderr = result.stderr.decode("utf-8", errors="replace").rstrip()
        return {
            "success":    result.returncode == 0,
            "stdout":     stdout,
            "stderr":     stderr,
            "returncode": result.returncode,
        }

    except FileNotFoundError:
        return {
            "success":    False,
            "stdout":     "",
            "stderr":     "git executable not found. Is Git installed and on PATH?",
            "returncode": -1,
        }
    except subprocess.TimeoutExpired:
        return {
            "success":    False,
            "stdout":     "",
            "stderr":     f"git command timed out after {timeout}s: git {' '.join(args)}",
            "returncode": -1,
        }
    except Exception as exc:
        return {
            "success":    False,
            "stdout":     "",
            "stderr":     str(exc),
            "returncode": -1,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def git_status(cwd: str = "", **_) -> dict:
    """
    Return the short branch + file-status summary for the working tree.

    Runs: ``git status --short --branch``
    """
    return _git(["status", "--short", "--branch"], cwd=cwd)


def git_diff(path: str = "", staged: bool = False, cwd: str = "", **_) -> dict:
    """
    Show a stat + patch diff of working-tree (or staged) changes.

    Runs: ``git diff --stat --patch [--staged] [-- <path>]``

    Args:
        path:   Optional file/directory to limit the diff to.
        staged: When True, diff the index against HEAD (``--staged``).
        cwd:    Repository root (defaults to ``os.getcwd()``).
    """
    args = ["diff", "--stat", "--patch"]
    if staged:
        args.append("--staged")
    if path:
        args += ["--", path]
    return _git(args, cwd=cwd)


def git_log(n: int = 10, oneline: bool = True, cwd: str = "", **_) -> dict:
    """
    Show recent commit history.

    When *oneline* is True runs: ``git log --oneline -<n>``
    Otherwise runs:              ``git log --format="%h %an %ar\\n  %s" -<n>``

    Args:
        n:       Number of commits to show (default 10).
        oneline: Use compact one-line format (default True).
        cwd:     Repository root (defaults to ``os.getcwd()``).
    """
    if oneline:
        fmt_args = ["--oneline"]
    else:
        fmt_args = ["--format=%h %an %ar%n  %s"]
    return _git(["log"] + fmt_args + [f"-{n}"], cwd=cwd)


def git_add(paths: Union[str, List[str]] = ".", cwd: str = "", **_) -> dict:
    """
    Stage files for the next commit.

    Args:
        paths: A single path string (space-separated paths are split
               automatically), a list of path strings, or ``"."`` to
               stage everything (default).
        cwd:   Repository root (defaults to ``os.getcwd()``).

    Runs: ``git add <path> [<path> ...]``
    """
    if isinstance(paths, str):
        path_list = paths.split() if paths.strip() != "." else ["."]
    else:
        path_list = list(paths)

    if not path_list:
        return {
            "success":    False,
            "stdout":     "",
            "stderr":     "No paths provided to git_add.",
            "returncode": -1,
        }

    return _git(["add"] + path_list, cwd=cwd)


def git_commit(message: str = "", cwd: str = "", **_) -> dict:
    """
    Commit staged changes with the given message.

    Args:
        message: Commit message (required; returns an error dict if empty).
        cwd:     Repository root (defaults to ``os.getcwd()``).

    Runs: ``git commit -m "<message>"``
    """
    if not message or not message.strip():
        return {
            "success":    False,
            "stdout":     "",
            "stderr":     "A non-empty commit message is required.",
            "returncode": -1,
        }
    return _git(["commit", "-m", message], cwd=cwd)


def git_checkout(branch: str = "", create: bool = False, cwd: str = "", **_) -> dict:
    """
    Switch to a branch or list all branches.

    - If *branch* is empty: runs ``git branch --list`` (all local branches).
    - If *create* is True:  runs ``git checkout -b <branch>``.
    - Otherwise:            runs ``git checkout <branch>``.

    Args:
        branch: Branch name to switch to (or create).
        create: When True, create the branch before switching.
        cwd:    Repository root (defaults to ``os.getcwd()``).
    """
    if not branch:
        return _git(["branch", "--list"], cwd=cwd)

    args = ["checkout"]
    if create:
        args.append("-b")
    args.append(branch)
    return _git(args, cwd=cwd)


def git_branch(name: str = "", delete: bool = False, cwd: str = "", **_) -> dict:
    """
    List, create, or delete a local branch.

    - *name* empty, *delete* False:  ``git branch --list`` (list all branches)
    - *name* given, *delete* False:  ``git branch <name>``  (create branch)
    - *name* given, *delete* True:   ``git branch -d <name>`` (delete branch)
    - *name* empty, *delete* True:   returns an error (nothing to delete)

    Args:
        name:   Branch name.
        delete: When True, delete the named branch.
        cwd:    Repository root (defaults to ``os.getcwd()``).
    """
    if delete:
        if not name:
            return {
                "success":    False,
                "stdout":     "",
                "stderr":     "A branch name is required when delete=True.",
                "returncode": -1,
            }
        return _git(["branch", "-d", name], cwd=cwd)

    if name:
        return _git(["branch", name], cwd=cwd)

    return _git(["branch", "--list"], cwd=cwd)


def git_stash(action: str = "push", message: str = "", cwd: str = "", **_) -> dict:
    """
    Manage the Git stash.

    Supported *action* values:
    - ``"push"``  – stash current changes; appends ``-m <message>`` when provided.
    - ``"pop"``   – apply the most recent stash and remove it.
    - ``"list"``  – list all stash entries.
    - ``"drop"``  – drop the most recent stash entry.

    Args:
        action:  One of ``"push"``, ``"pop"``, ``"list"``, ``"drop"``.
        message: Optional description (only used with ``action="push"``).
        cwd:     Repository root (defaults to ``os.getcwd()``).

    Runs: ``git stash <action> [-m <message>]``
    """
    valid_actions = {"push", "pop", "list", "drop"}
    if action not in valid_actions:
        return {
            "success":    False,
            "stdout":     "",
            "stderr":     f"Unknown stash action '{action}'. Valid: {sorted(valid_actions)}",
            "returncode": -1,
        }

    args = ["stash", action]
    if action == "push" and message:
        args += ["-m", message]

    return _git(args, cwd=cwd)
