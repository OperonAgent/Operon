"""
core/updater.py — self-update + version-check for Operon.

Closes a real product gap: a downloaded copy never auto-updates. This module
lets a user run `operon --update` to pull the latest code (when installed via
git clone) and refresh dependencies, and `operon --version` to see the current
version plus whether a newer release exists on GitHub.

Design:
  • Network-optional — version check degrades silently when offline.
  • Safe — never force-resets; aborts a git pull if the working tree is dirty.
  • Zero hard deps — stdlib urllib + the system `git` binary.

Public API:
    current_version() -> str
    latest_release(timeout) -> Optional[str]
    check_for_update() -> dict
    self_update() -> dict
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from core.version import __version__

_REPO   = "OperonAgent/Operon"
_ROOT   = Path(__file__).resolve().parent.parent
_LATEST = f"https://api.github.com/repos/{_REPO}/releases/latest"
_TAGS   = f"https://api.github.com/repos/{_REPO}/tags"


def current_version() -> str:
    return __version__


# ── version comparison (PEP 440-ish, dependency-free) ──────────────────────────

def _parse(v: str):
    v = v.lstrip("vV").strip()
    parts = []
    for chunk in v.replace("-", ".").split("."):
        num = ""
        for ch in chunk:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts) or (0,)


def _newer(remote: str, local: str) -> bool:
    a, b = _parse(remote), _parse(local)
    n = max(len(a), len(b))
    a += (0,) * (n - len(a))
    b += (0,) * (n - len(b))
    return a > b


# ── GitHub latest-release / tag lookup ─────────────────────────────────────────

def latest_release(timeout: float = 6.0) -> Optional[str]:
    """Return the latest release tag (e.g. 'v3.2.0') from GitHub, or None."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "operon-updater"}
    # Try the published release first.
    try:
        req = urllib.request.Request(_LATEST, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tag = data.get("tag_name")
            if tag:
                return tag
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError):
        pass
    # Fallback: newest tag.
    try:
        req = urllib.request.Request(_TAGS, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
            if tags:
                return tags[0].get("name")
    except Exception:
        pass
    return None


def check_for_update(timeout: float = 6.0) -> dict:
    """
    Compare the local version to the latest GitHub release.
    Returns {current, latest, update_available, offline}.
    """
    local  = current_version()
    remote = latest_release(timeout=timeout)
    if remote is None:
        return {"current": local, "latest": None,
                "update_available": False, "offline": True}
    return {
        "current": local,
        "latest": remote,
        "update_available": _newer(remote, local),
        "offline": False,
    }


# ── git helpers ─────────────────────────────────────────────────────────────────

def _git(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(_ROOT), *args],
                          capture_output=True, text=True, timeout=timeout)


def _is_git_clone() -> bool:
    try:
        r = _git("rev-parse", "--is-inside-work-tree", timeout=10)
        return r.returncode == 0 and r.stdout.strip() == "true"
    except Exception:
        return False


def _working_tree_dirty() -> bool:
    try:
        r = _git("status", "--porcelain", timeout=10)
        return bool(r.stdout.strip())
    except Exception:
        return False


# ── self-update ───────────────────────────────────────────────────────────────

def self_update(install_deps: bool = True) -> dict:
    """
    Update an Operon git checkout in place: git pull, then refresh deps.

    Returns {success, message, method}. Never force-resets; if the working
    tree has local changes, it aborts and tells the user (so we don't clobber
    their edits).
    """
    if not _is_git_clone():
        return {
            "success": False, "method": "none",
            "message": ("Not a git checkout — can't self-update. Re-download the "
                        "latest release, or reinstall with: "
                        "git clone https://github.com/OperonAgent/Operon.git"),
        }

    if _working_tree_dirty():
        return {
            "success": False, "method": "git",
            "message": ("You have uncommitted local changes — aborting to avoid "
                        "overwriting them. Commit or stash them, then retry "
                        "`operon --update`."),
        }

    try:
        pull = _git("pull", "--ff-only", timeout=120)
    except Exception as e:
        return {"success": False, "method": "git", "message": f"git pull failed: {e}"}

    if pull.returncode != 0:
        return {"success": False, "method": "git",
                "message": f"git pull failed:\n{pull.stderr.strip() or pull.stdout.strip()}"}

    out = pull.stdout.strip()
    already = "Already up to date" in out or "up-to-date" in out

    dep_msg = ""
    if install_deps and not already:
        try:
            from core.bootstrap import provision
            provision(full=False, browser=False)
            dep_msg = "  Dependencies refreshed."
        except Exception as e:
            dep_msg = f"  (dependency refresh skipped: {e})"

    return {
        "success": True, "method": "git",
        "message": ("Already on the latest version." if already
                    else f"Updated to latest.{dep_msg}\n  Restart Operon to use the new version."),
    }
