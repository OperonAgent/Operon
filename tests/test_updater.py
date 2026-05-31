"""tests/test_updater.py — self-update + version-check (fully mocked, no network/git)."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import updater
from core.version import __version__


class TestVersion:
    def test_current_version_matches_source(self):
        assert updater.current_version() == __version__

    def test_version_is_semver(self):
        parts = __version__.split(".")
        assert len(parts) >= 2 and all(p.isdigit() for p in parts[:2])


class TestVersionCompare:
    def test_newer_true(self):
        assert updater._newer("v3.2.0", "3.1.0") is True

    def test_newer_false_equal(self):
        assert updater._newer("3.1.0", "3.1.0") is False

    def test_newer_false_older(self):
        assert updater._newer("3.0.0", "3.1.0") is False

    def test_handles_v_prefix(self):
        assert updater._newer("v3.1.1", "v3.1.0") is True

    def test_patch_bump(self):
        assert updater._newer("3.1.10", "3.1.2") is True


class TestLatestRelease:
    def test_parses_release_tag(self):
        fake = MagicMock()
        fake.read.return_value = b'{"tag_name": "v3.2.0"}'
        fake.__enter__ = lambda s: fake
        fake.__exit__ = lambda *a: False
        with patch("core.updater.urllib.request.urlopen", return_value=fake):
            assert updater.latest_release() == "v3.2.0"

    def test_offline_returns_none(self):
        import urllib.error
        with patch("core.updater.urllib.request.urlopen",
                   side_effect=urllib.error.URLError("no net")):
            assert updater.latest_release() is None


class TestCheckForUpdate:
    def test_update_available(self):
        with patch("core.updater.latest_release", return_value="v9.9.9"):
            info = updater.check_for_update()
        assert info["update_available"] is True
        assert info["latest"] == "v9.9.9"
        assert info["offline"] is False

    def test_up_to_date(self):
        with patch("core.updater.latest_release", return_value=__version__):
            info = updater.check_for_update()
        assert info["update_available"] is False

    def test_offline(self):
        with patch("core.updater.latest_release", return_value=None):
            info = updater.check_for_update()
        assert info["offline"] is True
        assert info["update_available"] is False


class TestSelfUpdate:
    def test_aborts_when_not_git(self):
        with patch("core.updater._is_git_clone", return_value=False):
            r = updater.self_update()
        assert r["success"] is False
        assert "git" in r["message"].lower()

    def test_aborts_when_dirty(self):
        with patch("core.updater._is_git_clone", return_value=True), \
             patch("core.updater._working_tree_dirty", return_value=True):
            r = updater.self_update()
        assert r["success"] is False
        assert "local changes" in r["message"].lower()

    def test_success_already_up_to_date(self):
        proc = MagicMock(returncode=0, stdout="Already up to date.", stderr="")
        with patch("core.updater._is_git_clone", return_value=True), \
             patch("core.updater._working_tree_dirty", return_value=False), \
             patch("core.updater._git", return_value=proc):
            r = updater.self_update(install_deps=False)
        assert r["success"] is True
        assert "latest" in r["message"].lower()

    def test_success_pulls_updates(self):
        proc = MagicMock(returncode=0, stdout="Updating abc..def\n 3 files changed", stderr="")
        with patch("core.updater._is_git_clone", return_value=True), \
             patch("core.updater._working_tree_dirty", return_value=False), \
             patch("core.updater._git", return_value=proc):
            r = updater.self_update(install_deps=False)
        assert r["success"] is True

    def test_pull_failure_reported(self):
        proc = MagicMock(returncode=1, stdout="", stderr="merge conflict")
        with patch("core.updater._is_git_clone", return_value=True), \
             patch("core.updater._working_tree_dirty", return_value=False), \
             patch("core.updater._git", return_value=proc):
            r = updater.self_update()
        assert r["success"] is False
        assert "failed" in r["message"].lower()
