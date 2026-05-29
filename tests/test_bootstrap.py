"""
tests/test_bootstrap.py — Tests for core/bootstrap.py dependency provisioner.

These tests mock subprocess/import so they NEVER actually install anything.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import bootstrap


# ── Package group sanity ──────────────────────────────────────────────────────

class TestPackageGroups:
    def test_core_packages_nonempty(self):
        assert len(bootstrap.CORE_PACKAGES) >= 3

    def test_core_has_requests(self):
        assert any("requests" in p for p in bootstrap.CORE_PACKAGES)

    def test_recommended_has_playwright(self):
        assert any("playwright" in p for p in bootstrap.RECOMMENDED_PACKAGES)

    def test_recommended_has_prompt_toolkit(self):
        assert any("prompt_toolkit" in p for p in bootstrap.RECOMMENDED_PACKAGES)

    def test_full_superset_of_recommended(self):
        for p in bootstrap.RECOMMENDED_PACKAGES:
            assert p in bootstrap.FULL_PACKAGES

    def test_full_has_computer_use_deps(self):
        joined = " ".join(bootstrap.FULL_PACKAGES)
        assert "mss" in joined
        assert "pynput" in joined


# ── _is_importable ────────────────────────────────────────────────────────────

class TestIsImportable:
    def test_stdlib_importable(self):
        assert bootstrap._is_importable("os") is True

    def test_nonexistent_not_importable(self):
        assert bootstrap._is_importable("this_module_does_not_exist_xyz") is False

    def test_handles_version_spec(self):
        # Should strip >= specifier before checking
        assert bootstrap._is_importable("os>=1.0.0") is True

    def test_handles_extras_spec(self):
        # Should strip [extra] before checking
        assert bootstrap._is_importable("os[foo]") is True

    def test_beautifulsoup_alias(self):
        # beautifulsoup4 imports as bs4
        result = bootstrap._is_importable("beautifulsoup4")
        assert isinstance(result, bool)


# ── _pip_install ──────────────────────────────────────────────────────────────

class TestPipInstall:
    def test_empty_list_returns_true(self):
        assert bootstrap._pip_install([]) is True

    def test_calls_subprocess(self):
        with patch("core.bootstrap.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ok = bootstrap._pip_install(["fakepkg"])
        assert ok is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pip" in args
        assert "install" in args
        assert "fakepkg" in args

    def test_upgrade_flag(self):
        with patch("core.bootstrap.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            bootstrap._pip_install(["fakepkg"], upgrade=True)
        args = mock_run.call_args[0][0]
        assert "--upgrade" in args

    def test_returns_false_on_nonzero(self):
        with patch("core.bootstrap.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert bootstrap._pip_install(["fakepkg"]) is False

    def test_returns_false_on_exception(self):
        with patch("core.bootstrap.subprocess.run", side_effect=OSError("boom")):
            assert bootstrap._pip_install(["fakepkg"]) is False


# ── Browser binary detection ──────────────────────────────────────────────────

class TestBrowserBinary:
    def test_returns_false_when_playwright_missing(self):
        with patch("core.bootstrap._is_importable", return_value=False):
            assert bootstrap.is_browser_binary_installed() is False

    def test_ensure_skips_when_already_installed(self):
        with patch("core.bootstrap._is_importable", return_value=True), \
             patch("core.bootstrap.is_browser_binary_installed", return_value=True):
            ok, msg = bootstrap.ensure_browser_binary(quiet=True)
        assert ok is True
        assert "already" in msg.lower()

    def test_ensure_installs_when_missing(self):
        call_state = {"installed": False}

        def fake_installed():
            return call_state["installed"]

        def fake_run(cmd, check=False):
            call_state["installed"] = True   # simulate successful install
            return MagicMock(returncode=0)

        with patch("core.bootstrap._is_importable", return_value=True), \
             patch("core.bootstrap.is_browser_binary_installed", side_effect=fake_installed), \
             patch("core.bootstrap.subprocess.run", side_effect=fake_run):
            ok, msg = bootstrap.ensure_browser_binary(quiet=True, with_deps=False)
        assert ok is True

    def test_ensure_reports_failure(self):
        with patch("core.bootstrap._is_importable", return_value=True), \
             patch("core.bootstrap.is_browser_binary_installed", return_value=False), \
             patch("core.bootstrap.subprocess.run", return_value=MagicMock(returncode=1)):
            ok, msg = bootstrap.ensure_browser_binary(quiet=True, with_deps=False)
        assert ok is False

    def test_ensure_installs_package_if_missing(self):
        with patch("core.bootstrap._is_importable", return_value=False), \
             patch("core.bootstrap._pip_install", return_value=False) as mock_pip:
            ok, msg = bootstrap.ensure_browser_binary(quiet=True)
        assert ok is False
        mock_pip.assert_called()


# ── Status report ─────────────────────────────────────────────────────────────

class TestCheckStatus:
    def test_returns_dict_with_expected_keys(self):
        st = bootstrap.check_status()
        assert "core" in st
        assert "recommended" in st
        assert "browser_binary" in st
        assert "python" in st

    def test_core_status_has_all_packages(self):
        st = bootstrap.check_status()
        assert "requests" in st["core"]

    def test_python_version_string(self):
        st = bootstrap.check_status()
        assert isinstance(st["python"], str)
        assert "." in st["python"]

    def test_print_status_no_crash(self):
        with patch("builtins.print"):
            bootstrap.print_status()


# ── provision ─────────────────────────────────────────────────────────────────

class TestProvision:
    def test_provision_skips_installed_packages(self):
        with patch("core.bootstrap._is_importable", return_value=True), \
             patch("core.bootstrap.ensure_browser_binary", return_value=(True, "ok")), \
             patch("core.bootstrap._pip_install", return_value=True) as mock_pip, \
             patch("builtins.print"):
            ok = bootstrap.provision(full=False, browser=True)
        assert ok is True

    def test_provision_no_browser(self):
        with patch("core.bootstrap._is_importable", return_value=True), \
             patch("core.bootstrap.ensure_browser_binary") as mock_browser, \
             patch("core.bootstrap._pip_install", return_value=True), \
             patch("builtins.print"):
            bootstrap.provision(full=False, browser=False)
        mock_browser.assert_not_called()

    def test_provision_installs_missing(self):
        with patch("core.bootstrap._is_importable", return_value=False), \
             patch("core.bootstrap.ensure_browser_binary", return_value=(True, "ok")), \
             patch("core.bootstrap._pip_install", return_value=True) as mock_pip, \
             patch("builtins.print"):
            bootstrap.provision(full=False, browser=True)
        mock_pip.assert_called()

    def test_provision_full_uses_full_packages(self):
        captured = {}

        def fake_pip(pkgs, upgrade=False):
            captured["pkgs"] = pkgs
            return True

        with patch("core.bootstrap._is_importable", return_value=False), \
             patch("core.bootstrap.ensure_browser_binary", return_value=(True, "ok")), \
             patch("core.bootstrap._pip_install", side_effect=fake_pip), \
             patch("builtins.print"):
            bootstrap.provision(full=True, browser=False)
        joined = " ".join(captured.get("pkgs", []))
        assert "mss" in joined  # full-only package


# ── CLI ───────────────────────────────────────────────────────────────────────

class TestCLI:
    def test_check_flag(self):
        with patch.object(sys, "argv", ["bootstrap", "--check"]), \
             patch("core.bootstrap.print_status") as mock_status:
            rc = bootstrap._cli()
        assert rc == 0
        mock_status.assert_called_once()

    def test_browser_flag(self):
        with patch.object(sys, "argv", ["bootstrap", "--browser"]), \
             patch("core.bootstrap.ensure_browser_binary", return_value=(True, "ok")):
            rc = bootstrap._cli()
        assert rc == 0

    def test_browser_flag_failure_returns_1(self):
        with patch.object(sys, "argv", ["bootstrap", "--browser"]), \
             patch("core.bootstrap.ensure_browser_binary", return_value=(False, "fail")):
            rc = bootstrap._cli()
        assert rc == 1

    def test_default_provision(self):
        with patch.object(sys, "argv", ["bootstrap"]), \
             patch("core.bootstrap.provision", return_value=True) as mock_prov:
            rc = bootstrap._cli()
        assert rc == 0
        mock_prov.assert_called_once()
