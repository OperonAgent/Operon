"""
tests/test_cloud_exec.py — regression tests for tools/cloud_exec.py.

Guards the F821 bug where Daytona helpers referenced a bare `urllib` name that
was only imported under an alias, crashing with NameError whenever called.
These tests assert the functions return a clean {success: False} dict (no
credentials configured) instead of raising.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tools.cloud_exec as ce


@pytest.fixture(autouse=True)
def _no_cloud_creds(monkeypatch):
    for k in ("DAYTONA_API_KEY", "DAYTONA_SERVER_URL", "MODAL_TOKEN_ID",
              "MODAL_TOKEN_SECRET"):
        monkeypatch.delenv(k, raising=False)


class TestNoNameError:
    """The core regression: these must NOT raise NameError (the urllib bug)."""

    def test_daytona_list_workspaces_returns_dict(self):
        r = ce.daytona_list_workspaces()
        assert isinstance(r, dict)
        assert r.get("success") is False  # no creds → clean failure

    def test_daytona_run_returns_dict(self):
        r = ce.daytona_run(command="echo hi")
        assert isinstance(r, dict)
        assert r.get("success") is False

    def test_daytona_run_reports_missing_key(self):
        # With a valid command but no API key, it must reach the credential
        # check and fail cleanly (this is the path that hit the urllib bug).
        r = ce.daytona_run(command="echo hi")
        assert "error" in r and "DAYTONA" in str(r["error"]).upper()


class TestModal:
    def test_modal_status_returns_dict(self):
        r = ce.modal_status()
        assert isinstance(r, dict)

    def test_modal_run_returns_dict(self):
        r = ce.modal_run(code="print(1)")
        assert isinstance(r, dict)


class TestUrllibImport:
    def test_module_level_urllib_available(self):
        # The fix added module-level urllib.request / urllib.error.
        import urllib.request, urllib.error  # noqa: F401
        assert hasattr(ce, "urllib")
        assert hasattr(ce.urllib, "request")
        assert hasattr(ce.urllib, "error")
