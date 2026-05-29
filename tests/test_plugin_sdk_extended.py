"""Tests for plugin_sdk.py extensions: PluginSigner, PluginHotReloader, PluginMarketplace."""
import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from core.plugin_sdk import (
    PluginManager, PluginSigner, PluginHotReloader, PluginMarketplace,
    MarketplaceEntry, get_marketplace, get_signer, create_plugin_scaffold,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_plugin_dir(tmpdir: Path, name: str = "test-plugin") -> Path:
    """Create a minimal valid plugin directory."""
    pdir = tmpdir / name
    pdir.mkdir()
    (pdir / "plugin.json").write_text(json.dumps({
        "name": name,
        "version": "1.0.0",
        "description": "test",
        "tools": ["my_tool"],
        "skills": [],
    }))
    (pdir / "tools.py").write_text("def my_tool(**_): return {'success': True}\n")
    return pdir


# ── PluginSigner ──────────────────────────────────────────────────────────────

class TestPluginSigner:
    def test_sign_creates_sig_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_plugin_dir(Path(tmp))
            signer = PluginSigner()
            sig = signer.sign(pdir)
            assert (pdir / PluginSigner.SIG_FILE).exists()

    def test_sign_includes_all_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_plugin_dir(Path(tmp))
            signer = PluginSigner()
            sig = signer.sign(pdir)
            assert "plugin.json" in sig["files"]
            assert "tools.py" in sig["files"]

    def test_sign_records_name_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_plugin_dir(Path(tmp))
            signer = PluginSigner()
            sig = signer.sign(pdir)
            assert sig["name"] == "test-plugin"
            assert sig["version"] == "1.0.0"

    def test_sign_excludes_sig_file_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_plugin_dir(Path(tmp))
            signer = PluginSigner()
            sig = signer.sign(pdir)
            assert PluginSigner.SIG_FILE not in sig["files"]

    def test_verify_clean_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_plugin_dir(Path(tmp))
            signer = PluginSigner()
            signer.sign(pdir)
            valid, errors = signer.verify(pdir)
            assert valid
            assert errors == []

    def test_verify_no_sig_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_plugin_dir(Path(tmp))
            signer = PluginSigner()
            valid, errors = signer.verify(pdir)
            assert not valid
            assert len(errors) > 0

    def test_verify_modified_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_plugin_dir(Path(tmp))
            signer = PluginSigner()
            signer.sign(pdir)
            # Modify a file after signing
            (pdir / "tools.py").write_text("def my_tool(**_): return 999\n")
            valid, errors = signer.verify(pdir)
            assert not valid
            assert any("modified" in e for e in errors)

    def test_verify_deleted_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_plugin_dir(Path(tmp))
            signer = PluginSigner()
            signer.sign(pdir)
            (pdir / "tools.py").unlink()
            valid, errors = signer.verify(pdir)
            assert not valid
            assert any("missing" in e for e in errors)

    def test_verify_new_unsigned_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_plugin_dir(Path(tmp))
            signer = PluginSigner()
            signer.sign(pdir)
            # Add a new file after signing
            (pdir / "extra.py").write_text("# injected code\n")
            valid, errors = signer.verify(pdir)
            assert not valid
            assert any("unsigned" in e for e in errors)

    def test_sha256_different_for_different_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            f1 = Path(tmp) / "f1.txt"
            f2 = Path(tmp) / "f2.txt"
            f1.write_text("content A")
            f2.write_text("content B")
            signer = PluginSigner()
            assert signer._sha256(f1) != signer._sha256(f2)

    def test_sha256_same_for_same_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            f1 = Path(tmp) / "f1.txt"
            f2 = Path(tmp) / "f2.txt"
            f1.write_text("identical content")
            f2.write_text("identical content")
            signer = PluginSigner()
            assert signer._sha256(f1) == signer._sha256(f2)


# ── PluginHotReloader ─────────────────────────────────────────────────────────

class TestPluginHotReloader:
    def test_not_running_initially(self):
        mgr = PluginManager()
        hr = PluginHotReloader(mgr)
        assert not hr._running

    def test_stop_when_not_started(self):
        mgr = PluginManager()
        hr = PluginHotReloader(mgr)
        hr.stop()   # Should not raise
        assert not hr._running

    def test_dir_mtime_returns_float(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "file.txt").write_text("data")
            mtime = PluginHotReloader._dir_mtime(p)
            assert isinstance(mtime, float) and mtime > 0

    def test_dir_mtime_empty_dir_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            mtime = PluginHotReloader._dir_mtime(Path(tmp))
            assert mtime == 0.0

    def test_reload_now_unknown_plugin(self):
        mgr = PluginManager()
        hr = PluginHotReloader(mgr)
        ok, msg = hr.reload_now("nonexistent")
        assert not ok
        assert "not loaded" in msg

    def test_reload_now_known_plugin(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = _make_plugin_dir(Path(tmp))
            mgr = PluginManager(Path(tmp))
            mgr.load("test-plugin")
            hr = PluginHotReloader(mgr)
            ok, msg = hr.reload_now("test-plugin")
            assert ok
            assert "reloaded" in msg.lower()


# ── PluginMarketplace (offline) ───────────────────────────────────────────────

class TestPluginMarketplace:
    def _mp_with_entries(self) -> PluginMarketplace:
        mgr = PluginManager()
        mp = PluginMarketplace(mgr)
        mp._entries = [
            MarketplaceEntry("git-flow", "Git workflow tools", "alice",
                             tags=["git", "vcs"], stars=150, verified=True),
            MarketplaceEntry("slack-notify", "Send Slack messages", "bob",
                             tags=["slack", "notifications"], stars=80),
            MarketplaceEntry("data-viz", "Data visualization", "carol",
                             tags=["data", "charts"], stars=200, verified=True),
            MarketplaceEntry("sql-helper", "SQL query helper", "dave",
                             tags=["sql", "database"], stars=30),
        ]
        return mp

    def test_search_by_name(self):
        mp = self._mp_with_entries()
        results = mp.search("git")
        assert len(results) == 1
        assert results[0].name == "git-flow"

    def test_search_by_description(self):
        mp = self._mp_with_entries()
        results = mp.search("visualization")
        assert len(results) == 1

    def test_search_by_tag(self):
        mp = self._mp_with_entries()
        results = mp.search(tag="database")
        assert len(results) == 1
        assert results[0].name == "sql-helper"

    def test_search_verified_only(self):
        mp = self._mp_with_entries()
        results = mp.search(verified=True)
        assert all(r.verified for r in results)
        assert len(results) == 2

    def test_search_sorted_by_stars(self):
        mp = self._mp_with_entries()
        results = mp.search()
        assert results[0].stars >= results[-1].stars

    def test_search_empty_query_returns_all(self):
        mp = self._mp_with_entries()
        results = mp.search()
        assert len(results) == 4

    def test_search_no_match_returns_empty(self):
        mp = self._mp_with_entries()
        results = mp.search("zyxwvut")
        assert results == []

    def test_search_combined_query_and_tag(self):
        mp = self._mp_with_entries()
        results = mp.search("git", tag="vcs")
        assert len(results) == 1

    def test_add_tap_appends(self):
        mp = PluginMarketplace()
        mp.add_tap("https://example.com/registry.json")
        assert "https://example.com/registry.json" in mp._taps

    def test_remove_tap(self):
        mp = PluginMarketplace()
        url = "https://example.com/registry.json"
        mp.add_tap(url)
        mp.remove_tap(url)
        assert url not in mp._taps

    def test_remove_tap_clears_cache(self):
        mp = self._mp_with_entries()
        mp.remove_tap("https://example.com/nonexistent")
        assert mp._entries == []   # cache invalidated

    def test_install_by_name_not_found(self):
        mp = self._mp_with_entries()
        ok, msg = mp.install_by_name("nonexistent-plugin-xyz")
        assert not ok

    def test_install_by_name_no_repo_url(self):
        mp = self._mp_with_entries()
        mp._entries.append(MarketplaceEntry("no-url-plugin", "desc", "author"))
        ok, msg = mp.install_by_name("no-url-plugin")
        assert not ok
        assert "repo_url" in msg

    def test_fetch_uses_cache(self):
        mp = PluginMarketplace()
        mp._fetched_at = time.time()
        mp._entries = [MarketplaceEntry("cached", "cached plugin", "x")]
        with mock.patch.object(mp, "_fetch_tap", return_value=[]) as m:
            count = mp.fetch(force=False)
            m.assert_not_called()
        assert count == 1

    def test_fetch_force_bypasses_cache(self):
        mp = PluginMarketplace()
        mp._fetched_at = time.time()
        mp._entries = [MarketplaceEntry("stale", "stale", "x")]
        with mock.patch.object(mp, "_fetch_tap", return_value=[]):
            count = mp.fetch(force=True)
        assert count == 0   # fetch_tap returns []


# ── MarketplaceEntry ──────────────────────────────────────────────────────────

class TestMarketplaceEntry:
    def test_to_dict(self):
        e = MarketplaceEntry(
            name="my-plugin", description="does stuff", author="alice",
            tags=["tag1"], stars=42, verified=True,
        )
        d = e.to_dict()
        assert d["name"] == "my-plugin"
        assert d["stars"] == 42
        assert d["verified"] is True
        assert "tag1" in d["tags"]

    def test_defaults(self):
        e = MarketplaceEntry("minimal", "desc", "auth")
        assert e.version == "latest"
        assert e.repo_url == ""
        assert e.stars == 0
        assert e.verified is False


# ── get_* singletons ──────────────────────────────────────────────────────────

class TestSingletons:
    def test_get_marketplace_returns_instance(self):
        mp = get_marketplace()
        assert isinstance(mp, PluginMarketplace)

    def test_get_signer_returns_instance(self):
        s = get_signer()
        assert isinstance(s, PluginSigner)
