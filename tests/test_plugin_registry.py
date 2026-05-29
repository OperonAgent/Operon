"""
tests/test_plugin_registry.py — Tests for core/plugin_registry.py

Covers:
  - PluginEntry (dataclass, from_dict, matches, short_str, install_display)
  - RegistryCache (fresh check, load/save, installed ledger)
  - PluginIndex (add_local, get, plugins property, _parse)
  - RegistrySearcher (search, tag, author, verified, category, popular, suggest)
  - PluginInstaller (build_cmd, dry_run, is_installed, verify_sha)
  - PluginPublisher (create_manifest, read_manifest, validate_manifest, compute_sha256, package)
  - PluginRegistry (search, info, install dry_run, list_installed, summary)
  - Tool functions (plugin_search, plugin_install, plugin_info, plugin_list_installed)
  - _TOOL_DEFINITIONS + _DISPATCH
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import time
import unittest
from dataclasses import fields
from typing import Dict, Any
from unittest.mock import MagicMock, patch


from core.plugin_registry import (
    PluginEntry,
    RegistryCache,
    PluginIndex,
    RegistrySearcher,
    PluginInstaller,
    PluginPublisher,
    PluginRegistry,
    plugin_search,
    plugin_install,
    plugin_info,
    plugin_list_installed,
    _TOOL_DEFINITIONS,
    _DISPATCH,
    _DEFAULT_REGISTRY_URL,
    _MAX_RESULTS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_entry(**kw) -> PluginEntry:
    defaults = dict(
        name="test-plugin",
        version="1.0.0",
        description="A test plugin for testing",
        author="test-author",
        tags=["testing", "mock"],
        verified=False,
        downloads=50,
        category="tools",
        install_cmd="pip install test-plugin",
    )
    defaults.update(kw)
    return PluginEntry(**defaults)


def _make_index_with(*entries) -> PluginIndex:
    idx = PluginIndex()
    for e in entries:
        idx.add_local(e)
    return idx


# ===========================================================================
# PluginEntry
# ===========================================================================

class TestPluginEntry(unittest.TestCase):

    def test_default_fields(self):
        p = PluginEntry(name="my-plugin")
        assert p.version == "0.0.0"
        assert p.verified is False
        assert p.tags == []
        assert p.downloads == 0

    def test_matches_name(self):
        p = _make_entry(name="operon-web-search")
        assert p.matches("web")
        assert p.matches("operon")
        assert not p.matches("voice")

    def test_matches_description(self):
        p = _make_entry(description="Searches the web using DuckDuckGo")
        assert p.matches("duckduckgo")
        assert not p.matches("telegram")

    def test_matches_tag(self):
        p = _make_entry(tags=["search", "web", "scraping"])
        assert p.matches("scraping")
        assert not p.matches("memory")

    def test_matches_author(self):
        p = _make_entry(author="operon-team")
        assert p.matches("operon-team")
        assert p.matches("operon")

    def test_matches_case_insensitive(self):
        p = _make_entry(name="OpErOn-SeArCh")
        assert p.matches("operon")
        assert p.matches("OPERON")

    def test_to_dict_keys(self):
        p = _make_entry()
        d = p.to_dict()
        for field in fields(PluginEntry):
            assert field.name in d

    def test_from_dict_roundtrip(self):
        p = _make_entry(tags=["a", "b"])
        d = p.to_dict()
        p2 = PluginEntry.from_dict(d)
        assert p2.name == p.name
        assert p2.version == p.version
        assert p2.tags == p.tags

    def test_from_dict_tags_as_string(self):
        d = _make_entry().to_dict()
        d["tags"] = "search, web, mock"
        p = PluginEntry.from_dict(d)
        assert "search" in p.tags
        assert "web" in p.tags

    def test_from_dict_ignores_unknown_keys(self):
        d = _make_entry().to_dict()
        d["unknown_future_field"] = "value"
        # Should not raise
        p = PluginEntry.from_dict(d)
        assert p.name == "test-plugin"

    def test_install_display_verified(self):
        p = _make_entry(verified=True)
        display = p.install_display()
        assert "verified" in display.lower()
        assert p.name in display

    def test_install_display_unverified(self):
        p = _make_entry(verified=False)
        display = p.install_display()
        assert "unverified" in display.lower()

    def test_short_str(self):
        p = _make_entry()
        s = p.short_str()
        assert p.name in s
        assert p.version in s

    def test_install_cmd_default(self):
        p = PluginEntry(name="my-plugin")
        assert p.install_cmd == ""


# ===========================================================================
# RegistryCache
# ===========================================================================

class TestRegistryCache(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache  = RegistryCache(
            cache_dir=pathlib.Path(self.tmpdir), ttl=3600
        )

    def test_empty_not_fresh(self):
        assert not self.cache.is_fresh()

    def test_save_and_fresh(self):
        self.cache.save({"plugins": []})
        assert self.cache.is_fresh()

    def test_load_after_save(self):
        data = {"plugins": [_make_entry().to_dict()]}
        self.cache.save(data)
        loaded = self.cache.load()
        assert loaded is not None
        assert len(loaded["plugins"]) == 1

    def test_load_empty_returns_none(self):
        assert self.cache.load() is None

    def test_invalidate(self):
        self.cache.save({"plugins": []})
        assert self.cache.is_fresh()
        self.cache.invalidate()
        assert not self.cache.is_fresh()

    def test_record_installed(self):
        p = _make_entry()
        self.cache.record_installed(p)
        assert self.cache.is_installed("test-plugin")

    def test_not_installed_initially(self):
        assert not self.cache.is_installed("nonexistent-plugin")

    def test_load_installed_empty(self):
        installed = self.cache.load_installed()
        assert installed == {}

    def test_load_installed_after_record(self):
        p = _make_entry()
        self.cache.record_installed(p)
        installed = self.cache.load_installed()
        assert "test-plugin" in installed
        assert installed["test-plugin"]["version"] == "1.0.0"

    def test_stale_cache_ttl(self):
        cache = RegistryCache(
            cache_dir=pathlib.Path(self.tmpdir), ttl=0
        )
        cache.save({"plugins": []})
        # TTL=0 → immediately stale
        time.sleep(0.01)
        assert not cache.is_fresh()

    def test_save_creates_dir(self):
        new_dir = pathlib.Path(self.tmpdir) / "subdir" / "deep"
        cache = RegistryCache(cache_dir=new_dir, ttl=3600)
        cache.save({"plugins": []})
        assert cache.is_fresh()


# ===========================================================================
# PluginIndex
# ===========================================================================

class TestPluginIndex(unittest.TestCase):

    def test_add_local_and_get(self):
        idx = PluginIndex()
        p   = _make_entry()
        idx.add_local(p)
        assert idx.get("test-plugin") is not None
        assert idx.get("test-plugin").name == "test-plugin"

    def test_get_case_insensitive(self):
        idx = PluginIndex()
        idx.add_local(_make_entry(name="My-Plugin"))
        assert idx.get("my-plugin") is not None
        assert idx.get("MY-PLUGIN") is not None

    def test_get_not_found(self):
        idx = PluginIndex()
        assert idx.get("nonexistent") is None

    def test_plugins_property(self):
        idx = PluginIndex()
        idx.add_local(_make_entry(name="p1"))
        idx.add_local(_make_entry(name="p2"))
        assert len(idx.plugins) == 2

    def test_parse_valid(self):
        data = {"plugins": [_make_entry().to_dict()]}
        result = PluginIndex._parse(data)
        assert len(result) == 1
        assert result[0].name == "test-plugin"

    def test_parse_empty(self):
        result = PluginIndex._parse({"plugins": []})
        assert result == []

    def test_parse_missing_key(self):
        result = PluginIndex._parse({})
        assert result == []

    def test_parse_skips_invalid_entry(self):
        data = {"plugins": [{"bad": "entry"}, _make_entry().to_dict()]}
        result = PluginIndex._parse(data)
        # The valid entry should survive
        assert any(p.name == "test-plugin" for p in result)

    def test_fetch_network_failure_returns_none(self):
        idx = PluginIndex(registry_url="http://invalid.nonexistent.example.invalid/")
        result = idx._fetch()
        assert result is None

    def test_load_with_stale_cache(self):
        """If fetch fails and cache exists, uses stale cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = RegistryCache(cache_dir=pathlib.Path(tmpdir), ttl=0)  # TTL=0 → stale
            cache.save({"plugins": [_make_entry().to_dict()]})
            idx = PluginIndex(
                registry_url="http://invalid.nonexistent.example.invalid/",
                cache=cache,
            )
            ok = idx.load()
            assert ok
            assert len(idx.plugins) == 1


# ===========================================================================
# RegistrySearcher
# ===========================================================================

class TestRegistrySearcher(unittest.TestCase):

    def _make_searcher(self):
        idx = _make_index_with(
            _make_entry(name="operon-web-search", tags=["search","web"],
                        verified=True, downloads=1000, category="search",
                        author="operon-team"),
            _make_entry(name="operon-memory-store", tags=["memory","storage"],
                        verified=False, downloads=500, category="memory",
                        author="community"),
            _make_entry(name="operon-telegram-bot", tags=["telegram","chat"],
                        verified=True, downloads=250, category="integrations",
                        author="operon-team"),
        )
        return RegistrySearcher(idx)

    def test_search_all(self):
        s = self._make_searcher()
        results = s.search()
        assert len(results) == 3

    def test_search_by_keyword(self):
        s = self._make_searcher()
        results = s.search(query="web")
        assert len(results) == 1
        assert results[0].name == "operon-web-search"

    def test_search_by_tag(self):
        s = self._make_searcher()
        results = s.search(tag="memory")
        assert len(results) == 1
        assert results[0].name == "operon-memory-store"

    def test_search_by_author(self):
        s = self._make_searcher()
        results = s.search(author="operon-team")
        assert len(results) == 2

    def test_search_verified_only(self):
        s = self._make_searcher()
        results = s.search(verified_only=True)
        assert all(p.verified for p in results)
        assert len(results) == 2

    def test_search_by_category(self):
        s = self._make_searcher()
        results = s.search(category="memory")
        assert len(results) == 1

    def test_search_no_match(self):
        s = self._make_searcher()
        results = s.search(query="voice_pipeline_xyz_nonexistent")
        assert results == []

    def test_search_max_results(self):
        idx = _make_index_with(*[_make_entry(name=f"plugin-{i}") for i in range(25)])
        s = RegistrySearcher(idx)
        results = s.search(max_results=5)
        assert len(results) == 5

    def test_rank_verified_first(self):
        s = self._make_searcher()
        results = s.search()
        # Verified should come first
        verified = [p for p in results if p.verified]
        first_verified_idx = next(i for i, p in enumerate(results) if p.verified)
        first_unverified_idx = next(i for i, p in enumerate(results) if not p.verified)
        assert first_verified_idx < first_unverified_idx

    def test_get_by_tag(self):
        s = self._make_searcher()
        results = s.get_by_tag("telegram")
        assert len(results) == 1

    def test_get_verified(self):
        s = self._make_searcher()
        results = s.get_verified()
        assert all(p.verified for p in results)

    def test_get_popular(self):
        s = self._make_searcher()
        results = s.get_popular(n=2)
        assert len(results) == 2
        assert results[0].downloads >= results[1].downloads

    def test_suggest(self):
        s = self._make_searcher()
        suggestions = s.suggest("operon", n=5)
        assert all(isinstance(name, str) for name in suggestions)

    def test_get_by_category(self):
        s = self._make_searcher()
        results = s.get_by_category("search")
        assert len(results) == 1


# ===========================================================================
# PluginInstaller
# ===========================================================================

class TestPluginInstaller(unittest.TestCase):

    def test_build_cmd_from_entry(self):
        entry = _make_entry(install_cmd="pip install test-plugin")
        cmd = PluginInstaller._build_cmd(entry)
        assert cmd == "pip install test-plugin"

    def test_build_cmd_default_pip(self):
        entry = PluginEntry(name="my-pkg")
        cmd = PluginInstaller._build_cmd(entry)
        assert "pip" in cmd
        assert "my-pkg" in cmd

    def test_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            installer = PluginInstaller(
                cache=RegistryCache(cache_dir=pathlib.Path(tmpdir)),
                verify_sha256=False
            )
            entry = _make_entry()
            ok, msg = installer.install(entry, dry_run=True)
            assert ok
            assert "dry-run" in msg.lower()

    def test_is_installed_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            installer = PluginInstaller(
                cache=RegistryCache(cache_dir=pathlib.Path(tmpdir))
            )
            assert not installer.is_installed("nonexistent")

    def test_verify_sha_no_sha(self):
        entry = _make_entry(sha256="")
        ok, msg = PluginInstaller._verify_sha(entry)
        assert ok
        # Either "not provided" (if package found) or "skip" (if pkg not installed)
        assert "not provided" in msg or "skip" in msg.lower() or "unknown" in msg.lower()

    def test_install_failure(self):
        """Simulate a failed install command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            installer = PluginInstaller(
                cache=RegistryCache(cache_dir=pathlib.Path(tmpdir)),
                verify_sha256=False,
            )
            entry = _make_entry(
                install_cmd="false"   # Unix command that exits 1
            )
            ok, msg = installer.install(entry, dry_run=False)
            if sys.platform == "win32":
                self.skipTest("Skipping on Windows")
            assert not ok

    def test_uninstall_nonexistent(self):
        """Uninstalling a non-existent package should fail gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            installer = PluginInstaller(
                cache=RegistryCache(cache_dir=pathlib.Path(tmpdir))
            )
            ok, msg = installer.uninstall("nonexistent-pkg-xyz-abc-123")
            # pip uninstall returns 0 with --yes even for nonexistent pkgs
            # OR returns non-zero — just confirm no crash
            assert isinstance(ok, bool)


# ===========================================================================
# PluginPublisher
# ===========================================================================

class TestPluginPublisher(unittest.TestCase):

    def test_create_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pub = PluginPublisher()
            ok, msg = pub.create_manifest(
                tmpdir, "my-plugin", "1.0.0", "A test", "me", ["testing"]
            )
            assert ok
            manifest = pathlib.Path(tmpdir) / "operon_plugin.json"
            assert manifest.exists()

    def test_read_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pub = PluginPublisher()
            pub.create_manifest(tmpdir, "my-plugin", "1.0.0", "Desc", "Auth")
            entry = pub.read_manifest(tmpdir)
            assert entry is not None
            assert entry.name == "my-plugin"
            assert entry.version == "1.0.0"

    def test_read_manifest_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pub = PluginPublisher()
            assert pub.read_manifest(tmpdir) is None

    def test_validate_manifest_ok(self):
        pub   = PluginPublisher()
        entry = _make_entry()
        ok, errors = pub.validate_manifest(entry)
        assert ok
        assert errors == []

    def test_validate_manifest_missing_name(self):
        pub   = PluginPublisher()
        entry = PluginEntry(name="", version="1.0.0", description="Desc", author="Auth")
        ok, errors = pub.validate_manifest(entry)
        assert not ok
        assert any("name" in e.lower() for e in errors)

    def test_validate_manifest_bad_version(self):
        pub   = PluginPublisher()
        entry = _make_entry(version="beta")
        ok, errors = pub.validate_manifest(entry)
        assert not ok
        assert any("version" in e.lower() for e in errors)

    def test_validate_manifest_valid_semver(self):
        pub   = PluginPublisher()
        entry = _make_entry(version="2.3.4")
        ok, errors = pub.validate_manifest(entry)
        assert ok

    def test_compute_sha256(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pathlib.Path(tmpdir, "main.py").write_text("print('hello')")
            pub = PluginPublisher()
            sha = pub.compute_sha256(tmpdir)
            assert len(sha) == 64  # SHA-256 hex

    def test_compute_sha256_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pub = PluginPublisher()
            sha = pub.compute_sha256(tmpdir)
            # No .py files → empty hash
            assert isinstance(sha, str)

    def test_package_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pub = PluginPublisher()
            pub.create_manifest(tmpdir, "pkg", "1.0.0", "D", "A")
            pathlib.Path(tmpdir, "main.py").write_text("x=1")
            ok, msg, entry = pub.package(tmpdir)
            assert ok
            assert entry is not None
            assert len(entry.sha256) == 64

    def test_package_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pub = PluginPublisher()
            ok, msg, entry = pub.package(tmpdir)
            assert not ok
            assert entry is None

    def test_package_bad_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pub = PluginPublisher()
            pub.create_manifest(tmpdir, "pkg", "bad-version", "D", "A")
            ok, msg, entry = pub.package(tmpdir)
            assert not ok

    def test_publish_to_registry_stub(self):
        pub   = PluginPublisher()
        entry = _make_entry(sha256="a" * 64)
        ok, msg = pub.publish_to_registry(entry)
        assert ok
        assert "submission" in msg.lower() or "prepared" in msg.lower()


# ===========================================================================
# PluginRegistry (integration)
# ===========================================================================

class TestPluginRegistry(unittest.TestCase):

    def _make_registry(self):
        reg = PluginRegistry()
        # Populate with local entries (no HTTP)
        reg._index.add_local(_make_entry(
            name="operon-web", tags=["web"], verified=True, downloads=1000
        ))
        reg._index.add_local(_make_entry(
            name="operon-memory", tags=["memory"], verified=False, downloads=200
        ))
        return reg

    def test_search(self):
        reg = self._make_registry()
        results = reg.search(query="web")
        assert len(results) == 1
        assert results[0].name == "operon-web"

    def test_info_found(self):
        reg = self._make_registry()
        entry = reg.info("operon-web")
        assert entry is not None
        assert entry.name == "operon-web"

    def test_info_not_found(self):
        reg = self._make_registry()
        assert reg.info("nonexistent-xyz") is None

    def test_install_not_in_registry(self):
        reg = self._make_registry()
        ok, msg = reg.install("nonexistent-xyz-abc")
        assert not ok
        assert "not found" in msg.lower()

    def test_install_dry_run(self):
        reg = self._make_registry()
        ok, msg = reg.install("operon-web", dry_run=True)
        assert ok
        assert "dry-run" in msg.lower()

    def test_list_installed_empty(self):
        reg = self._make_registry()
        pkgs = reg.list_installed()
        assert isinstance(pkgs, list)

    def test_suggest(self):
        reg = self._make_registry()
        suggestions = reg.suggest("operon")
        assert isinstance(suggestions, list)

    def test_get_popular(self):
        reg = self._make_registry()
        popular = reg.get_popular(n=2)
        assert len(popular) == 2
        assert popular[0].downloads >= popular[1].downloads

    def test_summary(self):
        reg = self._make_registry()
        s = reg.summary()
        assert "total_plugins" in s
        assert s["total_plugins"] == 2
        assert "verified_plugins" in s

    def test_create_manifest_and_publish(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = PluginRegistry()
            ok, msg = reg.create_manifest(
                tmpdir, "my-plugin", "1.0.0", "Test plugin", "me"
            )
            assert ok
            assert pathlib.Path(tmpdir, "operon_plugin.json").exists()

    def test_uninstall_returns_tuple(self):
        reg = self._make_registry()
        ok, msg = reg.uninstall("nonexistent-xyz")
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_is_installed_false(self):
        reg = self._make_registry()
        assert not reg.is_installed("nonexistent-xyz")


# ===========================================================================
# Tool functions
# ===========================================================================

class TestToolFunctions(unittest.TestCase):

    def test_plugin_search_empty_query(self):
        with patch("core.plugin_registry.PluginRegistry") as MockReg:
            instance = MagicMock()
            instance.search.return_value = []
            MockReg.return_value = instance
            result = plugin_search()
        assert result["success"] is True
        assert result["count"] == 0

    def test_plugin_search_with_results(self):
        with patch("core.plugin_registry.PluginRegistry") as MockReg:
            instance = MagicMock()
            instance.search.return_value = [_make_entry()]
            MockReg.return_value = instance
            result = plugin_search(query="test")
        assert result["success"] is True
        assert result["count"] == 1

    def test_plugin_install_not_found(self):
        with patch("core.plugin_registry.PluginRegistry") as MockReg:
            instance = MagicMock()
            instance.install.return_value = (False, "not found in registry")
            MockReg.return_value = instance
            result = plugin_install("nonexistent")
        assert result["success"] is False

    def test_plugin_install_success(self):
        with patch("core.plugin_registry.PluginRegistry") as MockReg:
            instance = MagicMock()
            instance.install.return_value = (True, "✓ Installed test-plugin==1.0.0")
            MockReg.return_value = instance
            result = plugin_install("test-plugin", dry_run=True)
        assert result["success"] is True

    def test_plugin_info_not_found(self):
        with patch("core.plugin_registry.PluginRegistry") as MockReg:
            instance = MagicMock()
            instance.info.return_value = None
            MockReg.return_value = instance
            result = plugin_info("nonexistent")
        assert result["success"] is False
        assert result["plugin"] == {}

    def test_plugin_info_found(self):
        with patch("core.plugin_registry.PluginRegistry") as MockReg:
            instance = MagicMock()
            instance.info.return_value = _make_entry()
            MockReg.return_value = instance
            result = plugin_info("test-plugin")
        assert result["success"] is True
        assert "plugin" in result

    def test_plugin_list_installed_empty(self):
        with patch("core.plugin_registry.PluginRegistry") as MockReg:
            instance = MagicMock()
            instance.list_installed.return_value = []
            MockReg.return_value = instance
            result = plugin_list_installed()
        assert result["success"] is True
        assert result["count"] == 0

    def test_plugin_list_installed_with_results(self):
        with patch("core.plugin_registry.PluginRegistry") as MockReg:
            instance = MagicMock()
            instance.list_installed.return_value = [
                {"name": "p1", "version": "1.0.0", "installed_at": "", "verified": True}
            ]
            MockReg.return_value = instance
            result = plugin_list_installed()
        assert result["count"] == 1

    def test_search_output_field(self):
        with patch("core.plugin_registry.PluginRegistry") as MockReg:
            instance = MagicMock()
            instance.search.return_value = [_make_entry(), _make_entry(name="p2")]
            MockReg.return_value = instance
            result = plugin_search(query="test")
        assert "Found 2" in result["output"]

    def test_search_verified_only_passed(self):
        with patch("core.plugin_registry.PluginRegistry") as MockReg:
            instance = MagicMock()
            instance.search.return_value = []
            MockReg.return_value = instance
            plugin_search(verified_only=True)
        instance.search.assert_called_once_with(
            query="", tag="", verified_only=True
        )


# ===========================================================================
# _TOOL_DEFINITIONS + _DISPATCH
# ===========================================================================

class TestToolDefinitions(unittest.TestCase):

    def test_is_list(self):
        assert isinstance(_TOOL_DEFINITIONS, list)

    def test_four_tools(self):
        assert len(_TOOL_DEFINITIONS) == 4

    def test_tool_names(self):
        names = [t["name"] for t in _TOOL_DEFINITIONS]
        assert "plugin_search" in names
        assert "plugin_install" in names
        assert "plugin_info" in names
        assert "plugin_list_installed" in names

    def test_each_has_input_schema(self):
        for tool in _TOOL_DEFINITIONS:
            assert "input_schema" in tool
            assert "description" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_plugin_install_requires_name(self):
        t = next(t for t in _TOOL_DEFINITIONS if t["name"] == "plugin_install")
        assert "name" in t["input_schema"]["required"]


class TestDispatch(unittest.TestCase):

    def test_is_dict(self):
        assert isinstance(_DISPATCH, dict)

    def test_keys_match_definitions(self):
        def_names = {t["name"] for t in _TOOL_DEFINITIONS}
        assert def_names == set(_DISPATCH.keys())

    def test_values_callable(self):
        for fn in _DISPATCH.values():
            assert callable(fn)

    def test_dispatch_search(self):
        assert _DISPATCH["plugin_search"] is plugin_search

    def test_dispatch_install(self):
        assert _DISPATCH["plugin_install"] is plugin_install


# ===========================================================================
# Constants
# ===========================================================================

class TestConstants(unittest.TestCase):

    def test_registry_url_is_string(self):
        assert isinstance(_DEFAULT_REGISTRY_URL, str)
        assert len(_DEFAULT_REGISTRY_URL) > 0

    def test_max_results_positive(self):
        assert _MAX_RESULTS > 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
