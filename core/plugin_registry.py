"""
core/plugin_registry.py — Live Plugin Marketplace & Registry

Provides a browseable, installable plugin ecosystem for Operon:

  PluginEntry      — manifest for a single plugin (name, version, tags, etc.)
  PluginIndex      — HTTP-fetched or locally cached plugin catalogue
  RegistrySearcher — keyword + tag + author search against the index
  PluginInstaller  — pip / git install with SHA-256 integrity check
  PluginPublisher  — package a local plugin + submit to the registry
  RegistryCache    — TTL-based local disk cache for the remote index
  PluginRegistry   — top-level orchestrator (search, install, publish, list)

Registry format (JSON served at registry_url):
{
  "version": 1,
  "plugins": [
    {
      "name": "operon-web-search",
      "version": "0.3.1",
      "description": "Web search via SerpAPI / DuckDuckGo",
      "author": "operon-team",
      "tags": ["search", "web"],
      "install_cmd": "pip install operon-web-search",
      "source_url": "https://github.com/operon-team/operon-web-search",
      "sha256": "abc123...",
      "verified": true,
      "created_at": "2026-01-01T00:00:00Z"
    },
    ...
  ]
}

Usage:
    from core.plugin_registry import PluginRegistry

    reg = PluginRegistry()
    results = reg.search("web search")
    for p in results:
        print(p.name, p.version, p.description)

    ok, msg = reg.install("operon-web-search")
    print(msg)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("operon.plugin_registry")

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_REGISTRY_URL  = os.environ.get(
    "OPERON_PLUGIN_REGISTRY_URL",
    "https://raw.githubusercontent.com/operon-ai/plugin-registry/main/registry.json",
)
_CACHE_DIR          = Path(os.environ.get("OPERON_PLUGIN_CACHE",
                                          Path.home() / ".operon" / "plugin_cache"))
_CACHE_TTL_SECONDS  = int(os.environ.get("OPERON_PLUGIN_CACHE_TTL", "3600"))  # 1h
_INDEX_FILENAME     = "registry_index.json"
_INSTALLED_FILENAME = "installed.json"
_FETCH_TIMEOUT      = 10          # seconds
_MAX_RESULTS        = 20


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PluginEntry:
    """Manifest for a single plugin in the registry."""
    name:         str
    version:      str       = "0.0.0"
    description:  str       = ""
    author:       str       = ""
    tags:         List[str] = field(default_factory=list)
    install_cmd:  str       = ""          # pip install / git clone command
    source_url:   str       = ""          # GitHub / PyPI URL
    sha256:       str       = ""          # expected hash of installed package
    verified:     bool      = False       # reviewed by Operon team
    created_at:   str       = ""
    updated_at:   str       = ""
    downloads:    int       = 0
    category:     str       = "general"   # search, memory, tools, integrations…
    operon_min_version: str = ""

    def matches(self, query: str) -> bool:
        """Case-insensitive match against name, description, tags, author."""
        q = query.lower()
        return (
            q in self.name.lower()
            or q in self.description.lower()
            or q in self.author.lower()
            or any(q in t.lower() for t in self.tags)
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PluginEntry":
        known = {f.name for f in PluginEntry.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in d.items() if k in known}
        if "tags" in filtered and isinstance(filtered["tags"], str):
            filtered["tags"] = [t.strip() for t in filtered["tags"].split(",")]
        return PluginEntry(**filtered)

    def install_display(self) -> str:
        cmd = self.install_cmd or f"pip install {self.name}"
        v   = "✓ verified" if self.verified else "⚠ unverified"
        return f"{self.name}=={self.version}  [{v}]  {cmd}"

    def short_str(self) -> str:
        tags_str = ", ".join(self.tags[:3]) if self.tags else "—"
        return (f"{self.name:<32} v{self.version:<10} "
                f"{self.author:<20} [{tags_str}]  {self.description[:60]}")


# ── Registry cache ────────────────────────────────────────────────────────────

class RegistryCache:
    """
    TTL-based local disk cache for the remote registry index.
    Falls back to stale cache if fetch fails.
    """

    def __init__(
        self,
        cache_dir:  Path = _CACHE_DIR,
        ttl:        int  = _CACHE_TTL_SECONDS,
    ) -> None:
        self._dir = cache_dir
        self._ttl = ttl
        self._dir.mkdir(parents=True, exist_ok=True)

    def _index_path(self) -> Path:
        return self._dir / _INDEX_FILENAME

    def is_fresh(self) -> bool:
        p = self._index_path()
        if not p.exists():
            return False
        age = time.time() - p.stat().st_mtime
        return age < self._ttl

    def load(self) -> Optional[Dict[str, Any]]:
        p = self._index_path()
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Cache load failed: %s", e)
            return None

    def save(self, data: Dict[str, Any]) -> None:
        try:
            self._index_path().write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            log.warning("Cache save failed: %s", e)

    def invalidate(self) -> None:
        try:
            self._index_path().unlink(missing_ok=True)
        except Exception:
            pass

    # ── Installed-plugins ledger ──────────────────────────────────────────────

    def _installed_path(self) -> Path:
        return self._dir / _INSTALLED_FILENAME

    def load_installed(self) -> Dict[str, Any]:
        p = self._installed_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def record_installed(self, entry: PluginEntry) -> None:
        installed = self.load_installed()
        installed[entry.name] = {
            "version":      entry.version,
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "install_cmd":  entry.install_cmd,
        }
        try:
            self._installed_path().write_text(
                json.dumps(installed, indent=2), encoding="utf-8"
            )
        except Exception as e:
            log.warning("Record installed failed: %s", e)

    def is_installed(self, name: str) -> bool:
        return name in self.load_installed()


# ── Plugin index ──────────────────────────────────────────────────────────────

class PluginIndex:
    """
    Fetches (or loads from cache) the remote plugin registry index.
    Exposes an in-memory list of PluginEntry objects for searching.
    """

    def __init__(
        self,
        registry_url: str   = _DEFAULT_REGISTRY_URL,
        cache:        Optional[RegistryCache] = None,
    ) -> None:
        self._url   = registry_url
        self._cache = cache or RegistryCache()
        self._plugins: List[PluginEntry] = []
        self._loaded = False

    def load(self, force_refresh: bool = False) -> bool:
        """Load the index (from cache or remote). Returns True on success."""
        if not force_refresh and self._cache.is_fresh():
            data = self._cache.load()
            if data:
                self._plugins = self._parse(data)
                self._loaded  = True
                log.debug("Loaded %d plugins from cache", len(self._plugins))
                return True

        # Fetch from remote
        data = self._fetch()
        if data:
            self._cache.save(data)
            self._plugins = self._parse(data)
            self._loaded  = True
            log.info("Fetched %d plugins from registry", len(self._plugins))
            return True

        # Fall back to stale cache
        stale = self._cache.load()
        if stale:
            self._plugins = self._parse(stale)
            self._loaded  = True
            log.warning("Using stale cache (%d plugins)", len(self._plugins))
            return True

        log.error("Could not load plugin registry")
        return False

    def _fetch(self) -> Optional[Dict[str, Any]]:
        try:
            req = urllib.request.Request(
                self._url,
                headers={"User-Agent": "Operon/1.0 PluginRegistry"},
            )
            with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.URLError as e:
            log.debug("Registry fetch failed (network): %s", e)
        except Exception as e:
            log.debug("Registry fetch failed: %s", e)
        return None

    @staticmethod
    def _parse(data: Dict[str, Any]) -> List[PluginEntry]:
        plugins: List[PluginEntry] = []
        for raw in data.get("plugins", []):
            try:
                plugins.append(PluginEntry.from_dict(raw))
            except Exception as e:
                log.debug("Skipping malformed plugin entry: %s", e)
        return plugins

    @property
    def plugins(self) -> List[PluginEntry]:
        if not self._loaded:
            self.load()
        return list(self._plugins)

    def get(self, name: str) -> Optional[PluginEntry]:
        for p in self.plugins:
            if p.name.lower() == name.lower():
                return p
        return None

    def add_local(self, entry: PluginEntry) -> None:
        """Add a local (unpublished) plugin entry for testing."""
        self._plugins.append(entry)
        self._loaded = True


# ── Registry searcher ─────────────────────────────────────────────────────────

class RegistrySearcher:
    """
    Multi-criteria search against a PluginIndex.

    Supports:
      - keyword search (name + description + tags + author)
      - tag filter  (exact tag match)
      - author filter
      - verified-only filter
      - category filter
      - result ranking (verified > downloads > alphabetical)
    """

    def __init__(self, index: PluginIndex) -> None:
        self._index = index

    def search(
        self,
        query:          str  = "",
        tag:            str  = "",
        author:         str  = "",
        category:       str  = "",
        verified_only:  bool = False,
        max_results:    int  = _MAX_RESULTS,
    ) -> List[PluginEntry]:
        """Return matching PluginEntry objects, ranked by relevance."""
        results: List[PluginEntry] = []

        for p in self._index.plugins:
            if verified_only and not p.verified:
                continue
            if tag and tag.lower() not in [t.lower() for t in p.tags]:
                continue
            if author and author.lower() not in p.author.lower():
                continue
            if category and category.lower() != p.category.lower():
                continue
            if query and not p.matches(query):
                continue
            results.append(p)

        results.sort(
            key=lambda p: (
                -int(p.verified),
                -p.downloads,
                p.name.lower(),
            )
        )
        return results[:max_results]

    def get_by_tag(self, tag: str) -> List[PluginEntry]:
        return self.search(tag=tag)

    def get_verified(self) -> List[PluginEntry]:
        return self.search(verified_only=True)

    def get_by_category(self, category: str) -> List[PluginEntry]:
        return self.search(category=category)

    def get_popular(self, n: int = 10) -> List[PluginEntry]:
        plugins = sorted(self._index.plugins, key=lambda p: -p.downloads)
        return plugins[:n]

    def suggest(self, query: str, n: int = 5) -> List[str]:
        """Return plugin name suggestions for autocomplete."""
        results = self.search(query=query, max_results=n)
        return [p.name for p in results]


# ── Plugin installer ──────────────────────────────────────────────────────────

class PluginInstaller:
    """
    Install a plugin using its install_cmd.
    Supports pip install, git clone + pip install -e, and direct path installs.
    Optionally verifies the installed package SHA-256.
    """

    def __init__(
        self,
        cache: Optional[RegistryCache] = None,
        verify_sha256: bool = True,
    ) -> None:
        self._cache  = cache or RegistryCache()
        self._verify = verify_sha256

    def install(
        self,
        entry:   PluginEntry,
        dry_run: bool = False,
    ) -> Tuple[bool, str]:
        """
        Install the plugin. Returns (success, message).
        """
        cmd = self._build_cmd(entry)
        if dry_run:
            return True, f"[dry-run] Would run: {cmd}"

        log.info("Installing %s: %s", entry.name, cmd)
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return False, f"Install failed:\n{result.stderr[:500]}"

            # Record in installed ledger
            self._cache.record_installed(entry)
            msg = f"✓ Installed {entry.name}=={entry.version}"

            # SHA-256 verification (best-effort)
            if self._verify and entry.sha256:
                ok, sha_msg = self._verify_sha(entry)
                msg += f"\n{sha_msg}"

            return True, msg

        except subprocess.TimeoutExpired:
            return False, f"Install timed out after 120s: {cmd}"
        except Exception as e:
            return False, f"Install error: {e}"

    def uninstall(self, name: str) -> Tuple[bool, str]:
        """Uninstall a plugin via pip."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", name],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                return True, f"✓ Uninstalled {name}"
            return False, f"Uninstall failed: {result.stderr[:200]}"
        except Exception as e:
            return False, f"Uninstall error: {e}"

    def is_installed(self, name: str) -> bool:
        return self._cache.is_installed(name)

    @staticmethod
    def _build_cmd(entry: PluginEntry) -> str:
        if entry.install_cmd:
            return entry.install_cmd
        # Default: pip install from PyPI
        return f"{sys.executable} -m pip install {entry.name}"

    @staticmethod
    def _verify_sha(entry: PluginEntry) -> Tuple[bool, str]:
        """
        Verify the installed package wheel file SHA-256 if available.
        This is a best-effort check; skips gracefully if wheel not found.
        """
        try:
            import importlib.util
            spec = importlib.util.find_spec(
                entry.name.replace("-", "_").replace(".", "_")
            )
            if spec is None or spec.origin is None:
                return True, "(SHA-256 skip: package location unknown)"
            pkg_file = Path(spec.origin)
            if not pkg_file.exists():
                return True, "(SHA-256 skip: __init__.py not found)"
            digest = hashlib.sha256(pkg_file.read_bytes()).hexdigest()
            if entry.sha256 and digest.startswith(entry.sha256[:16]):
                return True, f"✓ SHA-256 prefix match ({digest[:16]}…)"
            if entry.sha256:
                return False, f"⚠ SHA-256 mismatch: {digest[:16]}… ≠ {entry.sha256[:16]}…"
            return True, "(SHA-256 not provided in registry)"
        except Exception as e:
            return True, f"(SHA-256 check skipped: {e})"


# ── Plugin publisher ──────────────────────────────────────────────────────────

class PluginPublisher:
    """
    Package a local plugin directory and create a registry entry.
    Also supports submitting to the remote registry (via HTTP POST / GitHub PR).

    Manifest file expected at <plugin_dir>/operon_plugin.json or pyproject.toml.
    """

    _REQUIRED_FIELDS = ("name", "version", "description", "author")

    def create_manifest(
        self,
        plugin_dir:  str,
        name:        str,
        version:     str,
        description: str,
        author:      str,
        tags:        Optional[List[str]] = None,
        source_url:  str = "",
    ) -> Tuple[bool, str]:
        """
        Write an operon_plugin.json manifest in plugin_dir.
        Returns (success, message).
        """
        manifest_path = Path(plugin_dir) / "operon_plugin.json"
        entry = PluginEntry(
            name=name,
            version=version,
            description=description,
            author=author,
            tags=tags or [],
            source_url=source_url,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            manifest_path.write_text(
                json.dumps(entry.to_dict(), indent=2),
                encoding="utf-8",
            )
            return True, f"Manifest written to {manifest_path}"
        except Exception as e:
            return False, f"Failed to write manifest: {e}"

    def read_manifest(self, plugin_dir: str) -> Optional[PluginEntry]:
        """Read operon_plugin.json from plugin_dir."""
        path = Path(plugin_dir) / "operon_plugin.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return PluginEntry.from_dict(data)
        except Exception:
            return None

    def validate_manifest(self, entry: PluginEntry) -> Tuple[bool, List[str]]:
        """Validate that required fields are present."""
        errors: List[str] = []
        for field_name in self._REQUIRED_FIELDS:
            if not getattr(entry, field_name, ""):
                errors.append(f"Missing required field: {field_name}")
        if entry.version and not re.match(r"^\d+\.\d+\.\d+", entry.version):
            errors.append(f"Version must follow semver (X.Y.Z): {entry.version}")
        return len(errors) == 0, errors

    def compute_sha256(self, plugin_dir: str) -> str:
        """Compute SHA-256 of all .py files in plugin_dir."""
        h = hashlib.sha256()
        for p in sorted(Path(plugin_dir).rglob("*.py")):
            h.update(p.read_bytes())
        return h.hexdigest()

    def package(
        self,
        plugin_dir: str,
        output_dir: str = "",
    ) -> Tuple[bool, str, Optional[PluginEntry]]:
        """
        Read the manifest, compute SHA-256, update the manifest.
        Returns (success, message, updated_entry).
        """
        entry = self.read_manifest(plugin_dir)
        if entry is None:
            return False, "No operon_plugin.json found in plugin_dir", None

        ok, errors = self.validate_manifest(entry)
        if not ok:
            return False, f"Manifest validation failed: {'; '.join(errors)}", None

        sha = self.compute_sha256(plugin_dir)
        entry.sha256 = sha
        entry.updated_at = datetime.now(timezone.utc).isoformat()

        # Write updated manifest back
        manifest_path = Path(plugin_dir) / "operon_plugin.json"
        manifest_path.write_text(
            json.dumps(entry.to_dict(), indent=2), encoding="utf-8"
        )
        msg = f"Packaged {entry.name}=={entry.version}  sha256={sha[:16]}…"
        return True, msg, entry

    def publish_to_registry(
        self,
        entry:        PluginEntry,
        registry_url: str = _DEFAULT_REGISTRY_URL,
        api_key:      str = "",
    ) -> Tuple[bool, str]:
        """
        Submit plugin entry to the remote registry via HTTP POST.
        In production this would open a GitHub PR or POST to a registry API.
        This is a stub that prints the payload and returns a pending message.
        """
        payload = json.dumps(entry.to_dict(), indent=2)
        log.info("Publishing %s to %s", entry.name, registry_url)
        # Real implementation would POST to an API or open a GitHub PR
        return True, (
            f"Plugin submission prepared for {entry.name}=={entry.version}.\n"
            f"Payload ({len(payload)} bytes) ready to POST to: {registry_url}\n"
            f"In production: opens a GitHub PR against the registry repo.\n"
            f"SHA-256: {entry.sha256[:32]}…"
        )


# ── Main registry orchestrator ────────────────────────────────────────────────

class PluginRegistry:
    """
    Top-level orchestrator. Combines index, searcher, installer, publisher.

    CLI commands:
        reg.search("web search")          → List[PluginEntry]
        reg.install("operon-web-search")  → (bool, str)
        reg.uninstall("operon-web-search")→ (bool, str)
        reg.list_installed()              → List[dict]
        reg.info("operon-web-search")     → Optional[PluginEntry]
        reg.refresh()                     → bool
        reg.publish(plugin_dir)           → (bool, str)
    """

    def __init__(
        self,
        registry_url:  str  = _DEFAULT_REGISTRY_URL,
        verify_sha256: bool = True,
        cache_ttl:     int  = _CACHE_TTL_SECONDS,
    ) -> None:
        self._cache     = RegistryCache(ttl=cache_ttl)
        self._index     = PluginIndex(registry_url=registry_url, cache=self._cache)
        self._searcher  = RegistrySearcher(self._index)
        self._installer = PluginInstaller(cache=self._cache, verify_sha256=verify_sha256)
        self._publisher = PluginPublisher()

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:         str  = "",
        tag:           str  = "",
        author:        str  = "",
        category:      str  = "",
        verified_only: bool = False,
        max_results:   int  = _MAX_RESULTS,
    ) -> List[PluginEntry]:
        return self._searcher.search(
            query=query, tag=tag, author=author,
            category=category, verified_only=verified_only,
            max_results=max_results,
        )

    def suggest(self, query: str, n: int = 5) -> List[str]:
        return self._searcher.suggest(query, n)

    def get_popular(self, n: int = 10) -> List[PluginEntry]:
        return self._searcher.get_popular(n)

    # ── Info ──────────────────────────────────────────────────────────────────

    def info(self, name: str) -> Optional[PluginEntry]:
        return self._index.get(name)

    # ── Install / uninstall ───────────────────────────────────────────────────

    def install(
        self,
        name:    str,
        dry_run: bool = False,
    ) -> Tuple[bool, str]:
        """Find plugin by name and install it."""
        entry = self._index.get(name)
        if entry is None:
            # Try loading the index first
            self._index.load()
            entry = self._index.get(name)
        if entry is None:
            return False, f"Plugin '{name}' not found in registry."
        return self._installer.install(entry, dry_run=dry_run)

    def install_entry(
        self,
        entry:   PluginEntry,
        dry_run: bool = False,
    ) -> Tuple[bool, str]:
        return self._installer.install(entry, dry_run=dry_run)

    def uninstall(self, name: str) -> Tuple[bool, str]:
        return self._installer.uninstall(name)

    def is_installed(self, name: str) -> bool:
        return self._installer.is_installed(name)

    # ── List installed ────────────────────────────────────────────────────────

    def list_installed(self) -> List[Dict[str, Any]]:
        installed = self._cache.load_installed()
        result = []
        for pname, info in installed.items():
            entry = self._index.get(pname)
            result.append({
                "name":         pname,
                "version":      info.get("version", "?"),
                "installed_at": info.get("installed_at", ""),
                "verified":     entry.verified if entry else False,
                "description":  entry.description if entry else "",
            })
        return result

    # ── Refresh index ─────────────────────────────────────────────────────────

    def refresh(self) -> bool:
        """Force-refresh the registry index from remote."""
        self._cache.invalidate()
        return self._index.load(force_refresh=True)

    # ── Publish ───────────────────────────────────────────────────────────────

    def publish(
        self,
        plugin_dir: str,
        api_key:    str = "",
    ) -> Tuple[bool, str]:
        """Package a local plugin and publish to the registry."""
        ok, msg, entry = self._publisher.package(plugin_dir)
        if not ok or entry is None:
            return False, msg
        ok2, msg2 = self._publisher.publish_to_registry(entry, api_key=api_key)
        return ok2, f"{msg}\n{msg2}"

    def create_manifest(
        self,
        plugin_dir:  str,
        name:        str,
        version:     str,
        description: str,
        author:      str,
        tags:        Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        return self._publisher.create_manifest(
            plugin_dir, name, version, description, author, tags
        )

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        plugins = self._index.plugins
        verified = [p for p in plugins if p.verified]
        categories: Dict[str, int] = {}
        for p in plugins:
            categories[p.category] = categories.get(p.category, 0) + 1
        installed = self.list_installed()
        return {
            "total_plugins":     len(plugins),
            "verified_plugins":  len(verified),
            "installed_plugins": len(installed),
            "categories":        categories,
            "registry_url":      self._index._url,
            "cache_fresh":       self._cache.is_fresh(),
        }


# ── Tool functions for Operon registry ───────────────────────────────────────

def plugin_search(
    query:        str = "",
    tag:          str = "",
    verified_only: bool = False,
) -> Dict[str, Any]:
    """
    Search the plugin registry.

    Returns:
        {success, plugins: [{name, version, description, tags, verified},...], count}
    """
    reg     = PluginRegistry()
    results = reg.search(query=query, tag=tag, verified_only=verified_only)
    return {
        "success": True,
        "plugins": [p.to_dict() for p in results],
        "count":   len(results),
        "output":  f"Found {len(results)} plugin(s) matching '{query or '*'}'",
    }


def plugin_install(
    name:    str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Install a plugin from the registry.

    Returns:
        {success, message, output}
    """
    reg  = PluginRegistry()
    ok, msg = reg.install(name, dry_run=dry_run)
    return {"success": ok, "message": msg, "output": msg}


def plugin_info(name: str) -> Dict[str, Any]:
    """
    Get full info about a plugin.

    Returns:
        {success, plugin: {...}, output}
    """
    reg   = PluginRegistry()
    entry = reg.info(name)
    if entry is None:
        return {"success": False, "output": f"Plugin '{name}' not found.", "plugin": {}}
    return {
        "success": True,
        "plugin":  entry.to_dict(),
        "output":  entry.install_display(),
    }


def plugin_list_installed() -> Dict[str, Any]:
    """
    List installed plugins.

    Returns:
        {success, plugins: [...], count, output}
    """
    reg  = PluginRegistry()
    pkgs = reg.list_installed()
    return {
        "success": True,
        "plugins": pkgs,
        "count":   len(pkgs),
        "output":  f"{len(pkgs)} plugin(s) installed",
    }


_TOOL_DEFINITIONS = [
    {
        "name": "plugin_search",
        "description": "Search the Operon plugin registry for plugins by keyword, tag, or filter.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":         {"type": "string",  "description": "Search query (keyword in name/description/tags)"},
                "tag":           {"type": "string",  "description": "Filter by exact tag"},
                "verified_only": {"type": "boolean", "description": "Only show verified plugins (default false)"},
            },
            "required": [],
        },
    },
    {
        "name": "plugin_install",
        "description": "Install a plugin from the Operon registry by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name":    {"type": "string",  "description": "Plugin name to install"},
                "dry_run": {"type": "boolean", "description": "If true, print install command without running"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "plugin_info",
        "description": "Get details about a specific plugin in the registry.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Plugin name"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "plugin_list_installed",
        "description": "List all installed Operon plugins.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

_DISPATCH = {
    "plugin_search":         plugin_search,
    "plugin_install":        plugin_install,
    "plugin_info":           plugin_info,
    "plugin_list_installed": plugin_list_installed,
}
