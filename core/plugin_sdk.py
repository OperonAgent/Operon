"""
Operon Plugin SDK — Community extension system.

Adapted from Hermes Agent skill_bundles / OpenClaw plugin-sdk architecture.

Plugins are directories in ~/.operon/plugins/<plugin_name>/ containing:
  plugin.json     — manifest (required)
  tools.py        — optional: Python file exporting tool functions
  skills/         — optional: Markdown skill files
  hooks/          — optional: hook scripts

plugin.json schema:
    {
      "name": "my-plugin",
      "version": "1.0.0",
      "description": "What this plugin does",
      "author": "Your Name",
      "tools": ["my_tool_1", "my_tool_2"],  // functions exported from tools.py
      "skills": ["skill1.md", "skill2.md"],  // skill files in skills/
      "hooks": {
        "pre_tool": "hooks/pre_tool.sh",
        "post_response": "hooks/post_response.sh"
      },
      "requires": ["requests>=2.28"],  // pip deps
      "operon_min_version": "1.0.0"
    }

Usage:
    from core.plugin_sdk import PluginManager
    pm = PluginManager()
    pm.load_all()           # load all plugins from ~/.operon/plugins/
    pm.install("path/to")  # install a plugin from a directory
    pm.list()              # list loaded plugins

    # Tool functions are auto-registered into the tool registry.
    # Skills are auto-loaded into the skill system.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("operon.plugin_sdk")

PLUGINS_DIR  = Path.home() / ".operon" / "plugins"
SKILLS_DIR   = Path.home() / ".operon" / "skills"


# ---------------------------------------------------------------------------
# Plugin manifest
# ---------------------------------------------------------------------------

@dataclass
class PluginManifest:
    name:               str
    version:            str  = "0.1.0"
    description:        str  = ""
    author:             str  = ""
    tools:              List[str] = field(default_factory=list)
    skills:             List[str] = field(default_factory=list)
    hooks:              Dict[str, str] = field(default_factory=dict)
    requires:           List[str] = field(default_factory=list)
    operon_min_version: str  = "0.0.0"

    @classmethod
    def from_dict(cls, d: Dict) -> "PluginManifest":
        return cls(
            name               = d.get("name", "unknown"),
            version            = d.get("version", "0.1.0"),
            description        = d.get("description", ""),
            author             = d.get("author", ""),
            tools              = d.get("tools", []),
            skills             = d.get("skills", []),
            hooks              = d.get("hooks", {}),
            requires           = d.get("requires", []),
            operon_min_version = d.get("operon_min_version", "0.0.0"),
        )


@dataclass
class LoadedPlugin:
    manifest:     PluginManifest
    plugin_dir:   Path
    tool_fns:     Dict[str, Callable] = field(default_factory=dict)
    skill_texts:  Dict[str, str]      = field(default_factory=dict)
    error:        str                 = ""
    loaded:       bool                = False


# ---------------------------------------------------------------------------
# Plugin manager
# ---------------------------------------------------------------------------

class PluginManager:
    """Manages loading, listing, and installing Operon plugins."""

    def __init__(self, plugins_dir: Optional[Path] = None) -> None:
        self._dir     = plugins_dir or PLUGINS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._hooks:   Dict[str, List[Callable]] = {}

    # ── Loading ────────────────────────────────────────────────────────────

    def load_all(self) -> int:
        """Load all plugins from the plugins directory. Returns count loaded."""
        loaded = 0
        for entry in sorted(self._dir.iterdir()):
            if entry.is_dir() and (entry / "plugin.json").exists():
                try:
                    plugin = self._load_plugin(entry)
                    if plugin.loaded:
                        loaded += 1
                except Exception as e:
                    log.warning("Failed to load plugin at %s: %s", entry, e)
        return loaded

    def load(self, name: str) -> Optional[LoadedPlugin]:
        """Load a specific plugin by name."""
        plugin_dir = self._dir / name
        if not plugin_dir.exists():
            log.error("Plugin directory not found: %s", plugin_dir)
            return None
        return self._load_plugin(plugin_dir)

    def _load_plugin(self, plugin_dir: Path) -> LoadedPlugin:
        manifest_path = plugin_dir / "plugin.json"
        try:
            raw      = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = PluginManifest.from_dict(raw)
        except Exception as e:
            log.warning("Failed to parse %s: %s", manifest_path, e)
            return LoadedPlugin(
                manifest=PluginManifest(name=plugin_dir.name),
                plugin_dir=plugin_dir,
                error=str(e),
            )

        plugin = LoadedPlugin(manifest=manifest, plugin_dir=plugin_dir)

        # Install pip deps (silently skip if already satisfied)
        if manifest.requires:
            self._install_deps(manifest.requires)

        # Load tools.py
        tools_file = plugin_dir / "tools.py"
        if tools_file.exists() and manifest.tools:
            try:
                spec   = importlib.util.spec_from_file_location(
                    f"operon_plugin_{manifest.name}", str(tools_file))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for fn_name in manifest.tools:
                    fn = getattr(module, fn_name, None)
                    if callable(fn):
                        plugin.tool_fns[fn_name] = fn
                    else:
                        log.warning("Plugin %s: tool %r not found in tools.py",
                                    manifest.name, fn_name)
            except Exception as e:
                plugin.error = f"tools.py import failed: {e}"
                log.warning("Plugin %s tools.py error: %s", manifest.name, e)

        # Load skill files
        skills_subdir = plugin_dir / "skills"
        for skill_file in manifest.skills:
            sf = (skills_subdir / skill_file
                  if not Path(skill_file).is_absolute()
                  else Path(skill_file))
            if sf.exists():
                try:
                    plugin.skill_texts[sf.stem] = sf.read_text(encoding="utf-8")
                except Exception as e:
                    log.warning("Plugin %s skill %s error: %s", manifest.name, skill_file, e)

        # Register hooks
        for hook_name, hook_script in manifest.hooks.items():
            hook_path = plugin_dir / hook_script
            if hook_path.exists():
                self._register_hook(hook_name, hook_path, manifest.name)

        plugin.loaded = True
        self._plugins[manifest.name] = plugin
        log.info("Plugin loaded: %s v%s (%d tools, %d skills)",
                 manifest.name, manifest.version,
                 len(plugin.tool_fns), len(plugin.skill_texts))
        return plugin

    # ── Installation ───────────────────────────────────────────────────────

    def install(self, source: str, name: str = "") -> Tuple[bool, str]:
        """
        Install a plugin from a local directory.

        Args:
            source : Path to a directory containing plugin.json
            name   : Override the plugin name (default: use manifest name)

        Returns:
            (success, message)
        """
        src = Path(source).expanduser().resolve()
        if not src.exists():
            return False, f"source not found: {src}"
        manifest_path = src / "plugin.json"
        if not manifest_path.exists():
            return False, f"plugin.json not found in {src}"

        try:
            raw      = json.loads(manifest_path.read_text())
            manifest = PluginManifest.from_dict(raw)
            dst_name = name or manifest.name
            dst      = self._dir / dst_name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            plugin = self._load_plugin(dst)
            if plugin.loaded:
                return True, f"Plugin '{dst_name}' installed ({len(plugin.tool_fns)} tools)"
            return False, f"Plugin installed but failed to load: {plugin.error}"
        except Exception as e:
            return False, str(e)

    def uninstall(self, name: str) -> Tuple[bool, str]:
        """Remove a plugin."""
        plugin_dir = self._dir / name
        if not plugin_dir.exists():
            return False, f"plugin '{name}' not found"
        try:
            shutil.rmtree(plugin_dir)
            self._plugins.pop(name, None)
            return True, f"Plugin '{name}' uninstalled"
        except Exception as e:
            return False, str(e)

    # ── Registry integration ───────────────────────────────────────────────

    def register_tools(self, registry=None) -> int:
        """
        Register all loaded plugin tools into the tool registry.

        Returns the number of tools registered.
        """
        if registry is None:
            try:
                from tools.registry import ToolRegistry
                registry = ToolRegistry()
            except Exception:
                return 0

        count = 0
        for plugin in self._plugins.values():
            for fn_name, fn in plugin.tool_fns.items():
                try:
                    registry.register_dynamic(fn_name, fn,
                                              description=fn.__doc__ or f"Plugin tool: {fn_name}")
                    count += 1
                except Exception as e:
                    log.warning("Failed to register plugin tool %s: %s", fn_name, e)
        return count

    # ── Hooks ──────────────────────────────────────────────────────────────

    def _register_hook(self, hook_name: str, hook_path: Path, plugin_name: str) -> None:
        def _runner(*args, **kwargs):
            try:
                subprocess.run([str(hook_path)], timeout=10, check=False)
            except Exception as e:
                log.debug("Hook %s from %s failed: %s", hook_name, plugin_name, e)
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(_runner)

    def run_hooks(self, hook_name: str, *args, **kwargs) -> None:
        """Fire all registered hooks for an event."""
        for fn in self._hooks.get(hook_name, []):
            try:
                fn(*args, **kwargs)
            except Exception as e:
                log.debug("Hook %s error: %s", hook_name, e)

    # ── Queries ────────────────────────────────────────────────────────────

    def list(self) -> List[Dict]:
        """List all loaded plugins."""
        result = []
        for plugin in self._plugins.values():
            m = plugin.manifest
            result.append({
                "name":        m.name,
                "version":     m.version,
                "description": m.description,
                "author":      m.author,
                "tools":       list(plugin.tool_fns.keys()),
                "skills":      list(plugin.skill_texts.keys()),
                "loaded":      plugin.loaded,
                "error":       plugin.error,
            })
        return result

    def get_all_skills(self) -> Dict[str, str]:
        """Return all skill texts from all loaded plugins."""
        all_skills: Dict[str, str] = {}
        for plugin in self._plugins.values():
            all_skills.update(plugin.skill_texts)
        return all_skills

    def get(self, name: str) -> Optional[LoadedPlugin]:
        return self._plugins.get(name)

    def __len__(self) -> int:
        return len(self._plugins)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _install_deps(requires: List[str]) -> None:
        """Silently pip-install missing dependencies."""
        for req in requires:
            try:
                pkg = req.split(">=")[0].split("==")[0].strip()
                importlib.import_module(pkg.replace("-", "_"))
            except ImportError:
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", req, "-q"],
                        timeout=60, check=False,
                    )
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Plugin scaffold generator
# ---------------------------------------------------------------------------

def create_plugin_scaffold(name: str, dest: Optional[Path] = None) -> Path:
    """
    Generate a boilerplate plugin directory structure.

    Args:
        name : Plugin name (lowercase, hyphen-separated)
        dest : Where to create the directory (default: cwd)

    Returns:
        Path to the created plugin directory.
    """
    dest = (dest or Path.cwd()) / name
    dest.mkdir(parents=True, exist_ok=True)

    # plugin.json
    (dest / "plugin.json").write_text(json.dumps({
        "name":        name,
        "version":     "0.1.0",
        "description": f"Description of {name}",
        "author":      "Your Name",
        "tools":       ["my_tool"],
        "skills":      ["my_skill.md"],
        "requires":    [],
    }, indent=2))

    # tools.py
    (dest / "tools.py").write_text(f'''"""Tools provided by the {name} plugin."""

def my_tool(param: str = "", **_) -> dict:
    """Example tool — replace with real implementation."""
    return {{"success": True, "result": f"my_tool called with {{param}}"}}
''')

    # skills/
    skills_dir = dest / "skills"
    skills_dir.mkdir(exist_ok=True)
    (skills_dir / "my_skill.md").write_text(f"""# {name.title()} Skill

## Description
What this skill does.

## Usage
When to use this skill.

## Steps
1. Step one
2. Step two
""")

    # README
    (dest / "README.md").write_text(f"""# {name.title()} Plugin for Operon

## Install
```bash
operon plugins install ./{name}
```

## Tools
- `my_tool` — example tool

## Skills
- `my_skill` — example skill
""")

    log.info("Plugin scaffold created at %s", dest)
    return dest


# ---------------------------------------------------------------------------
# Hot Reload
# ---------------------------------------------------------------------------

class PluginHotReloader:
    """
    Watch plugin directories for changes and reload automatically.
    Uses file mtime polling — no inotify / fsevents dependency required.
    """

    def __init__(self, manager: PluginManager, poll_interval: float = 2.0) -> None:
        self._manager  = manager
        self._interval = poll_interval
        self._mtimes:  Dict[str, float] = {}
        self._running  = False
        self._thread:  Optional[Any] = None

    def start(self, blocking: bool = False) -> None:
        """Start the hot-reload watcher."""
        import threading
        self._running = True
        self._snapshot_mtimes()

        def _loop() -> None:
            log.info("PluginHotReloader started (interval=%.1fs)", self._interval)
            while self._running:
                time.sleep(self._interval)
                self._check_changes()

        if blocking:
            _loop()
        else:
            self._thread = threading.Thread(
                target=_loop, daemon=True, name="operon-plugin-reloader"
            )
            self._thread.start()

    def stop(self) -> None:
        self._running = False

    def reload_now(self, name: str) -> Tuple[bool, str]:
        """Manually hot-reload a specific plugin."""
        plugin = self._manager.get(name)
        if not plugin:
            return False, f"plugin '{name}' not loaded"
        plugin_dir = plugin.plugin_dir
        try:
            # Remove old module from sys.modules to force reimport
            mod_key = f"operon_plugin_{name}"
            if mod_key in sys.modules:
                del sys.modules[mod_key]
            reloaded = self._manager._load_plugin(plugin_dir)
            if reloaded.loaded:
                log.info("Hot-reloaded plugin: %s", name)
                return True, f"Plugin '{name}' reloaded"
            return False, f"Reload failed: {reloaded.error}"
        except Exception as e:
            return False, str(e)

    def _snapshot_mtimes(self) -> None:
        for entry in self._manager._dir.iterdir():
            if entry.is_dir():
                mtime = self._dir_mtime(entry)
                self._mtimes[entry.name] = mtime

    def _check_changes(self) -> None:
        for entry in self._manager._dir.iterdir():
            if not entry.is_dir():
                continue
            name  = entry.name
            mtime = self._dir_mtime(entry)
            if mtime != self._mtimes.get(name, 0):
                self._mtimes[name] = mtime
                log.info("Plugin change detected: %s — reloading", name)
                ok, msg = self.reload_now(name)
                log.info("Hot reload %s: %s", name, msg)

    @staticmethod
    def _dir_mtime(path: Path) -> float:
        """Return the latest mtime across all files in a directory."""
        try:
            return max(
                (p.stat().st_mtime for p in path.rglob("*") if p.is_file()),
                default=0.0,
            )
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Plugin signing (SHA-256)
# ---------------------------------------------------------------------------

class PluginSigner:
    """
    Computes and verifies SHA-256 content hashes for plugin files.
    Not a PKI system — provides tamper detection for local/CI workflows.

    Signature format (plugin.sig.json):
        {
          "name": "my-plugin",
          "version": "1.0.0",
          "signed_at": "2025-01-01T00:00:00Z",
          "files": {
            "plugin.json": "<sha256>",
            "tools.py":    "<sha256>",
            ...
          }
        }
    """

    SIG_FILE = "plugin.sig.json"
    SKIP_FILES = {SIG_FILE, "__pycache__", ".DS_Store"}

    def sign(self, plugin_dir: Path) -> Dict[str, Any]:
        """
        Compute SHA-256 checksums for all plugin files.
        Writes plugin.sig.json and returns the signature dict.
        """
        plugin_dir = Path(plugin_dir)
        manifest   = self._load_manifest(plugin_dir)
        sig: Dict[str, Any] = {
            "name":      manifest.get("name", plugin_dir.name),
            "version":   manifest.get("version", "0.1.0"),
            "signed_at": self._iso_now(),
            "files":     {},
        }
        for fpath in sorted(plugin_dir.rglob("*")):
            if fpath.is_file() and fpath.name not in self.SKIP_FILES:
                rel = str(fpath.relative_to(plugin_dir))
                if "__pycache__" in rel:
                    continue
                sig["files"][rel] = self._sha256(fpath)

        sig_path = plugin_dir / self.SIG_FILE
        sig_path.write_text(json.dumps(sig, indent=2))
        log.info("Signed plugin %s: %d files", sig["name"], len(sig["files"]))
        return sig

    def verify(self, plugin_dir: Path) -> Tuple[bool, List[str]]:
        """
        Verify plugin files against plugin.sig.json.
        Returns (is_valid, list_of_errors).
        """
        plugin_dir = Path(plugin_dir)
        sig_path   = plugin_dir / self.SIG_FILE
        if not sig_path.exists():
            return False, ["plugin.sig.json not found — plugin is unsigned"]

        try:
            sig = json.loads(sig_path.read_text())
        except Exception as e:
            return False, [f"cannot read signature file: {e}"]

        errors: List[str] = []
        expected: Dict[str, str] = sig.get("files", {})

        # Check for missing or modified files
        for rel, expected_hash in expected.items():
            fpath = plugin_dir / rel
            if not fpath.exists():
                errors.append(f"missing: {rel}")
            elif self._sha256(fpath) != expected_hash:
                errors.append(f"modified: {rel}")

        # Check for new (unsigned) files
        for fpath in plugin_dir.rglob("*"):
            if fpath.is_file() and fpath.name not in self.SKIP_FILES:
                rel = str(fpath.relative_to(plugin_dir))
                if "__pycache__" in rel:
                    continue
                if rel not in expected:
                    errors.append(f"new unsigned file: {rel}")

        if errors:
            log.warning("Plugin %s verification FAILED: %s", plugin_dir.name, errors)
        else:
            log.info("Plugin %s verified OK (%d files)", plugin_dir.name, len(expected))

        return len(errors) == 0, errors

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def _load_manifest(plugin_dir: Path) -> Dict:
        mp = plugin_dir / "plugin.json"
        if mp.exists():
            try:
                return json.loads(mp.read_text())
            except Exception:
                pass
        return {}

    @staticmethod
    def _iso_now() -> str:
        import datetime
        return datetime.datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# Plugin Marketplace (GitHub tap registry)
# ---------------------------------------------------------------------------

@dataclass
class MarketplaceEntry:
    name:        str
    description: str
    author:      str
    version:     str = "latest"
    repo_url:    str = ""
    tags:        List[str] = field(default_factory=list)
    stars:       int = 0
    verified:    bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":        self.name,
            "description": self.description,
            "author":      self.author,
            "version":     self.version,
            "repo_url":    self.repo_url,
            "tags":        self.tags,
            "stars":       self.stars,
            "verified":    self.verified,
        }


class PluginMarketplace:
    """
    Plugin discovery and installation from GitHub tap registries.

    A "tap" is a GitHub repo containing a registry.json:
        [
          {
            "name": "my-plugin",
            "description": "...",
            "author": "github-user",
            "repo_url": "https://github.com/user/repo",
            "version": "1.0.0"
          }
        ]

    Default tap: https://raw.githubusercontent.com/operon-ai/plugin-registry/main/registry.json
    """

    DEFAULT_TAP = "https://raw.githubusercontent.com/operon-ai/plugin-registry/main/registry.json"

    def __init__(
        self,
        manager:     Optional[PluginManager] = None,
        cache_ttl:   int = 3600,   # seconds before re-fetching registry
    ) -> None:
        self._manager   = manager or PluginManager()
        self._taps:     List[str] = [self.DEFAULT_TAP]
        self._entries:  List[MarketplaceEntry] = []
        self._fetched_at: float = 0.0
        self._cache_ttl  = cache_ttl
        self._signer     = PluginSigner()
        self._verify_on_install = True

    def add_tap(self, registry_url: str) -> None:
        """Add a third-party registry URL."""
        if registry_url not in self._taps:
            self._taps.append(registry_url)
            self._entries.clear()  # invalidate cache

    def remove_tap(self, registry_url: str) -> None:
        self._taps = [t for t in self._taps if t != registry_url]
        self._entries.clear()

    def fetch(self, force: bool = False) -> int:
        """
        Fetch plugin listings from all taps.
        Returns total number of entries fetched.
        Silently skips unavailable taps.
        """
        now = time.time()
        if not force and self._entries and (now - self._fetched_at) < self._cache_ttl:
            return len(self._entries)

        self._entries.clear()
        for tap_url in self._taps:
            entries = self._fetch_tap(tap_url)
            self._entries.extend(entries)
            log.info("Fetched %d entries from %s", len(entries), tap_url)

        self._fetched_at = now
        return len(self._entries)

    def search(
        self,
        query:    str = "",
        tag:      str = "",
        verified: bool = False,
    ) -> List[MarketplaceEntry]:
        """Search the plugin registry."""
        if not self._entries:
            self.fetch()
        results = self._entries
        if query:
            q = query.lower()
            results = [e for e in results
                       if q in e.name.lower() or q in e.description.lower()
                       or q in e.author.lower()]
        if tag:
            results = [e for e in results if tag.lower() in [t.lower() for t in e.tags]]
        if verified:
            results = [e for e in results if e.verified]
        return sorted(results, key=lambda e: e.stars, reverse=True)

    def install_from_github(
        self,
        repo_url:      str,
        branch:        str = "main",
        verify_sig:    bool = True,
        plugin_name:   str = "",
    ) -> Tuple[bool, str]:
        """
        Clone or download a plugin from GitHub and install it.
        If git is available, uses git clone; otherwise falls back to zip download.
        Returns (success, message).
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Try git clone first
            if shutil.which("git"):
                result = subprocess.run(
                    ["git", "clone", "--depth=1", "--branch", branch, repo_url, str(tmpdir_path / "plugin")],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    # Try without branch (repo might use 'master')
                    result = subprocess.run(
                        ["git", "clone", "--depth=1", repo_url, str(tmpdir_path / "plugin")],
                        capture_output=True, text=True, timeout=120,
                    )
                if result.returncode != 0:
                    return False, f"git clone failed: {result.stderr[:200]}"
                plugin_src = tmpdir_path / "plugin"
            else:
                # Fallback: download zip
                ok, msg, plugin_src = self._download_zip(repo_url, branch, tmpdir_path)
                if not ok:
                    return False, msg

            # Verify signature if present
            if verify_sig and self._verify_on_install:
                valid, errors = self._signer.verify(plugin_src)
                if not valid:
                    log.warning("Plugin signature verification: %s", errors)
                    # Non-fatal warning — still install but log
                    log.warning("Installing unsigned/modified plugin from %s", repo_url)

            ok, msg = self._manager.install(str(plugin_src), name=plugin_name)
            return ok, msg

    def install_by_name(self, name: str) -> Tuple[bool, str]:
        """
        Install a plugin from the marketplace by name.
        Fetches registry if needed, then installs from GitHub.
        """
        if not self._entries:
            self.fetch()
        matches = [e for e in self._entries if e.name.lower() == name.lower()]
        if not matches:
            return False, f"plugin '{name}' not found in marketplace"
        entry = matches[0]
        if not entry.repo_url:
            return False, f"plugin '{name}' has no repo_url"
        return self.install_from_github(entry.repo_url)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _fetch_tap(self, url: str) -> List[MarketplaceEntry]:
        try:
            import requests
            resp = requests.get(url, timeout=10)
            if not resp.ok:
                log.warning("tap fetch failed %s: HTTP %d", url, resp.status_code)
                return []
            raw = resp.json()
            if not isinstance(raw, list):
                raw = raw.get("plugins", raw.get("entries", []))
            return [MarketplaceEntry(
                name        = e.get("name", ""),
                description = e.get("description", ""),
                author      = e.get("author", ""),
                version     = e.get("version", "latest"),
                repo_url    = e.get("repo_url", e.get("url", "")),
                tags        = e.get("tags", []),
                stars       = e.get("stars", 0),
                verified    = e.get("verified", False),
            ) for e in raw if e.get("name")]
        except Exception as e:
            log.warning("tap fetch error %s: %s", url, e)
            return []

    @staticmethod
    def _download_zip(
        repo_url: str, branch: str, dest: Path
    ) -> Tuple[bool, str, Path]:
        """Download and extract a GitHub zip archive."""
        # Convert https://github.com/user/repo → zip URL
        repo_url = repo_url.rstrip("/")
        if not repo_url.endswith(".git"):
            zip_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
        else:
            zip_url = repo_url.replace(".git", f"/archive/refs/heads/{branch}.zip")

        try:
            import requests, zipfile, io
            resp = requests.get(zip_url, timeout=60)
            if not resp.ok:
                return False, f"download failed: HTTP {resp.status_code}", dest
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            z.extractall(dest)
            # Extracted dir is usually repo-branch/
            dirs = [p for p in dest.iterdir() if p.is_dir()]
            if not dirs:
                return False, "zip extraction produced no directories", dest
            return True, "ok", dirs[0]
        except Exception as e:
            return False, str(e), dest


# ---------------------------------------------------------------------------
# Module-level default manager
# ---------------------------------------------------------------------------

_default_manager:     Optional[PluginManager]     = None
_default_marketplace: Optional[PluginMarketplace] = None
_default_signer:      Optional[PluginSigner]      = None


def get_manager() -> PluginManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = PluginManager()
    return _default_manager


def get_marketplace() -> PluginMarketplace:
    global _default_marketplace
    if _default_marketplace is None:
        _default_marketplace = PluginMarketplace(get_manager())
    return _default_marketplace


def get_signer() -> PluginSigner:
    global _default_signer
    if _default_signer is None:
        _default_signer = PluginSigner()
    return _default_signer
