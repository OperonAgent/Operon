"""
core/bootstrap.py — Operon dependency provisioner.

Solves the #1 fresh-clone problem: `pip install -r requirements.txt` installs
the Playwright *Python package* but NOT the Chromium *browser binary* (~120 MB),
which needs a separate `playwright install chromium` step that users never run.

This module:
  • Installs / verifies Python packages (core or full set).
  • Installs the Playwright Chromium browser binary (the missing piece).
  • Is idempotent — safe to run repeatedly; skips anything already present.
  • Works as an importable API, a CLI (`python -m core.bootstrap`), and as a
    runtime self-heal hook (browser tool calls ensure_browser_binary()).

Usage:
    python -m core.bootstrap            # interactive: install everything
    python -m core.bootstrap --full     # full optional feature set
    python -m core.bootstrap --check    # report status only, install nothing
    python -m core.bootstrap --browser  # only ensure the Chromium binary
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

# ── ANSI (degrade gracefully if not a TTY) ───────────────────────────────────
_TTY = sys.stdout.isatty()
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s
def _ok(s: str)   -> str: return _c("1;38;5;82",  f"  ✓ {s}")
def _warn(s: str) -> str: return _c("1;38;5;214", f"  ! {s}")
def _err(s: str)  -> str: return _c("1;38;5;196", f"  ✗ {s}")
def _info(s: str) -> str: return _c("1;38;5;81",  f"  → {s}")
def _head(s: str) -> str: return _c("1;38;5;141", s)

HERE = Path(__file__).resolve().parent.parent

# ── Package groups ────────────────────────────────────────────────────────────
# Core: the agent REPL refuses to start without these.
CORE_PACKAGES = [
    "requests>=2.31.0",
    "beautifulsoup4>=4.12.0",
    "duckduckgo_search>=6.0.0",
]

# Recommended: not strictly required, but expected by most users out of the box.
RECOMMENDED_PACKAGES = [
    "psutil>=5.9.8",          # telemetry (banner CPU/RAM)
    "pygments>=2.17.0",       # syntax highlighting
    "prompt_toolkit>=3.0.0",  # Claude Code-style TUI
    "pypdf>=4.0.0",           # PDF reading + RAG
    "reportlab>=4.0.0",       # PDF generation
    "playwright>=1.40.0",     # headless browser (binary handled separately)
    "paramiko>=3.4.0",        # SSH remote execution
]

# Full: every optional feature. Heavy (whisper pulls torch).
FULL_PACKAGES = RECOMMENDED_PACKAGES + [
    "mss>=9.0.0",             # fast cross-platform screenshots (computer use)
    "pynput>=1.7.6",          # mouse/keyboard automation (computer use)
    "mcp>=1.0.0",             # expose Operon as an MCP server
    "keyring>=24.0.0",        # OS keychain for secrets
    "cryptography>=41.0.0",   # Fernet encryption fallback
    "sounddevice>=0.4.6",     # voice recording
    "pyttsx3>=2.90",          # offline TTS
    "numpy>=1.26.0",          # vector math (RAG / vector memory)
]


# ── pip helpers ───────────────────────────────────────────────────────────────

def _pip_install(packages: List[str], upgrade: bool = False) -> bool:
    """Install a list of packages via the current interpreter's pip."""
    if not packages:
        return True
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd += packages
    print(_info(f"pip install {' '.join(packages[:3])}"
                f"{' …' if len(packages) > 3 else ''}"))
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(_err(f"pip install failed: {e}"))
        return False


def _is_importable(module: str) -> bool:
    import importlib.util
    # Map pip-name → import-name for the few that differ.
    alias = {
        "beautifulsoup4": "bs4",
        "duckduckgo_search": "duckduckgo_search",
        "psycopg2-binary": "psycopg2",
        "openai-whisper": "whisper",
        "py-cord": "discord",
        "slack-sdk": "slack_sdk",
        "Pillow": "PIL",
    }
    name = module.split(">=")[0].split("==")[0].split("[")[0].strip()
    imp  = alias.get(name, name.replace("-", "_"))
    try:
        return importlib.util.find_spec(imp) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


# ── Playwright browser binary (the critical bit) ──────────────────────────────

def is_browser_binary_installed() -> bool:
    """
    True if the Playwright Chromium *browser binary* is downloaded
    (separate from the playwright pip package).
    """
    if not _is_importable("playwright"):
        return False
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            exe = p.chromium.executable_path
            return bool(exe) and Path(exe).exists()
    except Exception:
        return False


def ensure_browser_binary(quiet: bool = False, with_deps: bool = None) -> Tuple[bool, str]:
    """
    Install the Playwright Chromium browser binary if missing. Idempotent.

    Returns (success, message). Safe to call at runtime — used by the browser
    tool to self-heal when a user tries to browse without the binary.
    """
    if not _is_importable("playwright"):
        ok = _pip_install(["playwright>=1.40.0"])
        if not ok:
            return False, "Could not install the playwright package."

    if is_browser_binary_installed():
        if not quiet:
            print(_ok("Chromium browser binary already installed"))
        return True, "already installed"

    if not quiet:
        print(_info("Downloading Chromium browser binary (~120 MB, one-time)…"))

    # On Linux, system libraries are often needed too.
    if with_deps is None:
        with_deps = sys.platform.startswith("linux")

    cmd = [sys.executable, "-m", "playwright", "install"]
    if with_deps:
        cmd.append("--with-deps")
    cmd.append("chromium")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0 and is_browser_binary_installed():
            if not quiet:
                print(_ok("Chromium browser binary installed"))
            return True, "installed"
        # --with-deps can fail without sudo; retry without it.
        if with_deps:
            if not quiet:
                print(_warn("--with-deps failed (needs sudo?); retrying without it"))
            result = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                check=False,
            )
            if result.returncode == 0 and is_browser_binary_installed():
                if not quiet:
                    print(_ok("Chromium browser binary installed"))
                    if sys.platform.startswith("linux"):
                        print(_warn("If browsing fails, run: "
                                    "sudo python -m playwright install-deps chromium"))
                return True, "installed (without system deps)"
        return False, "playwright install chromium failed"
    except Exception as e:
        return False, f"browser install error: {e}"


# ── Status report ─────────────────────────────────────────────────────────────

def check_status() -> dict:
    """Return a dict of what's installed vs missing (installs nothing)."""
    status = {"core": {}, "recommended": {}, "browser_binary": False, "python": sys.version.split()[0]}
    for pkg in CORE_PACKAGES:
        status["core"][pkg.split(">=")[0]] = _is_importable(pkg)
    for pkg in RECOMMENDED_PACKAGES:
        status["recommended"][pkg.split(">=")[0]] = _is_importable(pkg)
    status["browser_binary"] = is_browser_binary_installed()
    return status


def print_status() -> None:
    st = check_status()
    print(_head("\n  Operon dependency status"))
    print(_info(f"Python {st['python']}"))
    print()
    print(_head("  Core (required):"))
    for pkg, ok in st["core"].items():
        print(_ok(pkg) if ok else _err(f"{pkg}  — MISSING"))
    print()
    print(_head("  Recommended:"))
    for pkg, ok in st["recommended"].items():
        print(_ok(pkg) if ok else _warn(f"{pkg}  — not installed"))
    print()
    print(_head("  Browser binary (Chromium):"))
    if st["browser_binary"]:
        print(_ok("Chromium browser binary installed — browsing ready"))
    else:
        print(_warn("Chromium binary NOT installed — run: python -m core.bootstrap --browser"))
    print()


# ── Top-level provisioners ────────────────────────────────────────────────────

def provision(full: bool = False, browser: bool = True, upgrade: bool = False) -> bool:
    """
    Install everything Operon needs. Idempotent. Returns overall success.

    full=False → core + recommended (sensible default for most users)
    full=True  → every optional feature (heavy; pulls whisper/torch)
    browser    → also download the Chromium browser binary
    """
    print(_head("\n  ╭──────────────────────────────────────────────╮"))
    print(_head("  │   OPERON — Installing dependencies            │"))
    print(_head("  ╰──────────────────────────────────────────────╯\n"))

    pkgs = FULL_PACKAGES if full else (CORE_PACKAGES + RECOMMENDED_PACKAGES)

    # Only install what's missing (faster, clearer output) unless upgrading.
    if upgrade:
        to_install = pkgs
    else:
        to_install = [p for p in pkgs if not _is_importable(p)]
        already = [p for p in pkgs if _is_importable(p)]
        for p in already:
            print(_ok(f"{p.split('>=')[0]} already installed"))

    ok = True
    if to_install:
        print()
        ok = _pip_install(to_install, upgrade=upgrade)
        if not ok:
            print(_err("Some packages failed to install — see pip output above."))

    # The browser binary — the step everyone forgets.
    if browser:
        print()
        b_ok, b_msg = ensure_browser_binary()
        ok = ok and b_ok

    print()
    if ok:
        print(_ok("Bootstrap complete. Start Operon with:  operon   (or: python main.py)"))
    else:
        print(_warn("Bootstrap finished with warnings. Run  python -m core.bootstrap --check"))
    return ok


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="operon-bootstrap",
        description="Install Operon dependencies (packages + browser binary).",
    )
    ap.add_argument("--full",    action="store_true", help="Install every optional feature (heavy)")
    ap.add_argument("--check",   action="store_true", help="Report status only; install nothing")
    ap.add_argument("--browser", action="store_true", help="Only ensure the Chromium browser binary")
    ap.add_argument("--upgrade", action="store_true", help="Upgrade packages even if present")
    ap.add_argument("--no-browser", action="store_true", help="Skip the browser binary step")
    args = ap.parse_args()

    if args.check:
        print_status()
        return 0
    if args.browser:
        ok, msg = ensure_browser_binary()
        return 0 if ok else 1

    ok = provision(full=args.full, browser=not args.no_browser, upgrade=args.upgrade)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_cli())
