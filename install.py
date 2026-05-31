#!/usr/bin/env python3
"""
install.py — Operon one-shot installer.

The single command a fresh GitHub cloner runs:

    git clone https://github.com/OperonAgent/Operon.git
    cd operon
    python install.py          # installs EVERYTHING, including the browser binary

What it does (idempotent, safe to re-run):
  1. Verifies Python >= 3.9.
  2. Optionally creates a virtual environment (.venv) and re-execs inside it.
  3. Installs Operon + recommended dependencies (pip install -e .).
  4. Downloads the Playwright Chromium browser binary (~120 MB) — the step
     `pip install` does NOT do and everyone forgets.
  5. Installs Operon as the `operon` command (editable mode).
  6. Prints next steps.

Flags:
    python install.py --full        # every optional feature (voice, db, …)
    python install.py --no-venv     # install into the current environment
    python install.py --no-browser  # skip the Chromium download
    python install.py --check       # report status only
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_DIR = HERE / ".venv"

# ── ANSI ──────────────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()
def _c(code, s): return f"\033[{code}m{s}\033[0m" if _TTY else s
def ok(s):   return _c("1;38;5;82",  f"  ✓ {s}")
def warn(s): return _c("1;38;5;214", f"  ! {s}")
def err(s):  return _c("1;38;5;196", f"  ✗ {s}")
def info(s): return _c("1;38;5;81",  f"  → {s}")
def head(s): return _c("1;38;5;141", s)


def _python_ok() -> bool:
    if sys.version_info < (3, 9):
        print(err(f"Python 3.9+ required. You have {sys.version.split()[0]}."))
        print(info("Install a newer Python from https://python.org/downloads/"))
        return False
    print(ok(f"Python {sys.version.split()[0]}"))
    return True


def _in_venv() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("VIRTUAL_ENV") is not None
    )


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _create_and_reexec_venv(extra_args: list) -> None:
    """Create .venv and re-run this script inside it."""
    if not VENV_DIR.exists():
        print(info(f"Creating virtual environment at {VENV_DIR} …"))
        import venv
        venv.create(str(VENV_DIR), with_pip=True)
        print(ok("Virtual environment created"))
    else:
        print(ok(f"Using existing virtual environment at {VENV_DIR}"))

    vpy = _venv_python()
    if not vpy.exists():
        print(err(f"venv python not found at {vpy}; falling back to current env"))
        return  # caller continues in current interpreter

    print(info("Re-running installer inside the virtual environment …\n"))
    # Mark so the child doesn't recurse into venv creation again.
    env = dict(os.environ, OPERON_INSTALL_IN_VENV="1")
    result = subprocess.run([str(vpy), str(HERE / "install.py"),
                             "--no-venv", *extra_args], env=env)
    # Print activation hint and exit with child's status.
    if os.name == "nt":
        act = f"{VENV_DIR}\\Scripts\\activate"
    else:
        act = f"source {VENV_DIR}/bin/activate"
    print()
    print(head("  To use Operon, activate the environment first:"))
    print(info(act))
    print(info("then run:  operon"))
    sys.exit(result.returncode)


def _pip(args: list) -> bool:
    cmd = [sys.executable, "-m", "pip", *args]
    return subprocess.run(cmd, check=False).returncode == 0


def main() -> int:
    raw = sys.argv[1:]
    full       = "--full" in raw
    no_venv    = "--no-venv" in raw or os.environ.get("OPERON_INSTALL_IN_VENV")
    no_browser = "--no-browser" in raw
    check_only = "--check" in raw

    print(head("\n  ╭────────────────────────────────────────────────╮"))
    print(head("  │   OPERON INSTALLER  •  AI Terminal Cockpit      │"))
    print(head("  ╰────────────────────────────────────────────────╯\n"))

    if not _python_ok():
        return 1

    # ── Status-only mode ──────────────────────────────────────────────────────
    if check_only:
        sys.path.insert(0, str(HERE))
        from core.bootstrap import print_status
        print_status()
        return 0

    # ── Virtual environment ────────────────────────────────────────────────────
    if not no_venv and not _in_venv():
        # Forward all flags except --no-venv to the child.
        fwd = [a for a in raw if a != "--no-venv"]
        _create_and_reexec_venv(fwd)
        return 0  # _create_and_reexec_venv exits

    if _in_venv():
        print(ok("Running inside a virtual environment"))
    else:
        print(warn("Installing into the system/global environment (no venv)"))

    # ── Upgrade pip ────────────────────────────────────────────────────────────
    print(info("Upgrading pip …"))
    _pip(["install", "--upgrade", "pip"])

    # ── Install Operon itself (editable) + recommended/full extras ─────────────
    print(info("Installing Operon and dependencies …"))
    extra = "[full]" if full else ""
    if not _pip(["install", "-e", f".{extra}"]):
        # Fallback: requirements.txt if editable install fails.
        print(warn("Editable install failed; falling back to requirements.txt"))
        _pip(["install", "-r", "requirements.txt"])

    # ── Browser binary + recommended packages via bootstrap ────────────────────
    sys.path.insert(0, str(HERE))
    try:
        from core.bootstrap import provision
        provision(full=full, browser=not no_browser, upgrade=False)
    except Exception as e:
        print(warn(f"bootstrap step had an issue: {e}"))
        print(info("You can retry with:  python -m core.bootstrap"))

    # ── Done ────────────────────────────────────────────────────────────────────
    print(head("\n  ╭────────────────────────────────────────────────╮"))
    print(head("  │   INSTALL COMPLETE                              │"))
    print(head("  ╰────────────────────────────────────────────────╯\n"))
    print(ok("Start Operon:        operon        (or: python main.py)"))
    print(ok("First-run setup:     runs automatically on first launch"))
    print(ok("Verify everything:   operon  then type  /doctor"))
    print(ok("Re-check deps:       python -m core.bootstrap --check"))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
