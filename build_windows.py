"""
build_windows.py — Instructions and helper script for Windows packaging.

Operon Windows build must be run on a Windows machine (or Windows VM / CI).
This script prints the full build checklist when run on Windows,
and on macOS/Linux just prints instructions.

Usage (on Windows):
    pip install pyinstaller
    python build_windows.py       # validates environment
    pyinstaller operon.spec       # produces dist/operon.exe
"""

from __future__ import annotations
import sys
import subprocess
from pathlib import Path

VERSION = "3.1.0"
HERE    = Path(__file__).parent.resolve()


WINDOWS_CHECKLIST = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                Operon v{VERSION} — Windows Build Checklist                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Prerequisites (run on a Windows machine):
  • Python 3.11+ (64-bit)  https://python.org/downloads/
  • pip install -r requirements.txt
  • pip install pyinstaller

Build steps:
  1. cd {HERE}
  2. pyinstaller operon.spec --noconfirm --clean
  3. → dist\\operon.exe  (single-file portable Windows binary)

Optional: create NSIS installer
  • Install NSIS  https://nsis.sourceforge.io/
  • makensis build_nsis.nsi  (see build_nsis.nsi template)

CI/CD (GitHub Actions):
  Use the workflow in .github/workflows/build.yml (already configured).
  Push a tag  vX.Y.Z  to trigger a multi-platform release build.
"""

MACOS_INSTRUCTIONS = f"""
Windows build must run on Windows.
To build for Windows:
  1. Use a Windows VM, GitHub Actions, or Wine-based environment
  2. Run: pyinstaller operon.spec --noconfirm
  3. The output: dist\\operon.exe

GitHub Actions workflow (.github/workflows/build.yml) automates this
when you push a git tag like 'v{VERSION}'.
"""


def validate_windows() -> None:
    """On Windows: check environment and report readiness."""
    issues = []

    # Python version
    if sys.version_info < (3, 9):
        issues.append(f"Python ≥3.9 required, found {sys.version}")

    # PyInstaller
    try:
        import PyInstaller  # noqa: F401
        print(f"  ✓ PyInstaller {PyInstaller.__version__}")
    except ImportError:
        issues.append("PyInstaller not installed  (pip install pyinstaller)")

    # Key requirements
    for pkg in ["requests", "prompt_toolkit"]:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            issues.append(f"{pkg} not installed")

    if issues:
        print("\n!  Issues found:")
        for i in issues:
            print(f"   • {i}")
        print("\nFix the above, then run: pyinstaller operon.spec")
    else:
        print("\n✓  Environment ready. Run: pyinstaller operon.spec")


def main() -> None:
    if sys.platform == "win32":
        print(WINDOWS_CHECKLIST)
        validate_windows()
    else:
        print(MACOS_INSTRUCTIONS)


if __name__ == "__main__":
    main()
