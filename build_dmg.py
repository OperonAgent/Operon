"""
build_dmg.py — Create a macOS DMG installer for Operon.

Usage:
    pip install dmgbuild
    python build_dmg.py

Produces: dist/Operon-3.1.0.dmg (drag-to-Applications installer)

Prerequisites:
    1. pyinstaller operon.spec  (builds dist/Operon.app)
    2. pip install dmgbuild
"""

from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path

VERSION   = "3.1.0"
HERE      = Path(__file__).parent.resolve()
APP_PATH  = HERE / "dist" / "Operon.app"
DMG_PATH  = HERE / "dist" / f"Operon-{VERSION}.dmg"

# ── dmgbuild settings ─────────────────────────────────────────────────────────
SETTINGS = {
    "filename":     str(DMG_PATH),
    "volume_name":  "Operon",
    "format":       "UDZO",
    "compression_level": 9,
    "size":         None,  # auto
    "files":        [str(APP_PATH)],
    "symlinks":     {"Applications": "/Applications"},
    "icon_locations": {
        "Operon.app":   (150, 180),
        "Applications": (450, 180),
    },
    "background": "builtin-arrow",
    "icon_size":   128,
    "text_size":    12,
    "window_rect": ((200, 200), (640, 400)),
}

_SETTINGS_FILE = HERE / "build" / "_dmgbuild_settings.py"


def _write_settings() -> None:
    _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "from __future__ import unicode_literals",
        f"filename    = {SETTINGS['filename']!r}",
        f"volume_name = {SETTINGS['volume_name']!r}",
        f"format      = {SETTINGS['format']!r}",
        f"size        = None",
        f"files       = {SETTINGS['files']!r}",
        f"symlinks    = {SETTINGS['symlinks']!r}",
        f"icon_locations = {SETTINGS['icon_locations']!r}",
        f"background  = {SETTINGS['background']!r}",
        f"icon_size   = {SETTINGS['icon_size']!r}",
        f"text_size   = {SETTINGS['text_size']!r}",
        f"window_rect = {SETTINGS['window_rect']!r}",
    ]
    _SETTINGS_FILE.write_text("\n".join(lines) + "\n")


def build() -> None:
    if sys.platform != "darwin":
        print("✗  DMG builds only work on macOS.")
        sys.exit(1)

    if not APP_PATH.exists():
        print(f"✗  {APP_PATH} not found. Run  pyinstaller operon.spec  first.")
        sys.exit(1)

    try:
        import dmgbuild  # noqa: F401
    except ImportError:
        print("✗  dmgbuild not installed. Run: pip install dmgbuild")
        sys.exit(1)

    print(f"  Building {DMG_PATH.name} …")
    _write_settings()

    result = subprocess.run(
        ["dmgbuild", "-s", str(_SETTINGS_FILE),
         SETTINGS["volume_name"], str(DMG_PATH)],
        capture_output=False,
    )

    if result.returncode == 0 and DMG_PATH.exists():
        size_mb = DMG_PATH.stat().st_size / 1024 / 1024
        print(f"\n✓  Built: {DMG_PATH}  ({size_mb:.0f} MB)")
        print(f"   Drag Operon.app to Applications to install.")
    else:
        print(f"\n✗  DMG build failed (exit {result.returncode})")
        sys.exit(1)


if __name__ == "__main__":
    build()
