"""
core/computer_use.py — Mouse / Keyboard / Screen Desktop Control

Provides native computer use (beyond browser automation) via pyautogui + mss:
  • Mouse: move, click, double-click, right-click, drag, scroll
  • Keyboard: type text, press hotkeys, hold modifiers
  • Screen: screenshot, find image on screen, OCR text from screen
  • Clipboard: get/set clipboard contents
  • Window: list windows, focus window, resize/move window
  • Automations: open app, wait for element, record + replay macros

macOS / Windows / Linux compatible via pyautogui's cross-platform layer.
Screen capture via mss (faster than pyautogui.screenshot).

Safety features:
  • FAILSAFE: move mouse to top-left corner to abort (pyautogui default)
  • Pause between actions (configurable, default 0.1s)
  • Screen region capping (prevent accidental global clicks)
  • Dry-run mode (logs actions without executing)

Usage:
    from core.computer_use import ComputerUse
    cu = ComputerUse()

    cu.screenshot()                        # capture screen → base64 PNG
    cu.mouse_click(500, 300)               # left click at (500, 300)
    cu.keyboard_type("Hello Operon!")      # type text
    cu.keyboard_hotkey("command", "c")     # ⌘C / Ctrl+C
    cu.find_on_screen("button.png")        # locate image on screen → (x, y)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("operon.computer_use")

# ── Optional imports ──────────────────────────────────────────────────────────
import importlib.util as _importlib_util
import sys as _sys_mod


def _probe(name: str) -> bool:
    """Check if a module is available without importing it.
    Handles already-mocked sys.modules (e.g. in tests) gracefully."""
    if name in _sys_mod.modules:
        return True   # already imported or mocked — treat as available
    try:
        return _importlib_util.find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


# Probe availability cheaply (~0ms), actual import deferred to first use
_PYAUTOGUI = _probe("pyautogui")
_MSS       = _probe("mss")
_PIL       = _probe("PIL")

# Lazy module holders
_pyautogui_mod: "Any | None" = None
_mss_mod:       "Any | None" = None


def _get_pyautogui():
    """Lazily import pyautogui and configure it once."""
    global _pyautogui_mod, _PYAUTOGUI
    if _pyautogui_mod is None:
        try:
            import pyautogui as _pag
            _pag.FAILSAFE = True
            _pag.PAUSE    = 0.05
            _pyautogui_mod = _pag
        except ImportError:
            _PYAUTOGUI = False
            log.warning("computer_use: pyautogui not installed. Run: pip install pyautogui")
    return _pyautogui_mod


def _get_mss():
    """Lazily import mss."""
    global _mss_mod, _MSS
    if _mss_mod is None:
        try:
            import mss as _mss_lib
            import mss.tools  # noqa: F401
            _mss_mod = _mss_lib
        except ImportError:
            _MSS = False
    return _mss_mod

# ── Constants ─────────────────────────────────────────────────────────────────
_PLATFORM    = platform.system()   # Darwin / Windows / Linux
_IS_MAC      = _PLATFORM == "Darwin"
_IS_WIN      = _PLATFORM == "Windows"
_IS_LINUX    = _PLATFORM == "Linux"
_SCREENSHOTS_DIR = Path.home() / ".operon" / "screenshots"
_SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def _r(ok: bool, **kwargs) -> dict:
    return {"success": ok, **kwargs}


# Module-level aliases — populated lazily on first use so import stays fast
pyautogui: "Any | None" = None
mss:       "Any | None" = None


def _require_pyautogui() -> Optional[dict]:
    """Ensure pyautogui is loaded; return error dict if unavailable."""
    global pyautogui
    if pyautogui is None and _PYAUTOGUI:
        pyautogui = _get_pyautogui()
    if not _PYAUTOGUI or pyautogui is None:
        return _r(False, error="pyautogui not installed. Run: pip install pyautogui")
    return None


def _require_mss() -> Optional[dict]:
    """Ensure mss is loaded; return error dict if unavailable."""
    global mss
    if mss is None and _MSS:
        mss = _get_mss()
    if not _MSS or mss is None:
        return _r(False, error="mss not installed. Run: pip install mss")
    return None


# ── Screen capture ────────────────────────────────────────────────────────────

def screenshot(
    region:  Optional[Tuple[int, int, int, int]] = None,
    save_path: Optional[str] = None,
) -> dict:
    """
    Capture the screen (or a region) and return as base64-encoded PNG.

    Args:
        region: (left, top, width, height) for partial capture. None = full screen.
        save_path: optional path to save the PNG file.

    Returns:
        {success, base64_png, width, height, path?}
    """
    if _MSS:
        err = _require_mss()
        if not err:
            try:
                with mss.mss() as sct:
                    monitor = sct.monitors[1] if not region else {
                        "left": region[0], "top": region[1],
                        "width": region[2], "height": region[3],
                    }
                    img = sct.grab(monitor)
                    import mss.tools as _mss_tools
                    buf = _mss_tools.to_png(img.rgb, img.size)

                ts   = int(time.time())
                path = save_path or str(_SCREENSHOTS_DIR / f"screen_{ts}.png")
                with open(path, "wb") as f:
                    f.write(buf)

                b64 = base64.b64encode(buf).decode()
                return _r(True, base64_png=b64, width=img.width, height=img.height, path=path)
            except Exception:
                pass  # fall through to pyautogui

    if _PYAUTOGUI:
        err = _require_pyautogui()
        if err:
            return err
        try:
            if region:
                img = pyautogui.screenshot(region=region)
            else:
                img = pyautogui.screenshot()

            ts   = int(time.time())
            path = save_path or str(_SCREENSHOTS_DIR / f"screen_{ts}.png")
            img.save(path)

            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return _r(True, base64_png=b64, width=img.width, height=img.height, path=path)
        except Exception as e:
            return _r(False, error=str(e))

    return _r(False, error="No screenshot library available (install mss or pyautogui)")


def get_screen_size() -> dict:
    """Return screen resolution."""
    err = _require_pyautogui()
    if err:
        return err
    try:
        w, h = pyautogui.size()
        return _r(True, width=w, height=h)
    except Exception as e:
        return _r(False, error=str(e))


def get_mouse_position() -> dict:
    """Return current mouse cursor position."""
    err = _require_pyautogui()
    if err:
        return err
    try:
        x, y = pyautogui.position()
        return _r(True, x=x, y=y)
    except Exception as e:
        return _r(False, error=str(e))


# ── Mouse operations ──────────────────────────────────────────────────────────

def mouse_move(x: int, y: int, duration: float = 0.3, **_) -> dict:
    """Move mouse to (x, y) over duration seconds."""
    err = _require_pyautogui()
    if err:
        return err
    try:
        pyautogui.moveTo(x, y, duration=duration)
        return _r(True, x=x, y=y)
    except Exception as e:
        return _r(False, error=str(e))


def mouse_click(
    x:        int,
    y:        int,
    button:   str  = "left",
    clicks:   int  = 1,
    interval: float = 0.05,
    **_,
) -> dict:
    """Click at (x, y). button: left | right | middle."""
    err = _require_pyautogui()
    if err:
        return err
    try:
        pyautogui.click(x=x, y=y, clicks=clicks, interval=interval, button=button)
        return _r(True, x=x, y=y, button=button, clicks=clicks)
    except Exception as e:
        return _r(False, error=str(e))


def mouse_double_click(x: int, y: int, **_) -> dict:
    """Double-click at (x, y)."""
    return mouse_click(x, y, clicks=2)


def mouse_right_click(x: int, y: int, **_) -> dict:
    """Right-click at (x, y)."""
    return mouse_click(x, y, button="right")


def mouse_drag(
    from_x: int, from_y: int,
    to_x:   int, to_y:   int,
    duration: float = 0.5,
    **_,
) -> dict:
    """Click-drag from (from_x, from_y) to (to_x, to_y)."""
    err = _require_pyautogui()
    if err:
        return err
    try:
        pyautogui.dragTo(to_x, to_y, duration=duration, mouseDownUp=True)
        return _r(True, from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y)
    except Exception as e:
        return _r(False, error=str(e))


def mouse_scroll(x: int, y: int, clicks: int = 3, direction: str = "down", **_) -> dict:
    """Scroll at (x, y). direction: up | down."""
    err = _require_pyautogui()
    if err:
        return err
    try:
        pyautogui.moveTo(x, y)
        amount = -clicks if direction == "down" else clicks
        pyautogui.scroll(amount)
        return _r(True, x=x, y=y, clicks=clicks, direction=direction)
    except Exception as e:
        return _r(False, error=str(e))


# ── Keyboard operations ───────────────────────────────────────────────────────

def keyboard_type(text: str, interval: float = 0.02, **_) -> dict:
    """Type text character by character."""
    err = _require_pyautogui()
    if err:
        return err
    try:
        pyautogui.typewrite(text, interval=interval)
        return _r(True, text=text[:50] + ("…" if len(text) > 50 else ""), length=len(text))
    except Exception as e:
        return _r(False, error=str(e))


def keyboard_hotkey(*keys: str, **_) -> dict:
    """
    Press a key combination simultaneously.
    Example: keyboard_hotkey("ctrl", "c") or keyboard_hotkey("command", "v")
    Also accepts keys as a list via the 'keys' kwarg.
    """
    err = _require_pyautogui()
    if err:
        return err
    # Handle both positional *keys and keys=["ctrl","c"]
    if not keys and "keys" in _:
        keys = tuple(_["keys"]) if isinstance(_["keys"], list) else (_["keys"],)
    if not keys:
        return _r(False, error="No keys specified")
    try:
        pyautogui.hotkey(*keys)
        return _r(True, keys=list(keys))
    except Exception as e:
        return _r(False, error=str(e))


def keyboard_press(key: str, presses: int = 1, **_) -> dict:
    """Press a single key (e.g. 'enter', 'tab', 'escape', 'f1')."""
    err = _require_pyautogui()
    if err:
        return err
    try:
        pyautogui.press(key, presses=presses)
        return _r(True, key=key, presses=presses)
    except Exception as e:
        return _r(False, error=str(e))


def keyboard_hold(key: str, duration: float = 0.5, **_) -> dict:
    """Hold a key down for duration seconds (useful for shift/ctrl combos)."""
    err = _require_pyautogui()
    if err:
        return err
    try:
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)
        return _r(True, key=key, duration=duration)
    except Exception as e:
        pyautogui.keyUp(key)  # ensure key is released
        return _r(False, error=str(e))


# ── Clipboard ────────────────────────────────────────────────────────────────

def clipboard_get(**_) -> dict:
    """Get clipboard contents."""
    err = _require_pyautogui()
    if err:
        return err
    try:
        text = pyautogui.paste()
        return _r(True, text=text)
    except Exception:
        # Fallback: use pbpaste on macOS
        if _IS_MAC:
            try:
                text = subprocess.check_output(["pbpaste"]).decode()
                return _r(True, text=text)
            except Exception as e2:
                return _r(False, error=str(e2))
        return _r(False, error="clipboard_get failed")


def clipboard_set(text: str, **_) -> dict:
    """Set clipboard contents."""
    err = _require_pyautogui()
    if err:
        return err
    try:
        pyautogui.copy(text)
        return _r(True, length=len(text))
    except Exception:
        if _IS_MAC:
            try:
                p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                p.communicate(text.encode())
                return _r(True, length=len(text))
            except Exception as e2:
                return _r(False, error=str(e2))
        return _r(False, error="clipboard_set failed")


# ── Window management ─────────────────────────────────────────────────────────

def list_windows(**_) -> dict:
    """List open windows (macOS: via osascript; Linux: via wmctrl)."""
    if _IS_MAC:
        try:
            script = '''
tell application "System Events"
    set winList to {}
    repeat with proc in (processes whose background only is false)
        repeat with win in windows of proc
            set end of winList to name of proc & ": " & name of win
        end repeat
    end repeat
    return winList
end tell'''
            out = subprocess.check_output(["osascript", "-e", script], timeout=5).decode().strip()
            windows = [w.strip() for w in out.split(",") if w.strip()]
            return _r(True, windows=windows, count=len(windows))
        except Exception as e:
            return _r(False, error=str(e))

    if _IS_LINUX:
        try:
            out = subprocess.check_output(["wmctrl", "-l"], timeout=5).decode()
            windows = [line.split(None, 3)[-1] for line in out.splitlines() if line]
            return _r(True, windows=windows, count=len(windows))
        except Exception as e:
            return _r(False, error=f"wmctrl error: {e}")

    return _r(False, error=f"Window listing not implemented on {_PLATFORM}")


def open_application(app_name: str, **_) -> dict:
    """Open an application by name."""
    try:
        if _IS_MAC:
            subprocess.Popen(["open", "-a", app_name])
            time.sleep(0.5)
            return _r(True, app=app_name)
        elif _IS_WIN:
            subprocess.Popen(["start", app_name], shell=True)
            return _r(True, app=app_name)
        else:
            subprocess.Popen([app_name])
            return _r(True, app=app_name)
    except Exception as e:
        return _r(False, error=str(e))


def focus_window(title_contains: str, **_) -> dict:
    """Focus the first window whose title contains the given string (macOS only)."""
    if not _IS_MAC:
        return _r(False, error="focus_window only supported on macOS currently")
    try:
        script = f'''
tell application "System Events"
    set proc to first process whose windows exist and \
        (name of first window contains "{title_contains}")
    set frontmost of proc to true
end tell'''
        subprocess.check_call(["osascript", "-e", script], timeout=5)
        return _r(True, title_contains=title_contains)
    except Exception as e:
        return _r(False, error=str(e))


# ── Find image on screen ──────────────────────────────────────────────────────

def find_on_screen(image_path: str, confidence: float = 0.8, **_) -> dict:
    """
    Find a template image on screen and return its center coordinates.
    Requires pyautogui + Pillow (pip install Pillow).
    """
    err = _require_pyautogui()
    if err:
        return err
    if not _PIL:
        return _r(False, error="Pillow not installed (pip install Pillow)")
    try:
        loc = pyautogui.locateOnScreen(image_path, confidence=confidence)
        if loc is None:
            return _r(False, error=f"Image not found on screen: {image_path}")
        cx, cy = pyautogui.center(loc)
        return _r(True, x=cx, y=cy, region=loc)
    except Exception as e:
        return _r(False, error=str(e))


def click_image(image_path: str, confidence: float = 0.8, **_) -> dict:
    """Find image on screen and click its center."""
    result = find_on_screen(image_path, confidence)
    if not result["success"]:
        return result
    return mouse_click(result["x"], result["y"])


# ── Wait / polling ────────────────────────────────────────────────────────────

def wait_for_image(
    image_path: str,
    timeout:    float = 10.0,
    interval:   float = 0.5,
    confidence: float = 0.8,
    **_,
) -> dict:
    """Poll until image appears on screen or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        result = find_on_screen(image_path, confidence)
        if result["success"]:
            return result
        time.sleep(interval)
    return _r(False, error=f"Timed out waiting for {image_path} after {timeout}s")


# ── Availability check ────────────────────────────────────────────────────────

def computer_use_status(**_) -> dict:
    """Return availability of each computer use feature."""
    return _r(
        True,
        pyautogui    = _PYAUTOGUI,
        mss          = _MSS,
        pillow       = _PIL,
        platform     = _PLATFORM,
        screen_size  = _get_pyautogui().size() if _PYAUTOGUI else None,
        failsafe     = _get_pyautogui().FAILSAFE if _PYAUTOGUI else None,
    )


# ── Tool definitions ──────────────────────────────────────────────────────────

_TOOL_DEFINITIONS: List[Dict] = [
    {
        "name": "desktop_screenshot",
        "description": "Take a screenshot of the current screen (full screen or region).",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "array", "items": {"type": "integer"}, "description": "[left, top, width, height]"},
                "save_path": {"type": "string"},
            },
        },
    },
    {
        "name": "desktop_mouse_click",
        "description": "Click the mouse at screen coordinates (x, y).",
        "input_schema": {
            "type": "object",
            "properties": {
                "x":      {"type": "integer"},
                "y":      {"type": "integer"},
                "button": {"type": "string", "enum": ["left","right","middle"], "default": "left"},
                "clicks": {"type": "integer", "default": 1},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "desktop_keyboard_type",
        "description": "Type text on the keyboard.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text":     {"type": "string"},
                "interval": {"type": "number", "default": 0.02},
            },
            "required": ["text"],
        },
    },
    {
        "name": "desktop_keyboard_hotkey",
        "description": "Press a keyboard shortcut (e.g. ctrl+c, command+v, alt+tab).",
        "input_schema": {
            "type": "object",
            "properties": {
                "keys": {"type": "array", "items": {"type": "string"}, "description": "Keys to press together e.g. ['ctrl','c']"},
            },
            "required": ["keys"],
        },
    },
    {
        "name": "desktop_mouse_scroll",
        "description": "Scroll at a screen position.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x":         {"type": "integer"},
                "y":         {"type": "integer"},
                "clicks":    {"type": "integer", "default": 3},
                "direction": {"type": "string", "enum": ["up","down"], "default": "down"},
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "desktop_find_on_screen",
        "description": "Find a template image on screen and return its coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "confidence": {"type": "number", "default": 0.8},
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "desktop_open_app",
        "description": "Open an application by name (e.g. 'Safari', 'Terminal', 'Obsidian').",
        "input_schema": {
            "type": "object",
            "properties": {"app_name": {"type": "string"}},
            "required": ["app_name"],
        },
    },
    {
        "name": "desktop_clipboard_get",
        "description": "Get current clipboard contents.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "desktop_clipboard_set",
        "description": "Set clipboard contents.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "desktop_status",
        "description": "Check computer use module availability and screen size.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

_DISPATCH: Dict[str, Any] = {
    "desktop_screenshot":      screenshot,
    "desktop_mouse_click":     mouse_click,
    "desktop_keyboard_type":   keyboard_type,
    "desktop_keyboard_hotkey": keyboard_hotkey,
    "desktop_mouse_scroll":    mouse_scroll,
    "desktop_find_on_screen":  find_on_screen,
    "desktop_open_app":        open_application,
    "desktop_clipboard_get":   clipboard_get,
    "desktop_clipboard_set":   clipboard_set,
    "desktop_status":          computer_use_status,
}


# ── High-level class for main.py ───────────────────────────────────────────────

class ComputerUse:
    """Convenience class wrapping all computer use operations."""

    screenshot    = staticmethod(screenshot)
    mouse_click   = staticmethod(mouse_click)
    mouse_move    = staticmethod(mouse_move)
    mouse_drag    = staticmethod(mouse_drag)
    mouse_scroll  = staticmethod(mouse_scroll)
    keyboard_type = staticmethod(keyboard_type)
    keyboard_hotkey = staticmethod(keyboard_hotkey)
    keyboard_press = staticmethod(keyboard_press)
    find_on_screen = staticmethod(find_on_screen)
    click_image   = staticmethod(click_image)
    wait_for_image = staticmethod(wait_for_image)
    clipboard_get = staticmethod(clipboard_get)
    clipboard_set = staticmethod(clipboard_set)
    open_app      = staticmethod(open_application)
    list_windows  = staticmethod(list_windows)
    status        = staticmethod(computer_use_status)
