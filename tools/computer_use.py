"""
Operon Computer Use Tool — macOS desktop control.

Adapted from Hermes Agent computer_use/tool.py architecture.

Provides cross-platform GUI automation via:
  - mss   (screen capture — fast, zero deps on macOS/Linux/Windows)
  - pynput (mouse + keyboard control)
  - Pillow (image processing for SOM element detection)

Supported actions:
  capture      — screenshot (returns base64 PNG + element summary)
  click        — left/right/middle click at (x,y) or element index
  double_click — double-click
  right_click  — context menu click
  drag         — click-and-drag from src to dst
  scroll       — scroll wheel at position
  type         — type text (slow key-by-key with human delay)
  key          — press key or combo (cmd+c, ctrl+shift+esc, etc.)
  wait         — sleep N seconds
  list_apps    — list running application names (macOS only via osascript)
  focus_app    — bring an app window to front (macOS only)

Safety:
  - Blocked key combos: cmd+shift+q (logout), cmd+ctrl+q (lock), :(){ :|:& }; (fork bomb)
  - Blocked text patterns: curl|bash, rm -rf /, wget|bash
  - Approval callback: pluggable (auto-approve / auto-deny / interactive)
  - Hard-blocked actions can never be executed regardless of approval

Install:
    pip install mss pynput Pillow

Note: On macOS you may need to grant Accessibility + Screen Recording permissions
to Terminal / your terminal emulator in System Settings > Privacy & Security.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("operon.computer_use")

SCREENSHOTS_DIR = Path.home() / ".operon" / "screenshots"

# ---------------------------------------------------------------------------
# Safety: blocked key combos and text patterns
# ---------------------------------------------------------------------------

_KEY_ALIASES = {
    "command": "cmd", "control": "ctrl", "alt": "option",
    "⌘": "cmd", "⌥": "option", "⇧": "shift",
}

_BLOCKED_KEY_COMBOS: List[frozenset] = [
    frozenset({"cmd", "shift", "q"}),            # log out
    frozenset({"cmd", "ctrl", "q"}),             # lock screen
    frozenset({"cmd", "option", "shift", "q"}),  # force log out
    frozenset({"cmd", "shift", "backspace"}),    # empty trash
]

_BLOCKED_TYPE_PATTERNS: List[re.Pattern] = [
    re.compile(r"curl\s+[^|]*\|\s*bash", re.I),
    re.compile(r"curl\s+[^|]*\|\s*sh", re.I),
    re.compile(r"wget\s+[^|]*\|\s*bash", re.I),
    re.compile(r"\bsudo\s+rm\s+-[rf]", re.I),
    re.compile(r"\brm\s+-rf\s+/\s*$", re.I),
    re.compile(r":\s*\(\)\s*\{\s*:\|:\s*&\s*\}", re.I),   # fork bomb
]

_SAFE_ACTIONS = frozenset({"capture", "wait", "list_apps"})
_MUTATING_ACTIONS = frozenset({
    "click", "double_click", "right_click", "drag",
    "scroll", "type", "key", "focus_app",
})


def _canon_key(k: str) -> str:
    k = k.strip().lower()
    return _KEY_ALIASES.get(k, k)


def _canon_combo(keys: str) -> frozenset:
    return frozenset(_canon_key(p) for p in re.split(r"\s*\+\s*", keys) if p.strip())


def _check_type_safety(text: str) -> Optional[str]:
    for pat in _BLOCKED_TYPE_PATTERNS:
        if pat.search(text):
            return f"blocked text pattern: {pat.pattern!r}"
    return None


def _check_key_safety(keys: str) -> Optional[str]:
    combo = _canon_combo(keys)
    for blocked in _BLOCKED_KEY_COMBOS:
        if blocked.issubset(combo):
            return f"blocked key combo: {sorted(blocked)}"
    return None


# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------

_approval_callback = None
_session_auto_approve = False
_always_allow: set = set()


def set_approval_callback(cb) -> None:
    """Register a callback(action, args, summary) → 'approve_once' | 'approve_session' | 'always_approve' | 'deny'."""
    global _approval_callback
    _approval_callback = cb


def _request_approval(action: str, args: Dict) -> Optional[str]:
    """Return None if approved, or a JSON-encoded error string if denied."""
    global _session_auto_approve, _always_allow
    if _session_auto_approve or action in _always_allow:
        return None
    cb = _approval_callback
    if cb is None:
        return None   # No callback wired: default allow
    summary = _action_summary(action, args)
    try:
        verdict = cb(action, args, summary)
    except Exception:
        verdict = "deny"
    if verdict in ("approve_once",):
        return None
    if verdict in ("approve_session", "always_approve"):
        _always_allow.add(action)
        if verdict == "always_approve":
            _session_auto_approve = True
        return None
    return json.dumps({"error": "denied by user", "action": action})


def _action_summary(action: str, args: Dict) -> str:
    if action in ("click", "double_click", "right_click"):
        return f"{action} at ({args.get('x')},{args.get('y')})" + (
            f" element {args['element']}" if args.get("element") is not None else "")
    if action == "drag":
        return f"drag ({args.get('from_x')},{args.get('from_y')}) → ({args.get('to_x')},{args.get('to_y')})"
    if action == "type":
        t = args.get("text", "")
        return f"type {t[:60]!r}" + ("…" if len(t) > 60 else "")
    if action == "key":
        return f"key {args.get('keys', '')!r}"
    if action == "focus_app":
        return f"focus_app {args.get('app', '')!r}"
    return action


# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------

def _capture_screen(region: Optional[Dict] = None) -> Optional[bytes]:
    """Capture screen or a region to PNG bytes."""
    try:
        import mss
        import mss.tools
        with mss.mss() as sct:
            mon = region or sct.monitors[0]
            shot = sct.grab(mon)
            return mss.tools.to_png(shot.rgb, shot.size)
    except ImportError:
        pass

    # Fallback: macOS screencapture
    if sys.platform == "darwin":
        try:
            path = str(SCREENSHOTS_DIR / f"cua_{uuid.uuid4().hex[:8]}.png")
            SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(["screencapture", "-x", path], check=True, timeout=10)
            return Path(path).read_bytes()
        except Exception:
            pass
    return None


def _png_to_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode()


# ---------------------------------------------------------------------------
# Mouse / keyboard control (pynput)
# ---------------------------------------------------------------------------

def _mouse_click(x: float, y: float, button: str = "left",
                 click_count: int = 1) -> Tuple[bool, str]:
    try:
        from pynput.mouse import Button, Controller
        _BTN = {"left": Button.left, "right": Button.right, "middle": Button.middle}
        m = Controller()
        m.position = (x, y)
        time.sleep(0.05)
        btn = _BTN.get(button, Button.left)
        for _ in range(click_count):
            m.click(btn)
            time.sleep(0.05)
        return True, ""
    except ImportError:
        return False, "pynput not installed: pip install pynput"
    except Exception as e:
        return False, str(e)


def _mouse_drag(from_xy: Tuple, to_xy: Tuple, button: str = "left") -> Tuple[bool, str]:
    try:
        from pynput.mouse import Button, Controller
        _BTN = {"left": Button.left, "right": Button.right}
        m = Controller()
        btn = _BTN.get(button, Button.left)
        m.position = from_xy
        time.sleep(0.05)
        m.press(btn)
        time.sleep(0.05)
        # Smooth drag
        steps = 20
        fx, fy = from_xy
        tx, ty = to_xy
        for i in range(1, steps + 1):
            m.position = (fx + (tx - fx) * i / steps, fy + (ty - fy) * i / steps)
            time.sleep(0.01)
        m.release(btn)
        return True, ""
    except ImportError:
        return False, "pynput not installed: pip install pynput"
    except Exception as e:
        return False, str(e)


def _mouse_scroll(x: float, y: float, dx: int = 0, dy: int = -3) -> Tuple[bool, str]:
    try:
        from pynput.mouse import Controller
        m = Controller()
        m.position = (x, y)
        time.sleep(0.05)
        m.scroll(dx, dy)
        return True, ""
    except ImportError:
        return False, "pynput not installed"
    except Exception as e:
        return False, str(e)


def _keyboard_type(text: str, delay: float = 0.03) -> Tuple[bool, str]:
    try:
        from pynput.keyboard import Controller
        kb = Controller()
        for ch in text:
            kb.press(ch)
            kb.release(ch)
            time.sleep(delay)
        return True, ""
    except ImportError:
        return False, "pynput not installed"
    except Exception as e:
        return False, str(e)


_SPECIAL_KEYS = {
    "enter": "enter", "return": "enter", "tab": "tab",
    "space": "space", "escape": "esc", "esc": "esc",
    "backspace": "backspace", "delete": "delete",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "home": "home", "end": "end", "pageup": "page_up", "pagedown": "page_down",
    "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4",
    "f5": "f5", "f6": "f6", "f7": "f7", "f8": "f8",
    "f9": "f9", "f10": "f10", "f11": "f11", "f12": "f12",
}


def _keyboard_key(keys: str) -> Tuple[bool, str]:
    """Press a key or combination like ctrl+c, cmd+shift+4."""
    try:
        from pynput.keyboard import Key, Controller, HotKey
        kb   = Controller()
        parts = [_canon_key(p) for p in re.split(r"\s*\+\s*", keys) if p.strip()]

        def _resolve(k: str):
            # Special key name
            mapped = _SPECIAL_KEYS.get(k)
            if mapped:
                return getattr(Key, mapped)
            # Modifier keys
            if k in ("cmd", "command"):
                return Key.cmd
            if k in ("ctrl", "control"):
                return Key.ctrl
            if k in ("alt", "option"):
                return Key.alt
            if k == "shift":
                return Key.shift
            # Single character
            if len(k) == 1:
                return k
            # Fallback: try as Key attribute
            try:
                return getattr(Key, k)
            except AttributeError:
                return k

        resolved = [_resolve(p) for p in parts]
        if len(resolved) == 1:
            kb.press(resolved[0])
            kb.release(resolved[0])
        else:
            # Combo: press all, then release all in reverse
            for k in resolved:
                kb.press(k)
            for k in reversed(resolved):
                kb.release(k)
        return True, ""
    except ImportError:
        return False, "pynput not installed: pip install pynput"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# App management (macOS only)
# ---------------------------------------------------------------------------

def _list_apps_macos() -> List[Dict]:
    try:
        script = 'tell application "System Events" to get name of every process where background only is false'
        out = subprocess.check_output(["osascript", "-e", script], timeout=5)
        apps = [a.strip() for a in out.decode().split(",") if a.strip()]
        return [{"name": a} for a in sorted(apps)]
    except Exception as e:
        return [{"error": str(e)}]


def _focus_app_macos(app: str) -> Tuple[bool, str]:
    try:
        script = f'tell application "{app}" to activate'
        subprocess.run(["osascript", "-e", script], timeout=5, check=True)
        return True, ""
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def handle_computer_use(action: str, **args) -> Dict:
    """
    Dispatch a computer_use action.
    Returns a dict with success, message, and optionally base64_png / elements.
    """
    action = (action or "").strip().lower()
    if not action:
        return {"success": False, "error": "missing action"}

    # Safety checks before approval
    if action == "type":
        err = _check_type_safety(args.get("text", ""))
        if err:
            return {"success": False, "error": err}
    if action == "key":
        err = _check_key_safety(args.get("keys", ""))
        if err:
            return {"success": False, "error": err}

    # Approval gate for mutating actions
    if action in _MUTATING_ACTIONS:
        denied = _request_approval(action, args)
        if denied:
            return {"success": False, "error": "denied by approval gate"}

    # Dispatch
    if action == "capture":
        return _do_capture(args)
    if action in ("click", "double_click", "right_click"):
        return _do_click(action, args)
    if action == "drag":
        return _do_drag(args)
    if action == "scroll":
        return _do_scroll(args)
    if action == "type":
        return _do_type(args)
    if action == "key":
        return _do_key(args)
    if action == "wait":
        secs = float(args.get("seconds", 1.0))
        time.sleep(secs)
        return {"success": True, "action": "wait", "seconds": secs}
    if action == "list_apps":
        if sys.platform != "darwin":
            return {"success": False, "error": "list_apps is macOS only"}
        return {"success": True, "apps": _list_apps_macos()}
    if action == "focus_app":
        if sys.platform != "darwin":
            return {"success": False, "error": "focus_app is macOS only"}
        app = args.get("app", "")
        if not app:
            return {"success": False, "error": "focus_app requires app name"}
        ok, err = _focus_app_macos(app)
        return {"success": ok, "app": app, "error": err if not ok else ""}

    return {"success": False, "error": f"unknown action: {action!r}"}


def _do_capture(args: Dict) -> Dict:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    region = args.get("region")  # {"left": x, "top": y, "width": w, "height": h}
    png = _capture_screen(region=region)
    if not png:
        return {"success": False, "error": "screen capture failed (install mss or run on macOS)"}
    b64  = _png_to_base64(png)
    path = SCREENSHOTS_DIR / f"cua_{uuid.uuid4().hex[:8]}.png"
    path.write_bytes(png)
    return {
        "success": True,
        "action":  "capture",
        "saved_to": str(path),
        "base64_png": b64,
        "bytes":  len(png),
        "hint":   "Use this screenshot to identify element coordinates for clicks.",
    }


def _do_click(action: str, args: Dict) -> Dict:
    x = float(args.get("x", 0))
    y = float(args.get("y", 0))
    button_map = {"click": "left", "double_click": "left",
                  "right_click": "right"}
    button = args.get("button", button_map.get(action, "left"))
    count  = 2 if action == "double_click" else 1
    ok, err = _mouse_click(x, y, button=button, click_count=count)
    result = {"success": ok, "action": action, "x": x, "y": y}
    if not ok:
        result["error"] = err
    if ok and bool(args.get("capture_after")):
        cap = _do_capture({})
        result["after"] = cap
    return result


def _do_drag(args: Dict) -> Dict:
    fx = float(args.get("from_x", 0))
    fy = float(args.get("from_y", 0))
    tx = float(args.get("to_x", 0))
    ty = float(args.get("to_y", 0))
    button = args.get("button", "left")
    ok, err = _mouse_drag((fx, fy), (tx, ty), button=button)
    result = {"success": ok, "action": "drag",
              "from": [fx, fy], "to": [tx, ty]}
    if not ok:
        result["error"] = err
    return result


def _do_scroll(args: Dict) -> Dict:
    x = float(args.get("x", 640))
    y = float(args.get("y", 400))
    direction = args.get("direction", "down")
    amount    = int(args.get("amount", 3))
    dir_map   = {"down": (0, -amount), "up": (0, amount),
                 "right": (amount, 0), "left": (-amount, 0)}
    dx, dy = dir_map.get(direction, (0, -amount))
    ok, err = _mouse_scroll(x, y, dx=dx, dy=dy)
    result = {"success": ok, "action": "scroll", "direction": direction}
    if not ok:
        result["error"] = err
    return result


def _do_type(args: Dict) -> Dict:
    text  = args.get("text", "")
    delay = float(args.get("delay", 0.03))
    ok, err = _keyboard_type(text, delay=delay)
    result = {"success": ok, "action": "type", "chars": len(text)}
    if not ok:
        result["error"] = err
    return result


def _do_key(args: Dict) -> Dict:
    keys = args.get("keys", "")
    ok, err = _keyboard_key(keys)
    result = {"success": ok, "action": "key", "keys": keys}
    if not ok:
        result["error"] = err
    return result


# ---------------------------------------------------------------------------
# Single entry point for tool registry
# ---------------------------------------------------------------------------

def computer_use(action: str, x: float = None, y: float = None,
                 keys: str = "", text: str = "", button: str = "left",
                 from_x: float = None, from_y: float = None,
                 to_x: float = None, to_y: float = None,
                 direction: str = "down", amount: int = 3,
                 seconds: float = 1.0, app: str = "",
                 region: Dict = None, capture_after: bool = False,
                 click_count: int = 1, delay: float = 0.03,
                 **_) -> Dict:
    """
    Universal computer-use tool. action must be one of:
      capture, click, double_click, right_click, drag, scroll,
      type, key, wait, list_apps, focus_app
    """
    args = {k: v for k, v in {
        "x": x, "y": y, "keys": keys, "text": text, "button": button,
        "from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y,
        "direction": direction, "amount": amount, "seconds": seconds,
        "app": app, "region": region, "capture_after": capture_after,
        "click_count": click_count, "delay": delay,
    }.items() if v is not None}
    return handle_computer_use(action=action, **args)


def check_computer_use_requirements() -> bool:
    """Return True if computer_use can run on this system."""
    try:
        import mss        # noqa: F401
        import pynput     # noqa: F401
        return True
    except ImportError:
        pass
    return sys.platform == "darwin"  # can use screencapture + osascript fallback
