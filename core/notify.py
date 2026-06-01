"""
core/notify.py — turn-completion notifications.

Harvested from Hermes Agent's `bell_on_complete` and extended with optional
native desktop notifications. When a long-running turn finishes, Operon can:

  * ring the terminal bell (``\\a``) — propagates over SSH to the user's
    terminal, so you get a nudge even on a remote box; and/or
  * raise a native desktop notification (macOS ``osascript``, Linux
    ``notify-send``, Windows PowerShell toast).

Everything is best-effort and config-gated: every function swallows its own
errors and is a no-op unless explicitly enabled, so notifications can never
crash a turn or block the loop.

Config keys (core/config.py):
    notify_on_complete : bool  — ring the terminal bell when a turn finishes
    notify_desktop     : bool  — also raise a native desktop notification
    notify_min_seconds : float — only notify for turns at least this long
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from typing import Any, Optional

_APP_NAME = "Operon"


def _truthy(config: Any, key: str, default: bool = False) -> bool:
    try:
        return bool(config.get(key, default))
    except Exception:
        return default


def ring_bell() -> None:
    """Emit a terminal bell. Propagates over SSH. Never raises."""
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


def desktop_notification(message: str, title: str = _APP_NAME) -> bool:
    """
    Best-effort native desktop notification. Returns True if a notifier ran.
    Never raises; silently no-ops when no backend is available (e.g. headless).
    """
    msg = (message or "").replace('"', "'").strip()[:200] or "Task complete"
    ttl = (title or _APP_NAME).replace('"', "'")[:60]
    try:
        if sys.platform == "darwin" and shutil.which("osascript"):
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{msg}" with title "{ttl}"'],
                check=False, capture_output=True, timeout=5,
            )
            return True
        if sys.platform.startswith("linux") and shutil.which("notify-send"):
            subprocess.run(["notify-send", ttl, msg],
                           check=False, capture_output=True, timeout=5)
            return True
        if sys.platform.startswith("win") and shutil.which("powershell"):
            ps = (
                "[reflection.assembly]::loadwithpartialname('System.Windows.Forms')"
                ">$null; "
                "$n=New-Object System.Windows.Forms.NotifyIcon; "
                "$n.Icon=[System.Drawing.SystemIcons]::Information; "
                "$n.Visible=$true; "
                f"$n.ShowBalloonTip(5000,'{ttl}','{msg}',"
                "[System.Windows.Forms.ToolTipIcon]::Info)"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           check=False, capture_output=True, timeout=8)
            return True
    except Exception:
        pass
    return False


def notify_complete(config: Any, message: str = "Task complete",
                    elapsed: Optional[float] = None) -> None:
    """
    Fire the configured completion notifications for a finished turn.

    Honors ``notify_min_seconds`` so quick replies stay quiet. A no-op unless
    ``notify_on_complete`` is enabled. Never raises.
    """
    if not _truthy(config, "notify_on_complete", False):
        return
    try:
        min_s = float(config.get("notify_min_seconds", 0) or 0)
    except Exception:
        min_s = 0.0
    if elapsed is not None and min_s > 0 and elapsed < min_s:
        return

    ring_bell()
    if _truthy(config, "notify_desktop", False):
        desktop_notification(message)
