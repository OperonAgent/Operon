"""tests/test_notify.py — turn-completion notifications (harvested from Hermes)."""
from __future__ import annotations
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import notify
from core.notify import notify_complete, ring_bell, desktop_notification


def _cfg(**kw):
    c = MagicMock()
    c.get.side_effect = lambda k, d=None: kw.get(k, d)
    return c


def _capture(fn, *a, **k):
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn(*a, **k)
    finally:
        sys.stdout = old
    return buf.getvalue()


class TestRingBell:
    def test_emits_bell(self):
        assert "\a" in _capture(ring_bell)

    def test_never_raises(self):
        with patch.object(notify.sys, "stdout") as so:
            so.write.side_effect = RuntimeError("boom")
            ring_bell()  # must swallow


class TestNotifyComplete:
    def test_disabled_is_silent(self):
        out = _capture(notify_complete, _cfg(notify_on_complete=False))
        assert "\a" not in out

    def test_enabled_rings_bell(self):
        out = _capture(notify_complete, _cfg(notify_on_complete=True), "done", 5)
        assert "\a" in out

    def test_min_seconds_gate_blocks_quick_turns(self):
        cfg = _cfg(notify_on_complete=True, notify_min_seconds=30)
        out = _capture(notify_complete, cfg, "x", 2.0)
        assert "\a" not in out

    def test_min_seconds_allows_long_turns(self):
        cfg = _cfg(notify_on_complete=True, notify_min_seconds=10)
        out = _capture(notify_complete, cfg, "x", 99.0)
        assert "\a" in out

    def test_desktop_called_when_enabled(self):
        cfg = _cfg(notify_on_complete=True, notify_desktop=True)
        with patch("core.notify.desktop_notification", return_value=True) as dn:
            _capture(notify_complete, cfg, "hi", 1)
        dn.assert_called_once()

    def test_desktop_not_called_when_disabled(self):
        cfg = _cfg(notify_on_complete=True, notify_desktop=False)
        with patch("core.notify.desktop_notification") as dn:
            _capture(notify_complete, cfg, "hi", 1)
        dn.assert_not_called()

    def test_bad_config_never_raises(self):
        bad = MagicMock()
        bad.get.side_effect = RuntimeError("config exploded")
        notify_complete(bad)  # must swallow → no-op


class TestDesktopNotification:
    def test_macos_uses_osascript(self):
        with patch.object(notify.sys, "platform", "darwin"), \
             patch("core.notify.shutil.which", return_value="/usr/bin/osascript"), \
             patch("core.notify.subprocess.run") as run:
            assert desktop_notification("hello", "Operon") is True
        assert run.call_args[0][0][0] == "osascript"

    def test_linux_uses_notify_send(self):
        with patch.object(notify.sys, "platform", "linux"), \
             patch("core.notify.shutil.which", return_value="/usr/bin/notify-send"), \
             patch("core.notify.subprocess.run") as run:
            assert desktop_notification("hi") is True
        assert run.call_args[0][0][0] == "notify-send"

    def test_no_backend_returns_false(self):
        with patch("core.notify.shutil.which", return_value=None):
            assert desktop_notification("hi") is False

    def test_quotes_sanitised(self):
        with patch.object(notify.sys, "platform", "darwin"), \
             patch("core.notify.shutil.which", return_value="/usr/bin/osascript"), \
             patch("core.notify.subprocess.run") as run:
            desktop_notification('say "hi" now')
        # double quotes replaced so the AppleScript string stays valid
        assert '"hi"' not in run.call_args[0][0][2]

    def test_subprocess_error_swallowed(self):
        with patch.object(notify.sys, "platform", "darwin"), \
             patch("core.notify.shutil.which", return_value="/usr/bin/osascript"), \
             patch("core.notify.subprocess.run", side_effect=OSError("nope")):
            assert desktop_notification("hi") is False


class TestNotifyCommand:
    def test_registered(self):
        import cmd_handlers as ch
        assert "/notify" in ch.DISPATCH

    def test_on_off_persist(self):
        import cmd_handlers as ch
        cfg = MagicMock()
        store = {}
        cfg.get.side_effect = lambda k, d=None: store.get(k, d)
        cfg.set.side_effect = lambda k, v: store.__setitem__(k, v)
        theme = MagicMock()
        theme.success = theme.info = theme.error = lambda x: str(x)
        theme.box = lambda l, **k: "BOX"
        ctx = ch.CommandContext(command="/notify on", parts=["/notify", "on"],
                                cmd="/notify", args=["on"], config=cfg, theme=theme)
        with patch("builtins.print"):
            ch.dispatch(ctx)
        assert store["notify_on_complete"] is True

    def test_config_defaults_present(self):
        from core.config import _DEFAULTS
        assert "notify_on_complete" in _DEFAULTS
        assert "notify_desktop" in _DEFAULTS
        assert "notify_min_seconds" in _DEFAULTS
