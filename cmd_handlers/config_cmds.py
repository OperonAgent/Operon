"""
cmd_handlers/config_cmds.py — read-only configuration/profile commands.

Extracted verbatim from main.py. These read config/soul and never reassign
main.py module globals.
"""

from __future__ import annotations

import os

from cmd_handlers import command, CommandContext


@command("/models")
def cmd_models(ctx: CommandContext) -> None:
    config, theme = ctx.config, ctx.theme
    profiles = config.get("model_profiles", {})
    current  = config.get("default_model", "")
    lines    = ["  AVAILABLE MODEL PROFILES", "---"]
    for name, profile in sorted(profiles.items()):
        marker = ">> " if name == current else "   "
        lines.append(f"  {marker}{name:<28} [{profile.get('provider', '?')}]")
    print(theme.box(lines))


@command("/config")
def cmd_config(ctx: CommandContext) -> None:
    config, theme = ctx.config, ctx.theme
    cfg   = config.get_safe_display()
    lines = ["  CURRENT CONFIGURATION", "---"]
    for k, v in cfg.items():
        lines.append(f"  {k:<22} {str(v)[:44]}")
    print(theme.box(lines))


@command("/soul")
def cmd_soul(ctx: CommandContext) -> None:
    soul, theme, parts = ctx.soul, ctx.theme, ctx.parts
    if soul is None:
        print(theme.warning("Soul system not initialised."))
    elif len(parts) > 1 and parts[1] == "edit":
        import subprocess as _sp
        path   = soul.get_path()
        editor = os.environ.get("EDITOR", "nano")
        _sp.run([editor, path])
        print(theme.success(f"Soul updated: {path}"))
    else:
        print(theme.box(["  OPERON SOUL", "---"] +
                        ["  " + l for l in soul.read().splitlines()[:30]]))


@command("/notify")
def cmd_notify(ctx: CommandContext) -> None:
    """Toggle turn-completion notifications (terminal bell + optional desktop)."""
    config, theme, args = ctx.config, ctx.theme, ctx.args
    sub = (args[0].lower() if args else "status")

    if sub in ("on", "enable"):
        config.set("notify_on_complete", True)
        print(theme.success("Completion bell ON — Operon will ring the terminal "
                             "bell when a turn finishes."))
    elif sub in ("off", "disable"):
        config.set("notify_on_complete", False)
        config.set("notify_desktop", False)
        print(theme.success("Completion notifications OFF."))
    elif sub == "desktop":
        on = not bool(config.get("notify_desktop", False))
        config.set("notify_desktop", on)
        if on:
            config.set("notify_on_complete", True)
        # Fire a sample so the user sees it immediately.
        try:
            from core.notify import desktop_notification
            if on:
                desktop_notification("Desktop notifications enabled.", "Operon")
        except Exception:
            pass
        print(theme.success(f"Desktop notifications {'ON' if on else 'OFF'}."))
    elif sub == "test":
        try:
            from core.notify import ring_bell, desktop_notification
            ring_bell()
            if config.get("notify_desktop", False):
                desktop_notification("This is a test notification.", "Operon")
            print(theme.info("Sent a test bell" +
                             (" + desktop notification." if config.get("notify_desktop") else ".")))
        except Exception as e:
            print(theme.error(f"Notification test failed: {e}"))
    else:
        bell = "ON" if config.get("notify_on_complete", False) else "OFF"
        desk = "ON" if config.get("notify_desktop", False) else "OFF"
        mins = config.get("notify_min_seconds", 0)
        print(theme.box([
            "  NOTIFICATIONS", "---",
            f"  Completion bell   : {bell}",
            f"  Desktop alerts    : {desk}",
            f"  Min turn seconds  : {mins}",
            "---",
            "  /notify on | off | desktop | test",
        ]))
