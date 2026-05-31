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
