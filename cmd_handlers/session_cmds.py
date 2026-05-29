"""
cmd_handlers/session_cmds.py — session-history commands.

Extracted verbatim from main.py. These operate on ctx.session and never
reassign main.py module globals. (/retry stays in main.py because it calls
run_agent_loop, which lives there.)
"""

from __future__ import annotations

from cmd_handlers import command, CommandContext


@command("/clear")
def cmd_clear(ctx: CommandContext) -> None:
    ctx.session.clear()
    print(ctx.theme.success("Session history cleared."))


@command("/undo")
def cmd_undo(ctx: CommandContext) -> None:
    theme = ctx.theme
    if ctx.session.undo():
        print(theme.success(f"Last exchange removed. ({len(ctx.session)} messages remain)"))
    else:
        print(theme.warning("Nothing to undo."))


@command("/history")
def cmd_history(ctx: CommandContext) -> None:
    parts = ctx.parts
    n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
    lines = ctx.session.get_history_display(last_n=n)
    print(ctx.theme.box(["  SESSION HISTORY", "---"] + lines) if lines
          else ctx.theme.info("No history yet."))


@command("/compress")
def cmd_compress(ctx: CommandContext) -> None:
    theme = ctx.theme
    removed = ctx.session.compress(keep_first=4, keep_recent=30)
    if removed:
        print(theme.success(f"Context compressed — {removed} messages removed. "
                            f"{len(ctx.session)} remain."))
    else:
        print(theme.info("Context is already compact."))
