"""
cmd_handlers/info.py — read-only informational commands.

Extracted verbatim from main.py's handle_command. These handlers only read
their context; they never reassign main.py module globals.
"""

from __future__ import annotations

from cmd_handlers import command, CommandContext


@command("/tools")
def cmd_tools(ctx: CommandContext) -> None:
    from tools.registry import _TOOL_DEFINITIONS
    theme = ctx.theme
    tool_registry = ctx.tool_registry
    lines = [f"  AVAILABLE TOOLS  ({len(tool_registry.tools)})", "---"]
    for td in _TOOL_DEFINITIONS:
        desc = td.get("description", "")
        short = desc[:55] + "…" if len(desc) > 55 else desc
        lines.append(f"  {td['name']:<28} {short}")
    defined = {td["name"] for td in _TOOL_DEFINITIONS}
    for name in sorted(tool_registry.tools):
        if name not in defined:
            lines.append(f"  {name:<28} (dynamic)")
    print(theme.box(lines))


@command("/usage")
def cmd_usage(ctx: CommandContext) -> None:
    theme = ctx.theme
    try:
        s = ctx.session.get_usage_stats()
        print(theme.box([
            "  USAGE STATS", "---",
            f"  Turns              {s.get('turns', 0)}",
            f"  Messages           {s.get('messages', 0)}",
            f"  Characters         {int(s.get('chars', 0)):,}",
            f"  Est. tokens        {int(s.get('est_tokens', 0)):,}",
            f"  Est. cost (gpt-4o) ${float(s.get('est_cost_4o', 0.0)):.4f}",
            "---",
            "  Use /compress to trim context and reduce cost.",
        ]))
    except Exception as _e:
        print(theme.error(f"  Could not retrieve usage stats: {_e}"))


@command("/cost")
def cmd_cost(ctx: CommandContext) -> None:
    theme = ctx.theme
    cost_tracker = ctx.cost_tracker
    if cost_tracker is None or not cost_tracker._calls:
        print(theme.info("  No API calls recorded yet this session."))
    else:
        print(theme.box(cost_tracker.session_report()))
