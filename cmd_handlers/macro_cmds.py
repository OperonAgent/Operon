"""
cmd_handlers/macro_cmds.py — the /macro pipeline-macro command.

This is the first *stateful* command migrated out of main.py's elif-chain.
It owns the lazy-initialised ``main._macros`` MacroManager singleton, reading
and mutating it through the live ``main`` module so behaviour is identical to
the legacy handler (single source of truth for the global).
"""
from __future__ import annotations

from cmd_handlers import command, CommandContext


def _manager(ctx: CommandContext):
    """Return the shared MacroManager, creating it on first use."""
    import main  # lazy — avoids a circular import at package load time
    if main._macros is None:
        main._macros = main.MacroManager(
            tool_registry=ctx.tool_registry or main.ToolRegistry())
    return main._macros


@command("/macro")
def macro(ctx: CommandContext) -> None:
    theme = ctx.theme
    parts = ctx.parts
    mgr   = _manager(ctx)
    sub   = parts[1].lower() if len(parts) > 1 else "list"

    if sub == "list":
        macros = mgr.list_macros()
        if not macros:
            print(theme.info("No macros saved. Use /macro define to create one, "
                             "or use the macro_save tool."))
            return
        lines = ["  PIPELINE MACROS", "---"]
        for m in macros:
            lines.append(f"  {m['name']:<20}  {len(m.get('steps', []))} steps  "
                         f"{m.get('description', '')[:50]}")
        print(theme.box(lines))

    elif sub == "run":
        if len(parts) < 3:
            print(theme.warning("Usage: /macro run <name> [key=value ...]"))
            return
        name = parts[2]
        vars_dict = {}
        for token in parts[3:]:
            if "=" in token:
                k, v = token.split("=", 1)
                vars_dict[k] = v
        result = mgr.run(name, vars=vars_dict if vars_dict else None)
        if result.get("success"):
            print(theme.success(
                f"Macro '{name}' completed in {len(result.get('steps', []))} steps."))
            if result.get("output"):
                print(theme.info(str(result["output"])[:400]))
        else:
            print(theme.error(result.get("error", f"Macro '{name}' failed.")))

    elif sub in ("define", "create"):
        print(theme.info(
            "To create a macro, use the macro_save tool in a prompt:\n"
            '  Save a macro named "daily_report" with steps: ...'))

    elif sub == "delete":
        if len(parts) < 3:
            print(theme.warning("Usage: /macro delete <name>"))
            return
        result = mgr.delete(parts[2])
        if result.get("success"):
            print(theme.success(f"Macro '{parts[2]}' deleted."))
        else:
            print(theme.error(result.get("error", "Not found.")))

    else:
        print(theme.warning("Usage: /macro list|run <name>|delete <name>"))
