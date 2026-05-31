"""
cmd_handlers — modular slash-command handlers for Operon.

This package incrementally extracts the giant `handle_command` elif-chain in
main.py into small, individually-testable handler functions.

Design
------
Each handler is a function ``handler(ctx: CommandContext) -> None`` registered
under one or more command names via the ``@command(...)`` decorator. main.py
builds a CommandContext, then consults DISPATCH first; anything not yet
extracted falls through to the legacy elif-chain unchanged. This makes the
migration safe and reversible — no behaviour change, guarded by
tests/test_slash_commands.py.

Only read-only / stateless commands are extracted here. Commands that reassign
main.py module-level globals (e.g. /gateway, /dashboard, /approve) remain in
main.py for now and are tracked in ROADMAP.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Registry: command-name -> handler function
DISPATCH: Dict[str, Callable[["CommandContext"], None]] = {}


def command(*names: str) -> Callable:
    """Register a handler under one or more slash-command names."""
    def _wrap(fn: Callable[["CommandContext"], None]):
        for n in names:
            DISPATCH[n] = fn
        return fn
    return _wrap


@dataclass
class CommandContext:
    """Everything a command handler might need, in one object."""
    command: str
    parts:   List[str]
    cmd:     str
    args:    List[str]

    # Core services (mirror handle_command's parameters)
    config:       Any = None
    session:      Any = None
    memory:       Any = None
    theme:        Any = None
    soul:         Any = None
    scheduler:    Any = None
    tool_registry: Any = None
    router:       Any = None
    context_inject: str = ""
    planner:      Any = None
    skills:       Any = None
    curator:      Any = None
    cost_tracker: Any = None
    semantic_mem: Any = None
    knowledge:    Any = None
    extras:       Dict[str, Any] = field(default_factory=dict)


def dispatch(ctx: "CommandContext") -> bool:
    """
    Run the registered handler for ctx.cmd if one exists.
    Returns True if handled, False to signal main.py to use the legacy chain.
    """
    fn = DISPATCH.get(ctx.cmd)
    if fn is None:
        return False
    fn(ctx)
    return True


# Import handler modules so their @command registrations run.
from cmd_handlers import info as _info               # noqa: E402,F401
from cmd_handlers import session_cmds as _sess        # noqa: E402,F401
from cmd_handlers import config_cmds as _cfg          # noqa: E402,F401
from cmd_handlers import macro_cmds as _macro         # noqa: E402,F401
