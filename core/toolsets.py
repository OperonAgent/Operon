"""
Operon Toolset Distributions.

Adapted from Hermes Agent toolsets.py and toolset_distributions.py.

Named groups of tools for different task types.  Instead of giving the agent
all tools on every turn, a toolset distribution selects the right subset for
the current task, reducing prompt size and LLM confusion.

Toolsets:
  core         — minimal tools always available (shell, file, search)
  coding       — development workflow (git, code exec, shell, file)
  research     — information gathering (web, search, browser, vision)
  data         — data work (db_ops, code_exec, file_ops, cloud)
  devops       — infrastructure (docker, ssh, shell, cloud)
  comms        — communication tools (email, slack, discord, teams)
  full         — all registered tools

Distributions define which toolset is active for each agent personality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Tool names (must match registry keys) ─────────────────────────────────────

TOOL_GROUPS: dict[str, list[str]] = {
    "shell": [
        "shell_exec",
    ],
    "file": [
        "file_ops",
        "file_search",
    ],
    "search": [
        "web_search",
    ],
    "git": [
        "git_ops",
    ],
    "code": [
        "code_exec",
        "cloud_exec",
    ],
    "web": [
        "web_search",
        "http_client",
        "browser",
    ],
    "vision": [
        "vision",
    ],
    "data": [
        "db_ops",
        "code_exec",
    ],
    "devops": [
        "docker_exec",
        "ssh_exec",
        "shell_exec",
        "cloud_exec",
    ],
    "comms": [
        "email_draft",
        "slack_ops",
        "discord_ops",
        "teams_ops",
        "matrix_ops",
        "mattermost_ops",
        "irc_ops",
        "whatsapp_ops",
        "signal_ops",
        "messaging",
    ],
    "knowledge": [
        "knowledge_ops",
        "file_search",
        "web_search",
    ],
    "media": [
        "vision",
        "voice_input",
    ],
    "patch": [
        "apply_patch",
    ],
    "llm": [
        "llm_task",
    ],
}

# ── Named toolsets ─────────────────────────────────────────────────────────────

TOOLSETS: dict[str, list[str]] = {
    "core": [
        "shell_exec", "file_ops", "file_search", "web_search",
    ],
    "coding": [
        "shell_exec", "file_ops", "file_search", "git_ops",
        "code_exec", "cloud_exec", "apply_patch", "llm_task",
    ],
    "research": [
        "web_search", "http_client", "browser", "file_ops",
        "file_search", "vision", "llm_task",
    ],
    "data": [
        "db_ops", "code_exec", "file_ops", "file_search",
        "cloud_exec", "shell_exec", "llm_task",
    ],
    "devops": [
        "docker_exec", "ssh_exec", "shell_exec", "cloud_exec",
        "git_ops", "file_ops", "http_client",
    ],
    "comms": [
        "email_draft", "slack_ops", "discord_ops", "teams_ops",
        "matrix_ops", "whatsapp_ops", "signal_ops", "messaging",
    ],
    "writing": [
        "file_ops", "file_search", "web_search", "llm_task",
    ],
    "minimal": [
        "shell_exec",
    ],
    "full": [],   # Empty = load all from registry
}


# ── Toolset distribution ───────────────────────────────────────────────────────

@dataclass
class ToolsetDistribution:
    """
    Maps agent personas/contexts to their toolsets.

    Usage::

        dist = ToolsetDistribution()
        tools = dist.resolve("coding")  # ["shell_exec", "file_ops", ...]
    """
    base: str = "core"
    overrides: dict[str, str] = field(default_factory=dict)

    def resolve(self, context: str) -> list[str]:
        """
        Return the list of tool names for the given context.
        Falls back to base toolset if context not found.
        """
        name = self.overrides.get(context, context)
        return get_toolset(name)

    def resolve_names(self, context: str, registry_keys: Optional[set[str]] = None) -> list[str]:
        """
        Resolve toolset and optionally filter to tools that exist in the registry.
        """
        tools = self.resolve(context)
        if registry_keys:
            tools = [t for t in tools if t in registry_keys]
        return tools


# ── Default distributions per persona ─────────────────────────────────────────

PERSONA_DISTRIBUTIONS: dict[str, str] = {
    "general":    "core",
    "developer":  "coding",
    "researcher": "research",
    "analyst":    "data",
    "sre":        "devops",
    "writer":     "writing",
    "assistant":  "full",
    "minimal":    "minimal",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def get_toolset(name: str) -> list[str]:
    """Return the list of tool names for a named toolset."""
    if name == "full":
        # Collect all known tool names
        return sorted({
            t
            for tools in TOOLSETS.values() if tools
            for t in tools
        })
    return list(TOOLSETS.get(name, TOOLSETS["core"]))


def get_toolset_for_persona(persona: str) -> list[str]:
    """Return the toolset for an agent persona."""
    toolset_name = PERSONA_DISTRIBUTIONS.get(persona, "core")
    return get_toolset(toolset_name)


def add_toolset(name: str, tools: list[str]) -> None:
    """Register a custom toolset at runtime."""
    TOOLSETS[name] = list(tools)


def extend_toolset(name: str, extra_tools: list[str]) -> None:
    """Add tools to an existing toolset (creates it if missing)."""
    existing = TOOLSETS.get(name, [])
    TOOLSETS[name] = list(dict.fromkeys(existing + extra_tools))


def describe_toolsets() -> str:
    """Return a human-readable summary of all toolsets."""
    lines = []
    for name, tools in sorted(TOOLSETS.items()):
        if name == "full":
            lines.append(f"  {name:12} — all registered tools")
        else:
            lines.append(f"  {name:12} — {', '.join(tools)}")
    return "\n".join(lines)


@dataclass
class ActiveToolset:
    """
    Tracks which toolset is currently active for an agent session.
    Allows the user to switch toolsets mid-session.
    """
    name:     str        = "core"
    tools:    list[str]  = field(default_factory=lambda: get_toolset("core"))

    def switch(self, name: str) -> "ActiveToolset":
        """Switch to a different toolset. Returns self for chaining."""
        self.name  = name
        self.tools = get_toolset(name)
        return self

    def add(self, tool_name: str) -> "ActiveToolset":
        """Temporarily add a tool to the active set."""
        if tool_name not in self.tools:
            self.tools.append(tool_name)
        return self

    def remove(self, tool_name: str) -> "ActiveToolset":
        """Temporarily remove a tool from the active set."""
        self.tools = [t for t in self.tools if t != tool_name]
        return self

    def has(self, tool_name: str) -> bool:
        return tool_name in self.tools
