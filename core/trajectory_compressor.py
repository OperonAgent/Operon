"""
Operon Trajectory Compressor.

Adapted from Hermes Agent trajectory_compressor.py.

Compresses a session's full message trajectory into a compact skill file
at session end.  The skill file can be loaded at the start of a new session
to give the agent institutional memory of past sessions without blowing the
context window.

Skill file format (Markdown + YAML front-matter)::

    ---
    session_id: abc123
    compressed_at: 2024-01-01T00:00:00Z
    turns: 42
    tools_used: [shell_exec, file_ops, web_search]
    outcome: succeeded
    ---
    ## Summary
    <1-paragraph summary of what happened>

    ## Key decisions
    - Decision 1
    - Decision 2

    ## Artifacts
    - /path/to/file.py — created
    - /path/to/other.md — modified

    ## Learned patterns
    - Pattern 1
    - Pattern 2

    ## Open items
    - Item 1
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Skill file dataclass ───────────────────────────────────────────────────────

@dataclass
class SessionSkill:
    session_id:      str
    compressed_at:   str         # ISO8601
    turns:           int
    tools_used:      list[str]
    outcome:         str         # succeeded | failed | interrupted | unknown
    summary:         str
    key_decisions:   list[str]   = field(default_factory=list)
    artifacts:       list[str]   = field(default_factory=list)
    learned_patterns: list[str]  = field(default_factory=list)
    open_items:      list[str]   = field(default_factory=list)
    raw_goal:        str         = ""
    duration_s:      float       = 0.0

    def to_markdown(self) -> str:
        """Serialise to Markdown with YAML front-matter."""
        tools_yaml = json.dumps(self.tools_used)
        decisions  = "\n".join(f"- {d}" for d in self.key_decisions) or "- (none recorded)"
        artifacts  = "\n".join(f"- {a}" for a in self.artifacts)      or "- (none)"
        patterns   = "\n".join(f"- {p}" for p in self.learned_patterns) or "- (none)"
        open_items = "\n".join(f"- {i}" for i in self.open_items)     or "- (none)"

        return (
            f"---\n"
            f"session_id: {self.session_id}\n"
            f"compressed_at: {self.compressed_at}\n"
            f"turns: {self.turns}\n"
            f"tools_used: {tools_yaml}\n"
            f"outcome: {self.outcome}\n"
            f"duration_s: {self.duration_s:.1f}\n"
            f"---\n"
            f"## Goal\n{self.raw_goal or '(not recorded)'}\n\n"
            f"## Summary\n{self.summary}\n\n"
            f"## Key decisions\n{decisions}\n\n"
            f"## Artifacts\n{artifacts}\n\n"
            f"## Learned patterns\n{patterns}\n\n"
            f"## Open items\n{open_items}\n"
        )

    @classmethod
    def from_markdown(cls, text: str) -> "SessionSkill":
        """Parse a skill file back into a SessionSkill."""
        import yaml  # optional dep — gracefully degrade

        front_match = re.match(r"^---\n([\s\S]+?)\n---\n", text)
        meta: dict = {}
        body = text
        if front_match:
            try:
                meta = yaml.safe_load(front_match.group(1)) or {}
            except Exception:
                pass
            body = text[front_match.end():]

        def _extract_section(name: str) -> str:
            m = re.search(rf"## {name}\n([\s\S]+?)(?=\n## |\Z)", body)
            return m.group(1).strip() if m else ""

        def _extract_bullets(name: str) -> list[str]:
            section = _extract_section(name)
            return [
                l.lstrip("-• ").strip()
                for l in section.splitlines()
                if l.strip().startswith("-")
            ]

        return cls(
            session_id       = str(meta.get("session_id", "")),
            compressed_at    = str(meta.get("compressed_at", "")),
            turns            = int(meta.get("turns", 0)),
            tools_used       = list(meta.get("tools_used", [])),
            outcome          = str(meta.get("outcome", "unknown")),
            duration_s       = float(meta.get("duration_s", 0)),
            summary          = _extract_section("Summary"),
            key_decisions    = _extract_bullets("Key decisions"),
            artifacts        = _extract_bullets("Artifacts"),
            learned_patterns = _extract_bullets("Learned patterns"),
            open_items       = _extract_bullets("Open items"),
            raw_goal         = _extract_section("Goal"),
        )


# ── Compressor ─────────────────────────────────────────────────────────────────

class TrajectoryCompressor:
    """
    Compress a session trajectory into a SessionSkill.

    Typical usage (session-end hook)::

        compressor = TrajectoryCompressor(skills_dir=Path("~/.operon/skills"))
        skill      = compressor.compress(messages, session_id="abc123")
        path       = compressor.save(skill)
    """

    def __init__(
        self,
        skills_dir:  Optional[Path] = None,
        use_llm:     bool           = False,   # if True, use LLM for better summaries
    ) -> None:
        self.skills_dir = skills_dir or (Path.home() / ".operon" / "skills")
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.use_llm = use_llm

    # ── Public API ─────────────────────────────────────────────────────────────

    def compress(
        self,
        messages:    list[dict],
        session_id:  str           = "",
        outcome:     str           = "unknown",
        start_time:  Optional[float] = None,
    ) -> SessionSkill:
        """
        Compress a message trajectory into a SessionSkill.

        Parameters
        ----------
        messages   : full conversation message list
        session_id : identifier for this session
        outcome    : "succeeded" | "failed" | "interrupted" | "unknown"
        start_time : Unix timestamp when the session started
        """
        if not session_id:
            session_id = _short_id()

        turns       = sum(1 for m in messages if m.get("role") == "assistant")
        tools_used  = _extract_tools_used(messages)
        goal        = _extract_goal(messages)
        artifacts   = _extract_artifacts(messages)
        decisions   = _extract_decisions(messages)
        open_items  = _extract_open_items(messages)
        patterns    = _extract_patterns(messages)

        if self.use_llm:
            summary = self._llm_summary(messages, goal)
        else:
            summary = _heuristic_summary(messages, goal, tools_used, artifacts)

        duration_s = (time.time() - start_time) if start_time else 0.0

        return SessionSkill(
            session_id       = session_id,
            compressed_at    = datetime.now(timezone.utc).isoformat(),
            turns            = turns,
            tools_used       = sorted(tools_used),
            outcome          = outcome,
            summary          = summary,
            key_decisions    = decisions,
            artifacts        = artifacts,
            learned_patterns = patterns,
            open_items       = open_items,
            raw_goal         = goal,
            duration_s       = duration_s,
        )

    def save(self, skill: SessionSkill) -> Path:
        """Persist a SessionSkill to the skills directory. Returns the file path."""
        slug   = re.sub(r"[^a-zA-Z0-9_-]", "_", skill.session_id)[:32]
        ts     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname  = f"session_{ts}_{slug}.md"
        path   = self.skills_dir / fname
        path.write_text(skill.to_markdown(), encoding="utf-8")
        return path

    def load_recent(self, n: int = 5) -> list[SessionSkill]:
        """Load the `n` most recent skill files."""
        files = sorted(self.skills_dir.glob("session_*.md"), reverse=True)[:n]
        skills: list[SessionSkill] = []
        for f in files:
            try:
                skills.append(SessionSkill.from_markdown(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        return skills

    def load_context(self, n: int = 3) -> str:
        """
        Return a compact context string from the n most recent skills.
        Suitable for prepending to the system prompt.
        """
        skills = self.load_recent(n)
        if not skills:
            return ""
        parts  = ["## Recent session memory\n"]
        for sk in skills:
            parts.append(
                f"### Session {sk.session_id} ({sk.outcome})\n"
                f"Goal: {sk.raw_goal[:100]}\n"
                f"Summary: {sk.summary[:200]}\n"
            )
            if sk.open_items:
                items = "; ".join(sk.open_items[:3])
                parts.append(f"Open items: {items}\n")
        return "\n".join(parts)

    def _llm_summary(self, messages: list[dict], goal: str) -> str:
        """Use the LLM to generate a better summary (optional)."""
        try:
            from tools.llm_task import llm_summarize
            # Extract last 20 assistant turns as the "trajectory"
            turns = [
                m["content"] for m in messages
                if m.get("role") == "assistant"
            ][-20:]
            text  = "\n\n".join(turns)
            result = llm_summarize(text, max_bullets=5)
            if result.get("success"):
                return result["result"]
        except Exception:
            pass
        return _heuristic_summary(messages, goal, [], [])


# ── Extraction helpers ─────────────────────────────────────────────────────────

def _extract_tools_used(messages: list[dict]) -> list[str]:
    """Extract unique tool names from the message history."""
    tools: set[str] = set()
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            # Look for JSON tool calls embedded in content
            for match in re.finditer(r'"tool_name"\s*:\s*"([^"]+)"', content):
                tools.add(match.group(1))
        # tool_calls field (OpenAI style)
        for tc in m.get("tool_calls", []):
            name = tc.get("function", {}).get("name") or tc.get("name", "")
            if name:
                tools.add(name)
    return sorted(tools)


def _extract_goal(messages: list[dict]) -> str:
    """Extract the user's initial goal from the first user message."""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()[:300]
    return ""


def _extract_artifacts(messages: list[dict]) -> list[str]:
    """Extract file paths mentioned as created/modified/written."""
    paths: list[str] = []
    seen:  set[str]  = set()
    path_re = re.compile(
        r"(?:created?|wrote|modified?|updated?|saved?|written)\s+"
        r"(?:to\s+)?([`'\"]?)(/[^\s`'\"]+|[\w./]+\.\w{1,6})\1",
        re.IGNORECASE,
    )
    for m in messages:
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        for match in path_re.finditer(content):
            p = match.group(2)
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths[:20]


def _extract_decisions(messages: list[dict]) -> list[str]:
    """Extract key decisions from assistant messages."""
    decisions: list[str] = []
    decision_re = re.compile(
        r"(?:decided?|chose|selected?|going with|using|will use|opted for)\s+(.{10,80}?)(?:[.!]|$)",
        re.IGNORECASE,
    )
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        for match in decision_re.finditer(content):
            text = match.group(0).strip()
            if len(text) > 10 and text not in decisions:
                decisions.append(text[:120])
    return decisions[:10]


def _extract_open_items(messages: list[dict]) -> list[str]:
    """Extract TODO / open items from the last few assistant messages."""
    items: list[str] = []
    todo_re = re.compile(
        r"(?:TODO|FIXME|still need|haven't|not yet|remaining|pending|left to do)\s*:?\s*(.{10,100}?)(?:[.!]|$)",
        re.IGNORECASE,
    )
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    for m in assistant_msgs[-5:]:
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        for match in todo_re.finditer(content):
            text = match.group(0).strip()
            if text not in items:
                items.append(text[:120])
    return items[:8]


def _extract_patterns(messages: list[dict]) -> list[str]:
    """Extract reusable patterns / lessons learned from the session."""
    patterns: list[str] = []
    pattern_re = re.compile(
        r"(?:note that|remember|turns out|learned|pattern|best practice|tip)\s*:?\s*(.{10,120}?)(?:[.!]|$)",
        re.IGNORECASE,
    )
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        for match in pattern_re.finditer(content):
            text = match.group(0).strip()
            if text not in patterns:
                patterns.append(text[:120])
    return patterns[:6]


def _heuristic_summary(
    messages:  list[dict],
    goal:      str,
    tools:     list[str],
    artifacts: list[str],
) -> str:
    """Generate a simple heuristic summary without calling the LLM."""
    tool_str  = ", ".join(tools[:5]) if tools else "no tools"
    art_str   = ", ".join(artifacts[:3]) if artifacts else "no files"
    turns     = sum(1 for m in messages if m.get("role") == "assistant")

    # Use the last assistant message as a summary proxy
    last_turn = ""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if isinstance(content, str) and len(content.strip()) > 20:
                last_turn = content.strip()[:300]
                break

    summary = (
        f"Session ran {turns} turns using {tool_str}. "
        f"Files touched: {art_str}."
    )
    if last_turn:
        summary += f" Last response: {last_turn}"
    return summary


def _short_id() -> str:
    import secrets
    return secrets.token_hex(6)
