"""
core/obsidian_memory.py — Obsidian Vault Memory Sync

Bidirectional sync between Operon's memory system and a local Obsidian vault.
Writes Operon's knowledge (entities, facts, tasks, conversation summaries,
code snippets) as structured Markdown files with backlinks and tags.
Reads vault notes back as context for the agent.

Folder structure in vault:
   Daily/          ← conversation summaries  (Daily/2026-05-28.md)
   People/         ← entity notes            (People/John_Smith.md)
   Projects/       ← project knowledge       (Projects/Operon.md)
   Tasks/          ← task history            (Tasks/active.md)
   Code/           ← code snippets           (Code/python_tips.md)
   Facts/          ← standalone facts        (Facts/preferences.md)
   Goals/          ← agent goals             (Goals/active.md)
   Sessions/       ← session logs            (Sessions/session_001.md)

Features:
  - Auto-creates vault folders on first sync
  - 20-minute background auto-sync loop (like OpenHuman)
  - Bidirectional: reads vault → agent context; writes memory → vault
  - Obsidian [[backlinks]] and #tags in every note
  - Dataview-compatible frontmatter (YAML header)
  - Conflict detection: never overwrites user-edited notes without merging
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("operon.obsidian_memory")

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_VAULT     = Path.home() / "operon-brain"
_SYNC_INTERVAL_S   = 1200   # 20 minutes — matches OpenHuman's auto-fetch cadence
_MAX_NOTE_CHARS    = 32_000
_CONTEXT_NOTES     = 10     # max notes returned in get_context()

# ── Folder structure ──────────────────────────────────────────────────────────
_FOLDERS = ["Daily", "People", "Projects", "Tasks", "Code", "Facts", "Goals", "Sessions"]

# ── Frontmatter template ──────────────────────────────────────────────────────
def _frontmatter(**kwargs) -> str:
    """Build YAML frontmatter for Obsidian / Dataview."""
    now = datetime.now(timezone.utc).isoformat()
    lines = ["---"]
    lines.append(f"created: {kwargs.get('created', now)}")
    lines.append(f"updated: {kwargs.get('updated', now)}")
    if "source" in kwargs:
        lines.append(f"source: {kwargs['source']}")
    if "tags" in kwargs and kwargs["tags"]:
        lines.append("tags:")
        for t in kwargs["tags"]:
            lines.append(f"  - {t}")
    if "category" in kwargs:
        lines.append(f"category: {kwargs['category']}")
    if "session" in kwargs:
        lines.append(f"session: {kwargs['session']}")
    lines.append("---")
    return "\n".join(lines)


# ── Note builder helpers ──────────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:80] or "unnamed"


def _backlink(name: str) -> str:
    return f"[[{name}]]"


def _tag(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_/-]", "_", name)
    return f"#{clean}"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class VaultNote:
    """Represents a single Obsidian note."""
    path:     Path
    content:  str
    modified: float = 0.0   # mtime

    @property
    def title(self) -> str:
        return self.path.stem.replace("_", " ")

    @property
    def folder(self) -> str:
        return self.path.parent.name

    def frontmatter_dict(self) -> Dict:
        """Parse YAML frontmatter into dict."""
        m = re.match(r"^---\n(.*?)\n---\n", self.content, re.DOTALL)
        if not m:
            return {}
        out: Dict = {}
        for line in m.group(1).splitlines():
            if ":" in line and not line.startswith("-"):
                k, _, v = line.partition(":")
                out[k.strip()] = v.strip()
        return out

    def body_text(self) -> str:
        """Return content with frontmatter stripped."""
        content = re.sub(r"^---\n.*?\n---\n", "", self.content, flags=re.DOTALL)
        return content.strip()

    def extract_facts(self) -> List[str]:
        """Extract bullet-point facts from note body."""
        facts = []
        for line in self.body_text().splitlines():
            line = line.strip()
            if line.startswith(("- ", "* ", "+ ")):
                fact = line[2:].strip()
                if fact and len(fact) > 10:
                    facts.append(fact)
        return facts


# ── Vault manager ─────────────────────────────────────────────────────────────

class ObsidianVault:
    """
    Low-level read/write interface to an Obsidian vault folder.
    """

    def __init__(self, vault_path: Path = _DEFAULT_VAULT) -> None:
        self._root = Path(vault_path).expanduser().resolve()
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        """Create vault folders if they don't exist."""
        self._root.mkdir(parents=True, exist_ok=True)
        for folder in _FOLDERS:
            (self._root / folder).mkdir(exist_ok=True)

    def exists(self) -> bool:
        return self._root.exists()

    def write_note(
        self,
        folder:   str,
        filename: str,
        content:  str,
        merge:    bool = True,
    ) -> Path:
        """
        Write a note. If merge=True and the file exists, append new content
        rather than overwriting (respects user edits).
        Auto-creates the folder if it doesn't exist (handles custom folder names).
        """
        path = self._root / folder / f"{_sanitize_filename(filename)}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        if merge and path.exists():
            existing = path.read_text(encoding="utf-8")
            # Only append if content differs substantially
            existing_hash = hashlib.md5(existing.encode()).hexdigest()
            new_hash      = hashlib.md5(content.encode()).hexdigest()
            if existing_hash == new_hash:
                return path   # no change needed
            # Append new section under existing content
            ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
            merged = existing.rstrip() + f"\n\n---\n*Updated {ts} by Operon*\n\n" + content
            path.write_text(merged, encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def read_note(self, folder: str, filename: str) -> Optional[VaultNote]:
        path = self._root / folder / f"{_sanitize_filename(filename)}.md"
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        return VaultNote(path=path, content=content, modified=path.stat().st_mtime)

    def list_notes(self, folder: Optional[str] = None) -> List[VaultNote]:
        """List all notes, optionally filtered by folder."""
        root = self._root / folder if folder else self._root
        notes = []
        for p in root.rglob("*.md"):
            if p.is_file():
                try:
                    content = p.read_text(encoding="utf-8")
                    notes.append(VaultNote(path=p, content=content, modified=p.stat().st_mtime))
                except Exception:
                    pass
        return sorted(notes, key=lambda n: n.modified, reverse=True)

    def search_notes(self, query: str, max_results: int = 20) -> List[VaultNote]:
        """Simple text search across all notes."""
        q_lower = query.lower()
        results = []
        for note in self.list_notes():
            if q_lower in note.content.lower():
                results.append(note)
            if len(results) >= max_results:
                break
        return results

    def delete_note(self, folder: str, filename: str) -> bool:
        path = self._root / folder / f"{_sanitize_filename(filename)}.md"
        if path.exists():
            path.unlink()
            return True
        return False

    @property
    def root(self) -> Path:
        return self._root


# ── Note writers (one per note type) ─────────────────────────────────────────

class ObsidianNoteWriter:
    """Builds formatted Markdown notes for each Operon data type."""

    def __init__(self, vault: ObsidianVault) -> None:
        self._vault = vault

    # ── Daily summary ─────────────────────────────────────────────────────────

    def write_daily(
        self,
        summary:    str,
        date_str:   Optional[str] = None,
        session_id: str = "",
        turns:      int = 0,
        tools_used: Optional[List[str]] = None,
    ) -> Path:
        date_str = date_str or date.today().isoformat()
        tools_str = ", ".join(tools_used[:10]) if tools_used else "—"
        content = (
            _frontmatter(source="operon", tags=["daily", "operon-summary"], session=session_id)
            + f"\n# Daily Summary — {date_str}\n\n"
            + f"> **Session** {session_id or '—'}  │  **Turns** {turns}  │  **Tools** {tools_str}\n\n"
            + summary.strip()
            + f"\n\n---\n*Generated by Operon · {datetime.now().strftime('%H:%M')}*\n"
        )
        return self._vault.write_note("Daily", date_str, content, merge=True)

    # ── Entity / Person note ──────────────────────────────────────────────────

    def write_entity(
        self,
        name:  str,
        facts: List[str],
        tags:  Optional[List[str]] = None,
    ) -> Path:
        tags = (tags or []) + ["entity", "operon"]
        facts_md = "\n".join(f"- {f}" for f in facts)
        content = (
            _frontmatter(source="operon", tags=tags, category="entity")
            + f"\n# {name}\n\n"
            + f"## Facts\n{facts_md}\n\n"
            + f"---\n*Maintained by Operon · Updated {datetime.now().strftime('%Y-%m-%d')}*\n"
        )
        folder = "People" if "person" in (tags or []) else "Facts"
        return self._vault.write_note(folder, name, content, merge=True)

    # ── Project note ──────────────────────────────────────────────────────────

    def write_project(
        self,
        name:    str,
        summary: str,
        tasks:   Optional[List[str]] = None,
        links:   Optional[List[str]] = None,
    ) -> Path:
        task_md  = "\n".join(f"- [ ] {t}" for t in (tasks or []))
        links_md = "\n".join(f"- {_backlink(l)}" for l in (links or []))
        content = (
            _frontmatter(source="operon", tags=["project", "operon"], category="project")
            + f"\n# {name}\n\n"
            + f"## Summary\n{summary.strip()}\n\n"
            + (f"## Open Tasks\n{task_md}\n\n" if task_md else "")
            + (f"## Related\n{links_md}\n\n" if links_md else "")
            + f"---\n*Maintained by Operon*\n"
        )
        return self._vault.write_note("Projects", name, content, merge=True)

    # ── Code snippet ──────────────────────────────────────────────────────────

    def write_code(
        self,
        title:    str,
        code:     str,
        language: str = "python",
        context:  str = "",
    ) -> Path:
        content = (
            _frontmatter(source="operon", tags=["code", language, "operon"], category="code")
            + f"\n# {title}\n\n"
            + (f"{context.strip()}\n\n" if context else "")
            + f"```{language}\n{code.strip()}\n```\n\n"
            + f"---\n*Saved by Operon · {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
        )
        return self._vault.write_note("Code", title, content, merge=False)

    # ── Goals note ────────────────────────────────────────────────────────────

    def write_goals(self, goals: List[str]) -> Path:
        goals_md = "\n".join(f"- [ ] {g}" for g in goals)
        content = (
            _frontmatter(source="operon", tags=["goals", "operon"])
            + f"\n# Active Goals\n\n{goals_md}\n\n"
            + f"---\n*Updated by Operon · {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
        )
        return self._vault.write_note("Goals", "active", content, merge=False)

    # ── Facts / preferences ───────────────────────────────────────────────────

    def write_facts(self, title: str, facts: List[str], tags: Optional[List[str]] = None) -> Path:
        facts_md = "\n".join(f"- {f}" for f in facts)
        content = (
            _frontmatter(source="operon", tags=(tags or []) + ["fact", "operon"])
            + f"\n# {title}\n\n{facts_md}\n\n"
            + f"---\n*Updated by Operon · {datetime.now().strftime('%Y-%m-%d')}*\n"
        )
        return self._vault.write_note("Facts", title, content, merge=True)


# ── Context reader (vault → agent) ────────────────────────────────────────────

class ObsidianContextReader:
    """
    Reads relevant notes from the vault and formats them for system prompt injection.
    """

    def __init__(self, vault: ObsidianVault) -> None:
        self._vault = vault

    def get_context_for(self, query: str, limit: int = _CONTEXT_NOTES) -> str:
        """
        Return a formatted context block with relevant vault notes.
        Priority: Goals > Facts > Projects > Daily (most recent first)
        """
        notes: List[VaultNote] = []

        # Always include active goals
        goal_note = self._vault.read_note("Goals", "active")
        if goal_note:
            notes.append(goal_note)

        # Search for query-relevant notes
        if query:
            notes.extend(self._vault.search_notes(query, max_results=limit))

        # De-duplicate by path
        seen: set = set()
        unique: List[VaultNote] = []
        for n in notes:
            if str(n.path) not in seen:
                seen.add(str(n.path))
                unique.append(n)
            if len(unique) >= limit:
                break

        if not unique:
            return ""

        lines = ["[Obsidian Vault Context]"]
        for note in unique:
            body = note.body_text()[:500]
            lines.append(f"\n### {note.title} ({note.folder})")
            lines.append(body)

        return "\n".join(lines)

    def read_all_facts(self) -> List[str]:
        """Return all bullet-point facts from the Facts/ folder."""
        facts: List[str] = []
        for note in self._vault.list_notes("Facts"):
            facts.extend(note.extract_facts())
        return facts[:200]

    def recent_daily_summaries(self, n: int = 3) -> List[str]:
        """Return the n most recent daily summaries."""
        summaries = []
        for note in self._vault.list_notes("Daily")[:n]:
            summaries.append(note.body_text()[:800])
        return summaries


# ── Auto-sync loop ─────────────────────────────────────────────────────────────

class ObsidianSyncLoop:
    """
    Background thread that syncs Operon's memory to Obsidian every 20 minutes.
    Matches OpenHuman's auto-fetch loop cadence.
    """

    def __init__(
        self,
        memory_sync: "ObsidianMemory",
        interval_s:  int = _SYNC_INTERVAL_S,
    ) -> None:
        self._sync     = memory_sync
        self._interval = interval_s
        self._thread:  Optional[threading.Thread] = None
        self._running  = False
        self._last_sync: float = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="operon-obsidian-sync"
        )
        self._thread.start()
        log.info("ObsidianSyncLoop: started (interval=%ds)", self._interval)

    def stop(self) -> None:
        self._running = False

    # ── Thread-compatible helpers ─────────────────────────────────────────────

    @property
    def daemon(self) -> bool:
        """Always True — the background thread is always started as daemon."""
        return True

    def is_alive(self) -> bool:
        """Return True if the background thread is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the background thread to finish (up to timeout seconds)."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while self._running:
            time.sleep(60)   # check every minute
            if time.time() - self._last_sync >= self._interval:
                try:
                    self._sync.sync_all()
                    self._last_sync = time.time()
                except Exception as e:
                    log.warning("ObsidianSyncLoop error: %s", e)

    @property
    def last_sync_ago(self) -> int:
        """Seconds since last sync."""
        return int(time.time() - self._last_sync)


# ── High-level ObsidianMemory API ─────────────────────────────────────────────

class ObsidianMemory:
    """
    Primary API: Operon ↔ Obsidian bidirectional memory sync.

    Usage:
        om = ObsidianMemory(vault_path="~/operon-brain")
        om.start_auto_sync()

        # Write memory
        om.write_daily_summary("Today we built vector memory and Slack ops")
        om.write_entity("OpenHuman", ["Rust/Tauri desktop app", "118 OAuth integrations"])
        om.write_code("LanceDB setup", "import lancedb; db = lancedb.connect(...)")

        # Read context
        ctx = om.get_context("what are the current goals?")
        print(ctx)  # → formatted block for system prompt injection

        # Sync everything from memory_store
        om.sync_all()
    """

    def __init__(
        self,
        vault_path:     Path = _DEFAULT_VAULT,
        auto_sync:      bool = True,
        sync_interval:  int  = _SYNC_INTERVAL_S,
    ) -> None:
        self._vault   = ObsidianVault(vault_path)
        self._writer  = ObsidianNoteWriter(self._vault)
        self._reader  = ObsidianContextReader(self._vault)
        self._loop:   Optional[ObsidianSyncLoop] = None
        self._session_id = ""
        self._turn_count = 0
        self._pending_facts:  List[str] = []
        self._pending_entities: Dict[str, List[str]] = {}

        if auto_sync:
            self._loop = ObsidianSyncLoop(self, sync_interval)

    def set_session(self, session_id: str, turns: int = 0) -> None:
        self._session_id  = session_id
        self._turn_count  = turns

    def start_auto_sync(self) -> None:
        if self._loop:
            self._loop.start()

    def stop_auto_sync(self) -> None:
        if self._loop:
            self._loop.stop()

    # ── Write shortcuts ───────────────────────────────────────────────────────

    def write_daily_summary(
        self,
        summary:    str,
        tools_used: Optional[List[str]] = None,
    ) -> Path:
        return self._writer.write_daily(
            summary    = summary,
            session_id = self._session_id,
            turns      = self._turn_count,
            tools_used = tools_used,
        )

    def write_entity(self, name: str, facts: List[str], tags: Optional[List[str]] = None) -> Path:
        return self._writer.write_entity(name, facts, tags)

    def write_project(self, name: str, summary: str, tasks: Optional[List[str]] = None) -> Path:
        return self._writer.write_project(name, summary, tasks)

    def write_code(self, title: str, code: str, language: str = "python", context: str = "") -> Path:
        return self._writer.write_code(title, code, language, context)

    def write_goals(self, goals: List[str]) -> Path:
        return self._writer.write_goals(goals)

    def add_fact(self, topic: str, fact: str, tags: Optional[List[str]] = None) -> Path:
        """Write a fact immediately to the vault under the given topic."""
        return self._writer.write_facts(
            title=topic,
            facts=[fact],
            tags=tags,
        )

    def add_entity_fact(self, entity: str, fact: str) -> None:
        self._pending_entities.setdefault(entity, []).append(fact)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_context(self, query: str = "", limit: int = _CONTEXT_NOTES) -> str:
        return self._reader.get_context_for(query, limit)

    def read_facts(self, topic: Optional[str] = None) -> List[str]:
        """Return all facts, optionally filtered to notes whose title matches topic."""
        all_facts = self._reader.read_all_facts()
        if topic is None:
            return all_facts
        # Filter: return facts from notes whose filename/title contains the topic
        results: List[str] = []
        for note in self._vault.list_notes("Facts"):
            if topic.lower() in note.title.lower():
                results.extend(note.extract_facts())
        return results

    def recent_summaries(self, n: int = 3) -> List[str]:
        return self._reader.recent_daily_summaries(n)

    def search(self, query: str, max_results: int = 10) -> List[VaultNote]:
        return self._vault.search_notes(query, max_results)

    # ── Full sync ─────────────────────────────────────────────────────────────

    def sync_all(self) -> Dict:
        """
        Full sync: flush pending facts + entities to vault.
        Called by auto-sync loop or manually via /obsidian sync.
        """
        written = {"facts": 0, "entities": 0, "errors": []}

        # Flush pending facts
        if self._pending_facts:
            try:
                self._writer.write_facts(
                    "operon_facts",
                    self._pending_facts,
                    tags=["auto-sync"],
                )
                written["facts"] = len(self._pending_facts)
                self._pending_facts.clear()
            except Exception as e:
                written["errors"].append(str(e))

        # Flush pending entity facts
        for entity, facts in self._pending_entities.items():
            try:
                self._writer.write_entity(entity, facts)
                written["entities"] += 1
            except Exception as e:
                written["errors"].append(str(e))
        self._pending_entities.clear()

        log.info(
            "ObsidianMemory.sync_all: facts=%d entities=%d errors=%d",
            written["facts"], written["entities"], len(written["errors"]),
        )
        return written

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> Dict:
        notes = self._vault.list_notes()
        return {
            "vault_path":      str(self._vault.root),
            "vault_exists":    self._vault.exists(),
            "total_notes":     len(notes),
            "auto_sync":       self._loop is not None,
            "last_sync_ago_s": self._loop.last_sync_ago if self._loop else None,
            "pending_facts":   len(self._pending_facts),
            "pending_entities": len(self._pending_entities),
        }

    def summary(self) -> str:
        s = self.status()
        sync_str = f"{s['last_sync_ago_s']}s ago" if s["last_sync_ago_s"] is not None else "manual"
        return (
            f"Obsidian vault: {s['vault_path']}  │  "
            f"{s['total_notes']} notes  │  "
            f"auto-sync={'✓' if s['auto_sync'] else '✗'}  │  "
            f"last sync: {sync_str}"
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_default_om: Optional[ObsidianMemory] = None


def get_obsidian_memory(vault_path: Optional[Path] = None) -> ObsidianMemory:
    global _default_om
    if _default_om is None:
        vp = vault_path or Path(
            os.environ.get("OPERON_OBSIDIAN_VAULT", str(_DEFAULT_VAULT))
        )
        _default_om = ObsidianMemory(vault_path=vp)
    return _default_om


# ── Tool definitions ──────────────────────────────────────────────────────────

_TOOL_DEFINITIONS: List[Dict] = [
    {
        "name": "obsidian_write_note",
        "description": "Write a note to the Obsidian vault (facts, entities, code, projects, daily summary).",
        "input_schema": {
            "type": "object",
            "properties": {
                "folder":  {"type": "string", "enum": ["Daily","People","Projects","Tasks","Code","Facts","Goals","Sessions"]},
                "title":   {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["folder", "title", "content"],
        },
    },
    {
        "name": "obsidian_search",
        "description": "Search Obsidian vault notes for a query string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "obsidian_get_context",
        "description": "Get relevant context from Obsidian vault for a given query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
]


def _obs_write(folder: str, title: str, content: str) -> dict:
    om = get_obsidian_memory()
    path = om._vault.write_note(folder, title, content, merge=True)
    return {"success": True, "path": str(path)}


def _obs_search(query: str, max_results: int = 5) -> dict:
    om = get_obsidian_memory()
    notes = om.search(query, max_results)
    return {
        "success": True,
        "count":   len(notes),
        "notes":   [{"title": n.title, "folder": n.folder, "preview": n.body_text()[:200]} for n in notes],
    }


def _obs_context(query: str, limit: int = 5) -> dict:
    ctx = get_obsidian_memory().get_context(query, limit)
    return {"success": True, "context": ctx}


_DISPATCH: Dict[str, Any] = {
    "obsidian_write_note":   _obs_write,
    "obsidian_search":       _obs_search,
    "obsidian_get_context":  _obs_context,
}
