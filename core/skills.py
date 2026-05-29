"""
Operon SKILL.md System.

Loads instruction packs from (in priority order):
  1. ~/.operon/skills/          — personal / global skills
  2. .operon/skills/            — project-local skills (relative to cwd)
  3. (future: remote skill registry)

Each .md file is a named skill that gets injected into the system prompt.

Optional YAML-style frontmatter at the top of a skill file:
  ---
  name: My Skill
  description: What this skill does
  enabled: true
  ---

Usage:
    from core.skills import SkillLoader
    loader = SkillLoader()
    block  = loader.as_system_block()   # inject into system prompt
"""

import re
from pathlib import Path
from typing import Optional

SKILLS_DIR = Path.home() / ".operon" / "skills"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Parse optional YAML-style frontmatter from a skill file.
    Returns (metadata_dict, body_text).
    """
    meta = {}
    if not text.startswith("---"):
        return meta, text

    end = text.find("\n---", 3)
    if end == -1:
        return meta, text

    fm_block = text[3:end].strip()
    body     = text[end + 4:].strip()

    for line in fm_block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if v.lower() == "true":
                v = True
            elif v.lower() == "false":
                v = False
            meta[k] = v

    return meta, body


class SkillLoader:
    """Discovers, loads, and manages SKILL.md instruction packs."""

    def __init__(self):
        self._skills: list[dict] = []   # {name, description, path, enabled, body}
        self.reload()

    def reload(self) -> int:
        """
        Scan all skill directories and reload skills from disk.
        Returns the number of skills loaded.
        """
        self._skills = []
        skill_dirs = [
            SKILLS_DIR,                          # personal skills
            Path(".operon") / "skills",          # project-local
        ]

        for skill_dir in skill_dirs:
            if not skill_dir.exists():
                continue
            for path in sorted(skill_dir.rglob("*.md")):
                self._load_file(path)

        return len(self._skills)

    def _load_file(self, path: Path) -> None:
        try:
            raw  = path.read_text(encoding="utf-8").strip()
            meta, body = _parse_frontmatter(raw)

            # Default skill name = filename without extension
            name    = meta.get("name", path.stem.replace("-", " ").replace("_", " ").title())
            enabled = meta.get("enabled", True)

            # Skip disabled skills
            if enabled is False or str(enabled).lower() == "false":
                return

            self._skills.append({
                "name":        name,
                "description": meta.get("description", ""),
                "path":        str(path),
                "enabled":     True,
                "body":        body,
            })
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def as_system_block(self) -> str:
        """
        Return a formatted block of all active skills for injection into the
        system prompt. Empty string if no skills are loaded.

        Skill file paths have the home directory replaced with ~ to save
        ~400-600 tokens per skill in the system prompt.
        (Adapted from OpenClaw src/agents/skills/workspace.ts path compaction.)
        """
        if not self._skills:
            return ""

        import os as _os
        home    = str(Path(_os.path.expanduser("~")))

        def _compact_path(text: str) -> str:
            """Replace home dir prefix with ~ in any file paths found in text."""
            return text.replace(home, "~") if home and home != "~" else text

        parts = ["════════════════════════════════════════════════════",
                 "LOADED SKILLS",
                 "════════════════════════════════════════════════════"]
        for skill in self._skills:
            header = f"### {skill['name']}"
            if skill["description"]:
                header += f"  —  {skill['description']}"
            parts.append(header)
            parts.append(_compact_path(skill["body"]))
            parts.append("")

        return "\n".join(parts)

    def list_skills(self) -> list[dict]:
        """Return metadata for all loaded skills (no body)."""
        return [
            {
                "name":        s["name"],
                "description": s["description"],
                "path":        s["path"],
                "enabled":     s["enabled"],
                "size":        len(s["body"]),
            }
            for s in self._skills
        ]

    def install(self, name: str, content: str) -> str:
        """
        Write a new skill file to ~/.operon/skills/<name>.md.
        Auto-reloads. Returns the path.
        """
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w\-]", "_", name.lower())
        path = SKILLS_DIR / f"{safe}.md"
        path.write_text(content, encoding="utf-8")
        self.reload()
        return str(path)

    def remove(self, name: str) -> bool:
        """Delete a skill file by name or filename stem. Returns True if found and deleted."""
        def _norm(s: str) -> str:
            return s.lower().replace(" ", "").replace("-", "").replace("_", "")

        name_norm = _norm(name)
        for skill in self._skills:
            p = Path(skill["path"])
            if _norm(p.stem) == name_norm or _norm(skill["name"]) == name_norm:
                try:
                    p.unlink()
                    self.reload()
                    return True
                except Exception:
                    return False
        return False

    def __len__(self) -> int:
        return len(self._skills)

    def __repr__(self) -> str:
        return f"<SkillLoader loaded={len(self._skills)}>"
