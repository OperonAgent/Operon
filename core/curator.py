"""
Operon Curator — Autonomous Skill Creation Loop.

After a complex agentic task completes (≥N tool calls), the Curator
analyses the exchange and asks the active model to write a reusable
SKILL.md instruction pack that would help with similar tasks in future.

The generated skill is installed silently into ~/.operon/skills/.
On the next run, SkillLoader picks it up automatically and it becomes
part of Operon's system prompt.

Design goals:
  • Zero user friction — runs in a background thread, never blocks
  • Quality gate — only fires when there are enough tool calls to extract
    a meaningful pattern (default: 4+)
  • Deduplication — won't create a skill whose first 80 chars match an
    existing skill
  • Hard cap — at most 30 auto-generated skills (prevents prompt bloat)
"""

import json
import re
import threading
import time
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("operon.curator")

SKILLS_DIR      = Path.home() / ".operon" / "skills"
_GRADES_FILE    = Path.home() / ".operon" / "skill_grades.json"
_AUTO_PREFIX    = "auto__"
_MIN_TOOL_CALLS = 4      # minimum tool calls to trigger curation
_MAX_AUTO_SKILLS = 30    # max number of auto-generated skills
_COOLDOWN       = 120    # seconds between curator runs (prevents thrashing)
_REWRITE_THRESHOLD = 0.4  # rewrite skills with success_rate < 40%
_MIN_GRADES     = 3       # minimum recorded outcomes before considering rewrite

_last_run: float = 0.0


# ── Skill grading ─────────────────────────────────────────────────────────────

def _load_grades() -> dict:
    """Load the skill grade dictionary from disk."""
    if _GRADES_FILE.exists():
        try:
            return json.loads(_GRADES_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_grades(grades: dict) -> None:
    """Persist the skill grade dictionary to disk."""
    try:
        _GRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
        _GRADES_FILE.write_text(json.dumps(grades, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("Curator: could not save grades: %s", e)


def record_skill_outcome(skill_name: str, success: bool) -> None:
    """
    Record whether a task that used a particular skill succeeded or failed.
    Called from the agent loop when a curator-generated skill was active.
    """
    grades = _load_grades()
    if skill_name not in grades:
        grades[skill_name] = {"success": 0, "failure": 0}
    if success:
        grades[skill_name]["success"] += 1
    else:
        grades[skill_name]["failure"] += 1
    _save_grades(grades)


def get_skill_grades() -> dict:
    """Return all skill grades with computed success_rate."""
    grades = _load_grades()
    result = {}
    for name, counts in grades.items():
        total = counts.get("success", 0) + counts.get("failure", 0)
        rate  = counts["success"] / total if total > 0 else None
        result[name] = {
            "success":      counts.get("success", 0),
            "failure":      counts.get("failure", 0),
            "total":        total,
            "success_rate": round(rate, 2) if rate is not None else None,
        }
    return result


def _skills_needing_rewrite() -> list[str]:
    """Return names of auto-generated skills whose success rate is below threshold."""
    grades  = _load_grades()
    results = []
    for name, counts in grades.items():
        total = counts.get("success", 0) + counts.get("failure", 0)
        if total < _MIN_GRADES:
            continue
        rate = counts["success"] / total
        if rate < _REWRITE_THRESHOLD:
            # Check if the skill file still exists
            path = SKILLS_DIR / f"{name}.md"
            if not path.exists():
                # Try with auto__ prefix
                path = SKILLS_DIR / f"{_AUTO_PREFIX}{name}.md"
            if path.exists():
                results.append(name)
    return results


# ── Skill generation ──────────────────────────────────────────────────────────

_REWRITE_PROMPT = """\
You are the Operon Skill Curator. The following SKILL.md has a low success rate \
(too many task failures when this skill was active). Rewrite it to be more accurate, \
actionable, and useful.

ORIGINAL SKILL (low success rate: {success_rate:.0%}  failures: {failure}  successes: {success}):
{original_content}

---
Rewrite the skill with the same name and format but with:
1. More accurate and specific steps
2. Better error-handling tips
3. Clearer criteria for when this skill applies

Respond with ONLY the improved SKILL.md content — no commentary, no code blocks.
The response must start with --- (frontmatter) and be a complete skill file.
"""

_CURATOR_PROMPT = """\
You are the Operon Skill Curator. Your job is to analyse a completed agentic task \
and write a concise SKILL.md instruction pack that would help the agent perform \
similar tasks more efficiently in the future.

The exchange below shows the tool calls and results from the just-completed task.

COMPLETED TASK EXCHANGE:
{exchange}

---
Write a SKILL.md file in this exact format:

---
name: <Short descriptive name, 2-5 words>
description: <One sentence — what situation this skill helps with>
enabled: true
---

## Overview
<2-3 sentences describing the pattern / workflow this skill encapsulates>

## Key Steps
1. <First important step>
2. <Second important step>
3. <etc.>

## Tips
- <Practical tip 1>
- <Practical tip 2>

## Example Usage
<Brief example of when the agent should apply this skill>

Respond with ONLY the SKILL.md content — no commentary, no code blocks, no markdown fences.
The response must start with --- and end after the last line of the skill body.
"""


def _count_tool_calls(messages: list[dict]) -> int:
    count = 0
    for m in messages:
        if m.get("role") == "assistant":
            try:
                data = json.loads(m.get("content", "{}"))
                if isinstance(data, dict):
                    action = data.get("action", {})
                    if action.get("type") == "tool":
                        count += 1
            except Exception:
                pass
    return count


def _extract_exchange_summary(messages: list[dict], max_chars: int = 4000) -> str:
    """Build a readable summary of the tool calls and responses."""
    lines = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "assistant":
            try:
                data = json.loads(content)
                action = data.get("action", {})
                if action.get("type") == "tool":
                    name   = action.get("tool_name", "?")
                    params = action.get("params", {})
                    lines.append(f"TOOL_CALL: {name}({_fmt_params(params)})")
                elif action.get("type") == "response":
                    lines.append(f"RESPONSE: {action.get('content', '')[:200]}")
            except Exception:
                pass
        elif role == "user" and content.startswith("[TOOL_RESULT:"):
            # Trim long results
            short = content[:300].replace("\n", " ")
            lines.append(f"  → {short}")

    summary = "\n".join(lines)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n[...truncated]"
    return summary


def _fmt_params(params: dict) -> str:
    pairs = [f"{k}={repr(str(v)[:30])}" for k, v in list(params.items())[:3]]
    return ", ".join(pairs)


def _existing_skill_fingerprints() -> set[str]:
    """Return the first 80 chars of all existing auto-generated skills."""
    fingerprints = set()
    if not SKILLS_DIR.exists():
        return fingerprints
    for p in SKILLS_DIR.glob(f"{_AUTO_PREFIX}*.md"):
        try:
            text = p.read_text(encoding="utf-8").strip()
            fingerprints.add(text[:80])
        except Exception:
            pass
    return fingerprints


def _count_auto_skills() -> int:
    if not SKILLS_DIR.exists():
        return 0
    return len(list(SKILLS_DIR.glob(f"{_AUTO_PREFIX}*.md")))


def _parse_skill_name(content: str) -> str:
    """Extract name from frontmatter."""
    m = re.search(r"^name:\s*(.+)$", content, re.MULTILINE)
    if m:
        name = m.group(1).strip()
        safe = re.sub(r"[^\w\-]", "_", name.lower())
        return f"{_AUTO_PREFIX}{safe}"
    return f"{_AUTO_PREFIX}skill_{int(time.time())}"


def _install_skill(content: str) -> Optional[str]:
    """Write the skill file to disk. Returns path or None on failure."""
    content = content.strip()
    if not content.startswith("---"):
        log.warning("Curator: generated skill doesn't start with frontmatter, skipping.")
        return None

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    name = _parse_skill_name(content)
    path = SKILLS_DIR / f"{name}.md"

    # Check for near-duplicates
    fingerprints = _existing_skill_fingerprints()
    if content[:80] in fingerprints:
        log.info("Curator: skill already exists (duplicate), skipping.")
        return None

    try:
        path.write_text(content, encoding="utf-8")
        log.info("Curator: installed skill → %s", path)
        return str(path)
    except Exception as e:
        log.warning("Curator: failed to write skill: %s", e)
        return None


# ── Skill rewriting ───────────────────────────────────────────────────────────

def _rewrite_skill_async(skill_name: str, router, skills_loader) -> None:
    """
    Rewrite a low-performing auto-generated skill in a background thread.
    Replaces the existing skill file with an improved version.
    """
    def _run():
        # Locate the skill file
        path = SKILLS_DIR / f"{skill_name}.md"
        if not path.exists():
            path = SKILLS_DIR / f"{_AUTO_PREFIX}{skill_name}.md"
        if not path.exists():
            log.warning("Curator rewrite: skill file not found for '%s'", skill_name)
            return

        try:
            original_content = path.read_text(encoding="utf-8")
        except Exception as e:
            log.warning("Curator rewrite: could not read skill '%s': %s", skill_name, e)
            return

        grades = _load_grades()
        counts = grades.get(skill_name, {"success": 0, "failure": 0})
        total  = counts["success"] + counts["failure"]
        rate   = counts["success"] / total if total > 0 else 0.0

        prompt = _REWRITE_PROMPT.format(
            success_rate    = rate,
            success         = counts["success"],
            failure         = counts["failure"],
            original_content = original_content,
        )
        try:
            raw = router.complete(
                system   = "You are the Operon Skill Curator. Only output SKILL.md content.",
                messages = [{"role": "user", "content": prompt}],
            )
            if not raw:
                return

            content = re.sub(r"^```(?:md|markdown)?\n?", "", raw.strip(), flags=re.IGNORECASE)
            content = re.sub(r"\n?```$", "", content.strip())

            if not content.startswith("---"):
                log.warning("Curator rewrite: model output doesn't start with frontmatter, skipping.")
                return

            path.write_text(content, encoding="utf-8")
            log.info("Curator: rewrote low-performing skill '%s' → %s", skill_name, path)

            # Reset grade counts after rewrite so we start fresh
            grades[skill_name] = {"success": 0, "failure": 0}
            _save_grades(grades)

            if skills_loader is not None:
                skills_loader.reload()
        except Exception as e:
            log.error("Curator rewrite error for '%s': %s", skill_name, e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ── Main curator function ─────────────────────────────────────────────────────

def curate_skill_async(
    messages:  list[dict],
    router,
    skills_loader,
    min_tool_calls: int = _MIN_TOOL_CALLS,
) -> None:
    """
    Non-blocking entry point — fires in a background daemon thread.
    Checks whether the completed exchange is worth curating, then
    generates and installs a skill if so.
    """
    global _last_run

    # Gate: cooldown
    now = time.time()
    if now - _last_run < _COOLDOWN:
        return

    # Gate: enough tool calls to extract a pattern?
    n_tools = _count_tool_calls(messages)
    if n_tools < min_tool_calls:
        return

    # Gate: auto-skill cap
    if _count_auto_skills() >= _MAX_AUTO_SKILLS:
        log.info("Curator: auto-skill cap reached (%d), skipping.", _MAX_AUTO_SKILLS)
        return

    _last_run = now

    def _run():
        try:
            exchange  = _extract_exchange_summary(messages)
            prompt    = _CURATOR_PROMPT.format(exchange=exchange)
            # Use a minimal messages list — just the curator prompt
            raw = router.complete(
                system   = "You are the Operon Skill Curator. Only output SKILL.md content.",
                messages = [{"role": "user", "content": prompt}],
            )
            if not raw:
                return

            # Strip any accidental markdown fences
            content = re.sub(r"^```(?:md|markdown)?\n?", "", raw.strip(), flags=re.IGNORECASE)
            content = re.sub(r"\n?```$", "", content.strip())

            path = _install_skill(content)
            if path and skills_loader is not None:
                skills_loader.reload()
                log.info("Curator: skills reloaded — new skill at %s", path)
        except Exception as e:
            log.error("Curator error: %s", e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ── Curator management class ──────────────────────────────────────────────────

class Curator:
    """
    Thin wrapper that keeps references to the router and skills loader
    and exposes a single method to trigger async curation.
    """

    def __init__(self, router, skills_loader, min_tool_calls: int = _MIN_TOOL_CALLS):
        self._router       = router
        self._skills       = skills_loader
        self._min_tools    = min_tool_calls
        self._enabled      = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def maybe_curate(self, messages: list[dict]) -> None:
        """Call after each agent loop completion to trigger curation if warranted."""
        if not self._enabled:
            return
        curate_skill_async(messages, self._router, self._skills, self._min_tools)
        # Also rewrite any low-performing skills (fires at most once per cooldown)
        self._maybe_rewrite()

    def _maybe_rewrite(self) -> None:
        """Check for low-performing skills and trigger rewrites (non-blocking)."""
        if not self._enabled:
            return
        try:
            for skill_name in _skills_needing_rewrite():
                log.info("Curator: scheduling rewrite for low-performer '%s'", skill_name)
                _rewrite_skill_async(skill_name, self._router, self._skills)
        except Exception as e:
            log.debug("Curator rewrite check error: %s", e)

    def record_outcome(self, skill_name: str, success: bool) -> None:
        """Record whether a task using this skill succeeded."""
        record_skill_outcome(skill_name, success)

    def list_auto_skills(self) -> list[dict]:
        """Return metadata for all auto-generated skills."""
        result = []
        if not SKILLS_DIR.exists():
            return result
        for p in sorted(SKILLS_DIR.glob(f"{_AUTO_PREFIX}*.md")):
            try:
                text = p.read_text(encoding="utf-8")
                m    = re.search(r"^name:\s*(.+)$", text, re.MULTILINE)
                name = m.group(1).strip() if m else p.stem
                result.append({"name": name, "path": str(p), "size": len(text)})
            except Exception:
                pass
        return result

    def get_grades(self) -> dict:
        """Return success/failure grades for all tracked skills."""
        return get_skill_grades()

    def clear_auto_skills(self) -> int:
        """Delete all auto-generated skills. Returns count deleted."""
        removed = 0
        if not SKILLS_DIR.exists():
            return removed
        for p in SKILLS_DIR.glob(f"{_AUTO_PREFIX}*.md"):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
        if self._skills is not None:
            self._skills.reload()
        return removed

    def __repr__(self):
        return (f"<Curator enabled={self._enabled} "
                f"min_tools={self._min_tools} "
                f"auto_skills={_count_auto_skills()}>")
