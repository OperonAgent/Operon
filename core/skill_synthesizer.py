"""
core/skill_synthesizer.py — Self-Improvement Loop (Hermes-style)

After every successful multi-step task, Operon analyses the trajectory
and synthesizes a reusable skill. The next time a similar task arrives,
the skill is loaded automatically — Operon gets faster and better over
time without any manual intervention.

This closes the single largest competitive gap: Hermes scores 10/10 on
self-improvement; without this module Operon scores 4/10.

Pipeline:
  1. TaskTrajectory  — captures tool calls + results for one agent run
  2. TrajectoryAnalyser — extracts the "recipe" (steps, inputs, outputs)
  3. SkillWriter     — formats the recipe as a reusable Python skill stub
  4. SkillStore      — persists skills to ~/.operon/synthesized_skills/
  5. SkillMatcher    — at task start, retrieves similar skills for context
  6. SkillSynthesizer — orchestrates the full loop

Synthesized skill format (same as Operon's existing skills):
  ---
  name: <slug>
  description: <one-line>
  trigger: <keywords that activate this skill>
  ---
  ## Steps
  1. <step description> → tool: <tool_name>  params: <key params>
  2. ...
  ## Notes
  <any gotchas or conditions>
  ## Example
  User: <original user request>
  Result: <summary of outcome>

Usage:
    synth = SkillSynthesizer()

    # After agent run completes:
    trajectory = synth.get_trajectory()   # built during the run
    skill = synth.synthesize(trajectory, user_request="...", outcome="success")
    print(f"Synthesized: {skill.name}")

    # Before next agent run:
    hints = synth.get_hints_for("deploy my flask app to EC2")
    # → formatted skill context to inject into system prompt
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("operon.skill_synthesizer")

# ── Storage ───────────────────────────────────────────────────────────────────
_SKILLS_DIR      = Path.home() / ".operon" / "synthesized_skills"
_INDEX_FILE      = _SKILLS_DIR / "index.json"
_MAX_SKILLS      = 500
_MIN_STEPS       = 2      # don't synthesize single-step trivial tasks
_MIN_QUALITY     = 0.5    # trajectory quality threshold (0–1)
_SIMILARITY_THRESHOLD = 0.35   # keyword overlap for skill matching

# ── Step record ───────────────────────────────────────────────────────────────

@dataclass
class TrajectoryStep:
    """One tool call in an agent trajectory."""
    tool_name:   str
    params:      Dict[str, Any]
    result:      Any
    success:     bool
    duration_ms: float = 0.0
    timestamp:   float = field(default_factory=time.time)

    def param_summary(self, max_len: int = 80) -> str:
        """One-line param summary, truncated."""
        try:
            s = json.dumps(self.params, ensure_ascii=False)
            return s[:max_len] + ("…" if len(s) > max_len else "")
        except Exception:
            return str(self.params)[:max_len]

    def result_summary(self, max_len: int = 60) -> str:
        """One-line result summary."""
        if isinstance(self.result, dict):
            r = self.result.get("output", self.result.get("result", self.result.get("text", "")))
        else:
            r = self.result
        s = str(r)[:max_len]
        return s + ("…" if len(str(r)) > max_len else "")


# ── Trajectory ────────────────────────────────────────────────────────────────

@dataclass
class TaskTrajectory:
    """Full record of one agent run."""
    user_request: str = ""
    steps:        List[TrajectoryStep] = field(default_factory=list)
    final_answer: str = ""
    success:      bool = False
    start_time:   float = field(default_factory=time.time)
    end_time:     float = 0.0
    session_id:   str = ""

    def add_step(
        self,
        tool_name:   str,
        params:      Dict,
        result:      Any,
        success:     bool = True,
        duration_ms: float = 0.0,
    ) -> None:
        self.steps.append(TrajectoryStep(
            tool_name   = tool_name,
            params      = params,
            result      = result,
            success     = success,
            duration_ms = duration_ms,
        ))

    def finish(self, answer: str, success: bool = True) -> None:
        self.final_answer = answer
        self.success      = success
        self.end_time     = time.time()

    @property
    def duration_s(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def tools_used(self) -> List[str]:
        return list(dict.fromkeys(s.tool_name for s in self.steps))

    @property
    def successful_steps(self) -> List[TrajectoryStep]:
        return [s for s in self.steps if s.success]

    def quality_score(self) -> float:
        """
        Heuristic quality score (0–1) for this trajectory.
        High quality = many successful steps + clear outcome + not too long.
        """
        if not self.steps:
            return 0.0
        n_steps   = len(self.steps)
        n_success = len(self.successful_steps)
        success_ratio = n_success / max(n_steps, 1)
        has_answer    = 1.0 if len(self.final_answer) > 20 else 0.3
        length_penalty = max(0.5, 1.0 - (n_steps - 10) * 0.02)  # penalize very long runs
        task_quality  = 1.0 if self.success else 0.4
        return min(1.0, success_ratio * has_answer * length_penalty * task_quality)


# ── Trajectory analyser ───────────────────────────────────────────────────────

class TrajectoryAnalyser:
    """
    Extracts a structured recipe from a completed trajectory.
    """

    def analyse(self, traj: TaskTrajectory) -> Dict:
        """
        Returns a recipe dict:
          {
            "intent":      str,   # inferred task intent
            "steps":       [...], # (step_num, tool, params_summary, result_summary)
            "tools_used":  [...],
            "key_params":  {...}, # most important params extracted
            "quality":     float,
            "duration_s":  float,
          }
        """
        intent = self._infer_intent(traj.user_request)
        steps  = [
            {
                "n":      i + 1,
                "tool":   s.tool_name,
                "params": s.param_summary(),
                "result": s.result_summary(),
                "ok":     s.success,
            }
            for i, s in enumerate(traj.successful_steps[:20])  # cap at 20 steps
        ]
        key_params = self._extract_key_params(traj.steps)

        return {
            "intent":     intent,
            "steps":      steps,
            "tools_used": traj.tools_used,
            "key_params": key_params,
            "quality":    traj.quality_score(),
            "duration_s": traj.duration_s,
        }

    def _infer_intent(self, request: str) -> str:
        """Extract verb + object from user request."""
        request = request.strip()
        if not request:
            return "complete a task"
        # Take first sentence, max 80 chars
        first = re.split(r"[.!?\n]", request)[0].strip()
        return first[:80]

    def _extract_key_params(self, steps: List[TrajectoryStep]) -> Dict:
        """Pull out the most important parameter types across all steps."""
        key: Dict[str, Any] = {}
        for step in steps:
            for k, v in step.params.items():
                if k in ("cmd", "command", "code", "path", "url", "query", "message"):
                    if k not in key and isinstance(v, str):
                        key[k] = v[:60]
        return key


# ── Synthesized skill ─────────────────────────────────────────────────────────

@dataclass
class SynthesizedSkill:
    """A synthesized reusable skill."""
    name:         str
    description:  str
    trigger:      str          # comma-separated keywords
    steps_md:     str          # Markdown step list
    notes:        str = ""
    example_req:  str = ""
    example_out:  str = ""
    tools_used:   List[str] = field(default_factory=list)
    quality:      float = 0.0
    created_at:   str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    use_count:    int = 0
    skill_id:     str = ""

    def __post_init__(self):
        if not self.skill_id:
            self.skill_id = hashlib.sha256(
                (self.name + self.description).encode()
            ).hexdigest()[:12]

    def to_markdown(self) -> str:
        """Serialize to Operon skill markdown format."""
        return (
            f"---\n"
            f"name: {self.name}\n"
            f"description: {self.description}\n"
            f"trigger: {self.trigger}\n"
            f"tools: {', '.join(self.tools_used)}\n"
            f"quality: {self.quality:.2f}\n"
            f"use_count: {self.use_count}\n"
            f"created: {self.created_at}\n"
            f"skill_id: {self.skill_id}\n"
            f"---\n\n"
            f"## Steps\n{self.steps_md}\n\n"
            + (f"## Notes\n{self.notes}\n\n" if self.notes else "")
            + (f"## Example\n**User:** {self.example_req}\n\n**Result:** {self.example_out}\n" if self.example_req else "")
        )

    @staticmethod
    def from_markdown(text: str) -> Optional["SynthesizedSkill"]:
        """Parse a skill from its markdown representation."""
        fm_match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
        if not fm_match:
            return None
        fm: Dict[str, str] = {}
        for line in fm_match.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                fm[k.strip()] = v.strip()

        body  = text[fm_match.end():]
        steps = ""
        steps_m = re.search(r"## Steps\n(.*?)(?=\n## |\Z)", body, re.DOTALL)
        if steps_m:
            steps = steps_m.group(1).strip()

        return SynthesizedSkill(
            name        = fm.get("name", "unknown"),
            description = fm.get("description", ""),
            trigger     = fm.get("trigger", ""),
            steps_md    = steps,
            tools_used  = [t.strip() for t in fm.get("tools", "").split(",") if t.strip()],
            quality     = float(fm.get("quality", 0.0)),
            use_count   = int(fm.get("use_count", 0)),
            created_at  = fm.get("created", ""),
            skill_id    = fm.get("skill_id", ""),
        )

    def to_dict(self) -> Dict:
        return {
            "skill_id":   self.skill_id,
            "name":       self.name,
            "description": self.description,
            "trigger":    self.trigger,
            "tools_used": self.tools_used,
            "quality":    self.quality,
            "use_count":  self.use_count,
            "created_at": self.created_at,
        }


# ── Skill writer ──────────────────────────────────────────────────────────────

class SkillWriter:
    """
    Converts a trajectory recipe into a SynthesizedSkill.
    Uses heuristics only — no LLM call needed (works fully local).
    """

    def write(
        self,
        recipe:      Dict,
        user_request: str,
        outcome:     str = "",
    ) -> SynthesizedSkill:
        name        = self._make_name(user_request, recipe["tools_used"])
        description = self._make_description(recipe["intent"], recipe["tools_used"])
        trigger     = self._make_trigger(user_request, recipe["tools_used"])
        steps_md    = self._format_steps(recipe["steps"])
        notes       = self._make_notes(recipe)

        return SynthesizedSkill(
            name        = name,
            description = description,
            trigger     = trigger,
            steps_md    = steps_md,
            notes       = notes,
            example_req = user_request[:200],
            example_out = outcome[:200],
            tools_used  = recipe["tools_used"],
            quality     = recipe["quality"],
        )

    def _make_name(self, request: str, tools: List[str]) -> str:
        """Derive a slug name from the request."""
        words = re.findall(r"\b[a-z][a-z0-9]+\b", request.lower())
        stop  = {"the","a","an","to","of","in","on","for","and","with","my","your","i","we","it","is","are","do","get","a"}
        sig   = [w for w in words if w not in stop][:4]
        if not sig and tools:
            sig = [tools[0].replace("_", "-")]
        name  = "-".join(sig) or "operon-skill"
        return name[:50]

    def _make_description(self, intent: str, tools: List[str]) -> str:
        tool_str = "+".join(tools[:3]) if tools else "multi-step"
        return f"{intent} using {tool_str}"[:100]

    def _make_trigger(self, request: str, tools: List[str]) -> str:
        """Extract trigger keywords from request + tool names."""
        words = re.findall(r"\b[a-z][a-z0-9]+\b", request.lower())
        stop  = {"the","a","an","to","of","in","on","for","and","with","my","your","i","we","it","is","are","do"}
        kw    = [w for w in words if w not in stop and len(w) > 3][:6]
        # Add tool names as triggers
        for t in tools[:3]:
            base = t.split("_")[0]
            if base not in kw:
                kw.append(base)
        return ", ".join(kw[:8])

    def _format_steps(self, steps: List[Dict]) -> str:
        lines = []
        for s in steps:
            status = "✓" if s["ok"] else "✗"
            lines.append(
                f"{s['n']}. [{status}] `{s['tool']}` — params: `{s['params']}` → {s['result']}"
            )
        return "\n".join(lines)

    def _make_notes(self, recipe: Dict) -> str:
        notes = []
        if recipe["duration_s"] > 30:
            notes.append(f"Long-running task (~{recipe['duration_s']:.0f}s). Consider breaking into subtasks.")
        if recipe["quality"] < 0.7:
            notes.append("Partial success — some steps may need adjustment.")
        tools = recipe["tools_used"]
        if "shell_exec" in tools or "python_exec" in tools:
            notes.append("Contains code execution — verify environment before re-running.")
        return "\n".join(notes)


# ── Skill store ───────────────────────────────────────────────────────────────

class SkillStore:
    """Persists synthesized skills to ~/.operon/synthesized_skills/."""

    def __init__(self, skills_dir: Path = _SKILLS_DIR) -> None:
        self._dir = skills_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index: List[Dict] = self._load_index()

    def _load_index(self) -> List[Dict]:
        idx_file = self._dir / "index.json"
        if idx_file.exists():
            try:
                return json.loads(idx_file.read_text())
            except Exception:
                pass
        return []

    def _save_index(self) -> None:
        idx_file = self._dir / "index.json"
        idx_file.write_text(json.dumps(self._index, indent=2))

    def save(self, skill: SynthesizedSkill) -> Path:
        path = self._dir / f"{skill.skill_id}.md"
        path.write_text(skill.to_markdown(), encoding="utf-8")

        # Update index
        self._index = [e for e in self._index if e.get("skill_id") != skill.skill_id]
        self._index.append(skill.to_dict())
        # Sort by quality desc, cap at MAX_SKILLS
        self._index.sort(key=lambda e: e.get("quality", 0), reverse=True)
        if len(self._index) > _MAX_SKILLS:
            # Remove worst quality skills
            to_remove = self._index[_MAX_SKILLS:]
            for e in to_remove:
                p = self._dir / f"{e['skill_id']}.md"
                if p.exists():
                    p.unlink()
            self._index = self._index[:_MAX_SKILLS]

        self._save_index()
        return path

    def load(self, skill_id: str) -> Optional[SynthesizedSkill]:
        path = self._dir / f"{skill_id}.md"
        if not path.exists():
            return None
        try:
            return SynthesizedSkill.from_markdown(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def load_all(self) -> List[SynthesizedSkill]:
        skills = []
        for entry in self._index:
            sk = self.load(entry["skill_id"])
            if sk:
                skills.append(sk)
        return skills

    def increment_use(self, skill_id: str) -> None:
        for entry in self._index:
            if entry.get("skill_id") == skill_id:
                entry["use_count"] = entry.get("use_count", 0) + 1
                self._save_index()
                break

    def count(self) -> int:
        return len(self._index)

    def search_index(self, query: str, top_k: int = 5) -> List[Dict]:
        """Keyword search over the skill index."""
        q_words = set(re.findall(r"\b[a-z][a-z0-9]+\b", query.lower()))
        scored  = []
        for entry in self._index:
            trigger_words = set(re.findall(r"\b[a-z][a-z0-9]+\b", entry.get("trigger", "").lower()))
            desc_words    = set(re.findall(r"\b[a-z][a-z0-9]+\b", entry.get("description", "").lower()))
            all_words     = trigger_words | desc_words
            overlap       = len(q_words & all_words) / max(len(q_words | all_words), 1)
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]


# ── Skill matcher ─────────────────────────────────────────────────────────────

class SkillMatcher:
    """
    At the start of each agent run, retrieves relevant synthesized skills
    and formats them as a context hint for the system prompt.
    """

    def __init__(self, store: SkillStore) -> None:
        self._store = store

    def find_relevant(self, request: str, top_k: int = 3) -> List[SynthesizedSkill]:
        """Return the top-k most relevant skills for this request."""
        matches = self._store.search_index(request, top_k=top_k)
        skills  = []
        for m in matches:
            sk = self._store.load(m["skill_id"])
            if sk and m.get("overlap", 1.0) >= _SIMILARITY_THRESHOLD:
                skills.append(sk)
                self._store.increment_use(sk.skill_id)
        return skills

    def build_context_block(self, request: str) -> str:
        """Build a system-prompt block with relevant skill hints."""
        skills = self.find_relevant(request)
        if not skills:
            return ""
        lines = ["[Synthesized Skills — previous solutions for similar tasks]"]
        for sk in skills:
            lines.append(f"\n### {sk.name} (quality={sk.quality:.1f}, used {sk.use_count}×)")
            lines.append(f"**Goal:** {sk.description}")
            lines.append(f"**Steps:**\n{sk.steps_md}")
            if sk.notes:
                lines.append(f"**Notes:** {sk.notes}")
        lines.append("\n*Use these as a guide; adapt to the current task.*")
        return "\n".join(lines)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class SkillSynthesizer:
    """
    Primary API — manages the full self-improvement loop.

    Typical usage in main.py:
        synth = SkillSynthesizer()

        # Before agent run — get skill hints
        hints = synth.get_hints_for(user_request)
        if hints:
            system_prompt += "\n\n" + hints

        # Agent runs...
        # Record each tool call:
        synth.record_step(tool_name, params, result, success)

        # After agent run:
        skill = synth.synthesize_from_current(user_request, final_answer)
        if skill:
            print(f"New skill: {skill.name}")
    """

    def __init__(self, skills_dir: Path = _SKILLS_DIR) -> None:
        self._analyser  = TrajectoryAnalyser()
        self._writer    = SkillWriter()
        self._store     = SkillStore(skills_dir)
        self._matcher   = SkillMatcher(self._store)
        self._current:  Optional[TaskTrajectory] = None

    # ── Trajectory management ─────────────────────────────────────────────────

    def start_trajectory(self, user_request: str, session_id: str = "") -> TaskTrajectory:
        self._current = TaskTrajectory(
            user_request = user_request,
            session_id   = session_id,
        )
        return self._current

    def record_step(
        self,
        tool_name:   str,
        params:      Dict,
        result:      Any,
        success:     bool = True,
        duration_ms: float = 0.0,
    ) -> None:
        if self._current is None:
            self._current = TaskTrajectory()
        self._current.add_step(tool_name, params, result, success, duration_ms)

    def finish_trajectory(self, answer: str, success: bool = True) -> None:
        if self._current:
            self._current.finish(answer, success)

    def get_trajectory(self) -> Optional[TaskTrajectory]:
        return self._current

    def reset_trajectory(self) -> None:
        self._current = None

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        trajectory:   TaskTrajectory,
        user_request: str = "",
        outcome:      str = "",
    ) -> Optional[SynthesizedSkill]:
        """
        Analyse trajectory and produce a skill. Returns None if not worth saving.
        """
        request = user_request or trajectory.user_request

        # Quality gate
        if len(trajectory.successful_steps) < _MIN_STEPS:
            log.debug("SkillSynthesizer: too few steps (%d), skipping", len(trajectory.steps))
            return None
        quality = trajectory.quality_score()
        if quality < _MIN_QUALITY:
            log.debug("SkillSynthesizer: quality %.2f below threshold, skipping", quality)
            return None

        recipe = self._analyser.analyse(trajectory)
        skill  = self._writer.write(recipe, request, outcome)
        path   = self._store.save(skill)
        log.info(
            "SkillSynthesizer: synthesized '%s' (quality=%.2f) → %s",
            skill.name, skill.quality, path,
        )
        return skill

    def synthesize_from_current(
        self,
        user_request: str = "",
        outcome:      str = "",
    ) -> Optional[SynthesizedSkill]:
        """Synthesize from the currently-active trajectory, then reset."""
        if not self._current:
            return None
        skill = self.synthesize(self._current, user_request, outcome)
        self.reset_trajectory()
        return skill

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_hints_for(self, request: str) -> str:
        """Return a formatted context block of relevant skills for injection."""
        return self._matcher.build_context_block(request)

    def list_skills(self, top_n: int = 20) -> List[Dict]:
        return self._store._index[:top_n]

    def delete_skill(self, skill_id: str) -> bool:
        path = self._store._dir / f"{skill_id}.md"
        if path.exists():
            path.unlink()
        before = len(self._store._index)
        self._store._index = [e for e in self._store._index if e.get("skill_id") != skill_id]
        if len(self._store._index) < before:
            self._store._save_index()
            return True
        return False

    def stats(self) -> Dict:
        skills = self._store.load_all()
        avg_quality = sum(s.quality for s in skills) / max(len(skills), 1)
        total_uses  = sum(s.use_count for s in skills)
        return {
            "total_skills":  self._store.count(),
            "avg_quality":   round(avg_quality, 2),
            "total_uses":    total_uses,
            "skills_dir":    str(self._store._dir),
            "active_run":    self._current is not None,
            "current_steps": len(self._current.steps) if self._current else 0,
        }

    def summary(self) -> str:
        s = self.stats()
        return (
            f"SkillSynthesizer: {s['total_skills']} skills  │  "
            f"avg quality={s['avg_quality']}  │  "
            f"total uses={s['total_uses']}  │  "
            f"active={'yes' if s['active_run'] else 'no'}"
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_synth: Optional[SkillSynthesizer] = None


def get_synthesizer() -> SkillSynthesizer:
    global _synth
    if _synth is None:
        _synth = SkillSynthesizer()
    return _synth
