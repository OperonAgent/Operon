"""
core/model_router.py — Smart Multi-Model Routing

Automatically routes each task to the best available model based on
task type, available models, and user hints. Matches OpenHuman's
model routing layer.

Routing hints (add to any prompt):
  hint:reasoning  → best reasoning model (hermes3:8b)
  hint:code       → best coding model (qwen2.5-coder:7b)
  hint:fast       → fastest/cheapest model (qwen3:4b or qwen2.5:3b)
  hint:vision     → vision-capable model (if available)
  hint:local      → force local Ollama model
  hint:cloud      → force cloud API if configured

Auto-routing (no hint needed) classifies the task by:
  - Code keywords     → code model
  - Reasoning keywords → reasoning model
  - Short/simple tasks → fast model
  - Default           → configured default model

Available models (discovered from Ollama + configured API keys):
  hermes3:8b            — general reasoning + instruction following
  qwen2.5-coder:7b      — code generation, debugging, review
  qwen3:4b              — balanced speed/quality
  qwen2.5:3b            — fastest local option
  claude-* (if key set) — cloud reasoning, 200K context
  gpt-4o (if key set)   — cloud multimodal
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("operon.model_router")

# ── Routing hint patterns ─────────────────────────────────────────────────────
_HINT_RE = re.compile(
    r"\bhint:(reasoning|code|fast|vision|local|cloud|creative|analysis)\b",
    re.IGNORECASE,
)

# ── Task classification patterns ──────────────────────────────────────────────
_CODE_PATTERNS = re.compile(
    r"\b(code|function|class|def |import |python|javascript|typescript|rust|"
    r"debug|error|exception|traceback|script|program|implement|refactor|"
    r"unit test|pytest|unittest|sql|query|api|endpoint|bug|fix)\b",
    re.IGNORECASE,
)

_REASONING_PATTERNS = re.compile(
    r"\b(analyze|analyse|explain|compare|evaluate|assess|plan|strategy|"
    r"architecture|design|decision|pros and cons|trade.?off|summarize|"
    r"research|review|audit|think through|step.?by.?step|complex|deep dive)\b",
    re.IGNORECASE,
)

_FAST_PATTERNS = re.compile(
    r"\b(yes|no|quick|short|brief|one.?word|one.?line|simple|"
    r"what is|who is|when|where|list|name|define)\b",
    re.IGNORECASE,
)

_CREATIVE_PATTERNS = re.compile(
    r"\b(write|story|poem|creative|imagine|brainstorm|generate|draft|"
    r"compose|narrative|blog|article|essay)\b",
    re.IGNORECASE,
)


class TaskType(str, Enum):
    CODE       = "code"
    REASONING  = "reasoning"
    FAST       = "fast"
    VISION     = "vision"
    CREATIVE   = "creative"
    DEFAULT    = "default"


# ── Model profile ─────────────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    """Describes a model's capabilities and routing priority."""
    name:          str
    provider:      str        # ollama | anthropic | openai
    task_types:    List[TaskType] = field(default_factory=list)
    context_len:   int  = 4096
    speed_score:   int  = 5   # 1=slow, 10=fast
    quality_score: int  = 5   # 1=low, 10=high
    available:     bool = True
    is_vision:     bool = False
    cost_per_1k:   float = 0.0   # 0.0 = local/free

    @property
    def score_for(self) -> Dict[TaskType, float]:
        """Composite routing score per task type."""
        base = self.quality_score * 0.6 + self.speed_score * 0.4
        scores: Dict[TaskType, float] = {}
        for tt in TaskType:
            if tt in self.task_types:
                scores[tt] = base * 1.3   # bonus for specialty
            else:
                scores[tt] = base * 0.8
        return scores


# ── Built-in model profiles ───────────────────────────────────────────────────

_BUILT_IN_PROFILES: List[ModelProfile] = [
    ModelProfile(
        name="hermes3:8b",
        provider="ollama",
        task_types=[TaskType.REASONING, TaskType.DEFAULT, TaskType.CREATIVE],
        context_len=8192,
        speed_score=6,
        quality_score=8,
    ),
    ModelProfile(
        name="hermes3:8b-llama3.1-q4_K_M",
        provider="ollama",
        task_types=[TaskType.REASONING, TaskType.DEFAULT, TaskType.CREATIVE],
        context_len=8192,
        speed_score=6,
        quality_score=8,
    ),
    ModelProfile(
        name="qwen2.5-coder:7b",
        provider="ollama",
        task_types=[TaskType.CODE],
        context_len=16384,
        speed_score=7,
        quality_score=9,
    ),
    ModelProfile(
        name="qwen3:4b",
        provider="ollama",
        task_types=[TaskType.REASONING, TaskType.DEFAULT],
        context_len=8192,
        speed_score=8,
        quality_score=7,
    ),
    ModelProfile(
        name="qwen2.5:3b",
        provider="ollama",
        task_types=[TaskType.FAST],
        context_len=4096,
        speed_score=9,
        quality_score=5,
    ),
    ModelProfile(
        name="llama3.2:latest",
        provider="ollama",
        task_types=[TaskType.FAST],
        context_len=4096,
        speed_score=9,
        quality_score=4,
    ),
    ModelProfile(
        name="claude-sonnet-4-5",
        provider="anthropic",
        task_types=[TaskType.REASONING, TaskType.CREATIVE, TaskType.CODE, TaskType.DEFAULT],
        context_len=200000,
        speed_score=7,
        quality_score=10,
        cost_per_1k=0.003,
    ),
    ModelProfile(
        name="claude-opus-4-5",
        provider="anthropic",
        task_types=[TaskType.REASONING, TaskType.CREATIVE, TaskType.CODE],
        context_len=200000,
        speed_score=5,
        quality_score=10,
        cost_per_1k=0.015,
    ),
    ModelProfile(
        name="gpt-4o",
        provider="openai",
        task_types=[TaskType.REASONING, TaskType.CREATIVE, TaskType.VISION, TaskType.CODE],
        context_len=128000,
        speed_score=7,
        quality_score=9,
        is_vision=True,
        cost_per_1k=0.005,
    ),
]


# ── Ollama discovery ──────────────────────────────────────────────────────────

def _discover_ollama_models() -> List[str]:
    """Query Ollama for locally available models."""
    try:
        out = subprocess.check_output(
            ["ollama", "list"], stderr=subprocess.DEVNULL, timeout=5
        ).decode()
        models = []
        for line in out.splitlines()[1:]:   # skip header
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []


# ── Route decision ────────────────────────────────────────────────────────────

@dataclass
class RouteDecision:
    model:      str
    provider:   str
    task_type:  TaskType
    reason:     str
    hint_used:  Optional[str] = None
    fallback:   bool = False


# ── ModelRouter ───────────────────────────────────────────────────────────────

class SmartModelRouter:
    """
    Routes each prompt to the best available model.

    Priority order:
      1. Explicit hint in prompt (hint:code, hint:fast, etc.)
      2. Cloud model if API key set and task warrants it
      3. Classified task type → specialty local model
      4. Default model from config
    """

    def __init__(
        self,
        default_model:    str = "hermes3:8b",
        prefer_local:     bool = True,
        anthropic_key:    Optional[str] = None,
        openai_key:       Optional[str] = None,
    ) -> None:
        self._default      = default_model
        self._prefer_local = prefer_local
        self._anthro_key   = anthropic_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._openai_key   = openai_key    or os.environ.get("OPENAI_API_KEY", "")

        # Build model registry
        self._profiles: Dict[str, ModelProfile] = {p.name: p for p in _BUILT_IN_PROFILES}

        # Discover Ollama models in background so __init__ returns immediately
        self._available_local: set = set()
        self._discovery_done  = False
        import threading as _threading
        _t = _threading.Thread(target=self._discover_models, daemon=True, name="router-discovery")
        _t.start()

    def _discover_models(self) -> None:
        """Background thread: discover Ollama models and update profiles."""
        self._available_local = set(_discover_ollama_models())
        for name, profile in self._profiles.items():
            if profile.provider == "ollama":
                profile.available = name in self._available_local
            elif profile.provider == "anthropic":
                profile.available = bool(self._anthro_key)
            elif profile.provider == "openai":
                profile.available = bool(self._openai_key)
        self._discovery_done = True
        log.info(
            "SmartModelRouter: %d local models available: %s",
            len(self._available_local),
            ", ".join(sorted(self._available_local)),
        )

    # ── Classification ────────────────────────────────────────────────────────

    def classify(self, prompt: str) -> Tuple[TaskType, Optional[str]]:
        """
        Classify a prompt → (TaskType, hint_str_or_None).
        Hint takes highest priority.
        """
        m = _HINT_RE.search(prompt)
        if m:
            hint = m.group(1).lower()
            type_map = {
                "reasoning": TaskType.REASONING,
                "code":      TaskType.CODE,
                "fast":      TaskType.FAST,
                "vision":    TaskType.VISION,
                "creative":  TaskType.CREATIVE,
                "local":     TaskType.DEFAULT,
                "cloud":     TaskType.REASONING,
                "analysis":  TaskType.REASONING,
            }
            return type_map.get(hint, TaskType.DEFAULT), hint

        # Auto-classify
        if _CODE_PATTERNS.search(prompt):
            return TaskType.CODE, None
        if _REASONING_PATTERNS.search(prompt):
            return TaskType.REASONING, None
        if _FAST_PATTERNS.search(prompt) and len(prompt) < 120:
            return TaskType.FAST, None
        if _CREATIVE_PATTERNS.search(prompt):
            return TaskType.CREATIVE, None
        return TaskType.DEFAULT, None

    # ── Route ─────────────────────────────────────────────────────────────────

    def route(self, prompt: str) -> RouteDecision:
        """
        Determine the best model for this prompt.
        Returns RouteDecision with model name + provider + reasoning.
        """
        task_type, hint = self.classify(prompt)

        # Force cloud if hint:cloud and key available
        if hint == "cloud":
            if self._anthro_key:
                return RouteDecision(
                    model="claude-sonnet-4-5", provider="anthropic",
                    task_type=task_type, reason="hint:cloud → Claude Sonnet",
                    hint_used=hint,
                )
            if self._openai_key:
                return RouteDecision(
                    model="gpt-4o", provider="openai",
                    task_type=task_type, reason="hint:cloud → GPT-4o",
                    hint_used=hint,
                )

        # Force local if hint:local
        if hint == "local":
            model = self._best_local(task_type)
            return RouteDecision(
                model=model, provider="ollama",
                task_type=task_type, reason=f"hint:local → {model}",
                hint_used=hint,
            )

        # For vision tasks, check vision models
        if task_type == TaskType.VISION:
            if self._openai_key:
                return RouteDecision(
                    model="gpt-4o", provider="openai",
                    task_type=task_type, reason="vision → GPT-4o",
                    hint_used=hint,
                )
            # Fallback to default (most local models don't support vision)
            return RouteDecision(
                model=self._default, provider="ollama",
                task_type=task_type, reason="vision: no vision model, using default",
                hint_used=hint, fallback=True,
            )

        # Route by task type to best local model
        model = self._best_local(task_type)
        return RouteDecision(
            model=model, provider="ollama",
            task_type=task_type, reason=f"auto:{task_type.value} → {model}",
            hint_used=hint,
        )

    def _best_local(self, task_type: TaskType) -> str:
        """Return the best available local model for a task type."""
        candidates = [
            p for p in self._profiles.values()
            if p.provider == "ollama" and p.available
        ]
        if not candidates:
            return self._default

        # Score each candidate
        scored = sorted(
            candidates,
            key=lambda p: p.score_for.get(task_type, 5.0),
            reverse=True,
        )
        return scored[0].name if scored else self._default

    # ── Strip hints from prompt ───────────────────────────────────────────────

    @staticmethod
    def strip_hints(prompt: str) -> str:
        """Remove hint:xxx tokens from prompt before sending to model."""
        return _HINT_RE.sub("", prompt).strip()

    # ── Available models ──────────────────────────────────────────────────────

    def available_models(self) -> List[Dict]:
        """List all known models with availability status."""
        return [
            {
                "name":      p.name,
                "provider":  p.provider,
                "available": p.available,
                "tasks":     [t.value for t in p.task_types],
                "context":   p.context_len,
                "cost_per_1k": p.cost_per_1k,
            }
            for p in self._profiles.values()
        ]

    def best_for(self, task: str) -> str:
        """Convenience: get model name for a task type string."""
        try:
            tt = TaskType(task.lower())
        except ValueError:
            tt = TaskType.DEFAULT
        return self._best_local(tt)

    def status(self) -> str:
        available = [p.name for p in self._profiles.values() if p.available]
        return (
            f"SmartModelRouter: default={self._default}  │  "
            f"available={len(available)}: {', '.join(available[:5])}"
            + ("..." if len(available) > 5 else "")
        )

    def refresh(self) -> None:
        """Re-discover available Ollama models."""
        self._available_local = set(_discover_ollama_models())
        for name, profile in self._profiles.items():
            if profile.provider == "ollama":
                profile.available = name in self._available_local

    def add_profile(self, profile: ModelProfile) -> None:
        """Register a custom model profile."""
        self._profiles[profile.name] = profile


# ── Module-level singleton ────────────────────────────────────────────────────

_router: Optional[SmartModelRouter] = None


def get_smart_router(default_model: str = "hermes3:8b") -> SmartModelRouter:
    global _router
    if _router is None:
        _router = SmartModelRouter(default_model=default_model)
    return _router


def route_prompt(prompt: str, default_model: str = "hermes3:8b") -> RouteDecision:
    """Top-level routing function used by main.py agent loop."""
    return get_smart_router(default_model).route(prompt)


def classify_prompt(prompt: str) -> Tuple[str, Optional[str]]:
    """Return (task_type_str, hint_str) for a prompt."""
    tt, hint = get_smart_router().classify(prompt)
    return tt.value, hint


def strip_hints(prompt: str) -> str:
    return SmartModelRouter.strip_hints(prompt)
