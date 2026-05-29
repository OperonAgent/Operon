"""
cost_tracker.py — Token usage and cost tracking for the Operon AI agent.

Tracks per-call token counts and USD costs across all supported model providers.
Provides compact status lines for inline display and a detailed session report
suitable for rendering inside a terminal box.

Usage
-----
    from operon.core.cost_tracker import CostTracker

    tracker = CostTracker()
    tracker.record("gpt-4o", "openai", input_tokens=512, output_tokens=128)
    print(tracker.status_line())          # ↑512 ↓128 tokens | $0.0015
    for line in tracker.session_report():
        print(line)
    tracker.reset()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Pricing table
# Keys are canonical model name prefixes (lowercase).
# Values are (input_cost_per_1k_tokens, output_cost_per_1k_tokens) in USD.
# ---------------------------------------------------------------------------

_PRICES: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini":    (0.00015, 0.0006),
    "gpt-4o":         (0.0025,  0.01),
    "gpt-4-turbo":    (0.01,    0.03),
    "gpt-4":          (0.03,    0.06),
    "gpt-3.5-turbo":  (0.0005,  0.0015),
    "o1-mini":        (0.003,   0.012),
    "o3-mini":        (0.0011,  0.0044),
    "o1":             (0.015,   0.06),
    # Anthropic
    "claude-opus-4-5":   (0.015,  0.075),
    "claude-sonnet-4-5": (0.003,  0.015),
    "claude-haiku-3-5":  (0.0008, 0.004),
    "claude-3-opus":     (0.015,  0.075),
    "claude-3-sonnet":   (0.003,  0.015),
    "claude-3-haiku":    (0.0008, 0.004),
    # Google Gemini
    "gemini-2.0-flash": (0.0001,   0.0004),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-1.5-pro":   (0.00125,  0.005),
    # Local / free models
    "llama3.2":   (0.0, 0.0),
    "llama3.1":   (0.0, 0.0),
    "mistral":    (0.0, 0.0),
    "mixtral":    (0.0, 0.0),
    "phi3":       (0.0, 0.0),
    "qwen2.5":    (0.0, 0.0),
    "deepseek":   (0.0, 0.0),
}


def _PRICES_LOOKUP(model_name: str) -> Tuple[float, float]:
    """Return (input_cost_per_1k, output_cost_per_1k) for *model_name*.

    Matching is case-insensitive prefix matching so that versioned names like
    ``"claude-sonnet-4-5-20251022"`` still resolve to ``"claude-sonnet-4-5"``.

    Longer keys are tried before shorter ones to prevent ``"o1"`` from
    shadowing ``"o1-mini"`` or ``"o3-mini"``.  Falls back to ``(0.0, 0.0)``
    if no prefix matches.
    """
    normalised = model_name.strip().lower()
    for key in sorted(_PRICES, key=len, reverse=True):
        if normalised.startswith(key):
            return _PRICES[key]
    return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Internal record type
# ---------------------------------------------------------------------------

@dataclass
class _Call:
    """Single LLM call record."""
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    timestamp: datetime
    cost_usd: float


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Accumulate token usage and USD cost across multiple LLM calls.

    Thread-safety is intentionally out of scope; wrap externally if needed.
    """

    def __init__(self) -> None:
        self._calls: List[_Call] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        """Record one LLM call and accumulate its cost."""
        in_rate, out_rate = _PRICES_LOOKUP(model)
        cost = (input_tokens / 1_000) * in_rate + (output_tokens / 1_000) * out_rate
        self._calls.append(
            _Call(
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                timestamp=datetime.now(tz=timezone.utc),
                cost_usd=cost,
            )
        )

    def reset(self) -> None:
        """Clear all recorded calls."""
        self._calls.clear()

    # ------------------------------------------------------------------
    # Aggregate properties
    # ------------------------------------------------------------------

    @property
    def total_input(self) -> int:
        """Total input tokens across all recorded calls."""
        return sum(c.input_tokens for c in self._calls)

    @property
    def total_output(self) -> int:
        """Total output tokens across all recorded calls."""
        return sum(c.output_tokens for c in self._calls)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across all recorded calls."""
        return self.total_input + self.total_output

    @property
    def total_cost(self) -> float:
        """Total USD cost across all recorded calls."""
        return sum(c.cost_usd for c in self._calls)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @property
    def total_cache_read(self) -> int:
        return sum(c.cache_read_tokens for c in self._calls)

    @property
    def total_cache_write(self) -> int:
        return sum(c.cache_write_tokens for c in self._calls)

    @property
    def last_input(self) -> int:
        """Input tokens for the most recent LLM call (0 if no calls)."""
        return self._calls[-1].input_tokens if self._calls else 0

    @property
    def last_output(self) -> int:
        """Output tokens for the most recent LLM call (0 if no calls)."""
        return self._calls[-1].output_tokens if self._calls else 0

    @property
    def call_count(self) -> int:
        """Number of LLM calls recorded this session."""
        return len(self._calls)

    def status_line(self) -> str:
        """Return a compact one-line summary showing per-turn AND session totals.

        Single-call format:  ``"↑513 ↓13 tokens | free"``
        Multi-call format:   ``"↑513 ↓13 this turn | ↑2,341 ↓891 session | $0.0234"``
        With cache:          ``"↑513 ↓13 this turn | ↑2,341 ↓891 session | cache ✓1,200r/300w | $0.0234"``
        """
        if not self._calls:
            return "—"
        cost = self.total_cost
        cost_str = "free" if cost == 0.0 else f"${cost:.4f}"
        cache_r = self.total_cache_read
        cache_w = self.total_cache_write
        cache_str = (f" | cache ✓{cache_r:,}r/{cache_w:,}w"
                     if cache_r or cache_w else "")

        if len(self._calls) == 1:
            # First turn: no need to show session total separately
            return (
                f"↑{self.last_input:,} ↓{self.last_output:,} tokens"
                f"{cache_str} | {cost_str}"
            )
        else:
            # Subsequent turns: show per-turn then session cumulative
            return (
                f"↑{self.last_input:,} ↓{self.last_output:,} this turn"
                f" | ↑{self.total_input:,} ↓{self.total_output:,} session"
                f"{cache_str} | {cost_str}"
            )

    def session_report(self) -> List[str]:
        """Return a list of display lines for a full session cost report.

        The lines are plain strings with no box-drawing characters so the
        caller can embed them in whatever framing it prefers.

        Structure
        ---------
        - Section header
        - Blank separator
        - Total tokens + cost
        - Breakdown per model (sorted by descending cost then name)
        """
        lines: List[str] = []
        lines.append("Session Cost Report")
        lines.append("")

        if not self._calls:
            lines.append("No calls recorded.")
            return lines

        # Overall totals
        cost = self.total_cost
        cost_str = "free" if cost == 0.0 else f"${cost:.6f}"
        lines.append(
            f"Total  ↑{self.total_input:,} ↓{self.total_output:,} "
            f"({self.total_tokens:,} tokens)  {cost_str}"
        )
        lines.append("")
        lines.append("Breakdown by model:")

        # Aggregate per model
        model_stats: Dict[str, dict] = {}
        for c in self._calls:
            key = f"{c.provider}/{c.model}"
            if key not in model_stats:
                model_stats[key] = {
                    "input": 0,
                    "output": 0,
                    "cost": 0.0,
                    "calls": 0,
                }
            model_stats[key]["input"] += c.input_tokens
            model_stats[key]["output"] += c.output_tokens
            model_stats[key]["cost"] += c.cost_usd
            model_stats[key]["calls"] += 1

        # Sort: descending cost, then alphabetical key
        for key, stats in sorted(
            model_stats.items(),
            key=lambda kv: (-kv[1]["cost"], kv[0]),
        ):
            m_cost = stats["cost"]
            m_cost_str = "free" if m_cost == 0.0 else f"${m_cost:.6f}"
            call_word = "call" if stats["calls"] == 1 else "calls"
            lines.append(
                f"  {key}  "
                f"↑{stats['input']:,} ↓{stats['output']:,}  "
                f"{m_cost_str}  "
                f"({stats['calls']} {call_word})"
            )

        return lines
