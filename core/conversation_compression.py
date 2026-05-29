"""
core/conversation_compression.py — Dedicated Conversation Compression

Separate from context_compressor.py (which handles single-turn context windows),
this module handles multi-turn conversation history compression:

  - Iterative merge: fold prior summary + new turns → updated summary
  - Sliding window: keep N most-recent turns, summarise the rest
  - Turn pruning: strip bulky tool outputs before summarisation
  - Quality scoring: verify key facts survive compression
  - Budget-aware: stop compressing when token count fits the target

Architecture mirrors Hermes' conversation_compression.py design:
  ConversationSummarizer  — LLM-powered turn-group → summary
  IterativeMerger         — prior_summary + new_turns → merged_summary
  TurnPruner              — reduce token cost before LLM call
  CompressionQualityScorer — check information retention
  ConversationCompressor  — orchestrator with budget awareness
  RollingWindow           — always-on sliding window compression

Usage:
    from core.conversation_compression import ConversationCompressor

    compressor = ConversationCompressor()
    compressed = compressor.compress(
        conversation_history,    # list of {role, content} dicts
        target_tokens=8000,
        keep_last_n=6,
    )
    # compressed.messages — trimmed history
    # compressed.summary  — prose summary of what was cut
    # compressed.stats    — tokens_before, tokens_after, turns_removed
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("operon.conversation_compression")

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_TARGET_TOKENS   = 12_000
_DEFAULT_KEEP_LAST_N     = 8
_DEFAULT_WINDOW_SIZE     = 6
_MAX_TOOL_OUTPUT_CHARS   = 2_000
_CHARS_PER_TOKEN         = 4            # rough approximation
_QUALITY_PASS_THRESHOLD  = 0.60         # 60% of key entities must survive
_MIN_TURNS_TO_COMPRESS   = 4            # don't compress if fewer turns than this

_SUMMARISE_SYSTEM = """You are a conversation archivist. Summarise the provided
conversation turns into a compact, factual paragraph. Preserve:
  - All decisions made
  - All files/functions/errors mentioned
  - All user goals stated
  - All tool results that produced meaningful output
  - Any unresolved questions or pending tasks

Be specific. Include concrete names, file paths, and numbers.
Respond with plain prose (no bullet points, no headers). 1-3 paragraphs max."""

_MERGE_SYSTEM = """You are updating a running conversation summary.
Given an existing summary and new conversation turns, produce an updated summary
that incorporates the new information. If the new turns contradict earlier facts,
prefer the newer information. Preserve all unresolved items from the existing
summary unless the new turns resolve them. Plain prose, 2-4 paragraphs max."""

_QUALITY_SYSTEM = """You are a fact checker. Given a list of key facts and a
summary, return a JSON object:
{
  "retained": ["fact1", ...],    // facts that appear in the summary
  "missing":  ["fact2", ...],    // facts not mentioned in the summary
  "score": 0.85                  // fraction retained (0.0 – 1.0)
}
Respond ONLY with valid JSON."""


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """A single conversation turn."""
    role:    str                    # "user" | "assistant" | "tool"
    content: Any                    # str or list of content blocks
    index:   int = 0
    tokens:  int = 0                # estimated token count

    @property
    def text(self) -> str:
        """Flatten content to plain text."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            parts: List[str] = []
            for block in self.content:
                if isinstance(block, dict):
                    t = block.get("text") or block.get("output") or ""
                    if t:
                        parts.append(str(t))
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts)
        return str(self.content)

    def estimate_tokens(self) -> int:
        return max(1, len(self.text) // _CHARS_PER_TOKEN)

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class CompressionStats:
    tokens_before:  int = 0
    tokens_after:   int = 0
    turns_before:   int = 0
    turns_after:    int = 0
    turns_removed:  int = 0
    summary_tokens: int = 0
    quality_score:  float = 1.0
    duration_ms:    float = 0.0
    method:         str = "none"

    @property
    def compression_ratio(self) -> float:
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens_before":    self.tokens_before,
            "tokens_after":     self.tokens_after,
            "turns_before":     self.turns_before,
            "turns_after":      self.turns_after,
            "turns_removed":    self.turns_removed,
            "compression_ratio": round(self.compression_ratio, 3),
            "quality_score":    round(self.quality_score, 3),
            "duration_ms":      round(self.duration_ms, 1),
            "method":           self.method,
        }


@dataclass
class CompressedConversation:
    """Result of a compression pass."""
    messages:       List[Dict[str, Any]]    # compressed message list
    summary:        str                     # prose summary of removed turns
    prior_summary:  str                     # summary carried from previous pass
    stats:          CompressionStats = field(default_factory=CompressionStats)
    ok:             bool = True
    error:          str  = ""

    def total_tokens(self) -> int:
        total = 0
        for msg in self.messages:
            c = msg.get("content", "")
            text = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
            total += max(1, len(text) // _CHARS_PER_TOKEN)
        return total


# ── Turn pruner ──────────────────────────────────────────────────────────────

class TurnPruner:
    """
    Reduce token cost of turns before sending to the LLM for summarisation.

    Rules:
      - Truncate tool/assistant content blocks > _MAX_TOOL_OUTPUT_CHARS
      - Strip binary / base64 image data entirely
      - Collapse repeated assistant "Acknowledged" / "Done" turns
    """

    def __init__(self, max_tool_chars: int = _MAX_TOOL_OUTPUT_CHARS) -> None:
        self._max_tool_chars = max_tool_chars

    def prune(self, turns: List[Turn]) -> List[Turn]:
        """Return a pruned copy of the turns list."""
        pruned: List[Turn] = []
        prev_filler = False
        for t in turns:
            if self._is_filler(t):
                if prev_filler:
                    continue   # collapse consecutive filler turns
                prev_filler = True
            else:
                prev_filler = False
            pruned.append(Turn(
                role=t.role,
                content=self._prune_content(t.content),
                index=t.index,
            ))
        return pruned

    def _prune_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return self._truncate(content)
        if isinstance(content, list):
            result = []
            for block in content:
                if isinstance(block, dict):
                    # Drop image blocks
                    if block.get("type") == "image":
                        result.append({"type": "text",
                                       "text": "[image omitted in compression]"})
                        continue
                    # Truncate text blocks
                    if "text" in block:
                        result.append({**block,
                                        "text": self._truncate(block["text"])})
                        continue
                    # Truncate tool output
                    if "output" in block:
                        result.append({**block,
                                        "output": self._truncate(block["output"])})
                        continue
                result.append(block)
            return result
        return content

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_tool_chars:
            return text
        half = self._max_tool_chars // 2
        return (text[:half]
                + f"\n…[{len(text) - self._max_tool_chars} chars truncated]…\n"
                + text[-half:])

    @staticmethod
    def _is_filler(turn: Turn) -> bool:
        """Detect low-information turns (short acknowledgements)."""
        text = turn.text.strip().lower()
        if len(text) > 120:
            return False
        filler_patterns = (
            r"^(ok|okay|got it|understood|sure|done|alright|acknowledged)[\.\!]?$",
            r"^i('ll| will) (do|take care of|handle) that[\.\!]?$",
        )
        for p in filler_patterns:
            if re.match(p, text):
                return True
        return False


# ── Conversation summariser ──────────────────────────────────────────────────

class ConversationSummarizer:
    """
    LLM-powered: summarise a list of turns into a prose paragraph.
    Falls back to an extractive summary if LLM is unavailable.
    """

    def __init__(self, llm_fn: Optional[Callable[[str, str], str]] = None) -> None:
        self._llm = llm_fn or self._default_llm

    def summarise(self, turns: List[Turn], context: str = "") -> str:
        """Return a prose summary of the given turns."""
        if not turns:
            return ""
        # Build the transcript text
        lines: List[str] = []
        if context:
            lines.append(f"[Context: {context}]\n")
        for t in turns:
            prefix = {"user": "User", "assistant": "Agent",
                      "tool": "Tool"}.get(t.role, t.role.capitalize())
            lines.append(f"{prefix}: {t.text[:800]}")  # cap per turn
        transcript = "\n".join(lines)
        try:
            result = self._llm(_SUMMARISE_SYSTEM, transcript)
            return result.strip() if result else self._extractive(turns)
        except Exception as e:
            log.warning("ConversationSummarizer LLM call failed: %s", e)
            return self._extractive(turns)

    @staticmethod
    def _extractive(turns: List[Turn]) -> str:
        """Fallback: first sentence of each turn, deduped."""
        seen: set = set()
        parts: List[str] = []
        for t in turns:
            sentence = t.text.split(".")[0].strip()[:120]
            if sentence and sentence not in seen:
                seen.add(sentence)
                parts.append(f"[{t.role}] {sentence}")
        return " | ".join(parts[:10])

    def _default_llm(self, system: str, user: str) -> str:
        """Use Operon's model router if available."""
        try:
            from core.router import ModelRouter
            from core.config import ConfigManager
            cfg    = ConfigManager()
            router = ModelRouter(cfg)
            return router.complete(
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=512,
            ) or ""
        except Exception as e:
            log.debug("ConversationSummarizer default LLM unavailable: %s", e)
            return ""


# ── Iterative merger ─────────────────────────────────────────────────────────

class IterativeMerger:
    """
    Merge a prior summary + new turns into an updated summary.
    This is the key mechanism that prevents summary stacking (where each
    compression produces a longer and longer summary).
    """

    def __init__(self, llm_fn: Optional[Callable[[str, str], str]] = None) -> None:
        self._llm = llm_fn or self._default_llm
        self._summariser = ConversationSummarizer(llm_fn=llm_fn)

    def merge(self, prior_summary: str, new_turns: List[Turn]) -> str:
        """
        Fold prior_summary + new_turns into a single updated summary.
        If no prior summary, delegate to ConversationSummarizer.
        """
        if not prior_summary:
            return self._summariser.summarise(new_turns)
        if not new_turns:
            return prior_summary

        transcript_lines = [
            f"[Prior summary]: {prior_summary}\n",
            "[New conversation turns]:",
        ]
        for t in new_turns:
            prefix = {"user": "User", "assistant": "Agent",
                      "tool": "Tool"}.get(t.role, t.role.capitalize())
            transcript_lines.append(f"{prefix}: {t.text[:600]}")

        user_msg = "\n".join(transcript_lines)
        try:
            result = self._llm(_MERGE_SYSTEM, user_msg)
            return result.strip() if result else self._fallback_merge(prior_summary, new_turns)
        except Exception as e:
            log.warning("IterativeMerger LLM call failed: %s", e)
            return self._fallback_merge(prior_summary, new_turns)

    @staticmethod
    def _fallback_merge(prior: str, new_turns: List[Turn]) -> str:
        new_text = " | ".join(
            f"[{t.role}] {t.text[:80]}" for t in new_turns[:5]
        )
        return f"{prior} [New context: {new_text}]"

    def _default_llm(self, system: str, user: str) -> str:
        try:
            from core.router import ModelRouter
            from core.config import ConfigManager
            cfg    = ConfigManager()
            router = ModelRouter(cfg)
            return router.complete(
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=768,
            ) or ""
        except Exception:
            return ""


# ── Quality scorer ────────────────────────────────────────────────────────────

class CompressionQualityScorer:
    """
    After compression, verify that key facts from the original conversation
    are preserved in the resulting summary.

    Key fact extraction: simple regex-based NER (file paths, function names,
    error messages, numbers, quoted strings).
    """

    _FACT_PATTERNS = [
        re.compile(r"[`'\"]?([\w/.\-]+\.(?:py|js|ts|json|yml|md|txt|csv|sh))[`'\"]?"),  # files
        re.compile(r"`([\w_]{3,})\(`"),         # function calls
        re.compile(r"\b([A-Z][a-z]+Error|Exception)\b"),  # exceptions
        re.compile(r"\b(\d{3,})\b"),             # numbers ≥ 3 digits
        re.compile(r'"([^"]{4,40})"'),            # quoted strings
    ]

    def __init__(self, llm_fn: Optional[Callable[[str, str], str]] = None) -> None:
        self._llm = llm_fn or self._default_llm

    def extract_key_facts(self, turns: List[Turn]) -> List[str]:
        """Extract concrete facts (file names, functions, errors) from turns."""
        facts: List[str] = []
        seen: set = set()
        for t in turns:
            for pattern in self._FACT_PATTERNS:
                for m in pattern.finditer(t.text):
                    fact = m.group(1) if m.lastindex else m.group(0)
                    fact = fact.strip()
                    if fact and fact not in seen:
                        seen.add(fact)
                        facts.append(fact)
        return facts[:30]   # cap to avoid huge prompts

    def score(self, turns: List[Turn], summary: str) -> float:
        """
        Return fraction (0.0–1.0) of key facts that appear in summary.
        Falls back to string-match if LLM unavailable.
        """
        facts = self.extract_key_facts(turns)
        if not facts:
            return 1.0   # nothing to check

        # Try LLM-based scoring
        llm_score = self._llm_score(facts, summary)
        if llm_score is not None:
            return llm_score

        # Fallback: case-insensitive string match
        summary_lower = summary.lower()
        retained = sum(1 for f in facts if f.lower() in summary_lower)
        return retained / max(len(facts), 1)

    def _llm_score(self, facts: List[str], summary: str) -> Optional[float]:
        try:
            user_msg = (
                f"Key facts:\n{json.dumps(facts)}\n\n"
                f"Summary:\n{summary}"
            )
            raw = self._llm(_QUALITY_SYSTEM, user_msg)
            if not raw:
                return None
            clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
            data  = json.loads(clean)
            return float(data.get("score", 0.5))
        except Exception:
            return None

    def _default_llm(self, system: str, user: str) -> str:
        try:
            from core.router import ModelRouter
            from core.config import ConfigManager
            cfg    = ConfigManager()
            router = ModelRouter(cfg)
            return router.complete(
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=256,
            ) or ""
        except Exception:
            return ""


# ── Rolling window ────────────────────────────────────────────────────────────

class RollingWindow:
    """
    Always-on sliding window compression.
    Maintains a buffer of the last N turns.
    When the buffer exceeds max_turns, the oldest group is summarised and dropped.
    """

    def __init__(
        self,
        max_turns:   int = 20,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        summariser:  Optional[ConversationSummarizer] = None,
    ) -> None:
        self._max_turns    = max_turns
        self._window_size  = window_size
        self._summariser   = summariser or ConversationSummarizer()
        self._summary:      str = ""
        self._buffer:       List[Dict[str, Any]] = []

    def push(self, message: Dict[str, Any]) -> None:
        """Add a new message to the rolling buffer."""
        self._buffer.append(message)
        if len(self._buffer) > self._max_turns:
            self._flush_oldest()

    def flush_oldest(self, n: int = 0) -> None:
        """Summarise and remove the oldest n turns (default: window_size)."""
        self._flush_oldest(n or self._window_size)

    def _flush_oldest(self, n: Optional[int] = None) -> None:
        count = n or self._window_size
        to_summarise = self._buffer[:count]
        self._buffer  = self._buffer[count:]
        turns = [Turn(role=m["role"], content=m["content"], index=i)
                 for i, m in enumerate(to_summarise)]
        new_summary = self._summariser.summarise(turns)
        if self._summary:
            self._summary = f"{self._summary}\n\n{new_summary}"
        else:
            self._summary = new_summary

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def buffer(self) -> List[Dict[str, Any]]:
        return list(self._buffer)

    def get_context(self) -> List[Dict[str, Any]]:
        """
        Return the full context: summary as a system injection + live buffer.
        """
        result: List[Dict[str, Any]] = list(self._buffer)
        if self._summary:
            result.insert(0, {
                "role":    "user",
                "content": f"[Conversation history summary]\n{self._summary}",
            })
        return result

    def reset(self) -> None:
        self._summary = ""
        self._buffer  = []


# ── Main orchestrator ─────────────────────────────────────────────────────────

class ConversationCompressor:
    """
    High-level orchestrator for conversation compression.

    Strategy selection:
      1. If history fits in target_tokens → no-op
      2. If history is large → sliding window: keep last keep_last_n turns,
         summarise the rest using IterativeMerger
      3. Validate quality_score ≥ threshold; if not, expand keep_last_n and retry

    The summary is injected as a synthetic "user" turn at the top of the
    returned message list, clearly marked as a history summary.
    """

    _SUMMARY_PREFIX = "[Conversation History Summary]\n"

    def __init__(
        self,
        target_tokens:     int   = _DEFAULT_TARGET_TOKENS,
        keep_last_n:       int   = _DEFAULT_KEEP_LAST_N,
        quality_threshold: float = _QUALITY_PASS_THRESHOLD,
        llm_fn:            Optional[Callable[[str, str], str]] = None,
        enable_quality_check: bool = True,
    ) -> None:
        self._target        = target_tokens
        self._keep_last     = keep_last_n
        self._quality_thr   = quality_threshold
        self._enable_quality_check = enable_quality_check

        self._pruner     = TurnPruner()
        self._summariser = ConversationSummarizer(llm_fn=llm_fn)
        self._merger     = IterativeMerger(llm_fn=llm_fn)
        self._scorer     = CompressionQualityScorer(llm_fn=llm_fn)

        # running state (across calls)
        self._prior_summary: str = ""

    def compress(
        self,
        messages:    List[Dict[str, Any]],
        target_tokens: Optional[int] = None,
        keep_last_n:   Optional[int] = None,
        prior_summary: Optional[str] = None,
    ) -> CompressedConversation:
        """
        Compress the conversation history.

        Args:
            messages:      Full conversation history (list of {role, content})
            target_tokens: Token budget for compressed result
            keep_last_n:   Always keep this many recent turns verbatim
            prior_summary: Existing summary to merge into (overrides internal state)

        Returns:
            CompressedConversation with .messages, .summary, .stats
        """
        t0 = time.monotonic()
        target   = target_tokens or self._target
        keep_n   = keep_last_n   or self._keep_last
        prior    = prior_summary if prior_summary is not None else self._prior_summary

        # Wrap messages as Turn objects
        turns = [
            Turn(role=m.get("role", "user"),
                 content=m.get("content", ""),
                 index=i)
            for i, m in enumerate(messages)
        ]
        total_tokens_before = sum(t.estimate_tokens() for t in turns)

        stats = CompressionStats(
            tokens_before = total_tokens_before,
            turns_before  = len(turns),
        )

        # Already fits — no compression needed
        if total_tokens_before <= target:
            stats.tokens_after = total_tokens_before
            stats.turns_after  = len(turns)
            stats.method       = "none"
            stats.duration_ms  = (time.monotonic() - t0) * 1000
            return CompressedConversation(
                messages=[m for m in messages],
                summary=prior,
                prior_summary=prior,
                stats=stats,
            )

        # Too few turns to compress
        if len(turns) < _MIN_TURNS_TO_COMPRESS:
            stats.tokens_after = total_tokens_before
            stats.turns_after  = len(turns)
            stats.method       = "skip_too_few"
            stats.duration_ms  = (time.monotonic() - t0) * 1000
            return CompressedConversation(
                messages=[m for m in messages],
                summary=prior,
                prior_summary=prior,
                stats=stats,
            )

        # Sliding window: split into "old" and "recent"
        recent_turns = turns[-keep_n:]
        old_turns    = turns[:-keep_n]

        if not old_turns:
            # Nothing to compress — keep_n covers everything
            stats.tokens_after = total_tokens_before
            stats.turns_after  = len(turns)
            stats.method       = "skip_keep_n_covers_all"
            stats.duration_ms  = (time.monotonic() - t0) * 1000
            return CompressedConversation(
                messages=messages,
                summary=prior,
                prior_summary=prior,
                stats=stats,
            )

        # Prune old turns before summarisation (cheaper LLM call)
        pruned_old = self._pruner.prune(old_turns)

        # Iterative merge: prior_summary + old_turns → new summary
        new_summary = self._merger.merge(prior, pruned_old)

        # Quality check
        quality_score = 1.0
        if self._enable_quality_check and new_summary:
            quality_score = self._scorer.score(old_turns, new_summary)
            if quality_score < self._quality_thr:
                log.warning(
                    "Compression quality %.2f < threshold %.2f; expanding keep window",
                    quality_score, self._quality_thr
                )
                # Retry with bigger keep window
                expanded_keep = min(keep_n + 4, len(turns) - 2)
                if expanded_keep > keep_n:
                    return self.compress(
                        messages=messages,
                        target_tokens=target,
                        keep_last_n=expanded_keep,
                        prior_summary=prior,
                    )

        # Build compressed message list
        summary_message: Dict[str, Any] = {
            "role":    "user",
            "content": self._SUMMARY_PREFIX + new_summary,
        }
        compressed_messages: List[Dict[str, Any]] = (
            [summary_message] + [t.to_dict() for t in recent_turns]
        )
        tokens_after = sum(
            Turn(role=m["role"], content=m["content"]).estimate_tokens()
            for m in compressed_messages
        )

        # Update internal running summary state
        self._prior_summary = new_summary

        stats.tokens_after  = tokens_after
        stats.turns_after   = len(compressed_messages)
        stats.turns_removed = len(old_turns)
        stats.summary_tokens = len(new_summary) // _CHARS_PER_TOKEN
        stats.quality_score  = quality_score
        stats.method         = "sliding_window_merge"
        stats.duration_ms    = (time.monotonic() - t0) * 1000

        return CompressedConversation(
            messages      = compressed_messages,
            summary       = new_summary,
            prior_summary = prior,
            stats         = stats,
        )

    def compress_incremental(
        self,
        new_messages: List[Dict[str, Any]],
        existing_summary: str = "",
    ) -> CompressedConversation:
        """
        Incremental compression: summarise only new_messages and merge into
        existing_summary. Useful for streaming/online compression.
        """
        turns = [Turn(role=m["role"], content=m["content"], index=i)
                 for i, m in enumerate(new_messages)]
        pruned = self._pruner.prune(turns)
        merged = self._merger.merge(existing_summary, pruned)
        self._prior_summary = merged

        stats = CompressionStats(
            tokens_before  = sum(t.estimate_tokens() for t in turns),
            tokens_after   = len(merged) // _CHARS_PER_TOKEN,
            turns_before   = len(turns),
            turns_after    = 1,
            turns_removed  = len(turns),
            method         = "incremental_merge",
        )
        return CompressedConversation(
            messages      = [{"role": "user",
                               "content": self._SUMMARY_PREFIX + merged}],
            summary       = merged,
            prior_summary = existing_summary,
            stats         = stats,
        )

    def reset(self) -> None:
        """Clear running summary state (start fresh)."""
        self._prior_summary = ""

    @property
    def current_summary(self) -> str:
        return self._prior_summary

    @staticmethod
    def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
        """Quick token estimate for a message list."""
        total = 0
        for m in messages:
            c = m.get("content", "")
            text = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
            total += max(1, len(text) // _CHARS_PER_TOKEN)
        return total


# ── Convenience functions ─────────────────────────────────────────────────────

def compress_conversation(
    messages:      List[Dict[str, Any]],
    target_tokens: int = _DEFAULT_TARGET_TOKENS,
    keep_last_n:   int = _DEFAULT_KEEP_LAST_N,
    prior_summary: str = "",
) -> CompressedConversation:
    """One-shot convenience wrapper."""
    compressor = ConversationCompressor(
        target_tokens=target_tokens,
        keep_last_n=keep_last_n,
    )
    return compressor.compress(
        messages,
        prior_summary=prior_summary,
    )


def rolling_window_compress(
    messages:   List[Dict[str, Any]],
    max_turns:  int = 20,
    window_size: int = _DEFAULT_WINDOW_SIZE,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Rolling window compression: summarise oldest turns, return (summary, live_buffer).
    """
    window = RollingWindow(max_turns=max_turns, window_size=window_size)
    for msg in messages:
        window.push(msg)
    return window.summary, window.buffer


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    return ConversationCompressor.estimate_tokens(messages)


def extract_key_facts(messages: List[Dict[str, Any]]) -> List[str]:
    turns = [Turn(role=m["role"], content=m.get("content", ""), index=i)
             for i, m in enumerate(messages)]
    return CompressionQualityScorer().extract_key_facts(turns)
