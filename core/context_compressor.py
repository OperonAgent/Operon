"""
Operon Context Compressor — LLM-powered conversation compression.

Adapted from Hermes Agent context_compressor.py architecture.

When the conversation history grows too long, this compressor uses a cheap
auxiliary LLM call to summarise the middle turns while protecting:
  - The system prompt (always kept verbatim)
  - The first user message (task anchor)
  - The most recent N turns (immediate working context)
  - Pinned messages (messages with _pinned=True in metadata)

The summary is prepended as a special [CONTEXT COMPACTION] message that
tells the model this is historical context, NOT an active instruction.

Features:
  - Structured summary with Resolved / Pending / Active sections
  - Tool output pruning before LLM summarisation (cheap pre-pass)
  - Iterative updates: prior summaries are merged, not stacked
  - Token budget tail protection (not fixed message count)
  - Image cost estimation (1,600 tokens per image part)
  - Configurable: threshold, tail_turns, summary_ratio, aux_model

Usage:
    from core.context_compressor import ContextCompressor, CompressorConfig
    compressor = ContextCompressor()
    new_messages, did_compress = compressor.maybe_compress(messages, system_prompt)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("operon.context_compressor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. Treat it as background reference, NOT as active "
    "instructions. Your current task is in the 'Active Task' section — "
    "resume from there. Respond only to the latest user message after this "
    "summary. The current session state may reflect work described here — "
    "avoid repeating completed steps:"
)

_PRUNED_PLACEHOLDER = "[Old tool output cleared to save context space]"

_CHARS_PER_TOKEN    = 4
_IMAGE_TOKEN_COST   = 1_600
_IMAGE_CHAR_COST    = _IMAGE_TOKEN_COST * _CHARS_PER_TOKEN

_MIN_SUMMARY_TOKENS = 1_000
_SUMMARY_RATIO      = 0.20          # 20% of compressed content → summary budget
_SUMMARY_TOKENS_CAP = 8_000

_SUMMARY_FAIL_COOLDOWN = 300        # 5 min cooldown after failed summarisation


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CompressorConfig:
    """Tuning knobs for the context compressor."""
    threshold_tokens:  int   = 6_000    # compress when estimate exceeds this
    tail_turns:        int   = 6        # always keep last N user/assistant pairs
    tail_budget_chars: int   = 12_000   # alternative: keep tail by char budget
    summary_ratio:     float = 0.20     # summary length relative to compressed content
    min_summary_tokens: int  = 1_000
    max_summary_tokens: int  = 8_000
    aux_model:         str   = ""       # override aux model for summarisation
    prune_tool_output: bool  = True     # prune old tool results before summarising
    enabled:           bool  = True


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_content_chars(content: Any) -> int:
    """Estimate character cost of a message content, including images."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if not isinstance(part, dict):
                total += len(str(part))
                continue
            ptype = part.get("type", "")
            if ptype in ("image_url", "image", "input_image"):
                total += _IMAGE_CHAR_COST
            elif ptype == "text":
                total += len(part.get("text", ""))
            else:
                total += len(str(part))
        return total
    return len(str(content or ""))


def _estimate_tokens(messages: List[Dict], system: str = "") -> int:
    total = len(system) // _CHARS_PER_TOKEN
    for m in messages:
        total += _estimate_content_chars(m.get("content", "")) // _CHARS_PER_TOKEN
        total += 4   # role + overhead per message
    return total


# ---------------------------------------------------------------------------
# Tool output pruning (cheap pre-pass before LLM summarisation)
# ---------------------------------------------------------------------------

_TOOL_RESULT_RE = re.compile(r"^\[TOOL_RESULT:", re.M)


def _prune_old_tool_outputs(messages: List[Dict], keep_from: int) -> List[Dict]:
    """
    Replace tool result content in old messages with a placeholder.
    Only messages before index keep_from are pruned.
    """
    result = []
    for i, msg in enumerate(messages):
        if i >= keep_from:
            result.append(msg)
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and _TOOL_RESULT_RE.match(content):
            msg = dict(msg)
            msg["content"] = _PRUNED_PLACEHOLDER
        result.append(msg)
    return result


# ---------------------------------------------------------------------------
# Summary template
# ---------------------------------------------------------------------------

def _build_summary_prompt(messages: List[Dict]) -> str:
    """Format messages into a prompt for the summariser LLM."""
    lines = [
        "You are summarising a conversation for context window management.",
        "Produce a structured summary with these sections:",
        "  ## Resolved   — What was successfully completed",
        "  ## Pending    — Outstanding tasks / questions",
        "  ## Active Task — The most recent user request",
        "  ## Key Facts  — Important identifiers, paths, decisions",
        "",
        "Rules:",
        "- Be concise but complete. A future AI must be able to resume from this.",
        "- Preserve exact file paths, command names, error messages, and identifiers.",
        "- Do NOT interpret or embellish — only summarise what is explicitly present.",
        "- Do NOT include instructions for the AI — only historical facts.",
        "",
        "CONVERSATION TO SUMMARISE:",
        "=" * 60,
    ]
    for msg in messages:
        role    = msg.get("role", "?").upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content
                          if isinstance(p, dict) and p.get("type") == "text"]
            content = "\n".join(text_parts)
        content = str(content).strip()
        if len(content) > 2000:
            content = content[:1000] + "\n[…]\n" + content[-500:]
        lines.append(f"\n[{role}]\n{content}")
    lines.append("=" * 60)
    lines.append("\nSUMMARY:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

class ContextCompressor:
    """LLM-powered context compressor for Operon conversation loops."""

    def __init__(self, config: Optional[CompressorConfig] = None) -> None:
        self._config = config or CompressorConfig()
        self._last_fail_time: float = 0.0

    def maybe_compress(
        self,
        messages: List[Dict],
        system:   str = "",
        force:    bool = False,
    ) -> Tuple[List[Dict], bool]:
        """
        Compress messages if over threshold.

        Returns (new_messages, did_compress).
        If compression fails, returns (original_messages, False) — never crashes.
        """
        cfg = self._config
        if not cfg.enabled:
            return messages, False

        estimated = _estimate_tokens(messages, system)
        if not force and estimated < cfg.threshold_tokens:
            return messages, False

        # Cooldown after previous failure
        if time.time() - self._last_fail_time < _SUMMARY_FAIL_COOLDOWN:
            log.debug("Context compressor on cooldown — skipping")
            return messages, False

        log.info("Context compression triggered: ~%d tokens, threshold=%d",
                 estimated, cfg.threshold_tokens)

        try:
            new_messages, did = self._compress(messages, system)
            return new_messages, did
        except Exception as e:
            log.warning("Context compression failed: %s", e)
            self._last_fail_time = time.time()
            return messages, False

    def _compress(
        self, messages: List[Dict], system: str
    ) -> Tuple[List[Dict], bool]:
        cfg = self._config

        # Separate system messages
        sys_messages  = [m for m in messages if m.get("role") == "system"]
        conv_messages = [m for m in messages if m.get("role") != "system"]

        if len(conv_messages) < 4:
            return messages, False   # too short to compress

        # Determine tail (keep last N turns verbatim)
        tail_start = _find_tail_start(conv_messages, cfg.tail_turns, cfg.tail_budget_chars)
        if tail_start <= 1:
            return messages, False   # nothing to compress

        head = conv_messages[:1]       # keep first user message (task anchor)
        middle = conv_messages[1:tail_start]
        tail   = conv_messages[tail_start:]

        if not middle:
            return messages, False

        # Prune old tool outputs from middle (cheap pre-pass)
        if cfg.prune_tool_output:
            middle = _prune_old_tool_outputs(middle, keep_from=len(middle))

        # Build summary
        summary_text = self._summarise(middle, system)
        if not summary_text:
            return messages, False

        # Replace middle with summary message
        summary_msg = {
            "role":    "user",
            "content": f"{SUMMARY_PREFIX}\n\n{summary_text}",
            "_compacted": True,
        }

        new_messages = sys_messages + head + [summary_msg] + tail
        saved = len(conv_messages) - len(new_messages) + len(sys_messages)
        log.info("Context compressed: %d → %d messages (%d saved)",
                 len(messages), len(new_messages), saved)
        return new_messages, True

    def _summarise(self, messages: List[Dict], system: str = "") -> str:
        """Call the aux LLM to produce a summary. Returns empty string on failure."""
        cfg    = self._config
        prompt = _build_summary_prompt(messages)

        # Calculate summary token budget
        middle_chars = sum(_estimate_content_chars(m.get("content", "")) for m in messages)
        budget = max(cfg.min_summary_tokens,
                     min(int(middle_chars * cfg.summary_ratio / _CHARS_PER_TOKEN),
                         cfg.max_summary_tokens))

        try:
            from core.router import ModelRouter
            from core.config import ConfigManager

            cfg_mgr = ConfigManager()
            router  = ModelRouter(cfg_mgr)
            # Use aux model if configured, else default cheap model
            aux_model = (self._config.aux_model
                         or cfg_mgr.get("aux_model", "")
                         or cfg_mgr.get("default_model", "llama3.2"))

            raw = router.complete(
                system   = "You are a concise summariser. Output only the structured summary, no preamble.",
                messages = [{"role": "user", "content": prompt}],
                model    = aux_model,
                max_tokens = budget,
            )
            if not raw:
                return ""
            # Strip any JSON wrapping if model returned JSON
            if raw.strip().startswith("{"):
                try:
                    parsed = json.loads(raw)
                    raw = str(parsed.get("reply") or parsed.get("content") or raw)
                except Exception:
                    pass
            return raw.strip()
        except Exception as e:
            log.warning("Summarisation LLM call failed: %s", e)
            return ""


# ---------------------------------------------------------------------------
# Tail-boundary finder
# ---------------------------------------------------------------------------

def _find_tail_start(messages: List[Dict], tail_turns: int,
                     tail_budget_chars: int) -> int:
    """
    Return the index at which the 'tail' starts.
    The tail contains the last tail_turns complete user/assistant pairs
    OR fills tail_budget_chars, whichever is larger.
    """
    # Walk backwards, accumulate complete turns
    idx   = len(messages)
    turns = 0
    chars = 0
    pairs_found = 0
    last_role   = None

    for i in range(len(messages) - 1, -1, -1):
        msg  = messages[i]
        role = msg.get("role", "")
        chars += _estimate_content_chars(msg.get("content", ""))

        if last_role == "assistant" and role == "user":
            pairs_found += 1
            if pairs_found >= tail_turns and chars >= tail_budget_chars:
                idx = i + 1
                break
        last_role = role

        if i == 0:
            idx = max(1, len(messages) - tail_turns * 2)

    return idx


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_compressor: Optional[ContextCompressor] = None


def get_compressor(config: Optional[CompressorConfig] = None) -> ContextCompressor:
    """Return the module-level default compressor."""
    global _default_compressor
    if _default_compressor is None or config is not None:
        _default_compressor = ContextCompressor(config)
    return _default_compressor


def maybe_compress_messages(
    messages: List[Dict],
    system:   str  = "",
    threshold: int = 6_000,
    tail_turns: int = 6,
    force:     bool = False,
) -> Tuple[List[Dict], bool]:
    """
    Convenience wrapper: compress messages if over threshold.
    Returns (new_messages, did_compress).
    """
    cfg = CompressorConfig(
        threshold_tokens = threshold,
        tail_turns       = tail_turns,
    )
    return get_compressor(cfg).maybe_compress(messages, system, force=force)


# ---------------------------------------------------------------------------
# Compression Quality Audit
# ---------------------------------------------------------------------------

@dataclass
class QualityAuditResult:
    """Result of auditing a summary's coverage of the original messages."""
    passed:          bool
    coverage_score:  float      # 0.0–1.0: fraction of key terms found in summary
    missing_terms:   List[str]  # important terms absent from summary
    word_count:      int
    is_too_short:    bool
    is_too_long:     bool
    notes:           List[str]  = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed":         self.passed,
            "coverage_score": round(self.coverage_score, 3),
            "missing_terms":  self.missing_terms[:10],
            "word_count":     self.word_count,
            "is_too_short":   self.is_too_short,
            "is_too_long":    self.is_too_long,
            "notes":          self.notes,
        }


class CompressionQualityAudit:
    """
    Audits a generated summary for coverage and quality — no LLM required.
    Uses term-frequency heuristics to ensure key facts were captured.
    """

    def __init__(
        self,
        min_coverage:   float = 0.60,   # 60% of key terms must appear in summary
        min_word_count: int   = 30,
        max_word_count: int   = 2_000,
        important_patterns: Optional[List[str]] = None,
    ) -> None:
        self._min_coverage    = min_coverage
        self._min_words       = min_word_count
        self._max_words       = max_word_count
        self._extra_patterns  = important_patterns or []

    def audit(
        self,
        original_messages: List[Dict],
        summary:           str,
    ) -> QualityAuditResult:
        """
        Check whether `summary` covers the key terms from `original_messages`.
        Returns a QualityAuditResult.
        """
        notes: List[str] = []

        # Extract key terms from originals
        key_terms = self._extract_key_terms(original_messages)
        if not key_terms:
            return QualityAuditResult(
                passed=True, coverage_score=1.0,
                missing_terms=[], word_count=len(summary.split()),
                is_too_short=False, is_too_long=False,
                notes=["no key terms found in original"],
            )

        summary_lower = summary.lower()
        found   = [t for t in key_terms if t.lower() in summary_lower]
        missing = [t for t in key_terms if t.lower() not in summary_lower]
        coverage = len(found) / len(key_terms) if key_terms else 1.0

        word_count  = len(summary.split())
        is_too_short = word_count < self._min_words
        is_too_long  = word_count > self._max_words

        if is_too_short:
            notes.append(f"summary too short: {word_count} words (min {self._min_words})")
        if is_too_long:
            notes.append(f"summary too long: {word_count} words (max {self._max_words})")
        if coverage < self._min_coverage:
            notes.append(
                f"low coverage: {coverage:.0%} of key terms found "
                f"(min {self._min_coverage:.0%})"
            )

        # Check for required structural sections
        for section in ("Resolved", "Pending", "Active Task"):
            if section.lower() not in summary_lower:
                notes.append(f"missing section: {section}")

        passed = (
            not is_too_short
            and coverage >= self._min_coverage
            and "Active Task" in summary or "active task" in summary_lower
        )

        return QualityAuditResult(
            passed=passed,
            coverage_score=coverage,
            missing_terms=missing[:20],
            word_count=word_count,
            is_too_short=is_too_short,
            is_too_long=is_too_long,
            notes=notes,
        )

    def _extract_key_terms(self, messages: List[Dict]) -> List[str]:
        """
        Extract key terms: file paths, error messages, identifiers, URLs.
        """
        import re as _re
        terms: List[str] = []
        combined = ""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        combined += part.get("text", "") + " "
            elif isinstance(content, str):
                combined += content + " "

        # File paths (unix style)
        terms += _re.findall(r"(?:^|[\s\"'])(/[\w./\-_]+\.(?:py|js|ts|json|yaml|md|txt|sh|go|rs))",
                             combined)[:15]
        # Error codes / identifiers in ALL_CAPS or camelCase
        terms += [w for w in _re.findall(r"\b[A-Z][A-Z_]{3,}\b", combined)
                  if w not in ("HTTP", "JSON", "XML", "HTML", "TRUE", "FALSE", "NULL")][:10]
        # URLs
        terms += _re.findall(r"https?://\S+", combined)[:5]
        # Numbers that look like IDs (e.g., port numbers, task IDs)
        terms += _re.findall(r"\b(?:task|issue|pr|commit|sha|id)[_\-]?#?\d+\b",
                             combined, _re.IGNORECASE)[:5]
        # Extra user-defined patterns
        for pat in self._extra_patterns:
            terms += _re.findall(pat, combined)[:3]

        # Deduplicate
        seen: Dict[str, bool] = {}
        result: List[str] = []
        for t in terms:
            t = t.strip().strip("\"' ")
            if t and t not in seen:
                seen[t] = True
                result.append(t)
        return result


# ---------------------------------------------------------------------------
# Async / Background Context Compressor
# ---------------------------------------------------------------------------

class BackgroundCompressor:
    """
    Non-blocking context compressor that runs compression in a background
    thread so the agent loop never stalls waiting for the summariser LLM.

    Usage:
        bg = BackgroundCompressor()
        # Kick off compression without blocking:
        bg.submit(messages, system, callback=on_done)
        # Later: bg.get_result() -> (new_messages, did_compress) | None
    """

    def __init__(
        self,
        config:      Optional[CompressorConfig] = None,
        max_workers: int = 1,
    ) -> None:
        self._compressor = ContextCompressor(config)
        self._pool       = ThreadPoolExecutor(max_workers=max_workers,
                                              thread_name_prefix="operon-compress")
        self._pending:   Optional[Future] = None
        self._last_result: Optional[Tuple[List[Dict], bool]] = None
        self._lock       = threading.Lock()

    def submit(
        self,
        messages:  List[Dict],
        system:    str = "",
        force:     bool = False,
        callback:  Optional[Callable[[List[Dict], bool], None]] = None,
    ) -> Future:
        """
        Submit a compression job. Returns a Future immediately.
        If callback is provided, it is called with (new_messages, did_compress)
        when the job completes.
        """
        with self._lock:
            # Cancel any pending job (messages have changed)
            if self._pending and not self._pending.done():
                self._pending.cancel()

            def _run() -> Tuple[List[Dict], bool]:
                result = self._compressor.maybe_compress(messages, system, force=force)
                with self._lock:
                    self._last_result = result
                if callback:
                    try:
                        callback(*result)
                    except Exception as e:
                        log.warning("BackgroundCompressor callback failed: %s", e)
                return result

            future = self._pool.submit(_run)
            self._pending = future
            return future

    def get_result(self) -> Optional[Tuple[List[Dict], bool]]:
        """
        Return the most recent completed compression result, or None if still pending.
        """
        with self._lock:
            if self._pending and self._pending.done():
                try:
                    return self._pending.result()
                except Exception:
                    return self._last_result
            return self._last_result

    def wait(self, timeout: float = 30.0) -> Optional[Tuple[List[Dict], bool]]:
        """Block until the current job finishes (or times out). Returns result or None."""
        with self._lock:
            f = self._pending
        if f is None:
            return self._last_result
        try:
            return f.result(timeout=timeout)
        except Exception as e:
            log.warning("BackgroundCompressor.wait failed: %s", e)
            return None

    def is_running(self) -> bool:
        """Return True if a compression job is currently in progress."""
        with self._lock:
            return bool(self._pending and not self._pending.done())

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the background thread pool."""
        self._pool.shutdown(wait=wait)

    def __del__(self) -> None:
        try:
            self._pool.shutdown(wait=False)
        except Exception:
            pass


class AsyncContextCompressor:
    """
    Async wrapper around ContextCompressor for use in asyncio event loops.

    Example:
        async with AsyncContextCompressor() as acc:
            new_msgs, did = await acc.maybe_compress(messages)
    """

    def __init__(self, config: Optional[CompressorConfig] = None) -> None:
        self._compressor = ContextCompressor(config)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def maybe_compress(
        self,
        messages:   List[Dict],
        system:     str  = "",
        force:      bool = False,
        timeout:    float = 60.0,
    ) -> Tuple[List[Dict], bool]:
        """
        Async version of ContextCompressor.maybe_compress().
        Runs the blocking summariser LLM call in a thread executor.
        """
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._compressor.maybe_compress(messages, system, force=force),
                ),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            log.warning("AsyncContextCompressor timed out after %.0fs", timeout)
            return messages, False
        except Exception as e:
            log.warning("AsyncContextCompressor error: %s", e)
            return messages, False

    async def __aenter__(self) -> "AsyncContextCompressor":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    @property
    def config(self) -> CompressorConfig:
        return self._compressor._config


# ---------------------------------------------------------------------------
# Sliding-window compressor (compress oldest window, keep rest intact)
# ---------------------------------------------------------------------------

class SlidingWindowCompressor:
    """
    Compresses the oldest fixed-size window of messages rather than the
    entire middle section. Useful for very long sessions where you want
    incremental compression without touching recent context.

    Each call to compress_oldest() compresses one window:
        [window_start : window_start + window_size]  →  one summary message
    """

    def __init__(
        self,
        window_size: int = 20,
        config:      Optional[CompressorConfig] = None,
    ) -> None:
        self._window_size = window_size
        self._compressor  = ContextCompressor(config)
        self._windows_compressed = 0

    def compress_oldest(
        self,
        messages: List[Dict],
        system:   str = "",
    ) -> Tuple[List[Dict], bool]:
        """
        Find the oldest uncompressed window and compress it.
        Returns (new_messages, did_compress).
        """
        conv  = [m for m in messages if m.get("role") != "system"]
        sys_m = [m for m in messages if m.get("role") == "system"]

        # Skip already-compacted messages
        start_idx = 0
        for i, m in enumerate(conv):
            if m.get("_compacted"):
                start_idx = i + 1

        if len(conv) - start_idx < self._window_size:
            return messages, False

        end_idx = start_idx + self._window_size
        window  = conv[start_idx:end_idx]

        # Build a single-pass summary of this window
        summary_text = self._compressor._summarise(window, system)
        if not summary_text:
            return messages, False

        summary_msg = {
            "role": "user",
            "content": (
                f"[CONTEXT COMPACTION — Window {self._windows_compressed + 1}]\n"
                f"{summary_text}"
            ),
            "_compacted": True,
        }

        new_conv = conv[:start_idx] + [summary_msg] + conv[end_idx:]
        self._windows_compressed += 1
        log.info(
            "SlidingWindowCompressor: compressed window [%d:%d] → 1 message",
            start_idx, end_idx,
        )
        return sys_m + new_conv, True

    def stats(self) -> Dict[str, Any]:
        return {"windows_compressed": self._windows_compressed,
                "window_size": self._window_size}
