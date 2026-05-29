"""
Operon TokenJuice — Tool Result Compression Middleware.

Adapted from OpenClaw extensions/tokenjuice/tool-result-middleware.ts.

Post-execution middleware that compresses noisy tool outputs before they
are added to the conversation history, reducing context window usage.

Compression strategies (applied in priority order):
  1. Truncation     — hard cap on output length
  2. Line dedup     — collapse repeated/near-identical consecutive lines
  3. Whitespace squeeze  — normalise excess blank lines
  4. Log pattern strip   — remove common log noise (timestamps, PIDs, etc.)
  5. Stack trace fold    — keep first+last N lines of tracebacks
  6. JSON pretty-print   — detect and pretty-print embedded JSON
  7. Table trim          — if output looks tabular, keep header + sampled rows

The compressor is configurable per-tool via a ToolJuiceConfig.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class ToolJuiceConfig:
    """Per-tool compression configuration."""
    enabled:              bool  = True
    max_chars:            int   = 8_000     # hard cap after all other steps
    max_lines:            int   = 200       # hard cap on line count
    dedup_lines:          bool  = True
    squeeze_blank_lines:  bool  = True
    strip_log_noise:      bool  = True
    fold_tracebacks:      bool  = True
    traceback_head_lines: int   = 5
    traceback_tail_lines: int   = 5
    pretty_json:          bool  = True
    trim_tables:          bool  = True
    table_max_rows:       int   = 30
    add_summary_note:     bool  = True      # append "[N lines compressed]" when truncating


# Default configs for different tool categories
_CONFIGS: dict[str, ToolJuiceConfig] = {
    "shell_exec":   ToolJuiceConfig(max_chars=6_000, max_lines=150, fold_tracebacks=True),
    "file_ops":     ToolJuiceConfig(max_chars=8_000, max_lines=300, strip_log_noise=False),
    "web_search":   ToolJuiceConfig(max_chars=4_000, max_lines=100, pretty_json=False),
    "http_client":  ToolJuiceConfig(max_chars=5_000, max_lines=150),
    "db_ops":       ToolJuiceConfig(max_chars=6_000, max_lines=200, trim_tables=True),
    "git_ops":      ToolJuiceConfig(max_chars=5_000, max_lines=150, strip_log_noise=True),
    "code_exec":    ToolJuiceConfig(max_chars=6_000, max_lines=200, fold_tracebacks=True),
    "docker_exec":  ToolJuiceConfig(max_chars=5_000, max_lines=150, strip_log_noise=True),
    "_default":     ToolJuiceConfig(),
}


def get_config(tool_name: str) -> ToolJuiceConfig:
    """Return compression config for a tool, falling back to default."""
    return _CONFIGS.get(tool_name, _CONFIGS["_default"])


# ── Log noise patterns ─────────────────────────────────────────────────────────

_LOG_NOISE_RE = re.compile(
    r"(?:"
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"  # ISO timestamps
    r"|\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]"   # [2024-01-01 00:00:00]
    r"|^\s*(?:DEBUG|TRACE|VERBOSE)\s*:?"            # debug/trace log level prefix
    r"|^\s*(?:INFO|WARNING|ERROR|WARN)\s+\d+\s+"   # log-level + PID prefix
    r"|\bpid=\d+\b|\bthread=\d+\b"                  # PID/thread tags
    r"|\x1b\[[0-9;]*m"                              # ANSI color codes
    r")",
    re.IGNORECASE | re.MULTILINE,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]")

# ── Traceback detection ────────────────────────────────────────────────────────

_TRACEBACK_START_RE = re.compile(
    r"^(?:Traceback \(most recent call last\):|Error:|Exception:|"
    r".*Error: .*|.*Exception: .*)",
    re.MULTILINE,
)


# ── Core compressor ────────────────────────────────────────────────────────────

def compress(
    text:      str,
    tool_name: str = "_default",
    config:    Optional[ToolJuiceConfig] = None,
) -> str:
    """
    Compress `text` (a tool's output) using the configured strategies.

    Returns the compressed string.
    """
    if not text:
        return text

    cfg = config or get_config(tool_name)
    if not cfg.enabled:
        return text

    original_len = len(text)
    original_lines = text.count("\n") + 1

    # Strip ANSI codes first (always)
    text = _ANSI_RE.sub("", text)

    # Step 1: Strip log noise
    if cfg.strip_log_noise:
        text = _strip_log_noise(text)

    # Step 2: Dedup consecutive repeated lines
    if cfg.dedup_lines:
        text = _dedup_lines(text)

    # Step 3: Squeeze blank lines
    if cfg.squeeze_blank_lines:
        text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 4: Pretty-print embedded JSON
    if cfg.pretty_json:
        text = _pretty_json(text)

    # Step 5: Fold tracebacks
    if cfg.fold_tracebacks:
        text = _fold_tracebacks(text, cfg.traceback_head_lines, cfg.traceback_tail_lines)

    # Step 6: Trim tables
    if cfg.trim_tables:
        text = _trim_table(text, cfg.table_max_rows)

    # Step 7: Line cap
    lines = text.splitlines()
    if len(lines) > cfg.max_lines:
        kept  = lines[:cfg.max_lines]
        skipped = len(lines) - cfg.max_lines
        text = "\n".join(kept)
        if cfg.add_summary_note:
            text += f"\n... [{skipped} more lines omitted]"

    # Step 8: Char cap (hard limit, last resort)
    if len(text) > cfg.max_chars:
        text = text[:cfg.max_chars]
        if cfg.add_summary_note:
            compressed_by = original_len - cfg.max_chars
            text += f"\n... [{compressed_by} chars truncated]"

    return text


# ── Strategy implementations ───────────────────────────────────────────────────

def _strip_log_noise(text: str) -> str:
    """Remove common log noise patterns line-by-line."""
    cleaned_lines = []
    for line in text.splitlines():
        # Strip ANSI from individual lines first
        clean_line = _ANSI_RE.sub("", line)
        # Skip lines that are PURELY a timestamp/log-level prefix with nothing else
        if re.match(
            r"^\s*(?:\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|"
            r"\[DEBUG\]|\[TRACE\]|\[VERBOSE\])\s*$",
            clean_line, re.IGNORECASE,
        ):
            continue
        # Strip inline log noise but keep the rest of the line
        cleaned_lines.append(_LOG_NOISE_RE.sub("", clean_line))
    return "\n".join(cleaned_lines)


def _dedup_lines(text: str) -> str:
    """Collapse consecutive identical/near-identical lines."""
    lines = text.splitlines()
    out: list[str] = []
    repeat_count = 0
    prev: Optional[str] = None

    for line in lines:
        stripped = line.rstrip()
        if stripped == prev:
            repeat_count += 1
        else:
            if repeat_count > 0:
                out.append(f"  ... (repeated {repeat_count} more times)")
            out.append(stripped)
            prev = stripped
            repeat_count = 0

    if repeat_count > 0:
        out.append(f"  ... (repeated {repeat_count} more times)")

    return "\n".join(out)


def _pretty_json(text: str) -> str:
    """Detect and pretty-print embedded JSON blobs."""
    def _try_pretty(m: re.Match) -> str:
        try:
            obj = json.loads(m.group(0))
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            return m.group(0)

    # Only attempt on significant JSON blobs (> 100 chars of compressed JSON)
    return re.sub(r'\{(?:[^{}]|\{[^{}]*\}){50,}\}', _try_pretty, text)


def _fold_tracebacks(text: str, head: int, tail: int) -> str:
    """Fold long tracebacks: keep first `head` + last `tail` lines."""
    lines = text.splitlines()
    if len(lines) <= head + tail + 2:
        return text  # short enough already

    # Find traceback blocks
    in_tb   = False
    tb_start: Optional[int] = None
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not in_tb and _TRACEBACK_START_RE.match(line):
            in_tb    = True
            tb_start = i
            tb_lines = [line]
        elif in_tb:
            tb_lines.append(line)  # type: ignore[possibly-undefined]
            # Traceback ends at the first line that doesn't look like part of it
            is_frame = (
                line.strip().startswith("File ")
                or line.strip().startswith("at ")
                or line.strip().startswith("in ")
                or line.startswith("    ")
                or line.startswith("\t")
                or re.match(r"^\s+\^+\s*$", line)
            )
            if not is_frame and len(tb_lines) > 1:  # type: ignore[possibly-undefined]
                # End of traceback
                if len(tb_lines) > head + tail + 3:  # type: ignore[possibly-undefined]
                    kept = (tb_lines[:head]  # type: ignore[possibly-undefined]
                            + [f"  ... [{len(tb_lines) - head - tail} frames omitted] ..."]  # type: ignore
                            + tb_lines[-tail:])  # type: ignore[possibly-undefined]
                    result.extend(kept)
                else:
                    result.extend(tb_lines)  # type: ignore[possibly-undefined]
                in_tb  = False
                tb_start = None
                tb_lines = []  # type: ignore[assignment]
        else:
            result.append(line)
        i += 1

    if in_tb and tb_lines:  # type: ignore[possibly-undefined]
        result.extend(tb_lines)  # type: ignore[possibly-undefined]

    return "\n".join(result)


def _trim_table(text: str, max_rows: int) -> str:
    """
    If text looks like a text table (contains pipe-separated columns), keep
    header + at most `max_rows` data rows, appending a summary if truncated.
    """
    lines = text.splitlines()
    # Heuristic: at least 30% of lines contain a | and there are > max_rows lines
    pipe_lines = sum(1 for l in lines if "|" in l)
    if pipe_lines < 0.3 * len(lines) or len(lines) <= max_rows + 3:
        return text

    # Keep header (first 2 lines) + separator + max_rows data rows
    header = lines[:2]
    data   = [l for l in lines[2:] if l.strip() and not re.match(r"^[-+|= ]+$", l.strip())]
    sep    = [l for l in lines[:5]  if re.match(r"^[-+|= ]+$", l.strip())]

    if len(data) <= max_rows:
        return text

    trimmed_data = data[:max_rows]
    omitted = len(data) - max_rows
    result = header + sep[:1] + trimmed_data
    result.append(f"... [{omitted} more rows omitted]")
    return "\n".join(result)


# ── Middleware entry point ─────────────────────────────────────────────────────

def compress_tool_result(tool_name: str, result: dict) -> dict:
    """
    Apply tokenjuice compression to a tool result dict.
    Modifies "stdout" and "stderr" fields in-place if present,
    or "result" / "output" / "content" fields.

    Returns the (possibly mutated) result dict.
    """
    if not isinstance(result, dict):
        return result

    for key in ("stdout", "stderr", "output", "result", "content", "text"):
        if isinstance(result.get(key), str) and result[key]:
            result[key] = compress(result[key], tool_name=tool_name)

    return result
