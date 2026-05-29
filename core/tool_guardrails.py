"""
Operon Tool-Call Guardrails — ported and adapted from Hermes Agent.

Tracks per-turn tool call patterns and issues warn / block / halt decisions
to break runaway loops before they burn the full iteration budget.

Three failure modes detected:
  1. exact_failure   — same tool + same args (sha256) failing repeatedly
  2. same_tool       — same tool failing with ANY args repeatedly
  3. no_progress     — idempotent read tools returning the same result repeatedly

Actions:
  allow  — proceed normally
  warn   — append guidance to the tool result so the model re-reads it
  block  — return a synthetic error result; model must change strategy
  halt   — same as block but caller should also abort the turn

Usage:
    guardrail = ToolCallGuardrails()          # fresh instance per chat_turn()
    decision = guardrail.before_call(tool, params)
    if decision.should_block:
        # inject synthetic error and continue
    result = run_tool(tool, params)
    decision = guardrail.after_call(tool, params, result, failed=is_error(result))
    if decision.action == "warn":
        result += decision.guidance_suffix()
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


# ── Tool classification ────────────────────────────────────────────────────────

# Read-only tools: identical args + identical result = no progress
IDEMPOTENT_TOOLS = frozenset({
    "file_read", "file_exists", "file_info", "dir_list", "file_search",
    "duckduckgo_search", "web_scrape", "web_fetch",
    "browser_get_url", "browser_snapshot",
    "db_query", "knowledge_get", "knowledge_list",
    "git_status", "git_diff", "git_log",
    "x_search",
})

# Mutating tools: successful execution resets failure counters
MUTATING_TOOLS = frozenset({
    "shell_exec", "python_exec",
    "file_write", "file_append", "file_patch", "file_delete",
    "email_draft",
    "todo", "clarify",
    "browser_click", "browser_type", "browser_navigate",
    "git_commit", "git_push", "git_checkout",
    "docker_run", "docker_build",
    "ssh_exec",
    "sub_agent",
    "telegram_send", "discord_send", "slack_send", "matrix_send",
    "signal_send", "mattermost_send", "teams_send",
})


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GuardrailConfig:
    """Thresholds — configurable via /setup or operon config."""
    warnings_enabled:             bool = True
    hard_stop_enabled:            bool = False   # opt-in circuit breaker
    exact_failure_warn_after:     int  = 2
    exact_failure_block_after:    int  = 5
    same_tool_warn_after:         int  = 3
    same_tool_halt_after:         int  = 8
    no_progress_warn_after:       int  = 2
    no_progress_block_after:      int  = 4

    @classmethod
    def from_config(cls, cfg: dict) -> "GuardrailConfig":
        # cfg may be the raw full config dict OR the already-extracted inner dict.
        # Detect which: if it has a "tool_guardrails" sub-key, unwrap it;
        # otherwise it IS the inner dict (caller already did the first .get()).
        if isinstance(cfg, dict) and "tool_guardrails" in cfg:
            g = cfg["tool_guardrails"]
        else:
            g = cfg if isinstance(cfg, dict) else {}
        d = cls()
        return cls(
            warnings_enabled          = _as_bool(g.get("warnings_enabled"),        d.warnings_enabled),
            hard_stop_enabled         = _as_bool(g.get("hard_stop_enabled"),        d.hard_stop_enabled),
            exact_failure_warn_after  = _pos_int(g.get("exact_failure_warn_after"), d.exact_failure_warn_after),
            exact_failure_block_after = _pos_int(g.get("exact_failure_block_after"),d.exact_failure_block_after),
            same_tool_warn_after      = _pos_int(g.get("same_tool_warn_after"),     d.same_tool_warn_after),
            same_tool_halt_after      = _pos_int(g.get("same_tool_halt_after"),     d.same_tool_halt_after),
            no_progress_warn_after    = _pos_int(g.get("no_progress_warn_after"),   d.no_progress_warn_after),
            no_progress_block_after   = _pos_int(g.get("no_progress_block_after"),  d.no_progress_block_after),
        )


@dataclass
class GuardrailDecision:
    """Decision returned by the guardrail controller."""
    action:    str = "allow"   # allow | warn | block | halt
    code:      str = "allow"
    message:   str = ""
    tool_name: str = ""
    count:     int = 0

    @property
    def should_block(self) -> bool:
        return self.action in {"block", "halt"}

    @property
    def allows_execution(self) -> bool:
        return self.action in {"allow", "warn"}

    def synthetic_result(self) -> str:
        """Synthetic tool result JSON to inject when blocking."""
        return json.dumps({
            "error": self.message,
            "guardrail": {"action": self.action, "code": self.code, "count": self.count},
        }, ensure_ascii=False)

    def guidance_suffix(self) -> str:
        """Text appended to a real tool result to nudge the model."""
        if not self.message:
            return ""
        label = "Tool loop hard stop" if self.action == "halt" else "Tool loop warning"
        return f"\n\n[{label}: {self.code}; count={self.count}; {self.message}]"


# ── Main controller ────────────────────────────────────────────────────────────

class ToolCallGuardrails:
    """Per-turn guardrail controller — create one instance per chat_turn()."""

    def __init__(self, config: Optional[GuardrailConfig] = None):
        self._cfg = config or GuardrailConfig()
        # {signature_hash: fail_count}
        self._exact_failures:    dict[str, int] = {}
        # {tool_name: fail_count}
        self._tool_failures:     dict[str, int] = {}
        # {signature_hash: (result_hash, repeat_count)}
        self._no_progress:       dict[str, tuple[str, int]] = {}
        self._halt_decision: Optional[GuardrailDecision] = None

    @property
    def halt_decision(self) -> Optional[GuardrailDecision]:
        return self._halt_decision

    def before_call(self, tool_name: str, params: Any) -> GuardrailDecision:
        """Check before executing — may block without running the tool."""
        sig = _sig(tool_name, params)

        if self._cfg.hard_stop_enabled:
            exact_count = self._exact_failures.get(sig, 0)
            if exact_count >= self._cfg.exact_failure_block_after:
                d = GuardrailDecision(
                    action="block", code="exact_failure_block",
                    message=(
                        f"Blocked {tool_name}: the same call failed {exact_count} times "
                        "with identical arguments. Change strategy or explain the blocker."
                    ),
                    tool_name=tool_name, count=exact_count,
                )
                self._halt_decision = d
                return d

            if _is_idempotent(tool_name):
                record = self._no_progress.get(sig)
                if record is not None:
                    _, repeat = record
                    if repeat >= self._cfg.no_progress_block_after:
                        d = GuardrailDecision(
                            action="block", code="no_progress_block",
                            message=(
                                f"Blocked {tool_name}: returned the same result {repeat} times. "
                                "Use the result already provided or change the query."
                            ),
                            tool_name=tool_name, count=repeat,
                        )
                        self._halt_decision = d
                        return d

        return GuardrailDecision(tool_name=tool_name)

    def after_call(
        self,
        tool_name: str,
        params: Any,
        result: Optional[str],
        *,
        failed: Optional[bool] = None,
    ) -> GuardrailDecision:
        """Update counters after execution; may return warn/halt."""
        sig = _sig(tool_name, params)
        if failed is None:
            failed = _detect_failure(tool_name, result)

        if failed:
            exact_count = self._exact_failures.get(sig, 0) + 1
            self._exact_failures[sig] = exact_count
            self._no_progress.pop(sig, None)

            tool_count = self._tool_failures.get(tool_name, 0) + 1
            self._tool_failures[tool_name] = tool_count

            if self._cfg.hard_stop_enabled and tool_count >= self._cfg.same_tool_halt_after:
                d = GuardrailDecision(
                    action="halt", code="same_tool_halt",
                    message=(
                        f"Stopped {tool_name}: failed {tool_count} times this turn. "
                        "Stop retrying and choose a different approach."
                    ),
                    tool_name=tool_name, count=tool_count,
                )
                self._halt_decision = d
                return d

            if self._cfg.warnings_enabled and exact_count >= self._cfg.exact_failure_warn_after:
                return GuardrailDecision(
                    action="warn", code="exact_failure_warn",
                    message=(
                        f"{tool_name} failed {exact_count} times with the same arguments. "
                        "Inspect the error and change strategy instead of retrying unchanged."
                    ),
                    tool_name=tool_name, count=exact_count,
                )

            if self._cfg.warnings_enabled and tool_count >= self._cfg.same_tool_warn_after:
                return GuardrailDecision(
                    action="warn", code="same_tool_warn",
                    message=_recovery_hint(tool_name, tool_count),
                    tool_name=tool_name, count=tool_count,
                )

            return GuardrailDecision(tool_name=tool_name, count=exact_count)

        # Successful call — reset failure counters for this tool/sig
        self._exact_failures.pop(sig, None)
        self._tool_failures.pop(tool_name, None)

        if not _is_idempotent(tool_name):
            self._no_progress.pop(sig, None)
            return GuardrailDecision(tool_name=tool_name)

        # Track idempotent tool no-progress
        result_hash = _hash(result or "")
        previous = self._no_progress.get(sig)
        repeat = 1
        if previous is not None and previous[0] == result_hash:
            repeat = previous[1] + 1
        self._no_progress[sig] = (result_hash, repeat)

        if self._cfg.warnings_enabled and repeat >= self._cfg.no_progress_warn_after:
            return GuardrailDecision(
                action="warn", code="no_progress_warn",
                message=(
                    f"{tool_name} returned the same result {repeat} times. "
                    "Use the result already provided or change the query."
                ),
                tool_name=tool_name, count=repeat,
            )

        return GuardrailDecision(tool_name=tool_name, count=repeat)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sig(tool_name: str, params: Any) -> str:
    """Stable hash of tool_name + canonical params."""
    if not isinstance(params, Mapping):
        params = {}
    canonical = json.dumps(dict(params), ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), default=str)
    return hashlib.sha256(f"{tool_name}:{canonical}".encode()).hexdigest()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _is_idempotent(tool_name: str) -> bool:
    if tool_name in MUTATING_TOOLS:
        return False
    return tool_name in IDEMPOTENT_TOOLS


def _detect_failure(tool_name: str, result: Optional[str]) -> bool:
    """Heuristic to decide whether a tool result represents a failure."""
    if result is None:
        return True
    if tool_name == "shell_exec":
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                code = data.get("returncode", data.get("exit_code"))
                if code is not None and int(code) != 0:
                    return True
        except Exception:
            pass
    lower = result[:500].lower()
    if any(k in lower for k in ('"error"', '"failed"', '"success": false',
                                 '"success":false', 'traceback', 'exception')):
        if not any(ok in lower for ok in ('"success": true', '"success":true')):
            return True
    return False


def _recovery_hint(tool_name: str, count: int) -> str:
    base = (
        f"{tool_name} has failed {count} times this turn. "
        "Diagnose the error before retrying. "
    )
    if tool_name == "shell_exec":
        return base + "Try `pwd && ls -la` to verify the working directory and file paths."
    return base + "Try different arguments, an absolute path, or a different tool."


def _as_bool(val: Any, default: bool) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


def _pos_int(val: Any, default: int) -> int:
    try:
        n = int(val)
        return n if n >= 1 else default
    except Exception:
        return default
