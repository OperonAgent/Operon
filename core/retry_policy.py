"""
Operon Tenacity-Style Per-Tool Retry Policies.

Inspired by Hermes Agent's Tenacity integration. Each tool category (web,
SSH, database, messaging, etc.) can have its own retry policy with
configurable attempts, exponential backoff, and error-matching rules.

Usage
-----
    from core.retry_policy import execute_with_retry, DEFAULT_POLICIES

    result = execute_with_retry("web_scrape", web_scrape, {"url": "..."})

Policies are configured in ~/.operon/retry_policies.json or via the
/retry commands in the REPL.
"""

from __future__ import annotations

import json
import math
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("operon.retry")

_POLICIES_PATH = Path.home() / ".operon" / "retry_policies.json"


# ── RetryPolicy dataclass ─────────────────────────────────────────────────────

@dataclass
class RetryPolicy:
    """
    Configuration for how a tool should be retried on failure.

    Attributes
    ----------
    max_attempts      : Total attempts including the first (default 3)
    base_delay_s      : Seconds to wait before first retry (default 1.0)
    backoff_factor    : Multiplier applied each retry, e.g. 2 → 1s, 2s, 4s (default 2.0)
    max_delay_s       : Cap on retry delay (default 30s)
    retry_on_error    : Retry if result["error"] contains any of these strings (default any error)
    no_retry_on_error : NEVER retry if error contains any of these strings
    retry_on_success  : Retry even if success=True, e.g. if result is empty (unusual)
    """
    max_attempts:      int        = 3
    base_delay_s:      float      = 1.0
    backoff_factor:    float      = 2.0
    max_delay_s:       float      = 30.0
    retry_on_error:    List[str]  = field(default_factory=list)   # empty = retry on any error
    no_retry_on_error: List[str]  = field(default_factory=list)
    enabled:           bool       = True

    def delay_for_attempt(self, attempt: int) -> float:
        """Return sleep duration for attempt N (0-indexed)."""
        delay = self.base_delay_s * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay_s)

    def should_retry(self, result: dict, attempt: int) -> bool:
        """Return True if the result warrants another attempt."""
        if not self.enabled:
            return False
        if attempt >= self.max_attempts - 1:
            return False
        if result.get("success", False):
            return False  # success — no retry unless explicitly configured

        error_str = str(result.get("error", "")).lower()

        # Hard-stop patterns
        for pat in self.no_retry_on_error:
            if pat.lower() in error_str:
                return False

        # Selective retry
        if self.retry_on_error:
            return any(pat.lower() in error_str for pat in self.retry_on_error)

        # Retry on any error
        return bool(error_str)


# ── Default policies per tool category ───────────────────────────────────────

DEFAULT_POLICIES: Dict[str, RetryPolicy] = {
    # Web: retry on network errors, 429 rate limits, timeouts
    "duckduckgo_search":   RetryPolicy(max_attempts=3, base_delay_s=2.0, backoff_factor=2.0,
                                       no_retry_on_error=["invalid query"]),
    "web_scrape":          RetryPolicy(max_attempts=3, base_delay_s=1.5, backoff_factor=2.0,
                                       no_retry_on_error=["404", "403"]),
    "http_request":        RetryPolicy(max_attempts=3, base_delay_s=1.0, backoff_factor=2.0,
                                       retry_on_error=["timeout", "connection", "502", "503", "429"]),
    # SSH: retry on transient connection issues
    "ssh_exec":            RetryPolicy(max_attempts=2, base_delay_s=3.0, backoff_factor=1.5,
                                       retry_on_error=["connection reset", "timeout", "timed out"]),
    "ssh_upload":          RetryPolicy(max_attempts=2, base_delay_s=3.0, backoff_factor=1.5),
    "ssh_download":        RetryPolicy(max_attempts=2, base_delay_s=3.0, backoff_factor=1.5),
    # Database: retry on deadlocks and transient errors
    "db_query":            RetryPolicy(max_attempts=2, base_delay_s=0.5, backoff_factor=2.0,
                                       retry_on_error=["deadlock", "lock timeout", "connection"],
                                       no_retry_on_error=["syntax error", "no such table"]),
    # Messaging: retry on rate limits
    "discord_send":        RetryPolicy(max_attempts=3, base_delay_s=1.0, backoff_factor=2.0,
                                       retry_on_error=["rate limit", "429", "connection"]),
    "slack_send":          RetryPolicy(max_attempts=3, base_delay_s=1.0, backoff_factor=2.0,
                                       retry_on_error=["rate_limited", "timeout"]),
    "telegram_send":       RetryPolicy(max_attempts=3, base_delay_s=1.0, backoff_factor=2.0,
                                       retry_on_error=["flood", "timeout", "connection"]),
    "whatsapp_send":       RetryPolicy(max_attempts=2, base_delay_s=2.0, backoff_factor=2.0),
    # Browser: retry on navigation timeouts
    "browser_navigate":    RetryPolicy(max_attempts=2, base_delay_s=2.0, backoff_factor=1.0,
                                       retry_on_error=["timeout", "net::err"]),
    # Email: never retry automatically (requires user approval)
    "email_draft":         RetryPolicy(max_attempts=1, enabled=False),
    # Default for everything else
    "_default":            RetryPolicy(max_attempts=2, base_delay_s=1.0, backoff_factor=2.0),
}


# ── Policy persistence ────────────────────────────────────────────────────────

def _load_custom_policies() -> Dict[str, RetryPolicy]:
    if not _POLICIES_PATH.exists():
        return {}
    try:
        raw = json.loads(_POLICIES_PATH.read_text(encoding="utf-8"))
        return {k: RetryPolicy(**v) for k, v in raw.items()}
    except Exception as e:
        log.warning("Could not load custom retry policies: %s", e)
        return {}


def _save_custom_policies(custom: Dict[str, RetryPolicy]) -> None:
    _POLICIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _POLICIES_PATH.write_text(
        json.dumps({k: asdict(v) for k, v in custom.items()}, indent=2),
        encoding="utf-8",
    )


def get_policy(tool_name: str) -> RetryPolicy:
    """Return the effective RetryPolicy for a tool (custom overrides default)."""
    custom = _load_custom_policies()
    if tool_name in custom:
        return custom[tool_name]
    if tool_name in DEFAULT_POLICIES:
        return DEFAULT_POLICIES[tool_name]
    return DEFAULT_POLICIES["_default"]


def set_policy(tool_name: str, policy: RetryPolicy) -> None:
    """Persist a custom retry policy for a tool."""
    custom = _load_custom_policies()
    custom[tool_name] = policy
    _save_custom_policies(custom)


def reset_policy(tool_name: str) -> None:
    """Remove any custom override, reverting to the built-in default."""
    custom = _load_custom_policies()
    custom.pop(tool_name, None)
    _save_custom_policies(custom)


# ── Execution wrapper ─────────────────────────────────────────────────────────

def execute_with_retry(
    tool_name: str,
    fn:        Callable,
    params:    dict,
) -> dict:
    """
    Call fn(**params) and retry according to the tool's RetryPolicy.

    Returns the last result (success or not) after all attempts are exhausted.
    """
    policy  = get_policy(tool_name)
    result: dict = {}

    for attempt in range(policy.max_attempts):
        try:
            result = fn(**params) if params else fn()
        except Exception as e:
            result = {"success": False, "error": str(e), "output": None}

        if not policy.should_retry(result, attempt):
            break

        delay = policy.delay_for_attempt(attempt)
        log.info(
            "Tool '%s' failed (attempt %d/%d), retrying in %.1fs — %s",
            tool_name, attempt + 1, policy.max_attempts, delay,
            str(result.get("error", ""))[:60],
        )
        time.sleep(delay)

    if not result.get("success") and policy.max_attempts > 1:
        attempts_made = min(attempt + 1, policy.max_attempts)
        if attempts_made > 1:
            result["retry_attempts"] = attempts_made

    return result


# ── Policy manager ────────────────────────────────────────────────────────────

class RetryPolicyManager:
    """Thin management interface for the /retry slash command."""

    def list_policies(self) -> List[Dict[str, Any]]:
        custom = _load_custom_policies()
        all_names = sorted(set(list(DEFAULT_POLICIES.keys()) + list(custom.keys())))
        result = []
        for name in all_names:
            if name == "_default":
                continue
            p = get_policy(name)
            result.append({
                "tool":           name,
                "max_attempts":   p.max_attempts,
                "base_delay_s":   p.base_delay_s,
                "backoff_factor": p.backoff_factor,
                "enabled":        p.enabled,
                "custom":         name in custom,
            })
        return result

    def set(self, tool_name: str, max_attempts: int = 3,
            base_delay_s: float = 1.0, backoff_factor: float = 2.0,
            enabled: bool = True) -> None:
        p = RetryPolicy(
            max_attempts=max_attempts,
            base_delay_s=base_delay_s,
            backoff_factor=backoff_factor,
            enabled=enabled,
        )
        set_policy(tool_name, p)

    def reset(self, tool_name: str) -> None:
        reset_policy(tool_name)

    def get(self, tool_name: str) -> RetryPolicy:
        return get_policy(tool_name)
