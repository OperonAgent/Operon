"""
core/rate_limit_tracker.py — proactive rate-limit awareness.

Harvested and adapted from Hermes Agent's `agent/rate_limit_tracker.py`.

Operon's router already *reacts* to HTTP 429 (honors Retry-After + rotates
keys). This module adds the missing *proactive* half: it reads the
``x-ratelimit-*`` response headers that OpenAI, OpenRouter and Nous-style
endpoints return, and lets the router pre-emptively pause for a fraction of a
second when a window is nearly exhausted — avoiding the 429 round-trip
entirely. It is intentionally side-effect free except for an optional bounded
sleep, and degrades to a no-op when no headers are present.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

# Never block the turn for long: proactive pauses are capped hard.
_MAX_PROACTIVE_SLEEP = 5.0
# Throttle only when this fraction (or fewer) of a window's budget remains.
_LOW_WATER_FRACTION = 0.05


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_duration(value: Any) -> float:
    """
    Parse a reset value into seconds. Handles plain numbers ("12", "0.5"),
    OpenAI-style compound durations ("6m0s", "1m30s", "500ms", "2s"), and
    blanks. Returns 0.0 on anything unrecognised.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value).strip().lower()
    if not text:
        return 0.0
    # plain number (seconds)
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    total = 0.0
    matched = False
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", text):
        matched = True
        n = float(amount)
        total += {"ms": n / 1000.0, "s": n, "m": n * 60.0, "h": n * 3600.0}[unit]
    return total if matched else 0.0


@dataclass
class RateLimitBucket:
    """One rate-limit window (requests-per-minute, tokens-per-minute, …)."""
    limit:         int = 0
    remaining:     int = 0
    reset_seconds: float = 0.0
    captured_at:   float = field(default_factory=time.time)

    @property
    def has_data(self) -> bool:
        return self.limit > 0 or self.remaining > 0 or self.reset_seconds > 0

    def remaining_seconds_now(self) -> float:
        """Seconds until reset, adjusted for time elapsed since capture."""
        elapsed = time.time() - self.captured_at
        return max(0.0, self.reset_seconds - elapsed)

    def is_low(self) -> bool:
        """True when this window is at/under the low-water threshold."""
        if self.limit <= 0:
            return self.remaining == 0 and self.reset_seconds > 0
        return self.remaining <= max(0, int(self.limit * _LOW_WATER_FRACTION))


@dataclass
class RateLimitState:
    """Latest parsed rate-limit windows for one provider."""
    requests: RateLimitBucket = field(default_factory=RateLimitBucket)
    tokens:   RateLimitBucket = field(default_factory=RateLimitBucket)

    @property
    def has_data(self) -> bool:
        return self.requests.has_data or self.tokens.has_data

    def proactive_wait(self) -> float:
        """
        Seconds the caller should pause *before* the next request, bounded.
        Returns 0.0 unless a window is genuinely near-exhausted.
        """
        waits = [b.remaining_seconds_now()
                 for b in (self.requests, self.tokens)
                 if b.has_data and b.is_low()]
        if not waits:
            return 0.0
        return min(_MAX_PROACTIVE_SLEEP, max(waits))


def parse_rate_limit_headers(headers: Mapping[str, str]) -> RateLimitState:
    """Parse ``x-ratelimit-*`` headers into a RateLimitState (case-insensitive)."""
    # requests.Response.headers is already case-insensitive, but accept plain
    # dicts too by lower-casing keys defensively.
    if not isinstance(headers, Mapping):
        return RateLimitState()
    h = {str(k).lower(): v for k, v in headers.items()}

    def bucket(kind: str) -> RateLimitBucket:
        return RateLimitBucket(
            limit=_safe_int(h.get(f"x-ratelimit-limit-{kind}")),
            remaining=_safe_int(h.get(f"x-ratelimit-remaining-{kind}"), default=-1),
            reset_seconds=_parse_duration(h.get(f"x-ratelimit-reset-{kind}")),
        )

    req = bucket("requests")
    tok = bucket("tokens")
    # remaining defaulted to -1 to distinguish "absent" from "0"; normalise.
    if req.remaining < 0:
        req.remaining = req.limit
    if tok.remaining < 0:
        tok.remaining = tok.limit
    return RateLimitState(requests=req, tokens=tok)


class RateLimitTracker:
    """
    Per-provider proactive rate-limit tracker.

    Usage in the router:
        tracker.update(provider, resp.headers)   # after each response
        tracker.throttle(provider, sink=print)   # before the next request
    """

    def __init__(self) -> None:
        self._states: dict[str, RateLimitState] = {}

    def update(self, provider: str, headers: Optional[Mapping[str, str]]) -> None:
        if not headers:
            return
        state = parse_rate_limit_headers(headers)
        if state.has_data:
            self._states[provider] = state

    def state(self, provider: str) -> Optional[RateLimitState]:
        return self._states.get(provider)

    def seconds_to_wait(self, provider: str) -> float:
        state = self._states.get(provider)
        return state.proactive_wait() if state else 0.0

    def throttle(self, provider: str,
                 sink: Optional[Callable[[str], None]] = None,
                 sleep: Callable[[float], None] = time.sleep) -> float:
        """
        Pause (bounded) if the provider is near a rate-limit boundary.
        Returns the number of seconds actually waited.
        """
        wait = self.seconds_to_wait(provider)
        if wait <= 0:
            return 0.0
        if sink:
            sink(f"  [Router] {provider} rate window nearly exhausted — "
                 f"pausing {wait:.1f}s proactively to avoid a 429.")
        sleep(wait)
        return wait
