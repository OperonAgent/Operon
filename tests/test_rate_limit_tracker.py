"""tests/test_rate_limit_tracker.py — proactive rate-limit awareness (harvested from Hermes)."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.rate_limit_tracker import (
    RateLimitTracker, RateLimitState, RateLimitBucket,
    parse_rate_limit_headers, _parse_duration, _MAX_PROACTIVE_SLEEP,
)


class TestDurationParsing:
    @pytest.mark.parametrize("value,expected", [
        ("12", 12.0), ("0.5", 0.5), (5, 5.0), (2.5, 2.5),
        ("6m0s", 360.0), ("1m30s", 90.0), ("500ms", 0.5), ("2s", 2.0),
        ("1h", 3600.0), ("", 0.0), (None, 0.0), ("garbage", 0.0),
    ])
    def test_parse(self, value, expected):
        assert abs(_parse_duration(value) - expected) < 1e-6


class TestHeaderParsing:
    def test_empty(self):
        assert parse_rate_limit_headers({}).has_data is False

    def test_non_mapping(self):
        assert parse_rate_limit_headers(None).has_data is False  # type: ignore[arg-type]

    def test_case_insensitive(self):
        st = parse_rate_limit_headers({
            "X-RateLimit-Limit-Requests": "100",
            "X-RateLimit-Remaining-Requests": "40",
            "X-RateLimit-Reset-Requests": "30",
        })
        assert st.requests.limit == 100
        assert st.requests.remaining == 40

    def test_remaining_absent_defaults_to_limit(self):
        st = parse_rate_limit_headers({"x-ratelimit-limit-tokens": "1000"})
        assert st.tokens.remaining == 1000  # absent -> full, not 0

    def test_tokens_window(self):
        st = parse_rate_limit_headers({
            "x-ratelimit-limit-tokens": "1000",
            "x-ratelimit-remaining-tokens": "10",
            "x-ratelimit-reset-tokens": "4",
        })
        assert st.tokens.is_low() is True


class TestProactiveWait:
    def test_low_window_waits(self):
        st = parse_rate_limit_headers({
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "1",
            "x-ratelimit-reset-requests": "3",
        })
        assert 2.5 <= st.proactive_wait() <= 3.0

    def test_healthy_window_no_wait(self):
        st = parse_rate_limit_headers({
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "90",
            "x-ratelimit-reset-requests": "3",
        })
        assert st.proactive_wait() == 0.0

    def test_wait_is_bounded(self):
        st = parse_rate_limit_headers({
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "9999",
        })
        assert st.proactive_wait() == _MAX_PROACTIVE_SLEEP

    def test_zero_remaining_no_limit(self):
        # Some providers omit limit but send remaining=0 + reset.
        b = RateLimitBucket(limit=0, remaining=0, reset_seconds=2)
        assert b.is_low() is True


class TestTracker:
    def test_update_and_wait(self):
        t = RateLimitTracker()
        t.update("openai", {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "1",
            "x-ratelimit-reset-requests": "2",
        })
        assert t.seconds_to_wait("openai") > 0

    def test_update_ignores_empty_headers(self):
        t = RateLimitTracker()
        t.update("openai", {})
        t.update("openai", None)
        assert t.state("openai") is None
        assert t.seconds_to_wait("openai") == 0.0

    def test_per_provider_isolation(self):
        t = RateLimitTracker()
        t.update("openai", {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "1",
            "x-ratelimit-reset-requests": "2",
        })
        assert t.seconds_to_wait("anthropic") == 0.0

    def test_throttle_sleeps_and_reports(self):
        t = RateLimitTracker()
        t.update("openai", {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "2",
        })
        slept = []
        msgs = []
        waited = t.throttle("openai", sink=msgs.append, sleep=slept.append)
        assert waited > 0
        assert slept and slept[0] == waited
        assert msgs and "429" in msgs[0]

    def test_throttle_noop_when_healthy(self):
        t = RateLimitTracker()
        t.update("openai", {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
            "x-ratelimit-reset-requests": "2",
        })
        slept = []
        waited = t.throttle("openai", sleep=slept.append)
        assert waited == 0.0
        assert slept == []


class TestRouterIntegration:
    def test_router_has_tracker(self):
        from unittest.mock import MagicMock
        from core.router import ModelRouter
        r = ModelRouter(MagicMock())
        assert hasattr(r, "_rl_tracker")
        assert r._rl_tracker.seconds_to_wait("openai") == 0.0
