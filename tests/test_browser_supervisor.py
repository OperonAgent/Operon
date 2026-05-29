"""Tests for core/browser_supervisor.py"""
import asyncio
import time
import pytest
from unittest import mock

from core.browser_supervisor import (
    BrowserSupervisor, BrowserState, BrowserHealth, SupervisorResult,
    detect_captcha, detect_crash, supervised, get_supervisor,
    DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT_SEC,
)


# ── Detection helpers ─────────────────────────────────────────────────────────

class TestDetectCaptcha:
    def test_recaptcha_detected(self):
        assert detect_captcha("Please solve the recaptcha to continue") is True

    def test_hcaptcha_detected(self):
        assert detect_captcha("Complete the hcaptcha challenge") is True

    def test_cloudflare_detected(self):
        assert detect_captcha("Cloudflare challenge please verify you are human") is True

    def test_normal_page_not_captcha(self):
        assert detect_captcha("Welcome to our website, please log in") is False

    def test_empty_string_not_captcha(self):
        assert detect_captcha("") is False

    def test_case_insensitive(self):
        assert detect_captcha("RECAPTCHA required") is True


class TestDetectCrash:
    def test_target_closed_detected(self):
        assert detect_crash("Target closed: browser disconnected") is True

    def test_session_closed_detected(self):
        assert detect_crash("Session closed unexpectedly") is True

    def test_broken_pipe_detected(self):
        assert detect_crash("BrokenPipeError: broken pipe") is True

    def test_normal_error_not_crash(self):
        assert detect_crash("HTTP 404 Not Found") is False

    def test_empty_not_crash(self):
        assert detect_crash("") is False


# ── SupervisorResult ──────────────────────────────────────────────────────────

class TestSupervisorResult:
    def test_bool_true_on_success(self):
        r = SupervisorResult(success=True, value="ok")
        assert bool(r) is True

    def test_bool_false_on_failure(self):
        r = SupervisorResult(success=False, error="broken")
        assert bool(r) is False


# ── BrowserHealth ─────────────────────────────────────────────────────────────

class TestBrowserHealth:
    def test_running_is_healthy(self):
        h = BrowserHealth(state=BrowserState.RUNNING)
        assert h.is_healthy() is True

    def test_crashed_not_healthy(self):
        h = BrowserHealth(state=BrowserState.CRASHED)
        assert h.is_healthy() is False

    def test_closed_not_healthy(self):
        h = BrowserHealth(state=BrowserState.CLOSED)
        assert h.is_healthy() is False

    def test_to_dict_fields(self):
        h = BrowserHealth(state=BrowserState.RUNNING, pages_visited=5)
        d = h.to_dict()
        assert d["state"] == "running"
        assert d["pages_visited"] == 5


# ── BrowserSupervisor.execute ─────────────────────────────────────────────────

class TestBrowserSupervisorExecute:
    def _sup(self, max_retries: int = 2, backoff: float = 0.0) -> BrowserSupervisor:
        return BrowserSupervisor(max_retries=max_retries, backoff_base=backoff)

    def test_success_returns_value(self):
        sup = self._sup()
        r = sup.execute(lambda: "result_value")
        assert r.success is True
        assert r.value == "result_value"

    def test_success_increments_pages_visited(self):
        sup = self._sup()
        sup.execute(lambda: "ok")
        assert sup._health.pages_visited == 1

    def test_failure_retries_max_times(self):
        sup = self._sup(max_retries=2, backoff=0.0)
        call_count = [0]
        def task():
            call_count[0] += 1
            raise RuntimeError("always fails")
        r = sup.execute(task)
        assert r.success is False
        assert call_count[0] == 3   # initial + 2 retries

    def test_success_after_retries(self):
        sup = self._sup(max_retries=3, backoff=0.0)
        call_count = [0]
        def task():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("temp fail")
            return "eventually ok"
        r = sup.execute(task)
        assert r.success is True
        assert r.value == "eventually ok"

    def test_error_message_in_result(self):
        sup = self._sup(max_retries=0)
        def task():
            raise ValueError("specific error message")
        r = sup.execute(task)
        assert "specific error message" in r.error

    def test_captcha_page_returns_had_captcha(self):
        sup = BrowserSupervisor(max_retries=0, backoff_base=0.0)
        def task():
            return "please solve this recaptcha challenge to continue"
        # Captcha handler will retry, but with max_retries=0, will get captcha on first pass
        r = sup.execute(task)
        # Either had_captcha or success=False
        assert r.had_captcha or not r.success

    def test_crash_triggers_restart_callback(self):
        restart_called = [False]
        def on_restart():
            restart_called[0] = True
            return None  # return None to indicate mock browser
        sup = BrowserSupervisor(max_retries=1, backoff_base=0.0, on_restart=on_restart)
        def task():
            raise RuntimeError("Target closed session closed unexpectedly")
        sup.execute(task)
        assert restart_called[0]

    def test_duration_ms_recorded(self):
        sup = self._sup(max_retries=0)
        r = sup.execute(lambda: "ok")
        assert r.duration_ms >= 0

    def test_retries_count_in_result(self):
        sup = self._sup(max_retries=2, backoff=0.0)
        attempts = [0]
        def task():
            attempts[0] += 1
            if attempts[0] < 2:
                raise RuntimeError("fail")
            return "ok"
        r = sup.execute(task)
        assert r.retries >= 1


# ── BrowserSupervisor async ───────────────────────────────────────────────────

class TestBrowserSupervisorAsync:
    def test_async_success(self):
        sup = BrowserSupervisor(max_retries=1, backoff_base=0.0)
        async def task():
            return "async result"
        r = asyncio.run(sup.execute_async(task))
        assert r.success and r.value == "async result"

    def test_async_failure_retries(self):
        sup = BrowserSupervisor(max_retries=2, backoff_base=0.0)
        calls = [0]
        async def task():
            calls[0] += 1
            if calls[0] < 3:
                raise RuntimeError("temp")
            return "done"
        r = asyncio.run(sup.execute_async(task))
        assert r.success

    def test_async_captcha_detected(self):
        sup = BrowserSupervisor(max_retries=0, backoff_base=0.0)
        async def task():
            return "hcaptcha challenge required"
        r = asyncio.run(sup.execute_async(task))
        assert r.had_captcha or not r.success


# ── Health check & lifecycle ──────────────────────────────────────────────────

class TestHealthAndLifecycle:
    def test_health_check_returns_health(self):
        sup = BrowserSupervisor()
        h = sup.health_check()
        assert isinstance(h, BrowserHealth)

    def test_health_uptime_positive(self):
        sup = BrowserSupervisor()
        time.sleep(0.01)
        h = sup.health_check()
        assert h.uptime_seconds > 0

    def test_is_alive_no_browser(self):
        sup = BrowserSupervisor(browser=None)
        assert sup.is_alive() is False

    def test_restart_calls_callback(self):
        new_browser = object()
        def on_restart():
            return new_browser
        sup = BrowserSupervisor(on_restart=on_restart)
        ok = sup.restart()
        assert ok
        assert sup._browser is new_browser

    def test_restart_no_callback_still_ok(self):
        sup = BrowserSupervisor()
        ok = sup.restart()
        assert ok

    def test_close_no_browser(self):
        sup = BrowserSupervisor(browser=None)
        sup.close()  # should not raise
        assert sup._health.state == BrowserState.CLOSED

    def test_close_with_browser_close(self):
        browser = mock.MagicMock()
        sup = BrowserSupervisor(browser=browser)
        sup.close()
        browser.close.assert_called_once()


# ── Backoff ───────────────────────────────────────────────────────────────────

class TestBackoff:
    def test_backoff_increases_with_attempt(self):
        sup = BrowserSupervisor(backoff_base=1.0, backoff_max=100.0, jitter=False)
        assert sup._backoff(0) < sup._backoff(1) < sup._backoff(2)

    def test_backoff_caps_at_max(self):
        sup = BrowserSupervisor(backoff_base=1.0, backoff_max=5.0, jitter=False)
        assert sup._backoff(100) <= 5.0

    def test_backoff_with_jitter_in_range(self):
        sup = BrowserSupervisor(backoff_base=2.0, backoff_max=100.0, jitter=True)
        for _ in range(10):
            b = sup._backoff(0)
            assert 2.0 * 0.75 <= b <= 2.0 * 1.25 + 0.1


# ── @supervised decorator ─────────────────────────────────────────────────────

class TestSupervisedDecorator:
    def test_success_returns_value(self):
        @supervised(max_retries=0)
        def task():
            return 42
        assert task() == 42

    def test_failure_raises(self):
        @supervised(max_retries=0)
        def task():
            raise RuntimeError("broken")
        with pytest.raises(RuntimeError):
            task()

    def test_retry_on_failure_then_success(self):
        calls = [0]
        @supervised(max_retries=3)
        def task():
            calls[0] += 1
            if calls[0] < 3:
                raise RuntimeError("temp")
            return "ok"
        result = task()
        assert result == "ok"


# ── Module-level API ──────────────────────────────────────────────────────────

class TestModuleLevelAPI:
    def test_get_supervisor_returns_instance(self):
        s = get_supervisor()
        assert isinstance(s, BrowserSupervisor)

    def test_get_supervisor_sets_browser(self):
        browser = object()
        s = get_supervisor(browser=browser)
        assert s._browser is browser
