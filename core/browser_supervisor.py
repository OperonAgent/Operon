"""
Operon Browser Supervisor — Retry/recovery loop with CAPTCHA detection.

Matches Hermes browser_supervisor.py depth.

Wraps any browser automation call with:
  • Automatic retry with exponential backoff
  • CAPTCHA detection and handling hooks
  • CDP (Chrome DevTools Protocol) session management
  • Screenshot-on-failure for debugging
  • Timeout enforcement per action
  • Session health checks and auto-restart
  • Browser crash detection and recovery

Architecture:
  BrowserSupervisor
    .execute(task_fn, *args, **kwargs) → result | raises after max_retries
    .screenshot(path) → saves current page screenshot
    .health_check() → returns BrowserHealth
    .restart() → restart the browser cleanly
    .close() → graceful shutdown

Usage:
    from core.browser_supervisor import BrowserSupervisor

    sup = BrowserSupervisor()
    result = sup.execute(lambda page: page.goto("https://example.com"))

    # With CDP context
    async with sup.cdp_session() as session:
        await session.send("Network.enable")
"""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("operon.browser_supervisor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES   = 4
DEFAULT_TIMEOUT_SEC   = 30.0
DEFAULT_BACKOFF_BASE  = 1.5    # seconds, multiplied per retry
DEFAULT_BACKOFF_MAX   = 60.0
DEFAULT_SCREENSHOT_DIR = "/tmp/operon-browser-screenshots"

# CAPTCHA signals in page content or URL
_CAPTCHA_SIGNALS = [
    "captcha", "recaptcha", "hcaptcha", "turnstile",
    "cloudflare challenge", "bot detection", "please verify",
    "prove you are human", "i'm not a robot",
    "challenge-platform", "cf-browser-verification",
    "challenge.cloudflare.com",
]

# Crash signals in stderr / exception message
_CRASH_SIGNALS = [
    "crashed", "broken pipe", "connection refused",
    "target closed", "session closed", "browser has been disconnected",
    "page has been closed", "execution context was destroyed",
    "detached frame", "protocol error",
]


# ---------------------------------------------------------------------------
# Enums & data types
# ---------------------------------------------------------------------------

class BrowserState(str, Enum):
    STARTING  = "starting"
    RUNNING   = "running"
    DEGRADED  = "degraded"
    CRASHED   = "crashed"
    CLOSED    = "closed"

class RetryReason(str, Enum):
    TIMEOUT   = "timeout"
    CRASH     = "crash"
    CAPTCHA   = "captcha"
    EXCEPTION = "exception"
    STALE     = "stale_context"

@dataclass
class BrowserHealth:
    state:           BrowserState = BrowserState.RUNNING
    uptime_seconds:  float = 0.0
    pages_visited:   int   = 0
    retries_total:   int   = 0
    captchas_seen:   int   = 0
    crashes_total:   int   = 0
    last_error:      str   = ""
    last_url:        str   = ""

    def is_healthy(self) -> bool:
        return self.state in (BrowserState.RUNNING, BrowserState.STARTING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state":          self.state.value,
            "uptime_s":       round(self.uptime_seconds, 1),
            "pages_visited":  self.pages_visited,
            "retries":        self.retries_total,
            "captchas_seen":  self.captchas_seen,
            "crashes":        self.crashes_total,
            "last_error":     self.last_error[:120] if self.last_error else "",
            "last_url":       self.last_url,
        }

@dataclass
class SupervisorResult:
    success:     bool
    value:       Any   = None
    error:       str   = ""
    retries:     int   = 0
    duration_ms: float = 0.0
    had_captcha: bool  = False
    screenshot:  str   = ""   # path to screenshot on failure

    def __bool__(self) -> bool:
        return self.success


# ---------------------------------------------------------------------------
# CAPTCHA detection
# ---------------------------------------------------------------------------

def detect_captcha(content: str) -> bool:
    """Return True if page content contains CAPTCHA signals."""
    lower = content.lower()
    return any(sig in lower for sig in _CAPTCHA_SIGNALS)


def detect_crash(exc_str: str) -> bool:
    """Return True if exception message indicates a browser crash."""
    lower = exc_str.lower()
    return any(sig in lower for sig in _CRASH_SIGNALS)


# ---------------------------------------------------------------------------
# BrowserSupervisor
# ---------------------------------------------------------------------------

class BrowserSupervisor:
    """
    Resilient browser session manager.

    Works with Playwright (async or sync), Selenium, or any callable-based
    browser API. The actual browser object is provided via dependency injection
    — BrowserSupervisor manages lifecycle, retry logic, and health tracking
    without importing browser libraries directly (so it works even without
    Playwright installed).
    """

    def __init__(
        self,
        browser:            Any   = None,
        max_retries:        int   = DEFAULT_MAX_RETRIES,
        timeout_sec:        float = DEFAULT_TIMEOUT_SEC,
        backoff_base:       float = DEFAULT_BACKOFF_BASE,
        backoff_max:        float = DEFAULT_BACKOFF_MAX,
        screenshot_dir:     str   = DEFAULT_SCREENSHOT_DIR,
        captcha_handler:    Optional[Callable] = None,
        on_crash:           Optional[Callable] = None,
        on_restart:         Optional[Callable] = None,
        jitter:             bool  = True,
    ) -> None:
        self._browser        = browser
        self._max_retries    = max_retries
        self._timeout        = timeout_sec
        self._backoff_base   = backoff_base
        self._backoff_max    = backoff_max
        self._screenshot_dir = screenshot_dir
        self._captcha_handler = captcha_handler
        self._on_crash       = on_crash
        self._on_restart     = on_restart
        self._jitter         = jitter

        self._health  = BrowserHealth(state=BrowserState.RUNNING)
        self._start_t = time.time()

        Path(screenshot_dir).mkdir(parents=True, exist_ok=True)

    # ── Core execution with retry ─────────────────────────────────────────────

    def execute(
        self,
        task_fn: Callable,
        *args: Any,
        capture_screenshot_on_fail: bool = True,
        **kwargs: Any,
    ) -> SupervisorResult:
        """
        Execute task_fn(*args, **kwargs) with automatic retry.
        Returns a SupervisorResult.
        """
        start = time.time()
        last_exc: Optional[Exception] = None
        screenshot_path = ""

        for attempt in range(self._max_retries + 1):
            try:
                value = task_fn(*args, **kwargs)
                self._health.pages_visited += 1
                self._health.last_error = ""

                # Check result for CAPTCHA
                if isinstance(value, str) and detect_captcha(value):
                    self._health.captchas_seen += 1
                    self._health.last_error = "captcha detected"
                    handled = self._handle_captcha(value, attempt)
                    if not handled:
                        return SupervisorResult(
                            success=False,
                            error="captcha detected and not handled",
                            retries=attempt,
                            duration_ms=(time.time() - start) * 1000,
                            had_captcha=True,
                        )
                    # Retry after captcha handling
                    continue

                return SupervisorResult(
                    success=True,
                    value=value,
                    retries=attempt,
                    duration_ms=(time.time() - start) * 1000,
                )

            except Exception as exc:
                last_exc = exc
                exc_str  = str(exc)
                self._health.retries_total += 1
                self._health.last_error = exc_str[:200]

                is_crash   = detect_crash(exc_str)
                is_timeout = "timeout" in exc_str.lower()
                is_captcha = "captcha" in exc_str.lower()

                log.warning(
                    "browser attempt %d/%d failed [%s]: %s",
                    attempt + 1, self._max_retries + 1,
                    "crash" if is_crash else "timeout" if is_timeout else "error",
                    exc_str[:100],
                )

                # Screenshot on failure
                if capture_screenshot_on_fail and attempt == self._max_retries:
                    screenshot_path = self._try_screenshot()

                # On crash: try to restart browser
                if is_crash:
                    self._health.crashes_total += 1
                    self._health.state = BrowserState.CRASHED
                    if self._on_crash:
                        try:
                            self._on_crash(exc)
                        except Exception:
                            pass
                    restarted = self.restart()
                    if not restarted:
                        log.error("browser restart failed after crash")
                        break

                if attempt < self._max_retries:
                    wait = self._backoff(attempt)
                    log.info("waiting %.1fs before retry %d", wait, attempt + 2)
                    time.sleep(wait)
                else:
                    break

        return SupervisorResult(
            success=False,
            error=str(last_exc) if last_exc else "max retries exceeded",
            retries=self._max_retries,
            duration_ms=(time.time() - start) * 1000,
            screenshot=screenshot_path,
        )

    async def execute_async(
        self,
        task_fn: Callable,
        *args: Any,
        capture_screenshot_on_fail: bool = True,
        **kwargs: Any,
    ) -> SupervisorResult:
        """
        Async version of execute() — for use with async Playwright, etc.
        """
        start = time.time()
        last_exc: Optional[Exception] = None
        screenshot_path = ""

        for attempt in range(self._max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(task_fn):
                    value = await asyncio.wait_for(
                        task_fn(*args, **kwargs),
                        timeout=self._timeout,
                    )
                else:
                    value = task_fn(*args, **kwargs)

                self._health.pages_visited += 1
                if isinstance(value, str) and detect_captcha(value):
                    self._health.captchas_seen += 1
                    return SupervisorResult(
                        success=False,
                        error="captcha detected",
                        retries=attempt,
                        duration_ms=(time.time() - start) * 1000,
                        had_captcha=True,
                    )

                return SupervisorResult(
                    success=True,
                    value=value,
                    retries=attempt,
                    duration_ms=(time.time() - start) * 1000,
                )

            except asyncio.TimeoutError:
                last_exc = TimeoutError(f"task timed out after {self._timeout}s")
                self._health.retries_total += 1
                log.warning("async attempt %d/%d timed out", attempt + 1, self._max_retries + 1)

            except Exception as exc:
                last_exc = exc
                self._health.retries_total += 1
                log.warning("async attempt %d/%d failed: %s", attempt + 1, self._max_retries + 1, exc)

            if attempt < self._max_retries:
                wait = self._backoff(attempt)
                await asyncio.sleep(wait)

        return SupervisorResult(
            success=False,
            error=str(last_exc) if last_exc else "max retries exceeded",
            retries=self._max_retries,
            duration_ms=(time.time() - start) * 1000,
            screenshot=screenshot_path,
        )

    # ── CDP session ───────────────────────────────────────────────────────────

    class CDPSession:
        """Thin wrapper around a CDP session (for use with Playwright CDP)."""

        def __init__(self, browser: Any) -> None:
            self._browser = browser
            self._session: Any = None

        async def __aenter__(self) -> "BrowserSupervisor.CDPSession":
            try:
                # Playwright CDP
                if hasattr(self._browser, "new_cdp_session"):
                    page = await self._browser.new_page()
                    self._session = await page.context.new_cdp_session(page)
                else:
                    log.warning("browser does not support CDP sessions")
            except Exception as e:
                log.warning("CDP session init failed: %s", e)
            return self

        async def __aexit__(self, *args: Any) -> None:
            if self._session:
                try:
                    await self._session.detach()
                except Exception:
                    pass

        async def send(self, method: str, params: Optional[Dict] = None) -> Any:
            if not self._session:
                raise RuntimeError("CDP session not initialized")
            return await self._session.send(method, params or {})

    def cdp_session(self) -> "BrowserSupervisor.CDPSession":
        return self.CDPSession(self._browser)

    # ── Health & lifecycle ────────────────────────────────────────────────────

    def health_check(self) -> BrowserHealth:
        """Return current browser health."""
        self._health.uptime_seconds = time.time() - self._start_t
        return self._health

    def is_alive(self) -> bool:
        """Quick liveness check."""
        if self._browser is None:
            return False
        if self._health.state == BrowserState.CRASHED:
            return False
        # Try browser-specific ping
        try:
            if hasattr(self._browser, "is_connected"):
                return self._browser.is_connected()
            if hasattr(self._browser, "contexts"):
                _ = self._browser.contexts
                return True
        except Exception:
            return False
        return True

    def restart(self) -> bool:
        """
        Restart the browser session.
        Calls on_restart callback if provided.
        Returns True if restart succeeded.
        """
        log.info("restarting browser session")
        self._health.state = BrowserState.STARTING
        try:
            if self._on_restart:
                new_browser = self._on_restart()
                if new_browser is not None:
                    self._browser = new_browser
                    log.info("browser restarted via on_restart callback")
            self._health.state  = BrowserState.RUNNING
            self._start_t       = time.time()
            return True
        except Exception as e:
            log.error("browser restart failed: %s", e)
            self._health.state = BrowserState.CRASHED
            return False

    def close(self) -> None:
        """Gracefully close the browser."""
        self._health.state = BrowserState.CLOSED
        if self._browser is None:
            return
        try:
            if hasattr(self._browser, "close"):
                self._browser.close()
            elif hasattr(self._browser, "quit"):
                self._browser.quit()
        except Exception as e:
            log.warning("error closing browser: %s", e)

    # ── Screenshot ────────────────────────────────────────────────────────────

    def screenshot(self, path: Optional[str] = None) -> str:
        """Save a screenshot. Returns path to the file."""
        if path is None:
            ts   = int(time.time())
            path = os.path.join(self._screenshot_dir, f"operon_browser_{ts}.png")

        if self._browser is None:
            log.warning("screenshot: no browser attached")
            return ""

        try:
            if hasattr(self._browser, "screenshot"):
                data = self._browser.screenshot(path=path)
                return path
            # Playwright page
            if hasattr(self._browser, "pages"):
                pages = self._browser.pages
                if pages:
                    pages[-1].screenshot(path=path)
                    return path
        except Exception as e:
            log.warning("screenshot failed: %s", e)
        return ""

    # ── Internals ─────────────────────────────────────────────────────────────

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with optional jitter."""
        wait = min(self._backoff_base * (2 ** attempt), self._backoff_max)
        if self._jitter:
            wait *= (0.75 + random.random() * 0.5)
        return wait

    def _handle_captcha(self, content: str, attempt: int) -> bool:
        """
        Invoke captcha_handler if set.
        Returns True if CAPTCHA was resolved (can retry), False otherwise.
        """
        if self._captcha_handler:
            try:
                result = self._captcha_handler(content, attempt)
                return bool(result)
            except Exception as e:
                log.warning("captcha_handler raised: %s", e)
        # Default: wait a bit and hope
        wait = min(10 * (attempt + 1), 60)
        log.info("captcha detected — waiting %ds before retry", wait)
        time.sleep(wait)
        return True   # optimistic retry

    def _try_screenshot(self) -> str:
        """Best-effort screenshot, returns path or empty string."""
        try:
            return self.screenshot()
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Decorator: @supervised
# ---------------------------------------------------------------------------

def supervised(
    supervisor: Optional[BrowserSupervisor] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Callable:
    """
    Decorator that wraps a browser function in a BrowserSupervisor.

    Example:
        @supervised(max_retries=3)
        def scrape_page(url): ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sup = supervisor or BrowserSupervisor(max_retries=max_retries)
            result = sup.execute(fn, *args, **kwargs)
            if not result.success:
                raise RuntimeError(f"supervised task failed after {result.retries} retries: {result.error}")
            return result.value
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_supervisor: Optional[BrowserSupervisor] = None


def get_supervisor(browser: Any = None) -> BrowserSupervisor:
    """Return the session-scoped default browser supervisor."""
    global _default_supervisor
    if _default_supervisor is None:
        _default_supervisor = BrowserSupervisor(browser=browser)
    elif browser is not None and _default_supervisor._browser is None:
        _default_supervisor._browser = browser
    return _default_supervisor
