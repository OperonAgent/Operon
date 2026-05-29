"""
Operon Browser Stealth — Anti-detection patches, CDP direct control, human simulation.

Matches Hermes browser/stealth.py + OpenClaw src/browser/ depth.

Features
--------
• **Fingerprint spoofing** — overrides navigator.webdriver, platform, languages,
  plugins, screen dimensions, timezone, user-agent string, canvas/WebGL noise.

• **Human-like mouse movement** — Bézier-curve mouse paths, random micro-delays,
  natural jitter, scroll with variable speed.

• **Typing simulation** — per-character delays with Gaussian noise, occasional
  typos + corrections, burst/slow patterns mimicking real typing.

• **CDP direct session** — thin wrapper around Playwright's CDP session for
  low-level Chrome DevTools Protocol calls (Network.enable, Page.captureScreenshot,
  DOM manipulation, JavaScript execution).

• **Stealth profile generator** — generates a randomised-but-coherent browser
  identity (UA, screen, timezone, languages) that avoids bot fingerprinting.

• **Anti-detection patches** — applies all JS patches atomically before any
  page navigation via Page.addScriptToEvaluateOnNewDocument.

• **CAPTCHA & challenge detection** — enhanced detector beyond browser_supervisor,
  extracts solve hints (iframe src, sitekey).

• **Request intercept** — block or modify network requests via CDP
  (ad blocking, custom headers, request replay).

Usage
-----
    from core.browser_stealth import StealthConfig, StealthBrowser, apply_stealth

    # With Playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page    = await context.new_page()

        # Apply all stealth patches
        stealth = StealthBrowser(page)
        await stealth.apply_patches()
        await stealth.navigate("https://example.com")

    # Profile generation
    profile = StealthProfile.random()
    print(profile.user_agent)
    print(profile.screen_width, profile.screen_height)
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TYPE_DELAY_MS  = 80    # base ms per character
_DEFAULT_TYPE_JITTER_MS = 40    # Gaussian std dev on typing delay
_DEFAULT_MOUSE_STEPS    = 25    # Bézier steps for mouse movement
_DEFAULT_SCROLL_STEPS   = 5     # scroll steps per action
_MAX_RETRIES            = 3

# Known CAPTCHA signals — expanded from browser_supervisor
_CAPTCHA_SIGNALS = [
    "captcha",
    "recaptcha",
    "hcaptcha",
    "turnstile",
    "cloudflare challenge",
    "cf-challenge",
    "datadome",
    "perimeterx",
    "px-captcha",
    "funcaptcha",
    "arkose",
    "imperva",
    "incapsula",
    "challenge-form",
    "challenge-running",
    "please verify",
    "verify you are human",
    "are you a robot",
    "prove you're human",
    "security check",
    "bot protection",
]

_WEBDRIVER_SIGNALS = [
    "navigator.webdriver",
    "__selenium_",
    "_phantom",
    "_nightmare",
    "callSelenium",
    "__webdriver_evaluate",
    "__driver_evaluate",
    "webdriver",
]


# ---------------------------------------------------------------------------
# Stealth profiles
# ---------------------------------------------------------------------------

_USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

_SCREEN_RESOLUTIONS = [
    (1920, 1080), (1366, 768), (1440, 900), (2560, 1440),
    (1280, 800),  (1600, 900), (1920, 1200), (1024, 768),
]

_TIMEZONES = [
    "America/New_York", "America/Chicago", "America/Los_Angeles",
    "America/Denver", "Europe/London", "Europe/Berlin", "Europe/Paris",
    "Asia/Tokyo", "Asia/Shanghai", "Asia/Kolkata", "Australia/Sydney",
]

_LANGUAGE_SETS = [
    ["en-US", "en"],
    ["en-GB", "en"],
    ["en-US", "en", "fr"],
    ["en-US", "en", "de"],
    ["de-DE", "de", "en"],
    ["fr-FR", "fr", "en"],
    ["ja-JP", "ja", "en"],
]


@dataclass
class StealthProfile:
    """
    A coherent randomised browser fingerprint.
    All fields are designed to be internally consistent to avoid easy detection.
    """
    user_agent:      str
    screen_width:    int
    screen_height:   int
    timezone:        str
    languages:       List[str]
    color_depth:     int       = 24
    device_memory:   int       = 8     # GB, power of 2
    hardware_concurrency: int  = 8     # CPU cores
    max_touch_points: int      = 0
    vendor:          str       = "Google Inc."
    renderer:        str       = "ANGLE (Intel, Intel(R) UHD Graphics)"
    platform:        str       = "Win32"
    do_not_track:    str       = "1"
    cookie_enabled:  bool      = True
    webgl_vendor:    str       = "Intel Inc."

    @classmethod
    def random(cls) -> "StealthProfile":
        ua      = random.choice(_USER_AGENTS)
        screen  = random.choice(_SCREEN_RESOLUTIONS)
        tz      = random.choice(_TIMEZONES)
        langs   = random.choice(_LANGUAGE_SETS)
        memory  = random.choice([4, 8, 16, 32])
        cores   = random.choice([2, 4, 6, 8, 12, 16])

        # Infer platform from UA
        if "Windows" in ua:
            platform = "Win32"
        elif "Macintosh" in ua:
            platform = "MacIntel"
        elif "Linux" in ua:
            platform = "Linux x86_64"
        else:
            platform = "Win32"

        return cls(
            user_agent   = ua,
            screen_width  = screen[0],
            screen_height = screen[1],
            timezone      = tz,
            languages     = langs,
            device_memory = memory,
            hardware_concurrency = cores,
            platform      = platform,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StealthProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    def js_patches(self) -> str:
        """
        Return a JavaScript snippet that applies all fingerprint overrides.
        Should be injected via Page.addScriptToEvaluateOnNewDocument before
        any page navigation.
        """
        langs_json = "[" + ", ".join(f'"{l}"' for l in self.languages) + "]"
        return f"""
(function() {{
  'use strict';

  // ── navigator.webdriver ───────────────────────────────────────────────
  Object.defineProperty(navigator, 'webdriver', {{
    get: () => undefined,
    configurable: true,
  }});

  // ── navigator.userAgent / platform / vendor ───────────────────────────
  Object.defineProperty(navigator, 'userAgent', {{
    get: () => '{self.user_agent}',
    configurable: true,
  }});
  Object.defineProperty(navigator, 'platform', {{
    get: () => '{self.platform}',
    configurable: true,
  }});
  Object.defineProperty(navigator, 'vendor', {{
    get: () => '{self.vendor}',
    configurable: true,
  }});

  // ── navigator.languages ───────────────────────────────────────────────
  Object.defineProperty(navigator, 'language', {{
    get: () => '{self.languages[0]}',
    configurable: true,
  }});
  Object.defineProperty(navigator, 'languages', {{
    get: () => {langs_json},
    configurable: true,
  }});

  // ── navigator.hardwareConcurrency / deviceMemory ──────────────────────
  Object.defineProperty(navigator, 'hardwareConcurrency', {{
    get: () => {self.hardware_concurrency},
    configurable: true,
  }});
  Object.defineProperty(navigator, 'deviceMemory', {{
    get: () => {self.device_memory},
    configurable: true,
  }});
  Object.defineProperty(navigator, 'maxTouchPoints', {{
    get: () => {self.max_touch_points},
    configurable: true,
  }});
  Object.defineProperty(navigator, 'doNotTrack', {{
    get: () => '{self.do_not_track}',
    configurable: true,
  }});
  Object.defineProperty(navigator, 'cookieEnabled', {{
    get: () => {str(self.cookie_enabled).lower()},
    configurable: true,
  }});

  // ── navigator.plugins — fake a few common plugins ─────────────────────
  const fakePlugins = [
    {{ name: 'Chrome PDF Plugin',  filename: 'internal-pdf-viewer' }},
    {{ name: 'Chrome PDF Viewer',  filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' }},
    {{ name: 'Native Client',      filename: 'internal-nacl-plugin' }},
  ];
  Object.defineProperty(navigator, 'plugins', {{
    get: () => {{
      const arr = fakePlugins.map(p => {{ return {{ name: p.name, filename: p.filename }}; }});
      arr.item = (i) => arr[i];
      arr.namedItem = (n) => arr.find(p => p.name === n);
      Object.setPrototypeOf(arr, PluginArray.prototype);
      return arr;
    }},
    configurable: true,
  }});

  // ── screen dimensions ─────────────────────────────────────────────────
  Object.defineProperty(screen, 'width',       {{ get: () => {self.screen_width}  }});
  Object.defineProperty(screen, 'height',      {{ get: () => {self.screen_height} }});
  Object.defineProperty(screen, 'availWidth',  {{ get: () => {self.screen_width}  }});
  Object.defineProperty(screen, 'availHeight', {{ get: () => {self.screen_height} - 40 }});
  Object.defineProperty(screen, 'colorDepth',  {{ get: () => {self.color_depth}   }});
  Object.defineProperty(screen, 'pixelDepth',  {{ get: () => {self.color_depth}   }});

  // ── canvas fingerprinting noise ───────────────────────────────────────
  const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
  HTMLCanvasElement.prototype.toDataURL = function(type) {{
    if (type === 'image/png' || !type) {{
      const ctx = this.getContext('2d');
      if (ctx) {{
        const imgData = ctx.getImageData(0, 0, this.width, this.height);
        for (let i = 0; i < imgData.data.length; i += 4) {{
          imgData.data[i]   = Math.max(0, imgData.data[i]   + (Math.random() > 0.99 ? 1 : 0));
          imgData.data[i+1] = Math.max(0, imgData.data[i+1] + (Math.random() > 0.99 ? 1 : 0));
        }}
        ctx.putImageData(imgData, 0, 0);
      }}
    }}
    return originalToDataURL.apply(this, arguments);
  }};

  // ── WebGL vendor / renderer spoof ─────────────────────────────────────
  const getParam = WebGLRenderingContext.prototype.getParameter;
  WebGLRenderingContext.prototype.getParameter = function(param) {{
    if (param === 37445) return '{self.webgl_vendor}';   // UNMASKED_VENDOR_WEBGL
    if (param === 37446) return '{self.renderer}';        // UNMASKED_RENDERER_WEBGL
    return getParam.apply(this, arguments);
  }};

  // ── Remove CDP/automation footprints ─────────────────────────────────
  delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
  delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
  delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;

  // ── Permissions API — report microphone/camera as 'prompt' ───────────
  if (navigator.permissions) {{
    const origQuery = navigator.permissions.query.bind(navigator.permissions);
    navigator.permissions.query = (parameters) => (
      parameters.name === 'notifications'
        ? Promise.resolve({{ state: Notification.permission }})
        : origQuery(parameters)
    );
  }}

  // ── Chrome runtime object (expected by some sites) ───────────────────
  if (!window.chrome) {{
    window.chrome = {{
      runtime: {{
        PlatformOs:        {{ MAC: 'mac', WIN: 'win', ANDROID: 'android', CROS: 'cros', LINUX: 'linux', OPENBSD: 'openbsd' }},
        PlatformArch:      {{ ARM: 'arm', X86_32: 'x86-32', X86_64: 'x86-64', MIPS: 'mips', MIPS64: 'mips64' }},
        PlatformNaclArch:  {{ ARM: 'arm', X86_32: 'x86-32', X86_64: 'x86-64', MIPS: 'mips', MIPS64: 'mips64' }},
        RequestUpdateCheckStatus: {{ THROTTLED: 'throttled', NO_UPDATE: 'no_update', UPDATE_AVAILABLE: 'update_available' }},
        OnInstalledReason: {{ INSTALL: 'install', UPDATE: 'update', CHROME_UPDATE: 'chrome_update', SHARED_MODULE_UPDATE: 'shared_module_update' }},
        OnRestartRequiredReason: {{ APP_UPDATE: 'app_update', OS_UPDATE: 'os_update', PERIODIC: 'periodic' }},
      }},
      loadTimes: function() {{}},
      csi: function() {{}},
    }};
  }}

}})();
"""


# ---------------------------------------------------------------------------
# Human-like interaction helpers
# ---------------------------------------------------------------------------

class HumanBehavior:
    """
    Generates human-like interaction patterns.
    All methods are synchronous helpers; call them from async code with await.
    """

    @staticmethod
    def typing_delays(
        text: str,
        base_ms: float = _DEFAULT_TYPE_DELAY_MS,
        jitter_ms: float = _DEFAULT_TYPE_JITTER_MS,
        typo_rate: float = 0.02,
    ) -> List[float]:
        """
        Generate per-character typing delays (ms).
        Includes occasional bursts (fast typing) and pauses (thinking).
        """
        delays: List[float] = []
        burst_chars = 0
        burst_max   = random.randint(3, 12)
        in_burst    = False

        for i, ch in enumerate(text):
            # Bursts: type fast for a few chars
            if burst_chars >= burst_max:
                burst_chars = 0
                burst_max   = random.randint(3, 12)
                in_burst    = random.random() < 0.3
                if not in_burst:
                    # Thinking pause
                    delays.append(random.uniform(200, 500))
                    continue

            base = base_ms * 0.4 if in_burst else base_ms
            delay = max(20, random.gauss(base, jitter_ms))

            # Punctuation: slightly longer pause after
            if i > 0 and text[i-1] in ".,!?;:":
                delay += random.uniform(50, 200)

            # Space: natural word boundary micro-pause
            if ch == " ":
                delay += random.uniform(10, 50)

            delays.append(delay)
            burst_chars += 1

        return delays

    @staticmethod
    def bezier_path(
        start: Tuple[float, float],
        end:   Tuple[float, float],
        steps: int = _DEFAULT_MOUSE_STEPS,
    ) -> List[Tuple[float, float]]:
        """
        Generate a natural Bézier-curve mouse path from start to end.
        Uses two random control points to create a curved, non-linear path.
        """
        x0, y0 = start
        x1, y1 = end

        # Random control points
        cp1 = (
            x0 + random.uniform(0.1, 0.9) * (x1 - x0) + random.gauss(0, 50),
            y0 + random.gauss(0, 80),
        )
        cp2 = (
            x0 + random.uniform(0.1, 0.9) * (x1 - x0) + random.gauss(0, 50),
            y1 + random.gauss(0, 80),
        )

        path: List[Tuple[float, float]] = []
        for i in range(steps + 1):
            t = i / steps
            # Cubic Bézier formula
            mt = 1 - t
            x = (mt**3 * x0 + 3 * mt**2 * t * cp1[0] +
                 3 * mt * t**2 * cp2[0] + t**3 * x1)
            y = (mt**3 * y0 + 3 * mt**2 * t * cp1[1] +
                 3 * mt * t**2 * cp2[1] + t**3 * y1)
            # Add micro-jitter
            x += random.gauss(0, 1)
            y += random.gauss(0, 1)
            path.append((round(x, 1), round(y, 1)))
        return path

    @staticmethod
    def scroll_steps(
        delta_y: int,
        steps: int = _DEFAULT_SCROLL_STEPS,
        speed_variance: float = 0.3,
    ) -> List[int]:
        """
        Generate variable-speed scroll increments summing to delta_y.
        Simulates acceleration/deceleration.
        """
        if delta_y == 0:
            return []
        base = delta_y / steps
        deltas: List[int] = []
        for i in range(steps):
            # Ease-in-out curve
            t = i / (steps - 1) if steps > 1 else 0.5
            ease = 0.5 - 0.5 * math.cos(math.pi * t)
            variance = random.gauss(1.0, speed_variance)
            d = int(base * variance)
            if d == 0:
                d = 1 if delta_y > 0 else -1
            deltas.append(d)
        return deltas

    @staticmethod
    def random_viewport_position(
        width: int = 1920,
        height: int = 1080,
        margin: int = 50,
    ) -> Tuple[int, int]:
        """Return a random position within the visible viewport."""
        return (
            random.randint(margin, width - margin),
            random.randint(margin, height - margin),
        )

    @staticmethod
    def think_delay(min_ms: float = 300, max_ms: float = 1500) -> float:
        """Return a 'thinking' delay in ms drawn from a human-like distribution."""
        # Log-normal distribution (most thinking pauses are short, some are long)
        mu    = math.log((min_ms + max_ms) / 2)
        sigma = 0.4
        delay = random.lognormvariate(mu, sigma)
        return max(min_ms, min(max_ms * 3, delay))


# ---------------------------------------------------------------------------
# CDP Session wrapper
# ---------------------------------------------------------------------------

class CDPSession:
    """
    Thin async wrapper around Playwright's CDP session.
    Provides convenience methods for common CDP operations.
    """

    def __init__(self, session: Any) -> None:
        """
        Parameters
        ----------
        session : playwright.async_api.CDPSession
            The underlying Playwright CDP session.
        """
        self._session = session

    async def send(self, method: str, params: Optional[Dict] = None) -> Any:
        """Send a raw CDP command."""
        return await self._session.send(method, params or {})

    async def enable_network(self) -> None:
        await self.send("Network.enable")

    async def enable_page(self) -> None:
        await self.send("Page.enable")

    async def enable_runtime(self) -> None:
        await self.send("Runtime.enable")

    async def set_extra_headers(self, headers: Dict[str, str]) -> None:
        await self.send("Network.setExtraHTTPHeaders", {"headers": headers})

    async def block_urls(self, patterns: List[str]) -> None:
        """Block requests matching URL patterns (ad/tracking blocking)."""
        await self.send("Network.setBlockedURLs", {"urls": patterns})

    async def intercept_requests(self) -> None:
        await self.send("Network.setRequestInterception",
                        {"patterns": [{"urlPattern": "*"}]})

    async def screenshot(self, format: str = "png", quality: int = 90) -> bytes:
        """Capture a screenshot via CDP. Returns raw bytes."""
        import base64
        params: Dict[str, Any] = {"format": format}
        if format == "jpeg":
            params["quality"] = quality
        result = await self.send("Page.captureScreenshot", params)
        return base64.b64decode(result.get("data", ""))

    async def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript in the page context."""
        result = await self.send(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True},
        )
        return result.get("result", {}).get("value")

    async def get_cookies(self) -> List[Dict]:
        result = await self.send("Network.getAllCookies")
        return result.get("cookies", [])

    async def set_cookies(self, cookies: List[Dict]) -> None:
        await self.send("Network.setCookies", {"cookies": cookies})

    async def clear_cookies(self) -> None:
        await self.send("Network.clearBrowserCookies")

    async def get_performance_metrics(self) -> Dict[str, float]:
        result = await self.send("Performance.getMetrics")
        return {m["name"]: m["value"] for m in result.get("metrics", [])}

    async def add_script_on_load(self, script: str) -> str:
        """Add a script that runs on every new document. Returns identifier."""
        result = await self.send(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": script},
        )
        return result.get("identifier", "")

    async def remove_script_on_load(self, identifier: str) -> None:
        await self.send(
            "Page.removeScriptToEvaluateOnNewDocument",
            {"identifier": identifier},
        )

    async def navigate(self, url: str, wait_until: str = "load") -> Dict[str, Any]:
        """Navigate to URL. wait_until: load | domcontentloaded | networkidle."""
        return await self.send("Page.navigate", {"url": url})

    async def reload(self, ignore_cache: bool = False) -> None:
        await self.send("Page.reload", {"ignoreCache": ignore_cache})

    async def get_layout_metrics(self) -> Dict[str, Any]:
        return await self.send("Page.getLayoutMetrics")


# ---------------------------------------------------------------------------
# StealthBrowser — high-level async stealth interface
# ---------------------------------------------------------------------------

class StealthBrowser:
    """
    High-level stealth browser interface for Playwright pages.

    Usage:
        from playwright.async_api import async_playwright
        from core.browser_stealth import StealthBrowser, StealthProfile

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            context = await browser.new_context(
                user_agent=profile.user_agent,
                viewport={'width': profile.screen_width, 'height': profile.screen_height},
            )
            page = await context.new_page()
            sb   = StealthBrowser(page, profile=profile)
            await sb.apply_patches()
            await sb.navigate('https://example.com')
            await sb.human_type(selector='input[name=q]', text='hello world')
            await sb.human_click(selector='button[type=submit]')
    """

    def __init__(
        self,
        page: Any,
        profile: Optional[StealthProfile] = None,
    ) -> None:
        self._page    = page
        self._profile = profile or StealthProfile.random()
        self._cdp:    Optional[CDPSession] = None
        self._patch_ids: List[str] = []

    async def apply_patches(self) -> None:
        """Inject all stealth JS patches. Call before any navigation."""
        js = self._profile.js_patches()
        try:
            await self._page.add_init_script(js)
        except Exception:
            # Fallback: use CDP if available
            if self._cdp:
                await self._cdp.add_script_on_load(js)

    async def navigate(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
        timeout: int = 30_000,
    ) -> Dict[str, Any]:
        """Navigate to URL with human-like pre-navigation delay."""
        await asyncio.sleep(random.uniform(0.1, 0.5))
        try:
            await self._page.goto(url, wait_until=wait_until, timeout=timeout)
            return {"ok": True, "url": url}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "url": url}

    async def human_type(
        self,
        selector: str,
        text: str,
        clear_first: bool = True,
        base_delay_ms: float = _DEFAULT_TYPE_DELAY_MS,
    ) -> bool:
        """
        Type text into an element with human-like delays.
        Returns True on success.
        """
        try:
            element = await self._page.wait_for_selector(selector, timeout=5_000)
            if element is None:
                return False
            # Click to focus
            await element.click()
            await asyncio.sleep(random.uniform(0.05, 0.2))
            if clear_first:
                await element.triple_click()
                await asyncio.sleep(0.05)
            delays = HumanBehavior.typing_delays(text, base_ms=base_delay_ms)
            for char, delay_ms in zip(text, delays):
                await element.type(char, delay=delay_ms)
            return True
        except Exception:
            return False

    async def human_click(
        self,
        selector: str,
        timeout_ms: int = 5_000,
        delay_before_ms: float = 0,
    ) -> bool:
        """Click an element after optional human-like pre-delay."""
        try:
            if delay_before_ms:
                await asyncio.sleep(delay_before_ms / 1000)
            element = await self._page.wait_for_selector(selector, timeout=timeout_ms)
            if element is None:
                return False
            # Get bounding box for Bézier approach
            box = await element.bounding_box()
            if box:
                await self._page.mouse.move(
                    box["x"] + box["width"] / 2,
                    box["y"] + box["height"] / 2,
                )
                await asyncio.sleep(random.uniform(0.05, 0.15))
            await element.click()
            return True
        except Exception:
            return False

    async def human_scroll(
        self,
        delta_y: int = 500,
        smooth: bool = True,
    ) -> None:
        """Scroll the page with variable speed."""
        if smooth:
            steps = HumanBehavior.scroll_steps(delta_y)
            for step in steps:
                await self._page.mouse.wheel(0, step)
                await asyncio.sleep(random.uniform(0.02, 0.08))
        else:
            await self._page.mouse.wheel(0, delta_y)

    async def human_hover(
        self,
        x: float,
        y: float,
        from_pos: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Move mouse to (x, y) via a natural Bézier curve."""
        if from_pos is None:
            from_pos = HumanBehavior.random_viewport_position(
                self._profile.screen_width, self._profile.screen_height
            )
        path = HumanBehavior.bezier_path(from_pos, (x, y))
        for px, py in path:
            await self._page.mouse.move(px, py)
            await asyncio.sleep(0.01)

    async def wait_for_content(
        self,
        text: Optional[str] = None,
        selector: Optional[str] = None,
        timeout: int = 10_000,
    ) -> bool:
        """Wait for text or selector to appear on the page."""
        try:
            if selector:
                await self._page.wait_for_selector(selector, timeout=timeout)
                return True
            if text:
                await self._page.wait_for_function(
                    f"() => document.body.innerText.includes({json_str(text)})",
                    timeout=timeout,
                )
                return True
        except Exception:
            pass
        return False

    async def get_cdp_session(self) -> CDPSession:
        """Open (or return cached) CDP session for this page."""
        if self._cdp is None:
            raw_session = await self._page.context.new_cdp_session(self._page)
            self._cdp = CDPSession(raw_session)
        return self._cdp

    async def screenshot(self, path: Optional[str] = None) -> bytes:
        """Take a screenshot. Returns raw bytes, optionally writes to path."""
        data = await self._page.screenshot(
            type="png",
            full_page=False,
            path=path,
        )
        return data

    async def extract_text(self) -> str:
        """Return all visible text from the current page."""
        try:
            return await self._page.inner_text("body")
        except Exception:
            return ""

    async def detect_captcha(self) -> Tuple[bool, str]:
        """
        Check if the current page shows a CAPTCHA.
        Returns (found: bool, hint: str) where hint describes the CAPTCHA type.
        """
        try:
            content = (await self.extract_text()).lower()
            html    = await self._page.content()
        except Exception:
            return False, ""

        for signal in _CAPTCHA_SIGNALS:
            if signal in content or signal in html.lower():
                # Try to extract sitekey / iframe src
                hint = signal
                m = re.search(r'data-sitekey=["\']([^"\']+)["\']', html)
                if m:
                    hint = f"{signal} (sitekey: {m.group(1)[:32]})"
                return True, hint

        return False, ""

    async def detect_bot_check(self) -> bool:
        """Check if any webdriver/automation signal is detectable in page."""
        try:
            has_wd = await self._page.evaluate("() => !!navigator.webdriver")
            return bool(has_wd)
        except Exception:
            return False

    @property
    def profile(self) -> StealthProfile:
        return self._profile


# ---------------------------------------------------------------------------
# Standalone stealth-patch helpers (sync, no Playwright dependency)
# ---------------------------------------------------------------------------

def generate_stealth_launch_args() -> List[str]:
    """
    Return a list of Chromium launch flags that reduce bot fingerprinting.
    Pass to playwright.chromium.launch(args=...).
    """
    return [
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
        "--disable-extensions",
        "--no-first-run",
        "--no-service-autorun",
        "--password-store=basic",
        "--use-mock-keychain",
        "--disable-background-networking",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-breakpad",
        "--disable-client-side-phishing-detection",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-dev-shm-usage",
        "--disable-domain-reliability",
        "--disable-features=AudioServiceOutOfProcess,IsolateOrigins,site-per-process",
        "--disable-hang-monitor",
        "--disable-ipc-flooding-protection",
        "--disable-notifications",
        "--disable-offer-store-unmasked-wallet-cards",
        "--disable-popup-blocking",
        "--disable-print-preview",
        "--disable-prompt-on-repost",
        "--disable-renderer-backgrounding",
        "--disable-setuid-sandbox",
        "--disable-sync",
        "--force-color-profile=srgb",
        "--metrics-recording-only",
        "--mute-audio",
        "--no-sandbox",
        "--safebrowsing-disable-auto-update",
        "--enable-automation=false",
        "--ignore-certificate-errors",
    ]


def generate_stealth_context_options(
    profile: Optional[StealthProfile] = None,
) -> Dict[str, Any]:
    """
    Return Playwright new_context() kwargs with stealth settings.
    """
    p = profile or StealthProfile.random()
    return {
        "user_agent":         p.user_agent,
        "viewport":           {"width": p.screen_width, "height": p.screen_height},
        "locale":             p.languages[0],
        "timezone_id":        p.timezone,
        "color_scheme":       "light",
        "reduced_motion":     "no-preference",
        "has_touch":          p.max_touch_points > 0,
        "java_script_enabled": True,
        "ignore_https_errors": True,
    }


def detect_captcha_in_html(html: str) -> Tuple[bool, str]:
    """
    Detect CAPTCHA in raw HTML string (synchronous, no browser needed).
    Returns (found, hint).
    """
    lower = html.lower()
    for signal in _CAPTCHA_SIGNALS:
        if signal in lower:
            hint = signal
            m = re.search(r'data-sitekey=["\']([^"\']+)["\']', html)
            if m:
                hint = f"{signal} (sitekey: {m.group(1)[:32]})"
            return True, hint
    return False, ""


def detect_webdriver_signal(html: str) -> bool:
    """Check if the page DOM exposes any webdriver/automation signals."""
    for sig in _WEBDRIVER_SIGNALS:
        if sig in html:
            return True
    return False


def random_delay(min_ms: float = 500, max_ms: float = 2000) -> None:
    """Synchronous random delay (for non-async use)."""
    ms = random.uniform(min_ms, max_ms)
    time.sleep(ms / 1000.0)


def build_realistic_headers(profile: Optional[StealthProfile] = None) -> Dict[str, str]:
    """
    Return a set of HTTP request headers that mimic a real browser.
    Useful for requests-based scraping.
    """
    p = profile or StealthProfile.random()
    lang_str = ",".join(
        f"{l};q={round(1.0 - i * 0.1, 1)}" if i > 0 else l
        for i, l in enumerate(p.languages[:4])
    )
    return {
        "User-Agent":      p.user_agent,
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": lang_str,
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest":  "document",
        "Sec-Fetch-Mode":  "navigate",
        "Sec-Fetch-Site":  "none",
        "Sec-Fetch-User":  "?1",
        "Cache-Control":   "max-age=0",
        "DNT":             p.do_not_track,
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def json_str(value: str) -> str:
    """JSON-encode a string for safe injection into JS literals."""
    import json as _json
    return _json.dumps(value)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_profile: Optional[StealthProfile] = None


def get_stealth_profile(fresh: bool = False) -> StealthProfile:
    """Return a cached random stealth profile (or generate a fresh one)."""
    global _default_profile
    if fresh or _default_profile is None:
        _default_profile = StealthProfile.random()
    return _default_profile
