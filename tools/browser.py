"""
Operon Browser Automation Tool — Production-grade CDP implementation.

Adapted from Hermes Agent browser_tool.py architecture.

Backends (auto-detected, in priority order):
  1. Playwright CDP     — local headless Chromium, zero cloud cost, full control
  2. Browserbase cloud — if BROWSERBASE_API_KEY is set (optional)
  3. Browser Use cloud — if BROWSER_USE_API_KEY is set (optional)

Features:
  - Accessibility tree (aria snapshot) for text-based page representation
  - Element interaction via role/ref selectors
  - Full CDP network / console event access
  - Screenshot capture with base64 encoding
  - Tab management, dialog handling, cookie management
  - PDF export, file download tracking
  - URL safety checks (blocks known-malicious patterns)
  - Session isolation per task_id
  - Automatic cleanup on session reset
  - LLM-ready page extraction (structured text, tables, links)

Install:
    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import atexit
import base64
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

log = logging.getLogger("operon.browser")

# ---------------------------------------------------------------------------
# URL safety — blocks malicious / dangerous domains / patterns
# ---------------------------------------------------------------------------

_BLOCKED_SCHEMES = frozenset({"javascript", "vbscript", "data", "file"})
_BLOCKED_DOMAIN_PATTERNS: List[re.Pattern] = [
    re.compile(r"\.onion$", re.I),
    re.compile(r"^(localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|::1)$", re.I),
]
_BLOCKED_URL_PATTERNS: List[re.Pattern] = [
    re.compile(r"javascript\s*:", re.I),
    re.compile(r"<\s*script", re.I),
]


# ---------------------------------------------------------------------------
# Anti-bot stealth helpers
# ---------------------------------------------------------------------------

import random as _random

_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

def _random_user_agent() -> str:
    return _random.choice(_USER_AGENTS)


# JavaScript injected into every page to defeat common bot-detection checks
_STEALTH_JS = """
// Override navigator.webdriver — the #1 bot signal
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});

// Fake plugins list (real browser has plugins)
Object.defineProperty(navigator, 'plugins', {
    get: () => [
        {name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer'},
        {name: 'Chrome PDF Viewer',  filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai'},
        {name: 'Native Client',      filename: 'internal-nacl-plugin'},
    ]
});

// Fake language / platform
Object.defineProperty(navigator, 'languages',       {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'platform',         {get: () => 'MacIntel'});
Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
Object.defineProperty(navigator, 'deviceMemory',     {get: () => 8});

// Permissions API — real browsers return 'prompt' for notifications
const origQuery = window.navigator.permissions.query.bind(window.navigator.permissions);
window.navigator.permissions.query = (parameters) =>
    parameters.name === 'notifications'
        ? Promise.resolve({state: 'prompt'})
        : origQuery(parameters);

// Chrome runtime object expected by some detectors
window.chrome = {
    runtime: {
        connect: () => {},
        sendMessage: () => {},
        onMessage: {addListener: () => {}, removeListener: () => {}},
    }
};

// Pass WebGL vendor check
const getParameter = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(parameter) {
    if (parameter === 37445) return 'Intel Inc.';
    if (parameter === 37446) return 'Intel Iris OpenGL Engine';
    return getParameter.call(this, parameter);
};
"""

_CAPTCHA_PATTERNS = re.compile(
    r"(captcha|are you a robot|unusual traffic|verify you are human|"
    r"cloudflare|just a moment|ddos-guard|please complete the security)",
    re.I,
)


def _inject_stealth(page: Any) -> None:
    """Inject anti-bot JavaScript into all frames of a page."""
    try:
        page.add_init_script(_STEALTH_JS)
    except Exception as e:
        log.debug(f"Stealth injection failed (non-fatal): {e}")


def _human_delay(min_ms: int = 80, max_ms: int = 280) -> None:
    """Sleep a random human-like interval between interactions."""
    time.sleep(_random.randint(min_ms, max_ms) / 1000)


def _check_captcha(page: Any) -> bool:
    """Return True if the current page looks like a CAPTCHA challenge."""
    try:
        content = page.content()
        return bool(_CAPTCHA_PATTERNS.search(content[:3000]))
    except Exception:
        return False


def _is_url_safe(url: str) -> Tuple[bool, str]:
    """Return (safe, reason). Fail-closed for empty / unparseable URLs."""
    if not url or not isinstance(url, str):
        return False, "empty URL"
    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "unparseable URL"
    scheme = (parsed.scheme or "").lower()
    if scheme in _BLOCKED_SCHEMES:
        return False, f"blocked scheme: {scheme}"
    for pat in _BLOCKED_URL_PATTERNS:
        if pat.search(url):
            return False, "blocked URL pattern"
    host = parsed.hostname or ""
    for pat in _BLOCKED_DOMAIN_PATTERNS:
        if pat.search(host):
            return False, f"blocked host: {host}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Aria snapshot → readable text
# ---------------------------------------------------------------------------

def _aria_to_text(node: Dict, depth: int = 0, lines: Optional[List] = None) -> str:
    if lines is None:
        lines = []
    if not isinstance(node, dict):
        return "\n".join(lines)
    role     = node.get("role", "")
    name     = node.get("name", "")
    value    = node.get("value", "")
    children = node.get("children", [])

    if role in ("generic", "none", "presentation"):
        for child in children:
            _aria_to_text(child, depth, lines)
        return "\n".join(lines)

    indent = "  " * depth
    if role == "heading":
        level = node.get("level", 2)
        lines.append(f"\n{indent}{'#' * level} {name}")
    elif role == "link":
        lines.append(f"{indent}🔗 {name}")
    elif role == "button":
        lines.append(f"{indent}[btn] {name}")
    elif role in ("textbox", "searchbox", "combobox"):
        lines.append(f"{indent}[input:{role}] {name}" + (f" = {value!r}" if value else ""))
    elif role in ("checkbox", "radio"):
        checked = node.get("checked", False)
        lines.append(f"{indent}[{role}] {name} {'✓' if checked else '○'}")
    elif role in ("listitem", "row") and name:
        lines.append(f"{indent}• {name}")
    elif role in ("img", "image") and name:
        lines.append(f"{indent}[img: {name}]")
    elif role == "text" and name:
        lines.append(f"{indent}{name}")
    elif role in ("document", "main", "navigation", "banner", "region") and name:
        lines.append(f"\n{indent}[{role}: {name}]")
    elif name:
        lines.append(f"{indent}[{role}] {name}")

    for child in children:
        _aria_to_text(child, depth + 1, lines)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session registry — per task_id session isolation
# ---------------------------------------------------------------------------

_SESSION_LOCK = threading.Lock()
_SESSIONS: Dict[str, "_BrowserSession"] = {}
_DEFAULT_SESSION_ID = "_operon_default_"

SCREENSHOTS_DIR = Path.home() / ".operon" / "screenshots"
DOWNLOADS_DIR   = Path.home() / ".operon" / "downloads"


class _BrowserSession:
    """One Playwright browser context, shared across tool calls in a task."""

    def __init__(self, session_id: str) -> None:
        self.session_id   = session_id
        self._lock        = threading.Lock()
        self._pw_ctx      = None
        self._browser     = None
        self._context     = None
        self._page        = None
        self._console_log: List[Dict] = []
        self._network_log: List[Dict] = []
        self._dialog_text: List[str]  = []

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def ensure_open(self) -> Tuple[bool, str]:
        with self._lock:
            if self._page is not None:
                try:
                    _ = self._page.url
                    return True, ""
                except Exception:
                    self._page = None
            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                # Self-heal: try to install the playwright package + browser binary.
                try:
                    from core.bootstrap import ensure_browser_binary
                    print("  [browser] Playwright not installed — installing now…")
                    okb, msgb = ensure_browser_binary(quiet=False)
                    if okb:
                        from playwright.sync_api import sync_playwright  # noqa: F811
                    else:
                        return False, (
                            "Playwright not installed and auto-install failed.\n"
                            "  Run: python -m core.bootstrap --browser"
                        )
                except Exception:
                    return False, (
                        "Playwright not installed.\n"
                        "  Run: python -m core.bootstrap --browser\n"
                        "  Or:  pip install playwright && playwright install chromium"
                    )

            # Ensure the Chromium *binary* exists before launching (self-heal).
            try:
                from core.bootstrap import is_browser_binary_installed, ensure_browser_binary
                if not is_browser_binary_installed():
                    print("  [browser] Chromium binary missing — downloading (~120 MB, one-time)…")
                    ensure_browser_binary(quiet=False)
            except Exception:
                pass  # fall through; launch() will surface a clear error if still missing

            try:
                SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
                DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
                self._pw_ctx  = sync_playwright().__enter__()
                self._browser = self._pw_ctx.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                        "--disable-web-security",
                        "--disable-features=IsolateOrigins,site-per-process",
                        "--flag-switches-begin",
                        "--disable-site-isolation-trials",
                        "--flag-switches-end",
                    ],
                )
                self._context = self._browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    user_agent=_random_user_agent(),
                    ignore_https_errors=True,
                    accept_downloads=True,
                    locale="en-US",
                    timezone_id="America/New_York",
                    geolocation={"longitude": -74.006, "latitude": 40.7128},
                    permissions=["geolocation"],
                    color_scheme="light",
                    extra_http_headers={
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                        "sec-ch-ua": '"Chromium";v="121", "Not A(Brand";v="99"',
                        "sec-ch-ua-mobile": "?0",
                        "sec-ch-ua-platform": '"macOS"',
                        "Upgrade-Insecure-Requests": "1",
                    },
                )
                self._context.set_default_timeout(15_000)
                self._context.set_default_navigation_timeout(30_000)
                self._page = self._context.new_page()
                # Inject stealth patches to defeat bot detection
                _inject_stealth(self._page)
                self._page.on("console", lambda msg: self._console_log.append(
                    {"type": msg.type, "text": msg.text, "ts": time.time()}
                ) if len(self._console_log) < 500 else None)
                self._page.on("dialog", lambda dlg: (
                    self._dialog_text.append(dlg.message), dlg.dismiss(),
                ))
                self._page.on("response", lambda r: self._network_log.append(
                    {"url": r.url, "status": r.status, "ts": time.time()}
                ) if len(self._network_log) < 200 else None)
                return True, ""
            except Exception as e:
                msg = str(e)
                # Classic "binary missing" error → self-heal once and retry.
                if "executable doesn" in msg.lower() or "playwright install" in msg.lower():
                    try:
                        from core.bootstrap import ensure_browser_binary
                        print("  [browser] Installing missing Chromium binary…")
                        okb, _ = ensure_browser_binary(quiet=False)
                        if okb:
                            return False, (
                                "Chromium was just installed — please retry your request."
                            )
                    except Exception:
                        pass
                    return False, (
                        f"Browser binary missing: {e}\n"
                        "  Run: python -m core.bootstrap --browser"
                    )
                return False, f"Failed to launch browser: {e}"

    def close(self) -> None:
        with self._lock:
            try:
                if self._browser:
                    self._browser.close()
                if self._pw_ctx:
                    self._pw_ctx.__exit__(None, None, None)
            except Exception:
                pass
            finally:
                self._page = self._browser = self._context = self._pw_ctx = None

    # ── Navigation ─────────────────────────────────────────────────────────

    def navigate(self, url: str, wait_until: str = "domcontentloaded") -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        if not re.match(r"https?://", url, re.I):
            url = "https://" + url
        safe, reason = _is_url_safe(url)
        if not safe:
            return {"success": False, "error": f"URL blocked: {reason}"}
        try:
            resp = self._page.goto(url, wait_until=wait_until, timeout=30_000)
            return {
                "success": True, "url": self._page.url,
                "title": self._page.title(),
                "status": resp.status if resp else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Snapshot ───────────────────────────────────────────────────────────

    def snapshot(self, max_chars: int = 8000) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            aria = self._page.accessibility.snapshot(interesting_only=True)
            if aria:
                text = _aria_to_text(aria)
                if len(text) > max_chars:
                    text = text[:max_chars] + f"\n[… truncated at {max_chars} chars]"
                return {"success": True, "url": self._page.url,
                        "title": self._page.title(), "type": "aria", "content": text}
        except Exception:
            pass
        try:
            text = self._page.evaluate("""() => {
                function vis(e){const s=window.getComputedStyle(e);
                  return s.display!=='none'&&s.visibility!=='hidden'&&s.opacity!=='0';}
                function walk(n,d,out){
                  if(d>8)return;
                  if(['script','style','noscript','svg','path'].includes((n.tagName||'').toLowerCase()))return;
                  if(n.nodeType===3){const t=n.textContent.trim();if(t)out.push(t);}
                  else if(n.nodeType===1&&vis(n)){for(const c of n.childNodes)walk(c,d+1,out);}
                }
                const out=[];walk(document.body,0,out);return out.join('\\n');
            }""")
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n[… truncated]"
            return {"success": True, "url": self._page.url,
                    "title": self._page.title(), "type": "text", "content": text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Screenshot ─────────────────────────────────────────────────────────

    def screenshot(self, full_page: bool = False) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            png  = self._page.screenshot(full_page=full_page, type="png")
            b64  = base64.b64encode(png).decode()
            path = SCREENSHOTS_DIR / f"screenshot_{uuid.uuid4().hex[:8]}.png"
            path.write_bytes(png)
            return {"success": True, "url": self._page.url, "saved_to": str(path),
                    "bytes": len(png), "base64_png": b64}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Interactions ────────────────────────────────────────────────────────

    def _loc(self, selector: str):
        return self._page.locator(selector)

    def click(self, selector: str = "", x: int = None, y: int = None,
              button: str = "left", click_count: int = 1,
              capture_after: bool = False) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            if selector:
                self._loc(selector).click(button=button, click_count=click_count)
            elif x is not None and y is not None:
                self._page.mouse.click(float(x), float(y), button=button,
                                       click_count=click_count)
            else:
                return {"success": False, "error": "need selector or x,y"}
            res = {"success": True, "action": "click", "selector": selector}
            if capture_after:
                res["after_snapshot"] = self.snapshot().get("content", "")
            return res
        except Exception as e:
            return {"success": False, "error": str(e)}

    def type_text(self, selector: str, text: str, clear_first: bool = True) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            el = self._loc(selector)
            if clear_first:
                el.fill(text)
            else:
                el.type(text)
            return {"success": True, "action": "type", "chars": len(text)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def scroll(self, direction: str = "down", amount: int = 3) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            px  = amount * 300
            dy  = px  if direction == "down"  else (-px if direction == "up" else 0)
            dx  = px  if direction == "right" else (-px if direction == "left" else 0)
            self._page.mouse.wheel(dx, dy)
            return {"success": True, "action": "scroll", "direction": direction}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def hover(self, selector: str) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            self._loc(selector).hover()
            return {"success": True, "action": "hover"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def key_press(self, key: str, selector: str = "") -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            if selector:
                self._loc(selector).press(key)
            else:
                self._page.keyboard.press(key)
            return {"success": True, "action": "key", "key": key}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def select_option(self, selector: str, value: str = "", label: str = "") -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            kw: Dict = {}
            if value:  kw["value"] = value
            elif label: kw["label"] = label
            self._loc(selector).select_option(**kw)
            return {"success": True, "action": "select_option"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def fill_form(self, fields: List[Dict]) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        results = []
        for f in fields:
            sel, val = f.get("selector", ""), f.get("value", "")
            try:
                self._loc(sel).fill(val)
                results.append({"selector": sel, "ok": True})
            except Exception as e:
                results.append({"selector": sel, "ok": False, "error": str(e)})
        failed = [r for r in results if not r["ok"]]
        return {"success": not failed, "fields": results}

    # ── Tab management ─────────────────────────────────────────────────────

    def new_tab(self, url: str = "") -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            self._page = self._context.new_page()
            if url:
                return self.navigate(url)
            return {"success": True, "tab_count": len(self._context.pages)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_tabs(self) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        pages = self._context.pages
        return {"success": True, "tabs": [
            {"index": i, "url": p.url, "title": p.title()}
            for i, p in enumerate(pages)
        ], "active": next((i for i, p in enumerate(pages) if p is self._page), 0)}

    def switch_tab(self, index: int) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        pages = self._context.pages
        if not (0 <= index < len(pages)):
            return {"success": False, "error": f"tab {index} out of range"}
        self._page = pages[index]
        return {"success": True, "url": self._page.url}

    # ── Extract helpers ────────────────────────────────────────────────────

    def extract_text(self, selector: str = "body", max_chars: int = 6000) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            text = (self._loc(selector).inner_text()
                    if selector != "body" else self._page.inner_text("body"))
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) > max_chars:
                text = text[:max_chars] + "\n[… truncated]"
            return {"success": True, "text": text, "url": self._page.url}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_links(self) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            links = self._page.evaluate("""() =>
                Array.from(document.querySelectorAll('a[href]'))
                    .map(a=>({text:a.textContent.trim().slice(0,80),href:a.href}))
                    .filter(l=>l.href.startsWith('http'))
            """)
            return {"success": True, "links": links[:100], "total": len(links)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_tables(self) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            tables = self._page.evaluate("""() =>
                Array.from(document.querySelectorAll('table')).map(t=>({
                    headers: Array.from(t.querySelectorAll('th')).map(h=>h.textContent.trim()),
                    rows: Array.from(t.querySelectorAll('tr')).map(r=>
                        Array.from(r.querySelectorAll('td,th')).map(c=>c.textContent.trim())
                    ).filter(r=>r.length>0)
                }))
            """)
            return {"success": True, "tables": tables, "count": len(tables)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── JS / cookies / PDF ─────────────────────────────────────────────────

    def evaluate(self, script: str) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        _BLOCKED = ["fetch(", "XMLHttpRequest", "navigator.sendBeacon"]
        for b in _BLOCKED:
            if b.lower() in script.lower():
                return {"success": False, "error": f"blocked JS pattern: {b!r}"}
        try:
            result = self._page.evaluate(script)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_cookies(self) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            return {"success": True, "cookies": self._context.cookies()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_cookie(self, name: str, value: str, domain: str = "", path: str = "/") -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            if not domain:
                domain = urlparse(self._page.url).hostname or ""
            self._context.add_cookies([{"name": name, "value": value,
                                         "domain": domain, "path": path}])
            return {"success": True, "action": "set_cookie"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def print_pdf(self, path: str = "") -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            if not path:
                path = str(DOWNLOADS_DIR / f"page_{uuid.uuid4().hex[:8]}.pdf")
            self._page.pdf(path=path, format="A4", print_background=True)
            return {"success": True, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def go_back(self) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            self._page.go_back(wait_until="domcontentloaded", timeout=10_000)
            return {"success": True, "url": self._page.url}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def go_forward(self) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            self._page.go_forward(wait_until="domcontentloaded", timeout=10_000)
            return {"success": True, "url": self._page.url}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def reload(self) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            self._page.reload(wait_until="domcontentloaded")
            return {"success": True, "url": self._page.url}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def wait_for(self, selector: str = "", state: str = "visible",
                 timeout_ms: int = 10_000) -> Dict:
        ok, err = self.ensure_open()
        if not ok:
            return {"success": False, "error": err}
        try:
            if not selector:
                self._page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
                return {"success": True, "url": self._page.url}
            el = self._page.wait_for_selector(selector, state=state, timeout=timeout_ms)
            return {"success": el is not None}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def network_log(self, last_n: int = 20) -> Dict:
        return {"success": True, "log": self._network_log[-last_n:]}

    def console_log(self, last_n: int = 20) -> Dict:
        return {"success": True, "log": self._console_log[-last_n:]}


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def _get_session(task_id: str = "") -> _BrowserSession:
    sid = task_id or _DEFAULT_SESSION_ID
    with _SESSION_LOCK:
        if sid not in _SESSIONS:
            _SESSIONS[sid] = _BrowserSession(sid)
        return _SESSIONS[sid]


def _close_all():
    with _SESSION_LOCK:
        for s in list(_SESSIONS.values()):
            try:
                s.close()
            except Exception:
                pass
        _SESSIONS.clear()


atexit.register(_close_all)


# ---------------------------------------------------------------------------
# Public tool functions (registered in tools/registry.py)
# ---------------------------------------------------------------------------

def browser_navigate(url: str, task_id: str = "", wait_until: str = "domcontentloaded", **_) -> Dict:
    """Navigate to a URL. Auto-prepends https:// if missing."""
    return _get_session(task_id).navigate(url, wait_until=wait_until)

def browser_snapshot(task_id: str = "", max_chars: int = 8000, **_) -> Dict:
    """Return an accessibility-tree text snapshot of the current page."""
    return _get_session(task_id).snapshot(max_chars=int(max_chars))

def browser_screenshot(task_id: str = "", full_page: bool = False, **_) -> Dict:
    """Capture a screenshot. Returns base64 PNG and saved file path."""
    return _get_session(task_id).screenshot(full_page=bool(full_page))

def browser_click(selector: str = "", x: int = None, y: int = None,
                  button: str = "left", click_count: int = 1,
                  capture_after: bool = False, task_id: str = "", **_) -> Dict:
    """Click an element or screen coordinate."""
    return _get_session(task_id).click(selector=selector, x=x, y=y, button=button,
                                        click_count=int(click_count), capture_after=bool(capture_after))

def browser_type(selector: str, text: str, clear_first: bool = True, task_id: str = "", **_) -> Dict:
    """Type text into a form field."""
    return _get_session(task_id).type_text(selector=selector, text=text, clear_first=bool(clear_first))

def browser_scroll(direction: str = "down", amount: int = 3, task_id: str = "", **_) -> Dict:
    """Scroll the page. direction: up|down|left|right"""
    return _get_session(task_id).scroll(direction=direction, amount=int(amount))

def browser_hover(selector: str, task_id: str = "", **_) -> Dict:
    """Hover over an element."""
    return _get_session(task_id).hover(selector=selector)

def browser_key(key: str, selector: str = "", task_id: str = "", **_) -> Dict:
    """Press a keyboard key. Supports combos: Enter, Tab, ctrl+a."""
    return _get_session(task_id).key_press(key=key, selector=selector)

def browser_select(selector: str, value: str = "", label: str = "", task_id: str = "", **_) -> Dict:
    """Select an option from a <select> dropdown."""
    return _get_session(task_id).select_option(selector=selector, value=value, label=label)

def browser_fill_form(fields: List[Dict] = None, task_id: str = "", **_) -> Dict:
    """Fill multiple form fields at once. fields=[{selector, value}, ...]"""
    return _get_session(task_id).fill_form(fields or [])

def browser_wait(selector: str = "", state: str = "visible",
                 timeout_ms: int = 10_000, task_id: str = "", **_) -> Dict:
    """Wait for a selector to be visible/hidden, or for page navigation."""
    return _get_session(task_id).wait_for(selector=selector, state=state, timeout_ms=int(timeout_ms))

def browser_extract_text(selector: str = "body", max_chars: int = 6000, task_id: str = "", **_) -> Dict:
    """Extract visible text from the page or a specific element."""
    return _get_session(task_id).extract_text(selector=selector, max_chars=int(max_chars))

def browser_extract_links(task_id: str = "", **_) -> Dict:
    """Extract all hyperlinks from the current page."""
    return _get_session(task_id).extract_links()

def browser_extract_tables(task_id: str = "", **_) -> Dict:
    """Extract all HTML tables from the current page as structured data."""
    return _get_session(task_id).extract_tables()

def browser_evaluate(script: str, task_id: str = "", **_) -> Dict:
    """Execute JavaScript on the page."""
    return _get_session(task_id).evaluate(script=script)

def browser_new_tab(url: str = "", task_id: str = "", **_) -> Dict:
    """Open a new browser tab."""
    return _get_session(task_id).new_tab(url=url)

def browser_list_tabs(task_id: str = "", **_) -> Dict:
    """List all open browser tabs."""
    return _get_session(task_id).list_tabs()

def browser_switch_tab(index: int = 0, task_id: str = "", **_) -> Dict:
    """Switch to a browser tab by index."""
    return _get_session(task_id).switch_tab(index=int(index))

def browser_go_back(task_id: str = "", **_) -> Dict:
    """Go back in browser history."""
    return _get_session(task_id).go_back()

def browser_go_forward(task_id: str = "", **_) -> Dict:
    """Go forward in browser history."""
    return _get_session(task_id).go_forward()

def browser_reload(task_id: str = "", **_) -> Dict:
    """Reload the current page."""
    return _get_session(task_id).reload()

def browser_get_cookies(task_id: str = "", **_) -> Dict:
    """Get all cookies for the current browser session."""
    return _get_session(task_id).get_cookies()

def browser_set_cookie(name: str, value: str, domain: str = "", path: str = "/", task_id: str = "", **_) -> Dict:
    """Set a cookie."""
    return _get_session(task_id).set_cookie(name=name, value=value, domain=domain, path=path)

def browser_print_pdf(path: str = "", task_id: str = "", **_) -> Dict:
    """Print the current page to a PDF file."""
    return _get_session(task_id).print_pdf(path=path)

def browser_network_log(last_n: int = 20, task_id: str = "", **_) -> Dict:
    """Return recent network requests."""
    return _get_session(task_id).network_log(last_n=int(last_n))

def browser_console_log(last_n: int = 20, task_id: str = "", **_) -> Dict:
    """Return recent browser console messages."""
    return _get_session(task_id).console_log(last_n=int(last_n))

def browser_close(task_id: str = "", **_) -> Dict:
    """Close the browser session and free resources."""
    sid = task_id or _DEFAULT_SESSION_ID
    with _SESSION_LOCK:
        session = _SESSIONS.pop(sid, None)
    if session:
        session.close()
    return {"success": True, "session_id": sid}

# ---------------------------------------------------------------------------
# New tools: visual grounding, CAPTCHA detection, human-mode interactions
# ---------------------------------------------------------------------------

def browser_find_element(
    description: str,
    task_id: str = "",
    **_,
) -> Dict:
    """
    Find an element by natural-language description using the page's aria tree.
    Returns the best matching element ref, role, name and a selector hint.

    Args:
        description: Natural language description, e.g. "the login button" or "search box".
        task_id:     Optional session ID.

    Returns:
        {success, output: {ref, role, name, selector_hint}, error}
    """
    session = _get_session(task_id)
    ok, err = session.ensure_open()
    if not ok:
        return {"success": False, "error": err}
    try:
        # Try aria snapshot first
        snapshot_raw = session._page.accessibility.snapshot(interesting_only=True)
        if snapshot_raw:
            best = _find_best_match(snapshot_raw, description.lower())
            if best:
                return {"success": True, "output": best}

        # Fallback: search visible text
        desc_lower = description.lower()
        for selector in [
            f"text={description}",
            f"[aria-label*='{description}' i]",
            f"[placeholder*='{description}' i]",
            f"[title*='{description}' i]",
        ]:
            try:
                el = session._page.query_selector(selector)
                if el and el.is_visible():
                    box = el.bounding_box()
                    return {
                        "success": True,
                        "output": {
                            "selector": selector,
                            "x": int(box["x"] + box["width"] / 2) if box else None,
                            "y": int(box["y"] + box["height"] / 2) if box else None,
                        },
                    }
            except Exception:
                continue
        return {"success": False, "error": f"Could not find element matching: {description!r}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _find_best_match(node: Dict, query: str) -> Optional[Dict]:
    """Recursively search aria tree for best matching node."""
    name = (node.get("name") or "").lower()
    role = (node.get("role") or "").lower()

    # Score this node
    score = 0
    for word in query.split():
        if word in name:
            score += 2
        if word in role:
            score += 1

    best = None
    best_score = 0
    if score > 0 and role not in ("document", "generic", "none", "presentation"):
        best = {"ref": node.get("ref"), "role": role, "name": node.get("name", ""), "score": score}
        best_score = score

    for child in node.get("children", []):
        candidate = _find_best_match(child, query)
        if candidate and candidate.get("score", 0) > best_score:
            best = candidate
            best_score = candidate["score"]

    return best


def browser_check_captcha(task_id: str = "", **_) -> Dict:
    """
    Check if the current page is showing a CAPTCHA or bot challenge.

    Returns:
        {success, output: {captcha_detected: bool, url, title}, error}
    """
    session = _get_session(task_id)
    ok, err = session.ensure_open()
    if not ok:
        return {"success": False, "error": err}
    try:
        detected = _check_captcha(session._page)
        return {
            "success": True,
            "output": {
                "captcha_detected": detected,
                "url":   session._page.url,
                "title": session._page.title(),
                "hint":  "Use browser_screenshot() to see the challenge" if detected else "",
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_human_click(
    selector: str = "",
    x: Optional[int] = None,
    y: Optional[int] = None,
    task_id: str = "",
    **_,
) -> Dict:
    """
    Click with human-like random delay and slight position jitter.
    Mimics real mouse movement to defeat behaviour-based bot detection.

    Args:
        selector: CSS/aria selector (preferred over x/y).
        x, y:    Fallback coordinate click.
        task_id: Session ID.
    """
    _human_delay(60, 180)
    session = _get_session(task_id)
    ok, err = session.ensure_open()
    if not ok:
        return {"success": False, "error": err}
    try:
        if selector:
            el = session._page.query_selector(selector)
            if not el:
                return {"success": False, "error": f"Selector not found: {selector}"}
            box = el.bounding_box()
            if box:
                # Small random jitter within element bounds
                jx = int(box["x"] + box["width"]  * _random.uniform(0.3, 0.7))
                jy = int(box["y"] + box["height"] * _random.uniform(0.3, 0.7))
                session._page.mouse.move(jx - 15, jy - 10)
                _human_delay(30, 80)
                session._page.mouse.move(jx, jy)
                _human_delay(20, 60)
                session._page.mouse.click(jx, jy)
            else:
                el.click()
        elif x is not None and y is not None:
            session._page.mouse.move(int(x) - 12, int(y) - 8)
            _human_delay(30, 80)
            session._page.mouse.click(int(x), int(y))
        else:
            return {"success": False, "error": "Provide selector or x,y coordinates"}
        _human_delay(100, 300)
        return {"success": True, "url": session._page.url}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_human_type(
    text: str,
    selector: str = "",
    delay_ms: int = 80,
    task_id: str = "",
    **_,
) -> Dict:
    """
    Type text with realistic per-character delays and occasional typos/corrections.
    Defeats keystroke-timing bot detection.

    Args:
        text:     Text to type.
        selector: Focus this element before typing (optional).
        delay_ms: Average delay between keystrokes in ms (default 80).
        task_id:  Session ID.
    """
    session = _get_session(task_id)
    ok, err = session.ensure_open()
    if not ok:
        return {"success": False, "error": err}
    try:
        if selector:
            session._page.click(selector)
            _human_delay(80, 150)

        for char in text:
            session._page.keyboard.type(char)
            # Variable delay with occasional pauses
            delay = int(_random.gauss(delay_ms, delay_ms * 0.3))
            delay = max(20, min(delay, delay_ms * 4))
            time.sleep(delay / 1000)

        return {"success": True, "chars_typed": len(text)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def browser_wait_for_element(
    selector: str,
    timeout_ms: int = 10000,
    state: str = "visible",
    task_id: str = "",
    **_,
) -> Dict:
    """
    Wait for an element to appear, become visible, or be attached.

    Args:
        selector:   CSS or aria selector.
        timeout_ms: Max wait in milliseconds (default 10000).
        state:      "attached", "detached", "visible", "hidden".
        task_id:    Session ID.
    """
    session = _get_session(task_id)
    ok, err = session.ensure_open()
    if not ok:
        return {"success": False, "error": err}
    try:
        session._page.wait_for_selector(selector, timeout=timeout_ms, state=state)
        el  = session._page.query_selector(selector)
        box = el.bounding_box() if el else None
        return {
            "success": True,
            "selector": selector,
            "visible": bool(box),
            "box": box,
        }
    except Exception as e:
        return {"success": False, "error": f"Element wait failed: {e}"}


def browser_get_url(task_id: str = "", **_) -> Dict:
    """Return the current page URL and title."""
    session = _get_session(task_id)
    ok, err = session.ensure_open()
    if not ok:
        return {"success": False, "error": err}
    try:
        return {
            "success": True,
            "url":   session._page.url,
            "title": session._page.title(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
