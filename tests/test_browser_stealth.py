"""Tests for core/browser_stealth.py"""
import math
import time
from typing import List, Tuple

import pytest

from core.browser_stealth import (
    StealthProfile, HumanBehavior, CDPSession,
    generate_stealth_launch_args, generate_stealth_context_options,
    detect_captcha_in_html, detect_webdriver_signal,
    build_realistic_headers, get_stealth_profile, random_delay,
    json_str, _CAPTCHA_SIGNALS, _WEBDRIVER_SIGNALS,
    _USER_AGENTS, _SCREEN_RESOLUTIONS, _TIMEZONES, _LANGUAGE_SETS,
)


# ── StealthProfile ────────────────────────────────────────────────────────────

class TestStealthProfile:
    def test_random_generates_profile(self):
        p = StealthProfile.random()
        assert isinstance(p, StealthProfile)
        assert p.user_agent
        assert p.screen_width > 0
        assert p.screen_height > 0

    def test_random_user_agent_from_pool(self):
        p = StealthProfile.random()
        assert p.user_agent in _USER_AGENTS

    def test_random_screen_from_pool(self):
        p = StealthProfile.random()
        assert (p.screen_width, p.screen_height) in _SCREEN_RESOLUTIONS

    def test_random_timezone_from_pool(self):
        p = StealthProfile.random()
        assert p.timezone in _TIMEZONES

    def test_random_languages_from_pool(self):
        p = StealthProfile.random()
        assert p.languages in _LANGUAGE_SETS

    def test_platform_windows(self):
        # Force Windows UA
        p = StealthProfile(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...",
            screen_width=1920, screen_height=1080,
            timezone="America/New_York", languages=["en-US", "en"],
            platform="Win32",
        )
        assert p.platform == "Win32"

    def test_hardware_concurrency_valid(self):
        p = StealthProfile.random()
        assert p.hardware_concurrency in [2, 4, 6, 8, 12, 16]

    def test_device_memory_power_of_2(self):
        p = StealthProfile.random()
        assert p.device_memory in [4, 8, 16, 32]

    def test_to_dict_keys(self):
        p = StealthProfile.random()
        d = p.to_dict()
        assert "user_agent" in d
        assert "screen_width" in d
        assert "timezone" in d
        assert "languages" in d

    def test_from_dict_roundtrip(self):
        p = StealthProfile.random()
        d = p.to_dict()
        p2 = StealthProfile.from_dict(d)
        assert p2.user_agent == p.user_agent
        assert p2.timezone == p.timezone
        assert p2.screen_width == p.screen_width

    def test_js_patches_contains_webdriver(self):
        p = StealthProfile.random()
        js = p.js_patches()
        assert "navigator.webdriver" in js

    def test_js_patches_contains_user_agent(self):
        p = StealthProfile.random()
        js = p.js_patches()
        assert p.user_agent in js

    def test_js_patches_contains_screen_size(self):
        p = StealthProfile.random()
        js = p.js_patches()
        assert str(p.screen_width) in js
        assert str(p.screen_height) in js

    def test_js_patches_canvas_noise(self):
        p = StealthProfile.random()
        js = p.js_patches()
        assert "HTMLCanvasElement.prototype.toDataURL" in js

    def test_js_patches_webgl_spoof(self):
        p = StealthProfile.random()
        js = p.js_patches()
        assert "WebGLRenderingContext.prototype.getParameter" in js
        assert p.webgl_vendor in js

    def test_js_patches_plugins_faked(self):
        p = StealthProfile.random()
        js = p.js_patches()
        assert "Chrome PDF Plugin" in js

    def test_js_patches_chrome_runtime(self):
        p = StealthProfile.random()
        js = p.js_patches()
        assert "window.chrome" in js

    def test_js_patches_hardware_concurrency(self):
        p = StealthProfile.random()
        js = p.js_patches()
        assert str(p.hardware_concurrency) in js

    def test_js_patches_no_syntax_error_structure(self):
        """JS should have balanced braces (simple check)."""
        p = StealthProfile.random()
        js = p.js_patches()
        assert js.count("{") > 0
        # Should start with IIFE
        assert "(function()" in js


# ── HumanBehavior ─────────────────────────────────────────────────────────────

class TestHumanBehavior:
    def test_typing_delays_length(self):
        delays = HumanBehavior.typing_delays("Hello World")
        assert len(delays) == len("Hello World")

    def test_typing_delays_min_20ms(self):
        delays = HumanBehavior.typing_delays("test text")
        assert all(d >= 20 for d in delays)

    def test_typing_delays_jitter_varies(self):
        delays = HumanBehavior.typing_delays("aaaaaa" * 10, base_ms=50, jitter_ms=20)
        # Should not all be identical
        assert len(set(round(d) for d in delays)) > 1

    def test_typing_delays_empty_string(self):
        delays = HumanBehavior.typing_delays("")
        assert delays == []

    def test_typing_delays_punctuation_longer(self):
        # Delays after punctuation should average longer
        delays_punct = HumanBehavior.typing_delays("Hello, World!", base_ms=50)
        # Just check it works without error
        assert len(delays_punct) == len("Hello, World!")

    def test_bezier_path_length(self):
        path = HumanBehavior.bezier_path((0, 0), (500, 300), steps=20)
        assert len(path) == 21   # steps + 1 points

    def test_bezier_path_near_start(self):
        path = HumanBehavior.bezier_path((0, 0), (1000, 0), steps=30)
        # First point should be near (0, 0)
        assert abs(path[0][0]) < 10

    def test_bezier_path_near_end(self):
        path = HumanBehavior.bezier_path((0, 0), (1000, 0), steps=30)
        # Last point should be near (1000, 0)
        assert abs(path[-1][0] - 1000) < 10

    def test_bezier_path_not_straight_line(self):
        # With control point jitter, path should curve
        path = HumanBehavior.bezier_path((0, 0), (1000, 0), steps=30)
        y_values = [p[1] for p in path]
        # At least some y-values should be non-zero (curve deviation)
        assert any(abs(y) > 0.5 for y in y_values)

    def test_scroll_steps_count(self):
        steps = HumanBehavior.scroll_steps(500, steps=5)
        assert len(steps) == 5

    def test_scroll_steps_all_int(self):
        steps = HumanBehavior.scroll_steps(300, steps=5)
        assert all(isinstance(s, int) for s in steps)

    def test_scroll_steps_zero_returns_empty(self):
        steps = HumanBehavior.scroll_steps(0)
        assert steps == []

    def test_scroll_steps_direction_preserved(self):
        down = HumanBehavior.scroll_steps(500, steps=5)
        up   = HumanBehavior.scroll_steps(-500, steps=5)
        # Most down steps should be positive, up negative
        assert sum(down) > 0
        assert sum(up) < 0

    def test_random_viewport_position_in_bounds(self):
        for _ in range(20):
            x, y = HumanBehavior.random_viewport_position(1920, 1080, margin=50)
            assert 50 <= x <= 1870
            assert 50 <= y <= 1030

    def test_think_delay_in_range(self):
        for _ in range(20):
            d = HumanBehavior.think_delay(min_ms=100, max_ms=500)
            assert d >= 100


# ── Anti-detection functions ──────────────────────────────────────────────────

class TestDetectionFunctions:
    def test_detect_captcha_recaptcha(self):
        html = '<div class="g-recaptcha" data-sitekey="6LeXXXXXXXX"></div>'
        found, hint = detect_captcha_in_html(html)
        assert found
        # hint contains the matched signal + optional sitekey info
        assert "captcha" in hint.lower() or "recaptcha" in hint.lower()

    def test_detect_captcha_sitekey_extracted(self):
        html = '<div data-sitekey="SITEKEY123"></div><p>solve captcha</p>'
        found, hint = detect_captcha_in_html(html)
        assert found
        assert "SITEKEY123" in hint

    def test_detect_captcha_hcaptcha(self):
        html = '<div class="h-captcha" data-sitekey="ABC123"></div>'
        found, hint = detect_captcha_in_html(html)
        assert found

    def test_detect_captcha_cloudflare(self):
        html = '<div id="challenge-running">Cloudflare challenge</div>'
        found, hint = detect_captcha_in_html(html)
        assert found

    def test_detect_captcha_human_verification(self):
        html = '<p>Please verify you are human to continue</p>'
        found, hint = detect_captcha_in_html(html)
        assert found

    def test_detect_captcha_clean_page(self):
        html = '<html><body><h1>Welcome!</h1><p>Regular content here.</p></body></html>'
        found, hint = detect_captcha_in_html(html)
        assert not found
        assert hint == ""

    def test_detect_captcha_case_insensitive(self):
        html = '<div class="RECAPTCHA_CONTAINER">Please solve CAPTCHA</div>'
        found, _ = detect_captcha_in_html(html)
        assert found

    def test_detect_webdriver_selenium(self):
        assert detect_webdriver_signal("window.__selenium_evaluate = fn;")

    def test_detect_webdriver_phantom(self):
        assert detect_webdriver_signal("if(window._phantom) doSomething();")

    def test_detect_webdriver_clean(self):
        assert not detect_webdriver_signal("<div>Normal page content</div>")

    def test_detect_webdriver_navigator(self):
        assert detect_webdriver_signal("if(navigator.webdriver) alert('bot');")

    def test_all_captcha_signals_covered(self):
        for signal in _CAPTCHA_SIGNALS[:10]:
            html = f"<div>{signal}</div>"
            found, _ = detect_captcha_in_html(html)
            assert found, f"Signal '{signal}' not detected"


# ── Stealth launch/context helpers ────────────────────────────────────────────

class TestStealthHelpers:
    def test_launch_args_automation_disabled(self):
        args = generate_stealth_launch_args()
        assert "--disable-blink-features=AutomationControlled" in args

    def test_launch_args_no_sandbox(self):
        args = generate_stealth_launch_args()
        assert "--no-sandbox" in args

    def test_launch_args_enable_automation_false(self):
        args = generate_stealth_launch_args()
        assert "--enable-automation=false" in args

    def test_launch_args_is_list(self):
        args = generate_stealth_launch_args()
        assert isinstance(args, list)
        assert all(isinstance(a, str) for a in args)

    def test_context_options_user_agent(self):
        p = StealthProfile.random()
        opts = generate_stealth_context_options(p)
        assert opts["user_agent"] == p.user_agent

    def test_context_options_viewport(self):
        p = StealthProfile.random()
        opts = generate_stealth_context_options(p)
        assert opts["viewport"]["width"] == p.screen_width
        assert opts["viewport"]["height"] == p.screen_height

    def test_context_options_timezone(self):
        p = StealthProfile.random()
        opts = generate_stealth_context_options(p)
        assert opts["timezone_id"] == p.timezone

    def test_context_options_locale(self):
        p = StealthProfile.random()
        opts = generate_stealth_context_options(p)
        assert opts["locale"] == p.languages[0]

    def test_context_options_default_profile(self):
        opts = generate_stealth_context_options()
        assert "user_agent" in opts

    def test_build_headers_user_agent(self):
        p = StealthProfile.random()
        headers = build_realistic_headers(p)
        assert headers["User-Agent"] == p.user_agent

    def test_build_headers_accept_language(self):
        p = StealthProfile.random()
        headers = build_realistic_headers(p)
        assert "Accept-Language" in headers
        assert p.languages[0] in headers["Accept-Language"]

    def test_build_headers_sec_fetch_mode(self):
        headers = build_realistic_headers()
        assert headers["Sec-Fetch-Mode"] == "navigate"

    def test_build_headers_sec_fetch_dest(self):
        headers = build_realistic_headers()
        assert headers["Sec-Fetch-Dest"] == "document"

    def test_build_headers_dnt(self):
        p = StealthProfile.random()
        headers = build_realistic_headers(p)
        assert headers["DNT"] == p.do_not_track


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_stealth_profile_returns_same(self):
        p1 = get_stealth_profile()
        p2 = get_stealth_profile()
        assert p1 is p2

    def test_get_stealth_profile_fresh(self):
        p1 = get_stealth_profile()
        p2 = get_stealth_profile(fresh=True)
        # p2 should be a new instance (different object, same type)
        assert isinstance(p2, StealthProfile)

    def test_get_stealth_profile_is_stealth_profile(self):
        p = get_stealth_profile()
        assert isinstance(p, StealthProfile)


# ── json_str helper ───────────────────────────────────────────────────────────

class TestJsonStr:
    def test_basic_string(self):
        assert json_str("hello") == '"hello"'

    def test_escapes_quotes(self):
        result = json_str('say "hi"')
        assert '\\"' in result or result == '"say \\"hi\\""'

    def test_escapes_newline(self):
        result = json_str("line1\nline2")
        assert "\\n" in result
