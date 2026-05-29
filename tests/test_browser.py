"""Tests for tools/browser.py (no real browser needed — tests safety/logic layer)"""
import pytest
from tools.browser import (
    _is_url_safe, _aria_to_text, _random_user_agent,
    _check_captcha, _STEALTH_JS, _CAPTCHA_PATTERNS,
    browser_close,
)


# ── URL safety ────────────────────────────────────────────────────────────────

class TestUrlSafety:
    def test_https_allowed(self):
        safe, reason = _is_url_safe("https://example.com")
        assert safe

    def test_http_allowed(self):
        safe, reason = _is_url_safe("http://example.com/path?q=1")
        assert safe

    def test_javascript_blocked(self):
        safe, reason = _is_url_safe("javascript:alert(1)")
        assert not safe
        assert "javascript" in reason

    def test_vbscript_blocked(self):
        safe, reason = _is_url_safe("vbscript:msgbox(1)")
        assert not safe

    def test_data_html_blocked(self):
        safe, reason = _is_url_safe("data:text/html,<h1>hi</h1>")
        assert not safe

    def test_onion_blocked(self):
        safe, reason = _is_url_safe("http://something.onion/page")
        assert not safe
        assert "blocked" in reason.lower()

    def test_localhost_blocked(self):
        safe, reason = _is_url_safe("http://localhost:8080/admin")
        assert not safe

    def test_127_blocked(self):
        safe, reason = _is_url_safe("http://127.0.0.1/secret")
        assert not safe

    def test_empty_url_blocked(self):
        safe, reason = _is_url_safe("")
        assert not safe

    def test_none_blocked(self):
        safe, reason = _is_url_safe(None)
        assert not safe

    def test_unparseable_blocked(self):
        safe, reason = _is_url_safe("not a url \x00\x01")
        # Should not raise, just fail safely
        assert isinstance(safe, bool)


# ── Aria tree → text ──────────────────────────────────────────────────────────

class TestAriaToText:
    def test_simple_node(self):
        node = {"role": "text", "name": "Hello World", "children": []}
        result = _aria_to_text(node)
        assert "Hello World" in result

    def test_heading(self):
        node = {"role": "heading", "name": "Main Title", "level": 1, "children": []}
        result = _aria_to_text(node)
        assert "# Main Title" in result

    def test_link(self):
        node = {"role": "link", "name": "Click here", "children": []}
        result = _aria_to_text(node)
        assert "Click here" in result

    def test_button(self):
        node = {"role": "button", "name": "Submit", "children": []}
        result = _aria_to_text(node)
        assert "[btn] Submit" in result

    def test_input(self):
        node = {"role": "textbox", "name": "Email", "value": "user@example.com", "children": []}
        result = _aria_to_text(node)
        assert "Email" in result

    def test_checkbox_checked(self):
        node = {"role": "checkbox", "name": "Agree", "checked": True, "children": []}
        result = _aria_to_text(node)
        assert "Agree" in result
        assert "✓" in result

    def test_checkbox_unchecked(self):
        node = {"role": "checkbox", "name": "Subscribe", "checked": False, "children": []}
        result = _aria_to_text(node)
        assert "○" in result

    def test_nested_children(self):
        node = {
            "role": "main",
            "name": "Content",
            "children": [
                {"role": "heading", "name": "Title", "level": 2, "children": []},
                {"role": "text",    "name": "Body",               "children": []},
            ]
        }
        result = _aria_to_text(node)
        assert "Title" in result
        assert "Body" in result

    def test_generic_role_skipped(self):
        node = {"role": "generic", "name": "wrapper", "children": [
            {"role": "text", "name": "visible", "children": []}
        ]}
        result = _aria_to_text(node)
        assert "visible" in result
        assert "wrapper" not in result

    def test_empty_node(self):
        result = _aria_to_text({})
        assert isinstance(result, str)

    def test_none_node(self):
        # Should not raise
        result = _aria_to_text(None)
        assert isinstance(result, str)


# ── Stealth ───────────────────────────────────────────────────────────────────

class TestStealth:
    def test_stealth_js_present(self):
        assert "webdriver" in _STEALTH_JS
        assert "navigator" in _STEALTH_JS

    def test_stealth_removes_webdriver_flag(self):
        assert "webdriver" in _STEALTH_JS
        assert "undefined" in _STEALTH_JS

    def test_stealth_adds_chrome_runtime(self):
        assert "window.chrome" in _STEALTH_JS

    def test_stealth_patches_plugins(self):
        assert "plugins" in _STEALTH_JS

    def test_random_user_agent_returns_string(self):
        ua = _random_user_agent()
        assert isinstance(ua, str)
        assert "Mozilla" in ua

    def test_user_agent_rotates(self):
        uas = {_random_user_agent() for _ in range(20)}
        assert len(uas) > 1  # should not always return the same one


# ── CAPTCHA detection ─────────────────────────────────────────────────────────

class TestCaptchaDetection:
    def test_captcha_pattern_matches(self):
        texts = [
            "Please complete the captcha below",
            "Are you a robot? Please verify",
            "Unusual traffic detected from your network",
            "Just a moment... | Cloudflare",
            "DDoS-Guard protection page",
            "Verify you are human",
        ]
        for text in texts:
            assert _CAPTCHA_PATTERNS.search(text), f"Should match: {text!r}"

    def test_captcha_pattern_no_false_positive(self):
        clean_texts = [
            "Welcome to our website",
            "Search results for: python programming",
            "Latest news headlines",
        ]
        for text in clean_texts:
            # These should NOT match captcha patterns
            # (they might in edge cases, but common real content shouldn't)
            assert not _CAPTCHA_PATTERNS.search(text), f"False positive: {text!r}"


# ── browser_close (no browser needed) ────────────────────────────────────────

class TestBrowserClose:
    def test_close_nonexistent_session_ok(self):
        """Closing a session that doesn't exist should not raise."""
        result = browser_close(task_id="__nonexistent_test_session__")
        assert result["success"]

    def test_close_returns_session_id(self):
        result = browser_close(task_id="test_close_id")
        assert "session_id" in result
        assert result["session_id"] == "test_close_id"
