"""Tests for tools/http_client.py
Result shape: {success, status_code, headers, body, error}
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from tools.http_client import http_request


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_mock_response(data, status=200, content_type="application/json"):
    m = MagicMock()
    m.status_code = status
    m.ok = status < 400
    m.headers = {"Content-Type": content_type}
    if isinstance(data, (dict, list)):
        m.text = json.dumps(data)
        m.json.return_value = data
    else:
        m.text = str(data)
        m.json.side_effect = ValueError("not JSON")
    return m


# ── GET ───────────────────────────────────────────────────────────────────────

class TestHttpGet:
    def test_get_success(self):
        with patch("requests.request",
                   return_value=make_mock_response({"ok": True})):
            r = http_request("https://example.com/api")
        assert r["success"]

    def test_default_method_is_get(self):
        captured = {}
        def fake(method, url, **kw):
            captured["method"] = method
            return make_mock_response({})
        with patch("requests.request", side_effect=fake):
            http_request("https://example.com/api")
        assert captured.get("method", "GET").upper() == "GET"

    def test_url_required(self):
        r = http_request("")
        assert not r["success"]
        assert r["error"]

    def test_body_in_result(self):
        payload = {"name": "Alice", "score": 42}
        with patch("requests.request",
                   return_value=make_mock_response(payload)):
            r = http_request("https://example.com/data")
        assert r["success"]
        assert r["body"] == payload or "Alice" in str(r.get("body", ""))

    def test_status_code_in_result(self):
        with patch("requests.request",
                   return_value=make_mock_response({}, 200)):
            r = http_request("https://example.com")
        assert r["status_code"] == 200

    def test_query_params_passed(self):
        captured = {}
        def fake(method, url, **kw):
            captured["params"] = kw.get("params", {})
            return make_mock_response({})
        with patch("requests.request", side_effect=fake):
            http_request("https://api.com", params={"q": "test"})
        assert captured["params"].get("q") == "test"

    def test_result_has_required_keys(self):
        with patch("requests.request",
                   return_value=make_mock_response({})):
            r = http_request("https://api.com")
        for key in ("success", "status_code", "body", "error"):
            assert key in r, f"Missing key: {key}"


# ── POST / write methods ──────────────────────────────────────────────────────

class TestHttpPost:
    def test_post_method_sent(self):
        captured = {}
        def fake(method, url, **kw):
            captured["method"] = method
            return make_mock_response({"id": 1}, 201)
        with patch("requests.request", side_effect=fake):
            http_request("https://api.com/items", method="POST",
                         body={"name": "test"})
        assert captured["method"].upper() == "POST"

    def test_201_is_success(self):
        with patch("requests.request",
                   return_value=make_mock_response({"id": 1}, 201)):
            r = http_request("https://api.com/items", method="POST",
                             body={"name": "item"})
        assert r["success"]
        assert r["status_code"] == 201

    def test_body_sent_as_json(self):
        captured = {}
        def fake(method, url, **kw):
            captured.update(kw)
            return make_mock_response({})
        with patch("requests.request", side_effect=fake):
            http_request("https://api.com", method="POST",
                         body={"key": "value"})
        sent = captured.get("json") or captured.get("data")
        assert sent is not None


# ── Headers & Auth ────────────────────────────────────────────────────────────

class TestHttpHeaders:
    def test_custom_headers_forwarded(self):
        captured = {}
        def fake(method, url, **kw):
            captured["headers"] = kw.get("headers", {})
            return make_mock_response({})
        with patch("requests.request", side_effect=fake):
            http_request("https://api.com", headers={"X-Custom": "val123"})
        assert any("val123" in str(v) for v in captured["headers"].values())

    def test_bearer_token_in_auth_header(self):
        captured = {}
        def fake(method, url, **kw):
            captured["headers"] = kw.get("headers", {})
            return make_mock_response({})
        with patch("requests.request", side_effect=fake):
            http_request("https://api.com", bearer_token="secret_tok")
        assert "secret_tok" in str(captured["headers"])


# ── Error handling ────────────────────────────────────────────────────────────

class TestHttpErrors:
    def test_connection_error(self):
        import requests as _r
        with patch("requests.request",
                   side_effect=_r.exceptions.ConnectionError("no host")):
            r = http_request("https://nonexistent.invalid/")
        assert not r["success"]
        assert r["error"]

    def test_timeout_error(self):
        import requests as _r
        with patch("requests.request",
                   side_effect=_r.exceptions.Timeout("timed out")):
            r = http_request("https://slow.example.com/", timeout=1)
        assert not r["success"]

    def test_500_response(self):
        with patch("requests.request",
                   return_value=make_mock_response({"error": "oops"}, 500)):
            r = http_request("https://api.com/crash")
        assert r["status_code"] == 500

    def test_plain_text_response(self):
        with patch("requests.request",
                   return_value=make_mock_response("plain text here", 200,
                                                   "text/plain")):
            r = http_request("https://example.com/text")
        assert r["success"]
        assert "plain text here" in str(r.get("body", ""))


# ── HTTP methods ──────────────────────────────────────────────────────────────

class TestHttpMethods:
    @pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
    def test_method_forwarded(self, method):
        captured = {}
        def fake(m, url, **kw):
            captured["method"] = m
            return make_mock_response({})
        with patch("requests.request", side_effect=fake):
            http_request("https://api.com/resource", method=method)
        assert captured["method"].upper() == method.upper()
