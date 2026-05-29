"""Tests for tools/image_gen.py (no API calls — tests structure and error handling)"""
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from tools.image_gen import (
    image_generate, image_variation, image_list_generated,
    _save_image_bytes, _SAVE_DIR,
)


# ── No API key error handling ─────────────────────────────────────────────────

class TestMissingApiKey:
    def test_generate_fails_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        r = image_generate("a cat")
        assert not r["success"]
        assert "OPENAI_API_KEY" in r["error"]

    def test_stability_fails_without_key(self, monkeypatch):
        monkeypatch.delenv("STABILITY_API_KEY", raising=False)
        from tools.image_gen import image_generate_stability
        r = image_generate_stability("a dog")
        assert not r["success"]
        assert "STABILITY_API_KEY" in r["error"]

    def test_replicate_fails_without_key(self, monkeypatch):
        monkeypatch.delenv("REPLICATE_API_KEY", raising=False)
        from tools.image_gen import image_generate_replicate
        r = image_generate_replicate("a bird")
        assert not r["success"]
        assert "REPLICATE_API_KEY" in r["error"]

    def test_describe_fails_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from tools.image_gen import image_describe
        r = image_describe("https://example.com/img.png")
        assert not r["success"]
        assert "OPENAI_API_KEY" in r["error"]


# ── _save_image_bytes ─────────────────────────────────────────────────────────

class TestSaveImageBytes:
    def test_saves_file(self, tmp_path):
        # Temporarily redirect save dir
        import tools.image_gen as img_mod
        original = img_mod._SAVE_DIR
        img_mod._SAVE_DIR = tmp_path
        try:
            path = _save_image_bytes(b"\x89PNG\r\n", "test prompt", "png")
            assert Path(path).exists()
            assert Path(path).suffix == ".png"
        finally:
            img_mod._SAVE_DIR = original

    def test_different_prompts_different_files(self, tmp_path):
        import tools.image_gen as img_mod
        original = img_mod._SAVE_DIR
        img_mod._SAVE_DIR = tmp_path
        try:
            p1 = _save_image_bytes(b"data1", "prompt one", "png")
            p2 = _save_image_bytes(b"data2", "prompt two", "png")
            # Filenames contain prompt hash — should differ
            assert p1 != p2
        finally:
            img_mod._SAVE_DIR = original


# ── image_generate (mocked HTTP) ─────────────────────────────────────────────

class TestImageGenerateMocked:
    def _mock_response(self, url="https://cdn.example.com/image.png", b64=None):
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        if b64:
            mock.json.return_value = {"data": [{"b64_json": b64}]}
        else:
            mock.json.return_value = {"data": [{"url": url, "revised_prompt": "a cat"}]}
        return mock

    def test_url_response_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        import tools.image_gen as img_mod
        img_mod._SAVE_DIR = tmp_path

        # Mock the download response too
        url_resp = MagicMock()
        url_resp.raise_for_status.return_value = None
        url_resp.content = b"\x89PNG\r\n"
        url_resp.headers = {"content-type": "image/png"}

        with patch("tools.image_gen.requests.post", return_value=self._mock_response()):
            with patch("tools.image_gen.requests.get", return_value=url_resp):
                r = image_generate("a cat", save=True)

        assert r["success"], r.get("error")
        assert "url" in r["output"]
        assert r["output"]["model"] == "dall-e-3"

    def test_b64_response_success(self, tmp_path, monkeypatch):
        import base64
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        import tools.image_gen as img_mod
        img_mod._SAVE_DIR = tmp_path

        b64_data = base64.b64encode(b"\x89PNG\r\n").decode()
        with patch("tools.image_gen.requests.post", return_value=self._mock_response(b64=b64_data)):
            r = image_generate("a cat", return_base64=True, save=True)

        assert r["success"], r.get("error")
        assert "base64" in r["output"]

    def test_http_error_handled(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        import requests as reqs
        mock = MagicMock()
        mock.raise_for_status.side_effect = reqs.HTTPError(
            response=MagicMock(json=lambda: {"error": {"message": "billing limit"}})
        )
        with patch("tools.image_gen.requests.post", return_value=mock):
            r = image_generate("a cat")
        assert not r["success"]
        assert "billing" in r["error"]


# ── image_list_generated ──────────────────────────────────────────────────────

class TestImageListGenerated:
    def test_returns_list(self):
        r = image_list_generated(limit=5)
        assert r["success"]
        assert isinstance(r["output"], list)

    def test_limit_respected(self, tmp_path, monkeypatch):
        import tools.image_gen as img_mod
        original = img_mod._SAVE_DIR
        img_mod._SAVE_DIR = tmp_path
        try:
            for i in range(10):
                (tmp_path / f"img_{i:010d}_abcd1234.png").write_bytes(b"x")
            r = image_list_generated(limit=3)
            assert r["success"]
            assert len(r["output"]) <= 3
        finally:
            img_mod._SAVE_DIR = original

    def test_result_has_expected_keys(self, tmp_path, monkeypatch):
        import tools.image_gen as img_mod
        original = img_mod._SAVE_DIR
        img_mod._SAVE_DIR = tmp_path
        try:
            (tmp_path / "img_0000000001_abc12345.png").write_bytes(b"data")
            r = image_list_generated()
            assert r["success"]
            if r["output"]:
                entry = r["output"][0]
                assert "path" in entry
                assert "size_kb" in entry
                assert "created" in entry
        finally:
            img_mod._SAVE_DIR = original
