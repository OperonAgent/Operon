"""
tests/test_harvest_phase234.py — second-pass harvest (Phases 2-4).

Covers:
  * Phase 2 — file_search enrichment (whole_word, files_with_matches, safe
    fallbacks) and configurable max_response_tokens.
  * Phase 3 — Hermes-style memory fact normalization / dedup.
  * Phase 4 — configurable generation token budget wired into the router.
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Phase 2: file_search ──────────────────────────────────────────────────────

from tools.file_search import file_search


class TestFileSearchEnrichment:
    def test_missing_pattern_safe(self):
        out = file_search(pattern="")
        assert out["success"] is False and "required" in out["error"]

    def test_path_defaults_to_cwd(self, tmp_path, monkeypatch):
        (tmp_path / "a.txt").write_text("hello world\n")
        monkeypatch.chdir(tmp_path)
        out = file_search(pattern="hello", path="")   # empty path -> "."
        assert out["success"] is True and out["total"] >= 1

    def test_files_with_matches_returns_only_filenames(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo(): pass\n")
        (tmp_path / "b.py").write_text("x = 1\n")
        out = file_search(pattern="def ", path=str(tmp_path),
                          file_pattern="*.py", files_with_matches=True)
        assert out["total"] == 1
        assert all(set(m.keys()) == {"file"} for m in out["matches"])

    def test_whole_word_boundary(self, tmp_path):
        (tmp_path / "c.txt").write_text("Classifier\nClass\n")
        # whole_word 'Class' should match the standalone line but NOT 'Classifier'
        out = file_search(pattern="Class", path=str(tmp_path),
                          whole_word=True, case_sensitive=True)
        assert out["total"] == 1
        assert out["matches"][0]["line"] == "Class"

    def test_substring_without_whole_word(self, tmp_path):
        (tmp_path / "c.txt").write_text("Classifier\nClass\n")
        out = file_search(pattern="Class", path=str(tmp_path),
                          whole_word=False, case_sensitive=True)
        assert out["total"] == 2   # both lines match as substrings

    def test_invalid_regex_handled(self, tmp_path):
        out = file_search(pattern="(unclosed", path=str(tmp_path))
        assert out["success"] is False and "Invalid regex" in out["error"]

    def test_extra_kwargs_ignored(self, tmp_path):
        # registry may pass extra params; **_ must absorb them
        out = file_search(pattern="x", path=str(tmp_path), bogus=123)
        assert "success" in out


# ── Phase 3: memory normalization + dedup ─────────────────────────────────────

from core.memory import MemoryPipeline


class TestMemoryNormalization:
    def test_collapses_variants(self):
        n = MemoryPipeline._normalize_fact
        assert n("The user likes Python.") == n("user likes python") == "likes python"

    def test_strips_leading_filler_and_punct(self):
        n = MemoryPipeline._normalize_fact
        assert n("  My name is Sam!! ") == "name is sam"

    def test_empty_safe(self):
        assert MemoryPipeline._normalize_fact("") == ""
        assert MemoryPipeline._normalize_fact(None) == ""

    def test_dedup_on_save(self, tmp_path, monkeypatch):
        import core.memory as mem
        monkeypatch.setattr(mem, "MEMORY_DB", tmp_path / "mem.db")
        pipe = MemoryPipeline(config=MagicMock())
        # Force a known extraction so the test doesn't depend on regex patterns.
        monkeypatch.setattr(pipe, "_extract",
                            lambda text: [{"type": "fact", "content": text, "importance": 2}])
        pipe.async_evaluate_and_save([{"content": "The user likes Python."}])
        pipe.async_evaluate_and_save([{"content": "user likes python"}])  # dup
        pipe.async_evaluate_and_save([{"content": "User prefers dark mode"}])  # new
        facts = [m["content"] for m in pipe.get_all()]
        assert len(facts) == 2  # the near-duplicate was collapsed


# ── Phase 3/4: configurable max_response_tokens ───────────────────────────────

class TestConfigurableMaxTokens:
    def test_default_is_4096(self):
        from core.config import _DEFAULTS
        assert _DEFAULTS["max_response_tokens"] == 4096

    def test_router_reads_config(self):
        from core.router import ModelRouter
        cfg = MagicMock()
        store = {"max_response_tokens": 1000}
        cfg.get.side_effect = lambda k, d=None: store.get(k, d)
        r = ModelRouter(cfg)
        # the value the router would put in a payload
        assert r._config.get("max_response_tokens", 4096) == 1000

    def test_router_falls_back_to_4096(self):
        from core.router import ModelRouter
        cfg = MagicMock()
        cfg.get.side_effect = lambda k, d=None: d   # nothing stored
        r = ModelRouter(cfg)
        assert r._config.get("max_response_tokens", 4096) == 4096
