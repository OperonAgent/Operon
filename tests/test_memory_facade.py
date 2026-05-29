"""tests/test_memory_facade.py — unified memory facade."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.memory_facade import MemoryFacade, MemoryHit, get_memory_facade


@pytest.fixture
def fts_pipeline():
    pl = MagicMock()
    pl.search.return_value = [{"content": "User prefers dark mode"}]
    pl.get_all.return_value = [1, 2, 3]
    return pl


@pytest.fixture
def facade(fts_pipeline):
    return MemoryFacade(pipeline=fts_pipeline, enable_vector=False, enable_obsidian=False)


class TestRemember:
    def test_writes_to_fts5(self, facade, fts_pipeline):
        r = facade.remember("a fact", tags=["t"])
        assert r["fts5"] is True
        fts_pipeline.add_manual.assert_called_once()

    def test_vector_false_when_disabled(self, facade):
        r = facade.remember("a fact")
        assert r["vector"] is False

    def test_obsidian_skipped_by_default(self, facade):
        r = facade.remember("a fact")
        assert r["obsidian"] is False

    def test_tags_passed_as_csv(self, facade, fts_pipeline):
        facade.remember("x", tags=["a", "b"])
        _, kw = fts_pipeline.add_manual.call_args
        assert kw.get("tags") == "a,b"

    def test_fts5_error_tolerated(self):
        pl = MagicMock()
        pl.add_manual.side_effect = RuntimeError("db locked")
        f = MemoryFacade(pipeline=pl, enable_vector=False)
        r = f.remember("x")
        assert r["fts5"] is False  # tolerated, not raised


class TestRecall:
    def test_returns_memory_hits(self, facade):
        hits = facade.recall("theme")
        assert all(isinstance(h, MemoryHit) for h in hits)

    def test_fts5_hit_content(self, facade):
        hits = facade.recall("theme")
        assert any("dark mode" in h.content for h in hits)

    def test_dedup(self):
        pl = MagicMock()
        pl.search.return_value = [{"content": "dup"}, {"content": "dup"}]
        f = MemoryFacade(pipeline=pl, enable_vector=False)
        hits = f.recall("x")
        assert len(hits) == 1

    def test_limit_respected(self):
        pl = MagicMock()
        pl.search.return_value = [{"content": f"item {i}"} for i in range(20)]
        f = MemoryFacade(pipeline=pl, enable_vector=False)
        assert len(f.recall("x", limit=5)) == 5

    def test_empty_when_no_pipeline(self):
        f = MemoryFacade(pipeline=None, enable_vector=False)
        assert f.recall("x") == []

    def test_vector_ranked_first(self):
        pl = MagicMock()
        pl.search.return_value = [{"content": "kw hit"}]
        vec = MagicMock()
        entry = MagicMock(); entry.text = "semantic hit"; entry.score = 0.9; entry.category = "g"
        vec.recall.return_value = [entry]
        f = MemoryFacade(pipeline=pl, vector=vec)
        hits = f.recall("q")
        assert hits[0].source == "vector"


class TestContext:
    def test_context_is_string(self, facade):
        assert isinstance(facade.context("theme"), str)

    def test_context_empty_no_hits(self):
        f = MemoryFacade(pipeline=None, enable_vector=False)
        assert f.context("x") == ""

    def test_context_includes_content(self, facade):
        assert "dark mode" in facade.context("theme")


class TestIntrospection:
    def test_backends_available(self, facade):
        b = facade.backends_available()
        assert b["fts5"] is True
        assert b["vector"] is False

    def test_stats(self, facade):
        s = facade.stats()
        assert s["fts5_count"] == 3
        assert "backends" in s


class TestSingleton:
    def test_get_singleton_returns_same(self):
        import core.memory_facade as mf
        mf._facade_singleton = None
        a = get_memory_facade(pipeline=MagicMock())
        b = get_memory_facade()
        assert a is b

    def test_singleton_backfills_pipeline(self):
        import core.memory_facade as mf
        mf._facade_singleton = None
        a = get_memory_facade(pipeline=None, enable_vector=False)
        pl = MagicMock()
        get_memory_facade(pipeline=pl)
        assert a.pipeline is pl
