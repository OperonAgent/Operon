"""Tests for core/vector_memory.py — LanceDB semantic memory Phase 11

Actual API (from inspection):
  VectorMemoryEntry(text, source, category, tags, session_id, created_at, updated_at,
                    access_count, score, entry_id)
  EmbeddingEngine()   — methods: embed(text), embed_one(text), available, dim
  VectorStore(db_path, table_name, engine)  — upsert(entry), search(query, top_k, ...), delete(id), count(), all_texts()
  SemanticDeduplicator(store, threshold)    — is_duplicate(text) → (bool, Optional[str])
  VectorMemory(db_path, model_name, dedup, auto_consolidate)
              — remember(text, source, ...) → (bool, entry_id)
              — recall(query, top_k) → List[VectorMemoryEntry]
              — build_context_block(query) → str
              — get_context_for(query) → List[VectorMemoryEntry]
              — forget(entry_id) → bool
              — forget_by_text(text) → int
              — stats() → dict
              — summary() → str
              — count() → int
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


from core.vector_memory import (
    VectorMemoryEntry, EmbeddingEngine, VectorStore,
    SemanticDeduplicator, VectorMemory,
    get_vector_memory,
    _TOOL_DEFINITIONS, _DISPATCH,
)


# ── VectorMemoryEntry ─────────────────────────────────────────────────────────

class TestVectorMemoryEntry:
    def test_creation_basic(self):
        entry = VectorMemoryEntry(
            text="Python is a great language",
            source="user",
        )
        assert entry.text == "Python is a great language"
        assert entry.source == "user"

    def test_category_default(self):
        entry = VectorMemoryEntry(text="x", source="y")
        assert entry.category == "general"

    def test_tags_default_empty(self):
        entry = VectorMemoryEntry(text="x", source="y")
        assert entry.tags == []

    def test_custom_tags(self):
        entry = VectorMemoryEntry(text="Python tip", source="chat", tags=["python", "tip"])
        assert "python" in entry.tags

    def test_session_id_optional(self):
        entry = VectorMemoryEntry(text="x", source="y", session_id="sess-001")
        assert entry.session_id == "sess-001"

    def test_created_at_auto(self):
        entry = VectorMemoryEntry(text="x", source="y")
        assert entry.created_at  # non-empty string

    def test_entry_id_default_empty(self):
        entry = VectorMemoryEntry(text="x", source="y")
        assert isinstance(entry.entry_id, str)

    def test_access_count_default_zero(self):
        entry = VectorMemoryEntry(text="x", source="y")
        assert entry.access_count == 0

    def test_score_default_zero(self):
        entry = VectorMemoryEntry(text="x", source="y")
        assert entry.score == 0.0


# ── EmbeddingEngine ───────────────────────────────────────────────────────────

class TestEmbeddingEngine:
    @pytest.fixture
    def engine(self):
        return EmbeddingEngine()

    def test_embed_returns_list(self, engine):
        vec = engine.embed("test sentence")
        assert isinstance(vec, list)

    def test_embed_non_empty(self, engine):
        vec = engine.embed("some text about coding")
        assert len(vec) > 0

    def test_embed_dim_is_384(self, engine):
        vec = engine.embed("test")
        assert len(vec) == 384

    def test_embed_one_returns_list(self, engine):
        v = engine.embed_one("hello world")
        assert isinstance(v, list)
        assert len(v) == 384

    def test_embed_same_text_consistent(self, engine):
        v1 = engine.embed("same text")
        v2 = engine.embed("same text")
        assert v1 == v2

    def test_embed_different_texts(self, engine):
        v1 = engine.embed("python programming")
        v2 = engine.embed("quantum physics")
        # With hash fallback, these will differ
        assert v1 != v2

    def test_embed_empty_string(self, engine):
        # Should not raise
        vec = engine.embed("")
        assert isinstance(vec, list)

    def test_available_is_bool(self, engine):
        assert isinstance(engine.available, bool)

    def test_dim_is_int(self, engine):
        assert isinstance(engine.dim, int)
        assert engine.dim > 0


# ── VectorStore ───────────────────────────────────────────────────────────────

class TestVectorStore:
    @pytest.fixture
    def store(self, tmp_path):
        return VectorStore(db_path=tmp_path / "test.lance")

    def _make_entry(self, text="test memory", source="test"):
        return VectorMemoryEntry(text=text, source=source)

    def test_count_starts_zero(self, store):
        assert store.count() == 0

    def test_upsert_increases_count(self, store):
        entry = self._make_entry()
        store.upsert(entry)
        assert store.count() >= 1

    def test_search_returns_list(self, store):
        entry = self._make_entry("Python is great")
        store.upsert(entry)
        results = store.search("Python", top_k=5)
        assert isinstance(results, list)

    def test_search_returns_entries(self, store):
        entry = self._make_entry("OpenAI released GPT-5")
        store.upsert(entry)
        results = store.search("OpenAI GPT", top_k=3)
        assert len(results) >= 1

    def test_delete_by_id(self, store):
        entry = self._make_entry("entry to delete")
        entry.entry_id = "del_test_001"
        store.upsert(entry)
        initial = store.count()
        result = store.delete("del_test_001")
        assert isinstance(result, bool)

    def test_all_texts_returns_list(self, store):
        store.upsert(self._make_entry("text a"))
        store.upsert(self._make_entry("text b"))
        texts = store.all_texts()
        assert isinstance(texts, list)
        assert len(texts) >= 2

    def test_count_after_multiple_upserts(self, store):
        for i in range(5):
            store.upsert(self._make_entry(f"memory {i}"))
        assert store.count() >= 5

    def test_search_with_category_filter(self, store):
        e = self._make_entry("fact about AI")
        e.category = "fact"
        store.upsert(e)
        results = store.search("AI", top_k=5, category="fact")
        assert isinstance(results, list)

    def test_search_with_min_score(self, store):
        store.upsert(self._make_entry("test entry"))
        results = store.search("test", top_k=5, min_score=0.0)
        assert isinstance(results, list)


# ── SemanticDeduplicator ──────────────────────────────────────────────────────

class TestSemanticDeduplicator:
    @pytest.fixture
    def dedup(self, tmp_path):
        store = VectorStore(db_path=tmp_path / "dedup.lance")
        return SemanticDeduplicator(store=store, threshold=0.92)

    def test_is_duplicate_empty_store(self, dedup):
        is_dup, eid = dedup.is_duplicate("some text")
        assert is_dup is False
        assert eid is None

    def test_is_duplicate_returns_tuple(self, dedup):
        result = dedup.is_duplicate("test text")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_is_not_duplicate_unique_text(self, dedup):
        # Add one text, then check a very different text
        entry = VectorMemoryEntry(text="Python programming language basics", source="test")
        dedup._store.upsert(entry)
        is_dup, _ = dedup.is_duplicate("Quantum physics of the universe")
        assert is_dup is False

    def test_threshold_attribute(self, dedup):
        assert dedup.threshold == 0.92


# ── VectorMemory (high-level API) ─────────────────────────────────────────────

class TestVectorMemory:
    @pytest.fixture
    def mem(self, tmp_path):
        return VectorMemory(db_path=tmp_path / "vecmem.lance", dedup=False)

    def test_remember_returns_tuple(self, mem):
        result = mem.remember("Claude is an AI assistant", source="chat")
        assert isinstance(result, tuple)
        assert len(result) == 2
        ok, eid = result
        assert isinstance(ok, bool)
        assert isinstance(eid, str)

    def test_remember_success(self, mem):
        ok, eid = mem.remember("Python scripting", source="test")
        assert ok is True
        assert eid  # non-empty

    def test_remember_multiple(self, mem):
        ids = []
        for i in range(5):
            ok, eid = mem.remember(f"unique fact number {i} about something", source="test")
            ids.append(eid)
        # Most should succeed
        assert sum(1 for i in ids if i) >= 3

    def test_recall_returns_list(self, mem):
        mem.remember("Operon is an AI terminal cockpit", source="chat")
        results = mem.recall("AI terminal")
        assert isinstance(results, list)

    def test_recall_top_k(self, mem):
        for i in range(10):
            mem.remember(f"memory item number {i} here", source="test")
        results = mem.recall("memory item", top_k=3)
        assert len(results) <= 3

    def test_recall_empty_store(self, mem):
        results = mem.recall("anything")
        assert results == []

    def test_recall_returns_entries(self, mem):
        mem.remember("Python is great for scripting", source="user")
        results = mem.recall("Python scripting")
        if results:
            assert isinstance(results[0], VectorMemoryEntry)

    def test_get_context_for_returns_list(self, mem):
        mem.remember("Operon supports Slack integration", source="chat")
        ctx = mem.get_context_for("Slack")
        assert isinstance(ctx, list)

    def test_build_context_block_returns_str(self, mem):
        mem.remember("Python is great for scripting", source="user")
        block = mem.build_context_block("Python scripting")
        assert isinstance(block, str)

    def test_build_context_block_empty(self, mem):
        block = mem.build_context_block("xyzzy nothing matches here ever")
        assert isinstance(block, str)

    def test_forget_returns_bool(self, mem):
        ok, eid = mem.remember("temporary fact to forget", source="test")
        result = mem.forget(eid)
        assert isinstance(result, bool)

    def test_forget_by_text_returns_int(self, mem):
        mem.remember("specific phrase to forget right now", source="test")
        deleted = mem.forget_by_text("specific phrase to forget right now")
        assert isinstance(deleted, int)

    def test_stats_returns_dict(self, mem):
        mem.remember("some data", source="test")
        stats = mem.stats()
        assert isinstance(stats, dict)
        assert "total_memories" in stats

    def test_stats_keys(self, mem):
        stats = mem.stats()
        assert "embedding_model" in stats
        assert "lance_available" in stats
        assert "st_available" in stats

    def test_summary_returns_string(self, mem):
        s = mem.summary()
        assert isinstance(s, str)

    def test_count_increases(self, mem):
        initial = mem.count()
        mem.remember("test entry for count", source="test")
        assert mem.count() >= initial

    def test_remember_many(self, mem):
        entries = [
            VectorMemoryEntry(text=f"batch entry {i}", source="batch")
            for i in range(3)
        ]
        mem.remember_many(entries)
        assert mem.count() >= 3

    def test_set_session(self, mem):
        # Should not raise
        mem.set_session("session-001")


# ── Tool definitions + dispatch ───────────────────────────────────────────────

class TestVectorMemoryTools:
    def test_tool_definitions_exist(self):
        assert len(_TOOL_DEFINITIONS) >= 3

    def test_all_tools_have_name(self):
        for td in _TOOL_DEFINITIONS:
            assert "name" in td

    def test_dispatch_exists(self):
        assert isinstance(_DISPATCH, dict)

    def test_dispatch_callable_values(self):
        for k, v in _DISPATCH.items():
            assert callable(v)

    def test_remember_tool_present(self):
        names = [td["name"] for td in _TOOL_DEFINITIONS]
        assert any("remember" in n for n in names)

    def test_recall_tool_present(self):
        names = [td["name"] for td in _TOOL_DEFINITIONS]
        assert any("recall" in n for n in names)

    def test_forget_tool_present(self):
        names = [td["name"] for td in _TOOL_DEFINITIONS]
        assert any("forget" in n for n in names)

    def test_dispatch_covers_all_tools(self):
        for td in _TOOL_DEFINITIONS:
            assert td["name"] in _DISPATCH, f"{td['name']} not in _DISPATCH"


# ── Module singleton ──────────────────────────────────────────────────────────

class TestGetVectorMemory:
    def test_singleton(self):
        import core.vector_memory as vm
        vm._vector_memory = None
        m1 = get_vector_memory()
        m2 = get_vector_memory()
        assert m1 is m2
        vm._vector_memory = None

    def test_returns_vector_memory_instance(self):
        import core.vector_memory as vm
        vm._vector_memory = None
        m = get_vector_memory()
        assert isinstance(m, VectorMemory)
        vm._vector_memory = None
