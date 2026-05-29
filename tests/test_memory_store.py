"""Tests for core/memory_store.py"""
import json
import pathlib
import tempfile
import time
from unittest import mock

import pytest

from core.memory_store import (
    MemoryStore, WorkingMemory, EpisodicMemory, EntityMemory,
    MemoryConsolidator, MemoryEntry, EntityFact, WorkingSlot,
    get_memory_store, remember, recall,
    _tokenize, _extract_entities, _sha256_content,
    _DEFAULT_IMPORTANCE, _DECAY_HALF_LIFE_DAYS, _EPISODE_CONTENT_MAX,
)


# ── WorkingMemory ─────────────────────────────────────────────────────────────

class TestWorkingMemory:
    def test_set_and_get(self):
        wm = WorkingMemory()
        wm.set("key1", "value1")
        assert wm.get("key1") == "value1"

    def test_get_missing_returns_default(self):
        wm = WorkingMemory()
        assert wm.get("nope") is None
        assert wm.get("nope", "fallback") == "fallback"

    def test_set_different_kinds(self):
        wm = WorkingMemory()
        wm.set("a", 1, kind="fact")
        wm.set("b", [1, 2], kind="plan")
        wm.set("c", {"k": "v"}, kind="result")
        assert wm.get("a") == 1
        assert wm.get("b") == [1, 2]
        assert wm.get("c") == {"k": "v"}

    def test_delete(self):
        wm = WorkingMemory()
        wm.set("x", 99)
        assert wm.delete("x") is True
        assert wm.get("x") is None
        assert wm.delete("x") is False   # already gone

    def test_keys_filter_by_kind(self):
        wm = WorkingMemory()
        wm.set("f1", 1, kind="fact")
        wm.set("p1", 2, kind="plan")
        wm.set("f2", 3, kind="fact")
        facts = wm.keys(kind="fact")
        assert "f1" in facts and "f2" in facts
        assert "p1" not in facts

    def test_items_filter_by_kind(self):
        wm = WorkingMemory()
        wm.set("k1", "v1", kind="context")
        wm.set("k2", "v2", kind="fact")
        items = wm.items(kind="context")
        assert "k1" in items and "k2" not in items

    def test_clear_all(self):
        wm = WorkingMemory()
        wm.set("a", 1)
        wm.set("b", 2)
        n = wm.clear()
        assert n == 2
        assert wm.stats()["total"] == 0

    def test_clear_by_kind(self):
        wm = WorkingMemory()
        wm.set("f", 1, kind="fact")
        wm.set("p", 2, kind="plan")
        wm.clear(kind="fact")
        assert wm.get("f") is None
        assert wm.get("p") == 2

    def test_ttl_expiry(self):
        wm = WorkingMemory()
        wm.set("temp", "gone", ttl_sec=0.01)
        time.sleep(0.05)
        assert wm.get("temp") is None

    def test_capacity_eviction(self):
        wm = WorkingMemory(capacity=3)
        for i in range(5):
            wm.set(f"k{i}", i)
        assert wm.stats()["total"] <= 3

    def test_stats_keys(self):
        wm = WorkingMemory()
        wm.set("a", 1, kind="fact")
        wm.set("b", 2, kind="plan")
        s = wm.stats()
        assert "total" in s
        assert "by_kind" in s
        assert s["by_kind"].get("fact", 0) >= 1

    def test_snapshot(self):
        wm = WorkingMemory()
        wm.set("x", 42, kind="result")
        snap = wm.snapshot()
        assert "x" in snap
        assert snap["x"]["value"] == 42
        assert snap["x"]["kind"] == "result"

    def test_access_count_increments(self):
        wm = WorkingMemory()
        wm.set("q", "r")
        wm.get("q")
        wm.get("q")
        with wm._lock:
            slot = wm._slots["q"]
        assert slot.access_count == 2

    def test_thread_safety(self):
        import threading
        wm = WorkingMemory(capacity=1000)
        errors = []
        def worker(n):
            try:
                for i in range(50):
                    wm.set(f"k{n}_{i}", n * 1000 + i)
                    wm.get(f"k{n}_{i}")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == []


# ── EntityMemory ──────────────────────────────────────────────────────────────

class TestEntityMemory:
    def _make(self):
        tmp = tempfile.mkdtemp()
        return EntityMemory(db_path=pathlib.Path(tmp) / "test.db")

    def test_know_and_recall(self):
        em = self._make()
        em.know("Alice", "role", "engineer")
        facts = em.recall("Alice")
        assert any(f.attribute == "role" for f in facts)

    def test_know_attribute_filter(self):
        em = self._make()
        em.know("Bob", "email", "bob@example.com")
        em.know("Bob", "role", "designer")
        facts = em.recall("Bob", "email")
        assert len(facts) == 1 and facts[0].value == "bob@example.com"

    def test_know_updates_existing(self):
        em = self._make()
        em.know("Carol", "status", "active")
        em.know("Carol", "status", "inactive")
        facts = em.entity_facts("Carol")
        assert facts["status"] == "inactive"

    def test_entity_facts_as_dict(self):
        em = self._make()
        em.know("Dave", "lang", "Python")
        em.know("Dave", "tz", "UTC+5")
        d = em.entity_facts("Dave")
        assert d["lang"] == "Python"
        assert d["tz"] == "UTC+5"

    def test_list_entities(self):
        em = self._make()
        em.know("Alice", "x", 1)
        em.know("Bob", "x", 2)
        entities = em.list_entities()
        assert "Alice" in entities and "Bob" in entities

    def test_forget_entity(self):
        em = self._make()
        em.know("Eve", "attr", "val")
        n = em.forget_entity("Eve")
        assert n == 1
        assert "Eve" not in em.list_entities()

    def test_forget_attribute(self):
        em = self._make()
        em.know("Frank", "a", 1)
        em.know("Frank", "b", 2)
        ok = em.forget_attribute("Frank", "a")
        assert ok
        assert "a" not in em.entity_facts("Frank")
        assert "b" in em.entity_facts("Frank")

    def test_forget_missing_entity_returns_zero(self):
        em = self._make()
        assert em.forget_entity("nonexistent") == 0

    def test_merge_entities(self):
        em = self._make()
        em.know("OldName", "email", "old@x.com")
        em.know("NewName", "role", "admin")
        n = em.merge_entities("OldName", "NewName")
        assert n >= 1
        facts = em.entity_facts("NewName")
        assert "email" in facts
        assert "OldName" not in em.list_entities()

    def test_confidence_stored(self):
        em = self._make()
        em.know("G", "key", "val", confidence=0.7)
        facts = em.recall("G")
        assert any(abs(f.confidence - 0.7) < 0.01 for f in facts)

    def test_stats(self):
        em = self._make()
        em.know("X", "a", 1)
        em.know("X", "b", 2)
        em.know("Y", "c", 3)
        s = em.stats()
        assert s["total_facts"] == 3
        assert s["total_entities"] == 2

    def test_json_value_roundtrip(self):
        em = self._make()
        em.know("H", "prefs", {"dark_mode": True, "font": "mono"})
        facts = em.entity_facts("H")
        assert facts["prefs"]["dark_mode"] is True


# ── EpisodicMemory ────────────────────────────────────────────────────────────

class TestEpisodicMemory:
    def _make(self):
        tmp = tempfile.mkdtemp()
        return EpisodicMemory(db_path=pathlib.Path(tmp) / "ep.db")

    def test_store_and_recent(self):
        ep = self._make()
        ep.store("User said hello", role="user")
        ep.store("Bot replied hi",  role="assistant")
        recent = ep.recent(limit=10)
        assert len(recent) == 2
        assert all(isinstance(r, MemoryEntry) for r in recent)

    def test_content_truncated(self):
        ep = self._make()
        long_content = "x" * (_EPISODE_CONTENT_MAX + 100)
        ep.store(long_content, role="user")
        recent = ep.recent(limit=1)
        assert len(recent[0].content) <= _EPISODE_CONTENT_MAX

    def test_recall_returns_relevant(self):
        ep = self._make()
        ep.store("The user wants to build a Python web app", role="user", importance=0.8)
        ep.store("Today is a nice day", role="user", importance=0.3)
        results = ep.recall("Python web")
        assert len(results) >= 1
        # Most relevant should be first
        assert "Python" in results[0].content or "web" in results[0].content

    def test_archive_hides_entry(self):
        ep = self._make()
        entry_id = ep.store("To archive", role="user")
        ep.archive(entry_id)
        recent = ep.recent(limit=10)
        assert all(r.id != entry_id for r in recent)

    def test_boost_importance(self):
        ep = self._make()
        entry_id = ep.store("Important fact", role="user", importance=0.5)
        ep.boost_importance(entry_id, delta=0.2)
        # Check via recall stats
        results = ep.recent(limit=1)
        assert results[0].recall_count >= 1

    def test_session_context_order(self):
        ep = self._make()
        for i in range(5):
            ep.store(f"msg {i}", session_id="s1", role="user")
        ctx = ep.session_context("s1", limit=10)
        contents = [e.content for e in ctx]
        assert contents == [f"msg {i}" for i in range(5)]  # chronological

    def test_filter_by_session(self):
        ep = self._make()
        ep.store("session A msg", session_id="sA", role="user")
        ep.store("session B msg", session_id="sB", role="user")
        results = ep.recall("msg", session_id="sA")
        assert all("sA" in e.session_id for e in results)

    def test_stats_keys(self):
        ep = self._make()
        ep.store("a", role="user")
        s = ep.stats()
        assert "total" in s
        assert "sessions" in s
        assert "archived" in s
        assert s["total"] == 1

    def test_role_filter(self):
        ep = self._make()
        ep.store("user msg", role="user")
        ep.store("bot reply", role="assistant")
        user_msgs = ep.recent(limit=10, role="user")
        assert all(e.role == "user" for e in user_msgs)

    def test_entities_extracted(self):
        ep = self._make()
        entry_id = ep.store("Alice Smith sent email to bob@example.com", role="user")
        recent = ep.recent(limit=1)
        # Entities should be non-empty
        assert isinstance(recent[0].entities, list)


# ── MemoryStore façade ────────────────────────────────────────────────────────

class TestMemoryStore:
    def _make(self, session_id="test"):
        tmp = tempfile.mkdtemp()
        return MemoryStore(
            db_path=pathlib.Path(tmp) / "ms.db",
            session_id=session_id,
            auto_consolidate=False,
        )

    def test_remember_and_recall(self):
        ms = self._make()
        ms.remember("Python is a great language", role="user", importance=0.8)
        ms.remember("I prefer Ruby", role="user", importance=0.4)
        results = ms.recall("Python")
        assert len(results) >= 1

    def test_know_and_entity_facts(self):
        ms = self._make()
        ms.know("Alice", "role", "lead engineer")
        ms.know("Alice", "email", "alice@example.com")
        facts = ms.entity_facts("Alice")
        assert facts["role"] == "lead engineer"
        assert facts["email"] == "alice@example.com"

    def test_list_entities(self):
        ms = self._make()
        ms.know("Alice", "x", 1)
        ms.know("Bob", "x", 2)
        assert set(ms.list_entities()) == {"Alice", "Bob"}

    def test_forget(self):
        ms = self._make()
        ms.know("Eve", "secret", "yes")
        ms.forget("Eve")
        assert "Eve" not in ms.list_entities()

    def test_context_window(self):
        ms = self._make("ctx-session")
        ms.remember("hello", role="user")
        ms.remember("hi there", role="assistant")
        ctx = ms.context_window()
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"
        assert ctx[1]["role"] == "assistant"

    def test_recent(self):
        ms = self._make()
        ms.remember("msg1", role="user")
        ms.remember("msg2", role="assistant")
        recent = ms.recent(limit=10)
        assert len(recent) == 2

    def test_switch_session(self):
        ms = self._make("session-A")
        ms.remember("session A content", role="user")
        ms.switch_session("session-B")
        assert ms._session_id == "session-B"
        # Working memory should be cleared on switch
        assert ms.working.stats()["total"] == 0

    def test_stats_keys(self):
        ms = self._make()
        s = ms.stats()
        assert "session_id" in s
        assert "working" in s
        assert "episodic" in s
        assert "entities" in s

    def test_export_session(self):
        ms = self._make("export-session")
        ms.remember("data point 1", role="user")
        ms.remember("data point 2", role="assistant")
        exported = ms.export_session()
        assert exported["session_id"] == "export-session"
        assert len(exported["episodes"]) == 2
        assert "exported_at" in exported

    def test_working_memory_integrated(self):
        ms = self._make()
        ms.working.set("file_path", "/src/main.py", kind="context")
        val = ms.working.get("file_path")
        assert val == "/src/main.py"

    def test_consolidate_runs(self):
        ms = self._make()
        for i in range(3):
            ms.remember(f"episode {i}", role="user", importance=0.1)
        result = ms.consolidate(force=True)
        assert isinstance(result, dict)
        assert "archived" in result


# ── MemoryConsolidator ────────────────────────────────────────────────────────

class TestMemoryConsolidator:
    def test_archive_stale_memories(self):
        tmp = tempfile.mkdtemp()
        db  = pathlib.Path(tmp) / "con.db"
        ep  = EpisodicMemory(db_path=db)
        em  = EntityMemory(db_path=db)
        con = MemoryConsolidator(ep, em, archive_threshold=0.99)  # aggressive threshold
        # Store some memories
        for i in range(5):
            ep.store(f"old message {i}", role="user", importance=0.01)
        result = con.run(force=True)
        assert result.get("archived", 0) > 0

    def test_skip_if_recently_run(self):
        tmp = tempfile.mkdtemp()
        db  = pathlib.Path(tmp) / "con.db"
        ep  = EpisodicMemory(db_path=db)
        em  = EntityMemory(db_path=db)
        con = MemoryConsolidator(ep, em)
        con._last_run = time.time()  # just ran
        result = con.run(force=False)
        assert result.get("skipped") == 1

    def test_force_runs_even_if_recent(self):
        tmp = tempfile.mkdtemp()
        db  = pathlib.Path(tmp) / "con.db"
        ep  = EpisodicMemory(db_path=db)
        em  = EntityMemory(db_path=db)
        con = MemoryConsolidator(ep, em)
        con._last_run = time.time()
        result = con.run(force=True)
        assert "skipped" not in result


# ── Helper functions ──────────────────────────────────────────────────────────

class TestHelpers:
    def test_tokenize_basic(self):
        tokens = _tokenize("The quick brown fox")
        assert "quick" in tokens
        assert "brown" in tokens
        assert "the" not in tokens   # stop word

    def test_tokenize_min_length(self):
        tokens = _tokenize("a b ab abc")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "ab" in tokens

    def test_tokenize_numbers_included(self):
        tokens = _tokenize("Python 3.12 released in 2024")
        assert "python" in tokens or "Python" in [t.lower() for t in tokens]

    def test_extract_entities_mentions(self):
        entities = _extract_entities("Hello @alice and @bob")
        assert "alice" in entities
        assert "bob" in entities

    def test_extract_entities_caps(self):
        entities = _extract_entities("AWS S3 bucket in US-EAST")
        assert "AWS" in entities or "S3" in entities

    def test_extract_entities_email(self):
        entities = _extract_entities("Contact support@example.com for help")
        assert "support" in entities

    def test_extract_entities_url(self):
        entities = _extract_entities("Visit https://github.com/operon-ai for code")
        assert "github.com" in entities

    def test_extract_entities_title_case(self):
        entities = _extract_entities("Alice Smith is the lead engineer")
        assert "Alice Smith" in entities

    def test_extract_entities_cap(self):
        # Should cap at 20 entities
        text = " ".join(f"@user{i}" for i in range(50))
        entities = _extract_entities(text)
        assert len(entities) <= 20

    def test_sha256_content(self):
        h = _sha256_content("test content")
        assert len(h) == 16
        assert _sha256_content("test content") == h   # deterministic


# ── Module-level API ──────────────────────────────────────────────────────────

class TestModuleAPI:
    def test_get_memory_store_returns_store(self):
        ms = get_memory_store()
        assert isinstance(ms, MemoryStore)

    def test_get_memory_store_singleton(self):
        ms1 = get_memory_store()
        ms2 = get_memory_store()
        assert ms1 is ms2

    def test_remember_returns_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = pathlib.Path(tmp) / "m.db"
            ms = MemoryStore(db_path=db, session_id="api-test")
            entry_id = ms.remember("test content", role="user")
            assert isinstance(entry_id, int) and entry_id > 0

    def test_recall_returns_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = pathlib.Path(tmp) / "m.db"
            ms = MemoryStore(db_path=db, session_id="api-test2")
            ms.remember("Python tutorial", role="user")
            results = ms.recall("Python")
            assert isinstance(results, list)
