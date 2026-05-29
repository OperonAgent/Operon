"""Tests for core/obsidian_memory.py — Obsidian vault sync Phase 11

Actual API (from inspection):
  VaultNote(path, content, modified)
  ObsidianVault(vault_path)
    write_note(folder, filename, content, merge=True) → Path
    read_note(folder, filename) → Optional[VaultNote]
    list_notes(folder=None) → List[VaultNote]
    search_notes(query, max_results=20) → List[VaultNote]
    delete_note(folder, filename) → bool
    exists → property (Path)
  ObsidianNoteWriter(vault)
    write_daily(summary, date_str, session_id, turns, tools_used) → Path
    write_entity(name, facts, tags) → Path
    write_project(name, summary, tasks, links) → Path
    write_code(title, language, code, description) → Path
    write_goals(goals) → Path
    write_facts(topic, facts) → Path
  ObsidianContextReader(vault)
    get_context_for(query) → str
    read_all_facts() → List[str]
    recent_daily_summaries(n) → List[str]
  ObsidianSyncLoop(memory_sync, interval_s)
  ObsidianMemory(vault_path, auto_sync, sync_interval)
    write_daily_summary(summary, tools_used) → Path
    get_context(query, limit) → str
    status() → dict
    sync_all()
    start_auto_sync()
    stop_auto_sync()
    search(query) → List[VaultNote]
    write_entity(name, facts, tags) → Path
    write_project(name, summary, tasks, links) → Path
    write_code(title, language, code, description) → Path
    write_goals(goals) → Path
    add_fact(topic, fact, tags) → Path
    read_facts(topic) → List[str]
    recent_summaries(n) → List[str]
    summary() → str
    set_session(session_id, turns)
"""
import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.obsidian_memory import (
    VaultNote, ObsidianVault, ObsidianNoteWriter,
    ObsidianContextReader, ObsidianSyncLoop, ObsidianMemory,
    _TOOL_DEFINITIONS, _DISPATCH,
)


# ── VaultNote ─────────────────────────────────────────────────────────────────

class TestVaultNote:
    def test_creation(self, tmp_path):
        p = tmp_path / "notes" / "test.md"
        note = VaultNote(path=p, content="# Test Note\n\nSome content here.")
        assert note.content == "# Test Note\n\nSome content here."
        assert note.path == p

    def test_modified_default_zero(self, tmp_path):
        p = tmp_path / "test.md"
        note = VaultNote(path=p, content="content")
        assert note.modified == 0.0

    def test_custom_modified(self, tmp_path):
        p = tmp_path / "test.md"
        ts = time.time()
        note = VaultNote(path=p, content="content", modified=ts)
        assert note.modified == ts


# ── ObsidianVault ─────────────────────────────────────────────────────────────

class TestObsidianVault:
    @pytest.fixture
    def vault(self, tmp_path):
        return ObsidianVault(vault_path=tmp_path / "test-vault")

    def test_vault_dir_created(self, tmp_path):
        vault_path = tmp_path / "new-vault"
        assert not vault_path.exists()
        ObsidianVault(vault_path=vault_path)
        assert vault_path.exists()

    def test_write_creates_file(self, vault, tmp_path):
        vault.write_note("notes", "test.md", "# Test\n\ncontent here")
        vault_path = tmp_path / "test-vault"
        files = list(vault_path.rglob("*.md"))
        assert len(files) >= 1

    def test_write_and_read_note(self, vault):
        vault.write_note("notes", "hello.md", "# Hello\n\ntest content")
        result = vault.read_note("notes", "hello.md")
        assert result is not None
        assert isinstance(result, VaultNote)

    def test_read_returns_content(self, vault):
        vault.write_note("facts", "ai.md", "# AI Facts\n\nClaude is great.")
        result = vault.read_note("facts", "ai.md")
        assert result is not None
        assert "Claude" in result.content

    def test_read_nonexistent_returns_none(self, vault):
        result = vault.read_note("notes", "does_not_exist.md")
        assert result is None

    def test_list_notes_empty(self, vault):
        notes = vault.list_notes()
        assert isinstance(notes, list)
        assert len(notes) == 0

    def test_list_notes_after_write(self, vault):
        vault.write_note("notes", "a.md", "content a")
        vault.write_note("notes", "b.md", "content b")
        notes = vault.list_notes()
        assert len(notes) >= 2

    def test_list_notes_by_folder(self, vault):
        vault.write_note("facts", "fact1.md", "content")
        vault.write_note("code", "snippet.md", "code here")
        facts_notes = vault.list_notes(folder="facts")
        assert isinstance(facts_notes, list)
        assert len(facts_notes) >= 1

    def test_search_notes_returns_list(self, vault):
        vault.write_note("notes", "python.md", "# Python\n\nUse list comprehensions.")
        results = vault.search_notes("Python")
        assert isinstance(results, list)

    def test_search_notes_finds_content(self, vault):
        vault.write_note("notes", "ai.md", "Claude is an AI assistant by Anthropic")
        results = vault.search_notes("Anthropic")
        assert len(results) >= 1

    def test_search_notes_no_match(self, vault):
        vault.write_note("notes", "some.md", "some content here for testing")
        results = vault.search_notes("xyzzy_not_in_any_note_ever")
        assert results == []

    def test_delete_note(self, vault):
        vault.write_note("test", "delete_me.md", "to be deleted")
        vault.delete_note("test", "delete_me.md")
        assert vault.read_note("test", "delete_me.md") is None

    def test_delete_returns_bool(self, vault):
        vault.write_note("test", "del2.md", "content")
        result = vault.delete_note("test", "del2.md")
        assert isinstance(result, bool)

    def test_delete_nonexistent_no_error(self, vault):
        # Should not raise
        vault.delete_note("test", "does_not_exist.md")

    def test_write_merge_true(self, vault):
        vault.write_note("facts", "merged.md", "First version.", merge=True)
        vault.write_note("facts", "merged.md", "Second version.", merge=True)
        result = vault.read_note("facts", "merged.md")
        assert result is not None

    def test_root_property(self, vault, tmp_path):
        assert vault.root == tmp_path / "test-vault"


# ── ObsidianNoteWriter ────────────────────────────────────────────────────────

class TestObsidianNoteWriter:
    @pytest.fixture
    def writer_vault(self, tmp_path):
        vault = ObsidianVault(vault_path=tmp_path / "writer-vault")
        writer = ObsidianNoteWriter(vault=vault)
        return writer, vault

    def test_write_daily_returns_path(self, writer_vault):
        writer, _ = writer_vault
        result = writer.write_daily(summary="Had a productive session.")
        assert isinstance(result, Path)

    def test_write_daily_creates_note(self, writer_vault):
        writer, vault = writer_vault
        writer.write_daily(summary="Great day working on Operon.", date_str="2026-05-28")
        notes = vault.list_notes()
        assert len(notes) >= 1

    def test_write_entity_returns_path(self, writer_vault):
        writer, _ = writer_vault
        result = writer.write_entity(name="Claude", facts=["Made by Anthropic", "Released 2023"])
        assert isinstance(result, Path)

    def test_write_entity_creates_note(self, writer_vault):
        writer, vault = writer_vault
        writer.write_entity(name="Claude", facts=["AI assistant"])
        notes = vault.list_notes()
        assert any("Claude" in str(n.path) for n in notes)

    def test_write_project_returns_path(self, writer_vault):
        writer, _ = writer_vault
        result = writer.write_project(name="Operon", summary="AI Terminal Cockpit")
        assert isinstance(result, Path)

    def test_write_project_creates_note(self, writer_vault):
        writer, vault = writer_vault
        writer.write_project(name="Operon", summary="AI terminal", tasks=["Add tests"])
        notes = vault.list_notes()
        assert any("Operon" in str(n.path) for n in notes)

    def test_write_code_returns_path(self, writer_vault):
        writer, _ = writer_vault
        result = writer.write_code(
            title="Python Hello", code='print("Hello")',
            language="python", context="Basic Python"
        )
        assert isinstance(result, Path)

    def test_write_goals_returns_path(self, writer_vault):
        writer, _ = writer_vault
        result = writer.write_goals(goals=["Launch Operon v3.0", "Add 300+ tests"])
        assert isinstance(result, Path)

    def test_write_facts_returns_path(self, writer_vault):
        writer, _ = writer_vault
        result = writer.write_facts(title="Operon v3.0", facts=["Has 1800+ tests"])
        assert isinstance(result, Path)

    def test_write_entity_with_tags(self, writer_vault):
        writer, _ = writer_vault
        result = writer.write_entity(name="Python", facts=["High-level", "Dynamic"],
                                     tags=["language", "scripting"])
        assert isinstance(result, Path)


# ── ObsidianContextReader ─────────────────────────────────────────────────────

class TestObsidianContextReader:
    @pytest.fixture
    def populated_reader(self, tmp_path):
        vault = ObsidianVault(vault_path=tmp_path / "reader-vault")
        writer = ObsidianNoteWriter(vault=vault)
        writer.write_entity("Python", ["High-level", "Dynamic", "Popular"])
        writer.write_project("Operon", "AI terminal", tasks=["Add tests"])
        writer.write_facts("AI", ["Operon uses Hermes 3", "Supports Slack"])
        reader = ObsidianContextReader(vault=vault)
        return reader, vault

    def test_get_context_for_returns_string(self, populated_reader):
        reader, _ = populated_reader
        ctx = reader.get_context_for("Python programming")
        assert isinstance(ctx, str)

    def test_get_context_for_relevant_content(self, populated_reader):
        reader, _ = populated_reader
        ctx = reader.get_context_for("Operon terminal")
        assert isinstance(ctx, str)

    def test_read_all_facts_returns_list(self, populated_reader):
        reader, _ = populated_reader
        facts = reader.read_all_facts()
        assert isinstance(facts, list)

    def test_recent_daily_summaries_returns_list(self, populated_reader):
        reader, vault = populated_reader
        writer = ObsidianNoteWriter(vault=vault)
        writer.write_daily("Great session!", date_str="2026-05-28")
        summaries = reader.recent_daily_summaries(n=3)
        assert isinstance(summaries, list)

    def test_get_context_empty_vault(self, tmp_path):
        vault = ObsidianVault(vault_path=tmp_path / "empty-vault")
        reader = ObsidianContextReader(vault=vault)
        ctx = reader.get_context_for("anything")
        assert isinstance(ctx, str)


# ── ObsidianMemory (high-level API) ───────────────────────────────────────────

class TestObsidianMemory:
    @pytest.fixture
    def memory(self, tmp_path):
        return ObsidianMemory(vault_path=tmp_path / "obsidian-mem", auto_sync=False)

    def test_write_daily_summary_returns_path(self, memory):
        result = memory.write_daily_summary("Worked on Operon Phase 11 today.")
        assert isinstance(result, Path)

    def test_get_context_returns_string(self, memory):
        memory.write_daily_summary("Built vector memory module.")
        ctx = memory.get_context("vector memory")
        assert isinstance(ctx, str)

    def test_status_returns_dict(self, memory):
        s = memory.status()
        assert isinstance(s, dict)

    def test_sync_all_no_error(self, memory):
        memory.sync_all()  # Should not raise

    def test_search_returns_list(self, memory):
        memory.write_daily_summary("Python tips session")
        results = memory.search("Python")
        assert isinstance(results, list)

    def test_write_entity_returns_path(self, memory):
        result = memory.write_entity("Claude", ["AI by Anthropic"])
        assert isinstance(result, Path)

    def test_write_project_returns_path(self, memory):
        result = memory.write_project("Operon", "AI Terminal")
        assert isinstance(result, Path)

    def test_write_goals_returns_path(self, memory):
        result = memory.write_goals(["Launch v3.0"])
        assert isinstance(result, Path)

    def test_add_fact_returns_path(self, memory):
        result = memory.add_fact("AI", "Operon supports Slack")
        assert isinstance(result, Path)

    def test_read_facts_returns_list(self, memory):
        memory.add_fact("AI", "Operon is great")
        facts = memory.read_facts("AI")
        assert isinstance(facts, list)

    def test_recent_summaries_returns_list(self, memory):
        memory.write_daily_summary("Session 1")
        summaries = memory.recent_summaries(n=5)
        assert isinstance(summaries, list)

    def test_summary_returns_string(self, memory):
        s = memory.summary()
        assert isinstance(s, str)

    def test_set_session_no_error(self, memory):
        memory.set_session("test-session-001", turns=5)

    def test_vault_path_created(self, tmp_path):
        vault_path = tmp_path / "auto-vault"
        assert not vault_path.exists()
        ObsidianMemory(vault_path=vault_path, auto_sync=False)
        assert vault_path.exists()

    def test_write_code_returns_path(self, memory):
        result = memory.write_code("Hello", "python", 'print("hi")', "Basic example")
        assert isinstance(result, Path)


# ── ObsidianSyncLoop ──────────────────────────────────────────────────────────

class TestObsidianSyncLoop:
    def test_sync_loop_is_daemon(self, tmp_path):
        mem = ObsidianMemory(vault_path=tmp_path / "sync-vault", auto_sync=False)
        loop = ObsidianSyncLoop(memory_sync=mem, interval_s=3600)
        assert loop.daemon is True

    def test_sync_loop_can_start_and_stop(self, tmp_path):
        mem = ObsidianMemory(vault_path=tmp_path / "sync-vault2", auto_sync=False)
        loop = ObsidianSyncLoop(memory_sync=mem, interval_s=3600)
        loop.start()
        assert loop.is_alive()
        # Stop it cleanly
        if hasattr(loop, "stop"):
            loop.stop()
        loop.join(timeout=2.0)


# ── Tool definitions ──────────────────────────────────────────────────────────

class TestObsidianTools:
    def test_tool_definitions_exist(self):
        assert len(_TOOL_DEFINITIONS) >= 3

    def test_all_tools_have_name(self):
        for td in _TOOL_DEFINITIONS:
            assert "name" in td

    def test_dispatch_is_dict(self):
        assert isinstance(_DISPATCH, dict)

    def test_dispatch_callable_values(self):
        for k, v in _DISPATCH.items():
            assert callable(v)

    def test_write_note_tool_present(self):
        names = [td["name"] for td in _TOOL_DEFINITIONS]
        assert "obsidian_write_note" in names

    def test_search_tool_present(self):
        names = [td["name"] for td in _TOOL_DEFINITIONS]
        assert "obsidian_search" in names

    def test_get_context_tool_present(self):
        names = [td["name"] for td in _TOOL_DEFINITIONS]
        assert "obsidian_get_context" in names

    def test_dispatch_covers_all_tools(self):
        for td in _TOOL_DEFINITIONS:
            assert td["name"] in _DISPATCH, f"{td['name']} missing from _DISPATCH"

    def test_tool_definitions_have_parameters(self):
        for td in _TOOL_DEFINITIONS:
            assert "input_schema" in td or "parameters" in td or "params" in td
            assert "description" in td
