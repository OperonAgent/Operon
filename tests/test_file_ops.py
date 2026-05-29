"""Tests for tools/file_ops.py"""
import os
import pytest
from pathlib import Path

from tools.file_ops import (
    file_read, file_write, file_append, file_delete,
    file_exists, file_info, dir_list, file_patch,
)


# ── file_write / file_read round-trip ────────────────────────────────────────

class TestFileWrite:
    def test_creates_file(self, tmp_path):
        p = str(tmp_path / "hello.txt")
        r = file_write(p, "hello world")
        assert r["success"]
        assert Path(p).exists()

    def test_content_readable(self, tmp_path):
        p = str(tmp_path / "data.txt")
        file_write(p, "test content")
        assert Path(p).read_text() == "test content"

    def test_overwrites_existing(self, tmp_path):
        p = str(tmp_path / "file.txt")
        file_write(p, "original")
        file_write(p, "updated")
        assert Path(p).read_text() == "updated"

    def test_creates_parent_dirs(self, tmp_path):
        p = str(tmp_path / "a" / "b" / "c.txt")
        r = file_write(p, "deep")
        assert r["success"]
        assert Path(p).exists()

    def test_empty_content(self, tmp_path):
        p = str(tmp_path / "empty.txt")
        r = file_write(p, "")
        assert r["success"]
        assert Path(p).stat().st_size == 0

    def test_unicode_content(self, tmp_path):
        p = str(tmp_path / "unicode.txt")
        content = "日本語テスト 🚀 émojis"
        file_write(p, content)
        assert Path(p).read_text(encoding="utf-8") == content


class TestFileRead:
    def test_reads_existing_file(self, tmp_path):
        p = tmp_path / "read_me.txt"
        p.write_text("some content")
        r = file_read(str(p))
        assert r["success"]
        assert r["output"] == "some content"

    def test_missing_file_error(self):
        r = file_read("/nonexistent/path/file.txt")
        assert not r["success"]
        assert r["error"]

    def test_large_file_readable(self, tmp_path):
        p = tmp_path / "large.txt"
        p.write_text("x" * 50_000)
        r = file_read(str(p))
        assert r["success"]
        assert len(r["output"]) >= 50_000


# ── file_append ───────────────────────────────────────────────────────────────

class TestFileAppend:
    def test_appends_to_existing(self, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("line1\n")
        r = file_append(str(p), "line2\n")
        assert r["success"]
        assert p.read_text() == "line1\nline2\n"

    def test_creates_file_if_missing(self, tmp_path):
        p = str(tmp_path / "new.txt")
        r = file_append(p, "first")
        assert r["success"]
        assert Path(p).read_text() == "first"

    def test_multiple_appends(self, tmp_path):
        p = tmp_path / "multi.txt"
        for i in range(5):
            file_append(str(p), f"line{i}\n")
        lines = p.read_text().splitlines()
        assert len(lines) == 5


# ── file_delete ───────────────────────────────────────────────────────────────

class TestFileDelete:
    def test_deletes_file(self, tmp_path):
        p = tmp_path / "to_delete.txt"
        p.write_text("bye")
        r = file_delete(str(p))
        assert r["success"]
        assert not p.exists()

    def test_deletes_directory_recursively(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        (d / "child.txt").write_text("child")
        r = file_delete(str(d))
        assert r["success"]
        assert not d.exists()

    def test_missing_path_error(self):
        r = file_delete("/no/such/path/ever")
        assert not r["success"]


# ── file_exists ───────────────────────────────────────────────────────────────
# file_exists returns {"success": True, "output": {"exists": bool, "kind": ...}, "error": ""}

class TestFileExists:
    def test_existing_file(self, tmp_path):
        p = tmp_path / "yes.txt"
        p.write_text("hi")
        r = file_exists(str(p))
        assert r["success"]
        assert r["output"]["exists"] is True

    def test_missing_file(self):
        r = file_exists("/no/such/file.txt")
        assert r["success"]          # the call succeeds
        assert r["output"]["exists"] is False

    def test_existing_directory(self, tmp_path):
        r = file_exists(str(tmp_path))
        assert r["output"]["exists"] is True

    def test_kind_is_file(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("x")
        r = file_exists(str(p))
        assert r["output"]["kind"] == "file"

    def test_kind_is_directory(self, tmp_path):
        r = file_exists(str(tmp_path))
        assert r["output"]["kind"] == "directory"


# ── file_info ─────────────────────────────────────────────────────────────────
# file_info output keys: path, size, modified, is_dir, mode

class TestFileInfo:
    def test_returns_size(self, tmp_path):
        p = tmp_path / "info.txt"
        p.write_text("12345")
        r = file_info(str(p))
        assert r["success"]
        assert r["output"]["size"] == 5

    def test_is_dir_false_for_file(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("x")
        r = file_info(str(p))
        assert r["output"]["is_dir"] is False

    def test_is_dir_true_for_directory(self, tmp_path):
        r = file_info(str(tmp_path))
        assert r["success"]
        assert r["output"]["is_dir"] is True

    def test_missing_path_error(self):
        r = file_info("/nonexistent/x")
        assert not r["success"]

    def test_modified_field_present(self, tmp_path):
        p = tmp_path / "ts.txt"
        p.write_text("hi")
        r = file_info(str(p))
        assert "modified" in r["output"]


# ── dir_list ──────────────────────────────────────────────────────────────────
# dir_list output: {"tree": "...", ...} or the tree is in output directly

class TestDirList:
    def test_lists_contents(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        r = dir_list(str(tmp_path))
        assert r["success"]
        # tree may be in output["tree"] or output directly
        tree = r["output"]["tree"] if isinstance(r["output"], dict) else r["output"]
        assert "a.txt" in tree
        assert "b.txt" in tree

    def test_missing_dir_error(self):
        r = dir_list("/no/such/dir")
        assert not r["success"]

    def test_nested_structure_shown(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("n")
        r = dir_list(str(tmp_path), max_depth=2)
        assert r["success"]
        tree = r["output"]["tree"] if isinstance(r["output"], dict) else r["output"]
        assert "sub" in tree

    def test_returns_success_for_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        r = dir_list(str(empty))
        assert r["success"]


# ── file_patch ────────────────────────────────────────────────────────────────

class TestFilePatch:
    def test_basic_replacement(self, tmp_path):
        p = tmp_path / "patch_me.txt"
        p.write_text("Hello OLD world")
        r = file_patch(str(p), "OLD", "NEW")
        assert r["success"]
        assert p.read_text() == "Hello NEW world"

    def test_missing_old_text_error(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("content")
        r = file_patch(str(p), "NOTHERE", "replacement")
        assert not r["success"]

    def test_replaces_first_occurrence(self, tmp_path):
        p = tmp_path / "multi.txt"
        p.write_text("aaa bbb aaa")
        r = file_patch(str(p), "aaa", "ZZZ")
        assert r["success"]
        assert "ZZZ" in p.read_text()

    def test_missing_file_error(self):
        r = file_patch("/no/file.txt", "old", "new")
        assert not r["success"]
