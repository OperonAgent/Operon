"""Tests for tools/db_ops.py

All tests use SQLite :memory: databases — no external services required.
The PostgreSQL / MySQL / MongoDB paths are tested via mock to cover error
handling and URL routing without requiring live servers.
"""
import unittest.mock as mock
import pytest

from tools.db_ops import (
    db_query, db_list_tables, db_describe_table, mongo_query,
    _resolve_url, _fmt_result,
)


# ── _resolve_url ──────────────────────────────────────────────────────────────

class TestResolveUrl:
    def test_explicit_url_returned_as_is(self):
        assert _resolve_url("sqlite:///test.db") == "sqlite:///test.db"

    def test_empty_returns_empty_when_no_env(self, monkeypatch):
        for k in ("DATABASE_URL", "SQLITE_URL", "SQLITE_PATH",
                  "POSTGRES_URL", "MYSQL_URL", "MONGODB_URL"):
            monkeypatch.delenv(k, raising=False)
        assert _resolve_url("") == ""

    def test_database_url_env_respected(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:///env.db")
        assert _resolve_url("") == "sqlite:///env.db"

    def test_explicit_url_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:///env.db")
        assert _resolve_url("sqlite:///explicit.db") == "sqlite:///explicit.db"


# ── _fmt_result ───────────────────────────────────────────────────────────────

class TestFmtResult:
    def test_keys_present(self):
        r = _fmt_result([{"a": 1}], ["a"], "SELECT 1", 1, 0.001)
        for k in ("success", "query", "columns", "rows", "rowcount", "elapsed_ms", "error"):
            assert k in r

    def test_success_always_true(self):
        r = _fmt_result([], [], "SELECT 1", 0, 0.0)
        assert r["success"] is True

    def test_elapsed_ms_rounded(self):
        r = _fmt_result([], [], "q", 0, 0.0012345)
        # Should be rounded to 1 decimal place
        assert isinstance(r["elapsed_ms"], float)
        assert r["elapsed_ms"] == round(0.0012345 * 1000, 1)


# ── db_query — SQLite :memory: ────────────────────────────────────────────────

class TestDbQuerySQLite:
    def test_select_returns_rows(self):
        r = db_query("SELECT 1 AS n", db_url=":memory:")
        assert r["success"] is True
        assert r["columns"] == ["n"]
        assert r["rows"] == [{"n": 1}]

    def test_create_insert_select_roundtrip(self):
        # SQLite :memory: is stateless per connection — use a single file
        import tempfile, os
        f = tempfile.mktemp(suffix=".sqlite")
        try:
            r1 = db_query("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)",
                          db_url=f)
            assert r1["success"] is True
            r2 = db_query("INSERT INTO items (name) VALUES (?)", db_url=f,
                          params=["widget"])
            assert r2["success"] is True
            r3 = db_query("SELECT * FROM items", db_url=f)
            assert r3["success"] is True
            assert len(r3["rows"]) == 1
            assert r3["rows"][0]["name"] == "widget"
        finally:
            if os.path.exists(f):
                os.unlink(f)

    def test_empty_query_returns_error(self):
        r = db_query("")
        assert r["success"] is False
        assert "required" in r["error"].lower()

    def test_whitespace_query_returns_error(self):
        r = db_query("   ")
        assert r["success"] is False

    def test_syntax_error_returns_failure(self):
        r = db_query("SELECT * FROM", db_url=":memory:")
        assert r["success"] is False
        assert "error" in r

    def test_parameterised_query(self):
        import tempfile, os
        f = tempfile.mktemp(suffix=".sqlite")
        try:
            db_query("CREATE TABLE t (v INTEGER)", db_url=f)
            db_query("INSERT INTO t VALUES (?)", db_url=f, params=[42])
            r = db_query("SELECT v FROM t WHERE v = ?", db_url=f, params=[42])
            assert r["success"] is True
            assert r["rows"][0]["v"] == 42
        finally:
            if os.path.exists(f):
                os.unlink(f)

    def test_insert_returns_rowcount(self):
        import tempfile, os
        f = tempfile.mktemp(suffix=".sqlite")
        try:
            db_query("CREATE TABLE t2 (n INTEGER)", db_url=f)
            r = db_query("INSERT INTO t2 VALUES (1)", db_url=f)
            assert r["success"] is True
            assert r["rowcount"] == 1
        finally:
            if os.path.exists(f):
                os.unlink(f)

    def test_elapsed_ms_non_negative(self):
        r = db_query("SELECT 1", db_url=":memory:")
        assert r["elapsed_ms"] >= 0

    def test_sqlite_url_prefix_stripped(self):
        """sqlite:///path should work the same as bare path."""
        r = db_query("SELECT 1 AS x", db_url="sqlite:///:memory:")
        assert r["success"] is True

    def test_extra_kwargs_ignored(self):
        r = db_query("SELECT 2", db_url=":memory:", unknown_param="x")
        assert r["success"] is True


# ── db_query — Postgres/MySQL routing (mocked) ───────────────────────────────

class TestDbQueryBackendRouting:
    def test_postgres_url_routes_to_postgres(self):
        with mock.patch("tools.db_ops._postgres_query") as m:
            m.return_value = {"success": True, "columns": [], "rows": [],
                              "rowcount": 0, "elapsed_ms": 0.0, "query": "", "error": ""}
            db_query("SELECT 1", db_url="postgresql://user:pass@localhost/test")
            m.assert_called_once()

    def test_postgres_alt_scheme_routes(self):
        with mock.patch("tools.db_ops._postgres_query") as m:
            m.return_value = {"success": True, "columns": [], "rows": [],
                              "rowcount": 0, "elapsed_ms": 0.0, "query": "", "error": ""}
            db_query("SELECT 1", db_url="postgres://user:pass@localhost/test")
            m.assert_called_once()

    def test_mysql_url_routes_to_mysql(self):
        with mock.patch("tools.db_ops._mysql_query") as m:
            m.return_value = {"success": True, "columns": [], "rows": [],
                              "rowcount": 0, "elapsed_ms": 0.0, "query": "", "error": ""}
            db_query("SELECT 1", db_url="mysql://user:pass@localhost/test")
            m.assert_called_once()


# ── db_list_tables ────────────────────────────────────────────────────────────

class TestDbListTables:
    def test_empty_db_returns_empty_list(self):
        r = db_list_tables(db_url=":memory:")
        assert r["success"] is True
        assert isinstance(r["tables"], list)
        assert r["tables"] == []

    def test_created_table_appears(self):
        import tempfile, os
        f = tempfile.mktemp(suffix=".sqlite")
        try:
            db_query("CREATE TABLE alpha (id INTEGER)", db_url=f)
            db_query("CREATE TABLE beta  (id INTEGER)", db_url=f)
            r = db_list_tables(db_url=f)
            assert r["success"] is True
            assert "alpha" in r["tables"]
            assert "beta"  in r["tables"]
        finally:
            if os.path.exists(f):
                os.unlink(f)

    def test_error_key_empty_on_success(self):
        r = db_list_tables(db_url=":memory:")
        assert r["error"] == ""

    def test_postgres_routing(self):
        with mock.patch("tools.db_ops._postgres_query") as m:
            m.return_value = {"success": True, "rows": [{"tablename": "users"}],
                              "columns": ["tablename"], "rowcount": 1,
                              "elapsed_ms": 0.0, "query": "", "error": ""}
            r = db_list_tables(db_url="postgresql://u:p@host/db")
            assert r["success"] is True
            assert "users" in r["tables"]


# ── db_describe_table ─────────────────────────────────────────────────────────

class TestDbDescribeTable:
    def test_empty_table_name_returns_error(self):
        r = db_describe_table(table="")
        assert r["success"] is False
        assert "required" in r["error"].lower()

    def test_describe_existing_table(self):
        import tempfile, os
        f = tempfile.mktemp(suffix=".sqlite")
        try:
            db_query("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
                     db_url=f)
            r = db_describe_table(table="products", db_url=f)
            assert r["success"] is True
            assert r["table"] == "products"
            assert isinstance(r["columns"], list)
            assert len(r["columns"]) >= 2
        finally:
            if os.path.exists(f):
                os.unlink(f)

    def test_describe_nonexistent_table(self):
        r = db_describe_table(table="ghost_table", db_url=":memory:")
        # PRAGMA table_info returns empty list for nonexistent tables — success=True, empty columns
        assert r["success"] is True
        assert r["columns"] == []

    def test_postgres_routing(self):
        with mock.patch("tools.db_ops._postgres_query") as m:
            m.return_value = {"success": True, "rows": [
                {"name": "id", "type": "integer", "nullable": "NO", "default": None}
            ], "columns": ["name", "type", "nullable", "default"],
                              "rowcount": 1, "elapsed_ms": 0.0, "query": "", "error": ""}
            r = db_describe_table(table="users", db_url="postgresql://u:p@host/db")
            assert r["success"] is True
            assert r["table"] == "users"


# ── mongo_query (mocked — no real MongoDB) ────────────────────────────────────

class TestMongoQuery:
    def test_empty_collection_returns_error(self):
        r = mongo_query(collection="")
        assert r["success"] is False
        assert "collection" in r["error"].lower()

    def test_pymongo_not_installed_returns_friendly_error(self):
        import sys
        # Temporarily block pymongo import
        with mock.patch.dict(sys.modules, {"pymongo": None}):
            r = mongo_query(collection="test_col",
                            db_url="mongodb://localhost:27017/testdb")
        assert r["success"] is False
        assert "pymongo" in r["error"].lower()

    def test_connection_error_captured(self):
        with mock.patch("tools.db_ops._mongo_query") as m:
            m.return_value = {"success": False, "error": "connection refused"}
            r = mongo_query(collection="users",
                            db_url="mongodb://localhost:27017/testdb")
        assert r["success"] is False

    def test_defaults_to_localhost_url(self):
        """When no db_url and no env var, default URL is mongodb://localhost."""
        with mock.patch("tools.db_ops._mongo_query") as m:
            m.return_value = {"success": True, "columns": [], "rows": [],
                              "rowcount": 0, "elapsed_ms": 0.0, "query": "", "error": ""}
            mongo_query(collection="col")
            url_arg = m.call_args[0][0]
            assert "localhost" in url_arg or "mongo" in url_arg.lower()

    def test_extra_kwargs_ignored(self):
        with mock.patch("tools.db_ops._mongo_query") as m:
            m.return_value = {"success": True, "columns": [], "rows": [],
                              "rowcount": 0, "elapsed_ms": 0.0, "query": "", "error": ""}
            r = mongo_query(collection="c", ignored_extra="x")
            assert r["success"] is True
