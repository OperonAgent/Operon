"""
Operon Database Query Tools.

Supported backends (all gracefully degrade if driver not installed):
  • SQLite   — always available (Python stdlib)
  • PostgreSQL — requires psycopg2:  pip install psycopg2-binary
  • MySQL     — requires PyMySQL:    pip install pymysql
  • MongoDB   — requires pymongo:    pip install pymongo

Connection strings follow standard URL schemes:
  sqlite:///path/to/db.sqlite
  postgresql://user:pass@host:5432/dbname
  mysql://user:pass@host:3306/dbname
  mongodb://user:pass@host:27017/dbname

Credentials are also read from environment variables:
  DATABASE_URL — full connection URL (overrides all per-driver env vars)
  SQLITE_PATH, POSTGRES_URL, MYSQL_URL, MONGODB_URL

All functions accept **_ for registry compatibility.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Connection URL resolution
# ---------------------------------------------------------------------------

def _resolve_url(db_url: str = "", backend: str = "sqlite") -> str:
    if db_url:
        return db_url
    # Check environment
    for env_key in ("DATABASE_URL", f"{backend.upper()}_URL",
                    "SQLITE_PATH", "POSTGRES_URL", "MYSQL_URL", "MONGODB_URL"):
        v = os.environ.get(env_key, "")
        if v:
            return v
    return ""


# ---------------------------------------------------------------------------
# Shared result formatter
# ---------------------------------------------------------------------------

def _fmt_result(rows: List[Dict], columns: List[str], query: str,
                rowcount: int, elapsed: float) -> dict:
    return {
        "success":  True,
        "query":    query,
        "columns":  columns,
        "rows":     rows,
        "rowcount": rowcount,
        "elapsed_ms": round(elapsed * 1000, 1),
        "error":    "",
    }


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

def _sqlite_query(db_path: str, query: str, params: list, timeout: int) -> dict:
    import sqlite3
    # Expand ~ and relative paths
    db_path = os.path.expanduser(db_path) if db_path != ":memory:" else db_path
    t0 = time.monotonic()
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()
        cur.execute(query, params or [])
        if cur.description:
            cols = [d[0] for d in cur.description]
            rows = [dict(r) for r in cur.fetchall()]
            conn.commit()
            conn.close()
            return _fmt_result(rows, cols, query, len(rows), time.monotonic() - t0)
        else:
            conn.commit()
            rc = cur.rowcount
            conn.close()
            return _fmt_result([], [], query, rc, time.monotonic() - t0)
    except Exception as e:
        return {"success": False, "error": str(e), "query": query}


# ---------------------------------------------------------------------------
# PostgreSQL backend
# ---------------------------------------------------------------------------

def _postgres_query(url: str, query: str, params: list, timeout: int) -> dict:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        return {
            "success": False,
            "error": "psycopg2 not installed. Run: pip install psycopg2-binary",
        }
    t0 = time.monotonic()
    try:
        conn = psycopg2.connect(url, connect_timeout=timeout)
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(query, params or [])
        if cur.description:
            cols = [d.name for d in cur.description]
            rows = [dict(r) for r in cur.fetchall()]
            conn.commit()
            conn.close()
            return _fmt_result(rows, cols, query, len(rows), time.monotonic() - t0)
        else:
            conn.commit()
            rc = cur.rowcount
            conn.close()
            return _fmt_result([], [], query, rc, time.monotonic() - t0)
    except Exception as e:
        return {"success": False, "error": str(e), "query": query}


# ---------------------------------------------------------------------------
# MySQL backend
# ---------------------------------------------------------------------------

def _mysql_query(url: str, query: str, params: list, timeout: int) -> dict:
    try:
        import pymysql
        import pymysql.cursors
    except ImportError:
        return {
            "success": False,
            "error": "PyMySQL not installed. Run: pip install pymysql",
        }
    # Parse mysql://user:pass@host:port/dbname
    m = re.match(
        r"mysql://(?:([^:@]*)(?::([^@]*))?@)?([^:/]*)(?::(\d+))?/(.+)", url
    )
    if not m:
        return {"success": False, "error": f"Invalid MySQL URL: {url}"}
    user, pw, host, port, db = m.groups()
    t0 = time.monotonic()
    try:
        conn = pymysql.connect(
            host=host or "localhost", port=int(port or 3306),
            user=user or "", password=pw or "", database=db or "",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=timeout,
        )
        with conn.cursor() as cur:
            cur.execute(query, params or [])
            if cur.description:
                cols = [d[0] for d in cur.description]
                rows = list(cur.fetchall())
                conn.commit()
                conn.close()
                return _fmt_result(rows, cols, query, len(rows), time.monotonic() - t0)
            else:
                conn.commit()
                rc = cur.rowcount
                conn.close()
                return _fmt_result([], [], query, rc, time.monotonic() - t0)
    except Exception as e:
        return {"success": False, "error": str(e), "query": query}


# ---------------------------------------------------------------------------
# MongoDB backend
# ---------------------------------------------------------------------------

def _mongo_query(url: str, collection: str, operation: str,
                 filter_doc: dict, limit: int) -> dict:
    try:
        import pymongo
    except ImportError:
        return {
            "success": False,
            "error": "pymongo not installed. Run: pip install pymongo",
        }
    t0 = time.monotonic()
    try:
        # Extract db name from URL: mongodb://host/dbname
        m = re.search(r"/([^/?]+)(?:\?|$)", url.replace("mongodb://", "x://", 1))
        db_name = m.group(1) if m else "test"
        client = pymongo.MongoClient(url, serverSelectionTimeoutMS=5000)
        db     = client[db_name]
        col    = db[collection]

        op = operation.lower()
        if op in ("find", "query", "select"):
            rows = list(col.find(filter_doc or {}, limit=limit))
            # Convert ObjectId to string
            for r in rows:
                if "_id" in r:
                    r["_id"] = str(r["_id"])
            cols = list(rows[0].keys()) if rows else []
            client.close()
            return _fmt_result(rows, cols, f"find({filter_doc})", len(rows), time.monotonic() - t0)
        elif op in ("count", "count_documents"):
            count = col.count_documents(filter_doc or {})
            client.close()
            return _fmt_result([{"count": count}], ["count"], f"count({filter_doc})", count, time.monotonic() - t0)
        elif op in ("aggregate",):
            pipeline = filter_doc if isinstance(filter_doc, list) else [filter_doc]
            rows = list(col.aggregate(pipeline))
            for r in rows:
                if "_id" in r:
                    r["_id"] = str(r["_id"])
            cols = list(rows[0].keys()) if rows else []
            client.close()
            return _fmt_result(rows, cols, f"aggregate({pipeline})", len(rows), time.monotonic() - t0)
        else:
            client.close()
            return {"success": False, "error": f"Unknown MongoDB operation: {operation}. Use: find, count, aggregate"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def db_query(
    query: str = "",
    db_url: str = "",
    backend: str = "sqlite",
    params: list = None,
    timeout: int = 30,
    **_,
) -> dict:
    """
    Execute a SQL query against SQLite, PostgreSQL, or MySQL.

    Args:
        query   — SQL statement to execute (required)
        db_url  — connection URL or file path (optional — auto-read from DATABASE_URL env var)
                  SQLite:     /path/to/db.sqlite  OR  :memory:
                  PostgreSQL: postgresql://user:pass@host:5432/dbname
                  MySQL:      mysql://user:pass@host:3306/dbname
        backend — 'sqlite' | 'postgresql' | 'mysql' — auto-detected from db_url (optional)
        params  — list of query parameters for parameterised queries (optional)
        timeout — query timeout in seconds, default 30 (optional)

    Returns:
        {success, columns, rows, rowcount, elapsed_ms, error}
    """
    if not query or not query.strip():
        return {"success": False, "error": "query is required."}

    url = _resolve_url(db_url, backend)

    # Auto-detect backend from URL
    if url.startswith("postgresql") or url.startswith("postgres"):
        return _postgres_query(url, query, params or [], timeout)
    if url.startswith("mysql"):
        return _mysql_query(url, query, params or [], timeout)
    # SQLite — url might be a plain file path
    if not url:
        url = ":memory:"
    sqlite_path = url.replace("sqlite:///", "").replace("sqlite://", "")
    return _sqlite_query(sqlite_path, query, params or [], timeout)


def db_list_tables(
    db_url: str = "",
    backend: str = "sqlite",
    **_,
) -> dict:
    """
    List all tables (or collections) in a database.

    Args:
        db_url  — connection URL (optional — auto-read from DATABASE_URL)
        backend — 'sqlite' | 'postgresql' | 'mysql' (optional, auto-detected)

    Returns:
        {success, tables: [str], error}
    """
    url = _resolve_url(db_url, backend)

    if url.startswith("postgresql") or url.startswith("postgres"):
        result = _postgres_query(
            url,
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename",
            [], 15,
        )
    elif url.startswith("mysql"):
        result = _mysql_query(url, "SHOW TABLES", [], 15)
    else:
        sqlite_path = (url or ":memory:").replace("sqlite:///", "").replace("sqlite://", "")
        result = _sqlite_query(
            sqlite_path,
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            [], 15,
        )

    if not result["success"]:
        return result

    # Flatten first column of each row to a plain list
    tables = []
    for row in result.get("rows", []):
        if isinstance(row, dict):
            tables.append(list(row.values())[0])
        else:
            tables.append(str(row))
    return {"success": True, "tables": tables, "error": ""}


def db_describe_table(
    table: str = "",
    db_url: str = "",
    backend: str = "sqlite",
    **_,
) -> dict:
    """
    Return the schema / column definitions for a table.

    Args:
        table   — table name (required)
        db_url  — connection URL (optional — auto-read from DATABASE_URL)
        backend — 'sqlite' | 'postgresql' | 'mysql' (optional, auto-detected)

    Returns:
        {success, table, columns: [{name, type, nullable, default, primary_key}], error}
    """
    if not table:
        return {"success": False, "error": "table name is required."}

    url = _resolve_url(db_url, backend)

    if url.startswith("postgresql") or url.startswith("postgres"):
        result = _postgres_query(
            url,
            """
            SELECT column_name AS name,
                   data_type   AS type,
                   is_nullable AS nullable,
                   column_default AS "default"
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = 'public'
            ORDER BY ordinal_position
            """,
            [table], 15,
        )
    elif url.startswith("mysql"):
        result = _mysql_query(url, f"DESCRIBE `{table}`", [], 15)
    else:
        sqlite_path = (url or ":memory:").replace("sqlite:///", "").replace("sqlite://", "")
        result = _sqlite_query(sqlite_path, f"PRAGMA table_info({table})", [], 15)

    if not result["success"]:
        return result

    return {"success": True, "table": table, "columns": result.get("rows", []), "error": ""}


def mongo_query(
    collection: str = "",
    operation: str = "find",
    filter: dict = None,
    limit: int = 20,
    db_url: str = "",
    **_,
) -> dict:
    """
    Query a MongoDB collection.

    Args:
        collection — collection name (required)
        operation  — 'find' | 'count' | 'aggregate' (optional, default 'find')
        filter     — filter document for find/count, or pipeline list for aggregate (optional)
        limit      — max documents to return for find (optional, default 20)
        db_url     — MongoDB connection URL (optional — auto-read from MONGODB_URL)

    Returns:
        {success, columns, rows, rowcount, elapsed_ms, error}
    """
    if not collection:
        return {"success": False, "error": "collection is required."}

    url = _resolve_url(db_url, "mongodb")
    if not url:
        url = "mongodb://localhost:27017/test"

    return _mongo_query(url, collection, operation, filter or {}, limit)
