"""
Cross-session persistent semantic memory for the Operon AI agent.

Stores every conversation turn in a local SQLite database and retrieves
semantically relevant memories at the start of each new session turn.

Embedding priority order
------------------------
1. Ollama (local)  — POSTs to http://localhost:11434/api/embeddings using the
   ``nomic-embed-text`` model.  Preferred when an Ollama daemon is running
   because it is fully offline and free.  5-second timeout; silently skipped
   on any network or HTTP error.

2. OpenAI          — POSTs to https://api.openai.com/v1/embeddings using the
   ``text-embedding-3-small`` model.  Requires the ``OPENAI_API_KEY``
   environment variable.  10-second timeout; silently skipped when the key is
   absent or the API call fails.

3. BM25 fallback   — Pure-Python TF-IDF/BM25-style sparse-dense vector built
   entirely from the tokens already stored in the database.  Always available;
   used when both network backends are unreachable.

The backend is resolved lazily on the first call that needs an embedding and
then cached for the lifetime of the ``SemanticMemory`` instance.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# numpy is optional — used only in _cosine() if available, otherwise falls
# back to a pure-Python dot-product.  Never imported at module level so that
# Operon starts without the ~80 MB overhead on memory-constrained machines.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DB_DIR = Path.home() / ".operon"
_DB_PATH = _DB_DIR / "memory.sqlite"

_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    timestamp  REAL    NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    embedding  BLOB
);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories (session_id);
"""

_CONTENT_MAX = 500

# BM25 hyper-parameters
_BM25_K1 = 1.5
_BM25_B = 0.75


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

def _embed_ollama(text: str) -> Optional[List[float]]:
    """Attempt to get an embedding from a local Ollama instance."""
    url = "http://localhost:11434/api/embeddings"
    payload = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            embedding = data.get("embedding")
            if isinstance(embedding, list) and embedding:
                return [float(v) for v in embedding]
    except Exception:
        pass
    return None


def _embed_openai(text: str) -> Optional[List[float]]:
    """Attempt to get an embedding from the OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    url = "https://api.openai.com/v1/embeddings"
    payload = json.dumps({"model": "text-embedding-3-small", "input": text}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            embedding = data["data"][0]["embedding"]
            if isinstance(embedding, list) and embedding:
                return [float(v) for v in embedding]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# BM25 fallback
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-z]{3,}\b", text.lower())


def _build_bm25_vector(
    tokens: List[str],
    vocab: Dict[str, int],
    df: Dict[str, int],
    num_docs: int,
    avg_dl: float,
) -> List[float]:
    """Return a BM25-weighted, L2-normalised vector in vocab space.
    Pure Python — no numpy dependency."""
    tf: Dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    dl = len(tokens)

    vec = [0.0] * len(vocab)
    for term, idx in vocab.items():
        if term not in tf:
            continue
        idf = math.log((num_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
        numerator = tf[term] * (_BM25_K1 + 1)
        denominator = tf[term] + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(avg_dl, 1))
        vec[idx] = idf * numerator / denominator

    # Pure-Python L2 normalisation (no numpy)
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

class SemanticMemory:
    """
    Persistent cross-session memory store backed by SQLite.

    Parameters
    ----------
    config : dict, optional
        Reserved for future configuration options (e.g. custom db_path).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config: Dict[str, Any] = config or {}
        db_path = Path(self._config.get("db_path", _DB_PATH))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        from core.sqlite_utils import configure_wal, reconcile_columns
        configure_wal(self._conn)
        self._conn.executescript(_DDL)
        reconcile_columns(self._conn, _DDL)
        self._conn.commit()
        # Lazy embedding backend; resolved once, then cached.
        self._embed_fn = None  # type: Optional[str]  # "ollama" | "openai" | "bm25"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_backend(self) -> str:
        """Resolve which embedding backend to use and cache the result."""
        if self._embed_fn is not None:
            return self._embed_fn

        # Try Ollama
        if _embed_ollama("test") is not None:
            self._embed_fn = "ollama"
            return self._embed_fn

        # Try OpenAI
        if _embed_openai("test") is not None:
            self._embed_fn = "openai"
            return self._embed_fn

        # Fall back to BM25
        self._embed_fn = "bm25"
        return self._embed_fn

    def _bm25_stats(self) -> tuple:
        """
        Return (vocab, df, num_docs, avg_dl) computed from all stored memories.
        """
        rows = self._conn.execute("SELECT content FROM memories").fetchall()
        num_docs = len(rows)
        if num_docs == 0:
            return {}, {}, 0, 1.0

        vocab: Dict[str, int] = {}
        df: Dict[str, int] = {}
        total_len = 0
        for (content,) in rows:
            tokens = _tokenize(content)
            total_len += len(tokens)
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
            for term in tokens:
                if term not in vocab:
                    vocab[term] = len(vocab)

        avg_dl = total_len / num_docs
        return vocab, df, num_docs, avg_dl

    def _embed(self, text: str) -> Optional[List[float]]:
        """Produce an embedding for *text* using the resolved backend."""
        backend = self._resolve_backend()
        try:
            if backend == "ollama":
                return _embed_ollama(text)
            if backend == "openai":
                return _embed_openai(text)
            # BM25
            vocab, df, num_docs, avg_dl = self._bm25_stats()
            if not vocab:
                return None
            tokens = _tokenize(text)
            return _build_bm25_vector(tokens, vocab, df, num_docs, avg_dl)
        except Exception:
            return None

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """Cosine similarity. Uses numpy if available, otherwise pure Python."""
        try:
            import numpy as _np
            va = _np.array(a, dtype=_np.float32)
            vb = _np.array(b, dtype=_np.float32)
            na = float(_np.linalg.norm(va))
            nb = float(_np.linalg.norm(vb))
            if na == 0 or nb == 0:
                return 0.0
            return float(_np.dot(va, vb) / (na * nb))
        except ImportError:
            # Pure-Python fallback — no numpy needed
            dot  = sum(x * y for x, y in zip(a, b))
            na   = math.sqrt(sum(x * x for x in a))
            nb   = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

    @staticmethod
    def _fmt_ts(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, session_id: str, role: str, content: str) -> int:
        """
        Persist one conversation turn and return its assigned row id.

        Parameters
        ----------
        session_id : str
            Unique identifier for the current conversation session.
        role : str
            ``"user"`` or ``"assistant"``.
        content : str
            Message text; automatically truncated to 500 characters.

        Returns
        -------
        int
            The ``id`` of the newly inserted row.
        """
        content = content[:_CONTENT_MAX]
        embedding = self._embed(content)
        embedding_blob = json.dumps(embedding) if embedding is not None else None
        ts = time.time()
        cur = self._conn.execute(
            "INSERT INTO memories (session_id, timestamp, role, content, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, ts, role, content, embedding_blob),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def recall(
        self,
        query: str,
        top_k: int = 5,
        session_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories semantically relevant to *query*.

        Only memories from *other* sessions are returned (to avoid surfacing
        the current conversation back to itself).

        Parameters
        ----------
        query : str
            The text to search for.
        top_k : int
            Maximum number of results to return.
        session_id : str
            The current session id; memories with this id are excluded.

        Returns
        -------
        list[dict]
            Each dict contains ``id``, ``session_id``, ``timestamp``,
            ``role``, ``content``, and ``similarity`` keys.  Ordered by
            descending similarity.  Only entries with similarity >= 0.4 are
            included.
        """
        q_emb = self._embed(query)

        rows = self._conn.execute(
            "SELECT id, session_id, timestamp, role, content, embedding "
            "FROM memories "
            "WHERE session_id != ?",
            (session_id,),
        ).fetchall()

        if not rows:
            return []

        scored: List[tuple] = []
        for row_id, sess, ts, role, content, emb_blob in rows:
            if q_emb is None or emb_blob is None:
                sim = 0.0
            else:
                try:
                    mem_emb = json.loads(emb_blob)
                    # Align vector dimensions (BM25 vocab may grow between calls)
                    if len(q_emb) != len(mem_emb):
                        min_len = min(len(q_emb), len(mem_emb))
                        sim = self._cosine(q_emb[:min_len], mem_emb[:min_len])
                    else:
                        sim = self._cosine(q_emb, mem_emb)
                except Exception:
                    sim = 0.0
            if sim >= 0.4:
                scored.append((sim, row_id, sess, ts, role, content))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "id": row_id,
                "session_id": sess,
                "timestamp": ts,
                "role": role,
                "content": content,
                "similarity": round(sim, 4),
            }
            for sim, row_id, sess, ts, role, content in scored[:top_k]
        ]

    def as_context_block(
        self,
        query: str,
        session_id: str = "",
        max_chars: int = 1200,
    ) -> str:
        """
        Return a formatted string of relevant past memories for injection into
        a prompt, or an empty string if nothing relevant was found.

        Parameters
        ----------
        query : str
            The current user message used as the search query.
        session_id : str
            The active session id (excluded from results).
        max_chars : int
            Hard cap on the total character length of the returned block
            (default 1 200, ~300 tokens).  Memories are included in
            descending-similarity order; the block is truncated with a
            ``"[… truncated]"`` marker when the cap is reached.  Set to 0
            to disable the cap entirely.

        Returns
        -------
        str
            A multi-line block ready for prepending to a system prompt, or
            ``""`` if there are no memories with similarity >= 0.4.
        """
        memories = self.recall(query, session_id=session_id)
        if not memories:
            return ""

        role_label = {"user": "User", "assistant": "Operon"}
        lines = ["[LONG-TERM MEMORY — relevant past context]"]
        used_chars = len(lines[0]) + 1  # header + newline

        for mem in memories:
            date = self._fmt_ts(mem["timestamp"])
            label = role_label.get(mem["role"], mem["role"].capitalize())
            entry = f"[{date}] {label}: {mem['content']}"
            entry_chars = len(entry) + 1  # +1 for the joining newline

            if max_chars and used_chars + entry_chars > max_chars:
                lines.append("[… memory truncated to stay within token budget]")
                break
            lines.append(entry)
            used_chars += entry_chars

        lines.append("[END MEMORY]")
        return "\n".join(lines)

    def forget(self, memory_id: int) -> None:
        """Delete a single memory by its row id."""
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    def forget_session(self, session_id: str) -> None:
        """Delete all memories belonging to *session_id*."""
        self._conn.execute("DELETE FROM memories WHERE session_id = ?", (session_id,))
        self._conn.commit()

    def forget_all(self) -> None:
        """Wipe the entire memory store."""
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()

    def stats(self) -> Dict[str, Any]:
        """
        Return summary statistics about the memory store.

        Returns
        -------
        dict
            Keys: ``total_memories``, ``total_sessions``, ``with_embeddings``,
            ``oldest_ts`` (float or None), ``newest_ts`` (float or None).
        """
        row = self._conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT session_id), "
            "SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END), "
            "MIN(timestamp), MAX(timestamp) "
            "FROM memories"
        ).fetchone()
        total, sessions, with_emb, oldest, newest = row
        return {
            "total_memories": total or 0,
            "total_sessions": sessions or 0,
            "with_embeddings": with_emb or 0,
            "oldest_ts": oldest,
            "newest_ts": newest,
        }

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass
