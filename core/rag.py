"""
Operon RAG (Retrieval-Augmented Generation) Pipeline.

Chunks documents, embeds them, stores in SQLite, and retrieves relevant
passages at query time using the same embedding backends as SemanticMemory
(Ollama → OpenAI → BM25 pure-Python fallback).

Supported document types (auto-detected by extension):
  .txt .md .rst .py .js .ts .html .csv .json .pdf (requires pypdf)

Usage
-----
  from core.rag import RAGPipeline

  rag = RAGPipeline()
  rag.index_file("/path/to/docs/report.pdf")
  rag.index_directory("/path/to/project", glob="*.md")
  results = rag.query("How does the authentication system work?", top_k=5)
  context = rag.as_context_block("How does auth work?")
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_RAG_DB_PATH = Path.home() / ".operon" / "rag.sqlite"

_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding   BLOB,
    indexed_at  REAL NOT NULL,
    UNIQUE(source, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_rag_source ON documents (source);
CREATE INDEX IF NOT EXISTS idx_rag_hash   ON documents (content_hash);
"""

_CHUNK_SIZE    = 800    # target chars per chunk
_CHUNK_OVERLAP = 100    # overlap chars between consecutive chunks


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text(path: Path) -> str:
    """Extract plain text from a file. Handles PDF if pypdf installed."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            return f"[PDF extraction requires pypdf: pip install pypdf] {path}"
        except Exception as e:
            return f"[PDF error: {e}]"

    if suffix in (".html", ".htm"):
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(path.read_text(errors="replace"), "html.parser")
            return soup.get_text(separator="\n")
        except ImportError:
            pass

    # Default: read as text
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[Read error: {e}]"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE,
                overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks on paragraph / sentence boundaries."""
    # Split on paragraph breaks first
    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    buf = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(buf) + len(para) + 1 <= chunk_size:
            buf = (buf + "\n\n" + para).strip() if buf else para
        else:
            if buf:
                chunks.append(buf)
                # Keep overlap
                buf = buf[-overlap:] if len(buf) > overlap else buf
            if len(para) <= chunk_size:
                buf = (buf + "\n\n" + para).strip() if buf else para
            else:
                # Paragraph is itself too long — split by sentence
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if len(buf) + len(sent) + 1 <= chunk_size:
                        buf = (buf + " " + sent).strip()
                    else:
                        if buf:
                            chunks.append(buf)
                            buf = buf[-overlap:] if len(buf) > overlap else ""
                        buf = sent

    if buf.strip():
        chunks.append(buf.strip())

    # Filter out very short chunks (< 40 chars)
    return [c for c in chunks if len(c) >= 40]


# ---------------------------------------------------------------------------
# Embedding — reuse SemanticMemory's backends
# ---------------------------------------------------------------------------

def _embed(text: str) -> Optional[List[float]]:
    """Embed text using the same priority chain as SemanticMemory."""
    from core.semantic_memory import _embed_ollama, _embed_openai, _tokenize, _build_bm25_vector
    vec = _embed_ollama(text)
    if vec:
        return vec
    vec = _embed_openai(text)
    if vec:
        return vec
    return None   # caller falls back to BM25 at query time


def _cosine(a: List[float], b: List[float]) -> float:
    try:
        import numpy as _np
        va, vb = _np.array(a, dtype=_np.float32), _np.array(b, dtype=_np.float32)
        na, nb = float(_np.linalg.norm(va)), float(_np.linalg.norm(vb))
        return 0.0 if (na == 0 or nb == 0) else float(_np.dot(va, vb) / (na * nb))
    except ImportError:
        dot = sum(x * y for x, y in zip(a, b))
        na  = math.sqrt(sum(x * x for x in a))
        nb  = math.sqrt(sum(x * x for x in b))
        return 0.0 if (na == 0 or nb == 0) else dot / (na * nb)


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Document indexer and retriever for Retrieval-Augmented Generation.

    Parameters
    ----------
    db_path : Path, optional
        SQLite database path. Defaults to ~/.operon/rag.sqlite.
    chunk_size : int, optional
        Target characters per chunk (default 800).
    chunk_overlap : int, optional
        Overlap characters between chunks (default 100).
    """

    def __init__(
        self,
        db_path:       Path = _RAG_DB_PATH,
        chunk_size:    int  = _CHUNK_SIZE,
        chunk_overlap: int  = _CHUNK_OVERLAP,
    ) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn          = sqlite3.connect(str(db_path), check_same_thread=False)
        self._chunk_size    = chunk_size
        self._chunk_overlap = chunk_overlap
        from core.sqlite_utils import configure_wal, reconcile_columns
        configure_wal(self._conn)
        self._conn.executescript(_DDL)
        reconcile_columns(self._conn, _DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_file(self, path: str, force: bool = False) -> Dict[str, Any]:
        """
        Index a single file.

        Parameters
        ----------
        path  : file path
        force : re-index even if file hash hasn't changed

        Returns
        -------
        dict with keys: source, chunks_added, chunks_skipped, error
        """
        p = Path(path).expanduser()
        if not p.exists():
            return {"source": str(p), "chunks_added": 0, "error": f"File not found: {p}"}

        text   = _extract_text(p)
        chunks = _chunk_text(text, self._chunk_size, self._chunk_overlap)
        source = str(p)

        added   = 0
        skipped = 0
        for idx, chunk in enumerate(chunks):
            h = hashlib.sha256(chunk.encode()).hexdigest()
            existing = self._conn.execute(
                "SELECT id FROM documents WHERE source = ? AND chunk_index = ?",
                (source, idx),
            ).fetchone()

            if existing and not force:
                # Check if content changed
                old_hash = self._conn.execute(
                    "SELECT content_hash FROM documents WHERE id = ?", (existing[0],)
                ).fetchone()
                if old_hash and old_hash[0] == h:
                    skipped += 1
                    continue
                # Content changed — update
                emb = _embed(chunk)
                emb_blob = json.dumps(emb) if emb else None
                self._conn.execute(
                    "UPDATE documents SET content=?, content_hash=?, embedding=?, indexed_at=? WHERE id=?",
                    (chunk, h, emb_blob, time.time(), existing[0]),
                )
            else:
                emb = _embed(chunk)
                emb_blob = json.dumps(emb) if emb else None
                self._conn.execute(
                    "INSERT OR REPLACE INTO documents (source, chunk_index, content, content_hash, embedding, indexed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (source, idx, chunk, h, emb_blob, time.time()),
                )
            added += 1

        # Remove stale chunks if file shrank
        self._conn.execute(
            "DELETE FROM documents WHERE source = ? AND chunk_index >= ?",
            (source, len(chunks)),
        )
        self._conn.commit()
        return {"source": source, "chunks_added": added, "chunks_skipped": skipped, "error": ""}

    def index_directory(
        self,
        directory: str,
        glob: str = "**/*.{txt,md,py,js,ts,html,rst,csv,json,pdf}",
        recursive: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Index all matching files in a directory.

        Returns
        -------
        dict with keys: files_indexed, chunks_added, errors
        """
        base = Path(directory).expanduser()
        if not base.is_dir():
            return {"files_indexed": 0, "chunks_added": 0, "errors": [f"Not a directory: {directory}"]}

        # Build file list
        extensions = re.findall(r"\w+", glob.split("{")[-1].split("}")[0]) if "{" in glob else []
        files: List[Path] = []
        if extensions:
            for ext in extensions:
                pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
                files.extend(base.glob(pattern))
        else:
            pattern = glob if not recursive else f"**/{glob}"
            files.extend(base.glob(pattern))

        files = sorted(set(files))
        total_added  = 0
        errors: List[str] = []

        for f in files:
            result = self.index_file(str(f), force=force)
            if result.get("error"):
                errors.append(result["error"])
            else:
                total_added += result.get("chunks_added", 0)

        return {
            "files_indexed": len(files),
            "chunks_added":  total_added,
            "errors":        errors,
        }

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: float = 0.3,
        sources: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant document chunks for a query.

        Parameters
        ----------
        query_text : search query
        top_k      : max results to return
        min_score  : minimum similarity threshold
        sources    : limit to these source paths (optional)

        Returns
        -------
        List of dicts: {source, chunk_index, content, score}
        """
        q_emb = _embed(query_text)

        where = "WHERE 1=1"
        args: list = []
        if sources:
            placeholders = ",".join("?" * len(sources))
            where += f" AND source IN ({placeholders})"
            args.extend(sources)

        rows = self._conn.execute(
            f"SELECT source, chunk_index, content, embedding FROM documents {where}",
            args,
        ).fetchall()

        if not rows:
            return []

        # BM25 fallback stats if needed
        bm25_stats = None

        scored: List[Tuple[float, str, int, str]] = []
        for source, idx, content, emb_blob in rows:
            if q_emb and emb_blob:
                try:
                    mem_emb = json.loads(emb_blob)
                    if len(q_emb) != len(mem_emb):
                        min_len = min(len(q_emb), len(mem_emb))
                        sim = _cosine(q_emb[:min_len], mem_emb[:min_len])
                    else:
                        sim = _cosine(q_emb, mem_emb)
                except Exception:
                    sim = 0.0
            else:
                # BM25 keyword fallback
                if bm25_stats is None:
                    bm25_stats = self._build_bm25_stats()
                sim = self._bm25_score(query_text, content, *bm25_stats)

            if sim >= min_score:
                scored.append((sim, source, idx, content))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "source":      src,
                "chunk_index": ci,
                "content":     cont,
                "score":       round(sc, 4),
            }
            for sc, src, ci, cont in scored[:top_k]
        ]

    def as_context_block(self, query: str, top_k: int = 4) -> str:
        """
        Return a formatted context block ready to prepend to a system prompt.
        Returns empty string if no relevant chunks found.
        """
        results = self.query(query, top_k=top_k)
        if not results:
            return ""

        lines = [f"[DOCUMENT CONTEXT — {len(results)} relevant passages]"]
        for r in results:
            source_name = Path(r["source"]).name
            lines.append(f"\n[Source: {source_name} | chunk {r['chunk_index']} | score {r['score']}]")
            lines.append(r["content"])
        lines.append("[END DOCUMENT CONTEXT]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # BM25 fallback
    # ------------------------------------------------------------------

    def _build_bm25_stats(self):
        rows = self._conn.execute("SELECT content FROM documents").fetchall()
        from core.semantic_memory import _tokenize
        num_docs  = len(rows)
        vocab: Dict[str, int] = {}
        df: Dict[str, int]    = {}
        total_len = 0
        for (content,) in rows:
            tokens = _tokenize(content)
            total_len += len(tokens)
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
            for term in tokens:
                if term not in vocab:
                    vocab[term] = len(vocab)
        avg_dl = total_len / num_docs if num_docs else 1.0
        return vocab, df, num_docs, avg_dl

    def _bm25_score(self, query: str, doc: str, vocab, df, num_docs, avg_dl,
                    k1=1.5, b=0.75) -> float:
        from core.semantic_memory import _tokenize
        q_tokens = _tokenize(query)
        d_tokens = _tokenize(doc)
        dl = len(d_tokens)
        tf: Dict[str, int] = {}
        for t in d_tokens:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        for term in q_tokens:
            if term not in tf:
                continue
            idf = math.log((num_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
            num = tf[term] * (k1 + 1)
            den = tf[term] + k1 * (1 - b + b * dl / max(avg_dl, 1))
            score += idf * num / den
        # Normalise to [0, 1] range approximately
        return min(score / (len(q_tokens) * 5 + 1), 1.0) if q_tokens else 0.0

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def list_sources(self) -> List[Dict[str, Any]]:
        """List all indexed document sources with chunk counts."""
        rows = self._conn.execute(
            "SELECT source, COUNT(*) as chunks, MAX(indexed_at) as last_indexed "
            "FROM documents GROUP BY source ORDER BY source"
        ).fetchall()
        return [
            {
                "source":       r[0],
                "chunks":       r[1],
                "last_indexed": r[2],
            }
            for r in rows
        ]

    def remove_source(self, source: str) -> int:
        """Remove all chunks for a source. Returns number of chunks deleted."""
        cur = self._conn.execute("DELETE FROM documents WHERE source = ?", (source,))
        self._conn.commit()
        return cur.rowcount

    def clear(self) -> None:
        """Remove all indexed documents."""
        self._conn.execute("DELETE FROM documents")
        self._conn.commit()

    def stats(self) -> Dict[str, Any]:
        row = self._conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT source), "
            "SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM documents"
        ).fetchone()
        return {
            "total_chunks":    row[0] or 0,
            "total_sources":   row[1] or 0,
            "with_embeddings": row[2] or 0,
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LightRAG optional backend
# ---------------------------------------------------------------------------

class LightRAGBackend:
    """
    Optional LightRAG backend (graph-based RAG with entity/relation extraction).
    Requires: pip install lightrag-hku

    Provides the same interface as RAGPipeline so it can be used as a drop-in.
    Falls back gracefully if lightrag-hku is not installed.

    Environment
    -----------
      LIGHTRAG_WORK_DIR — working directory for LightRAG graph files
                          (default: ~/.operon/lightrag)
      OPENAI_API_KEY    — required for LightRAG's default LLM/embedding
      (or configure with a custom llm_func / embedding_func at init time)

    Usage
    -----
      from core.rag import LightRAGBackend

      lrag = LightRAGBackend()
      if lrag.available():
          lrag.index_text("LightRAG is a graph-based RAG system...", source="intro")
          results = lrag.query("What is LightRAG?")
          context = lrag.as_context_block("What is LightRAG?")
    """

    def __init__(
        self,
        work_dir: Optional[str] = None,
        mode: str = "hybrid",    # naive | local | global | hybrid
    ) -> None:
        self._mode     = mode
        self._work_dir = Path(
            work_dir
            or os.environ.get("LIGHTRAG_WORK_DIR", "")
            or (Path.home() / ".operon" / "lightrag")
        )
        self._rag      = None
        self._error    = ""
        self._init()

    def _init(self) -> None:
        try:
            from lightrag import LightRAG, QueryParam
            from lightrag.llm import gpt_4o_mini_complete, openai_embed
            self._work_dir.mkdir(parents=True, exist_ok=True)
            self._rag = LightRAG(
                working_dir=str(self._work_dir),
                llm_model_func=gpt_4o_mini_complete,
                embedding_func=openai_embed,
            )
            self._QueryParam = QueryParam
        except ImportError:
            self._error = "lightrag-hku not installed. Run: pip install lightrag-hku"
        except Exception as e:
            self._error = str(e)

    def available(self) -> bool:
        """Return True if LightRAG is installed and initialised."""
        return self._rag is not None

    def index_text(self, text: str, source: str = "") -> Dict[str, Any]:
        """Insert text into the LightRAG graph."""
        if not self.available():
            return {"success": False, "error": self._error}
        try:
            import asyncio
            asyncio.get_event_loop().run_until_complete(self._rag.ainsert(text))
            return {"success": True, "source": source, "error": ""}
        except Exception as e:
            return {"success": False, "source": source, "error": str(e)}

    def index_file(self, path: str, force: bool = False) -> Dict[str, Any]:
        """Index a file using LightRAG."""
        if not self.available():
            return {"source": path, "chunks_added": 0, "error": self._error}
        p = Path(path).expanduser()
        if not p.exists():
            return {"source": str(p), "chunks_added": 0, "error": f"File not found: {p}"}
        text = _extract_text(p)
        result = self.index_text(text, source=str(p))
        return {
            "source":       str(p),
            "chunks_added": 1 if result["success"] else 0,
            "error":        result.get("error", ""),
        }

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: float = 0.0,
        sources: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query using LightRAG graph traversal."""
        if not self.available():
            return []
        try:
            import asyncio
            response = asyncio.get_event_loop().run_until_complete(
                self._rag.aquery(
                    query_text,
                    param=self._QueryParam(mode=self._mode, top_k=top_k),
                )
            )
            return [{"source": "lightrag", "chunk_index": 0, "content": str(response), "score": 1.0}]
        except Exception:
            return []

    def as_context_block(self, query: str, top_k: int = 4) -> str:
        """Return a formatted context block for system prompt injection."""
        results = self.query(query, top_k=top_k)
        if not results:
            return ""
        lines = [f"[LIGHTRAG CONTEXT — graph-aware retrieval]"]
        for r in results:
            lines.append(r["content"])
        lines.append("[END LIGHTRAG CONTEXT]")
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        return {
            "backend":   "lightrag",
            "available": self.available(),
            "mode":      self._mode,
            "work_dir":  str(self._work_dir),
            "error":     self._error,
        }


# ---------------------------------------------------------------------------
# Factory — pick the best available backend
# ---------------------------------------------------------------------------

def create_rag_pipeline(
    prefer_lightrag: bool = False,
    lightrag_mode:   str  = "hybrid",
    **kwargs,
) -> "RAGPipeline | LightRAGBackend":
    """
    Return the best available RAG backend.

    Parameters
    ----------
    prefer_lightrag : If True, attempt LightRAG first (requires lightrag-hku + OPENAI_API_KEY)
    lightrag_mode   : LightRAG query mode — 'naive' | 'local' | 'global' | 'hybrid'
    **kwargs        : Passed to RAGPipeline constructor if used as fallback
    """
    if prefer_lightrag:
        backend = LightRAGBackend(mode=lightrag_mode)
        if backend.available():
            return backend
    return RAGPipeline(**kwargs)
