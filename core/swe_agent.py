"""
core/swe_agent.py — Operon Software Engineering Agent

Full SWE loop:
  1. Issue parsing (GitHub issue / free-text task)
  2. Codebase localisation (relevant files via AST + grep)
  3. Fix planning (LLM-generated step-by-step plan)
  4. Patch generation (unified-diff patches)
  5. Patch application (apply to working copy)
  6. Test execution (run existing + generated tests)
  7. Verification loop (iterate until tests pass, max N retries)
  8. PR creation (via github_ops.py)
  9. Trajectory recording (every action logged for debugging)

Inspired by Hermes SWE-bench agent and OpenClaw code-agent.

Usage:
    from core.swe_agent import SWEAgent, SWETask

    agent = SWEAgent(repo_path="/path/to/repo")
    result = agent.solve(SWETask(
        title="Fix off-by-one error in pagination",
        body="When page=0 is passed, the query returns page 1 instead of page 0...",
        issue_number=42,
        repo="owner/repo",
    ))
    print(result.summary())
"""

from __future__ import annotations

import ast
import difflib
import fnmatch
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("operon.swe_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_FIX_RETRIES    = 5
_MAX_CONTEXT_FILES  = 10
_MAX_FILE_CHARS     = 8_000      # chars read per file for context
_PATCH_APPLY_TRIES  = 3
_TEST_TIMEOUT_SEC   = 120

_EXCLUDE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", "dist", "build", "*.egg-info", ".mypy_cache",
}

_CODE_EXTS = {
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c",
    ".rb", ".php", ".cs", ".swift", ".kt", ".scala",
}

_TEST_PATTERNS = [
    "test_*.py", "*_test.py", "tests/**/*.py",
    "spec/**/*.js", "**/*.test.ts", "**/*.spec.ts",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class SWEState(str, Enum):
    PENDING     = "pending"
    LOCALISING  = "localising"
    PLANNING    = "planning"
    PATCHING    = "patching"
    TESTING     = "testing"
    VERIFYING   = "verifying"
    DONE        = "done"
    FAILED      = "failed"


@dataclass
class SWETask:
    """Input task / issue description."""
    title:        str
    body:         str             = ""
    issue_number: Optional[int]   = None
    repo:         str             = ""          # "owner/repo"
    branch:       str             = ""          # target branch
    labels:       List[str]       = field(default_factory=list)
    assignees:    List[str]       = field(default_factory=list)

    def full_text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()


@dataclass
class FileHunk:
    """A file + relevant line range identified during localisation."""
    path:       str
    start_line: int               = 1
    end_line:   int               = -1          # -1 = EOF
    relevance:  float             = 1.0
    reason:     str               = ""

    def read(self, repo_path: Path) -> str:
        abs_path = repo_path / self.path
        if not abs_path.exists():
            return ""
        try:
            text = abs_path.read_text(errors="replace")
            lines = text.splitlines()
            end = self.end_line if self.end_line != -1 else len(lines)
            snippet = "\n".join(lines[max(0, self.start_line - 1):end])
            if len(snippet) > _MAX_FILE_CHARS:
                snippet = snippet[:_MAX_FILE_CHARS] + "\n[...truncated...]"
            return snippet
        except Exception as e:
            log.warning("FileHunk.read failed for %s: %s", self.path, e)
            return ""


@dataclass
class FilePatch:
    """A unified diff patch for one file."""
    path:      str
    diff:      str               # unified diff text
    is_new:    bool = False      # True if creating a new file
    is_delete: bool = False      # True if deleting a file


@dataclass
class TestRun:
    """Result of running the test suite."""
    passed:   int   = 0
    failed:   int   = 0
    errors:   int   = 0
    skipped:  int   = 0
    output:   str   = ""
    duration: float = 0.0
    cmd:      str   = ""

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.errors == 0

    def summary(self) -> str:
        s = f"passed={self.passed} failed={self.failed} errors={self.errors}"
        if self.skipped:
            s += f" skipped={self.skipped}"
        s += f" ({self.duration:.1f}s)"
        return s


@dataclass
class TrajectoryEvent:
    """One action in the SWE agent trajectory."""
    step:      int
    action:    str           # e.g. "localise", "patch", "test"
    detail:    str
    timestamp: float = field(default_factory=time.time)
    ok:        bool  = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":      self.step,
            "action":    self.action,
            "detail":    self.detail[:500],
            "timestamp": self.timestamp,
            "ok":        self.ok,
        }


@dataclass
class SWEResult:
    """Final result of a SWE agent run."""
    task:           SWETask
    state:          SWEState        = SWEState.PENDING
    hunks:          List[FileHunk]  = field(default_factory=list)
    patches:        List[FilePatch] = field(default_factory=list)
    test_runs:      List[TestRun]   = field(default_factory=list)
    trajectory:     List[TrajectoryEvent] = field(default_factory=list)
    pr_url:         str             = ""
    branch_name:    str             = ""
    error:          str             = ""
    retries:        int             = 0
    started_at:     float           = field(default_factory=time.time)
    finished_at:    float           = 0.0

    @property
    def duration(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def succeeded(self) -> bool:
        return self.state == SWEState.DONE

    @property
    def last_test(self) -> Optional[TestRun]:
        return self.test_runs[-1] if self.test_runs else None

    def summary(self) -> str:
        lines = [
            f"SWE Task: {self.task.title}",
            f"State: {self.state.value}",
            f"Duration: {self.duration:.1f}s",
            f"Retries: {self.retries}",
            f"Files patched: {len(self.patches)}",
        ]
        if self.last_test:
            lines.append(f"Final tests: {self.last_test.summary()}")
        if self.pr_url:
            lines.append(f"PR: {self.pr_url}")
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task":       {"title": self.task.title, "repo": self.task.repo},
            "state":      self.state.value,
            "duration":   round(self.duration, 2),
            "retries":    self.retries,
            "patches":    [{"path": p.path, "lines": len(p.diff.splitlines())}
                           for p in self.patches],
            "last_test":  self.last_test.summary() if self.last_test else None,
            "pr_url":     self.pr_url,
            "error":      self.error,
            "trajectory": [e.to_dict() for e in self.trajectory[-20:]],
        }


# ---------------------------------------------------------------------------
# Issue parser
# ---------------------------------------------------------------------------

class IssueParser:
    """Extract structured info from a GitHub issue or free-text task."""

    # Regex patterns that suggest relevant files or symbols
    _FILE_RE  = re.compile(r"[`'\"]?([\w/.\-]+\.(?:py|js|ts|go|rs|java|cpp|rb|php|cs))[`'\"]?")
    _FUNC_RE  = re.compile(r"`([\w_]+)\(`")
    _CLASS_RE = re.compile(r"`((?:[A-Z][a-z]+){2,})`")
    _LINE_RE  = re.compile(r"(?:line|L)[\s#]?(\d+)")
    _ERROR_RE = re.compile(
        r"((?:Error|Exception|Traceback|Warning|FAIL)[:\s][^\n]{0,120})",
        re.IGNORECASE,
    )

    def parse(self, task: SWETask) -> Dict[str, Any]:
        text = task.full_text()
        return {
            "mentioned_files":   self._FILE_RE.findall(text),
            "mentioned_funcs":   self._FUNC_RE.findall(text),
            "mentioned_classes": self._CLASS_RE.findall(text),
            "mentioned_lines":   [int(n) for n in self._LINE_RE.findall(text)],
            "error_messages":    self._ERROR_RE.findall(text),
            "keywords":          self._extract_keywords(text),
            "is_bug":            self._is_bug(text),
            "is_feature":        self._is_feature(text),
            "is_refactor":       self._is_refactor(text),
        }

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        stop = {
            "the", "a", "an", "is", "in", "on", "at", "to", "of", "and",
            "or", "but", "for", "with", "this", "that", "when", "it", "be",
            "not", "from", "by", "as", "are", "was", "were", "have", "has",
        }
        words = re.findall(r"\b[a-zA-Z_]\w{3,}\b", text.lower())
        freq: Dict[str, int] = {}
        for w in words:
            if w not in stop:
                freq[w] = freq.get(w, 0) + 1
        return sorted(freq, key=lambda k: -freq[k])[:20]

    @staticmethod
    def _is_bug(text: str) -> bool:
        return bool(re.search(
            r"\b(bug|fix|error|fail|broken|crash|traceback|exception|regression)\b",
            text, re.I,
        ))

    @staticmethod
    def _is_feature(text: str) -> bool:
        return bool(re.search(
            r"\b(feature|add|implement|support|enhance|new|request)\b",
            text, re.I,
        ))

    @staticmethod
    def _is_refactor(text: str) -> bool:
        return bool(re.search(
            r"\b(refactor|cleanup|clean.?up|improve|optimise|optimize|simplify)\b",
            text, re.I,
        ))


# ---------------------------------------------------------------------------
# Code localiser
# ---------------------------------------------------------------------------

class CodeLocaliser:
    """
    Find the most relevant source files for a task.
    Strategy (in order):
      1. Exact file matches from issue text
      2. Keyword grep across the codebase
      3. AST import graph (files importing mentioned symbols)
      4. Fallback: most recently modified files
    """

    def __init__(self, repo_path: Path) -> None:
        self.repo = repo_path

    def locate(self, parsed: Dict[str, Any], max_files: int = _MAX_CONTEXT_FILES) -> List[FileHunk]:
        results: List[FileHunk] = []
        seen: set = set()

        # 1. Direct file mentions
        for fname in parsed.get("mentioned_files", []):
            for p in self._find_file(fname):
                if p not in seen:
                    seen.add(p)
                    results.append(FileHunk(path=p, relevance=1.0, reason="mentioned in issue"))

        # 2. Keyword grep
        keywords = parsed.get("keywords", [])[:8]
        for kw in keywords:
            for p, lineno in self._grep(kw, max_results=3):
                if p not in seen:
                    seen.add(p)
                    results.append(FileHunk(
                        path=p, start_line=max(1, lineno - 5),
                        end_line=lineno + 20, relevance=0.8,
                        reason=f"keyword match: {kw}",
                    ))

        # 3. AST import scan for mentioned function/class names
        symbols = parsed.get("mentioned_funcs", []) + parsed.get("mentioned_classes", [])
        for sym in symbols[:5]:
            for p, lineno in self._grep(sym, max_results=2):
                if p not in seen:
                    seen.add(p)
                    results.append(FileHunk(
                        path=p, start_line=max(1, lineno - 3),
                        end_line=lineno + 30, relevance=0.9,
                        reason=f"symbol match: {sym}",
                    ))

        # 4. Fallback: most recently modified Python files
        if len(results) < 3:
            for p in self._recent_files(n=5):
                if p not in seen:
                    seen.add(p)
                    results.append(FileHunk(path=p, relevance=0.3, reason="recently modified"))

        # Sort by relevance, cap at max_files
        results.sort(key=lambda h: -h.relevance)
        return results[:max_files]

    def _find_file(self, fname: str) -> List[str]:
        results: List[str] = []
        for root, dirs, files in os.walk(self.repo):
            dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIRS]
            for f in files:
                if f == fname or f == os.path.basename(fname):
                    rel = os.path.relpath(os.path.join(root, f), self.repo)
                    results.append(rel)
        return results[:3]

    def _grep(self, keyword: str, max_results: int = 5) -> List[Tuple[str, int]]:
        """Simple in-process grep over source files."""
        results: List[Tuple[str, int]] = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        for root, dirs, files in os.walk(self.repo):
            dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIRS]
            for fname in files:
                if Path(fname).suffix not in _CODE_EXTS:
                    continue
                abs_path = Path(root) / fname
                try:
                    for lineno, line in enumerate(
                        abs_path.read_text(errors="replace").splitlines(), 1
                    ):
                        if pattern.search(line):
                            rel = str(abs_path.relative_to(self.repo))
                            results.append((rel, lineno))
                            if len(results) >= max_results * 3:
                                break
                except Exception:
                    continue
            if len(results) >= max_results * 3:
                break
        return results[:max_results]

    def _recent_files(self, n: int = 5) -> List[str]:
        files: List[Tuple[float, str]] = []
        for root, dirs, fnames in os.walk(self.repo):
            dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIRS]
            for fname in fnames:
                if Path(fname).suffix not in _CODE_EXTS:
                    continue
                abs_path = Path(root) / fname
                try:
                    mtime = abs_path.stat().st_mtime
                    rel = str(abs_path.relative_to(self.repo))
                    files.append((mtime, rel))
                except Exception:
                    continue
        files.sort(reverse=True)
        return [f for _, f in files[:n]]


# ---------------------------------------------------------------------------
# Patch generator
# ---------------------------------------------------------------------------

class PatchGenerator:
    """
    Use an LLM router to generate unified diff patches.
    Falls back to a stub generator when no LLM is available.
    """

    def __init__(self, repo_path: Path) -> None:
        self.repo = repo_path

    def generate(
        self,
        task: SWETask,
        hunks: List[FileHunk],
        plan: str,
        prior_error: str = "",
    ) -> List[FilePatch]:
        """Ask LLM to produce patches for the relevant files."""
        context = self._build_context(hunks)
        prompt  = self._build_prompt(task, context, plan, prior_error)

        try:
            raw = self._call_llm(prompt)
            return self._parse_patches(raw)
        except Exception as e:
            log.warning("PatchGenerator LLM call failed: %s", e)
            return []

    def _build_context(self, hunks: List[FileHunk]) -> str:
        parts: List[str] = []
        for hunk in hunks:
            snippet = hunk.read(self.repo)
            if snippet:
                parts.append(
                    f"=== FILE: {hunk.path} (lines {hunk.start_line}–) ===\n"
                    f"{snippet}\n"
                )
        return "\n".join(parts) if parts else "(no relevant files found)"

    def _build_prompt(
        self,
        task: SWETask,
        context: str,
        plan: str,
        prior_error: str,
    ) -> str:
        parts = [
            "You are a senior software engineer tasked with fixing a codebase issue.",
            "",
            f"## Issue\n{task.full_text()}",
            "",
            f"## Fix Plan\n{plan}",
            "",
            "## Relevant Code\n" + context,
        ]
        if prior_error:
            parts += ["", f"## Previous Attempt Error\n{prior_error}"]
        parts += [
            "",
            "## Instructions",
            "Produce ONLY unified diff patches in this EXACT format:",
            "```diff",
            "--- a/path/to/file.py",
            "+++ b/path/to/file.py",
            "@@ -10,6 +10,8 @@",
            " context line",
            "-removed line",
            "+added line",
            "```",
            "Output one diff block per file. Nothing else — no explanations.",
        ]
        return "\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        from core.router import ModelRouter
        from core.config import ConfigManager
        cfg = ConfigManager()
        router = ModelRouter(cfg)
        return router.complete(
            system="You are a precise code patcher. Respond only with unified diff blocks.",
            messages=[{"role": "user", "content": prompt}],
            model=cfg.get("default_model", ""),
            max_tokens=4096,
        )

    @staticmethod
    def _parse_patches(raw: str) -> List[FilePatch]:
        """Extract ```diff ... ``` blocks from LLM output."""
        patches: List[FilePatch] = []
        # Find all diff code blocks
        block_re = re.compile(
            r"```(?:diff)?\s*\n(.*?)```",
            re.S,
        )
        for match in block_re.finditer(raw):
            diff_text = match.group(1).strip()
            if not diff_text.startswith("---"):
                continue
            # Extract path from --- line
            first_line = diff_text.splitlines()[0]
            m = re.match(r"^---\s+(?:a/)?(.+)$", first_line)
            path = m.group(1).strip() if m else "unknown"
            patches.append(FilePatch(path=path, diff=diff_text))

        # Also try bare diffs (no code fence)
        if not patches:
            bare_re = re.compile(
                r"(^---\s+\S+\n\+\+\+\s+\S+\n(?:@@.*\n(?:[+\- ].*\n)*)+)",
                re.M,
            )
            for match in bare_re.finditer(raw):
                diff_text = match.group(1).strip()
                first_line = diff_text.splitlines()[0]
                m = re.match(r"^---\s+(?:a/)?(.+)$", first_line)
                path = m.group(1).strip() if m else "unknown"
                patches.append(FilePatch(path=path, diff=diff_text))

        return patches


# ---------------------------------------------------------------------------
# Patch applier
# ---------------------------------------------------------------------------

class PatchApplier:
    """Apply unified diff patches to a working directory."""

    def __init__(self, repo_path: Path) -> None:
        self.repo = repo_path

    def apply(self, patch: FilePatch) -> Tuple[bool, str]:
        """
        Try to apply a patch. Returns (success, error_message).
        Strategy: write to temp file, then call `patch -p1`.
        Falls back to Python apply if `patch` binary missing.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".diff", delete=False, dir="/tmp"
        ) as f:
            # Normalise --- a/... → a/... so `patch -p1` strips the leading a/
            f.write(self._normalise_diff(patch.diff))
            tmp_path = f.name

        try:
            # Try GNU patch
            result = subprocess.run(
                ["patch", "-p1", "--forward", "--no-backup-if-mismatch",
                 "-i", tmp_path],
                cwd=str(self.repo),
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return True, ""
            # If already applied, treat as success
            if "already applied" in result.stdout.lower():
                return True, ""
            return False, (result.stdout + result.stderr).strip()
        except FileNotFoundError:
            # `patch` not available — use Python fallback
            return self._python_apply(patch)
        except Exception as e:
            return False, str(e)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def apply_all(self, patches: List[FilePatch]) -> Tuple[int, List[str]]:
        """Apply all patches. Returns (applied_count, errors)."""
        applied = 0
        errors: List[str] = []
        for p in patches:
            ok, err = self.apply(p)
            if ok:
                applied += 1
            else:
                errors.append(f"{p.path}: {err}")
        return applied, errors

    @staticmethod
    def _normalise_diff(diff: str) -> str:
        """Ensure --- a/path and +++ b/path have the `a/` `b/` prefix."""
        lines: List[str] = []
        for line in diff.splitlines():
            if line.startswith("--- ") and not line.startswith("--- a/"):
                line = "--- a/" + line[4:].lstrip("/")
            elif line.startswith("+++ ") and not line.startswith("+++ b/"):
                line = "+++ b/" + line[4:].lstrip("/")
            lines.append(line)
        return "\n".join(lines) + "\n"

    def _python_apply(self, patch: FilePatch) -> Tuple[bool, str]:
        """
        Minimal Python-level patch applier for simple +/- blocks.
        Only handles single-file, single-hunk patches without context.
        """
        try:
            target = self.repo / patch.path
            if not target.exists():
                if patch.is_new:
                    # Extract added lines
                    added = [l[1:] for l in patch.diff.splitlines()
                             if l.startswith("+") and not l.startswith("+++")]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("\n".join(added) + "\n")
                    return True, ""
                return False, f"Target file not found: {patch.path}"
            original = target.read_text(errors="replace")
            patched  = self._apply_hunks(original, patch.diff)
            target.write_text(patched)
            return True, ""
        except Exception as e:
            return False, str(e)

    @staticmethod
    def _apply_hunks(original: str, diff: str) -> str:
        """Apply unified diff hunks to original text. Best-effort."""
        orig_lines = original.splitlines(keepends=True)
        result     = list(orig_lines)
        offset     = 0

        hunk_re = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        diff_lines = diff.splitlines()

        i = 0
        while i < len(diff_lines):
            m = hunk_re.match(diff_lines[i])
            if not m:
                i += 1
                continue
            src_start = int(m.group(1)) - 1   # 0-indexed
            i += 1
            hunk_old: List[str] = []
            hunk_new: List[str] = []
            while i < len(diff_lines):
                l = diff_lines[i]
                if l.startswith("@@") or l.startswith("---") or l.startswith("+++"):
                    break
                if l.startswith("-"):
                    hunk_old.append(l[1:])
                elif l.startswith("+"):
                    hunk_new.append(l[1:])
                elif l.startswith(" "):
                    hunk_old.append(l[1:])
                    hunk_new.append(l[1:])
                i += 1

            # Replace old lines with new lines at src_start + offset
            pos = src_start + offset
            result[pos: pos + len(hunk_old)] = [l + "\n" if not l.endswith("\n") else l
                                                  for l in hunk_new]
            offset += len(hunk_new) - len(hunk_old)

        return "".join(result)


# ---------------------------------------------------------------------------
# Static analysis (LSP-style) — runs before/after a fix to catch errors fast
# ---------------------------------------------------------------------------

@dataclass
class Diagnostic:
    file:     str
    line:     int
    col:      int
    code:     str
    message:  str
    severity: str = "error"   # "error" | "warning"

    def as_line(self) -> str:
        return f"{self.file}:{self.line}:{self.col} {self.severity} [{self.code}] {self.message}"


@dataclass
class AnalysisResult:
    tool:        str
    diagnostics: List["Diagnostic"] = field(default_factory=list)
    ran:         bool = False
    raw:         str = ""

    @property
    def error_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == "error")

    @property
    def clean(self) -> bool:
        return self.ran and not self.diagnostics

    def summary(self) -> str:
        if not self.ran:
            return f"{self.tool}: not available (skipped)"
        if not self.diagnostics:
            return f"{self.tool}: clean ✓"
        errs = self.error_count
        warns = len(self.diagnostics) - errs
        return f"{self.tool}: {errs} error(s), {warns} warning(s)"


class StaticAnalyzer:
    """
    Lightweight LSP-style static analysis for the SWE loop.

    Prefers `ruff` (fast, JSON output), falls back to `pyflakes`, then to
    Python's own `compile()` for a syntax-only check. Always degrades
    gracefully — a missing linter yields a skipped (not failed) result.
    """

    def __init__(self, repo_path: Path) -> None:
        self.repo = Path(repo_path)

    def _have(self, *cmd: str) -> bool:
        try:
            subprocess.run(list(cmd) + ["--version"], capture_output=True,
                           timeout=5, cwd=str(self.repo))
            return True
        except Exception:
            return False

    def analyze(self, files: Optional[List[str]] = None) -> AnalysisResult:
        """Analyze the given Python files (or the whole repo)."""
        py_files = [f for f in (files or []) if f.endswith(".py")]
        if self._have("ruff"):
            return self._run_ruff(py_files)
        if self._have("python", "-m", "pyflakes"):
            return self._run_pyflakes(py_files)
        return self._run_syntax_check(py_files)

    def _run_ruff(self, files: List[str]) -> AnalysisResult:
        target = files or ["."]
        try:
            proc = subprocess.run(
                ["ruff", "check", "--output-format", "json", *target],
                cwd=str(self.repo), capture_output=True, text=True, timeout=60,
            )
            res = AnalysisResult(tool="ruff", ran=True, raw=proc.stdout)
            try:
                for item in json.loads(proc.stdout or "[]"):
                    res.diagnostics.append(Diagnostic(
                        file=item.get("filename", "?"),
                        line=(item.get("location") or {}).get("row", 0),
                        col=(item.get("location") or {}).get("column", 0),
                        code=item.get("code") or "RUFF",
                        message=item.get("message", ""),
                        severity="error",
                    ))
            except Exception:
                pass
            return res
        except Exception as e:
            return AnalysisResult(tool="ruff", ran=False, raw=str(e))

    def _run_pyflakes(self, files: List[str]) -> AnalysisResult:
        target = files or [str(self.repo)]
        try:
            proc = subprocess.run(
                ["python", "-m", "pyflakes", *target],
                cwd=str(self.repo), capture_output=True, text=True, timeout=60,
            )
            res = AnalysisResult(tool="pyflakes", ran=True, raw=proc.stdout + proc.stderr)
            for line in (proc.stdout + proc.stderr).splitlines():
                # format: path:line:col message
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    try:
                        res.diagnostics.append(Diagnostic(
                            file=parts[0], line=int(parts[1]),
                            col=int(parts[2]), code="F",
                            message=parts[3].strip(), severity="error"))
                    except ValueError:
                        continue
            return res
        except Exception as e:
            return AnalysisResult(tool="pyflakes", ran=False, raw=str(e))

    def _run_syntax_check(self, files: List[str]) -> AnalysisResult:
        """Last-resort: compile each file to catch syntax errors only."""
        res = AnalysisResult(tool="py_compile", ran=True)
        for f in files:
            path = (self.repo / f) if not Path(f).is_absolute() else Path(f)
            if not path.exists():
                continue
            try:
                compile(path.read_text(encoding="utf-8", errors="replace"), str(path), "exec")
            except SyntaxError as e:
                res.diagnostics.append(Diagnostic(
                    file=str(f), line=e.lineno or 0, col=e.offset or 0,
                    code="E999", message=f"SyntaxError: {e.msg}", severity="error"))
            except Exception:
                continue
        return res


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestRunner:
    """
    Discover and run tests for the repository.
    Supports pytest, unittest, node test runners.
    """

    def __init__(self, repo_path: Path) -> None:
        self.repo = repo_path

    def run(
        self,
        changed_files: Optional[List[str]] = None,
        timeout: int = _TEST_TIMEOUT_SEC,
    ) -> TestRun:
        """Run the test suite and return a TestRun result."""
        cmd = self._pick_cmd(changed_files)
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.repo),
                capture_output=True, text=True,
                timeout=timeout,
            )
            output = proc.stdout + proc.stderr
            duration = time.time() - start
            return self._parse_output(output, cmd=shlex_join(cmd), duration=duration)
        except subprocess.TimeoutExpired:
            return TestRun(
                errors=1, output="Test run timed out",
                duration=timeout, cmd=shlex_join(cmd),
            )
        except FileNotFoundError as e:
            return TestRun(errors=1, output=f"Runner not found: {e}", cmd=str(cmd))
        except Exception as e:
            return TestRun(errors=1, output=str(e), cmd=str(cmd))

    def _pick_cmd(self, changed_files: Optional[List[str]]) -> List[str]:
        """Chose the best test command for this repo."""
        # If pytest is available and we changed specific files, focus on them
        if self._has_pytest():
            if changed_files:
                # Find test files related to changed files
                related = self._find_related_tests(changed_files)
                if related:
                    return ["python", "-m", "pytest"] + related + ["-x", "-q", "--tb=short"]
            return ["python", "-m", "pytest", "-x", "-q", "--tb=short"]

        # Fallback: unittest discovery
        if (self.repo / "tests").is_dir():
            return ["python", "-m", "unittest", "discover", "-s", "tests", "-q"]
        return ["python", "-m", "unittest", "discover", "-q"]

    def _has_pytest(self) -> bool:
        try:
            subprocess.run(
                ["python", "-m", "pytest", "--version"],
                capture_output=True, timeout=5, cwd=str(self.repo),
            )
            return True
        except Exception:
            return False

    def _find_related_tests(self, changed: List[str]) -> List[str]:
        related: List[str] = []
        for changed_file in changed:
            stem = Path(changed_file).stem
            # test_<stem>.py or <stem>_test.py
            for pattern in [f"test_{stem}.py", f"{stem}_test.py"]:
                for root, _, files in os.walk(self.repo):
                    if any(d in root for d in _EXCLUDE_DIRS):
                        continue
                    if pattern in files:
                        rel = os.path.relpath(os.path.join(root, pattern), self.repo)
                        related.append(rel)
        return related[:5]

    @staticmethod
    def _parse_output(output: str, cmd: str, duration: float) -> TestRun:
        """Parse pytest/unittest output for pass/fail counts."""
        tr = TestRun(output=output[:4000], cmd=cmd, duration=duration)

        # pytest short summary: "X passed, Y failed, Z error"
        m = re.search(
            r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) error)?(?:, (\d+) skipped)?",
            output,
        )
        if m:
            tr.passed  = int(m.group(1) or 0)
            tr.failed  = int(m.group(2) or 0)
            tr.errors  = int(m.group(3) or 0)
            tr.skipped = int(m.group(4) or 0)
            return tr

        # unittest: "Ran X tests in Y.Ys\nOK" or "FAILED (failures=X)"
        m2 = re.search(r"Ran (\d+) test", output)
        if m2:
            tr.passed = int(m2.group(1))
        if "FAILED" in output:
            mf = re.search(r"failures=(\d+)", output)
            me = re.search(r"errors=(\d+)", output)
            tr.failed = int(mf.group(1)) if mf else 1
            tr.errors = int(me.group(1)) if me else 0
            tr.passed = max(0, tr.passed - tr.failed - tr.errors)
        return tr


def shlex_join(parts: List[str]) -> str:
    import shlex
    return " ".join(shlex.quote(p) for p in parts)


# ---------------------------------------------------------------------------
# Fix planner
# ---------------------------------------------------------------------------

class FixPlanner:
    """Generate a step-by-step fix plan using the LLM."""

    def plan(
        self,
        task: SWETask,
        parsed: Dict[str, Any],
        hunks: List[FileHunk],
        repo_path: Path,
    ) -> str:
        """Returns a textual fix plan."""
        context = "\n".join(
            f"- {h.path} ({h.reason})" for h in hunks[:6]
        )
        prompt = (
            f"## Issue\n{task.full_text()}\n\n"
            f"## Relevant files\n{context}\n\n"
            f"## Task type\n"
            f"bug={parsed.get('is_bug')}, "
            f"feature={parsed.get('is_feature')}, "
            f"refactor={parsed.get('is_refactor')}\n\n"
            "Write a numbered step-by-step fix plan (max 8 steps). "
            "Be concrete: name exact functions/methods to change and what to do. "
            "No code yet — just the plan."
        )
        try:
            from core.router import ModelRouter
            from core.config import ConfigManager
            cfg = ConfigManager()
            return ModelRouter(cfg).complete(
                system="You are a senior software engineer. Be concise and precise.",
                messages=[{"role": "user", "content": prompt}],
                model=cfg.get("default_model", ""),
                max_tokens=1024,
            ) or "(plan unavailable)"
        except Exception as e:
            log.warning("FixPlanner LLM failed: %s", e)
            return f"1. Review {hunks[0].path if hunks else 'relevant files'}\n2. Apply minimal targeted fix\n3. Run tests"


# ---------------------------------------------------------------------------
# Branch manager
# ---------------------------------------------------------------------------

class BranchManager:
    """Git branch operations for the SWE workflow."""

    def __init__(self, repo_path: Path) -> None:
        self.repo = repo_path

    def create_fix_branch(self, task: SWETask) -> str:
        """Create and checkout a fix branch. Returns branch name."""
        slug = re.sub(r"[^\w-]", "-", task.title.lower())[:40].strip("-")
        num  = f"issue-{task.issue_number}" if task.issue_number else "fix"
        branch = f"{num}/{slug}"
        try:
            self._git(["checkout", "-b", branch])
            return branch
        except Exception as e:
            log.warning("BranchManager: could not create branch %s: %s", branch, e)
            return ""

    def commit(self, patches: List[FilePatch], task: SWETask) -> bool:
        """Stage patched files and commit them."""
        paths = [p.path for p in patches]
        if not paths:
            return False
        try:
            self._git(["add"] + paths)
            msg = f"fix: {task.title}"
            if task.issue_number:
                msg += f"\n\nCloses #{task.issue_number}"
            self._git(["commit", "-m", msg,
                       "--author", "Operon SWE Agent <operon@agents.ai>"])
            return True
        except Exception as e:
            log.warning("BranchManager.commit failed: %s", e)
            return False

    def _git(self, args: List[str]) -> str:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(self.repo),
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()


# ---------------------------------------------------------------------------
# PR creator
# ---------------------------------------------------------------------------

class PRCreator:
    """Open a pull request on GitHub after the fix is committed."""

    def create(
        self,
        task: SWETask,
        branch: str,
        test_run: Optional[TestRun],
        patches: List[FilePatch],
    ) -> str:
        """Returns PR URL or empty string on failure."""
        try:
            from tools.github_ops import create_pull_request
            body = self._build_body(task, test_run, patches)
            result = create_pull_request(
                repo=task.repo,
                title=task.title,
                body=body,
                head=branch,
                base=task.branch or "main",
            )
            if result.get("success"):
                return result.get("output", {}).get("html_url", "")
        except Exception as e:
            log.warning("PRCreator failed: %s", e)
        return ""

    @staticmethod
    def _build_body(
        task: SWETask,
        test_run: Optional[TestRun],
        patches: List[FilePatch],
    ) -> str:
        lines = [
            f"## Summary\n{task.body[:500]}" if task.body else "## Summary\nAutomated fix by Operon SWE Agent.",
            "",
            f"## Changes\n" + "\n".join(f"- `{p.path}`" for p in patches[:10]),
        ]
        if test_run:
            status = "✓ Passing" if test_run.ok else "✗ Failing"
            lines += ["", f"## Tests\n{status} — {test_run.summary()}"]
        if task.issue_number:
            lines += ["", f"Closes #{task.issue_number}"]
        lines += ["", "---", "_ Generated by [Operon SWE Agent](https://github.com/operon)_"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SWE Agent
# ---------------------------------------------------------------------------

class SWEAgent:
    """
    Full software engineering agent.

    solve() runs the complete loop:
      parse → localise → plan → patch → apply → test → verify → PR
    """

    def __init__(
        self,
        repo_path:       str   = ".",
        max_retries:     int   = _MAX_FIX_RETRIES,
        open_pr:         bool  = False,
        create_branch:   bool  = False,
        on_event:        Optional[Callable[[TrajectoryEvent], None]] = None,
    ) -> None:
        self.repo          = Path(repo_path).resolve()
        self.max_retries   = max_retries
        self.open_pr       = open_pr
        self.create_branch = create_branch
        self.on_event      = on_event

        self._parser    = IssueParser()
        self._localiser = CodeLocaliser(self.repo)
        self._planner   = FixPlanner()
        self._patcher   = PatchGenerator(self.repo)
        self._applier   = PatchApplier(self.repo)
        self._tester    = TestRunner(self.repo)
        self._branches  = BranchManager(self.repo)
        self._pr        = PRCreator()

    # ── Public API ─────────────────────────────────────────────────────────

    def solve(self, task: SWETask) -> SWEResult:
        """Run the full SWE loop and return a SWEResult."""
        result = SWEResult(task=task)
        step   = 0

        def emit(action: str, detail: str, ok: bool = True) -> None:
            nonlocal step
            step += 1
            ev = TrajectoryEvent(step=step, action=action, detail=detail, ok=ok)
            result.trajectory.append(ev)
            if self.on_event:
                try:
                    self.on_event(ev)
                except Exception:
                    pass
            log.info("[SWE %d] %s: %s", step, action.upper(), detail[:120])

        try:
            # 1. Parse issue
            result.state = SWEState.LOCALISING
            emit("parse", f"title={task.title!r}")
            parsed = self._parser.parse(task)
            emit("parse_result", f"keywords={parsed['keywords'][:5]}, "
                 f"files={parsed['mentioned_files'][:3]}, bug={parsed['is_bug']}")

            # 2. Localise relevant files
            hunks = self._localiser.locate(parsed)
            result.hunks = hunks
            emit("localise", f"found {len(hunks)} relevant file(s): "
                 f"{[h.path for h in hunks[:5]]}")

            # 3. Plan the fix
            result.state = SWEState.PLANNING
            plan = self._planner.plan(task, parsed, hunks, self.repo)
            emit("plan", plan[:300])

            # 4. Create a fix branch (optional)
            if self.create_branch:
                branch = self._branches.create_fix_branch(task)
                result.branch_name = branch
                emit("branch", f"created branch: {branch}", ok=bool(branch))

            # 5. Patch + test loop
            result.state = SWEState.PATCHING
            prior_error  = ""
            final_test: Optional[TestRun] = None

            for attempt in range(1, self.max_retries + 1):
                emit("attempt", f"attempt {attempt}/{self.max_retries}")

                # Generate patches
                patches = self._patcher.generate(task, hunks, plan, prior_error)
                if not patches:
                    emit("patch_gen", "no patches produced by LLM", ok=False)
                    prior_error = "LLM produced no patches — check prompt"
                    result.retries += 1
                    continue

                # Apply patches
                result.state = SWEState.PATCHING
                applied, errors = self._applier.apply_all(patches)
                result.patches = patches
                emit("apply", f"applied {applied}/{len(patches)} patches, "
                     f"errors={errors[:3]}", ok=applied > 0)

                if applied == 0:
                    prior_error = "; ".join(errors[:3])
                    result.retries += 1
                    emit("apply_failed", prior_error, ok=False)
                    continue

                # Run tests
                result.state = SWEState.TESTING
                changed = [p.path for p in patches]
                test_run = self._tester.run(changed_files=changed)
                result.test_runs.append(test_run)
                final_test = test_run
                emit("test", test_run.summary(), ok=test_run.ok)

                if test_run.ok:
                    emit("verify_pass", f"all tests passed on attempt {attempt}")
                    break

                # Tests failed — refine plan for next attempt
                result.state = SWEState.VERIFYING
                prior_error  = f"Tests failed: {test_run.output[-800:]}"
                result.retries += 1
                emit("verify_fail", f"retrying ({attempt}/{self.max_retries}): "
                     f"{test_run.failed} failures", ok=False)

            # 6. Commit + PR
            if result.patches:
                if self.create_branch and result.branch_name:
                    committed = self._branches.commit(result.patches, task)
                    emit("commit", f"committed to {result.branch_name}", ok=committed)

                    if self.open_pr and committed and final_test:
                        pr_url = self._pr.create(task, result.branch_name, final_test, result.patches)
                        result.pr_url = pr_url
                        emit("pr", f"PR created: {pr_url}", ok=bool(pr_url))

            # Final state
            if final_test and final_test.ok:
                result.state = SWEState.DONE
            elif result.patches:
                result.state = SWEState.DONE  # patches applied, tests may still warn
            else:
                result.state = SWEState.FAILED
                result.error = "No patches could be generated or applied"

        except Exception as e:
            result.state = SWEState.FAILED
            result.error = str(e)
            emit("error", str(e), ok=False)
            log.exception("SWEAgent.solve crashed")

        result.finished_at = time.time()
        emit("done", result.summary()[:300])
        return result

    def dry_run(self, task: SWETask) -> Dict[str, Any]:
        """
        Run parse + localise + plan but don't apply any patches.
        Returns a preview of what would happen.
        """
        parsed = self._parser.parse(task)
        hunks  = self._localiser.locate(parsed)
        plan   = self._planner.plan(task, parsed, hunks, self.repo)
        return {
            "parsed":  parsed,
            "files":   [{"path": h.path, "reason": h.reason, "relevance": h.relevance}
                        for h in hunks],
            "plan":    plan,
        }

    def run_tests(self, timeout: int = _TEST_TIMEOUT_SEC) -> TestRun:
        """Run the test suite without making any changes."""
        return self._tester.run(timeout=timeout)


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_default_agent: Optional[SWEAgent] = None


def get_swe_agent(
    repo_path: str = ".",
    max_retries: int = _MAX_FIX_RETRIES,
    open_pr: bool = False,
) -> SWEAgent:
    """Return (or create) the module-level default SWE agent."""
    global _default_agent
    if _default_agent is None:
        _default_agent = SWEAgent(
            repo_path=repo_path,
            max_retries=max_retries,
            open_pr=open_pr,
        )
    return _default_agent


def solve_issue(
    title: str,
    body: str = "",
    repo: str = "",
    issue_number: Optional[int] = None,
    repo_path: str = ".",
) -> SWEResult:
    """Convenience function: solve a single issue."""
    agent = SWEAgent(repo_path=repo_path)
    task  = SWETask(title=title, body=body, repo=repo, issue_number=issue_number)
    return agent.solve(task)
