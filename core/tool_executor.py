"""
core/tool_executor.py — Dedicated Tool Dispatch & Result Management

Extracted from the main.py monolith to provide:
  - ToolResult classification (code / text / binary / image / error)
  - ErrorClassifier  — retry vs fatal vs user-visible errors
  - SchemaSanitizer  — strips unsupported JSON Schema fields before sending to models
  - OutputCapEnforcer — per-tool output limits, truncation, storage offload
  - ToolDispatcher   — single entry point for executing any registered tool
  - RetryPolicy      — per-tool retry config with exponential backoff

Inspired by Hermes tool_executor.py (910 LOC) architecture.

Usage:
    from core.tool_executor import ToolDispatcher, ToolResult, ErrorClassifier

    dispatcher = ToolDispatcher(registry)
    result = dispatcher.execute("shell_exec", {"cmd": "ls -la"})
    if result.should_retry:
        result = dispatcher.execute("shell_exec", {"cmd": "ls -la"})
    print(result.text_content())
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

log = logging.getLogger("operon.tool_executor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_OUTPUT_CHARS = 32_000   # 8k tokens @ 4 chars/token
_CODE_OUTPUT_CAP_CHARS    = 48_000
_IMAGE_OUTPUT_CAP_CHARS   = 64_000
_BINARY_OUTPUT_CAP_CHARS  = 16_000
_TRUNCATION_MARKER        = "\n[...output truncated at {n:,} chars — full result in {path}...]"

_RETRY_ERRORS = frozenset({
    "timeout", "rate limit", "rate_limit", "429", "503", "overloaded",
    "connection reset", "network", "temporary", "transient", "retriable",
    "request failed", "502 bad gateway", "500 internal",
})

_FATAL_ERRORS = frozenset({
    "permission denied", "not found", "no such file", "invalid argument",
    "bad request", "authentication", "unauthorized", "403", "401",
    "syntax error", "unsupported", "not implemented",
})

_SCHEMA_UNSUPPORTED_KEYS = frozenset({
    "default", "examples", "$schema", "$id", "$ref", "allOf", "anyOf", "oneOf",
    "if", "then", "else", "const", "contains", "propertyNames",
    "minContains", "maxContains", "additionalItems", "unevaluatedProperties",
    "unevaluatedItems", "deprecated", "readOnly", "writeOnly",
    "contentMediaType", "contentEncoding",
})


# ---------------------------------------------------------------------------
# ToolResult classification
# ---------------------------------------------------------------------------

class ResultType(str, Enum):
    TEXT   = "text"
    CODE   = "code"
    IMAGE  = "image"
    BINARY = "binary"
    JSON   = "json"
    ERROR  = "error"
    EMPTY  = "empty"


@dataclass
class ToolResult:
    """
    Structured output from a tool execution.
    Carries the raw result plus classification metadata.
    """
    tool_name:   str
    success:     bool
    raw:         Any                    = None   # raw return value from tool fn
    result_type: ResultType             = ResultType.TEXT
    content:     str                    = ""     # text representation
    image_b64:   Optional[str]         = None   # base64 if image
    storage_path: Optional[str]        = None   # path if offloaded to disk
    tokens_approx: int                 = 0
    duration_ms:   float               = 0.0
    error:         Optional[str]       = None
    should_retry:  bool                = False
    retry_after:   float               = 0.0    # seconds (for rate-limit)
    metadata:      Dict[str, Any]      = field(default_factory=dict)

    def text_content(self, max_chars: int = _DEFAULT_MAX_OUTPUT_CHARS) -> str:
        """Return content truncated to max_chars."""
        if not self.content:
            return self.error or ""
        if len(self.content) <= max_chars:
            return self.content
        half = max_chars // 2
        return self.content[:half] + f"\n[...truncated {len(self.content) - max_chars:,} chars...]"

    def to_context_string(self) -> str:
        """Format for insertion into the LLM context."""
        if not self.success:
            return f"[TOOL_ERROR: {self.tool_name}]\n{self.error or 'unknown error'}"
        if self.result_type == ResultType.IMAGE:
            return f"[TOOL_RESULT: {self.tool_name}] (image — {len(self.image_b64 or '')} b64 chars)"
        prefix = f"[TOOL_RESULT: {self.tool_name}]\n"
        return prefix + self.text_content()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool":         self.tool_name,
            "success":      self.success,
            "result_type":  self.result_type.value,
            "tokens_approx": self.tokens_approx,
            "duration_ms":  round(self.duration_ms, 1),
            "should_retry": self.should_retry,
            "error":        self.error,
            "storage_path": self.storage_path,
        }


# ---------------------------------------------------------------------------
# Result classifier
# ---------------------------------------------------------------------------

class ResultClassifier:
    """
    Classify a tool's raw output into a ResultType.
    Also estimates approximate token cost.
    """

    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
    _CODE_TOOLS = {
        "shell_exec", "code_exec", "python_exec", "docker_exec",
        "ssh_exec", "cloud_exec",
    }
    _JSON_TOOLS = {
        "github_ops", "http_client", "web_search", "data_analysis",
        "db_ops", "llm_task",
    }

    def classify(self, tool_name: str, raw: Any) -> Tuple[ResultType, str, Optional[str]]:
        """
        Returns (result_type, content_string, image_b64_or_none).
        """
        if raw is None:
            return ResultType.EMPTY, "", None

        # Already a dict with "output" key (standard Operon tool return)
        if isinstance(raw, dict):
            return self._classify_dict(tool_name, raw)

        # Already a string
        if isinstance(raw, str):
            return self._classify_string(tool_name, raw)

        # Bytes → image or binary
        if isinstance(raw, bytes):
            return self._classify_bytes(tool_name, raw)

        # Fallback
        return ResultType.TEXT, str(raw), None

    def _classify_dict(
        self, tool_name: str, d: Dict
    ) -> Tuple[ResultType, str, Optional[str]]:
        """Standard Operon format: {success, output, error}"""
        if not d.get("success", True):
            err = str(d.get("error", d.get("output", "tool failed")))
            return ResultType.ERROR, err, None

        raw_out = d.get("output", d.get("result", ""))

        # Image tool
        if isinstance(raw_out, bytes):
            return self._classify_bytes(tool_name, raw_out)

        # Dict inside output → JSON
        if isinstance(raw_out, (dict, list)):
            try:
                return ResultType.JSON, json.dumps(raw_out, indent=2), None
            except Exception:
                return ResultType.TEXT, str(raw_out), None

        text = str(raw_out) if raw_out is not None else ""

        # Tool-based heuristics
        if tool_name in self._CODE_TOOLS:
            return ResultType.CODE, text, None
        if tool_name in self._JSON_TOOLS:
            stripped = text.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                return ResultType.JSON, stripped, None

        return ResultType.TEXT, text, None

    def _classify_string(
        self, tool_name: str, s: str
    ) -> Tuple[ResultType, str, Optional[str]]:
        stripped = s.strip()
        # Base64 image data URI?
        if stripped.startswith("data:image/"):
            b64 = stripped.split(",", 1)[-1]
            return ResultType.IMAGE, f"[image data: {len(b64)} b64 chars]", b64
        # JSON?
        if stripped.startswith(("{", "[")):
            try:
                json.loads(stripped)
                return ResultType.JSON, stripped, None
            except Exception:
                pass
        if tool_name in self._CODE_TOOLS:
            return ResultType.CODE, s, None
        return ResultType.TEXT, s, None

    @staticmethod
    def _classify_bytes(
        tool_name: str, data: bytes
    ) -> Tuple[ResultType, str, Optional[str]]:
        # Check for common image magic bytes
        is_image = (
            data[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1")
            or data[:6] in (b"GIF87a", b"GIF89a")
            or data[:4] == b"RIFF"  # WebP
        )
        if is_image:
            b64 = base64.b64encode(data).decode()
            return ResultType.IMAGE, f"[binary image: {len(data):,} bytes]", b64
        # Generic binary
        b64 = base64.b64encode(data).decode()
        return ResultType.BINARY, f"[binary data: {len(data):,} bytes]", None

    @staticmethod
    def estimate_tokens(content: str) -> int:
        return max(1, len(content) // 4)


# ---------------------------------------------------------------------------
# Error classifier
# ---------------------------------------------------------------------------

class ErrorClassifier:
    """
    Classify an exception or error string into:
      - RETRY     → transient, should retry with backoff
      - FATAL     → permanent, don't retry (invalid input, permissions)
      - USER      → should be shown to user (not an agent bug)
      - UNKNOWN   → unclear, treat as fatal
    """

    class ErrorKind(str, Enum):
        RETRY   = "retry"
        FATAL   = "fatal"
        USER    = "user"
        UNKNOWN = "unknown"

    def classify(
        self,
        error: Exception | str,
        tool_name: str = "",
    ) -> Tuple["ErrorClassifier.ErrorKind", float]:
        """
        Returns (ErrorKind, retry_after_seconds).
        retry_after is 0.0 for non-retry errors.
        """
        msg = str(error).lower()

        # Rate-limit: extract retry-after if present
        if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
            wait = self._extract_retry_after(str(error))
            return self.ErrorKind.RETRY, wait or 60.0

        # Other transient
        for pat in _RETRY_ERRORS:
            if pat in msg:
                return self.ErrorKind.RETRY, 5.0

        # Fatal
        for pat in _FATAL_ERRORS:
            if pat in msg:
                return self.ErrorKind.FATAL, 0.0

        # Keyboard / user interruptions
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            return self.ErrorKind.USER, 0.0

        return self.ErrorKind.UNKNOWN, 0.0

    @staticmethod
    def _extract_retry_after(msg: str) -> float:
        """Try to parse a 'retry after N seconds' hint from an error message."""
        m = re.search(r"retry.{0,20}?(\d+(?:\.\d+)?)\s*s", msg, re.I)
        if m:
            return float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s*second", msg, re.I)
        if m:
            return float(m.group(1))
        return 0.0

    def should_retry(self, error: Exception | str, tool_name: str = "") -> bool:
        kind, _ = self.classify(error, tool_name)
        return kind == self.ErrorKind.RETRY

    def is_fatal(self, error: Exception | str, tool_name: str = "") -> bool:
        kind, _ = self.classify(error, tool_name)
        return kind in (self.ErrorKind.FATAL, self.ErrorKind.UNKNOWN)


# ---------------------------------------------------------------------------
# Schema sanitizer
# ---------------------------------------------------------------------------

class SchemaSanitizer:
    """
    Strips JSON Schema fields that are not supported by the Anthropic / OpenAI
    tool API, preventing validation errors from unsupported keywords.
    """

    def sanitize(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Return a new schema with unsupported keys removed (deep copy)."""
        return self._clean(schema)

    def _clean(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: self._clean(v)
                for k, v in obj.items()
                if k not in _SCHEMA_UNSUPPORTED_KEYS
            }
        if isinstance(obj, list):
            return [self._clean(item) for item in obj]
        return obj

    def sanitize_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sanitize all tool definition schemas in a list."""
        result: List[Dict[str, Any]] = []
        for defn in definitions:
            clean = dict(defn)
            if "input_schema" in clean:
                clean["input_schema"] = self.sanitize(clean["input_schema"])
            if "parameters" in clean:
                clean["parameters"] = self.sanitize(clean["parameters"])
            result.append(clean)
        return result


# ---------------------------------------------------------------------------
# Output cap enforcer
# ---------------------------------------------------------------------------

class OutputCapEnforcer:
    """
    Enforce per-type output character caps, offloading large outputs to disk
    rather than truncating and losing data.
    """

    # Type-specific caps
    _CAPS: Dict[ResultType, int] = {
        ResultType.CODE:   _CODE_OUTPUT_CAP_CHARS,
        ResultType.TEXT:   _DEFAULT_MAX_OUTPUT_CHARS,
        ResultType.JSON:   _DEFAULT_MAX_OUTPUT_CHARS,
        ResultType.IMAGE:  _IMAGE_OUTPUT_CAP_CHARS,
        ResultType.BINARY: _BINARY_OUTPUT_CAP_CHARS,
        ResultType.ERROR:  8_000,
        ResultType.EMPTY:  0,
    }

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        tool_overrides: Optional[Dict[str, int]] = None,
    ) -> None:
        self._storage_dir    = Path(storage_dir) if storage_dir else None
        self._tool_overrides = tool_overrides or {}

    def enforce(self, result: ToolResult) -> ToolResult:
        """Enforce output cap on a ToolResult. Mutates and returns it."""
        if not result.content:
            return result

        cap = self._tool_overrides.get(result.tool_name,
              self._CAPS.get(result.result_type, _DEFAULT_MAX_OUTPUT_CHARS))

        if len(result.content) <= cap:
            return result

        # Offload to storage if dir configured
        if self._storage_dir:
            path = self._offload(result)
            if path:
                result.storage_path = path
                result.content = result.content[:cap // 4] + _TRUNCATION_MARKER.format(
                    n=cap, path=path
                )
                result.metadata["offloaded"] = True
                return result

        # Otherwise just truncate
        result.content = result.content[:cap] + f"\n[...{len(result.content) - cap:,} chars truncated]"
        return result

    def _offload(self, result: ToolResult) -> Optional[str]:
        """Write content to a temp file and return its path."""
        try:
            import hashlib
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{result.tool_name}_{int(time.time())}_{hashlib.md5(result.content[:64].encode()).hexdigest()[:8]}.txt"
            path  = self._storage_dir / fname
            path.write_text(result.content, errors="replace")
            return str(path)
        except Exception as e:
            log.warning("OutputCapEnforcer: failed to offload: %s", e)
            return None


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

@dataclass
class RetryConfig:
    max_attempts: int   = 3
    base_delay:   float = 1.0      # seconds
    backoff:      float = 2.0      # exponential factor
    max_delay:    float = 60.0
    enabled:      bool  = True
    jitter:       bool  = True

    def delay_for(self, attempt: int) -> float:
        """Return wait time in seconds for given attempt (0-indexed)."""
        import random
        d = min(self.base_delay * (self.backoff ** attempt), self.max_delay)
        if self.jitter:
            d *= (0.8 + random.random() * 0.4)
        return d


class RetryPolicyManager:
    """Manages per-tool retry configurations."""

    _DEFAULTS: Dict[str, RetryConfig] = {
        "shell_exec":    RetryConfig(max_attempts=1, enabled=False),
        "docker_exec":   RetryConfig(max_attempts=1, enabled=False),
        "http_client":   RetryConfig(max_attempts=3, base_delay=2.0),
        "web_search":    RetryConfig(max_attempts=3, base_delay=1.0),
        "github_ops":    RetryConfig(max_attempts=3, base_delay=5.0),
        "llm_task":      RetryConfig(max_attempts=3, base_delay=10.0),
    }

    def __init__(self) -> None:
        self._policies: Dict[str, RetryConfig] = dict(self._DEFAULTS)
        self._default = RetryConfig()

    def get(self, tool_name: str) -> RetryConfig:
        return self._policies.get(tool_name, self._default)

    def set(self, tool_name: str, config: RetryConfig) -> None:
        self._policies[tool_name] = config

    def enable(self, tool_name: str) -> None:
        self.get(tool_name).enabled = True

    def disable(self, tool_name: str) -> None:
        self.get(tool_name).enabled = False

    def list(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "max_attempts": cfg.max_attempts,
                "base_delay":   cfg.base_delay,
                "backoff":      cfg.backoff,
                "enabled":      cfg.enabled,
            }
            for name, cfg in self._policies.items()
        }


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """
    Central tool execution entry point.

    Handles:
      - Looking up the tool function by name
      - Input validation
      - Result classification
      - Output cap enforcement
      - Retry with backoff on transient errors
      - Duration tracking
    """

    def __init__(
        self,
        tool_registry: Optional[Any]   = None,    # core.tools.registry.ToolRegistry
        storage_dir:   Optional[str]   = None,
        tool_overrides: Optional[Dict[str, int]] = None,
    ) -> None:
        self._registry       = tool_registry
        self._classifier     = ResultClassifier()
        self._error_clf      = ErrorClassifier()
        self._sanitizer      = SchemaSanitizer()
        self._cap_enforcer   = OutputCapEnforcer(storage_dir, tool_overrides)
        self._retry_mgr      = RetryPolicyManager()
        self._call_counts:   Dict[str, int]   = {}
        self._error_counts:  Dict[str, int]   = {}

    def execute(
        self,
        tool_name:  str,
        params:     Dict[str, Any],
        timeout:    Optional[float] = None,
    ) -> ToolResult:
        """
        Execute a tool and return a ToolResult.
        Retries on transient errors per per-tool policy.
        """
        retry_cfg  = self._retry_mgr.get(tool_name)
        last_error: Optional[Exception] = None

        for attempt in range(retry_cfg.max_attempts if retry_cfg.enabled else 1):
            result = self._execute_once(tool_name, params, timeout)

            if result.success or not result.should_retry:
                return result

            last_error = Exception(result.error or "tool failed")
            if attempt < retry_cfg.max_attempts - 1:
                wait = max(result.retry_after, retry_cfg.delay_for(attempt))
                log.info(
                    "Tool %s retry %d/%d in %.1fs: %s",
                    tool_name, attempt + 1, retry_cfg.max_attempts, wait, result.error,
                )
                time.sleep(wait)

        # All retries exhausted
        return ToolResult(
            tool_name=tool_name,
            success=False,
            result_type=ResultType.ERROR,
            error=str(last_error),
            should_retry=False,
        )

    def _execute_once(
        self,
        tool_name: str,
        params:    Dict[str, Any],
        timeout:   Optional[float],
    ) -> ToolResult:
        """Single (non-retrying) tool execution."""
        self._call_counts[tool_name] = self._call_counts.get(tool_name, 0) + 1
        start = time.time()

        try:
            fn = self._resolve(tool_name)
            if fn is None:
                return ToolResult(
                    tool_name=tool_name, success=False,
                    result_type=ResultType.ERROR,
                    error=f"Tool '{tool_name}' not found in registry",
                )

            # Execute (with optional timeout)
            if timeout:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(fn, **params)
                    try:
                        raw = future.result(timeout=timeout)
                    except concurrent.futures.TimeoutError:
                        return ToolResult(
                            tool_name=tool_name, success=False,
                            result_type=ResultType.ERROR,
                            error=f"Tool '{tool_name}' timed out after {timeout:.0f}s",
                            should_retry=True,
                        )
            else:
                raw = fn(**params)

            duration_ms = (time.time() - start) * 1000
            result_type, content, image_b64 = self._classifier.classify(tool_name, raw)

            # Build result
            res = ToolResult(
                tool_name=tool_name,
                success=True,
                raw=raw,
                result_type=result_type,
                content=content,
                image_b64=image_b64,
                tokens_approx=ResultClassifier.estimate_tokens(content),
                duration_ms=duration_ms,
            )

            # Check if the tool itself reported failure
            if isinstance(raw, dict) and not raw.get("success", True):
                err = str(raw.get("error", raw.get("output", "tool reported failure")))
                kind, wait = self._error_clf.classify(err, tool_name)
                res.success      = False
                res.error        = err
                res.result_type  = ResultType.ERROR
                res.should_retry = kind == ErrorClassifier.ErrorKind.RETRY
                res.retry_after  = wait
                self._error_counts[tool_name] = self._error_counts.get(tool_name, 0) + 1

            # Enforce output cap then inline-compress (TokenJuice)
            res = self._cap_enforcer.enforce(res)
            if res.content:
                try:
                    from core.tokenjuice import compress as _tj_compress
                    res.content = _tj_compress(res.content, tool_name=tool_name)
                    res.tokens_approx = ResultClassifier.estimate_tokens(res.content)
                except Exception:
                    pass  # compression is best-effort; never break tool execution
            return res

        except Exception as exc:
            duration_ms = (time.time() - start) * 1000
            kind, wait  = self._error_clf.classify(exc, tool_name)
            self._error_counts[tool_name] = self._error_counts.get(tool_name, 0) + 1
            log.warning("Tool %s error (%s): %s", tool_name, kind.value, exc)

            return ToolResult(
                tool_name=tool_name,
                success=False,
                result_type=ResultType.ERROR,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=duration_ms,
                should_retry=kind == ErrorClassifier.ErrorKind.RETRY,
                retry_after=wait,
            )

    def _resolve(self, tool_name: str) -> Optional[Callable]:
        """Resolve a tool name to a callable."""
        if self._registry is None:
            return None
        # Support both dict-of-fn and ToolRegistry objects
        if isinstance(self._registry, dict):
            return self._registry.get(tool_name)
        # Try standard Operon registry API
        try:
            return self._registry.get_tool(tool_name)
        except Exception:
            pass
        try:
            return self._registry.resolve(tool_name)
        except Exception:
            pass
        return None

    def sanitize_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sanitize a list of tool definitions for LLM submission."""
        return self._sanitizer.sanitize_tool_definitions(definitions)

    def stats(self) -> Dict[str, Any]:
        """Return call and error statistics."""
        return {
            "total_calls":  sum(self._call_counts.values()),
            "total_errors": sum(self._error_counts.values()),
            "per_tool":     {
                name: {
                    "calls":  self._call_counts.get(name, 0),
                    "errors": self._error_counts.get(name, 0),
                }
                for name in set(list(self._call_counts) + list(self._error_counts))
            },
        }

    def set_retry_policy(self, tool_name: str, config: RetryConfig) -> None:
        """Override the retry policy for a specific tool."""
        self._retry_mgr.set(tool_name, config)

    def list_retry_policies(self) -> Dict[str, Dict[str, Any]]:
        """List all configured retry policies."""
        return self._retry_mgr.list()


# ---------------------------------------------------------------------------
# Parallel tool dispatcher
# ---------------------------------------------------------------------------

class ParallelToolDispatcher(ToolDispatcher):
    """
    Executes multiple tool calls concurrently using a thread pool.
    Falls back to sequential on exceptions.
    """

    def __init__(
        self,
        tool_registry: Optional[Any] = None,
        max_workers:   int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(tool_registry, **kwargs)
        self._max_workers = max_workers

    def execute_parallel(
        self,
        calls: List[Tuple[str, Dict[str, Any]]],
        timeout: Optional[float] = None,
    ) -> List[ToolResult]:
        """
        Execute multiple (tool_name, params) pairs concurrently.
        Returns results in the same order as `calls`.
        """
        import concurrent.futures

        results: List[Optional[ToolResult]] = [None] * len(calls)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(calls))
        ) as pool:
            futures = {
                pool.submit(self.execute, name, params, timeout): i
                for i, (name, params) in enumerate(calls)
            }
            done_iter = concurrent.futures.as_completed(
                futures, timeout=(timeout or 120)
            )
            for future in done_iter:
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = ToolResult(
                        tool_name=calls[idx][0],
                        success=False,
                        result_type=ResultType.ERROR,
                        error=str(e),
                    )

        # Fill any missing (timeout)
        for i, (name, _) in enumerate(calls):
            if results[i] is None:
                results[i] = ToolResult(
                    tool_name=name, success=False,
                    result_type=ResultType.ERROR,
                    error="Parallel execution timed out",
                )

        return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_dispatcher: Optional[ToolDispatcher] = None


def get_dispatcher(
    tool_registry: Optional[Any] = None,
    storage_dir:   Optional[str] = None,
) -> ToolDispatcher:
    """Return (or create) the module-level default dispatcher."""
    global _default_dispatcher
    if _default_dispatcher is None or tool_registry is not None:
        _default_dispatcher = ToolDispatcher(
            tool_registry=tool_registry,
            storage_dir=storage_dir,
        )
    return _default_dispatcher


def classify_result(tool_name: str, raw: Any) -> ToolResult:
    """Convenience: classify a raw tool result without dispatch."""
    clf = ResultClassifier()
    rt, content, img = clf.classify(tool_name, raw)
    return ToolResult(
        tool_name=tool_name,
        success=True,
        raw=raw,
        result_type=rt,
        content=content,
        image_b64=img,
        tokens_approx=ResultClassifier.estimate_tokens(content),
    )


def sanitize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience: sanitize a single JSON Schema dict."""
    return SchemaSanitizer().sanitize(schema)
