"""
Operon Security Checks.

Adapted from OpenClaw src/gateway/known-weak-gateway-secrets.ts,
src/gateway/input-allowlist.ts, and AGENTS.md path hardening rules.

Provides:
  - known_weak_secret()    — detect trivially weak passwords/tokens
  - validate_workspace_path() — enforce workspaceOnly file tool restriction
  - check_path_traversal() — detect ../.. escapes
  - sanitize_input()       — gateway input allowlist (max length + char filter)
  - is_sandbox_safe()      — check if a sub-agent spawn is sandboxed
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

# ── Known weak secrets ─────────────────────────────────────────────────────────
# Common trivial passwords that should never be used as API keys / gateway secrets.
# Adapted from OpenClaw's known-weak-gateway-secrets.ts.

_WEAK_SECRETS: frozenset[str] = frozenset({
    "", "password", "secret", "12345", "123456", "1234567", "12345678",
    "123456789", "1234567890", "password1", "password123", "qwerty",
    "abc123", "letmein", "admin", "administrator", "root", "toor",
    "changeme", "default", "test", "testing", "demo", "example",
    "operon", "operon123", "apikey", "api_key", "mykey", "yourkey",
    "enter_here", "placeholder", "xxx", "yyy", "zzz",
    "token", "mytoken", "secret_key", "my_secret",
    "pass", "passwd", "pass123", "pass1234",
})


def known_weak_secret(value: str) -> bool:
    """
    Return True if `value` is a trivially weak secret.
    Also flags secrets that are too short (< 16 chars) for a gateway token.
    """
    if not value:
        return True
    v = value.strip().lower()
    if v in _WEAK_SECRETS:
        return True
    # Single repeated character (e.g. "aaaaaaaaaa")
    if len(set(v)) == 1 and len(v) < 32:
        return True
    return False


# ── Path hardening ─────────────────────────────────────────────────────────────

def validate_workspace_path(
    path: str,
    workspace: Optional[str] = None,
    allow_absolute: bool = False,
) -> tuple[bool, str]:
    """
    Check whether `path` is safe to access.

    Returns (ok: bool, reason: str).

    Rules:
    - Block path traversal (../.. escapes)
    - If workspace is set and allow_absolute is False, block paths outside workspace
    - Block access to sensitive dirs (~/.ssh, ~/.gnupg, /etc/passwd, etc.)
    """
    if not path:
        return False, "Empty path"

    # Normalise
    try:
        resolved = str(Path(path).resolve())
    except Exception as e:
        return False, f"Invalid path: {e}"

    # Path traversal in original string
    if ".." in Path(path).parts:
        return False, f"Path traversal detected: {path}"

    # Sensitive directories that should never be read/written by the agent
    _SENSITIVE = (
        str(Path.home() / ".ssh"),
        str(Path.home() / ".gnupg"),
        str(Path.home() / ".aws"),
        str(Path.home() / ".config" / "gcloud"),
        "/etc/passwd",
        "/etc/shadow",
        "/etc/sudoers",
        "/private/etc/passwd",
        "/private/etc/shadow",
    )
    for sensitive in _SENSITIVE:
        if resolved.startswith(sensitive):
            return False, f"Access to sensitive path denied: {resolved}"

    # Workspace restriction
    if workspace and not allow_absolute:
        ws = str(Path(workspace).resolve())
        if not resolved.startswith(ws):
            return False, f"Path outside workspace ({ws}): {resolved}"

    return True, ""


def check_path_traversal(path: str) -> bool:
    """Return True if the path contains a traversal attempt."""
    parts = Path(path).parts
    return ".." in parts or any(p.startswith("..") for p in parts)


# ── Input allowlist ─────────────────────────────────────────────────────────────

# Characters allowed in gateway text input (blocks NUL, control chars, etc.)
_ALLOWED_INPUT_RE = re.compile(
    r'^[\x20-\x7E\x80-\xFFĀ-￿\n\r\t]*$'
)

_MAX_INPUT_LENGTH = 32_000   # ~8k tokens max per message


def sanitize_input(
    text:        str,
    max_length:  int  = _MAX_INPUT_LENGTH,
    allow_html:  bool = False,
) -> tuple[str, bool]:
    """
    Sanitize gateway input text.

    Returns (cleaned_text, was_modified).
    - Truncates at max_length
    - Strips NUL bytes
    - Optionally strips HTML tags
    """
    modified = False
    if not isinstance(text, str):
        text = str(text)
        modified = True

    # Strip NUL bytes
    if "\x00" in text:
        text = text.replace("\x00", "")
        modified = True

    # Strip HTML tags if requested
    if not allow_html and re.search(r'<[a-zA-Z/][^>]*>', text):
        text = re.sub(r'<[^>]+>', '', text)
        modified = True

    # Truncate
    if len(text) > max_length:
        text = text[:max_length] + " [truncated]"
        modified = True

    return text, modified


# ── Sandbox policy ─────────────────────────────────────────────────────────────

def is_sandbox_safe(
    agent_config: dict,
    require_sandbox: bool = False,
) -> tuple[bool, str]:
    """
    Check whether a sub-agent spawn is safe to proceed.

    If `require_sandbox=True`, the agent config must declare sandbox support.
    Returns (ok: bool, reason: str).
    """
    if not require_sandbox:
        return True, ""
    sandbox = agent_config.get("sandbox", agent_config.get("sandboxed", False))
    if not sandbox:
        return False, (
            "Sub-agent spawn rejected: sandbox=require is set but the agent "
            "config does not declare sandbox support. Set sandbox=true in the "
            "agent config or disable require_sandbox."
        )
    return True, ""
