"""
Operon Context Cache-TTL Pruner.

Adapted from OpenClaw src/agents/pi-hooks/context-pruning/.

Manages the conversation message list to prevent context window overflow
by pruning old tool results when they become stale.

Strategy:
  1. Soft-trim  — replace old tool-result content with a summary stub
                  after cache_ttl_seconds (default 5 minutes / 300s)
  2. Hard-clear — drop messages entirely once they exceed hard_ttl_seconds
                  (default 30 minutes)
  3. Preserve   — never prune the system prompt, user messages, or the
                  most recent N assistant turns
  4. Pin        — messages tagged {"pinned": True} are never pruned
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class PrunerConfig:
    cache_ttl_seconds:   float = 300.0    # 5 minutes — soft trim after this
    hard_ttl_seconds:    float = 1800.0   # 30 minutes — hard drop after this
    keep_last_n_turns:   int   = 6        # always keep the last N assistant turns
    max_messages:        int   = 200      # hard cap on total messages before pruning
    stub_template:       str   = "[tool result — pruned after {age:.0f}s to save context]"
    soft_trim_keys:      tuple = ("stdout", "output", "result", "content", "text")


_DEFAULT_CONFIG = PrunerConfig()


# ── Message timestamping ───────────────────────────────────────────────────────

def stamp_message(message: dict) -> dict:
    """
    Add a _ts (Unix timestamp) field to a message dict for TTL tracking.
    Returns the mutated message.
    """
    if "_ts" not in message:
        message["_ts"] = time.time()
    return message


def stamp_messages(messages: list[dict]) -> list[dict]:
    """Stamp all messages in a list that lack timestamps."""
    for m in messages:
        stamp_message(m)
    return messages


# ── Core pruner ────────────────────────────────────────────────────────────────

def prune_messages(
    messages:  list[dict],
    config:    Optional[PrunerConfig] = None,
    now:       Optional[float]        = None,
) -> tuple[list[dict], int, int]:
    """
    Prune stale tool results from the conversation history.

    Returns:
        (pruned_messages, soft_trimmed_count, hard_dropped_count)

    Parameters
    ----------
    messages  : full conversation message list (will NOT be mutated)
    config    : pruner configuration; uses defaults if None
    now       : current Unix timestamp; uses time.time() if None
    """
    cfg = config or _DEFAULT_CONFIG
    now = now or time.time()
    soft_trim  = 0
    hard_drop  = 0

    if not messages:
        return messages, 0, 0

    # Identify the most recent N assistant turns (their indices)
    assistant_indices: list[int] = [
        i for i, m in enumerate(messages) if m.get("role") == "assistant"
    ]
    protected_indices = set(assistant_indices[-cfg.keep_last_n_turns:])
    # Always protect the very last message
    protected_indices.add(len(messages) - 1)

    result: list[dict] = []

    for i, msg in enumerate(messages):
        # Never prune system messages or pinned messages
        role = msg.get("role", "")
        if role == "system" or msg.get("pinned"):
            result.append(msg)
            continue

        # Protected recent turns — keep as-is
        if i in protected_indices:
            result.append(msg)
            continue

        ts  = msg.get("_ts")
        age = (now - ts) if ts else 0.0

        # User messages: only hard-drop (never soft-trim user input)
        if role == "user":
            if ts and age > cfg.hard_ttl_seconds:
                hard_drop += 1
                continue   # drop it
            result.append(msg)
            continue

        # Tool result messages (typically role="tool" or tagged)
        is_tool_result = (
            role in ("tool", "tool_result")
            or msg.get("type") == "tool_result"
            or (role == "user" and msg.get("content", "").startswith("[TOOL_RESULT:"))
        )

        if is_tool_result:
            if ts and age > cfg.hard_ttl_seconds:
                hard_drop += 1
                continue   # hard drop
            elif ts and age > cfg.cache_ttl_seconds:
                # Soft trim — replace content with stub
                stub = cfg.stub_template.format(age=age)
                pruned_msg = dict(msg)
                content = msg.get("content", "")
                if isinstance(content, str):
                    pruned_msg["content"] = stub
                elif isinstance(content, list):
                    pruned_msg["content"] = [{"type": "text", "text": stub}]
                else:
                    pruned_msg["content"] = stub
                # Also trim nested tool result payloads
                for key in cfg.soft_trim_keys:
                    if key in pruned_msg and isinstance(pruned_msg[key], str):
                        pruned_msg[key] = stub
                result.append(pruned_msg)
                soft_trim += 1
                continue

        result.append(msg)

    # Hard message cap — if still over limit, drop oldest non-protected messages
    if len(result) > cfg.max_messages:
        overflow = len(result) - cfg.max_messages
        # Find oldest pruneable messages (non-system, non-pinned, non-protected)
        to_drop: list[int] = []
        for i, m in enumerate(result):
            if len(to_drop) >= overflow:
                break
            if m.get("role") == "system" or m.get("pinned"):
                continue
            to_drop.append(i)
        drop_set = set(to_drop)
        hard_drop += len(drop_set)
        result = [m for i, m in enumerate(result) if i not in drop_set]

    return result, soft_trim, hard_drop


# ── Convenience wrapper ────────────────────────────────────────────────────────

class ContextPruner:
    """
    Stateful wrapper around prune_messages for use inside an agent loop.

    Usage::

        pruner = ContextPruner()
        messages = pruner.maybe_prune(messages)
    """

    def __init__(self, config: Optional[PrunerConfig] = None) -> None:
        self.config  = config or _DEFAULT_CONFIG
        self.stats   = {"soft_trimmed": 0, "hard_dropped": 0, "prune_runs": 0}

    def maybe_prune(
        self,
        messages: list[dict],
        force:    bool = False,
    ) -> list[dict]:
        """
        Prune messages if necessary.

        Pruning is triggered when:
          - force=True, OR
          - len(messages) > max_messages, OR
          - the oldest timestamped message exceeds hard_ttl_seconds
        """
        if not messages:
            return messages

        # Stamp any unstamped messages
        stamp_messages(messages)

        should_prune = force or len(messages) > self.config.max_messages
        if not should_prune:
            # Check if oldest message is past hard TTL
            now = time.time()
            for m in messages:
                ts = m.get("_ts")
                if ts and (now - ts) > self.config.hard_ttl_seconds:
                    should_prune = True
                    break

        if not should_prune:
            return messages

        pruned, soft, hard = prune_messages(messages, self.config)
        self.stats["soft_trimmed"] += soft
        self.stats["hard_dropped"] += hard
        self.stats["prune_runs"]   += 1

        if soft + hard > 0:
            pass  # caller can check stats if interested

        return pruned
