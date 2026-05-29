"""
Operon Background Review Daemon.

Adapted from Hermes Agent agent/background_review.py.

After each completed exchange, spawns a lightweight background thread that
runs the agent in "review mode" to self-improve skills and knowledge based
on what just happened.

Key design points from Hermes:
- Fork uses quiet_mode=True (no output to terminal)
- Fork uses a restricted tool whitelist (read/write/knowledge only)
- Fork uses the parent's cached system prompt → immediate prefix cache hit
- Runs max 8 iterations (not full budget)
- ContextVar "background_review" distinguishes writes from foreground writes

Usage::

    reviewer = BackgroundReviewer()
    # Call after each successful exchange:
    reviewer.maybe_spawn(session_messages, agent_runner_factory=...)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

log = logging.getLogger("operon.background_review")

# Only these tools are available to the background review agent.
# Prevents the review fork from spending tokens on network, spawning agents, etc.
REVIEW_TOOL_WHITELIST: frozenset[str] = frozenset({
    "file_read", "file_write", "file_append",
    "knowledge_set", "knowledge_get", "knowledge_list",
    "dir_list", "file_exists",
})

_REVIEW_PROMPT = """You are running a background self-improvement review.

Review the recent exchange between the user and assistant. Your goal:
1. If the assistant made an error or used a suboptimal approach, update an
   existing SKILL.md to document the better approach.
2. If a new reusable pattern was discovered, add it to an appropriate skill.
3. If important facts were learned about the user or project, save them with
   knowledge_set.

Priority order for skill improvements:
  1. Update an already-loaded skill file in place (preferred)
  2. Add a support note to an existing related skill
  3. Only create a NEW skill file as a last resort

DO NOT document:
  - One-off commands specific only to this session
  - Environment errors or tool failures
  - Negative claims ("this tool doesn't work for X")
  - Anything that only applies to this single task

RECENT EXCHANGE:
{exchange_summary}

Now review and improve skills/knowledge as needed. When done, reply with a
brief summary of any changes made, or "No improvements needed." if nothing
was worth capturing.
"""

# Minimum seconds between review runs (avoid rapid-fire reviews)
_MIN_REVIEW_INTERVAL_SECONDS = 120

# Maximum consecutive review threads at once (don't pile up)
_MAX_CONCURRENT_REVIEWS = 1


class BackgroundReviewer:
    """
    Spawns a background review thread after qualifying exchanges.

    Parameters
    ----------
    min_exchange_turns : int
        Minimum number of tool calls in an exchange before reviewing.
        Avoids wasting tokens on trivial one-liner exchanges.
    """

    def __init__(self, min_exchange_turns: int = 2) -> None:
        self._min_turns     = min_exchange_turns
        self._last_review   = 0.0
        self._active_count  = 0
        self._lock          = threading.Lock()

    def maybe_spawn(
        self,
        messages:              list[dict],
        agent_runner_factory:  Optional[Callable] = None,
    ) -> bool:
        """
        Decide whether to spawn a background review, and if so, do it.

        Parameters
        ----------
        messages:
            Full session message list (not just the last exchange).
        agent_runner_factory:
            Callable that returns a function `(prompt: str) -> str`.
            If None, the review is skipped (no runner available).

        Returns True if a review was spawned.
        """
        if agent_runner_factory is None:
            return False

        with self._lock:
            now = time.time()
            # Rate-limit to avoid spinning up a review after every message
            if now - self._last_review < _MIN_REVIEW_INTERVAL_SECONDS:
                return False
            if self._active_count >= _MAX_CONCURRENT_REVIEWS:
                return False

            # Count recent tool calls — if too few, not worth reviewing
            recent = messages[-20:] if len(messages) > 20 else messages
            tool_calls = sum(
                1 for m in recent
                if m.get("role") == "user" and m.get("content", "").startswith("[TOOL_RESULT:")
            )
            if tool_calls < self._min_turns:
                return False

            self._active_count += 1
            self._last_review = now

        t = threading.Thread(
            target=self._review_worker,
            args=(messages, agent_runner_factory),
            daemon=True,
            name="operon-background-review",
        )
        t.start()
        log.debug("Background review spawned (tool_calls=%d)", tool_calls)
        return True

    def _review_worker(
        self,
        messages:             list[dict],
        agent_runner_factory: Callable,
    ) -> None:
        try:
            # Build exchange summary from last 10 messages
            exchange_lines = []
            for m in messages[-10:]:
                role    = m.get("role", "?")
                content = m.get("content", "")[:200]
                exchange_lines.append(f"[{role.upper()}] {content}")
            exchange_summary = "\n".join(exchange_lines)

            prompt = _REVIEW_PROMPT.format(exchange_summary=exchange_summary)

            # Get a runner with the review tool whitelist applied
            runner = agent_runner_factory(
                tool_whitelist=REVIEW_TOOL_WHITELIST,
                quiet_mode=True,
                max_iters=8,
                context_tag="background_review",
            )

            result = runner(prompt)
            if result and "no improvements needed" not in result.lower():
                log.info("Background review completed: %s", result[:100])
        except Exception as e:
            log.debug("Background review error (non-critical): %s", e)
        finally:
            with self._lock:
                self._active_count = max(0, self._active_count - 1)
