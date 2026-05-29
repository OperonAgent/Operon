"""
Operon Sub-Agent Delegation Tool.

Adapted from Hermes Agent delegate_tool.py architecture.

Spawns child Operon agent instances with:
  - Isolated conversation context (no parent history leaked)
  - Restricted toolsets (configurable, dangerous tools always blocked)
  - Parallel execution via ThreadPoolExecutor
  - ACP event emission for observability
  - Automatic result summarization for parent context efficiency

Supports:
  delegate_task  — spawn a single focused sub-agent
  delegate_batch — spawn N sub-agents in parallel (max 5 concurrent)

Blocked tools in sub-agents (never delegatable):
  delegate_task, delegate_batch — no recursion
  email_draft, email_send      — no cross-platform side effects
  computer_use                 — no GUI in headless sub-agents
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("operon.delegate")

# Tools sub-agents must NEVER have access to
DELEGATE_BLOCKED_TOOLS = frozenset({
    "delegate_task",     # no recursive delegation
    "delegate_batch",    # no recursive delegation
    "email_draft",       # no cross-platform side effects
    "email_send",        # same
    "computer_use",      # no GUI in headless sub-agents
})

_MAX_CONCURRENT = 5
_DEFAULT_TIMEOUT = 300   # 5 minutes per sub-agent
_MAX_DEPTH = 1           # parent (0) → child (1); grandchild rejected

# Thread-local depth counter to detect recursion
_tls = threading.local()


def _current_depth() -> int:
    return getattr(_tls, "delegate_depth", 0)


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n[…{len(text)-max_chars} chars omitted…]\n" + text[-half:]


def _format_result(result: Any, max_chars: int = 4000) -> str:
    if isinstance(result, dict):
        if "reply" in result:
            return _truncate(str(result["reply"]), max_chars)
        if "content" in result:
            return _truncate(str(result["content"]), max_chars)
        return _truncate(json.dumps(result, ensure_ascii=False), max_chars)
    return _truncate(str(result), max_chars)


# ---------------------------------------------------------------------------
# Sub-agent executor
# ---------------------------------------------------------------------------

def _run_subagent(
    task: str,
    toolset: str = "core",
    model: Optional[str] = None,
    context: str = "",
    timeout: int = _DEFAULT_TIMEOUT,
    task_id: str = "",
) -> Dict:
    """
    Run a single sub-agent synchronously.
    Returns {"success": bool, "result": str, "duration_s": float, "task_id": str}.
    """
    task_id = task_id or str(uuid.uuid4())[:12]
    _tls.delegate_depth = _current_depth() + 1
    t0 = time.time()

    try:
        # Emit ACP started event
        try:
            from core.acp_adapter import make_agent
            acp = make_agent(agent_id=f"delegate_{task_id}")
            acp.started(task=task[:200])
        except Exception:
            acp = None

        # Build sub-agent system prompt
        system_parts = [
            "You are a focused sub-agent spawned by Operon to complete a specific task.",
            "Complete the task and return a clear, factual result.",
            "Do NOT ask clarifying questions — make reasonable assumptions.",
            "Be concise: your output will be embedded in the parent's context.",
        ]
        if context:
            system_parts.append(f"\nContext provided by parent agent:\n{context}")
        system = "\n".join(system_parts)

        # Import and run the router
        from core.router import ModelRouter
        from core.config import ConfigManager

        cfg    = ConfigManager()
        chosen = model or cfg.get("default_model", "llama3.2")
        router = ModelRouter(cfg)

        # Get allowed tools
        allowed_tools = _get_allowed_tools(toolset)

        # Build tool descriptions for sub-agent
        from tools.registry import ToolRegistry
        registry = ToolRegistry()
        # Restrict registry to allowed tools only
        all_defs = registry.get_definitions()
        sub_defs = [d for d in all_defs
                    if d.get("name") in allowed_tools
                    and d.get("name") not in DELEGATE_BLOCKED_TOOLS]

        tools_block = "\n".join(
            f"- {d['name']}: {d.get('description','')}"
            for d in sub_defs
        )
        full_system = system + f"\n\nAvailable tools:\n{tools_block}\n\n" + \
                      _get_format_instructions()

        messages = [{"role": "user", "content": task}]

        # Run the agent loop (up to 10 iterations)
        final_result = ""
        for iteration in range(10):
            try:
                raw = router.complete(system=full_system, messages=messages)
            except Exception as e:
                final_result = f"[sub-agent error on iteration {iteration}: {e}]"
                break

            parsed = router.parse_response(raw) if raw else None
            if not parsed or not isinstance(parsed, dict):
                # Try raw text
                if raw:
                    final_result = raw.strip()
                break

            action = parsed.get("action", {})
            a_type = action.get("type", "")

            if a_type == "response":
                final_result = str(action.get("content", "")).strip()
                break

            elif a_type == "tool_call":
                tool_name = action.get("tool", "")
                params    = action.get("params", {})

                if tool_name in DELEGATE_BLOCKED_TOOLS:
                    tool_result = {"error": f"tool {tool_name!r} is blocked in sub-agents"}
                elif tool_name not in allowed_tools:
                    tool_result = {"error": f"tool {tool_name!r} not in sub-agent toolset"}
                else:
                    try:
                        from tools.registry import _DISPATCH
                        fn = _DISPATCH.get(tool_name)
                        if fn:
                            tool_result = fn(**params)
                        else:
                            tool_result = {"error": f"tool {tool_name!r} not dispatched"}
                    except Exception as e:
                        tool_result = {"error": str(e)}

                result_str = (json.dumps(tool_result, ensure_ascii=False)
                              if isinstance(tool_result, dict) else str(tool_result))
                messages.append({"role": "assistant", "content": json.dumps(parsed)})
                messages.append({"role": "user",
                                 "content": f"[TOOL_RESULT: {tool_name}]\n{result_str}"})
            else:
                # Unknown action — try to extract any content
                content = (action.get("content") or parsed.get("reply")
                           or parsed.get("response") or "")
                if content:
                    final_result = str(content).strip()
                break

        duration = round(time.time() - t0, 2)
        if acp:
            acp.finished("success", summary=final_result[:200])
        return {
            "success":    True,
            "result":     _format_result(final_result),
            "task_id":    task_id,
            "duration_s": duration,
            "iterations": iteration + 1,
        }

    except Exception as e:
        duration = round(time.time() - t0, 2)
        log.exception("Sub-agent %s failed: %s", task_id, e)
        return {
            "success":    False,
            "error":      str(e),
            "task_id":    task_id,
            "duration_s": duration,
        }
    finally:
        _tls.delegate_depth = max(0, _current_depth() - 1)


def _get_format_instructions() -> str:
    return """
STRICT RESPONSE FORMAT — ALWAYS VALID JSON
Every response MUST be a single JSON object:

For a final answer:
{"action": {"type": "response", "content": "your answer here"}}

For a tool call:
{"action": {"type": "tool_call", "tool": "tool_name", "params": {"key": "value"}}}
"""


def _get_allowed_tools(toolset: str) -> set:
    """Return the set of tools allowed for a given toolset name."""
    try:
        from core.toolsets import get_toolset
        ts = get_toolset(toolset)
        return set(ts) - DELEGATE_BLOCKED_TOOLS
    except Exception:
        pass
    # Fallback: safe core tools
    return {
        "shell_exec", "file_ops", "file_search", "web_search",
        "http_client", "git_ops", "code_exec", "db_ops", "llm_task",
        "apply_patch",
    } - DELEGATE_BLOCKED_TOOLS


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def delegate_task(
    task: str,
    toolset: str = "core",
    model: str = "",
    context: str = "",
    timeout: int = _DEFAULT_TIMEOUT,
    **_,
) -> Dict:
    """
    Spawn a focused sub-agent to complete a specific task.

    The sub-agent runs in isolation with its own context, executes the task
    using the allowed toolset, and returns a summary result to the parent.
    The parent's context only sees the delegation call and the summary.

    Args:
        task    : What the sub-agent must accomplish (be specific).
        toolset : Which toolset to give the sub-agent (core/coding/research/data/devops).
        model   : Override model (default: same as parent's configured model).
        context : Additional context/instructions for the sub-agent.
        timeout : Max seconds to wait (default 300).

    Returns:
        dict with success, result (str), duration_s, task_id.
    """
    if _current_depth() >= _MAX_DEPTH:
        return {
            "success": False,
            "error": f"delegation depth limit ({_MAX_DEPTH}) reached — no recursive delegation",
        }
    if not task or not task.strip():
        return {"success": False, "error": "task is required"}

    task_id = str(uuid.uuid4())[:12]
    log.info("Delegating task %s (toolset=%s, timeout=%ds)", task_id, toolset, timeout)

    # Run with timeout
    result: Dict = {}
    exc_holder: List = []

    def _run():
        try:
            result.update(_run_subagent(
                task=task, toolset=toolset, model=model or "",
                context=context, timeout=timeout, task_id=task_id,
            ))
        except Exception as e:
            exc_holder.append(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout + 5)

    if t.is_alive():
        return {
            "success": False,
            "error":   f"sub-agent timed out after {timeout}s",
            "task_id": task_id,
        }
    if exc_holder:
        return {"success": False, "error": str(exc_holder[0]), "task_id": task_id}
    return result


def delegate_batch(
    tasks: List[Dict],
    toolset: str = "core",
    model: str = "",
    max_concurrent: int = 3,
    timeout: int = _DEFAULT_TIMEOUT,
    **_,
) -> Dict:
    """
    Spawn multiple sub-agents in parallel, each handling a different task.

    Args:
        tasks : List of task dicts. Each must have "task" (str) and optionally
                "context" (str), "toolset" (str), "id" (str).
        toolset : Default toolset (can be overridden per task).
        model   : Override model for all tasks.
        max_concurrent : Max simultaneous sub-agents (1–5).
        timeout : Per-task timeout in seconds.

    Returns:
        dict with tasks: [{task_id, success, result, duration_s}, ...], totals.
    """
    if _current_depth() >= _MAX_DEPTH:
        return {
            "success": False,
            "error": "delegation depth limit reached",
        }
    if not tasks:
        return {"success": False, "error": "tasks list is required"}

    max_concurrent = max(1, min(max_concurrent, _MAX_CONCURRENT))
    results = [None] * len(tasks)

    def _run_one(idx: int, task_dict: Dict):
        _tls.delegate_depth = _current_depth() + 1
        try:
            return _run_subagent(
                task    = task_dict.get("task", ""),
                toolset = task_dict.get("toolset", toolset),
                model   = model or "",
                context = task_dict.get("context", ""),
                timeout = timeout,
                task_id = task_dict.get("id", str(uuid.uuid4())[:12]),
            )
        finally:
            _tls.delegate_depth = max(0, _current_depth() - 1)

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(_run_one, i, t): i for i, t in enumerate(tasks)}
        for fut, idx in futures.items():
            try:
                results[idx] = fut.result(timeout=timeout + 10)
            except FutTimeoutError:
                results[idx] = {"success": False,
                                "error": "timed out", "task_id": tasks[idx].get("id", "")}
            except Exception as e:
                results[idx] = {"success": False, "error": str(e),
                                "task_id": tasks[idx].get("id", "")}

    succeeded = sum(1 for r in results if r and r.get("success"))
    return {
        "success": succeeded == len(tasks),
        "tasks":   results,
        "summary": f"{succeeded}/{len(tasks)} tasks succeeded",
    }
