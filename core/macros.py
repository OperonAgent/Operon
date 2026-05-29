"""
Operon Pipeline Macros — Lobster-style composable tool pipelines.

Inspired by OpenClaw's Lobster macro engine. A macro is a named sequence
of tool calls where each step can reference the previous step's output
via template variables. Macros are saved to ~/.operon/macros.json and
are callable by the agent via the run_macro tool or by the user via
/macro run <name>.

Macro format (JSON)
-------------------
{
  "name": "daily_digest",
  "description": "Search news, summarise, and send to Telegram",
  "steps": [
    {
      "tool": "duckduckgo_search",
      "params": {"query": "{topic} latest news", "max_results": 5}
    },
    {
      "tool": "python_exec",
      "params": {"code": "print('''Summarise: {prev_output}''')"}
    },
    {
      "tool": "telegram_send",
      "params": {"chat_id": "{chat_id}", "text": "Daily digest:\n{prev_output}"}
    }
  ],
  "vars": {"topic": "AI", "chat_id": "12345"}
}

Template variables
------------------
  {prev_output}   — full JSON string of previous step's result
  {prev_text}     — result["output"] or result["stdout"] or str(result)
  {var_name}      — any key from the macro's "vars" dict
  {step_N}        — result of step N (0-indexed), as JSON
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_MACROS_PATH = Path.home() / ".operon" / "macros.json"


# ── Storage ───────────────────────────────────────────────────────────────────

def _load() -> List[Dict[str, Any]]:
    if _MACROS_PATH.exists():
        try:
            return json.loads(_MACROS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save(macros: List[Dict[str, Any]]) -> None:
    _MACROS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MACROS_PATH.write_text(
        json.dumps(macros, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Template substitution ─────────────────────────────────────────────────────

def _resolve(template: Any, context: dict) -> Any:
    """
    Recursively substitute {variable} placeholders in strings, dicts, and lists.
    """
    if isinstance(template, str):
        def _sub(m):
            key = m.group(1)
            return str(context.get(key, m.group(0)))
        return re.sub(r"\{(\w+)\}", _sub, template)
    if isinstance(template, dict):
        return {k: _resolve(v, context) for k, v in template.items()}
    if isinstance(template, list):
        return [_resolve(item, context) for item in template]
    return template


def _result_text(result: dict) -> str:
    """Extract a human-readable string from a tool result."""
    for key in ("output", "stdout", "response", "text", "content",
                "transcript", "stdout", "rows"):
        if key in result and result[key] is not None:
            v = result[key]
            if isinstance(v, (list, dict)):
                return json.dumps(v, ensure_ascii=False)[:2000]
            return str(v)[:2000]
    return json.dumps(result, ensure_ascii=False)[:2000]


# ── Macro CRUD ────────────────────────────────────────────────────────────────

def macro_save(
    name:        str,
    steps:       List[Dict[str, Any]],
    description: str = "",
    vars:        Dict[str, str] = None,
    **_,
) -> dict:
    """
    Save (create or update) a pipeline macro.

    Args:
        name        — unique macro name (required)
        steps       — list of {tool, params} dicts (required)
        description — human-readable description (optional)
        vars        — default variable values (optional)

    Returns:
        {success, name, step_count, error}
    """
    if not name:
        return {"success": False, "error": "name is required."}
    if not steps or not isinstance(steps, list):
        return {"success": False, "error": "steps must be a non-empty list."}

    macros = _load()
    # Remove existing with same name
    macros = [m for m in macros if m.get("name") != name]
    macros.append({
        "name":        name,
        "description": description,
        "steps":       steps,
        "vars":        vars or {},
        "created_at":  time.time(),
    })
    _save(macros)
    return {"success": True, "name": name, "step_count": len(steps), "error": ""}


def macro_delete(name: str = "", **_) -> dict:
    """Delete a saved macro by name."""
    if not name:
        return {"success": False, "error": "name is required."}
    macros = _load()
    before = len(macros)
    macros = [m for m in macros if m.get("name") != name]
    if len(macros) == before:
        return {"success": False, "error": f"Macro '{name}' not found."}
    _save(macros)
    return {"success": True, "name": name, "error": ""}


def macro_list(**_) -> dict:
    """List all saved macros."""
    macros = _load()
    return {
        "success": True,
        "macros": [
            {
                "name":        m["name"],
                "description": m.get("description", ""),
                "step_count":  len(m.get("steps", [])),
                "vars":        list(m.get("vars", {}).keys()),
            }
            for m in macros
        ],
        "count":   len(macros),
        "error":   "",
    }


# ── Macro execution ───────────────────────────────────────────────────────────

def run_macro(
    name:         str  = "",
    vars:         dict = None,
    tool_registry = None,
    **_,
) -> dict:
    """
    Execute a saved pipeline macro.

    Args:
        name         — macro name (required)
        vars         — variable overrides (optional — merged with macro defaults)
        tool_registry — ToolRegistry instance injected at call time

    Returns:
        {success, name, steps_run, results: [...], final_output, error}
    """
    if not name:
        return {"success": False, "error": "name is required."}

    macros   = _load()
    macro    = next((m for m in macros if m["name"] == name), None)
    if macro is None:
        available = [m["name"] for m in macros]
        return {
            "success": False,
            "error":   f"Macro '{name}' not found. Available: {available}",
        }

    if tool_registry is None:
        return {"success": False, "error": "tool_registry not available — macro cannot run."}

    # Build variable context: macro defaults → caller overrides
    context: Dict[str, str] = dict(macro.get("vars", {}))
    if vars and isinstance(vars, dict):
        context.update(vars)

    steps        = macro.get("steps", [])
    step_results = []
    prev_result  = {}
    prev_text    = ""

    for i, step in enumerate(steps):
        tool_name = step.get("tool", "")
        if not tool_name:
            step_results.append({"step": i, "error": "missing tool name"})
            continue

        # Build context for this step
        step_ctx = dict(context)
        step_ctx["prev_output"] = json.dumps(prev_result, ensure_ascii=False)
        step_ctx["prev_text"]   = prev_text
        # Expose each previous step's result as step_N
        for j, r in enumerate(step_results):
            step_ctx[f"step_{j}"] = json.dumps(r, ensure_ascii=False)

        # Resolve template variables in params
        raw_params    = step.get("params", {})
        resolved_params = _resolve(raw_params, step_ctx)
        if not isinstance(resolved_params, dict):
            resolved_params = {}

        # Execute
        result = tool_registry.execute(tool_name, resolved_params)
        step_results.append({
            "step":      i,
            "tool":      tool_name,
            "params":    resolved_params,
            "result":    result,
        })

        prev_result = result
        prev_text   = _result_text(result)

        if not result.get("success", True) is False:
            pass  # continue pipeline even on soft failures

    final_output = prev_text
    return {
        "success":      True,
        "name":         name,
        "steps_run":    len(step_results),
        "results":      step_results,
        "final_output": final_output,
        "error":        "",
    }


# ── MacroManager ─────────────────────────────────────────────────────────────

class MacroManager:
    """Management wrapper used by /macro slash commands."""

    def __init__(self, tool_registry=None):
        self._registry = tool_registry

    def list_macros(self) -> List[Dict[str, Any]]:
        return macro_list()["macros"]

    def save(self, name: str, steps: list, description: str = "",
             vars: dict = None) -> dict:
        return macro_save(name=name, steps=steps,
                          description=description, vars=vars)

    def delete(self, name: str) -> dict:
        return macro_delete(name=name)

    def run(self, name: str, vars: dict = None) -> dict:
        return run_macro(name=name, vars=vars, tool_registry=self._registry)

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return next((m for m in _load() if m["name"] == name), None)
