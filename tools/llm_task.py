"""
Operon LLM Task Tool.

Adapted from OpenClaw extensions/llm-task/src/llm-task-tool.ts.

Lightweight in-band sub-model call — lets the agent invoke a cheap/fast
model for a single focused task (classification, summarisation, extraction)
without spinning up a full sub-agent.

Usage via registry::

    result = llm_task(
        prompt   = "Summarise this text in 3 bullets: ...",
        model    = "gpt-4o-mini",     # optional — defaults to config
        provider = "openai",          # optional
        max_tokens = 512,             # optional
        temperature = 0.1,            # optional
    )
    # Returns {"success": bool, "result": str, "tokens": int}
"""

from __future__ import annotations

import time
from typing import Optional


def llm_task(
    prompt:      str,
    model:       Optional[str]  = None,
    provider:    Optional[str]  = None,
    max_tokens:  int            = 1024,
    temperature: float          = 0.1,
    system:      Optional[str]  = None,
    timeout:     int            = 60,
) -> dict:
    """
    Run a single LLM call for a focused sub-task.

    Returns::

        {
            "success":   bool,
            "result":    str,    # model response text
            "tokens":    int,    # total tokens used
            "model":     str,    # model that ran
            "provider":  str,    # provider used
            "latency_ms": float, # wall-clock ms
        }
    """
    if not prompt or not prompt.strip():
        return {"success": False, "result": "Empty prompt", "tokens": 0,
                "model": "", "provider": "", "latency_ms": 0.0}

    # Lazy import to avoid circular deps at module load
    try:
        from core.config import ConfigManager
        from core.router import ModelRouter
    except ImportError as e:
        return {"success": False, "result": f"Import error: {e}", "tokens": 0,
                "model": "", "provider": "", "latency_ms": 0.0}

    cfg     = ConfigManager()
    router  = ModelRouter(cfg)

    # Override model/provider if requested
    if model or provider:
        # Patch config temporarily
        original_model = cfg.get("default_model")
        if model:
            cfg._data["default_model"] = model

    sys_prompt = system or (
        "You are a focused sub-task assistant. "
        "Answer concisely and exactly as requested. "
        "Do not add preamble or sign-offs."
    )

    messages = [{"role": "user", "content": prompt}]

    t0 = time.monotonic()
    try:
        response = router.complete(sys_prompt, messages)
    finally:
        # Restore original model if we patched it
        if model or provider:
            if model:
                cfg._data["default_model"] = original_model

    latency_ms = (time.monotonic() - t0) * 1000

    if response is None:
        return {
            "success":    False,
            "result":     "LLM call failed — see router logs",
            "tokens":     0,
            "model":      model or cfg.get("default_model", ""),
            "provider":   provider or "",
            "latency_ms": latency_ms,
        }

    usage = router.last_usage
    total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

    return {
        "success":    True,
        "result":     response,
        "tokens":     total_tokens,
        "model":      usage.get("model", model or ""),
        "provider":   usage.get("provider", provider or ""),
        "latency_ms": latency_ms,
    }


# ── Convenience wrappers ───────────────────────────────────────────────────────

def llm_classify(text: str, categories: list[str], **kwargs) -> dict:
    """
    Classify `text` into one of `categories` using a fast LLM call.
    Returns {"success": bool, "category": str, ...}
    """
    cats_str = ", ".join(f'"{c}"' for c in categories)
    prompt = (
        f"Classify the following text into exactly one of these categories: {cats_str}\n\n"
        f"Text:\n{text}\n\n"
        f"Reply with ONLY the category name, nothing else."
    )
    result = llm_task(prompt, max_tokens=32, temperature=0.0, **kwargs)
    if result["success"]:
        raw = result["result"].strip().strip('"').strip("'")
        # Find best matching category
        for cat in categories:
            if cat.lower() in raw.lower():
                result["category"] = cat
                return result
        result["category"] = raw  # return whatever the model said
    return result


def llm_summarize(text: str, max_bullets: int = 5, **kwargs) -> dict:
    """
    Summarise `text` into up to `max_bullets` bullet points.
    Returns {"success": bool, "result": str, ...}
    """
    prompt = (
        f"Summarise the following in at most {max_bullets} concise bullet points:\n\n{text}"
    )
    return llm_task(prompt, max_tokens=512, **kwargs)


def llm_extract(text: str, fields: list[str], **kwargs) -> dict:
    """
    Extract named fields from `text` as a JSON object.
    Returns {"success": bool, "result": str (JSON), "parsed": dict | None, ...}
    """
    fields_str = ", ".join(f'"{f}"' for f in fields)
    prompt = (
        f"Extract these fields from the text: {fields_str}\n\n"
        f"Return a JSON object with these exact keys. "
        f"Use null for missing fields.\n\n"
        f"Text:\n{text}"
    )
    result = llm_task(prompt, max_tokens=512, **kwargs)
    if result["success"]:
        import json, re
        try:
            m = re.search(r"\{[\s\S]+\}", result["result"])
            if m:
                result["parsed"] = json.loads(m.group(0))
            else:
                result["parsed"] = None
        except Exception:
            result["parsed"] = None
    return result
