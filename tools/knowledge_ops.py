"""
Operon knowledge_set / knowledge_get / knowledge_delete tools.

These are wired to the KnowledgeBase singleton via set_knowledge_base().
"""

from typing import Optional
from core.knowledge import KnowledgeBase

_kb: Optional[KnowledgeBase] = None


def set_knowledge_base(kb: KnowledgeBase) -> None:
    global _kb
    _kb = kb


def knowledge_set(key: str = "", value: str = "", **_) -> dict:
    """Save a permanent fact."""
    if _kb is None:
        return {"success": False, "output": None, "error": "KnowledgeBase not initialised."}
    if not key:
        return {"success": False, "output": None, "error": "key is required."}
    if not value:
        return {"success": False, "output": None, "error": "value is required."}
    try:
        _kb.set(key, value)
        return {
            "success": True,
            "output":  f"Remembered permanently: {key} = {value}",
            "error":   "",
        }
    except Exception as e:
        return {"success": False, "output": None, "error": str(e)}


def knowledge_get(key: str = "", **_) -> dict:
    """Retrieve a specific permanent fact by key."""
    if _kb is None:
        return {"success": False, "output": None, "error": "KnowledgeBase not initialised."}
    if not key:
        return {"success": False, "output": None, "error": "key is required."}
    val = _kb.get(key)
    if val is None:
        return {"success": False, "output": None, "error": f"No fact stored for key '{key}'."}
    return {"success": True, "output": val, "error": ""}


def knowledge_delete(key: str = "", **_) -> dict:
    """Delete a permanent fact by key."""
    if _kb is None:
        return {"success": False, "output": None, "error": "KnowledgeBase not initialised."}
    if not key:
        return {"success": False, "output": None, "error": "key is required."}
    if _kb.delete(key):
        return {"success": True, "output": f"Deleted fact: {key}", "error": ""}
    return {"success": False, "output": None, "error": f"Key '{key}' not found."}


def knowledge_list(**_) -> dict:
    """List all stored permanent facts."""
    if _kb is None:
        return {"success": False, "output": None, "error": "KnowledgeBase not initialised."}
    data = _kb.get_all()
    if not data:
        return {"success": True, "output": "(no facts stored yet)", "error": ""}
    lines = [f"{k}: {v}" for k, v in data.items()]
    return {"success": True, "output": "\n".join(lines), "error": ""}
