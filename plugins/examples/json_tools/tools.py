"""json_tools plugin — validate / pretty-print / minify JSON. Dependency-free."""

import json as _json


def json_validate(text: str = "") -> dict:
    """Check whether a string is valid JSON; report the error if not."""
    try:
        _json.loads(text)
        return {"success": True, "valid": True}
    except Exception as e:
        return {"success": True, "valid": False, "error": str(e)}


def json_pretty(text: str = "", indent: int = 2, sort_keys: bool = False) -> dict:
    """Pretty-print a JSON string with the given indentation."""
    try:
        obj = _json.loads(text)
        out = _json.dumps(obj, indent=int(indent), sort_keys=bool(sort_keys),
                          ensure_ascii=False)
        return {"success": True, "result": out}
    except Exception as e:
        return {"success": False, "error": str(e)}


def json_minify(text: str = "") -> dict:
    """Minify a JSON string (remove all insignificant whitespace)."""
    try:
        obj = _json.loads(text)
        out = _json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        return {"success": True, "result": out}
    except Exception as e:
        return {"success": False, "error": str(e)}
