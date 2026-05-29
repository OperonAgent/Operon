"""codec plugin — Base64 and URL encoding/decoding. Dependency-free."""

import base64 as _b64
import urllib.parse as _up


def base64_encode(text: str = "") -> dict:
    """Base64-encode a UTF-8 string."""
    try:
        out = _b64.b64encode(str(text).encode("utf-8")).decode("ascii")
        return {"success": True, "result": out}
    except Exception as e:
        return {"success": False, "error": str(e)}


def base64_decode(data: str = "") -> dict:
    """Base64-decode to a UTF-8 string."""
    try:
        out = _b64.b64decode(str(data).encode("ascii")).decode("utf-8", "replace")
        return {"success": True, "result": out}
    except Exception as e:
        return {"success": False, "error": str(e)}


def url_encode(text: str = "") -> dict:
    """Percent-encode a string for safe use in URLs."""
    return {"success": True, "result": _up.quote(str(text), safe="")}


def url_decode(text: str = "") -> dict:
    """Decode a percent-encoded URL string."""
    return {"success": True, "result": _up.unquote(str(text))}
