"""Optional anonymous web search and page extraction via Parallel Search MCP."""

import uuid
from urllib.parse import urlsplit

from core.mcp import _HttpTransport
from core.version import __version__


_ENDPOINT = "https://search.parallel.ai/mcp"
_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "User-Agent": f"Operon/{__version__} (+https://github.com/OperonAgent/Operon)",
}
# Reuse one anonymous identity throughout this Operon process, including fetches.
_SESSION_ID = uuid.uuid4().hex


def _anonymous_auth(request):
    """Prevent requests from picking up unrelated credentials from .netrc."""
    request.headers.pop("Authorization", None)
    return request


def _result(response: dict) -> dict:
    if "error" in response:
        raise ValueError(str(response["error"]))
    result = response.get("result")
    if not isinstance(result, dict):
        raise ValueError("Invalid MCP result")
    return result


def _call(tool_name: str, arguments: dict) -> dict:
    transport = _HttpTransport(_ENDPOINT, headers=_HEADERS, exact_url=True,
                               auth=_anonymous_auth)
    try:
        _result(transport.send_request("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "operon", "version": __version__},
        }))
        notification = transport.send_notification("notifications/initialized")
        if "error" in notification:
            raise ValueError(str(notification["error"]))
        tools = _result(transport.send_request("tools/list")).get("tools", [])
        if not any(tool.get("name") == tool_name for tool in tools):
            raise ValueError(f"Parallel MCP does not advertise {tool_name}")
        result = _result(transport.send_request("tools/call", {
            "name": tool_name,
            "arguments": {**arguments, "session_id": _SESSION_ID},
        }, timeout=30.0))
        text = "\n".join(c.get("text", "") for c in result.get("content", [])
                         if c.get("type") == "text")
        if result.get("isError"):
            return {"success": False, "output": None,
                    "error": text or "Parallel MCP tool failed"}
        output = result.get("structuredContent", text)
        if not output:
            raise ValueError("Parallel MCP returned no content")
        return {"success": True, "output": output, "error": ""}
    except Exception as e:
        return {"success": False, "output": None, "error": str(e)}
    finally:
        transport.close()


def parallel_search(objective: str, search_queries: list) -> dict:
    """Return focused web excerpts, titles and source URLs without an API key."""
    if (not isinstance(objective, str) or not objective.strip()
            or not isinstance(search_queries, list) or not search_queries
            or any(not isinstance(q, str) or not q.strip() for q in search_queries)):
        return {"success": False, "output": None,
                "error": "Provide an objective and a nonempty list of search_queries."}
    return _call("web_search", {"objective": objective, "search_queries": search_queries})


def parallel_fetch(urls: list, objective: str = "", full_content: bool = False) -> dict:
    """Extract excerpts from up to 20 HTTP(S) source pages."""
    try:
        if not isinstance(urls, list) or not 1 <= len(urls) <= 20:
            raise ValueError("Provide between 1 and 20 HTTP(S) URLs.")
        for url in urls:
            parsed = urlsplit(url) if isinstance(url, str) else None
            if (parsed is None or parsed.scheme not in ("http", "https")
                    or not parsed.hostname or parsed.username or parsed.password):
                raise ValueError("Each URL must be HTTP(S) without embedded credentials.")
        if not isinstance(objective, str) or len(objective) > 200:
            raise ValueError("Fetch objective must be a string of at most 200 characters.")
        if not isinstance(full_content, bool):
            raise ValueError("full_content must be a boolean.")
    except ValueError as e:
        return {"success": False, "output": None, "error": str(e)}
    return _call("web_fetch", {"urls": urls, "objective": objective or None,
                               "full_content": full_content})
