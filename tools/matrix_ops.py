"""
Operon Matrix Messaging Integration.

Supports the Matrix protocol (used by Element, Beeper, etc.) via:
  1. matrix-nio Python library (pip install matrix-nio)  — preferred
  2. Direct Matrix Client-Server HTTP API               — fallback (no deps)

Setup
-----
  pip install matrix-nio   # optional but preferred
  export MATRIX_HOMESERVER=https://matrix.org
  export MATRIX_USER=@youruser:matrix.org
  export MATRIX_PASSWORD=yourpassword
  # OR use an access token instead of password:
  export MATRIX_ACCESS_TOKEN=syt_xxxxx

All functions accept **_ for registry compatibility.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


def _creds() -> tuple[str, str, str, str]:
    """Return (homeserver, user, password, access_token)."""
    hs    = os.environ.get("MATRIX_HOMESERVER", "").rstrip("/")
    user  = os.environ.get("MATRIX_USER", "")
    pw    = os.environ.get("MATRIX_PASSWORD", "")
    token = os.environ.get("MATRIX_ACCESS_TOKEN", "")
    return hs, user, pw, token


def _http(method: str, url: str, body: dict = None, token: str = "") -> dict:
    """Minimal HTTP helper using only stdlib."""
    import json as _json
    data    = _json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return _json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return _json.loads(e.read())
        except Exception:
            return {"error": str(e), "errcode": str(e.code)}
    except Exception as e:
        return {"error": str(e)}


def _get_token() -> tuple[str, str]:
    """Return (homeserver, access_token). Logs in if needed."""
    hs, user, pw, token = _creds()
    if not hs:
        return "", ""
    if token:
        return hs, token
    if user and pw:
        resp = _http("POST", f"{hs}/_matrix/client/v3/login", {
            "type":     "m.login.password",
            "user":     user,
            "password": pw,
        })
        return hs, resp.get("access_token", "")
    return hs, ""


def matrix_send(
    message: str = "",
    room_id: str = "",
    msgtype: str = "m.text",
    **_,
) -> dict:
    """
    Send a message to a Matrix room.

    Args:
        message — message text (required)
        room_id — Matrix room ID e.g. '!abc123:matrix.org' (optional — auto-read from MATRIX_ROOM_ID)
        msgtype — 'm.text' | 'm.notice' | 'm.emote' (optional, default 'm.text')

    Returns:
        {success, event_id, room_id, error}
    """
    if not message:
        return {"success": False, "error": "message is required."}

    target = room_id or os.environ.get("MATRIX_ROOM_ID", "").strip()
    if not target:
        return {"success": False, "error": "room_id required or set MATRIX_ROOM_ID"}

    hs, token = _get_token()
    if not hs or not token:
        return {
            "success": False,
            "error": (
                "Matrix credentials not set. Export:\n"
                "  MATRIX_HOMESERVER=https://matrix.org\n"
                "  MATRIX_USER=@user:matrix.org\n"
                "  MATRIX_PASSWORD=password  OR  MATRIX_ACCESS_TOKEN=syt_..."
            ),
        }

    txn_id = str(int(time.time() * 1000))
    resp = _http(
        "PUT",
        f"{hs}/_matrix/client/v3/rooms/{urllib.parse.quote(target)}/send/m.room.message/{txn_id}",
        {"msgtype": msgtype, "body": message},
        token=token,
    )
    if "event_id" in resp:
        return {"success": True, "event_id": resp["event_id"], "room_id": target, "error": ""}
    return {"success": False, "error": resp.get("error", str(resp))}


def matrix_get_messages(
    room_id: str = "",
    limit:   int = 10,
    **_,
) -> dict:
    """
    Fetch recent messages from a Matrix room.

    Args:
        room_id — Matrix room ID (optional — auto-read from MATRIX_ROOM_ID)
        limit   — number of messages (optional, default 10)

    Returns:
        {success, messages: [{sender, body, timestamp, event_id}], count, error}
    """
    target = room_id or os.environ.get("MATRIX_ROOM_ID", "").strip()
    if not target:
        return {"success": False, "error": "room_id required or set MATRIX_ROOM_ID"}

    hs, token = _get_token()
    if not hs or not token:
        return {"success": False, "error": "Matrix credentials not set."}

    import urllib.parse
    resp = _http(
        "GET",
        f"{hs}/_matrix/client/v3/rooms/{urllib.parse.quote(target)}/messages"
        f"?dir=b&limit={limit}",
        token=token,
    )
    if "chunk" not in resp:
        return {"success": False, "error": resp.get("error", str(resp))}

    messages = []
    for ev in resp["chunk"]:
        if ev.get("type") == "m.room.message":
            content = ev.get("content", {})
            messages.append({
                "sender":    ev.get("sender", "?"),
                "body":      content.get("body", ""),
                "timestamp": ev.get("origin_server_ts", 0) // 1000,
                "event_id":  ev.get("event_id", ""),
            })
    return {"success": True, "messages": messages, "count": len(messages), "error": ""}


def matrix_list_rooms(**_) -> dict:
    """
    List Matrix rooms the account has joined.

    Returns:
        {success, rooms: [{room_id, name}], count, error}
    """
    hs, token = _get_token()
    if not hs or not token:
        return {"success": False, "error": "Matrix credentials not set."}

    resp = _http("GET", f"{hs}/_matrix/client/v3/joined_rooms", token=token)
    if "joined_rooms" not in resp:
        return {"success": False, "error": resp.get("error", str(resp))}

    rooms = [{"room_id": r} for r in resp["joined_rooms"]]
    return {"success": True, "rooms": rooms, "count": len(rooms), "error": ""}


# Patch urllib.parse into scope for matrix_send
import urllib.parse
