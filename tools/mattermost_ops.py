"""
Operon Mattermost Integration.

Uses the Mattermost REST API (v4) — no extra Python packages needed,
only the stdlib urllib.

Setup
-----
  export MATTERMOST_URL=https://your-instance.mattermost.com
  export MATTERMOST_TOKEN=your_personal_access_token
  export MATTERMOST_DEFAULT_CHANNEL=town-square   # optional

All functions accept **_ for registry compatibility.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
import urllib.parse
from typing import Any, Dict, List, Optional


def _creds() -> tuple[str, str]:
    """Return (base_url, token)."""
    url   = os.environ.get("MATTERMOST_URL", "").rstrip("/")
    token = os.environ.get("MATTERMOST_TOKEN", "").strip()
    return url, token


def _api(method: str, path: str, body: dict = None,
         base_url: str = "", token: str = "") -> dict:
    """Minimal REST helper."""
    if not base_url or not token:
        base_url, token = _creds()
    if not base_url or not token:
        return {"error": "Mattermost credentials not set."}

    url  = f"{base_url}/api/v4{path}"
    data = json.dumps(body).encode() if body else None
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {token}",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return {"error": str(e), "status_code": e.code}
    except Exception as e:
        return {"error": str(e)}


def _resolve_channel_id(channel: str, base_url: str, token: str) -> str:
    """Resolve channel name → channel ID. Returns the ID or empty string."""
    if channel.startswith("~"):
        channel = channel[1:]
    # Try direct lookup by name
    teams_resp = _api("GET", "/teams", base_url=base_url, token=token)
    if not isinstance(teams_resp, list):
        return ""
    for team in teams_resp[:5]:
        team_id = team.get("id", "")
        resp = _api("GET", f"/teams/{team_id}/channels/name/{urllib.parse.quote(channel)}",
                    base_url=base_url, token=token)
        if resp.get("id"):
            return resp["id"]
    return ""


def mattermost_send(
    message:    str  = "",
    channel:    str  = "",
    channel_id: str  = "",
    root_id:    str  = "",
    props:      dict = None,
    **_,
) -> dict:
    """
    Post a message to a Mattermost channel.

    Args:
        message    — message text in Markdown (required)
        channel    — channel name e.g. 'town-square' (optional — uses MATTERMOST_DEFAULT_CHANNEL)
        channel_id — direct channel ID (optional — faster than name lookup)
        root_id    — post ID to reply in a thread (optional)
        props      — extra Mattermost post props dict (optional)

    Returns:
        {success, post_id, channel_id, error}
    """
    if not message:
        return {"success": False, "error": "message is required."}

    base_url, token = _creds()
    if not base_url or not token:
        return {
            "success": False,
            "error": (
                "Mattermost credentials not set. Export:\n"
                "  MATTERMOST_URL=https://your-instance.mattermost.com\n"
                "  MATTERMOST_TOKEN=your_personal_access_token"
            ),
        }

    cid = channel_id
    if not cid:
        ch_name = channel or os.environ.get("MATTERMOST_DEFAULT_CHANNEL", "")
        if ch_name:
            cid = _resolve_channel_id(ch_name, base_url, token)
    if not cid:
        return {"success": False, "error": "Could not resolve channel. Pass channel_id or channel name."}

    body: Dict[str, Any] = {"channel_id": cid, "message": message}
    if root_id:
        body["root_id"] = root_id
    if props:
        body["props"] = props

    resp = _api("POST", "/posts", body=body, base_url=base_url, token=token)
    if resp.get("id"):
        return {"success": True, "post_id": resp["id"], "channel_id": cid, "error": ""}
    return {"success": False, "error": resp.get("message", str(resp))}


def mattermost_get_messages(
    channel:    str  = "",
    channel_id: str  = "",
    limit:      int  = 10,
    **_,
) -> dict:
    """
    Retrieve recent posts from a Mattermost channel.

    Args:
        channel    — channel name (optional)
        channel_id — direct channel ID (optional — faster)
        limit      — number of posts (optional, default 10)

    Returns:
        {success, messages: [{id, user_id, message, create_at}], count, error}
    """
    base_url, token = _creds()
    if not base_url or not token:
        return {"success": False, "error": "Mattermost credentials not set."}

    cid = channel_id
    if not cid:
        ch_name = channel or os.environ.get("MATTERMOST_DEFAULT_CHANNEL", "")
        if ch_name:
            cid = _resolve_channel_id(ch_name, base_url, token)
    if not cid:
        return {"success": False, "error": "Could not resolve channel."}

    resp = _api("GET", f"/channels/{cid}/posts?per_page={min(limit, 200)}",
                base_url=base_url, token=token)
    if "error" in resp:
        return {"success": False, "error": resp["error"]}

    order  = resp.get("order", [])
    posts  = resp.get("posts", {})
    result = []
    for pid in order[:limit]:
        p = posts.get(pid, {})
        result.append({
            "id":        pid,
            "user_id":   p.get("user_id", ""),
            "message":   p.get("message", ""),
            "create_at": p.get("create_at", 0) // 1000,
        })
    return {"success": True, "messages": result, "count": len(result), "error": ""}


def mattermost_list_channels(team: str = "", **_) -> dict:
    """
    List public channels on the Mattermost instance.

    Args:
        team — team name or ID to filter (optional)

    Returns:
        {success, channels: [{id, name, display_name, type}], count, error}
    """
    base_url, token = _creds()
    if not base_url or not token:
        return {"success": False, "error": "Mattermost credentials not set."}

    teams_resp = _api("GET", "/teams", base_url=base_url, token=token)
    if not isinstance(teams_resp, list):
        return {"success": False, "error": str(teams_resp)}

    channels = []
    for t in teams_resp[:3]:
        tid  = t.get("id", "")
        name = t.get("name", "")
        if team and team not in (tid, name):
            continue
        resp = _api("GET", f"/teams/{tid}/channels?per_page=100",
                    base_url=base_url, token=token)
        if isinstance(resp, list):
            for ch in resp:
                channels.append({
                    "id":           ch.get("id", ""),
                    "name":         ch.get("name", ""),
                    "display_name": ch.get("display_name", ""),
                    "type":         ch.get("type", ""),
                    "team":         name,
                })
    return {"success": True, "channels": channels, "count": len(channels), "error": ""}
