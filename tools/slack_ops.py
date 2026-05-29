"""
Operon Slack Integration.

Two modes — credentials resolved from env vars automatically:

  1. Incoming Webhook — POST to a Slack incoming webhook URL.
                        No OAuth required.
                        Set SLACK_WEBHOOK_URL env var.

  2. Web API (SDK)    — Full Slack API via slack-sdk (optional).
                        Install: pip install slack-sdk
                        Set SLACK_BOT_TOKEN env var.

All functions accept **_ so the registry can safely pass extra params.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_webhook(url: str = "") -> str:
    return url or os.environ.get("SLACK_WEBHOOK_URL", "")


def _get_token(token: str = "") -> str:
    return token or os.environ.get("SLACK_BOT_TOKEN", "")


def _slack_api(method: str, payload: dict, token: str, *, timeout: int = 10) -> dict:
    """Call a Slack Web API method. Returns parsed JSON."""
    url  = f"https://slack.com/api/{method}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {token}",
            "User-Agent": "Operon/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            if not result.get("ok"):
                return {"success": False, "error": result.get("error", "unknown slack error")}
            return {"success": True, "data": result}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def slack_send(
    message: str = "",
    channel: str = "",
    webhook_url: str = "",
    bot_token: str = "",
    username: str = "Operon",
    icon_emoji: str = ":robot_face:",
    thread_ts: str = "",
    blocks: list = None,
    **_,
) -> dict:
    """
    Send a message to a Slack channel or webhook.

    Tries incoming webhook first (no OAuth needed), then falls back to
    the Slack Web API (chat.postMessage) with a bot token.

    Args:
        message     — message text (required unless blocks is set)
        channel     — channel name or ID, e.g. '#general' or 'C0123456' (required for bot mode)
        webhook_url — Slack incoming webhook URL (optional — auto-read from SLACK_WEBHOOK_URL)
        bot_token   — Slack bot OAuth token (optional — auto-read from SLACK_BOT_TOKEN)
        username    — display name override (optional, webhook mode only)
        icon_emoji  — emoji icon, e.g. ':robot_face:' (optional, webhook mode only)
        thread_ts   — reply in a thread (optional — provide parent message timestamp)
        blocks      — Slack Block Kit blocks list for rich messages (optional)

    Returns:
        {success, ts, channel, error}
    """
    if not message and not blocks:
        return {"success": False, "error": "message or blocks is required."}

    # ── Webhook mode ─────────────────────────────────────────────────────────
    wh = _get_webhook(webhook_url)
    if wh:
        payload: Dict[str, Any] = {"username": username, "icon_emoji": icon_emoji}
        if message:
            payload["text"] = message
        if blocks:
            payload["blocks"] = blocks
        if channel:
            payload["channel"] = channel

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            wh,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read().decode()
                if body == "ok":
                    return {"success": True, "ts": "", "channel": channel, "mode": "webhook", "error": ""}
                return {"success": False, "error": f"Slack webhook returned: {body}"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Bot API mode ──────────────────────────────────────────────────────────
    token = _get_token(bot_token)
    if not token:
        return {
            "success": False,
            "error": (
                "No Slack credentials found. Set one of:\n"
                "  SLACK_WEBHOOK_URL — for incoming webhook (simple, no OAuth)\n"
                "  SLACK_BOT_TOKEN   — for full bot API (requires Slack app)"
            ),
        }

    ch = channel or os.environ.get("SLACK_DEFAULT_CHANNEL", "#general")
    payload = {"channel": ch, "text": message or ""}
    if blocks:
        payload["blocks"] = blocks
    if thread_ts:
        payload["thread_ts"] = thread_ts

    result = _slack_api("chat.postMessage", payload, token)
    if result["success"]:
        data = result.get("data", {})
        return {
            "success": True,
            "ts":      data.get("ts", ""),
            "channel": data.get("channel", ch),
            "mode":    "bot_api",
            "error":   "",
        }
    return result


def slack_get_messages(
    channel: str = "",
    bot_token: str = "",
    limit: int = 10,
    oldest: str = "",
    **_,
) -> dict:
    """
    Fetch recent messages from a Slack channel.
    Requires SLACK_BOT_TOKEN with channels:history scope.

    Args:
        channel   — channel ID or name (optional — auto-read from SLACK_DEFAULT_CHANNEL)
        bot_token — bot OAuth token (optional — auto-read from SLACK_BOT_TOKEN)
        limit     — number of messages to fetch, max 200 (optional, default 10)
        oldest    — fetch messages after this Unix timestamp (optional)

    Returns:
        {success, messages: [{ts, user, text, thread_ts}], error}
    """
    token = _get_token(bot_token)
    if not token:
        return {"success": False, "error": "SLACK_BOT_TOKEN required to read messages."}

    ch = channel or os.environ.get("SLACK_DEFAULT_CHANNEL", "")
    if not ch:
        return {"success": False, "error": "channel is required."}

    payload: Dict[str, Any] = {"channel": ch, "limit": min(200, max(1, int(limit)))}
    if oldest:
        payload["oldest"] = oldest

    result = _slack_api("conversations.history", payload, token)
    if not result["success"]:
        return result

    raw_msgs = result["data"].get("messages", [])
    messages = [
        {
            "ts":        m.get("ts", ""),
            "user":      m.get("user", m.get("bot_id", "unknown")),
            "text":      m.get("text", ""),
            "thread_ts": m.get("thread_ts", ""),
        }
        for m in raw_msgs
    ]
    return {"success": True, "messages": messages, "channel": ch, "error": ""}


def slack_list_channels(
    bot_token: str = "",
    **_,
) -> dict:
    """
    List all public channels the bot has access to.
    Requires SLACK_BOT_TOKEN with channels:read scope.

    Returns:
        {success, channels: [{id, name, member_count}], error}
    """
    token = _get_token(bot_token)
    if not token:
        return {"success": False, "error": "SLACK_BOT_TOKEN required."}

    result = _slack_api("conversations.list", {"exclude_archived": True, "limit": 200}, token)
    if not result["success"]:
        return result

    channels = [
        {
            "id":           ch.get("id", ""),
            "name":         ch.get("name", ""),
            "member_count": ch.get("num_members", 0),
        }
        for ch in result["data"].get("channels", [])
    ]
    return {"success": True, "channels": channels, "error": ""}


def slack_upload_file(
    file_path: str = "",
    channel: str = "",
    title: str = "",
    message: str = "",
    bot_token: str = "",
    **_,
) -> dict:
    """
    Upload a local file to a Slack channel.
    Requires SLACK_BOT_TOKEN with files:write scope.

    Args:
        file_path — path to the local file (required)
        channel   — channel to post into (optional — auto-read from SLACK_DEFAULT_CHANNEL)
        title     — file title shown in Slack (optional)
        message   — message text to accompany the file (optional)
        bot_token — bot OAuth token (optional — auto-read from SLACK_BOT_TOKEN)

    Returns:
        {success, file_id, permalink, error}
    """
    import os as _os
    token = _get_token(bot_token)
    if not token:
        return {"success": False, "error": "SLACK_BOT_TOKEN required."}
    if not file_path or not _os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}

    ch = channel or _os.environ.get("SLACK_DEFAULT_CHANNEL", "")

    # Use multipart/form-data upload
    import urllib.parse as _up
    url = "https://slack.com/api/files.upload"
    boundary = "OperonBoundary12345"
    fname    = _os.path.basename(file_path)

    with open(file_path, "rb") as fh:
        file_bytes = fh.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="channels"\r\n\r\n{ch}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="filename"\r\n\r\n{fname}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="title"\r\n\r\n{title or fname}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="initial_comment"\r\n\r\n{message}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            if not result.get("ok"):
                return {"success": False, "error": result.get("error", "upload failed")}
            f = result.get("file", {})
            return {
                "success":   True,
                "file_id":   f.get("id", ""),
                "permalink": f.get("permalink", ""),
                "error":     "",
            }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Extended operations (Phase 11)
# ---------------------------------------------------------------------------

def slack_send_dm(user_id: str = "", text: str = "", bot_token: str = "", **_) -> dict:
    """Send a direct message to a Slack user by user ID (U...)."""
    token = _get_token(bot_token)
    if not token:
        return {"success": False, "error": "SLACK_BOT_TOKEN required."}
    open_res = _slack_api("conversations.open", {"users": user_id}, token)
    if not open_res["success"]:
        return open_res
    channel_id = open_res["data"]["channel"]["id"]
    return _slack_api("chat.postMessage", {"channel": channel_id, "text": text}, token)


def slack_list_users(limit: int = 100, bot_token: str = "", **_) -> dict:
    """List all workspace members."""
    token = _get_token(bot_token)
    if not token:
        return {"success": False, "error": "SLACK_BOT_TOKEN required."}
    res = _slack_api("users.list", {"limit": limit}, token)
    if not res["success"]:
        return res
    members = [
        {"id": u["id"], "name": u.get("name", ""), "real_name": u.get("real_name", ""), "is_bot": u.get("is_bot", False)}
        for u in res["data"].get("members", [])
        if not u.get("deleted", False)
    ]
    return {"success": True, "users": members, "count": len(members)}


def slack_add_reaction(channel: str = "", ts: str = "", emoji: str = "", bot_token: str = "", **_) -> dict:
    """Add a reaction emoji to a message."""
    token = _get_token(bot_token)
    if not token:
        return {"success": False, "error": "SLACK_BOT_TOKEN required."}
    return _slack_api("reactions.add", {"channel": channel, "timestamp": ts, "name": emoji.strip(":")}, token)


def slack_search_messages(query: str = "", count: int = 10, bot_token: str = "", **_) -> dict:
    """Search messages across the workspace (requires search:read scope)."""
    token = _get_token(bot_token)
    if not token:
        return {"success": False, "error": "SLACK_BOT_TOKEN required."}
    res = _slack_api("search.messages", {"query": query, "count": count}, token)
    if not res["success"]:
        return res
    matches = res["data"].get("messages", {}).get("matches", [])
    return {
        "success": True, "count": len(matches),
        "results": [{"ts": m.get("ts",""), "user": m.get("username",""), "channel": m.get("channel",{}).get("name",""), "text": m.get("text","")[:200]} for m in matches],
    }


def slack_delete_message(channel: str = "", ts: str = "", bot_token: str = "", **_) -> dict:
    """Delete a message sent by the bot."""
    token = _get_token(bot_token)
    if not token:
        return {"success": False, "error": "SLACK_BOT_TOKEN required."}
    return _slack_api("chat.delete", {"channel": channel, "ts": ts}, token)


def slack_create_channel(name: str = "", is_private: bool = False, bot_token: str = "", **_) -> dict:
    """Create a new Slack channel."""
    token = _get_token(bot_token)
    if not token:
        return {"success": False, "error": "SLACK_BOT_TOKEN required."}
    res = _slack_api("conversations.create", {"name": name, "is_private": is_private}, token)
    if not res["success"]:
        return res
    ch = res["data"].get("channel", {})
    return {"success": True, "id": ch.get("id", ""), "name": ch.get("name", "")}


def slack_status(bot_token: str = "", **_) -> dict:
    """Check Slack connection status and auth info."""
    token = _get_token(bot_token)
    if not token:
        return {"success": False, "error": "No SLACK_BOT_TOKEN configured", "token_set": False}
    res = _slack_api("auth.test", {}, token)
    if not res["success"]:
        return res
    d = res["data"]
    return {"success": True, "user": d.get("user",""), "team": d.get("team",""), "team_id": d.get("team_id",""), "url": d.get("url","")}


# ---------------------------------------------------------------------------
# Tool definitions + dispatch
# ---------------------------------------------------------------------------

_TOOL_DEFINITIONS: List[Dict] = [
    {
        "name": "slack_send",
        "description": "Send a message to a Slack channel via webhook or Bot API.",
        "input_schema": {"type": "object", "properties": {"message": {"type": "string"}, "channel": {"type": "string"}}, "required": ["message"]},
    },
    {
        "name": "slack_get_messages",
        "description": "Fetch recent messages from a Slack channel.",
        "input_schema": {"type": "object", "properties": {"channel": {"type": "string"}, "limit": {"type": "integer", "default": 20}}, "required": ["channel"]},
    },
    {
        "name": "slack_list_channels",
        "description": "List all accessible Slack channels in the workspace.",
        "input_schema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 50}}},
    },
    {
        "name": "slack_upload_file",
        "description": "Upload a local file to a Slack channel.",
        "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "channel": {"type": "string"}, "title": {"type": "string"}}, "required": ["file_path"]},
    },
    {
        "name": "slack_send_dm",
        "description": "Send a direct message to a Slack user by user ID.",
        "input_schema": {"type": "object", "properties": {"user_id": {"type": "string"}, "text": {"type": "string"}}, "required": ["user_id", "text"]},
    },
    {
        "name": "slack_list_users",
        "description": "List all Slack workspace members.",
        "input_schema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 100}}},
    },
    {
        "name": "slack_search_messages",
        "description": "Search messages across the Slack workspace.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "count": {"type": "integer", "default": 10}}, "required": ["query"]},
    },
    {
        "name": "slack_add_reaction",
        "description": "Add a reaction emoji to a Slack message.",
        "input_schema": {"type": "object", "properties": {"channel": {"type": "string"}, "ts": {"type": "string"}, "emoji": {"type": "string"}}, "required": ["channel", "ts", "emoji"]},
    },
    {
        "name": "slack_create_channel",
        "description": "Create a new Slack channel.",
        "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "is_private": {"type": "boolean", "default": False}}, "required": ["name"]},
    },
    {
        "name": "slack_status",
        "description": "Check Slack connection status and authentication.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

_DISPATCH: Dict[str, Any] = {
    "slack_send":            slack_send,
    "slack_get_messages":    slack_get_messages,
    "slack_list_channels":   slack_list_channels,
    "slack_upload_file":     slack_upload_file,
    "slack_send_dm":         slack_send_dm,
    "slack_list_users":      slack_list_users,
    "slack_search_messages": slack_search_messages,
    "slack_add_reaction":    slack_add_reaction,
    "slack_create_channel":  slack_create_channel,
    "slack_status":          slack_status,
}
