"""
Operon Microsoft Teams Integration.

Sending via Incoming Webhooks (no OAuth, no app registration needed).
Reading messages requires Microsoft Graph API (Azure app registration).

Setup — Sending (webhook, easiest)
-----------------------------------
  1. In Teams, open a channel → ... → Connectors → Incoming Webhook
  2. Create a webhook, copy the URL
  3. export TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/...

Setup — Reading messages (Graph API, optional)
----------------------------------------------
  1. Register an app in Azure AD
  2. Grant Channel.ReadBasic.All, ChannelMessage.Read.All permissions
  3. export TEAMS_TENANT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
  4. export TEAMS_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
  5. export TEAMS_CLIENT_SECRET=your_secret

All functions accept **_ for registry compatibility.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


def _webhook_url() -> str:
    return os.environ.get("TEAMS_WEBHOOK_URL", "").strip()


def _graph_creds() -> tuple[str, str, str]:
    return (
        os.environ.get("TEAMS_TENANT_ID", "").strip(),
        os.environ.get("TEAMS_CLIENT_ID", "").strip(),
        os.environ.get("TEAMS_CLIENT_SECRET", "").strip(),
    )


def _post_json(url: str, payload: dict, token: str = "") -> dict:
    """POST JSON payload, return parsed response."""
    data    = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read()
            return json.loads(body) if body else {"success": True}
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return {"error": str(e), "code": e.code}
    except Exception as e:
        return {"error": str(e)}


def _get_graph_token() -> str:
    """Acquire an OAuth2 access token from Azure AD for Graph API."""
    tenant, client_id, client_secret = _graph_creds()
    if not all([tenant, client_id, client_secret]):
        return ""
    url  = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    data = (
        f"grant_type=client_credentials"
        f"&client_id={client_id}"
        f"&client_secret={urllib.parse.quote(client_secret)}"
        f"&scope=https%3A%2F%2Fgraph.microsoft.com%2F.default"
    ).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read()).get("access_token", "")
    except Exception:
        return ""


def teams_send(
    message:     str  = "",
    webhook_url: str  = "",
    title:       str  = "",
    color:       str  = "#0078D7",
    facts:       list = None,
    **_,
) -> dict:
    """
    Send a message to a Microsoft Teams channel via Incoming Webhook.

    Args:
        message     — message body in Markdown (required)
        webhook_url — Teams incoming webhook URL (optional — auto-read from TEAMS_WEBHOOK_URL)
        title       — card title (optional)
        color       — accent colour hex (optional, default Teams blue)
        facts       — list of {name, value} dicts for a facts card (optional)

    Returns:
        {success, error}
    """
    if not message:
        return {"success": False, "error": "message is required."}

    url = webhook_url or _webhook_url()
    if not url:
        return {
            "success": False,
            "error": (
                "No Teams webhook URL. Set TEAMS_WEBHOOK_URL or pass webhook_url.\n"
                "Create one: Teams channel → ... → Connectors → Incoming Webhook"
            ),
        }

    # Build an Adaptive Card payload (works with all Teams clients)
    body_blocks: List[Dict[str, Any]] = []
    if title:
        body_blocks.append({
            "type":   "TextBlock",
            "text":   title,
            "weight": "Bolder",
            "size":   "Medium",
        })
    body_blocks.append({
        "type": "TextBlock",
        "text": message,
        "wrap": True,
    })
    if facts:
        fact_set = {
            "type":  "FactSet",
            "facts": [{"title": f.get("name", ""), "value": f.get("value", "")}
                      for f in facts],
        }
        body_blocks.append(fact_set)

    payload = {
        "type":        "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type":    "AdaptiveCard",
                    "version": "1.4",
                    "body":    body_blocks,
                    "msteams": {"width": "Full"},
                },
            }
        ],
    }

    resp = _post_json(url, payload)
    if resp.get("error"):
        return {"success": False, "error": resp["error"]}
    return {"success": True, "error": ""}


def teams_get_messages(
    team_id:    str  = "",
    channel_id: str  = "",
    limit:      int  = 10,
    **_,
) -> dict:
    """
    Retrieve recent messages from a Teams channel using the Graph API.
    Requires TEAMS_TENANT_ID, TEAMS_CLIENT_ID, TEAMS_CLIENT_SECRET.

    Args:
        team_id    — Teams group/team ID (required)
        channel_id — Teams channel ID (required)
        limit      — max messages (optional, default 10)

    Returns:
        {success, messages: [{id, sender, body, created_at}], count, error}
    """
    if not team_id or not channel_id:
        return {"success": False, "error": "team_id and channel_id are required."}

    token = _get_graph_token()
    if not token:
        return {
            "success": False,
            "error": (
                "Graph API credentials not set. Export:\n"
                "  TEAMS_TENANT_ID, TEAMS_CLIENT_ID, TEAMS_CLIENT_SECRET"
            ),
        }

    url = (
        f"https://graph.microsoft.com/v1.0/teams/{team_id}"
        f"/channels/{channel_id}/messages?$top={min(limit, 50)}"
    )
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data  = json.loads(resp.read())
            items = data.get("value", [])
            messages = [
                {
                    "id":         m.get("id", ""),
                    "sender":     (m.get("from") or {}).get("user", {}).get("displayName", "?"),
                    "body":       (m.get("body") or {}).get("content", ""),
                    "created_at": m.get("createdDateTime", ""),
                }
                for m in items
            ]
            return {"success": True, "messages": messages, "count": len(messages), "error": ""}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.read()[:200].decode()}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def teams_list_teams(**_) -> dict:
    """
    List Microsoft Teams groups accessible to the app.
    Requires Graph API credentials.

    Returns:
        {success, teams: [{id, display_name, description}], count, error}
    """
    token = _get_graph_token()
    if not token:
        return {"success": False, "error": "Graph API credentials not set."}

    url = "https://graph.microsoft.com/v1.0/groups?$filter=resourceProvisioningOptions/Any(x:x eq 'Team')&$select=id,displayName,description"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data  = json.loads(resp.read())
            items = data.get("value", [])
            teams = [
                {
                    "id":           t.get("id", ""),
                    "display_name": t.get("displayName", ""),
                    "description":  t.get("description", ""),
                }
                for t in items
            ]
            return {"success": True, "teams": teams, "count": len(teams), "error": ""}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Ensure urllib.parse is available for _get_graph_token
import urllib.parse
