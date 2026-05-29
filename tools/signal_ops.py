"""
Operon Signal Messaging Integration.

Uses signal-cli (https://github.com/AsamK/signal-cli) as a local bridge.
signal-cli is a Java command-line tool that interfaces with the Signal
protocol. No unofficial API — uses the official Signal protocol.

Setup
-----
  1. Install signal-cli: https://github.com/AsamK/signal-cli/releases
  2. Register your number:
       signal-cli -u +1XXXXXXXXXX register
       signal-cli -u +1XXXXXXXXXX verify <code>
  3. Set environment variables:
       export SIGNAL_NUMBER=+1XXXXXXXXXX          # your registered number
       export SIGNAL_CLI_PATH=/usr/local/bin/signal-cli  # optional

All functions accept **_ for registry compatibility.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional


def _signal_cli() -> str:
    """Return the path to the signal-cli binary."""
    return os.environ.get("SIGNAL_CLI_PATH", "signal-cli")


def _signal_number() -> str:
    """Return the registered Signal number from env var."""
    return os.environ.get("SIGNAL_NUMBER", "").strip()


def signal_send(
    message:    str  = "",
    recipient:  str  = "",
    group_id:   str  = "",
    attachment: str  = "",
    **_,
) -> dict:
    """
    Send a Signal message to a contact or group.

    Args:
        message    — message text (required)
        recipient  — recipient phone number e.g. '+1234567890' (required unless group_id given)
        group_id   — Signal group ID in base64 (optional — use instead of recipient)
        attachment — local file path to attach (optional)

    Returns:
        {success, timestamp, recipient, error}
    """
    if not message:
        return {"success": False, "error": "message is required."}

    number = _signal_number()
    if not number:
        return {
            "success": False,
            "error": (
                "SIGNAL_NUMBER env var not set.\n"
                "  export SIGNAL_NUMBER=+1XXXXXXXXXX\n"
                "  Then register: signal-cli -u +1XXXXXXXXXX register"
            ),
        }

    if not recipient and not group_id:
        return {"success": False, "error": "recipient or group_id is required."}

    cmd = [_signal_cli(), "-u", number, "send"]
    if group_id:
        cmd += ["-g", group_id]
    else:
        cmd += [recipient]
    cmd += ["-m", message]
    if attachment and Path(attachment).exists():
        cmd += ["-a", attachment]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return {
                "success":   True,
                "recipient": recipient or group_id,
                "error":     "",
            }
        return {
            "success": False,
            "error":   result.stderr.strip() or result.stdout.strip() or "signal-cli failed",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": (
                "signal-cli not found. Install from: "
                "https://github.com/AsamK/signal-cli/releases\n"
                "Then set SIGNAL_CLI_PATH=/path/to/signal-cli"
            ),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "signal-cli timed out after 30s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def signal_receive(
    limit:  int  = 10,
    timeout: int = 5,
    **_,
) -> dict:
    """
    Receive pending Signal messages.

    Args:
        limit   — max messages to return (optional, default 10)
        timeout — seconds to wait for messages (optional, default 5)

    Returns:
        {success, messages: [{sender, timestamp, message, attachments}], count, error}
    """
    number = _signal_number()
    if not number:
        return {"success": False, "error": "SIGNAL_NUMBER env var not set."}

    cmd = [_signal_cli(), "-u", number, "receive",
           "--timeout", str(timeout), "-o", "json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout + 10)
        messages = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            try:
                obj   = json.loads(line)
                env   = obj.get("envelope", {})
                msg   = env.get("dataMessage", {})
                if msg.get("message"):
                    messages.append({
                        "sender":      env.get("source", "?"),
                        "timestamp":   env.get("timestamp", 0),
                        "message":     msg.get("message", ""),
                        "attachments": [a.get("filename", "") for a in msg.get("attachments", [])],
                    })
            except Exception:
                pass

        return {
            "success":  True,
            "messages": messages[:limit],
            "count":    len(messages[:limit]),
            "error":    "",
        }
    except FileNotFoundError:
        return {"success": False, "error": "signal-cli not found."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def signal_list_groups(**_) -> dict:
    """
    List Signal groups the registered number belongs to.

    Returns:
        {success, groups: [{id, name, members}], count, error}
    """
    number = _signal_number()
    if not number:
        return {"success": False, "error": "SIGNAL_NUMBER env var not set."}

    cmd = [_signal_cli(), "-u", number, "listGroups", "-o", "json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return {"success": False, "error": result.stderr.strip()}
        groups = []
        for line in result.stdout.strip().splitlines():
            try:
                g = json.loads(line)
                groups.append({
                    "id":      g.get("id", ""),
                    "name":    g.get("name", ""),
                    "members": g.get("members", []),
                })
            except Exception:
                pass
        return {"success": True, "groups": groups, "count": len(groups), "error": ""}
    except FileNotFoundError:
        return {"success": False, "error": "signal-cli not found."}
    except Exception as e:
        return {"success": False, "error": str(e)}
