"""
Operon WhatsApp Integration.

Uses the Twilio API to send and receive WhatsApp messages.
No special business account required — Twilio's sandbox works for testing.

Setup
-----
  pip install twilio
  export TWILIO_ACCOUNT_SID=ACxxxxxxxx
  export TWILIO_AUTH_TOKEN=your_token
  export TWILIO_WHATSAPP_FROM=whatsapp:+14155238886   # Twilio sandbox number
  export TWILIO_WHATSAPP_TO=whatsapp:+1XXXXXXXXXX     # default recipient

Twilio Sandbox
--------------
  1. Create a free Twilio account at https://console.twilio.com
  2. In Messaging → Try it Out → WhatsApp, follow the sandbox setup
  3. Send "join <sandbox-keyword>" from your WhatsApp to the Twilio number

All functions accept **_ for registry compatibility.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional


def _get_twilio_creds() -> tuple[str, str, str]:
    """Return (account_sid, auth_token, from_number) from env vars."""
    sid   = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    from_ = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886").strip()
    return sid, token, from_


def whatsapp_send(
    message:   str = "",
    to:        str = "",
    from_:     str = "",
    media_url: str = "",
    **_,
) -> dict:
    """
    Send a WhatsApp message via Twilio.

    Args:
        message   — message text (required unless media_url is set)
        to        — recipient WhatsApp number e.g. 'whatsapp:+1234567890'
                    (optional — auto-read from TWILIO_WHATSAPP_TO)
        from_     — sender Twilio WhatsApp number
                    (optional — auto-read from TWILIO_WHATSAPP_FROM)
        media_url — public URL of an image/video to send (optional)

    Returns:
        {success, sid, status, to, error}
    """
    sid, token, default_from = _get_twilio_creds()
    if not sid or not token:
        return {
            "success": False,
            "error": (
                "Twilio credentials not set. Export:\n"
                "  TWILIO_ACCOUNT_SID=ACxxxxxxxxxx\n"
                "  TWILIO_AUTH_TOKEN=your_token"
            ),
        }

    to_num  = to  or os.environ.get("TWILIO_WHATSAPP_TO", "").strip()
    from_num = from_ or default_from

    if not to_num:
        return {
            "success": False,
            "error": (
                "Recipient number required. Pass 'to' param or set "
                "TWILIO_WHATSAPP_TO=whatsapp:+1XXXXXXXXXX"
            ),
        }

    if not message and not media_url:
        return {"success": False, "error": "Either 'message' or 'media_url' is required."}

    # Normalise numbers — prepend whatsapp: if missing
    if not to_num.startswith("whatsapp:"):
        to_num = f"whatsapp:{to_num}"
    if not from_num.startswith("whatsapp:"):
        from_num = f"whatsapp:{from_num}"

    try:
        from twilio.rest import Client
        client = Client(sid, token)
        kwargs: dict = {"from_": from_num, "to": to_num}
        if message:
            kwargs["body"] = message
        if media_url:
            kwargs["media_url"] = [media_url]
        msg = client.messages.create(**kwargs)
        return {
            "success": True,
            "sid":     msg.sid,
            "status":  msg.status,
            "to":      to_num,
            "error":   "",
        }
    except ImportError:
        return {
            "success": False,
            "error": "twilio package not installed. Run: pip install twilio",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def whatsapp_get_messages(
    limit:      int = 10,
    to:         str = "",
    from_:      str = "",
    **_,
) -> dict:
    """
    Retrieve recent WhatsApp messages from Twilio.

    Args:
        limit  — number of messages to return (optional, default 10)
        to     — filter by recipient number (optional)
        from_  — filter by sender number (optional)

    Returns:
        {success, messages: [{sid, from, to, body, status, date_sent}], count, error}
    """
    sid, token, _ = _get_twilio_creds()
    if not sid or not token:
        return {"success": False, "error": "Twilio credentials not set."}

    try:
        from twilio.rest import Client
        client = Client(sid, token)
        kwargs: dict = {"limit": min(limit, 100)}
        if to:
            kwargs["to"] = to if to.startswith("whatsapp:") else f"whatsapp:{to}"
        if from_:
            kwargs["from_"] = from_ if from_.startswith("whatsapp:") else f"whatsapp:{from_}"
        messages_raw = client.messages.list(**kwargs)
        messages = [
            {
                "sid":       m.sid,
                "from":      m.from_,
                "to":        m.to,
                "body":      m.body,
                "status":    m.status,
                "date_sent": str(m.date_sent),
            }
            for m in messages_raw
        ]
        return {
            "success":  True,
            "messages": messages,
            "count":    len(messages),
            "error":    "",
        }
    except ImportError:
        return {"success": False, "error": "pip install twilio"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def whatsapp_status(message_sid: str = "", **_) -> dict:
    """
    Check the delivery status of a specific WhatsApp message.

    Args:
        message_sid — Twilio message SID (required)

    Returns:
        {success, sid, status, error_code, error_message, error}
    """
    if not message_sid:
        return {"success": False, "error": "message_sid is required."}

    sid, token, _ = _get_twilio_creds()
    if not sid or not token:
        return {"success": False, "error": "Twilio credentials not set."}

    try:
        from twilio.rest import Client
        client = Client(sid, token)
        msg = client.messages(message_sid).fetch()
        return {
            "success":       True,
            "sid":           msg.sid,
            "status":        msg.status,
            "error_code":    msg.error_code,
            "error_message": msg.error_message,
            "error":         "",
        }
    except ImportError:
        return {"success": False, "error": "pip install twilio"}
    except Exception as e:
        return {"success": False, "error": str(e)}
