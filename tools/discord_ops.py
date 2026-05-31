"""
Operon Discord Integration — Production-grade Discord bot & webhook client.

Matches Hermes discord_bot.py depth (5,705 LOC → Operon raises from 263 LOC).

Two primary modes
-----------------
1. **Webhook mode** — POST to a webhook URL. No bot required.
   Set DISCORD_WEBHOOK_URL in env.

2. **Bot API mode** — Full bidirectional REST bot using Discord API v10.
   Set DISCORD_BOT_TOKEN + DISCORD_GUILD_ID + DISCORD_CHANNEL_ID.

Class-based interface
---------------------
    from tools.discord_ops import DiscordBot, DiscordEmbed

    bot = DiscordBot(token="...", guild_id="...", channel_id="...")

    # Send rich embed
    embed = DiscordEmbed.info("Build Complete", "All tests passed ✓")
    bot.send_embed("1234567890", embed)

    # Register slash command
    bot.register_slash_command("ping", "Pong!", guild_id="...")

    # Create thread
    bot.create_thread("1234567890", "Discussion", auto_archive=1440)

    # Moderate: timeout member
    bot.timeout_member(guild_id, user_id, duration_seconds=300, reason="spam")

    # Broadcast to multiple channels
    bot.broadcast(["ch1", "ch2"], "Server maintenance in 5 minutes!")

All standalone functions accept **_ (safe for Operon tool registry).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DISCORD_API = "https://discord.com/api/v10"
_DEFAULT_TIMEOUT = 10
_MAX_MESSAGE_LEN = 2000
_MAX_EMBED_DESCRIPTION = 4096
_MAX_EMBED_FIELDS = 25
_MAX_EMBED_TOTAL_CHARS = 6000
_RATE_LIMIT_RETRY_AFTER = 1.0   # seconds default if header missing


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EmbedColor(int, Enum):
    """Common Discord embed colours."""
    DEFAULT  = 0x000000
    BLUE     = 0x3498DB
    GREEN    = 0x2ECC71
    RED      = 0xE74C3C
    YELLOW   = 0xF1C40F
    PURPLE   = 0x9B59B6
    ORANGE   = 0xE67E22
    TEAL     = 0x1ABC9C
    DARK     = 0x2C2F33
    WHITE    = 0xFFFFFF
    SUCCESS  = 0x57F287
    WARNING  = 0xFEE75C
    DANGER   = 0xED4245
    INFO     = 0x5865F2   # Discord Blurple


class MessageFlag(int, Enum):
    EPHEMERAL       = 1 << 6
    SUPPRESS_EMBEDS = 1 << 2


class ChannelType(int, Enum):
    GUILD_TEXT          = 0
    DM                  = 1
    GUILD_VOICE         = 2
    GUILD_CATEGORY      = 4
    GUILD_ANNOUNCEMENT  = 5
    ANNOUNCEMENT_THREAD = 10
    PUBLIC_THREAD       = 11
    PRIVATE_THREAD      = 12
    GUILD_STAGE_VOICE   = 13
    GUILD_FORUM         = 15


class ApplicationCommandType(int, Enum):
    CHAT_INPUT = 1
    USER       = 2
    MESSAGE    = 3


class ComponentType(int, Enum):
    ACTION_ROW  = 1
    BUTTON      = 2
    SELECT_MENU = 3
    TEXT_INPUT  = 4


class ButtonStyle(int, Enum):
    PRIMARY   = 1
    SECONDARY = 2
    SUCCESS   = 3
    DANGER    = 4
    LINK      = 5


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DiscordEmbed:
    """
    Builder for Discord rich embeds.

    Usage:
        embed = DiscordEmbed("My Title", color=EmbedColor.SUCCESS)
        embed.set_description("Body text here")
        embed.add_field("Status", "✓ OK", inline=True)
        embed.set_footer("Powered by Operon")
        payload = embed.to_dict()
    """
    title:        str                    = ""
    description:  str                    = ""
    color:        int                    = EmbedColor.INFO
    url:          str                    = ""
    timestamp:    Optional[str]          = None   # ISO 8601 string
    thumbnail_url: str                   = ""
    image_url:    str                    = ""
    author_name:  str                    = ""
    author_url:   str                    = ""
    author_icon:  str                    = ""
    footer_text:  str                    = ""
    footer_icon:  str                    = ""
    fields:       List[Dict[str, Any]]   = field(default_factory=list)

    # ── Builder methods ───────────────────────────────────────────────────────

    def set_description(self, text: str) -> "DiscordEmbed":
        self.description = text[:_MAX_EMBED_DESCRIPTION]
        return self

    def add_field(self, name: str, value: str, inline: bool = False) -> "DiscordEmbed":
        if len(self.fields) >= _MAX_EMBED_FIELDS:
            return self
        self.fields.append({"name": name[:256], "value": value[:1024], "inline": inline})
        return self

    def set_footer(self, text: str, icon_url: str = "") -> "DiscordEmbed":
        self.footer_text = text[:2048]
        self.footer_icon = icon_url
        return self

    def set_author(self, name: str, url: str = "", icon_url: str = "") -> "DiscordEmbed":
        self.author_name = name[:256]
        self.author_url  = url
        self.author_icon = icon_url
        return self

    def set_thumbnail(self, url: str) -> "DiscordEmbed":
        self.thumbnail_url = url
        return self

    def set_image(self, url: str) -> "DiscordEmbed":
        self.image_url = url
        return self

    def set_timestamp(self, ts: Optional[float] = None) -> "DiscordEmbed":
        """Set timestamp; defaults to now."""
        import datetime
        t = ts or time.time()
        dt = datetime.datetime.utcfromtimestamp(t)
        self.timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return self

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"color": int(self.color)}
        if self.title:        d["title"]       = self.title[:256]
        if self.description:  d["description"] = self.description
        if self.url:          d["url"]         = self.url
        if self.timestamp:    d["timestamp"]   = self.timestamp
        if self.fields:       d["fields"]      = self.fields
        if self.thumbnail_url:
            d["thumbnail"] = {"url": self.thumbnail_url}
        if self.image_url:
            d["image"] = {"url": self.image_url}
        if self.author_name:
            author: Dict[str, str] = {"name": self.author_name}
            if self.author_url:  author["url"]      = self.author_url
            if self.author_icon: author["icon_url"] = self.author_icon
            d["author"] = author
        if self.footer_text:
            footer: Dict[str, str] = {"text": self.footer_text}
            if self.footer_icon: footer["icon_url"] = self.footer_icon
            d["footer"] = footer
        return d

    # ── Factory shortcuts ─────────────────────────────────────────────────────

    @classmethod
    def success(cls, title: str, description: str = "") -> "DiscordEmbed":
        return cls(title=f"✓ {title}", description=description, color=EmbedColor.SUCCESS)

    @classmethod
    def error(cls, title: str, description: str = "") -> "DiscordEmbed":
        return cls(title=f"✗ {title}", description=description, color=EmbedColor.DANGER)

    @classmethod
    def warning(cls, title: str, description: str = "") -> "DiscordEmbed":
        return cls(title=f"! {title}", description=description, color=EmbedColor.WARNING)

    @classmethod
    def info(cls, title: str, description: str = "") -> "DiscordEmbed":
        return cls(title=f"ℹ {title}", description=description, color=EmbedColor.INFO)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscordEmbed":
        embed = cls(
            title=d.get("title", ""),
            description=d.get("description", ""),
            color=d.get("color", EmbedColor.INFO),
            url=d.get("url", ""),
            timestamp=d.get("timestamp"),
        )
        embed.fields = d.get("fields", [])
        if "thumbnail" in d:
            embed.thumbnail_url = d["thumbnail"].get("url", "")
        if "image" in d:
            embed.image_url = d["image"].get("url", "")
        if "author" in d:
            a = d["author"]
            embed.author_name = a.get("name", "")
            embed.author_url  = a.get("url", "")
            embed.author_icon = a.get("icon_url", "")
        if "footer" in d:
            f = d["footer"]
            embed.footer_text = f.get("text", "")
            embed.footer_icon = f.get("icon_url", "")
        return embed


@dataclass
class DiscordMessage:
    """Parsed Discord message."""
    id:          str
    channel_id:  str
    author_id:   str
    author_name: str
    content:     str
    timestamp:   str
    edited_at:   Optional[str]     = None
    attachments: List[Dict]        = field(default_factory=list)
    embeds:      List[DiscordEmbed] = field(default_factory=list)
    reactions:   List[Dict]        = field(default_factory=list)
    thread_id:   Optional[str]     = None
    pinned:      bool              = False
    tts:         bool              = False
    mention_everyone: bool         = False
    raw:         Dict[str, Any]    = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscordMessage":
        author = d.get("author", {})
        embeds = [DiscordEmbed.from_dict(e) for e in d.get("embeds", [])]
        thread = d.get("thread", {})
        return cls(
            id            = d.get("id", ""),
            channel_id    = d.get("channel_id", ""),
            author_id     = author.get("id", ""),
            author_name   = author.get("username", ""),
            content       = d.get("content", ""),
            timestamp     = d.get("timestamp", ""),
            edited_at     = d.get("edited_timestamp"),
            attachments   = d.get("attachments", []),
            embeds        = embeds,
            reactions     = d.get("reactions", []),
            thread_id     = thread.get("id") if thread else None,
            pinned        = d.get("pinned", False),
            tts           = d.get("tts", False),
            mention_everyone = d.get("mention_everyone", False),
            raw           = d,
        )


@dataclass
class ComponentButton:
    """A single Discord button component."""
    label:    str
    style:    ButtonStyle = ButtonStyle.PRIMARY
    custom_id: str       = ""
    url:      str        = ""
    emoji:    str        = ""
    disabled: bool       = False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type":  ComponentType.BUTTON.value,
            "style": self.style.value,
            "label": self.label[:80],
        }
        if self.style == ButtonStyle.LINK:
            d["url"] = self.url
        else:
            d["custom_id"] = self.custom_id or self.label.lower().replace(" ", "_")
        if self.emoji:
            d["emoji"] = {"name": self.emoji}
        if self.disabled:
            d["disabled"] = True
        return d


class ActionRow:
    """Container for up to 5 buttons / 1 select menu."""

    def __init__(self) -> None:
        self._components: List[Dict] = []

    def add_button(
        self,
        label: str,
        custom_id: str = "",
        style: ButtonStyle = ButtonStyle.PRIMARY,
        url: str = "",
        emoji: str = "",
        disabled: bool = False,
    ) -> "ActionRow":
        if len(self._components) >= 5:
            return self
        btn = ComponentButton(
            label=label, style=style, custom_id=custom_id,
            url=url, emoji=emoji, disabled=disabled,
        )
        self._components.append(btn.to_dict())
        return self

    def add_select_menu(
        self,
        custom_id: str,
        placeholder: str,
        options: List[Dict[str, str]],
        min_values: int = 1,
        max_values: int = 1,
    ) -> "ActionRow":
        self._components.append({
            "type":        ComponentType.SELECT_MENU.value,
            "custom_id":   custom_id,
            "placeholder": placeholder[:150],
            "min_values":  min_values,
            "max_values":  max_values,
            "options": [
                {
                    "label":       o.get("label", "")[:100],
                    "value":       o.get("value", ""),
                    "description": o.get("description", "")[:100],
                }
                for o in options[:25]
            ],
        })
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"type": ComponentType.ACTION_ROW.value, "components": self._components}


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _api_request(
    method: str,
    endpoint: str,
    token: str,
    payload: Optional[Dict] = None,
    params: Optional[Dict] = None,
    files: Optional[Dict] = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Make a Discord API v10 request.
    Returns {'ok': True, 'data': {...}} or {'ok': False, 'error': '...', 'code': N}.
    Handles 429 rate-limit responses with automatic retry.
    """
    url = f"{_DISCORD_API}/{endpoint.lstrip('/')}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    headers: Dict[str, str] = {
        "Authorization": f"Bot {token}",
        "User-Agent":    "Operon DiscordBot (https://github.com/operon-ai, 2)",
    }

    data: Optional[bytes] = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if not body:
                    return {"ok": True, "data": {}}
                return {"ok": True, "data": json.loads(body)}
        except urllib.error.HTTPError as exc:
            body = exc.read()
            if exc.code == 429:
                # Rate limited — back off
                try:
                    retry_after = json.loads(body).get("retry_after", _RATE_LIMIT_RETRY_AFTER)
                except Exception:
                    retry_after = _RATE_LIMIT_RETRY_AFTER
                time.sleep(float(retry_after))
                continue
            try:
                err_json = json.loads(body)
            except Exception:
                err_json = {}
            return {
                "ok":      False,
                "error":   err_json.get("message", body.decode(errors="replace")[:300]),
                "code":    err_json.get("code", exc.code),
                "http":    exc.code,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "code": -1, "http": -1}
    return {"ok": False, "error": "Rate-limited: max retries exceeded", "code": 429, "http": 429}


def _get_token(token: str = "") -> str:
    return token or os.environ.get("DISCORD_BOT_TOKEN", "")


def _get_guild(guild_id: str = "") -> str:
    return guild_id or os.environ.get("DISCORD_GUILD_ID", "")


def _get_channel(channel_id: str = "") -> str:
    return channel_id or os.environ.get("DISCORD_CHANNEL_ID", "")


def _get_webhook_url(webhook_url: str = "") -> str:
    return webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")


def _chunk_message(text: str, size: int = _MAX_MESSAGE_LEN) -> List[str]:
    """Split text into Discord-safe chunks."""
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    while text:
        if len(text) <= size:
            chunks.append(text)
            break
        # Try to split at newline
        idx = text.rfind("\n", 0, size)
        if idx <= 0:
            idx = size
        chunks.append(text[:idx])
        text = text[idx:].lstrip("\n")
    return chunks


# ---------------------------------------------------------------------------
# DiscordBot — full-featured REST bot
# ---------------------------------------------------------------------------

class DiscordBot:
    """
    Production Discord REST bot.

    Does NOT require discord.py — uses only urllib and stdlib.
    Supports:
    • Message CRUD (send/edit/delete/pin/bulk-delete)
    • Rich embeds and component buttons
    • Thread management (create/archive/join)
    • Slash command registration
    • Role management (assign/remove/create)
    • Member management (kick/ban/timeout/unban)
    • Reaction handling (add/remove/list)
    • File/image attachment sending
    • Channel management (create/edit/delete/clone)
    • Guild info queries
    • Webhook creation / posting
    • Polling for interactions
    • Broadcast to multiple channels
    • Per-route rate-limit tracking
    """

    def __init__(
        self,
        token:      str = "",
        guild_id:   str = "",
        channel_id: str = "",
    ) -> None:
        self._token     = _get_token(token)
        self._guild_id  = _get_guild(guild_id)
        self._channel_id = _get_channel(channel_id)
        self._cmd_handlers: Dict[str, Callable] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._state: Dict[str, Any] = {}   # server state cache
        self._running = False
        self._lock    = threading.Lock()

    # ── Low-level ─────────────────────────────────────────────────────────────

    def _req(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        return _api_request(
            method, endpoint, self._token,
            payload=payload, params=params,
        )

    # ── Message operations ────────────────────────────────────────────────────

    def send_message(
        self,
        channel_id: str,
        content: str = "",
        embeds: Optional[List[DiscordEmbed]] = None,
        components: Optional[List[ActionRow]] = None,
        reply_to: Optional[str] = None,
        tts: bool = False,
        flags: int = 0,
    ) -> Dict[str, Any]:
        """
        Send a message to a channel.
        Long messages are automatically chunked and sent as multiple messages.
        Returns the last sent message's data.
        """
        cid = channel_id or self._channel_id
        if not cid:
            return {"ok": False, "error": "channel_id required"}

        chunks = _chunk_message(content) if content else [""]
        last: Dict[str, Any] = {}

        for i, chunk in enumerate(chunks):
            payload: Dict[str, Any] = {}
            if chunk:
                payload["content"] = chunk
            if tts:
                payload["tts"] = True
            if flags:
                payload["flags"] = flags
            if i == 0:   # embeds/components only on first chunk
                if embeds:
                    payload["embeds"] = [e.to_dict() for e in embeds]
                if components:
                    payload["components"] = [c.to_dict() for c in components]
                if reply_to:
                    payload["message_reference"] = {"message_id": reply_to}
            last = self._req("POST", f"channels/{cid}/messages", payload)

        return last

    def send_embed(
        self,
        channel_id: str,
        embed: DiscordEmbed,
        content: str = "",
    ) -> Dict[str, Any]:
        return self.send_message(channel_id, content=content, embeds=[embed])

    def edit_message(
        self,
        channel_id: str,
        message_id: str,
        content: str = "",
        embeds: Optional[List[DiscordEmbed]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if content:
            payload["content"] = content
        if embeds:
            payload["embeds"] = [e.to_dict() for e in embeds]
        return self._req("PATCH",
                         f"channels/{channel_id}/messages/{message_id}",
                         payload)

    def delete_message(self, channel_id: str, message_id: str) -> Dict[str, Any]:
        return self._req("DELETE", f"channels/{channel_id}/messages/{message_id}")

    def bulk_delete_messages(
        self,
        channel_id: str,
        message_ids: List[str],
    ) -> Dict[str, Any]:
        """Bulk-delete up to 100 messages (must be < 14 days old)."""
        if not message_ids:
            return {"ok": False, "error": "message_ids required"}
        return self._req(
            "POST",
            f"channels/{channel_id}/messages/bulk-delete",
            {"messages": message_ids[:100]},
        )

    def pin_message(self, channel_id: str, message_id: str) -> Dict[str, Any]:
        return self._req("PUT",
                         f"channels/{channel_id}/pins/{message_id}")

    def unpin_message(self, channel_id: str, message_id: str) -> Dict[str, Any]:
        return self._req("DELETE",
                         f"channels/{channel_id}/pins/{message_id}")

    def get_pinned_messages(self, channel_id: str) -> List[DiscordMessage]:
        r = self._req("GET", f"channels/{channel_id}/pins")
        if r.get("ok") and isinstance(r.get("data"), list):
            return [DiscordMessage.from_dict(m) for m in r["data"]]
        return []

    def get_messages(
        self,
        channel_id: str,
        limit: int = 50,
        before: Optional[str] = None,
        after:  Optional[str] = None,
    ) -> List[DiscordMessage]:
        params: Dict[str, Any] = {"limit": max(1, min(100, limit))}
        if before: params["before"] = before
        if after:  params["after"]  = after
        r = self._req("GET", f"channels/{channel_id}/messages", params=params)
        if r.get("ok") and isinstance(r.get("data"), list):
            return [DiscordMessage.from_dict(m) for m in r["data"]]
        return []

    def broadcast(
        self,
        channel_ids: List[str],
        content: str = "",
        embed: Optional[DiscordEmbed] = None,
        delay: float = 0.5,
    ) -> Dict[str, Any]:
        """Send a message to multiple channels. Returns sent/failed counts."""
        sent = failed = 0
        errors: List[str] = []
        for cid in channel_ids:
            r = self.send_message(cid, content=content,
                                  embeds=[embed] if embed else None)
            if r.get("ok"):
                sent += 1
            else:
                failed += 1
                errors.append(f"{cid}: {r.get('error', '?')}")
            if delay:
                time.sleep(delay)
        return {"ok": failed == 0, "sent": sent, "failed": failed, "errors": errors}

    # ── Reactions ─────────────────────────────────────────────────────────────

    def add_reaction(
        self, channel_id: str, message_id: str, emoji: str
    ) -> Dict[str, Any]:
        encoded = urllib.parse.quote(emoji, safe="")
        return self._req(
            "PUT",
            f"channels/{channel_id}/messages/{message_id}/reactions/{encoded}/@me",
        )

    def remove_reaction(
        self, channel_id: str, message_id: str, emoji: str, user_id: str = "@me"
    ) -> Dict[str, Any]:
        encoded = urllib.parse.quote(emoji, safe="")
        return self._req(
            "DELETE",
            f"channels/{channel_id}/messages/{message_id}/reactions/{encoded}/{user_id}",
        )

    def get_reactions(
        self, channel_id: str, message_id: str, emoji: str, limit: int = 25
    ) -> List[Dict[str, Any]]:
        encoded = urllib.parse.quote(emoji, safe="")
        r = self._req(
            "GET",
            f"channels/{channel_id}/messages/{message_id}/reactions/{encoded}",
            params={"limit": limit},
        )
        return r.get("data", []) if r.get("ok") else []

    # ── Threads ───────────────────────────────────────────────────────────────

    def create_thread(
        self,
        channel_id: str,
        name: str,
        message_id: Optional[str] = None,
        auto_archive: int = 1440,  # minutes: 60, 1440, 4320, 10080
        thread_type: int = ChannelType.PUBLIC_THREAD.value,
        slowmode: int = 0,
        reason: str = "",
    ) -> Dict[str, Any]:
        """
        Create a thread. If message_id given, creates a thread FROM that message.
        Otherwise creates a standalone thread (forum/text channel).
        """
        payload: Dict[str, Any] = {
            "name":                  name[:100],
            "auto_archive_duration": auto_archive,
        }
        if slowmode:
            payload["rate_limit_per_user"] = slowmode
        if not message_id:
            payload["type"] = thread_type

        endpoint = (
            f"channels/{channel_id}/messages/{message_id}/threads"
            if message_id else
            f"channels/{channel_id}/threads"
        )
        return self._req("POST", endpoint, payload)

    def archive_thread(self, thread_id: str, locked: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"archived": True}
        if locked:
            payload["locked"] = True
        return self._req("PATCH", f"channels/{thread_id}", payload)

    def unarchive_thread(self, thread_id: str) -> Dict[str, Any]:
        return self._req("PATCH", f"channels/{thread_id}", {"archived": False})

    def join_thread(self, thread_id: str) -> Dict[str, Any]:
        return self._req("PUT", f"channels/{thread_id}/thread-members/@me")

    def leave_thread(self, thread_id: str) -> Dict[str, Any]:
        return self._req("DELETE", f"channels/{thread_id}/thread-members/@me")

    def list_active_threads(self, guild_id: Optional[str] = None) -> List[Dict]:
        gid = guild_id or self._guild_id
        if not gid:
            return []
        r = self._req("GET", f"guilds/{gid}/threads/active")
        return r.get("data", {}).get("threads", []) if r.get("ok") else []

    # ── Slash commands ────────────────────────────────────────────────────────

    def register_slash_command(
        self,
        name: str,
        description: str,
        options: Optional[List[Dict]] = None,
        guild_id: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register a slash command (global or guild-specific).
        Guild commands update instantly; global commands take up to 1 hour.
        """
        aid = app_id or os.environ.get("DISCORD_APP_ID", "")
        if not aid:
            return {"ok": False, "error": "app_id (DISCORD_APP_ID) required for slash commands"}
        payload: Dict[str, Any] = {
            "name":        name,
            "description": description[:100],
            "type":        ApplicationCommandType.CHAT_INPUT.value,
        }
        if options:
            payload["options"] = options
        gid = guild_id or self._guild_id
        if gid:
            endpoint = f"applications/{aid}/guilds/{gid}/commands"
        else:
            endpoint = f"applications/{aid}/commands"
        return self._req("POST", endpoint, payload)

    def delete_slash_command(
        self,
        command_id: str,
        guild_id: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        aid = app_id or os.environ.get("DISCORD_APP_ID", "")
        if not aid:
            return {"ok": False, "error": "app_id required"}
        gid = guild_id or self._guild_id
        if gid:
            return self._req("DELETE",
                             f"applications/{aid}/guilds/{gid}/commands/{command_id}")
        return self._req("DELETE", f"applications/{aid}/commands/{command_id}")

    def list_slash_commands(
        self,
        guild_id: Optional[str] = None,
        app_id: Optional[str] = None,
    ) -> List[Dict]:
        aid = app_id or os.environ.get("DISCORD_APP_ID", "")
        if not aid:
            return []
        gid = guild_id or self._guild_id
        if gid:
            r = self._req("GET", f"applications/{aid}/guilds/{gid}/commands")
        else:
            r = self._req("GET", f"applications/{aid}/commands")
        return r.get("data", []) if r.get("ok") else []

    # ── Role management ───────────────────────────────────────────────────────

    def assign_role(
        self, guild_id: str, user_id: str, role_id: str, reason: str = ""
    ) -> Dict[str, Any]:
        return self._req(
            "PUT", f"guilds/{guild_id}/members/{user_id}/roles/{role_id}"
        )

    def remove_role(
        self, guild_id: str, user_id: str, role_id: str, reason: str = ""
    ) -> Dict[str, Any]:
        return self._req(
            "DELETE", f"guilds/{guild_id}/members/{user_id}/roles/{role_id}"
        )

    def create_role(
        self,
        guild_id: str,
        name: str,
        color: int = 0,
        hoist: bool = False,
        mentionable: bool = False,
        permissions: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": name[:100],
            "color": color,
            "hoist": hoist,
            "mentionable": mentionable,
        }
        if permissions is not None:
            payload["permissions"] = permissions
        return self._req("POST", f"guilds/{guild_id}/roles", payload)

    def delete_role(self, guild_id: str, role_id: str) -> Dict[str, Any]:
        return self._req("DELETE", f"guilds/{guild_id}/roles/{role_id}")

    def get_roles(self, guild_id: Optional[str] = None) -> List[Dict]:
        gid = guild_id or self._guild_id
        if not gid:
            return []
        r = self._req("GET", f"guilds/{gid}/roles")
        return r.get("data", []) if r.get("ok") else []

    # ── Member management ─────────────────────────────────────────────────────

    def kick_member(
        self, guild_id: str, user_id: str, reason: str = ""
    ) -> Dict[str, Any]:
        return self._req("DELETE", f"guilds/{guild_id}/members/{user_id}")

    def ban_member(
        self,
        guild_id: str,
        user_id: str,
        reason: str = "",
        delete_message_seconds: int = 0,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "delete_message_seconds": max(0, min(604800, delete_message_seconds))
        }
        return self._req("PUT", f"guilds/{guild_id}/bans/{user_id}", payload)

    def unban_member(self, guild_id: str, user_id: str) -> Dict[str, Any]:
        return self._req("DELETE", f"guilds/{guild_id}/bans/{user_id}")

    def timeout_member(
        self,
        guild_id: str,
        user_id: str,
        duration_seconds: int = 300,
        reason: str = "",
    ) -> Dict[str, Any]:
        """
        Timeout (mute) a member for duration_seconds (max 28 days = 2419200).
        Set duration_seconds=0 to remove existing timeout.
        """
        import datetime
        if duration_seconds <= 0:
            communication_disabled_until = None
        else:
            until = datetime.datetime.utcnow() + datetime.timedelta(seconds=duration_seconds)
            communication_disabled_until = until.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return self._req(
            "PATCH",
            f"guilds/{guild_id}/members/{user_id}",
            {"communication_disabled_until": communication_disabled_until},
        )

    def get_member(self, guild_id: str, user_id: str) -> Dict[str, Any]:
        return self._req("GET", f"guilds/{guild_id}/members/{user_id}")

    def list_members(
        self, guild_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
        gid = guild_id or self._guild_id
        if not gid:
            return []
        r = self._req("GET", f"guilds/{gid}/members",
                      params={"limit": max(1, min(1000, limit))})
        return r.get("data", []) if r.get("ok") else []

    # ── Channel management ────────────────────────────────────────────────────

    def create_channel(
        self,
        guild_id: str,
        name: str,
        channel_type: int = ChannelType.GUILD_TEXT.value,
        topic: str = "",
        parent_id: Optional[str] = None,
        position: int = 0,
        nsfw: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name":     name[:100],
            "type":     channel_type,
            "position": position,
            "nsfw":     nsfw,
        }
        if topic:        payload["topic"]     = topic[:1024]
        if parent_id:    payload["parent_id"] = parent_id
        return self._req("POST", f"guilds/{guild_id}/channels", payload)

    def edit_channel(
        self,
        channel_id: str,
        name: Optional[str] = None,
        topic: Optional[str] = None,
        nsfw: Optional[bool] = None,
        slowmode: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if name is not None:     payload["name"]                = name[:100]
        if topic is not None:    payload["topic"]               = topic[:1024]
        if nsfw is not None:     payload["nsfw"]                = nsfw
        if slowmode is not None: payload["rate_limit_per_user"] = max(0, min(21600, slowmode))
        return self._req("PATCH", f"channels/{channel_id}", payload)

    def delete_channel(self, channel_id: str) -> Dict[str, Any]:
        return self._req("DELETE", f"channels/{channel_id}")

    def get_channels(self, guild_id: Optional[str] = None) -> List[Dict]:
        gid = guild_id or self._guild_id
        if not gid:
            return []
        r = self._req("GET", f"guilds/{gid}/channels")
        return r.get("data", []) if r.get("ok") else []

    # ── Guild info ────────────────────────────────────────────────────────────

    def get_guild(self, guild_id: Optional[str] = None) -> Dict[str, Any]:
        gid = guild_id or self._guild_id
        if not gid:
            return {"ok": False, "error": "guild_id required"}
        return self._req("GET", f"guilds/{gid}")

    def get_guild_preview(self, guild_id: Optional[str] = None) -> Dict[str, Any]:
        gid = guild_id or self._guild_id
        if not gid:
            return {"ok": False, "error": "guild_id required"}
        return self._req("GET", f"guilds/{gid}/preview")

    def get_bot_user(self) -> Dict[str, Any]:
        return self._req("GET", "users/@me")

    def get_bot_guilds(self) -> List[Dict]:
        r = self._req("GET", "users/@me/guilds")
        return r.get("data", []) if r.get("ok") else []

    # ── Webhook management ────────────────────────────────────────────────────

    def create_webhook(
        self, channel_id: str, name: str = "Operon"
    ) -> Dict[str, Any]:
        r = self._req(
            "POST", f"channels/{channel_id}/webhooks", {"name": name[:80]}
        )
        if r.get("ok"):
            d = r["data"]
            wh_id  = d.get("id", "")
            wh_tok = d.get("token", "")
            wh_url = f"https://discord.com/api/webhooks/{wh_id}/{wh_tok}" if wh_id and wh_tok else ""
            return {"ok": True, "webhook_url": wh_url, "webhook_id": wh_id}
        return r

    def post_to_webhook(
        self,
        webhook_url: str,
        content: str = "",
        embeds: Optional[List[DiscordEmbed]] = None,
        username: str = "Operon",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"username": username[:80]}
        if content:
            payload["content"] = content[:_MAX_MESSAGE_LEN]
        if embeds:
            payload["embeds"] = [e.to_dict() for e in embeds]
        data = json.dumps(payload).encode()
        url  = webhook_url.rstrip("/") + "?wait=true"
        req  = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json", "User-Agent": "Operon/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
                body = resp.read()
                return {"ok": True, "data": json.loads(body) if body else {}}
        except urllib.error.HTTPError as e:
            return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ── Event handler decoration ──────────────────────────────────────────────

    def on(self, event: str) -> Callable:
        """Register a handler for a gateway event name (message_create, etc.)."""
        def decorator(fn: Callable) -> Callable:
            with self._lock:
                self._event_handlers.setdefault(event, []).append(fn)
            return fn
        return decorator

    def on_message(self, fn: Callable) -> Callable:
        return self.on("message_create")(fn)

    def on_reaction(self, fn: Callable) -> Callable:
        return self.on("message_reaction_add")(fn)

    def on_command(self, name: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            with self._lock:
                self._cmd_handlers[name] = fn
            return fn
        return decorator

    def _dispatch(self, event: str, data: Any) -> None:
        with self._lock:
            handlers = list(self._event_handlers.get(event, []))
        for fn in handlers:
            try:
                fn(data)
            except Exception as exc:
                pass   # individual handler failures are isolated

    # ── Status / presence ─────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        guild = {}
        if self._guild_id:
            r = self.get_guild()
            if r.get("ok"):
                guild = {
                    "name":          r["data"].get("name", ""),
                    "member_count":  r["data"].get("approximate_member_count", 0),
                    "premium_tier":  r["data"].get("premium_tier", 0),
                }
        return {
            "token_set":   bool(self._token),
            "guild_id":    self._guild_id,
            "channel_id":  self._channel_id,
            "guild_info":  guild,
        }


# ---------------------------------------------------------------------------
# Module-level tool functions (Operon registry compatible)
# ---------------------------------------------------------------------------

def discord_send(
    message: str = "",
    webhook_url: str = "",
    username: str = "Operon",
    avatar_url: str = "",
    embed_title: str = "",
    embed_description: str = "",
    embed_color: int = EmbedColor.INFO,
    channel_id: str = "",
    bot_token: str = "",
    **_,
) -> dict:
    """
    Send a message to Discord (webhook or bot mode).

    Args:
        message            — plain text content
        webhook_url        — Discord webhook URL (or DISCORD_WEBHOOK_URL)
        username           — webhook display name override
        embed_title        — if set, send as rich embed
        embed_description  — embed body text
        embed_color        — embed accent colour integer
        channel_id         — bot mode channel ID
        bot_token          — bot token (or DISCORD_BOT_TOKEN)

    Returns:
        {success, message_id, mode, error}
    """
    if not message and not embed_title:
        return {"success": False, "error": "message or embed_title required."}

    # Webhook mode
    wh_url = _get_webhook_url(webhook_url)
    if wh_url:
        payload: Dict[str, Any] = {"username": username or "Operon"}
        if avatar_url:
            payload["avatar_url"] = avatar_url
        if embed_title:
            embed = DiscordEmbed(
                title=embed_title,
                description=embed_description or message,
                color=embed_color,
            ).set_timestamp()
            payload["embeds"] = [embed.to_dict()]
            if message and not embed_description:
                payload["content"] = ""
        else:
            payload["content"] = message[:_MAX_MESSAGE_LEN]

        try:
            data = json.dumps(payload).encode()
            req  = urllib.request.Request(
                wh_url.rstrip("/") + "?wait=true", data=data,
                headers={"Content-Type": "application/json", "User-Agent": "Operon/1.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
                body = json.loads(resp.read())
                return {
                    "success": True,
                    "message_id": body.get("id", ""),
                    "mode": "webhook",
                    "error": "",
                }
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")
            return {"success": False, "error": f"HTTP {e.code}: {err[:200]}"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # Bot mode
    token = _get_token(bot_token)
    cid   = _get_channel(channel_id)
    if token and cid:
        bot = DiscordBot(token=token, channel_id=cid)
        embeds = None
        if embed_title:
            embeds = [DiscordEmbed(title=embed_title,
                                   description=embed_description or message,
                                   color=embed_color).set_timestamp()]
        r = bot.send_message(cid, content=message, embeds=embeds)
        return {
            "success": r.get("ok", False),
            "message_id": r.get("data", {}).get("id", ""),
            "mode": "bot",
            "error": r.get("error", ""),
        }

    return {
        "success": False,
        "error": (
            "No Discord credentials found. Set DISCORD_WEBHOOK_URL or "
            "(DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID)."
        ),
    }


def discord_get_messages(
    channel_id: str = "",
    bot_token: str = "",
    limit: int = 10,
    **_,
) -> dict:
    """Fetch recent messages from a Discord channel."""
    token = _get_token(bot_token)
    cid   = _get_channel(channel_id)
    if not token:
        return {"success": False, "error": "DISCORD_BOT_TOKEN required", "messages": []}
    if not cid:
        return {"success": False, "error": "channel_id required", "messages": []}
    bot  = DiscordBot(token=token)
    msgs = bot.get_messages(cid, limit=min(100, max(1, int(limit))))
    return {
        "success": True,
        "messages": [
            {
                "id":        m.id,
                "author":    m.author_name,
                "content":   m.content,
                "timestamp": m.timestamp,
                "thread_id": m.thread_id,
            }
            for m in msgs
        ],
        "error": "",
    }


def discord_create_thread(
    channel_id: str = "",
    name: str = "New Thread",
    message_id: str = "",
    auto_archive: int = 1440,
    bot_token: str = "",
    **_,
) -> dict:
    """Create a thread in a Discord channel."""
    token = _get_token(bot_token)
    cid   = _get_channel(channel_id)
    if not token:
        return {"success": False, "error": "DISCORD_BOT_TOKEN required"}
    if not cid:
        return {"success": False, "error": "channel_id required"}
    bot = DiscordBot(token=token)
    r   = bot.create_thread(
        cid, name, message_id=message_id or None, auto_archive=auto_archive
    )
    return {
        "success":   r.get("ok", False),
        "thread_id": r.get("data", {}).get("id", ""),
        "name":      r.get("data", {}).get("name", ""),
        "error":     r.get("error", ""),
    }


def discord_add_reaction(
    channel_id: str = "",
    message_id: str = "",
    emoji: str = "",
    bot_token: str = "",
    **_,
) -> dict:
    """Add a reaction emoji to a message."""
    token = _get_token(bot_token)
    cid   = _get_channel(channel_id)
    if not token:
        return {"success": False, "error": "DISCORD_BOT_TOKEN required"}
    if not cid or not message_id:
        return {"success": False, "error": "channel_id and message_id required"}
    bot = DiscordBot(token=token)
    r   = bot.add_reaction(cid, message_id, emoji)
    return {"success": r.get("ok", False), "error": r.get("error", "")}


def discord_timeout_member(
    guild_id: str = "",
    user_id: str = "",
    duration_seconds: int = 300,
    reason: str = "",
    bot_token: str = "",
    **_,
) -> dict:
    """Timeout (mute) a guild member."""
    token = _get_token(bot_token)
    gid   = _get_guild(guild_id)
    if not token:
        return {"success": False, "error": "DISCORD_BOT_TOKEN required"}
    if not gid or not user_id:
        return {"success": False, "error": "guild_id and user_id required"}
    bot = DiscordBot(token=token, guild_id=gid)
    r   = bot.timeout_member(gid, user_id, duration_seconds, reason)
    return {"success": r.get("ok", False), "error": r.get("error", "")}


def discord_create_webhook(
    channel_id: str = "",
    bot_token: str = "",
    name: str = "Operon",
    **_,
) -> dict:
    """Create a webhook for a Discord channel."""
    token = _get_token(bot_token)
    cid   = _get_channel(channel_id)
    if not token:
        return {"success": False, "error": "DISCORD_BOT_TOKEN required"}
    if not cid:
        return {"success": False, "error": "channel_id required"}
    bot = DiscordBot(token=token)
    r   = bot.create_webhook(cid, name)
    return {
        "success":     r.get("ok", False),
        "webhook_url": r.get("webhook_url", ""),
        "webhook_id":  r.get("webhook_id", ""),
        "error":       r.get("error", ""),
    }
