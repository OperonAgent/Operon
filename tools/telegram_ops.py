"""
Operon Telegram Bot — Full bot with polling/webhook, inline keyboards, media.

Matches Hermes telegram_ops.py depth.

Provides:
  • TelegramBot      — core bot class: send/receive, polling, webhook
  • InlineKeyboard   — builder for inline keyboard markup
  • TelegramHandler  — message routing with command registration
  • Media support    — send/receive photos, documents, audio, video
  • Session state    — per-chat state tracking (FSM-style)
  • Broadcast        — send message to multiple chats
  • Rate limiting    — per-chat send throttling
  • Webhook server   — built-in aiohttp-based webhook server

Usage:
    from tools.telegram_ops import TelegramBot, send_message

    bot = TelegramBot(token=os.environ["TELEGRAM_BOT_TOKEN"])
    bot.send_message(chat_id=12345, text="Hello from Operon!")

    # Polling loop
    bot.start_polling(handler=my_handler_fn)

    # One-shot send (no bot instance needed)
    result = send_message(token="...", chat_id=12345, text="hi")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

log = logging.getLogger("operon.telegram_ops")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
_DEFAULT_TIMEOUT    = 30
_DEFAULT_POLL_LIMIT = 100     # max updates per long-poll request
_LONG_POLL_TIMEOUT  = 30      # seconds to wait in long-poll
_RETRY_DELAY_SEC    = 5.0
_MAX_MESSAGE_LEN    = 4_096
_MAX_CAPTION_LEN    = 1_024

# Parse modes
class ParseMode(str, Enum):
    MARKDOWN   = "Markdown"
    MARKDOWNV2 = "MarkdownV2"
    HTML       = "HTML"
    NONE       = ""

# Update types
class UpdateType(str, Enum):
    MESSAGE            = "message"
    EDITED_MESSAGE     = "edited_message"
    CALLBACK_QUERY     = "callback_query"
    INLINE_QUERY       = "inline_query"
    CHANNEL_POST       = "channel_post"
    DOCUMENT           = "document"
    PHOTO              = "photo"
    VOICE              = "voice"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TelegramUser:
    id:         int
    first_name: str
    last_name:  str = ""
    username:   str = ""
    is_bot:     bool = False

    @classmethod
    def from_dict(cls, d: Dict) -> "TelegramUser":
        return cls(
            id=d.get("id", 0),
            first_name=d.get("first_name", ""),
            last_name=d.get("last_name", ""),
            username=d.get("username", ""),
            is_bot=d.get("is_bot", False),
        )

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def mention(self) -> str:
        return f"@{self.username}" if self.username else self.full_name


@dataclass
class TelegramMessage:
    message_id:  int
    chat_id:     int
    chat_type:   str
    text:        str = ""
    from_user:   Optional[TelegramUser] = None
    date:        int = 0
    reply_to:    Optional[int] = None    # message_id of replied-to message
    document:    Optional[Dict] = None
    photo:       Optional[List[Dict]] = None
    voice:       Optional[Dict] = None
    video:       Optional[Dict] = None
    raw:         Dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: Dict) -> "TelegramMessage":
        chat = d.get("chat", {})
        from_ = d.get("from", {})
        reply = d.get("reply_to_message", {})
        return cls(
            message_id=d.get("message_id", 0),
            chat_id=chat.get("id", 0),
            chat_type=chat.get("type", "private"),
            text=d.get("text", d.get("caption", "")),
            from_user=TelegramUser.from_dict(from_) if from_ else None,
            date=d.get("date", 0),
            reply_to=reply.get("message_id") if reply else None,
            document=d.get("document"),
            photo=d.get("photo"),
            voice=d.get("voice"),
            video=d.get("video"),
            raw=d,
        )

    @property
    def is_command(self) -> bool:
        return self.text.startswith("/")

    @property
    def command(self) -> str:
        if not self.is_command:
            return ""
        return self.text.split()[0].lstrip("/").split("@")[0]

    @property
    def command_args(self) -> List[str]:
        parts = self.text.split()
        return parts[1:] if len(parts) > 1 else []


@dataclass
class CallbackQuery:
    id:          str
    from_user:   TelegramUser
    message:     Optional[TelegramMessage]
    data:        str = ""
    inline_msg_id: str = ""

    @classmethod
    def from_dict(cls, d: Dict) -> "CallbackQuery":
        return cls(
            id=d.get("id", ""),
            from_user=TelegramUser.from_dict(d.get("from", {})),
            message=TelegramMessage.from_dict(d["message"]) if "message" in d else None,
            data=d.get("data", ""),
            inline_msg_id=d.get("inline_message_id", ""),
        )


@dataclass
class TelegramUpdate:
    update_id: int
    message:   Optional[TelegramMessage] = None
    edited:    Optional[TelegramMessage] = None
    callback:  Optional[CallbackQuery]   = None
    raw:       Dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: Dict) -> "TelegramUpdate":
        return cls(
            update_id=d.get("update_id", 0),
            message=TelegramMessage.from_dict(d["message"]) if "message" in d else None,
            edited=TelegramMessage.from_dict(d["edited_message"]) if "edited_message" in d else None,
            callback=CallbackQuery.from_dict(d["callback_query"]) if "callback_query" in d else None,
            raw=d,
        )


# ---------------------------------------------------------------------------
# Inline keyboard builder
# ---------------------------------------------------------------------------

class InlineKeyboard:
    """Fluent builder for Telegram InlineKeyboardMarkup."""

    def __init__(self) -> None:
        self._rows: List[List[Dict]] = []
        self._current_row: List[Dict] = []

    def button(self, text: str, callback_data: str = "", url: str = "") -> "InlineKeyboard":
        """Add a button to the current row."""
        btn: Dict[str, str] = {"text": text}
        if url:
            btn["url"] = url
        elif callback_data:
            btn["callback_data"] = callback_data[:64]
        else:
            btn["callback_data"] = text[:64]
        self._current_row.append(btn)
        return self

    def row(self) -> "InlineKeyboard":
        """Finish the current row and start a new one."""
        if self._current_row:
            self._rows.append(self._current_row)
            self._current_row = []
        return self

    def build(self) -> Dict:
        """Return the markup dict ready to pass to the API."""
        if self._current_row:
            self._rows.append(self._current_row)
            self._current_row = []
        return {"inline_keyboard": self._rows}

    @staticmethod
    def yes_no(yes_data: str = "yes", no_data: str = "no") -> Dict:
        """Convenience: two-button Yes/No keyboard."""
        return InlineKeyboard().button("✅ Yes", yes_data).button("❌ No", no_data).build()

    @staticmethod
    def confirm_cancel(confirm_data: str = "confirm", cancel_data: str = "cancel") -> Dict:
        return InlineKeyboard().button("✅ Confirm", confirm_data).button("🚫 Cancel", cancel_data).build()

    @staticmethod
    def from_list(buttons: List[Tuple[str, str]], row_width: int = 2) -> Dict:
        """Build keyboard from list of (label, data) tuples."""
        kb = InlineKeyboard()
        for i, (label, data) in enumerate(buttons):
            kb.button(label, data)
            if (i + 1) % row_width == 0:
                kb.row()
        return kb.build()


# ---------------------------------------------------------------------------
# TelegramBot
# ---------------------------------------------------------------------------

class TelegramBot:
    """
    Full Telegram bot — polling, webhook, messages, media, inline keyboards.
    All API calls use the `requests` library (sync) or `aiohttp` (async).
    """

    def __init__(
        self,
        token:             str = "",
        timeout:           int = _DEFAULT_TIMEOUT,
        parse_mode:        ParseMode = ParseMode.HTML,
        auto_retry:        bool = True,
        max_retries:       int = 3,
        rate_limit_sec:    float = 0.05,   # min seconds between sends per chat
    ) -> None:
        self._token        = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._timeout      = timeout
        self._parse_mode   = parse_mode
        self._auto_retry   = auto_retry
        self._max_retries  = max_retries
        self._rate_limit   = rate_limit_sec
        self._offset       = 0
        self._running      = False
        self._poll_thread: Optional[threading.Thread] = None
        self._handlers:    Dict[str, List[Callable]] = {}
        self._cmd_handlers: Dict[str, Callable] = {}
        self._chat_state:  Dict[int, Dict[str, Any]] = {}
        self._last_send:   Dict[int, float] = {}     # chat_id → last send time
        self._bot_info:    Optional[TelegramUser] = None

    # ── Bot info ──────────────────────────────────────────────────────────────

    def get_me(self) -> Optional[TelegramUser]:
        """Fetch bot info from Telegram."""
        result = self._call("getMe")
        if result and result.get("ok"):
            self._bot_info = TelegramUser.from_dict(result["result"])
            return self._bot_info
        return None

    # ── Sending messages ──────────────────────────────────────────────────────

    def send_message(
        self,
        chat_id:         Union[int, str],
        text:            str,
        parse_mode:      Optional[str] = None,
        reply_to:        Optional[int] = None,
        reply_markup:    Optional[Dict] = None,
        disable_preview: bool = False,
        disable_notify:  bool = False,
    ) -> Optional[Dict]:
        """Send a text message."""
        self._throttle(chat_id)
        # Auto-split long messages
        if len(text) > _MAX_MESSAGE_LEN:
            chunks = self._chunk_text(text, _MAX_MESSAGE_LEN)
            last_result = None
            for chunk in chunks:
                last_result = self.send_message(chat_id, chunk, parse_mode, reply_to,
                                                reply_markup, disable_preview, disable_notify)
                reply_to = None   # only first chunk is a reply
            return last_result

        params: Dict[str, Any] = {
            "chat_id":                  chat_id,
            "text":                     text,
            "parse_mode":               parse_mode or self._parse_mode.value,
            "disable_web_page_preview": disable_preview,
            "disable_notification":     disable_notify,
        }
        if reply_to:
            params["reply_to_message_id"] = reply_to
        if reply_markup:
            params["reply_markup"] = json.dumps(reply_markup)

        return self._call("sendMessage", params)

    def send_photo(
        self,
        chat_id:      Union[int, str],
        photo:        Union[str, bytes],
        caption:      str = "",
        reply_markup: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Send a photo (file_id, URL, or bytes)."""
        self._throttle(chat_id)
        params: Dict[str, Any] = {"chat_id": chat_id}
        if caption:
            params["caption"] = caption[:_MAX_CAPTION_LEN]
            params["parse_mode"] = self._parse_mode.value
        if reply_markup:
            params["reply_markup"] = json.dumps(reply_markup)

        if isinstance(photo, bytes):
            return self._call("sendPhoto", params, files={"photo": photo})
        else:
            params["photo"] = photo
            return self._call("sendPhoto", params)

    def send_document(
        self,
        chat_id:   Union[int, str],
        document:  Union[str, bytes, Path],
        filename:  str = "file.txt",
        caption:   str = "",
    ) -> Optional[Dict]:
        """Send a document (file path, URL, file_id, or bytes)."""
        self._throttle(chat_id)
        params: Dict[str, Any] = {"chat_id": chat_id}
        if caption:
            params["caption"] = caption[:_MAX_CAPTION_LEN]

        if isinstance(document, Path) or (isinstance(document, str) and Path(document).exists()):
            data = Path(document).read_bytes()
            return self._call("sendDocument", params,
                              files={"document": (filename, data)})
        elif isinstance(document, bytes):
            return self._call("sendDocument", params,
                              files={"document": (filename, document)})
        else:
            params["document"] = document
            return self._call("sendDocument", params)

    def send_audio(
        self,
        chat_id: Union[int, str],
        audio:   Union[str, bytes],
        title:   str = "",
        duration: int = 0,
    ) -> Optional[Dict]:
        """Send an audio file."""
        self._throttle(chat_id)
        params: Dict[str, Any] = {"chat_id": chat_id}
        if title:
            params["title"] = title
        if duration:
            params["duration"] = duration
        if isinstance(audio, bytes):
            return self._call("sendAudio", params, files={"audio": audio})
        params["audio"] = audio
        return self._call("sendAudio", params)

    def send_video(
        self,
        chat_id:  Union[int, str],
        video:    Union[str, bytes],
        caption:  str = "",
        duration: int = 0,
        width:    int = 0,
        height:   int = 0,
    ) -> Optional[Dict]:
        """Send a video file."""
        self._throttle(chat_id)
        params: Dict[str, Any] = {"chat_id": chat_id}
        if caption:
            params["caption"] = caption[:_MAX_CAPTION_LEN]
        if duration:
            params["duration"] = duration
        if width:
            params["width"] = width
        if height:
            params["height"] = height
        if isinstance(video, bytes):
            return self._call("sendVideo", params, files={"video": video})
        params["video"] = video
        return self._call("sendVideo", params)

    def edit_message(
        self,
        chat_id:    Union[int, str],
        message_id: int,
        text:       str,
        reply_markup: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Edit an existing message."""
        params: Dict[str, Any] = {
            "chat_id":    chat_id,
            "message_id": message_id,
            "text":       text[:_MAX_MESSAGE_LEN],
            "parse_mode": self._parse_mode.value,
        }
        if reply_markup:
            params["reply_markup"] = json.dumps(reply_markup)
        return self._call("editMessageText", params)

    def delete_message(
        self, chat_id: Union[int, str], message_id: int
    ) -> Optional[Dict]:
        """Delete a message."""
        return self._call("deleteMessage", {"chat_id": chat_id, "message_id": message_id})

    def answer_callback(
        self,
        callback_query_id: str,
        text:              str = "",
        show_alert:        bool = False,
    ) -> Optional[Dict]:
        """Respond to an inline keyboard button press."""
        return self._call("answerCallbackQuery", {
            "callback_query_id": callback_query_id,
            "text": text[:200] if text else "",
            "show_alert": show_alert,
        })

    def pin_message(
        self, chat_id: Union[int, str], message_id: int, notify: bool = False
    ) -> Optional[Dict]:
        return self._call("pinChatMessage", {
            "chat_id": chat_id,
            "message_id": message_id,
            "disable_notification": not notify,
        })

    def set_typing(self, chat_id: Union[int, str]) -> Optional[Dict]:
        """Send 'typing' action indicator."""
        return self._call("sendChatAction", {"chat_id": chat_id, "action": "typing"})

    # ── Broadcasts ────────────────────────────────────────────────────────────

    def broadcast(
        self,
        chat_ids: List[Union[int, str]],
        text:     str,
        delay:    float = 0.05,
    ) -> Dict[str, Any]:
        """Send a message to multiple chats. Returns per-chat results."""
        results: Dict[str, Any] = {"sent": 0, "failed": 0, "errors": {}}
        for cid in chat_ids:
            r = self.send_message(cid, text)
            if r and r.get("ok"):
                results["sent"] += 1
            else:
                results["failed"] += 1
                results["errors"][str(cid)] = r
            if delay > 0:
                time.sleep(delay)
        return results

    # ── Updates & polling ─────────────────────────────────────────────────────

    def get_updates(
        self,
        offset:  int = 0,
        limit:   int = _DEFAULT_POLL_LIMIT,
        timeout: int = 0,
    ) -> List[TelegramUpdate]:
        """Fetch pending updates. Returns list of TelegramUpdate objects."""
        result = self._call("getUpdates", {
            "offset": offset, "limit": limit, "timeout": timeout,
        })
        if not result or not result.get("ok"):
            return []
        updates = []
        for raw in result.get("result", []):
            try:
                updates.append(TelegramUpdate.from_dict(raw))
            except Exception as e:
                log.warning("failed to parse update: %s", e)
        return updates

    def start_polling(
        self,
        handler:   Optional[Callable[[TelegramUpdate], None]] = None,
        blocking:  bool = False,
        interval:  float = 0.5,
    ) -> None:
        """
        Start the polling loop.
        handler receives each TelegramUpdate.
        If blocking=True, run in the calling thread; else spawn a daemon thread.
        """
        self._running = True

        def _loop() -> None:
            log.info("Telegram polling loop started")
            while self._running:
                try:
                    updates = self.get_updates(
                        offset=self._offset,
                        timeout=_LONG_POLL_TIMEOUT,
                    )
                    for update in updates:
                        self._offset = update.update_id + 1
                        self._dispatch(update, handler)
                    if not updates and interval > 0:
                        time.sleep(interval)
                except Exception as e:
                    log.warning("polling error: %s", e)
                    time.sleep(_RETRY_DELAY_SEC)
            log.info("Telegram polling loop stopped")

        if blocking:
            _loop()
        else:
            self._poll_thread = threading.Thread(
                target=_loop, daemon=True, name="operon-telegram-poll"
            )
            self._poll_thread.start()

    def stop_polling(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5)

    # ── Webhook ───────────────────────────────────────────────────────────────

    def set_webhook(
        self,
        url:             str,
        certificate:     Optional[bytes] = None,
        max_connections: int = 40,
        allowed_updates: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """Register a webhook URL with Telegram."""
        params: Dict[str, Any] = {
            "url": url,
            "max_connections": max_connections,
        }
        if allowed_updates:
            params["allowed_updates"] = json.dumps(allowed_updates)
        files = {"certificate": certificate} if certificate else None
        return self._call("setWebhook", params, files=files)

    def delete_webhook(self) -> Optional[Dict]:
        """Remove the webhook (switch to polling mode)."""
        return self._call("deleteWebhook")

    def get_webhook_info(self) -> Optional[Dict]:
        """Return current webhook info."""
        return self._call("getWebhookInfo")

    def process_webhook_update(
        self,
        raw_update: Dict,
        handler:    Optional[Callable[[TelegramUpdate], None]] = None,
    ) -> None:
        """Process a single update received via webhook."""
        try:
            update = TelegramUpdate.from_dict(raw_update)
            self._dispatch(update, handler)
        except Exception as e:
            log.warning("webhook update processing failed: %s", e)

    async def start_webhook_server(
        self,
        host:    str = "0.0.0.0",
        port:    int = 8443,
        path:    str = "/webhook",
        handler: Optional[Callable] = None,
    ) -> None:
        """
        Start a minimal aiohttp webhook server.
        Requires: pip install aiohttp
        """
        try:
            import aiohttp.web as web
        except ImportError:
            log.error("aiohttp not installed — cannot start webhook server")
            return

        bot = self

        async def handle_update(request: web.Request) -> web.Response:
            try:
                data = await request.json()
                bot.process_webhook_update(data, handler)
            except Exception as e:
                log.warning("webhook request failed: %s", e)
            return web.Response(text="OK")

        app = web.Application()
        app.router.add_post(path, handle_update)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        log.info("Telegram webhook server listening on %s:%d%s", host, port, path)

    # ── Command & event handlers ──────────────────────────────────────────────

    def on_command(self, command: str) -> Callable:
        """Decorator to register a command handler: @bot.on_command('start')"""
        def decorator(fn: Callable) -> Callable:
            self._cmd_handlers[command.lstrip("/")] = fn
            return fn
        return decorator

    def on(self, event: str) -> Callable:
        """Decorator to register an event handler: @bot.on('message')"""
        def decorator(fn: Callable) -> Callable:
            self._handlers.setdefault(event, []).append(fn)
            return fn
        return decorator

    # ── Chat state (FSM) ──────────────────────────────────────────────────────

    def get_state(self, chat_id: int) -> Dict[str, Any]:
        return self._chat_state.setdefault(chat_id, {})

    def set_state(self, chat_id: int, key: str, value: Any) -> None:
        self._chat_state.setdefault(chat_id, {})[key] = value

    def clear_state(self, chat_id: int) -> None:
        self._chat_state.pop(chat_id, None)

    # ── File download ─────────────────────────────────────────────────────────

    def get_file(self, file_id: str) -> Optional[str]:
        """Get the download URL for a file_id."""
        result = self._call("getFile", {"file_id": file_id})
        if result and result.get("ok"):
            fp = result["result"].get("file_path", "")
            if fp:
                return f"https://api.telegram.org/file/bot{self._token}/{fp}"
        return None

    def download_file(self, file_id: str, dest: Union[str, Path]) -> bool:
        """Download a file to dest. Returns True on success."""
        url = self.get_file(file_id)
        if not url:
            return False
        try:
            import requests
            r = requests.get(url, timeout=self._timeout)
            if r.ok:
                Path(dest).write_bytes(r.content)
                return True
        except Exception as e:
            log.warning("download_file failed: %s", e)
        return False

    # ── Internals ─────────────────────────────────────────────────────────────

    def _call(
        self,
        method: str,
        params: Optional[Dict] = None,
        files:  Optional[Dict] = None,
        attempt: int = 0,
    ) -> Optional[Dict]:
        """Make a Telegram Bot API call."""
        if not self._token:
            log.error("Telegram bot token not set")
            return {"ok": False, "error": "token not set"}

        url = _TELEGRAM_API.format(token=self._token, method=method)
        try:
            import requests as req
            resp = req.post(
                url,
                data=params,
                files=files,
                timeout=self._timeout,
            )
            data = resp.json()
            if not data.get("ok"):
                err = data.get("description", "unknown error")
                retry_after = data.get("parameters", {}).get("retry_after", 0)
                log.warning("Telegram API error [%s]: %s", method, err)
                if self._auto_retry and attempt < self._max_retries:
                    wait = retry_after or (_RETRY_DELAY_SEC * (attempt + 1))
                    log.info("retrying %s in %.1fs", method, wait)
                    time.sleep(wait)
                    return self._call(method, params, files, attempt + 1)
            return data
        except Exception as e:
            log.warning("Telegram API call failed [%s]: %s", method, e)
            if self._auto_retry and attempt < self._max_retries:
                time.sleep(_RETRY_DELAY_SEC)
                return self._call(method, params, files, attempt + 1)
            return {"ok": False, "error": str(e)}

    def _dispatch(
        self,
        update: TelegramUpdate,
        external_handler: Optional[Callable],
    ) -> None:
        """Route an update to registered handlers."""
        try:
            if external_handler:
                external_handler(update)

            # Command handlers
            msg = update.message or update.edited
            if msg and msg.is_command:
                cmd_fn = self._cmd_handlers.get(msg.command)
                if cmd_fn:
                    try:
                        cmd_fn(msg)
                    except Exception as e:
                        log.warning("command handler %s raised: %s", msg.command, e)

            # Generic message handlers
            for fn in self._handlers.get("message", []):
                try:
                    fn(update)
                except Exception as e:
                    log.warning("message handler raised: %s", e)

            # Callback query handlers
            if update.callback:
                for fn in self._handlers.get("callback_query", []):
                    try:
                        fn(update.callback)
                    except Exception as e:
                        log.warning("callback handler raised: %s", e)

        except Exception as e:
            log.error("dispatch error: %s", e)

    def _throttle(self, chat_id: Union[int, str]) -> None:
        """Enforce per-chat rate limit."""
        if self._rate_limit <= 0:
            return
        key = str(chat_id)
        last = self._last_send.get(key, 0.0)
        elapsed = time.time() - last
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_send[key] = time.time()

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> List[str]:
        """Split text into chunks at newline boundaries."""
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        while text:
            if len(text) <= chunk_size:
                chunks.append(text)
                break
            # Try to split at newline
            split_at = text.rfind("\n", 0, chunk_size)
            if split_at <= 0:
                split_at = chunk_size
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        return chunks


# ---------------------------------------------------------------------------
# Convenience functions (no bot instance needed)
# ---------------------------------------------------------------------------

def send_message(
    token:     str,
    chat_id:   Union[int, str],
    text:      str,
    parse_mode: str = "HTML",
    reply_markup: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    One-shot send — creates a temporary bot, sends one message, returns result.
    """
    bot = TelegramBot(token=token)
    result = bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode=parse_mode,
        reply_markup=reply_markup,
    )
    return result or {"ok": False, "error": "send failed"}


def send_photo(
    token:   str,
    chat_id: Union[int, str],
    photo:   Union[str, bytes],
    caption: str = "",
) -> Dict[str, Any]:
    """One-shot photo send."""
    bot = TelegramBot(token=token)
    result = bot.send_photo(chat_id, photo, caption)
    return result or {"ok": False, "error": "send_photo failed"}


def send_file(
    token:    str,
    chat_id:  Union[int, str],
    filepath: Union[str, Path],
    caption:  str = "",
) -> Dict[str, Any]:
    """One-shot file send."""
    bot = TelegramBot(token=token)
    result = bot.send_document(chat_id, Path(filepath), caption=caption)
    return result or {"ok": False, "error": "send_file failed"}


# ---------------------------------------------------------------------------
# Operon tool interface
# ---------------------------------------------------------------------------

def telegram_send(
    text:         str,
    chat_id:      Optional[Union[int, str]] = None,
    token:        Optional[str] = None,
    parse_mode:   str = "HTML",
    **_: Any,
) -> Dict[str, Any]:
    """
    Tool-registry-compatible wrapper.
    Reads token from TELEGRAM_BOT_TOKEN env var if not provided.
    Reads chat_id from TELEGRAM_CHAT_ID env var if not provided.
    """
    tok  = token  or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    cid  = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not tok:
        return {"success": False, "error": "TELEGRAM_BOT_TOKEN not set"}
    if not cid:
        return {"success": False, "error": "chat_id not provided and TELEGRAM_CHAT_ID not set"}
    result = send_message(tok, cid, text, parse_mode)
    return {
        "success":    result.get("ok", False),
        "message_id": result.get("result", {}).get("message_id"),
        "error":      result.get("description", ""),
    }


def telegram_get_updates(
    token: Optional[str] = None,
    limit: int = 10,
    **_: Any,
) -> Dict[str, Any]:
    """
    Tool-registry-compatible: fetch recent updates.
    """
    tok = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not tok:
        return {"success": False, "error": "TELEGRAM_BOT_TOKEN not set"}
    bot     = TelegramBot(token=tok)
    updates = bot.get_updates(limit=limit, timeout=0)
    return {
        "success": True,
        "count":   len(updates),
        "updates": [
            {
                "update_id":  u.update_id,
                "chat_id":    u.message.chat_id if u.message else None,
                "text":       u.message.text    if u.message else "",
                "from":       u.message.from_user.full_name if (u.message and u.message.from_user) else "",
            }
            for u in updates
        ],
    }
