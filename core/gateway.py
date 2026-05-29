"""
Operon Messaging Gateway — Telegram bot backend.

Runs as a background daemon thread. Polls Telegram for new messages,
feeds each one through a fresh agent loop, and sends the reply back.

No library required — uses raw requests to the Telegram Bot API.

Usage (from main.py):
    gateway = TelegramGateway(token, agent_runner, config)
    gateway.start()   # non-blocking
    gateway.stop()
    gateway.status()  # dict

The agent_runner callable must accept (prompt: str) -> str | None and
return the agent's final response text.
"""

import json
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import requests

GATEWAY_LOG = Path.home() / ".operon" / "gateway.log"


class TelegramGateway:
    """
    Polls the Telegram Bot API (long-polling getUpdates) and feeds each
    incoming message through the agent loop.
    """

    POLL_TIMEOUT = 30     # long-poll seconds
    RETRY_DELAY  = 5      # seconds between error retries

    def __init__(
        self,
        token:        str,
        agent_runner: Callable[[str], Optional[str]],
        config=None,
        allowed_users: list = None,  # list of Telegram user_id ints; None = allow all
    ):
        self._token         = token
        self._agent_runner  = agent_runner
        self._config        = config
        self._allowed_users = set(allowed_users) if allowed_users else None

        self._running       = False
        self._thread: Optional[threading.Thread] = None
        self._offset        = 0
        self._msg_count     = 0
        self._err_count     = 0
        self._start_time: Optional[float] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running    = True
        self._start_time = time.time()
        self._thread     = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        self._log(f"Gateway started (token ends …{self._token[-6:]})")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.POLL_TIMEOUT + 2)
        self._log("Gateway stopped.")

    def status(self) -> dict:
        uptime = int(time.time() - self._start_time) if self._start_time else 0
        return {
            "running":       self._running,
            "messages_recv": self._msg_count,
            "errors":        self._err_count,
            "uptime_sec":    uptime,
        }

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while self._running:
            try:
                updates = self._get_updates()
                for upd in updates:
                    self._handle_update(upd)
            except requests.exceptions.ConnectionError:
                self._err_count += 1
                self._log("Connection error — retrying in 5s…", level="WARN")
                time.sleep(self.RETRY_DELAY)
            except requests.exceptions.Timeout:
                pass   # normal for long polling
            except Exception as e:
                self._err_count += 1
                self._log(f"Unexpected error: {e}", level="ERROR")
                time.sleep(self.RETRY_DELAY)

    def _get_updates(self) -> list:
        resp = requests.get(
            f"https://api.telegram.org/bot{self._token}/getUpdates",
            params={"offset": self._offset, "timeout": self.POLL_TIMEOUT},
            timeout=self.POLL_TIMEOUT + 5,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            return []
        updates = data.get("result", [])
        if updates:
            self._offset = updates[-1]["update_id"] + 1
        return updates

    def _handle_update(self, upd: dict) -> None:
        msg = upd.get("message") or upd.get("edited_message")
        if not msg:
            return

        chat_id   = msg["chat"]["id"]
        user_id   = msg.get("from", {}).get("id")
        user_name = msg.get("from", {}).get("first_name", "User")
        text      = msg.get("text", "").strip()

        if not text:
            return   # ignore non-text messages (stickers, photos, etc.)

        # ── Access control ────────────────────────────────────────────────────
        if self._allowed_users and user_id not in self._allowed_users:
            self._send(chat_id, "⛔ You are not authorised to use this Operon instance.")
            return

        self._msg_count += 1
        self._log(f"[{user_name}/{user_id}] {text[:80]}")

        # Show typing indicator
        try:
            requests.post(
                f"https://api.telegram.org/bot{self._token}/sendChatAction",
                json={"chat_id": chat_id, "action": "typing"},
                timeout=5,
            )
        except Exception:
            pass

        # ── Run agent loop ────────────────────────────────────────────────────
        try:
            response = self._agent_runner(text)
            if response:
                self._send(chat_id, response)
            else:
                self._send(chat_id, "_(no response)_")
        except Exception as e:
            self._send(chat_id, f"⚠ Agent error: {e}")
            self._log(f"Agent error for '{text[:40]}': {e}", level="ERROR")

    def _send(self, chat_id: int, text: str, parse_mode: str = "Markdown") -> None:
        max_len = 4000
        if not text:
            return
        chunks  = [text[i:i + max_len] for i in range(0, len(text), max_len)]
        for chunk in chunks:
            try:
                resp = requests.post(
                    f"https://api.telegram.org/bot{self._token}/sendMessage",
                    json={"chat_id": chat_id, "text": chunk, "parse_mode": parse_mode},
                    timeout=15,
                )
                # Retry without markdown on parse error
                if resp.status_code == 400:
                    requests.post(
                        f"https://api.telegram.org/bot{self._token}/sendMessage",
                        json={"chat_id": chat_id, "text": chunk},
                        timeout=15,
                    )
            except Exception as e:
                self._log(f"Send error: {e}", level="WARN")

    def _log(self, msg: str, level: str = "INFO") -> None:
        line = f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}"
        try:
            GATEWAY_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(GATEWAY_LOG, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass
        if level in ("WARN", "ERROR"):
            print(f"\n  [Gateway] {msg}", file=sys.stderr)
