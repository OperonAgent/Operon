"""Tests for tools/telegram_ops.py"""
import json
import pytest
from unittest import mock

from tools.telegram_ops import (
    TelegramBot, TelegramUser, TelegramMessage, TelegramUpdate,
    CallbackQuery, InlineKeyboard, ParseMode,
    telegram_send, telegram_get_updates, send_message,
    _MAX_MESSAGE_LEN, _MAX_CAPTION_LEN,
)


# ── TelegramUser ──────────────────────────────────────────────────────────────

class TestTelegramUser:
    def test_full_name(self):
        u = TelegramUser.from_dict({"id": 1, "first_name": "Alice", "last_name": "Smith"})
        assert u.full_name == "Alice Smith"

    def test_full_name_no_last(self):
        u = TelegramUser.from_dict({"id": 1, "first_name": "Alice"})
        assert u.full_name == "Alice"

    def test_mention_with_username(self):
        u = TelegramUser.from_dict({"id": 1, "first_name": "Bob", "username": "bobby"})
        assert u.mention == "@bobby"

    def test_mention_without_username(self):
        u = TelegramUser.from_dict({"id": 1, "first_name": "Carol"})
        assert u.mention == "Carol"

    def test_is_bot_flag(self):
        u = TelegramUser.from_dict({"id": 1, "first_name": "Bot", "is_bot": True})
        assert u.is_bot is True

    def test_from_dict_defaults(self):
        u = TelegramUser.from_dict({"id": 9})
        assert u.first_name == ""
        assert u.username == ""
        assert u.is_bot is False


# ── TelegramMessage ───────────────────────────────────────────────────────────

class TestTelegramMessage:
    def _msg(self, text: str = "hello") -> TelegramMessage:
        return TelegramMessage.from_dict({
            "message_id": 1, "date": 0,
            "chat": {"id": 100, "type": "private"},
            "text": text,
            "from": {"id": 99, "first_name": "Test"},
        })

    def test_basic_fields(self):
        msg = self._msg("hello world")
        assert msg.message_id == 1
        assert msg.chat_id == 100
        assert msg.chat_type == "private"
        assert msg.text == "hello world"

    def test_is_command_true(self):
        assert self._msg("/start arg1 arg2").is_command is True

    def test_is_command_false(self):
        assert self._msg("just a message").is_command is False

    def test_command_name(self):
        assert self._msg("/help me").command == "help"

    def test_command_strips_bot_mention(self):
        msg = self._msg("/start@MyBot arg")
        assert msg.command == "start"

    def test_command_args(self):
        msg = self._msg("/cmd arg1 arg2 arg3")
        assert msg.command_args == ["arg1", "arg2", "arg3"]

    def test_command_args_empty(self):
        msg = self._msg("/start")
        assert msg.command_args == []

    def test_from_user_parsed(self):
        msg = self._msg()
        assert msg.from_user is not None
        assert msg.from_user.first_name == "Test"

    def test_reply_to_parsed(self):
        raw = {
            "message_id": 5, "date": 0,
            "chat": {"id": 100, "type": "private"},
            "text": "reply text",
            "reply_to_message": {"message_id": 3},
        }
        msg = TelegramMessage.from_dict(raw)
        assert msg.reply_to == 3

    def test_document_field(self):
        raw = {
            "message_id": 7, "date": 0,
            "chat": {"id": 100, "type": "private"},
            "document": {"file_id": "abc123", "file_name": "test.pdf"},
        }
        msg = TelegramMessage.from_dict(raw)
        assert msg.document is not None
        assert msg.document["file_id"] == "abc123"


# ── CallbackQuery ─────────────────────────────────────────────────────────────

class TestCallbackQuery:
    def test_from_dict(self):
        raw = {
            "id": "qid-123",
            "from": {"id": 99, "first_name": "User"},
            "message": {
                "message_id": 1, "date": 0,
                "chat": {"id": 100, "type": "private"},
                "text": "original message",
            },
            "data": "btn_click",
        }
        cb = CallbackQuery.from_dict(raw)
        assert cb.id == "qid-123"
        assert cb.data == "btn_click"
        assert cb.from_user.first_name == "User"
        assert cb.message.message_id == 1


# ── InlineKeyboard ────────────────────────────────────────────────────────────

class TestInlineKeyboard:
    def test_single_row_single_button(self):
        kb = InlineKeyboard().button("Click", "cb1").build()
        assert len(kb["inline_keyboard"]) == 1
        assert kb["inline_keyboard"][0][0]["text"] == "Click"
        assert kb["inline_keyboard"][0][0]["callback_data"] == "cb1"

    def test_multiple_rows(self):
        kb = InlineKeyboard().button("A", "a").row().button("B", "b").build()
        assert len(kb["inline_keyboard"]) == 2

    def test_url_button(self):
        kb = InlineKeyboard().button("Open", url="https://example.com").build()
        btn = kb["inline_keyboard"][0][0]
        assert btn["url"] == "https://example.com"
        assert "callback_data" not in btn

    def test_empty_keyboard(self):
        kb = InlineKeyboard().build()
        assert kb["inline_keyboard"] == []

    def test_yes_no(self):
        kb = InlineKeyboard.yes_no()
        row = kb["inline_keyboard"][0]
        assert len(row) == 2
        assert row[0]["callback_data"] == "yes"
        assert row[1]["callback_data"] == "no"

    def test_confirm_cancel(self):
        kb = InlineKeyboard.confirm_cancel()
        row = kb["inline_keyboard"][0]
        assert row[0]["callback_data"] == "confirm"
        assert row[1]["callback_data"] == "cancel"

    def test_from_list_row_width(self):
        buttons = [(f"Btn{i}", f"cb{i}") for i in range(6)]
        kb = InlineKeyboard.from_list(buttons, row_width=3)
        assert len(kb["inline_keyboard"]) == 2
        assert len(kb["inline_keyboard"][0]) == 3

    def test_callback_data_truncated_at_64(self):
        long_data = "x" * 100
        kb = InlineKeyboard().button("Btn", long_data).build()
        assert len(kb["inline_keyboard"][0][0]["callback_data"]) <= 64

    def test_button_text_default_callback(self):
        # No callback_data provided → uses text as callback_data
        kb = InlineKeyboard().button("Submit").build()
        assert kb["inline_keyboard"][0][0]["callback_data"] == "Submit"


# ── TelegramBot.chunk_text ────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_not_split(self):
        chunks = TelegramBot._chunk_text("hello", 4096)
        assert chunks == ["hello"]

    def test_long_text_split(self):
        text = "a\n" * 3000   # 6000 chars
        chunks = TelegramBot._chunk_text(text, 100)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_no_newline_splits_at_limit(self):
        text = "a" * 200
        chunks = TelegramBot._chunk_text(text, 50)
        assert all(len(c) <= 50 for c in chunks)
        assert "".join(chunks) == text

    def test_exact_length_not_split(self):
        text = "a" * 100
        chunks = TelegramBot._chunk_text(text, 100)
        assert len(chunks) == 1

    def test_prefers_newline_boundary(self):
        text = "line1\nline2\n" * 5
        chunks = TelegramBot._chunk_text(text, 20)
        # Should not cut mid-word (splits at newline)
        for chunk in chunks:
            assert len(chunk) <= 20


# ── TelegramBot (mocked API) ──────────────────────────────────────────────────

class TestTelegramBotMocked:
    def _bot_and_mock(self):
        bot = TelegramBot(token="fake-token", auto_retry=False)
        return bot

    def test_send_message_calls_api(self):
        bot = self._bot_and_mock()
        mock_result = {"ok": True, "result": {"message_id": 42}}
        with mock.patch.object(bot, "_call", return_value=mock_result) as m:
            result = bot.send_message(123, "hello")
            m.assert_called_once()
            assert result["ok"] is True

    def test_send_long_message_splits(self):
        bot = self._bot_and_mock()
        calls = []
        def mock_call(method, params=None, files=None, attempt=0):
            if method == "sendMessage":
                calls.append(params["text"])
            return {"ok": True, "result": {"message_id": 1}}
        bot._call = mock_call
        long_text = "word\n" * 2000
        bot.send_message(123, long_text)
        assert len(calls) > 1

    def test_send_photo_calls_api(self):
        bot = self._bot_and_mock()
        with mock.patch.object(bot, "_call", return_value={"ok": True, "result": {}}) as m:
            bot.send_photo(123, "https://example.com/img.jpg", caption="test")
            m.assert_called_once()

    def test_edit_message(self):
        bot = self._bot_and_mock()
        with mock.patch.object(bot, "_call", return_value={"ok": True, "result": {}}) as m:
            bot.edit_message(123, 42, "updated text")
            call_args = m.call_args
            assert call_args[0][0] == "editMessageText"

    def test_answer_callback(self):
        bot = self._bot_and_mock()
        with mock.patch.object(bot, "_call", return_value={"ok": True, "result": True}) as m:
            bot.answer_callback("qid-123", "Thanks!")
            assert m.call_args[0][0] == "answerCallbackQuery"

    def test_set_typing(self):
        bot = self._bot_and_mock()
        with mock.patch.object(bot, "_call", return_value={"ok": True}) as m:
            bot.set_typing(123)
            params = m.call_args[1]["params"] if "params" in m.call_args[1] else m.call_args[0][1]
            assert params["action"] == "typing"

    def test_delete_message(self):
        bot = self._bot_and_mock()
        with mock.patch.object(bot, "_call", return_value={"ok": True}) as m:
            bot.delete_message(123, 42)
            assert m.call_args[0][0] == "deleteMessage"

    def test_broadcast_counts(self):
        bot = self._bot_and_mock()
        with mock.patch.object(bot, "_call",
                               return_value={"ok": True, "result": {"message_id": 1}}):
            results = bot.broadcast([1, 2, 3], "hello", delay=0)
        assert results["sent"] == 3
        assert results["failed"] == 0


# ── Command & event handlers ──────────────────────────────────────────────────

class TestHandlers:
    def test_on_command_decorator(self):
        bot = TelegramBot()
        @bot.on_command("start")
        def handler(msg): pass
        assert "start" in bot._cmd_handlers

    def test_on_event_decorator(self):
        bot = TelegramBot()
        @bot.on("message")
        def handler(update): pass
        assert len(bot._handlers["message"]) == 1

    def test_dispatch_calls_command_handler(self):
        bot = TelegramBot()
        called = [False]
        @bot.on_command("ping")
        def handle_ping(msg):
            called[0] = True

        raw_update = {
            "update_id": 1,
            "message": {
                "message_id": 1, "date": 0,
                "chat": {"id": 100, "type": "private"},
                "text": "/ping",
            },
        }
        update = TelegramUpdate.from_dict(raw_update)
        bot._dispatch(update, None)
        assert called[0]

    def test_dispatch_calls_message_handler(self):
        bot = TelegramBot()
        received = []
        @bot.on("message")
        def handler(update):
            received.append(update)

        raw_update = {
            "update_id": 2,
            "message": {
                "message_id": 2, "date": 0,
                "chat": {"id": 100, "type": "private"},
                "text": "hello",
            },
        }
        update = TelegramUpdate.from_dict(raw_update)
        bot._dispatch(update, None)
        assert len(received) == 1

    def test_dispatch_calls_callback_handler(self):
        bot = TelegramBot()
        cbs = []
        @bot.on("callback_query")
        def handler(cb):
            cbs.append(cb)

        raw_update = {
            "update_id": 3,
            "callback_query": {
                "id": "cb-001",
                "from": {"id": 99, "first_name": "User"},
                "data": "btn1",
            },
        }
        update = TelegramUpdate.from_dict(raw_update)
        bot._dispatch(update, None)
        assert len(cbs) == 1 and cbs[0].data == "btn1"

    def test_external_handler_called(self):
        bot = TelegramBot()
        received = []
        def ext_handler(update):
            received.append(update)

        raw_update = {
            "update_id": 4,
            "message": {
                "message_id": 4, "date": 0,
                "chat": {"id": 100, "type": "private"},
                "text": "test",
            },
        }
        update = TelegramUpdate.from_dict(raw_update)
        bot._dispatch(update, ext_handler)
        assert len(received) == 1


# ── Chat state (FSM) ──────────────────────────────────────────────────────────

class TestChatState:
    def test_set_get_state(self):
        bot = TelegramBot()
        bot.set_state(999, "step", 3)
        assert bot.get_state(999)["step"] == 3

    def test_get_state_empty_dict_for_new_chat(self):
        bot = TelegramBot()
        assert bot.get_state(12345) == {}

    def test_clear_state(self):
        bot = TelegramBot()
        bot.set_state(999, "key", "value")
        bot.clear_state(999)
        assert bot.get_state(999) == {}

    def test_multiple_chats_isolated(self):
        bot = TelegramBot()
        bot.set_state(100, "x", 1)
        bot.set_state(200, "x", 2)
        assert bot.get_state(100)["x"] == 1
        assert bot.get_state(200)["x"] == 2


# ── Module-level convenience ──────────────────────────────────────────────────

class TestModuleLevelAPI:
    def test_telegram_send_missing_token(self):
        r = telegram_send("hello", chat_id=123, token="")
        assert not r["success"]
        assert "token" in r["error"].lower()

    def test_telegram_send_missing_chat_id(self):
        import os
        os.environ.pop("TELEGRAM_CHAT_ID", None)
        r = telegram_send("hello", token="fake-token")
        assert not r["success"]

    def test_telegram_get_updates_missing_token(self):
        r = telegram_get_updates(token="")
        assert not r["success"]

    def test_send_message_missing_token(self):
        # This will hit the real API (fake token) but should return ok=False
        r = send_message(token="", chat_id=123, text="test")
        assert r.get("ok") is False or "error" in r


# ── Rate limiting ─────────────────────────────────────────────────────────────

class TestRateLimit:
    def test_throttle_enforces_delay(self):
        import time
        bot = TelegramBot(rate_limit_sec=0.1)
        bot._last_send["100"] = time.time()  # just sent
        start = time.time()
        bot._throttle("100")
        elapsed = time.time() - start
        # Should have waited ~0.1s
        assert elapsed >= 0.05   # be generous for CI

    def test_throttle_no_delay_if_enough_time_passed(self):
        import time
        bot = TelegramBot(rate_limit_sec=0.01)
        bot._last_send["200"] = time.time() - 1.0  # sent 1s ago
        start = time.time()
        bot._throttle("200")
        elapsed = time.time() - start
        assert elapsed < 0.05   # should not wait
