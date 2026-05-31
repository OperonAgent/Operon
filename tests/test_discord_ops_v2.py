"""Tests for tools/discord_ops.py — production Discord bot."""
import json
from unittest import mock

import pytest

from tools.discord_ops import (
    DiscordEmbed, DiscordMessage, ActionRow, ComponentButton,
    DiscordBot, EmbedColor, ButtonStyle, ChannelType, MessageFlag,
    ApplicationCommandType, ComponentType,
    discord_send, discord_get_messages, discord_create_thread,
    discord_add_reaction, discord_timeout_member, discord_create_webhook,
    _chunk_message, _api_request, _get_token, _get_guild, _get_channel,
    _get_webhook_url, _MAX_MESSAGE_LEN,
)


# ── DiscordEmbed ──────────────────────────────────────────────────────────────

class TestDiscordEmbed:
    def test_basic_to_dict(self):
        e = DiscordEmbed(title="Test", description="Desc", color=EmbedColor.SUCCESS)
        d = e.to_dict()
        assert d["title"] == "Test"
        assert d["description"] == "Desc"
        assert d["color"] == EmbedColor.SUCCESS

    def test_set_description_truncates(self):
        e = DiscordEmbed()
        long_desc = "x" * 5000
        e.set_description(long_desc)
        assert len(e.description) <= 4096

    def test_add_field(self):
        e = DiscordEmbed()
        e.add_field("Name", "Value", inline=True)
        d = e.to_dict()
        assert len(d["fields"]) == 1
        assert d["fields"][0]["name"] == "Name"
        assert d["fields"][0]["inline"] is True

    def test_add_field_max_25(self):
        e = DiscordEmbed()
        for i in range(30):
            e.add_field(f"Field {i}", "val")
        assert len(e.fields) == 25

    def test_field_value_truncated(self):
        e = DiscordEmbed()
        e.add_field("Name", "x" * 2000)
        assert len(e.fields[0]["value"]) <= 1024

    def test_set_footer(self):
        e = DiscordEmbed()
        e.set_footer("Footer text", icon_url="https://example.com/icon.png")
        d = e.to_dict()
        assert d["footer"]["text"] == "Footer text"
        assert d["footer"]["icon_url"] == "https://example.com/icon.png"

    def test_set_author(self):
        e = DiscordEmbed()
        e.set_author("Alice", url="https://example.com", icon_url="https://example.com/a.png")
        d = e.to_dict()
        assert d["author"]["name"] == "Alice"
        assert d["author"]["url"] == "https://example.com"

    def test_set_thumbnail(self):
        e = DiscordEmbed()
        e.set_thumbnail("https://example.com/thumb.png")
        d = e.to_dict()
        assert d["thumbnail"]["url"] == "https://example.com/thumb.png"

    def test_set_image(self):
        e = DiscordEmbed()
        e.set_image("https://example.com/img.png")
        d = e.to_dict()
        assert d["image"]["url"] == "https://example.com/img.png"

    def test_set_timestamp_auto(self):
        e = DiscordEmbed()
        e.set_timestamp()
        d = e.to_dict()
        assert "timestamp" in d
        assert "T" in d["timestamp"]

    def test_factory_success(self):
        e = DiscordEmbed.success("All Good", "Tests pass")
        assert "✓" in e.title
        assert e.color == EmbedColor.SUCCESS

    def test_factory_error(self):
        e = DiscordEmbed.error("Failed", "Build broken")
        assert "✗" in e.title
        assert e.color == EmbedColor.DANGER

    def test_factory_warning(self):
        e = DiscordEmbed.warning("Slow", "High latency")
        assert "!" in e.title
        assert e.color == EmbedColor.WARNING

    def test_factory_info(self):
        e = DiscordEmbed.info("FYI", "Just info")
        assert "ℹ" in e.title
        assert e.color == EmbedColor.INFO

    def test_from_dict_roundtrip(self):
        original = DiscordEmbed(
            title="T", description="D", color=0x3498DB,
            url="https://example.com",
        )
        original.add_field("k", "v")
        original.set_footer("f")
        original.set_author("A")
        original.set_thumbnail("https://example.com/t.png")

        d = original.to_dict()
        restored = DiscordEmbed.from_dict(d)
        assert restored.title == "T"
        assert restored.description == "D"
        assert len(restored.fields) == 1
        assert restored.footer_text == "f"
        assert restored.author_name == "A"
        assert restored.thumbnail_url == "https://example.com/t.png"

    def test_empty_embed_to_dict(self):
        e = DiscordEmbed()
        d = e.to_dict()
        assert "color" in d
        assert "title" not in d   # not included if empty
        assert "description" not in d


# ── DiscordMessage ────────────────────────────────────────────────────────────

class TestDiscordMessage:
    def _raw(self, content="hello", **kwargs):
        return {
            "id": "123",
            "channel_id": "456",
            "author": {"id": "789", "username": "testuser"},
            "content": content,
            "timestamp": "2026-05-26T12:00:00.000Z",
            "embeds": [],
            "attachments": [],
            "reactions": [],
            "pinned": False,
            "tts": False,
            "mention_everyone": False,
            **kwargs,
        }

    def test_basic_parse(self):
        msg = DiscordMessage.from_dict(self._raw("hello"))
        assert msg.id == "123"
        assert msg.channel_id == "456"
        assert msg.author_name == "testuser"
        assert msg.content == "hello"

    def test_attachments_parsed(self):
        msg = DiscordMessage.from_dict(self._raw(
            attachments=[{"filename": "test.png", "url": "https://cdn/test.png"}]
        ))
        assert len(msg.attachments) == 1

    def test_embeds_parsed(self):
        msg = DiscordMessage.from_dict(self._raw(
            embeds=[{"title": "embed", "color": 0x3498DB}]
        ))
        assert len(msg.embeds) == 1
        assert isinstance(msg.embeds[0], DiscordEmbed)

    def test_pinned_flag(self):
        msg = DiscordMessage.from_dict(self._raw(pinned=True))
        assert msg.pinned is True

    def test_thread_id_from_thread_field(self):
        msg = DiscordMessage.from_dict(self._raw(
            thread={"id": "thread-123", "name": "Discussion"}
        ))
        assert msg.thread_id == "thread-123"

    def test_no_thread(self):
        msg = DiscordMessage.from_dict(self._raw())
        assert msg.thread_id is None


# ── ActionRow & Buttons ───────────────────────────────────────────────────────

class TestActionRow:
    def test_single_button(self):
        ar = ActionRow()
        ar.add_button("Click Me", custom_id="click_me")
        d = ar.to_dict()
        assert d["type"] == ComponentType.ACTION_ROW.value
        assert len(d["components"]) == 1
        assert d["components"][0]["label"] == "Click Me"

    def test_button_styles(self):
        ar = ActionRow()
        for style in [ButtonStyle.PRIMARY, ButtonStyle.SECONDARY,
                      ButtonStyle.SUCCESS, ButtonStyle.DANGER]:
            ar.add_button(f"Btn", custom_id="x", style=style)
        d = ar.to_dict()
        styles = {c["style"] for c in d["components"]}
        assert len(styles) == 4

    def test_link_button_has_url(self):
        ar = ActionRow()
        ar.add_button("Docs", style=ButtonStyle.LINK, url="https://example.com")
        d = ar.to_dict()
        btn = d["components"][0]
        assert "url" in btn
        assert "custom_id" not in btn

    def test_max_5_buttons(self):
        ar = ActionRow()
        for i in range(10):
            ar.add_button(f"Btn{i}", custom_id=f"btn{i}")
        assert len(ar.to_dict()["components"]) == 5

    def test_select_menu(self):
        ar = ActionRow()
        ar.add_select_menu("pick", "Choose one", [
            {"label": "Option A", "value": "a"},
            {"label": "Option B", "value": "b"},
        ])
        d = ar.to_dict()
        assert d["components"][0]["type"] == ComponentType.SELECT_MENU.value
        assert len(d["components"][0]["options"]) == 2

    def test_select_menu_max_25_options(self):
        ar = ActionRow()
        options = [{"label": f"opt{i}", "value": str(i)} for i in range(30)]
        ar.add_select_menu("x", "Pick", options)
        assert len(ar.to_dict()["components"][0]["options"]) == 25

    def test_button_emoji(self):
        # Discord supports attaching an emoji to a button; custom emoji are
        # referenced by text name, so the plumbing is exercised without any
        # emoji character living in the project.
        ar = ActionRow()
        ar.add_button("React", custom_id="react", emoji="thumbsup")
        d = ar.to_dict()
        assert "emoji" in d["components"][0]
        assert d["components"][0]["emoji"]["name"] == "thumbsup"

    def test_disabled_button(self):
        ar = ActionRow()
        ar.add_button("Disabled", custom_id="d", disabled=True)
        d = ar.to_dict()
        assert d["components"][0].get("disabled") is True

    def test_custom_id_defaults_to_lowercase_label(self):
        btn = ComponentButton(label="Hello World")
        d = btn.to_dict()
        # custom_id is lowercased label (spaces replaced with underscores)
        assert d["custom_id"] == "hello world" or d["custom_id"] == "hello_world"


# ── _chunk_message ────────────────────────────────────────────────────────────

class TestChunkMessage:
    def test_short_not_split(self):
        assert _chunk_message("hello") == ["hello"]

    def test_long_split_at_limit(self):
        text = "a" * 4100
        chunks = _chunk_message(text)
        assert all(len(c) <= _MAX_MESSAGE_LEN for c in chunks)

    def test_prefers_newline_boundary(self):
        text = "line\n" * 500  # 2500 chars
        chunks = _chunk_message(text, 100)
        assert all(len(c) <= 100 for c in chunks)

    def test_chunk_preserves_all_content(self):
        text = "x" * 5000
        chunks = _chunk_message(text)
        assert "".join(chunks) == text

    def test_empty_string(self):
        assert _chunk_message("") == [""]


# ── EmbedColor & Enums ────────────────────────────────────────────────────────

class TestEnums:
    def test_embed_color_values(self):
        assert EmbedColor.SUCCESS == 0x57F287
        assert EmbedColor.DANGER  == 0xED4245
        assert EmbedColor.INFO    == 0x5865F2

    def test_channel_type_values(self):
        assert ChannelType.GUILD_TEXT.value == 0
        assert ChannelType.GUILD_VOICE.value == 2

    def test_button_style_values(self):
        assert ButtonStyle.PRIMARY.value == 1
        assert ButtonStyle.LINK.value == 5

    def test_message_flag_ephemeral(self):
        assert MessageFlag.EPHEMERAL == 64

    def test_component_type_action_row(self):
        assert ComponentType.ACTION_ROW.value == 1
        assert ComponentType.BUTTON.value == 2


# ── DiscordBot (mocked API) ───────────────────────────────────────────────────

class TestDiscordBotMocked:
    def _bot(self):
        return DiscordBot(token="fake-token", guild_id="guild-123", channel_id="ch-456")

    def _ok(self, data=None):
        return {"ok": True, "data": data or {}}

    def test_send_message_calls_api(self):
        bot = self._bot()
        with mock.patch.object(bot, "_req", return_value=self._ok({"id": "m1"})) as m:
            r = bot.send_message("ch-1", content="hello")
            m.assert_called_once()
            assert r["data"]["id"] == "m1"

    def test_send_embed(self):
        bot = self._bot()
        embed = DiscordEmbed.success("Done", "All good")
        with mock.patch.object(bot, "_req", return_value=self._ok()) as m:
            bot.send_embed("ch-1", embed)
            call_kwargs = m.call_args[0]
            payload = call_kwargs[2]
            assert "embeds" in payload
            assert len(payload["embeds"]) == 1

    def test_edit_message(self):
        bot = self._bot()
        with mock.patch.object(bot, "_req", return_value=self._ok()) as m:
            bot.edit_message("ch-1", "m1", content="updated")
            assert m.call_args[0][0] == "PATCH"
            assert "messages/m1" in m.call_args[0][1]

    def test_delete_message(self):
        bot = self._bot()
        with mock.patch.object(bot, "_req", return_value=self._ok()) as m:
            bot.delete_message("ch-1", "m1")
            assert m.call_args[0][0] == "DELETE"

    def test_add_reaction(self):
        bot = self._bot()
        with mock.patch.object(bot, "_req", return_value=self._ok()) as m:
            bot.add_reaction("ch-1", "m1", "")
            assert m.call_args[0][0] == "PUT"
            assert "reactions" in m.call_args[0][1]

    def test_create_thread(self):
        bot = self._bot()
        with mock.patch.object(bot, "_req",
                               return_value=self._ok({"id": "t1", "name": "My Thread"})) as m:
            r = bot.create_thread("ch-1", "My Thread", auto_archive=1440)
            payload = m.call_args[0][2]
            assert payload["name"] == "My Thread"
            assert payload["auto_archive_duration"] == 1440

    def test_create_thread_from_message(self):
        bot = self._bot()
        with mock.patch.object(bot, "_req", return_value=self._ok({"id": "t1"})) as m:
            bot.create_thread("ch-1", "Reply Thread", message_id="m999")
            endpoint = m.call_args[0][1]
            assert "m999" in endpoint

    def test_assign_role(self):
        bot = self._bot()
        with mock.patch.object(bot, "_req", return_value=self._ok()) as m:
            bot.assign_role("g1", "u1", "r1")
            assert m.call_args[0][0] == "PUT"
            assert "roles/r1" in m.call_args[0][1]

    def test_ban_member(self):
        bot = self._bot()
        with mock.patch.object(bot, "_req", return_value=self._ok()) as m:
            bot.ban_member("g1", "u1", reason="spamming", delete_message_seconds=86400)
            assert m.call_args[0][0] == "PUT"
            assert "bans/u1" in m.call_args[0][1]

    def test_timeout_member(self):
        bot = self._bot()
        with mock.patch.object(bot, "_req", return_value=self._ok()) as m:
            bot.timeout_member("g1", "u1", duration_seconds=300)
            payload = m.call_args[0][2]
            assert "communication_disabled_until" in payload

    def test_timeout_remove(self):
        bot = self._bot()
        with mock.patch.object(bot, "_req", return_value=self._ok()) as m:
            bot.timeout_member("g1", "u1", duration_seconds=0)
            payload = m.call_args[0][2]
            assert payload["communication_disabled_until"] is None

    def test_pin_message(self):
        bot = self._bot()
        with mock.patch.object(bot, "_req", return_value=self._ok()) as m:
            bot.pin_message("ch-1", "m1")
            assert m.call_args[0][0] == "PUT"
            assert "pins/m1" in m.call_args[0][1]

    def test_bulk_delete(self):
        bot = self._bot()
        with mock.patch.object(bot, "_req", return_value=self._ok()) as m:
            bot.bulk_delete_messages("ch-1", ["m1", "m2", "m3"])
            payload = m.call_args[0][2]
            assert "messages" in payload
            assert len(payload["messages"]) == 3

    def test_broadcast_counts(self):
        bot = self._bot()
        with mock.patch.object(bot, "send_message",
                               return_value={"ok": True, "data": {"id": "x"}}):
            result = bot.broadcast(["ch1", "ch2", "ch3"], content="hello", delay=0)
        assert result["sent"] == 3
        assert result["failed"] == 0

    def test_broadcast_failure_counted(self):
        bot = self._bot()
        def mock_send(cid, **kwargs):
            return {"ok": cid != "ch2", "error": "forbidden" if cid == "ch2" else ""}
        with mock.patch.object(bot, "send_message", side_effect=mock_send):
            result = bot.broadcast(["ch1", "ch2"], content="hi", delay=0)
        assert result["sent"] == 1
        assert result["failed"] == 1

    def test_create_channel(self):
        bot = self._bot()
        with mock.patch.object(bot, "_req", return_value=self._ok({"id": "c1"})) as m:
            bot.create_channel("g1", "new-channel", topic="Test channel")
            payload = m.call_args[0][2]
            assert payload["name"] == "new-channel"
            assert payload["topic"] == "Test channel"

    def test_list_members_empty_guild(self):
        bot = DiscordBot()   # no guild_id
        result = bot.list_members()
        assert result == []

    def test_no_channel_guard(self):
        bot = DiscordBot()
        r = bot.send_message("", content="test")
        assert not r.get("ok")

    def test_event_handler_registration(self):
        bot = self._bot()
        received = []

        @bot.on("message_create")
        def handler(data):
            received.append(data)

        bot._dispatch("message_create", {"content": "hello"})
        assert len(received) == 1

    def test_command_handler_registration(self):
        bot = self._bot()
        calls = []

        @bot.on_command("ping")
        def handle_ping(data):
            calls.append(data)

        assert "ping" in bot._cmd_handlers

    def test_stats_structure(self):
        bot = self._bot()
        with mock.patch.object(bot, "get_guild",
                               return_value={"ok": False, "error": "not found"}):
            s = bot.stats()
        assert "token_set" in s
        assert s["token_set"] is True


# ── Module-level tool functions ───────────────────────────────────────────────

class TestModuleLevelFunctions:
    def test_discord_send_no_creds(self):
        import os
        os.environ.pop("DISCORD_WEBHOOK_URL", None)
        os.environ.pop("DISCORD_BOT_TOKEN", None)
        r = discord_send("hello", webhook_url="", bot_token="")
        assert not r["success"]

    def test_discord_send_requires_content(self):
        r = discord_send(message="", embed_title="", webhook_url="", bot_token="")
        assert not r["success"]
        assert "required" in r["error"]

    def test_discord_get_messages_no_token(self):
        import os
        os.environ.pop("DISCORD_BOT_TOKEN", None)
        r = discord_get_messages(bot_token="", channel_id="123")
        assert not r["success"]
        assert "DISCORD_BOT_TOKEN" in r["error"]

    def test_discord_get_messages_no_channel(self):
        import os
        os.environ.pop("DISCORD_CHANNEL_ID", None)
        r = discord_get_messages(bot_token="fake-token", channel_id="")
        assert not r["success"]

    def test_discord_create_thread_no_token(self):
        r = discord_create_thread(bot_token="", channel_id="123")
        assert not r["success"]

    def test_discord_add_reaction_no_token(self):
        r = discord_add_reaction(bot_token="", channel_id="123", message_id="456")
        assert not r["success"]

    def test_discord_timeout_no_guild(self):
        r = discord_timeout_member(bot_token="fake", guild_id="", user_id="u1")
        assert not r["success"]

    def test_discord_create_webhook_no_token(self):
        r = discord_create_webhook(bot_token="", channel_id="123")
        assert not r["success"]

    def test_discord_send_via_webhook_mock(self):
        """Test webhook path using a mock HTTP response."""
        import urllib.request
        import io

        response_data = json.dumps({"id": "msg-123"}).encode()
        mock_response = mock.MagicMock()
        mock_response.read.return_value = response_data
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = mock.Mock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            r = discord_send(
                "Hello Discord!",
                webhook_url="https://discord.com/api/webhooks/fake/url",
            )
        assert r["success"] is True
        assert r["mode"] == "webhook"
        assert r["message_id"] == "msg-123"


# ── _api_request error handling ───────────────────────────────────────────────

class TestApiRequest:
    def test_returns_ok_on_success(self):
        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps({"id": "123"}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = mock.Mock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=mock_response):
            r = _api_request("GET", "channels/123", token="tok")
        assert r["ok"] is True
        assert r["data"]["id"] == "123"

    def test_returns_error_on_http_error(self):
        import urllib.error
        err = urllib.error.HTTPError(
            url="https://discord.com/api/v10/test",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=io.BytesIO(json.dumps({"message": "Missing Permissions", "code": 50013}).encode()),
        )
        with mock.patch("urllib.request.urlopen", side_effect=err):
            r = _api_request("POST", "channels/123/messages", token="tok", payload={})
        assert r["ok"] is False
        assert r["http"] == 403

    def test_retries_on_429(self):
        import urllib.error
        call_count = {"n": 0}
        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            raise urllib.error.HTTPError(
                url="x", code=429, msg="Too Many Requests", hdrs=None,
                fp=io.BytesIO(json.dumps({"retry_after": 0.01}).encode())
            )
        with mock.patch("urllib.request.urlopen", side_effect=side_effect):
            with mock.patch("time.sleep"):
                r = _api_request("GET", "test", token="tok")
        assert r["ok"] is False
        assert call_count["n"] == 3   # 3 retries


import io
