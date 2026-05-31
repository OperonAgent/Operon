"""tests/test_slack_ops.py — Slack integration (mocked, no network/token)."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import slack_ops as s


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    for k in ("SLACK_WEBHOOK_URL", "SLACK_BOT_TOKEN", "SLACK_DEFAULT_CHANNEL"):
        monkeypatch.delenv(k, raising=False)
    yield


def _ok(data=None):
    return {"success": True, "data": data or {}}


# ── registry consistency ─────────────────────────────────────────────────────

class TestRegistryConsistency:
    def test_definitions_match_dispatch(self):
        defs = {d["name"] for d in s._TOOL_DEFINITIONS}
        assert defs == set(s._DISPATCH)

    def test_all_dispatch_callable(self):
        assert all(callable(fn) for fn in s._DISPATCH.values())

    def test_expected_depth_tools_present(self):
        for name in ("slack_get_thread", "slack_update_message",
                     "slack_schedule_message", "slack_pin_message",
                     "slack_set_topic", "slack_build_blocks",
                     "slack_delete_message"):
            assert name in s._DISPATCH


# ── credential gating (graceful, offline) ────────────────────────────────────

class TestCredentialGating:
    @pytest.mark.parametrize("fn,args", [
        (s.slack_get_thread, ("C1", "123")),
        (s.slack_update_message, ("C1", "123")),
        (s.slack_pin_message, ("C1", "123")),
        (s.slack_set_topic, ("C1",)),
        (s.slack_delete_message, ("C1", "123")),
    ])
    def test_requires_token(self, fn, args):
        out = fn(*args)
        assert out["success"] is False
        assert "SLACK_BOT_TOKEN" in out["error"]

    def test_schedule_requires_token(self):
        out = s.slack_schedule_message("C1", post_at=9999999999)
        assert out["success"] is False


# ── thread reading ───────────────────────────────────────────────────────────

class TestGetThread:
    def test_missing_args(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "x")
        assert s.slack_get_thread("", "")["success"] is False

    def test_parses_parent_and_replies(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "x")
        api = _ok({"messages": [
            {"ts": "1", "user": "U1", "text": "root"},
            {"ts": "2", "user": "U2", "text": "reply a"},
            {"ts": "3", "user": "U3", "text": "reply b"},
        ]})
        with patch.object(s, "_slack_api", return_value=api):
            out = s.slack_get_thread("C1", "1")
        assert out["success"] is True
        assert out["parent"]["text"] == "root"
        assert out["reply_count"] == 2
        assert out["replies"][1]["text"] == "reply b"


# ── editing / scheduling ──────────────────────────────────────────────────────

class TestUpdateAndSchedule:
    def test_update_message(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "x")
        with patch.object(s, "_slack_api",
                          return_value=_ok({"ts": "9", "channel": "C1"})) as m:
            out = s.slack_update_message("C1", "9", text="edited")
        assert out["success"] is True and out["ts"] == "9"
        assert m.call_args[0][0] == "chat.update"

    def test_update_requires_ts(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "x")
        assert s.slack_update_message("C1", "")["success"] is False

    def test_schedule_message(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "x")
        with patch.object(s, "_slack_api",
                          return_value=_ok({"scheduled_message_id": "Q1", "post_at": 123})) as m:
            out = s.slack_schedule_message("C1", text="later", post_at=123)
        assert out["success"] is True
        assert out["scheduled_message_id"] == "Q1"
        assert m.call_args[0][0] == "chat.scheduleMessage"


# ── pin / topic ───────────────────────────────────────────────────────────────

class TestPinAndTopic:
    def test_pin(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "x")
        with patch.object(s, "_slack_api", return_value=_ok()) as m:
            out = s.slack_pin_message("C1", "9")
        assert out["success"] is True and out["pinned"] is True
        assert m.call_args[0][0] == "pins.add"

    def test_unpin_uses_remove(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "x")
        with patch.object(s, "_slack_api", return_value=_ok()) as m:
            out = s.slack_pin_message("C1", "9", unpin=True)
        assert out["pinned"] is False
        assert m.call_args[0][0] == "pins.remove"

    def test_set_topic(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "x")
        api = _ok({"channel": {"topic": {"value": "new topic"}}})
        with patch.object(s, "_slack_api", return_value=api):
            out = s.slack_set_topic("C1", "new topic")
        assert out["success"] is True and out["topic"] == "new topic"


# ── Block Kit builder (pure, offline) ─────────────────────────────────────────

class TestBuildBlocks:
    def test_empty_fails(self):
        assert s.slack_build_blocks()["success"] is False

    def test_full_payload(self):
        out = s.slack_build_blocks(title="Deploy", body="*ok*",
                                   fields={"env": "prod", "rev": "abc"},
                                   context="by operon")
        assert out["success"] is True
        types = [b["type"] for b in out["blocks"]]
        assert types == ["header", "section", "section", "context"]

    def test_title_truncated(self):
        out = s.slack_build_blocks(title="x" * 300)
        assert len(out["blocks"][0]["text"]["text"]) <= 150

    def test_fields_capped_at_10(self):
        out = s.slack_build_blocks(fields={f"k{i}": i for i in range(20)})
        section = next(b for b in out["blocks"] if b.get("fields"))
        assert len(section["fields"]) == 10
