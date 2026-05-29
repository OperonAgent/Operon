"""Tests for core/context_compressor.py

All LLM summarisation calls are mocked — no model required.
"""
import time
import unittest.mock as mock
import pytest

from core.context_compressor import (
    ContextCompressor, CompressorConfig, SUMMARY_PREFIX,
    _estimate_content_chars, _estimate_tokens,
    _prune_old_tool_outputs, _build_summary_prompt,
    _find_tail_start, get_compressor, maybe_compress_messages,
    _PRUNED_PLACEHOLDER, _CHARS_PER_TOKEN, _IMAGE_CHAR_COST,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _msgs(n: int, alternating: bool = True) -> list:
    """Build n simple messages, alternating user/assistant."""
    msgs = []
    for i in range(n):
        role = "user" if (i % 2 == 0 or not alternating) else "assistant"
        msgs.append({"role": role, "content": f"message {i}"})
    return msgs


def _long_msgs(n: int, chars_each: int = 5_000) -> list:
    """Build n long messages (to trigger compression threshold)."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": "x" * chars_each})
    return msgs


# ── _estimate_content_chars ───────────────────────────────────────────────────

class TestEstimateContentChars:
    def test_string_content(self):
        assert _estimate_content_chars("hello world") == 11

    def test_empty_string(self):
        assert _estimate_content_chars("") == 0

    def test_none_like_content(self):
        # None is treated as falsy → "" → 0 chars  (not str(None)="None"=4)
        assert _estimate_content_chars(None) == 0

    def test_list_with_text_parts(self):
        parts = [{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]
        assert _estimate_content_chars(parts) == 11

    def test_image_part_costs_fixed_amount(self):
        parts = [{"type": "image", "url": "http://x.com/img.png"}]
        cost = _estimate_content_chars(parts)
        assert cost == _IMAGE_CHAR_COST

    def test_mixed_text_and_image(self):
        parts = [
            {"type": "text", "text": "description"},
            {"type": "image", "url": "http://x.com/img.png"},
        ]
        cost = _estimate_content_chars(parts)
        assert cost == len("description") + _IMAGE_CHAR_COST

    def test_unknown_part_type_uses_str_repr(self):
        parts = [{"type": "unknown_type", "data": "xyz"}]
        cost = _estimate_content_chars(parts)
        assert cost > 0


# ── _estimate_tokens ──────────────────────────────────────────────────────────

class TestEstimateTokens:
    def test_empty_messages_and_system(self):
        assert _estimate_tokens([], "") == 0

    def test_system_prompt_counted(self):
        tokens = _estimate_tokens([], "a" * 400)
        assert tokens == 100   # 400 chars / 4

    def test_messages_counted(self):
        msgs = [{"role": "user", "content": "a" * 400}]
        tokens = _estimate_tokens(msgs, "")
        # 400/4 + 4 overhead = 104
        assert tokens == 104

    def test_overhead_per_message(self):
        """Each message adds 4 tokens of overhead."""
        msgs = [{"role": "user", "content": ""}]   # 0 content chars
        assert _estimate_tokens(msgs, "") == 4

    def test_multiple_messages(self):
        msgs = [
            {"role": "user",      "content": "a" * 400},
            {"role": "assistant", "content": "b" * 400},
        ]
        tokens = _estimate_tokens(msgs, "")
        # (400+400)/4 + 2*4 = 200 + 8 = 208
        assert tokens == 208


# ── _prune_old_tool_outputs ───────────────────────────────────────────────────

class TestPruneOldToolOutputs:
    def test_tool_result_before_keep_index_pruned(self):
        msgs = [
            {"role": "user",      "content": "[TOOL_RESULT: shell_exec]\nhello world"},
            {"role": "assistant", "content": "done"},
            {"role": "user",      "content": "next message"},
        ]
        result = _prune_old_tool_outputs(msgs, keep_from=2)
        assert result[0]["content"] == _PRUNED_PLACEHOLDER

    def test_messages_at_or_after_keep_from_preserved(self):
        msgs = [
            {"role": "user", "content": "[TOOL_RESULT: x]\ndata"},
            {"role": "user", "content": "[TOOL_RESULT: y]\ndata"},
        ]
        result = _prune_old_tool_outputs(msgs, keep_from=1)
        assert result[0]["content"] == _PRUNED_PLACEHOLDER
        assert result[1]["content"] == "[TOOL_RESULT: y]\ndata"   # preserved

    def test_non_tool_messages_not_pruned(self):
        msgs = [
            {"role": "user",      "content": "regular user message"},
            {"role": "assistant", "content": "I'll help with that"},
        ]
        result = _prune_old_tool_outputs(msgs, keep_from=0)
        for r, o in zip(result, msgs):
            assert r["content"] == o["content"]

    def test_returns_same_length(self):
        msgs = _msgs(6)
        result = _prune_old_tool_outputs(msgs, keep_from=3)
        assert len(result) == len(msgs)


# ── _build_summary_prompt ─────────────────────────────────────────────────────

class TestBuildSummaryPrompt:
    def test_returns_string(self):
        msgs = _msgs(4)
        prompt = _build_summary_prompt(msgs)
        assert isinstance(prompt, str)

    def test_contains_structure_headers(self):
        prompt = _build_summary_prompt(_msgs(2))
        assert "Resolved" in prompt
        assert "Pending" in prompt
        assert "Active Task" in prompt

    def test_truncates_long_message_content(self):
        msgs = [{"role": "user", "content": "x" * 5000}]
        prompt = _build_summary_prompt(msgs)
        # Content > 2000 chars should be truncated
        assert "…" in prompt or "[…]" in prompt

    def test_includes_role_labels(self):
        msgs = [
            {"role": "user",      "content": "user input"},
            {"role": "assistant", "content": "assistant response"},
        ]
        prompt = _build_summary_prompt(msgs)
        assert "USER" in prompt
        assert "ASSISTANT" in prompt

    def test_handles_list_content(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "text part"},
        ]}]
        prompt = _build_summary_prompt(msgs)
        assert "text part" in prompt


# ── _find_tail_start ──────────────────────────────────────────────────────────

class TestFindTailStart:
    def test_small_conversation_returns_sensible_boundary(self):
        msgs = _msgs(8)
        idx = _find_tail_start(msgs, tail_turns=2, tail_budget_chars=0)
        assert 0 < idx <= len(msgs)

    def test_tail_turns_respected(self):
        msgs = _msgs(10, alternating=True)
        idx = _find_tail_start(msgs, tail_turns=2, tail_budget_chars=1_000_000)
        # Tail should include at least 2 pairs = 4 messages
        tail_len = len(msgs) - idx
        assert tail_len >= 4

    def test_empty_messages_returns_boundary(self):
        idx = _find_tail_start([], tail_turns=3, tail_budget_chars=0)
        assert idx >= 0

    def test_single_message_returns_boundary(self):
        msgs = [{"role": "user", "content": "hi"}]
        idx = _find_tail_start(msgs, tail_turns=3, tail_budget_chars=0)
        assert 0 <= idx <= 1


# ── ContextCompressor.maybe_compress ─────────────────────────────────────────

class TestMaybeCompress:
    def _compressor(self, threshold=500, tail_turns=2) -> ContextCompressor:
        cfg = CompressorConfig(
            threshold_tokens=threshold,
            tail_turns=tail_turns,
            prune_tool_output=False,
        )
        return ContextCompressor(cfg)

    def _mock_summarise(self, compressor, summary="## Summary\nDone."):
        compressor._summarise = mock.MagicMock(return_value=summary)

    def test_disabled_config_returns_unchanged(self):
        cfg = CompressorConfig(enabled=False)
        c = ContextCompressor(cfg)
        msgs = _long_msgs(10)
        new_msgs, did = c.maybe_compress(msgs)
        assert not did
        assert new_msgs is msgs

    def test_below_threshold_not_compressed(self):
        c = self._compressor(threshold=100_000)
        msgs = _msgs(6)
        new_msgs, did = c.maybe_compress(msgs)
        assert not did

    def test_force_triggers_compression_regardless_of_threshold(self):
        c = self._compressor(threshold=100_000)
        self._mock_summarise(c)
        msgs = _long_msgs(20)
        new_msgs, did = c.maybe_compress(msgs, force=True)
        assert did

    def test_above_threshold_triggers_compression(self):
        c = self._compressor(threshold=10)  # very low threshold
        self._mock_summarise(c)
        msgs = _long_msgs(20)
        new_msgs, did = c.maybe_compress(msgs)
        assert did

    def test_compression_reduces_message_count(self):
        c = self._compressor(threshold=10)
        self._mock_summarise(c)
        msgs = _long_msgs(20)
        new_msgs, did = c.maybe_compress(msgs)
        if did:
            assert len(new_msgs) < len(msgs)

    def test_summary_prefix_appears_in_result(self):
        c = self._compressor(threshold=10)
        self._mock_summarise(c)
        msgs = _long_msgs(20)
        new_msgs, did = c.maybe_compress(msgs)
        if did:
            combined = " ".join(m["content"] for m in new_msgs if isinstance(m.get("content"), str))
            assert "CONTEXT COMPACTION" in combined or SUMMARY_PREFIX[:20] in combined

    def test_never_crashes_on_summarise_failure(self):
        c = self._compressor(threshold=10)
        c._summarise = mock.MagicMock(side_effect=RuntimeError("LLM down"))
        msgs = _long_msgs(20)
        new_msgs, did = c.maybe_compress(msgs)
        assert not did
        assert new_msgs == msgs   # original returned unchanged

    def test_system_messages_preserved(self):
        c = self._compressor(threshold=10)
        self._mock_summarise(c)
        msgs = [
            {"role": "system",    "content": "You are Operon."},
            *_long_msgs(16),
        ]
        new_msgs, did = c.maybe_compress(msgs)
        if did:
            system_msgs = [m for m in new_msgs if m["role"] == "system"]
            assert len(system_msgs) >= 1
            assert system_msgs[0]["content"] == "You are Operon."

    def test_first_user_message_preserved(self):
        c = self._compressor(threshold=10)
        self._mock_summarise(c)
        msgs = [
            {"role": "user", "content": "ORIGINAL FIRST MESSAGE"},
            *_long_msgs(16),
        ]
        new_msgs, did = c.maybe_compress(msgs)
        if did:
            combined = " ".join(
                m.get("content", "") for m in new_msgs
                if isinstance(m.get("content"), str)
            )
            assert "ORIGINAL FIRST MESSAGE" in combined

    def test_tail_messages_always_preserved(self):
        c = self._compressor(threshold=10, tail_turns=2)
        self._mock_summarise(c)
        tail_content = "TAIL_MARKER_UNIQUE_XYZ"
        msgs = [
            *_long_msgs(14),
            {"role": "user",      "content": tail_content},
            {"role": "assistant", "content": "last response"},
        ]
        new_msgs, did = c.maybe_compress(msgs)
        if did:
            combined = " ".join(
                m.get("content", "") for m in new_msgs
                if isinstance(m.get("content"), str)
            )
            assert tail_content in combined

    def test_cooldown_prevents_retry_after_failure(self):
        c = self._compressor(threshold=10)
        c._summarise = mock.MagicMock(return_value="")   # empty = failure path
        msgs = _long_msgs(20)
        # First attempt fails
        c.maybe_compress(msgs, force=True)
        fail_time = c._last_fail_time
        # Second attempt should be blocked by cooldown
        _, did2 = c.maybe_compress(msgs, force=True)
        if fail_time > 0:
            assert not did2   # cooldown blocked it

    def test_too_short_conversation_not_compressed(self):
        c = self._compressor(threshold=10)
        msgs = _msgs(3)   # less than 4 conv messages
        new_msgs, did = c.maybe_compress(msgs, force=True)
        assert not did


# ── CompressorConfig ──────────────────────────────────────────────────────────

class TestCompressorConfig:
    def test_defaults(self):
        cfg = CompressorConfig()
        assert cfg.threshold_tokens  == 6_000
        assert cfg.tail_turns        == 6
        assert cfg.enabled           is True
        assert cfg.prune_tool_output is True

    def test_custom_values(self):
        cfg = CompressorConfig(threshold_tokens=1_000, tail_turns=3, enabled=False)
        assert cfg.threshold_tokens == 1_000
        assert cfg.tail_turns       == 3
        assert cfg.enabled          is False


# ── get_compressor / maybe_compress_messages ─────────────────────────────────

class TestModuleLevelAPI:
    def test_get_compressor_returns_instance(self):
        c = get_compressor()
        assert isinstance(c, ContextCompressor)

    def test_get_compressor_with_config_creates_new(self):
        cfg = CompressorConfig(threshold_tokens=999)
        c = get_compressor(cfg)
        assert c._config.threshold_tokens == 999

    def test_maybe_compress_messages_convenience(self):
        msgs = _msgs(4)
        new_msgs, did = maybe_compress_messages(msgs, threshold=100_000)
        assert isinstance(new_msgs, list)
        assert isinstance(did, bool)

    def test_maybe_compress_messages_force(self):
        with mock.patch.object(ContextCompressor, "_summarise", return_value="summary"):
            msgs = _long_msgs(20)
            new_msgs, did = maybe_compress_messages(msgs, threshold=10, force=True)
        assert isinstance(did, bool)
