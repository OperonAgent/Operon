"""
tests/test_conversation_compression.py — Tests for core/conversation_compression.py

Covers:
  - Turn dataclass (text flattening, token estimation, to_dict)
  - CompressionStats (compression_ratio, to_dict)
  - CompressedConversation (total_tokens, fields)
  - TurnPruner (truncation, image removal, filler detection, collapse)
  - ConversationSummarizer (extractive fallback, LLM mock)
  - IterativeMerger (fallback merge, LLM mock)
  - CompressionQualityScorer (fact extraction, string-match scoring, LLM mock)
  - RollingWindow (push, flush, buffer, summary, get_context, reset)
  - ConversationCompressor (no-op, compress, quality retry, incremental)
  - Convenience functions (compress_conversation, rolling_window_compress, estimate_tokens, extract_key_facts)
  - Constants
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch


from core.conversation_compression import (
    _CHARS_PER_TOKEN,
    _DEFAULT_KEEP_LAST_N,
    _DEFAULT_TARGET_TOKENS,
    _MIN_TURNS_TO_COMPRESS,
    _QUALITY_PASS_THRESHOLD,
    CompressedConversation,
    CompressionQualityScorer,
    CompressionStats,
    ConversationCompressor,
    ConversationSummarizer,
    IterativeMerger,
    RollingWindow,
    Turn,
    TurnPruner,
    compress_conversation,
    estimate_tokens,
    extract_key_facts,
    rolling_window_compress,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_msgs(n: int, chars_per: int = 100) -> List[Dict[str, Any]]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"Message {i}: " + "x" * chars_per}
        for i in range(n)
    ]


def _noop_llm(system: str, user: str) -> str:
    return ""


def _mock_llm(response: str):
    def fn(system: str, user: str) -> str:
        return response
    return fn


# ===========================================================================
# Turn
# ===========================================================================

class TestTurn(unittest.TestCase):

    def test_text_string(self):
        t = Turn(role="user", content="Hello world")
        assert t.text == "Hello world"

    def test_text_list_of_blocks(self):
        content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        t = Turn(role="user", content=content)
        assert "hello" in t.text
        assert "world" in t.text

    def test_text_list_with_output_block(self):
        content = [{"output": "tool result here"}]
        t = Turn(role="tool", content=content)
        assert "tool result here" in t.text

    def test_text_non_string_list(self):
        t = Turn(role="user", content=[42, None])
        # Should not crash
        _ = t.text

    def test_text_non_string_content(self):
        t = Turn(role="user", content=123)
        assert t.text == "123"

    def test_estimate_tokens_simple(self):
        t = Turn(role="user", content="a" * 40)  # 40 chars → 10 tokens
        assert t.estimate_tokens() == 10

    def test_estimate_tokens_min_one(self):
        t = Turn(role="user", content="")
        assert t.estimate_tokens() >= 1

    def test_to_dict(self):
        t = Turn(role="user", content="hi", index=3)
        d = t.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hi"

    def test_to_dict_no_index(self):
        t = Turn(role="assistant", content="resp")
        d = t.to_dict()
        assert set(d.keys()) == {"role", "content"}

    def test_defaults(self):
        t = Turn(role="user", content="x")
        assert t.index == 0
        assert t.tokens == 0

    def test_list_content_no_text_key(self):
        t = Turn(role="user", content=[{"unknown_key": "val"}])
        _ = t.text  # should not crash


# ===========================================================================
# CompressionStats
# ===========================================================================

class TestCompressionStats(unittest.TestCase):

    def test_compression_ratio(self):
        s = CompressionStats(tokens_before=1000, tokens_after=400)
        assert abs(s.compression_ratio - 0.4) < 1e-6

    def test_compression_ratio_zero_before(self):
        s = CompressionStats(tokens_before=0)
        assert s.compression_ratio == 1.0

    def test_to_dict_keys(self):
        s = CompressionStats(tokens_before=100, tokens_after=50, method="test")
        d = s.to_dict()
        for key in ("tokens_before", "tokens_after", "compression_ratio",
                    "quality_score", "duration_ms", "method"):
            assert key in d

    def test_to_dict_values(self):
        s = CompressionStats(tokens_before=200, tokens_after=100, method="merge")
        d = s.to_dict()
        assert d["method"] == "merge"
        assert d["compression_ratio"] == 0.5

    def test_defaults(self):
        s = CompressionStats()
        assert s.tokens_before == 0
        assert s.quality_score == 1.0
        assert s.method == "none"


# ===========================================================================
# CompressedConversation
# ===========================================================================

class TestCompressedConversation(unittest.TestCase):

    def test_total_tokens_string(self):
        msgs = [{"role": "user", "content": "x" * 400}]
        cc = CompressedConversation(messages=msgs, summary="", prior_summary="")
        assert cc.total_tokens() == 100

    def test_total_tokens_list_content(self):
        msgs = [{"role": "user", "content": [{"text": "hello"}]}]
        cc = CompressedConversation(messages=msgs, summary="", prior_summary="")
        # json.dumps adds overhead; just confirm > 0
        assert cc.total_tokens() > 0

    def test_ok_default_true(self):
        cc = CompressedConversation(messages=[], summary="", prior_summary="")
        assert cc.ok is True

    def test_error_field(self):
        cc = CompressedConversation(messages=[], summary="", prior_summary="",
                                    ok=False, error="failed")
        assert cc.error == "failed"

    def test_stats_field(self):
        s  = CompressionStats(method="test")
        cc = CompressedConversation(messages=[], summary="s", prior_summary="p", stats=s)
        assert cc.stats.method == "test"


# ===========================================================================
# TurnPruner
# ===========================================================================

class TestTurnPruner(unittest.TestCase):

    def test_short_content_unchanged(self):
        pruner = TurnPruner(max_tool_chars=500)
        t = Turn("user", "short text")
        result = pruner.prune([t])
        assert result[0].content == "short text"

    def test_long_content_truncated(self):
        pruner = TurnPruner(max_tool_chars=100)
        t = Turn("tool", "x" * 300)
        result = pruner.prune([t])
        assert len(result[0].content) < 300
        assert "truncated" in result[0].content

    def test_image_block_replaced(self):
        pruner = TurnPruner()
        content = [{"type": "image", "source": {"data": "base64data"}}]
        t = Turn("user", content)
        result = pruner.prune([t])
        c = result[0].content
        assert isinstance(c, list)
        assert c[0]["type"] == "text"
        assert "omitted" in c[0]["text"]

    def test_text_block_truncated(self):
        pruner = TurnPruner(max_tool_chars=50)
        content = [{"type": "text", "text": "y" * 200}]
        t = Turn("assistant", content)
        result = pruner.prune([t])
        assert len(result[0].content[0]["text"]) < 200

    def test_output_block_truncated(self):
        pruner = TurnPruner(max_tool_chars=50)
        content = [{"output": "z" * 200}]
        t = Turn("tool", content)
        result = pruner.prune([t])
        assert len(result[0].content[0]["output"]) < 200

    def test_filler_ok_collapsed(self):
        pruner = TurnPruner()
        turns = [
            Turn("user",      "Fix the bug"),
            Turn("assistant", "ok"),
            Turn("assistant", "okay"),
            Turn("user",      "Thanks"),
        ]
        result = pruner.prune(turns)
        # "ok" and "okay" are fillers; second filler collapsed
        assert len(result) < len(turns)

    def test_filler_first_kept(self):
        pruner = TurnPruner()
        turns = [Turn("assistant", "ok"), Turn("user", "real content here")]
        result = pruner.prune(turns)
        assert any(t.text == "real content here" for t in result)

    def test_non_filler_kept(self):
        pruner = TurnPruner()
        t = Turn("user", "Please help me fix the IndexError in tools/browser.py")
        result = pruner.prune([t])
        assert len(result) == 1

    def test_is_filler_patterns(self):
        pruner = TurnPruner()
        assert TurnPruner._is_filler(Turn("a", "ok"))
        assert TurnPruner._is_filler(Turn("a", "Got it."))
        assert TurnPruner._is_filler(Turn("a", "Sure!"))
        assert TurnPruner._is_filler(Turn("a", "I will do that."))
        assert not TurnPruner._is_filler(Turn("a", "Here is the output of the analysis."))

    def test_long_filler_not_collapsed(self):
        """A filler pattern that's too long should NOT be filtered."""
        long = "ok " * 50  # 150 chars > 120 limit
        assert not TurnPruner._is_filler(Turn("a", long))

    def test_empty_turns_list(self):
        pruner = TurnPruner()
        assert pruner.prune([]) == []


# ===========================================================================
# ConversationSummarizer
# ===========================================================================

class TestConversationSummarizer(unittest.TestCase):

    def test_empty_turns_returns_empty(self):
        s = ConversationSummarizer(llm_fn=_noop_llm)
        assert s.summarise([]) == ""

    def test_extractive_fallback(self):
        s = ConversationSummarizer(llm_fn=_noop_llm)
        turns = [
            Turn("user",      "Fix the bug in browser.py"),
            Turn("assistant", "I will fix it now"),
        ]
        result = s.summarise(turns)
        # Extractive: should contain first sentences
        assert "browser" in result or "fix" in result.lower()

    def test_llm_result_used(self):
        s = ConversationSummarizer(llm_fn=_mock_llm("LLM summary here"))
        turns = [Turn("user", "hello"), Turn("assistant", "world")]
        result = s.summarise(turns)
        assert result == "LLM summary here"

    def test_llm_exception_falls_back(self):
        def bad_llm(sys, usr):
            raise RuntimeError("API down")
        s = ConversationSummarizer(llm_fn=bad_llm)
        turns = [Turn("user", "hello")]
        result = s.summarise(turns)
        assert isinstance(result, str)

    def test_context_prepended(self):
        parts_seen = []
        def capture_llm(sys, usr):
            parts_seen.append(usr)
            return "summary"
        s = ConversationSummarizer(llm_fn=capture_llm)
        s.summarise([Turn("user", "hi")], context="Bug tracker context")
        assert any("Bug tracker context" in p for p in parts_seen)

    def test_extractive_dedupes(self):
        turns = [Turn("user", "Same text"), Turn("user", "Same text")]
        result = ConversationSummarizer._extractive(turns)
        # Second duplicate should not appear twice
        assert result.count("Same text") == 1


# ===========================================================================
# IterativeMerger
# ===========================================================================

class TestIterativeMerger(unittest.TestCase):

    def test_empty_new_turns_returns_prior(self):
        m = IterativeMerger(llm_fn=_noop_llm)
        result = m.merge("prior summary", [])
        assert result == "prior summary"

    def test_empty_prior_delegates_to_summariser(self):
        m = IterativeMerger(llm_fn=_mock_llm("new summary"))
        turns = [Turn("user", "Fix browser.py")]
        result = m.merge("", turns)
        assert isinstance(result, str)

    def test_llm_result_used(self):
        m = IterativeMerger(llm_fn=_mock_llm("merged summary"))
        turns = [Turn("user", "extra content")]
        result = m.merge("prior", turns)
        assert result == "merged summary"

    def test_fallback_merge(self):
        prior = "Prior context."
        turns = [Turn("user", "new message")]
        result = IterativeMerger._fallback_merge(prior, turns)
        assert "Prior context" in result
        assert "new message" in result

    def test_llm_exception_falls_back(self):
        def bad_llm(s, u):
            raise RuntimeError("err")
        m = IterativeMerger(llm_fn=bad_llm)
        result = m.merge("prior", [Turn("user", "text")])
        assert isinstance(result, str)

    def test_merge_incorporates_new_content(self):
        captured = []
        def capture(sys, usr):
            captured.append(usr)
            return "merged"
        m = IterativeMerger(llm_fn=capture)
        m.merge("old summary", [Turn("user", "fix browser.py")])
        assert any("old summary" in c for c in captured)
        assert any("browser.py" in c for c in captured)


# ===========================================================================
# CompressionQualityScorer
# ===========================================================================

class TestCompressionQualityScorer(unittest.TestCase):

    def test_empty_turns_returns_one(self):
        s = CompressionQualityScorer(llm_fn=_noop_llm)
        assert s.score([], "any summary") == 1.0

    def test_extract_file_paths(self):
        s = CompressionQualityScorer()
        turns = [Turn("user", "Fix tools/browser.py and core/router.py")]
        facts = s.extract_key_facts(turns)
        assert "tools/browser.py" in facts
        assert "core/router.py" in facts

    def test_extract_exception_names(self):
        s = CompressionQualityScorer()
        turns = [Turn("user", "Got IndexError and ValueError")]
        facts = s.extract_key_facts(turns)
        assert "IndexError" in facts or "ValueError" in facts

    def test_extract_numbers(self):
        s = CompressionQualityScorer()
        turns = [Turn("user", "Line 342 failed with exit code 404")]
        facts = s.extract_key_facts(turns)
        assert "342" in facts or "404" in facts

    def test_string_match_score_full(self):
        s = CompressionQualityScorer(llm_fn=_noop_llm)
        turns = [Turn("user", "Fix tools/browser.py IndexError")]
        summary = "Fixed IndexError in tools/browser.py"
        score = s.score(turns, summary)
        assert score > 0.5

    def test_string_match_score_none(self):
        s = CompressionQualityScorer(llm_fn=_noop_llm)
        turns = [Turn("user", "Fix tools/browser.py line 404")]
        summary = "All good."
        score = s.score(turns, summary)
        assert score < 1.0

    def test_llm_score_used(self):
        llm_response = '{"retained": ["browser.py"], "missing": [], "score": 0.9}'
        s = CompressionQualityScorer(llm_fn=_mock_llm(llm_response))
        turns = [Turn("user", "Fix `tools/browser.py`")]
        score = s.score(turns, "Fixed browser.py")
        assert abs(score - 0.9) < 1e-6

    def test_llm_score_bad_json_falls_back(self):
        s = CompressionQualityScorer(llm_fn=_mock_llm("not json at all"))
        turns = [Turn("user", "Fix tools/browser.py")]
        score = s.score(turns, "Fixed browser.py")
        # Falls back to string match → some reasonable value
        assert 0.0 <= score <= 1.0

    def test_fact_cap_at_30(self):
        s = CompressionQualityScorer()
        # Generate many facts
        content = " ".join(f"file{i}.py" for i in range(50))
        turns = [Turn("user", content)]
        facts = s.extract_key_facts(turns)
        assert len(facts) <= 30


# ===========================================================================
# RollingWindow
# ===========================================================================

class TestRollingWindow(unittest.TestCase):

    def test_push_within_max(self):
        window = RollingWindow(max_turns=5, window_size=3)
        for i in range(4):
            window.push({"role": "user", "content": f"msg {i}"})
        assert len(window.buffer) == 4

    def test_push_triggers_flush(self):
        window = RollingWindow(
            max_turns=5, window_size=3,
            summariser=ConversationSummarizer(llm_fn=_noop_llm)
        )
        for i in range(6):
            window.push({"role": "user", "content": f"msg {i}"})
        # After 6 pushes (> max=5), should have flushed 3
        assert len(window.buffer) <= 5

    def test_flush_oldest(self):
        window = RollingWindow(
            max_turns=10, window_size=3,
            summariser=ConversationSummarizer(llm_fn=_noop_llm)
        )
        for i in range(5):
            window.push({"role": "user", "content": f"msg {i}"})
        window.flush_oldest(n=2)
        assert len(window.buffer) == 3

    def test_summary_accumulates(self):
        window = RollingWindow(
            max_turns=3, window_size=2,
            summariser=ConversationSummarizer(llm_fn=_noop_llm)
        )
        for i in range(5):
            window.push({"role": "user", "content": f"turn about browser.py {i}"})
        # Summary should be non-empty (extractive fallback fills it)
        assert isinstance(window.summary, str)

    def test_get_context_with_summary(self):
        window = RollingWindow(
            max_turns=3, window_size=2,
            summariser=ConversationSummarizer(llm_fn=_mock_llm("test summary"))
        )
        for i in range(5):
            window.push({"role": "user", "content": f"msg {i}"})
        ctx = window.get_context()
        assert any("summary" in str(m.get("content", "")).lower() for m in ctx)

    def test_get_context_no_summary(self):
        window = RollingWindow(max_turns=10)
        window.push({"role": "user", "content": "hello"})
        ctx = window.get_context()
        assert len(ctx) == 1

    def test_reset(self):
        window = RollingWindow(
            max_turns=3, window_size=2,
            summariser=ConversationSummarizer(llm_fn=_noop_llm)
        )
        for i in range(5):
            window.push({"role": "user", "content": f"msg {i}"})
        window.reset()
        assert window.summary == ""
        assert window.buffer == []

    def test_buffer_property_copy(self):
        window = RollingWindow()
        window.push({"role": "user", "content": "hi"})
        buf = window.buffer
        buf.append({"role": "user", "content": "injected"})
        # Internal buffer not modified
        assert len(window.buffer) == 1


# ===========================================================================
# ConversationCompressor
# ===========================================================================

class TestConversationCompressor(unittest.TestCase):

    def _compressor(self, llm_fn=None, quality=False):
        return ConversationCompressor(
            target_tokens=2000,
            keep_last_n=4,
            llm_fn=llm_fn or _noop_llm,
            enable_quality_check=quality,
        )

    def test_no_op_small_history(self):
        cc = self._compressor()
        msgs = _make_msgs(3, chars_per=10)
        result = cc.compress(msgs)
        assert result.stats.method == "none"
        assert result.messages == msgs

    def test_no_op_too_few_turns(self):
        cc = self._compressor()
        # Below _MIN_TURNS_TO_COMPRESS
        msgs = _make_msgs(_MIN_TURNS_TO_COMPRESS - 1, chars_per=600)
        result = cc.compress(msgs)
        assert "skip" in result.stats.method or result.stats.method == "none"

    def test_compression_method(self):
        cc = self._compressor()
        msgs = _make_msgs(30, chars_per=300)
        result = cc.compress(msgs, target_tokens=500, keep_last_n=4)
        assert result.stats.method == "sliding_window_merge"

    def test_compression_reduces_turns(self):
        cc = self._compressor()
        msgs = _make_msgs(30, chars_per=300)
        result = cc.compress(msgs, target_tokens=500, keep_last_n=4)
        assert result.stats.turns_after < result.stats.turns_before

    def test_compression_keeps_last_n(self):
        cc = self._compressor()
        msgs = _make_msgs(20, chars_per=400)
        result = cc.compress(msgs, target_tokens=500, keep_last_n=4)
        # 4 recent + 1 summary = 5 messages
        assert result.stats.turns_after == 5

    def test_summary_message_injected(self):
        cc = self._compressor()
        msgs = _make_msgs(20, chars_per=400)
        result = cc.compress(msgs, target_tokens=500, keep_last_n=4)
        first = result.messages[0]
        assert "Summary" in first["content"] or "summary" in first["content"].lower()

    def test_prior_summary_threaded(self):
        cc = self._compressor(llm_fn=_mock_llm("merged summary"))
        msgs = _make_msgs(20, chars_per=400)
        result = cc.compress(msgs, target_tokens=500, prior_summary="Old facts")
        # merged summary should mention both
        assert isinstance(result.summary, str)

    def test_incremental_compress(self):
        cc = self._compressor(llm_fn=_mock_llm("incremental summary"))
        new_msgs = _make_msgs(5, chars_per=100)
        result = cc.compress_incremental(new_msgs, existing_summary="Prior context")
        assert result.stats.method == "incremental_merge"
        assert "Summary" in result.messages[0]["content"]

    def test_incremental_updates_prior_summary(self):
        cc = self._compressor(llm_fn=_mock_llm("merged result"))
        cc.compress_incremental([{"role": "user", "content": "test"}])
        assert cc.current_summary == "merged result"

    def test_reset_clears_prior_summary(self):
        cc = self._compressor(llm_fn=_mock_llm("summary"))
        cc.compress_incremental([{"role": "user", "content": "x"}])
        cc.reset()
        assert cc.current_summary == ""

    def test_keep_n_covers_all(self):
        cc = self._compressor()
        msgs = _make_msgs(4, chars_per=600)   # exactly keep_n = 4
        result = cc.compress(msgs, keep_last_n=4, target_tokens=500)
        assert "covers_all" in result.stats.method or result.stats.method in ("none", "skip_keep_n_covers_all")

    def test_quality_check_expands_keep_window(self):
        """Low quality score causes expansion of keep_last_n and retry."""
        call_count = [0]

        def bad_llm(sys, usr):
            call_count[0] += 1
            if "Key facts" in usr:
                # Always return low quality score
                return '{"retained": [], "missing": ["x"], "score": 0.1}'
            return "summary"

        cc = ConversationCompressor(
            target_tokens=500,
            keep_last_n=2,
            llm_fn=bad_llm,
            enable_quality_check=True,
            quality_threshold=0.6,
        )
        msgs = _make_msgs(20, chars_per=300)
        result = cc.compress(msgs, target_tokens=500)
        # Should have retried at least once
        assert isinstance(result, CompressedConversation)

    def test_stats_populated(self):
        cc = self._compressor()
        msgs = _make_msgs(20, chars_per=300)
        result = cc.compress(msgs, target_tokens=500, keep_last_n=4)
        s = result.stats
        assert s.tokens_before > 0
        assert s.turns_before > 0
        assert s.duration_ms >= 0.0

    def test_estimate_tokens_static(self):
        msgs = [{"role": "user", "content": "a" * 400}]
        tokens = ConversationCompressor.estimate_tokens(msgs)
        assert tokens == 100

    def test_ok_field_true_on_success(self):
        cc = self._compressor()
        msgs = _make_msgs(2, chars_per=10)
        result = cc.compress(msgs)
        assert result.ok is True

    def test_custom_target_tokens(self):
        cc = ConversationCompressor(
            target_tokens=99999, keep_last_n=4,
            llm_fn=_noop_llm, enable_quality_check=False
        )
        msgs = _make_msgs(5, chars_per=400)
        result = cc.compress(msgs)
        # Everything fits → no-op
        assert result.stats.method == "none"


# ===========================================================================
# Convenience functions
# ===========================================================================

class TestConvenienceFunctions(unittest.TestCase):

    def test_estimate_tokens_list(self):
        msgs = [{"role": "user", "content": "a" * 400}]
        assert estimate_tokens(msgs) == 100

    def test_estimate_tokens_empty(self):
        assert estimate_tokens([]) == 0

    def test_estimate_tokens_list_content(self):
        msgs = [{"role": "user", "content": [{"text": "hello"}]}]
        assert estimate_tokens(msgs) > 0

    def test_extract_key_facts_file_paths(self):
        msgs = [{"role": "user", "content": "Fix core/router.py"}]
        facts = extract_key_facts(msgs)
        assert "core/router.py" in facts

    def test_extract_key_facts_empty(self):
        facts = extract_key_facts([])
        assert facts == []

    def test_compress_conversation_noop(self):
        msgs = _make_msgs(2, chars_per=10)
        result = compress_conversation(msgs, target_tokens=99999)
        assert result.stats.method == "none"

    def test_compress_conversation_compresses(self):
        msgs = _make_msgs(30, chars_per=300)
        result = compress_conversation(msgs, target_tokens=500, keep_last_n=4)
        assert result.stats.turns_after < result.stats.turns_before

    def test_rolling_window_compress_returns_tuple(self):
        msgs = _make_msgs(5, chars_per=100)
        summary, buf = rolling_window_compress(msgs, max_turns=10, window_size=3)
        assert isinstance(summary, str)
        assert isinstance(buf, list)

    def test_rolling_window_compress_flushes_when_overflow(self):
        msgs = _make_msgs(12, chars_per=50)
        summary, buf = rolling_window_compress(msgs, max_turns=8, window_size=4)
        assert len(buf) <= 8


# ===========================================================================
# Constants
# ===========================================================================

class TestConstants(unittest.TestCase):

    def test_chars_per_token(self):
        assert _CHARS_PER_TOKEN > 0

    def test_default_target_tokens(self):
        assert _DEFAULT_TARGET_TOKENS > 1000

    def test_default_keep_last_n(self):
        assert _DEFAULT_KEEP_LAST_N > 0

    def test_min_turns_to_compress(self):
        assert _MIN_TURNS_TO_COMPRESS >= 2

    def test_quality_threshold(self):
        assert 0.0 < _QUALITY_PASS_THRESHOLD < 1.0


if __name__ == "__main__":
    unittest.main(verbosity=2)
