"""tests/test_background_compact.py — non-blocking context compaction wiring."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main


@pytest.fixture(autouse=True)
def _reset_bg():
    main._bg_compressor = None
    main._bg_submit_len = 0
    yield
    main._bg_compressor = None
    main._bg_submit_len = 0


def _theme():
    t = MagicMock(); t.dim = lambda x: x
    return t


class TestThresholds:
    def test_below_soft_no_action(self):
        msgs = [{"role": "user", "content": "hi"}]
        with patch("main._estimate_tokens", return_value=100):
            out, applied = main._background_compact(msgs, "sys", soft=6000, hard=9000, theme=_theme())
        assert applied is False
        assert out == msgs
        # no background job should have been submitted
        assert not main._bg_compressor.is_running()

    def test_soft_threshold_submits_background_job(self):
        msgs = [{"role": "user", "content": "x" * 100} for _ in range(8)]
        with patch("main._estimate_tokens", return_value=7000):
            fake_bg = MagicMock()
            fake_bg.is_running.return_value = False
            fake_bg.get_result.return_value = None
            main._bg_compressor = fake_bg
            out, applied = main._background_compact(msgs, "sys", soft=6000, hard=9000, theme=_theme())
        # submitted a snapshot for later, did not block/apply this turn
        fake_bg.submit.assert_called_once()
        assert applied is False
        assert main._bg_submit_len == len(msgs)

    def test_hard_ceiling_compacts_synchronously(self):
        msgs = [{"role": "user", "content": "x"} for _ in range(10)]
        with patch("main._estimate_tokens", return_value=15000), \
             patch("main.maybe_compress_messages",
                   return_value=([{"role": "system", "content": "summary"}], True)) as mc:
            out, applied = main._background_compact(msgs, "sys", soft=6000, hard=9000, theme=_theme())
        assert applied is True
        mc.assert_called_once()
        assert out == [{"role": "system", "content": "summary"}]


class TestMergeReadyResult:
    def test_merges_compacted_prefix_with_appended(self):
        # Simulate: a previous job compacted the first 5 msgs into 2; since then
        # 3 new msgs were appended (total 8 now).
        compacted = [{"role": "system", "content": "summary"},
                     {"role": "user", "content": "recent"}]
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(8)]
        fake_bg = MagicMock()
        fake_bg.is_running.return_value = False
        fake_bg.get_result.return_value = (compacted, True)
        main._bg_compressor = fake_bg
        main._bg_submit_len = 5  # job was submitted when there were 5 msgs
        with patch("main._estimate_tokens", return_value=100):
            out, applied = main._background_compact(msgs, "sys", soft=6000, hard=9000, theme=_theme())
        assert applied is True
        # compacted (2) + messages appended after index 5 (3) = 5
        assert len(out) == len(compacted) + 3
        assert out[0]["content"] == "summary"
        assert out[-1] == msgs[-1]

    def test_no_merge_when_not_reduced(self):
        # Result didn't actually reduce -> don't apply
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        fake_bg = MagicMock()
        fake_bg.is_running.return_value = False
        fake_bg.get_result.return_value = (msgs, True)  # same length
        main._bg_compressor = fake_bg
        main._bg_submit_len = 5
        with patch("main._estimate_tokens", return_value=100):
            out, applied = main._background_compact(msgs, "sys", soft=6000, hard=9000, theme=_theme())
        assert applied is False

    def test_no_merge_while_running(self):
        fake_bg = MagicMock()
        fake_bg.is_running.return_value = True  # still working
        main._bg_compressor = fake_bg
        msgs = [{"role": "user", "content": "x"}]
        with patch("main._estimate_tokens", return_value=100):
            out, applied = main._background_compact(msgs, "sys", soft=6000, hard=9000, theme=_theme())
        assert applied is False
        fake_bg.get_result.assert_not_called()
