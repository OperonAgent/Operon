"""tests/test_video_and_swe_lint.py — video gen tool, SWE static analysis, streaming voice."""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Video generation tool ──────────────────────────────────────────────────────

class TestVideoGen:
    def setup_method(self):
        for k in ("REPLICATE_API_TOKEN", "LUMA_API_KEY", "RUNWAY_API_KEY"):
            os.environ.pop(k, None)

    def test_no_provider_errors_cleanly(self):
        from tools.video_gen import video_generate
        r = video_generate("a cat playing piano")
        assert r["success"] is False
        assert "provider" in r["error"].lower()

    def test_empty_prompt(self):
        from tools.video_gen import video_generate
        r = video_generate("")
        assert r["success"] is False
        assert "prompt" in r["error"].lower()

    def test_image_to_video_needs_url(self):
        from tools.video_gen import video_from_image
        r = video_from_image("")
        assert r["success"] is False

    def test_list_generated_empty(self):
        from tools.video_gen import video_list_generated
        r = video_list_generated()
        assert r["success"] is True
        assert "videos" in r["output"]

    def test_registered_in_dispatch(self):
        from tools.registry import _DISPATCH, _TOOL_DEFINITIONS
        names = {t["name"] for t in _TOOL_DEFINITIONS}
        for t in ("video_generate", "video_from_image", "video_list_generated"):
            assert t in _DISPATCH
            assert t in names

    def test_in_vision_toolset(self):
        from tools.registry import TOOLSETS
        assert "video_generate" in TOOLSETS["vision"]

    def test_replicate_selected_when_token_present(self):
        os.environ["REPLICATE_API_TOKEN"] = "fake"
        from tools.video_gen import video_generate
        with patch("tools.video_gen._replicate_run",
                   return_value={"success": True, "output": {"video_url": "x"}, "error": None}) as m:
            r = video_generate("test", provider="auto")
        assert r["success"] is True
        m.assert_called_once()
        os.environ.pop("REPLICATE_API_TOKEN", None)


# ── SWE static analysis ────────────────────────────────────────────────────────

class TestStaticAnalyzer:
    @pytest.fixture
    def repo(self):
        d = Path(tempfile.mkdtemp())
        (d / "good.py").write_text("x = 1\nprint(x)\n")
        return d

    def test_clean_file(self, repo):
        from core.swe_agent import StaticAnalyzer
        sa = StaticAnalyzer(repo)
        r = sa.analyze(["good.py"])
        assert r.ran is True

    def test_detects_syntax_error(self, repo):
        from core.swe_agent import StaticAnalyzer
        (repo / "bad.py").write_text("def f(:\n  pass\n")
        sa = StaticAnalyzer(repo)
        r = sa.analyze(["bad.py"])
        # ruff/pyflakes/py_compile all catch this
        assert len(r.diagnostics) >= 1

    def test_summary_string(self, repo):
        from core.swe_agent import StaticAnalyzer
        r = StaticAnalyzer(repo).analyze(["good.py"])
        assert isinstance(r.summary(), str)

    def test_diagnostic_as_line(self):
        from core.swe_agent import Diagnostic
        d = Diagnostic(file="x.py", line=3, col=5, code="E999", message="bad")
        assert "x.py:3:5" in d.as_line()

    def test_analysis_result_error_count(self):
        from core.swe_agent import AnalysisResult, Diagnostic
        r = AnalysisResult(tool="t", ran=True)
        r.diagnostics.append(Diagnostic("a", 1, 1, "E", "e", "error"))
        r.diagnostics.append(Diagnostic("a", 2, 1, "W", "w", "warning"))
        assert r.error_count == 1
        assert r.clean is False

    def test_clean_property_true(self):
        from core.swe_agent import AnalysisResult
        assert AnalysisResult(tool="t", ran=True).clean is True

    def test_syntax_check_fallback_directly(self, repo):
        from core.swe_agent import StaticAnalyzer
        (repo / "syn.py").write_text("this is (not python\n")
        sa = StaticAnalyzer(repo)
        r = sa._run_syntax_check(["syn.py"])
        assert r.tool == "py_compile"
        assert any(d.code == "E999" for d in r.diagnostics)


# ── Streaming voice ─────────────────────────────────────────────────────────────

class TestStreamingVoice:
    def test_stream_listen_exists(self):
        from core.voice_pipeline import VoicePipeline
        assert hasattr(VoicePipeline, "stream_listen")

    def test_streaming_transcriber_windows(self):
        from core.voice_pipeline import StreamingTranscriber, VoiceConfig
        st = StreamingTranscriber(VoiceConfig(), window_sec=0.1)
        # push silence; should not crash and returns None or str
        out = st.push(b"\x00" * 100)
        assert out is None or isinstance(out, str)

    def test_streaming_flush_returns_str(self):
        from core.voice_pipeline import StreamingTranscriber, VoiceConfig
        st = StreamingTranscriber(VoiceConfig())
        assert isinstance(st.flush(), str)

    def test_stream_listen_no_pyaudio_fallback(self):
        from core.voice_pipeline import VoicePipeline
        vp = VoicePipeline()
        # recorder.record returns b"" without PyAudio; transcriber returns "" → empty
        with patch.object(vp.recorder, "record", return_value=b""):
            result = vp.stream_listen()
        assert isinstance(result, str)
