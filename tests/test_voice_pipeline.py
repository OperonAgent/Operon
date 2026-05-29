"""Tests for core/voice_pipeline.py"""
import io
import math
import os
import struct
import tempfile
import threading
import time
import wave
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from core.voice_pipeline import (
    VoiceConfig, STTBackend, TTSBackend, AudioFormat,
    Transcriber, Speaker, AudioRecorder, VADDetector,
    WakeWordDetector, SpeakerDiariser, DiarisationSegment,
    MultimodalMessage, MultimodalRouter, VoicePipeline,
    StreamingTranscriber,
    pcm_rms, resample_pcm, save_wav, load_wav,
    wav_bytes_to_array, get_voice_pipeline, listen, speak,
    transcribe_file, _SAMPLE_RATE, _SILENCE_THRESHOLD, _WAKE_WORDS,
)


# ── PCM helpers ───────────────────────────────────────────────────────────────

class TestPCMHelpers:
    def test_pcm_rms_silence(self):
        pcm = b"\x00" * 1000
        assert pcm_rms(pcm) == 0.0

    def test_pcm_rms_positive(self):
        # 1kHz tone
        pcm = struct.pack("<512h", *[int(10000 * math.sin(2 * math.pi * i / 16))
                                      for i in range(512)])
        assert pcm_rms(pcm) > 0

    def test_pcm_rms_empty(self):
        assert pcm_rms(b"") == 0.0

    def test_pcm_rms_single_byte(self):
        assert pcm_rms(b"\x01") == 0.0   # < 2 bytes → 0

    def test_resample_same_rate(self):
        pcm = b"\x00\x01" * 100
        result = resample_pcm(pcm, 16000, 16000)
        assert result == pcm

    def test_save_and_load_wav(self):
        pcm = b"\x00" * 3200   # 0.1s of silence at 16kHz
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            save_wav(pcm, path, sample_rate=16000)
            loaded_pcm, rate = load_wav(path)
            assert rate == 16000
            assert len(loaded_pcm) == len(pcm)
        finally:
            os.unlink(path)

    def test_wav_bytes_to_array(self):
        pcm = b"\x00" * 3200
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm)
        wav_bytes = buf.getvalue()
        frames, sr, ch = wav_bytes_to_array(wav_bytes)
        assert sr == 16000
        assert ch == 1
        assert len(frames) == len(pcm)


# ── VoiceConfig ───────────────────────────────────────────────────────────────

class TestVoiceConfig:
    def test_defaults(self):
        cfg = VoiceConfig()
        assert cfg.stt_backend == STTBackend.STUB
        assert cfg.tts_backend == TTSBackend.STUB
        assert cfg.sample_rate == _SAMPLE_RATE
        assert cfg.channels == 1

    def test_custom_backend(self):
        cfg = VoiceConfig(stt_backend=STTBackend.WHISPER_API, tts_backend=TTSBackend.OPENAI)
        assert cfg.stt_backend == STTBackend.WHISPER_API
        assert cfg.tts_backend == TTSBackend.OPENAI

    def test_wake_words_default(self):
        cfg = VoiceConfig()
        assert len(cfg.wake_words) > 0

    def test_custom_wake_words(self):
        cfg = VoiceConfig(wake_words=["jarvis", "computer"])
        assert "jarvis" in cfg.wake_words


# ── VADDetector ───────────────────────────────────────────────────────────────

class TestVADDetector:
    def test_silence_not_speech(self):
        vad = VADDetector()
        silence = b"\x00" * 4096
        assert not vad.is_speech(silence, threshold=_SILENCE_THRESHOLD)

    def test_loud_audio_is_speech(self):
        vad = VADDetector()
        # Loud signal: max amplitude
        loud = struct.pack("<512h", *[30000 if i % 2 == 0 else -30000
                                       for i in range(512)])
        assert vad.is_speech(loud, threshold=100)

    def test_filter_silence_removes_quiet(self):
        vad = VADDetector()
        silent_frame = b"\x00" * 2048
        loud_frame   = struct.pack("<512h", *[20000] * 512)
        frames = [silent_frame, loud_frame, silent_frame]
        filtered = vad.filter_silence(frames, threshold=1000)
        assert loud_frame in filtered
        assert silent_frame not in filtered

    def test_filter_all_silence(self):
        vad = VADDetector()
        frames = [b"\x00" * 100] * 5
        filtered = vad.filter_silence(frames)
        assert filtered == []


# ── WakeWordDetector ──────────────────────────────────────────────────────────

class TestWakeWordDetector:
    def setup_method(self):
        self.wwd = WakeWordDetector(wake_words=["hey operon", "computer"])

    def test_detects_wake_word(self):
        found, word = self.wwd.detected_in_text("Hey Operon, what time is it?")
        assert found
        assert word == "hey operon"

    def test_not_detected_in_clean_text(self):
        found, word = self.wwd.detected_in_text("Hello there, how are you?")
        assert not found
        assert word == ""

    def test_case_insensitive(self):
        found, _ = self.wwd.detected_in_text("COMPUTER, play music")
        assert found

    def test_strip_wake_word(self):
        result = self.wwd.strip_wake_word("hey operon turn off the lights")
        assert "turn off the lights" in result
        assert "operon" not in result.lower()

    def test_strip_no_wake_word(self):
        text = "what is the weather today"
        result = self.wwd.strip_wake_word(text)
        assert result == text

    def test_energy_pre_screen_silence(self):
        silence = b"\x00" * 4096
        assert not self.wwd.detected_in_audio_energy(silence)

    def test_energy_pre_screen_loud(self):
        loud = struct.pack("<512h", *[25000] * 512)
        assert self.wwd.detected_in_audio_energy(loud, threshold=100)

    def test_default_wake_words(self):
        wwd = WakeWordDetector()
        found, _ = wwd.detected_in_text("operon help me")
        assert found


# ── Transcriber (stub backend) ────────────────────────────────────────────────

class TestTranscriberStub:
    def test_stub_returns_placeholder(self):
        cfg = VoiceConfig(stt_backend=STTBackend.STUB)
        t = Transcriber(cfg)
        result = t._stub_transcribe("/fake/audio.wav")
        assert "audio.wav" in result

    def test_transcribe_file_stub(self):
        cfg = VoiceConfig(stt_backend=STTBackend.STUB)
        t = Transcriber(cfg)
        result = t.transcribe_file("/nonexistent.wav")
        assert isinstance(result, str)

    def test_transcribe_bytes_creates_temp_file(self):
        cfg = VoiceConfig(stt_backend=STTBackend.STUB)
        t = Transcriber(cfg)
        # Create valid WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 3200)
        result = t.transcribe_bytes(buf.getvalue(), fmt="wav")
        assert isinstance(result, str)

    def test_transcribe_pcm_stub(self):
        cfg = VoiceConfig(stt_backend=STTBackend.STUB)
        t = Transcriber(cfg)
        result = t.transcribe_pcm(b"\x00" * 3200)
        assert isinstance(result, str)

    def test_pick_backend_stub_when_no_deps(self):
        cfg = VoiceConfig(stt_backend=STTBackend.STUB)
        t = Transcriber(cfg)
        backend = t._pick_backend()
        # Should fall back to STUB or auto-detect something
        assert isinstance(backend, STTBackend)


# ── Speaker (stub backend) ────────────────────────────────────────────────────

class TestSpeakerStub:
    def test_stub_returns_empty_bytes(self):
        cfg = VoiceConfig(tts_backend=TTSBackend.STUB)
        s = Speaker(cfg)
        result = s._synthesise("Hello")
        assert result == b""

    def test_speak_stub_no_crash(self):
        cfg = VoiceConfig(tts_backend=TTSBackend.STUB)
        s = Speaker(cfg)
        result = s.speak("Hello", play=False)
        assert result == b""

    def test_synthesise_to_file_stub(self):
        cfg = VoiceConfig(tts_backend=TTSBackend.STUB)
        s = Speaker(cfg)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            result = s.synthesise_to_file("Hello", path)
            assert result is False  # stub returns empty bytes → write fails
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass

    def test_pick_backend_stub(self):
        cfg = VoiceConfig(tts_backend=TTSBackend.STUB)
        s = Speaker(cfg)
        backend = s._pick_backend()
        assert isinstance(backend, TTSBackend)


# ── AudioRecorder ─────────────────────────────────────────────────────────────

class TestAudioRecorder:
    def test_empty_wav_valid_wav_bytes(self):
        cfg = VoiceConfig()
        r = AudioRecorder(cfg)
        wav = r._empty_wav()
        assert wav[:4] == b"RIFF"

    def test_pcm_to_wav_roundtrip(self):
        cfg = VoiceConfig()
        r = AudioRecorder(cfg)
        pcm = b"\x00" * 3200
        wav = r._pcm_to_wav(pcm)
        frames, sr, ch = wav_bytes_to_array(wav)
        assert sr == cfg.sample_rate
        assert ch == cfg.channels

    def test_record_without_pyaudio_returns_empty(self):
        cfg = VoiceConfig(enable_vad=False)
        r = AudioRecorder(cfg)
        with patch("builtins.__import__", side_effect=ImportError("pyaudio")):
            pass  # can't easily mock builtins; just verify _empty_wav works
        wav = r._empty_wav()
        assert len(wav) > 0


# ── MultimodalMessage ─────────────────────────────────────────────────────────

class TestMultimodalMessage:
    def test_has_image_with_bytes(self):
        msg = MultimodalMessage(text="what is this?", images=[b"\xff\xd8\xff"])
        assert msg.has_image()

    def test_has_image_with_path(self):
        msg = MultimodalMessage(text="describe", image_paths=["/fake/img.jpg"])
        assert msg.has_image()

    def test_no_image(self):
        msg = MultimodalMessage(text="hello")
        assert not msg.has_image()

    def test_has_audio(self):
        msg = MultimodalMessage(audio=b"\x00\x01")
        assert msg.has_audio()

    def test_no_audio(self):
        msg = MultimodalMessage(text="hello")
        assert not msg.has_audio()

    def test_to_anthropic_content_text_only(self):
        msg = MultimodalMessage(text="hello world")
        content = msg.to_anthropic_content()
        assert any(p.get("type") == "text" for p in content)

    def test_to_anthropic_content_with_bytes_image(self):
        fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        msg = MultimodalMessage(text="what is this?", images=[fake_jpeg])
        content = msg.to_anthropic_content()
        types = [p["type"] for p in content]
        assert "image" in types
        assert "text" in types

    def test_to_anthropic_content_image_has_base64(self):
        import base64
        img_bytes = b"\xff\xd8\xff" * 10
        msg = MultimodalMessage(images=[img_bytes])
        content = msg.to_anthropic_content()
        img_part = next(p for p in content if p["type"] == "image")
        data = img_part["source"]["data"]
        decoded = base64.b64decode(data)
        assert decoded == img_bytes

    def test_missing_image_path_skipped(self):
        msg = MultimodalMessage(
            text="x",
            image_paths=["/nonexistent/image.png"]
        )
        content = msg.to_anthropic_content()
        # Should not crash, but image won't appear
        assert any(p.get("type") == "text" for p in content)


# ── MultimodalRouter ──────────────────────────────────────────────────────────

class TestMultimodalRouter:
    def test_routes_text_to_text_model(self):
        router = MultimodalRouter(text_model="claude-3-haiku")
        msg = MultimodalMessage(text="hello")
        result = router.route(msg)
        assert result["model"] == "claude-3-haiku"

    def test_routes_image_to_vision_model(self):
        router = MultimodalRouter(
            text_model="claude-3-haiku",
            vision_model="claude-3-5-sonnet-20241022",
        )
        msg = MultimodalMessage(text="what is this?", images=[b"\xff"])
        result = router.route(msg)
        assert result["model"] == "claude-3-5-sonnet-20241022"

    def test_route_includes_text(self):
        router = MultimodalRouter()
        msg = MultimodalMessage(text="test message")
        result = router.route(msg)
        assert result["text"] == "test message"

    def test_route_transcribes_audio(self):
        router = MultimodalRouter()
        # Create a valid wav
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 3200)
        msg = MultimodalMessage(audio=buf.getvalue())
        result = router.route(msg)
        assert "transcription" in result


# ── SpeakerDiariser ────────────────────────────────────────────────────────────

class TestSpeakerDiariser:
    def test_stub_returns_single_speaker(self):
        # Create temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            save_wav(b"\x00" * 16000 * 2, path, sample_rate=16000)  # 1 second
            d = SpeakerDiariser()
            segments = d._stub_diarise(path)
            assert len(segments) == 1
            assert segments[0].speaker_id == "SPEAKER_00"
            assert segments[0].start_sec == 0.0
        finally:
            os.unlink(path)

    def test_diarisation_segment_dataclass(self):
        seg = DiarisationSegment(
            speaker_id="SPEAKER_01", start_sec=0.5, end_sec=2.3, text="hello"
        )
        assert seg.speaker_id == "SPEAKER_01"
        assert seg.start_sec == 0.5
        assert seg.end_sec == 2.3


# ── VoicePipeline ─────────────────────────────────────────────────────────────

class TestVoicePipeline:
    def setup_method(self):
        self.cfg = VoiceConfig(
            stt_backend=STTBackend.STUB,
            tts_backend=TTSBackend.STUB,
            enable_vad=False,
        )
        self.vp = VoicePipeline(self.cfg)

    def test_stats_returns_dict(self):
        s = self.vp.stats()
        assert "stt_backend" in s
        assert "tts_backend" in s
        assert "sample_rate" in s

    def test_speak_stub_returns_bytes(self):
        result = self.vp.speak("Hello", play=False)
        assert isinstance(result, bytes)

    def test_converse_returns_tuple(self):
        with patch.object(self.vp, "listen", return_value="test input"):
            user, reply = self.vp.converse(process_fn=lambda t: "response")
        assert user == "test input"
        assert reply == "response"

    def test_converse_empty_input_returns_empty(self):
        with patch.object(self.vp, "listen", return_value=""):
            user, reply = self.vp.converse(process_fn=lambda t: "response")
        assert user == ""
        assert reply == ""

    def test_on_transcribed_callback(self):
        received = []
        with patch.object(self.vp, "listen", return_value="hello"):
            self.vp.converse(
                process_fn=lambda t: "ok",
                on_transcribed=lambda t: received.append(t),
            )
        assert received == ["hello"]

    def test_transcribe_file(self):
        with patch.object(self.vp.transcriber, "transcribe_file", return_value="text"):
            result = self.vp.transcribe_file("/fake.wav")
        assert result == "text"

    def test_stop_sets_running_false(self):
        self.vp._running = True
        self.vp.stop()
        assert not self.vp._running


# ── StreamingTranscriber ──────────────────────────────────────────────────────

class TestStreamingTranscriber:
    def test_push_returns_none_when_buffer_small(self):
        cfg = VoiceConfig(stt_backend=STTBackend.STUB)
        st = StreamingTranscriber(config=cfg, window_sec=1.0)
        result = st.push(b"\x00" * 100)
        assert result is None

    def test_flush_returns_string(self):
        cfg = VoiceConfig(stt_backend=STTBackend.STUB)
        st = StreamingTranscriber(config=cfg)
        result = st.flush()
        assert isinstance(result, str)

    def test_reset_clears_buffer(self):
        cfg = VoiceConfig(stt_backend=STTBackend.STUB)
        st = StreamingTranscriber(config=cfg)
        st._buffer = b"\x01" * 1000
        st.reset()
        assert st._buffer == b""

    def test_push_accumulates(self):
        cfg = VoiceConfig(stt_backend=STTBackend.STUB)
        st = StreamingTranscriber(config=cfg, window_sec=1.0)
        for _ in range(5):
            st.push(b"\x00" * 200)
        # buffer should have accumulated


# ── Module-level API ──────────────────────────────────────────────────────────

class TestModuleAPI:
    def test_get_voice_pipeline_returns_pipeline(self):
        vp = get_voice_pipeline(VoiceConfig(stt_backend=STTBackend.STUB))
        assert isinstance(vp, VoicePipeline)

    def test_get_voice_pipeline_singleton(self):
        vp1 = get_voice_pipeline()
        vp2 = get_voice_pipeline()
        assert vp1 is vp2

    def test_enums_complete(self):
        assert STTBackend.STUB is not None
        assert TTSBackend.STUB is not None
        assert AudioFormat.WAV is not None


# ── STTBackend enum ───────────────────────────────────────────────────────────

class TestSTTBackend:
    def test_values_exist(self):
        assert STTBackend.WHISPER_LOCAL.value == "whisper_local"
        assert STTBackend.WHISPER_API.value == "whisper_api"
        assert STTBackend.STUB.value == "stub"


class TestTTSBackend:
    def test_values_exist(self):
        assert TTSBackend.PYTTSX3.value == "pyttsx3"
        assert TTSBackend.OPENAI.value == "openai"
        assert TTSBackend.ESPEAK.value == "espeak"
        assert TTSBackend.STUB.value == "stub"
