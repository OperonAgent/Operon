"""
core/voice_pipeline.py — Operon Voice & Multimodal Pipeline

Handles:
  - Speech-to-Text  (Whisper local / OpenAI Whisper API / stub fallback)
  - Text-to-Speech  (pyttsx3 local / OpenAI TTS / espeak / stub)
  - Audio streaming (chunked PCM ingestion, silence detection)
  - Speaker diarisation (stub — hook for pyannote.audio)
  - Multimodal routing (image + text → vision model)
  - Wake-word detection (simple energy-based + keyword scan)
  - Voice activity detection (VAD via webrtcvad or energy threshold)

All classes degrade gracefully when optional deps (openai, whisper, pyttsx3,
pyaudio, webrtcvad) are absent — they log a warning and use stubs/fallbacks.

Usage:
    from core.voice_pipeline import VoicePipeline, VoiceConfig

    vp = VoicePipeline()
    text = vp.listen()            # record microphone → transcribe → return text
    vp.speak("Hello, I'm Operon") # text → audio → play

    # Or individual components:
    from core.voice_pipeline import Transcriber, Speaker, AudioRecorder
    t = Transcriber()
    text = t.transcribe_file("audio.wav")
"""

from __future__ import annotations

import audioop
import base64
import hashlib
import io
import json
import logging
import math
import os
import struct
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

log = logging.getLogger("operon.voice_pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE   = 16_000       # Hz — Whisper optimal sample rate
_CHANNELS      = 1
_SAMPLE_WIDTH  = 2            # bytes per sample (16-bit PCM)
_CHUNK_SIZE    = 1_024        # frames per PyAudio read
_VAD_AGGRESSIVENESS = 2       # 0–3, higher = more aggressive silence detection
_SILENCE_THRESHOLD  = 500     # RMS energy threshold for simple VAD
_MAX_RECORD_SEC     = 60      # max recording before auto-stop
_SILENCE_TIMEOUT_SEC = 2.5    # seconds of silence before stopping

_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
_DEFAULT_WHISPER_MODEL = "base"

_WAKE_WORDS = ["hey operon", "operon", "hey agent", "computer"]

# ---------------------------------------------------------------------------
# Enums / config
# ---------------------------------------------------------------------------

class STTBackend(str, Enum):
    WHISPER_LOCAL  = "whisper_local"   # openai-whisper Python package
    WHISPER_API    = "whisper_api"     # OpenAI Whisper API
    VOSK           = "vosk"            # vosk offline
    DEEPGRAM       = "deepgram"        # real-time streaming WebSocket
    STUB           = "stub"            # returns placeholder text


class TTSBackend(str, Enum):
    PYTTSX3  = "pyttsx3"   # cross-platform local TTS
    OPENAI   = "openai"    # OpenAI TTS API
    ESPEAK   = "espeak"    # espeak/espeak-ng command-line
    BARK     = "bark"      # suno/bark (GPU)
    STUB     = "stub"      # no-op


class AudioFormat(str, Enum):
    WAV  = "wav"
    MP3  = "mp3"
    OGG  = "ogg"
    RAW  = "raw"   # raw PCM bytes


@dataclass
class VoiceConfig:
    """Tuning configuration for the voice pipeline."""
    stt_backend:        STTBackend  = STTBackend.STUB
    tts_backend:        TTSBackend  = TTSBackend.STUB
    whisper_model:      str         = _DEFAULT_WHISPER_MODEL
    sample_rate:        int         = _SAMPLE_RATE
    channels:           int         = _CHANNELS
    chunk_size:         int         = _CHUNK_SIZE
    silence_threshold:  int         = _SILENCE_THRESHOLD
    silence_timeout_sec: float      = _SILENCE_TIMEOUT_SEC
    max_record_sec:     float       = _MAX_RECORD_SEC
    wake_words:         List[str]   = field(default_factory=lambda: list(_WAKE_WORDS))
    language:           str         = "en"      # BCP-47 language code
    tts_voice:          str         = "alloy"   # OpenAI voice name
    tts_speed:          float       = 1.0
    openai_api_key:     str         = ""        # falls back to env OPENAI_API_KEY
    audio_device_index: Optional[int] = None   # None = default mic
    enable_vad:         bool        = True
    enable_wake_word:   bool        = False
    output_format:      AudioFormat = AudioFormat.WAV


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def pcm_rms(data: bytes) -> float:
    """Compute RMS energy of raw 16-bit PCM bytes."""
    if len(data) < 2:
        return 0.0
    n = len(data) // 2
    squares = sum(s * s for s in struct.unpack(f"<{n}h", data[:n * 2]))
    return math.sqrt(squares / n) if n else 0.0


def wav_bytes_to_array(wav_bytes: bytes) -> Tuple[bytes, int, int]:
    """
    Parse WAV bytes → (raw_pcm, sample_rate, channels).
    Returns raw PCM frames.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr    = wf.getframerate()
        ch    = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    return frames, sr, ch


def resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample mono 16-bit PCM from src_rate to dst_rate using audioop."""
    if src_rate == dst_rate:
        return pcm
    try:
        resampled, _ = audioop.ratecv(pcm, 2, 1, src_rate, dst_rate, None)
        return resampled
    except Exception:
        return pcm


def save_wav(pcm: bytes, path: str, sample_rate: int = _SAMPLE_RATE) -> None:
    """Write raw PCM to a WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def load_wav(path: str) -> Tuple[bytes, int]:
    """Read a WAV file → (raw_pcm, sample_rate)."""
    with wave.open(path, "rb") as wf:
        rate  = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    return frames, rate


# ---------------------------------------------------------------------------
# Voice Activity Detection
# ---------------------------------------------------------------------------

class VADDetector:
    """
    Voice Activity Detection.
    Prefers webrtcvad for high accuracy; falls back to simple RMS energy.
    """

    def __init__(self, aggressiveness: int = _VAD_AGGRESSIVENESS,
                 sample_rate: int = _SAMPLE_RATE) -> None:
        self._rate  = sample_rate
        self._agg   = aggressiveness
        self._vad   = None
        self._frame_ms = 30    # webrtcvad works in 10/20/30 ms frames

        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(aggressiveness)
            log.debug("VAD: using webrtcvad (aggressiveness=%d)", aggressiveness)
        except ImportError:
            log.debug("VAD: webrtcvad not available, using RMS energy fallback")

    def is_speech(self, frame: bytes, threshold: int = _SILENCE_THRESHOLD) -> bool:
        """Return True if the audio frame contains speech."""
        if self._vad is not None:
            # webrtcvad requires exactly 10/20/30 ms of 16-bit mono audio
            frame_len = int(self._rate * self._frame_ms / 1000) * 2
            if len(frame) < frame_len:
                return False
            try:
                return self._vad.is_speech(frame[:frame_len], self._rate)
            except Exception:
                pass
        # Fallback: RMS energy
        return pcm_rms(frame) > threshold

    def filter_silence(
        self, frames: List[bytes], threshold: int = _SILENCE_THRESHOLD
    ) -> List[bytes]:
        """Remove silent frames from a list."""
        return [f for f in frames if self.is_speech(f, threshold)]


# ---------------------------------------------------------------------------
# Speech-to-Text
# ---------------------------------------------------------------------------

class Transcriber:
    """
    Speech-to-text transcription with multiple backend support.
    All methods return plain text strings.
    """

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        self._cfg    = config or VoiceConfig()
        self._model  = None      # lazy-loaded Whisper model
        self._lock   = threading.Lock()

    # ── Public methods ──────────────────────────────────────────────────────

    def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file (WAV/MP3/OGG). Returns text."""
        backend = self._pick_backend()
        try:
            if backend == STTBackend.WHISPER_LOCAL:
                return self._whisper_local(path)
            elif backend == STTBackend.WHISPER_API:
                return self._whisper_api_file(path)
            elif backend == STTBackend.VOSK:
                return self._vosk_file(path)
            else:
                return self._stub_transcribe(path)
        except Exception as e:
            log.warning("Transcriber.transcribe_file error (%s): %s", backend, e)
            return ""

    def transcribe_bytes(self, audio: bytes, fmt: str = "wav") -> str:
        """Transcribe raw audio bytes. fmt: 'wav', 'mp3', 'ogg'."""
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
            f.write(audio)
            tmp = f.name
        try:
            return self.transcribe_file(tmp)
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = _SAMPLE_RATE) -> str:
        """Transcribe raw 16-bit mono PCM bytes."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            save_wav(pcm, tmp, sample_rate)
            return self.transcribe_file(tmp)
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    # ── Backends ────────────────────────────────────────────────────────────

    def _whisper_local(self, path: str) -> str:
        with self._lock:
            if self._model is None:
                import whisper  # type: ignore
                model_name = self._cfg.whisper_model
                log.info("Loading Whisper model: %s", model_name)
                self._model = whisper.load_model(model_name)
        result = self._model.transcribe(path, language=self._cfg.language or None)
        return result.get("text", "").strip()

    def _whisper_api_file(self, path: str) -> str:
        import urllib.request
        key = self._cfg.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY not set for Whisper API")

        with open(path, "rb") as f:
            audio_bytes = f.read()

        # Build multipart form manually (no requests dependency)
        boundary = "----OperonBoundary" + hashlib.md5(audio_bytes[:64]).hexdigest()[:8]
        fname    = os.path.basename(path)
        body     = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="model"\r\n\r\n'
            f"whisper-1\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="language"\r\n\r\n'
            f"{self._cfg.language}\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode() + audio_bytes + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data.get("text", "").strip()

    def _vosk_file(self, path: str) -> str:
        from vosk import Model, KaldiRecognizer  # type: ignore
        import json as _json

        model_path = os.environ.get("VOSK_MODEL_PATH", "model")
        m = Model(model_path)
        rec = KaldiRecognizer(m, self._cfg.sample_rate)

        pcm, sr = load_wav(path)
        if sr != self._cfg.sample_rate:
            pcm = resample_pcm(pcm, sr, self._cfg.sample_rate)

        words: List[str] = []
        chunk = 4000
        for i in range(0, len(pcm), chunk):
            if rec.AcceptWaveform(pcm[i: i + chunk]):
                res = _json.loads(rec.Result())
                words.append(res.get("text", ""))
        final = _json.loads(rec.FinalResult())
        words.append(final.get("text", ""))
        return " ".join(w for w in words if w).strip()

    @staticmethod
    def _stub_transcribe(path: str) -> str:
        log.debug("Transcriber stub: returning placeholder for %s", path)
        return f"[transcription of {os.path.basename(path)}]"

    def _pick_backend(self) -> STTBackend:
        if self._cfg.stt_backend != STTBackend.STUB:
            return self._cfg.stt_backend
        # Auto-detect: try whisper first
        try:
            import whisper  # noqa
            return STTBackend.WHISPER_LOCAL
        except ImportError:
            pass
        if os.environ.get("OPENAI_API_KEY"):
            return STTBackend.WHISPER_API
        return STTBackend.STUB


# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------

class Speaker:
    """
    Text-to-speech synthesis + playback.
    Returns audio bytes and optionally plays via system audio.
    """

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        self._cfg    = config or VoiceConfig()
        self._engine = None    # pyttsx3 engine, lazy-loaded
        self._lock   = threading.Lock()

    def speak(self, text: str, play: bool = True) -> bytes:
        """Synthesise text to speech. Returns WAV bytes. Plays if play=True."""
        audio = self._synthesise(text)
        if play and audio:
            self._play(audio)
        return audio

    def synthesise_to_file(self, text: str, path: str) -> bool:
        """Synthesise text and write to path. Returns True on success."""
        audio = self._synthesise(text)
        if audio:
            with open(path, "wb") as f:
                f.write(audio)
            return True
        return False

    def _synthesise(self, text: str) -> bytes:
        backend = self._pick_backend()
        try:
            if backend == TTSBackend.PYTTSX3:
                return self._pyttsx3_synth(text)
            elif backend == TTSBackend.OPENAI:
                return self._openai_tts(text)
            elif backend == TTSBackend.ESPEAK:
                return self._espeak_synth(text)
            elif backend == TTSBackend.BARK:
                return self._bark_synth(text)
            else:
                log.debug("Speaker stub: would speak: %s", text[:60])
                return b""
        except Exception as e:
            log.warning("Speaker._synthesise error (%s): %s", backend, e)
            return b""

    def _pyttsx3_synth(self, text: str) -> bytes:
        import pyttsx3  # type: ignore
        with self._lock:
            if self._engine is None:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", int(150 * self._cfg.tts_speed))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            self._engine.save_to_file(text, tmp)
            self._engine.runAndWait()
            if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                with open(tmp, "rb") as f:
                    return f.read()
            return b""
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def _openai_tts(self, text: str) -> bytes:
        import urllib.request
        key = self._cfg.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY not set for OpenAI TTS")
        payload = json.dumps({
            "model": "tts-1",
            "input": text[:4096],
            "voice": self._cfg.tts_voice,
            "speed": self._cfg.tts_speed,
            "response_format": "wav",
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/speech",
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

    def _espeak_synth(self, text: str) -> bytes:
        import subprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            subprocess.run(
                ["espeak-ng", "-w", tmp, "-s", str(int(150 * self._cfg.tts_speed)),
                 "-v", self._cfg.language, text[:500]],
                check=True, capture_output=True, timeout=20,
            )
            if os.path.exists(tmp):
                with open(tmp, "rb") as f:
                    return f.read()
            return b""
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def _bark_synth(self, text: str) -> bytes:
        from bark import generate_audio, SAMPLE_RATE  # type: ignore
        import numpy as np
        audio_array = generate_audio(text)
        pcm = (audio_array * 32767).astype(np.int16).tobytes()
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm)
            return buf.getvalue()

    def _play(self, audio: bytes) -> None:
        """Play WAV audio bytes via system player."""
        try:
            import pyaudio  # type: ignore
            pcm, sr, ch = wav_bytes_to_array(audio)
            p  = pyaudio.PyAudio()
            st = p.open(format=pyaudio.paInt16, channels=ch, rate=sr, output=True)
            st.write(pcm)
            st.stop_stream()
            st.close()
            p.terminate()
        except ImportError:
            self._play_cli(audio)
        except Exception as e:
            log.warning("Speaker._play failed: %s", e)

    @staticmethod
    def _play_cli(audio: bytes) -> None:
        """Fallback: write to temp file and use aplay/afplay/sox."""
        import subprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            tmp = f.name
        try:
            for cmd in [["aplay", tmp], ["afplay", tmp], ["sox", tmp, "-d"]]:
                try:
                    subprocess.run(cmd, check=True, capture_output=True, timeout=30)
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def _pick_backend(self) -> TTSBackend:
        if self._cfg.tts_backend != TTSBackend.STUB:
            return self._cfg.tts_backend
        # Auto-detect
        for dep, backend in [
            ("pyttsx3", TTSBackend.PYTTSX3),
        ]:
            try:
                __import__(dep)
                return backend
            except ImportError:
                pass
        try:
            import subprocess
            subprocess.run(["espeak-ng", "--version"], capture_output=True, timeout=3)
            return TTSBackend.ESPEAK
        except Exception:
            pass
        if os.environ.get("OPENAI_API_KEY"):
            return TTSBackend.OPENAI
        return TTSBackend.STUB


# ---------------------------------------------------------------------------
# Audio Recorder
# ---------------------------------------------------------------------------

class AudioRecorder:
    """
    Record from microphone using PyAudio with VAD-based stop trigger.
    Gracefully stubs when PyAudio is unavailable.
    """

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        self._cfg = config or VoiceConfig()
        self._vad = VADDetector(sample_rate=self._cfg.sample_rate)

    def record(
        self,
        on_chunk: Optional[Callable[[bytes], None]] = None,
    ) -> bytes:
        """
        Record until silence timeout or max duration.
        Returns raw WAV bytes.
        on_chunk: called with each PCM chunk as it arrives.
        """
        try:
            import pyaudio  # type: ignore
            return self._record_pyaudio(on_chunk)
        except ImportError:
            log.warning("AudioRecorder: PyAudio not installed — returning empty audio")
            return self._empty_wav()

    def _record_pyaudio(
        self, on_chunk: Optional[Callable[[bytes], None]]
    ) -> bytes:
        import pyaudio  # type: ignore
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self._cfg.channels,
            rate=self._cfg.sample_rate,
            input=True,
            input_device_index=self._cfg.audio_device_index,
            frames_per_buffer=self._cfg.chunk_size,
        )

        frames: List[bytes] = []
        silence_start: Optional[float] = None
        start_time = time.time()
        speech_detected = False

        log.debug("AudioRecorder: recording started (VAD=%s)", self._cfg.enable_vad)
        try:
            while True:
                chunk = stream.read(self._cfg.chunk_size, exception_on_overflow=False)
                frames.append(chunk)
                if on_chunk:
                    try:
                        on_chunk(chunk)
                    except Exception:
                        pass

                elapsed = time.time() - start_time
                if elapsed >= self._cfg.max_record_sec:
                    log.debug("AudioRecorder: max duration reached")
                    break

                if self._cfg.enable_vad:
                    is_speech = self._vad.is_speech(chunk, self._cfg.silence_threshold)
                    if is_speech:
                        speech_detected = True
                        silence_start   = None
                    else:
                        if speech_detected:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > self._cfg.silence_timeout_sec:
                                log.debug("AudioRecorder: silence timeout — stopping")
                                break
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        pcm = b"".join(frames)
        return self._pcm_to_wav(pcm)

    def _pcm_to_wav(self, pcm: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self._cfg.channels)
            wf.setsampwidth(_SAMPLE_WIDTH)
            wf.setframerate(self._cfg.sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()

    def _empty_wav(self) -> bytes:
        """Return an empty (0.1s) WAV — stub when no mic available."""
        silence = b"\x00" * int(self._cfg.sample_rate * 0.1) * 2
        return self._pcm_to_wav(silence)


# ---------------------------------------------------------------------------
# Wake-word detector
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """
    Detect wake words in a transcribed string.
    Can also do energy-based pre-screening before expensive STT.
    """

    def __init__(self, wake_words: Optional[List[str]] = None) -> None:
        self._words = [w.lower() for w in (wake_words or _WAKE_WORDS)]

    def detected_in_text(self, text: str) -> Tuple[bool, str]:
        """Check if any wake word appears in transcribed text."""
        lower = text.lower()
        for word in self._words:
            if word in lower:
                return True, word
        return False, ""

    def detected_in_audio_energy(
        self, pcm: bytes, threshold: int = _SILENCE_THRESHOLD * 3
    ) -> bool:
        """
        Quick energy pre-screen: returns True if the audio is likely to
        contain speech (used to skip quiet noise before STT).
        """
        return pcm_rms(pcm) > threshold

    def strip_wake_word(self, text: str) -> str:
        """Remove the wake word prefix from a command string."""
        lower = text.lower()
        for word in self._words:
            if lower.startswith(word):
                return text[len(word):].lstrip(" ,")
        return text


# ---------------------------------------------------------------------------
# Speaker diarisation stub
# ---------------------------------------------------------------------------

@dataclass
class DiarisationSegment:
    speaker_id: str
    start_sec:  float
    end_sec:    float
    text:       str = ""


class SpeakerDiariser:
    """
    Speaker diarisation — identifies who is speaking when.
    Full implementation requires pyannote.audio (GPU-heavy).
    This stub assigns all speech to 'SPEAKER_00'.
    """

    def __init__(self) -> None:
        self._model = None
        self._available = self._try_load()

    def _try_load(self) -> bool:
        try:
            from pyannote.audio import Pipeline  # type: ignore
            hf_token = os.environ.get("HF_TOKEN", "")
            if hf_token:
                self._model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token,
                )
                return True
        except ImportError:
            pass
        return False

    def diarise(self, wav_path: str) -> List[DiarisationSegment]:
        """Return diarisation segments for the given WAV file."""
        if self._available and self._model is not None:
            return self._pyannote_diarise(wav_path)
        return self._stub_diarise(wav_path)

    def _pyannote_diarise(self, wav_path: str) -> List[DiarisationSegment]:
        diarization = self._model(wav_path)
        segments: List[DiarisationSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarisationSegment(
                speaker_id=speaker,
                start_sec=turn.start,
                end_sec=turn.end,
            ))
        return segments

    @staticmethod
    def _stub_diarise(wav_path: str) -> List[DiarisationSegment]:
        pcm, sr = load_wav(wav_path)
        duration = len(pcm) / (sr * 2)
        return [DiarisationSegment(
            speaker_id="SPEAKER_00",
            start_sec=0.0,
            end_sec=duration,
        )]


# ---------------------------------------------------------------------------
# Multimodal router
# ---------------------------------------------------------------------------

@dataclass
class MultimodalMessage:
    """A message that may contain text + images."""
    text:   str               = ""
    images: List[bytes]       = field(default_factory=list)   # raw image bytes
    image_paths: List[str]    = field(default_factory=list)
    audio:  Optional[bytes]   = None

    def has_image(self) -> bool:
        return bool(self.images or self.image_paths)

    def has_audio(self) -> bool:
        return self.audio is not None

    def to_anthropic_content(self) -> List[Dict[str, Any]]:
        """Build Anthropic message content list with image_url parts."""
        parts: List[Dict[str, Any]] = []

        for img_bytes in self.images:
            b64 = base64.b64encode(img_bytes).decode()
            parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            })

        for path in self.image_paths:
            try:
                with open(path, "rb") as f:
                    img_bytes = f.read()
                b64 = base64.b64encode(img_bytes).decode()
                ext = Path(path).suffix.lower().lstrip(".")
                media_type = {
                    "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "gif": "image/gif",
                    "webp": "image/webp",
                }.get(ext, "image/jpeg")
                parts.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                })
            except Exception as e:
                log.warning("MultimodalMessage: failed to load image %s: %s", path, e)

        if self.text:
            parts.append({"type": "text", "text": self.text})

        return parts


class MultimodalRouter:
    """
    Routes messages to the appropriate model based on modality.
    - text-only     → configured default model
    - text + image  → vision model (claude-3-5-sonnet / gpt-4-vision)
    - audio         → Transcriber first, then text model
    """

    def __init__(
        self,
        config:     Optional[VoiceConfig] = None,
        text_model:   str = "",
        vision_model: str = "",
    ) -> None:
        self._cfg          = config or VoiceConfig()
        self._text_model   = text_model
        self._vision_model = vision_model or "claude-3-5-sonnet-20241022"
        self._transcriber  = Transcriber(config)

    def route(self, msg: MultimodalMessage) -> Dict[str, Any]:
        """
        Process a multimodal message.
        Returns {"model": ..., "content": ..., "text": ..., "transcription": ...}
        """
        result: Dict[str, Any] = {"transcription": ""}

        # Transcribe audio first if present
        if msg.has_audio() and msg.audio:
            transcription = self._transcriber.transcribe_bytes(msg.audio)
            result["transcription"] = transcription
            if transcription and not msg.text:
                msg = MultimodalMessage(
                    text=transcription,
                    images=msg.images,
                    image_paths=msg.image_paths,
                )

        # Pick model
        if msg.has_image():
            result["model"]   = self._vision_model
            result["content"] = msg.to_anthropic_content()
        else:
            result["model"]   = self._text_model
            result["content"] = [{"type": "text", "text": msg.text}]

        result["text"] = msg.text
        return result

    def describe_image(self, image_bytes: bytes, prompt: str = "Describe this image.") -> str:
        """Convenience: describe a single image using the vision model."""
        msg = MultimodalMessage(text=prompt, images=[image_bytes])
        routed = self.route(msg)
        try:
            from core.router import ModelRouter
            from core.config import ConfigManager
            cfg    = ConfigManager()
            router = ModelRouter(cfg)
            return router.complete(
                system="You are a helpful vision assistant.",
                messages=[{"role": "user", "content": routed["content"]}],
                model=routed["model"],
                max_tokens=1024,
            ) or ""
        except Exception as e:
            log.warning("MultimodalRouter.describe_image failed: %s", e)
            return ""


# ---------------------------------------------------------------------------
# Full VoicePipeline (high-level API)
# ---------------------------------------------------------------------------

class VoicePipeline:
    """
    High-level voice I/O pipeline combining recorder, transcriber, and speaker.

    listen()  → record mic → transcribe → return text
    speak()   → synthesise text → play
    converse() → listen → process(fn) → speak reply → return (user_text, reply)
    """

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        self._cfg        = config or VoiceConfig()
        self.recorder    = AudioRecorder(self._cfg)
        self.transcriber = Transcriber(self._cfg)
        self.speaker     = Speaker(self._cfg)
        self.vad         = VADDetector(sample_rate=self._cfg.sample_rate)
        self.wake_word   = WakeWordDetector(self._cfg.wake_words)
        self.diariser    = SpeakerDiariser()
        self.multimodal  = MultimodalRouter(self._cfg)
        self._running    = False

    def listen(
        self,
        on_chunk: Optional[Callable[[bytes], None]] = None,
    ) -> str:
        """Record microphone and return transcribed text."""
        wav = self.recorder.record(on_chunk=on_chunk)
        if not wav:
            return ""
        return self.transcriber.transcribe_bytes(wav, fmt="wav")

    def speak(self, text: str, play: bool = True) -> bytes:
        """Synthesise and optionally play text. Returns WAV bytes."""
        return self.speaker.speak(text, play=play)

    def stream_listen(
        self,
        on_partial: Optional[Callable[[str], None]] = None,
        max_seconds: float = 30.0,
        window_sec:  float = 3.0,
    ) -> str:
        """
        Real-time streaming transcription: capture the microphone and emit
        partial transcripts via *on_partial* as soon as each audio window is
        ready, instead of waiting for the full recording.

        Returns the final, complete transcript. Falls back to a single
        listen()+transcribe when the recorder can't stream (no PyAudio).

        This is the first-class streaming-STT entry point. Prefers true
        real-time cloud streaming (Deepgram) when DEEPGRAM_API_KEY +
        websocket-client are present; otherwise falls back to windowed local
        transcription.
        """
        # Preferred path: real-time cloud streaming (sub-second latency).
        cloud = CloudStreamingTranscriber(self._cfg, on_partial=on_partial)
        if cloud.available() and cloud.start():
            start_c = time.time()

            def _feed_cloud(chunk: bytes) -> None:
                cloud.push(chunk)
                if (time.time() - start_c) > max_seconds:
                    self.recorder.stop()

            try:
                self.recorder.record(on_chunk=_feed_cloud)
            except Exception:
                pass
            return cloud.finish()

        # Fallback: windowed local streaming.
        st = StreamingTranscriber(self._cfg, window_sec=window_sec)
        start = time.time()
        streamed_any = False

        def _feed(chunk: bytes) -> None:
            nonlocal streamed_any
            streamed_any = True
            partial = st.push(chunk)
            if partial and on_partial:
                on_partial(partial)
            if (time.time() - start) > max_seconds:
                # Signal the recorder to stop by raising StopIteration-like flag.
                self.recorder.stop()

        try:
            wav = self.recorder.record(on_chunk=_feed)
        except Exception:
            wav = b""

        if streamed_any:
            tail = st.flush()
            if tail:
                return tail
        # Fallback path: nothing streamed (no PyAudio) → transcribe the whole clip.
        if wav:
            return self.transcriber.transcribe_bytes(wav, fmt="wav")
        return st.flush()

    def converse(
        self,
        process_fn: Callable[[str], str],
        on_listen_start: Optional[Callable[[], None]] = None,
        on_transcribed:  Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, str]:
        """
        Full round-trip: listen → process → speak reply.
        Returns (user_text, reply_text).
        """
        if on_listen_start:
            on_listen_start()
        user_text = self.listen()
        if on_transcribed:
            on_transcribed(user_text)
        if not user_text.strip():
            return "", ""
        reply = process_fn(user_text)
        if reply:
            self.speak(reply)
        return user_text, reply

    def run_loop(
        self,
        process_fn: Callable[[str], str],
        on_user:    Optional[Callable[[str], None]] = None,
        on_agent:   Optional[Callable[[str], None]] = None,
        stop_phrases: Optional[List[str]] = None,
    ) -> None:
        """
        Continuous voice loop until stop phrase or self._running = False.
        process_fn(user_text) → reply text
        """
        self._running    = True
        stop_phrases     = [p.lower() for p in (stop_phrases or ["stop", "exit", "quit"])]

        log.info("VoicePipeline: starting continuous loop")
        while self._running:
            try:
                user_text = self.listen()
                if not user_text.strip():
                    continue

                # Strip wake word if enabled
                if self._cfg.enable_wake_word:
                    detected, _ = self.wake_word.detected_in_text(user_text)
                    if not detected:
                        continue
                    user_text = self.wake_word.strip_wake_word(user_text)

                if on_user:
                    on_user(user_text)

                # Check stop phrases
                if any(p in user_text.lower() for p in stop_phrases):
                    log.info("VoicePipeline: stop phrase detected — exiting loop")
                    break

                reply = process_fn(user_text)
                if reply:
                    if on_agent:
                        on_agent(reply)
                    self.speak(reply)

            except KeyboardInterrupt:
                break
            except Exception as e:
                log.warning("VoicePipeline loop error: %s", e)

        self._running = False
        log.info("VoicePipeline: loop stopped")

    def stop(self) -> None:
        """Stop the voice loop."""
        self._running = False

    def transcribe_file(self, path: str) -> str:
        """Transcribe a pre-recorded audio file."""
        return self.transcriber.transcribe_file(path)

    def stats(self) -> Dict[str, Any]:
        return {
            "stt_backend": self._cfg.stt_backend.value,
            "tts_backend": self._cfg.tts_backend.value,
            "sample_rate": self._cfg.sample_rate,
            "vad_enabled": self._cfg.enable_vad,
            "wake_word_enabled": self._cfg.enable_wake_word,
            "wake_words": self._cfg.wake_words,
        }


# ---------------------------------------------------------------------------
# Streaming transcription (chunked, for real-time use)
# ---------------------------------------------------------------------------

class StreamingTranscriber:
    """
    Accumulates audio chunks and transcribes incrementally.
    Useful for real-time transcription with partial results.
    """

    def __init__(
        self,
        config:          Optional[VoiceConfig] = None,
        window_sec:      float = 3.0,
        overlap_sec:     float = 0.5,
    ) -> None:
        self._cfg      = config or VoiceConfig()
        self._tx        = Transcriber(config)
        self._window    = int(self._cfg.sample_rate * window_sec) * 2
        self._overlap   = int(self._cfg.sample_rate * overlap_sec) * 2
        self._buffer    = b""
        self._results:  List[str] = []
        self._lock      = threading.Lock()

    def push(self, chunk: bytes) -> Optional[str]:
        """
        Push a PCM audio chunk. Returns a partial transcript when a full
        window is accumulated, otherwise None.
        """
        with self._lock:
            self._buffer += chunk
            if len(self._buffer) >= self._window:
                window = self._buffer[:self._window]
                self._buffer = self._buffer[self._window - self._overlap:]
                text = self._tx.transcribe_pcm(window, self._cfg.sample_rate)
                if text:
                    self._results.append(text)
                    return text
        return None

    def flush(self) -> str:
        """Transcribe any remaining buffered audio and return full transcript."""
        with self._lock:
            remaining = self._buffer
            self._buffer = b""

        parts = list(self._results)
        if remaining and len(remaining) > self._cfg.sample_rate // 4:
            text = self._tx.transcribe_pcm(remaining, self._cfg.sample_rate)
            if text:
                parts.append(text)
        self._results.clear()
        return " ".join(parts)

    def reset(self) -> None:
        with self._lock:
            self._buffer = b""
            self._results.clear()


# ---------------------------------------------------------------------------
# Real-time cloud streaming transcription (Deepgram WebSocket)
# ---------------------------------------------------------------------------

class CloudStreamingTranscriber:
    """
    True real-time streaming STT over Deepgram's WebSocket API.

    Sends raw PCM chunks as they arrive and yields interim + final transcripts
    with sub-second latency — unlike the windowed local StreamingTranscriber,
    which batches whisper passes. Requires DEEPGRAM_API_KEY and the optional
    `websocket-client` package; degrades gracefully (available()==False) when
    either is missing, so callers can fall back to the local path.

        st = CloudStreamingTranscriber(on_partial=print)
        if st.available():
            st.start(); st.push(pcm); ...; text = st.finish()
    """

    DG_URL = ("wss://api.deepgram.com/v1/listen"
              "?encoding=linear16&sample_rate={rate}&channels=1"
              "&interim_results=true&punctuate=true&language={lang}&model=nova-2")

    def __init__(
        self,
        config:     Optional[VoiceConfig] = None,
        api_key:    str = "",
        on_partial: Optional[Callable[[str], None]] = None,
        on_final:   Optional[Callable[[str], None]] = None,
    ) -> None:
        self._cfg     = config or VoiceConfig()
        self._key     = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        self._on_part = on_partial
        self._on_fin  = on_final
        self._ws      = None
        self._thread  = None
        self._finals:  List[str] = []
        self._lock     = threading.Lock()
        self._closed   = False

    @staticmethod
    def _have_ws() -> bool:
        try:
            import websocket  # noqa: F401  (websocket-client)
            return True
        except ImportError:
            return False

    def available(self) -> bool:
        """True only if both the API key and the websocket dep are present."""
        return bool(self._key) and self._have_ws()

    def start(self) -> bool:
        """Open the streaming connection. Returns True on success."""
        if not self.available():
            return False
        import json as _json
        import websocket  # type: ignore

        url = self.DG_URL.format(rate=self._cfg.sample_rate, lang=self._cfg.language)

        def _on_message(_ws, message):
            try:
                data = _json.loads(message)
                alt = (data.get("channel", {}).get("alternatives") or [{}])[0]
                text = (alt.get("transcript") or "").strip()
                if not text:
                    return
                if data.get("is_final"):
                    with self._lock:
                        self._finals.append(text)
                    if self._on_fin:
                        self._on_fin(text)
                elif self._on_part:
                    self._on_part(text)
            except Exception:
                pass

        try:
            self._ws = websocket.WebSocketApp(
                url,
                header=[f"Authorization: Token {self._key}"],
                on_message=_on_message,
            )
            self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
            self._thread.start()
            time.sleep(0.3)   # let the socket establish
            return True
        except Exception as e:
            log.warning("Deepgram stream start failed: %s", e)
            return False

    def push(self, pcm: bytes) -> None:
        """Send a raw PCM16 chunk to the live stream."""
        if self._ws is None or self._closed:
            return
        try:
            import websocket  # type: ignore
            self._ws.send(pcm, opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception:
            pass

    def finish(self, timeout: float = 3.0) -> str:
        """Close the stream and return the full concatenated final transcript."""
        self._closed = True
        try:
            if self._ws is not None:
                # Deepgram flush: send an empty frame then close.
                try:
                    self._ws.send(b"", opcode=__import__("websocket").ABNF.OPCODE_BINARY)
                except Exception:
                    pass
                time.sleep(min(timeout, 1.0))
                self._ws.close()
        except Exception:
            pass
        with self._lock:
            return " ".join(self._finals).strip()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_pipeline: Optional[VoicePipeline] = None


def get_voice_pipeline(config: Optional[VoiceConfig] = None) -> VoicePipeline:
    """Return (or create) the module-level default VoicePipeline."""
    global _default_pipeline
    if _default_pipeline is None or config is not None:
        _default_pipeline = VoicePipeline(config)
    return _default_pipeline


def listen(config: Optional[VoiceConfig] = None) -> str:
    """Record and transcribe once."""
    return get_voice_pipeline(config).listen()


def speak(text: str, config: Optional[VoiceConfig] = None) -> bytes:
    """Speak text once."""
    return get_voice_pipeline(config).speak(text, play=True)


def transcribe_file(path: str) -> str:
    """Transcribe an audio file at path."""
    return get_voice_pipeline().transcribe_file(path)
