"""
Operon Voice Input / Output.

Speech-to-Text (STT):
  • Local Whisper   — openai-whisper package (pip install openai-whisper)
  • OpenAI Whisper API — uses OPENAI_API_KEY, no local model download needed
  • Fallback        — returns an error with install instructions

Audio recording:
  • sounddevice    — pip install sounddevice (recommended)
  • pyaudio        — pip install pyaudio (fallback)
  • System arecord — Linux fallback (no Python dep)

Text-to-Speech (TTS):
  • pyttsx3        — fully offline, cross-platform (pip install pyttsx3)
  • OpenAI TTS     — existing tts_speak() tool (uses API key)
  • macOS say      — built-in, no deps needed

All functions accept **_ for registry compatibility.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Audio recording helpers
# ---------------------------------------------------------------------------

def _record_sounddevice(duration: int, sample_rate: int) -> Optional[bytes]:
    """Record audio using sounddevice. Returns WAV bytes or None."""
    try:
        import sounddevice as sd
        import numpy as np

        print(f"\n  🎙  Recording {duration}s… (press Ctrl+C to stop early)", flush=True)
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        print("  ✓  Recording complete.", flush=True)

        # Encode as WAV in memory
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()
    except KeyboardInterrupt:
        print("\n  Recording stopped early.", flush=True)
        return None
    except ImportError:
        return None
    except Exception as e:
        print(f"  sounddevice error: {e}", flush=True)
        return None


def _record_pyaudio(duration: int, sample_rate: int) -> Optional[bytes]:
    """Record audio using PyAudio. Returns WAV bytes or None."""
    try:
        import pyaudio
        import wave

        CHUNK = 1024
        pa    = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=CHUNK,
        )

        print(f"\n  🎙  Recording {duration}s… (press Ctrl+C to stop early)", flush=True)
        frames = []
        try:
            for _ in range(0, int(sample_rate / CHUNK * duration)):
                frames.append(stream.read(CHUNK, exception_on_overflow=False))
        except KeyboardInterrupt:
            print("\n  Recording stopped early.", flush=True)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

        if not frames:
            return None

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()
    except ImportError:
        return None
    except Exception as e:
        print(f"  pyaudio error: {e}", flush=True)
        return None


def _record_arecord(duration: int, sample_rate: int) -> Optional[bytes]:
    """Record using Linux arecord. Returns WAV bytes or None."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        print(f"\n  🎙  Recording {duration}s via arecord…", flush=True)
        subprocess.run(
            ["arecord", "-d", str(duration), "-r", str(sample_rate),
             "-f", "S16_LE", "-c", "1", tmp],
            timeout=duration + 5, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        data = Path(tmp).read_bytes()
        Path(tmp).unlink(missing_ok=True)
        return data if data else None
    except Exception:
        return None


def _record_audio(duration: int = 5, sample_rate: int = 16000) -> Optional[bytes]:
    """Try recording backends in priority order. Returns WAV bytes or None."""
    # sounddevice preferred (lower latency, numpy-backed)
    data = _record_sounddevice(duration, sample_rate)
    if data:
        return data
    # PyAudio fallback
    data = _record_pyaudio(duration, sample_rate)
    if data:
        return data
    # Linux arecord
    data = _record_arecord(duration, sample_rate)
    return data


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------

def _transcribe_local_whisper(wav_bytes: bytes, model_size: str = "base") -> Optional[str]:
    """Transcribe using local openai-whisper model."""
    try:
        import whisper
        import numpy as np
        import wave

        # Decode WAV to float32 numpy array
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            n_frames = wf.getnframes()
            raw      = wf.readframes(n_frames)
            sr       = wf.getframerate()

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        # Whisper expects 16kHz; resample if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        model  = whisper.load_model(model_size)
        result = model.transcribe(audio, fp16=False)
        return result.get("text", "").strip()
    except ImportError:
        return None
    except Exception as e:
        print(f"  Whisper local error: {e}", flush=True)
        return None


def _transcribe_openai_api(wav_bytes: bytes, language: str = "") -> Optional[str]:
    """Transcribe using OpenAI Whisper API."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import urllib.request as _req
        url      = "https://api.openai.com/v1/audio/transcriptions"
        boundary = "OperonVoiceBoundary"

        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n'
            f"--{boundary}\r\n"
        )
        if language:
            body += (
                f'Content-Disposition: form-data; name="language"\r\n\r\n{language}\r\n'
                f"--{boundary}\r\n"
            )
        body += (
            'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        )
        payload = body.encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()

        request = _req.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        import json
        with _req.urlopen(request, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("text", "").strip()
    except Exception as e:
        print(f"  OpenAI Whisper API error: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# TTS helpers
# ---------------------------------------------------------------------------

def _speak_pyttsx3(text: str, rate: int = 175, voice_id: str = "") -> bool:
    """Speak using pyttsx3 (fully offline)."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        if voice_id:
            engine.setProperty("voice", voice_id)
        engine.say(text)
        engine.runAndWait()
        return True
    except ImportError:
        return False
    except Exception:
        return False


def _speak_macos_say(text: str, voice: str = "", rate: int = 175) -> bool:
    """Speak using macOS built-in 'say' command."""
    try:
        cmd = ["say", "-r", str(rate)]
        if voice:
            cmd += ["-v", voice]
        cmd.append(text)
        subprocess.run(cmd, timeout=120, check=False)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def voice_record_and_transcribe(
    duration: int = 5,
    model: str = "base",
    language: str = "",
    use_api: bool = False,
    sample_rate: int = 16000,
    **_,
) -> dict:
    """
    Record audio from the microphone and transcribe it to text using Whisper.

    Args:
        duration    — recording duration in seconds (optional, default 5)
        model       — local Whisper model size: tiny / base / small / medium / large
                      (optional, default 'base')
        language    — language hint e.g. 'en', 'fr', 'de' (optional)
        use_api     — force OpenAI Whisper API instead of local model (optional)
        sample_rate — audio sample rate in Hz (optional, default 16000)

    Returns:
        {success, transcript, duration_s, model_used, error}
    """
    # Check if we're in a terminal (can't record from a non-interactive session)
    if not sys.stdin.isatty():
        return {
            "success": False,
            "error": "voice_record requires an interactive terminal (stdin is not a tty).",
        }

    # Record audio
    wav_bytes = _record_audio(duration, sample_rate)
    if not wav_bytes:
        return {
            "success": False,
            "error": (
                "Could not record audio. Install a recording backend:\n"
                "  pip install sounddevice  (recommended)\n"
                "  pip install pyaudio\n"
                "  Or ensure 'arecord' is installed (Linux)"
            ),
        }

    # Transcribe
    transcript = None
    model_used = ""

    if use_api or not _transcribe_local_whisper.__wrapped__ if hasattr(_transcribe_local_whisper, "__wrapped__") else False:
        transcript = _transcribe_openai_api(wav_bytes, language)
        model_used = "openai-whisper-api"

    if transcript is None and not use_api:
        transcript = _transcribe_local_whisper(wav_bytes, model)
        model_used = f"whisper-{model}-local"

    if transcript is None:
        transcript = _transcribe_openai_api(wav_bytes, language)
        model_used = "openai-whisper-api"

    if transcript is None:
        return {
            "success": False,
            "error": (
                "Transcription failed. Install one of:\n"
                "  pip install openai-whisper  (local, no API key needed)\n"
                "  Set OPENAI_API_KEY for cloud Whisper API"
            ),
        }

    return {
        "success":    True,
        "transcript": transcript,
        "duration_s": duration,
        "model_used": model_used,
        "error":      "",
    }


def voice_transcribe_file(
    file_path: str = "",
    model: str = "base",
    language: str = "",
    use_api: bool = False,
    **_,
) -> dict:
    """
    Transcribe an existing audio file (WAV, MP3, M4A, OGG, etc.) to text.

    Args:
        file_path — path to the audio file (required)
        model     — local Whisper model: tiny/base/small/medium/large (optional, default 'base')
        language  — language hint e.g. 'en' (optional)
        use_api   — use OpenAI Whisper API instead of local model (optional)

    Returns:
        {success, transcript, file, model_used, error}
    """
    if not file_path:
        return {"success": False, "error": "file_path is required."}

    path = Path(file_path)
    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    file_bytes = path.read_bytes()

    # For local Whisper, use the file path directly (it handles format conversion)
    transcript = None
    model_used = ""

    if not use_api:
        try:
            import whisper
            m      = whisper.load_model(model)
            result = m.transcribe(str(path), language=language or None, fp16=False)
            transcript = result.get("text", "").strip()
            model_used = f"whisper-{model}-local"
        except ImportError:
            pass
        except Exception as e:
            print(f"  Whisper local error: {e}", flush=True)

    if transcript is None:
        transcript = _transcribe_openai_api(file_bytes, language)
        model_used = "openai-whisper-api"

    if transcript is None:
        return {
            "success": False,
            "error": "Transcription failed. Install openai-whisper or set OPENAI_API_KEY.",
        }

    return {
        "success":    True,
        "transcript": transcript,
        "file":       str(path),
        "model_used": model_used,
        "error":      "",
    }


def voice_speak(
    text: str = "",
    voice: str = "",
    rate: int = 175,
    engine: str = "auto",
    save_path: str = "",
    play: bool = True,
    **_,
) -> dict:
    """
    Convert text to speech using an offline engine or OpenAI TTS.

    Args:
        text      — text to speak (required)
        voice     — voice name or ID (optional, engine-specific)
        rate      — speech rate in words per minute (optional, default 175)
        engine    — 'auto' | 'pyttsx3' | 'say' | 'openai' (optional, default 'auto')
        save_path — save audio to this file path (optional)
        play      — play audio immediately, default True (optional)

    Returns:
        {success, engine_used, error}
    """
    if not text:
        return {"success": False, "error": "text is required."}

    # If save_path or openai explicitly requested, delegate to existing tts_speak
    if engine == "openai" or save_path:
        try:
            from tools.vision import tts_speak
            return tts_speak(text=text, voice=voice or "alloy", save_path=save_path, play=play)
        except Exception as e:
            return {"success": False, "error": str(e)}

    used = ""
    if engine in ("auto", "pyttsx3"):
        if _speak_pyttsx3(text, rate, voice):
            used = "pyttsx3"
    if not used and engine in ("auto", "say") and sys.platform == "darwin":
        if _speak_macos_say(text, voice, rate):
            used = "macos-say"
    if not used:
        # Last resort: OpenAI TTS
        try:
            from tools.vision import tts_speak
            result = tts_speak(text=text, voice=voice or "alloy", play=play)
            if result.get("success"):
                used = "openai-tts"
            else:
                return result
        except Exception as e:
            return {
                "success": False,
                "error": (
                    f"No TTS engine available ({e}). Install one of:\n"
                    "  pip install pyttsx3  (offline, cross-platform)\n"
                    "  macOS: 'say' built-in\n"
                    "  Set OPENAI_API_KEY for OpenAI TTS"
                ),
            }

    return {"success": True, "engine_used": used, "error": ""}


def voice_list_voices(**_) -> dict:
    """
    List available TTS voices on this system.

    Returns:
        {success, voices: [{id, name, languages}], engine, error}
    """
    # Try pyttsx3 first
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        result = [
            {
                "id":        v.id,
                "name":      v.name,
                "languages": [l.decode() if isinstance(l, bytes) else l for l in (v.languages or [])],
            }
            for v in voices
        ]
        engine.stop()
        return {"success": True, "voices": result, "engine": "pyttsx3", "error": ""}
    except ImportError:
        pass
    except Exception:
        pass

    # macOS say
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["say", "-v", "?"], text=True, timeout=5)
            voices = []
            for line in out.strip().splitlines():
                parts = line.split()
                if parts:
                    voices.append({"id": parts[0], "name": parts[0], "languages": []})
            return {"success": True, "voices": voices, "engine": "macos-say", "error": ""}
        except Exception:
            pass

    return {
        "success": False,
        "error": "No TTS engine found. Install: pip install pyttsx3",
        "voices": [],
    }
