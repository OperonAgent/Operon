"""
Operon Vision, Image Generation, and TTS Tools.

vision_analyze  — describe or query an image (OpenAI or Anthropic vision)
image_generate  — generate an image with DALL-E 3
tts_speak       — convert text to speech (OpenAI TTS; falls back to macOS say)

All tools read API keys from environment variables or ~/.operon/config.json.
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

import requests


# ── API key helper ────────────────────────────────────────────────────────────

def _get_api_key(provider: str = "openai") -> str:
    env_map = {
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    key = os.environ.get(env_map.get(provider, ""), "")
    if key:
        return key
    # Fall back to ~/.operon/config.json
    try:
        cfg_path = Path.home() / ".operon" / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            return cfg.get("api_keys", {}).get(provider, "")
    except Exception:
        pass
    return ""


def _r(success: bool, output=None, error: str = "") -> dict:
    return {"success": success, "output": output, "error": error}


# ── Vision ────────────────────────────────────────────────────────────────────

def vision_analyze(
    image_path: str = "",
    image_url:  str = "",
    prompt:     str = "Describe this image in detail.",
    provider:   str = "auto",
    **_,
) -> dict:
    """
    Analyze or query an image using a vision-capable model.

    Provide either image_path (local file) or image_url (public URL).
    provider: 'openai' | 'anthropic' | 'auto' (tries OpenAI first, then Anthropic)
    """
    if not image_path and not image_url:
        return _r(False, error="Provide image_path or image_url.")

    # ── Encode local image ────────────────────────────────────────────────────
    img_b64 = ""
    img_ext = "jpeg"
    if image_path:
        p = Path(image_path).expanduser()
        if not p.exists():
            return _r(False, error=f"File not found: {image_path}")
        img_b64 = base64.b64encode(p.read_bytes()).decode()
        ext = p.suffix.lower().lstrip(".")
        img_ext = "jpeg" if ext in ("jpg", "jpeg") else ext

    # ── Determine provider ────────────────────────────────────────────────────
    if provider == "auto":
        provider = "openai" if _get_api_key("openai") else "anthropic"

    # ── OpenAI vision ─────────────────────────────────────────────────────────
    if provider == "openai":
        api_key = _get_api_key("openai")
        if not api_key:
            return _r(False, error="No OpenAI API key. Set OPENAI_API_KEY or run /setup.")

        image_content: dict
        if img_b64:
            image_content = {
                "type":      "image_url",
                "image_url": {"url": f"data:image/{img_ext};base64,{img_b64}"},
            }
        else:
            image_content = {"type": "image_url", "image_url": {"url": image_url}}

        payload = {
            "model":    "gpt-4o",
            "messages": [{
                "role":    "user",
                "content": [image_content, {"type": "text", "text": prompt}],
            }],
            "max_tokens": 1200,
        }
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
            return _r(True, {"analysis": answer, "provider": "openai"})
        except requests.exceptions.HTTPError as e:
            return _r(False, error=f"OpenAI error: {e.response.text[:300]}")
        except Exception as e:
            return _r(False, error=str(e))

    # ── Anthropic vision ──────────────────────────────────────────────────────
    if provider == "anthropic":
        api_key = _get_api_key("anthropic")
        if not api_key:
            return _r(False, error="No Anthropic API key. Set ANTHROPIC_API_KEY or run /setup.")

        if img_b64:
            image_block = {
                "type":   "image",
                "source": {"type": "base64", "media_type": f"image/{img_ext}", "data": img_b64},
            }
        else:
            image_block = {
                "type":   "image",
                "source": {"type": "url", "url": image_url},
            }

        payload = {
            "model":      "claude-3-5-sonnet-20241022",
            "max_tokens": 1200,
            "messages":   [{
                "role":    "user",
                "content": [image_block, {"type": "text", "text": prompt}],
            }],
        }
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type":      "application/json",
                },
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            answer = resp.json()["content"][0]["text"]
            return _r(True, {"analysis": answer, "provider": "anthropic"})
        except requests.exceptions.HTTPError as e:
            return _r(False, error=f"Anthropic error: {e.response.text[:300]}")
        except Exception as e:
            return _r(False, error=str(e))

    return _r(False, error=f"Unknown provider: {provider}")


# ── Image generation ──────────────────────────────────────────────────────────

def image_generate(
    prompt:     str,
    model:      str = "dall-e-3",
    size:       str = "1024x1024",
    quality:    str = "standard",
    save_path:  str = "",
    **_,
) -> dict:
    """
    Generate an image from a text prompt using DALL-E 3.

    size:    '1024x1024' | '1792x1024' | '1024x1792'
    quality: 'standard' | 'hd'
    save_path: where to save the PNG (default: ~/Desktop/operon_img_<ts>.png)
    """
    api_key = _get_api_key("openai")
    if not api_key:
        return _r(False, error="No OpenAI API key. Set OPENAI_API_KEY or run /setup.")

    payload = {
        "model":           model,
        "prompt":          prompt,
        "n":               1,
        "size":            size,
        "quality":         quality,
        "response_format": "b64_json",
    }
    try:
        resp = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()["data"][0]
        b64  = data["b64_json"]

        if not save_path:
            desktop = Path.home() / "Desktop"
            if not desktop.exists():
                desktop = Path.home()
            save_path = str(desktop / f"operon_img_{int(time.time())}.png")

        save_path = str(Path(save_path).expanduser())
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(base64.b64decode(b64))

        return _r(True, {
            "path":           save_path,
            "revised_prompt": data.get("revised_prompt", prompt),
            "size":           size,
        })
    except requests.exceptions.HTTPError as e:
        return _r(False, error=f"OpenAI error: {e.response.text[:300]}")
    except Exception as e:
        return _r(False, error=str(e))


# ── Text-to-speech ────────────────────────────────────────────────────────────

def tts_speak(
    text:      str,
    voice:     str = "alloy",
    save_path: str = "",
    play:      bool = True,
    **_,
) -> dict:
    """
    Convert text to speech.

    Tries OpenAI TTS first (voices: alloy, echo, fable, onyx, nova, shimmer).
    Falls back to macOS 'say' command if no API key is available.

    play: True to also play the audio immediately (macOS afplay / Linux mpg123)
    """
    api_key = _get_api_key("openai")

    # ── OpenAI TTS ────────────────────────────────────────────────────────────
    if api_key:
        payload = {"model": "tts-1", "input": text[:4096], "voice": voice}
        try:
            resp = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()

            if not save_path:
                audio_dir = Path.home() / ".operon" / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(audio_dir / f"speech_{int(time.time())}.mp3")

            save_path = str(Path(save_path).expanduser())
            with open(save_path, "wb") as f:
                f.write(resp.content)

            if play:
                import subprocess
                player = "afplay" if sys.platform == "darwin" else "mpg123"
                subprocess.Popen(
                    [player, save_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            return _r(True, {
                "path":   save_path,
                "chars":  len(text),
                "voice":  voice,
                "played": play,
                "method": "openai_tts",
            })
        except requests.exceptions.HTTPError as e:
            # Fall through to system fallback
            pass
        except Exception:
            pass

    # ── macOS say fallback ────────────────────────────────────────────────────
    if sys.platform == "darwin":
        try:
            import subprocess
            subprocess.Popen(["say", text])
            return _r(True, {"played": True, "method": "macos_say",
                             "note": "Using system TTS — set OpenAI key for better voices."})
        except Exception as e:
            return _r(False, error=str(e))

    return _r(False, error=(
        "No TTS available. Either set OPENAI_API_KEY for OpenAI TTS, "
        "or run on macOS for system TTS."
    ))
