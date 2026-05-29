"""
tools/video_gen.py — Text-to-video and image-to-video generation.

Provider-agnostic with graceful degradation:
  • Replicate  (REPLICATE_API_TOKEN)  — many open video models (e.g. Zeroscope,
    Stable Video Diffusion, LTX, Kling-community mirrors).
  • Luma Dream Machine (LUMA_API_KEY)
  • Runway  (RUNWAY_API_KEY)
  • Fallback: returns a clear, actionable error telling the user which key to set.

No heavy SDKs required — uses stdlib urllib for HTTP. Generated videos are
polled to completion and the resulting URL (and a downloaded local file when
possible) is returned.

Tools exposed:  video_generate, video_from_image, video_list_generated
"""

from __future__ import annotations

import os
import json
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_OUT_DIR = Path.home() / ".operon" / "generated" / "video"


def _ok(output: Any) -> dict:  return {"success": True,  "output": output, "error": None}
def _err(msg: str)   -> dict:  return {"success": False, "output": None,   "error": msg}


def _http_json(url: str, method: str = "GET", headers: Optional[dict] = None,
               body: Optional[dict] = None, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download(url: str, prompt: str) -> Optional[str]:
    try:
        _OUT_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "".join(c for c in prompt[:30] if c.isalnum() or c in " -_").strip().replace(" ", "_")
        dest = _OUT_DIR / f"video_{slug}_{ts}.mp4"
        with urllib.request.urlopen(url, timeout=120) as resp:
            dest.write_bytes(resp.read())
        return str(dest)
    except Exception:
        return None


# ── Replicate ───────────────────────────────────────────────────────────────────

def _replicate_run(model_version: str, inputs: dict, prompt: str,
                   poll_timeout: float = 300.0) -> dict:
    token = os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        return _err("REPLICATE_API_TOKEN not set")
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    try:
        pred = _http_json("https://api.replicate.com/v1/predictions", "POST",
                          headers, {"version": model_version, "input": inputs})
    except urllib.error.HTTPError as e:
        return _err(f"Replicate error {e.code}: {e.read().decode('utf-8', 'replace')[:200]}")
    except Exception as e:
        return _err(f"Replicate request failed: {e}")

    get_url = pred.get("urls", {}).get("get")
    deadline = time.time() + poll_timeout
    while time.time() < deadline:
        status = pred.get("status")
        if status == "succeeded":
            out = pred.get("output")
            video_url = out[-1] if isinstance(out, list) and out else out
            local = _download(video_url, prompt) if isinstance(video_url, str) else None
            return _ok({"video_url": video_url, "local_path": local, "provider": "replicate"})
        if status in ("failed", "canceled"):
            return _err(f"Replicate prediction {status}: {pred.get('error')}")
        time.sleep(3)
        try:
            pred = _http_json(get_url, "GET", headers)
        except Exception as e:
            return _err(f"Replicate poll failed: {e}")
    return _err("Replicate timed out waiting for video")


# ── Luma ─────────────────────────────────────────────────────────────────────────

def _luma_run(prompt: str, image_url: Optional[str] = None,
              poll_timeout: float = 300.0) -> dict:
    key = os.environ.get("LUMA_API_KEY", "")
    if not key:
        return _err("LUMA_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body: Dict[str, Any] = {"prompt": prompt}
    if image_url:
        body["keyframes"] = {"frame0": {"type": "image", "url": image_url}}
    try:
        gen = _http_json("https://api.lumalabs.ai/dream-machine/v1/generations",
                         "POST", headers, body)
    except Exception as e:
        return _err(f"Luma request failed: {e}")
    gid = gen.get("id")
    deadline = time.time() + poll_timeout
    while gid and time.time() < deadline:
        if gen.get("state") == "completed":
            video_url = (gen.get("assets") or {}).get("video")
            local = _download(video_url, prompt) if video_url else None
            return _ok({"video_url": video_url, "local_path": local, "provider": "luma"})
        if gen.get("state") == "failed":
            return _err(f"Luma generation failed: {gen.get('failure_reason')}")
        time.sleep(3)
        try:
            gen = _http_json(
                f"https://api.lumalabs.ai/dream-machine/v1/generations/{gid}",
                "GET", headers)
        except Exception as e:
            return _err(f"Luma poll failed: {e}")
    return _err("Luma timed out waiting for video")


# ── Public tools ──────────────────────────────────────────────────────────────────

# A widely-mirrored open text-to-video model version on Replicate (Zeroscope v2 XL).
_DEFAULT_REPLICATE_T2V = (
    "anotherjesse/zeroscope-v2-xl:"
    "9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351"
)


def video_generate(prompt: str, provider: str = "auto",
                   duration: int = 4, fps: int = 24) -> dict:
    """
    Generate a short video from a text prompt.

    Args:
        prompt:   text description of the video.
        provider: 'auto' | 'replicate' | 'luma' (default auto — picks by available key).
        duration: target length in seconds (provider-dependent best-effort).
        fps:      frames per second hint.
    """
    if not prompt or not str(prompt).strip():
        return _err("prompt is required")

    provider = (provider or "auto").lower()
    have_replicate = bool(os.environ.get("REPLICATE_API_TOKEN"))
    have_luma      = bool(os.environ.get("LUMA_API_KEY"))

    if provider == "auto":
        provider = "replicate" if have_replicate else ("luma" if have_luma else "")

    if provider == "replicate":
        return _replicate_run(_DEFAULT_REPLICATE_T2V,
                              {"prompt": prompt, "num_frames": max(1, duration * fps),
                               "fps": fps}, prompt)
    if provider == "luma":
        return _luma_run(prompt)

    return _err("No video provider configured. Set REPLICATE_API_TOKEN or "
                "LUMA_API_KEY, then retry (or pass provider= explicitly).")


def video_from_image(image_url: str, prompt: str = "", provider: str = "auto") -> dict:
    """
    Animate a still image into a short video (image-to-video).

    Args:
        image_url: public URL of the source image.
        prompt:    optional motion/scene guidance.
        provider:  'auto' | 'luma' | 'replicate'.
    """
    if not image_url:
        return _err("image_url is required")
    provider = (provider or "auto").lower()
    have_luma = bool(os.environ.get("LUMA_API_KEY"))
    if provider in ("auto", "luma") and have_luma:
        return _luma_run(prompt or "animate this image", image_url=image_url)
    if provider in ("auto", "replicate") and os.environ.get("REPLICATE_API_TOKEN"):
        # Stable Video Diffusion (img2vid)
        svd = ("stability-ai/stable-video-diffusion:"
               "3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438")
        return _replicate_run(svd, {"input_image": image_url}, prompt or "svd")
    return _err("No image-to-video provider configured. Set LUMA_API_KEY or "
                "REPLICATE_API_TOKEN.")


def video_list_generated(limit: int = 20) -> dict:
    """List previously generated videos saved under ~/.operon/generated/video."""
    try:
        if not _OUT_DIR.exists():
            return _ok({"count": 0, "videos": []})
        files = sorted(_OUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        vids = [{"path": str(f), "size_kb": f.stat().st_size // 1024} for f in files[:limit]]
        return _ok({"count": len(vids), "videos": vids})
    except Exception as e:
        return _err(str(e))
