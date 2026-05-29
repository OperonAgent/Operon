"""
Operon Image Generation Tool.

Multi-backend image generation:
  1. OpenAI DALL-E 3 / DALL-E 2   — highest quality, requires OPENAI_API_KEY
  2. Stability AI (stable-diffusion) — via REST API, requires STABILITY_API_KEY
  3. Replicate                        — any model on replicate.com, requires REPLICATE_API_KEY
  4. Ollama / LLaVA (local)          — experimental, no key needed

Also provides:
  - Image editing (DALL-E inpainting)
  - Image variations (DALL-E)
  - Style transfer description helpers
  - Local save + base64 return

All functions return a dict: {success, output, error}
output contains: {url, path, base64, width, height, model, prompt_used}
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

# ── Config helpers ────────────────────────────────────────────────────────────

_SAVE_DIR = Path.home() / ".operon" / "images"
_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def _ok(output: Any)  -> dict: return {"success": True,  "output": output, "error": None}
def _err(msg: str)    -> dict: return {"success": False, "output": None,   "error": msg}

def _save_image_bytes(data: bytes, prompt: str, fmt: str = "png") -> str:
    """Save image bytes locally and return path."""
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)
    slug  = hashlib.md5(prompt.encode()).hexdigest()[:8]
    ts    = int(time.time())
    fname = f"img_{ts}_{slug}.{fmt}"
    path  = _SAVE_DIR / fname
    path.write_bytes(data)
    return str(path)

def _save_from_url(url: str, prompt: str) -> str:
    """Download image from URL and save locally."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    fmt = "png"
    if "jpeg" in resp.headers.get("content-type", "") or url.endswith(".jpg"):
        fmt = "jpg"
    return _save_image_bytes(resp.content, prompt, fmt)


# ── DALL-E (OpenAI) ───────────────────────────────────────────────────────────

def image_generate(
    prompt: str,
    model: str = "dall-e-3",
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid",
    n: int = 1,
    save: bool = True,
    return_base64: bool = False,
) -> dict:
    """
    Generate an image using DALL-E 3 (default) or DALL-E 2.

    Args:
        prompt:        Text description of the image to generate.
        model:         "dall-e-3" (default) or "dall-e-2".
        size:          Image size. DALL-E 3: "1024x1024", "1792x1024", "1024x1792".
                       DALL-E 2: "256x256", "512x512", "1024x1024".
        quality:       "standard" or "hd" (DALL-E 3 only, hd costs more).
        style:         "vivid" (default) or "natural" (DALL-E 3 only).
        n:             Number of images (DALL-E 3: max 1, DALL-E 2: max 10).
        save:          Save image locally (default True).
        return_base64: Include base64-encoded image in output.

    Returns:
        {success, output: {url, path, base64, width, height, model, revised_prompt}, error}
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return _err("OPENAI_API_KEY not set. Required for DALL-E image generation.")

    payload: dict = {
        "model":   model,
        "prompt":  prompt,
        "size":    size,
        "n":       min(n, 1 if model == "dall-e-3" else 10),
    }
    if model == "dall-e-3":
        payload["quality"] = quality
        payload["style"]   = style
    if return_base64:
        payload["response_format"] = "b64_json"
    else:
        payload["response_format"] = "url"

    try:
        resp = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data   = resp.json()
        result = data["data"][0]

        out: Dict[str, Any] = {
            "model":          model,
            "prompt_used":    prompt,
            "revised_prompt": result.get("revised_prompt", prompt),
            "width":  int(size.split("x")[0]),
            "height": int(size.split("x")[1]),
        }

        if return_base64:
            b64 = result.get("b64_json", "")
            out["base64"] = b64
            if save:
                image_bytes = base64.b64decode(b64)
                out["path"] = _save_image_bytes(image_bytes, prompt)
        else:
            url = result.get("url", "")
            out["url"] = url
            if save:
                out["path"] = _save_from_url(url, prompt)

        return _ok(out)

    except requests.HTTPError as e:
        try:
            err_body = e.response.json()
            return _err(f"OpenAI API error: {err_body.get('error', {}).get('message', str(e))}")
        except Exception:
            return _err(f"OpenAI API error: {e}")
    except Exception as e:
        return _err(f"image_generate failed: {e}")


def image_edit(
    image_path: str,
    prompt: str,
    mask_path: Optional[str] = None,
    size: str = "1024x1024",
    n: int = 1,
    save: bool = True,
) -> dict:
    """
    Edit an existing image using DALL-E inpainting.
    The image and mask must be square PNG files.

    Args:
        image_path: Path to the original image (PNG, max 4 MB).
        prompt:     Description of the desired edit.
        mask_path:  Path to mask PNG (transparent areas = edit areas). None = full edit.
        size:       Output size: "256x256", "512x512", "1024x1024".
        n:          Number of variations (1-10).
        save:       Save locally.

    Returns:
        {success, output: {url, path, model}, error}
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return _err("OPENAI_API_KEY not set.")

    try:
        files: Dict[str, Any] = {
            "image":  ("image.png", open(image_path, "rb"), "image/png"),
            "prompt": (None, prompt),
            "size":   (None, size),
            "n":      (None, str(n)),
            "model":  (None, "dall-e-2"),
        }
        if mask_path:
            files["mask"] = ("mask.png", open(mask_path, "rb"), "image/png")

        resp = requests.post(
            "https://api.openai.com/v1/images/edits",
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            timeout=120,
        )
        resp.raise_for_status()
        url = resp.json()["data"][0]["url"]
        out: Dict[str, Any] = {"url": url, "model": "dall-e-2", "prompt": prompt}
        if save:
            out["path"] = _save_from_url(url, prompt)
        return _ok(out)
    except FileNotFoundError as e:
        return _err(f"File not found: {e}")
    except Exception as e:
        return _err(f"image_edit failed: {e}")


def image_variation(
    image_path: str,
    n: int = 1,
    size: str = "1024x1024",
    save: bool = True,
) -> dict:
    """
    Generate variations of an existing image using DALL-E 2.

    Args:
        image_path: Path to source image (PNG, max 4 MB, square).
        n:          Number of variations (1-10).
        size:       Output size.
        save:       Save locally.

    Returns:
        {success, output: list of {url, path}, error}
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return _err("OPENAI_API_KEY not set.")

    try:
        resp = requests.post(
            "https://api.openai.com/v1/images/variations",
            headers={"Authorization": f"Bearer {api_key}"},
            files={
                "image": ("image.png", open(image_path, "rb"), "image/png"),
                "n":     (None, str(n)),
                "size":  (None, size),
                "model": (None, "dall-e-2"),
            },
            timeout=120,
        )
        resp.raise_for_status()
        results = []
        for item in resp.json()["data"]:
            url = item["url"]
            entry: Dict[str, Any] = {"url": url}
            if save:
                entry["path"] = _save_from_url(url, "variation")
            results.append(entry)
        return _ok(results)
    except FileNotFoundError as e:
        return _err(f"File not found: {e}")
    except Exception as e:
        return _err(f"image_variation failed: {e}")


# ── Stability AI ──────────────────────────────────────────────────────────────

def image_generate_stability(
    prompt: str,
    negative_prompt: str = "",
    model: str = "stable-diffusion-xl-1024-v1-0",
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    cfg_scale: float = 7.0,
    samples: int = 1,
    save: bool = True,
) -> dict:
    """
    Generate images using Stability AI (Stable Diffusion).
    Requires STABILITY_API_KEY environment variable.

    Args:
        prompt:          Text description.
        negative_prompt: What to avoid in the image.
        model:           Engine ID (default: stable-diffusion-xl-1024-v1-0).
        width/height:    Image dimensions (must match engine requirements).
        steps:           Inference steps (more = higher quality, slower).
        cfg_scale:       How closely to follow the prompt (1-35, default 7).
        samples:         Number of images to generate.
        save:            Save locally.

    Returns:
        {success, output: list of {path, seed}, error}
    """
    api_key = os.environ.get("STABILITY_API_KEY", "")
    if not api_key:
        return _err("STABILITY_API_KEY not set. Get one at platform.stability.ai")

    payload = {
        "text_prompts": [
            {"text": prompt,          "weight": 1.0},
            {"text": negative_prompt, "weight": -1.0},
        ] if negative_prompt else [{"text": prompt, "weight": 1.0}],
        "width":     width,
        "height":    height,
        "steps":     steps,
        "cfg_scale": cfg_scale,
        "samples":   samples,
    }

    try:
        resp = requests.post(
            f"https://api.stability.ai/v1/generation/{model}/text-to-image",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        results = []
        for artifact in resp.json().get("artifacts", []):
            img_bytes = base64.b64decode(artifact["base64"])
            path = _save_image_bytes(img_bytes, prompt)
            results.append({"path": path, "seed": artifact.get("seed")})
        return _ok(results)
    except requests.HTTPError as e:
        try:
            return _err(f"Stability AI error: {e.response.json()}")
        except Exception:
            return _err(f"Stability AI error: {e}")
    except Exception as e:
        return _err(f"image_generate_stability failed: {e}")


# ── Replicate ─────────────────────────────────────────────────────────────────

def image_generate_replicate(
    prompt: str,
    model: str = "stability-ai/sdxl:latest",
    input_params: Optional[dict] = None,
    save: bool = True,
) -> dict:
    """
    Generate images using any Replicate model.
    Requires REPLICATE_API_KEY environment variable.

    Args:
        prompt:       Text description.
        model:        Replicate model string "owner/model:version" or "owner/model:latest".
        input_params: Additional model-specific parameters.
        save:         Save locally.

    Returns:
        {success, output: {url, path, model}, error}
    """
    api_key = os.environ.get("REPLICATE_API_KEY", "")
    if not api_key:
        return _err("REPLICATE_API_KEY not set. Get one at replicate.com")

    params = {"prompt": prompt}
    if input_params:
        params.update(input_params)

    try:
        # Create prediction
        resp = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            },
            json={"version": model.split(":")[-1] if ":" in model else model,
                  "input": params},
            timeout=30,
        )
        resp.raise_for_status()
        prediction = resp.json()
        pred_id    = prediction["id"]

        # Poll until complete
        for _ in range(60):  # max 60 * 2 = 120s
            time.sleep(2)
            poll = requests.get(
                f"https://api.replicate.com/v1/predictions/{pred_id}",
                headers={"Authorization": f"Token {api_key}"},
                timeout=15,
            )
            poll.raise_for_status()
            status = poll.json()
            if status["status"] == "succeeded":
                urls = status.get("output", [])
                url  = urls[0] if isinstance(urls, list) and urls else str(urls)
                out: Dict[str, Any] = {"url": url, "model": model}
                if save:
                    out["path"] = _save_from_url(url, prompt)
                return _ok(out)
            elif status["status"] in ("failed", "canceled"):
                return _err(f"Replicate prediction {status['status']}: {status.get('error')}")

        return _err("Replicate prediction timed out after 120s")

    except requests.HTTPError as e:
        return _err(f"Replicate API error: {e}")
    except Exception as e:
        return _err(f"image_generate_replicate failed: {e}")


# ── Utility ───────────────────────────────────────────────────────────────────

def image_list_generated(limit: int = 20) -> dict:
    """
    List recently generated images saved in ~/.operon/images/.

    Args:
        limit: Maximum number of results (default 20, newest first).

    Returns:
        {success, output: list of {path, size_kb, created}, error}
    """
    try:
        _SAVE_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(_SAVE_DIR.glob("img_*"), key=lambda f: f.stat().st_mtime, reverse=True)
        result = []
        for f in files[:limit]:
            st = f.stat()
            result.append({
                "path":     str(f),
                "size_kb":  round(st.st_size / 1024, 1),
                "created":  time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
            })
        return _ok(result)
    except Exception as e:
        return _err(f"image_list_generated failed: {e}")


def image_describe(
    path_or_url: str,
    detail: str = "low",
) -> dict:
    """
    Describe an image using GPT-4 Vision (OpenAI).

    Args:
        path_or_url: Local file path or public URL.
        detail:      "low" (faster, cheaper) or "high" (more detailed).

    Returns:
        {success, output: str (description), error}
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return _err("OPENAI_API_KEY not set.")

    try:
        if path_or_url.startswith("http"):
            image_content = {"type": "image_url", "image_url": {"url": path_or_url, "detail": detail}}
        else:
            data = Path(path_or_url).read_bytes()
            b64  = base64.b64encode(data).decode()
            ext  = Path(path_or_url).suffix.lstrip(".").lower() or "png"
            mime = f"image/{ext}"
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail}
            }

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": [
                    image_content,
                    {"type": "text", "text": "Describe this image in detail."}
                ]}],
                "max_tokens": 500,
            },
            timeout=30,
        )
        resp.raise_for_status()
        description = resp.json()["choices"][0]["message"]["content"]
        return _ok(description)
    except FileNotFoundError:
        return _err(f"File not found: {path_or_url}")
    except Exception as e:
        return _err(f"image_describe failed: {e}")
