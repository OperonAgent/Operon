"""
tools/visual_grounding.py — Visual DOM Grounding

Finds UI elements on a screenshot by description and returns their
pixel coordinates for automated clicking/interaction.

Strategies (in priority order):
  1. Vision LLM (Anthropic claude-3-5-sonnet vision)
     → Best accuracy; prompts model to return {x, y, confidence}
  2. Template matching (OpenCV)
     → Finds a template image inside a larger screenshot
  3. Text OCR (pytesseract)
     → Finds clickable text on screen by OCR
  4. Accessibility tree (Playwright page.accessibility.snapshot)
     → Finds elements by text in the A11Y tree

All functions return {success, x, y, confidence, method, output/error}.

Usage:
    from tools.visual_grounding import find_element, click_element

    result = find_element("Submit button", screenshot_bytes)
    if result["success"]:
        x, y = result["x"], result["y"]
        # click at (x, y) with Playwright / PyAutoGUI
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("operon.visual_grounding")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CONFIDENCE_THRESHOLD = 0.5
_VISION_MODEL = "claude-3-5-sonnet-20241022"
_TEMPLATE_MATCH_THRESHOLD = 0.75   # OpenCV TM_CCOEFF_NORMED
_OCR_MARGIN = 10                    # pixels to add around OCR text bounds

_FIND_ELEMENT_SYSTEM = """You are a visual UI assistant. Given a screenshot,
find the location of the requested UI element. Respond ONLY with valid JSON
in this exact format (no markdown, no explanation):
{"x": <int>, "y": <int>, "confidence": <float 0.0-1.0>, "description": "<what you found>"}

If the element is not visible, respond:
{"x": -1, "y": -1, "confidence": 0.0, "description": "not found"}"""

_FIND_ALL_SYSTEM = """You are a visual UI assistant. Find ALL instances of the
requested element type in the screenshot. Respond ONLY with valid JSON:
[{"x": <int>, "y": <int>, "confidence": <float>, "label": "<text or description>"},...]
Return an empty array [] if none found."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class GroundingMethod(str, Enum):
    VISION_LLM = "vision_llm"
    TEMPLATE   = "template"
    OCR        = "ocr"
    A11Y       = "accessibility"
    STUB       = "stub"


@dataclass
class GroundingResult:
    """Result of a visual element search."""
    found:       bool
    x:           int     = -1
    y:           int     = -1
    confidence:  float   = 0.0
    method:      GroundingMethod = GroundingMethod.STUB
    description: str     = ""
    bbox:        Optional[Tuple[int, int, int, int]] = None   # (x1, y1, x2, y2)
    label:       str     = ""
    error:       str     = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success":     self.found,
            "x":           self.x,
            "y":           self.y,
            "confidence":  round(self.confidence, 3),
            "method":      self.method.value,
            "description": self.description,
            "label":       self.label,
            "error":       self.error,
            "output":      f"({self.x},{self.y}) conf={self.confidence:.2f}" if self.found else self.error,
        }


# ---------------------------------------------------------------------------
# Vision LLM grounding
# ---------------------------------------------------------------------------

class VisionGrounder:
    """
    Find UI elements using the Claude Vision API.
    Returns pixel coordinates from the model's response.
    """

    def __init__(
        self,
        model: str = _VISION_MODEL,
        api_key: str = "",
    ) -> None:
        self._model   = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def find_element(
        self,
        description: str,
        screenshot:  bytes,
        image_type:  str = "png",
    ) -> GroundingResult:
        """Find a single element by description."""
        prompt = f"Find this element in the screenshot: {description}"
        response = self._call_vision(
            prompt, screenshot, image_type, system=_FIND_ELEMENT_SYSTEM
        )
        if response is None:
            return GroundingResult(found=False, method=GroundingMethod.VISION_LLM,
                                   error="Vision API call failed")
        return self._parse_single_response(response)

    def find_all_elements(
        self,
        element_type: str,
        screenshot:   bytes,
        image_type:   str = "png",
    ) -> List[GroundingResult]:
        """Find all elements of a given type."""
        prompt = f"Find all {element_type} elements in the screenshot."
        response = self._call_vision(
            prompt, screenshot, image_type, system=_FIND_ALL_SYSTEM
        )
        if response is None:
            return []
        return self._parse_all_response(response)

    def describe_screenshot(
        self,
        screenshot: bytes,
        image_type: str = "png",
        question:   str = "Describe what you see on this screen.",
    ) -> str:
        """Get a textual description of a screenshot."""
        result = self._call_vision(
            question, screenshot, image_type,
            system="You are a UI analyst. Describe screen content concisely.",
        )
        return result or ""

    def _call_vision(
        self,
        prompt:     str,
        image_data: bytes,
        image_type: str,
        system:     str = "",
    ) -> Optional[str]:
        """Call the vision model via the Operon router."""
        try:
            from core.router import ModelRouter
            from core.config import ConfigManager

            b64 = base64.b64encode(image_data).decode()
            mime = f"image/{image_type.lower().replace('jpg', 'jpeg')}"

            cfg    = ConfigManager()
            router = ModelRouter(cfg)

            content = [
                {"type": "image",
                 "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": prompt},
            ]

            return router.complete(
                system=system,
                messages=[{"role": "user", "content": content}],
                model=self._model,
                max_tokens=512,
            )
        except Exception as e:
            log.warning("VisionGrounder._call_vision failed: %s", e)
            return None

    @staticmethod
    def _parse_single_response(text: str) -> GroundingResult:
        """Parse a JSON response from the vision model."""
        # Strip any markdown fences
        clean = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text.strip())
        # Extract first JSON object
        m = re.search(r"\{[^{}]*\}", clean, re.S)
        if not m:
            return GroundingResult(
                found=False, method=GroundingMethod.VISION_LLM,
                error=f"Could not parse model response: {text[:100]}"
            )
        try:
            d = json.loads(m.group(0))
            x = int(d.get("x", -1))
            y = int(d.get("y", -1))
            conf = float(d.get("confidence", 0.0))
            desc = str(d.get("description", ""))
            if x < 0 or y < 0 or conf < _DEFAULT_CONFIDENCE_THRESHOLD:
                return GroundingResult(
                    found=False, method=GroundingMethod.VISION_LLM,
                    confidence=conf, description=desc,
                    error="Element not found or low confidence"
                )
            return GroundingResult(
                found=True, x=x, y=y, confidence=conf,
                method=GroundingMethod.VISION_LLM, description=desc,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return GroundingResult(
                found=False, method=GroundingMethod.VISION_LLM,
                error=f"JSON parse error: {e}"
            )

    @staticmethod
    def _parse_all_response(text: str) -> List[GroundingResult]:
        """Parse a JSON array response."""
        clean = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text.strip())
        m = re.search(r"\[.*?\]", clean, re.S)
        if not m:
            return []
        try:
            items = json.loads(m.group(0))
            results: List[GroundingResult] = []
            for item in items:
                x    = int(item.get("x", -1))
                y    = int(item.get("y", -1))
                conf = float(item.get("confidence", 0.8))
                label = str(item.get("label", ""))
                if x >= 0 and y >= 0:
                    results.append(GroundingResult(
                        found=True, x=x, y=y, confidence=conf,
                        method=GroundingMethod.VISION_LLM, label=label,
                    ))
            return results
        except (json.JSONDecodeError, ValueError):
            return []


# ---------------------------------------------------------------------------
# Template-matching grounding (OpenCV)
# ---------------------------------------------------------------------------

class TemplateGrounder:
    """
    Find a template image inside a larger screenshot using OpenCV.
    Useful for finding icons, logos, or known button images.
    """

    def find_template(
        self,
        template_path: str,
        screenshot:    bytes,
        threshold:     float = _TEMPLATE_MATCH_THRESHOLD,
    ) -> GroundingResult:
        """
        Find template_path inside screenshot bytes.
        Returns the center coordinates of the best match.
        """
        try:
            import cv2  # type: ignore
            import numpy as np

            # Decode screenshot
            nparr = np.frombuffer(screenshot, np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return GroundingResult(found=False, method=GroundingMethod.TEMPLATE,
                                       error="Failed to decode screenshot")

            # Load template
            tmpl  = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if tmpl is None:
                return GroundingResult(found=False, method=GroundingMethod.TEMPLATE,
                                       error=f"Template not found: {template_path}")

            # Template match
            result = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val < threshold:
                return GroundingResult(
                    found=False, method=GroundingMethod.TEMPLATE,
                    confidence=float(max_val),
                    error=f"Best match confidence {max_val:.2f} below threshold {threshold:.2f}",
                )

            th, tw = tmpl.shape[:2]
            cx = max_loc[0] + tw // 2
            cy = max_loc[1] + th // 2
            return GroundingResult(
                found=True, x=cx, y=cy,
                confidence=float(max_val),
                method=GroundingMethod.TEMPLATE,
                bbox=(max_loc[0], max_loc[1], max_loc[0] + tw, max_loc[1] + th),
                description=f"Template match at ({cx},{cy})",
            )
        except ImportError:
            return GroundingResult(
                found=False, method=GroundingMethod.TEMPLATE,
                error="cv2 (OpenCV) not installed. pip install opencv-python",
            )
        except Exception as e:
            return GroundingResult(
                found=False, method=GroundingMethod.TEMPLATE, error=str(e)
            )

    def find_all_templates(
        self,
        template_path: str,
        screenshot:    bytes,
        threshold:     float = _TEMPLATE_MATCH_THRESHOLD,
        max_results:   int   = 10,
    ) -> List[GroundingResult]:
        """Find all occurrences of a template in a screenshot."""
        try:
            import cv2
            import numpy as np

            nparr = np.frombuffer(screenshot, np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            tmpl  = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if img is None or tmpl is None:
                return []

            result = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
            th, tw = tmpl.shape[:2]

            results: List[GroundingResult] = []
            for _ in range(max_results):
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val < threshold:
                    break
                cx = max_loc[0] + tw // 2
                cy = max_loc[1] + th // 2
                results.append(GroundingResult(
                    found=True, x=cx, y=cy,
                    confidence=float(max_val),
                    method=GroundingMethod.TEMPLATE,
                ))
                # Suppress this match region
                x1 = max(0, max_loc[0] - tw // 2)
                y1 = max(0, max_loc[1] - th // 2)
                x2 = min(result.shape[1], max_loc[0] + tw)
                y2 = min(result.shape[0], max_loc[1] + th)
                result[y1:y2, x1:x2] = 0

            return results
        except Exception as e:
            log.warning("TemplateGrounder.find_all: %s", e)
            return []


# ---------------------------------------------------------------------------
# OCR-based grounding (pytesseract)
# ---------------------------------------------------------------------------

class OCRGrounder:
    """
    Find clickable text on screen using OCR.
    Returns the center coordinates of the matched text bounding box.
    """

    def find_text(
        self,
        text:       str,
        screenshot: bytes,
        margin:     int = _OCR_MARGIN,
        case_sensitive: bool = False,
    ) -> GroundingResult:
        """Find text in screenshot, return its center coordinates."""
        try:
            import pytesseract  # type: ignore
            from PIL import Image

            img  = Image.open(io.BytesIO(screenshot))
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            search = text if case_sensitive else text.lower()

            for i, word in enumerate(data["text"]):
                if not word:
                    continue
                w = word if case_sensitive else word.lower()
                if search in w:
                    conf = float(data["conf"][i])
                    if conf < 0:
                        continue
                    x = data["left"][i] + data["width"][i] // 2
                    y = data["top"][i]  + data["height"][i] // 2
                    return GroundingResult(
                        found=True, x=x, y=y,
                        confidence=conf / 100.0,
                        method=GroundingMethod.OCR,
                        label=word,
                        bbox=(
                            data["left"][i] - margin,
                            data["top"][i]  - margin,
                            data["left"][i] + data["width"][i]  + margin,
                            data["top"][i]  + data["height"][i] + margin,
                        ),
                    )

            return GroundingResult(
                found=False, method=GroundingMethod.OCR,
                error=f"Text '{text}' not found on screen",
            )
        except ImportError:
            return GroundingResult(
                found=False, method=GroundingMethod.OCR,
                error="pytesseract/Pillow not installed. pip install pytesseract pillow",
            )
        except Exception as e:
            return GroundingResult(found=False, method=GroundingMethod.OCR, error=str(e))

    def get_all_text(self, screenshot: bytes) -> List[Dict[str, Any]]:
        """Extract all text blocks with their bounding boxes."""
        try:
            import pytesseract
            from PIL import Image
            img  = Image.open(io.BytesIO(screenshot))
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            blocks: List[Dict[str, Any]] = []
            for i, word in enumerate(data["text"]):
                if word and float(data["conf"][i]) > 0:
                    blocks.append({
                        "text": word,
                        "x": data["left"][i] + data["width"][i] // 2,
                        "y": data["top"][i]  + data["height"][i] // 2,
                        "confidence": float(data["conf"][i]) / 100.0,
                        "bbox": (data["left"][i], data["top"][i],
                                 data["left"][i] + data["width"][i],
                                 data["top"][i]  + data["height"][i]),
                    })
            return blocks
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Accessibility-tree grounding (Playwright)
# ---------------------------------------------------------------------------

class A11YGrounder:
    """
    Find elements using Playwright's accessibility tree.
    More reliable than pixel coordinates for semantic elements.
    """

    def find_by_text(self, page: Any, text: str) -> GroundingResult:
        """Find element by visible text in Playwright page."""
        try:
            elem = page.locator(f"text={text}").first
            box  = elem.bounding_box()
            if not box:
                return GroundingResult(found=False, method=GroundingMethod.A11Y,
                                       error=f"Element with text '{text}' has no bounding box")
            x = int(box["x"] + box["width"] / 2)
            y = int(box["y"] + box["height"] / 2)
            return GroundingResult(
                found=True, x=x, y=y, confidence=0.95,
                method=GroundingMethod.A11Y, label=text,
                bbox=(int(box["x"]), int(box["y"]),
                      int(box["x"] + box["width"]),
                      int(box["y"] + box["height"])),
            )
        except Exception as e:
            return GroundingResult(found=False, method=GroundingMethod.A11Y,
                                   error=str(e))

    def find_by_role(
        self, page: Any, role: str, name: str = ""
    ) -> GroundingResult:
        """Find element by ARIA role and accessible name."""
        try:
            locator = page.get_by_role(role, name=name) if name else page.get_by_role(role)
            box = locator.first.bounding_box()
            if not box:
                return GroundingResult(found=False, method=GroundingMethod.A11Y,
                                       error=f"role={role} name={name!r} has no bbox")
            x = int(box["x"] + box["width"] / 2)
            y = int(box["y"] + box["height"] / 2)
            return GroundingResult(
                found=True, x=x, y=y, confidence=0.95,
                method=GroundingMethod.A11Y, label=f"{role}:{name}",
            )
        except Exception as e:
            return GroundingResult(found=False, method=GroundingMethod.A11Y,
                                   error=str(e))

    def snapshot(self, page: Any) -> Dict[str, Any]:
        """Get the full accessibility tree snapshot."""
        try:
            return page.accessibility.snapshot() or {}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# High-level composite grounder
# ---------------------------------------------------------------------------

class VisualGrounder:
    """
    Composite grounder that tries multiple strategies in order.
    Returns the first confident result.
    """

    def __init__(
        self,
        vision_model:     str   = _VISION_MODEL,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        prefer_vision:    bool  = True,
    ) -> None:
        self._vision    = VisionGrounder(model=vision_model)
        self._template  = TemplateGrounder()
        self._ocr       = OCRGrounder()
        self._a11y      = A11YGrounder()
        self._threshold = confidence_threshold
        self._prefer_vision = prefer_vision

    def find(
        self,
        description: str,
        screenshot:  Optional[bytes] = None,
        page:        Optional[Any]   = None,
        template:    Optional[str]   = None,
    ) -> GroundingResult:
        """
        Find an element using the best available strategy.

        Priority:
          1. Accessibility tree (if page provided) — most reliable
          2. Vision LLM (if screenshot provided + prefer_vision)
          3. OCR text search (if screenshot provided)
          4. Template match (if template path provided)
        """
        # 1. A11Y tree (most precise for web)
        if page is not None:
            result = self._a11y.find_by_text(page, description)
            if result.found and result.confidence >= self._threshold:
                return result

        # 2. Vision LLM
        if screenshot is not None and self._prefer_vision:
            result = self._vision.find_element(description, screenshot)
            if result.found and result.confidence >= self._threshold:
                return result

        # 3. OCR text match (description as text query)
        if screenshot is not None:
            result = self._ocr.find_text(description, screenshot)
            if result.found and result.confidence >= self._threshold:
                return result

        # 4. Template matching
        if template is not None and screenshot is not None:
            result = self._template.find_template(template, screenshot)
            if result.found:
                return result

        return GroundingResult(
            found=False,
            error=f"Element '{description}' not found by any strategy",
        )

    def find_and_click(
        self,
        description: str,
        screenshot:  Optional[bytes],
        page:        Optional[Any] = None,
        delay_ms:    float = 100,
    ) -> Dict[str, Any]:
        """Find element and return click result dict."""
        result = self.find(description, screenshot=screenshot, page=page)
        if not result.found:
            return {"success": False, "error": result.error, "output": result.error}

        if page is not None:
            try:
                time.sleep(delay_ms / 1000)
                page.mouse.click(result.x, result.y)
                return {"success": True, "x": result.x, "y": result.y,
                        "output": f"Clicked ({result.x},{result.y}) — {result.description}"}
            except Exception as e:
                return {"success": False, "error": str(e),
                        "output": f"Click failed: {e}"}

        # Return coordinates for external use
        return result.to_dict()


# ---------------------------------------------------------------------------
# Tool functions (Operon tool API)
# ---------------------------------------------------------------------------

def find_element(
    description:  str,
    screenshot:   Optional[bytes] = None,
    screenshot_b64: str = "",
    page:         Optional[Any] = None,
    template:     str = "",
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Find a UI element by description.

    Args:
        description: Text description of the element to find
        screenshot:  Raw PNG/JPG bytes of the screenshot (optional)
        screenshot_b64: Base64-encoded screenshot (alternative to screenshot)
        page:        Playwright page object for A11Y grounding (optional)
        template:    Path to a template image for template matching (optional)
        confidence_threshold: Minimum confidence (0.0–1.0)

    Returns:
        {success, x, y, confidence, method, description, error}
    """
    if screenshot_b64 and not screenshot:
        try:
            screenshot = base64.b64decode(screenshot_b64)
        except Exception:
            return {"success": False, "error": "Invalid base64 screenshot data",
                    "output": "Invalid base64 screenshot data"}

    grounder = VisualGrounder(confidence_threshold=confidence_threshold)
    result = grounder.find(
        description,
        screenshot=screenshot,
        page=page,
        template=template or None,
    )
    return result.to_dict()


def find_all_elements(
    element_type: str,
    screenshot:   Optional[bytes] = None,
    screenshot_b64: str = "",
) -> Dict[str, Any]:
    """
    Find all elements of a given type on screen.

    Returns:
        {success, elements: [{x, y, confidence, label},...], count, output}
    """
    if screenshot_b64 and not screenshot:
        try:
            screenshot = base64.b64decode(screenshot_b64)
        except Exception:
            return {"success": False, "error": "Invalid base64",
                    "output": "Invalid base64"}

    if not screenshot:
        return {"success": False, "error": "No screenshot provided",
                "output": "No screenshot provided"}

    grounder = VisionGrounder()
    results  = grounder.find_all_elements(element_type, screenshot)
    elements = [r.to_dict() for r in results]
    return {
        "success": True,
        "elements": elements,
        "count": len(elements),
        "output": f"Found {len(elements)} {element_type} element(s)",
    }


def describe_screen(
    screenshot:     Optional[bytes] = None,
    screenshot_b64: str = "",
    question:       str = "What do you see on this screen?",
) -> Dict[str, Any]:
    """
    Get a text description of a screenshot.

    Returns:
        {success, description, output}
    """
    if screenshot_b64 and not screenshot:
        try:
            screenshot = base64.b64decode(screenshot_b64)
        except Exception:
            return {"success": False, "error": "Invalid base64", "output": ""}

    if not screenshot:
        return {"success": False, "error": "No screenshot provided", "output": ""}

    grounder = VisionGrounder()
    desc = grounder.describe_screenshot(screenshot, question=question)
    return {
        "success": bool(desc),
        "description": desc,
        "output": desc,
    }


def ocr_screenshot(
    screenshot:     Optional[bytes] = None,
    screenshot_b64: str = "",
) -> Dict[str, Any]:
    """
    Extract all text from a screenshot using OCR.

    Returns:
        {success, text_blocks: [{text, x, y, confidence},...], output}
    """
    if screenshot_b64 and not screenshot:
        try:
            screenshot = base64.b64decode(screenshot_b64)
        except Exception:
            return {"success": False, "error": "Invalid base64", "output": ""}

    if not screenshot:
        return {"success": False, "error": "No screenshot provided", "output": ""}

    grounder = OCRGrounder()
    blocks   = grounder.get_all_text(screenshot)
    text_out = " ".join(b["text"] for b in blocks)
    return {
        "success": True,
        "text_blocks": blocks,
        "text": text_out,
        "output": f"OCR extracted {len(blocks)} text blocks: {text_out[:200]}",
    }


# ── Tool definitions for LLM registry ────────────────────────────────────────

_TOOL_DEFINITIONS = [
    {
        "name": "find_element",
        "description": (
            "Find a UI element on screen by description. Returns pixel coordinates (x, y) "
            "for clicking. Requires a screenshot (bytes or base64). Uses vision LLM, "
            "OCR, or accessibility tree depending on available inputs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Text description of the element to find (e.g. 'Submit button', 'Search box', 'Login link')",
                },
                "screenshot_b64": {
                    "type": "string",
                    "description": "Base64-encoded screenshot PNG/JPG",
                },
                "template": {
                    "type": "string",
                    "description": "Path to a template image file for template matching (optional)",
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Minimum confidence 0.0–1.0 (default 0.5)",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "find_all_elements",
        "description": "Find all UI elements of a given type in a screenshot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "element_type": {
                    "type": "string",
                    "description": "Type of element to find (e.g. 'button', 'text input', 'link')",
                },
                "screenshot_b64": {
                    "type": "string",
                    "description": "Base64-encoded screenshot",
                },
            },
            "required": ["element_type"],
        },
    },
    {
        "name": "describe_screen",
        "description": "Get a text description of what is visible on a screenshot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "screenshot_b64": {
                    "type": "string",
                    "description": "Base64-encoded screenshot",
                },
                "question": {
                    "type": "string",
                    "description": "Specific question to answer about the screen",
                },
            },
            "required": [],
        },
    },
    {
        "name": "ocr_screenshot",
        "description": "Extract all text from a screenshot using OCR.",
        "input_schema": {
            "type": "object",
            "properties": {
                "screenshot_b64": {
                    "type": "string",
                    "description": "Base64-encoded screenshot",
                },
            },
            "required": [],
        },
    },
]

_DISPATCH = {
    "find_element":     find_element,
    "find_all_elements": find_all_elements,
    "describe_screen":  describe_screen,
    "ocr_screenshot":   ocr_screenshot,
}
