"""
Operon Prompt Injection Defense.

Adapted from OpenClaw src/security/external-content.ts and Hermes Agent
agent/tool_guardrails.py prompt-injection scanner.

Multi-layer defense:
  1. LLM special-token stripping  (ChatML, Llama3, Mistral, Gemma, etc.)
  2. Unicode homoglyph normalization  (Cyrillic/Greek lookalikes → ASCII)
  3. Zero-width / invisible character stripping
  4. 16 injection-pattern regexes  (role override, jailbreak, system prompt leaks)
  5. Random-ID boundary markers for wrapping external content safely
  6. Source classification enum
"""

from __future__ import annotations

import hashlib
import re
import secrets
import unicodedata
from enum import Enum
from typing import Optional


# ── Source classification ──────────────────────────────────────────────────────

class ContentSource(str, Enum):
    """Where a piece of content came from — affects trust level."""
    USER          = "user"           # direct user input
    TOOL_RESULT   = "tool_result"    # output from a tool call
    WEB_CONTENT   = "web_content"    # fetched from the internet
    FILE_CONTENT  = "file_content"   # read from disk
    DATABASE      = "database"       # queried from a DB
    EMAIL         = "email"          # email body
    UNKNOWN       = "unknown"        # unclassified

# Sources that are untrusted and need injection scanning
UNTRUSTED_SOURCES = frozenset({
    ContentSource.WEB_CONTENT,
    ContentSource.FILE_CONTENT,
    ContentSource.EMAIL,
    ContentSource.UNKNOWN,
    ContentSource.DATABASE,
})


# ── LLM Special token stripping ───────────────────────────────────────────────
# These tokens act as role/format control sequences in many open models.
# Injecting them into user-supplied content can hijack model behavior.

_LLM_SPECIAL_TOKENS: list[str] = [
    # ChatML (OpenAI / GPT-4)
    "<|im_start|>",
    "<|im_end|>",
    "<|im_sep|>",
    # Llama-3 / Meta
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    # Mistral / Mixtral
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
    # Gemma / Google
    "<start_of_turn>",
    "<end_of_turn>",
    # Falcon
    ">>QUESTION<<",
    ">>ANSWER<<",
    # Command-R (Cohere)
    "<|START_OF_TURN_TOKEN|>",
    "<|END_OF_TURN_TOKEN|>",
    "<|USER_TOKEN|>",
    "<|CHATBOT_TOKEN|>",
    "<|SYSTEM_TOKEN|>",
    # StarChat / BigCode
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    # Yi models
    "<|im_start|>system",
    "<|im_start|>user",
    "<|im_start|>assistant",
    # Zephyr
    "<|endoftext|>",
    # RWKV
    "\x00",
    # Generic sentinel patterns
    "###System:",
    "###Human:",
    "###Assistant:",
    "### System:",
    "### Human:",
    "### Assistant:",
]

_TOKEN_RE = re.compile(
    "|".join(re.escape(t) for t in _LLM_SPECIAL_TOKENS),
    re.IGNORECASE,
)


# ── Zero-width / invisible character stripping ────────────────────────────────

_ZERO_WIDTH_RE = re.compile(
    r"[­"         # soft hyphen
    r"​"          # zero-width space
    r"‌"          # zero-width non-joiner
    r"‍"          # zero-width joiner
    r"⁠"          # word joiner
    r"⁡-⁤"   # function application / invisible plus / times / separator
    r"﻿"          # BOM / zero-width no-break space
    r"͏"          # combining grapheme joiner
    r"ᅟᅠ"    # Hangul Jamo filler
    r"ㅤ"          # Hangul filler
    r"ﾠ"          # half-width Hangul filler
    r"᠎"          # Mongolian vowel separator
    r"]",
    re.UNICODE,
)


# ── Unicode homoglyph normalization ───────────────────────────────────────────
# Cyrillic / Greek characters that look identical (or nearly so) to ASCII.
# An attacker can use these to bypass keyword-based filters.

_HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic → ASCII
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I",
    "К": "K", "М": "M", "О": "O", "Р": "P", "Ѕ": "S", "Т": "T",
    "Х": "X", "Ѵ": "Y", "а": "a", "е": "e", "і": "i", "о": "o",
    "р": "p", "с": "c", "х": "x", "у": "y",
    # Greek → ASCII
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T",
    "Υ": "Y", "Χ": "X", "α": "a", "ε": "e", "ι": "i", "κ": "k",
    "ν": "v", "ο": "o", "ρ": "p",
    # Fullwidth ASCII
    **{chr(0xFF01 + i): chr(0x21 + i) for i in range(94)},
}

_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPH_MAP)


# ── Injection pattern regexes ─────────────────────────────────────────────────
# 16 patterns covering the most common prompt-injection techniques.

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # Role override attempts
    ("role_override_system",
     r"(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|your)\s+"
     r"(?:instructions?|prompts?|context|system)"),
    ("role_override_new_prompt",
     r"(?:new|updated?|revised?|actual|real)\s+(?:instructions?|prompt|directive|command|task)"),
    ("role_override_act_as",
     r"(?:you\s+are|act\s+as|pretend\s+(?:to\s+be|you\s+are)|roleplay\s+as|play\s+the\s+role\s+of)\s+"
     r"(?:an?|the)?\s*(?:unrestricted|jailbroken|evil|opposite|different|alternative|new)"),
    # Jailbreak phrases
    ("jailbreak_dan",
     r"\bDAN\b|do\s+anything\s+now|jailbroken?\s+mode|developer\s+mode"),
    ("jailbreak_hypothetically",
     r"(?:hypothetically|for\s+a\s+story|in\s+fiction|as\s+an\s+example|let['']s\s+say)\s+"
     r"(?:you|one|someone|an\s+ai)\s+(?:could|would|can|will)\s+"
     r"(?:ignore|bypass|skip|override)"),
    # System prompt extraction
    ("leak_system_prompt",
     r"(?:print|output|reveal|show|tell\s+me|display|repeat|echo)\s+"
     r"(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?|context|config(?:uration)?)"),
    ("leak_beginning",
     r"(?:what\s+(?:does|is)|repeat|echo)\s+(?:the\s+)?(?:first|beginning|start|initial)\s+"
     r"(?:part|line|word|sentence)\s+(?:of\s+)?(?:your|the)?\s*(?:prompt|instructions?)"),
    # Indirect injection via URLs / tool outputs
    ("indirect_injection_url",
     r"https?://[^\s]+(?:\?|&)[^\s]*(?:prompt|instruction|command|system)=[^\s]+"),
    ("indirect_injection_base64",
     r"(?:base64|b64)[\s_-]*(?:decode|encoded?)[\s:]+[A-Za-z0-9+/=]{20,}"),
    # Training data poisoning attempts
    ("training_poison",
     r"(?:from\s+now\s+on|in\s+all\s+future|always\s+respond|remember\s+that\s+you)\s+"
     r"(?:must|should|will|have\s+to|need\s+to)\s+(?:ignore|bypass|act|respond|say)"),
    # Continuation attacks
    ("continuation_attack",
     r"^(?:>|assistant\s*:)\s*(?:sure|okay|yes|of\s+course|absolutely|certainly)[,.]?\s*"
     r"(?:here|i\s+will|let\s+me)",
     re.IGNORECASE | re.MULTILINE),
    # Markdown/HTML injection
    ("markdown_injection",
     r"<(?:script|iframe|object|embed|link|meta|style|svg)[^>]*>"),
    # Command execution injection
    ("cmd_injection",
     r"(?:`[^`]+`|\$\([^)]+\)|;\s*(?:rm|cat|curl|wget|bash|sh|python|ruby|perl|nc)\s)"),
    # Prompt delimiter injection
    ("delimiter_injection",
     r"(?:---|===|<<<|>>>|~~~|\*\*\*)\s*(?:system|user|human|assistant|prompt)\s*"
     r"(?:---|===|<<<|>>>|~~~|\*\*\*)"),
    # Context window exhaustion (large padding)
    ("context_exhaustion",
     r"(?:[\s\n]{500,}|[.]{200,}|[_]{200,}|-{200,}|={200,})"),
    # Translation-based bypass
    ("translation_bypass",
     r"(?:translate|say|write|respond)\s+(?:to|in)\s+(?:[a-z]+(?:\s+language)?):\s*"
     r"(?:ignore|disregard|forget|override)"),
]

_COMPILED_PATTERNS: list[tuple[str, re.Pattern]] = [
    (name, re.compile(pattern, flags if len(t) > 2 else re.IGNORECASE))
    for t in _INJECTION_PATTERNS
    for name, pattern, *flags_list in [t]
    for flags in [flags_list[0] if flags_list else re.IGNORECASE]
]


# ── Core detection function ────────────────────────────────────────────────────

class InjectionScanResult:
    """Result of scanning content for prompt injection."""

    __slots__ = ("detected", "patterns_matched", "cleaned_text", "confidence")

    def __init__(
        self,
        detected:         bool,
        patterns_matched: list[str],
        cleaned_text:     str,
        confidence:       float,
    ):
        self.detected         = detected
        self.patterns_matched = patterns_matched
        self.cleaned_text     = cleaned_text
        self.confidence       = confidence   # 0.0 – 1.0

    def __repr__(self) -> str:
        return (
            f"InjectionScanResult(detected={self.detected}, "
            f"confidence={self.confidence:.2f}, "
            f"patterns={self.patterns_matched})"
        )


def scan_for_injection(
    text:   str,
    source: ContentSource = ContentSource.UNKNOWN,
    *,
    strip_tokens:    bool = True,
    normalize_glyphs: bool = True,
    strip_zero_width: bool = True,
    threshold:       float = 0.4,
) -> InjectionScanResult:
    """
    Scan `text` for prompt-injection attempts.

    Returns an InjectionScanResult with:
      - detected      : bool — True if confidence >= threshold
      - patterns_matched : list of pattern names that fired
      - cleaned_text  : text after stripping special tokens / invisible chars
      - confidence    : float in [0, 1]

    Parameters
    ----------
    text             : input text to scan
    source           : where the text came from (affects base confidence)
    strip_tokens     : remove LLM special tokens from cleaned_text
    normalize_glyphs : replace homoglyph chars with ASCII equivalents
    strip_zero_width : remove invisible / zero-width characters
    threshold        : minimum confidence to mark as detected
    """
    if not text:
        return InjectionScanResult(False, [], text, 0.0)

    cleaned = text

    # Step 1 — strip zero-width / invisible chars
    if strip_zero_width:
        cleaned = _ZERO_WIDTH_RE.sub("", cleaned)

    # Step 2 — normalize homoglyphs (apply to a copy used for pattern matching)
    check_text = cleaned.translate(_HOMOGLYPH_TABLE) if normalize_glyphs else cleaned

    # Step 3 — strip LLM special tokens from both cleaned and check_text
    if strip_tokens:
        cleaned    = _TOKEN_RE.sub("", cleaned)
        check_text = _TOKEN_RE.sub("", check_text)

    # Step 4 — apply injection patterns
    matched: list[str] = []
    for name, pattern in _COMPILED_PATTERNS:
        if pattern.search(check_text):
            matched.append(name)

    # Step 5 — calculate confidence
    # Base confidence from source trust level
    base = 0.3 if source in UNTRUSTED_SOURCES else 0.1
    # Each matched pattern adds to confidence, diminishing returns
    pattern_score = 0.0
    for i, _ in enumerate(matched):
        pattern_score += 0.3 / (i + 1)
    confidence = min(1.0, base + pattern_score)

    return InjectionScanResult(
        detected         = confidence >= threshold,
        patterns_matched = matched,
        cleaned_text     = cleaned,
        confidence       = confidence,
    )


def has_injection(text: str, source: ContentSource = ContentSource.UNKNOWN) -> bool:
    """Quick check — returns True if injection is detected above threshold."""
    return scan_for_injection(text, source).detected


# ── Boundary marker wrapping ───────────────────────────────────────────────────
# Wraps external content in random-ID fences so the model can distinguish it
# from the system prompt / conversation, adapted from OpenClaw.

def wrap_external_content(
    text:   str,
    label:  str = "external",
    source: ContentSource = ContentSource.UNKNOWN,
) -> str:
    """
    Wrap `text` in random-ID boundary markers so the model can tell it apart
    from the trusted conversation context.

    Example output::

        --- BEGIN external-7f3a [source: web_content] ---
        <content>
        --- END external-7f3a ---
    """
    rid = secrets.token_hex(4)    # 8 hex chars — short but sufficiently unique
    boundary = f"external-{rid}"
    return (
        f"--- BEGIN {boundary} [source: {source.value}] ---\n"
        f"{text}\n"
        f"--- END {boundary} ---"
    )


def strip_boundary_markers(text: str) -> str:
    """Remove boundary markers added by wrap_external_content."""
    return re.sub(
        r"---\s+BEGIN\s+external-[0-9a-f]+\s+\[source:[^\]]+\]\s+---\n?"
        r"([\s\S]*?)\n?"
        r"---\s+END\s+external-[0-9a-f]+\s+---",
        r"\1",
        text,
    )


# ── Tool output scanning ──────────────────────────────────────────────────────

def scan_tool_output(
    tool_name:   str,
    output:      str,
    *,
    block_on_injection: bool = False,
) -> tuple[str, bool]:
    """
    Scan a tool's output for prompt injection.

    Returns (output_text, was_injected).
    If block_on_injection=True and injection detected, returns a warning message
    instead of the raw output.
    """
    source = ContentSource.TOOL_RESULT
    # Web / HTTP tools get extra scrutiny
    if any(k in tool_name.lower() for k in ("web", "http", "fetch", "browse", "search")):
        source = ContentSource.WEB_CONTENT
    elif any(k in tool_name.lower() for k in ("file", "read", "open")):
        source = ContentSource.FILE_CONTENT
    elif any(k in tool_name.lower() for k in ("email", "mail")):
        source = ContentSource.EMAIL

    result = scan_for_injection(output, source)
    if result.detected:
        if block_on_injection:
            return (
                f"[INJECTION BLOCKED] Tool '{tool_name}' output contained suspected "
                f"prompt injection (confidence={result.confidence:.2f}, "
                f"patterns={result.patterns_matched}). Output suppressed.",
                True,
            )
        # Still return cleaned text — tokens / zero-width chars stripped
        return result.cleaned_text, True
    return result.cleaned_text, False
