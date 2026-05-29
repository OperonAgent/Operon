"""
Operon Multi-Provider Model Router.

Routes API payloads to OpenAI, Anthropic, or OpenRouter.
Features:
  • Pure requests — no SDK hard-dependency
  • JSON response extraction + auto-repair
  • Graceful error messages with actionable hints
  • API key rotation — round-robin across multiple keys on rate-limit
    (set OPENAI_API_KEYS, ANTHROPIC_API_KEYS as comma-separated env vars,
     or add api_keys list to model config)
"""

import json
import os
import re
import time
import requests
from typing import Optional

from core.config import ConfigManager, PROVIDER_URLS, LOCAL_PROVIDERS

# Retry settings for rate-limit and transient server errors
_RETRY_MAX      = 3


# ── API key rotation pool ──────────────────────────────────────────────────────

class _KeyPool:
    """Round-robin pool of API keys for a provider."""

    def __init__(self, keys: list[str]) -> None:
        self._keys  = [k for k in keys if k]
        self._index = 0

    def __len__(self) -> int:
        return len(self._keys)

    def current(self) -> Optional[str]:
        if not self._keys:
            return None
        return self._keys[self._index % len(self._keys)]

    def rotate(self) -> Optional[str]:
        """Advance to next key and return it."""
        if not self._keys:
            return None
        self._index = (self._index + 1) % len(self._keys)
        return self._keys[self._index]

    def all_keys(self) -> list[str]:
        return list(self._keys)


def _load_key_pool(provider: str, primary_key: str) -> _KeyPool:
    """
    Load all configured keys for a provider.

    Reads from:
      1. Environment variable  <PROVIDER>_API_KEYS  (comma-separated)
      2. The primary key passed in
    """
    env_var   = f"{provider.upper()}_API_KEYS"
    env_value = os.environ.get(env_var, "")
    extra     = [k.strip() for k in env_value.split(",") if k.strip()]

    # Deduplicate while preserving order; primary key first
    seen: set[str] = set()
    keys: list[str] = []
    for k in ([primary_key] + extra):
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return _KeyPool(keys)


def _apply_message_cache_control(messages: list[dict], n: int = 3) -> list[dict]:
    """
    Mark the last `n` messages with Anthropic cache_control so prefix-cache
    breakpoints are set at the most recent conversation turns.
    Content is converted from plain string to a content-block list when needed.
    """
    if not messages:
        return messages
    patched = list(messages)
    marked = 0
    for i in range(len(patched) - 1, -1, -1):
        if marked >= n:
            break
        msg = patched[i]
        content = msg.get("content", "")
        if isinstance(content, str):
            patched[i] = {
                "role": msg["role"],
                "content": [{"type": "text", "text": content,
                              "cache_control": {"type": "ephemeral"}}],
            }
            marked += 1
        elif isinstance(content, list) and content:
            new_content = list(content)
            last_block = dict(new_content[-1])
            last_block["cache_control"] = {"type": "ephemeral"}
            new_content[-1] = last_block
            patched[i] = dict(msg)
            patched[i]["content"] = new_content
            marked += 1
    return patched
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_RETRY_BASE     = 1.5   # seconds — doubled each attempt


class ModelRouter:

    def __init__(self, config: ConfigManager):
        self._config = config
        # Last call's token usage — populated after every successful API call.
        # Keys: input_tokens (int), output_tokens (int), model (str), provider (str)
        self.last_usage: dict = {}
        # stop_reason from the last Anthropic call — "end_turn" | "max_tokens" | etc.
        self._last_stop_reason: str = "end_turn"
        # API key pools — keyed by provider, lazily initialized
        self._key_pools: dict[str, _KeyPool] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def complete(self, system: str, messages: list[dict]) -> Optional[str]:
        """
        Send a chat completion request to the active model.
        Returns the raw response text, or None on failure.
        """
        model_name = self._config.get("default_model", "gpt-4o")
        info       = self._config.resolve_model(model_name)
        provider   = info["provider"]
        model_id   = info["model_id"]
        api_key    = info["api_key"]
        timeout    = self._config.get("request_timeout", 120)

        is_local = provider in LOCAL_PROVIDERS
        if not api_key and not is_local:
            print(f"\n  [Router] No API key for provider '{provider}'. Run /setup or set the env var.")
            return None

        # Build / reuse key pool for this provider
        if provider not in self._key_pools:
            self._key_pools[provider] = _load_key_pool(provider, api_key or "")
        pool = self._key_pools[provider]
        active_key = pool.current() or api_key or ""

        last_exc = None
        for attempt in range(1, _RETRY_MAX + 1):
            try:
                if provider == "anthropic":
                    return self._call_anthropic(active_key, model_id, system, messages, timeout)
                else:
                    base_url = PROVIDER_URLS.get(provider, PROVIDER_URLS["openai"])
                    return self._call_openai_compat(active_key, model_id, system, messages, timeout, base_url, provider)

            except requests.exceptions.Timeout:
                print(
                    f"\n  ⏱  Model took too long — request timed out after {timeout}s.\n"
                    f"  Try a shorter prompt, a faster model (/model), or raise the\n"
                    f"  timeout with /setup → Step 5 (current: {timeout}s)."
                )
                return None

            except requests.exceptions.ConnectionError as e:
                print(f"\n  [Router] Connection error: {e}")
                return None

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                body   = e.response.text[:300]  if e.response is not None else ""
                if status in _RETRY_STATUSES and attempt < _RETRY_MAX:
                    wait = _RETRY_BASE * (2 ** (attempt - 1))
                    if status == 429:
                        # Honor Retry-After header if present
                        ra = e.response.headers.get("Retry-After") if e.response is not None else None
                        if ra:
                            try:
                                wait = float(ra)
                            except ValueError:
                                pass
                        # Key rotation on rate-limit — try the next key immediately
                        if len(pool) > 1:
                            rotated = pool.rotate()
                            active_key = rotated or active_key
                            print(f"\n  [Router] Rate-limited — rotating to next API key (attempt {attempt}/{_RETRY_MAX})…")
                            time.sleep(min(wait, 2.0))   # shorter wait when rotating
                        else:
                            print(f"\n  [Router] HTTP {status} — retrying in {wait:.1f}s (attempt {attempt}/{_RETRY_MAX})…")
                            time.sleep(wait)
                    else:
                        print(f"\n  [Router] HTTP {status} — retrying in {wait:.1f}s (attempt {attempt}/{_RETRY_MAX})…")
                        time.sleep(wait)
                    last_exc = e
                    continue
                print(f"\n  [Router] HTTP {status}: {body}")
                return None

            except Exception as e:
                print(f"\n  [Router] Unexpected error: {e}")
                return None

        # Exhausted retries
        if last_exc is not None:
            status = last_exc.response.status_code if last_exc.response is not None else "?"
            print(f"\n  [Router] Gave up after {_RETRY_MAX} retries (last HTTP {status}).")
        return None

    # ── Streaming completion ──────────────────────────────────────────────────

    def stream_complete(
        self,
        system:   str,
        messages: list[dict],
    ):
        """
        Stream a chat completion, yielding text chunks as they arrive.

        Usage:
            chunks = []
            for chunk in router.stream_complete(system, messages):
                print(chunk, end="", flush=True)
                chunks.append(chunk)
            full_text = "".join(chunks)

        Falls back to non-streaming complete() on providers that don't
        support SSE (returns the full text as a single chunk).
        """
        model_name = self._config.get("default_model", "gpt-4o")
        info       = self._config.resolve_model(model_name)
        provider   = info["provider"]
        model_id   = info["model_id"]
        api_key    = info["api_key"]
        timeout    = self._config.get("request_timeout", 120)
        is_local   = provider in LOCAL_PROVIDERS

        if not api_key and not is_local:
            return  # nothing to stream

        try:
            if provider == "anthropic":
                yield from self._stream_anthropic(api_key, model_id, system, messages, timeout)
            else:
                base_url = PROVIDER_URLS.get(provider, PROVIDER_URLS["openai"])
                yield from self._stream_openai_compat(
                    api_key, model_id, system, messages, timeout, base_url, provider
                )
        except Exception as e:
            # On any streaming error, fall back to the full non-streaming path
            import logging as _log
            _log.getLogger("operon.router").warning("Streaming failed (%s), falling back to non-streaming: %s", provider, e)
            result = self.complete(system=system, messages=messages)
            if result:
                yield result

    def _stream_openai_compat(
        self,
        api_key: str, model_id: str, system: str,
        messages: list[dict], timeout: int, base_url: str, provider: str,
    ):
        """Stream SSE chunks from an OpenAI-compatible endpoint."""
        import json as _json

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://operon.local"
            headers["X-Title"]      = "Operon Terminal Cockpit"

        payload = {
            "model":       model_id,
            "messages":    [{"role": "system", "content": system}] + messages,
            "max_tokens":  4096,
            "temperature": 0.2,
            "stream":      True,
        }

        # Local / smaller providers often don't support JSON mode with streaming
        _JSON_MODE_TAGS  = ("gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o-mini")
        _NO_JSON_MODE_TAGS = ("o1", "o3", "o4")
        if provider == "openai" and any(t in model_id for t in _JSON_MODE_TAGS) \
                and not any(t in model_id for t in _NO_JSON_MODE_TAGS):
            payload["response_format"] = {"type": "json_object"}

        full_text  = []
        input_tok  = 0
        output_tok = 0
        stop_reason = "stop"

        with requests.post(base_url, headers=headers, json=payload,
                           timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() == "[DONE]":
                    break
                try:
                    obj = _json.loads(line)
                except Exception:
                    continue

                # Extract usage if present (sometimes in last chunk)
                usage = obj.get("usage") or {}
                if usage:
                    input_tok  = usage.get("prompt_tokens", input_tok)
                    output_tok = usage.get("completion_tokens", output_tok)

                choices = obj.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta  = choice.get("delta", {})
                chunk  = delta.get("content") or ""
                if chunk:
                    full_text.append(chunk)
                    yield chunk
                fr = choice.get("finish_reason")
                if fr:
                    stop_reason = fr

        self.last_usage = {
            "input_tokens":  input_tok,
            "output_tokens": output_tok or len("".join(full_text)) // 4,
            "model":    model_id,
            "provider": provider,
        }
        self._last_stop_reason = stop_reason

    def _stream_anthropic(
        self,
        api_key: str, model_id: str, system: str,
        messages: list[dict], timeout: int,
    ):
        """Stream SSE chunks from the Anthropic Messages API."""
        import json as _json

        headers = {
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta":    "prompt-caching-2024-07-31",
            "Content-Type":      "application/json",
        }

        normalized = self._normalize_for_anthropic(messages)
        system_blocks  = [{"type": "text", "text": system,
                           "cache_control": {"type": "ephemeral"}}]
        cached_messages = _apply_message_cache_control(normalized, n=3)

        payload = {
            "model":      model_id,
            "max_tokens": 4096,
            "system":     system_blocks,
            "messages":   cached_messages,
            "stream":     True,
        }

        input_tok  = 0
        output_tok = 0
        cache_read = 0
        cache_write = 0
        stop_reason = "end_turn"
        event_type  = ""

        with requests.post(
            PROVIDER_URLS["anthropic"], headers=headers, json=payload,
            timeout=timeout, stream=True,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")

                if line.startswith("event: "):
                    event_type = line[7:].strip()
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        obj = _json.loads(data_str)
                    except Exception:
                        continue

                    etype = obj.get("type", event_type)

                    if etype == "message_start":
                        usage  = obj.get("message", {}).get("usage", {})
                        input_tok   = usage.get("input_tokens", 0)
                        cache_read  = usage.get("cache_read_input_tokens", 0)
                        cache_write = usage.get("cache_creation_input_tokens", 0)

                    elif etype == "content_block_delta":
                        delta = obj.get("delta", {})
                        chunk = delta.get("text") or ""
                        if chunk:
                            yield chunk

                    elif etype == "message_delta":
                        delta = obj.get("delta", {})
                        stop_reason = delta.get("stop_reason") or stop_reason
                        usage  = obj.get("usage", {})
                        output_tok = usage.get("output_tokens", 0)

                    elif etype == "message_stop":
                        break

        self.last_usage = {
            "input_tokens":       input_tok,
            "output_tokens":      output_tok,
            "cache_read_tokens":  cache_read,
            "cache_write_tokens": cache_write,
            "model":    model_id,
            "provider": "anthropic",
        }
        self._last_stop_reason = stop_reason

    # ── Provider implementations ──────────────────────────────────────────────

    def _call_openai_compat(
        self, api_key: str, model_id: str, system: str,
        messages: list[dict], timeout: int, base_url: str, provider: str
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://operon.local"
            headers["X-Title"]      = "Operon Terminal Cockpit"

        payload: dict = {
            "model":      model_id,
            "messages":   [{"role": "system", "content": system}] + messages,
            "max_tokens": 4096,
            "temperature": 0.2,
        }

        # Enable native JSON mode for OpenAI models that support it
        # (local providers and OpenRouter do not reliably support response_format)
        _JSON_MODE_TAGS = (
            "gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o-mini",
        )
        _NO_JSON_MODE_TAGS = ("o1", "o3", "o4")   # reasoning models use different API
        if provider == "openai" and any(tag in model_id for tag in _JSON_MODE_TAGS) \
                and not any(tag in model_id for tag in _NO_JSON_MODE_TAGS):
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        self.last_usage = {
            "input_tokens":  usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "model":    model_id,
            "provider": provider,
        }
        choice = data["choices"][0]
        self._last_stop_reason = choice.get("finish_reason", "stop") or "stop"
        return choice["message"]["content"]

    def _call_anthropic(
        self, api_key: str, model_id: str, system: str,
        messages: list[dict], timeout: int
    ) -> str:
        headers = {
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta":    "prompt-caching-2024-07-31",
            "Content-Type":      "application/json",
        }
        # Anthropic requires alternating user/assistant turns
        # and the first message must be from the user.
        normalized = self._normalize_for_anthropic(messages)

        # Prompt caching: system_and_3 strategy (up to 4 cache breakpoints).
        # Wrap system as a content-block list so we can attach cache_control,
        # then mark the last 3 messages for caching too.
        system_blocks = [{"type": "text", "text": system,
                          "cache_control": {"type": "ephemeral"}}]
        cached_messages = _apply_message_cache_control(normalized, n=3)

        payload = {
            "model":      model_id,
            "max_tokens": 4096,
            "system":     system_blocks,
            "messages":   cached_messages,
        }

        resp = requests.post(
            PROVIDER_URLS["anthropic"],
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        self.last_usage = {
            "input_tokens":        usage.get("input_tokens", 0),
            "output_tokens":       usage.get("output_tokens", 0),
            "cache_read_tokens":   usage.get("cache_read_input_tokens", 0),
            "cache_write_tokens":  usage.get("cache_creation_input_tokens", 0),
            "model":    model_id,
            "provider": "anthropic",
        }
        # Expose stop_reason so the caller can detect max_tokens truncation
        self._last_stop_reason = data.get("stop_reason", "end_turn")
        return data["content"][0]["text"]

    @staticmethod
    def _normalize_for_anthropic(messages: list[dict]) -> list[dict]:
        """
        Anthropic strictly requires alternating user/assistant messages
        starting with 'user'.  Merge consecutive same-role messages and
        ensure the list starts with a user turn.

        KV cache normalization: embedded JSON in assistant messages is
        re-serialized with sort_keys=True so identical tool calls produce
        identical byte sequences across turns — maximizing Anthropic prefix
        cache hits.  (Adapted from Hermes Agent conversation_loop.py.)
        """
        import json as _json
        if not messages:
            return [{"role": "user", "content": "(start)"}]

        def _normalize_content(role: str, content: str) -> str:
            if role != "assistant":
                return content
            try:
                obj = _json.loads(content)
                return _json.dumps(obj, ensure_ascii=False,
                                   sort_keys=True, separators=(",", ":"))
            except Exception:
                return content

        merged: list[dict] = []
        for msg in messages:
            role    = msg["role"]
            content = _normalize_content(role, msg["content"])
            # Treat tool results embedded as user messages
            if role == "system":
                role = "user"
            if merged and merged[-1]["role"] == role:
                merged[-1]["content"] += "\n\n" + content
            else:
                merged.append({"role": role, "content": content})

        if merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "(begin session)"})

        # Ensure final message is from user (model needs to respond)
        # If last message is assistant, that's fine — the API call will extend it.
        return merged

    # ── JSON extraction & repair ──────────────────────────────────────────────

    @staticmethod
    def _extract_first_json(text: str) -> Optional[str]:
        """
        Walk character-by-character to extract the first *complete* balanced
        JSON object `{…}` or array `[…]` from text.

        Unlike a greedy regex, this stops at the correct closing bracket even
        when multiple JSON objects appear back-to-back in the same response.
        String contents (including escaped chars) are handled correctly.
        """
        for open_ch, close_ch in (('{', '}'), ('[', ']')):
            start = text.find(open_ch)
            if start == -1:
                continue
            depth     = 0
            in_str    = False
            escaped   = False
            for i, ch in enumerate(text[start:], start):
                if escaped:
                    escaped = False
                    continue
                if ch == '\\' and in_str:
                    escaped = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        return None

    @staticmethod
    def parse_response(text: str) -> Optional[dict]:
        """
        Extract a JSON object from the model's response.
        Handles:
          1. Pure JSON
          2. JSON inside ```json … ``` fences
          3. Multiple JSON objects in one response — picks the FIRST complete one
          4. JSON embedded in prose
          5. Trailing commas
          6. Python-style True/False/None
          7. Single-quoted strings
        """
        if not text:
            return None

        text = text.strip()

        def _as_dict(obj) -> Optional[dict]:
            """Return obj if it is a dict; unwrap single-element lists; else None."""
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list):
                return next((i for i in obj if isinstance(i, dict)), None)
            return None   # string, int, bool, None → not a valid response dict

        # Pass 1 — direct parse (covers clean single-object responses)
        try:
            result = json.loads(text)
            d = _as_dict(result)
            if d is not None:
                return d
            # Parsed OK but not a dict/list-of-dicts — fall through to extraction
        except json.JSONDecodeError:
            pass

        # Pass 2 — strip markdown fences
        fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                d = _as_dict(json.loads(candidate))
                if d is not None:
                    return d
            except json.JSONDecodeError:
                pass
            text = candidate  # keep repairing on the fenced block

        # Pass 3 — balanced-brace extraction (handles multiple JSON objects
        #           back-to-back; always picks the FIRST complete one)
        extracted = ModelRouter._extract_first_json(text)
        if extracted:
            try:
                d = _as_dict(json.loads(extracted))
                if d is not None:
                    return d
            except json.JSONDecodeError:
                pass
            text = extracted   # hand off to repair passes

        # Pass 4 — repair trailing commas
        repaired = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            d = _as_dict(json.loads(repaired))
            if d is not None:
                return d
        except json.JSONDecodeError:
            pass

        # Pass 5 — Python literal fixups (True, False, None)
        py_fixed = (
            repaired
            .replace(": True",  ": true")
            .replace(": False", ": false")
            .replace(": None",  ": null")
            .replace(":True",   ":true")
            .replace(":False",  ":false")
            .replace(":None",   ":null")
        )
        try:
            d = _as_dict(json.loads(py_fixed))
            if d is not None:
                return d
        except json.JSONDecodeError:
            pass

        # Pass 6 — single-quoted → double-quoted strings
        # Guard: protect apostrophes that sit BETWEEN letters (contractions like
        # don't, it's, can't, I'm, they're) before running the quote-swap regex.
        # Without this, `'([^']*)'` matches `'I don'` (stopping at the apostrophe
        # in "don't") and produces `"I don"` — truncating the response.
        _APOS_GUARD  = ""   # Unicode private-use char — won't appear in output
        py_protected = re.sub(r"(?<=[A-Za-z])'(?=[A-Za-z])", _APOS_GUARD, py_fixed)
        single_fixed = re.sub(r"'([^']*)'", r'"\1"', py_protected)
        single_fixed = single_fixed.replace(_APOS_GUARD, "'")   # restore before parsing
        try:
            d = _as_dict(json.loads(single_fixed))
            if d is not None:
                return d
        except json.JSONDecodeError:
            pass

        return None   # give up; caller handles raw-text fallback
