"""
Operon Configuration Manager.
Stores config in ~/.operon/config.json — never in the project directory.
"""

import json
import os
from pathlib import Path

CONFIG_DIR  = Path.home() / ".operon"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Canonical provider → base URL mapping
PROVIDER_URLS = {
    "openai":     "https://api.openai.com/v1/chat/completions",
    "anthropic":  "https://api.anthropic.com/v1/messages",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    # ── Local runners (no API key required) ───────────────────────────────────
    "ollama":     "http://localhost:11434/v1/chat/completions",
    "lmstudio":   "http://localhost:1234/v1/chat/completions",
    "jan":        "http://localhost:1337/v1/chat/completions",
    "local":      "http://localhost:11434/v1/chat/completions",  # alias for ollama
}

# Health-check URLs for local providers (GET → 200 means server is running)
LOCAL_HEALTH_URLS = {
    "ollama":   "http://localhost:11434/api/tags",
    "lmstudio": "http://localhost:1234/v1/models",
    "jan":      "http://localhost:1337/v1/models",
    "local":    "http://localhost:11434/api/tags",
}

# Providers that need no API key
LOCAL_PROVIDERS = {"ollama", "lmstudio", "jan", "local"}

# Bundled model profiles: name → {provider, model_id}
DEFAULT_PROFILES = {
    # ── Cloud ─────────────────────────────────────────────────────────────────
    "gpt-4o":                 {"provider": "openai",     "model_id": "gpt-4o"},
    "gpt-4o-mini":            {"provider": "openai",     "model_id": "gpt-4o-mini"},
    "gpt-4-turbo":            {"provider": "openai",     "model_id": "gpt-4-turbo"},
    "claude-3-5-sonnet":      {"provider": "anthropic",  "model_id": "claude-3-5-sonnet-20241022"},
    "claude-3-5-haiku":       {"provider": "anthropic",  "model_id": "claude-3-5-haiku-20241022"},
    "claude-opus-4":          {"provider": "anthropic",  "model_id": "claude-opus-4-5"},
    "claude-sonnet-4":        {"provider": "anthropic",  "model_id": "claude-sonnet-4-5"},
    "mistral-large":          {"provider": "openrouter", "model_id": "mistralai/mistral-large"},
    "llama-3.1-70b":          {"provider": "openrouter", "model_id": "meta-llama/llama-3.1-70b-instruct"},
    "deepseek-v3":            {"provider": "openrouter", "model_id": "deepseek/deepseek-chat"},
    "gemini-pro-1.5":         {"provider": "openrouter", "model_id": "google/gemini-pro-1.5"},
    "qwen-2.5-72b":           {"provider": "openrouter", "model_id": "qwen/qwen-2.5-72b-instruct"},
    # ── Local — Ollama (pre-built; model must be pulled first) ────────────────
    "ollama:llama3.2":        {"provider": "ollama",     "model_id": "llama3.2"},
    "ollama:llama3.1":        {"provider": "ollama",     "model_id": "llama3.1"},
    "ollama:mistral":         {"provider": "ollama",     "model_id": "mistral"},
    "ollama:codellama":       {"provider": "ollama",     "model_id": "codellama"},
    "ollama:deepseek-r1":     {"provider": "ollama",     "model_id": "deepseek-r1"},
    "ollama:qwen2.5":         {"provider": "ollama",     "model_id": "qwen2.5"},
    "ollama:phi4":            {"provider": "ollama",     "model_id": "phi4"},
    "ollama:gemma3":          {"provider": "ollama",     "model_id": "gemma3"},
    # ── Local — LM Studio (model_id = whatever is loaded in LM Studio) ────────
    "lmstudio:loaded":        {"provider": "lmstudio",   "model_id": "local-model"},
}

_DEFAULTS = {
    "configured":        False,
    "default_model":     "gpt-4o",
    "memory_enabled":    True,
    "max_tool_iters":    12,
    "request_timeout":   120,
    "model_profiles":    DEFAULT_PROFILES,
    "api_keys":          {
        "openai":     "",
        "anthropic":  "",
        "openrouter": "",
    },
    "active_provider":   "openai",
}


class ConfigManager:

    def __init__(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self._cfg = self._load()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                # Backfill any missing defaults
                for k, v in _DEFAULTS.items():
                    if k not in data:
                        data[k] = v
                # Backfill missing profiles
                for name, profile in DEFAULT_PROFILES.items():
                    data.setdefault("model_profiles", {})[name] = profile
                return data
            except json.JSONDecodeError:
                import sys
                print(
                    f"  [Config] Warning: {CONFIG_FILE} is corrupted — "
                    "starting with defaults. Run /setup to reconfigure.",
                    file=sys.stderr,
                )
            except Exception:
                pass
        # First run — use defaults
        return dict(_DEFAULTS)

    def save(self) -> None:
        with open(CONFIG_FILE, "w") as f:
            json.dump(self._cfg, f, indent=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, key: str, default=None):
        return self._cfg.get(key, default)

    def set(self, key: str, value) -> None:
        self._cfg[key] = value
        self.save()

    def is_configured(self) -> bool:
        return bool(self._cfg.get("configured", False))

    def get_api_key(self, provider: str) -> str:
        keys = self._cfg.get("api_keys", {})
        # Also allow environment variable overrides
        env_map = {
            "openai":     "OPENAI_API_KEY",
            "anthropic":  "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_key = os.environ.get(env_map.get(provider, ""), "")
        return env_key or keys.get(provider, "")

    def set_api_key(self, provider: str, key: str) -> None:
        if "api_keys" not in self._cfg:
            self._cfg["api_keys"] = {}
        self._cfg["api_keys"][provider] = key
        self.save()

    def resolve_model(self, name: str) -> dict:
        """Return {provider, model_id, api_key} for the given profile name."""
        profiles = self._cfg.get("model_profiles", DEFAULT_PROFILES)
        if name in profiles:
            profile = profiles[name]
            provider = profile["provider"]
            return {
                "provider": provider,
                "model_id": profile["model_id"],
                "api_key":  self.get_api_key(provider),
            }
        # Explicit local prefix: "ollama:phi4", "lmstudio:my-model", etc.
        for local_prov in LOCAL_PROVIDERS:
            prefix = f"{local_prov}:"
            if name.lower().startswith(prefix):
                model_id = name[len(prefix):]
                return {"provider": local_prov, "model_id": model_id, "api_key": ""}
        # Fallback: treat `name` as a raw model id, guess provider
        provider = "openai"
        if "claude" in name.lower():
            provider = "anthropic"
        elif "/" in name:
            provider = "openrouter"
        return {
            "provider": provider,
            "model_id": name,
            "api_key":  self.get_api_key(provider),
        }

    def get_safe_display(self) -> dict:
        display = {}
        for k, v in self._cfg.items():
            if k == "api_keys":
                masked = {}
                for prov, key in v.items():
                    masked[prov] = ("●●●●" + key[-4:]) if len(key) > 4 else ("SET" if key else "NOT SET")
                display["api_keys"] = masked
            elif k == "model_profiles":
                display[k] = f"({len(v)} profiles)"
            else:
                display[k] = v
        return display
