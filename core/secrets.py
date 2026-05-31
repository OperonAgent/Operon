"""
Operon Secrets Manager — OS-keychain-backed credential storage.

Priority chain (most secure → least secure):
  1. OS Keychain   — macOS Keychain / Linux Secret Service / Windows Credential Manager
                     via the `keyring` library (pip install keyring).
  2. Fernet        — AES-128 symmetric encryption stored in ~/.operon/secrets.enc
                     via the `cryptography` library (pip install cryptography).
                     Encryption key derived from machine-unique bytes + a salt.
  3. Plain JSON    — Falls back to the existing ~/.operon/knowledge.json behaviour,
                     but prints a warning advising users to install keyring.

Usage
-----
    from core.secrets import SecretsManager

    sm = SecretsManager()
    sm.set("app_password", "mypassword")
    pw = sm.get("app_password")
    sm.delete("app_password")
    sm.list_keys()         → ["app_password", "sender_email", ...]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_OPERON_DIR = Path.home() / ".operon"
_OPERON_APP = "operon"


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _try_keyring() -> bool:
    """Return True if keyring is available and functional."""
    try:
        import keyring as _kr
        # Probe — some headless environments raise RuntimeError on get_password
        _kr.get_password(_OPERON_APP, "__probe__")
        return True
    except Exception:
        return False


def _try_fernet() -> bool:
    """Return True if cryptography (Fernet) is available."""
    try:
        from cryptography.fernet import Fernet  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Machine-unique key derivation (for Fernet fallback)
# ---------------------------------------------------------------------------

def _derive_fernet_key() -> bytes:
    """
    Derive a 32-byte key from machine-unique bytes using PBKDF2-HMAC-SHA256.
    Stores the salt in ~/.operon/.salt so decryption is stable across restarts.
    """
    import hashlib
    import hmac

    salt_path = _OPERON_DIR / ".salt"
    _OPERON_DIR.mkdir(parents=True, exist_ok=True)

    if salt_path.exists():
        salt = salt_path.read_bytes()
    else:
        salt = os.urandom(32)
        salt_path.write_bytes(salt)
        salt_path.chmod(0o600)

    # Collect machine-unique bytes
    machine_id = ""
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            machine_id = open(path).read().strip()
            break
        except Exception:
            pass
    if not machine_id:
        import uuid
        machine_id = str(uuid.getnode())   # MAC address as fallback

    key = hashlib.pbkdf2_hmac(
        "sha256",
        machine_id.encode() + b"operon-secrets",
        salt,
        iterations=100_000,
        dklen=32,
    )
    # Fernet needs URL-safe base64-encoded 32 bytes
    import base64
    return base64.urlsafe_b64encode(key)


# ---------------------------------------------------------------------------
# SecretsManager
# ---------------------------------------------------------------------------

class SecretsManager:
    """
    Cross-platform secrets manager with automatic backend selection.

    All secrets are namespaced under the 'operon' application name.
    """

    def __init__(self) -> None:
        _OPERON_DIR.mkdir(parents=True, exist_ok=True)
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        if _try_keyring():
            return "keyring"
        if _try_fernet():
            return "fernet"
        return "plain"

    @property
    def backend(self) -> str:
        """Active backend name: 'keyring', 'fernet', or 'plain'."""
        return self._backend

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def set(self, key: str, value: str) -> bool:
        """Store a secret. Returns True on success."""
        if not key or value is None:
            return False

        if self._backend == "keyring":
            try:
                import keyring
                keyring.set_password(_OPERON_APP, key, str(value))
                return True
            except Exception:
                self._backend = "fernet" if _try_fernet() else "plain"

        if self._backend == "fernet":
            return self._fernet_set(key, str(value))

        # Plain fallback
        return self._plain_set(key, str(value))

    def get(self, key: str) -> Optional[str]:
        """Retrieve a secret by key. Returns None if not found."""
        if not key:
            return None

        if self._backend == "keyring":
            try:
                import keyring
                return keyring.get_password(_OPERON_APP, key)
            except Exception:
                self._backend = "fernet" if _try_fernet() else "plain"

        if self._backend == "fernet":
            return self._fernet_get(key)

        return self._plain_get(key)

    def delete(self, key: str) -> bool:
        """Delete a secret by key. Returns True if it existed."""
        if not key:
            return False

        if self._backend == "keyring":
            try:
                import keyring
                existing = keyring.get_password(_OPERON_APP, key)
                if existing is not None:
                    keyring.delete_password(_OPERON_APP, key)
                    return True
                return False
            except Exception:
                pass

        if self._backend == "fernet":
            return self._fernet_delete(key)

        return self._plain_delete(key)

    def list_keys(self) -> List[str]:
        """
        List all stored secret keys (NOT their values).
        Note: keyring does not support listing — falls back to index file.
        """
        index = self._load_index()
        return sorted(index)

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    # ------------------------------------------------------------------
    # Index file (used for list_keys across all backends)
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return _OPERON_DIR / ".secrets_index"

    def _load_index(self) -> List[str]:
        p = self._index_path()
        try:
            return json.loads(p.read_text()) if p.exists() else []
        except Exception:
            return []

    def _save_index(self, keys: List[str]) -> None:
        try:
            self._index_path().write_text(json.dumps(sorted(set(keys))))
            self._index_path().chmod(0o600)
        except Exception:
            pass

    def _add_to_index(self, key: str) -> None:
        keys = self._load_index()
        if key not in keys:
            keys.append(key)
            self._save_index(keys)

    def _remove_from_index(self, key: str) -> None:
        keys = self._load_index()
        if key in keys:
            keys.remove(key)
            self._save_index(keys)

    # ------------------------------------------------------------------
    # Fernet backend
    # ------------------------------------------------------------------

    def _fernet_path(self) -> Path:
        return _OPERON_DIR / "secrets.enc"

    def _fernet_load(self) -> Dict[str, str]:
        from cryptography.fernet import Fernet, InvalidToken
        path = self._fernet_path()
        if not path.exists():
            return {}
        try:
            f    = Fernet(_derive_fernet_key())
            data = f.decrypt(path.read_bytes())
            return json.loads(data)
        except (InvalidToken, Exception):
            return {}

    def _fernet_save(self, store: Dict[str, str]) -> None:
        from cryptography.fernet import Fernet
        f    = Fernet(_derive_fernet_key())
        data = f.encrypt(json.dumps(store).encode())
        path = self._fernet_path()
        path.write_bytes(data)
        path.chmod(0o600)

    def _fernet_set(self, key: str, value: str) -> bool:
        try:
            store = self._fernet_load()
            store[key] = value
            self._fernet_save(store)
            self._add_to_index(key)
            return True
        except Exception:
            return False

    def _fernet_get(self, key: str) -> Optional[str]:
        try:
            return self._fernet_load().get(key)
        except Exception:
            return None

    def _fernet_delete(self, key: str) -> bool:
        try:
            store = self._fernet_load()
            if key in store:
                del store[key]
                self._fernet_save(store)
                self._remove_from_index(key)
                return True
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Plain JSON fallback (existing knowledge.json behaviour)
    # ------------------------------------------------------------------

    def _plain_path(self) -> Path:
        return _OPERON_DIR / "knowledge.json"

    def _plain_load(self) -> Dict[str, Any]:
        p = self._plain_path()
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:
            return {}

    def _plain_save(self, store: Dict[str, Any]) -> None:
        self._plain_path().write_text(
            json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _plain_set(self, key: str, value: str) -> bool:
        """Stores in plain JSON — prints a security warning."""
        print(
            f"\n  !  Storing secret '{key}' as plain text. "
            "For encryption, run:  pip install keyring  or  pip install cryptography\n",
            file=sys.stderr,
        )
        try:
            from datetime import datetime
            store = self._plain_load()
            store[key] = {"value": value, "updated": datetime.now().isoformat()}
            self._plain_save(store)
            self._add_to_index(key)
            return True
        except Exception:
            return False

    def _plain_get(self, key: str) -> Optional[str]:
        try:
            entry = self._plain_load().get(key)
            if entry is None:
                return None
            if isinstance(entry, dict):
                return entry.get("value")
            return str(entry)
        except Exception:
            return None

    def _plain_delete(self, key: str) -> bool:
        try:
            store = self._plain_load()
            if key in store:
                del store[key]
                self._plain_save(store)
                self._remove_from_index(key)
                return True
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------

    def migrate_from_knowledge_json(self) -> int:
        """
        One-time migration: read secrets from the plain knowledge.json store
        and re-save them with the currently active (hopefully encrypted) backend.
        Returns the number of keys migrated.
        """
        plain_store = self._plain_load()
        migrated = 0
        _SENSITIVE_KEYS = {
            "app_password", "gmail_app_password", "email_password",
            "openai_api_key", "anthropic_api_key", "telegram_token",
            "discord_bot_token", "slack_bot_token",
        }
        for key, entry in plain_store.items():
            if key in _SENSITIVE_KEYS:
                value = entry.get("value", "") if isinstance(entry, dict) else str(entry)
                if value and self._backend != "plain":
                    self.set(key, value)
                    migrated += 1
        return migrated

    def status(self) -> Dict[str, Any]:
        """Return info about the current backend and stored key count."""
        return {
            "backend":    self._backend,
            "key_count":  len(self.list_keys()),
            "keys":       self.list_keys(),
            "encrypted":  self._backend in ("keyring", "fernet"),
            "storage":    {
                "keyring": "OS Keychain (macOS Keychain / GNOME Keyring / Windows Credential Manager)",
                "fernet":  str(self._fernet_path()),
                "plain":   str(self._plain_path()),
            }.get(self._backend, "unknown"),
        }


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init)
# ---------------------------------------------------------------------------
_instance: Optional[SecretsManager] = None


def get_secrets() -> SecretsManager:
    """Return the shared SecretsManager singleton."""
    global _instance
    if _instance is None:
        _instance = SecretsManager()
    return _instance
