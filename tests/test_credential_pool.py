"""Tests for core/credential_pool.py"""
import os
import time
import pytest
from unittest import mock

from core.credential_pool import (
    CredentialPool, KeySlot, KeyStatus, ErrorKind,
    get_pool,
    _BACKOFF, _ERROR_PATTERNS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pool_with_keys(n: int = 3, provider: str = "openai") -> CredentialPool:
    pool = CredentialPool()
    for i in range(n):
        pool.add(provider, f"sk-test-key-{i:03d}", label=f"key{i}")
    return pool


# ── KeySlot ───────────────────────────────────────────────────────────────────

class TestKeySlot:
    def test_active_is_available(self):
        slot = KeySlot(provider="openai", key="sk-x")
        assert slot.is_available is True

    def test_banned_not_available(self):
        slot = KeySlot(provider="openai", key="sk-x", status=KeyStatus.BANNED)
        assert slot.is_available is False

    def test_cooling_not_available_before_cooldown(self):
        slot = KeySlot(provider="openai", key="sk-x", status=KeyStatus.COOLING,
                       cooldown_until=time.time() + 1000)
        assert slot.is_available is False

    def test_cooling_available_after_cooldown(self):
        slot = KeySlot(provider="openai", key="sk-x", status=KeyStatus.COOLING,
                       cooldown_until=time.time() - 1)
        assert slot.is_available is True
        assert slot.status == KeyStatus.ACTIVE   # auto-recovered

    def test_masked_key(self):
        slot = KeySlot(provider="openai", key="sk-abcdefghij")
        assert "…" in slot.masked_key
        assert "sk-a" in slot.masked_key
        assert "ghij" in slot.masked_key

    def test_masked_key_short(self):
        slot = KeySlot(provider="openai", key="short")
        assert slot.masked_key == "****"

    def test_to_dict_fields(self):
        slot = KeySlot(provider="openai", key="sk-test-123",
                       label="my-key", use_count=5, error_count=1)
        d = slot.to_dict()
        assert d["label"] == "my-key"
        assert d["use_count"] == 5
        assert d["error_count"] == 1
        assert "available" in d


# ── CredentialPool.add ────────────────────────────────────────────────────────

class TestPoolAdd:
    def test_add_single_key(self):
        pool = CredentialPool()
        pool.add("openai", "sk-001")
        assert pool.key_count("openai") == 1

    def test_add_multiple_keys(self):
        pool = _pool_with_keys(3)
        assert pool.key_count("openai") == 3

    def test_duplicate_key_ignored(self):
        pool = CredentialPool()
        pool.add("openai", "sk-001")
        pool.add("openai", "sk-001")
        assert pool.key_count("openai") == 1

    def test_different_providers_isolated(self):
        pool = CredentialPool()
        pool.add("openai", "sk-001")
        pool.add("anthropic", "ant-001")
        assert pool.key_count("openai") == 1
        assert pool.key_count("anthropic") == 1

    def test_add_with_label(self):
        pool = CredentialPool()
        pool.add("openai", "sk-001", label="production")
        s = pool.status()
        assert s["openai"]["keys"][0]["label"] == "production"

    def test_provider_name_normalized_lowercase(self):
        pool = CredentialPool()
        pool.add("OpenAI", "sk-001")
        assert pool.key_count("openai") == 1


# ── CredentialPool.load_from_env ─────────────────────────────────────────────

class TestLoadFromEnv:
    def test_loads_bare_prefix(self):
        pool = CredentialPool()
        with mock.patch.dict(os.environ, {"MY_API_KEY": "sk-bare"}):
            count = pool.load_from_env("test", "MY_API_KEY")
        assert count == 1
        assert pool.key_count("test") == 1

    def test_loads_numbered_variants(self):
        pool = CredentialPool()
        env = {"MY_KEY_1": "sk-001", "MY_KEY_2": "sk-002", "MY_KEY_3": "sk-003"}
        with mock.patch.dict(os.environ, env):
            count = pool.load_from_env("test", "MY_KEY")
        assert count == 3

    def test_missing_env_var_loads_nothing(self):
        pool = CredentialPool()
        # Ensure env var not set
        os.environ.pop("NONEXISTENT_KEY_XYZ", None)
        count = pool.load_from_env("test", "NONEXISTENT_KEY_XYZ")
        assert count == 0

    def test_bare_and_numbered_combined(self):
        pool = CredentialPool()
        env = {"TEST_KEY": "bare", "TEST_KEY_1": "sk-001", "TEST_KEY_2": "sk-002"}
        with mock.patch.dict(os.environ, env):
            count = pool.load_from_env("test", "TEST_KEY")
        assert count == 3


# ── CredentialPool.get ────────────────────────────────────────────────────────

class TestPoolGet:
    def test_get_returns_first_key(self):
        pool = _pool_with_keys(3)
        k = pool.get("openai")
        assert k == "sk-test-key-000"

    def test_get_unknown_provider_returns_none(self):
        pool = CredentialPool()
        assert pool.get("nonexistent") is None

    def test_get_skips_banned_key(self):
        pool = _pool_with_keys(2)
        pool.report_error("openai", "sk-test-key-000", Exception("401 unauthorized invalid_api_key"))
        k = pool.get("openai")
        assert k == "sk-test-key-001"

    def test_get_skips_cooling_key(self):
        pool = _pool_with_keys(2)
        pool._slots["openai"][0].status = KeyStatus.COOLING
        pool._slots["openai"][0].cooldown_until = time.time() + 1000
        k = pool.get("openai")
        assert k == "sk-test-key-001"

    def test_get_all_banned_returns_none(self):
        pool = _pool_with_keys(2)
        for slot in pool._slots["openai"]:
            slot.status = KeyStatus.BANNED
        assert pool.get("openai") is None

    def test_get_increments_use_count(self):
        pool = _pool_with_keys(1)
        pool.get("openai")
        assert pool._slots["openai"][0].use_count == 1

    def test_get_updates_last_used(self):
        pool = _pool_with_keys(1)
        pool.get("openai")
        assert pool._slots["openai"][0].last_used > 0


# ── CredentialPool.rotate ─────────────────────────────────────────────────────

class TestPoolRotate:
    def test_rotate_advances_to_next(self):
        pool = _pool_with_keys(3)
        next_k = pool.rotate("openai")
        assert next_k == "sk-test-key-001"

    def test_rotate_marks_current_as_degraded(self):
        pool = _pool_with_keys(3)
        pool.rotate("openai")
        assert pool._slots["openai"][0].status == KeyStatus.DEGRADED

    def test_rotate_wraps_around(self):
        pool = _pool_with_keys(2)
        pool._current["openai"] = 1   # point to last key
        # both keys active, should wrap to 0
        next_k = pool.rotate("openai")
        assert next_k is not None

    def test_rotate_no_keys_returns_none(self):
        pool = CredentialPool()
        assert pool.rotate("nonexistent") is None

    def test_rotate_all_banned_returns_none(self):
        pool = _pool_with_keys(2)
        for slot in pool._slots["openai"]:
            slot.status = KeyStatus.BANNED
        assert pool.rotate("openai") is None


# ── CredentialPool.report_error ──────────────────────────────────────────────

class TestReportError:
    def test_rate_limit_sets_cooling(self):
        pool = _pool_with_keys(2)
        status, _ = pool.report_error("openai", "sk-test-key-000",
                                      Exception("429 rate limit exceeded"))
        assert status == KeyStatus.COOLING

    def test_auth_error_sets_banned(self):
        pool = _pool_with_keys(2)
        status, _ = pool.report_error("openai", "sk-test-key-000",
                                      Exception("401 Unauthorized invalid_api_key"))
        assert status == KeyStatus.BANNED

    def test_quota_error_sets_banned(self):
        pool = _pool_with_keys(2)
        status, _ = pool.report_error("openai", "sk-test-key-000",
                                      Exception("insufficient_quota billing issue"))
        assert status == KeyStatus.BANNED

    def test_timeout_error_sets_cooling(self):
        pool = _pool_with_keys(2)
        status, _ = pool.report_error("openai", "sk-test-key-000",
                                      Exception("connection timed out"))
        assert status == KeyStatus.COOLING

    def test_error_rotates_to_next_key(self):
        pool = _pool_with_keys(3)
        _, next_k = pool.report_error("openai", "sk-test-key-000",
                                      Exception("429 rate limit"))
        assert next_k == "sk-test-key-001"

    def test_error_increments_error_count(self):
        pool = _pool_with_keys(2)
        pool.report_error("openai", "sk-test-key-000", Exception("server error 500"))
        assert pool._slots["openai"][0].error_count == 1

    def test_error_records_message(self):
        pool = _pool_with_keys(2)
        pool.report_error("openai", "sk-test-key-000", Exception("some specific error"))
        assert "some specific error" in pool._slots["openai"][0].last_error_msg


# ── CredentialPool.report_success ────────────────────────────────────────────

class TestReportSuccess:
    def test_success_resets_to_active(self):
        pool = _pool_with_keys(1)
        pool._slots["openai"][0].status = KeyStatus.DEGRADED
        pool.report_success("openai", "sk-test-key-000")
        assert pool._slots["openai"][0].status == KeyStatus.ACTIVE

    def test_success_decrements_error_count(self):
        pool = _pool_with_keys(1)
        pool._slots["openai"][0].error_count = 3
        pool.report_success("openai", "sk-test-key-000")
        assert pool._slots["openai"][0].error_count == 2


# ── CredentialPool.call_with_retry ───────────────────────────────────────────

class TestCallWithRetry:
    def test_success_on_first_try(self):
        pool = _pool_with_keys(2)
        fn = mock.MagicMock(return_value="ok")
        result, ok = pool.call_with_retry("openai", fn, max_tries=3)
        assert ok and result == "ok"
        assert fn.call_count == 1

    def test_rotates_on_error(self):
        pool = _pool_with_keys(3)
        call_count = [0]
        def fn(key):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("rate limit 429")
            return "success"
        result, ok = pool.call_with_retry("openai", fn, max_tries=3)
        assert ok and result == "success"
        assert call_count[0] == 2

    def test_all_keys_fail_returns_none(self):
        pool = _pool_with_keys(2)
        fn = mock.MagicMock(side_effect=Exception("auth error 401 unauthorized"))
        result, ok = pool.call_with_retry("openai", fn, max_tries=2)
        assert not ok

    def test_no_keys_returns_false(self):
        pool = CredentialPool()
        result, ok = pool.call_with_retry("openai", lambda k: k)
        assert not ok


# ── CredentialPool.status ─────────────────────────────────────────────────────

class TestPoolStatus:
    def test_status_shows_all_providers(self):
        pool = CredentialPool()
        pool.add("openai", "sk-001")
        pool.add("anthropic", "ant-001")
        s = pool.status()
        assert "openai" in s
        assert "anthropic" in s

    def test_status_counts_active(self):
        pool = _pool_with_keys(3)
        s = pool.status()
        assert s["openai"]["active"] == 3
        assert s["openai"]["banned"] == 0

    def test_available_count(self):
        pool = _pool_with_keys(3)
        assert pool.available_count("openai") == 3
        pool._slots["openai"][0].status = KeyStatus.BANNED
        assert pool.available_count("openai") == 2

    def test_providers_list(self):
        pool = CredentialPool()
        pool.add("openai", "sk-001")
        pool.add("cohere", "co-001")
        assert set(pool.providers()) == {"openai", "cohere"}


# ── Error classification ──────────────────────────────────────────────────────

class TestErrorClassification:
    def test_classify_rate_limit(self):
        assert CredentialPool._classify_error("429 too many requests") == ErrorKind.RATE_LIMIT

    def test_classify_auth(self):
        assert CredentialPool._classify_error("401 unauthorized") == ErrorKind.AUTH

    def test_classify_quota(self):
        assert CredentialPool._classify_error("insufficient_quota exceeded billing") == ErrorKind.QUOTA

    def test_classify_timeout(self):
        assert CredentialPool._classify_error("connection timed out") == ErrorKind.TIMEOUT

    def test_classify_server_error(self):
        assert CredentialPool._classify_error("500 internal server error") == ErrorKind.SERVER

    def test_classify_unknown(self):
        assert CredentialPool._classify_error("something weird happened") == ErrorKind.UNKNOWN


# ── Module-level API ──────────────────────────────────────────────────────────

class TestModuleLevelAPI:
    def test_get_pool_returns_instance(self):
        pool = get_pool()
        assert isinstance(pool, CredentialPool)
