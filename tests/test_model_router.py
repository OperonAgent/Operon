"""Tests for core/model_router.py — SmartModelRouter Phase 11"""
import pytest
from unittest.mock import patch, MagicMock

from core.model_router import (
    TaskType, ModelProfile, RouteDecision, SmartModelRouter,
    get_smart_router, route_prompt, classify_prompt, strip_hints,
    _HINT_RE, _CODE_PATTERNS, _REASONING_PATTERNS, _FAST_PATTERNS,
    _CREATIVE_PATTERNS, _BUILT_IN_PROFILES,
)


# ── TaskType ──────────────────────────────────────────────────────────────────

class TestTaskType:
    def test_values_are_strings(self):
        assert TaskType.CODE == "code"
        assert TaskType.REASONING == "reasoning"
        assert TaskType.FAST == "fast"
        assert TaskType.VISION == "vision"
        assert TaskType.CREATIVE == "creative"
        assert TaskType.DEFAULT == "default"

    def test_all_members(self):
        members = {t.value for t in TaskType}
        assert members == {"code", "reasoning", "fast", "vision", "creative", "default"}


# ── Pattern matching ──────────────────────────────────────────────────────────

class TestPatterns:
    def test_code_pattern_python(self):
        assert _CODE_PATTERNS.search("write a python function")

    def test_code_pattern_debug(self):
        assert _CODE_PATTERNS.search("debug this traceback")

    def test_code_pattern_sql(self):
        assert _CODE_PATTERNS.search("write a SQL query to fetch users")

    def test_code_pattern_api(self):
        assert _CODE_PATTERNS.search("implement a REST api endpoint")

    def test_code_pattern_no_match(self):
        assert not _CODE_PATTERNS.search("what is the weather like today")

    def test_reasoning_pattern_analyze(self):
        assert _REASONING_PATTERNS.search("analyze the trade-offs of microservices")

    def test_reasoning_pattern_compare(self):
        assert _REASONING_PATTERNS.search("compare these two architectures")

    def test_reasoning_pattern_step_by_step(self):
        assert _REASONING_PATTERNS.search("think step by step through this problem")

    def test_reasoning_pattern_no_match(self):
        assert not _REASONING_PATTERNS.search("hello world")

    def test_fast_pattern_yes(self):
        assert _FAST_PATTERNS.search("yes that is correct")

    def test_fast_pattern_list(self):
        assert _FAST_PATTERNS.search("list the planets")

    def test_fast_pattern_what_is(self):
        assert _FAST_PATTERNS.search("what is the capital of France")

    def test_creative_pattern_write(self):
        assert _CREATIVE_PATTERNS.search("write a short story about AI")

    def test_creative_pattern_brainstorm(self):
        assert _CREATIVE_PATTERNS.search("brainstorm ideas for a product launch")

    def test_hint_re_reasoning(self):
        m = _HINT_RE.search("please hint:reasoning explain this")
        assert m and m.group(1).lower() == "reasoning"

    def test_hint_re_code(self):
        m = _HINT_RE.search("hint:code write a parser")
        assert m and m.group(1).lower() == "code"

    def test_hint_re_case_insensitive(self):
        m = _HINT_RE.search("HINT:FAST answer this")
        assert m and m.group(1).lower() == "fast"

    def test_hint_re_not_in_word(self):
        # "thint:code" should not match since \b is required
        assert not _HINT_RE.search("thint:code")


# ── ModelProfile ──────────────────────────────────────────────────────────────

class TestModelProfile:
    def test_default_values(self):
        p = ModelProfile(name="test-model", provider="ollama")
        assert p.context_len == 4096
        assert p.speed_score == 5
        assert p.quality_score == 5
        assert p.available is True
        assert p.is_vision is False
        assert p.cost_per_1k == 0.0

    def test_score_for_specialty(self):
        p = ModelProfile(
            name="code-model", provider="ollama",
            task_types=[TaskType.CODE],
            quality_score=8, speed_score=6,
        )
        scores = p.score_for
        # Code should score higher (specialty bonus)
        assert scores[TaskType.CODE] > scores[TaskType.FAST]

    def test_score_for_all_types_present(self):
        p = ModelProfile(name="m", provider="ollama")
        scores = p.score_for
        for tt in TaskType:
            assert tt in scores

    def test_score_for_positive(self):
        p = ModelProfile(name="m", provider="ollama", quality_score=7, speed_score=7)
        for score in p.score_for.values():
            assert score > 0

    def test_task_types_default_empty(self):
        p = ModelProfile(name="m", provider="ollama")
        assert p.task_types == []


# ── Built-in profiles ─────────────────────────────────────────────────────────

class TestBuiltInProfiles:
    def test_profiles_exist(self):
        assert len(_BUILT_IN_PROFILES) >= 6

    def test_hermes_is_in_profiles(self):
        names = [p.name for p in _BUILT_IN_PROFILES]
        assert "hermes3:8b" in names

    def test_code_model_in_profiles(self):
        names = [p.name for p in _BUILT_IN_PROFILES]
        assert "qwen2.5-coder:7b" in names

    def test_cloud_models_have_cost(self):
        for p in _BUILT_IN_PROFILES:
            if p.provider in ("anthropic", "openai"):
                assert p.cost_per_1k > 0, f"{p.name} should have cost"

    def test_local_models_free(self):
        for p in _BUILT_IN_PROFILES:
            if p.provider == "ollama":
                assert p.cost_per_1k == 0.0, f"{p.name} should be free"

    def test_gpt4o_is_vision(self):
        gpt4o = next((p for p in _BUILT_IN_PROFILES if p.name == "gpt-4o"), None)
        assert gpt4o is not None
        assert gpt4o.is_vision is True

    def test_claude_has_large_context(self):
        claude = next((p for p in _BUILT_IN_PROFILES if "claude" in p.name), None)
        assert claude is not None
        assert claude.context_len >= 100000


# ── SmartModelRouter ──────────────────────────────────────────────────────────

@pytest.fixture
def router_no_ollama():
    """Router with no Ollama models available, no cloud keys."""
    with patch("core.model_router._discover_ollama_models", return_value=[]):
        r = SmartModelRouter(default_model="hermes3:8b", prefer_local=True)
    return r


@pytest.fixture
def router_with_models():
    """Router with several local models available."""
    local = ["hermes3:8b", "qwen2.5-coder:7b", "qwen3:4b", "qwen2.5:3b"]
    with patch("core.model_router._discover_ollama_models", return_value=local):
        r = SmartModelRouter(default_model="hermes3:8b", prefer_local=True)
    return r


class TestSmartModelRouterClassify:
    def test_hint_code(self, router_with_models):
        tt, hint = router_with_models.classify("hint:code write a function")
        assert tt == TaskType.CODE
        assert hint == "code"

    def test_hint_fast(self, router_with_models):
        tt, hint = router_with_models.classify("hint:fast what is 2+2")
        assert tt == TaskType.FAST
        assert hint == "fast"

    def test_hint_reasoning(self, router_with_models):
        tt, hint = router_with_models.classify("hint:reasoning analyze this")
        assert tt == TaskType.REASONING
        assert hint == "reasoning"

    def test_hint_vision(self, router_with_models):
        tt, hint = router_with_models.classify("hint:vision describe this image")
        assert tt == TaskType.VISION
        assert hint == "vision"

    def test_hint_cloud_maps_to_reasoning(self, router_with_models):
        tt, hint = router_with_models.classify("hint:cloud solve this")
        assert tt == TaskType.REASONING
        assert hint == "cloud"

    def test_hint_analysis_maps_to_reasoning(self, router_with_models):
        tt, hint = router_with_models.classify("hint:analysis look at this data")
        assert tt == TaskType.REASONING
        assert hint == "analysis"

    def test_auto_code(self, router_with_models):
        tt, hint = router_with_models.classify("write a python class for parsing JSON")
        assert tt == TaskType.CODE
        assert hint is None

    def test_auto_reasoning(self, router_with_models):
        tt, hint = router_with_models.classify("analyze the trade-offs of this architecture design")
        assert tt == TaskType.REASONING
        assert hint is None

    def test_auto_fast_short(self, router_with_models):
        tt, hint = router_with_models.classify("what is the capital")
        assert tt == TaskType.FAST
        assert hint is None

    def test_auto_creative(self, router_with_models):
        tt, hint = router_with_models.classify("write a poem about the ocean waves")
        assert tt == TaskType.CREATIVE
        assert hint is None

    def test_auto_default(self, router_with_models):
        tt, hint = router_with_models.classify("hey there operon")
        assert tt == TaskType.DEFAULT
        assert hint is None

    def test_fast_long_prompt_not_fast(self, router_with_models):
        long = "what is " + "a very long question " * 10
        tt, _ = router_with_models.classify(long)
        # Long prompts shouldn't be FAST even if they have fast keywords
        assert tt != TaskType.FAST


class TestSmartModelRouterRoute:
    def test_route_code_returns_code_model(self, router_with_models):
        decision = router_with_models.route("write a python function")
        assert decision.provider == "ollama"
        assert "qwen2.5-coder" in decision.model or "qwen" in decision.model

    def test_route_returns_route_decision(self, router_with_models):
        d = router_with_models.route("hello")
        assert isinstance(d, RouteDecision)
        assert d.model
        assert d.provider
        assert d.task_type
        assert d.reason

    def test_route_cloud_hint_no_key(self, router_no_ollama):
        # No cloud keys, should fall through to local
        d = router_no_ollama.route("hint:cloud solve this problem")
        # Should not raise, returns something
        assert isinstance(d, RouteDecision)

    def test_route_cloud_hint_with_anthropic_key(self, router_with_models):
        router_with_models._anthro_key = "sk-test-key"
        router_with_models._profiles["claude-sonnet-4-5"].available = True
        d = router_with_models.route("hint:cloud analyze this deeply")
        assert d.provider == "anthropic"
        assert "claude" in d.model

    def test_route_local_hint(self, router_with_models):
        d = router_with_models.route("hint:local answer this")
        assert d.provider == "ollama"
        assert d.hint_used == "local"

    def test_route_vision_no_key(self, router_with_models):
        d = router_with_models.route("hint:vision describe this screenshot")
        # No OpenAI key, should fall back
        assert isinstance(d, RouteDecision)
        assert d.task_type == TaskType.VISION

    def test_route_vision_with_openai_key(self, router_with_models):
        router_with_models._openai_key = "sk-openai-test"
        router_with_models._profiles["gpt-4o"].available = True
        d = router_with_models.route("hint:vision analyze this image")
        assert d.model == "gpt-4o"
        assert d.provider == "openai"

    def test_route_no_models_returns_default(self, router_no_ollama):
        d = router_no_ollama.route("write some code")
        assert d.model == "hermes3:8b"  # default fallback

    def test_route_reason_is_set(self, router_with_models):
        d = router_with_models.route("analyze this data")
        assert len(d.reason) > 0

    def test_route_fallback_flag(self, router_with_models):
        router_with_models._openai_key = ""
        d = router_with_models.route("hint:vision image here")
        assert d.fallback is True


class TestSmartModelRouterBestLocal:
    def test_best_local_code(self, router_with_models):
        model = router_with_models._best_local(TaskType.CODE)
        assert "coder" in model

    def test_best_local_fast(self, router_with_models):
        model = router_with_models._best_local(TaskType.FAST)
        # Should return a fast model
        assert model in {"qwen2.5:3b", "qwen3:4b", "hermes3:8b", "qwen2.5-coder:7b"}

    def test_best_local_no_candidates_returns_default(self, router_no_ollama):
        model = router_no_ollama._best_local(TaskType.CODE)
        assert model == "hermes3:8b"


class TestSmartModelRouterStripHints:
    def test_strip_single_hint(self):
        result = SmartModelRouter.strip_hints("hint:code write a function")
        assert "hint:" not in result
        assert "write a function" in result

    def test_strip_multiple_hints(self):
        result = SmartModelRouter.strip_hints("hint:fast hint:local what is 2+2")
        assert "hint:" not in result

    def test_strip_no_hints(self):
        result = SmartModelRouter.strip_hints("normal prompt")
        assert result == "normal prompt"

    def test_strip_hint_only(self):
        result = SmartModelRouter.strip_hints("hint:reasoning")
        assert result == ""

    def test_module_strip_hints(self):
        result = strip_hints("hint:cloud analyze this")
        assert "hint:" not in result


class TestSmartModelRouterMethods:
    def test_available_models_returns_list(self, router_with_models):
        models = router_with_models.available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_available_models_structure(self, router_with_models):
        models = router_with_models.available_models()
        m = models[0]
        assert "name" in m
        assert "provider" in m
        assert "available" in m
        assert "tasks" in m
        assert "context" in m
        assert "cost_per_1k" in m

    def test_best_for_code(self, router_with_models):
        model = router_with_models.best_for("code")
        assert model  # returns something

    def test_best_for_invalid_falls_back(self, router_with_models):
        model = router_with_models.best_for("invalid_task_type_xyz")
        assert model  # doesn't raise

    def test_status_returns_string(self, router_with_models):
        s = router_with_models.status()
        assert isinstance(s, str)
        assert "SmartModelRouter" in s

    def test_refresh_updates_availability(self):
        with patch("core.model_router._discover_ollama_models", return_value=[]):
            r = SmartModelRouter()
        with patch("core.model_router._discover_ollama_models", return_value=["hermes3:8b"]):
            r.refresh()
        assert r._profiles["hermes3:8b"].available is True

    def test_add_profile(self, router_with_models):
        custom = ModelProfile(
            name="my-custom:7b",
            provider="ollama",
            task_types=[TaskType.CODE],
        )
        router_with_models.add_profile(custom)
        assert "my-custom:7b" in router_with_models._profiles


# ── Module-level functions ────────────────────────────────────────────────────

class TestModuleFunctions:
    def test_get_smart_router_singleton(self):
        import core.model_router as mr
        mr._router = None
        with patch("core.model_router._discover_ollama_models", return_value=[]):
            r1 = get_smart_router()
            r2 = get_smart_router()
        assert r1 is r2
        mr._router = None

    def test_route_prompt_returns_decision(self):
        with patch("core.model_router._discover_ollama_models", return_value=["hermes3:8b"]):
            import core.model_router as mr
            mr._router = None
            d = route_prompt("analyze this code")
        assert isinstance(d, RouteDecision)
        mr._router = None

    def test_classify_prompt_returns_tuple(self):
        import core.model_router as mr
        mr._router = None
        with patch("core.model_router._discover_ollama_models", return_value=[]):
            tt, hint = classify_prompt("hint:code write a function")
        assert tt == "code"
        assert hint == "code"
        mr._router = None

    def test_classify_prompt_no_hint(self):
        import core.model_router as mr
        mr._router = None
        with patch("core.model_router._discover_ollama_models", return_value=[]):
            tt, hint = classify_prompt("what is the weather today")
        assert hint is None
        mr._router = None


# ── Ollama discovery ──────────────────────────────────────────────────────────

class TestOllamaDiscovery:
    def test_discover_parses_output(self):
        from core.model_router import _discover_ollama_models
        fake_output = b"NAME\t\t\tID\t\tSIZE\n hermes3:8b\t\tabc\t4GB\nqwen2.5-coder:7b\txyz\t4GB\n"
        with patch("subprocess.check_output", return_value=fake_output):
            models = _discover_ollama_models()
        assert "hermes3:8b" in models or len(models) >= 1

    def test_discover_handles_error_gracefully(self):
        from core.model_router import _discover_ollama_models
        with patch("subprocess.check_output", side_effect=Exception("ollama not found")):
            models = _discover_ollama_models()
        assert models == []

    def test_discover_handles_timeout(self):
        from core.model_router import _discover_ollama_models
        import subprocess
        with patch("subprocess.check_output", side_effect=subprocess.TimeoutExpired("ollama", 5)):
            models = _discover_ollama_models()
        assert models == []
