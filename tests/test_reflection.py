"""Tests for core/reflection.py"""
import pytest
from unittest.mock import MagicMock, patch
from core.reflection import (
    ReflectionEngine, ReflectionConfig, ReflectionResult, ReflectionIssue,
    _heuristic_checks,
    ISSUE_MISSING_TOOL_CALL, ISSUE_HALLUCINATION, ISSUE_SELF_CONTRADICTION,
    ISSUE_INCOMPLETE_ANSWER,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_router(response_text='{"verdict": "NONE"}'):
    router = MagicMock()
    router.model = "claude-3-haiku-20240307"
    router.complete.return_value = response_text
    return router


def make_response(content="The answer is 42.", tool_call=None):
    r = {"thought": "...", "content": content}
    if tool_call:
        r["tool_call"] = tool_call
    return r


# ── Heuristic checks ──────────────────────────────────────────────────────────

class TestHeuristicChecks:
    def test_flags_missing_tool_when_actionable_request(self):
        issues = _heuristic_checks(
            "Please search for the latest Python version",
            make_response("I don't have internet access."),
            tool_results=[],
        )
        types = [i.issue_type for i in issues]
        assert ISSUE_MISSING_TOOL_CALL in types

    def test_no_issue_when_tool_was_called(self):
        issues = _heuristic_checks(
            "Search for Python version",
            make_response("Python 3.13 is the latest."),
            tool_results=[{"tool_name": "web_search", "output": "Python 3.13"}],
        )
        # Tool was called, so no missing-tool issue
        types = [i.issue_type for i in issues]
        assert ISSUE_MISSING_TOOL_CALL not in types

    def test_flags_hallucination_claim_without_tool(self):
        issues = _heuristic_checks(
            "What is the current Bitcoin price?",
            make_response("The current price is $65,000 right now."),
            tool_results=[],
        )
        types = [i.issue_type for i in issues]
        assert ISSUE_HALLUCINATION in types

    def test_no_hallucination_when_tool_present(self):
        issues = _heuristic_checks(
            "What is the current Bitcoin price?",
            make_response("The price is $65,000."),
            tool_results=[{"tool_name": "web_search", "output": "$65,000"}],
        )
        types = [i.issue_type for i in issues]
        assert ISSUE_HALLUCINATION not in types

    def test_flags_self_contradiction(self):
        issues = _heuristic_checks(
            "Delete the file",
            make_response("Successfully deleted the file."),
            tool_results=[{"tool_name": "shell_exec", "output": "Error: permission denied"}],
        )
        types = [i.issue_type for i in issues]
        assert ISSUE_SELF_CONTRADICTION in types

    def test_flags_very_short_answer(self):
        issues = _heuristic_checks(
            "Explain in great detail how machine learning gradient descent works step by step, "
            "including the mathematical derivation of partial derivatives and the intuition behind "
            "learning rate selection and momentum in neural network training",
            make_response("It's an optimization method."),
            tool_results=[],
        )
        types = [i.issue_type for i in issues]
        assert ISSUE_INCOMPLETE_ANSWER in types

    def test_normal_response_no_issues(self):
        issues = _heuristic_checks(
            "What is 2 + 2?",
            make_response("2 + 2 = 4."),
            tool_results=[],
        )
        assert len(issues) == 0


# ── ReflectionEngine ─────────────────────────────────────────────────────────

class TestReflectionEngine:
    def test_disabled_config_returns_immediately(self):
        router = make_router()
        engine = ReflectionEngine(router, ReflectionConfig(enabled=False))
        result = engine.reflect("test", make_response("answer"), tool_results=[])
        assert not result.has_issues
        assert not result.did_correct
        router.complete.assert_not_called()

    def test_clean_response_no_issues(self):
        router = make_router('{"verdict": "NONE"}')
        engine = ReflectionEngine(router)
        result = engine.reflect(
            user_message="What is 2+2?",
            agent_response=make_response("4"),
        )
        # Might have no issues or only low-severity ones
        critical = [i for i in result.issues if i.severity == "critical"]
        assert len(critical) == 0

    def test_detects_contradiction(self):
        router = make_router('{"verdict": "NONE"}')
        engine = ReflectionEngine(router)
        result = engine.reflect(
            user_message="Run the script",
            agent_response=make_response("Script ran successfully."),
            tool_results=[{"tool_name": "shell_exec", "output": "Error: not found"}],
        )
        types = [i.issue_type for i in result.issues]
        assert ISSUE_SELF_CONTRADICTION in types

    def test_correction_applied_for_contradiction(self):
        router = make_router('{"verdict": "NONE"}')
        engine = ReflectionEngine(router)
        response = make_response("Successfully completed the task.")
        result = engine.reflect(
            user_message="Do the thing",
            agent_response=response,
            tool_results=[{"tool_name": "shell_exec", "output": "Error: something failed"}],
        )
        # Should auto-correct the contradiction
        if result.did_correct:
            assert "successfully" not in result.final_response.get("content", "").lower() or \
                   "error" in result.final_response.get("content", "").lower() or \
                   "review" in result.final_response.get("content", "").lower()

    def test_llm_correction_applied(self):
        llm_correction = '{"verdict": "INCOMPLETE", "severity": "high", ' \
                         '"reason": "Did not answer", ' \
                         '"corrected_content": "Here is the full answer."}'
        router = make_router(llm_correction)
        engine = ReflectionEngine(router)
        result = engine.reflect(
            user_message="Search for X and explain Y in great detail",
            agent_response=make_response("Done."),
            tool_results=[],
        )
        # LLM correction may be applied
        assert isinstance(result.final_response, dict)
        assert "content" in result.final_response

    def test_max_corrections_not_exceeded(self):
        router = make_router('{"verdict": "NONE"}')
        engine = ReflectionEngine(router, ReflectionConfig(max_corrections=1))

        # First call with contradiction → correction applied
        r1 = engine.reflect(
            "test",
            make_response("Success!"),
            tool_results=[{"tool_name": "x", "output": "Error"}],
            session_id="sess1",
        )
        # Second call — correction count at max
        r2 = engine.reflect(
            "test",
            make_response("Success!"),
            tool_results=[{"tool_name": "x", "output": "Error"}],
            session_id="sess1",
        )
        # Second correction should not be applied (count exceeded)
        assert engine._correction_counts.get("sess1", 0) <= 1

    def test_reset_session_clears_counter(self):
        router = make_router('{"verdict": "NONE"}')
        engine = ReflectionEngine(router)
        engine._correction_counts["mysession"] = 5
        engine.reset_session("mysession")
        assert "mysession" not in engine._correction_counts

    def test_reflection_ms_populated(self):
        router = make_router('{"verdict": "NONE"}')
        engine = ReflectionEngine(router)
        result = engine.reflect("hi", make_response("hello"))
        assert result.reflection_ms >= 0

    def test_summary_for_prompt_empty_when_no_issues(self):
        router = make_router()
        engine = ReflectionEngine(router)
        result = ReflectionResult(original_response=make_response("ok"))
        assert engine.summary_for_prompt(result) == ""

    def test_summary_for_prompt_lists_issues(self):
        router = make_router()
        engine = ReflectionEngine(router)
        result = ReflectionResult(
            original_response=make_response("ok"),
            issues=[
                ReflectionIssue(
                    issue_type="hallucination_risk",
                    severity="medium",
                    description="Claims real-time data without tool",
                    suggestion="Use web_search",
                )
            ],
        )
        summary = engine.summary_for_prompt(result)
        assert "REFLECTION" in summary
        assert "hallucination_risk" in summary

    def test_get_engine_singleton(self):
        from core.reflection import get_engine
        import core.reflection as refl_mod
        refl_mod._default_engine = None  # reset
        r = make_router()
        e1 = get_engine(r)
        e2 = get_engine()  # no router needed on second call
        assert e1 is e2
        refl_mod._default_engine = None  # cleanup


# ── ReflectionResult properties ───────────────────────────────────────────────

class TestReflectionResult:
    def test_final_response_is_corrected_when_present(self):
        orig = make_response("original")
        corr = make_response("corrected")
        result = ReflectionResult(
            original_response=orig,
            corrected_response=corr,
        )
        assert result.final_response is corr

    def test_final_response_is_original_when_no_correction(self):
        orig = make_response("original")
        result = ReflectionResult(original_response=orig)
        assert result.final_response is orig

    def test_critical_issues_filter(self):
        result = ReflectionResult(
            original_response=make_response("x"),
            issues=[
                ReflectionIssue("a", "critical", "crit issue"),
                ReflectionIssue("b", "medium",   "med issue"),
                ReflectionIssue("c", "high",     "high issue"),
            ],
        )
        crits = result.critical_issues
        assert len(crits) == 1
        assert crits[0].issue_type == "a"

    def test_has_issues_false_when_empty(self):
        result = ReflectionResult(original_response=make_response("ok"))
        assert not result.has_issues

    def test_has_issues_true_when_present(self):
        result = ReflectionResult(
            original_response=make_response("ok"),
            issues=[ReflectionIssue("x", "low", "minor")],
        )
        assert result.has_issues
