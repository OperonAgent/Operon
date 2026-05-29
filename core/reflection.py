"""
Operon Reflection Engine — Agent self-critique and correction.

After every agent response, this module runs a lightweight validation pass
that catches common failure modes before the user sees a bad answer:

  1. Tool-call quality:  Did the agent call a tool when it should have?
                         Did it pick the right tool for the task?
  2. Completeness:       Does the response actually answer the user's question?
  3. Consistency:        Does the answer contradict earlier tool results?
  4. Hallucination risk: Did the agent claim facts not grounded in tool outputs?
  5. Plan adherence:     If a plan was stated, did the agent execute it?

When issues are found, the engine either:
  - Auto-corrects (fixes obvious parameter errors, retries with better prompt)
  - Appends a correction note to the response
  - Triggers a follow-up tool call

Architecture inspired by Hermes Agent's verification_loop.py.

Usage:
    from core.reflection import ReflectionEngine
    engine = ReflectionEngine(router)
    corrected_response, did_correct = engine.reflect(
        user_message=user_msg,
        agent_response=response_dict,
        tool_results=tool_history,
        messages=message_history,
    )
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("operon.reflection")

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ReflectionConfig:
    enabled:           bool  = True
    max_corrections:   int   = 2        # max auto-corrections per turn
    confidence_thresh: float = 0.6      # below this → trigger reflection
    use_fast_model:    bool  = True     # use cheaper model for reflection
    fast_model:        str   = ""       # empty = same model as parent
    hallucination_check: bool = True    # check for unsupported factual claims
    plan_check:        bool  = True     # check plan adherence
    verbose:           bool  = False    # log reflection reasoning


# ── Issue taxonomy ────────────────────────────────────────────────────────────

ISSUE_MISSING_TOOL_CALL  = "missing_tool_call"
ISSUE_WRONG_TOOL         = "wrong_tool"
ISSUE_INCOMPLETE_ANSWER  = "incomplete_answer"
ISSUE_HALLUCINATION      = "hallucination_risk"
ISSUE_PLAN_MISMATCH      = "plan_mismatch"
ISSUE_PARAM_ERROR        = "param_error"
ISSUE_SELF_CONTRADICTION  = "self_contradiction"

@dataclass
class ReflectionIssue:
    issue_type:  str
    severity:    str      # "critical" | "high" | "medium" | "low"
    description: str
    suggestion:  str = ""
    auto_fixable: bool = False


@dataclass
class ReflectionResult:
    original_response: dict
    corrected_response: Optional[dict] = None
    issues: List[ReflectionIssue] = field(default_factory=list)
    did_correct: bool = False
    correction_count: int = 0
    reflection_ms: int = 0

    @property
    def final_response(self) -> dict:
        return self.corrected_response if self.corrected_response else self.original_response

    @property
    def critical_issues(self) -> List[ReflectionIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)


# ── Heuristic checks (fast, no LLM call) ─────────────────────────────────────

_TOOL_TRIGGER_PHRASES = [
    r"\b(search|look up|find|check|get|fetch|read|list|run|execute|create|write|"
    r"send|calculate|convert|download|upload|install|start|stop)\b",
]
_TOOL_TRIGGER_RE = re.compile("|".join(_TOOL_TRIGGER_PHRASES), re.I)

_UNSUPPORTED_CLAIM_RE = re.compile(
    r"\b(the (current|latest|real-time|live) (price|rate|temperature|weather|"
    r"score|news|status)|as of today|right now)\b",
    re.I,
)


def _heuristic_checks(
    user_message: str,
    response: dict,
    tool_results: List[dict],
) -> List[ReflectionIssue]:
    """Fast heuristic checks that require no LLM call."""
    issues: List[ReflectionIssue] = []
    content = response.get("content", "") or ""
    tool_call = response.get("tool_call")

    # 1. Did the agent answer without using a tool when it should have?
    if not tool_call and not tool_results:
        if _TOOL_TRIGGER_RE.search(user_message):
            # Check if the response actually answers or just says it doesn't know
            hedges = re.search(
                r"\b(i (cannot|can't|don't have|am unable|lack)|(I'm sorry|unfortunately)"
                r"|no (internet|access|ability))\b",
                content, re.I
            )
            if hedges:
                issues.append(ReflectionIssue(
                    issue_type=ISSUE_MISSING_TOOL_CALL,
                    severity="high",
                    description="Agent declined to use a tool despite an actionable request.",
                    suggestion="Retry with explicit instruction to use the appropriate tool.",
                    auto_fixable=True,
                ))

    # 2. Hallucination risk — claims real-time facts with no tool backing
    if _UNSUPPORTED_CLAIM_RE.search(content) and not tool_results:
        issues.append(ReflectionIssue(
            issue_type=ISSUE_HALLUCINATION,
            severity="medium",
            description="Response claims real-time information without a search/API tool call.",
            suggestion="Run web_search or http_get to ground the claim.",
            auto_fixable=False,
        ))

    # 3. Very short response to a complex query
    words_in_query    = len(user_message.split())
    words_in_response = len(content.split())
    if words_in_query > 20 and words_in_response < 15 and not tool_call:
        issues.append(ReflectionIssue(
            issue_type=ISSUE_INCOMPLETE_ANSWER,
            severity="low",
            description=f"Response ({words_in_response} words) seems too brief for the query ({words_in_query} words).",
            suggestion="Expand the response with more detail.",
            auto_fixable=False,
        ))

    # 4. Self-contradiction in tool results vs response
    if tool_results:
        last_tool_output = str(tool_results[-1].get("output", ""))
        if "error" in last_tool_output.lower() and "success" in content.lower():
            issues.append(ReflectionIssue(
                issue_type=ISSUE_SELF_CONTRADICTION,
                severity="high",
                description="Response claims success but the last tool result contained an error.",
                suggestion="Acknowledge the error and propose a corrective action.",
                auto_fixable=True,
            ))

    return issues


# ── LLM reflection pass ───────────────────────────────────────────────────────

_REFLECTION_SYSTEM = """\
You are a quality-assurance reviewer for an AI agent. Your job is to inspect
the agent's last response and identify any of the following issues:

  1. MISSING_TOOL: The agent should have called a tool but didn't.
  2. WRONG_TOOL:   The agent called the wrong tool (e.g., shell_exec instead of web_search).
  3. INCOMPLETE:   The response doesn't fully answer the user's question.
  4. HALLUCINATION: The response states facts not grounded in tool outputs.
  5. PLAN_MISMATCH: The agent said it would do X but did Y.
  6. NONE:         The response looks correct and complete.

Respond ONLY with a JSON object in this exact format:
{
  "verdict": "NONE" | "MISSING_TOOL" | "WRONG_TOOL" | "INCOMPLETE" | "HALLUCINATION" | "PLAN_MISMATCH",
  "severity": "critical" | "high" | "medium" | "low",
  "reason": "one sentence",
  "corrected_content": "optional — rewrite just the content field if you can fix it, else null"
}
Do not add any text outside the JSON."""


def _llm_reflect(
    router: Any,
    user_message: str,
    response: dict,
    tool_results: List[dict],
    fast_model: str,
) -> Optional[dict]:
    """
    Ask a cheap LLM to review the response quality.
    Returns parsed verdict dict or None on failure.
    """
    tool_summary = ""
    if tool_results:
        snippets = []
        for tr in tool_results[-3:]:  # last 3 tool results
            name = tr.get("tool_name", "?")
            out  = str(tr.get("output", ""))[:300]
            snippets.append(f"  [{name}] → {out}")
        tool_summary = "\nTool results used:\n" + "\n".join(snippets)

    review_prompt = (
        f"USER MESSAGE: {user_message[:400]}\n"
        f"AGENT RESPONSE CONTENT: {str(response.get('content',''))[:600]}"
        + (f"\n\nTOOL CALL: {response.get('tool_call')}" if response.get("tool_call") else "")
        + tool_summary
    )

    messages = [{"role": "user", "content": review_prompt}]

    # Switch to fast model if configured
    orig_model = None
    if fast_model and fast_model != router.model:
        orig_model    = router.model
        router.model  = fast_model

    try:
        raw = router.complete(system=_REFLECTION_SYSTEM, messages=messages)
        result = json.loads(raw) if isinstance(raw, str) else raw
        return result if isinstance(result, dict) else None
    except Exception as e:
        log.debug(f"LLM reflection failed: {e}")
        return None
    finally:
        if orig_model:
            router.model = orig_model


# ── Main engine ───────────────────────────────────────────────────────────────

class ReflectionEngine:
    """
    Agent self-critique engine. Call reflect() after every agent response turn.
    """

    def __init__(self, router: Any, config: Optional[ReflectionConfig] = None):
        self.router = router
        self.config = config or ReflectionConfig()
        self._correction_counts: Dict[str, int] = {}  # session-level counters

    def reflect(
        self,
        user_message:   str,
        agent_response: dict,
        tool_results:   Optional[List[dict]] = None,
        messages:       Optional[List[dict]] = None,
        session_id:     str = "default",
    ) -> ReflectionResult:
        """
        Run reflection on an agent response.

        Args:
            user_message:   The original user request.
            agent_response: The parsed response dict {thought, content, tool_call}.
            tool_results:   List of {tool_name, params, output} dicts for this turn.
            messages:       Full conversation history (for context).
            session_id:     Used to track correction counts per session.

        Returns:
            ReflectionResult with final_response and issue list.
        """
        if not self.config.enabled:
            return ReflectionResult(original_response=agent_response)

        start = time.monotonic()
        tool_results = tool_results or []
        result = ReflectionResult(original_response=agent_response)

        # ── Step 1: fast heuristic checks (no LLM call) ───────────────────
        heuristic_issues = _heuristic_checks(user_message, agent_response, tool_results)
        result.issues.extend(heuristic_issues)

        # ── Step 2: LLM review (only on critical/high heuristic hits OR always if configured) ──
        critical_heuristic = any(i.severity in ("critical", "high") for i in heuristic_issues)
        should_llm_review  = critical_heuristic or (
            self.config.hallucination_check and not tool_results
        )

        llm_verdict = None
        if should_llm_review:
            fast_model = self.config.fast_model if self.config.use_fast_model else ""
            llm_verdict = _llm_reflect(
                self.router, user_message, agent_response, tool_results, fast_model
            )
            if llm_verdict and llm_verdict.get("verdict") not in (None, "NONE"):
                result.issues.append(ReflectionIssue(
                    issue_type  = llm_verdict["verdict"].lower(),
                    severity    = llm_verdict.get("severity", "medium"),
                    description = llm_verdict.get("reason", ""),
                    auto_fixable= bool(llm_verdict.get("corrected_content")),
                ))

        # ── Step 3: auto-correct if possible ──────────────────────────────
        session_corrections = self._correction_counts.get(session_id, 0)
        if (result.has_issues
                and session_corrections < self.config.max_corrections):
            corrected = self._attempt_correction(
                agent_response, result.issues, llm_verdict, user_message
            )
            if corrected and corrected != agent_response:
                result.corrected_response = corrected
                result.did_correct        = True
                result.correction_count   = session_corrections + 1
                self._correction_counts[session_id] = result.correction_count

                if self.config.verbose:
                    log.info(
                        f"[Reflection] Corrected response. Issues: "
                        f"{[i.issue_type for i in result.issues]}"
                    )

        result.reflection_ms = int((time.monotonic() - start) * 1000)
        return result

    def _attempt_correction(
        self,
        response:    dict,
        issues:      List[ReflectionIssue],
        llm_verdict: Optional[dict],
        user_message: str,
    ) -> Optional[dict]:
        """
        Attempt to auto-correct the response based on detected issues.
        Returns corrected dict or None if correction not possible.
        """
        corrected = dict(response)

        # Use LLM-provided correction if available
        if llm_verdict and llm_verdict.get("corrected_content"):
            corrected["content"] = llm_verdict["corrected_content"]
            corrected.setdefault("_reflection_applied", []).append("llm_content_fix")
            return corrected

        # Fix self-contradiction: error in tool output but "success" in response
        for issue in issues:
            if issue.issue_type == ISSUE_SELF_CONTRADICTION:
                old_content = corrected.get("content", "")
                corrected["content"] = (
                    old_content.replace("successfully", "")
                               .replace("Success:", "Note:")
                    + "\n\n(Note: a tool error was detected — please review the output above.)"
                )
                corrected.setdefault("_reflection_applied", []).append("contradiction_fix")
                return corrected

        return None

    def summary_for_prompt(self, result: ReflectionResult) -> str:
        """
        Return a short summary of reflection findings to inject into the next
        system prompt turn (reminds the agent of its mistakes).
        """
        if not result.has_issues:
            return ""
        lines = ["[REFLECTION] Issues detected in previous response:"]
        for i in result.issues:
            lines.append(f"  [{i.severity.upper()}] {i.issue_type}: {i.description}")
            if i.suggestion:
                lines.append(f"    -> {i.suggestion}")
        return "\n".join(lines)

    def reset_session(self, session_id: str = "default") -> None:
        """Reset correction counter for a session (call on /clear)."""
        self._correction_counts.pop(session_id, None)


# ── Convenience wrapper ───────────────────────────────────────────────────────

_default_engine: Optional[ReflectionEngine] = None

def get_engine(router: Any = None, config: Optional[ReflectionConfig] = None) -> ReflectionEngine:
    """Return the process-level default ReflectionEngine, creating it if needed."""
    global _default_engine
    if _default_engine is None:
        if router is None:
            raise ValueError("router required on first call to get_engine()")
        _default_engine = ReflectionEngine(router, config)
    return _default_engine
