"""
Operon Multi-Agent Mesh — parallel specialised agent coordination.

Architecture inspired by OpenClaw's Lobster engine and Hermes's team_runner.py.

Unlike delegate.py (which runs anonymous sub-agents), the mesh assigns
named, specialised agents with distinct system prompts and toolsets:

  Researcher    — web search, knowledge retrieval, fact checking
  Coder         — code generation, execution, file operations, git
  Analyst       — data analysis, statistical reasoning, chart generation
  Writer        — long-form content, reports, emails, summarisation
  Reviewer      — quality assurance, critique, fact-check outputs
  Planner       — task decomposition, dependency ordering

Usage:
    from core.multi_agent import AgentMesh, AgentRole

    mesh = AgentMesh(router=router, tool_registry=registry)

    # Run a single specialised agent
    result = mesh.run_agent(AgentRole.RESEARCHER, "What is the current state of fusion energy?")

    # Run a pipeline: researcher → writer → reviewer
    pipeline_result = mesh.run_pipeline(
        task="Write a detailed report on fusion energy breakthroughs",
        roles=[AgentRole.RESEARCHER, AgentRole.WRITER, AgentRole.REVIEWER],
    )

    # Run agents in parallel and synthesise results
    parallel_result = mesh.run_parallel(
        task="Analyse this dataset and create a summary report",
        roles=[AgentRole.ANALYST, AgentRole.WRITER],
    )
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("operon.multi_agent")

# ── Agent roles ────────────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    CODER      = "coder"
    ANALYST    = "analyst"
    WRITER     = "writer"
    REVIEWER   = "reviewer"
    PLANNER    = "planner"
    GENERALIST = "generalist"
    # Hierarchical worker tier (constructive-tension trio)
    ENGINEER   = "engineer"   # The Execution Engineer — local file edit + code sandbox
    AUDITOR    = "auditor"    # The Quality Auditor — cynical review, linters, vuln scan


# Toolsets each role gets access to
_ROLE_TOOLSETS: Dict[AgentRole, List[str]] = {
    AgentRole.RESEARCHER: ["web_search", "http_get", "browser_navigate",
                           "browser_extract_text", "knowledge_save", "knowledge_query"],
    AgentRole.CODER:      ["shell_exec", "file_read", "file_write", "file_list",
                           "code_exec", "git_status", "git_diff", "apply_patch"],
    AgentRole.ANALYST:    ["file_read", "data_load", "data_describe", "data_query",
                           "data_groupby", "data_chart", "data_correlations", "data_anomalies"],
    AgentRole.WRITER:     ["file_read", "file_write", "pdf_create"],
    AgentRole.REVIEWER:   ["file_read", "web_search"],
    AgentRole.PLANNER:    ["file_read", "file_write"],
    AgentRole.GENERALIST: [],  # filled with "default" toolset at runtime
    # Execution Engineer — can read/write/patch files and run code in the sandbox,
    # but is steered (by persona) toward minimal, reviewable diffs.
    AgentRole.ENGINEER:   ["file_read", "file_write", "file_append", "file_patch",
                           "apply_patch", "dir_list", "file_search", "code_exec",
                           "shell_exec", "git_diff", "git_status"],
    # Quality Auditor — read-only-leaning: inspects code, runs linters/tests via
    # shell, greps logs, and diffs. Deliberately has NO write/patch tools so its
    # review stays independent of the Engineer's edits (constructive tension).
    AgentRole.AUDITOR:    ["file_read", "file_search", "dir_list", "shell_exec",
                           "code_exec", "git_diff", "git_status"],
}

# System prompt per role
_ROLE_SYSTEM_PROMPTS: Dict[AgentRole, str] = {
    AgentRole.RESEARCHER: (
        "You are a specialised research agent. Your job is to find accurate, "
        "up-to-date information using web search and knowledge tools. "
        "Always cite your sources. Prefer primary sources over secondary. "
        "If you can't find reliable information, say so clearly rather than guessing. "
        "Return well-structured findings with key facts highlighted."
    ),
    AgentRole.CODER: (
        "You are a specialised coding agent. Your job is to write, debug, and execute code. "
        "Always write clean, well-commented code. Test before submitting. "
        "Handle errors gracefully. Prefer idiomatic patterns for the language. "
        "When modifying existing code, preserve style and avoid unnecessary changes."
    ),
    AgentRole.ANALYST: (
        "You are a specialised data analysis agent. Your job is to load, explore, "
        "clean, and draw insights from tabular data. "
        "Always start with data_load to understand the shape and types. "
        "Detect anomalies and null values. Generate charts when they add clarity. "
        "Return statistical summaries with actionable insights."
    ),
    AgentRole.WRITER: (
        "You are a specialised writing agent. Your job is to produce clear, "
        "well-structured prose — reports, summaries, documentation, emails. "
        "Adapt your tone to the context (formal/informal). "
        "Use headings and bullet points for long documents. "
        "Always proofread for clarity before submitting."
    ),
    AgentRole.REVIEWER: (
        "You are a specialised quality-review agent. Your job is to critique "
        "the output of other agents. Check for: factual errors, missing steps, "
        "unclear explanations, logical inconsistencies, and completeness. "
        "Be constructive — for each issue, suggest a specific fix. "
        "Rate overall quality: PASS, NEEDS_REVISION, or FAIL."
    ),
    AgentRole.PLANNER: (
        "You are a specialised planning agent. Your job is to decompose complex "
        "tasks into ordered steps, identify dependencies, and assign each step "
        "to the most appropriate agent role. "
        "Return your plan as a structured JSON with: "
        '{\"steps\": [{\"step\": 1, \"role\": \"researcher\", \"task\": \"...\"}]}'
    ),
    AgentRole.GENERALIST: (
        "You are a general-purpose agent. Complete the given task using any "
        "available tools. Think step-by-step and verify your work."
    ),
    AgentRole.ENGINEER: (
        "You are the Execution Engineer in a hierarchical engineering team. "
        "Your job is to implement the objective by editing local files and "
        "running code in the sandbox. Work in SMALL, reviewable steps: read the "
        "relevant code first, make the minimal change that satisfies the "
        "objective, then run/compile it to confirm it works. Preserve existing "
        "style and never make sweeping unrelated edits. "
        "A cynical Quality Auditor WILL review your work and send back explicit "
        "fix instructions — when you receive auditor feedback, apply exactly the "
        "requested fixes and re-verify, do not argue or expand scope. "
        "When the change compiles/tests clean, summarise precisely what you "
        "changed and which files you touched."
    ),
    AgentRole.AUDITOR: (
        "You are the Quality Auditor — a deliberately cynical, detail-obsessed "
        "reviewer. You did NOT write the code, and you assume it is broken until "
        "proven otherwise. Your job is to find what is wrong: run linters and "
        "tests via the shell, read the fault logs, grep for error patterns, and "
        "inspect diffs. Look specifically for: compile/lint errors, failing "
        "tests, unhandled exceptions, security issues (injection, secrets, unsafe "
        "shell), missing edge cases, and silent failures. "
        "You have NO file-write tools on purpose — you critique, you do not patch. "
        "Return a verdict line of exactly PASS, NEEDS_REVISION, or FAIL, followed "
        "by a numbered list of CONCRETE fix instructions the Execution Engineer "
        "can act on directly (file + what to change + why). If it genuinely "
        "passes, say PASS and stop."
    ),
}


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    role:       AgentRole
    task:       str
    output:     str
    success:    bool
    error:      Optional[str] = None
    duration_s: float = 0.0
    tool_calls_made: int = 0

    def to_dict(self) -> dict:
        return {
            "role":       self.role.value,
            "task":       self.task[:200],
            "output":     self.output[:4000],
            "success":    self.success,
            "error":      self.error,
            "duration_s": round(self.duration_s, 2),
        }


@dataclass
class MeshResult:
    task:    str
    results: List[AgentResult] = field(default_factory=list)
    synthesis: str = ""
    success:  bool = True
    mode:     str = "single"   # "single" | "pipeline" | "parallel"

    @property
    def final_output(self) -> str:
        if self.synthesis:
            return self.synthesis
        if self.results:
            return self.results[-1].output
        return ""

    def to_dict(self) -> dict:
        return {
            "task":      self.task[:200],
            "mode":      self.mode,
            "success":   self.success,
            "synthesis": self.synthesis[:4000],
            "steps":     [r.to_dict() for r in self.results],
        }


# ── Core mesh ─────────────────────────────────────────────────────────────────

class AgentMesh:
    """
    Coordinates multiple specialised agents to complete complex tasks.
    Each agent runs in its own thread with its own system prompt and toolset.
    """

    def __init__(
        self,
        router:        Any,
        tool_registry: Any,
        max_workers:   int = 4,
        max_iterations: int = 8,
    ):
        self.router         = router
        self.tool_registry  = tool_registry
        self.max_workers    = max_workers
        self.max_iterations = max_iterations
        self._lock          = threading.Lock()

    # ── Single agent ──────────────────────────────────────────────────────────

    def run_agent(
        self,
        role:    AgentRole,
        task:    str,
        context: str = "",
        timeout: int = 120,
    ) -> AgentResult:
        """
        Run a single specialised agent on a task.

        Args:
            role:    Agent specialisation.
            task:    The task description.
            context: Additional context to inject into the agent's prompt.
            timeout: Max seconds to wait (default 120).

        Returns:
            AgentResult with output and success flag.
        """
        start = time.monotonic()
        try:
            output = self._run_with_timeout(role, task, context, timeout)
            return AgentResult(
                role=role, task=task, output=output, success=True,
                duration_s=time.monotonic() - start,
            )
        except Exception as e:
            return AgentResult(
                role=role, task=task, output="", success=False,
                error=str(e), duration_s=time.monotonic() - start,
            )

    # ── Pipeline (sequential, each step sees previous output) ────────────────

    def run_pipeline(
        self,
        task:    str,
        roles:   List[AgentRole],
        timeout_per_step: int = 120,
    ) -> MeshResult:
        """
        Run a chain of agents where each step receives the previous step's output.

        Example: PLANNER → RESEARCHER → WRITER → REVIEWER

        Args:
            task:             The initial task.
            roles:            Ordered list of agent roles.
            timeout_per_step: Per-step timeout (default 120s).

        Returns:
            MeshResult with all step outputs and the final agent's output.
        """
        mesh_result = MeshResult(task=task, mode="pipeline")
        current_context = task

        for role in roles:
            step_task = (
                f"{task}\n\n"
                f"Previous step output:\n{current_context[:3000]}"
                if mesh_result.results else task
            )
            result = self.run_agent(role, step_task, timeout=timeout_per_step)
            mesh_result.results.append(result)

            if not result.success:
                mesh_result.success = False
                log.warning(f"Pipeline step {role.value} failed: {result.error}")
                break  # stop pipeline on failure

            current_context = result.output
            log.info(f"Pipeline step {role.value} done ({result.duration_s:.1f}s)")

        mesh_result.synthesis = current_context
        return mesh_result

    # ── Parallel (concurrent, then synthesise) ───────────────────────────────

    def run_parallel(
        self,
        task:    str,
        roles:   List[AgentRole],
        timeout: int = 120,
        synthesise: bool = True,
    ) -> MeshResult:
        """
        Run multiple agents concurrently on the same task, then synthesise.

        Args:
            task:       The task (same for all agents).
            roles:      List of agent roles to run in parallel.
            timeout:    Per-agent timeout.
            synthesise: Call WRITER to synthesise outputs (default True).

        Returns:
            MeshResult with all agent outputs and a synthesised summary.
        """
        mesh_result = MeshResult(task=task, mode="parallel")

        if not roles:
            return mesh_result

        with ThreadPoolExecutor(max_workers=min(len(roles), self.max_workers)) as pool:
            futures = {
                pool.submit(self.run_agent, role, task, timeout=timeout): role
                for role in roles
            }
            for fut in as_completed(futures):
                result = fut.result()
                mesh_result.results.append(result)
                if not result.success:
                    log.warning(f"Parallel agent {result.role.value} failed: {result.error}")

        # Sort results by role order for determinism
        role_order = {r: i for i, r in enumerate(roles)}
        mesh_result.results.sort(key=lambda r: role_order.get(r.role, 99))

        # Synthesise if requested
        if synthesise and any(r.success for r in mesh_result.results):
            synthesis = self._synthesise(task, mesh_result.results)
            mesh_result.synthesis = synthesis
        elif mesh_result.results:
            mesh_result.synthesis = mesh_result.results[0].output

        mesh_result.success = any(r.success for r in mesh_result.results)
        return mesh_result

    # ── Auto-plan and execute ─────────────────────────────────────────────────

    def run_auto(
        self,
        task:    str,
        timeout_per_step: int = 120,
    ) -> MeshResult:
        """
        Automatically plan and execute a complex task using the PLANNER agent.

        The planner decomposes the task, assigns roles, then the mesh executes
        the plan either sequentially or in parallel based on dependencies.

        Args:
            task:             Complex multi-step task description.
            timeout_per_step: Per-step timeout.

        Returns:
            MeshResult with plan and execution results.
        """
        # Step 1: Plan
        plan_result = self.run_agent(
            AgentRole.PLANNER,
            f"Create an execution plan for this task:\n{task}",
            timeout=60,
        )

        if not plan_result.success:
            log.warning(f"Planning failed, falling back to generalist: {plan_result.error}")
            result = self.run_agent(AgentRole.GENERALIST, task)
            return MeshResult(
                task=task, mode="auto_fallback",
                results=[result], synthesis=result.output,
                success=result.success,
            )

        # Step 2: Parse plan
        steps = self._parse_plan(plan_result.output)
        if not steps:
            # Planner didn't return parseable JSON — use pipeline default
            return self.run_pipeline(task, [AgentRole.RESEARCHER, AgentRole.WRITER])

        # Step 3: Execute plan
        mesh_result = MeshResult(task=task, mode="auto")
        current_context = ""
        for step_info in steps[:6]:  # cap at 6 steps
            role_name = step_info.get("role", "generalist").lower()
            step_task = step_info.get("task", task)
            try:
                role = AgentRole(role_name)
            except ValueError:
                role = AgentRole.GENERALIST

            full_task = step_task + (f"\n\nPrevious output:\n{current_context[:2000]}"
                                     if current_context else "")
            result = self.run_agent(role, full_task, timeout=timeout_per_step)
            mesh_result.results.append(result)
            if result.success:
                current_context = result.output

        mesh_result.synthesis = current_context
        mesh_result.success    = any(r.success for r in mesh_result.results)
        return mesh_result

    # ── Agent factory: spawn a worker with an explicit tool allocation ─────────

    def run_with_tools(
        self,
        persona:         str,
        objective:       str,
        allocated_tools: Optional[List[str]] = None,
        context:         str = "",
        timeout:         int = 120,
    ) -> AgentResult:
        """
        Spawn a sandboxed sub-agent for *objective*, restricted to exactly
        *allocated_tools*. This is the runtime behind the ``spawn_agent`` meta-tool.

        *persona* may be a role name ("engineer", "auditor", "researcher", …) or
        free text. A recognised role supplies its persona prompt and — when no
        tools are explicitly allocated — its default toolset. Unknown personas
        fall back to the generalist prompt.

        All arguments degrade gracefully: a missing/empty tool list falls back to
        the persona's role toolset (or the default set), so the factory never
        raises on sparse input.
        """
        objective = (objective or "").strip() or "(no objective provided)"
        try:
            role = AgentRole((persona or "").strip().lower())
        except (ValueError, AttributeError):
            role = AgentRole.GENERALIST

        # Normalise the allocation: accept list, comma string, or None.
        if isinstance(allocated_tools, str):
            allocated_tools = [t.strip() for t in allocated_tools.split(",") if t.strip()]
        explicit = [t for t in (allocated_tools or []) if t]

        # If persona was free text (not a role) keep its description as a prefix.
        extra_context = context
        if role is AgentRole.GENERALIST and persona and persona.strip().lower() not in {r.value for r in AgentRole}:
            extra_context = (f"You are acting as: {persona}.\n\n" + context).strip()

        start = time.monotonic()
        try:
            output = self._run_with_timeout(
                role, objective, extra_context, timeout,
                explicit_tools=explicit or None,
            )
            return AgentResult(role=role, task=objective, output=output,
                               success=True, duration_s=time.monotonic() - start)
        except Exception as e:
            return AgentResult(role=role, task=objective, output="", success=False,
                               error=str(e), duration_s=time.monotonic() - start)

    # ── Autonomous self-correction: Engineer ⇄ Auditor loop ────────────────────

    def run_self_correction(
        self,
        objective:  str,
        verify_cmd: str = "",
        context:    str = "",
        max_rounds: int = 3,
        timeout_per_step: int = 150,
    ) -> MeshResult:
        """
        Drive an autonomous fix loop:

            Engineer drafts/edits → verify (shell command or Auditor review) →
            if it fails, the Auditor turns the fault log into explicit fix
            instructions → Engineer applies them → re-verify. Repeat until the
            objective verifies clean or *max_rounds* is reached.

        Args:
            objective:  What the Engineer must accomplish.
            verify_cmd: Optional shell command whose exit code / output decides
                        pass/fail (e.g. "python -m pytest -q"). When empty, the
                        Auditor's PASS/FAIL verdict is used instead.
            max_rounds: Hard cap on Engineer⇄Auditor iterations (anti-infinite-loop).

        Returns:
            MeshResult with one AgentResult per Engineer/Auditor turn and a
            synthesis describing the final state.
        """
        objective = (objective or "").strip() or "(no objective provided)"
        max_rounds = max(1, int(max_rounds or 1))
        mesh = MeshResult(task=objective, mode="self_correction")
        auditor_feedback = ""
        passed = False

        for rnd in range(1, max_rounds + 1):
            # 1) Engineer implements (or applies the auditor's fixes).
            eng_task = objective if rnd == 1 else (
                f"{objective}\n\nThe Quality Auditor reviewed the previous attempt "
                f"and returned these required fixes — apply them exactly:\n{auditor_feedback}"
            )
            eng = self.run_agent(AgentRole.ENGINEER, eng_task, context=context,
                                 timeout=timeout_per_step)
            mesh.results.append(eng)
            if not eng.success:
                mesh.synthesis = f"Engineer failed on round {rnd}: {eng.error}"
                break

            # 2) Verify — prefer a concrete command, else fall back to the Auditor.
            verify_report, verify_ok = self._verify(verify_cmd)

            # 3) Auditor review (always runs; sees code + any verify output).
            audit_task = (
                f"Review the Execution Engineer's work for this objective:\n{objective}\n\n"
                f"Engineer's report:\n{eng.output[:2000]}\n\n"
                + (f"Verification command `{verify_cmd}` output:\n{verify_report[:2000]}\n\n"
                   if verify_cmd else "")
                + "Give your verdict (PASS / NEEDS_REVISION / FAIL) and concrete fixes."
            )
            audit = self.run_agent(AgentRole.AUDITOR, audit_task, timeout=timeout_per_step)
            mesh.results.append(audit)
            auditor_feedback = audit.output

            verdict_pass = "PASS" in (audit.output or "").upper()[:400] and \
                "NEEDS_REVISION" not in (audit.output or "").upper()[:400] and \
                "FAIL" not in (audit.output or "").upper()[:400]

            # Pass when either the command verifies clean (if provided) AND/OR the
            # auditor signs off. A concrete verify_cmd is authoritative when present.
            if verify_cmd:
                passed = verify_ok and verdict_pass
            else:
                passed = verdict_pass

            if passed:
                mesh.synthesis = (
                    f"Verified clean after {rnd} round(s).\n\n"
                    f"Final engineer report:\n{eng.output[:1500]}"
                )
                break
            mesh.synthesis = (
                f"Did not verify after {rnd} round(s). Latest auditor feedback:\n"
                f"{auditor_feedback[:1500]}"
            )

        mesh.success = passed
        return mesh

    def _verify(self, verify_cmd: str) -> tuple:
        """Run an optional verification command; return (report, ok)."""
        if not verify_cmd:
            return "", True
        try:
            from tools.registry import _DISPATCH
            fn = _DISPATCH.get("shell_exec")
            if not fn:
                return "[verify] shell_exec unavailable", False
            res = fn(command=verify_cmd)
            if isinstance(res, dict):
                rc = res.get("exit_code", res.get("returncode", 0))
                out = (str(res.get("stdout", "")) + "\n" + str(res.get("stderr", ""))).strip()
                return out, (rc == 0)
            return str(res), True
        except Exception as e:
            return f"[verify] command raised: {e}", False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_with_timeout(
        self, role: AgentRole, task: str, context: str, timeout: int,
        explicit_tools: Optional[List[str]] = None,
    ) -> str:
        """
        Run a sub-agent with its own tool loop and return the final answer.

        If *explicit_tools* is given, the sub-agent is sandboxed to exactly that
        tool list (intersected with what's registered, minus the always-blocked
        delegate tools). Otherwise it falls back to the role's default toolset.
        A per-sub-agent ToolCallGuardrails instance is active so a worker can't
        get stuck in an infinite tool-calling loop.
        """
        from tools.registry import _DISPATCH, _TOOL_DEFINITIONS, TOOLSETS, DELEGATE_BLOCKED_TOOLS

        # Build toolset: explicit allocation wins; else fall back to role default.
        if explicit_tools:
            role_tools = [t for t in explicit_tools if t]   # drop empties gracefully
        else:
            role_tools = _ROLE_TOOLSETS.get(role, [])
        if not role_tools:
            # Generalist / empty allocation: use default toolset
            role_tools = list(TOOLSETS.get("default", {}).keys() if hasattr(TOOLSETS.get("default", {}), "keys") else TOOLSETS.get("default", []))

        allowed_names = set(role_tools)
        allowed_tools = [
            td for td in _TOOL_DEFINITIONS
            if td["name"] in allowed_names
            and td["name"] not in DELEGATE_BLOCKED_TOOLS
        ]

        # Per-sub-agent loop guardrails (anti-infinite-loop, shared design with
        # the primary agent loop in main.py).
        try:
            from core.tool_guardrails import ToolCallGuardrails
            guardrail = ToolCallGuardrails()
        except Exception:
            guardrail = None

        system = _ROLE_SYSTEM_PROMPTS.get(role, _ROLE_SYSTEM_PROMPTS[AgentRole.GENERALIST])
        if context:
            system += f"\n\nContext from parent agent:\n{context[:2000]}"

        # Build tool description for system prompt
        if allowed_tools:
            tool_desc = "\n".join(
                f"  {td['name']}: {td.get('description', '')}"
                for td in allowed_tools[:20]
            )
            system += f"\n\nAvailable tools:\n{tool_desc}"

        system += (
            "\n\nAlways respond with JSON: "
            '{"thought": "...", "content": "...", "tool_call": {"name": "...", "parameters": {...}}}'
            '\nOmit tool_call if no tool is needed.'
        )

        messages = [{"role": "user", "content": task}]
        output   = ""

        for iteration in range(self.max_iterations):
            try:
                raw = self.router.complete(system=system, messages=messages)
            except Exception as e:
                raise RuntimeError(f"LLM call failed: {e}")

            # Parse response
            parsed = None
            if hasattr(self.router, "parse_response"):
                parsed = self.router.parse_response(raw)
            if parsed is None:
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    parsed = {"content": str(raw)}

            content   = parsed.get("content", "") or ""
            tool_call = parsed.get("tool_call")

            if content:
                output = content

            if not tool_call:
                break  # done

            # Execute tool
            tool_name   = tool_call.get("name", "")
            tool_params = tool_call.get("parameters", {}) or {}

            # Sandbox enforcement: a worker may ONLY call tools in its allocation.
            if tool_name not in allowed_names:
                tool_result_str = (
                    f"[Error] Tool '{tool_name}' is not in this agent's allocated "
                    f"toolset. Allowed: {', '.join(sorted(allowed_names)) or '(none)'}."
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"[TOOL_RESULT: {tool_name}]\n{tool_result_str}"})
                continue

            # Guardrail: block before executing if this looks like a loop.
            if guardrail is not None:
                pre = guardrail.before_call(tool_name, tool_params)
                if pre.should_block:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": f"[TOOL_RESULT: {tool_name}]\n{pre.synthetic_result()}"})
                    if pre.action == "halt":
                        output = output or pre.message
                        break
                    continue

            if tool_name in DELEGATE_BLOCKED_TOOLS:
                tool_result_str = f"[Error] Tool '{tool_name}' is not available in sub-agents."
            else:
                try:
                    fn = _DISPATCH.get(tool_name)
                    if fn:
                        res = fn(**{k: v for k, v in tool_params.items() if v is not None})
                        tool_result_str = json.dumps(res) if isinstance(res, dict) else str(res)
                    else:
                        tool_result_str = f"[Error] Unknown tool: {tool_name}"
                except Exception as e:
                    tool_result_str = f"[Error] Tool {tool_name} failed: {e}"

            # Guardrail: record the result; append loop-warning guidance if any.
            if guardrail is not None:
                post = guardrail.after_call(tool_name, tool_params, tool_result_str)
                if post.action in ("warn", "halt"):
                    tool_result_str += post.guidance_suffix()
                if post.action == "halt":
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": f"[TOOL_RESULT: {tool_name}]\n{tool_result_str}"})
                    output = output or f"[{role.value}] Halted: repeated non-progressing tool calls."
                    break

            # Truncate large tool outputs
            if len(tool_result_str) > 4000:
                tool_result_str = tool_result_str[:2000] + "\n...[truncated]...\n" + tool_result_str[-500:]

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",      "content": f"[TOOL_RESULT: {tool_name}]\n{tool_result_str}"})

        return output or f"[{role.value}] Task completed (no text output)"

    def _synthesise(self, original_task: str, results: List[AgentResult]) -> str:
        """Ask the model to synthesise outputs from multiple parallel agents."""
        if not results:
            return ""
        # Build a combined context
        parts = [f"Original task: {original_task}\n"]
        for r in results:
            if r.success:
                parts.append(f"--- {r.role.value.upper()} output ---\n{r.output[:1500]}\n")

        synthesis_prompt = (
            "\n".join(parts)
            + "\n---\nSynthesize the above into a single coherent, complete response. "
            "Avoid repetition. Include all key findings."
        )
        try:
            raw = self.router.complete(
                system="You are a synthesis agent. Combine multiple expert outputs into one clear response.",
                messages=[{"role": "user", "content": synthesis_prompt}],
            )
            if hasattr(self.router, "parse_response"):
                parsed = self.router.parse_response(raw)
                if parsed:
                    return parsed.get("content", raw)
            return raw
        except Exception as e:
            log.warning(f"Synthesis failed: {e}")
            # Fall back to last successful result
            for r in reversed(results):
                if r.success:
                    return r.output
            return ""

    def _parse_plan(self, plan_text: str) -> List[dict]:
        """Extract steps from planner output."""
        import re
        # Try to find JSON
        json_match = re.search(r"\{[\s\S]+\}", plan_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("steps", [])
            except Exception:
                pass
        return []


# ── Convenience functions ──────────────────────────────────────────────────────

def create_mesh(router: Any, tool_registry: Any) -> AgentMesh:
    """Create an AgentMesh with default configuration."""
    return AgentMesh(router=router, tool_registry=tool_registry)
