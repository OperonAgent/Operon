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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_with_timeout(
        self, role: AgentRole, task: str, context: str, timeout: int
    ) -> str:
        """Run a sub-agent with its own tool loop and return the final answer."""
        from tools.registry import _DISPATCH, _TOOL_DEFINITIONS, TOOLSETS, DELEGATE_BLOCKED_TOOLS

        # Build toolset for this role
        role_tools = _ROLE_TOOLSETS.get(role, [])
        if not role_tools:
            # Generalist: use default toolset
            role_tools = list(TOOLSETS.get("default", {}).keys() if hasattr(TOOLSETS.get("default", {}), "keys") else TOOLSETS.get("default", []))

        allowed_tools = [
            td for td in _TOOL_DEFINITIONS
            if td["name"] in role_tools
            and td["name"] not in DELEGATE_BLOCKED_TOOLS
        ]

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
