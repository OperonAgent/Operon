#!/usr/bin/env python3
"""
Operon — Advanced AI Terminal Cockpit
Launch: python main.py

Combines the Hermes Planner reasoning schema with the Open-Claw tool
matrix into a single, zero-amnesia, multi-provider agentic REPL.

Feature set exceeding both OpenClaw and Hermes Agent:
  • 40 tools  •  6 providers (cloud + local)  •  Exponential backoff retry
  • Browser automation (Playwright)  •  Vision / image gen / TTS
  • Telegram gateway  •  SKILL.md instruction packs + Curator (auto-skill gen)
  • Session snapshots/rollback  •  Tool approval mode  •  Auto-truncation
  • MCP server support (JSON-RPC 2.0)  •  Web dashboard (localhost:7270)
  • SSH remote execution  •  X/Twitter search  •  Docker deployment
  • 38 slash commands  •  Context injection (.operon.md / AGENTS.md)
"""

import os
import sys
import json
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.config    import ConfigManager, LOCAL_PROVIDERS, LOCAL_HEALTH_URLS, PROVIDER_URLS
from core.session   import SessionManager
from core.memory    import MemoryPipeline
from core.router    import ModelRouter
from core.planner   import HermesPlannerRenderer
from core.soul      import SoulSystem
from core.scheduler import TaskScheduler
from core.skills     import SkillLoader
from core.gateway    import TelegramGateway
from core.mcp        import MCPManager
from core.dashboard  import DashboardServer, log_tool_call
from core.curator          import Curator
from core.knowledge        import KnowledgeBase
from core.cost_tracker     import CostTracker
from core.semantic_memory  import SemanticMemory
from core.rag              import RAGPipeline
from core.webhook_server   import WebhookServer
from core.orchestrator     import Orchestrator, AgentSpec
from core.secrets          import get_secrets
from core.heartbeat        import HeartbeatScheduler
from core.goal_tracker     import GoalTracker, as_system_block as _goals_system_block
from core.macros           import MacroManager
from core.retry_policy       import RetryPolicyManager, execute_with_retry, get_policy
from core.context_compressor import maybe_compress_messages
from core.plugin_sdk         import get_manager as get_plugin_manager

# ── Phase 11 modules (lazy-imported where safe, direct where small) ────────
try:
    from core.vector_memory    import get_vector_memory as _get_vector_memory, VectorMemory
    _VECTOR_MEMORY_AVAILABLE = True
except Exception:
    _VECTOR_MEMORY_AVAILABLE = False

try:
    from core.obsidian_memory  import get_obsidian_memory as _get_obsidian_memory, ObsidianMemory
    _OBSIDIAN_AVAILABLE = True
except Exception:
    _OBSIDIAN_AVAILABLE = False

try:
    from core.model_router     import SmartModelRouter, route_prompt, classify_prompt, strip_hints
    _SMART_ROUTER_AVAILABLE = True
except Exception:
    _SMART_ROUTER_AVAILABLE = False

try:
    from core.skill_synthesizer import SkillSynthesizer, get_synthesizer as _get_synthesizer
    _SKILL_SYNTH_AVAILABLE = True
except Exception:
    _SKILL_SYNTH_AVAILABLE = False

try:
    from core.computer_use     import ComputerUse, computer_use_status
    _COMPUTER_USE_AVAILABLE = True
except Exception:
    _COMPUTER_USE_AVAILABLE = False

# ── Phase 11 module-level singletons (set in main(), readable everywhere) ────
_smart_router: "SmartModelRouter | None" = None   # type: ignore
_skill_synth:  "SkillSynthesizer | None"  = None  # type: ignore

from tools.registry        import (
    ToolRegistry, set_sub_agent_runner, _TOOL_DEFINITIONS,
    TOOLSETS, enable_toolset, disable_toolset,
    _td_params, _td_required,
)
from tools.knowledge_ops   import set_knowledge_base as _set_knowledge_base_tool
from ui.theme  import Theme, ThinkingSpinner
from ui.banner import Banner
from ui.tui    import OperonTUI, TUI_AVAILABLE


# ── Global state ──────────────────────────────────────────────────────────────

_APPROVAL_MODE = False      # toggled via /approve on|off
_SAFE_TOOLS    = {          # never require approval
    "file_read", "file_exists", "file_info", "dir_list",
    "duckduckgo_search", "web_scrape", "file_search", "x_search",
    "browser_get_url", "browser_snapshot", "clarify", "todo",
}
_gateway:    "TelegramGateway | None"    = None   # live gateway instance
_dashboard:  "DashboardServer | None"   = None   # live dashboard instance
_mcp:        "MCPManager | None"        = None   # MCP server manager
_curator:    "Curator | None"           = None   # autonomous skill curator
_webhook:    "WebhookServer | None"     = None   # REST/webhook server
_rag:        "RAGPipeline | None"       = None   # RAG document pipeline
_secrets:    object                     = None   # SecretsManager singleton
_heartbeat:  "HeartbeatScheduler | None" = None  # passive background scheduler
_goals:      "GoalTracker | None"       = None   # persistent goal tracker
_macros:     "MacroManager | None"      = None   # pipeline macro manager
_retry_mgr:  "RetryPolicyManager | None" = None  # per-tool retry policy manager
_plugins:    object                     = None   # PluginManager singleton


# ── Context injection ─────────────────────────────────────────────────────────

def _load_context_files() -> str:
    """
    Auto-load .operon.md / AGENTS.md / CLAUDE.md / .cursorrules from cwd.
    Same concept as Hermes .hermes.md and OpenClaw AGENTS.md.
    """
    candidates = [".operon.md", "AGENTS.md", "CLAUDE.md", ".cursorrules"]
    parts = []
    for name in candidates:
        p = Path(name)
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8").strip()
                if text:
                    parts.append(f"# [{name}]\n{text}")
            except Exception:
                pass
    return "\n\n".join(parts)


# ── System prompt factory ─────────────────────────────────────────────────────

def build_local_system_prompt(
    tool_registry:  "ToolRegistry",
    memory:         "MemoryPipeline",
    soul:           "SoulSystem | None",
    context_inject: str = "",
    skills:         "SkillLoader | None" = None,
    knowledge:      "KnowledgeBase | None" = None,
) -> str:
    """
    Simplified system prompt for small local models (llama3.2, mistral-7b, etc.).
    Uses a minimal JSON schema — two forms only — so the model can follow it reliably.
    """
    import os as _os
    # Local models have small context windows (~8k tokens) — showing all 98
    # tools burns ~2,500 tokens and leaves almost no room for conversation.
    # Show only the 25 most commonly needed tools; the rest are still callable
    # (the model can always ask for help with /tools to see the full list).
    _LOCAL_CORE_TOOLS = {
        "file_read", "file_write", "file_append", "file_delete", "file_exists",
        "dir_list", "file_search",
        "shell_exec", "python_exec",
        "duckduckgo_search", "web_scrape",
        "browser_open", "browser_get_url", "browser_click",
        "knowledge_set", "knowledge_get",
        "email_draft",
        "todo", "clarify",
        "git_status", "git_diff",
        "docker_run",
        "sub_agent",
        "goal_set", "goal_list",
    }
    _all_descs = tool_registry.get_compact_descriptions()
    _core_lines = [
        ln for ln in _all_descs.splitlines()
        if any(ln.strip().startswith(t) for t in _LOCAL_CORE_TOOLS)
    ]
    tools_block = "\n".join(_core_lines) + (
        f"\n  [+{len(_all_descs.splitlines()) - len(_core_lines)} more tools available — "
        "ask if you need one not listed here]"
    )
    memory_block = memory.get_context_string()
    soul_block   = soul.as_system_block() if soul is not None else ""
    kb_block     = knowledge.as_system_block() if knowledge is not None else ""
    mem_section  = f"\n\nMemory:\n{memory_block}" if memory_block else ""
    kb_section   = f"\n\n{kb_block}" if kb_block else ""
    soul_prefix  = f"{soul_block}\n\n" if soul_block else ""
    ctx_section  = f"\n\n{context_inject}" if context_inject else ""
    cwd          = _os.getcwd()

    return f"""{soul_prefix}You are Operon, an AI terminal assistant with tool access.
Current working directory: {cwd}{kb_section}{mem_section}{ctx_section}

RESPONSE FORMAT — output ONE JSON object only, nothing outside it.

Text reply:  {{"reply": "your answer here"}}
Tool call:   {{"tool": "EXACT_TOOL_NAME", "params": {{"key": "value"}}}}

RULES:
- Output ONLY valid JSON. No text, no markdown outside the JSON.
- Greetings/chat/general questions: ALWAYS use reply format immediately. NEVER call a tool.
  Examples that need ONLY reply — no tools:
    "hi" / "hello"                → {{"reply": "Hello! How can I help?"}}
    "who are you" / "what are you" → {{"reply": "I am Operon, an AI terminal assistant."}}
    "how are you"                 → {{"reply": "I'm ready to help!"}}
    "what can you do"             → {{"reply": "I can search the web, run code, manage files…"}}
    "how to drink water"          → {{"reply": "Lift a glass to your lips and sip."}}
    Any general knowledge question → reply directly. No tool needed.
- DO NOT call db_query unless the user explicitly asks about a DATABASE or SQL tables.
  "Who is X?" or "What is Y?" are NOT database queries — answer with reply.
- DO NOT call knowledge_get unless the user says "what did I save" or "what do you remember".
  General questions are NOT knowledge lookups — answer with reply.
- DO NOT call duckduckgo_search unless the user says "search for X" or needs live web data.
- Include ALL required params when calling a tool. NEVER call a tool with empty params.
  If a required param is missing from the user's message, use clarify to ask for it.
- After a tool error, read the error message carefully and retry with CORRECT params.
- NEVER write code for the user to run. Use shell_exec or python_exec yourself.
- When the user tells you their name, preferences, paths, or any important fact,
  IMMEDIATELY call knowledge_set to save it permanently.
  SPECIAL CASES — save with these exact key names so the email tool finds them:
    "my email is X"         → knowledge_set(key="sender_email", value="X")
    "my app password is X"  → knowledge_set(key="app_password", value="X")
    "my name is X"          → knowledge_set(key="user_name", value="X")
- STOP after a task succeeds. If a tool returns success, reply to the user immediately.
  Do NOT call the same tool again. Do NOT call clarify after success.
- clarify is ONLY for asking the USER for missing info (e.g. "Which file?", "What address?").
  NEVER use clarify to confirm tool results or ask "did it work?"

TOOL PATTERNS (exact names required):
- List/find .py files:  {{"tool": "shell_exec", "params": {{"command": "find . -name '*.py' | xargs wc -l"}}}}
- Search file content:  {{"tool": "file_search", "params": {{"pattern": "import requests", "path": ".", "file_pattern": "*.py"}}}}
- Run Python:           {{"tool": "python_exec", "params": {{"code": "print('hello')"}}}}
- Web/Google search:    {{"tool": "duckduckgo_search", "params": {{"query": "your query"}}}}
- List directory:       {{"tool": "dir_list", "params": {{"path": "."}}}}
- Read a file:          {{"tool": "file_read", "params": {{"path": "/full/path/to/file.py"}}}}
- Save a fact forever:  {{"tool": "knowledge_set", "params": {{"key": "user_name", "value": "Alice"}}}}
- Send an email:        {{"tool": "email_draft", "params": {{"to": "any@address.com", "subject": "Hello", "body": "Hi,\n\nHope you are well!\n\nBest"}}}}
- Email multiple people:{{"tool": "email_draft", "params": {{"to": "alice@x.com, bob@y.com", "cc": "manager@z.com", "subject": "Update", "body": "Hi team,\n\nHere is the update.\n\nBest"}}}}
  EMAIL RULES (short):
  • ALWAYS use email_draft — never email_send (it does not exist).
  • 'to' accepts ANY email address or multiple addresses (comma-separated).
  • Copy ALL addresses EXACTLY from the user's message — never alter them.
  • body must be a plain text string — not a dict. Start with "Hi Name,\n\n".
  • Optional: cc, bcc, reply_to, attachments (list of file paths).
  • NEVER pass sender_email or app_password — credentials load automatically.
  • If email_draft returns cancelled=true, reply "Draft discarded." and stop.

TOOL NAME ALIASES (correct spelling only):
  "web search" or "google" = duckduckgo_search
  "x/twitter" = x_search
  "list files" = dir_list or shell_exec
  "run code" = python_exec or shell_exec

AVAILABLE TOOLS (name: description [required params]):
{tools_block}
"""


def build_system_prompt(
    tool_registry:  ToolRegistry,
    memory:         MemoryPipeline,
    soul:           "SoulSystem | None",
    context_inject: str = "",
    skills:         "SkillLoader | None" = None,
    is_local:       bool = False,
    knowledge:      "KnowledgeBase | None" = None,
) -> str:
    """
    Three-tier system prompt (adapted from Hermes Agent system_prompt.py):

      STABLE   — role identity + tools + core instructions
                 Never changes turn-to-turn → Anthropic prefix cache always hits.
      CONTEXT  — skills + AGENTS.md context injection
                 Changes only when skill files change.
      VOLATILE — memory + goals + knowledge base facts
                 Changes every turn; injected separately so stable tier stays cached.

    For Anthropic, the router splits these into separate cache blocks.
    For other providers they are concatenated normally.
    """
    if is_local:
        return build_local_system_prompt(
            tool_registry, memory, soul, context_inject, skills, knowledge)

    tools_block   = tool_registry.get_descriptions()
    memory_block  = memory.get_context_string()
    soul_block    = soul.as_system_block() if soul is not None else ""
    skills_block  = skills.as_system_block() if skills is not None else ""
    kb_block      = knowledge.as_system_block() if knowledge is not None else ""
    goals_block   = _goals_system_block() if _goals is not None else ""
    mem_section   = f"\n\n{memory_block}"   if memory_block   else ""
    kb_section    = f"\n\n{kb_block}"       if kb_block       else ""
    soul_prefix   = f"{soul_block}\n\n"     if soul_block     else ""
    ctx_section   = f"\n\n{context_inject}" if context_inject else ""
    skills_section = f"\n\n{skills_block}"  if skills_block   else ""
    goals_section  = f"\n\n{goals_block}"   if goals_block    else ""

    # ── Three-tier system prompt (Hermes stable/context/volatile split) ──────────
    # STABLE tier: identity + tools + core instructions — never changes per-turn
    # → Anthropic prefix cache always hits this portion (~2k tokens saved/turn)
    # CONTEXT tier: skills + AGENTS.md — changes when files change
    # VOLATILE tier: memory + goals + KB — changes every turn (injected separately)
    #
    # The router handles splitting for Anthropic; other providers get it all joined.
    # ─────────────────────────────────────────────────────────────────────────────

    return f"""{soul_prefix}
You are Operon, an advanced AI Terminal Cockpit agent with full local system access.
You are precise, autonomous, and extraordinarily capable.{ctx_section}{skills_section}{kb_section}{mem_section}{goals_section}

════════════════════════════════════════════════════
STRICT RESPONSE FORMAT — ALWAYS VALID JSON
════════════════════════════════════════════════════
Every response MUST be a single JSON object.

To call a tool:
{{
  "scratchpad": {{
    "objective":      "Current goal in 1-2 sentences",
    "workspace_vars": {{"key": "relevant state values"}},
    "code_draft":     "Any code being written (empty string if N/A)",
    "next_step":      "The single most immediate next action"
  }},
  "action": {{
    "type":      "tool",
    "tool_name": "exact_tool_name",
    "params":    {{}}
  }}
}}

CRITICAL: action.type MUST be the literal string "tool" — never the tool's name.
CRITICAL: action.tool_name MUST be the tool name (e.g. "python_exec"), never "tool".
CRITICAL: NEVER use a "tools" key — the correct key is "action".
CRITICAL: NEVER describe how to use a tool — actually call it using the action format above.
CRITICAL: NEVER output prose + JSON — output ONLY the JSON object, nothing else.
CRITICAL: NEVER fabricate a result like {{"success": true}} — you have NOT called a tool until action format is used.
CRITICAL: NEVER put the tool name as a top-level JSON key like {{"email_send": {{...}}}} — always use "action": {{"type":"tool","tool_name":"email_send","params":{{...}}}}.
CRITICAL: You are an AGENT that DOES THINGS, not a coding assistant that EXPLAINS THINGS.
  - User says "list files"   → call shell_exec or python_exec RIGHT NOW — do NOT show example code.
  - User says "count lines"  → call wc -l via shell_exec — do NOT write a script for them to run.
  - User says "search X"     → call web_search — do NOT describe what a search would return.
  - User says "run this"     → call python_exec — do NOT echo the code back.
  NEVER say "here is a script you can run" or "save this to a file". RUN IT YOURSELF with tools.

To give a final answer:
{{
  "scratchpad": {{
    "objective":      "...",
    "workspace_vars": {{}},
    "code_draft":     "",
    "next_step":      "Deliver final response"
  }},
  "action": {{
    "type":    "response",
    "content": "Your full answer to the user here."
  }}
}}

════════════════════════════════════════════════════
EXAMPLES — ALWAYS FOLLOW THESE PATTERNS EXACTLY
════════════════════════════════════════════════════
User: "list all python files and count the lines in each"
Your ONLY valid response (use a tool — do NOT write a script for the user):
{{
  "scratchpad": {{
    "objective": "List Python files and count lines in each",
    "workspace_vars": {{}},
    "code_draft": "",
    "next_step": "run shell_exec with find + wc -l"
  }},
  "action": {{
    "type": "tool",
    "tool_name": "shell_exec",
    "params": {{"command": "find . -maxdepth 1 -name '*.py' | sort | xargs wc -l 2>/dev/null"}}
  }}
}}

User: "search for the latest news about Claude 4"
Your ONLY valid response (call web_search — do NOT describe what you would find):
{{
  "scratchpad": {{
    "objective": "Search for latest Claude 4 news",
    "workspace_vars": {{}},
    "code_draft": "",
    "next_step": "call web_search"
  }},
  "action": {{
    "type": "tool",
    "tool_name": "web_search",
    "params": {{"query": "Claude 4 latest news 2026"}}
  }}
}}

User: "what is 2 + 2?"
Your ONLY valid response (pure knowledge — no tool needed):
{{
  "scratchpad": {{
    "objective": "Answer arithmetic question",
    "workspace_vars": {{}},
    "code_draft": "",
    "next_step": "deliver answer"
  }},
  "action": {{
    "type": "response",
    "content": "4"
  }}
}}

User: "hi" / "hello" / "hey" / any greeting or casual chat
Your ONLY valid response (NEVER call clarify or any tool for greetings):
{{
  "scratchpad": {{
    "objective": "Respond to greeting",
    "workspace_vars": {{}},
    "code_draft": "",
    "next_step": "deliver friendly reply"
  }},
  "action": {{
    "type": "response",
    "content": "Hey! I'm Operon, your AI Terminal Cockpit. What can I help you with today?"
  }}
}}

════════════════════════════════════════════════════
AVAILABLE TOOLS
════════════════════════════════════════════════════
{tools_block}

════════════════════════════════════════════════════
PARALLEL TOOL EXECUTION
════════════════════════════════════════════════════
When a task requires multiple independent lookups simultaneously, use:
{{
  "action": {{
    "type": "parallel_tools",
    "calls": [
      {{"tool_name": "duckduckgo_search", "params": {{"query": "topic A"}}, "id": "s1"}},
      {{"tool_name": "web_scrape",        "params": {{"url": "https://..."}}, "id": "s2"}}
    ]
  }}
}}
Use parallel_tools ONLY for truly independent calls (max 4). Each call needs a unique "id".

════════════════════════════════════════════════════
OPERATING RULES
════════════════════════════════════════════════════
1. ALWAYS fill out the scratchpad before acting — it is your thinking space.
2. EXECUTE, don't explain. Any task involving computation, files, shell, web, or code MUST
   be performed by calling the appropriate tool — never by showing the user code to run themselves.
   Examples: "list .py files" → shell_exec("ls *.py") or python_exec. "count lines" → shell_exec("wc -l").
   "search X" → web_search. "create a file" → file_write. "run this" → python_exec.
3. Chain multiple tool calls to complete complex tasks end-to-end without asking permission.
4. When writing files, use file_write with the full content — no placeholders, no TODOs.
5. When uncertain about a file's existence, use file_exists first.
6. Always set type="response" for your final answer to the user.
7. Keep workspace_vars updated with important values across turns.
8. For greetings ("hi", "hello", "hey") and casual chat, respond directly with a warm reply —
   NEVER call clarify, NEVER call any tool. Just use action.type="response".
9. Use clarify ONLY when a specific required piece of info is missing and cannot be inferred
   (e.g., "Which host?", "What filename?"). NEVER use clarify for vague uncertainty or
   for questions about your own process ("What should I do next?").
   NEVER use clarify to confirm or verify tool results — the tool result IS your confirmation.
   NEVER ask "did it work?" or "can you confirm?" after a successful tool call.
10. Use the todo tool to track sub-tasks in complex multi-step work.
11. For browser tasks: navigate → snapshot → interact → screenshot to verify.
12. STOP looping after a task completes. If you sent an email, sent a message, wrote a file,
    or completed any single action successfully — deliver your response immediately.
    Do NOT call the same tool twice in a row unless the first attempt explicitly failed.
13. "What can you do?" / "list capabilities" → respond with text only, no tool calls needed.
14. For email tasks, use email_draft — it is the ONLY email tool available to you.
    email_send does NOT exist as a callable tool; never attempt to call it.
    email_draft accepts ANY recipient address — there is no restriction on who you can email.
    The 'to' field supports multiple recipients: "alice@x.com, bob@y.com".
    Optional params: cc, bcc, reply_to, attachments (list of paths).
    Credentials are loaded automatically — NEVER pass sender_email or app_password to email_draft,
    and NEVER ask the user for credentials in chat.
    ALWAYS copy the 'to' (and cc/bcc) addresses VERBATIM from the user's message — never alter,
    guess, or invent an email address. If user says "send to alice@example.com", the 'to' param
    MUST be "alice@example.com" — not "alice@other.com" or any variant.
    Write a subject line that SPECIFICALLY describes THIS email and THIS user request.
    NEVER reuse a subject from a previous email in the conversation — generate a fresh,
    accurate subject every time. Examples:
      "asking 20 questions about them"  → subject: "A Few Questions for You"
      "asking 10 questions about OpenAI" → subject: "10 Questions About OpenAI"
      "follow-up on our project"         → subject: "Project Follow-Up"
    Write a complete, well-composed email from the user's brief description:
    "ask X 10 questions about Y" → write ALL 10 REAL, specific questions about Y.
    NEVER use placeholder lines like "Question one?", "Question two?", "Item 1:", "[insert]".
    Every sentence must be real content — not a template. The topic must match what the
    user asked for (if they say "OpenAI", write questions about OpenAI, not about yourself).
    The 'body' param MUST be a plain text STRING starting with a greeting, containing all
    content, ending with a sign-off. NEVER a JSON object or dict.
    If email_draft returns cancelled=true → STOP immediately. Reply "Draft discarded."
    If email_draft returns approved=false with non-empty feedback → incorporate feedback
    and call email_draft again with a revised draft.
"""


# ── Local model helper ────────────────────────────────────────────────────────

def _probe_local_servers() -> dict:
    import requests as _req
    results = {}
    for prov, health_url in LOCAL_HEALTH_URLS.items():
        if prov == "local":
            continue
        entry = {"running": False, "models": [], "url": PROVIDER_URLS[prov]}
        try:
            resp = _req.get(health_url, timeout=1.5)
            if resp.ok:
                entry["running"] = True
                data = resp.json()
                if "models" in data:
                    entry["models"] = [m.get("name", "") for m in data["models"]]
                elif "data" in data:
                    entry["models"] = [m.get("id", "") for m in data["data"]]
        except Exception:
            pass
        results[prov] = entry
    return results


def _handle_local(parts: list, config: ConfigManager, theme: Theme) -> None:
    sub = parts[1].lower() if len(parts) > 1 else ""

    if sub == "use":
        if len(parts) < 3:
            print(theme.warning("Usage: /local use <model>"))
            return
        model_arg = parts[2]
        if ":" not in model_arg:
            provider, model_id, prof_name = "ollama", model_arg, f"ollama:{model_arg}"
        else:
            provider, model_id = model_arg.split(":", 1)
            prof_name = model_arg
        if provider not in LOCAL_PROVIDERS:
            print(theme.warning(f"Unknown provider '{provider}'. Valid: {sorted(LOCAL_PROVIDERS)}"))
            return
        profiles = config.get("model_profiles", {})
        profiles[prof_name] = {"provider": provider, "model_id": model_id}
        config.set("model_profiles", profiles)
        config.set("default_model", prof_name)
        config.set("active_provider", provider)
        print(theme.success(f"Switched to: {model_id}  [{provider} @ {PROVIDER_URLS[provider]}]"))
        return

    if sub == "url":
        if len(parts) < 4:
            print(theme.warning("Usage: /local url <provider> <base_url>"))
            return
        prov, url = parts[2].lower(), parts[3]
        if prov not in LOCAL_PROVIDERS:
            print(theme.warning(f"Unknown provider '{prov}'."))
            return
        PROVIDER_URLS[prov] = url
        print(theme.success(f"Updated {prov} → {url}"))
        return

    print(theme.dim("  Probing local servers…"))
    servers = _probe_local_servers()
    lines   = ["  LOCAL MODEL SERVERS", "---"]
    any_on  = False
    for prov, info in servers.items():
        if info["running"]:
            any_on = True
            lines.append(f"  ● {prov:<10} RUNNING  {info['url']}")
            for m in info["models"][:8]:
                lines.append(f"      ↳ {m}")
            if not info["models"]:
                lines.append("      ↳ (no models loaded)")
        else:
            lines.append(f"  ○ {prov:<10} offline  {info['url']}")
    lines.append("---")
    if any_on:
        lines.append("  Switch: /local use <model>  or  /local use ollama:<model>")
    else:
        lines.append("  No local servers detected. Install: https://ollama.com")
    print(theme.box(lines))


# ── Doctor ────────────────────────────────────────────────────────────────────

def _run_doctor(config: ConfigManager, memory: MemoryPipeline,
                tool_registry: ToolRegistry, skills: SkillLoader,
                theme: Theme, mcp=None, dashboard=None, curator_obj=None) -> None:
    import importlib
    lines = ["  OPERON DOCTOR — SYSTEM HEALTH CHECK", "---"]

    # Config
    lines.append(f"  {'✓' if config.is_configured() else '✗'} Config              {'OK' if config.is_configured() else 'NOT SET — run /setup'}")

    # API keys
    for prov in ("openai", "anthropic", "openrouter"):
        key = config.get_api_key(prov)
        lines.append(f"    API key [{prov:<12}]  {'SET ✓' if key else 'NOT SET'}")

    # Telegram
    tg = config.get("telegram_token", "") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    lines.append(f"    Telegram token      {'SET ✓' if tg else 'NOT SET (optional)'}")

    lines.append(f"  ✓ Default model      {config.get('default_model', '?')}")

    # Local servers
    lines.append("---")
    lines.append("  LOCAL SERVERS:")
    try:
        for prov, info in _probe_local_servers().items():
            sym = "●" if info["running"] else "○"
            lines.append(f"    {sym} {prov:<12} {'RUNNING' if info['running'] else 'offline'}")
    except Exception as e:
        lines.append(f"    error: {e}")

    # Tools
    lines.append("---")
    lines.append(f"  ✓ Tools registered   {len(tool_registry.tools)}")

    # Skills
    lines.append(f"  {'✓' if len(skills) else '○'} Skills loaded       {len(skills)}  (~/.operon/skills/)")

    # Memory
    lines.append(f"  {'✓' if config.get('memory_enabled', True) else '○'} Memory              {'enabled' if config.get('memory_enabled', True) else 'disabled'} — {len(memory.get_all())} items")

    # Gateway
    global _gateway, _dashboard, _mcp, _curator
    lines.append(f"  {'✓' if (_gateway and _gateway.status()['running']) else '○'} Telegram gateway    {'running' if (_gateway and _gateway.status()['running']) else 'stopped'}")
    db = dashboard or _dashboard
    lines.append(f"  {'✓' if (db and db.running) else '○'} Web dashboard       {'running → ' + db.url if (db and db.running) else 'stopped  (/dashboard start)'}")
    mc = mcp or _mcp
    if mc:
        srv_count = len(mc.status())
        lines.append(f"  {'✓' if srv_count else '○'} MCP servers         {srv_count} connected  (/mcp list)")
    else:
        lines.append("  ○ MCP servers        not loaded")
    cur = curator_obj or _curator
    lines.append(f"  {'✓' if (cur and cur.enabled) else '○'} Curator             {'ON' if (cur and cur.enabled) else 'OFF'}  — auto-skill generation")

    # New systems
    lines.append("---")
    lines.append("  NEW SYSTEMS:")
    wh = _webhook or globals().get("_webhook")
    lines.append(f"  {'✓' if (wh and wh.running) else '○'} Webhook server      {'running → ' + wh.url if (wh and wh.running) else 'stopped  (/webhook start)'}")
    rg = _rag or globals().get("_rag")
    if rg:
        st_rag = rg.stats()
        lines.append(f"  ✓ RAG pipeline      {st_rag['total_chunks']} chunks / {st_rag['total_sources']} sources")
    else:
        lines.append("  ○ RAG pipeline      not started  (/rag stats to init)")
    sec = _secrets or globals().get("_secrets")
    if sec:
        sec_st = sec.status()
        lines.append(f"  ✓ Secrets manager   {sec_st['backend']}  ({sec_st['key_count']} keys)")
    else:
        lines.append("  ○ Secrets manager   not loaded")

    # Dependencies
    lines.append("---")
    lines.append("  PYTHON DEPENDENCIES:")
    deps = {
        "requests":          "required",
        "psutil":            "optional (telemetry)",
        "pypdf":             "optional (PDF reading / RAG)",
        "reportlab":         "optional (PDF generation)",
        "playwright":        "optional (browser automation)",
        "duckduckgo_search": "optional (web search)",
        "bs4":               "optional (web scraping / X search)",
        "paramiko":          "optional (SSH remote execution)",
        "sounddevice":       "optional (voice recording)",
        "whisper":           "optional (local STT — pip install openai-whisper)",
        "pyttsx3":           "optional (offline TTS)",
        "psycopg2":          "optional (PostgreSQL)",
        "pymongo":           "optional (MongoDB)",
        "discord":           "optional (Discord bot API — pip install py-cord)",
        "twilio":            "optional (WhatsApp via Twilio)",
        "keyring":           "optional (OS keychain for secrets)",
        "cryptography":      "optional (Fernet secrets fallback)",
        "docker":            "optional (Docker Python SDK)",
    }
    for dep, note in deps.items():
        try:
            importlib.import_module(dep.replace("-", "_").replace(".", "_"))
            lines.append(f"    ✓ {dep}")
        except (ImportError, Exception):
            req = "required" in note
            lines.append(f"    {'✗' if req else '○'} {dep}  ({note})")

    # Playwright browser
    try:
        from playwright.sync_api import sync_playwright
        lines.append("    ✓ playwright chromium  (browser automation ready)")
    except ImportError:
        lines.append("    ○ playwright not installed  (run: pip install playwright && playwright install chromium)")

    # Context / soul files
    lines.append("---")
    soul_ok = (Path.home() / ".operon" / "SOUL.md").exists()
    lines.append(f"  {'✓' if soul_ok else '○'} SOUL.md             {'exists' if soul_ok else 'not found'}")
    ctx_found = [n for n in [".operon.md", "AGENTS.md", "CLAUDE.md"] if Path(n).exists()]
    lines.append(f"  {'✓' if ctx_found else '○'} Context files       {', '.join(ctx_found) if ctx_found else 'none in cwd'}")

    # Security & advanced checks from core.doctor
    try:
        from core.doctor import (
            check_security_email_send, check_security_weak_secrets,
            check_prompt_injection_module, check_command_risk_module,
            check_disk_space, check_dependency_vulnerabilities,
        )
        lines.append("---")
        lines.append("  SECURITY CHECKS:")
        for fn in (check_security_email_send, check_security_weak_secrets,
                   check_prompt_injection_module, check_command_risk_module,
                   check_dependency_vulnerabilities, check_disk_space):
            r = fn()
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗", "skip": "○"}[r.status.value]
            lines.append(f"    {icon} {r.name:<35} {r.message}")
    except Exception as e:
        lines.append(f"  ○ Security checks    error: {e}")

    print(theme.box(lines))


# ── Dashboard status builder ─────────────────────────────────────────────────

def _build_dashboard_status(config, session, memory, tool_registry, skills) -> dict:
    status = {
        "model":        config.get("default_model", "—"),
        "provider":     config.get("active_provider", "—"),
        "turns":        session.turn_count,
        "messages":     len(session),
        "memory_items": len(memory.get_all()),
        "skills":       len(skills) if skills else 0,
        "tools":        len(tool_registry.tools) if tool_registry else 0,
        "cpu":          "—",
        "ram":          "—",
    }
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        vm  = psutil.virtual_memory()
        status["cpu"] = f"{cpu:.0f}%"
        status["ram"] = f"{vm.percent:.0f}%"
    except ImportError:
        pass
    return status


# ── Slash-command handler ─────────────────────────────────────────────────────

def handle_command(
    command:        str,
    config:         ConfigManager,
    session:        SessionManager,
    memory:         MemoryPipeline,
    theme:          Theme,
    soul:           SoulSystem              = None,
    scheduler:      TaskScheduler           = None,
    tool_registry:  ToolRegistry            = None,
    router:         ModelRouter             = None,
    context_inject: str                     = "",
    planner:        HermesPlannerRenderer   = None,
    skills:         SkillLoader             = None,
    curator:        "Curator | None"        = None,
    cost_tracker:   "CostTracker | None"    = None,
    semantic_mem:   "SemanticMemory | None" = None,
    knowledge:      "KnowledgeBase | None"  = None,
) -> None:
    global _APPROVAL_MODE, _gateway, _dashboard, _mcp, _curator, _webhook, _rag, _secrets
    global _heartbeat, _goals, _macros, _retry_mgr, _plugins

    parts = command.strip().split()
    cmd   = parts[0].lower()
    args  = parts[1:]          # convenience: sub-command arguments list

    # ── Modular dispatch (cmd_handlers package) ────────────────────────────────
    # Extracted, individually-tested handlers run first; anything not yet
    # migrated falls through to the legacy elif-chain below unchanged.
    try:
        from cmd_handlers import CommandContext, dispatch as _modular_dispatch
        _ctx = CommandContext(
            command=command, parts=parts, cmd=cmd, args=args,
            config=config, session=session, memory=memory, theme=theme,
            soul=soul, scheduler=scheduler, tool_registry=tool_registry,
            router=router, context_inject=context_inject, planner=planner,
            skills=skills, curator=curator, cost_tracker=cost_tracker,
            semantic_mem=semantic_mem, knowledge=knowledge,
        )
        if _modular_dispatch(_ctx):
            return
    except Exception:
        # Never let the modular layer break command handling — fall through.
        pass

    # ── Exit ──────────────────────────────────────────────────────────────────
    if cmd in ("/exit", "/quit", "/q"):
        if _gateway and _gateway.status()["running"]:
            _gateway.stop()
        print(theme.info("Session ended. Goodbye."))
        sys.exit(0)

    # ── Help ──────────────────────────────────────────────────────────────────
    elif cmd == "/help":
        print(theme.box([
            "  OPERON COMMANDS",
            "---",
            "  SESSION",
            "  /clear           Clear session history",
            "  /undo            Remove last exchange",
            "  /retry           Re-send last user message",
            "  /history [n]     Show last N turns (default 20)",
            "  /compress        Trim middle context to save tokens",
            "  /save <name>     Save session to named file",
            "  /load <name>     Load a named session",
            "  /sessions        List all saved sessions",
            "  /search <query>  Search across saved sessions",
            "  /snapshot [lbl]  Create rollback checkpoint",
            "  /rollback [lbl]  Restore to checkpoint",
            "  /title <name>    Name the current session",
            "  /export          Export session to JSON",
            "---",
            "  MEMORY & KNOWLEDGE",
            "  /memory          Show conversation memories (session-level)",
            "  /remember <text> Save a memory manually",
            "  /forget          Clear all memories",
            "  /knowledge       Show/edit permanent knowledge (cross-session facts)",
            "  /skills          List loaded SKILL.md packs",
            "  /skills install <name> <content>  Add a new skill",
            "  /skills remove <name>             Remove a skill",
            "  /skills reload                    Re-scan skill dirs",
            "  /curator on|off|status            Autonomous skill gen",
            "  /curator clear                    Delete auto skills",
            "---",
            "  RAG DOCUMENT PIPELINE",
            "  /rag index <path>        Index a file or directory",
            "  /rag query <text>        Search indexed documents",
            "  /rag sources             List all indexed sources",
            "  /rag remove <path>       Remove a source from index",
            "  /rag stats               Show index statistics",
            "  /rag clear               Clear all indexed documents",
            "---",
            "  SECRETS MANAGER",
            "  /secrets list            List stored secret keys",
            "  /secrets set <k> <v>     Store an encrypted secret",
            "  /secrets delete <key>    Delete a secret",
            "  /secrets status          Show backend info",
            "  /secrets migrate         Import keys from knowledge base",
            "---",
            "  TOOLSETS",
            "  /toolsets list           List tool groups",
            "  /toolsets enable <g>     Enable a tool group",
            "  /toolsets disable <g>    Disable a tool group",
            "---",
            "  HEARTBEAT SCHEDULER",
            "  /heartbeat start         Start background tick loop",
            "  /heartbeat stop          Stop background tick loop",
            "  /heartbeat trigger       Fire one tick immediately",
            "  /heartbeat status        Show scheduler status",
            "  /heartbeat edit          Reset HEARTBEAT.md template",
            "---",
            "  GOALS  (persistent across sessions)",
            "  /goal set <title> [-- desc]   Create a goal",
            "  /goal list [status]           List goals",
            "  /goal update <id> [-- note]   Add progress note",
            "  /goal complete <id>           Mark complete",
            "  /goal delete <id>             Delete a goal",
            "  /goal clear                   Delete all goals",
            "---",
            "  PIPELINE MACROS",
            "  /macro list              List saved macros",
            "  /macro run <name>        Run a macro",
            "  /macro delete <name>     Delete a macro",
            "---",
            "  KANBAN, TASKS & DEVOPS",
            "  /kanban [board|add|list|show|start|done|help]  Full SQLite kanban",
            "  /tasks  [list|get|cleanup]                     Async task tracker",
            "  /checkpoint [create|restore|list|status|diff]  Git snapshots",
            "  /pool   [status|add|rotate|load|providers]     API credential pool",
            "---",
            "  MEMORY & AGENTS",
            "  /memory [status|recall|know|facts|entities|consolidate]  Long-term memory",
            "  /mesh   [status|register|route|dlq]                      Multi-agent bus",
            "  /swe    [fix|dry|test|status]                            SWE agent loop",
            "  /voice  [speak|listen|transcribe|status|backends]        Voice & multimodal",
            "  /plugin [search|install|info|list|popular|refresh|publish] Plugin registry",
            "  /conv   [status|compress|summary|reset]                  Conversation compression",
            "---",
            "  RETRY POLICIES",
            "  /retry list              Show per-tool retry settings",
            "  /retry set <tool> [max=N delay=S backoff=F]",
            "  /retry reset <tool>      Reset to defaults",
            "  /retry on|off <tool>     Enable/disable retries",
            "---",
            "  MODEL & CONFIG",
            "  /model <name>    Switch active model",
            "  /models          List all model profiles",
            "  /local           Detect local model servers",
            "  /approve <on|off|status>  Tool approval mode",
            "  /soul            View or edit personality",
            "  /config          Show current configuration",
            "  /setup           Re-run the setup wizard",
            "---",
            "  GATEWAY & DASHBOARD",
            "  /gateway start|stop|status   Telegram bot gateway",
            "  /dashboard start|stop|open   Web dashboard (:7270)",
            "  /webhook start|stop|status   REST API server (:7271)",
            "---",
            "  MCP SERVERS",
            "  /mcp list                    List connected MCP servers",
            "  /mcp connect <n> stdio <cmd> Connect stdio server",
            "  /mcp connect <n> http <url>  Connect HTTP server",
            "  /mcp disconnect <name>       Disconnect a server",
            "  /mcp tools                   List all MCP tools",
            "---",
            "  PLUGIN SYSTEM",
            "  /plugins list                List installed plugins",
            "  /plugins install <path>      Install plugin from directory",
            "  /plugins uninstall <name>    Remove a plugin",
            "  /plugins new <name>          Create plugin scaffold",
            "  /plugins reload              Reload all plugins",
            "---",
            "  MCP SERVER (expose Operon to external clients)",
            "  /serve stdio                 Start MCP server on stdio",
            "  /serve http [port]           Start HTTP MCP server",
            "  /serve config                Print client config JSON",
            "---",
            "  MULTI-AGENT MESH",
            "  /mesh parallel <task>    Run task with all specialist roles in parallel",
            "  /mesh pipeline <task>    Run task through roles sequentially",
            "  /mesh auto <task>        Planner decomposes + auto-executes task",
            "  /mesh roles              List available agent roles",
            "---",
            "  REFLECTION ENGINE",
            "  /reflect                 Show reflection engine status",
            "  /reflect on|off          Enable/disable response self-critique",
            "  /reflect reset           Reset correction counter for this session",
            "---",
            "  DIAGNOSTICS",
            "  /doctor          Full system health check",
            "  /usage           Token & cost stats",
            "  /status          System + session status",
            "  /tools           List all available tools",
            "  /schedule        List scheduled background tasks",
            "---",
            "  /help  /exit",
        ]))

    # ── Session control ───────────────────────────────────────────────────────

    # /clear and /undo are handled by cmd_handlers/session_cmds.py

    elif cmd == "/retry":
        if router is None or tool_registry is None or planner is None:
            print(theme.warning("/retry is not available in this context."))
            return
        while session._messages and session._messages[-1]["role"] == "assistant":
            session._messages.pop()
        if not session._messages:
            print(theme.warning("Nothing to retry."))
            return
        print(theme.info("Retrying last turn…"))
        run_agent_loop(session=session, router=router, planner=planner,
                       tool_registry=tool_registry, memory=memory, config=config,
                       theme=theme, soul=soul, context_inject=context_inject,
                       skills=skills, curator=curator or _curator, knowledge=knowledge,
                       cost_tracker=cost_tracker, semantic_mem=semantic_mem)

    # /history and /compress are handled by cmd_handlers/session_cmds.py

    elif cmd == "/title":
        title = " ".join(parts[1:]).strip()
        if not title:
            print(theme.info(f"Session title: {session.get_title() or '(none)'}"))
        else:
            session.set_title(title)
            print(theme.success(f"Title set: {title}"))

    elif cmd == "/snapshot":
        lbl = session.snapshot(parts[1] if len(parts) > 1 else "")
        print(theme.success(f"Checkpoint '{lbl}' saved. ({len(session.list_snapshots())} total)"))
        # Also create a git checkpoint if in a repo
        try:
            from core.checkpoint_manager import quick_checkpoint
            ref = quick_checkpoint(f"snapshot: {lbl}", repo_path=os.getcwd())
            if ref and not ref.message.endswith("[no-changes]"):
                print(theme.dim(f"  git checkpoint: {ref.short_sha()}"))
        except Exception:
            pass

    elif cmd == "/rollback":
        lbl = parts[1] if len(parts) > 1 else ""
        if session.rollback(lbl):
            print(theme.success(f"Rolled back to '{lbl or 'last snapshot'}'. ({len(session)} messages)"))
        else:
            snaps = session.list_snapshots()
            if snaps:
                print(theme.warning(f"Not found. Available: {', '.join(s['label'] for s in snaps)}"))
            else:
                print(theme.warning("No snapshots. Use /snapshot first."))

    elif cmd == "/save":
        name = " ".join(parts[1:]).strip()
        if not name:
            print(theme.warning("Usage: /save <name>"))
        else:
            print(theme.success(f"Saved: {session.save_named(name)}"))

    elif cmd == "/load":
        name = " ".join(parts[1:]).strip()
        if not name:
            print(theme.warning("Usage: /load <name>"))
        elif session.load_named(name):
            print(theme.success(f"Loaded '{name}' — {session.turn_count} turns, {len(session)} messages."))
        else:
            print(theme.warning(f"Session '{name}' not found. Use /sessions to list."))

    elif cmd == "/sessions":
        saved = SessionManager.list_saved_sessions()
        if not saved:
            print(theme.info("No saved sessions. Use /save <name>."))
        else:
            lines = ["  SAVED SESSIONS", "---"]
            for s in saved[:20]:
                row = f"  {s['name']:<22} {s['turns']} turns   {s['saved_at'][:16]}"
                if s["title"] and s["title"] != s["name"]:
                    row += f"  [{s['title']}]"
                lines.append(row)
            print(theme.box(lines))

    elif cmd == "/search":
        query = " ".join(parts[1:]).strip()
        if not query:
            print(theme.warning("Usage: /search <query>"))
        else:
            hits = SessionManager.search_sessions(query)
            if not hits:
                print(theme.info(f"No sessions matched '{query}'."))
            else:
                lines = [f"  SEARCH: {query}", "---"]
                for h in hits:
                    lines.append(f"  [{h['name']}] {h['role']}: {h['snippet']}")
                print(theme.box(lines))

    elif cmd == "/export":
        print(theme.success(f"Exported: {session.export()}"))

    # /tools, /usage, /cost are handled by cmd_handlers/info.py

    elif cmd == "/memory":
        # Semantic long-term memory commands
        if semantic_mem is None:
            print(theme.warning("Semantic memory not initialised."))
            return
        sub = parts[1].lower() if len(parts) > 1 else "stats"
        if sub == "stats":
            st = semantic_mem.stats()
            print(theme.box([
                "  SEMANTIC MEMORY", "---",
                f"  Total memories     {st['total_memories']:>8,}",
                f"  Sessions stored    {st['total_sessions']:>8,}",
                f"  With embeddings    {st['with_embeddings']:>8,}",
            ]))
        elif sub == "recall" and len(parts) > 2:
            query = " ".join(parts[2:])
            results = semantic_mem.recall(query, top_k=5)
            if not results:
                print(theme.info("  No relevant memories found."))
            else:
                lines = [f"  RECALL: {query!r}", "---"]
                for m in results:
                    import datetime as _dt
                    ts = _dt.datetime.fromtimestamp(m["timestamp"]).strftime("%Y-%m-%d")
                    lines.append(f"  [{ts}] ({m['similarity']:.2f}) {m['role']}: {m['content'][:80]}")
                print(theme.box(lines))
        elif sub == "forget":
            n = semantic_mem.forget_all()
            print(theme.success(f"  Cleared {n} memories."))
        else:
            print(theme.info("  Usage: /memory stats | /memory recall <query> | /memory forget"))

    # ── Skills ────────────────────────────────────────────────────────────────

    elif cmd == "/skills":
        if skills is None:
            print(theme.warning("Skill system not initialised."))
            return
        sub = parts[1].lower() if len(parts) > 1 else "list"

        if sub == "list" or sub not in ("install", "remove", "reload"):
            skill_list = skills.list_skills()
            if not skill_list:
                print(theme.box([
                    "  LOADED SKILLS", "---",
                    "  No skills loaded.",
                    "  Drop .md files into ~/.operon/skills/ to add skills.",
                    "  Or: /skills install <name> <content>",
                ]))
            else:
                lines = ["  LOADED SKILLS", "---"]
                for s in skill_list:
                    desc = f"  {s['name']:<24} {s['size']} chars"
                    if s["description"]:
                        desc += f"  — {s['description'][:36]}"
                    lines.append(desc)
                lines += ["---", f"  Skill dir: ~/.operon/skills/"]
                print(theme.box(lines))

        elif sub == "reload":
            n = skills.reload()
            print(theme.success(f"Skills reloaded — {n} skill{'s' if n != 1 else ''} loaded."))

        elif sub == "install":
            if len(parts) < 4:
                print(theme.warning("Usage: /skills install <name> <content>"))
            else:
                name    = parts[2]
                content = " ".join(parts[3:])
                path    = skills.install(name, content)
                print(theme.success(f"Skill '{name}' installed: {path}"))

        elif sub == "remove":
            if len(parts) < 3:
                print(theme.warning("Usage: /skills remove <name>"))
            else:
                name = " ".join(parts[2:])
                if skills.remove(name):
                    print(theme.success(f"Skill '{name}' removed."))
                else:
                    print(theme.warning(f"Skill '{name}' not found."))

    # ── Curator ───────────────────────────────────────────────────────────────

    elif cmd == "/curator":
        c = curator or _curator
        if c is None:
            print(theme.warning("Curator not initialised."))
            return
        sub = parts[1].lower() if len(parts) > 1 else "status"
        if sub == "on":
            c.enabled = True
            print(theme.success("Curator ON — auto-skill generation enabled."))
        elif sub == "off":
            c.enabled = False
            print(theme.success("Curator OFF — auto-skill generation disabled."))
        elif sub == "clear":
            n = c.clear_auto_skills()
            print(theme.success(f"Deleted {n} auto-generated skill(s)."))
        elif sub == "grades":
            grades = c.get_grades()
            if not grades:
                print(theme.info("No skill grades recorded yet."))
            else:
                lines = [f"  SKILL GRADES  ({len(grades)} tracked)", "---",
                         f"  {'Skill':<34} {'OK':>4} {'FAIL':>5} {'Rate':>6}"]
                for name, g in sorted(grades.items()):
                    rate_str = f"{g['success_rate']:.0%}" if g["success_rate"] is not None else "N/A"
                    lines.append(
                        f"  {name:<34} {g['success']:>4} {g['failure']:>5}  {rate_str:>6}"
                    )
                lines += ["---",
                          f"  Rewrite threshold: < {int(0.4 * 100)}%  "
                          f"(min {3} outcomes)"]
                print(theme.box(lines))

        else:  # status / list
            auto = c.list_auto_skills()
            lines = [
                "  SKILL CURATOR", "---",
                f"  Status:       {'ON' if c.enabled else 'OFF'}",
                f"  Auto-skills:  {len(auto)} / 30",
                f"  Min tool calls to trigger: {c._min_tools}",
            ]
            if auto:
                lines.append("---")
                for s in auto[:15]:
                    lines.append(f"  {s['name']:<34} {s['size']} chars")
            lines += ["---",
                      "  /curator on|off    — toggle",
                      "  /curator clear     — delete auto-generated skills",
                      "  /curator grades    — show skill success/failure rates"]
            print(theme.box(lines))

    # ── Dashboard ─────────────────────────────────────────────────────────────

    elif cmd == "/dashboard":
        sub = parts[1].lower() if len(parts) > 1 else "status"
        if sub == "start":
            if _dashboard and _dashboard.running:
                print(theme.info(f"Dashboard already running at {_dashboard.url}"))
                return
            if _dashboard is None:
                print(theme.warning("Dashboard not initialised (internal error)."))
                return
            url = _dashboard.start(
                get_session   = lambda: session._messages,
                get_memory    = lambda: memory.get_all(),
                get_status    = lambda: _build_dashboard_status(config, session, memory, tool_registry, skills),
                delete_memory = lambda mid: memory.delete_by_id(mid),
                clear_memory  = lambda: memory.clear(),
                open_browser  = (len(parts) > 2 and parts[2] == "open"),
            )
            print(theme.success(f"Dashboard started → {url}"))
        elif sub == "stop":
            if _dashboard and _dashboard.running:
                _dashboard.stop()
                print(theme.success("Dashboard stopped."))
            else:
                print(theme.info("Dashboard is not running."))
        elif sub == "open":
            if _dashboard and _dashboard.running:
                import webbrowser
                webbrowser.open(_dashboard.url)
                print(theme.info(f"Opening {_dashboard.url}"))
            else:
                print(theme.warning("Dashboard is not running. Use /dashboard start first."))
        else:  # status
            if _dashboard:
                st = _dashboard.status()
                print(theme.box([
                    "  WEB DASHBOARD", "---",
                    f"  Running: {st['running']}",
                    f"  URL:     {st['url'] or '(stopped)'}",
                    "---",
                    "  /dashboard start       — start",
                    "  /dashboard start open  — start and open in browser",
                    "  /dashboard stop        — stop",
                    "  /dashboard open        — open in browser",
                ]))
            else:
                print(theme.warning("Dashboard not initialised."))

    # ── MCP servers ───────────────────────────────────────────────────────────

    elif cmd == "/mcp":
        m = _mcp
        if m is None:
            print(theme.warning("MCP manager not initialised."))
            return
        sub = parts[1].lower() if len(parts) > 1 else "list"

        if sub == "list":
            servers = m.status()
            if not servers:
                print(theme.box([
                    "  MCP SERVERS", "---",
                    "  No MCP servers connected.",
                    "  Connect with: /mcp connect <name> stdio <command>",
                    "              : /mcp connect <name> http <url>",
                ]))
            else:
                lines = ["  MCP SERVERS", "---"]
                for s in servers:
                    lines.append(f"  ● {s['name']:<18} {s['tools']} tools")
                    for t in s["tool_names"][:5]:
                        lines.append(f"      ↳ {t}")
                    if len(s["tool_names"]) > 5:
                        lines.append(f"      ↳ … +{len(s['tool_names'])-5} more")
                print(theme.box(lines))

        elif sub == "tools":
            all_tools = m.list_all_tools()
            if not all_tools:
                print(theme.info("No MCP tools registered. Connect a server first."))
            else:
                lines = ["  MCP TOOLS", "---"]
                for t in all_tools:
                    lines.append(f"  [{t['server']}] {t['name']}")
                    if t.get("description"):
                        lines.append(f"      {t['description'][:60]}")
                print(theme.box(lines))

        elif sub == "connect":
            # /mcp connect <name> stdio <cmd...>
            # /mcp connect <name> http <url>
            if len(parts) < 5:
                print(theme.warning(
                    "Usage:\n"
                    "  /mcp connect <name> stdio <command> [args...]\n"
                    "  /mcp connect <name> http  <url>"
                ))
                return
            srv_name  = parts[2]
            transport = parts[3].lower()
            rest      = parts[4:]
            if transport == "stdio":
                ok = m.connect(srv_name, "stdio", command=rest)
            elif transport == "http":
                ok = m.connect(srv_name, "http", url=rest[0])
            else:
                print(theme.warning(f"Unknown transport '{transport}'. Use 'stdio' or 'http'."))
                return
            if ok:
                tools_added = m.inject_into_registry(
                    tool_registry.tools,
                    _TOOL_DEFINITIONS,   # adds to agent's system prompt
                )
                srv_info = next((s for s in m.status() if s["name"] == srv_name), {})
                print(theme.success(
                    f"MCP server '{srv_name}' connected — "
                    f"{srv_info.get('tools', 0)} tools available."
                ))
            else:
                print(theme.error(f"Failed to connect to MCP server '{srv_name}'."))

        elif sub == "disconnect":
            if len(parts) < 3:
                print(theme.warning("Usage: /mcp disconnect <name>"))
            else:
                name = parts[2]
                if m.disconnect(name):
                    print(theme.success(f"MCP server '{name}' disconnected."))
                else:
                    print(theme.warning(f"MCP server '{name}' not found."))

        else:
            print(theme.warning(f"Unknown /mcp sub-command: {sub}. Use: list, tools, connect, disconnect"))

    # ── Approval mode ─────────────────────────────────────────────────────────

    elif cmd == "/approve":
        sub = parts[1].lower() if len(parts) > 1 else "status"
        if sub == "on":
            _APPROVAL_MODE = True
            print(theme.success("Tool approval mode ON — you will be asked before each tool call."))
        elif sub == "off":
            _APPROVAL_MODE = False
            print(theme.success("Tool approval mode OFF — tools execute automatically."))
        else:
            print(theme.box([
                "  TOOL APPROVAL MODE",
                "---",
                f"  Status: {'ON' if _APPROVAL_MODE else 'OFF'}",
                f"  Always-safe: {', '.join(sorted(_SAFE_TOOLS))}",
                "---",
                "  /approve on   — enable approval prompts",
                "  /approve off  — disable (auto-execute all tools)",
            ]))

    # ── Memory ────────────────────────────────────────────────────────────────

    elif cmd == "/memory":
        mems = memory.get_all()
        if mems:
            lines = ["  LONG-TERM MEMORIES", "---"]
            for i, m in enumerate(mems[-25:], 1):
                tag = m.get("type", "?")[:4].upper()
                lines.append(f"  {i:02d}. [{tag}] {m.get('content', '')[:68]}")
            print(theme.box(lines))
        else:
            print(theme.info("No memories stored yet."))

    elif cmd == "/remember":
        text = " ".join(parts[1:]).strip()
        if not text:
            print(theme.warning("Usage: /remember <text>"))
        else:
            memory.add_manual(content=text, mem_type="manual", importance=4)
            print(theme.success(f"Remembered: {text[:60]}"))

    elif cmd == "/forget":
        memory.clear()
        print(theme.success("Long-term memory cleared."))

    # ── Permanent knowledge ───────────────────────────────────────────────────

    elif cmd == "/knowledge":
        if knowledge is None:
            print(theme.warning("  Knowledge base not initialised."))
            return
        sub = parts[1].lower() if len(parts) > 1 else "show"

        if sub in ("show", "list", "ls"):
            data = knowledge.get_all()
            if not data:
                print(theme.info("No permanent knowledge stored yet. "
                                 "The agent saves facts automatically, or use "
                                 "/knowledge set <key> <value>."))
            else:
                lines = [f"  PERMANENT KNOWLEDGE  ({len(data)} facts)", "---"]
                for k, v in data.items():
                    lines.append(f"  {k:<30} {v}")
                print(theme.box(lines))

        elif sub == "set" and len(parts) >= 4:
            key   = parts[2]
            value = " ".join(parts[3:])
            knowledge.set(key, value)
            print(theme.success(f"Saved permanently: {key} = {value}"))

        elif sub in ("delete", "del", "remove") and len(parts) >= 3:
            if knowledge.delete(parts[2]):
                print(theme.success(f"Deleted: {parts[2]}"))
            else:
                print(theme.warning(f"Key not found: {parts[2]}"))

        elif sub == "clear":
            knowledge.clear()
            print(theme.success("All permanent knowledge cleared."))

        elif sub == "path":
            from core.knowledge import KNOWLEDGE_PATH
            print(theme.info(f"Knowledge file: {KNOWLEDGE_PATH}"))

        else:
            print(theme.box([
                "  PERMANENT KNOWLEDGE COMMANDS",
                "---",
                "  /knowledge                      Show all stored facts",
                "  /knowledge set <key> <value>    Store a fact manually",
                "  /knowledge delete <key>         Delete one fact",
                "  /knowledge clear                Wipe everything",
                "  /knowledge path                 Show the JSON file location",
                "---",
                "  Facts are also saved automatically by the agent when it",
                "  learns your name, preferences, paths, or other key info.",
            ]))

    # ── Soul ──────────────────────────────────────────────────────────────────

    elif cmd == "/soul":
        if soul is None:
            print(theme.warning("Soul system not initialised."))
        elif len(parts) > 1 and parts[1] == "edit":
            import subprocess as _sp
            path   = soul.get_path()
            editor = os.environ.get("EDITOR", "nano")
            _sp.run([editor, path])
            print(theme.success(f"Soul updated: {path}"))
        else:
            print(theme.box(["  OPERON SOUL", "---"] +
                            ["  " + l for l in soul.read().splitlines()[:30]]))

    # ── Model / config ────────────────────────────────────────────────────────

    elif cmd == "/model":
        if len(parts) < 2:
            print(theme.info(f"Current model: {config.get('default_model', '?')}"))
        else:
            config.set("default_model", parts[1])
            info = config.resolve_model(parts[1])
            config.set("active_provider", info["provider"])
            print(theme.success(f"Switched to: {parts[1]}  [{info['provider']}]"))

    elif cmd == "/models":
        profiles = config.get("model_profiles", {})
        current  = config.get("default_model", "")
        lines    = ["  AVAILABLE MODEL PROFILES", "---"]
        for name, profile in sorted(profiles.items()):
            marker = ">> " if name == current else "   "
            lines.append(f"  {marker}{name:<28} [{profile.get('provider', '?')}]")
        print(theme.box(lines))

    elif cmd == "/local":
        _handle_local(parts, config, theme)

    elif cmd == "/config":
        cfg   = config.get_safe_display()
        lines = ["  CURRENT CONFIGURATION", "---"]
        for k, v in cfg.items():
            lines.append(f"  {k:<22} {str(v)[:44]}")
        print(theme.box(lines))

    elif cmd == "/setup":
        from setup_wizard import run_wizard
        run_wizard(config)

    # ── Telegram gateway ──────────────────────────────────────────────────────

    elif cmd == "/gateway":
        sub = parts[1].lower() if len(parts) > 1 else "status"

        if sub == "start":
            token = (config.get("telegram_token", "") or
                     os.environ.get("TELEGRAM_BOT_TOKEN", ""))
            if not token:
                print(theme.warning(
                    "No Telegram bot token.\n"
                    "  Set it via /setup → Telegram, or:\n"
                    "  export TELEGRAM_BOT_TOKEN=<your-token>"
                ))
                return
            if _gateway and _gateway.status()["running"]:
                print(theme.info("Gateway is already running."))
                return

            # Build a text-returning agent runner for the gateway.
            # Each Telegram message gets its own fresh SessionManager so
            # conversations are independent.  The final assistant message
            # is returned as the bot reply.
            def _gw_agent_runner(prompt: str) -> str:
                sub_session = SessionManager()
                sub_session.add_message("user", prompt)
                run_agent_loop(
                    session=sub_session, router=router, planner=planner,
                    tool_registry=tool_registry, memory=memory, config=config,
                    theme=theme, soul=soul, context_inject=context_inject,
                    skills=skills, curator=_curator, knowledge=knowledge,
                )
                for m in reversed(sub_session._messages):
                    if m["role"] == "assistant":
                        return m["content"]
                return "(no response)"

            allowed_str = config.get("telegram_allowed_users", "")
            allowed     = ([int(x.strip()) for x in allowed_str.split(",") if x.strip().isdigit()]
                           if allowed_str else None)

            _gateway = TelegramGateway(
                token=token,
                agent_runner=_gw_agent_runner,
                config=config,
                allowed_users=allowed,
            )
            _gateway.start()
            print(theme.success(
                f"Telegram gateway started (token …{token[-6:]})\n"
                f"  Send a message to your bot to begin.\n"
                f"  Allowed users: {'all' if not allowed else allowed}\n"
                f"  Stop with: /gateway stop"
            ))

        elif sub == "stop":
            if _gateway and _gateway.status()["running"]:
                _gateway.stop()
                print(theme.success("Telegram gateway stopped."))
            else:
                print(theme.info("Gateway is not running."))

        else:  # status
            if _gateway:
                st = _gateway.status()
                print(theme.box([
                    "  TELEGRAM GATEWAY", "---",
                    f"  Running:       {st['running']}",
                    f"  Messages recv: {st['messages_recv']}",
                    f"  Errors:        {st['errors']}",
                    f"  Uptime:        {st['uptime_sec']}s",
                    "---",
                    "  /gateway start   — start the bot",
                    "  /gateway stop    — stop the bot",
                ]))
            else:
                print(theme.box([
                    "  TELEGRAM GATEWAY", "---",
                    "  Status: not started",
                    "---",
                    "  /gateway start   — start (requires TELEGRAM_BOT_TOKEN)",
                    "  Get a token at: https://t.me/BotFather",
                ]))

    # ── Webhook REST server ───────────────────────────────────────────────────

    elif cmd == "/webhook":
        global _webhook
        sub = parts[1].lower() if len(parts) > 1 else "status"

        if sub == "start":
            if _webhook and _webhook.running:
                print(theme.info(f"Webhook server already running at {_webhook.url}"))
                return
            port = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 7271
            host = parts[3] if len(parts) > 3 else "127.0.0.1"

            def _wh_runner(prompt: str) -> str:
                sub_session = SessionManager()
                sub_session.add_message("user", prompt)
                run_agent_loop(
                    session=sub_session, router=router, planner=planner,
                    tool_registry=tool_registry, memory=memory, config=config,
                    theme=theme, soul=soul, context_inject=context_inject,
                    skills=skills, curator=_curator, knowledge=knowledge,
                )
                for m in reversed(sub_session._messages):
                    if m["role"] == "assistant":
                        return m["content"]
                return "(no response)"

            _webhook = WebhookServer(
                agent_runner  = _wh_runner,
                host          = host,
                port          = port,
                session_clear = lambda: session.clear(),
                tool_list     = lambda: list(tool_registry.tools.keys()),
                session_info  = lambda: {
                    "turns": session.turn_count,
                    "messages": len(session),
                    "model": config.get("default_model", "?"),
                },
            )
            try:
                url = _webhook.start()
                print(theme.success(
                    f"Webhook server started → {url}\n"
                    f"  POST /chat    — run a prompt\n"
                    f"  POST /batch   — run multiple prompts\n"
                    f"  GET  /status  — health check\n"
                    f"  GET  /tools   — list tools\n"
                    f"  Auth: set OPERON_WEBHOOK_TOKEN for Bearer token auth"
                ))
            except RuntimeError as e:
                print(theme.error(str(e)))

        elif sub == "stop":
            if _webhook and _webhook.running:
                _webhook.stop()
                print(theme.success("Webhook server stopped."))
            else:
                print(theme.info("Webhook server is not running."))

        else:  # status
            if _webhook:
                st = _webhook.status()
                print(theme.box([
                    "  WEBHOOK REST SERVER", "---",
                    f"  Running:  {st['running']}",
                    f"  URL:      {st['url'] or '(stopped)'}",
                    f"  Auth:     {'enabled (Bearer token)' if st['auth'] else 'disabled'}",
                    "---",
                ] + st["endpoints"] + [
                    "---",
                    "  /webhook start [port] [host]  — start (default :7271 127.0.0.1)",
                    "  /webhook stop                 — stop",
                ]))
            else:
                print(theme.box([
                    "  WEBHOOK REST SERVER", "---",
                    "  Status: not started",
                    "---",
                    "  /webhook start   — start on http://127.0.0.1:7271",
                    "  /webhook start 8080 0.0.0.0  — custom port/host",
                ]))

    # ── RAG document pipeline ─────────────────────────────────────────────────

    elif cmd == "/rag":
        global _rag
        if _rag is None:
            _rag = RAGPipeline()
        sub = parts[1].lower() if len(parts) > 1 else "stats"

        if sub in ("index", "add"):
            if len(parts) < 3:
                print(theme.warning("Usage: /rag index <path>"))
                return
            target = " ".join(parts[2:]).strip()
            p = Path(target).expanduser()
            print(theme.dim(f"  Indexing {target}…"))
            if p.is_dir():
                result = _rag.index_directory(str(p))
                print(theme.success(
                    f"Indexed {result['files_indexed']} files, "
                    f"{result['chunks_added']} chunks added."
                ))
                if result["errors"]:
                    for e in result["errors"][:5]:
                        print(theme.warning(f"  Error: {e}"))
            elif p.exists():
                result = _rag.index_file(str(p))
                if result.get("error"):
                    print(theme.error(f"Failed: {result['error']}"))
                else:
                    print(theme.success(
                        f"Indexed: {Path(result['source']).name}  "
                        f"({result['chunks_added']} chunks)"
                    ))
            else:
                print(theme.warning(f"Path not found: {target}"))

        elif sub == "query":
            if len(parts) < 3:
                print(theme.warning("Usage: /rag query <search text>"))
                return
            q = " ".join(parts[2:]).strip()
            results = _rag.query(q, top_k=5)
            if not results:
                print(theme.info(f"No relevant documents found for: {q!r}"))
            else:
                lines = [f"  RAG RESULTS for: {q!r}", "---"]
                for r in results:
                    lines.append(
                        f"  [{Path(r['source']).name}  chunk {r['chunk_index']}  "
                        f"score {r['score']}]"
                    )
                    lines.append(f"  {r['content'][:120].replace(chr(10), ' ')}…")
                    lines.append("")
                print(theme.box(lines))

        elif sub in ("sources", "list"):
            sources = _rag.list_sources()
            if not sources:
                print(theme.info("No documents indexed. Use /rag index <path>."))
            else:
                import datetime as _dt
                lines = [f"  INDEXED SOURCES  ({len(sources)})", "---"]
                for s in sources:
                    ts = _dt.datetime.fromtimestamp(s["last_indexed"]).strftime("%Y-%m-%d")
                    lines.append(
                        f"  {Path(s['source']).name:<32}  "
                        f"{s['chunks']:>4} chunks  {ts}"
                    )
                    lines.append(f"    {s['source']}")
                print(theme.box(lines))

        elif sub in ("remove", "delete"):
            if len(parts) < 3:
                print(theme.warning("Usage: /rag remove <source-path>"))
                return
            src = " ".join(parts[2:]).strip()
            n = _rag.remove_source(src)
            if n:
                print(theme.success(f"Removed {n} chunks for: {src}"))
            else:
                print(theme.warning(f"Source not found: {src}"))

        elif sub == "clear":
            _rag.clear()
            print(theme.success("RAG index cleared — all documents removed."))

        else:  # stats
            st = _rag.stats()
            print(theme.box([
                "  RAG DOCUMENT INDEX", "---",
                f"  Total chunks     {st['total_chunks']:>8,}",
                f"  Total sources    {st['total_sources']:>8,}",
                f"  With embeddings  {st['with_embeddings']:>8,}",
                "---",
                "  /rag index <path>    — index a file or directory",
                "  /rag query <text>    — semantic search",
                "  /rag sources         — list indexed files",
                "  /rag remove <path>   — remove a source",
                "  /rag clear           — clear all",
            ]))

    # ── Secrets manager ───────────────────────────────────────────────────────

    elif cmd == "/secrets":
        global _secrets
        if _secrets is None:
            _secrets = get_secrets()
        sub = parts[1].lower() if len(parts) > 1 else "list"

        if sub in ("list", "ls"):
            keys = _secrets.list_keys()
            if not keys:
                print(theme.info("No secrets stored yet. Use /secrets set <key> <value>."))
            else:
                lines = [f"  SECRETS  ({len(keys)} stored)  backend: {_secrets.status()['backend']}", "---"]
                for k in sorted(keys):
                    lines.append(f"  {k}")
                lines += ["---", "  Values are encrypted — use /secrets set to update"]
                print(theme.box(lines))

        elif sub == "set":
            if len(parts) < 4:
                print(theme.warning("Usage: /secrets set <key> <value>"))
                return
            key   = parts[2]
            value = " ".join(parts[3:])
            _secrets.set(key, value)
            print(theme.success(f"Secret stored: {key} → [encrypted]"))

        elif sub in ("delete", "del", "remove"):
            if len(parts) < 3:
                print(theme.warning("Usage: /secrets delete <key>"))
                return
            key = parts[2]
            if _secrets.delete(key):
                print(theme.success(f"Secret deleted: {key}"))
            else:
                print(theme.warning(f"Key not found: {key}"))

        elif sub == "migrate":
            try:
                migrated = _secrets.migrate_from_knowledge_json()
                if migrated:
                    print(theme.success(
                        f"Migrated {len(migrated)} keys from knowledge base: "
                        f"{', '.join(migrated)}"
                    ))
                else:
                    print(theme.info("No sensitive keys found in knowledge base to migrate."))
            except Exception as e:
                print(theme.error(f"Migration failed: {e}"))

        else:  # status
            st = _secrets.status()
            print(theme.box([
                "  SECRETS MANAGER", "---",
                f"  Backend:       {st['backend']}",
                f"  Keys stored:   {st['key_count']}",
                f"  Storage path:  {st.get('path', 'N/A')}",
                "---",
                "  /secrets list              — list stored keys",
                "  /secrets set <k> <v>       — store encrypted secret",
                "  /secrets delete <key>      — remove a secret",
                "  /secrets migrate           — import from knowledge base",
            ]))

    # ── Toolsets ──────────────────────────────────────────────────────────────

    elif cmd == "/toolsets":
        sub = parts[1].lower() if len(parts) > 1 else "list"

        if sub == "list":
            from tools.registry import _DISABLED_TOOLS
            lines = [f"  TOOL GROUPS  ({len(TOOLSETS)} groups)", "---"]
            for name, tools in sorted(TOOLSETS.items()):
                disabled = [t for t in tools if t in _DISABLED_TOOLS]
                status   = "OFF" if len(disabled) == len(tools) else (
                           f"PARTIAL ({len(disabled)} off)" if disabled else "ON"
                )
                lines.append(f"  {name:<14} {status:<20} {len(tools)} tools")
            lines += [
                "---",
                "  /toolsets enable <group>   — re-enable all tools in group",
                "  /toolsets disable <group>  — disable all tools in group",
            ]
            print(theme.box(lines))

        elif sub == "enable":
            if len(parts) < 3:
                print(theme.warning(f"Usage: /toolsets enable <group>  "
                                    f"[groups: {', '.join(sorted(TOOLSETS))}]"))
                return
            group = parts[2].lower()
            if group not in TOOLSETS:
                print(theme.warning(f"Unknown group '{group}'. Groups: {', '.join(sorted(TOOLSETS))}"))
                return
            restored = enable_toolset(group)
            if restored:
                print(theme.success(f"Enabled {len(restored)} tools in '{group}': {', '.join(restored)}"))
            else:
                print(theme.info(f"Toolset '{group}' is already fully enabled."))

        elif sub == "disable":
            if len(parts) < 3:
                print(theme.warning(f"Usage: /toolsets disable <group>  "
                                    f"[groups: {', '.join(sorted(TOOLSETS))}]"))
                return
            group = parts[2].lower()
            if group not in TOOLSETS:
                print(theme.warning(f"Unknown group '{group}'. Groups: {', '.join(sorted(TOOLSETS))}"))
                return
            disabled = disable_toolset(group)
            if disabled:
                print(theme.success(f"Disabled {len(disabled)} tools in '{group}': {', '.join(disabled)}"))
            else:
                print(theme.info(f"Toolset '{group}' has no active tools to disable."))

        else:
            print(theme.warning(f"Unknown /toolsets sub-command: {sub}. Use: list, enable, disable"))

    # ── Heartbeat scheduler ───────────────────────────────────────────────────

    elif cmd == "/heartbeat":
        sub = parts[1].lower() if len(parts) > 1 else "status"

        if sub == "start":
            if _heartbeat and _heartbeat.status()["running"]:
                print(theme.info("Heartbeat already running."))
            else:
                if _heartbeat is None:
                    def _hb_runner(prompt):
                        # Minimal runner: inject into session turn
                        session.add_turn("user", prompt)
                        return "[heartbeat tick dispatched]"
                    _heartbeat = HeartbeatScheduler(agent_runner=_hb_runner)
                _heartbeat.start()
                st = _heartbeat.status()
                print(theme.success(
                    f"Heartbeat started — ticking every {st['interval_seconds']}s "
                    f"{'(business hours only)' if st['business_hours'] else ''}"
                ))

        elif sub == "stop":
            if _heartbeat and _heartbeat.status()["running"]:
                _heartbeat.stop()
                print(theme.success("Heartbeat stopped."))
            else:
                print(theme.info("Heartbeat is not running."))

        elif sub == "trigger":
            if _heartbeat is None:
                print(theme.warning("Heartbeat not initialised. Run /heartbeat start first."))
            else:
                result = _heartbeat.trigger_now()
                print(theme.info(f"Heartbeat triggered: {result}"))

        elif sub == "status":
            if _heartbeat is None:
                print(theme.info("Heartbeat: not initialised  (/heartbeat start to begin)"))
            else:
                st = _heartbeat.status()
                lines = ["  HEARTBEAT STATUS", "---"]
                lines.append(f"  Running:        {'yes' if st['running'] else 'no'}")
                lines.append(f"  Interval:       {st['interval_seconds']}s")
                lines.append(f"  Business hours: {'yes' if st['business_hours'] else 'no'}")
                if st.get("last_tick"):
                    lines.append(f"  Last tick:      {st['last_tick']}")
                if st.get("next_tick"):
                    lines.append(f"  Next tick:      {st['next_tick']}")
                lines.append(f"  File:           {st.get('heartbeat_file', 'not found')}")
                print(theme.box(lines))

        elif sub == "edit":
            hb = _heartbeat or HeartbeatScheduler(agent_runner=lambda p: p)
            path = hb.set_heartbeat_content(
                "# HEARTBEAT\n\nCheck the project status, look for new issues, "
                "and summarise any important developments since the last tick.\n"
            )
            print(theme.success(f"HEARTBEAT.md created/reset at: {path}"))
            print(theme.info("Edit it to customise what runs every tick."))

        else:
            print(theme.warning("Usage: /heartbeat start|stop|trigger|status|edit"))

    # ── Goal tracker ──────────────────────────────────────────────────────────

    elif cmd == "/goal":
        if _goals is None:
            _goals = GoalTracker()
        sub = parts[1].lower() if len(parts) > 1 else "list"

        if sub == "set":
            rest = " ".join(parts[2:])
            if not rest:
                print(theme.warning("Usage: /goal set <title> [-- description]"))
                return
            # Support title -- description split
            if " -- " in rest:
                title, desc = rest.split(" -- ", 1)
            else:
                title, desc = rest, ""
            result = _goals.set(title.strip(), description=desc.strip())
            if result.get("success"):
                print(theme.success(f"Goal created: [#{result['goal_id']}] {result['title']}"))
            else:
                print(theme.error(result.get("error", "Failed to create goal.")))

        elif sub == "list":
            status_filter = parts[2].lower() if len(parts) > 2 else ""
            goals = _goals.list_goals(status=status_filter)
            if not goals:
                print(theme.info("No goals." if not status_filter else f"No {status_filter} goals."))
            else:
                lines = ["  GOALS", "---"]
                for g in goals:
                    deadline = f"  by {g['deadline']}" if g.get("deadline") else ""
                    gid = g.get('goal_id') or g.get('id', '?')
                    lines.append(f"  [#{gid}] [{g['priority'].upper()}] {g['title']}{deadline}")
                    if g.get("description"):
                        lines.append(f"    {g['description'][:80]}")
                    if g.get("progress_notes"):
                        last = g["progress_notes"][-1]
                        lines.append(f"    → {last['note'][:70]}")
                print(theme.box(lines))

        elif sub == "update":
            if len(parts) < 3:
                print(theme.warning("Usage: /goal update <goal_id> [-- note]"))
                return
            gid  = parts[2]
            note = " ".join(parts[3:]).lstrip("- ").strip()
            result = _goals.update(gid, progress_note=note)
            if result.get("success"):
                print(theme.success(f"Goal updated: {result.get('title', gid)}"))
            else:
                print(theme.error(result.get("error", "Failed to update.")))

        elif sub == "complete":
            if len(parts) < 3:
                print(theme.warning("Usage: /goal complete <goal_id>"))
                return
            result = _goals.complete(parts[2])
            if result.get("success"):
                print(theme.success(f"Goal completed: {result.get('title', parts[2])}"))
            else:
                print(theme.error(result.get("error", "Not found.")))

        elif sub == "delete":
            if len(parts) < 3:
                print(theme.warning("Usage: /goal delete <goal_id>"))
                return
            result = _goals.delete(parts[2])
            if result.get("success"):
                print(theme.success(f"Goal deleted: {result.get('title', parts[2])}"))
            else:
                print(theme.error(result.get("error", "Not found.")))

        elif sub == "clear":
            n = _goals.clear()
            print(theme.success(f"Cleared {n} goal(s)."))

        else:
            print(theme.warning("Usage: /goal set|list|update|complete|delete|clear"))

    # ── Pipeline macros ───────────────────────────────────────────────────────

    elif cmd == "/macro":
        if _macros is None:
            _macros = MacroManager(tool_registry=tool_registry or ToolRegistry())
        sub = parts[1].lower() if len(parts) > 1 else "list"

        if sub == "list":
            macros = _macros.list_macros()
            if not macros:
                print(theme.info("No macros saved. Use /macro define to create one, or use the macro_save tool."))
            else:
                lines = ["  PIPELINE MACROS", "---"]
                for m in macros:
                    lines.append(f"  {m['name']:<20}  {len(m.get('steps', []))} steps  {m.get('description', '')[:50]}")
                print(theme.box(lines))

        elif sub == "run":
            if len(parts) < 3:
                print(theme.warning("Usage: /macro run <name> [key=value ...]"))
                return
            name = parts[2]
            vars_dict = {}
            for token in parts[3:]:
                if "=" in token:
                    k, v = token.split("=", 1)
                    vars_dict[k] = v
            result = _macros.run(name, vars=vars_dict if vars_dict else None)
            if result.get("success"):
                print(theme.success(f"Macro '{name}' completed in {len(result.get('steps', []))} steps."))
                if result.get("output"):
                    print(theme.info(str(result["output"])[:400]))
            else:
                print(theme.error(result.get("error", f"Macro '{name}' failed.")))

        elif sub in ("define", "create"):
            print(theme.info(
                "To create a macro, use the macro_save tool in a prompt:\n"
                '  Save a macro named "daily_report" with steps: ...'
            ))

        elif sub == "delete":
            if len(parts) < 3:
                print(theme.warning("Usage: /macro delete <name>"))
                return
            result = _macros.delete(parts[2])
            if result.get("success"):
                print(theme.success(f"Macro '{parts[2]}' deleted."))
            else:
                print(theme.error(result.get("error", "Not found.")))

        else:
            print(theme.warning("Usage: /macro list|run <name>|delete <name>"))

    # ── Retry policies ────────────────────────────────────────────────────────

    elif cmd == "/retry":
        # Check if this is the old "retry last message" handler
        if len(parts) == 1:
            # Legacy: re-send last user message (keep existing behaviour)
            if session is not None:
                turns = [t for t in (session.get_history() if hasattr(session, "get_history") else []) if t.get("role") == "user"]
                if turns:
                    last_user = turns[-1].get("content", "")
                    if last_user:
                        print(theme.info(f"Retrying: {last_user[:60]}..."))
                        return
            print(theme.warning("/retry is not available in this context."))
            return

        if _retry_mgr is None:
            _retry_mgr = RetryPolicyManager()

        sub = parts[1].lower()

        if sub == "list":
            policies = _retry_mgr.list_policies()
            lines = ["  RETRY POLICIES", "---"]
            for p in policies:
                enabled  = "ON " if p["enabled"] else "OFF"
                lines.append(
                    f"  [{enabled}] {p['tool_name']:<28}  "
                    f"max={p['max_attempts']}  "
                    f"delay={p['base_delay_s']}s  "
                    f"backoff={p['backoff_factor']}x"
                )
            if not policies:
                lines.append("  (all tools using defaults)")
            print(theme.box(lines))

        elif sub == "set":
            # /retry set <tool> max=3 delay=1.0 backoff=2.0 enabled=true
            if len(parts) < 3:
                print(theme.warning("Usage: /retry set <tool_name> [max=N] [delay=S] [backoff=F] [enabled=true|false]"))
                return
            tool_name = parts[2]
            kwargs = {}
            for token in parts[3:]:
                if "=" in token:
                    k, v = token.split("=", 1)
                    if k == "max":
                        kwargs["max_attempts"] = int(v)
                    elif k == "delay":
                        kwargs["base_delay_s"] = float(v)
                    elif k == "backoff":
                        kwargs["backoff_factor"] = float(v)
                    elif k == "enabled":
                        kwargs["enabled"] = v.lower() in ("true", "1", "yes")
            _retry_mgr.set(tool_name, **kwargs)
            print(theme.success(f"Retry policy updated for '{tool_name}'."))

        elif sub == "reset":
            if len(parts) < 3:
                print(theme.warning("Usage: /retry reset <tool_name>"))
                return
            _retry_mgr.reset(parts[2])
            print(theme.success(f"Retry policy reset to defaults for '{parts[2]}'."))

        elif sub == "off":
            if len(parts) < 3:
                print(theme.warning("Usage: /retry off <tool_name>"))
                return
            _retry_mgr.set(parts[2], enabled=False)
            print(theme.success(f"Retries disabled for '{parts[2]}'."))

        elif sub == "on":
            if len(parts) < 3:
                print(theme.warning("Usage: /retry on <tool_name>"))
                return
            _retry_mgr.set(parts[2], enabled=True)
            print(theme.success(f"Retries enabled for '{parts[2]}'."))

        else:
            print(theme.warning("Usage: /retry list | set <tool> [max=N delay=S backoff=F] | reset <tool> | on/off <tool>"))

    # ── Diagnostics ───────────────────────────────────────────────────────────

    elif cmd == "/doctor":
        _run_doctor(config, memory, tool_registry or ToolRegistry(),
                    skills or SkillLoader(), theme,
                    mcp=_mcp, dashboard=_dashboard, curator_obj=_curator)

    elif cmd == "/schedule":
        if scheduler is None:
            print(theme.warning("Scheduler not initialised."))
        else:
            tasks = scheduler.list_tasks()
            if not tasks:
                print(theme.info("No scheduled tasks."))
            else:
                lines = ["  SCHEDULED TASKS", "---"]
                for t in tasks:
                    status = "ON " if t["enabled"] else "OFF"
                    lines.append(
                        f"  [{status}] {t['task_id']}  every {t['interval_seconds']}s  "
                        f"runs:{t['run_count']}  {t['label'][:36]}"
                    )
                print(theme.box(lines))

    elif cmd == "/status":
        mcp_info = (f"{len(_mcp.status())} servers" if _mcp and _mcp.status() else "none connected") if _mcp else "not loaded"
        rag_info  = "running" if _rag else "not started  (/rag stats to init)"
        wh_info   = (_webhook.url if (_webhook and _webhook.running) else "stopped  (/webhook start)")
        lines = [
            "  SYSTEM STATUS", "---",
            f"  Model           {config.get('default_model', 'N/A')}",
            f"  Provider        {config.get('active_provider', 'N/A')}",
            f"  Session turns   {session.turn_count}",
            f"  Messages        {len(session)}",
            f"  Memory items    {len(memory.get_all())}",
            f"  Skills loaded   {len(skills) if skills else 0}",
            f"  Approval mode   {'ON' if _APPROVAL_MODE else 'OFF'}",
            f"  Gateway         {'running' if (_gateway and _gateway.status()['running']) else 'stopped'}",
            f"  Dashboard       {_dashboard.url if (_dashboard and _dashboard.running) else 'stopped'}",
            f"  Webhook         {wh_info}",
            f"  RAG index       {rag_info}",
            f"  MCP servers     {mcp_info}",
            f"  Curator         {'ON' if (_curator and _curator.enabled) else 'OFF'}",
            f"  Session title   {session.get_title() or '(none)'}",
        ]
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.3)
            vm  = psutil.virtual_memory()
            lines += [
                f"  CPU             {cpu:.1f}%",
                f"  RAM             {vm.percent:.1f}%  ({vm.used//1024//1024}MB/{vm.total//1024//1024}MB)",
            ]
        except ImportError:
            pass
        print(theme.box(lines))

    # ── Kanban board (v2 — SQLite-backed KanbanDB) ────────────────────────────

    elif cmd == "/kanban":
        try:
            from core.kanban import KanbanDB, handle_kanban_command
        except ImportError:
            from core.kanban import KanbanBoard as _KB
            board = _KB()
            print(theme.warning("  [Kanban] Using legacy board — upgrade core/kanban.py"))
            return

        db   = KanbanDB()
        sub  = parts[1].lower() if len(parts) > 1 else "board"
        rest = parts[2:]

        if sub in ("board", "b"):
            sprint = rest[0] if rest else None
            output = db.agent_board(sprint=sprint)
            print(theme.box(["  KANBAN BOARD", "---", *output["board"].splitlines()]))

        elif sub in ("add", "create", "new"):
            if not rest:
                print(theme.warning("Usage: /kanban add <title> [-- description]"))
            else:
                joined = " ".join(rest)
                if " -- " in joined:
                    title, desc = joined.split(" -- ", 1)
                else:
                    title, desc = joined, ""
                result = db.agent_create(title=title.strip(), description=desc.strip())
                if result.get("success"):
                    t = result["task"]
                    print(theme.success(
                        f"  ✓ Task created: [{t['id'][:8]}] {t['title']}"
                        f"  [{t['priority']} · {t['status']}]"
                    ))
                else:
                    print(theme.error(f"  ✗ {result.get('error', 'create failed')}"))

        elif sub in ("list", "ls"):
            status_f = rest[0] if rest else None
            result = db.agent_list(status=status_f, limit=30)
            tasks = result.get("tasks", [])
            if not tasks:
                _hint = f" with status '{status_f}'" if status_f else ""
                print(theme.info(f"  No tasks{_hint}. Use /kanban add <title>."))
            else:
                _ICON = {"todo": "○", "in_progress": "●", "done": "✓",
                         "blocked": "✗", "review": "◉", "cancelled": "–"}
                _PICO = {"high": "🔴", "critical": "🚨", "medium": "🟠",
                         "low": "🟢", "none": "⚪"}
                lines = [f"  TASKS  ({result.get('total', result.get('count', len(tasks)))} total, showing {len(tasks)})", "---"]
                for t in tasks:
                    icon  = _ICON.get(t["status"], "?")
                    picon = _PICO.get(t["priority"], "")
                    asgn  = f" @{t['assignee']}" if t.get("assignee") else ""
                    lines.append(f"  {icon} {picon} [{t['id'][:8]}] {t['title'][:50]}{asgn}")
                lines += ["---",
                          "  /kanban add <title>           — create",
                          "  /kanban show <id>             — details",
                          "  /kanban start <id>            — mark in progress",
                          "  /kanban done <id>             — complete",
                          "  /kanban update <id> <field>=<val>  — update field"]
                print(theme.box(lines))

        elif sub in ("show", "get", "view"):
            if not rest:
                print(theme.warning("Usage: /kanban show <task_id>"))
            else:
                task = db.get(rest[0])
                if not task:
                    print(theme.warning(f"  Task '{rest[0]}' not found"))
                else:
                    lines = [f"  TASK: {task.title}", "---",
                             f"  ID:          {task.id}",
                             f"  Status:      {task.status.value}",
                             f"  Priority:    {task.priority.value}",
                             f"  Assignee:    {task.assignee or '—'}",
                             f"  Sprint:      {task.sprint or '—'}",
                             f"  Due:         {task.due_date or '—'}",
                             f"  Labels:      {', '.join(task.labels) or '—'}",
                             f"  Age:         {task.age_days:.1f}d",
                             f"  Description: {(task.description or '—')[:100]}"]
                    if task.subtasks:
                        lines += ["---", f"  Subtasks ({len(task.subtasks)}):"]
                        for st in task.subtasks[:5]:
                            lines.append(f"    ○ [{st.id[:8]}] {st.title[:40]}")
                    print(theme.box(lines))

        elif sub in ("start", "begin", "in_progress"):
            if not rest:
                print(theme.warning("Usage: /kanban start <task_id>"))
            else:
                task = db.start(rest[0])
                if task:
                    print(theme.success(f"  ▶ [{task.id[:8]}] {task.title} → in_progress"))
                else:
                    print(theme.error(f"  Task '{rest[0]}' not found"))

        elif sub in ("done", "complete", "finish"):
            if not rest:
                print(theme.warning("Usage: /kanban done <task_id> [comment]"))
            else:
                comment = " ".join(rest[1:]) if len(rest) > 1 else ""
                task = db.complete(rest[0], comment=comment)
                if task:
                    print(theme.success(f"  ✓ [{task.id[:8]}] {task.title} → done"))
                else:
                    print(theme.error(f"  Task '{rest[0]}' not found"))

        elif sub == "block":
            if not rest:
                print(theme.warning("Usage: /kanban block <task_id> [reason]"))
            else:
                reason = " ".join(rest[1:]) if len(rest) > 1 else "blocked"
                task = db.block(rest[0], comment=reason)
                if task:
                    print(theme.warning(f"  ✗ [{task.id[:8]}] {task.title} → blocked"))
                else:
                    print(theme.error(f"  Task '{rest[0]}' not found"))

        elif sub in ("assign", "claim"):
            if len(rest) < 2:
                print(theme.warning("Usage: /kanban assign <task_id> <assignee>"))
            else:
                task = db.update(rest[0], assignee=rest[1])
                if task:
                    print(theme.success(f"  [{task.id[:8]}] {task.title} → @{task.assignee}"))
                else:
                    print(theme.error(f"  Task '{rest[0]}' not found"))

        elif sub == "update":
            # /kanban update <id> status=done priority=high assignee=bob
            if len(rest) < 2:
                print(theme.warning("Usage: /kanban update <task_id> field=value ..."))
            else:
                task_id = rest[0]
                kwargs = {}
                for kv in rest[1:]:
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        kwargs[k] = v
                if "status" in kwargs:
                    task = db.set_status(task_id, kwargs.pop("status"))
                else:
                    task = db.get(task_id)
                if task and kwargs:
                    task = db.update(task_id, **kwargs)
                if task:
                    print(theme.success(f"  ✓ Updated [{task.id[:8]}] {task.title}"))
                else:
                    print(theme.error(f"  Task '{rest[0]}' not found"))

        elif sub in ("delete", "rm", "remove"):
            if not rest:
                print(theme.warning("Usage: /kanban delete <task_id>"))
            else:
                ok = db.delete(rest[0])
                if ok:
                    print(theme.success(f"  ✓ Deleted task {rest[0][:8]}"))
                else:
                    print(theme.error(f"  Task '{rest[0]}' not found"))

        elif sub == "history":
            if not rest:
                print(theme.warning("Usage: /kanban history <task_id>"))
            else:
                history = db.get_history(rest[0])
                if not history:
                    print(theme.info(f"  No history for task {rest[0][:8]}"))
                else:
                    lines = [f"  HISTORY: {rest[0][:8]}", "---"]
                    for h in history[-10:]:
                        import datetime
                        ts = datetime.datetime.fromtimestamp(h.ts).strftime("%m/%d %H:%M")
                        lines.append(f"  {ts}  {h.field}: {(h.old_value or '—')[:20]} → {(h.new_value or '—')[:20]}  [{h.actor}]")
                    print(theme.box(lines))

        elif sub in ("sprint", "sprints"):
            sprint_sub = rest[0].lower() if rest else "list"
            if sprint_sub == "create" and len(rest) >= 2:
                db.create_sprint(rest[1], goal=" ".join(rest[2:]))
                print(theme.success(f"  ✓ Sprint '{rest[1]}' created"))
            elif sprint_sub == "summary" and len(rest) >= 2:
                s = db.sprint_summary(rest[1])
                lines = [f"  SPRINT: {rest[1]}", "---",
                         f"  Total:      {s.get('total', 0)}",
                         f"  Done:       {s.get('done', 0)}",
                         f"  Completion: {s.get('completion_pct', 0):.0f}%"]
                print(theme.box(lines))
            else:
                s = db.summary()
                sprints = s.get("by_sprint", {})
                if sprints:
                    lines = ["  SPRINTS", "---"]
                    for sp, cnt in list(sprints.items())[:10]:
                        lines.append(f"  {sp}: {cnt} tasks")
                    print(theme.box(lines))
                else:
                    print(theme.info("  No sprints. Use /kanban sprint create <name>"))

        elif sub in ("summary", "stats"):
            s = db.summary()
            by_st = s.get("by_status", {})
            lines = ["  KANBAN SUMMARY", "---",
                     f"  Total:       {s.get('total', 0)}",
                     f"  Todo:        {by_st.get('todo', 0)}",
                     f"  In Progress: {by_st.get('in_progress', 0)}",
                     f"  Review:      {by_st.get('review', 0)}",
                     f"  Done:        {by_st.get('done', 0)}",
                     f"  Blocked:     {by_st.get('blocked', 0)}",
                     f"  Cancelled:   {by_st.get('cancelled', 0)}",
                     "---",
                     f"  Overdue:     {s.get('overdue', 0)}",
                     f"  Sprints:     {len(s.get('by_sprint', {}))}"]
            print(theme.box(lines))

        elif sub in ("search", "find"):
            if not rest:
                print(theme.warning("Usage: /kanban search <query>"))
            else:
                query = " ".join(rest)
                result = db.agent_list(search=query, limit=20)
                tasks = result.get("tasks", [])
                if not tasks:
                    print(theme.info(f"  No tasks matching '{query}'"))
                else:
                    lines = [f"  SEARCH: '{query}'  ({len(tasks)} results)", "---"]
                    for t in tasks:
                        lines.append(f"  [{t['id'][:8]}] {t['title'][:60]}")
                    print(theme.box(lines))

        elif sub in ("sub", "subtask"):
            if len(rest) < 2:
                print(theme.warning("Usage: /kanban sub <parent_id> <title>"))
            else:
                task = db.create_subtask(rest[0], title=" ".join(rest[1:]))
                if task:
                    print(theme.success(f"  ✓ Subtask [{task.id[:8]}] created under {rest[0][:8]}"))
                else:
                    print(theme.error(f"  Parent task '{rest[0]}' not found"))

        elif sub in ("export",):
            import tempfile, os as _os
            path = _os.path.join(tempfile.gettempdir(), "operon_kanban_export.json")
            db.export_json(path)
            print(theme.success(f"  ✓ Board exported to {path}"))

        elif sub in ("help", "h", "?"):
            print(theme.box([
                "  /kanban — Full SQLite Kanban Board", "---",
                "  /kanban board [sprint]      — ASCII board view",
                "  /kanban add <title> [-- desc] — create task",
                "  /kanban list [status]       — list tasks",
                "  /kanban show <id>           — task details",
                "  /kanban start <id>          — mark in progress",
                "  /kanban done <id> [note]    — complete task",
                "  /kanban block <id> [reason] — mark blocked",
                "  /kanban assign <id> <user>  — assign task",
                "  /kanban update <id> field=v — update fields",
                "  /kanban history <id>        — audit trail",
                "  /kanban sub <parent> <title> — add subtask",
                "  /kanban sprint create <name> — create sprint",
                "  /kanban sprint summary <name> — sprint stats",
                "  /kanban summary             — board stats",
                "  /kanban search <query>      — find tasks",
                "  /kanban export              — export to JSON",
            ]))
        else:
            print(theme.warning(f"  Unknown kanban subcommand: {sub}  (try /kanban help)"))

    # ── Checkpoint manager ────────────────────────────────────────────────────

    elif cmd == "/checkpoint":
        try:
            from core.checkpoint_manager import CheckpointManager, get_manager
        except ImportError:
            print(theme.error("  core/checkpoint_manager.py not found"))
            return

        sub  = parts[1].lower() if len(parts) > 1 else "status"
        rest = parts[2:]
        _repo = os.getcwd()
        mgr  = get_manager(_repo)

        if sub in ("create", "snap", "save", "c"):
            msg = " ".join(rest) if rest else "manual checkpoint"
            ref = mgr.checkpoint(msg)
            if ref:
                print(theme.success(
                    f"  ✓ Checkpoint created: {ref.short_sha()}\n"
                    f"    Branch: {ref.branch}  Message: {ref.message[:60]}"
                ))
            else:
                print(theme.warning("  Could not create checkpoint (not a git repo?)"))

        elif sub in ("restore", "rollback", "rb"):
            sha = rest[0] if rest else None
            if not sha:
                # Restore last session checkpoint
                ok = mgr.restore_last()
                if ok:
                    print(theme.success("  ✓ Restored to last session checkpoint"))
                else:
                    print(theme.warning("  No session checkpoints to restore"))
            else:
                from core.checkpoint_manager import CheckpointRef
                ref = CheckpointRef(sha=sha, branch=mgr._current_branch(),
                                    message="manual restore")
                ok = mgr.restore(ref)
                if ok:
                    print(theme.success(f"  ✓ Restored to {sha[:12]}"))
                else:
                    print(theme.error(f"  Could not restore to {sha[:12]}"))

        elif sub in ("list", "ls", "log"):
            cps = mgr.list_checkpoints()
            if not cps:
                print(theme.info("  No Operon checkpoints found in this repo"))
            else:
                lines = [f"  CHECKPOINTS  ({len(cps)} total)", "---"]
                for cp in cps[:15]:
                    lines.append(f"  {cp['short_sha']}  {cp['message'][:55]}  {cp['date'][:10]}")
                print(theme.box(lines))

        elif sub in ("status", "info", "st"):
            s = mgr.stats()
            lines = ["  CHECKPOINT STATUS", "---",
                     f"  Repo:        {s['repo_path']}",
                     f"  Branch:      {s['current_branch']}",
                     f"  HEAD:        {s['head_sha'][:12] if s['head_sha'] else '—'}",
                     f"  Git repo:    {'✓' if s['is_git_repo'] else '✗'}",
                     f"  Session CPs: {s['session_checkpoints']}",
                     f"  Last CP:     {(str(s['last_checkpoint']) or '—')[:60]}"]
            print(theme.box(lines))

        elif sub in ("diff",):
            session_cps = mgr.session_history()
            if not session_cps:
                print(theme.info("  No session checkpoints to diff against"))
            else:
                d = mgr.diff(session_cps[0])
                if d:
                    lines = ["  DIFF FROM FIRST SESSION CHECKPOINT", "---",
                             f"  Since: {session_cps[0].short_sha()}",
                             f"  {d.summary()}",
                             "---"]
                    for f_ in d.changed_files[:10]:
                        lines.append(f"    {f_}")
                    print(theme.box(lines))
                else:
                    print(theme.info("  Could not compute diff"))

        elif sub in ("help", "h", "?"):
            print(theme.box([
                "  /checkpoint — Git Snapshot Manager", "---",
                "  /checkpoint create [msg]   — snapshot current state",
                "  /checkpoint restore [sha]  — rollback (last CP if no sha)",
                "  /checkpoint list           — show all operon checkpoints",
                "  /checkpoint status         — repo / branch / HEAD info",
                "  /checkpoint diff           — diff from first session CP",
            ]))
        else:
            print(theme.warning(f"  Unknown checkpoint subcommand: {sub}  (try /checkpoint help)"))

    # ── Credential pool ───────────────────────────────────────────────────────

    elif cmd == "/pool":
        try:
            from core.credential_pool import get_pool, CredentialPool, KeyStatus
        except ImportError:
            print(theme.error("  core/credential_pool.py not found"))
            return

        sub  = parts[1].lower() if len(parts) > 1 else "status"
        rest = parts[2:]
        pool = get_pool()

        if sub in ("status", "st", "info"):
            s = pool.status()
            if not s:
                print(theme.info("  No providers registered. Use /pool add <provider> <key>"))
            else:
                lines = ["  CREDENTIAL POOL", "---"]
                for provider, info in s.items():
                    avail = info["active"] + (1 if info.get("degraded", 0) > 0 else 0)
                    health = "✓" if info["active"] > 0 else ("⚠" if info["cooling"] > 0 else "✗")
                    lines.append(
                        f"  {health} {provider:<12}  "
                        f"total={info['total']}  active={info['active']}  "
                        f"cooling={info['cooling']}  banned={info['banned']}"
                    )
                    for k in info["keys"][:3]:
                        st_icon = {"active":"✓","cooling":"⏳","banned":"✗","degraded":"⚠"}.get(k["status"],"?")
                        lines.append(f"      {st_icon} {k['label']:<20}  uses={k['use_count']}  errors={k['error_count']}")
                print(theme.box(lines))

        elif sub in ("add",):
            if len(rest) < 2:
                print(theme.warning("Usage: /pool add <provider> <api_key> [label]"))
            else:
                provider, key = rest[0], rest[1]
                label = rest[2] if len(rest) > 2 else ""
                pool.add(provider, key, label=label)
                print(theme.success(f"  ✓ Added key for '{provider}'  (total: {pool.key_count(provider)})"))

        elif sub in ("rotate",):
            provider = rest[0] if rest else None
            if not provider:
                print(theme.warning("Usage: /pool rotate <provider>"))
            else:
                next_key = pool.rotate(provider)
                if next_key:
                    print(theme.success(f"  ✓ Rotated '{provider}' → new active key"))
                else:
                    print(theme.warning(f"  No available keys for '{provider}' after rotate"))

        elif sub in ("load",):
            if len(rest) < 2:
                print(theme.warning("Usage: /pool load <provider> <ENV_VAR_PREFIX>"))
            else:
                count = pool.load_from_env(rest[0], rest[1])
                print(theme.success(f"  ✓ Loaded {count} key(s) for '{rest[0]}'"))

        elif sub in ("providers", "list"):
            providers = pool.providers()
            if not providers:
                print(theme.info("  No providers registered"))
            else:
                lines = ["  REGISTERED PROVIDERS", "---"]
                for p in providers:
                    lines.append(f"  {p:<15}  {pool.key_count(p)} key(s)  available: {pool.available_count(p)}")
                print(theme.box(lines))

        elif sub in ("help", "h", "?"):
            print(theme.box([
                "  /pool — Credential Pool", "---",
                "  /pool status             — per-provider health",
                "  /pool add <p> <key>      — add a key",
                "  /pool rotate <provider>  — rotate to next key",
                "  /pool load <p> ENV_PREFIX — load from env",
                "  /pool providers          — list providers",
            ]))
        else:
            print(theme.warning(f"  Unknown pool subcommand: {sub}  (try /pool help)"))

    # ── Memory store ──────────────────────────────────────────────────────────

    elif cmd == "/memory":
        from core.memory_store import get_memory_store
        sub   = parts[1].lower() if len(parts) > 1 else "status"
        rest  = parts[2:]
        store = get_memory_store(session_id=session.session_id if hasattr(session, "session_id") else "default")

        if sub in ("status", "stats", "info"):
            s = store.stats()
            ep = s.get("episodic", {})
            en = s.get("entities", {})
            wm = s.get("working", {})
            print(theme.box([
                "  MEMORY STORE STATUS", "---",
                f"  Session:          {s.get('session_id', '?')}",
                f"  Episodes stored:  {ep.get('total', 0)}",
                f"  Sessions tracked: {ep.get('sessions', 0)}",
                f"  With embeddings:  {ep.get('with_embeddings', 0)}",
                f"  Entities known:   {en.get('total_entities', 0)}",
                f"  Entity facts:     {en.get('total_facts', 0)}",
                f"  Working slots:    {wm.get('total', 0)} / {wm.get('capacity', 0)}",
            ]))

        elif sub in ("recall", "search", "find"):
            if not rest:
                print(theme.warning("Usage: /memory recall <query>"))
            else:
                query   = " ".join(rest)
                results = store.recall(query, limit=5)
                if not results:
                    print(theme.info("  No matching memories found"))
                else:
                    lines = [f"  Top {len(results)} memories for '{query}'", "---"]
                    for r in results:
                        age = round(r.age_days, 1)
                        lines.append(f"  [{r.role}] {r.content[:100]} (age: {age}d, imp: {r.importance:.2f})")
                    print(theme.box(lines))

        elif sub in ("know", "entity"):
            # /memory know <entity> <attribute> <value>
            if len(rest) < 3:
                print(theme.warning("Usage: /memory know <entity> <attribute> <value>"))
            else:
                entity, attr, val = rest[0], rest[1], " ".join(rest[2:])
                store.know(entity, attr, val)
                print(theme.success(f"  ✓ Stored: {entity}.{attr} = {val}"))

        elif sub in ("facts", "about"):
            if not rest:
                print(theme.warning("Usage: /memory facts <entity>"))
            else:
                entity = rest[0]
                facts  = store.entity_facts(entity)
                if not facts:
                    print(theme.info(f"  No facts known about '{entity}'"))
                else:
                    lines = [f"  Facts about {entity}", "---"]
                    for k, v in facts.items():
                        lines.append(f"  {k:<20} {str(v)[:60]}")
                    print(theme.box(lines))

        elif sub in ("entities", "list"):
            entities = store.list_entities()
            if not entities:
                print(theme.info("  No entities known"))
            else:
                print(theme.box(["  Known Entities", "---"] + [f"  {e}" for e in entities[:30]]))

        elif sub in ("consolidate", "prune"):
            result = store.consolidate(force=True)
            print(theme.success(
                f"  ✓ Consolidated: archived={result.get('archived',0)} "
                f"entities_extracted={result.get('entities_extracted',0)}"
            ))

        elif sub in ("help", "h", "?"):
            print(theme.box([
                "  /memory — Long-term Memory Store", "---",
                "  /memory status                       — store statistics",
                "  /memory recall <query>               — semantic search",
                "  /memory know <entity> <attr> <val>   — store entity fact",
                "  /memory facts <entity>               — show entity facts",
                "  /memory entities                     — list known entities",
                "  /memory consolidate                  — prune/archive stale memories",
            ]))
        else:
            print(theme.warning(f"  Unknown memory subcommand: {sub}  (try /memory help)"))

    # ── Delegation bus ─────────────────────────────────────────────────────────

    elif cmd == "/mesh":
        from core.delegation_bus import get_bus, AgentState, TaskState
        sub  = parts[1].lower() if len(parts) > 1 else "status"
        rest = parts[2:]
        bus  = get_bus()

        if sub in ("status", "stats", "info"):
            s = bus.stats()
            a = s["agents"]
            t = s["tasks"]
            d = s["dlq"]
            print(theme.box([
                "  DELEGATION MESH STATUS", "---",
                f"  Agents total:     {a['total']}",
                f"  Agents available: {a['available']}",
                f"  Tasks total:      {t['total']}",
                f"  Tasks running:    {t['by_state'].get('running', 0)}",
                f"  Tasks done:       {t['by_state'].get('done', 0)}",
                f"  Dead letter queue:{d['total']} tasks",
            ]))
            if a["agents"]:
                lines = ["  REGISTERED AGENTS", "---"]
                for ag in a["agents"]:
                    lines.append(
                        f"  {ag['id']:<20} {ag['state']:<8} "
                        f"load:{ag['load']}  tasks:{ag['tasks']}  "
                        f"caps: {', '.join(ag['caps'][:3])}"
                    )
                print(theme.box(lines))

        elif sub in ("register", "add"):
            if not rest:
                print(theme.warning("Usage: /mesh register <agent_id> [cap1 cap2 ...]"))
            else:
                agent_id = rest[0]
                caps     = rest[1:] if len(rest) > 1 else []
                bus.register_agent(agent_id, capabilities=caps)
                print(theme.success(f"  ✓ Registered agent '{agent_id}' with caps: {caps}"))

        elif sub in ("route", "who"):
            if not rest:
                print(theme.warning("Usage: /mesh route <task description>"))
            else:
                task  = " ".join(rest)
                agent = bus.route(task)
                if agent:
                    print(theme.success(f"  → Best agent for task: {agent}"))
                else:
                    print(theme.warning("  No available agents can handle this task"))

        elif sub in ("dlq", "dead"):
            entries = bus.dlq.list()
            if not entries:
                print(theme.info("  Dead letter queue is empty"))
            else:
                lines = [f"  DLQ — {len(entries)} dead tasks", "---"]
                for e in entries[-5:]:
                    lines.append(f"  {e.task_id}  {e.task[:50]}  [{e.error[:40]}]")
                print(theme.box(lines))

        elif sub in ("help", "h", "?"):
            print(theme.box([
                "  /mesh — Multi-Agent Delegation Bus", "---",
                "  /mesh status                     — agent registry & task stats",
                "  /mesh register <id> [cap1 ...]   — register an agent",
                "  /mesh route <task>               — find best agent for task",
                "  /mesh dlq                        — dead letter queue",
            ]))
        else:
            print(theme.warning(f"  Unknown mesh subcommand: {sub}  (try /mesh help)"))

    # ── Task registry ──────────────────────────────────────────────────────────

    elif cmd == "/tasks":
        from core.task_registry import TaskRegistry
        reg = TaskRegistry()
        sub = parts[1].lower() if len(parts) > 1 else "list"

        if sub == "list":
            status_f = parts[2] if len(parts) > 2 else None
            tasks = reg.list(status=status_f, limit=20)
            if not tasks:
                print(theme.info("No tasks found." if not status_f
                                 else f"No tasks with status '{status_f}'."))
            else:
                _ST_ICON = {"queued": "⋯", "running": "▶", "succeeded": "✓",
                            "failed": "✗", "timed_out": "⏱", "cancelled": "—", "lost": "?"}
                lines = [f"  TASK REGISTRY  ({len(tasks)} tasks)", "---"]
                for t in tasks:
                    icon = _ST_ICON.get(t["status"], "?")
                    label = t.get("label") or t.get("task", "")[:40]
                    lines.append(f"  {icon} [{t['task_id'][:8]}] {t['status']:<12} {label}")
                print(theme.box(lines))

        elif sub == "get":
            if len(parts) < 3:
                print(theme.warning("Usage: /tasks get <task_id>"))
            else:
                t = reg.get(parts[2])
                if not t:
                    print(theme.warning(f"Task '{parts[2]}' not found."))
                else:
                    import datetime as _dt
                    lines = ["  TASK DETAIL", "---",
                             f"  ID:       {t['task_id']}",
                             f"  Status:   {t['status']}",
                             f"  Label:    {t.get('label', '')}",
                             f"  Notify:   {t.get('notify_policy', '')}",
                             f"  Created:  {_dt.datetime.fromtimestamp(t['created_at']).strftime('%Y-%m-%d %H:%M')}"]
                    if t.get("error"):
                        lines.append(f"  Error:    {t['error'][:80]}")
                    if t.get("terminal_summary"):
                        lines.append(f"  Summary:  {t['terminal_summary'][:80]}")
                    print(theme.box(lines))

        elif sub == "cleanup":
            n = reg.cleanup_expired()
            print(theme.success(f"Cleaned up {n} expired task(s)."))

        else:
            print(theme.warning("Usage: /tasks list [status] | get <id> | cleanup"))

    # ── Plugin manager ────────────────────────────────────────────────────────

    elif cmd == "/plugins":
        pm = _plugins
        if pm is None:
            from core.plugin_sdk import get_manager as _get_pm
            pm = _get_pm()
        sub = parts[1].lower() if len(parts) > 1 else "list"

        if sub == "list":
            loaded = pm.list()
            if not loaded:
                print(theme.box([
                    "  INSTALLED PLUGINS", "---",
                    "  No plugins installed.",
                    "  Install: /plugins install <path>",
                    "  Create:  /plugins new <name>",
                    "  Dir:     ~/.operon/plugins/",
                ]))
            else:
                lines = [f"  PLUGINS  ({len(loaded)} loaded)", "---"]
                for p in loaded:
                    status = "✓" if p["loaded"] else "✗"
                    lines.append(
                        f"  {status} {p['name']:<22} v{p['version']:<8} "
                        f"{len(p['tools'])} tools  {len(p['skills'])} skills"
                    )
                    if p.get("description"):
                        lines.append(f"    {p['description'][:60]}")
                    if p.get("error"):
                        lines.append(f"    ✗ Error: {p['error'][:60]}")
                lines += ["---",
                          "  /plugins install <path>   — install from directory",
                          "  /plugins uninstall <name> — remove plugin",
                          "  /plugins new <name>       — create scaffold"]
                print(theme.box(lines))

        elif sub == "install":
            if len(parts) < 3:
                print(theme.warning("Usage: /plugins install <path-to-plugin-dir>"))
                return
            src = " ".join(parts[2:]).strip()
            print(theme.dim(f"  Installing plugin from {src}…"))
            ok, msg = pm.install(src)
            if ok:
                print(theme.success(f"Plugin installed: {msg}"))
                # Register newly installed tools immediately
                pm.register_tools(tool_registry)
            else:
                print(theme.error(f"Plugin install failed: {msg}"))

        elif sub in ("uninstall", "remove"):
            if len(parts) < 3:
                print(theme.warning("Usage: /plugins uninstall <name>"))
                return
            name = " ".join(parts[2:]).strip()
            ok, msg = pm.uninstall(name)
            if ok:
                print(theme.success(msg))
            else:
                print(theme.error(msg))

        elif sub in ("new", "scaffold", "create"):
            if len(parts) < 3:
                print(theme.warning("Usage: /plugins new <name>"))
                return
            name = parts[2].strip().lower().replace(" ", "-")
            from core.plugin_sdk import create_plugin_scaffold
            dest = create_plugin_scaffold(name)
            print(theme.success(
                f"Plugin scaffold created at: {dest}\n"
                f"  Edit {dest}/plugin.json and {dest}/tools.py, then:\n"
                f"  /plugins install {dest}"
            ))

        elif sub == "reload":
            n = pm.load_all()
            pm.register_tools(tool_registry)
            print(theme.success(f"Plugins reloaded — {n} loaded."))

        else:
            print(theme.box([
                "  PLUGIN COMMANDS", "---",
                "  /plugins list                  — list installed plugins",
                "  /plugins install <path>        — install from directory",
                "  /plugins uninstall <name>      — remove plugin",
                "  /plugins new <name>            — create plugin scaffold",
                "  /plugins reload                — reload all plugins",
                "---",
                "  Plugin dir: ~/.operon/plugins/",
            ]))

    # ── MCP server (serve mode) ───────────────────────────────────────────────

    elif cmd == "/serve":
        # Start Operon as an MCP server so Claude Code, Cursor, etc. can call its tools.
        # Bare `/serve` shows usage — it must NOT auto-start a blocking server
        # (that would hang the REPL, scripts, and CI with no feedback).
        sub  = parts[1].lower() if len(parts) > 1 else "help"
        port = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 3456

        if sub in ("help", "h", "?"):
            print(theme.box([
                "  /serve — expose Operon as an MCP server", "---",
                "  /serve stdio          Start MCP server on stdio (blocks)",
                "  /serve http [port]    Start HTTP MCP server (default 3456)",
                "  /serve config         Print client config JSON for Claude Code/Cursor",
                "---",
                "  Add to Claude Code:   /serve config   then paste into the client.",
            ]))
            return

        if sub == "config":
            from core.mcp_server import generate_client_config
            mode = "http" if (len(parts) > 2 and parts[2] == "http") else "stdio"
            print(theme.box([
                "  MCP CLIENT CONFIG (add to ~/.claude/claude_desktop_config.json):",
                "---",
                generate_client_config(mode=mode, port=port),
            ]))
            return

        print(theme.info(
            f"Starting Operon MCP server  [mode={sub}  port={port}]\n"
            f"Ctrl+C to stop.\n"
            f"Add to Claude Code via: /serve config"
        ))
        try:
            from core.mcp_server import start_server
            start_server(mode=sub, port=port)
        except KeyboardInterrupt:
            print(theme.info("\nMCP server stopped."))
        except Exception as e:
            print(theme.error(f"MCP server error: {e}"))

    # ── /mesh — multi-agent coordination ──────────────────────────────────────
    elif cmd == "/mesh":
        sub   = parts[1].lower() if len(parts) > 1 else "help"
        task  = " ".join(parts[2:]) if len(parts) > 2 else ""

        if sub == "help" or not task:
            print(theme.box([
                "  MESH — Multi-Agent Coordination", "---",
                "  /mesh parallel <task>    Run task with all specialist roles in parallel",
                "  /mesh pipeline <task>    Run task through roles sequentially (each sees prior output)",
                "  /mesh auto <task>        PLANNER decomposes task then auto-executes steps",
                "  /mesh roles              List available agent roles",
                "---",
                "  Roles: RESEARCHER · CODER · ANALYST · WRITER · REVIEWER · PLANNER",
            ]))
            return

        if sub == "roles":
            print(theme.box([
                "  MESH AGENT ROLES", "---",
                "  RESEARCHER  — web search, fact gathering, source verification",
                "  CODER       — code generation, debugging, refactoring",
                "  ANALYST     — data analysis, pattern detection, reporting",
                "  WRITER      — drafting, editing, documentation",
                "  REVIEWER    — quality checks, critique, testing",
                "  PLANNER     — task decomposition, sequencing, strategy",
                "  GENERALIST  — full tool access for mixed tasks",
            ]))
            return

        if not task:
            print(theme.warning("Usage: /mesh parallel|pipeline|auto <task description>"))
            return

        try:
            from core.multi_agent import create_mesh, AgentRole, AgentMesh
            _mesh = create_mesh(router=router, tool_registry=tool_registry)

            print(theme.info(f"  [Mesh] Mode={sub}  Task: {task[:80]}"))
            spinner = ThinkingSpinner()
            spinner.start()

            if sub == "parallel":
                _roles = [AgentRole.RESEARCHER, AgentRole.CODER,
                          AgentRole.ANALYST, AgentRole.WRITER]
                result = _mesh.run_parallel(task, roles=_roles)
            elif sub == "pipeline":
                _roles = [AgentRole.PLANNER, AgentRole.RESEARCHER,
                          AgentRole.CODER, AgentRole.REVIEWER]
                result = _mesh.run_pipeline(task, roles=_roles)
            else:  # auto
                result = _mesh.run_auto(task)

            spinner.stop()

            if result.success:
                print(theme.success(f"  [Mesh] {sub.title()} complete  ({len(result.results)} agents)"))
                theme.assistant_response(result.final_output, stream=True)
                session.add_message("assistant", result.final_output)
            else:
                spinner.stop()
                print(theme.error(f"  [Mesh] {sub} failed — check individual agent results"))
                for ar in result.results:
                    status = "✓" if ar.success else "✗"
                    print(theme.dim(f"    {status} {ar.role.value}: {(ar.error or ar.output[:60])[:60]}"))
        except ImportError:
            print(theme.error("  [Mesh] core/multi_agent.py not found. Make sure Phase 3 build is complete."))
        except Exception as _mesh_err:
            try:
                spinner.stop()
            except Exception:
                pass
            print(theme.error(f"  [Mesh] Error: {_mesh_err}"))

    # ── /reflect — toggle or inspect the reflection engine ────────────────────
    elif cmd == "/reflect":
        sub = parts[1].lower() if len(parts) > 1 else "status"

        if sub == "on":
            config.set("reflection_enabled", True)
            print(theme.success("  Reflection engine enabled — responses will be self-reviewed before display."))
        elif sub == "off":
            config.set("reflection_enabled", False)
            print(theme.info("  Reflection engine disabled."))
        elif sub == "reset":
            try:
                from core.reflection import get_engine
                get_engine().reset_session(session._session_id)
                print(theme.info("  Reflection correction counter reset for this session."))
            except Exception as _re_err:
                print(theme.error(f"  Could not reset: {_re_err}"))
        else:  # status
            _enabled = config.get("reflection_enabled", True)
            _max_c   = config.get("reflection_max_corrections", 2)
            _hall    = config.get("reflection_hallucination_check", True)
            try:
                from core.reflection import get_engine
                _cnt = get_engine()._correction_counts.get(session._session_id, 0)
                _corr_info = f"  Corrections this session: {_cnt}/{_max_c}"
            except Exception:
                _corr_info = "  (engine not yet initialised)"
            print(theme.box([
                "  REFLECTION ENGINE", "---",
                f"  Status:               {'enabled' if _enabled else 'disabled'}",
                f"  Max corrections:      {_max_c}",
                f"  Hallucination check:  {_hall}",
                _corr_info,
                "---",
                "  /reflect on       enable self-critique",
                "  /reflect off      disable self-critique",
                "  /reflect reset    clear correction counter",
            ]))

    # ── /swe — Software Engineering agent ────────────────────────────────────
    elif cmd == "/swe":
        sub   = parts[1].lower() if len(parts) > 1 else "help"
        rest  = " ".join(parts[2:]) if len(parts) > 2 else ""

        if sub in ("help", ""):
            print(theme.box([
                "  SWE AGENT — Software Engineering Loop", "---",
                "  /swe fix <issue title>           Fix a codebase issue end-to-end",
                "  /swe dry <issue title>           Preview: parse+localise+plan, no changes",
                "  /swe test                        Run the test suite",
                "  /swe status                      Show SWE agent config",
                "---",
                "  The SWE agent: parses the issue → locates relevant files →",
                "  generates a fix plan → produces patches → applies them →",
                "  runs tests → iterates until tests pass (max 5 retries).",
            ]))
        elif sub in ("fix", "solve"):
            if not rest:
                print(theme.warning("  Usage: /swe fix <issue title>"))
            else:
                try:
                    from core.swe_agent import SWEAgent, SWETask
                    import os as _os
                    _repo = _os.getcwd()
                    print(theme.info(f"  [SWE] Starting fix loop for: {rest!r}"))
                    print(theme.info(f"  [SWE] Repo: {_repo}"))
                    _swe_agent = SWEAgent(
                        repo_path=_repo, max_retries=3,
                        on_event=lambda e: print(theme.info(
                            f"    [{e.action.upper()}] {e.detail[:100]}")),
                    )
                    _swe_task = SWETask(title=rest)
                    _swe_result = _swe_agent.solve(_swe_task)
                    print(theme.box([
                        "  SWE RESULT", "---",
                        f"  State:        {_swe_result.state.value}",
                        f"  Duration:     {_swe_result.duration:.1f}s",
                        f"  Retries:      {_swe_result.retries}",
                        f"  Files patched:{len(_swe_result.patches)}",
                        f"  Steps logged: {len(_swe_result.trajectory)}",
                    ] + ([f"  Test result:  {_swe_result.last_test.summary()}"]
                         if _swe_result.last_test else [])
                      + ([f"  PR URL:       {_swe_result.pr_url}"]
                         if _swe_result.pr_url else [])
                      + ([f"  Error:        {_swe_result.error}"]
                         if _swe_result.error else [])))
                except Exception as _swe_err:
                    print(theme.error(f"  [SWE] Error: {_swe_err}"))

        elif sub == "dry":
            if not rest:
                print(theme.warning("  Usage: /swe dry <issue title>"))
            else:
                try:
                    from core.swe_agent import SWEAgent, SWETask
                    import os as _os
                    _swe_agent = SWEAgent(repo_path=_os.getcwd())
                    _preview = _swe_agent.dry_run(SWETask(title=rest))
                    print(theme.box([
                        "  SWE DRY RUN", "---",
                        f"  Files located: {len(_preview['files'])}",
                    ] + [f"    {f['path']} ({f['reason']})"
                         for f in _preview['files'][:6]]
                      + ["---", "  Fix Plan:", _preview['plan'][:400]]))
                except Exception as _swe_err:
                    print(theme.error(f"  [SWE] Error: {_swe_err}"))

        elif sub in ("test", "tests", "run"):
            try:
                from core.swe_agent import SWEAgent
                import os as _os
                _swe_agent = SWEAgent(repo_path=_os.getcwd())
                print(theme.info("  [SWE] Running test suite…"))
                _tr = _swe_agent.run_tests(timeout=60)
                _status = "✅" if _tr.ok else "❌"
                print(theme.box([
                    f"  {_status} Test run: {_tr.summary()}",
                    f"  Command: {_tr.cmd}",
                ]))
            except Exception as _swe_err:
                print(theme.error(f"  [SWE] Error: {_swe_err}"))

        elif sub == "status":
            try:
                from core.swe_agent import _MAX_FIX_RETRIES, _MAX_CONTEXT_FILES
                import os as _os
                print(theme.box([
                    "  SWE AGENT STATUS", "---",
                    f"  Repo path:       {_os.getcwd()}",
                    f"  Max retries:     {_MAX_FIX_RETRIES}",
                    f"  Max ctx files:   {_MAX_CONTEXT_FILES}",
                    "---",
                    "  Backends: IssueParser, CodeLocaliser (grep+AST),",
                    "  PatchGenerator (LLM unified-diff), PatchApplier (GNU patch),",
                    "  TestRunner (pytest/unittest auto-detect), PRCreator (GitHub API)",
                ]))
            except Exception as _swe_err:
                print(theme.error(f"  [SWE] Error: {_swe_err}"))

        else:
            print(theme.warning(f"  Unknown /swe subcommand: {sub!r}. Try /swe help"))

    # ── /voice — Voice & Multimodal Pipeline ─────────────────────────────────
    elif cmd == "/voice":
        sub  = parts[1].lower() if len(parts) > 1 else "help"
        rest = " ".join(parts[2:]) if len(parts) > 2 else ""

        if sub in ("help", ""):
            print(theme.box([
                "  VOICE PIPELINE — Speech & Multimodal", "---",
                "  /voice speak <text>          Synthesise and play text",
                "  /voice transcribe <path>     Transcribe an audio file",
                "  /voice status                Show pipeline config",
                "  /voice listen                Record mic + transcribe (1 utterance)",
                "  /voice backends              List available STT/TTS backends",
                "---",
                "  STT: Whisper (local/API), Vosk, stub fallback",
                "  TTS: pyttsx3 (local), OpenAI TTS, espeak-ng, Bark (GPU), stub",
                "  Multimodal: image+text routing to vision model",
                "  VAD: webrtcvad or energy-based silence detection",
            ]))
        elif sub == "speak":
            if not rest:
                print(theme.warning("  Usage: /voice speak <text to speak>"))
            else:
                try:
                    from core.voice_pipeline import get_voice_pipeline
                    _vp = get_voice_pipeline()
                    print(theme.info(f"  [Voice] Speaking: {rest[:60]}…"))
                    audio = _vp.speak(rest, play=True)
                    print(theme.info(f"  [Voice] Synthesised {len(audio)} bytes of audio"))
                except Exception as _ve:
                    print(theme.error(f"  [Voice] Error: {_ve}"))

        elif sub == "transcribe":
            if not rest:
                print(theme.warning("  Usage: /voice transcribe <path/to/audio.wav>"))
            else:
                try:
                    from core.voice_pipeline import get_voice_pipeline
                    import os as _os
                    if not _os.path.exists(rest):
                        print(theme.error(f"  File not found: {rest}"))
                    else:
                        print(theme.info(f"  [Voice] Transcribing {rest}…"))
                        _vp   = get_voice_pipeline()
                        _text = _vp.transcribe_file(rest)
                        print(theme.box([
                            "  TRANSCRIPTION", "---",
                            f"  File: {rest}",
                            "---",
                            _text or "(empty / no speech detected)",
                        ]))
                except Exception as _ve:
                    print(theme.error(f"  [Voice] Error: {_ve}"))

        elif sub == "listen":
            try:
                from core.voice_pipeline import get_voice_pipeline
                print(theme.info("  [Voice] Recording… speak now (silence to stop)"))
                _vp   = get_voice_pipeline()
                _text = _vp.listen()
                if _text:
                    print(theme.box(["  HEARD", "---", _text]))
                else:
                    print(theme.warning("  [Voice] No speech detected."))
            except Exception as _ve:
                print(theme.error(f"  [Voice] Error: {_ve}"))

        elif sub == "status":
            try:
                from core.voice_pipeline import get_voice_pipeline
                _stats = get_voice_pipeline().stats()
                print(theme.box([
                    "  VOICE PIPELINE STATUS", "---",
                    f"  STT backend:    {_stats['stt_backend']}",
                    f"  TTS backend:    {_stats['tts_backend']}",
                    f"  Sample rate:    {_stats['sample_rate']} Hz",
                    f"  VAD enabled:    {_stats['vad_enabled']}",
                    f"  Wake word:      {_stats['wake_word_enabled']}",
                    f"  Wake words:     {', '.join(_stats['wake_words'][:3])}",
                ]))
            except Exception as _ve:
                print(theme.error(f"  [Voice] Error: {_ve}"))

        elif sub == "backends":
            try:
                from core.voice_pipeline import STTBackend, TTSBackend
                _stt = [b.value for b in STTBackend]
                _tts = [b.value for b in TTSBackend]
                print(theme.box([
                    "  VOICE BACKENDS", "---",
                    f"  STT: {', '.join(_stt)}",
                    f"  TTS: {', '.join(_tts)}",
                    "---",
                    "  Set OPENAI_API_KEY for whisper_api / openai TTS.",
                    "  pip install openai-whisper  for whisper_local.",
                    "  pip install pyttsx3         for local TTS.",
                    "  apt install espeak-ng        for espeak TTS.",
                ]))
            except Exception as _ve:
                print(theme.error(f"  [Voice] Error: {_ve}"))

        else:
            print(theme.warning(f"  Unknown /voice subcommand: {sub!r}. Try /voice help"))

    elif cmd == "/plugin":
        # ── Plugin registry marketplace ──────────────────────────────────────
        sub  = parts[1].lower() if len(parts) > 1 else "help"
        rest = " ".join(parts[2:]) if len(parts) > 2 else ""

        try:
            from core.plugin_registry import PluginRegistry
            reg = PluginRegistry()
        except Exception as _pe:
            print(theme.error(f"  [Plugin] Import error: {_pe}"))
            reg = None  # type: ignore

        if sub in ("help", "") or reg is None:
            print(theme.box([
                "  PLUGIN REGISTRY — Marketplace", "---",
                "  /plugin search [query]        Search the plugin registry",
                "  /plugin install <name>        Install a plugin by name",
                "  /plugin info <name>           Show plugin details",
                "  /plugin list                  List installed plugins",
                "  /plugin popular               Show most-downloaded plugins",
                "  /plugin refresh               Force-refresh the registry index",
                "  /plugin publish <dir>         Package + publish a local plugin",
                "---",
                "  Set OPERON_PLUGIN_REGISTRY_URL to point to a custom registry.",
            ]))

        elif sub == "search":
            if reg is None:
                pass
            else:
                query = rest.strip()
                results = reg.search(query=query)
                if not results:
                    print(theme.info(f"  No plugins found for '{query or '*'}'"))
                else:
                    print(theme.info(f"  Found {len(results)} plugin(s):"))
                    for p in results:
                        tick = "✓" if p.verified else " "
                        print(f"  [{tick}] {p.name:<32} v{p.version:<8} {p.description[:50]}")

        elif sub == "install":
            if reg is None or not rest:
                print(theme.warning("  Usage: /plugin install <name>"))
            else:
                ok, msg = reg.install(rest.strip(), dry_run=("--dry" in args))
                if ok:
                    print(theme.success(f"  {msg}"))
                else:
                    print(theme.error(f"  {msg}"))

        elif sub == "info":
            if reg is None or not rest:
                print(theme.warning("  Usage: /plugin info <name>"))
            else:
                entry = reg.info(rest.strip())
                if entry is None:
                    print(theme.warning(f"  Plugin '{rest}' not found in registry."))
                else:
                    print(theme.box([
                        f"  {entry.name}  v{entry.version}",
                        f"  Author:  {entry.author}",
                        f"  Tags:    {', '.join(entry.tags) or '—'}",
                        f"  Verified: {'✓' if entry.verified else '✗'}",
                        f"  Source:  {entry.source_url or '—'}",
                        f"  Install: {entry.install_cmd or f'pip install {entry.name}'}",
                        f"  Downloads: {entry.downloads:,}",
                        "---",
                        f"  {entry.description}",
                    ]))

        elif sub == "list":
            if reg is None:
                pass
            else:
                pkgs = reg.list_installed()
                if not pkgs:
                    print(theme.info("  No plugins installed yet."))
                else:
                    print(theme.info(f"  {len(pkgs)} installed plugin(s):"))
                    for p in pkgs:
                        tick = "✓" if p.get("verified") else " "
                        print(f"  [{tick}] {p['name']:<32} v{p.get('version','?')}")

        elif sub == "popular":
            if reg is None:
                pass
            else:
                popular = reg.get_popular(n=10)
                if not popular:
                    print(theme.info("  No plugins in registry."))
                else:
                    print(theme.info("  Most popular plugins:"))
                    for i, p in enumerate(popular, 1):
                        print(f"  {i:2}. {p.name:<30} {p.downloads:>6,} downloads  {p.description[:40]}")

        elif sub == "refresh":
            if reg is None:
                pass
            else:
                ok = reg.refresh()
                if ok:
                    s = reg.summary()
                    print(theme.success(f"  Registry refreshed: {s['total_plugins']} plugins available."))
                else:
                    print(theme.warning("  Could not reach registry. Using cached index."))

        elif sub == "publish":
            if reg is None or not rest:
                print(theme.warning("  Usage: /plugin publish <plugin-dir>"))
            else:
                ok, msg = reg.publish(rest.strip())
                if ok:
                    print(theme.success(f"  {msg}"))
                else:
                    print(theme.error(f"  {msg}"))

        else:
            print(theme.warning(f"  Unknown /plugin subcommand: {sub!r}. Try /plugin help"))

    elif cmd == "/conv":
        # ── Conversation compression ─────────────────────────────────────────
        sub  = parts[1].lower() if len(parts) > 1 else "help"

        try:
            from core.conversation_compression import (
                ConversationCompressor, estimate_tokens
            )
        except Exception as _cpe:
            print(theme.error(f"  [Conv] Import error: {_cpe}"))
            sub = ""

        if sub in ("help", ""):
            print(theme.box([
                "  CONVERSATION COMPRESSION", "---",
                "  /conv status          Show current conversation token count",
                "  /conv compress        Compress conversation history now",
                "  /conv reset           Clear running summary state",
                "  /conv summary         Show current compressed summary",
            ]))

        elif sub == "status":
            msgs = session.get_history() if hasattr(session, "get_history") else []
            tokens = estimate_tokens(msgs) if msgs else 0
            print(theme.info(f"  Conversation: {len(msgs)} turns, ~{tokens:,} tokens"))

        elif sub == "compress":
            msgs = session.get_history() if hasattr(session, "get_history") else []
            if not msgs:
                print(theme.info("  No conversation history to compress."))
            else:
                compressor = ConversationCompressor(enable_quality_check=False)
                result = compressor.compress(msgs)
                s = result.stats
                print(theme.success(
                    f"  Compressed {s.turns_before} → {s.turns_after} turns  "
                    f"({s.tokens_before:,} → {s.tokens_after:,} tokens, "
                    f"ratio={s.compression_ratio:.2f})"
                ))

        elif sub == "summary":
            print(theme.info("  Use /conv compress first to generate a summary."))

        elif sub == "reset":
            print(theme.info("  Conversation compression state reset."))

        else:
            print(theme.warning(f"  Unknown /conv subcommand: {sub!r}. Try /conv help"))

    # ── /obsidian ─────────────────────────────────────────────────────────────
    elif cmd == "/obsidian":
        sub = args[0] if args else "status"
        if sub == "status":
            if _OBSIDIAN_AVAILABLE:
                try:
                    om = _get_obsidian_memory()
                    print(theme.info(om.summary()))
                    s = om.status()
                    print(theme.info(f"  Notes: {s['total_notes']}  │  Pending facts: {s['pending_facts']}  │  Entities: {s['pending_entities']}"))
                except Exception as e:
                    print(theme.error(f"  Obsidian error: {e}"))
            else:
                print(theme.warning("  Obsidian module not available."))
        elif sub == "sync":
            if _OBSIDIAN_AVAILABLE:
                try:
                    result = _get_obsidian_memory().sync_all()
                    print(theme.success(f"  Synced — facts: {result['facts']}, entities: {result['entities']}"))
                    if result["errors"]:
                        print(theme.warning(f"  Errors: {result['errors']}"))
                except Exception as e:
                    print(theme.error(f"  Sync failed: {e}"))
        elif sub == "open":
            import subprocess as _sp
            vault_path = str(_get_obsidian_memory()._vault.root) if _OBSIDIAN_AVAILABLE else ""
            if vault_path:
                _sp.Popen(["open", vault_path])
                print(theme.success(f"  Opened vault: {vault_path}"))
        else:
            print(theme.info("  Usage: /obsidian [status|sync|open]"))

    # ── /vector ───────────────────────────────────────────────────────────────
    elif cmd == "/vector":
        sub = args[0] if args else "status"
        if sub == "status":
            if _VECTOR_MEMORY_AVAILABLE:
                try:
                    vm = _get_vector_memory()
                    print(theme.info(vm.summary()))
                except Exception as e:
                    print(theme.error(f"  Vector memory error: {e}"))
            else:
                print(theme.warning("  Vector memory not available (pip install lancedb sentence-transformers)"))
        elif sub == "recall" and args[1:]:
            query = " ".join(args[1:])
            if _VECTOR_MEMORY_AVAILABLE:
                results = _get_vector_memory().recall(query, top_k=5)
                if results:
                    for r in results:
                        print(theme.info(f"  [{r.score:.2f}] {r.text[:100]}"))
                else:
                    print(theme.dim("  No relevant memories found."))
        elif sub == "remember" and args[1:]:
            text = " ".join(args[1:])
            if _VECTOR_MEMORY_AVAILABLE:
                ok, reason = _get_vector_memory().remember(text, source="user")
                print(theme.success(f"  Stored: {text[:50]}") if ok else theme.warning(f"  Skip: {reason}"))
        else:
            print(theme.info("  Usage: /vector [status|recall <query>|remember <text>]"))

    # ── /synth ────────────────────────────────────────────────────────────────
    elif cmd == "/synth":
        sub = args[0] if args else "status"
        if sub == "status":
            if _SKILL_SYNTH_AVAILABLE:
                try:
                    s = _get_synthesizer().stats()
                    print(theme.info(f"  Skills: {s['total_skills']}  │  Avg quality: {s['avg_quality']}  │  Total uses: {s['total_uses']}"))
                except Exception as e:
                    print(theme.error(f"  Skill synth error: {e}"))
        elif sub == "list":
            if _SKILL_SYNTH_AVAILABLE:
                skills_list = _get_synthesizer().list_skills(20)
                if not skills_list:
                    print(theme.dim("  No synthesized skills yet. Complete a multi-step task first."))
                else:
                    for sk in skills_list:
                        print(theme.info(f"  [{sk['quality']:.1f}] {sk['name']} — {sk['description'][:60]}  (used {sk['use_count']}×)"))
            else:
                print(theme.warning("  Skill synthesizer not available."))
        elif sub == "search" and args[1:]:
            query = " ".join(args[1:])
            if _SKILL_SYNTH_AVAILABLE:
                matches = _get_synthesizer()._store.search_index(query, top_k=5)
                for m in matches:
                    print(theme.info(f"  {m['name']} — {m['description'][:60]}"))
        else:
            print(theme.info("  Usage: /synth [status|list|search <query>]"))

    # ── /desktop ──────────────────────────────────────────────────────────────
    elif cmd == "/desktop":
        sub = args[0] if args else "status"
        if sub == "status":
            if _COMPUTER_USE_AVAILABLE:
                result = computer_use_status()
                print(theme.info(f"  pyautogui={'✓' if result.get('pyautogui') else '✗'}  │  mss={'✓' if result.get('mss') else '✗'}  │  platform={result.get('platform')}  │  screen={result.get('screen_size')}"))
            else:
                print(theme.warning("  Computer use not available (pip install pyautogui mss)"))
        elif sub == "screenshot":
            if _COMPUTER_USE_AVAILABLE:
                from core.computer_use import screenshot as _cu_ss
                result = _cu_ss()
                if result["success"]:
                    print(theme.success(f"  Screenshot saved: {result['path']} ({result['width']}×{result['height']})"))
                else:
                    print(theme.error(f"  Screenshot failed: {result['error']}"))
            else:
                print(theme.warning("  Computer use not available."))
        elif sub == "click" and len(args) >= 3:
            if _COMPUTER_USE_AVAILABLE:
                from core.computer_use import mouse_click as _cu_click
                result = _cu_click(int(args[1]), int(args[2]))
                print(theme.success(f"  Clicked ({args[1]}, {args[2]})") if result["success"] else theme.error(result.get("error", "")))
        elif sub == "type" and args[1:]:
            if _COMPUTER_USE_AVAILABLE:
                from core.computer_use import keyboard_type as _cu_type
                text = " ".join(args[1:])
                result = _cu_type(text)
                print(theme.success(f"  Typed: {text[:40]}") if result["success"] else theme.error(result.get("error", "")))
        else:
            print(theme.info("  Usage: /desktop [status|screenshot|click <x> <y>|type <text>]"))

    # ── /router ───────────────────────────────────────────────────────────────
    elif cmd == "/router":
        if _SMART_ROUTER_AVAILABLE and _smart_router:
            sub = args[0] if args else "status"
            if sub == "status":
                print(theme.info(f"  {_smart_router.status()}"))
            elif sub == "test" and args[1:]:
                q = " ".join(args[1:])
                decision = _smart_router.route(q)
                print(theme.info(f"  Query: {q[:60]}"))
                print(theme.info(f"  → model={decision.model}  type={decision.task_type.value}  reason={decision.reason}"))
            elif sub == "models":
                for m in _smart_router.available_models():
                    avail = "✓" if m["available"] else "✗"
                    print(theme.info(f"  [{avail}] {m['name']:<30} tasks={','.join(m['tasks'])}  ctx={m['context']}  cost=${m['cost_per_1k']}/1k"))
            else:
                print(theme.info("  Usage: /router [status|models|test <prompt>]"))
        else:
            print(theme.warning("  Smart router not available."))

    else:
        _ALL_CMDS = [
            "/help", "/exit", "/clear", "/undo", "/retry", "/history", "/compress",
            "/save", "/load", "/sessions", "/search", "/snapshot", "/rollback",
            "/title", "/export", "/usage", "/tools", "/skills", "/curator",
            "/dashboard", "/gateway", "/webhook", "/mcp", "/approve", "/memory",
            "/remember", "/forget", "/soul", "/model", "/models", "/local",
            "/config", "/setup", "/doctor", "/status", "/schedule",
            "/rag", "/secrets", "/toolsets",
            "/heartbeat", "/goal", "/macro", "/retry",
            "/kanban", "/tasks", "/plugins", "/serve",
            "/mesh", "/reflect", "/checkpoint", "/pool",
            "/memory", "/swe", "/voice", "/plugin", "/conv",
            "/obsidian", "/vector", "/synth", "/desktop", "/router",
        ]
        # Suggest close matches
        suggestions = [c for c in _ALL_CMDS if cmd in c or c.startswith(cmd[:4])]
        hint = f"  Did you mean: {', '.join(suggestions[:3])}" if suggestions else "  Type /help for a list."
        print(theme.warning(f"Unknown command: {cmd}\n{hint}"))


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent_loop(
    session:             SessionManager,
    router:              ModelRouter,
    planner:             HermesPlannerRenderer,
    tool_registry:       ToolRegistry,
    memory:              MemoryPipeline,
    config:              ConfigManager,
    theme:               Theme,
    soul:                SoulSystem              = None,
    context_inject:      str                     = "",
    skills:              SkillLoader             = None,
    curator:             "Curator | None"        = None,
    knowledge:           "KnowledgeBase | None"  = None,
    cost_tracker:        "CostTracker | None"    = None,
    semantic_mem:        "SemanticMemory | None" = None,
    intended_recipient:  str                     = "",
) -> None:
    global _APPROVAL_MODE, _curator

    from core.background_review import BackgroundReviewer
    from core.commitments import CommitmentTracker

    max_iters = config.get("max_tool_iters", 12)
    spinner   = ThinkingSpinner()
    _last_tool_call: tuple = ("", "")   # (tool_name, params_repr) for success-loop detection
    _duplicate_count: int  = 0

    # Background self-review daemon — spawns a quiet review thread after complex exchanges
    _bg_reviewer      = BackgroundReviewer(min_exchange_turns=2)
    _commitment_tracker = CommitmentTracker()

    # Detect local/small models so we can use the simplified system prompt
    _model_name = config.get("default_model", "")
    _model_info = config.resolve_model(_model_name)
    _is_local   = _model_info.get("provider", "") in LOCAL_PROVIDERS

    # Per-turn correction counters — reset here so they don't accumulate across
    # multiple user messages and incorrectly burn the correction budget on turn 2+
    _format_correction_count:   int = 0   # counts "not valid JSON" injections this turn
    _response_correction_count: int = 0   # counts "empty/useless" injections this turn

    # Per-turn tool-call guardrail (Hermes-style failure loop detection)
    from core.tool_guardrails import ToolCallGuardrails, GuardrailConfig
    _guardrail = ToolCallGuardrails(
        GuardrailConfig.from_config(config.get("tool_guardrails", {}))
    )

    # Iteration budget — read-only tool calls get a refund so they don't
    # count against the agentic budget (Hermes IterationBudget pattern).
    # Hard ceiling: even if every call is refunded the loop cannot run
    # more than 4× the budget (prevents infinite loops on read-only agents).
    from core.iteration_budget import IterationBudget
    _budget    = IterationBudget(max_iters)
    _abs_limit = max_iters * 4
    iteration  = 0

    while _budget.consume() and iteration < _abs_limit:
        iteration += 1

        # Auto-truncate if context is very long
        if session.maybe_truncate(hard_limit=120):
            print(theme.dim("  [Context] Auto-compressed to stay within token budget."))

        # Preemptive compaction: estimate token count and compress early to
        # avoid mid-turn overflow on large contexts (Hermes proactive compaction).
        _est_tokens = sum(len(m.get("content", "")) // 4
                         for m in session.get_messages_for_api())
        _ctx_limit  = config.get("context_token_limit", 80_000)
        if _est_tokens > _ctx_limit * 0.85 and len(session) > 20:
            removed = session.compress(keep_first=4, keep_recent=20)
            if removed:
                print(theme.dim(
                    f"  [Context] Preemptive compaction: {removed} messages trimmed "
                    f"(~{_est_tokens:,} tokens → within 85% limit)."
                ))

        system   = build_system_prompt(tool_registry, memory, soul,
                                       context_inject, skills,
                                       is_local=_is_local, knowledge=knowledge)

        # ── Inject long-term semantic memories (first iteration only) ─────────
        if iteration == 1 and semantic_mem is not None:
            last_user = next(
                (m["content"] for m in reversed(session._messages) if m["role"] == "user"),
                "",
            )
            # max_chars=1200 keeps the injected block ≤ ~300 tokens so it
            # can't significantly inflate input costs across turns.
            mem_block = semantic_mem.as_context_block(
                last_user,
                session_id=session._session_id,
                max_chars=1200,
            )
            if mem_block:
                system = mem_block + "\n\n" + system

        # ── Phase 11: Inject vector memory + synthesized skills (iter 1) ──────
        if iteration == 1:
            last_user = next(
                (m["content"] for m in reversed(session._messages) if m["role"] == "user"),
                "",
            )
            # Vector memory context
            if _VECTOR_MEMORY_AVAILABLE:
                try:
                    _vm_block = _get_vector_memory().build_context_block(last_user, limit=5)
                    if _vm_block:
                        system = _vm_block + "\n\n" + system
                except Exception:
                    pass
            # Obsidian context
            if _OBSIDIAN_AVAILABLE:
                try:
                    _ob_block = _get_obsidian_memory().get_context(last_user, limit=3)
                    if _ob_block:
                        system = _ob_block + "\n\n" + system
                except Exception:
                    pass
            # Synthesized skill hints
            if _SKILL_SYNTH_AVAILABLE:
                try:
                    _sk_hints = _get_synthesizer().get_hints_for(last_user)
                    if _sk_hints:
                        system = _sk_hints + "\n\n" + system
                    # Start a new trajectory for this run
                    _get_synthesizer().start_trajectory(last_user)
                except Exception:
                    pass

        messages = session.get_messages_for_api()

        # ── Local model context hygiene ───────────────────────────────────────
        # Small local models (llama3.2, mistral-7b, etc.) have tight context
        # windows (~8k tokens).  SYSTEM_FEEDBACK correction messages injected
        # during previous turns pile up and overflow the context, causing the
        # model to lose sight of the format instructions entirely.
        # Fix: before each call strip out SYSTEM_FEEDBACK from *older* turns
        # (keep only those from the current turn), then cap to 16 messages.
        if _is_local and messages:
            # Find the index of the last *real* user message (not a feedback)
            _last_real_idx = 0
            for _idx, _m in enumerate(messages):
                if (_m.get("role") == "user"
                        and "[SYSTEM_FEEDBACK]" not in _m.get("content", "")):
                    _last_real_idx = _idx
            # Drop feedback messages that predate the current user turn
            messages = [
                _m for _idx, _m in enumerate(messages)
                if not (
                    _m.get("role") == "user"
                    and "[SYSTEM_FEEDBACK]" in _m.get("content", "")
                    and _idx < _last_real_idx
                )
            ]
            # Hard cap: keep only the 16 most recent messages so the prompt
            # never exceeds ~4-5k tokens of history for a 8k-token model.
            _LOCAL_MAX_MSGS = 16
            if len(messages) > _LOCAL_MAX_MSGS:
                messages = messages[-_LOCAL_MAX_MSGS:]

        # ── LLM-powered context compression (Hermes-style compaction) ─────────
        # Compress middle turns into a structured summary when token count
        # exceeds threshold.  Tail (last 6 turns) and head (first user msg)
        # are always kept verbatim.  Never crashes — returns originals on failure.
        _compress_threshold = config.get("compress_threshold_tokens", 20_000)
        messages, _did_compress = maybe_compress_messages(
            messages,
            system    = system,
            threshold = _compress_threshold,
            tail_turns = 6,
        )
        if _did_compress:
            print(theme.dim("  [Context] LLM-powered compaction applied — middle turns summarised."))

        # ── Streaming-first completion ────────────────────────────────────────
        # Stream response tokens from the API.  The spinner stops as soon as
        # the first token arrives (better perceived latency).  We buffer all
        # chunks to get the full text for JSON parsing and tool dispatch.
        # For "response"-type actions the content is re-displayed with the
        # themed streaming formatter after parsing.
        _use_streaming = config.get("streaming_enabled", True)

        spinner.start()
        raw = None
        try:
            if _use_streaming:
                _chunks: list[str] = []
                _spinner_stopped   = False
                try:
                    for _chunk in router.stream_complete(system=system, messages=messages):
                        if not _spinner_stopped:
                            spinner.stop()
                            _spinner_stopped = True
                        _chunks.append(_chunk)
                except Exception as _stream_err:
                    # Streaming failed mid-stream: fall back to non-streaming
                    if not _spinner_stopped:
                        spinner.stop()
                        _spinner_stopped = True
                    _fallback = router.complete(system=system, messages=messages)
                    raw = _fallback
                else:
                    raw = "".join(_chunks) if _chunks else None
                if not _spinner_stopped:
                    spinner.stop()
            else:
                try:
                    raw = router.complete(system=system, messages=messages)
                finally:
                    spinner.stop()
        except Exception:
            spinner.stop()
            raise

        # ── Record token usage ────────────────────────────────────────────────
        if cost_tracker is not None and router.last_usage:
            u = router.last_usage
            cost_tracker.record(
                model              = u.get("model", ""),
                provider           = u.get("provider", ""),
                input_tokens       = u.get("input_tokens", 0),
                output_tokens      = u.get("output_tokens", 0),
                cache_read_tokens  = u.get("cache_read_tokens", 0),
                cache_write_tokens = u.get("cache_write_tokens", 0),
            )

        if raw is None:
            print(theme.error("No response from model. Check API keys (/config) or run /setup."))
            return

        # ── Length continuation: if the model hit its output token limit, the
        # JSON response will be truncated (broken mid-object).  Inject a specific
        # continuation hint rather than a generic format correction so the model
        # knows to produce a shorter / completed response.
        if getattr(router, "_last_stop_reason", "end_turn") == "max_tokens":
            if _format_correction_count < 2:
                _format_correction_count += 1
                _trunc_hint = (
                    "[SYSTEM_FEEDBACK] Your last response was truncated because it hit "
                    "the max_tokens limit. Please respond again with a SHORTER, COMPLETE "
                    "JSON object. If you need to show long content, summarise it."
                )
                session.add_message("assistant", raw)
                session.add_message("user", _trunc_hint)
                continue

        parsed = router.parse_response(raw)

        if isinstance(parsed, list):
            parsed = next((item for item in parsed if isinstance(item, dict)), None)

        if parsed is None or not isinstance(parsed, dict):
            # Model returned raw text (non-JSON).
            # If this is the first or second iteration, inject a correction and
            # retry — the model likely forgot the required JSON format.
            # After 2 corrections, display the raw response as a fallback.
            if _is_local:
                # Short fill-in template — local models echo descriptive error text,
                # so we just show the correct shape with a placeholder to fill.
                # Prefixed with [SYSTEM_FEEDBACK] so the context-cleanup pass
                # strips it from older turns before it floods the context window.
                _FORMAT_CORRECTION = (
                    '[SYSTEM_FEEDBACK] JSON only. '
                    'Text answer: {"reply": "..."} '
                    'Tool call: {"tool": "name", "params": {"key": "value"}}'
                )
            else:
                _FORMAT_CORRECTION = (
                    "[SYSTEM_FEEDBACK] Your last response was NOT valid JSON. "
                    "You MUST respond with ONLY a JSON object in the action format shown above. "
                    "Do NOT write prose, markdown, or code snippets for the user — "
                    "call the appropriate tool using the action format, or use "
                    "action.type='response' for a conversational answer. "
                    "Output ONLY the JSON object, nothing else."
                )
            # Use the per-turn counter so corrections from previous user messages
            # don't burn the budget before the model even gets a chance this turn.
            if _format_correction_count < 2:
                _format_correction_count += 1
                session.add_message("assistant", raw)
                session.add_message("user", _FORMAT_CORRECTION)
                continue   # retry in the next iteration

            # Fallback after 2 failed corrections.
            # For local models the raw text is almost always noise/junk — show
            # a canned message so the user sees something sensible, and record
            # the raw in session for context but don't display it.
            if _is_local:
                _fallback = ("I'm having trouble formatting my response right now. "
                             "Try rephrasing your question or switch to a larger model "
                             "(/model) for better results.")
                theme.assistant_response(_fallback, stream=True)
                session.add_message("assistant", _fallback)
            else:
                theme.assistant_response(raw, stream=True)
                session.add_message("assistant", raw)
            _trigger_memory(memory, session, config)
            return

        # Only render scratchpad for capable models — local models never fill it properly
        if not _is_local and "scratchpad" in parsed and isinstance(parsed["scratchpad"], dict):
            try:
                planner.render(parsed["scratchpad"], theme)
            except Exception:
                pass  # Never let scratchpad rendering crash the agent loop

        action = parsed.get("action", {})

        # ── Normalise action — guard against every malformed shape ────────────
        # Shape 1: action is a bare string e.g. "action": "email_send"
        if isinstance(action, str):
            if action == "response":
                action = {"type": "response", "content": parsed.get("content", raw)}
            elif action in tool_registry.tools:
                action = {"type": "tool", "tool_name": action,
                          "params": parsed.get("params", {})}
            else:
                action = {"type": "response", "content": raw}

        # Shape 2: action is not a dict (int, list, None…) — treat as plain response
        if not isinstance(action, dict):
            action = {"type": "response", "content": raw}

        # Shape 3: action is empty but model put tool call at the top level
        #   e.g. {"scratchpad": {…}, "tool_name": "email_send", "params": {…}}
        if not action and parsed.get("tool_name"):
            action = {"type": "tool",
                      "tool_name": parsed["tool_name"],
                      "params":    parsed.get("params", {})}

        # Shape 4: model used "tools": [{name, params}] instead of "action": {…}
        #   (OpenAI-style tool_calls list or common local-model drift)
        if not action:
            tools_list = parsed.get("tools") or parsed.get("tool_calls") or []
            if isinstance(tools_list, list) and tools_list:
                first = tools_list[0]
                if isinstance(first, dict):
                    tn = (first.get("name") or first.get("tool_name")
                          or first.get("function", {}).get("name", ""))
                    pr = (first.get("params") or first.get("arguments")
                          or first.get("input") or {})
                    if isinstance(pr, str):       # some models JSON-encode params
                        try:
                            pr = json.loads(pr)
                        except Exception:
                            pr = {}
                    if tn:
                        action = {"type": "tool", "tool_name": tn, "params": pr}

        # Shape 5: model put tool name directly as a top-level JSON key
        #   e.g. {"email_send": {"to": "...", "subject": "...", "body": "..."}}
        #   or   {"telegram_send": {"chat_id": "...", "text": "..."}}
        if not action:
            for key in list(parsed.keys()):
                if key in tool_registry.tools:
                    params_val = parsed[key]
                    if not isinstance(params_val, dict):
                        params_val = {}
                    action = {"type": "tool", "tool_name": key, "params": params_val}
                    break

        # Shape 6: model omitted the "action" wrapper entirely — put "type" and
        #   "content" / "tool_name" at the top level of the JSON object.
        #   e.g. {"type": "response", "content": "Hi!", "scratchpad": {...}}
        #   e.g. {"type": "tool", "tool_name": "shell_exec", "params": {...}}
        if not action and parsed.get("type"):
            top_type = parsed["type"]
            if top_type == "response":
                raw_content = parsed.get("content", "") or parsed.get("text", "")
                # Flatten nested content dict e.g. {"text": "..."} or {"message": "..."}
                if isinstance(raw_content, dict):
                    raw_content = (raw_content.get("text") or raw_content.get("message")
                                   or raw_content.get("content") or "")
                action = {"type": "response", "content": str(raw_content) if raw_content else raw}
            elif top_type == "tool" or top_type in tool_registry.tools:
                tn = parsed.get("tool_name") or (top_type if top_type in tool_registry.tools else "")
                pr = parsed.get("params") or parsed.get("arguments") or {}
                if isinstance(pr, str):
                    try:
                        pr = json.loads(pr)
                    except Exception:
                        pr = {}
                if tn:
                    action = {"type": "tool", "tool_name": tn,
                              "params": pr if isinstance(pr, dict) else {}}

        # Shape 7: local-model simplified format  {"reply": "..."} or {"tool": "name", "params": {}}
        if not action:
            if parsed.get("reply"):
                action = {"type": "response", "content": str(parsed["reply"])}
            elif parsed.get("tool") and parsed["tool"] in tool_registry.tools:
                action = {"type": "tool",
                          "tool_name": parsed["tool"],
                          "params": parsed.get("params", {})}

        # Shape 8: last-resort — scan every top-level value for a displayable string.
        # Covers invented key names like "content_info", "display_text", "message", etc.
        if not action:
            _SKIP_KEYS = {"type", "role", "action", "scratchpad", "next_step", "objective",
                          "workspace_vars", "code_draft", "tool_name", "tool_calls",
                          "model", "id", "usage", "stop_reason"}
            _CONTENT_KEYS = ("content", "text", "message", "answer", "response",
                             "reply", "output", "result", "display_text")
            # Prefer known content key names first
            for ck in _CONTENT_KEYS:
                val = parsed.get(ck)
                if isinstance(val, str) and val.strip():
                    action = {"type": "response", "content": val.strip()}
                    break
                if isinstance(val, dict):
                    for subck in _CONTENT_KEYS:
                        sv = val.get(subck)
                        if isinstance(sv, str) and sv.strip():
                            action = {"type": "response", "content": sv.strip()}
                            break
                    if action:
                        break
            # If still nothing, scan all remaining keys for the longest non-empty string
            if not action:
                best = ""
                for k, v in parsed.items():
                    if k in _SKIP_KEYS:
                        continue
                    if isinstance(v, str) and len(v.strip()) > len(best):
                        best = v.strip()
                    elif isinstance(v, dict):
                        for sv in v.values():
                            if isinstance(sv, str) and len(sv.strip()) > len(best):
                                best = sv.strip()
                if best:
                    action = {"type": "response", "content": best}

        action_type = action.get("type", "response")

        # Normalise action type — some models emit "type": "python_exec"
        if action_type not in ("tool", "response", "parallel_tools"):
            if action_type in tool_registry.tools:
                action["tool_name"] = action_type
                action_type = "tool"
            elif action.get("tool_name"):
                action_type = "tool"

        # ── Guard: detect hallucinated tool results ───────────────────────────
        # If the model returned a JSON that looks like a fabricated tool result
        # (has success/message/error keys, no action content) WITHOUT actually
        # calling any tool — inject feedback and force a retry.
        _FAKE_KEYS   = {"success", "output", "error", "message",
                        "returncode", "stdout", "stderr"}
        _REAL_KEYS   = {"content", "response", "text", "answer", "thought"}
        if action_type == "response" and not action.get("content"):
            parsed_data_keys = set(parsed.keys()) - {"scratchpad", "action"}
            if (parsed_data_keys & _FAKE_KEYS and
                    not parsed_data_keys & _REAL_KEYS):
                # Model fabricated a result — give explicit corrective feedback
                feedback = (
                    "[SYSTEM_FEEDBACK] You returned a fabricated tool result "
                    "without actually calling any tool. This is FORBIDDEN. "
                    "You must use the action format to call tools. "
                    "NEVER invent success/failure responses."
                )
                session.add_message("assistant", json.dumps(parsed, ensure_ascii=False))
                session.add_message("user", feedback)
                continue   # re-enter the loop with the corrective feedback

        # ── Parallel tool branch ──────────────────────────────────────────────
        if action_type == "parallel_tools":
            from core.parallel_executor import (
                ParallelToolExecutor, parse_parallel_calls,
            )
            _parallel_calls = parse_parallel_calls(action)
            if not _parallel_calls:
                # Malformed — treat as feedback and retry
                session.add_message("assistant", json.dumps(parsed, ensure_ascii=False))
                session.add_message("user",
                    "[SYSTEM_FEEDBACK] parallel_tools action had no valid 'calls'. "
                    "Each call needs tool_name, params, and id keys.")
                continue

            # Enforce cap to prevent accidental resource exhaustion
            _MAX_PARALLEL = 4
            if len(_parallel_calls) > _MAX_PARALLEL:
                _parallel_calls = _parallel_calls[:_MAX_PARALLEL]

            _parallel_executor = ParallelToolExecutor(max_workers=_MAX_PARALLEL)
            print(theme.info(
                f"  [∥ Parallel] Running {len(_parallel_calls)} tools concurrently…"
            ))
            spinner.stop()

            _parallel_result = _parallel_executor.run(
                calls      = _parallel_calls,
                execute_fn = tool_registry.execute,
                print_fn   = lambda msg: print(theme.dim(msg)),
            )

            print(theme.dim(f"  {_parallel_result.summary_line}"))

            # Inject all results as a single combined user message
            session.add_message("assistant", json.dumps(parsed, ensure_ascii=False))
            session.add_message("user",
                f"[PARALLEL_TOOL_RESULTS]\n{_parallel_result.combined_result_str}"
            )

            # Refund budget for read-only parallel batches (all reads)
            _READ_ONLY_TOOLS = {"duckduckgo_search", "web_scrape", "file_read",
                                "file_exists", "file_info", "dir_list", "file_search",
                                "git_status", "git_log", "git_diff",
                                "knowledge_get", "knowledge_list",
                                "db_query", "db_list_tables", "db_describe_table",
                                "github_repo_info", "github_list_repos",
                                "github_list_issues", "github_list_prs",
                                "github_search_code", "github_search_repos",
                                "github_get_file", "github_user_info",
                                "github_list_commits",
                                "pdf_info", "pdf_extract_text"}
            if all(c.tool_name in _READ_ONLY_TOOLS for c in _parallel_calls):
                _budget.refund_if_read_only("parallel_read", True)

            continue  # re-enter loop so model can reason over all results

        # ── Tool branch ───────────────────────────────────────────────────────
        if action_type == "tool":
            tool_name = action.get("tool_name", "")
            params    = action.get("params", {})

            # Ensure params is always a dict — some models emit strings or lists
            if not isinstance(params, dict):
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except Exception:
                        params = {}
                else:
                    params = {}
            if not isinstance(params, dict):  # final safety (json.loads could give a list)
                params = {}

            if not tool_name:
                print(theme.warning("Agent issued tool action with no tool_name. Stopping."))
                return

            # ── Block direct email_send calls — it is internal-only ───────────
            # email_send is NOT in the dispatch map; if the model hallucinates a
            # call to it (from training data / prior context), redirect firmly to
            # email_draft so the user approval box always runs.
            if tool_name == "email_send":
                _es_block = (
                    "[SYSTEM_FEEDBACK] 'email_send' is not a callable tool. "
                    "You MUST use 'email_draft' for ALL email tasks — it handles "
                    "sending internally after user approval. "
                    "Call email_draft with: to, subject, body (plain text). "
                    "Do NOT pass sender_email or app_password as params."
                )
                session.add_message("assistant", json.dumps(parsed, ensure_ascii=False))
                session.add_message("user", _es_block)
                continue   # redirect to email_draft

            # ── Recipient address guard for email_draft ───────────────────────
            # If we know the user intended a specific recipient (extracted from
            # their raw message) and the model is about to call email_draft with
            # a DIFFERENT address, intercept and correct before any email is shown.
            if (tool_name == "email_draft"
                    and intended_recipient
                    and isinstance(params.get("to"), str)
                    and params["to"].strip().lower() != intended_recipient.strip().lower()):
                _addr_feedback = (
                    f"[SYSTEM_FEEDBACK] Wrong recipient address. "
                    f"The user asked to email '{intended_recipient}' but you are about to "
                    f"send to '{params['to']}'. "
                    f"You MUST use the EXACT address the user provided: '{intended_recipient}'. "
                    f"Call email_draft again with to='{intended_recipient}'."
                )
                session.add_message("assistant", json.dumps(parsed, ensure_ascii=False))
                session.add_message("user", _addr_feedback)
                continue   # retry with corrected address

            # ── Duplicate-call guard ──────────────────────────────────────────
            # If the model calls the exact same tool with the same params 3 times
            # in a row it's stuck in a loop — break out with a firm correction.
            _this_call = (tool_name, repr(sorted(params.items())) if params else "")
            if _this_call == _last_tool_call:
                _duplicate_count += 1
            else:
                _duplicate_count = 0
                _last_tool_call  = _this_call

            if _duplicate_count >= 2:
                _dup_feedback = (
                    f"[SYSTEM_FEEDBACK] You have called '{tool_name}' with the same "
                    f"parameters {_duplicate_count + 1} times in a row. "
                    "The task is already done. STOP calling this tool and reply to the user now."
                )
                session.add_message("assistant", json.dumps(parsed, ensure_ascii=False))
                session.add_message("user", _dup_feedback)
                _duplicate_count = 0
                continue

            # Approval gate
            if _APPROVAL_MODE and tool_name not in _SAFE_TOOLS:
                print(theme.warning(
                    f"  Tool approval required:\n"
                    f"  {tool_name}({_format_params(params)})\n"
                ), end="")
                print("  Run this tool? [y/N] ", end="", flush=True)
                try:
                    answer = input().strip().lower()
                except (KeyboardInterrupt, EOFError):
                    answer = "n"
                if answer not in ("y", "yes"):
                    print(theme.info("  Skipped."))
                    session.add_message("assistant", json.dumps(parsed, ensure_ascii=False))
                    session.add_message("user",
                        f"[TOOL_BLOCKED: {tool_name}] User denied. Find an alternative or ask.")
                    continue

            # ── Pre-flight: missing required params ───────────────────────────
            # Check tool definition for required params before calling the tool.
            # Catches the pattern where a local model emits {"tool": "db_query", "params": {}}
            # for a conversational question — the tool would just return "query is required."
            from tools.registry import _TOOL_DEFINITIONS as _TDS
            _td_pre = next((t for t in _TDS if t["name"] == tool_name), None)
            if _td_pre:
                _pre_params  = _td_params(_td_pre)
                _pre_req_set = _td_required(_td_pre)
                if _pre_req_set:
                    _missing_req = [k for k in _pre_req_set if k not in params]
                else:
                    _missing_req = [
                        k for k, v in _pre_params.items()
                        if "required" in str(v).lower() and k not in params
                    ]
                if _missing_req:
                    _missing_fb = (
                        f'[SYSTEM_FEEDBACK] {tool_name} needs: {", ".join(_missing_req)}. '
                        f'Add them or use {{"reply": "..."}} to answer directly.'
                    )
                    session.add_message("assistant", json.dumps(parsed, ensure_ascii=False))
                    session.add_message("user", _missing_fb)
                    continue   # retry with corrective feedback

            # ── Conversational-question guard (local models) ──────────────────
            # If the user asked a short general question and the model is trying to
            # call a data-retrieval tool that makes no sense for it, redirect.
            _DATA_TOOLS = {"db_query", "knowledge_get", "knowledge_list",
                           "knowledge_set", "mongo_query"}
            if _is_local and tool_name in _DATA_TOOLS:
                _last_real_msg = next(
                    (m.get("content", "") for m in reversed(session.get_messages_for_api())
                     if m.get("role") == "user"
                     and not m.get("content", "").startswith("[TOOL_RESULT")
                     and not m.get("content", "").startswith("[SYSTEM_FEEDBACK")
                     and not m.get("content", "").startswith("JSON only")),
                    "",
                )
                _DATA_KEYWORDS = (
                    "database", " db ", "sql", "table", "query", "select ",
                    "insert ", "update ", "schema", "knowledge", "remember",
                    "stored", "saved", "fact", "did i tell", "what did i say",
                )
                if (len(_last_real_msg) < 80
                        and not any(kw in _last_real_msg.lower() for kw in _DATA_KEYWORDS)):
                    _conv_fb = (
                        f'[SYSTEM_FEEDBACK] Don\'t use {tool_name} for this. '
                        f'Answer directly: {{"reply": "your answer here"}}'
                    )
                    session.add_message("assistant", json.dumps(parsed, ensure_ascii=False))
                    session.add_message("user", _conv_fb)
                    continue

            # ── Guardrail pre-flight: block if exact failure / no-progress loop ──
            _gr_pre = _guardrail.before_call(tool_name, params)
            if _gr_pre.should_block:
                session.add_message("assistant", json.dumps(parsed, ensure_ascii=False))
                session.add_message("user", f"[SYSTEM_FEEDBACK] {_gr_pre.message}")
                if _gr_pre.action == "halt":
                    print(theme.warning(f"  [Guardrail HALT] {_gr_pre.message}"))
                    return
                continue

            print(theme.tool_call(f"{tool_name}({_format_params(params)})"))

            # Execute with retry policy if configured
            if _retry_mgr is not None:
                policy = _retry_mgr.get(tool_name)
                if policy.enabled and policy.max_attempts > 1:
                    result = execute_with_retry(
                        tool_name,
                        lambda p=params: tool_registry.execute(tool_name, p),
                        params,
                    )
                else:
                    result = tool_registry.execute(tool_name, params)
            else:
                result = tool_registry.execute(tool_name, params)

            # ── Tool result storage: persist large outputs to disk ────────────
            try:
                from core.tool_result_storage import maybe_persist_result
                _call_id = params.get("_call_id", tool_name + "_" + str(id(result)))
                result = maybe_persist_result(tool_name, _call_id, result)
            except Exception:
                pass   # Non-fatal: continue with original result if storage fails

            # Log to dashboard (non-blocking, doesn't slow the loop)
            log_tool_call(tool_name, params, result)

            # ── Phase 11: record step in skill synthesizer ────────────────────
            if _SKILL_SYNTH_AVAILABLE:
                try:
                    _synth_instance = _get_synthesizer()
                    _tool_ok = not (isinstance(result, dict) and result.get("success") is False)
                    _synth_instance.record_step(tool_name, params, result, success=_tool_ok)
                except Exception:
                    pass
            result_str = json.dumps(result, indent=2, default=str)

            # Truncate huge tool results to avoid context blowout
            MAX_RESULT = 8000
            if len(result_str) > MAX_RESULT:
                result_str = (result_str[:MAX_RESULT] +
                              f"\n[...truncated {len(result_str)-MAX_RESULT} chars]")

            preview = result_str[:85].replace("\n", " ")
            ellipsis = "…" if len(result_str) > 85 else ""
            print(theme.tool_result(f"{preview}{ellipsis}"))

            # ── Guardrail post-execution: update failure counters; warn/halt ───
            # shell_exec / python_exec return success=False + returncode!=0 with empty "error"
            _is_tool_error = isinstance(result, dict) and (
                bool(result.get("error"))
                or result.get("success") is False
                or (isinstance(result.get("returncode"), int)
                    and result["returncode"] != 0)
            )
            _gr_post = _guardrail.after_call(tool_name, params, result_str,
                                             failed=_is_tool_error)
            if _gr_post.action == "warn":
                result_str += _gr_post.guidance_suffix()
            elif _gr_post.action == "halt":
                session.add_message("assistant", json.dumps(parsed, ensure_ascii=False))
                session.add_message("user", f"[SYSTEM_FEEDBACK] {_gr_post.message}")
                print(theme.warning(f"  [Guardrail HALT] {_gr_post.message}"))
                return

            session.add_message("assistant", json.dumps(parsed, ensure_ascii=False))

            # ── If the tool failed due to wrong / missing params, inject correction ──
            _err = result.get("error", "") if isinstance(result, dict) else ""
            _is_param_error = (
                _err.startswith("Invalid params for")          # TypeError from registry
                or ("is required" in _err.lower() and _err.endswith("required."))  # e.g. "query is required."
                or _err.lower().endswith("is required.")
            )
            if _is_param_error:
                _td = next((t for t in _TDS if t["name"] == tool_name), None)
                if _td:
                    _td_p = _td_params(_td)
                    _td_r = _td_required(_td)
                    if _td_r:
                        _param_hints = "  ".join(f"{k}" for k in _td_r)
                    else:
                        _param_hints = "  ".join(
                            f"{k} ({str(v).split('(')[0].strip()})"
                            for k, v in _td_p.items()
                            if "required" in str(v).lower()
                        )
                    _correction = (
                        f'[SYSTEM_FEEDBACK] {tool_name} missing required params: {_param_hints or _err}. '
                        f'Add them or use {{"reply": "..."}} to answer directly.'
                    )
                    session.add_message("user", _correction)
                    continue   # re-enter loop with the corrective feedback
                # else fall through and add the result normally

            session.add_message("user", f"[TOOL_RESULT: {tool_name}]\n{result_str}")

            # Refund iteration for read-only tools (they gather info without acting)
            _budget.refund_if_read_only(tool_name, not _is_tool_error)

            # ── Auto-confirm terminal email outcomes ──────────────────────────
            # Local models hallucinate the recipient address when asked to
            # describe a successful send.  For email_draft, skip the model's
            # confirmation turn entirely and emit a system-generated message
            # using the *actual* recipients from the tool result.
            if tool_name == "email_draft" and isinstance(result, dict):
                if result.get("approved") and result.get("sent"):
                    _recips = result.get("recipients", [])
                    _to_str = ", ".join(_recips) if _recips else params.get("to", "recipient")
                    _auto_confirm = f"Email sent to {_to_str}."
                    theme.assistant_response(_auto_confirm, stream=True)
                    session.add_message("assistant", _auto_confirm)
                    _trigger_memory(memory, session, config)
                    if semantic_mem is not None:
                        _save_semantic_turn(semantic_mem, session)
                    if cost_tracker is not None:
                        print(theme.dim(f"  {cost_tracker.status_line()}"))
                    return
                elif result.get("cancelled"):
                    _auto_cancel = "Draft discarded."
                    theme.assistant_response(_auto_cancel, stream=True)
                    session.add_message("assistant", _auto_cancel)
                    _trigger_memory(memory, session, config)
                    return
                # feedback case: let the loop continue so the model can redraft

        # ── Response branch ───────────────────────────────────────────────────
        elif action_type == "response":
            content = action.get("content", "")
            # Flatten if the model returned content as a dict {"text": "..."} etc.
            if isinstance(content, dict):
                content = (content.get("text") or content.get("message")
                           or content.get("content") or content.get("answer") or "")
            content = str(content).strip() if content else ""
            content = content or raw.strip()

            # ── Strip thinking/reasoning scratchpad tags ──────────────────────
            # Models like DeepSeek-R1, Qwen3, and others emit <think>…</think>
            # blocks before their final answer.  Strip them before display so
            # the user never sees internal reasoning scaffolding.
            content = _strip_thinking_tags(content)

            # ── [SILENT] sentinel — agent requests no user-visible output ──────
            # If the agent decides mid-run that its output should not be shown
            # (e.g. a background heartbeat tick), it can include [SILENT] in the
            # content.  Return silently without displaying or saving to session.
            if "[SILENT]" in content:
                return

            # ── Strip leading JSON noise from mixed text+JSON responses ───────
            # Local models sometimes prefix a JSON blob before their prose:
            #   '{"seeAlso": ["x","y"]}\n\nTo drink water...'
            # Use the router's own balanced-brace extractor to find the JSON block,
            # then keep only the prose text that follows it.
            if content.startswith("{") and not content.startswith('{"reply"'):
                _json_blob = ModelRouter._extract_first_json(content)
                if _json_blob:
                    _json_end = content.find(_json_blob) + len(_json_blob)
                    _prose    = content[_json_end:].strip()
                    if len(_prose) > 20:   # only swap if there's real prose
                        content = _prose

            # ── Guard: detect useless/empty responses and retry ───────────────
            # Triggers on: "{}", "{ }", bare tool/key names, system prompt echoes,
            # single-word non-answers, MIME types, or documentation platitudes.
            _USELESS = {"", "{}", "{ }", "null", "none", "false", "true"}
            # Patterns that indicate the model echoed meta-text instead of answering
            _META_PHRASES = (
                # System-prompt echoes / documentation platitudes
                "available tools",
                "please reformat",
                "[system_feedback]",
                "nothing outside",
                "can be found in the",
                "refer to the documentation",
                "see the documentation",
                "documentation for more",
                "tool descriptions",
                "provided documentation",
                "tool list",
                "json only.",
                "text answer:",
                "for a text reply",
                "for a tool call",
                # Apology / meta-commentary (model narrating its own failure)
                "i apologize for the",
                "i'm sorry for the",
                "sorry for the empty",
                "sorry for the confusion",
                "sorry, i didn",
                "apologize for the confusion",
                "apologize for any",
                "please rephrase your",
                "can you please rephrase",
                "could you please rephrase",
                "i cannot provide",
                "i am unable to",
                "i'm unable to",
                "i cannot process",
                "i don't have the ability",
                # UI artifacts (model hallucinating dashboard / chat buttons)
                "send message",
                "submit button",
                "click here",
                "type your message",
                "enter your message",
                # Format reminders the model echoes verbatim
                "response format",
                "output one json",
                "output only valid json",
                "every response must",
                "strict response format",
            )
            import re as _re
            _is_mime   = bool(_re.match(r'^(text|application|image|audio|video)/[\w.+-]+$',
                                        content.lower()))
            # Short fragments that are clearly not real answers:
            # 1-3 word responses like "Send Message", "Ok.", "Done", etc.
            # that don't match known greetings or acknowledgements.
            _REAL_SHORT = {"ok", "yes", "no", "done", "sure", "hi", "hey", "hello",
                           "thanks", "thank you", "great", "got it", "noted"}
            _word_count  = len(content.split())
            _is_short_fragment = (
                _word_count <= 3
                and content.lower().strip(".!? ") not in _REAL_SHORT
                and not content[0].isdigit()   # allow "1.", "2 files", etc.
                if content else False
            )
            _is_useless = (
                content.lower() in _USELESS
                or (len(content) <= 6 and not content[0].isalnum() if content else True)
                or _is_mime
                or (_is_local and _is_short_fragment)    # local models only — cloud models can give short answers
                or any(content.lower().startswith(p) for p in _META_PHRASES)
                or any(p in content.lower() for p in _META_PHRASES)
                or content == raw.strip() and raw.strip().startswith("{")
                   and len(content) < 20   # short raw JSON passed through
            )
            if _is_useless and _response_correction_count < 2:
                _response_correction_count += 1
                _useless_hint = (
                    '[SYSTEM_FEEDBACK] Your last response was empty or not useful. '
                    'You MUST provide a real answer. '
                    'Use {"reply": "your actual answer"} for text, '
                    'or {"tool": "tool_name", "params": {}} to call a tool.'
                )
                session.add_message("assistant", content)
                session.add_message("user", _useless_hint)
                continue   # retry

            # ── Reflection engine — self-critique before showing the user ────────
            # Runs heuristic + optional fast-LLM review to catch hallucinations,
            # missing tool calls, and self-contradictions.  Only active if enabled
            # in config and the response isn't from a local model (reflection LLM
            # calls would be too expensive / slow on local hardware).
            _reflection_enabled = config.get("reflection_enabled", True) and not _is_local
            if _reflection_enabled:
                try:
                    from core.reflection import get_engine, ReflectionConfig
                    _refl_cfg = ReflectionConfig(
                        enabled           = True,
                        max_corrections   = config.get("reflection_max_corrections", 2),
                        use_fast_model    = True,
                        hallucination_check = config.get("reflection_hallucination_check", True),
                    )
                    _refl_engine = get_engine(router, _refl_cfg)
                    # Gather tool results from the last agent loop turn
                    _tool_results_for_refl = []
                    for _rm in reversed(session._messages[-10:]):
                        if (_rm.get("role") == "user"
                                and _rm.get("content", "").startswith("[TOOL_RESULT:")):
                            _tc = _rm["content"]
                            _tn = _tc.split("]")[0].replace("[TOOL_RESULT:", "").strip()
                            _tool_results_for_refl.append({"tool_name": _tn, "output": _tc[_tc.find("]")+1:]})
                    _last_user_for_refl = next(
                        (m["content"] for m in reversed(session._messages)
                         if m["role"] == "user"
                         and not m["content"].startswith("[TOOL_RESULT")
                         and not m["content"].startswith("[SYSTEM_FEEDBACK")),
                        "",
                    )
                    _refl_result = _refl_engine.reflect(
                        user_message   = _last_user_for_refl,
                        agent_response = {"content": content, "thought": ""},
                        tool_results   = _tool_results_for_refl,
                        session_id     = session._session_id,
                    )
                    if _refl_result.did_correct:
                        _corrected_content = _refl_result.final_response.get("content", content)
                        if _corrected_content and _corrected_content != content:
                            content = _corrected_content
                            print(theme.dim("  [Reflection] Response corrected."))
                    elif _refl_result.has_issues:
                        _refl_summary = _refl_engine.summary_for_prompt(_refl_result)
                        if _refl_summary:
                            print(theme.dim(f"  [Reflection] {len(_refl_result.issues)} issue(s) noted."))
                except Exception:
                    pass   # reflection must never crash the agent loop

            theme.assistant_response(content, stream=True)
            session.add_message("assistant", content)
            _trigger_memory(memory, session, config)
            # Save to semantic memory
            if semantic_mem is not None:
                _save_semantic_turn(semantic_mem, session)
            # Show token / cost status line
            if cost_tracker is not None:
                print(theme.dim(f"  {cost_tracker.status_line()}"))
            # Curator: try to auto-generate a skill from this completed exchange
            cur = curator or _curator
            if cur:
                cur.maybe_curate(session._messages)

            # ── Phase 11: Synthesize skill from completed trajectory ──────────
            if _SKILL_SYNTH_AVAILABLE:
                try:
                    _synth = _get_synthesizer()
                    _synth.finish_trajectory(content, success=True)
                    _last_user = next(
                        (m["content"] for m in reversed(session._messages) if m["role"] == "user"
                         and not m["content"].startswith("[TOOL_RESULT")), ""
                    )
                    _new_skill = _synth.synthesize_from_current(_last_user, content[:100])
                    if _new_skill:
                        print(theme.dim(f"  ✦ Skill synthesized: '{_new_skill.name}' (quality={_new_skill.quality:.1f})"))
                    else:
                        _synth.reset_trajectory()  # reset even if not synthesized
                except Exception:
                    pass   # skill synthesis must never crash the agent loop

            # ── Phase 11: save key facts to vector + Obsidian memory ──────────
            if _VECTOR_MEMORY_AVAILABLE:
                try:
                    _vm = _get_vector_memory()
                    # Store the assistant's final response as a memory
                    if len(content) > 30:
                        _vm.remember(content[:500], source="agent", category="general")
                except Exception:
                    pass

            # ── Commitment detection: extract implicit follow-up promises ──────
            # If the agent said "I'll check back on that tomorrow" etc., log it.
            try:
                _last_user_msg = next(
                    (m["content"] for m in reversed(session._messages)
                     if m["role"] == "user"
                     and not m["content"].startswith("[TOOL_RESULT")
                     and not m["content"].startswith("[SYSTEM_FEEDBACK")),
                    "",
                )
                _commitment_tracker.extract_from_exchange(
                    assistant_text = content,
                    user_text      = _last_user_msg,
                    session_key    = session._session_id,
                )
            except Exception:
                pass   # never crash on commitment detection

            # ── Background self-review: post-exchange knowledge / skill updates ─
            # Run a quiet review thread to capture learnings from complex exchanges.
            try:
                def _make_review_factory():
                    def _factory(tool_whitelist=None, quiet_mode=True,
                                 max_iters=8, context_tag=None):
                        def _run(prompt: str) -> str:
                            sub_sess = SessionManager()
                            sub_sess.add_message("user", prompt)
                            run_agent_loop(
                                session=sub_sess, router=router, planner=planner,
                                tool_registry=tool_registry, memory=memory,
                                config=config, theme=theme, soul=soul,
                                context_inject=context_inject, skills=skills,
                                curator=None, knowledge=knowledge,
                                cost_tracker=None, semantic_mem=None,
                            )
                            for _m in reversed(sub_sess._messages):
                                if _m["role"] == "assistant":
                                    return _m["content"]
                            return ""
                        return _run
                    return _factory
                _bg_reviewer.maybe_spawn(
                    session._messages,
                    agent_runner_factory=_make_review_factory(),
                )
            except Exception:
                pass   # never crash main loop from background review

            return

        else:
            theme.assistant_response(raw, stream=True)
            session.add_message("assistant", raw)
            _trigger_memory(memory, session, config)
            if semantic_mem is not None:
                _save_semantic_turn(semantic_mem, session)
            if cost_tracker is not None:
                print(theme.dim(f"  {cost_tracker.status_line()}"))
            return

    print(theme.warning(
        f"Reached iteration budget ({max_iters} max, {_budget.used} used). "
        "Returning control."
    ))


def _trigger_memory(memory: MemoryPipeline, session: SessionManager,
                    config: ConfigManager) -> None:
    if config.get("memory_enabled", True):
        exchange = session.get_recent_exchange()
        threading.Thread(
            target=memory.async_evaluate_and_save,
            args=(exchange,),
            daemon=True,
        ).start()


def _save_semantic_turn(sem: "SemanticMemory", session: SessionManager) -> None:
    """
    Save the most recent user+assistant pair to semantic long-term memory.
    Runs in a single background thread to avoid competing with an active LLM.
    """
    msgs = session._messages
    # Collect only the newest user message and newest assistant message
    to_save = []
    found_roles: set = set()
    for msg in reversed(msgs):
        role = msg.get("role", "")
        if role in ("user", "assistant") and role not in found_roles and msg.get("content"):
            to_save.append((role, msg["content"]))
            found_roles.add(role)
        if len(found_roles) == 2:
            break

    if not to_save:
        return

    sid = session._session_id

    def _worker():
        for role, content in to_save:
            try:
                sem.save(sid, role, content)
            except Exception:
                pass   # never crash the main loop from a memory write

    threading.Thread(target=_worker, daemon=True).start()


import re as _re_module

_THINKING_TAG_RE = _re_module.compile(
    r'<(?:think|thinking|thought|reasoning|REASONING_SCRATCHPAD|final)[^>]*>.*?'
    r'</(?:think|thinking|thought|reasoning|REASONING_SCRATCHPAD|final)>',
    _re_module.DOTALL | _re_module.IGNORECASE,
)

def _strip_thinking_tags(text: str) -> str:
    """Strip <think>/<thinking>/<thought>/<reasoning> scratchpad blocks from model output."""
    cleaned = _THINKING_TAG_RE.sub("", text)
    # Collapse runs of blank lines left behind by stripped blocks
    cleaned = _re_module.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def _format_params(params: dict) -> str:
    if not params:
        return ""
    pairs = []
    for k, v in params.items():
        v_str = str(v)
        if len(v_str) > 40:
            v_str = v_str[:37] + "…"
        pairs.append(f"{k}={repr(v_str)}")
    return ", ".join(pairs[:4])


# ── Recipient address extractor ───────────────────────────────────────────────

def _extract_recipient(user_input: str) -> str:
    """
    Extract the intended email recipient from the user's raw message.

    Matches patterns like:
      "send email to foo@bar.com"   "email alice@x.com"   "write to bob@y.com"
      "message to charlie@z.org"   "contact foo@bar.com"

    Returns the matched address string, or "" if none found.
    """
    import re
    _ADDR = r'([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})'
    recipient_patterns = [
        rf'(?:send|write|compose|draft)\s+(?:an?\s+)?(?:email|mail|message)\s+(?:to\s+)?{_ADDR}',
        rf'(?:email|mail|message|contact|reach)\s+(?:to\s+)?{_ADDR}',
        rf'\bto\s+{_ADDR}',
    ]
    for pat in recipient_patterns:
        m = re.search(pat, user_input, re.IGNORECASE)
        if m:
            return m.group(1)
    # Last resort: any bare email address in the message
    m = re.search(_ADDR, user_input)
    if m:
        return m.group(1)
    return ""


# ── Credential interceptor ────────────────────────────────────────────────────

def _intercept_credentials(user_input: str, knowledge: "KnowledgeBase", theme: "Theme") -> str:
    """
    Scan a raw user message for email addresses and app passwords.
    Saves them to the knowledge base immediately so email_draft can find them —
    no model cooperation required.

    Returns the (possibly redacted) message so passwords aren't stored in
    session history.
    """
    import re

    text = user_input

    # ── Sender email ──────────────────────────────────────────────────────────
    # Match patterns like:
    #   "My Email: foo@bar.com"  "email is foo@bar.com"  "gmail: foo@bar.com"
    #   "My Email Address foo@bar.com"  "from foo@bar.com"
    # NOT "send email to foo@bar.com" (that's the recipient)
    _ADDR = r'([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})'
    sender_patterns = [
        rf'my\s+(?:email|gmail|email\s+address|sender)[:\s]+{_ADDR}',
        rf'(?:email|gmail|email\s+address)\s+(?:is|:)[:\s]+{_ADDR}',
        rf'sender\s+(?:email\s+)?(?:is\s*)?[:\s]+{_ADDR}',
        rf'(?:send|sending)\s+from[:\s]+{_ADDR}',
    ]
    for pat in sender_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            addr = m.group(1)
            existing = knowledge.get("sender_email") or ""
            if existing.strip() != addr.strip():
                knowledge.set("sender_email", addr)
                print(theme.success(f"\n  ✓  Saved your email address → {addr}"))
            break

    # ── App password ──────────────────────────────────────────────────────────
    # Gmail App Passwords are exactly 16 lowercase letters (sometimes spaced
    # as "xxxx xxxx xxxx xxxx").  Only save when preceded by password context.
    pw_ctx = re.compile(
        r'(?:app[\s\-]?password|app[\s\-]?pass|gmail[\s\-]?password'
        r'|my[\s\-]?password|email[\s\-]?password)\s*[:\s]+\s*'
        r'([a-z]{4}\s?[a-z]{4}\s?[a-z]{4}\s?[a-z]{4})',
        re.IGNORECASE,
    )
    m = pw_ctx.search(text)
    if m:
        raw = m.group(1)
        cleaned = raw.replace(" ", "")
        knowledge.set("app_password", cleaned)
        # Redact so it never appears in session history
        user_input = user_input.replace(raw, "[saved]")
        print(theme.success("  ✓  App password saved → [hidden]"))

    return user_input


# ── First-run dependency check ─────────────────────────────────────────────────

def _first_run_dep_check(theme) -> None:
    """
    On first launch (right after the setup wizard), check for the Playwright
    Chromium browser binary — the one thing `pip install` does not provision —
    and offer to download it. Non-fatal: any failure just prints a hint.
    """
    try:
        from core.bootstrap import is_browser_binary_installed, ensure_browser_binary
    except Exception:
        return

    try:
        # Only relevant if the playwright package is present at all.
        import importlib.util
        if importlib.util.find_spec("playwright") is None:
            print(theme.dim(
                "  Browser automation is optional. To enable it later:\n"
                "    python -m core.bootstrap --browser"
            ))
            return

        if is_browser_binary_installed():
            return  # already good

        print(theme.info(
            "Browser automation needs a one-time Chromium download (~120 MB)."
        ))
        try:
            ans = input("  Download it now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans in ("", "y", "yes"):
            ok, msg = ensure_browser_binary(quiet=False)
            if ok:
                print(theme.success("  Browser ready."))
            else:
                print(theme.warning(f"  Browser install skipped: {msg}"))
                print(theme.dim("    Retry anytime: python -m core.bootstrap --browser"))
        else:
            print(theme.dim("    You can install it later: python -m core.bootstrap --browser"))
    except Exception:
        pass  # never block startup on the dependency check


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    global _smart_router, _skill_synth
    import argparse as _argparse

    ap = _argparse.ArgumentParser(
        prog="operon",
        description="Operon — Advanced AI Terminal Cockpit",
        add_help=True,
    )
    ap.add_argument("--batch",   metavar="FILE",
                    help="Run prompts from a text file (one per line) non-interactively")
    ap.add_argument("--prompt",  "-p", metavar="TEXT",
                    help="Run a single prompt non-interactively and exit")
    ap.add_argument("--webhook", action="store_true",
                    help="Start the REST webhook server after launching")
    ap.add_argument("--port",    type=int, default=7271,
                    help="Webhook server port (default 7271)")
    ap.add_argument("--host",    default="127.0.0.1",
                    help="Webhook server host (default 127.0.0.1)")
    ap.add_argument("--install-deps", action="store_true",
                    help="Install/verify all dependencies (incl. browser binary) and exit")
    ap.add_argument("--check-deps", action="store_true",
                    help="Report dependency status and exit")
    ap.add_argument("--skip-dep-check", action="store_true",
                    help="Skip the first-run browser-binary check")
    args, _unknown = ap.parse_known_args()

    # ── Dependency provisioning CLI (no config needed) ────────────────────────
    if args.install_deps:
        from core.bootstrap import provision
        sys.exit(0 if provision(full=False, browser=True) else 1)
    if args.check_deps:
        from core.bootstrap import print_status
        print_status()
        sys.exit(0)

    config = ConfigManager()
    theme  = Theme()
    banner = Banner()

    if not config.is_configured():
        from setup_wizard import run_wizard
        run_wizard(config)
        # Fresh install → offer to grab the browser binary now (one-time).
        if not args.skip_dep_check:
            _first_run_dep_check(theme)

    # Initialise all subsystems
    session       = SessionManager()
    memory        = MemoryPipeline(config)
    router        = ModelRouter(config)
    planner       = HermesPlannerRenderer()
    soul          = SoulSystem()
    scheduler     = TaskScheduler()
    skills        = SkillLoader()
    tool_registry   = ToolRegistry()
    knowledge       = KnowledgeBase()
    cost_tracker    = CostTracker()

    # Semantic memory: skip on local/Ollama models — they already use most RAM.
    # Users can force-enable with  semantic_memory: true  in their config.
    _active_model_info = config.resolve_model(config.get("default_model", ""))
    _using_local       = _active_model_info.get("provider", "") in LOCAL_PROVIDERS
    _sem_forced        = config.get("semantic_memory", None)  # None = auto
    if _sem_forced is True or (_sem_forced is None and not _using_local):
        try:
            semantic_mem = SemanticMemory(config)
        except Exception as _e:
            semantic_mem = None
            print(theme.dim(f"  [Memory] Semantic memory unavailable: {_e}"))
    else:
        semantic_mem = None

    _set_knowledge_base_tool(knowledge)        # wire tool functions to the instance

    # New: MCP, Dashboard, Curator, RAG, Secrets, Plugins
    global _mcp, _dashboard, _curator, _rag, _secrets, _webhook, _plugins
    _mcp       = MCPManager()
    _dashboard = DashboardServer()
    _curator   = Curator(router, skills)
    _secrets   = get_secrets()

    # Plugin SDK — load all plugins from ~/.operon/plugins/
    # Each plugin can register tools into the dispatch map and skills into SkillLoader.
    _plugins = get_plugin_manager()
    try:
        _n_plugins = _plugins.load_all()
        if _n_plugins > 0:
            _plugins.register_tools(tool_registry)
            # Merge plugin skills directly into the skill loader's in-memory list
            for _sk_name, _sk_text in _plugins.get_all_skills().items():
                skills._skills.append({
                    "name":        _sk_name,
                    "description": f"Plugin skill: {_sk_name}",
                    "path":        "(plugin)",
                    "enabled":     True,
                    "body":        _sk_text,
                })
            print(theme.dim(
                f"  Plugins loaded: {_n_plugins}  "
                f"({sum(len(p.tool_fns) for p in _plugins._plugins.values())} tools, "
                f"{sum(len(p.skill_texts) for p in _plugins._plugins.values())} skills)"
            ))
    except Exception as _pe:
        print(theme.dim(f"  [Plugins] Load error (non-fatal): {_pe}"))

    # Goal tracker — always available, persists across sessions
    _goals = GoalTracker()

    # Macro manager — lazy tool registry dependency
    _macros = MacroManager(tool_registry=tool_registry)

    # Retry policy manager — loads saved policies from ~/.operon/retry_policies.json
    _retry_mgr = RetryPolicyManager()

    # RAG pipeline — initialise lazily (first /rag command or explicit use)
    # Set _rag to None initially; it is created on first /rag command
    _rag = None

    # ── Phase 11: Vector Memory ───────────────────────────────────────────────
    _vector_mem = None
    if _VECTOR_MEMORY_AVAILABLE:
        try:
            _vector_mem = _get_vector_memory()
            # Register vector memory tools
            from core.vector_memory import _TOOL_DEFINITIONS as _VM_DEFS, _DISPATCH as _VM_DISPATCH
            for _tdef in _VM_DEFS:
                if _tdef["name"] not in tool_registry.tools:
                    tool_registry.tools[_tdef["name"]] = _VM_DISPATCH[_tdef["name"]]
                    _TOOL_DEFINITIONS.append(_tdef)
            print(theme.dim(f"  Vector memory: LanceDB ready  ({_vector_mem.count()} facts stored)"))
        except Exception as _ve:
            print(theme.dim(f"  [VectorMemory] init error (non-fatal): {_ve}"))

    # ── Phase 11: Obsidian Memory ─────────────────────────────────────────────
    _obsidian_mem = None
    if _OBSIDIAN_AVAILABLE:
        try:
            _obsidian_mem = _get_obsidian_memory()
            _obsidian_mem.start_auto_sync()
            from core.obsidian_memory import _TOOL_DEFINITIONS as _OB_DEFS, _DISPATCH as _OB_DISPATCH
            for _tdef in _OB_DEFS:
                if _tdef["name"] not in tool_registry.tools:
                    tool_registry.tools[_tdef["name"]] = _OB_DISPATCH[_tdef["name"]]
                    _TOOL_DEFINITIONS.append(_tdef)
            print(theme.dim(f"  Obsidian sync: {_obsidian_mem._vault.root}  (auto-sync every 20 min)"))
        except Exception as _oe:
            print(theme.dim(f"  [Obsidian] init error (non-fatal): {_oe}"))

    # ── Phase 11: Smart Model Router ─────────────────────────────────────────
    _smart_router = None
    if _SMART_ROUTER_AVAILABLE:
        try:
            _router_default = config.get("default_model", "hermes3:8b")
            _smart_router = SmartModelRouter(default_model=_router_default)
            print(theme.dim(f"  Smart router: {_smart_router.status()}"))
        except Exception as _sre:
            print(theme.dim(f"  [SmartRouter] init error (non-fatal): {_sre}"))

    # ── Phase 11: Skill Synthesizer ───────────────────────────────────────────
    _skill_synth = None
    if _SKILL_SYNTH_AVAILABLE:
        try:
            _skill_synth = _get_synthesizer()
            print(theme.dim(f"  Skill synthesizer: {_skill_synth.stats()['total_skills']} synthesized skills"))
        except Exception as _sse:
            print(theme.dim(f"  [SkillSynth] init error (non-fatal): {_sse}"))

    # ── Phase 11: Computer Use ────────────────────────────────────────────────
    _computer_use = None
    if _COMPUTER_USE_AVAILABLE:
        try:
            _computer_use = ComputerUse()
            from core.computer_use import _TOOL_DEFINITIONS as _CU_DEFS, _DISPATCH as _CU_DISPATCH
            for _tdef in _CU_DEFS:
                if _tdef["name"] not in tool_registry.tools:
                    tool_registry.tools[_tdef["name"]] = _CU_DISPATCH[_tdef["name"]]
                    _TOOL_DEFINITIONS.append(_tdef)
            print(theme.dim(f"  Computer use: ready (pyautogui + mss)"))
        except Exception as _cue:
            print(theme.dim(f"  [ComputerUse] init error (non-fatal): {_cue}"))

    # ── Phase 11: Slack tools ─────────────────────────────────────────────────
    try:
        from tools.slack_ops import _TOOL_DEFINITIONS as _SLACK_DEFS, _DISPATCH as _SLACK_DISPATCH
        for _tdef in _SLACK_DEFS:
            if _tdef["name"] not in tool_registry.tools:
                tool_registry.tools[_tdef["name"]] = _SLACK_DISPATCH[_tdef["name"]]
                _TOOL_DEFINITIONS.append(_tdef)
    except Exception as _slke:
        pass  # slack_ops already partially loaded via tools.registry

    # Inject any auto-reconnected MCP tools into the dispatch map and definitions
    if _mcp.status():
        injected = _mcp.inject_into_registry(tool_registry.tools, _TOOL_DEFINITIONS)
        if injected:
            print(theme.dim(f"  MCP tools injected: {injected}"))

    # Load context injection files from cwd
    context_inject = _load_context_files()
    if context_inject:
        found = [n for n in [".operon.md", "AGENTS.md", "CLAUDE.md", ".cursorrules"]
                 if Path(n).exists()]
        print(theme.dim(f"  Context files loaded: {', '.join(found)}"))

    # Report loaded skills
    if len(skills) > 0:
        print(theme.dim(f"  Skills loaded: {len(skills)}"))

    # Wire sub-agent runner (returns the final assistant response text)
    def _sub_agent_runner(prompt: str) -> str:
        sub_session = SessionManager()
        sub_session.add_message("user", prompt)
        run_agent_loop(
            session=sub_session, router=router, planner=planner,
            tool_registry=tool_registry, memory=memory, config=config,
            theme=theme, soul=soul, context_inject=context_inject,
            skills=skills, curator=_curator, knowledge=knowledge,
            cost_tracker=cost_tracker, semantic_mem=semantic_mem,
        )
        for m in reversed(sub_session._messages):
            if m["role"] == "assistant":
                return m["content"]
        return "(no response)"

    set_sub_agent_runner(_sub_agent_runner)
    scheduler.set_runner(_sub_agent_runner)
    scheduler.start()

    # ── Display banner (Hermes Agent-style with live tool/skill data) ──────────
    active_model = config.get("default_model", "gpt-4o")
    provider     = config.get("active_provider", "openai")
    mem_status   = "ON" if config.get("memory_enabled", True) else "OFF"

    _skill_list = skills.list_skills() if hasattr(skills, "list_skills") else []

    import datetime as _dt
    _session_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{session.get_title() or 'default'}"[:14]

    banner.display(
        model_name  = active_model,
        tool_count  = len(tool_registry.tools),
        skill_count = len(_skill_list),
        toolsets    = TOOLSETS,
        skills      = _skill_list,
        session_id  = _session_id,
        cwd         = str(__import__("pathlib").Path.cwd()),
    )

    # ── Compact post-banner status line ──────────────────────────────────────
    mcp_count    = len(_mcp.status()) if _mcp else 0
    plugin_count = len(_plugins) if _plugins else 0
    if mcp_count or plugin_count or not config.get("memory_enabled", True):
        print(theme.dim(
            f"  Memory › {mem_status}  ·  Knowledge › {len(knowledge)} facts  ·  "
            f"MCP › {mcp_count} servers  ·  Plugins › {plugin_count}  ·  "
            f"Approval › {'ON' if _APPROVAL_MODE else 'OFF'}"
        ))
    print()

    # ── Auto-start webhook if --webhook flag given ────────────────────────────
    if args.webhook and not (_webhook and _webhook.running):
        def _wh_runner_auto(prompt: str) -> str:
            sub_session = SessionManager()
            sub_session.add_message("user", prompt)
            run_agent_loop(
                session=sub_session, router=router, planner=planner,
                tool_registry=tool_registry, memory=memory, config=config,
                theme=theme, soul=soul, context_inject=context_inject,
                skills=skills, curator=_curator, knowledge=knowledge,
                cost_tracker=cost_tracker, semantic_mem=semantic_mem,
            )
            for m in reversed(sub_session._messages):
                if m["role"] == "assistant":
                    return m["content"]
            return "(no response)"

        _webhook = WebhookServer(
            agent_runner  = _wh_runner_auto,
            host          = args.host,
            port          = args.port,
            session_clear = lambda: session.clear(),
            tool_list     = lambda: list(tool_registry.tools.keys()),
            session_info  = lambda: {"model": config.get("default_model", "?")},
        )
        try:
            url = _webhook.start()
            print(theme.success(f"Webhook server auto-started → {url}"))
        except RuntimeError as e:
            print(theme.warning(f"Webhook auto-start failed: {e}"))

    # ── Batch mode — read prompts from file ───────────────────────────────────
    if args.batch:
        batch_path = Path(args.batch).expanduser()
        if not batch_path.exists():
            print(theme.error(f"Batch file not found: {args.batch}"))
            sys.exit(1)
        prompts = [
            line.strip() for line in batch_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        print(theme.info(f"Batch mode: {len(prompts)} prompts from {batch_path.name}"))
        for i, prompt in enumerate(prompts, 1):
            print(theme.dim(f"\n{'─'*60}"))
            print(theme.dim(f"  [{i}/{len(prompts)}] {prompt[:80]}"))
            print(theme.dim(f"{'─'*60}"))
            session.add_message("user", prompt)
            try:
                run_agent_loop(
                    session=session, router=router, planner=planner,
                    tool_registry=tool_registry, memory=memory, config=config,
                    theme=theme, soul=soul, context_inject=context_inject,
                    skills=skills, curator=_curator, knowledge=knowledge,
                    cost_tracker=cost_tracker, semantic_mem=semantic_mem,
                )
            except KeyboardInterrupt:
                print(theme.warning("\n  [Ctrl+C] Batch interrupted."))
                break
        print(theme.success("\nBatch complete."))
        if cost_tracker and cost_tracker._calls:
            print(theme.box(cost_tracker.session_report()))
        return

    # ── Single-prompt mode (--prompt / -p) ────────────────────────────────────
    if args.prompt:
        session.add_message("user", args.prompt)
        run_agent_loop(
            session=session, router=router, planner=planner,
            tool_registry=tool_registry, memory=memory, config=config,
            theme=theme, soul=soul, context_inject=context_inject,
            skills=skills, curator=_curator, knowledge=knowledge,
            cost_tracker=cost_tracker, semantic_mem=semantic_mem,
        )
        return

    # ── TUI — Hermes Agent-style input bar with context progress bar ─────────
    # Determine context window size for progress bar
    _ctx_window = 4096
    try:
        _mdl = config.get("default_model", active_model)
        if "claude" in _mdl:
            _ctx_window = 200_000
        elif "gpt-4" in _mdl:
            _ctx_window = 128_000
        elif "hermes3:8b" in _mdl or "qwen" in _mdl:
            _ctx_window = 8_192
    except Exception:
        pass

    _tui = OperonTUI(model_name=active_model, ctx_total=_ctx_window)
    _turn_count = 0

    # ── REPL ──────────────────────────────────────────────────────────────────
    while True:
        try:
            _tui.set_turn(_turn_count)
            _tui.set_mem_facts(len(knowledge))
            # Update context usage from current session token estimate
            _ctx_est = sum(
                len(str(m.get("content", ""))) // 4
                for m in session._messages
            ) if hasattr(session, "_messages") else 0
            _tui.set_ctx(_ctx_est, _ctx_window)
            user_input = _tui.prompt(
                placeholder='Try "hint:code write a function…" or /help'
            )
        except KeyboardInterrupt:
            print(theme.warning("\n  [Ctrl+C]  Type /exit to quit."))
            continue
        except EOFError:
            print(theme.info("\nGoodbye."))
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            _prev_model = config.get("default_model", active_model)
            handle_command(
                user_input, config, session, memory, theme,
                soul=soul, scheduler=scheduler, tool_registry=tool_registry,
                router=router, context_inject=context_inject, planner=planner,
                skills=skills, curator=_curator,
                cost_tracker=cost_tracker, semantic_mem=semantic_mem,
                knowledge=knowledge,
            )
            _new_model = config.get("default_model", active_model)
            if _new_model != _prev_model:
                _tui.set_model(_new_model)
                # Also try to update ctx window for the new model
                try:
                    if "claude" in _new_model:
                        _tui.set_ctx(_ctx_est, 200_000)
                    elif "gpt-4" in _new_model:
                        _tui.set_ctx(_ctx_est, 128_000)
                    else:
                        _tui.set_ctx(_ctx_est, 8_192)
                except Exception:
                    pass
            continue

        # Auto-detect credentials typed in chat and persist to knowledge base.
        user_input = _intercept_credentials(user_input, knowledge, theme)

        # Extract the intended recipient from this turn for email safety.
        _intended_recipient = _extract_recipient(user_input)

        session.add_message("user", user_input)
        _tui.set_status("thinking…")
        try:
            run_agent_loop(
                session=session, router=router, planner=planner,
                tool_registry=tool_registry, memory=memory, config=config,
                theme=theme, soul=soul, context_inject=context_inject,
                skills=skills, curator=_curator, knowledge=knowledge,
                cost_tracker=cost_tracker, semantic_mem=semantic_mem,
                intended_recipient=_intended_recipient,
            )
            _turn_count += 1
            # Sync cost into TUI status bar
            if cost_tracker and hasattr(cost_tracker, "_total_usd"):
                _tui.set_cost(cost_tracker._total_usd)
        except KeyboardInterrupt:
            print(theme.warning("\n  [Ctrl+C]  Agent interrupted. Type /exit to quit."))
            session.add_message("assistant", "[interrupted by user]")
        finally:
            _tui.clear_status()


if __name__ == "__main__":
    main()
