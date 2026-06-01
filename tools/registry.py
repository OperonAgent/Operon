"""
Operon Tool Registry.

All tools callable by the agent are registered here.
Dispatches JSON params → Python functions and returns normalised results.
"""

import json
from typing import Any, Callable, Optional

from tools.file_ops import (
    file_read, file_write, file_append, file_patch,
    file_delete, dir_list, file_exists, file_info,
)
from tools.shell_exec  import shell_exec
from tools.web_search  import duckduckgo_search, web_scrape, x_search
from tools.code_exec   import python_exec
from tools.http_client import http_request
from tools.file_search import file_search
from tools.email_draft import email_draft   # email_send is imported by email_draft internally
from tools.browser     import (
    browser_navigate, browser_snapshot, browser_screenshot,
    browser_click, browser_type, browser_scroll,
    browser_hover, browser_key, browser_select, browser_fill_form,
    browser_wait, browser_extract_text, browser_extract_links,
    browser_extract_tables, browser_evaluate,
    browser_new_tab, browser_list_tabs, browser_switch_tab,
    browser_go_back, browser_go_forward, browser_reload,
    browser_get_cookies, browser_set_cookie, browser_print_pdf,
    browser_network_log, browser_console_log, browser_close,
    # New browser tools
    browser_find_element, browser_check_captcha,
    browser_human_click, browser_human_type,
    browser_wait_for_element, browser_get_url,
)
from tools.pdf_ops import (
    pdf_create, pdf_extract_text, pdf_info, pdf_merge, pdf_split,
    pdf_rotate, pdf_watermark, pdf_encrypt, pdf_decrypt, pdf_extract_pages,
)
from tools.image_gen import (
    image_generate as dalle_generate,
    image_edit, image_variation, image_list_generated, image_describe,
)
from tools.video_gen import (
    video_generate, video_from_image, video_list_generated,
)
from tools.data_analysis import (
    data_load, data_save, data_convert, data_describe as data_describe_stats,
    data_query, data_groupby, data_clean, data_merge, data_pivot,
    data_anomalies, data_correlations, data_chart,
)
from tools.computer_use import computer_use
from tools.delegate     import delegate_task, delegate_batch
from tools.vision      import vision_analyze, image_generate, tts_speak
from tools.messaging   import telegram_send, clarify, todo
from tools.telegram_ops import (
    telegram_get_updates, telegram_edit_message, telegram_delete_message,
    telegram_pin_message, telegram_send_photo, telegram_send_document,
)
from tools.ssh_exec    import ssh_exec, ssh_upload, ssh_download
from tools.knowledge_ops import (
    knowledge_set, knowledge_get, knowledge_delete, knowledge_list,
    set_knowledge_base,
)
from tools.git_ops import (
    git_status, git_diff, git_log, git_add, git_commit,
    git_checkout, git_branch, git_stash,
)
from tools.discord_ops import (
    discord_send, discord_get_messages, discord_create_webhook,
)
from tools.slack_ops import (
    slack_send, slack_get_messages, slack_list_channels, slack_upload_file,
    slack_send_dm, slack_list_users, slack_add_reaction, slack_search_messages,
    slack_create_channel, slack_status, slack_delete_message, slack_get_thread,
    slack_update_message, slack_schedule_message, slack_pin_message,
    slack_set_topic, slack_build_blocks,
)
from tools.db_ops import (
    db_query, db_list_tables, db_describe_table, mongo_query,
)
from tools.voice_input import (
    voice_record_and_transcribe, voice_transcribe_file,
    voice_speak, voice_list_voices,
)
from tools.whatsapp_ops import (
    whatsapp_send, whatsapp_get_messages, whatsapp_status,
)
from tools.docker_exec import (
    docker_run, docker_run_code, docker_list_containers, docker_pull,
)
from tools.signal_ops import (
    signal_send, signal_receive, signal_list_groups,
)
from tools.matrix_ops import (
    matrix_send, matrix_get_messages, matrix_list_rooms,
)
from tools.irc_ops import (
    irc_send, irc_get_messages,
)
from tools.mattermost_ops import (
    mattermost_send, mattermost_get_messages, mattermost_list_channels,
)
from tools.teams_ops import (
    teams_send, teams_get_messages, teams_list_teams,
)
from tools.cloud_exec import (
    modal_run, modal_status, daytona_run, daytona_list_workspaces,
)
from core.macros import run_macro, macro_save, macro_delete, macro_list
from core.goal_tracker import (
    goal_set, goal_update, goal_list, goal_complete, goal_delete,
)
from tools.llm_task import llm_task, llm_classify, llm_summarize, llm_extract
from tools.apply_patch import apply_patch
from tools.github_ops import (
    github_repo_info, github_list_repos, github_list_issues, github_create_issue,
    github_list_prs, github_search_code, github_search_repos,
    github_get_file, github_create_gist, github_user_info, github_list_commits,
)


# ── Helper: normalise tool-definition param access ───────────────────────────
# Registry-native tools use a flat "params" dict.
# Phase 11 tools (computer_use, obsidian_memory, vector_memory…) use the
# Anthropic "input_schema" / "parameters" dict-of-properties format.
# Both surfaces need to work everywhere _TOOL_DEFINITIONS is iterated.

def _td_params(td: dict) -> dict:
    """Return a flat {param_name: description} dict for any tool definition format."""
    if "params" in td:
        return td["params"]
    for key in ("input_schema", "parameters"):
        if key in td and isinstance(td[key], dict):
            return td[key].get("properties", {})
    return {}


def _td_required(td: dict) -> set:
    """Return the set of required parameter names, if declared."""
    for key in ("input_schema", "parameters"):
        if key in td and isinstance(td[key], dict):
            req = td[key].get("required", [])
            if req:
                return set(req)
    return set()


# ── Tool descriptors (name, description, params schema) ──────────────────────

_TOOL_DEFINITIONS = [
    # ── File system ──────────────────────────────────────────────────────────
    {
        "name": "file_read",
        "description": "Read the full contents of a file and return them as a string.",
        "params": {
            "path":     "string — absolute or relative file path (required)",
            "encoding": "string — file encoding, default utf-8 (optional)",
        },
    },
    {
        "name": "file_write",
        "description": "Create or overwrite a file with the provided content. Parent directories are created automatically.",
        "params": {
            "path":    "string — file path (required)",
            "content": "string — full file content to write (required)",
        },
    },
    {
        "name": "file_append",
        "description": "Append content to the end of a file. Creates the file if it does not exist.",
        "params": {
            "path":    "string — file path (required)",
            "content": "string — text to append (required)",
        },
    },
    {
        "name": "file_patch",
        "description": "Find-and-replace: replaces the first occurrence of old_text with new_text inside a file.",
        "params": {
            "path":     "string — file path (required)",
            "old_text": "string — exact text to find (required)",
            "new_text": "string — replacement text (required)",
        },
    },
    {
        "name": "file_delete",
        "description": "Delete a file or directory (directories removed recursively).",
        "params": {
            "path": "string — file or directory path (required)",
        },
    },
    {
        "name": "dir_list",
        "description": "List the contents of a directory as an ASCII tree.",
        "params": {
            "path":      "string — directory path, default '.' (optional)",
            "max_depth": "integer — max recursion depth, default 3 (optional)",
        },
    },
    {
        "name": "file_exists",
        "description": "Check whether a file or directory exists at the given path.",
        "params": {
            "path": "string — path to check (required)",
        },
    },
    {
        "name": "file_info",
        "description": "Return metadata (size, modification time, permissions) for a file or directory.",
        "params": {
            "path": "string — file or directory path (required)",
        },
    },
    # ── Shell ─────────────────────────────────────────────────────────────────
    {
        "name": "shell_exec",
        "description": (
            "Execute a bash/shell command and capture stdout and stderr. "
            "Use for running scripts, installing packages, git operations, compiling code, etc."
        ),
        "params": {
            "command": "string — the shell command to run (required)",
            "cwd":     "string — working directory (optional)",
            "timeout": "integer — max seconds before killing, default 30 (optional)",
        },
    },
    # ── Web ───────────────────────────────────────────────────────────────────
    {
        "name": "duckduckgo_search",
        "description": "Search the web via DuckDuckGo and return titles, URLs, and snippets. No API key required.",
        "params": {
            "query":       "string — search query (required)",
            "max_results": "integer — number of results, default 6 (optional)",
        },
    },
    {
        "name": "web_scrape",
        "description": "Fetch a URL and extract its readable text content. Good for documentation, articles, GitHub READMEs.",
        "params": {
            "url":       "string — full URL to fetch (required)",
            "max_chars": "integer — max characters to return, default 8000 (optional)",
        },
    },
    # ── Code execution ────────────────────────────────────────────────────────
    {
        "name": "python_exec",
        "description": "Execute arbitrary Python code in a subprocess sandbox and capture stdout/stderr.",
        "params": {
            "code":    "string — Python source code to run (required)",
            "timeout": "integer — max seconds, default 30 (optional)",
            "cwd":     "string — working directory (optional)",
        },
    },
    # ── HTTP client ───────────────────────────────────────────────────────────
    {
        "name": "http_request",
        "description": "Make an HTTP request (GET/POST/PUT/PATCH/DELETE) to any URL. Supports JSON, headers, Bearer auth.",
        "params": {
            "url":          "string — target URL (required)",
            "method":       "string — HTTP verb, default GET (optional)",
            "headers":      "object — extra HTTP headers (optional)",
            "body":         "object — request body dict sent as JSON (optional)",
            "params":       "object — URL query params (optional)",
            "bearer_token": "string — Authorization Bearer token (optional)",
            "timeout":      "integer — seconds before timeout, default 20 (optional)",
        },
    },
    # ── File search ───────────────────────────────────────────────────────────
    {
        "name": "file_search",
        "description": "Search file contents recursively for a pattern (regex or plain text). Returns file paths, line numbers, and matches.",
        "params": {
            "pattern":        "string — regex or plain-text pattern (required)",
            "path":           "string — directory or file to search, default '.' (optional)",
            "recursive":      "boolean — recurse into subdirs, default true (optional)",
            "case_sensitive": "boolean — case-sensitive match, default false (optional)",
            "file_pattern":   "string — glob filter e.g. '*.py', default '*' (optional)",
            "max_results":    "integer — max matches, default 50 (optional)",
            "context_lines":  "integer — extra context lines per match, default 0 (optional)",
            "whole_word":     "boolean — match whole words only (grep -w), default false (optional)",
            "files_with_matches": "boolean — return only filenames that contain a match (grep -l), default false (optional)",
        },
    },
    # ── Email ─────────────────────────────────────────────────────────────────
    {
        "name": "email_draft",
        "description": (
            "Compose an email draft, show a formatted preview to the user, and send only after "
            "they approve it. ALWAYS use this — NEVER use email_send — when a user asks to send "
            "an email to ANY recipient. The 'to' field accepts any valid email address or multiple "
            "comma-separated addresses. Credentials are loaded from env vars or knowledge base "
            "automatically; NEVER pass sender_email or app_password as params. "
            "The 'body' MUST be a complete plain-text email matching exactly what the user asked "
            "for (correct recipient, correct topic, correct number of questions/items). "
            "Returns {approved, sent, recipients, feedback, cancelled}. "
            "If cancelled=true → STOP, do NOT redraft. "
            "If approved=false and feedback is non-empty → incorporate feedback and call email_draft "
            "again with a revised draft. "
            "NEVER call email_send before or after email_draft."
        ),
        "params": {
            "to":          "string — recipient address(es), comma-separated — any valid email (required)",
            "subject":     "string — your drafted subject line (required)",
            "body":        (
                "string — full email body as plain text prose (required). "
                "Must start with a greeting (e.g. 'Hi,'), contain all content the user requested "
                "(e.g. all 10 questions if the user said '10 questions'), and end with a sign-off. "
                "NEVER pass a JSON object, dict, or list — plain text only."
            ),
            "cc":          "string — CC recipients, comma-separated (optional)",
            "bcc":         "string — BCC recipients, comma-separated (optional)",
            "reply_to":    "string — Reply-To address (optional)",
            "attachments": "list — file paths to attach, e.g. [\"/path/to/file.pdf\"] (optional)",
        },
    },
    # email_send is intentionally NOT listed here — it is an internal helper
    # called only by email_draft after the user approves the preview.
    # Exposing it to the model would allow it to bypass the approval step
    # and pass credentials as plaintext params.
    # ── Browser automation (production Playwright CDP) ────────────────────────
    {
        "name": "browser_navigate",
        "description": "Navigate the browser to a URL. Returns page title and HTTP status. Requires Playwright.",
        "params": {
            "url":        "string — URL to navigate to (required)",
            "task_id":    "string — session ID for tab isolation (optional)",
            "wait_until": "string — 'domcontentloaded' | 'load' | 'networkidle' (optional)",
            "timeout":    "integer — ms before timeout, default 20000 (optional)",
        },
    },
    {
        "name": "browser_snapshot",
        "description": "Get the accessible text and structure of the current page as a readable outline (aria snapshot). Better than raw HTML.",
        "params": {
            "task_id":   "string — session ID (optional)",
            "max_chars": "integer — max characters to return, default 8000 (optional)",
        },
    },
    {
        "name": "browser_screenshot",
        "description": "Take a screenshot of the current browser page. Returns base64 PNG and saves to disk.",
        "params": {
            "task_id":   "string — session ID (optional)",
            "full_page": "boolean — capture full scrollable page, default false (optional)",
        },
    },
    {
        "name": "browser_click",
        "description": "Click an element on the current page using a CSS selector, text locator, or (x, y) coordinates.",
        "params": {
            "selector": "string — CSS selector or 'text=...' locator (optional if x/y given)",
            "x":        "number — page X coordinate for coordinate click (optional)",
            "y":        "number — page Y coordinate for coordinate click (optional)",
            "task_id":  "string — session ID (optional)",
            "timeout":  "integer — ms to wait for element, default 8000 (optional)",
        },
    },
    {
        "name": "browser_type",
        "description": "Type text into an input field on the current browser page.",
        "params": {
            "selector":    "string — CSS selector for the input (required)",
            "text":        "string — text to type (required)",
            "clear_first": "boolean — clear existing text first, default true (optional)",
            "task_id":     "string — session ID (optional)",
        },
    },
    {
        "name": "browser_scroll",
        "description": "Scroll the current browser page up or down.",
        "params": {
            "direction": "string — 'up' or 'down', default 'down' (optional)",
            "amount":    "integer — scroll amount in notches, default 3 (optional)",
            "task_id":   "string — session ID (optional)",
        },
    },
    {
        "name": "browser_hover",
        "description": "Hover the mouse over an element (useful for revealing dropdown menus).",
        "params": {
            "selector": "string — CSS selector or text='...' locator (required)",
            "task_id":  "string — session ID (optional)",
        },
    },
    {
        "name": "browser_key",
        "description": "Press a keyboard key or chord on the current page (e.g. 'Enter', 'Tab', 'Control+a').",
        "params": {
            "key":      "string — key name or chord, e.g. 'Enter', 'Escape', 'Control+a' (required)",
            "selector": "string — focus this element before pressing (optional)",
            "task_id":  "string — session ID (optional)",
        },
    },
    {
        "name": "browser_select",
        "description": "Select an option from a <select> dropdown element.",
        "params": {
            "selector": "string — CSS selector for the <select> (required)",
            "value":    "string — option value attribute to select (optional)",
            "label":    "string — option display text to select (optional)",
            "task_id":  "string — session ID (optional)",
        },
    },
    {
        "name": "browser_fill_form",
        "description": "Fill multiple form fields in one call. Each field dict must have 'selector' and 'value' keys.",
        "params": {
            "fields":  "list — list of {selector, value} dicts (required)",
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_wait",
        "description": "Wait for a CSS selector to appear or become visible, or just sleep for a duration.",
        "params": {
            "selector":   "string — CSS selector to wait for (optional — omit to sleep by duration)",
            "state":      "string — 'visible' | 'attached' | 'hidden', default 'visible' (optional)",
            "timeout_ms": "integer — max ms to wait, default 8000 (optional)",
            "sleep_ms":   "integer — unconditional sleep ms if no selector given (optional)",
            "task_id":    "string — session ID (optional)",
        },
    },
    {
        "name": "browser_extract_text",
        "description": "Extract all visible text from a CSS selector or the whole page.",
        "params": {
            "selector":  "string — CSS selector to extract from, default 'body' (optional)",
            "max_chars": "integer — max characters, default 6000 (optional)",
            "task_id":   "string — session ID (optional)",
        },
    },
    {
        "name": "browser_extract_links",
        "description": "Extract all hyperlinks (href + text) from the current page.",
        "params": {
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_extract_tables",
        "description": "Extract all HTML tables from the current page as lists of rows.",
        "params": {
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_evaluate",
        "description": "Execute arbitrary JavaScript in the browser page context and return the result.",
        "params": {
            "script":  "string — JavaScript expression or statement (required)",
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_new_tab",
        "description": "Open a new browser tab, optionally navigating to a URL.",
        "params": {
            "url":     "string — URL to open in the new tab (optional)",
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_list_tabs",
        "description": "List all open browser tabs with their index, title, and URL.",
        "params": {
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_switch_tab",
        "description": "Switch to a browser tab by its index (0-based).",
        "params": {
            "index":   "integer — tab index from browser_list_tabs (required)",
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_go_back",
        "description": "Navigate back one page in the browser history.",
        "params": {
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_go_forward",
        "description": "Navigate forward one page in the browser history.",
        "params": {
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_reload",
        "description": "Reload the current browser page.",
        "params": {
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_get_cookies",
        "description": "Get all cookies for the current browser session.",
        "params": {
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_set_cookie",
        "description": "Set a cookie in the current browser session.",
        "params": {
            "name":    "string — cookie name (required)",
            "value":   "string — cookie value (required)",
            "domain":  "string — cookie domain (optional)",
            "path":    "string — cookie path, default '/' (optional)",
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_print_pdf",
        "description": "Print the current page to a PDF file. Requires Playwright.",
        "params": {
            "path":    "string — output PDF file path (optional — auto-named if omitted)",
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_network_log",
        "description": "Return the most recent network requests captured during the browser session.",
        "params": {
            "last_n":  "integer — number of recent requests to return, default 20 (optional)",
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_console_log",
        "description": "Return browser console messages (console.log, errors, warnings) from the current session.",
        "params": {
            "last_n":  "integer — number of messages to return, default 20 (optional)",
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_close",
        "description": "Close the browser session and free all resources.",
        "params": {
            "task_id": "string — session ID to close (optional — closes all if omitted)",
        },
    },
    # ── Computer use (screen capture + mouse/keyboard) ────────────────────────
    {
        "name": "computer_use",
        "description": (
            "Control the local computer: capture the screen, click, type, press keys, "
            "scroll, drag, and manage application focus. Uses mss (screen capture) and "
            "pynput (mouse/keyboard). Requires user approval for potentially dangerous actions."
        ),
        "params": {
            "action": (
                "string — one of: capture, click, double_click, right_click, drag, scroll, "
                "type, key, wait, list_apps, focus_app (required)"
            ),
            "x":      "number — screen X coordinate for click/scroll (optional)",
            "y":      "number — screen Y coordinate for click/scroll (optional)",
            "text":   "string — text to type (required for 'type' action)",
            "keys":   "string — key name or chord e.g. 'cmd+c', 'enter' (required for 'key' action)",
            "app":    "string — application name to focus (required for 'focus_app' action)",
            "region": "object — {x,y,width,height} for partial screen capture (optional)",
            "to_x":   "number — drag destination X (required for 'drag' action)",
            "to_y":   "number — drag destination Y (required for 'drag' action)",
            "dx":     "integer — horizontal scroll delta (optional)",
            "dy":     "integer — vertical scroll delta (optional)",
            "duration": "number — drag duration in seconds, default 0.5 (optional)",
        },
    },
    # ── Sub-agent delegation ──────────────────────────────────────────────────
    {
        "name": "delegate_task",
        "description": (
            "Spawn a focused sub-agent to complete a specific task in isolation. "
            "The sub-agent runs with its own context (no parent history leaked), uses a "
            "restricted toolset, and returns a concise result. Use for parallelisable or "
            "sandboxed sub-tasks. Cannot be nested (sub-agents cannot delegate further)."
        ),
        "params": {
            "task":    "string — exact task description for the sub-agent to complete (required)",
            "toolset": "string — 'core' | 'coding' | 'research' | 'data' | 'devops' (optional, default 'core')",
            "model":   "string — override model for the sub-agent (optional)",
            "context": "string — additional context or constraints to pass to the sub-agent (optional)",
            "timeout": "integer — max seconds to wait, default 300 (optional)",
        },
    },
    {
        "name": "delegate_batch",
        "description": (
            "Spawn multiple sub-agents in parallel, each handling a different task. "
            "Returns results for all tasks. Max 5 concurrent sub-agents. "
            "Cannot be nested."
        ),
        "params": {
            "tasks":          "list — list of task dicts, each with 'task' (str) and optionally 'context', 'toolset', 'id' (required)",
            "toolset":        "string — default toolset for all tasks (optional)",
            "model":          "string — override model for all tasks (optional)",
            "max_concurrent": "integer — max parallel sub-agents, 1-5, default 3 (optional)",
            "timeout":        "integer — per-task timeout in seconds, default 300 (optional)",
        },
    },
    # ── Vision ────────────────────────────────────────────────────────────────
    {
        "name": "vision_analyze",
        "description": (
            "Analyze or query an image using a vision-capable model (GPT-4o or Claude). "
            "Provide image_path for a local file or image_url for a public URL."
        ),
        "params": {
            "image_path": "string — local file path to the image (optional if image_url given)",
            "image_url":  "string — public URL of the image (optional if image_path given)",
            "prompt":     "string — question or instruction about the image, default 'Describe this image' (optional)",
            "provider":   "string — 'openai' | 'anthropic' | 'auto', default 'auto' (optional)",
        },
    },
    {
        "name": "image_generate",
        "description": "Generate an image from a text prompt using DALL-E 3. Saves to disk and returns the file path.",
        "params": {
            "prompt":    "string — image description (required)",
            "size":      "string — '1024x1024' | '1792x1024' | '1024x1792', default '1024x1024' (optional)",
            "quality":   "string — 'standard' | 'hd', default 'standard' (optional)",
            "save_path": "string — output file path (optional, defaults to ~/Desktop/operon_img_<ts>.png)",
        },
    },
    {
        "name": "video_generate",
        "description": (
            "Generate a short video from a text prompt (text-to-video). Uses Replicate "
            "or Luma depending on which API key is set. Returns a video URL + local path."
        ),
        "params": {
            "prompt":   "string — description of the video (required)",
            "provider": "string — 'auto' | 'replicate' | 'luma', default 'auto' (optional)",
            "duration": "int — target length in seconds, default 4 (optional)",
            "fps":      "int — frames per second hint, default 24 (optional)",
        },
    },
    {
        "name": "video_from_image",
        "description": (
            "Animate a still image into a short video (image-to-video). Needs LUMA_API_KEY "
            "or REPLICATE_API_TOKEN."
        ),
        "params": {
            "image_url": "string — public URL of the source image (required)",
            "prompt":    "string — optional motion/scene guidance (optional)",
            "provider":  "string — 'auto' | 'luma' | 'replicate', default 'auto' (optional)",
        },
    },
    {
        "name": "video_list_generated",
        "description": "List previously generated videos saved under ~/.operon/generated/video.",
        "params": {
            "limit": "int — max results, default 20 (optional)",
        },
    },
    {
        "name": "tts_speak",
        "description": (
            "Convert text to speech. Uses OpenAI TTS (voices: alloy, echo, fable, onyx, nova, shimmer) "
            "if an API key is available, otherwise falls back to macOS system TTS."
        ),
        "params": {
            "text":      "string — text to speak (required)",
            "voice":     "string — voice name, default 'alloy' (optional)",
            "save_path": "string — where to save the audio file (optional)",
            "play":      "boolean — play the audio immediately, default true (optional)",
        },
    },
    # ── Messaging ─────────────────────────────────────────────────────────────
    {
        "name": "telegram_send",
        "description": (
            "Send a message to a Telegram chat. Requires a bot token configured via /setup → Telegram "
            "or the TELEGRAM_BOT_TOKEN environment variable."
        ),
        "params": {
            "chat_id":    "string — Telegram chat ID or @username (required)",
            "text":       "string — message body, Markdown supported (required)",
            "parse_mode": "string — 'Markdown' | 'HTML' | '', default 'Markdown' (optional)",
        },
    },
    {
        "name": "telegram_get_updates",
        "description": "Fetch recent inbound Telegram updates (messages). Requires TELEGRAM_BOT_TOKEN.",
        "params": {
            "limit": "integer — max updates to fetch (optional, default 10)",
        },
    },
    {
        "name": "telegram_edit_message",
        "description": "Edit the text of a message the bot previously sent. Requires TELEGRAM_BOT_TOKEN.",
        "params": {
            "chat_id":    "string — chat ID (optional — auto-read from TELEGRAM_CHAT_ID)",
            "message_id": "integer — id of the message to edit (required)",
            "text":       "string — new message text (required)",
        },
    },
    {
        "name": "telegram_delete_message",
        "description": "Delete a Telegram message by id. Requires TELEGRAM_BOT_TOKEN.",
        "params": {
            "chat_id":    "string — chat ID (optional — auto-read from TELEGRAM_CHAT_ID)",
            "message_id": "integer — id of the message to delete (required)",
        },
    },
    {
        "name": "telegram_pin_message",
        "description": "Pin a message in a Telegram chat. Requires TELEGRAM_BOT_TOKEN.",
        "params": {
            "chat_id":    "string — chat ID (optional — auto-read from TELEGRAM_CHAT_ID)",
            "message_id": "integer — id of the message to pin (required)",
            "notify":     "boolean — notify chat members (optional, default false)",
        },
    },
    {
        "name": "telegram_send_photo",
        "description": "Send a photo to a Telegram chat by URL or file_id. Requires TELEGRAM_BOT_TOKEN.",
        "params": {
            "chat_id": "string — chat ID (optional — auto-read from TELEGRAM_CHAT_ID)",
            "photo":   "string — image URL or Telegram file_id (required)",
            "caption": "string — optional caption text",
        },
    },
    {
        "name": "telegram_send_document",
        "description": "Send a document to a Telegram chat by local path, URL, or file_id. Requires TELEGRAM_BOT_TOKEN.",
        "params": {
            "chat_id":  "string — chat ID (optional — auto-read from TELEGRAM_CHAT_ID)",
            "document": "string — local file path, URL, or file_id (required)",
            "filename": "string — filename to present (optional, default 'file.txt')",
            "caption":  "string — optional caption text",
        },
    },
    {
        "name": "clarify",
        "description": (
            "Ask the user ONE specific blocking question when a required piece of information "
            "is missing and cannot be inferred. Examples of GOOD use: 'Which server hostname?', "
            "'What filename should I write to?', 'Which of these three options do you want?'. "
            "NEVER use clarify for: greetings, casual chat, or simple questions with obvious answers. "
            "NEVER use clarify to confirm or verify a tool result — the tool result shows "
            "success/failure directly; act on it. NEVER ask 'did it work?', 'can you confirm?', "
            "'was the email sent?' after a successful tool call. "
            "NEVER use clarify when an email tool returns an error — email_draft handles "
            "credential setup interactively on its own; just call email_draft once and wait. "
            "The question must be directed AT THE USER to get info you cannot get any other way."
        ),
        "params": {
            "question": "string — a specific question for the USER (required)",
        },
    },
    {
        "name": "todo",
        "description": "Manage a session-scoped task list. Use to track sub-tasks during complex multi-step work.",
        "params": {
            "action": "string — 'add' | 'list' | 'complete' | 'remove' | 'clear' (required)",
            "item":   "string — task text for 'add' (required for add)",
            "index":  "integer — 1-based task index for 'complete' or 'remove' (required for those actions)",
        },
    },
    # ── X / Twitter search ────────────────────────────────────────────────────
    {
        "name": "x_search",
        "description": (
            "Search X (Twitter/social media) for posts matching a query. "
            "Use this ONLY for social media posts. "
            "For general web/Google searches, use duckduckgo_search instead. "
            "Tries public Nitter instances, falls back to DuckDuckGo site:x.com. "
            "No API key required."
        ),
        "params": {
            "query":       "string — search query (required)",
            "max_results": "integer — number of results, default 8 (optional)",
        },
    },
    # ── SSH remote execution ──────────────────────────────────────────────────
    {
        "name": "ssh_exec",
        "description": (
            "Execute a shell command on a remote host over SSH. "
            "Uses paramiko if installed, otherwise falls back to the system ssh binary. "
            "Connections are reused within a session. Supports password and key-based auth."
        ),
        "params": {
            "host":     "string — hostname or IP address (required)",
            "command":  "string — shell command to run (required)",
            "port":     "integer — SSH port, default 22 (optional)",
            "user":     "string — SSH username, default current user (optional)",
            "password": "string — SSH password (optional, prefer key auth)",
            "key_path": "string — path to private key file, e.g. ~/.ssh/id_rsa (optional)",
            "timeout":  "integer — seconds before timeout, default 30 (optional)",
            "cwd":      "string — remote working directory (optional)",
        },
    },
    {
        "name": "ssh_upload",
        "description": "Upload a local file to a remote host via SFTP/SCP.",
        "params": {
            "host":        "string — hostname or IP address (required)",
            "local_path":  "string — path to the local file to upload (required)",
            "remote_path": "string — destination path on the remote host (required)",
            "port":        "integer — SSH port, default 22 (optional)",
            "user":        "string — SSH username (optional)",
            "password":    "string — SSH password (optional)",
            "key_path":    "string — path to private key file (optional)",
        },
    },
    {
        "name": "ssh_download",
        "description": "Download a file from a remote host to local disk via SFTP/SCP.",
        "params": {
            "host":        "string — hostname or IP address (required)",
            "remote_path": "string — path on the remote host to download (required)",
            "local_path":  "string — where to save the file locally (required)",
            "port":        "integer — SSH port, default 22 (optional)",
            "user":        "string — SSH username (optional)",
            "password":    "string — SSH password (optional)",
            "key_path":    "string — path to private key file (optional)",
        },
    },
    # ── Git version control ───────────────────────────────────────────────────
    {
        "name": "git_status",
        "description": "Show the git working tree status (staged, unstaged, untracked files and current branch).",
        "params": {"cwd": "string — repo directory (optional, defaults to current dir)"},
    },
    {
        "name": "git_diff",
        "description": "Show changes in the working tree or staged area as a unified diff.",
        "params": {
            "path":   "string — file or directory to diff (optional)",
            "staged": "boolean — if true, show staged (--cached) diff (optional)",
            "cwd":    "string — repo directory (optional)",
        },
    },
    {
        "name": "git_log",
        "description": "Show recent commit history.",
        "params": {
            "n":       "integer — number of commits to show, default 10 (optional)",
            "oneline": "boolean — compact one-line format, default true (optional)",
            "cwd":     "string — repo directory (optional)",
        },
    },
    {
        "name": "git_add",
        "description": "Stage file(s) for the next commit.",
        "params": {
            "paths": "string — space-separated file paths or '.' for all (required)",
            "cwd":   "string — repo directory (optional)",
        },
    },
    {
        "name": "git_commit",
        "description": "Commit staged changes with a message.",
        "params": {
            "message": "string — commit message (required)",
            "cwd":     "string — repo directory (optional)",
        },
    },
    {
        "name": "git_checkout",
        "description": "Switch to a branch. Omit branch to list all branches.",
        "params": {
            "branch": "string — branch name to switch to (optional — omit to list)",
            "create": "boolean — create the branch if it doesn't exist (optional)",
            "cwd":    "string — repo directory (optional)",
        },
    },
    {
        "name": "git_branch",
        "description": "List, create, or delete git branches.",
        "params": {
            "name":   "string — branch name to create or delete (optional — omit to list)",
            "delete": "boolean — delete the named branch (optional)",
            "cwd":    "string — repo directory (optional)",
        },
    },
    {
        "name": "git_stash",
        "description": "Stash or restore uncommitted changes.",
        "params": {
            "action":  "string — one of: push, pop, list, drop (default: push)",
            "message": "string — stash description for push (optional)",
            "cwd":     "string — repo directory (optional)",
        },
    },
    # ── Permanent knowledge store ─────────────────────────────────────────────
    {
        "name": "knowledge_set",
        "description": (
            "Save an important fact to PERMANENT memory — persists across ALL future sessions. "
            "Use this proactively when the user shares: their name, email, preferences, project "
            "paths, API base URLs, coding style, timezone, or any long-lived setting. "
            "Facts stored here are injected into every system prompt automatically. "
            "ALWAYS call this when you learn something important about the user or their environment."
        ),
        "params": {
            "key":   "string — short snake_case key e.g. 'user_name', 'project_path' (required)",
            "value": "string — the value to remember permanently (required)",
        },
    },
    {
        "name": "knowledge_get",
        "description": "Retrieve a specific permanent fact by key. Returns the stored value.",
        "params": {
            "key": "string — the key to look up (required)",
        },
    },
    {
        "name": "knowledge_delete",
        "description": "Delete a single permanent fact by key.",
        "params": {
            "key": "string — the key to delete (required)",
        },
    },
    {
        "name": "knowledge_list",
        "description": "List all currently stored permanent facts.",
        "params": {},
    },
    # ── Discord ───────────────────────────────────────────────────────────────
    {
        "name": "discord_send",
        "description": (
            "Send a message to a Discord channel via webhook or bot API. "
            "Set DISCORD_WEBHOOK_URL env var for webhook mode (no bot required), "
            "or DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID for bot mode."
        ),
        "params": {
            "message":     "string — message text (required)",
            "webhook_url": "string — Discord webhook URL (optional — auto-read from DISCORD_WEBHOOK_URL)",
            "username":    "string — display name override (optional)",
            "embed_title": "string — wrap message in a rich embed with this title (optional)",
            "channel_id":  "string — channel snowflake ID for bot mode (optional)",
        },
    },
    {
        "name": "discord_get_messages",
        "description": "Fetch recent messages from a Discord channel. Requires DISCORD_BOT_TOKEN.",
        "params": {
            "channel_id": "string — channel ID (optional — auto-read from DISCORD_CHANNEL_ID)",
            "limit":      "integer — number of messages (optional, default 10)",
        },
    },
    {
        "name": "discord_create_webhook",
        "description": "Create a Discord webhook for a channel and return its URL.",
        "params": {
            "channel_id": "string — channel ID (required)",
            "name":       "string — webhook display name (optional, default 'Operon')",
        },
    },
    # ── Slack ─────────────────────────────────────────────────────────────────
    {
        "name": "slack_send",
        "description": (
            "Send a message to a Slack channel. "
            "Set SLACK_WEBHOOK_URL for webhook mode, or SLACK_BOT_TOKEN for full bot API."
        ),
        "params": {
            "message":     "string — message text (required)",
            "channel":     "string — channel name or ID e.g. '#general' (optional)",
            "webhook_url": "string — Slack incoming webhook URL (optional — auto-read from SLACK_WEBHOOK_URL)",
            "thread_ts":   "string — reply in thread (optional — provide parent message timestamp)",
        },
    },
    {
        "name": "slack_get_messages",
        "description": "Fetch recent messages from a Slack channel. Requires SLACK_BOT_TOKEN.",
        "params": {
            "channel": "string — channel name or ID (optional — auto-read from SLACK_DEFAULT_CHANNEL)",
            "limit":   "integer — number of messages (optional, default 10)",
        },
    },
    {
        "name": "slack_list_channels",
        "description": "List all Slack channels the bot has access to. Requires SLACK_BOT_TOKEN.",
        "params": {},
    },
    {
        "name": "slack_upload_file",
        "description": "Upload a local file to a Slack channel. Requires SLACK_BOT_TOKEN.",
        "params": {
            "file_path": "string — local file path to upload (required)",
            "channel":   "string — target channel (optional)",
            "title":     "string — file title in Slack (optional)",
            "message":   "string — message to accompany the file (optional)",
        },
    },
    {
        "name": "slack_send_dm",
        "description": "Send a direct message to a Slack user by user ID (U...). Requires SLACK_BOT_TOKEN.",
        "params": {
            "user_id": "string — Slack user ID, e.g. U0123456 (required)",
            "text":    "string — message text (required)",
        },
    },
    {
        "name": "slack_get_thread",
        "description": "Read a full Slack thread: the parent message plus all replies. Requires SLACK_BOT_TOKEN.",
        "params": {
            "channel":   "string — channel ID (required)",
            "thread_ts": "string — parent message timestamp / thread root (required)",
            "limit":     "integer — max replies, 1-200 (optional, default 50)",
        },
    },
    {
        "name": "slack_update_message",
        "description": "Edit a message the bot previously posted (chat.update). Requires SLACK_BOT_TOKEN.",
        "params": {
            "channel": "string — channel ID where the message lives (required)",
            "ts":      "string — timestamp of the message to edit (required)",
            "text":    "string — new message text (optional)",
            "blocks":  "array — new Block Kit blocks (optional)",
        },
    },
    {
        "name": "slack_schedule_message",
        "description": "Schedule a message for future delivery at a Unix epoch timestamp. Requires SLACK_BOT_TOKEN.",
        "params": {
            "channel": "string — channel ID or name (required)",
            "text":    "string — message text (optional if blocks given)",
            "post_at": "integer — Unix epoch seconds for delivery, must be future (required)",
            "blocks":  "array — Block Kit blocks (optional)",
        },
    },
    {
        "name": "slack_add_reaction",
        "description": "Add a reaction emoji to a Slack message. Requires SLACK_BOT_TOKEN.",
        "params": {
            "channel": "string — channel ID (required)",
            "ts":      "string — message timestamp (required)",
            "emoji":   "string — emoji name without colons, e.g. 'thumbsup' (required)",
        },
    },
    {
        "name": "slack_pin_message",
        "description": "Pin or unpin a message in a Slack channel. Requires SLACK_BOT_TOKEN.",
        "params": {
            "channel": "string — channel ID (required)",
            "ts":      "string — message timestamp (required)",
            "unpin":   "boolean — set true to remove the pin (optional, default false)",
        },
    },
    {
        "name": "slack_set_topic",
        "description": "Set the topic of a Slack channel. Requires SLACK_BOT_TOKEN.",
        "params": {
            "channel": "string — channel ID (required)",
            "topic":   "string — the new channel topic (required)",
        },
    },
    {
        "name": "slack_build_blocks",
        "description": "Compose a Slack Block Kit payload from simple parts (header/body/fields/context). Returns blocks to pass to slack_send.",
        "params": {
            "title":   "string — header text (optional)",
            "body":    "string — main markdown body (optional)",
            "fields":  "object — {label: value} rendered as a field grid (optional)",
            "context": "string — small footnote line (optional)",
        },
    },
    {
        "name": "slack_search_messages",
        "description": "Search messages across the Slack workspace. Requires SLACK_BOT_TOKEN with search:read.",
        "params": {
            "query": "string — search query (required)",
            "count": "integer — max results (optional, default 10)",
        },
    },
    {
        "name": "slack_list_users",
        "description": "List all active Slack workspace members. Requires SLACK_BOT_TOKEN.",
        "params": {
            "limit": "integer — max users (optional, default 100)",
        },
    },
    {
        "name": "slack_create_channel",
        "description": "Create a new Slack channel. Requires SLACK_BOT_TOKEN with channels:manage.",
        "params": {
            "name":       "string — channel name (required)",
            "is_private": "boolean — create a private channel (optional, default false)",
        },
    },
    {
        "name": "slack_delete_message",
        "description": "Delete a message previously posted by the bot. Requires SLACK_BOT_TOKEN.",
        "params": {
            "channel": "string — channel ID (required)",
            "ts":      "string — message timestamp (required)",
        },
    },
    {
        "name": "slack_status",
        "description": "Check Slack connection status and authentication (auth.test).",
        "params": {},
    },
    # ── Database ──────────────────────────────────────────────────────────────
    {
        "name": "db_query",
        "description": (
            "Execute a SQL query against SQLite, PostgreSQL, or MySQL. "
            "Returns columns, rows, rowcount, and elapsed time. "
            "Connection URL auto-read from DATABASE_URL env var, or pass db_url directly."
        ),
        "params": {
            "query":   "string — SQL query to execute (required)",
            "db_url":  "string — connection URL or SQLite file path (optional — auto-read from DATABASE_URL)",
            "backend": "string — 'sqlite' | 'postgresql' | 'mysql' — auto-detected from URL (optional)",
            "params":  "array — parameterised query values e.g. ['Alice', 30] (optional)",
            "timeout": "integer — query timeout in seconds, default 30 (optional)",
        },
    },
    {
        "name": "db_list_tables",
        "description": "List all tables in a database (SQLite, PostgreSQL, or MySQL).",
        "params": {
            "db_url":  "string — connection URL (optional — auto-read from DATABASE_URL)",
            "backend": "string — 'sqlite' | 'postgresql' | 'mysql' (optional)",
        },
    },
    {
        "name": "db_describe_table",
        "description": "Return column definitions and types for a database table.",
        "params": {
            "table":   "string — table name (required)",
            "db_url":  "string — connection URL (optional — auto-read from DATABASE_URL)",
            "backend": "string — 'sqlite' | 'postgresql' | 'mysql' (optional)",
        },
    },
    {
        "name": "mongo_query",
        "description": (
            "Query a MongoDB collection. Supports find, count, and aggregate operations. "
            "Requires pymongo: pip install pymongo. "
            "Connection URL auto-read from MONGODB_URL env var."
        ),
        "params": {
            "collection": "string — collection name (required)",
            "operation":  "string — 'find' | 'count' | 'aggregate' (optional, default 'find')",
            "filter":     "object — filter document or aggregation pipeline (optional)",
            "limit":      "integer — max documents for find (optional, default 20)",
            "db_url":     "string — MongoDB URL (optional — auto-read from MONGODB_URL)",
        },
    },
    # ── Voice ─────────────────────────────────────────────────────────────────
    {
        "name": "voice_record_and_transcribe",
        "description": (
            "Record audio from the microphone and transcribe it to text using Whisper. "
            "Requires sounddevice (pip install sounddevice) for recording and either "
            "openai-whisper (local) or OPENAI_API_KEY for transcription."
        ),
        "params": {
            "duration":   "integer — recording duration in seconds (optional, default 5)",
            "model":      "string — local Whisper model: tiny/base/small/medium/large (optional, default 'base')",
            "language":   "string — language hint e.g. 'en', 'fr' (optional)",
            "use_api":    "boolean — use OpenAI API instead of local model (optional)",
        },
    },
    {
        "name": "voice_transcribe_file",
        "description": "Transcribe an existing audio file (WAV, MP3, M4A, OGG) to text using Whisper.",
        "params": {
            "file_path": "string — path to the audio file (required)",
            "model":     "string — Whisper model size: tiny/base/small/medium/large (optional, default 'base')",
            "language":  "string — language hint (optional)",
            "use_api":   "boolean — use OpenAI API (optional)",
        },
    },
    {
        "name": "voice_speak",
        "description": (
            "Convert text to speech using pyttsx3 (offline), macOS 'say', or OpenAI TTS. "
            "Plays audio immediately. Install pyttsx3 for offline use: pip install pyttsx3."
        ),
        "params": {
            "text":      "string — text to speak (required)",
            "voice":     "string — voice name (optional, engine-specific)",
            "rate":      "integer — speech rate in WPM (optional, default 175)",
            "engine":    "string — 'auto' | 'pyttsx3' | 'say' | 'openai' (optional, default 'auto')",
            "save_path": "string — save audio to file instead of playing (optional)",
        },
    },
    {
        "name": "voice_list_voices",
        "description": "List all available TTS voices on this system.",
        "params": {},
    },
    # ── WhatsApp ──────────────────────────────────────────────────────────────
    {
        "name": "whatsapp_send",
        "description": (
            "Send a WhatsApp message via Twilio. "
            "Requires TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables. "
            "Supports text messages and media (images/videos via URL)."
        ),
        "params": {
            "message":   "string — message text (required unless media_url given)",
            "to":        "string — recipient number e.g. 'whatsapp:+1234567890' (optional — auto-read from TWILIO_WHATSAPP_TO)",
            "media_url": "string — public URL of image/video to send (optional)",
        },
    },
    {
        "name": "whatsapp_get_messages",
        "description": "Retrieve recent WhatsApp messages from Twilio. Requires Twilio credentials.",
        "params": {
            "limit": "integer — number of messages to return (optional, default 10)",
            "to":    "string — filter by recipient number (optional)",
        },
    },
    {
        "name": "whatsapp_status",
        "description": "Check the delivery status of a WhatsApp message by its Twilio SID.",
        "params": {
            "message_sid": "string — Twilio message SID from whatsapp_send (required)",
        },
    },
    # ── Docker sandboxed execution ─────────────────────────────────────────────
    {
        "name": "docker_run",
        "description": (
            "Run a shell command inside a sandboxed Docker container. "
            "Safer than shell_exec for untrusted code — isolated filesystem and network. "
            "Requires Docker to be running."
        ),
        "params": {
            "command": "string — shell command to run (required)",
            "image":   "string — Docker image (optional, default 'python:3.12-slim')",
            "env":     "object — environment variables dict (optional)",
            "timeout": "integer — seconds before killing container (optional, default 30)",
            "workdir": "string — working directory inside container (optional)",
        },
    },
    {
        "name": "docker_run_code",
        "description": (
            "Execute code in a sandboxed Docker container. "
            "Supports Python, Node.js, Bash, Ruby. "
            "Better than python_exec for security-sensitive tasks."
        ),
        "params": {
            "code":     "string — source code to execute (required)",
            "language": "string — 'python' | 'node' | 'bash' | 'ruby' (optional, default 'python')",
            "image":    "string — override Docker image (optional)",
            "timeout":  "integer — seconds before timeout (optional, default 30)",
        },
    },
    {
        "name": "docker_list_containers",
        "description": "List Docker containers running on the host.",
        "params": {
            "running_only": "boolean — only show running containers, default false (optional)",
        },
    },
    {
        "name": "docker_pull",
        "description": "Pull a Docker image from Docker Hub.",
        "params": {
            "image": "string — Docker image name e.g. 'python:3.12-slim' (required)",
        },
    },
    # ── Signal ────────────────────────────────────────────────────────────────
    {
        "name": "signal_send",
        "description": "Send a message via Signal using signal-cli. Requires signal-cli installed and SIGNAL_NUMBER env var.",
        "params": {
            "message":    "string — message text (required)",
            "recipient":  "string — recipient phone number e.g. '+15551234567' (optional if SIGNAL_RECIPIENT set)",
            "group_id":   "string — Signal group ID for group messages (optional)",
            "attachment": "string — file path to attach (optional)",
        },
    },
    {
        "name": "signal_receive",
        "description": "Receive pending Signal messages via signal-cli.",
        "params": {
            "limit":   "integer — max messages to return, default 10 (optional)",
            "timeout": "integer — receive timeout seconds, default 5 (optional)",
        },
    },
    {
        "name": "signal_list_groups",
        "description": "List all Signal groups the account belongs to.",
        "params": {},
    },
    # ── Matrix ────────────────────────────────────────────────────────────────
    {
        "name": "matrix_send",
        "description": "Send a message to a Matrix room. Requires MATRIX_HOMESERVER and MATRIX_ACCESS_TOKEN (or MATRIX_USER + MATRIX_PASSWORD).",
        "params": {
            "message": "string — message text (required)",
            "room_id": "string — Matrix room ID e.g. '!abc:matrix.org' (optional if MATRIX_ROOM_ID set)",
            "msgtype": "string — Matrix message type, default 'm.text' (optional)",
        },
    },
    {
        "name": "matrix_get_messages",
        "description": "Retrieve recent messages from a Matrix room.",
        "params": {
            "room_id": "string — Matrix room ID (optional if MATRIX_ROOM_ID set)",
            "limit":   "integer — max messages to return, default 10 (optional)",
        },
    },
    {
        "name": "matrix_list_rooms",
        "description": "List Matrix rooms the account has joined.",
        "params": {},
    },
    # ── IRC ───────────────────────────────────────────────────────────────────
    {
        "name": "irc_send",
        "description": "Send a message to an IRC channel. Uses stdlib socket — no extra packages needed. Reads IRC_SERVER, IRC_PORT, IRC_NICK, IRC_CHANNEL.",
        "params": {
            "message": "string — message text (required)",
            "channel": "string — IRC channel e.g. '#general' (optional if IRC_CHANNEL set)",
            "server":  "string — IRC server hostname (optional if IRC_SERVER set)",
            "port":    "integer — IRC server port, default 6667 (optional)",
            "nick":    "string — IRC nickname (optional if IRC_NICK set)",
            "password": "string — NickServ password (optional)",
        },
    },
    {
        "name": "irc_get_messages",
        "description": "Connect to an IRC channel and collect messages for a short window.",
        "params": {
            "channel": "string — IRC channel (optional if IRC_CHANNEL set)",
            "server":  "string — IRC server hostname (optional if IRC_SERVER set)",
            "port":    "integer — IRC port (optional)",
            "nick":    "string — IRC nickname (optional)",
            "wait_s":  "integer — seconds to listen, default 5 (optional)",
        },
    },
    # ── Mattermost ────────────────────────────────────────────────────────────
    {
        "name": "mattermost_send",
        "description": "Post a message to a Mattermost channel. Requires MATTERMOST_URL and MATTERMOST_TOKEN.",
        "params": {
            "message":    "string — message text in Markdown (required)",
            "channel":    "string — channel name e.g. 'town-square' (optional)",
            "channel_id": "string — direct channel ID, faster than name lookup (optional)",
            "root_id":    "string — post ID to reply in a thread (optional)",
        },
    },
    {
        "name": "mattermost_get_messages",
        "description": "Retrieve recent posts from a Mattermost channel.",
        "params": {
            "channel":    "string — channel name (optional)",
            "channel_id": "string — direct channel ID (optional)",
            "limit":      "integer — number of posts, default 10 (optional)",
        },
    },
    {
        "name": "mattermost_list_channels",
        "description": "List public channels on the Mattermost instance.",
        "params": {
            "team": "string — team name or ID to filter (optional)",
        },
    },
    # ── Microsoft Teams ───────────────────────────────────────────────────────
    {
        "name": "teams_send",
        "description": "Send a message to a Microsoft Teams channel via Incoming Webhook. Requires TEAMS_WEBHOOK_URL.",
        "params": {
            "message":     "string — message body in Markdown (required)",
            "webhook_url": "string — Teams webhook URL (optional if TEAMS_WEBHOOK_URL set)",
            "title":       "string — card title (optional)",
            "color":       "string — accent colour hex, default '#0078D7' (optional)",
            "facts":       "list — list of {name, value} dicts for a facts card (optional)",
        },
    },
    {
        "name": "teams_get_messages",
        "description": "Retrieve recent messages from a Teams channel via Graph API. Requires TEAMS_TENANT_ID, TEAMS_CLIENT_ID, TEAMS_CLIENT_SECRET.",
        "params": {
            "team_id":    "string — Teams group/team ID (required)",
            "channel_id": "string — Teams channel ID (required)",
            "limit":      "integer — max messages, default 10 (optional)",
        },
    },
    {
        "name": "teams_list_teams",
        "description": "List Microsoft Teams groups accessible to the app via Graph API.",
        "params": {},
    },
    # ── Cloud execution ───────────────────────────────────────────────────────
    {
        "name": "modal_run",
        "description": "Run Python code on a Modal serverless worker (cloud GPU/CPU). Requires: pip install modal && modal setup.",
        "params": {
            "code":         "string — Python code to execute (required)",
            "requirements": "list — pip packages to install e.g. ['numpy', 'pandas'] (optional)",
            "image":        "string — base Docker image, default 'python:3.12' (optional)",
            "timeout":      "integer — seconds before timeout, default 60 (optional)",
            "gpu":          "string — GPU type e.g. 'T4', 'A10G', 'A100' — leave empty for CPU (optional)",
        },
    },
    {
        "name": "modal_status",
        "description": "Check whether Modal is installed and authenticated.",
        "params": {},
    },
    {
        "name": "daytona_run",
        "description": "Run a shell command in a Daytona managed workspace. Requires DAYTONA_API_KEY.",
        "params": {
            "command":      "string — shell command to execute (required)",
            "workspace_id": "string — existing workspace ID, creates ephemeral one if omitted (optional)",
            "image":        "string — Docker image for new workspace, default 'ubuntu:22.04' (optional)",
            "timeout":      "integer — seconds before timeout, default 60 (optional)",
        },
    },
    {
        "name": "daytona_list_workspaces",
        "description": "List Daytona managed workspaces. Requires DAYTONA_API_KEY.",
        "params": {},
    },
    # ── Pipeline macros ───────────────────────────────────────────────────────
    {
        "name": "macro_save",
        "description": "Save a named pipeline macro — a sequence of tool steps that run in order, with output passed between steps via {prev_text} template variables.",
        "params": {
            "name":        "string — macro name (required)",
            "steps":       "list — list of {tool, params} step dicts (required)",
            "description": "string — human description of what the macro does (optional)",
            "vars":        "dict — default variable values e.g. {'query': 'default'} (optional)",
        },
    },
    {
        "name": "macro_delete",
        "description": "Delete a saved macro by name.",
        "params": {
            "name": "string — macro name to delete (required)",
        },
    },
    {
        "name": "macro_list",
        "description": "List all saved pipeline macros with their descriptions and step counts.",
        "params": {},
    },
    {
        "name": "run_macro",
        "description": "Execute a saved pipeline macro. Steps run in order; {prev_text} in params is replaced with the previous step's output.",
        "params": {
            "name": "string — macro name to run (required)",
            "vars": "dict — override variable values for this run (optional)",
        },
    },
    # ── Goals ─────────────────────────────────────────────────────────────────
    {
        "name": "goal_set",
        "description": "Create a new persistent goal that will be injected into every system prompt until completed.",
        "params": {
            "title":       "string — goal title (required)",
            "description": "string — detailed description (optional)",
            "deadline":    "string — deadline date string e.g. '2025-12-31' (optional)",
            "priority":    "string — 'high', 'medium', or 'low', default 'medium' (optional)",
        },
    },
    {
        "name": "goal_update",
        "description": "Update an existing goal's status or add a progress note.",
        "params": {
            "goal_id":      "string — goal ID (required)",
            "progress_note": "string — note about progress made (optional)",
            "status":       "string — new status: 'active', 'paused', or 'abandoned' (optional)",
        },
    },
    {
        "name": "goal_list",
        "description": "List persistent goals, optionally filtered by status.",
        "params": {
            "status": "string — filter by 'active', 'complete', 'paused', or 'abandoned' (optional)",
        },
    },
    {
        "name": "goal_complete",
        "description": "Mark a goal as completed.",
        "params": {
            "goal_id": "string — goal ID to mark complete (required)",
        },
    },
    {
        "name": "goal_delete",
        "description": "Permanently delete a goal.",
        "params": {
            "goal_id": "string — goal ID to delete (required)",
        },
    },
    # ── Sub-agent ─────────────────────────────────────────────────────────────
    {
        "name": "sub_agent",
        "description": "Spawn a focused sub-agent to execute a specific prompt autonomously. Use for parallelisable sub-tasks.",
        "params": {
            "prompt": "string — the complete task prompt for the sub-agent (required)",
        },
    },
    {
        "name": "spawn_agent",
        "description": (
            "Spawn a SANDBOXED specialist worker with an explicit tool allocation "
            "(the multi-agent factory). Pick a persona and hand it ONLY the tools "
            "it needs — e.g. a 'researcher' for web/read tools, an 'engineer' for "
            "file-edit/code tools, an 'auditor' for lint/test/log review (no write "
            "tools). The worker runs its own bounded, guardrailed tool loop and "
            "returns a report. Use this to decompose a large task across workers."
        ),
        "params": {
            "persona":         "string — researcher | engineer | auditor | analyst | writer | reviewer | planner | coder | generalist (required)",
            "objective":       "string — what the worker must accomplish (required)",
            "allocated_tools": "array — explicit tool names the worker may use; omit to use the persona's default toolset (optional)",
        },
    },
    # ── LLM task (in-band sub-call) ───────────────────────────────────────────
    {
        "name": "llm_task",
        "description": (
            "Make a lightweight in-band LLM call for a focused sub-task (classification, "
            "summarisation, extraction) without spinning up a full sub-agent. "
            "Cheaper and faster than sub_agent for simple one-shot prompts."
        ),
        "params": {
            "prompt":      "string — the focused prompt for the LLM (required)",
            "model":       "string — model to use, defaults to config default (optional)",
            "max_tokens":  "integer — max response tokens, default 1024 (optional)",
            "temperature": "float — sampling temperature, default 0.1 (optional)",
            "system":      "string — custom system prompt (optional)",
        },
    },
    # ── Apply patch ───────────────────────────────────────────────────────────
    {
        "name": "apply_patch",
        "description": (
            "Apply a structured patch to files. Supports three formats: "
            "1) unified_diff (standard --- / +++ diff), "
            "2) search_replace (list of {search, replace} ops), "
            "3) json_patch (RFC 6902 JSON Patch operations). "
            "Safer than file_patch for multi-hunk changes."
        ),
        "params": {
            "patch":      "string | list — patch content (required)",
            "workspace":  "string — root dir for unified diffs, default '.' (optional)",
            "file_path":  "string — target file for search_replace and json_patch (optional)",
            "format":     "string — 'auto' | 'unified_diff' | 'search_replace' | 'json_patch' (optional)",
            "dry_run":    "boolean — validate but don't write changes (optional)",
            "backup":     "boolean — create .bak files before modifying, default true (optional)",
        },
    },
    # ── PDF operations ────────────────────────────────────────────────────────
    {
        "name": "pdf_create",
        "description": (
            "Create a PDF document from Markdown-like text content using reportlab. "
            "Supports headings (#/##/###), **bold**, and ```code blocks```. "
            "Parent directories are created automatically."
        ),
        "params": {
            "output":     "string — output PDF file path (required)",
            "content":    "string — Markdown-like text content (required)",
            "title":      "string — document title metadata (optional)",
            "author":     "string — document author metadata (optional)",
            "font_size":  "integer — base font size, default 11 (optional)",
            "page_size":  "string — 'letter' | 'a4', default 'letter' (optional)",
            "margins":    "integer — page margins in points, default 72 (optional)",
        },
    },
    {
        "name": "pdf_info",
        "description": "Return metadata for a PDF file: page count, encryption status, file size, title, and author.",
        "params": {
            "path": "string — path to the PDF file (required)",
        },
    },
    {
        "name": "pdf_extract_text",
        "description": (
            "Extract text from a PDF file, optionally limiting to specific pages. "
            "Page markers [Page N] are inserted between pages."
        ),
        "params": {
            "path":  "string — path to the PDF file (required)",
            "pages": "string — page range e.g. '1-3,5,7' (optional — all pages if omitted)",
        },
    },
    {
        "name": "pdf_merge",
        "description": "Merge multiple PDF files into a single output PDF. All input files must exist.",
        "params": {
            "paths":  "list — ordered list of PDF file paths to merge (required)",
            "output": "string — output PDF file path (required)",
        },
    },
    {
        "name": "pdf_split",
        "description": "Split a PDF into multiple files, grouping N pages per output file.",
        "params": {
            "path":           "string — source PDF path (required)",
            "output_dir":     "string — directory for output files (required)",
            "pages_per_file": "integer — pages per output file, default 1 (optional)",
        },
    },
    {
        "name": "pdf_rotate",
        "description": "Rotate pages in a PDF by 90, 180, or 270 degrees.",
        "params": {
            "path":    "string — source PDF path (required)",
            "output":  "string — output PDF path (required)",
            "degrees": "integer — rotation: 90 | 180 | 270 (optional, default 90)",
            "pages":   "string — page range e.g. '1-3,5' (optional — all pages if omitted)",
        },
    },
    {
        "name": "pdf_watermark",
        "description": "Stamp a diagonal text watermark on every page of a PDF.",
        "params": {
            "path":      "string — source PDF path (required)",
            "output":    "string — output PDF path (required)",
            "text":      "string — watermark text, e.g. 'CONFIDENTIAL' (required)",
            "opacity":   "number — text opacity 0.0–1.0, default 0.3 (optional)",
            "font_size": "integer — watermark font size, default 48 (optional)",
            "color":     "string — text color name, default 'gray' (optional)",
        },
    },
    {
        "name": "pdf_encrypt",
        "description": "Password-protect a PDF with a user password (and optional owner password).",
        "params": {
            "path":           "string — source PDF path (required)",
            "output":         "string — output encrypted PDF path (required)",
            "user_password":  "string — password required to open the PDF (required)",
            "owner_password": "string — owner/permissions password (optional)",
        },
    },
    {
        "name": "pdf_decrypt",
        "description": "Remove password protection from an encrypted PDF.",
        "params": {
            "path":     "string — source encrypted PDF path (required)",
            "output":   "string — output decrypted PDF path (required)",
            "password": "string — the user or owner password (required)",
        },
    },
    {
        "name": "pdf_extract_pages",
        "description": "Extract specific pages from a PDF into a new file.",
        "params": {
            "path":   "string — source PDF path (required)",
            "output": "string — output PDF path (required)",
            "pages":  "string — page range e.g. '1,3,5-8' (required)",
        },
    },
    # ── Advanced image generation ─────────────────────────────────────────────
    {
        "name": "dalle_generate",
        "description": (
            "Generate an image with DALL-E 3 (OpenAI). More options than image_generate: "
            "size, quality, style, and number of images. Saves to ~/.operon/images/. "
            "Requires OPENAI_API_KEY."
        ),
        "params": {
            "prompt":         "string — detailed image description (required)",
            "model":          "string — 'dall-e-3' | 'dall-e-2', default 'dall-e-3' (optional)",
            "size":           "string — '1024x1024' | '1792x1024' | '1024x1792', default '1024x1024' (optional)",
            "quality":        "string — 'standard' | 'hd', default 'standard' (optional)",
            "style":          "string — 'vivid' | 'natural', default 'vivid' (optional)",
            "n":              "integer — number of images, default 1 (optional)",
            "save":           "boolean — save to disk, default true (optional)",
            "return_base64":  "boolean — include base64 in response, default false (optional)",
        },
    },
    {
        "name": "image_edit",
        "description": (
            "Edit an existing image with DALL-E 2 based on a text prompt. "
            "Optionally provide a mask PNG to restrict edits to a region. "
            "Requires OPENAI_API_KEY."
        ),
        "params": {
            "image_path": "string — path to source PNG image (required)",
            "prompt":     "string — description of the desired edit (required)",
            "mask_path":  "string — path to mask PNG (transparent = edit region, optional)",
            "size":       "string — output size, default '1024x1024' (optional)",
            "n":          "integer — number of variations, default 1 (optional)",
            "save":       "boolean — save outputs to disk, default true (optional)",
        },
    },
    {
        "name": "image_variation",
        "description": "Generate variations of an existing image using DALL-E 2. Requires OPENAI_API_KEY.",
        "params": {
            "image_path": "string — path to source PNG image (required)",
            "n":          "integer — number of variations, default 1 (optional)",
            "size":       "string — output size, default '1024x1024' (optional)",
            "save":       "boolean — save outputs to disk, default true (optional)",
        },
    },
    {
        "name": "image_list_generated",
        "description": "List previously generated images saved in ~/.operon/images/.",
        "params": {
            "limit": "integer — max files to return, default 20 (optional)",
        },
    },
    {
        "name": "image_describe",
        "description": "Describe an image using GPT-4 Vision. Accepts a local file path or public URL.",
        "params": {
            "path_or_url": "string — local file path or public image URL (required)",
            "detail":      "string — 'low' | 'high', default 'low' (optional)",
        },
    },
    # ── Data analysis ─────────────────────────────────────────────────────────
    {
        "name": "data_load",
        "description": (
            "Load a data file and return a preview (first 10 rows). "
            "Supports CSV, TSV, JSON, JSONL, Excel (xlsx/xls), Parquet, and SQLite."
        ),
        "params": {
            "path":      "string — file path (required)",
            "format":    "string — 'csv' | 'tsv' | 'json' | 'jsonl' | 'excel' | 'parquet' | 'sqlite' (optional — auto-detected)",
            "sheet":     "string — Excel sheet name or index (optional)",
            "encoding":  "string — text encoding, default 'utf-8' (optional)",
            "delimiter": "string — CSV delimiter, default ',' (optional)",
        },
    },
    {
        "name": "data_save",
        "description": "Save data (list of dicts or JSON string) to a file. Format auto-detected from extension.",
        "params": {
            "path":   "string — output file path (required)",
            "data":   "string | list — JSON string or list of dicts (required)",
            "format": "string — 'csv' | 'json' | 'excel' | 'parquet' (optional — auto-detected)",
            "index":  "boolean — include DataFrame index, default false (optional)",
        },
    },
    {
        "name": "data_convert",
        "description": "Convert a data file from one format to another (e.g. CSV → Excel, JSON → Parquet).",
        "params": {
            "input_path":     "string — source file path (required)",
            "output_path":    "string — destination file path (required)",
            "input_format":   "string — source format (optional — auto-detected)",
            "output_format":  "string — target format (optional — auto-detected from extension)",
        },
    },
    {
        "name": "data_describe_stats",
        "description": (
            "Compute descriptive statistics for a dataset: shape, dtypes, null percentages, "
            "min/max/mean/std, and top value counts per column."
        ),
        "params": {
            "path":    "string — file path (required)",
            "columns": "list — specific columns to describe (optional — all columns if omitted)",
            "format":  "string — file format (optional — auto-detected)",
        },
    },
    {
        "name": "data_query",
        "description": (
            "Filter rows using a pandas query expression. "
            "Example: 'age > 30 and salary < 100000' or 'status == \"active\"'."
        ),
        "params": {
            "path":   "string — file path (required)",
            "query":  "string — pandas query expression (required)",
            "format": "string — file format (optional — auto-detected)",
            "limit":  "integer — max rows to return, default 100 (optional)",
        },
    },
    {
        "name": "data_groupby",
        "description": "Group a dataset by one or more columns and apply aggregate functions.",
        "params": {
            "path":       "string — file path (required)",
            "group_cols": "list — column name(s) to group by (required)",
            "agg":        "object — {column: function} e.g. {\"salary\": \"sum\", \"count\": \"mean\"} (required)",
            "format":     "string — file format (optional)",
        },
    },
    {
        "name": "data_clean",
        "description": (
            "Clean a dataset: drop nulls/duplicates, fill missing values, "
            "strip whitespace, and normalize column names to snake_case."
        ),
        "params": {
            "path":                   "string — source file path (required)",
            "output":                 "string — output file path (required)",
            "drop_nulls":             "boolean — drop rows with any null, default false (optional)",
            "fill_nulls":             "string | number — fill value for nulls (optional)",
            "drop_duplicates":        "boolean — remove duplicate rows, default true (optional)",
            "strip_whitespace":       "boolean — strip leading/trailing whitespace, default true (optional)",
            "normalize_column_names": "boolean — convert column names to snake_case, default true (optional)",
            "format":                 "string — file format (optional)",
        },
    },
    {
        "name": "data_merge",
        "description": "Join two datasets on one or more key columns (inner/left/right/outer join).",
        "params": {
            "left_path":  "string — left dataset file path (required)",
            "right_path": "string — right dataset file path (required)",
            "output":     "string — output file path (required)",
            "on":         "string | list — shared key column(s) (optional — use left_on/right_on for different names)",
            "left_on":    "string | list — key column(s) in left dataset (optional)",
            "right_on":   "string | list — key column(s) in right dataset (optional)",
            "how":        "string — 'inner' | 'left' | 'right' | 'outer', default 'inner' (optional)",
        },
    },
    {
        "name": "data_pivot",
        "description": "Create a pivot table from a dataset.",
        "params": {
            "path":    "string — file path (required)",
            "index":   "string — column(s) for the pivot row index (required)",
            "columns": "string — column to spread as pivot columns (required)",
            "values":  "string — column with the values to aggregate (required)",
            "aggfunc": "string — aggregation function: 'sum' | 'mean' | 'count' | 'min' | 'max', default 'sum' (optional)",
            "output":  "string — save result to this path (optional — printed if omitted)",
        },
    },
    {
        "name": "data_anomalies",
        "description": "Detect outliers/anomalies in numeric columns using Z-score or IQR method.",
        "params": {
            "path":      "string — file path (required)",
            "columns":   "list — column names to check (optional — all numeric columns if omitted)",
            "method":    "string — 'zscore' | 'iqr', default 'zscore' (optional)",
            "threshold": "number — Z-score threshold or IQR multiplier, default 3.0 (optional)",
        },
    },
    {
        "name": "data_correlations",
        "description": "Compute pairwise correlations between numeric columns and highlight strong relationships.",
        "params": {
            "path":      "string — file path (required)",
            "method":    "string — 'pearson' | 'spearman' | 'kendall', default 'pearson' (optional)",
            "threshold": "number — report pairs with |correlation| ≥ this value, default 0.7 (optional)",
        },
    },
    {
        "name": "data_chart",
        "description": (
            "Generate and save a chart (PNG) from a dataset. "
            "Saves to ~/.operon/charts/. Returns the saved file path."
        ),
        "params": {
            "path":       "string — data file path (required)",
            "chart_type": "string — 'bar' | 'barh' | 'line' | 'scatter' | 'hist' | 'box' | 'pie' | 'area' | 'heatmap' (required)",
            "x":          "string — column name for X axis (required for most chart types)",
            "y":          "string | list — column name(s) for Y axis (optional)",
            "title":      "string — chart title (optional)",
            "output":     "string — output PNG path (optional — auto-named if omitted)",
            "figsize":    "string — 'WIDTHxHEIGHT' in inches e.g. '12x6', default '10x6' (optional)",
        },
    },
    # ── Browser stealth / visual grounding ────────────────────────────────────
    {
        "name": "browser_find_element",
        "description": (
            "Find a page element by natural-language description (e.g. 'the login button', "
            "'search box'). Returns matching selector(s) from the aria snapshot tree. "
            "Use when you don't know the exact CSS selector."
        ),
        "params": {
            "description": "string — natural-language description of the element (required)",
            "task_id":     "string — session ID (optional)",
        },
    },
    {
        "name": "browser_check_captcha",
        "description": (
            "Detect whether the current page contains a CAPTCHA or bot-challenge. "
            "Returns {captcha_detected: bool, hint, url, title}."
        ),
        "params": {
            "task_id": "string — session ID (optional)",
        },
    },
    {
        "name": "browser_human_click",
        "description": (
            "Click an element with randomised human-like jitter and delays to avoid bot detection. "
            "Provide a CSS selector OR absolute x/y coordinates."
        ),
        "params": {
            "selector": "string — CSS selector or 'text=...' locator (optional if x/y given)",
            "x":        "number — page X coordinate (optional)",
            "y":        "number — page Y coordinate (optional)",
            "task_id":  "string — session ID (optional)",
        },
    },
    {
        "name": "browser_human_type",
        "description": (
            "Type text character by character with gaussian inter-key delays to simulate human typing. "
            "Use instead of browser_type when stealth is important."
        ),
        "params": {
            "text":     "string — text to type (required)",
            "selector": "string — CSS selector to focus before typing (optional)",
            "delay_ms": "integer — average delay between keystrokes in ms, default 80 (optional)",
            "task_id":  "string — session ID (optional)",
        },
    },
    {
        "name": "browser_wait_for_element",
        "description": "Wait for a CSS selector to reach a given state (visible/attached/hidden) with a timeout.",
        "params": {
            "selector":   "string — CSS selector to wait for (required)",
            "timeout_ms": "integer — max wait time in ms, default 10000 (optional)",
            "state":      "string — 'visible' | 'attached' | 'hidden', default 'visible' (optional)",
            "task_id":    "string — session ID (optional)",
        },
    },
    {
        "name": "browser_get_url",
        "description": "Return the current browser page URL and title.",
        "params": {
            "task_id": "string — session ID (optional)",
        },
    },
    # ── GitHub API ────────────────────────────────────────────────────────────
    {
        "name": "github_repo_info",
        "description": (
            "Return metadata about a GitHub repository: stars, forks, language, "
            "open issues, license, topics, and more. No authentication needed for public repos."
        ),
        "params": {
            "owner": "string — repository owner (user or org) (required)",
            "repo":  "string — repository name (required)",
            "token": "string — GitHub personal access token (optional — auto-read from GITHUB_TOKEN)",
        },
    },
    {
        "name": "github_list_repos",
        "description": "List repositories for a GitHub user or organisation.",
        "params": {
            "username": "string — GitHub username (optional — provide username OR org)",
            "org":      "string — GitHub org name (optional — provide username OR org)",
            "sort":     "string — 'created' | 'updated' | 'pushed' | 'full_name', default 'updated' (optional)",
            "limit":    "integer — max repositories to return, default 20 (optional)",
            "token":    "string — GitHub token (optional)",
        },
    },
    {
        "name": "github_list_issues",
        "description": "List issues for a GitHub repository. Excludes pull requests.",
        "params": {
            "owner":  "string — repository owner (required)",
            "repo":   "string — repository name (required)",
            "state":  "string — 'open' | 'closed' | 'all', default 'open' (optional)",
            "labels": "string — comma-separated label names to filter by (optional)",
            "limit":  "integer — max issues to return, default 20 (optional)",
            "token":  "string — GitHub token (optional)",
        },
    },
    {
        "name": "github_create_issue",
        "description": "Create a new issue in a GitHub repository. Requires GITHUB_TOKEN with repo scope.",
        "params": {
            "owner":     "string — repository owner (required)",
            "repo":      "string — repository name (required)",
            "title":     "string — issue title (required)",
            "body":      "string — issue description in Markdown (optional)",
            "labels":    "list — label names to apply e.g. ['bug', 'help wanted'] (optional)",
            "assignees": "list — GitHub usernames to assign (optional)",
            "token":     "string — GitHub token with repo scope (optional — auto-read from GITHUB_TOKEN)",
        },
    },
    {
        "name": "github_list_prs",
        "description": "List pull requests for a GitHub repository.",
        "params": {
            "owner": "string — repository owner (required)",
            "repo":  "string — repository name (required)",
            "state": "string — 'open' | 'closed' | 'all', default 'open' (optional)",
            "limit": "integer — max PRs to return, default 20 (optional)",
            "token": "string — GitHub token (optional)",
        },
    },
    {
        "name": "github_search_code",
        "description": (
            "Search code across GitHub repositories by keyword. "
            "Optionally filter by language, owner, or repo. "
            "Rate-limited without authentication."
        ),
        "params": {
            "query":    "string — search keywords (required)",
            "language": "string — programming language filter e.g. 'python' (optional)",
            "owner":    "string — GitHub user/org to scope search (optional)",
            "repo":     "string — specific repo to search e.g. 'myrepo' (optional, requires owner)",
            "limit":    "integer — max results, default 10 (optional)",
            "token":    "string — GitHub token for higher rate limits (optional)",
        },
    },
    {
        "name": "github_search_repos",
        "description": "Search GitHub repositories by keyword, optionally filtered by language.",
        "params": {
            "query":    "string — search keywords (required)",
            "language": "string — programming language filter (optional)",
            "sort":     "string — 'stars' | 'forks' | 'updated', default 'stars' (optional)",
            "limit":    "integer — max results, default 10 (optional)",
            "token":    "string — GitHub token (optional)",
        },
    },
    {
        "name": "github_get_file",
        "description": (
            "Fetch the decoded text content of a file from a GitHub repository, "
            "or list a directory's contents. No authentication needed for public repos."
        ),
        "params": {
            "owner": "string — repository owner (required)",
            "repo":  "string — repository name (required)",
            "path":  "string — file or directory path in the repo (required)",
            "ref":   "string — branch, tag, or commit SHA (optional — default branch if omitted)",
            "token": "string — GitHub token for private repos (optional)",
        },
    },
    {
        "name": "github_create_gist",
        "description": (
            "Create a GitHub Gist (public or private) with one or more files. "
            "Returns the Gist URL. Requires GITHUB_TOKEN with gist scope."
        ),
        "params": {
            "description": "string — gist description (optional)",
            "files":       "object — {filename: content} dict of files to include (required)",
            "public":      "boolean — make gist public, default false (optional)",
            "token":       "string — GitHub token with gist scope (optional — auto-read from GITHUB_TOKEN)",
        },
    },
    {
        "name": "github_user_info",
        "description": (
            "Return public profile information for a GitHub user "
            "(name, bio, repos, followers, etc.). "
            "Omit username to get the authenticated user's profile (requires token)."
        ),
        "params": {
            "username": "string — GitHub username (optional — omit for authenticated user)",
            "token":    "string — GitHub token (optional)",
        },
    },
    {
        "name": "github_list_commits",
        "description": "List recent commits for a repository, optionally filtered by branch or author.",
        "params": {
            "owner":  "string — repository owner (required)",
            "repo":   "string — repository name (required)",
            "branch": "string — branch name (optional — default branch if omitted)",
            "author": "string — filter by GitHub username or email (optional)",
            "limit":  "integer — max commits to return, default 20 (optional)",
            "token":  "string — GitHub token (optional)",
        },
    },
]

# ── Dispatch map: tool name → callable ───────────────────────────────────────

_DISPATCH: dict[str, Callable] = {
    # File system
    "file_read":         file_read,
    "file_write":        file_write,
    "file_append":       file_append,
    "file_patch":        file_patch,
    "file_delete":       file_delete,
    "dir_list":          dir_list,
    "file_exists":       file_exists,
    "file_info":         file_info,
    # Shell
    "shell_exec":        shell_exec,
    # Web
    "duckduckgo_search": duckduckgo_search,
    "web_scrape":        web_scrape,
    # Code
    "python_exec":       python_exec,
    # HTTP
    "http_request":      http_request,
    # Search
    "file_search":       file_search,
    # Email  (email_send is internal-only — not model-callable)
    "email_draft":       email_draft,
    # Browser automation (production Playwright CDP)
    "browser_navigate":      browser_navigate,
    "browser_snapshot":      browser_snapshot,
    "browser_screenshot":    browser_screenshot,
    "browser_click":         browser_click,
    "browser_type":          browser_type,
    "browser_scroll":        browser_scroll,
    "browser_hover":         browser_hover,
    "browser_key":           browser_key,
    "browser_select":        browser_select,
    "browser_fill_form":     browser_fill_form,
    "browser_wait":          browser_wait,
    "browser_extract_text":  browser_extract_text,
    "browser_extract_links": browser_extract_links,
    "browser_extract_tables": browser_extract_tables,
    "browser_evaluate":      browser_evaluate,
    "browser_new_tab":       browser_new_tab,
    "browser_list_tabs":     browser_list_tabs,
    "browser_switch_tab":    browser_switch_tab,
    "browser_go_back":       browser_go_back,
    "browser_go_forward":    browser_go_forward,
    "browser_reload":        browser_reload,
    "browser_get_cookies":   browser_get_cookies,
    "browser_set_cookie":    browser_set_cookie,
    "browser_print_pdf":     browser_print_pdf,
    "browser_network_log":   browser_network_log,
    "browser_console_log":   browser_console_log,
    "browser_close":             browser_close,
    # New browser stealth + visual grounding tools
    "browser_find_element":      browser_find_element,
    "browser_check_captcha":     browser_check_captcha,
    "browser_human_click":       browser_human_click,
    "browser_human_type":        browser_human_type,
    "browser_wait_for_element":  browser_wait_for_element,
    "browser_get_url":           browser_get_url,
    # Computer use (screen + mouse/keyboard)
    "computer_use":          computer_use,
    # Sub-agent delegation
    "delegate_task":         delegate_task,
    "delegate_batch":        delegate_batch,
    # Vision / media
    "vision_analyze":    vision_analyze,
    "image_generate":    image_generate,
    "video_generate":      video_generate,
    "video_from_image":    video_from_image,
    "video_list_generated": video_list_generated,
    "tts_speak":         tts_speak,
    # Messaging
    "telegram_send":          telegram_send,
    "telegram_get_updates":   telegram_get_updates,
    "telegram_edit_message":  telegram_edit_message,
    "telegram_delete_message": telegram_delete_message,
    "telegram_pin_message":   telegram_pin_message,
    "telegram_send_photo":    telegram_send_photo,
    "telegram_send_document": telegram_send_document,
    "clarify":           clarify,
    "todo":              todo,
    # Permanent knowledge
    "knowledge_set":     knowledge_set,
    "knowledge_get":     knowledge_get,
    "knowledge_delete":  knowledge_delete,
    "knowledge_list":    knowledge_list,
    # X / Twitter search
    "x_search":          x_search,
    # SSH
    "ssh_exec":          ssh_exec,
    "ssh_upload":        ssh_upload,
    "ssh_download":      ssh_download,
    # Git
    "git_status":        git_status,
    "git_diff":          git_diff,
    "git_log":           git_log,
    "git_add":           git_add,
    "git_commit":        git_commit,
    "git_checkout":      git_checkout,
    "git_branch":        git_branch,
    "git_stash":         git_stash,
    # Discord
    "discord_send":               discord_send,
    "discord_get_messages":       discord_get_messages,
    "discord_create_webhook":     discord_create_webhook,
    # Slack
    "slack_send":                 slack_send,
    "slack_get_messages":         slack_get_messages,
    "slack_list_channels":        slack_list_channels,
    "slack_upload_file":          slack_upload_file,
    "slack_send_dm":              slack_send_dm,
    "slack_get_thread":           slack_get_thread,
    "slack_update_message":       slack_update_message,
    "slack_schedule_message":     slack_schedule_message,
    "slack_add_reaction":         slack_add_reaction,
    "slack_pin_message":          slack_pin_message,
    "slack_set_topic":            slack_set_topic,
    "slack_build_blocks":         slack_build_blocks,
    "slack_search_messages":      slack_search_messages,
    "slack_list_users":           slack_list_users,
    "slack_create_channel":       slack_create_channel,
    "slack_delete_message":       slack_delete_message,
    "slack_status":               slack_status,
    # Database
    "db_query":                   db_query,
    "db_list_tables":             db_list_tables,
    "db_describe_table":          db_describe_table,
    "mongo_query":                mongo_query,
    # Voice
    "voice_record_and_transcribe": voice_record_and_transcribe,
    "voice_transcribe_file":       voice_transcribe_file,
    "voice_speak":                 voice_speak,
    "voice_list_voices":           voice_list_voices,
    # WhatsApp
    "whatsapp_send":               whatsapp_send,
    "whatsapp_get_messages":       whatsapp_get_messages,
    "whatsapp_status":             whatsapp_status,
    # Docker
    "docker_run":                  docker_run,
    "docker_run_code":             docker_run_code,
    "docker_list_containers":      docker_list_containers,
    "docker_pull":                 docker_pull,
    # Signal
    "signal_send":                 signal_send,
    "signal_receive":              signal_receive,
    "signal_list_groups":          signal_list_groups,
    # Matrix
    "matrix_send":                 matrix_send,
    "matrix_get_messages":         matrix_get_messages,
    "matrix_list_rooms":           matrix_list_rooms,
    # IRC
    "irc_send":                    irc_send,
    "irc_get_messages":            irc_get_messages,
    # Mattermost
    "mattermost_send":             mattermost_send,
    "mattermost_get_messages":     mattermost_get_messages,
    "mattermost_list_channels":    mattermost_list_channels,
    # Microsoft Teams
    "teams_send":                  teams_send,
    "teams_get_messages":          teams_get_messages,
    "teams_list_teams":            teams_list_teams,
    # Cloud execution
    "modal_run":                   modal_run,
    "modal_status":                modal_status,
    "daytona_run":                 daytona_run,
    "daytona_list_workspaces":     daytona_list_workspaces,
    # Pipeline macros
    "macro_save":                  macro_save,
    "macro_delete":                macro_delete,
    "macro_list":                  macro_list,
    "run_macro":                   run_macro,
    # Goals
    "goal_set":                    goal_set,
    "goal_update":                 goal_update,
    "goal_list":                   goal_list,
    "goal_complete":               goal_complete,
    "goal_delete":                 goal_delete,
    # LLM task
    "llm_task":                    llm_task,
    # Apply patch
    "apply_patch":                 apply_patch,
    # PDF operations
    "pdf_create":          pdf_create,
    "pdf_extract_text":    pdf_extract_text,
    "pdf_info":            pdf_info,
    "pdf_merge":           pdf_merge,
    "pdf_split":           pdf_split,
    "pdf_rotate":          pdf_rotate,
    "pdf_watermark":       pdf_watermark,
    "pdf_encrypt":         pdf_encrypt,
    "pdf_decrypt":         pdf_decrypt,
    "pdf_extract_pages":   pdf_extract_pages,
    # Image generation
    "dalle_generate":      dalle_generate,
    "image_edit":          image_edit,
    "image_variation":     image_variation,
    "image_list_generated": image_list_generated,
    "image_describe":      image_describe,
    # Data analysis
    "data_load":           data_load,
    "data_save":           data_save,
    "data_convert":        data_convert,
    "data_describe_stats": data_describe_stats,
    "data_query":          data_query,
    "data_groupby":        data_groupby,
    "data_clean":          data_clean,
    "data_merge":          data_merge,
    "data_pivot":          data_pivot,
    "data_anomalies":      data_anomalies,
    "data_correlations":   data_correlations,
    "data_chart":          data_chart,
    # GitHub API
    "github_repo_info":    github_repo_info,
    "github_list_repos":   github_list_repos,
    "github_list_issues":  github_list_issues,
    "github_create_issue": github_create_issue,
    "github_list_prs":     github_list_prs,
    "github_search_code":  github_search_code,
    "github_search_repos": github_search_repos,
    "github_get_file":     github_get_file,
    "github_create_gist":  github_create_gist,
    "github_user_info":    github_user_info,
    "github_list_commits": github_list_commits,
}

_sub_agent_runner: Optional[Callable[[str], None]] = None
# Factory hook for the spawn_agent meta-tool: (persona, objective, allocated_tools) -> dict
_agent_factory: Optional[Callable[[str, str, list], dict]] = None

# Tools that sub-agents (delegated workers) must NOT call:
#   sub_agent / delegate_task / delegate_batch — no recursive delegation
#   clarify    — sub-agents have no live user to ask
#   email_draft — approval dialog doesn't work in a headless sub-agent
#   computer_use — no GUI in headless sub-agents
DELEGATE_BLOCKED_TOOLS: frozenset[str] = frozenset({
    "sub_agent",
    "delegate_task",
    "delegate_batch",
    "spawn_agent",      # no recursive worker spawning (prevents fork bombs)
    "clarify",
    "email_draft",
    "computer_use",
})


def set_sub_agent_runner(runner: Callable[[str], None]) -> None:
    global _sub_agent_runner
    _sub_agent_runner = runner


def _sub_agent(prompt: str, **_) -> dict:
    if _sub_agent_runner is None:
        return {"success": False, "output": None,
                "error": "Sub-agent runner not wired up."}
    try:
        # Inject a reminder about blocked tools so the sub-agent knows its limits
        _blocked_hint = (
            "\n\n[SYSTEM] You are running as a sub-agent. "
            f"The following tools are NOT available to you: "
            f"{', '.join(sorted(DELEGATE_BLOCKED_TOOLS))}. "
            "Do not attempt to call them."
        )
        scoped_prompt = prompt + _blocked_hint
        result = _sub_agent_runner(scoped_prompt)
        output = result if isinstance(result, str) else "Sub-agent completed."
        return {"success": True, "output": output, "error": ""}
    except Exception as e:
        return {"success": False, "output": None, "error": str(e)}


_DISPATCH["sub_agent"] = _sub_agent


def set_agent_factory(factory: Callable[[str, str, list], dict]) -> None:
    """Wire the spawn_agent meta-tool to an AgentMesh-backed factory."""
    global _agent_factory
    _agent_factory = factory


def spawn_agent(persona: str = "generalist", objective: str = "",
                allocated_tools: Any = None, **_) -> dict:
    """
    Spawn a sandboxed specialist sub-agent (the multi-agent factory meta-tool).

    Args:
        persona         — worker role: 'researcher' | 'engineer' | 'auditor' |
                          'analyst' | 'writer' | 'reviewer' | 'planner' |
                          'coder' | 'generalist' (or free-text description).
        objective       — what the worker must accomplish (required).
        allocated_tools — explicit list (or comma-separated string) of tool names
                          the worker may use. The worker is hard-sandboxed to this
                          set; omit it to use the persona's default toolset.

    Returns:
        {success, persona, output, error}
    """
    objective = (objective or "").strip()
    if not objective:
        return {"success": False, "persona": persona,
                "output": None, "error": "objective is required."}
    if _agent_factory is None:
        return {"success": False, "persona": persona,
                "output": None, "error": "Agent factory not wired up."}
    # Normalise allocated_tools to a list; tolerate None / str / list gracefully.
    if isinstance(allocated_tools, str):
        allocated_tools = [t.strip() for t in allocated_tools.split(",") if t.strip()]
    elif not isinstance(allocated_tools, list):
        allocated_tools = []
    # Defensive sandbox: a spawned worker can never receive delegate/spawn tools.
    allocated_tools = [t for t in allocated_tools if t not in DELEGATE_BLOCKED_TOOLS]
    try:
        return _agent_factory(persona or "generalist", objective, allocated_tools)
    except Exception as e:
        return {"success": False, "persona": persona, "output": None, "error": str(e)}


_DISPATCH["spawn_agent"] = spawn_agent


# ── Toolset groups ────────────────────────────────────────────────────────────
# Logical groupings of tools that can be enabled/disabled as a set.
# /toolsets list|enable <group>|disable <group>

TOOLSETS: dict[str, list[str]] = {
    "filesystem":  ["file_read", "file_write", "file_append", "file_patch",
                    "file_delete", "dir_list", "file_exists", "file_info", "file_search"],
    "shell":       ["shell_exec", "python_exec"],
    "web":         ["duckduckgo_search", "web_scrape", "x_search", "http_request"],
    "email":       ["email_draft"],
    "browser":     ["browser_navigate", "browser_snapshot", "browser_screenshot",
                    "browser_click", "browser_type", "browser_scroll", "browser_hover",
                    "browser_key", "browser_select", "browser_fill_form", "browser_wait",
                    "browser_extract_text", "browser_extract_links", "browser_extract_tables",
                    "browser_evaluate", "browser_new_tab", "browser_list_tabs",
                    "browser_switch_tab", "browser_go_back", "browser_go_forward",
                    "browser_reload", "browser_get_cookies", "browser_set_cookie",
                    "browser_print_pdf", "browser_network_log", "browser_console_log",
                    "browser_close"],
    "computer":    ["computer_use"],
    "delegation":  ["delegate_task", "delegate_batch"],
    "vision":      ["vision_analyze", "image_generate", "tts_speak",
                    "video_generate", "video_from_image", "video_list_generated"],
    "messaging":   ["telegram_send", "discord_send", "discord_get_messages",
                    "discord_create_webhook", "slack_send", "slack_get_messages",
                    "slack_list_channels", "slack_upload_file",
                    "whatsapp_send", "whatsapp_get_messages", "whatsapp_status",
                    "signal_send", "signal_receive", "signal_list_groups",
                    "matrix_send", "matrix_get_messages", "matrix_list_rooms",
                    "irc_send", "irc_get_messages",
                    "mattermost_send", "mattermost_get_messages", "mattermost_list_channels",
                    "teams_send", "teams_get_messages", "teams_list_teams"],
    "database":    ["db_query", "db_list_tables", "db_describe_table", "mongo_query"],
    "voice":       ["voice_record_and_transcribe", "voice_transcribe_file",
                    "voice_speak", "voice_list_voices"],
    "ssh":         ["ssh_exec", "ssh_upload", "ssh_download"],
    "git":         ["git_status", "git_diff", "git_log", "git_add", "git_commit",
                    "git_checkout", "git_branch", "git_stash"],
    "knowledge":   ["knowledge_set", "knowledge_get", "knowledge_delete", "knowledge_list"],
    "docker":      ["docker_run", "docker_run_code", "docker_list_containers", "docker_pull"],
    "cloud":       ["modal_run", "modal_status", "daytona_run", "daytona_list_workspaces"],
    "macros":      ["macro_save", "macro_delete", "macro_list", "run_macro"],
    "goals":       ["goal_set", "goal_update", "goal_list", "goal_complete", "goal_delete"],
    "agent":       ["sub_agent", "spawn_agent", "todo", "clarify"],
}

# Tools disabled in the current session (populated by enable/disable commands)
_DISABLED_TOOLS: set[str] = set()


def enable_toolset(group: str) -> list[str]:
    """Re-enable all tools in a toolset group. Returns list of newly enabled tool names."""
    names = TOOLSETS.get(group, [])
    restored = [n for n in names if n in _DISABLED_TOOLS]
    _DISABLED_TOOLS.difference_update(names)
    # Restore to _DISPATCH if they were removed
    for n in restored:
        fn = _get_original_fn(n)
        if fn is not None:
            _DISPATCH[n] = fn
    return restored


def disable_toolset(group: str) -> list[str]:
    """Disable all tools in a toolset group for this session. Returns list of disabled names."""
    names = TOOLSETS.get(group, [])
    newly_disabled = [n for n in names if n in _DISPATCH]
    _DISABLED_TOOLS.update(newly_disabled)
    for n in newly_disabled:
        _DISPATCH.pop(n, None)
    return newly_disabled


def _get_original_fn(tool_name: str):
    """Look up the original function for a tool name (for re-enabling)."""
    # Build once from module imports
    _MAP = {
        "file_read": file_read, "file_write": file_write, "file_append": file_append,
        "file_patch": file_patch, "file_delete": file_delete, "dir_list": dir_list,
        "file_exists": file_exists, "file_info": file_info, "shell_exec": shell_exec,
        "duckduckgo_search": duckduckgo_search, "web_scrape": web_scrape,
        "x_search": x_search, "python_exec": python_exec, "http_request": http_request,
        "file_search": file_search, "email_draft": email_draft,
        "browser_navigate": browser_navigate, "browser_snapshot": browser_snapshot,
        "browser_screenshot": browser_screenshot, "browser_click": browser_click,
        "browser_type": browser_type, "browser_scroll": browser_scroll,
        "browser_hover": browser_hover, "browser_key": browser_key,
        "browser_select": browser_select, "browser_fill_form": browser_fill_form,
        "browser_wait": browser_wait, "browser_extract_text": browser_extract_text,
        "browser_extract_links": browser_extract_links,
        "browser_extract_tables": browser_extract_tables,
        "browser_evaluate": browser_evaluate, "browser_new_tab": browser_new_tab,
        "browser_list_tabs": browser_list_tabs, "browser_switch_tab": browser_switch_tab,
        "browser_go_back": browser_go_back, "browser_go_forward": browser_go_forward,
        "browser_reload": browser_reload, "browser_get_cookies": browser_get_cookies,
        "browser_set_cookie": browser_set_cookie, "browser_print_pdf": browser_print_pdf,
        "browser_network_log": browser_network_log, "browser_console_log": browser_console_log,
        "browser_close": browser_close,
        "computer_use": computer_use,
        "delegate_task": delegate_task, "delegate_batch": delegate_batch,
        "vision_analyze": vision_analyze,
        "image_generate": image_generate, "tts_speak": tts_speak,
        "telegram_send": telegram_send, "clarify": clarify, "todo": todo,
        "knowledge_set": knowledge_set, "knowledge_get": knowledge_get,
        "knowledge_delete": knowledge_delete, "knowledge_list": knowledge_list,
        "ssh_exec": ssh_exec, "ssh_upload": ssh_upload, "ssh_download": ssh_download,
        "git_status": git_status, "git_diff": git_diff, "git_log": git_log,
        "git_add": git_add, "git_commit": git_commit, "git_checkout": git_checkout,
        "git_branch": git_branch, "git_stash": git_stash,
        "discord_send": discord_send, "discord_get_messages": discord_get_messages,
        "discord_create_webhook": discord_create_webhook,
        "slack_send": slack_send, "slack_get_messages": slack_get_messages,
        "slack_list_channels": slack_list_channels, "slack_upload_file": slack_upload_file,
        "db_query": db_query, "db_list_tables": db_list_tables,
        "db_describe_table": db_describe_table, "mongo_query": mongo_query,
        "voice_record_and_transcribe": voice_record_and_transcribe,
        "voice_transcribe_file": voice_transcribe_file, "voice_speak": voice_speak,
        "voice_list_voices": voice_list_voices,
        "whatsapp_send": whatsapp_send, "whatsapp_get_messages": whatsapp_get_messages,
        "whatsapp_status": whatsapp_status,
        "docker_run": docker_run, "docker_run_code": docker_run_code,
        "docker_list_containers": docker_list_containers, "docker_pull": docker_pull,
        "signal_send": signal_send, "signal_receive": signal_receive,
        "signal_list_groups": signal_list_groups,
        "matrix_send": matrix_send, "matrix_get_messages": matrix_get_messages,
        "matrix_list_rooms": matrix_list_rooms,
        "irc_send": irc_send, "irc_get_messages": irc_get_messages,
        "mattermost_send": mattermost_send, "mattermost_get_messages": mattermost_get_messages,
        "mattermost_list_channels": mattermost_list_channels,
        "teams_send": teams_send, "teams_get_messages": teams_get_messages,
        "teams_list_teams": teams_list_teams,
        "modal_run": modal_run, "modal_status": modal_status,
        "daytona_run": daytona_run, "daytona_list_workspaces": daytona_list_workspaces,
        "macro_save": macro_save, "macro_delete": macro_delete,
        "macro_list": macro_list, "run_macro": run_macro,
        "goal_set": goal_set, "goal_update": goal_update, "goal_list": goal_list,
        "goal_complete": goal_complete, "goal_delete": goal_delete,
        "llm_task": llm_task, "apply_patch": apply_patch,
    }
    return _MAP.get(tool_name)


class ToolRegistry:

    @property
    def tools(self) -> dict:
        return _DISPATCH

    def get_descriptions(self) -> str:
        """
        Return a formatted string listing all tools for the system prompt.
        Includes both built-in tools (_TOOL_DEFINITIONS) and any dynamically
        injected tools (e.g. from MCP servers) that are in _DISPATCH but not
        yet in _TOOL_DEFINITIONS.
        """
        lines = []
        defined_names = {td["name"] for td in _TOOL_DEFINITIONS}

        for td in _TOOL_DEFINITIONS:
            params_dict = _td_params(td)
            params_str = "\n".join(
                f"      {k}: {v}" for k, v in params_dict.items()
            )
            lines.append(
                f"  tool_name: \"{td['name']}\"\n"
                f"  description: {td['description']}\n"
                f"  params:\n{params_str if params_str else '      (none)'}"
            )

        # Include any dynamically registered tools (e.g. MCP tools) that are
        # in the dispatch map but don't have a static definition entry.
        for name in _DISPATCH:
            if name not in defined_names:
                lines.append(
                    f"  tool_name: \"{name}\"\n"
                    f"  description: (dynamically registered tool)\n"
                    f"  params:\n      (see server documentation)"
                )

        return "\n\n".join(lines)

    def get_compact_descriptions(self) -> str:
        """
        Single-line-per-tool listing for use in small-model (local) system prompts.
        Keeps token count low while still telling the model what tools exist.
        """
        lines = []
        for td in _TOOL_DEFINITIONS:
            # First sentence of description only
            short_desc = td["description"].split(".")[0].strip()
            # Required params — works for both "params" format and "input_schema" format
            params_dict = _td_params(td)
            required_set = _td_required(td)
            if required_set:
                req_params = [k for k in params_dict if k in required_set]
            else:
                req_params = [k for k, v in params_dict.items()
                              if "required" in str(v).lower()]
            params_hint = ", ".join(req_params) if req_params else "no required params"
            lines.append(f"  {td['name']:<28} {short_desc}  [{params_hint}]")
        return "\n".join(lines)

    def execute(self, tool_name: str, params: dict) -> Any:
        """
        Dispatch a tool call. Returns a dict with {success, output, error}.
        """
        fn = _DISPATCH.get(tool_name)
        if fn is None:
            return {
                "success": False,
                "output":  None,
                "error":   f"Unknown tool: '{tool_name}'. Available: {list(_DISPATCH.keys())}",
            }
        try:
            if not isinstance(params, dict):
                params = {}
            clean = {k: v for k, v in params.items() if v is not None}
            return fn(**clean)
        except TypeError as e:
            return {
                "success": False,
                "output":  None,
                "error":   f"Invalid params for '{tool_name}': {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "output":  None,
                "error":   f"Tool '{tool_name}' raised: {type(e).__name__}: {e}",
            }
