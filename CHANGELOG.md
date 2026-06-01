# Changelog

All notable changes to Operon are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — 3.1.x — Competitiveness pass

### Added — Turn-completion notifications (harvested from Hermes)
- **`core/notify.py`**: rings the terminal bell (`\a`, propagates over SSH) and
  optionally raises a native desktop notification (macOS `osascript`, Linux
  `notify-send`, Windows PowerShell toast) when a turn finishes. Best-effort and
  config-gated — never raises, no-op unless enabled.
- Config keys `notify_on_complete` / `notify_desktop` / `notify_min_seconds`
  (the last suppresses alerts for quick turns).
- **`/notify on | off | desktop | test`** command; wired into the REPL turn
  boundary (interactive only — sub-agents stay silent).

### Added — Hierarchical multi-agent orchestration
- **Worker-tier personas** in `core/multi_agent.py`: `AgentRole.ENGINEER`
  (execution: file edits + code sandbox, steered to minimal diffs) and
  `AgentRole.AUDITOR` (cynical QA: linters/tests/logs/vuln review, with **no
  write tools** so its critique stays independent — constructive tension).
- **`spawn_agent(persona, objective, allocated_tools)` meta-tool** — a
  multi-agent factory that spins up a sandboxed worker restricted to exactly
  the tools it's handed. Wired via `set_agent_factory` to an `AgentMesh`. Added
  to `DELEGATE_BLOCKED_TOOLS` so workers can't recursively spawn (no fork bombs).
- **Autonomous self-correction loop** — `AgentMesh.run_self_correction()` and
  `/mesh fix <objective> [|| verify-cmd]`: Engineer drafts → verify (shell
  command and/or Auditor verdict) → Auditor turns the fault log into explicit
  fixes → Engineer applies them → re-verify, bounded by `max_rounds`.
- Sub-agent loops now run with `core/tool_guardrails.py` active (per-worker
  block/warn/halt) and hard sandbox enforcement, preventing multi-agent
  infinite loops. 4096 max_tokens preserved across sub-agent calls.



Real-engineering improvements raising Operon's depth in its weakest categories
(context handling, voice, messaging, architecture). No cosmetic changes.

### Added
- **Real-time cloud streaming STT** — `CloudStreamingTranscriber` in
  `core/voice_pipeline.py` streams microphone PCM to Deepgram's WebSocket API
  for sub-second interim + final transcripts. `VoicePipeline.stream_listen`
  now prefers the cloud streamer (when `DEEPGRAM_API_KEY` + `websocket-client`
  are present) and falls back to the windowed local transcriber otherwise.
  New `STTBackend.DEEPGRAM`.
- **Deeper Slack toolset** — thread reading (`slack_get_thread`), message
  editing (`slack_update_message`), scheduled sends (`slack_schedule_message`),
  pin/unpin (`slack_pin_message`), channel topic (`slack_set_topic`), and a
  Block Kit builder (`slack_build_blocks`).
- **Telegram agent tools** — exposed the existing `TelegramBot` capabilities as
  registry tools: `telegram_get_updates`, `telegram_edit_message`,
  `telegram_delete_message`, `telegram_pin_message`, `telegram_send_photo`,
  `telegram_send_document`.

### Changed
- **Non-blocking context compaction** wired into the agent loop
  (`main._background_compact` + `BackgroundCompressor`): summarization runs
  off-thread and merges on a later turn, with a synchronous hard-ceiling
  fallback.
- **`/macro` migrated** out of `main.py`'s elif-chain into
  `cmd_handlers/macro_cmds.py` — the first stateful command extracted, proving
  the live-module pattern for the rest of the modularization.

### Fixed
- Registry wiring gaps: only 4 of 17 Slack tools and 1 of 7 Telegram tools were
  actually reachable by the agent; all are now registered (185 tools, with
  definitions and dispatch kept in sync).

### Tests
- +66 tests: streaming voice (11), Slack depth (21), Telegram wrappers (20),
  `/macro` migration (14).

## [3.1.0] — 2026-05-29 — Initial Public Beta

First public release. Operon is an agentic AI terminal cockpit with 185+ tools,
8+ AI providers, persistent memory, browser & desktop automation, a SWE agent
loop, multi-agent delegation, and a self-improvement skill synthesizer.

### Added
- **One-shot installer** (`install.py`, `install.sh`, `install.ps1`) that
  provisions Python deps **and** the Playwright Chromium browser binary.
- **`core/bootstrap.py`** dependency provisioner with runtime self-heal:
  the browser tool auto-installs Chromium on first use if missing.
- **CLI flags**: `operon --install-deps`, `--check-deps`, `--skip-dep-check`.
- **Claude Code-style TUI** (`ui/tui.py`) with persistent input bar, token
  counter, and model HUD (prompt_toolkit).
- **Semantic vector memory** (LanceDB + SentenceTransformers) and **Obsidian
  vault sync**.
- **Smart model router** with per-turn `hint:code/fast/reasoning` selection and
  background Ollama discovery.
- **Skill synthesizer** — generates new Python tools from conversation
  trajectories at runtime.
- **Desktop computer use** (pyautogui + mss): mouse, keyboard, screenshots,
  template matching.
- **SWE agent loop**, **voice pipeline** (STT/TTS/VAD), **multi-agent mesh**.
- **SQLite Kanban board**, **git checkpoint manager**, **credential pool**.
- **macOS `.app` / Windows `.exe` / Linux** packaging via PyInstaller, plus a
  GitHub Actions release workflow.
- **1,896 passing tests** across 42 test files, including new
  `test_slash_commands.py`, `test_phase11_coverage.py`, and `test_bootstrap.py`.

### Fixed
- `/vector`, `/desktop`, `/synth` crashed with `NameError: args` — added the
  missing `args` binding in `handle_command`.
- `/checkpoint` crashed with `UnboundLocalError: os` — a bare `import os`
  inside the handler shadowed the module-level import.
- `/kanban list` crashed with `KeyError: 'total'` — handler now reads `count`.
- `/usage` crashed on malformed stats — values are now defensively coerced.
- `/knowledge` crashed when no knowledge base was initialised — added a guard.
- `ui/tui.py` restored the `_extra_status` back-compat alias.

### Known limitations (tracked for post-beta — see ROADMAP.md)
- `main.py` is a large monolith; command handlers will be modularised.
- Tool dispatch is synchronous; async dispatch is planned.
- Messaging channels (Discord/Telegram/Slack) are functional but not yet at
  production depth.

[3.1.0]: https://github.com/OperonAgent/Operon/releases/tag/v3.1.0
