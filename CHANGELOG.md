# Changelog

All notable changes to Operon are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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

[3.1.0]: https://github.com/OWNER/operon/releases/tag/v3.1.0
