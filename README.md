<h1 align="center">OPERON</h1>
<p align="center"><strong>AI Terminal Cockpit</strong> — a fully agentic, multi-provider terminal AI with 185+ tools, persistent memory, browser &amp; desktop automation, and a self-improvement loop.</p>

<p align="center">
  <code>v3.1.0</code> · Phase 11 Build · 1,896 tests passing · MIT License
</p>

---

## Quick Install

The recommended installer handles **everything** — Python dependencies **and** the
Playwright Chromium browser binary (the ~120 MB download that `pip install` skips).

### macOS / Linux

```bash
git clone https://github.com/OWNER/operon.git
cd operon
./install.sh
```

### Windows

```powershell
git clone https://github.com/OWNER/operon.git
cd operon
powershell -ExecutionPolicy Bypass -File install.ps1
```

### Any platform (Python)

```bash
python install.py            # core + recommended + browser binary
python install.py --full     # also voice, databases, screen capture, etc.
```

That's it. The installer:
1. Verifies Python ≥ 3.9
2. Creates a `.venv` virtual environment
3. Installs Operon and its dependencies
4. **Downloads the Chromium browser binary** (the step everyone forgets)
5. Registers the `operon` command

Then launch:

```bash
operon          # or: python main.py
```

On first run a setup wizard configures your AI providers and API keys.

---

## What gets installed

| Component | Default install | `--full` install |
|-----------|:---------------:|:----------------:|
| Core REPL (requests, bs4, web search) | ✓ | ✓ |
| Telemetry, syntax highlighting, TUI | ✓ | ✓ |
| PDF reading + generation | ✓ | ✓ |
| **Playwright + Chromium browser binary** | ✓ | ✓ |
| SSH remote execution (paramiko) | ✓ | ✓ |
| Desktop computer use (mss, pynput) | — | ✓ |
| Voice (whisper STT, TTS) | — | ✓ |
| Databases (Postgres, MongoDB) | — | ✓ |
| Secrets keychain, MCP server | — | ✓ |

> **Why the browser is a separate step:** `pip install playwright` installs only the
> Python *package*. The actual Chromium *browser binary* requires a separate
> `playwright install chromium`. Operon's installer (and a runtime self-heal hook)
> do this for you — if you ever try to browse without it, Operon downloads it on demand.

---

## Verify your install

```bash
operon --check-deps          # dependency status report
operon  →  then type /doctor # full in-app health check
```

Re-run dependency provisioning anytime:

```bash
python -m core.bootstrap            # core + recommended + browser
python -m core.bootstrap --full     # everything
python -m core.bootstrap --browser  # just the Chromium binary
python -m core.bootstrap --check    # status only
```

---

## Makefile shortcuts

```bash
make install        # full install + browser binary
make install-full   # every optional feature
make browser        # just the Chromium binary
make check          # dependency status
make run            # launch Operon
make test           # run the test suite
```

---

## Pre-built apps (no Python needed)

Each tagged release ships standalone binaries via GitHub Actions:

| Platform | Download |
|----------|----------|
| macOS (Apple Silicon) | `Operon.app` (drag to Applications) |
| Windows (x64) | `operon.exe` |
| Linux (x64) | `operon` |

These bundle Python and all dependencies. *(Browser automation in the bundled app
still downloads Chromium on first use.)*

---

## Highlights

- **185+ tools** — files, shell, git, HTTP, databases, PDF, vision, image gen, TTS
- **8+ AI providers** — OpenAI, Anthropic, OpenRouter, Ollama, LM Studio, Jan, and more
- **Persistent memory** — FTS5 + LanceDB semantic vectors + Obsidian vault sync
- **Browser & computer use** — Playwright automation + pyautogui desktop control
- **SWE agent** — automated issue → fix → test → PR loop
- **Multi-agent mesh** — parallel/pipeline delegation with named specialist roles
- **Voice pipeline** — speech-to-text, text-to-speech, multimodal
- **Self-improvement** — synthesizes new Python tools from conversation trajectories
- **60+ slash commands** — `/kanban`, `/checkpoint`, `/mesh`, `/swe`, `/voice`, …
- **Claude Code-style TUI** — persistent input bar, token counter, model HUD

---

## Requirements

- Python 3.9+ (3.11+ recommended)
- macOS, Linux, or Windows
- Optional: an Ollama install for fully-offline local models (no API key needed)

---

## Documentation

Full technical reference: `Operon_Documentation.pdf` (generate with `python generate_docs.py`).

## License

MIT © 2026 Operon Project
