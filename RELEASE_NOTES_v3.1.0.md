# Operon v3.1.0 — Initial Public Beta

**Operon is an advanced autonomous AI Terminal Cockpit** — a fully agentic,
multi-provider terminal AI with 185+ tools, persistent memory, browser &
desktop automation, a multi-agent mesh, and a self-improvement loop.

This is the first public release. It's feature-rich and well-tested
(2,750+ tests passing), runs on macOS/Linux/Windows, and works fully offline
with local models via Ollama.

---

## ⚡ Install in one command

```bash
git clone https://github.com/OperonAgent/Operon.git
cd Operon
./install.sh            # macOS/Linux   (Windows: install.ps1)
operon
```

The installer provisions **everything** — Python deps **and** the Playwright
Chromium browser binary (the ~120 MB download a plain `pip install` skips).
Pre-built `Operon.app` / `operon.exe` / Linux binaries are attached below.

---

## ✨ Highlights

- **185+ tools** — files, shell, git, HTTP, databases, PDF, vision, image gen, TTS, **video gen**
- **8+ AI providers** — OpenAI, Anthropic, OpenRouter, Ollama, LM Studio, Jan, and more
- **Persistent memory** — FTS5 keyword + LanceDB semantic vectors + Obsidian vault sync, unified behind one facade
- **Browser & computer use** — Playwright automation + pyautogui desktop control, with runtime self-heal
- **SWE agent** — automated issue → fix → test → PR loop, now with LSP-style static analysis
- **Multi-agent mesh** — parallel/pipeline delegation with named specialist roles
- **Concurrent tool dispatch** — independent tool calls run in parallel
- **Voice pipeline** — speech-to-text, text-to-speech, real-time streaming STT
- **Self-improvement** — synthesizes new Python tools from conversation trajectories
- **Security** — `email_send` is structurally un-callable by the model; OSV.dev CVE dependency scanning in `/doctor`
- **60+ slash commands** — `/kanban`, `/checkpoint`, `/mesh`, `/swe`, `/voice`, `/doctor`, …
- **Claude Code-style TUI** — persistent input bar, token counter, model HUD
- **Plugin SDK** — manifest + tools, with 5 ready-to-use example plugins

---

## 🆕 Since the project's internal builds

- One-command installer + dependency bootstrapper with browser self-heal
- Supply-chain CVE auditing (`core/dep_audit.py`) wired into `/doctor`
- Unified memory facade over FTS5 + vector + Obsidian
- 5 seed example plugins (11 tools) demonstrating the SDK
- Video generation tool (Replicate / Luma)
- Static analysis in the SWE loop
- Began breaking up the monolithic `main.py` into a `cmd_handlers/` package
- 120+ new tests, including real agent-loop integration tests
- Full repo hygiene: LICENSE, README, CHANGELOG, CONTRIBUTING, SECURITY, ROADMAP

---

## 🩹 Fixes

- Repaired 6 broken slash commands (`/vector`, `/desktop`, `/synth`,
  `/checkpoint`, `/kanban list`, `/usage`, `/knowledge`)
- Restored the TUI `_extra_status` back-compat alias
- Hardened `/doctor` against import-time errors in optional deps

---

## 📋 Known limitations (tracked in [ROADMAP.md](ROADMAP.md))

- `main.py` modularization is in progress (read-only commands extracted; global-state commands pending)
- Messaging channels (Discord/Telegram/Slack) are functional but not yet at production depth
- Voice streaming to cloud realtime APIs is partial

---

## 🔧 Requirements

- Python 3.9+ (3.11+ recommended), macOS / Linux / Windows
- Optional: Ollama for fully-offline local models (no API key)

---

## 📚 Docs

- **Setup Guide:** `Operon_Setup_Guide.pdf`
- **Full reference:** `Operon_Documentation.pdf`
- **Comparison:** `Operon_Comparison.pdf`
- Build them: `make docs`

**Full Changelog:** https://github.com/OperonAgent/Operon/blob/main/CHANGELOG.md
