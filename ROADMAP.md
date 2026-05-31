# Operon Roadmap

Operon v3.1.0 is an **early public beta**. It is feature-rich and well-tested
(1,896 passing tests), but a few architectural improvements are intentionally
deferred to keep the beta stable rather than risk a last-minute rewrite.

## Known beta limitations

### 1. `main.py` is a monolith (~5,300 lines) — IN PROGRESS
The command dispatcher (`handle_command`) and the agent loop live in a single
large module.

**Done (v3.1.x):** Introduced the `cmd_handlers/` package with a
`CommandContext` + dispatch registry. `handle_command` now consults the modular
dispatch first and falls back to the legacy elif-chain for anything not yet
migrated. Read-only commands (`/clear`, `/undo`, `/history`, `/compress`,
`/tools`, `/usage`, `/cost`) are extracted into `cmd_handlers/info.py` and
`cmd_handlers/session_cmds.py`, covered by `tests/test_cmd_handlers.py`.

**Remaining:** Migrate the global-state-mutating commands (`/gateway`,
`/dashboard`, `/approve`, `/mcp`, `/webhook`, `/rag`, `/secrets`, `/heartbeat`,
`/goal`, `/macro`, `/plugin`, `/curator`) — these reassign main.py module
globals and need the context to carry mutable service handles. Then move the
agent loop into its own module.

**Target:** v3.2.0

### 2. Synchronous tool dispatch  (context compaction now async — done)
Tool calls run synchronously, so one slow network tool blocks the turn.

**Plan:** Move the tool-dispatch layer in `run_agent_loop()` to `asyncio`, using
`asyncio.gather()` for independent parallel tool calls.

**Target:** v3.2.0

### 3. Messaging channels not yet production-depth
Discord, Telegram, and Slack integrations are functional but lighter than
dedicated bots (no full thread/voice/embed parity).

**Plan:** Deepen each channel incrementally based on beta feedback.

**Target:** v3.3.0

## Post-beta direction (subject to feedback)

- Background (non-blocking) context compression wired into the main loop.
- Unified memory facade over FTS5 + vector + Obsidian backends.
- Streaming real-time voice (OpenAI Realtime / Deepgram).
- Plugin marketplace with seed community plugins.
- LSP / static-analysis integration in the SWE agent.
- Supply-chain dependency scanning in `/doctor`.

## How to influence the roadmap

Open a [feature request](.github/ISSUE_TEMPLATE/feature_request.md) or start a
GitHub Discussion. Beta feedback directly shapes priorities.
