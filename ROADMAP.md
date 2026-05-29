# Operon Roadmap

Operon v3.1.0 is an **early public beta**. It is feature-rich and well-tested
(1,896 passing tests), but a few architectural improvements are intentionally
deferred to keep the beta stable rather than risk a last-minute rewrite.

## Known beta limitations

### 1. `main.py` is a monolith (~5,300 lines)
The command dispatcher (`handle_command`) and the agent loop live in a single
large module. It works and is tested, but it's hard to navigate and contribute
to.

**Plan:** Extract command handlers into a `cmd_handlers/` package
(`session.py`, `memory.py`, `infra.py`, `agents.py`, `services.py`) behind a
small dispatch table, keeping shared global state in one place. Pure refactor —
no behaviour change, guarded by the existing `test_slash_commands.py` suite.

**Target:** v3.2.0

### 2. Synchronous tool dispatch
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
