"""
Operon Setup Wizard — V3.

Comprehensive first-run (and re-run) configuration covering:
   1. Provider selection     — cloud and/or local
   2. API keys               — OpenAI, Anthropic, OpenRouter
   3. Local server           — Ollama / LM Studio auto-detection
   4. Default model          — picked from everything available
   5. Agent behaviour        — max iterations, request timeout
   6. Memory & Mem0          — built-in + Mem0 external provider
   7. Soul / personality     — create or keep existing
   8. Messaging channels     — Telegram, Discord, Slack, WhatsApp,
                               Signal, Matrix, IRC, Mattermost, Teams
   9. Cloud execution        — Modal, Daytona, Docker check
  10. Webhook & remote       — REST API server, bearer token
  11. Heartbeat scheduler    — passive background tick loop
  12. Summary                — all settings confirmed, cheat-sheet printed

Re-entrant: press ENTER to keep any existing value.
"""

import os
import re
import sys
import time

from core.config import (
    ConfigManager, DEFAULT_PROFILES,
    LOCAL_PROVIDERS, LOCAL_HEALTH_URLS, PROVIDER_URLS,
)

# ── ANSI helpers ──────────────────────────────────────────────────────────────
_R  = "\033[0m"
_B  = "\033[1m"
_D  = "\033[2m"
_C  = "\033[1;38;5;81m"    # neon cyan
_P  = "\033[1;38;5;99m"    # purple
_PL = "\033[1;38;5;141m"   # light purple
_A  = "\033[38;5;214m"     # amber
_G  = "\033[38;5;82m"      # green
_DG = "\033[38;5;244m"     # dim gray
_W  = "\033[1;38;5;255m"   # white
_RE = "\033[38;5;196m"     # red

def _h(text):   return f"{_B}{_C}{text}{_R}"
def _label(t):  return f"{_A}{t}{_R}"
def _ok(t):     return f"  {_G}✓{_R}  {t}"
def _skip(t):   return f"  {_DG}–{_R}  {_DG}{t}{_R}"
def _warn(t):   return f"  {_A}⚠{_R}  {t}"
def _dot(t):    return f"  {_PL}●{_R}  {t}"
def _sep():     print(f"\n{_DG}{'─' * 62}{_R}\n")
def _plain(s):  return re.sub(r'\033\[[0-9;]*m', '', s)


def _ask(prompt: str, default: str = "") -> str:
    hint = f" {_DG}[{default}]{_R}" if default else ""
    try:
        val = input(f"  {_A}▸{_R} {_W}{prompt}{hint}{_R}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return val if val else default


def _ask_secret(prompt: str) -> str:
    import getpass
    try:
        val = getpass.getpass(f"  {_A}▸{_R} {_W}{prompt}{_R}: ").strip()
    except Exception:
        val = _ask(prompt)
    return val


def _ask_yn(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = _ask(f"{prompt} [{hint}]", "").lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


def _banner():
    print(f"""
{_P}╔══════════════════════════════════════════════════════════╗{_R}
{_P}║{_R}                                                          {_P}║{_R}
{_P}║{_R}   {_C}{_B}  ██████╗ ██████╗ ███████╗██████╗  ██████╗ ███╗{_R}   {_C}{_B}██╗{_R}  {_P}║{_R}
{_P}║{_R}   {_C}{_B} ██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔═══██╗████╗{_R}  {_C}{_B}██║{_R}  {_P}║{_R}
{_P}║{_R}   {_C}{_B} ██║   ██║██████╔╝█████╗  ██████╔╝██║   ██║██╔██╗{_R} {_C}{_B}██║{_R}  {_P}║{_R}
{_P}║{_R}   {_C}{_B} ██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██║   ██║██║╚██╗{_R}{_C}{_B}██║{_R}  {_P}║{_R}
{_P}║{_R}   {_C}{_B} ╚██████╔╝██║     ███████╗██║  ██║╚██████╔╝██║ ╚████║{_R}  {_P}║{_R}
{_P}║{_R}   {_C}{_B}  ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝{_R}  {_P}║{_R}
{_P}║{_R}                                                          {_P}║{_R}
{_P}║{_R}          {_PL}S E T U P   W I Z A R D   V 3{_R}                  {_P}║{_R}
{_P}║{_R}                                                          {_P}║{_R}
{_P}╚══════════════════════════════════════════════════════════╝{_R}
""")


def _section(n: int, total: int, title: str):
    bar = "─" * max(0, 62 - len(f"── Step {n}/{total}  {title} ") - 2)
    print(f"{_C}{_B}── Step {n}/{total}  {title} {bar}{_R}")


# ── Local server detection ────────────────────────────────────────────────────

def _probe_servers() -> dict:
    """Quick probe of all local servers. Returns {prov: {running, models}}."""
    try:
        import requests as _req
    except ImportError:
        return {}

    results = {}
    for prov, url in LOCAL_HEALTH_URLS.items():
        if prov == "local":
            continue
        entry = {"running": False, "models": []}
        try:
            r = _req.get(url, timeout=1.0)
            if r.ok:
                entry["running"] = True
                data = r.json()
                if "models" in data:
                    entry["models"] = [m.get("name", "") for m in data.get("models", [])]
                elif "data" in data:
                    entry["models"] = [m.get("id", "") for m in data.get("data", [])]
        except Exception:
            pass
        results[prov] = entry
    return results


# ── Step helpers ──────────────────────────────────────────────────────────────

def _step_cloud_keys(config: ConfigManager, use_cloud: bool) -> dict:
    """Returns {provider: key_was_set}."""
    saved = {}
    if not use_cloud:
        return saved

    providers = [
        ("openai",     "OpenAI",     "gpt-4o, gpt-4o-mini, gpt-4-turbo",         "OPENAI_API_KEY"),
        ("anthropic",  "Anthropic",  "claude-sonnet-4, claude-opus-4",            "ANTHROPIC_API_KEY"),
        ("openrouter", "OpenRouter", "mistral, llama, deepseek, gemini, qwen …",  "OPENROUTER_API_KEY"),
    ]

    for prov, label, models, env_var in providers:
        print(f"\n  {_W}{label}{_R}  {_DG}({models}){_R}")
        existing = config.get_api_key(prov)

        if os.environ.get(env_var):
            print(_ok(f"Key found in env var {env_var} — skipping entry."))
            saved[prov] = True
            continue

        hint = ("●●●●" + existing[-4:]) if len(existing) > 4 else ""
        prompt = f"{label} API key{' (current: ' + hint + ')' if hint else ' (leave blank to skip)'}"
        key = _ask_secret(prompt)
        if key:
            config.set_api_key(prov, key)
            print(_ok(f"{label} key saved."))
            saved[prov] = True
        elif existing:
            print(_ok(f"Keeping existing {label} key."))
            saved[prov] = True
        else:
            print(_skip(f"{label} skipped."))
            saved[prov] = False

    return saved


def _step_local(config: ConfigManager, servers: dict) -> list:
    """Configure local providers. Returns list of usable local model names."""
    local_models = []
    running = {p: s for p, s in servers.items() if s["running"]}

    if running:
        print(f"\n  {_G}Detected running local servers:{_R}")
        for prov, info in running.items():
            url = PROVIDER_URLS.get(prov, "")
            print(f"    {_G}●{_R} {_W}{prov:<10}{_R} {_DG}{url}{_R}")
            for m in info["models"][:6]:
                print(f"        {_DG}↳{_R} {m}")
                local_models.append(f"{prov}:{m.split(':')[0]}")
    else:
        print(f"\n  {_DG}No local servers detected.{_R}")
        print(f"  {_DG}Install Ollama: https://ollama.com  |  then: ollama pull llama3.2{_R}")

    current_ollama = PROVIDER_URLS.get("ollama", "http://localhost:11434/v1/chat/completions")
    if _ask_yn(f"\n  Customise the Ollama server URL?", default=False):
        new_url = _ask("Ollama base URL", current_ollama)
        PROVIDER_URLS["ollama"] = new_url
        PROVIDER_URLS["local"]  = new_url
        print(_ok(f"Ollama URL set to {new_url}"))

    if _ask_yn(f"  Customise the LM Studio server URL?", default=False):
        current = PROVIDER_URLS.get("lmstudio", "http://localhost:1234/v1/chat/completions")
        new_url = _ask("LM Studio base URL", current)
        PROVIDER_URLS["lmstudio"] = new_url
        print(_ok(f"LM Studio URL set to {new_url}"))

    return local_models


def _step_default_model(config: ConfigManager, cloud_saved: dict,
                        local_models: list) -> str:
    print()
    options = []
    cloud_groups = [
        ("openai",     ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]),
        ("anthropic",  ["claude-sonnet-4", "claude-opus-4", "claude-3-5-sonnet", "claude-3-5-haiku"]),
        ("openrouter", ["deepseek-v3", "llama-3.1-70b", "mistral-large", "qwen-2.5-72b"]),
    ]
    for prov, names in cloud_groups:
        if cloud_saved.get(prov):
            for n in names:
                if n in DEFAULT_PROFILES:
                    options.append((n, DEFAULT_PROFILES[n]["model_id"], prov))

    for lm in local_models[:8]:
        prov, model_id = lm.split(":", 1)
        options.append((lm, model_id, prov))

    if not options:
        for name, p in DEFAULT_PROFILES.items():
            options.append((name, p["model_id"], p["provider"]))

    current = config.get("default_model", "gpt-4o")
    print(f"  {_W}Available models:{_R}")
    for i, (profile, model_id, prov) in enumerate(options, 1):
        marker = f"{_G}>>>{_R}" if profile == current else "   "
        local_tag = f" {_A}[local]{_R}" if prov in LOCAL_PROVIDERS else ""
        print(f"    {marker} {_DG}{i:2d}.{_R} {_W}{profile:<30}{_R} {_DG}{prov}{_R}{local_tag}")

    print()
    choice = _ask("Model number or name", current)

    if choice.isdigit():
        idx = int(choice) - 1
        chosen = options[idx][0] if 0 <= idx < len(options) else current
    else:
        chosen = choice

    if ":" in chosen and chosen not in DEFAULT_PROFILES:
        prov, model_id = chosen.split(":", 1)
        if prov in LOCAL_PROVIDERS:
            profiles = config.get("model_profiles", {})
            profiles[chosen] = {"provider": prov, "model_id": model_id}
            config.set("model_profiles", profiles)

    return chosen


def _step_agent_behaviour(config: ConfigManager):
    print()
    iters = _ask(
        "Max tool iterations per prompt  (higher = more autonomous, default 12)",
        str(config.get("max_tool_iters", 12)),
    )
    try:
        config.set("max_tool_iters", max(1, min(50, int(iters))))
        print(_ok(f"Max iterations: {config.get('max_tool_iters')}"))
    except ValueError:
        print(_skip("Keeping current max iterations."))

    timeout = _ask(
        "API request timeout in seconds  (default 120)",
        str(config.get("request_timeout", 120)),
    )
    try:
        config.set("request_timeout", max(10, int(timeout)))
        print(_ok(f"Request timeout: {config.get('request_timeout')}s"))
    except ValueError:
        print(_skip("Keeping current timeout."))


def _step_memory(config: ConfigManager):
    """Step 6 — built-in memory + Mem0 external provider."""
    print()
    current = config.get("memory_enabled", True)
    on = _ask_yn(
        "Enable long-term memory?  "
        f"{_DG}(remembers preferences & facts across sessions){_R}",
        default=current,
    )
    config.set("memory_enabled", on)
    print(_ok(f"Memory pipeline: {'ON' if on else 'OFF'}"))

    # Mem0 external provider
    print(f"\n  {_DG}Mem0 is an optional cloud memory layer (cross-session, semantic).{_R}")
    print(f"  {_DG}Get a free API key at https://mem0.ai{_R}\n")
    existing_key = os.environ.get("MEM0_API_KEY", "") or config.get("mem0_api_key", "")
    if existing_key:
        print(_ok(f"Mem0 API key already set (●●●●{existing_key[-4:]})."))
        if _ask_yn("  Update it?", default=False):
            new_key = _ask_secret("New Mem0 API key (m0-...)")
            if new_key:
                config.set("mem0_api_key", new_key)
                print(_ok("Mem0 key updated."))
    else:
        wants_mem0 = _ask_yn("  Configure Mem0?", default=False)
        if wants_mem0:
            key = _ask_secret("Mem0 API key (from mem0.ai)")
            if key:
                config.set("mem0_api_key", key)
                print(_ok("Mem0 key saved.  Set MEM0_API_KEY in your shell to activate."))
                uid = _ask("Mem0 user ID  (leave blank for default 'operon_user')", "operon_user")
                config.set("mem0_user_id", uid)
                print(_ok(f"Mem0 user ID: {uid}"))
            else:
                print(_skip("Mem0 skipped."))
        else:
            print(_skip("Mem0 skipped — built-in SemanticMemory is active."))


def _step_soul(config: ConfigManager):
    from pathlib import Path
    soul_path = Path.home() / ".operon" / "soul.md"
    print()
    if soul_path.exists():
        size = soul_path.stat().st_size
        print(f"  {_DG}Soul file exists: {soul_path}  ({size} bytes){_R}")
        if _ask_yn("  Open it in your editor now?", default=False):
            editor = os.environ.get("EDITOR", "nano")
            import subprocess
            subprocess.run([editor, str(soul_path)])
            print(_ok("Soul file updated."))
        else:
            print(_skip("Keeping existing soul file.  Edit anytime with /soul edit"))
    else:
        print(f"  {_DG}No soul file yet — a default will be created on first launch.{_R}")
        print(f"  {_DG}Customise Operon's personality with /soul edit after launch.{_R}")


def _step_messaging(config: ConfigManager):
    """Step 8 — configure all supported messaging channels."""
    print(f"\n  {_DG}Configure messaging channels so Operon can send and receive messages.{_R}")
    print(f"  {_DG}Leave blank to skip any channel — all can be set later via env vars.{_R}\n")

    # ── Telegram ──────────────────────────────────────────────────────────────
    print(f"  {_PL}● Telegram{_R}  {_DG}(Create a bot at https://t.me/BotFather){_R}")
    existing_token = config.get("telegram_token", "") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if existing_token:
        print(_ok(f"Token already set (●●●●{existing_token[-6:]})."))
        if _ask_yn("  Update it?", default=False):
            t = _ask_secret("New Telegram bot token")
            if t:
                config.set("telegram_token", t)
                print(_ok("Telegram token updated."))
    else:
        if _ask_yn("  Set up Telegram?", default=False):
            t = _ask_secret("Telegram bot token")
            if t:
                config.set("telegram_token", t)
                allowed = _ask(
                    "Allowed Telegram user IDs (comma-separated, blank = all)", "")
                if allowed.strip():
                    config.set("telegram_allowed_users", allowed.strip())
                print(_ok("Telegram configured."))
            else:
                print(_skip("Skipped."))
        else:
            print(_skip("Skipped."))

    # ── Discord ───────────────────────────────────────────────────────────────
    print(f"\n  {_PL}● Discord{_R}  {_DG}(DISCORD_BOT_TOKEN — https://discord.com/developers){_R}")
    if os.environ.get("DISCORD_BOT_TOKEN"):
        print(_ok("DISCORD_BOT_TOKEN found in environment."))
    else:
        if _ask_yn("  Configure Discord?", default=False):
            token = _ask_secret("Discord bot token")
            if token:
                config.set("discord_bot_token", token)
                gid = _ask("Default Guild (server) ID (optional)", "")
                if gid:
                    config.set("discord_guild_id", gid)
                cid = _ask("Default Channel ID (optional)", "")
                if cid:
                    config.set("discord_channel_id", cid)
                print(_ok("Discord configured.  "
                          f"{_DG}Set DISCORD_BOT_TOKEN in your shell env.{_R}"))
            else:
                print(_skip("Skipped."))
        else:
            print(_skip("Skipped."))

    # ── Slack ─────────────────────────────────────────────────────────────────
    print(f"\n  {_PL}● Slack{_R}  {_DG}(SLACK_BOT_TOKEN — https://api.slack.com/apps){_R}")
    if os.environ.get("SLACK_BOT_TOKEN"):
        print(_ok("SLACK_BOT_TOKEN found in environment."))
    else:
        if _ask_yn("  Configure Slack?", default=False):
            token = _ask_secret("Slack bot token (xoxb-...)")
            if token:
                config.set("slack_bot_token", token)
                ch = _ask("Default Slack channel (e.g. general)", "general")
                config.set("slack_default_channel", ch)
                print(_ok(f"Slack configured.  Default channel: {ch}"))
            else:
                print(_skip("Skipped."))
        else:
            print(_skip("Skipped."))

    # ── WhatsApp ──────────────────────────────────────────────────────────────
    print(f"\n  {_PL}● WhatsApp{_R}  {_DG}(via Twilio — https://twilio.com){_R}")
    if os.environ.get("TWILIO_ACCOUNT_SID"):
        print(_ok("TWILIO_ACCOUNT_SID found in environment."))
    else:
        if _ask_yn("  Configure WhatsApp / Twilio?", default=False):
            sid = _ask_secret("Twilio Account SID")
            tok = _ask_secret("Twilio Auth Token")
            if sid and tok:
                config.set("twilio_account_sid", sid)
                config.set("twilio_auth_token", tok)
                from_num = _ask("WhatsApp From number (e.g. whatsapp:+14155238886)",
                                "whatsapp:+14155238886")
                config.set("twilio_whatsapp_from", from_num)
                to_num   = _ask("Default To number (e.g. whatsapp:+15551234567)", "")
                if to_num:
                    config.set("twilio_whatsapp_to", to_num)
                print(_ok("WhatsApp via Twilio configured."))
            else:
                print(_skip("Skipped."))
        else:
            print(_skip("Skipped."))

    # ── Signal ────────────────────────────────────────────────────────────────
    print(f"\n  {_PL}● Signal{_R}  {_DG}(requires signal-cli — https://github.com/AsamK/signal-cli){_R}")
    if os.environ.get("SIGNAL_NUMBER"):
        print(_ok(f"SIGNAL_NUMBER found: {os.environ.get('SIGNAL_NUMBER')}"))
    else:
        if _ask_yn("  Configure Signal?", default=False):
            cli_path = _ask("signal-cli binary path", "/usr/local/bin/signal-cli")
            number   = _ask("Your registered Signal phone number (e.g. +15551234567)", "")
            if number:
                config.set("signal_cli_path", cli_path)
                config.set("signal_number",   number)
                recipient = _ask("Default recipient number (optional)", "")
                if recipient:
                    config.set("signal_recipient", recipient)
                print(_ok("Signal configured.  "
                          f"{_DG}Set SIGNAL_NUMBER & SIGNAL_CLI_PATH in your shell.{_R}"))
            else:
                print(_skip("Skipped."))
        else:
            print(_skip("Skipped."))

    # ── Matrix ────────────────────────────────────────────────────────────────
    print(f"\n  {_PL}● Matrix{_R}  {_DG}(MATRIX_HOMESERVER + credentials — no pip install needed){_R}")
    if os.environ.get("MATRIX_HOMESERVER"):
        print(_ok("MATRIX_HOMESERVER found in environment."))
    else:
        if _ask_yn("  Configure Matrix?", default=False):
            homeserver = _ask("Matrix homeserver URL", "https://matrix.org")
            user       = _ask("Matrix user ID (e.g. @you:matrix.org)", "")
            if homeserver and user:
                config.set("matrix_homeserver", homeserver)
                config.set("matrix_user", user)
                access_tok = _ask_secret(
                    "Access token (leave blank to use password login instead)")
                if access_tok:
                    config.set("matrix_access_token", access_tok)
                else:
                    password = _ask_secret("Matrix password")
                    config.set("matrix_password", password)
                room_id = _ask("Default room ID (e.g. !abc:matrix.org)", "")
                if room_id:
                    config.set("matrix_room_id", room_id)
                print(_ok("Matrix configured."))
            else:
                print(_skip("Skipped."))
        else:
            print(_skip("Skipped."))

    # ── Mattermost ────────────────────────────────────────────────────────────
    print(f"\n  {_PL}● Mattermost{_R}  {_DG}(MATTERMOST_URL + token — no pip install needed){_R}")
    if os.environ.get("MATTERMOST_URL"):
        print(_ok("MATTERMOST_URL found in environment."))
    else:
        if _ask_yn("  Configure Mattermost?", default=False):
            url   = _ask("Mattermost URL (e.g. https://mattermost.yourco.com)", "")
            token = _ask_secret("Personal access token")
            if url and token:
                config.set("mattermost_url",   url)
                config.set("mattermost_token", token)
                ch = _ask("Default channel name (e.g. town-square)", "town-square")
                config.set("mattermost_default_channel", ch)
                print(_ok("Mattermost configured."))
            else:
                print(_skip("Skipped."))
        else:
            print(_skip("Skipped."))

    # ── Microsoft Teams ───────────────────────────────────────────────────────
    print(f"\n  {_PL}● Microsoft Teams{_R}  {_DG}(webhook mode — no app registration needed){_R}")
    if os.environ.get("TEAMS_WEBHOOK_URL"):
        print(_ok("TEAMS_WEBHOOK_URL found in environment."))
    else:
        if _ask_yn("  Configure Teams?", default=False):
            webhook = _ask(
                "Incoming webhook URL  "
                "(Teams channel → ··· → Connectors → Incoming Webhook)", "")
            if webhook:
                config.set("teams_webhook_url", webhook)
                print(_ok("Teams webhook configured.  "
                          "Add TEAMS_WEBHOOK_URL to your shell env."))
                if _ask_yn(
                    "  Also configure Graph API (needed to read messages)?",
                    default=False
                ):
                    tid = _ask("Azure Tenant ID", "")
                    cid = _ask("Azure App Client ID", "")
                    sec = _ask_secret("Azure App Client Secret")
                    if tid and cid and sec:
                        config.set("teams_tenant_id",     tid)
                        config.set("teams_client_id",     cid)
                        config.set("teams_client_secret", sec)
                        print(_ok("Teams Graph API configured."))
                    else:
                        print(_skip("Graph API skipped."))
            else:
                print(_skip("Skipped."))
        else:
            print(_skip("Skipped."))

    # ── IRC ───────────────────────────────────────────────────────────────────
    print(f"\n  {_PL}● IRC{_R}  {_DG}(pure socket — no pip install needed){_R}")
    if os.environ.get("IRC_SERVER"):
        print(_ok("IRC_SERVER found in environment."))
    else:
        if _ask_yn("  Configure IRC?", default=False):
            server  = _ask("IRC server hostname (e.g. irc.libera.chat)", "")
            if server:
                port    = _ask("Port (6667 plain / 6697 SSL)", "6697")
                nick    = _ask("Nickname", "operon_bot")
                channel = _ask("Default channel (e.g. #general)", "")
                config.set("irc_server",  server)
                config.set("irc_port",    port)
                config.set("irc_nick",    nick)
                if channel:
                    config.set("irc_channel", channel)
                password = _ask_secret("NickServ password (optional, blank to skip)")
                if password:
                    config.set("irc_password", password)
                print(_ok(f"IRC configured: {nick}@{server}:{port}"))
            else:
                print(_skip("Skipped."))
        else:
            print(_skip("Skipped."))


def _step_cloud_execution(config: ConfigManager):
    """Step 9 — Modal, Daytona, Docker check."""
    print(f"\n  {_DG}Cloud execution lets Operon run code on remote workers with no server setup.{_R}\n")

    # ── Docker check ──────────────────────────────────────────────────────────
    print(f"  {_PL}● Docker{_R}  {_DG}(sandboxed local execution — no credentials needed){_R}")
    import shutil
    if shutil.which("docker"):
        print(_ok("docker binary found in PATH."))
    else:
        print(_warn("docker not found.  Install from https://docs.docker.com/get-docker/"))

    try:
        import docker as _docker_sdk
        print(_ok("docker Python SDK installed  (pip install docker)."))
    except ImportError:
        print(f"  {_DG}ℹ docker SDK not installed — Operon falls back to docker CLI.{_R}")
        print(f"  {_DG}  pip install docker{_R}")

    # ── Modal ─────────────────────────────────────────────────────────────────
    print(f"\n  {_PL}● Modal{_R}  {_DG}(serverless Python — GPU optional, modal.com){_R}")
    modal_ok = False
    try:
        import modal  # noqa: F401
        import subprocess
        r = subprocess.run(["modal", "profile", "current"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            print(_ok(f"Modal installed and authenticated  ({r.stdout.strip()})."))
            modal_ok = True
        else:
            print(_warn("Modal installed but not authenticated.  Run: modal setup"))
    except ImportError:
        if _ask_yn("  Install Modal?", default=False):
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "modal"],
                           check=False)
            print(_ok("modal installed.  Run 'modal setup' to authenticate."))
        else:
            print(_skip("Skipped.  Run: pip install modal && modal setup"))

    # ── Daytona ───────────────────────────────────────────────────────────────
    print(f"\n  {_PL}● Daytona{_R}  {_DG}(managed dev workspaces — daytona.io){_R}")
    existing_key = os.environ.get("DAYTONA_API_KEY", "") or config.get("daytona_api_key", "")
    if existing_key:
        print(_ok(f"DAYTONA_API_KEY already set (●●●●{existing_key[-6:]})."))
    else:
        if _ask_yn("  Configure Daytona?", default=False):
            key = _ask_secret("Daytona API key (from https://app.daytona.io)")
            if key:
                config.set("daytona_api_key", key)
                srv = _ask("Daytona server URL", "https://app.daytona.io")
                config.set("daytona_server_url", srv)
                print(_ok("Daytona configured.  "
                          "Set DAYTONA_API_KEY and DAYTONA_SERVER_URL in your shell."))
            else:
                print(_skip("Skipped."))
        else:
            print(_skip("Skipped."))


def _step_webhook(config: ConfigManager):
    """Step 10 — REST webhook server & bearer token."""
    print(f"\n  {_DG}The webhook server exposes Operon as a REST API at http://0.0.0.0:7271.{_R}")
    print(f"  {_DG}Send prompts via POST /chat or run batches via POST /batch.{_R}\n")

    current_enabled = config.get("webhook_autostart", False)
    autostart = _ask_yn("  Auto-start webhook server on launch?", default=current_enabled)
    config.set("webhook_autostart", autostart)

    if autostart:
        port = _ask("Port", str(config.get("webhook_port", 7271)))
        try:
            config.set("webhook_port", int(port))
        except ValueError:
            pass
        print(_ok(f"Webhook server will start on port {config.get('webhook_port', 7271)}."))

    existing_tok = config.get("webhook_token", "") or os.environ.get("OPERON_WEBHOOK_TOKEN", "")
    if existing_tok:
        print(_ok(f"Bearer token already set (●●●●{existing_tok[-6:]})."))
        if _ask_yn("  Regenerate it?", default=False):
            import secrets as _sec
            new_tok = _sec.token_urlsafe(32)
            config.set("webhook_token", new_tok)
            print(_ok(f"New token: {_A}{new_tok}{_R}"))
            print(f"  {_DG}Export as OPERON_WEBHOOK_TOKEN to use it.{_R}")
    else:
        if _ask_yn("  Generate a bearer token for webhook auth?", default=True):
            import secrets as _sec
            tok = _sec.token_urlsafe(32)
            config.set("webhook_token", tok)
            print(_ok(f"Bearer token: {_A}{tok}{_R}"))
            print(f"  {_DG}Save this somewhere safe — it won't be shown again.{_R}")
            print(f"  {_DG}Set OPERON_WEBHOOK_TOKEN in your shell or .env file.{_R}")
        else:
            print(_skip("No bearer token — webhook will be unauthenticated."))


def _step_heartbeat(config: ConfigManager):
    """Step 11 — passive background heartbeat scheduler."""
    from pathlib import Path
    hb_path = Path.home() / ".operon" / "HEARTBEAT.md"

    print(f"\n  {_DG}The heartbeat scheduler fires at a regular interval and runs the{_R}")
    print(f"  {_DG}contents of HEARTBEAT.md as an agent prompt — like a cron job that{_R}")
    print(f"  {_DG}has full agent context, memory, and tool access.{_R}\n")

    current_enabled = config.get("heartbeat_enabled", False)
    enable = _ask_yn("  Enable heartbeat scheduler?", default=current_enabled)
    config.set("heartbeat_enabled", enable)

    if enable:
        interval = _ask(
            "Interval between ticks in seconds  (1800 = 30 min, 3600 = 1 hr)",
            str(config.get("heartbeat_interval", 1800)),
        )
        try:
            config.set("heartbeat_interval", max(60, int(interval)))
            print(_ok(f"Heartbeat interval: {config.get('heartbeat_interval')}s"))
        except ValueError:
            pass

        business_hours = _ask_yn(
            "  Only run during business hours?  (Mon–Fri 09:00–18:00 local)",
            default=config.get("heartbeat_business_hours", False),
        )
        config.set("heartbeat_business_hours", business_hours)
        if business_hours:
            print(_ok("Business-hours-only mode enabled."))

        if not hb_path.exists():
            hb_path.parent.mkdir(parents=True, exist_ok=True)
            hb_path.write_text(
                "# HEARTBEAT\n\n"
                "Check the project status, look for anything urgent or blocked,\n"
                "and surface a short summary of what needs attention.\n",
                encoding="utf-8",
            )
            print(_ok(f"HEARTBEAT.md created at {hb_path}"))
            print(f"  {_DG}Edit it to customise what runs on each tick.{_R}")
        else:
            print(_ok(f"HEARTBEAT.md already exists at {hb_path}"))
            if _ask_yn("  Open it in your editor now?", default=False):
                editor = os.environ.get("EDITOR", "nano")
                import subprocess
                subprocess.run([editor, str(hb_path)])
                print(_ok("HEARTBEAT.md updated."))
    else:
        print(_skip("Heartbeat disabled.  Enable later with /heartbeat start"))


def _print_summary(config: ConfigManager, servers: dict):
    """Full configuration summary box."""
    model    = config.get("default_model", "?")
    prov     = config.get("active_provider", "?")
    mem      = "ON" if config.get("memory_enabled", True) else "OFF"
    mem0_key = config.get("mem0_api_key", "") or os.environ.get("MEM0_API_KEY", "")
    iters    = config.get("max_tool_iters", 12)
    timeout  = config.get("request_timeout", 120)

    keys = {}
    for p in ("openai", "anthropic", "openrouter"):
        k = config.get_api_key(p)
        keys[p] = "SET" if k else "—"

    local_status = [name for name, info in servers.items() if info["running"]]
    local_str    = "  ".join(f"{n} ✓" for n in local_status) or "none detected"

    # Messaging channel status
    channels = []
    tg_tok = config.get("telegram_token", "") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if tg_tok:             channels.append("Telegram")
    if (config.get("discord_bot_token") or os.environ.get("DISCORD_BOT_TOKEN")):
                           channels.append("Discord")
    if (config.get("slack_bot_token") or os.environ.get("SLACK_BOT_TOKEN")):
                           channels.append("Slack")
    if (config.get("twilio_account_sid") or os.environ.get("TWILIO_ACCOUNT_SID")):
                           channels.append("WhatsApp")
    if (config.get("signal_number") or os.environ.get("SIGNAL_NUMBER")):
                           channels.append("Signal")
    if (config.get("matrix_homeserver") or os.environ.get("MATRIX_HOMESERVER")):
                           channels.append("Matrix")
    if (config.get("mattermost_url") or os.environ.get("MATTERMOST_URL")):
                           channels.append("Mattermost")
    if (config.get("teams_webhook_url") or os.environ.get("TEAMS_WEBHOOK_URL")):
                           channels.append("Teams")
    if (config.get("irc_server") or os.environ.get("IRC_SERVER")):
                           channels.append("IRC")
    channel_str = ", ".join(channels) if channels else "none configured"

    # Cloud execution status
    cloud_backends = []
    import shutil
    if shutil.which("docker"):         cloud_backends.append("Docker")
    if config.get("daytona_api_key") or os.environ.get("DAYTONA_API_KEY"):
                                       cloud_backends.append("Daytona")
    try:
        import modal; cloud_backends.append("Modal")  # noqa: E401
    except ImportError:
        pass
    cloud_str = ", ".join(cloud_backends) if cloud_backends else "none"

    # Webhook & heartbeat
    webhook_port = config.get("webhook_port", 7271)
    webhook_on   = config.get("webhook_autostart", False)
    webhook_str  = f"auto-start :{webhook_port}" if webhook_on else f"manual (/webhook start)"
    hb_on        = config.get("heartbeat_enabled", False)
    hb_str       = (f"ON  every {config.get('heartbeat_interval', 1800)}s"
                    + (" [biz hrs]" if config.get("heartbeat_business_hours") else "")
                    if hb_on else "OFF  (/heartbeat start)")

    width = 64
    P = _P; R = _R; A = _A; G = _G; DG = _DG; C = _C; B = _B; PL = _PL; W = _W

    def _row(label, value, vcolor=W):
        inner = f"{A}{label:<24}{R} {vcolor}{value}{R}"
        pad   = (width - 4) - len(_plain(inner))
        return f"{P}║{R}  {inner}{' ' * max(0, pad)}  {P}║{R}"

    top    = f"{P}╔{'═' * (width - 2)}╗{R}"
    bottom = f"{P}╚{'═' * (width - 2)}╝{R}"
    div    = f"{P}╠{'═' * (width - 2)}╣{R}"

    print(f"\n{top}")
    print(f"{P}║{R}  {C}{B}{'OPERON CONFIGURATION SUMMARY':<{width-4}}{R}  {P}║{R}")
    print(div)
    print(_row("Default model",      model))
    print(_row("Provider",           prov))
    print(_row("OpenAI key",         keys['openai'],     G if keys['openai']     == 'SET' else DG))
    print(_row("Anthropic key",      keys['anthropic'],  G if keys['anthropic']  == 'SET' else DG))
    print(_row("OpenRouter key",     keys['openrouter'], G if keys['openrouter'] == 'SET' else DG))
    print(_row("Local servers",      local_str,          G if local_status       else DG))
    print(div)
    print(_row("Memory",             mem,       G if mem == 'ON'    else DG))
    print(_row("Mem0 provider",      "SET" if mem0_key else "—",
                                               G if mem0_key        else DG))
    print(div)
    print(_row("Messaging channels", channel_str[:36],   G if channels           else DG))
    print(div)
    print(_row("Cloud execution",    cloud_str[:36],      G if cloud_backends     else DG))
    print(div)
    print(_row("Webhook server",     webhook_str,         G if webhook_on         else DG))
    print(_row("Heartbeat",          hb_str,              G if hb_on              else DG))
    print(div)
    print(_row("Max iterations",     str(iters)))
    print(_row("Request timeout",    f"{timeout}s"))
    print(div)

    # Quick reference
    print(f"{P}║{R}  {DG}{'Quick reference':<{width-4}}{R}  {P}║{R}")
    quick_refs = [
        ("/model <name>",        "Switch LLM model"),
        ("/local",               "Detect & switch local models"),
        ("/soul edit",           "Edit personality file"),
        ("/memory",              "View stored memories"),
        ("/goal set <title>",    "Create a persistent goal"),
        ("/macro list",          "List pipeline macros"),
        ("/heartbeat start",     "Start background scheduler"),
        ("/webhook start",       "Start REST API server"),
        ("/toolsets list",       "Show 17 tool groups"),
        ("/retry list",          "View per-tool retry policies"),
        ("/rag index <path>",    "Index docs for RAG"),
        ("/secrets list",        "View encrypted secrets"),
        ("/doctor",              "Full system health check"),
        ("/status",              "Live telemetry"),
        ("/help",                "Full command list"),
    ]
    for cmd, desc in quick_refs:
        line = f"{PL}{cmd:<24}{R} {DG}{desc}{R}"
        pad  = (width - 4) - len(_plain(line))
        print(f"{P}║{R}  {line}{' ' * max(0, pad)}  {P}║{R}")

    print(f"{bottom}\n")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_wizard(config: ConfigManager) -> None:
    os.system("clear" if os.name != "nt" else "cls")
    _banner()

    is_rerun = config.is_configured()
    if is_rerun:
        print(f"  {_DG}Re-running setup. Press ENTER to keep any existing value.{_R}\n")
    else:
        print(f"  {_W}Welcome to Operon. Let's get you configured in a few steps.{_R}")
        print(f"  {_DG}Press ENTER to accept defaults.  All settings saved to ~/.operon/config.json{_R}\n")

    TOTAL = 11

    # ── Step 1: Provider selection ────────────────────────────────────────────
    _sep()
    _section(1, TOTAL, "Provider Selection")
    print(f"\n  {_DG}Operon can use cloud APIs, local models, or both.{_R}\n")
    use_cloud = _ask_yn("  Use cloud providers? (OpenAI / Anthropic / OpenRouter)", default=True)
    use_local = _ask_yn("  Use local models?    (Ollama / LM Studio / Jan)",         default=False)

    # ── Step 2: Cloud API keys ────────────────────────────────────────────────
    _sep()
    _section(2, TOTAL, "Cloud API Keys")
    cloud_saved = _step_cloud_keys(config, use_cloud)

    # ── Step 3: Local server setup ────────────────────────────────────────────
    _sep()
    _section(3, TOTAL, "Local Model Servers")
    print(f"\n  {_DG}Probing localhost for running servers...{_R}", end="", flush=True)
    servers = _probe_servers()
    running_count = sum(1 for s in servers.values() if s["running"])
    print(f"  {_G if running_count else _DG}{running_count} server(s) found{_R}")

    if use_local or running_count:
        local_models = _step_local(config, servers)
    else:
        local_models = []
        print(f"\n  {_DG}Skipping local setup. Run /local at any time to configure.{_R}")

    # ── Step 4: Default model ─────────────────────────────────────────────────
    _sep()
    _section(4, TOTAL, "Default Model")
    print(f"\n  {_DG}This model is used for every prompt unless you switch with /model.{_R}")
    chosen_model = _step_default_model(config, cloud_saved, local_models)
    info = config.resolve_model(chosen_model)
    config.set("default_model", chosen_model)
    config.set("active_provider", info["provider"])
    print(_ok(f"Default model: {chosen_model}  [{info['provider']}]"))

    # ── Step 5: Agent behaviour ───────────────────────────────────────────────
    _sep()
    _section(5, TOTAL, "Agent Behaviour")
    print(f"\n  {_DG}Controls how aggressively the agent loops through tool calls.{_R}")
    _step_agent_behaviour(config)

    # ── Step 6: Memory & Mem0 ─────────────────────────────────────────────────
    _sep()
    _section(6, TOTAL, "Memory & External Providers")
    _step_memory(config)

    # ── Step 7: Soul / Personality ────────────────────────────────────────────
    _sep()
    _section(7, TOTAL, "Soul & Personality")
    print(f"\n  {_DG}The soul file defines Operon's identity, tone, and operating rules.{_R}")
    print(f"  {_DG}It is injected into every system prompt.{_R}")
    _step_soul(config)

    # ── Step 8: Messaging channels ────────────────────────────────────────────
    _sep()
    _section(8, TOTAL, "Messaging Channels")
    print(f"\n  {_DG}Operon supports 9 messaging platforms out-of-the-box.{_R}")
    print(f"  {_DG}Configure just the ones you use — all others can be set later via env vars.{_R}")
    _step_messaging(config)

    # ── Step 9: Cloud execution ───────────────────────────────────────────────
    _sep()
    _section(9, TOTAL, "Cloud Execution Backends")
    _step_cloud_execution(config)

    # ── Step 10: Webhook & remote access ─────────────────────────────────────
    _sep()
    _section(10, TOTAL, "Webhook & Remote Access")
    _step_webhook(config)

    # ── Step 11: Heartbeat scheduler ─────────────────────────────────────────
    _sep()
    _section(11, TOTAL, "Heartbeat Scheduler")
    _step_heartbeat(config)

    # ── Step 12: Summary ──────────────────────────────────────────────────────
    config.set("configured", True)

    _sep()
    print(f"  {_G}{_B}Operon is ready.{_R}\n")
    _print_summary(config, servers)

    _ask("Press ENTER to launch", "")
    os.system("clear" if os.name != "nt" else "cls")
