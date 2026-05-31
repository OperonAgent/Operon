"""
Operon Messaging Tools.

telegram_send — send a message to a Telegram chat (requires bot token + chat_id)
clarify       — agent asks the user a blocking question and waits for the answer
todo          — in-session task list (add / complete / remove / list / clear)

For Telegram gateway (receiving messages and running the agent loop),
see core/gateway.py and use the /gateway command.
"""

import os
import json
from pathlib import Path

import requests

# ── Shared helper ─────────────────────────────────────────────────────────────

def _r(success: bool, output=None, error: str = "") -> dict:
    return {"success": success, "output": output, "error": error}


def _get_telegram_token() -> str:
    """Read Telegram bot token from env var or ~/.operon/config.json."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if token:
        return token
    try:
        cfg = json.loads((Path.home() / ".operon" / "config.json").read_text())
        return cfg.get("telegram_token", "")
    except Exception:
        return ""


# ── Telegram send ─────────────────────────────────────────────────────────────

def telegram_send(
    chat_id:    str,
    text:       str,
    parse_mode: str = "Markdown",
    token:      str = "",
    **_,
) -> dict:
    """
    Send a message to a Telegram chat.

    chat_id:    Telegram chat ID or @username.
    text:       Message body (Markdown supported by default).
    parse_mode: 'Markdown' | 'HTML' | '' (plain).
    token:      Bot token — if omitted, reads from TELEGRAM_BOT_TOKEN env var
                or ~/.operon/config.json telegram_token field.
    """
    token = token or _get_telegram_token()
    if not token:
        return _r(False, error=(
            "No Telegram bot token. Set TELEGRAM_BOT_TOKEN env var "
            "or configure it via /setup → Telegram."
        ))

    # Split long messages (Telegram max is 4096 chars)
    max_len = 4000
    chunks  = [text[i:i + max_len] for i in range(0, len(text), max_len)]
    sent    = []

    for chunk in chunks:
        payload: dict = {"chat_id": chat_id, "text": chunk}
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            sent.append(data.get("result", {}).get("message_id"))
        except requests.exceptions.HTTPError as e:
            # Retry without parse_mode on formatting errors
            if e.response is not None and e.response.status_code == 400:
                try:
                    plain_payload = {"chat_id": chat_id, "text": chunk}
                    resp2 = requests.post(
                        f"https://api.telegram.org/bot{token}/sendMessage",
                        json=plain_payload,
                        timeout=15,
                    )
                    resp2.raise_for_status()
                    sent.append(resp2.json().get("result", {}).get("message_id"))
                except Exception as e2:
                    return _r(False, error=str(e2))
            else:
                return _r(False, error=str(e))
        except Exception as e:
            return _r(False, error=str(e))

    return _r(True, {
        "sent_to":    chat_id,
        "message_ids": sent,
        "chunks":     len(chunks),
    })


# ── Clarify ───────────────────────────────────────────────────────────────────

def clarify(question: str, **_) -> dict:
    """
    Ask the user a blocking clarifying question and return their typed answer.
    Use when the task is ambiguous and proceeding blindly would be wrong.
    """
    print(f"\n   Operon needs clarification:\n  {question}")
    print()
    try:
        answer = input("  Your answer ❯ ").strip()
        if not answer:
            return _r(True, {"answer": "(no answer provided)", "cancelled": False})
        return _r(True, {"answer": answer, "cancelled": False})
    except (KeyboardInterrupt, EOFError):
        return _r(True, {"answer": "", "cancelled": True,
                         "note": "User cancelled — proceed with best judgement."})


# ── Session todo list ─────────────────────────────────────────────────────────

_TODOS: list[dict] = []   # module-level; lives for the session


def todo(action: str, item: str = "", index: int = 0, **_) -> dict:
    """
    Manage a session-scoped task list.

    action: 'add'      — add a new task (requires item)
            'list'     — show all tasks
            'complete' — mark task done (requires index, 1-based)
            'remove'   — delete a task (requires index, 1-based)
            'clear'    — remove all tasks
    """
    action = action.lower()

    if action == "add":
        if not item:
            return _r(False, error="Provide item text.")
        _TODOS.append({"text": item.strip(), "done": False})
        return _r(True, {
            "added": item,
            "total": len(_TODOS),
            "todos": _todos_display(),
        })

    elif action == "list":
        if not _TODOS:
            return _r(True, {"output": "No todos.", "todos": []})
        return _r(True, {"todos": _todos_display(), "total": len(_TODOS)})

    elif action == "complete":
        idx = int(index) - 1
        if not (0 <= idx < len(_TODOS)):
            return _r(False, error=f"Index {index} out of range (1–{len(_TODOS)}).")
        _TODOS[idx]["done"] = True
        return _r(True, {"completed": _TODOS[idx]["text"], "todos": _todos_display()})

    elif action == "remove":
        idx = int(index) - 1
        if not (0 <= idx < len(_TODOS)):
            return _r(False, error=f"Index {index} out of range (1–{len(_TODOS)}).")
        removed = _TODOS.pop(idx)
        return _r(True, {"removed": removed["text"], "todos": _todos_display()})

    elif action == "clear":
        _TODOS.clear()
        return _r(True, {"output": "All todos cleared."})

    return _r(False, error=f"Unknown action '{action}'. Use: add|list|complete|remove|clear")


def _todos_display() -> list[str]:
    return [
        f"{'✓' if t['done'] else '○'} {i + 1}. {t['text']}"
        for i, t in enumerate(_TODOS)
    ]
