"""
Operon Email Draft Tool.

Shows a formatted email preview in the terminal and asks for approval
before sending.  The model generates subject + body from context; this
tool handles the display → approve/reject → send loop so the user never
has to dictate every word of an email.

Flow:
  1. Model calls email_draft(to, subject, body, sender_email, app_password)
  2. Tool renders a colour-formatted preview box
  3. User types:
       y / yes / send   → sends immediately, returns success
       n / no           → returns {cancelled: true}  (model must STOP, not redraft)
       <anything else>  → treated as feedback, returned so model can redraft
  4. Model sees the result and either confirms or calls email_draft again
     with a revised draft.

Credentials are resolved automatically (no user action needed in chat):
  1. Passed explicitly as params
  2. GMAIL_SENDER_EMAIL / GMAIL_APP_PASSWORD environment variables
  3. ~/.operon/knowledge.json
  4. Interactive first-time setup prompt (if still missing)
"""

import os
import re
import sys
import getpass
from tools.email_send import email_send   # reuse the actual SMTP logic

# ── ANSI helpers (keep self-contained — no import from ui.theme) ──────────────
_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_PURPLE  = "\033[1;38;5;141m"
_CYAN    = "\033[1;38;5;81m"
_WHITE   = "\033[1;38;5;255m"
_GRAY    = "\033[38;5;244m"
_AMBER   = "\033[38;5;214m"
_GREEN   = "\033[1;38;5;82m"
_RED     = "\033[38;5;203m"
_WIDTH   = 78
_INNER   = _WIDTH - 2   # 76

# Strip ANSI escape codes when measuring visible width
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _vlen(s: str) -> int:
    """Visible length of a string (ignores ANSI escape codes)."""
    return len(_ANSI_RE.sub("", s))


def _box_line(text: str) -> str:
    """
    Pad *text* so its visible width equals _INNER, then wrap in box borders.
    ANSI codes are excluded from the width measurement so alignment is exact.
    """
    vl = _vlen(text)
    if vl < _INNER:
        padded = text + " " * (_INNER - vl)
    else:
        # Over-long: trim trailing visible chars until we fit
        # (rare — only hits if content is very wide)
        padded = text
    return f"{_PURPLE}│{_RESET}{padded}{_PURPLE}│{_RESET}"


# ── Credential helpers ────────────────────────────────────────────────────────

def _save_credentials_to_kb(email: str, password: str) -> None:
    """Persist email credentials to ~/.operon/knowledge.json."""
    try:
        import json as _json
        from pathlib import Path as _Path
        from datetime import datetime as _dt
        kb_path = _Path.home() / ".operon" / "knowledge.json"
        kb_path.parent.mkdir(parents=True, exist_ok=True)
        kb = _json.loads(kb_path.read_text(encoding="utf-8")) if kb_path.exists() else {}
        now = _dt.now().isoformat()
        if email:
            kb["sender_email"] = {"value": email,    "updated": now}
        if password:
            kb["app_password"] = {"value": password, "updated": now}
        kb_path.write_text(_json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass  # non-fatal — credentials still work for this session


def _prompt_credentials(need_email: bool, need_password: bool) -> tuple:
    """
    Interactive one-time credential setup displayed in the terminal.
    Only prompts for what is missing. Saves to knowledge base so it's
    never asked again. Returns (sender_email, app_password).
    """
    top    = f"{_PURPLE}╭{'─' * _INNER}╮{_RESET}"
    sep    = f"{_PURPLE}├{'─' * _INNER}┤{_RESET}"
    bottom = f"{_PURPLE}╰{'─' * _INNER}╯{_RESET}"

    print()
    print(top)
    print(_box_line(f"  {_AMBER}{_BOLD}⚙  EMAIL SETUP — one-time configuration{_RESET}"))
    print(sep)
    if need_email:
        print(_box_line(f"  {_WHITE}Your Gmail address is needed to send emails.{_RESET}"))
    if need_password:
        print(_box_line(f"  {_WHITE}A Gmail App Password is needed (not your regular password).{_RESET}"))
        print(_box_line(f"  {_GRAY}Get one at: myaccount.google.com/apppasswords{_RESET}"))
    print(_box_line(f"  {_GREEN}Credentials are saved locally — never sent to any AI model.{_RESET}"))
    print(bottom)
    print()

    email    = ""
    password = ""

    if need_email:
        sys.stdout.write(f"  {_CYAN}Your Gmail address{_RESET} ❯ ")
        sys.stdout.flush()
        try:
            email = input().strip()
        except (KeyboardInterrupt, EOFError):
            return "", ""

    if need_password:
        sys.stdout.write(f"  {_CYAN}App Password{_RESET} ❯ ")
        sys.stdout.flush()
        try:
            password = getpass.getpass("").strip()   # hidden input
        except (KeyboardInterrupt, EOFError):
            return email, ""

    if email or password:
        _save_credentials_to_kb(email, password)
        print(f"\n  {_GREEN}✓  Credentials saved — you won't be asked again.{_RESET}\n")

    return email, password


# ── Draft renderer ────────────────────────────────────────────────────────────

def _render_draft(
    sender: str, to: str, subject: str, body: str,
    cc: str = "", bcc: str = "", reply_to: str = "",
    attachments: list = None,
) -> None:
    """Print a formatted email preview box to stdout."""
    top    = f"{_PURPLE}╭{'─' * _INNER}╮{_RESET}"
    sep    = f"{_PURPLE}├{'─' * _INNER}┤{_RESET}"
    bottom = f"{_PURPLE}╰{'─' * _INNER}╯{_RESET}"

    print()
    print(top)
    print(_box_line(f"  {_AMBER}{_BOLD}✉  EMAIL DRAFT — PREVIEW{_RESET}"))
    print(sep)
    print(_box_line(f"  {_GRAY}From   :{_RESET}  {_WHITE}{sender}{_RESET}"))
    print(_box_line(f"  {_GRAY}To     :{_RESET}  {_WHITE}{to}{_RESET}"))
    if cc:
        print(_box_line(f"  {_GRAY}CC     :{_RESET}  {_WHITE}{cc}{_RESET}"))
    if bcc:
        print(_box_line(f"  {_GRAY}BCC    :{_RESET}  {_WHITE}{bcc}{_RESET}"))
    if reply_to:
        print(_box_line(f"  {_GRAY}Reply-To:{_RESET} {_WHITE}{reply_to}{_RESET}"))
    print(_box_line(f"  {_GRAY}Subject:{_RESET}  {_CYAN}{_BOLD}{subject}{_RESET}"))
    if attachments:
        from pathlib import Path as _Path
        att_names = ", ".join(_Path(a).name for a in attachments)
        print(_box_line(f"  {_GRAY}Attach :{_RESET}  {_WHITE}{att_names}{_RESET}"))
    print(sep)

    # Word-wrap body at (_INNER - 4) visible chars
    wrap_w = _INNER - 4
    lines  = []
    for paragraph in body.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words   = paragraph.split()
        cur     = []
        cur_len = 0
        for word in words:
            if cur_len + len(word) + (1 if cur else 0) > wrap_w:
                lines.append(" ".join(cur))
                cur     = [word]
                cur_len = len(word)
            else:
                cur.append(word)
                cur_len += len(word) + (1 if len(cur) > 1 else 0)
        if cur:
            lines.append(" ".join(cur))

    for ln in lines:
        print(_box_line(f"  {_WHITE}{ln}{_RESET}"))

    print(sep)
    print(_box_line(
        f"  {_GREEN}[y] Send{_RESET}   "
        f"{_RED}[n] Discard{_RESET}   "
        f"{_GRAY}or type feedback to request a new draft{_RESET}"
    ))
    print(bottom)


# ── Main tool function ────────────────────────────────────────────────────────

def email_draft(
    to:           str = "",
    subject:      str = "",
    body:         str = "",
    cc:           str = "",
    bcc:          str = "",
    reply_to:     str = "",
    attachments:  list = None,
    sender_email: str = "",
    app_password: str = "",
    **_,
) -> dict:
    """
    Preview an email draft in the terminal and send on user approval.

    Args:
        to           — Recipient address(es), comma-separated (required)
        subject      — Draft subject line (required)
        body         — Draft body as PLAIN TEXT prose (required — NOT a JSON object)
        cc           — CC recipients, comma-separated (optional)
        bcc          — BCC recipients, comma-separated (optional)
        reply_to     — Reply-To address (optional)
        attachments  — List of file paths to attach (optional)
        sender_email — Sender address (optional — auto-loaded from env/knowledge base)
        app_password — SMTP/App password (optional — auto-loaded from env/knowledge base)

    Returns on approval + send:
        {"approved": true,  "sent": true,  "recipients": [...], "error": ""}
    Returns on user cancellation (n/no):
        {"approved": false, "sent": false, "cancelled": true,   "error": ""}
    Returns on feedback:
        {"approved": false, "sent": false, "feedback": "<text>","error": ""}
    """
    if attachments is None:
        attachments = []

    # ── Coerce body to string ─────────────────────────────────────────────────
    # Local models sometimes pass body as a dict or list instead of plain text.
    # Extract usable text rather than crashing.
    if isinstance(body, dict):
        parts = []
        for key in ("greeting", "intro", "introduction", "opening", "salutation"):
            if key in body:
                parts.append(str(body[key]))
        for key in ("questions", "body", "content", "text", "paragraphs", "items"):
            if key in body:
                val = body[key]
                if isinstance(val, list):
                    parts.extend(
                        f"{i + 1}. {q}" for i, q in enumerate(val)
                    )
                elif val:
                    parts.append(str(val))
        for key in ("closing", "sign_off", "signature"):
            if key in body:
                parts.append(str(body[key]))
        body = "\n\n".join(parts) if parts else ""
    elif not isinstance(body, str):
        body = str(body) if body else ""

    # ── Validate body is real prose, not a JSON fragment ─────────────────────
    body_stripped = body.strip()
    if body_stripped in ("{", "}", "{}", "{ }", "[]", "") or (
        len(body_stripped) < 20 and body_stripped.startswith(("{", "["))
    ):
        return {
            "approved": False, "sent": False, "feedback": "",
            "error": (
                "The email body is empty or invalid (looks like a broken JSON fragment). "
                "Write the full email body as plain prose text — numbered questions, "
                "paragraphs, sign-off — NOT as a JSON object or dict. "
                "Example body: 'Hi,\\n\\n1. What is OpenAI?\\n2. ...\\n\\nBest regards'"
            ),
        }

    # ── Detect placeholder / template content ────────────────────────────────
    # Reject generic filler like "Question one?", "1. Question two...",
    # "Item one:", "[Placeholder]", "TODO:" etc.  The model must write real
    # content that matches what the user actually asked for.
    _PLACEHOLDER_RE = re.compile(
        r'question\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)'
        r'|\bitem\s+(?:one|two|three|four|five|\d+)'
        r'|\[placeholder\]|\[insert\]|\[your\s'
        r'|\bTODO\b|\bFILL\s+IN\b',
        re.IGNORECASE,
    )
    _placeholder_hits = _PLACEHOLDER_RE.findall(body_stripped)
    if len(_placeholder_hits) >= 2:
        return {
            "approved": False, "sent": False, "feedback": "",
            "error": (
                "The email body contains placeholder text "
                f"(e.g. {_placeholder_hits[0]!r}) instead of real content. "
                "Write ACTUAL, specific questions or content that directly addresses "
                "what the user asked for. Do NOT use generic placeholders like "
                "'Question one', 'Question two', 'Item one', etc. "
                "Every sentence must be real, meaningful content."
            ),
        }

    # ── Detect subject/body topic mismatch ───────────────────────────────────
    # Catch cases where the model recycled a subject from a previous email
    # (e.g. subject says "OpenAI Questions" but body is personal questions).
    # Heuristic: extract keywords from the subject and check whether at least
    # one of them appears in the body; if zero match, flag a mismatch.
    _subj_words = set(re.findall(r'\b[a-z]{4,}\b', subject.lower()))
    _body_words = set(re.findall(r'\b[a-z]{4,}\b', body_stripped.lower()))
    # Ignore very generic words that appear in almost every email
    _STOPWORDS = {
        "have", "this", "that", "with", "from", "will", "your", "been",
        "they", "what", "when", "here", "like", "some", "more", "each",
        "just", "about", "also", "best", "regards", "thank", "hello",
        "dear", "hope", "would", "could", "should", "please",
    }
    _subj_keywords = _subj_words - _STOPWORDS
    _body_keywords = _body_words - _STOPWORDS
    if len(_subj_keywords) >= 2 and _subj_keywords.isdisjoint(_body_keywords):
        return {
            "approved": False, "sent": False, "feedback": "",
            "error": (
                f"The subject '{subject}' does not match the email body content. "
                "The subject appears to have been copied from a previous email. "
                "Write a NEW subject line that accurately describes THIS specific email. "
                f"Example: if the body asks personal questions, use a subject like "
                "'A Few Questions for You', not a topic from a previous email."
            ),
        }

    # ── Validate recipient address ────────────────────────────────────────────
    if to and "@" not in to:
        return {
            "approved": False, "sent": False, "feedback": "",
            "error": (
                f"'{to}' is not a valid email address (missing @). "
                "Ask the user for the correct recipient address."
            ),
        }

    # ── Resolve credentials: params → env vars → knowledge base ──────────────
    if not sender_email:
        sender_email = os.environ.get("GMAIL_SENDER_EMAIL", "")
    if not app_password:
        app_password = os.environ.get("GMAIL_APP_PASSWORD", "")

    if not sender_email or not app_password:
        try:
            import json as _json
            from pathlib import Path as _Path
            kb_path = _Path.home() / ".operon" / "knowledge.json"
            if kb_path.exists():
                kb = _json.loads(kb_path.read_text(encoding="utf-8"))

                def _kbget(*keys):
                    for k in keys:
                        v = kb.get(k, {})
                        val = v.get("value", "").strip() if isinstance(v, dict) else str(v).strip()
                        if val:
                            return val
                    return ""

                if not sender_email:
                    sender_email = _kbget(
                        "sender_email", "gmail_sender", "gmail_address",
                        "email_address", "email", "gmail", "my_email",
                    )
                if not app_password:
                    app_password = _kbget(
                        "app_password", "gmail_app_password", "email_password",
                        "gmail_password", "email_app_password",
                    )
        except Exception:
            pass

    # ── Validate required fields ──────────────────────────────────────────────
    if not to or not subject or not body:
        missing = [f for f, v in [("to", to), ("subject", subject), ("body", body)] if not v]
        return {
            "approved": False, "sent": False, "feedback": "",
            "error": f"Missing required fields: {', '.join(missing)}. All three are required.",
        }

    # ── Guard: sender ≠ recipient ─────────────────────────────────────────────
    if sender_email and to.strip().lower() == sender_email.strip().lower():
        return {
            "approved": False, "sent": False, "feedback": "",
            "error": (
                f"The 'to' address ({to}) is the same as the sender's email. "
                "Do not use the sender's address as the recipient. "
                "Re-read the user's message for the correct recipient address."
            ),
        }

    # ── Interactive first-time setup if credentials still missing ─────────────
    if (not sender_email or not app_password) and sys.stdin.isatty():
        fetched_email, fetched_pw = _prompt_credentials(
            need_email    = not sender_email,
            need_password = not app_password,
        )
        if fetched_email:
            sender_email = fetched_email
        if fetched_pw:
            app_password = fetched_pw

    if not sender_email:
        return {
            "approved": False, "sent": False, "feedback": "",
            "error": "Sender email not provided. Setup was cancelled.",
        }
    if not app_password:
        return {
            "approved": False, "sent": False, "feedback": "",
            "error": "App password not provided. Setup was cancelled.",
        }

    # ── Show the draft ────────────────────────────────────────────────────────
    _render_draft(
        sender=sender_email, to=to, subject=subject, body=body,
        cc=cc, bcc=bcc, reply_to=reply_to, attachments=attachments,
    )

    # ── Get user decision ─────────────────────────────────────────────────────
    try:
        sys.stdout.write(f"\n  {_PURPLE}Your decision{_RESET} ❯ ")
        sys.stdout.flush()
        decision = input().strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return {"approved": False, "sent": False, "cancelled": True, "error": ""}

    decision_lower = decision.lower()

    # ── Send ──────────────────────────────────────────────────────────────────
    if decision_lower in ("y", "yes", "send", "ok", "approve", "approved", "looks good", "send it"):
        result = email_send(
            sender_email=sender_email,
            app_password=app_password,
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
            reply_to=reply_to,
            attachments=attachments or [],
        )
        if result.get("success"):
            print(f"\n  {_GREEN}✓  Email sent to {to}{_RESET}\n")
            return {
                "approved":   True,
                "sent":       True,
                "recipients": result.get("recipients", []),
                "error":      "",
            }
        else:
            return {
                "approved": True,
                "sent":     False,
                "feedback": "",
                "error":    result.get("error", "Send failed."),
            }

    # ── Cancel (n / no) ───────────────────────────────────────────────────────
    if decision_lower in ("n", "no", ""):
        print(f"\n  {_AMBER}Draft discarded.{_RESET}\n")
        return {
            "approved": False,
            "sent":     False,
            "cancelled": True,   # model must STOP here — do NOT redraft
            "feedback": "",
            "error":    "",
        }

    # ── Feedback — model should redraft ───────────────────────────────────────
    print(f"\n  {_AMBER}Draft discarded.{_RESET}  Feedback: {decision}\n")
    return {
        "approved": False,
        "sent":     False,
        "cancelled": False,
        "feedback": decision,
        "error":    "",
    }
