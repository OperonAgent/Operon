"""
Operon Email Send Tool.

Sends email via SMTP with support for:
  • Gmail, Outlook, Yahoo, and any custom SMTP server
  • TLS (port 587, recommended) and SSL (port 465)
  • Plain text and HTML bodies
  • CC, BCC, and Reply-To headers
  • File attachments

No external dependencies beyond the Python standard library.
"""

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Optional


# ── Known provider defaults ────────────────────────────────────────────────────

_PROVIDER_DEFAULTS = {
    "gmail":   {"host": "smtp.gmail.com",        "tls_port": 587, "ssl_port": 465},
    "outlook": {"host": "smtp-mail.outlook.com", "tls_port": 587, "ssl_port": 587},
    "hotmail": {"host": "smtp-mail.outlook.com", "tls_port": 587, "ssl_port": 587},
    "yahoo":   {"host": "smtp.mail.yahoo.com",   "tls_port": 587, "ssl_port": 465},
    "icloud":  {"host": "smtp.mail.me.com",      "tls_port": 587, "ssl_port": 587},
    "zoho":    {"host": "smtp.zoho.com",         "tls_port": 587, "ssl_port": 465},
    "ses":     {"host": "email-smtp.us-east-1.amazonaws.com", "tls_port": 587, "ssl_port": 465},
}


def _infer_provider(sender_email: str) -> Optional[str]:
    """Guess the provider from the sender's email domain."""
    domain = sender_email.split("@")[-1].lower() if "@" in sender_email else ""
    for key in _PROVIDER_DEFAULTS:
        if key in domain:
            return key
    return None


def email_send(
    sender_email: str,
    app_password: str,
    to: str,
    subject: str,
    body: str,
    *,
    body_type: str = "plain",
    cc: str = "",
    bcc: str = "",
    reply_to: str = "",
    smtp_host: str = "",
    smtp_port: int = 0,
    use_ssl: bool = False,
    attachments: list = None,
) -> dict:
    """
    Send an email.

    Args:
        sender_email  — From address (required)
        app_password  — SMTP password or Gmail App Password (required)
        to            — Recipient(s), comma-separated (required)
        subject       — Email subject (required)
        body          — Email body text (required)
        body_type     — "plain" or "html", default "plain" (optional)
        cc            — CC recipients, comma-separated (optional)
        bcc           — BCC recipients, comma-separated (optional)
        reply_to      — Reply-To address (optional)
        smtp_host     — Override SMTP host (auto-detected from sender domain) (optional)
        smtp_port     — Override SMTP port (optional)
        use_ssl       — Use SSL instead of TLS, default False (optional)
        attachments   — List of file paths to attach (optional)

    Returns:
        {success, recipients, message_id, error}
    """
    if not sender_email or not app_password or not to or not subject or not body:
        return {
            "success": False, "recipients": [], "message_id": "",
            "error": "sender_email, app_password, to, subject, and body are all required.",
        }

    # ── Resolve SMTP settings ────────────────────────────────────────────────
    provider = _infer_provider(sender_email)
    defaults = _PROVIDER_DEFAULTS.get(provider or "", {
        "host": smtp_host or "smtp.gmail.com",
        "tls_port": 587,
        "ssl_port": 465,
    })

    host = smtp_host or defaults["host"]
    if smtp_port:
        port = smtp_port
    elif use_ssl:
        port = defaults["ssl_port"]
    else:
        port = defaults["tls_port"]

    # ── Build message ────────────────────────────────────────────────────────
    msg = MIMEMultipart("alternative" if body_type == "html" else "mixed")
    msg["From"]    = sender_email
    msg["To"]      = to
    msg["Subject"] = subject
    if cc:
        msg["Cc"] = cc
    if reply_to:
        msg["Reply-To"] = reply_to

    msg.attach(MIMEText(body, body_type))

    # ── Attachments ──────────────────────────────────────────────────────────
    if attachments:
        for path_str in attachments:
            path = Path(path_str)
            if not path.exists():
                return {
                    "success": False, "recipients": [], "message_id": "",
                    "error": f"Attachment not found: {path_str}",
                }
            part = MIMEBase("application", "octet-stream")
            part.set_payload(path.read_bytes())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{path.name}"')
            msg.attach(part)

    # ── Collect all recipients ───────────────────────────────────────────────
    all_recipients = [r.strip() for r in to.split(",") if r.strip()]
    if cc:
        all_recipients += [r.strip() for r in cc.split(",") if r.strip()]
    if bcc:
        all_recipients += [r.strip() for r in bcc.split(",") if r.strip()]

    # ── Connect and send ─────────────────────────────────────────────────────
    server = None
    try:
        if use_ssl:
            context = ssl.create_default_context()
            server  = smtplib.SMTP_SSL(host, port, context=context, timeout=15)
        else:
            server = smtplib.SMTP(host, port, timeout=15)
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()

        server.login(sender_email, app_password)
        server.sendmail(sender_email, all_recipients, msg.as_string())

        return {
            "success":    True,
            "recipients": all_recipients,
            "message_id": msg.get("Message-ID", ""),
            "error":      "",
        }

    except smtplib.SMTPAuthenticationError:
        return {
            "success": False, "recipients": [], "message_id": "",
            "error": (
                "Authentication failed. For Gmail, use an App Password "
                "(not your regular password). "
                "Enable it at: myaccount.google.com/apppasswords"
            ),
        }
    except smtplib.SMTPRecipientsRefused as e:
        return {
            "success": False, "recipients": [], "message_id": "",
            "error": f"Recipient(s) refused by server: {e.recipients}",
        }
    except smtplib.SMTPException as e:
        return {
            "success": False, "recipients": [], "message_id": "",
            "error": f"SMTP error: {e}",
        }
    except Exception as e:
        return {
            "success": False, "recipients": [], "message_id": "",
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                pass
