"""
Operon IRC Messaging Integration.

Pure Python socket-based IRC client — zero dependencies.

Setup
-----
  export IRC_SERVER=irc.libera.chat
  export IRC_PORT=6667
  export IRC_NICK=operonbot
  export IRC_CHANNEL=#general
  export IRC_PASSWORD=         # optional NickServ password

All functions accept **_ for registry compatibility.
"""

from __future__ import annotations

import os
import socket
import ssl
import time
from typing import List, Optional


def _defaults() -> tuple[str, int, str, str, str]:
    """Return (server, port, nick, channel, password) from env vars."""
    server  = os.environ.get("IRC_SERVER",   "irc.libera.chat")
    port    = int(os.environ.get("IRC_PORT", "6697"))
    nick    = os.environ.get("IRC_NICK",     "operon_bot")
    channel = os.environ.get("IRC_CHANNEL",  "")
    pw      = os.environ.get("IRC_PASSWORD", "")
    return server, port, nick, channel, pw


def _connect(server: str, port: int, nick: str, password: str,
             use_ssl: bool = True) -> socket.socket:
    """Open an IRC socket, authenticate, and return it."""
    raw = socket.create_connection((server, port), timeout=15)
    if use_ssl or port in (6697, 7000, 7070):
        ctx = ssl.create_default_context()
        sock = ctx.wrap_socket(raw, server_hostname=server)
    else:
        sock = raw

    def _send(line: str):
        sock.sendall((line + "\r\n").encode("utf-8", errors="replace"))

    if password:
        _send(f"PASS {password}")
    _send(f"NICK {nick}")
    _send(f"USER {nick} 0 * :Operon Bot")

    # Wait for welcome (001) or error
    buf = b""
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            text = buf.decode("utf-8", errors="replace")
            if " 001 " in text:  # RPL_WELCOME
                break
            if "PING" in text:
                # Handle PING/PONG during registration
                for line in text.splitlines():
                    if line.startswith("PING"):
                        _send(f"PONG {line.split(':', 1)[-1].strip()}")
        except socket.timeout:
            pass

    return sock


def _recv_lines(sock: socket.socket, timeout: float = 3.0) -> List[str]:
    """Read available lines from socket."""
    sock.settimeout(timeout)
    buf = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        except socket.timeout:
            break
    return buf.decode("utf-8", errors="replace").splitlines()


def irc_send(
    message:  str  = "",
    channel:  str  = "",
    server:   str  = "",
    port:     int  = 0,
    nick:     str  = "",
    password: str  = "",
    **_,
) -> dict:
    """
    Send a message to an IRC channel.

    Args:
        message  — message text (required)
        channel  — IRC channel e.g. '#general' (optional — auto-read from IRC_CHANNEL)
        server   — IRC server hostname (optional — auto-read from IRC_SERVER)
        port     — IRC port (optional — default 6697 for SSL, 6667 plain)
        nick     — bot nickname (optional — auto-read from IRC_NICK)
        password — NickServ password (optional — auto-read from IRC_PASSWORD)

    Returns:
        {success, channel, server, error}
    """
    if not message:
        return {"success": False, "error": "message is required."}

    d_server, d_port, d_nick, d_channel, d_pw = _defaults()
    server   = server   or d_server
    port     = port     or d_port
    nick     = nick     or d_nick
    channel  = channel  or d_channel
    password = password or d_pw

    if not channel:
        return {"success": False, "error": "channel required or set IRC_CHANNEL."}

    sock = None
    try:
        sock = _connect(server, port, nick, password)

        def _send(line: str):
            sock.sendall((line + "\r\n").encode("utf-8", errors="replace"))

        _send(f"JOIN {channel}")
        time.sleep(0.5)
        _send(f"PRIVMSG {channel} :{message}")
        time.sleep(0.3)
        _send("QUIT :Operon done")
        return {"success": True, "channel": channel, "server": server, "error": ""}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass


def irc_get_messages(
    channel:  str  = "",
    server:   str  = "",
    port:     int  = 0,
    nick:     str  = "",
    password: str  = "",
    wait_s:   int  = 5,
    **_,
) -> dict:
    """
    Join an IRC channel and collect messages for a few seconds.

    Args:
        channel — IRC channel (optional — auto-read from IRC_CHANNEL)
        server  — IRC server (optional)
        wait_s  — seconds to listen for messages (optional, default 5)

    Returns:
        {success, messages: [{nick, text, channel}], count, error}
    """
    d_server, d_port, d_nick, d_channel, d_pw = _defaults()
    server   = server   or d_server
    port     = port     or d_port
    nick     = nick     or d_nick
    channel  = channel  or d_channel
    password = password or d_pw

    if not channel:
        return {"success": False, "error": "channel required or set IRC_CHANNEL."}

    sock = None
    try:
        sock = _connect(server, port, nick, password)

        def _send(line: str):
            sock.sendall((line + "\r\n").encode("utf-8", errors="replace"))

        _send(f"JOIN {channel}")
        lines   = _recv_lines(sock, timeout=float(wait_s))
        messages = []
        for line in lines:
            # :nick!user@host PRIVMSG #channel :text
            if "PRIVMSG" in line and channel in line:
                try:
                    sender = line.split("!")[0].lstrip(":")
                    text   = line.split("PRIVMSG", 1)[1].split(":", 1)[1].strip()
                    messages.append({"nick": sender, "text": text, "channel": channel})
                except Exception:
                    pass
        _send("QUIT")
        return {"success": True, "messages": messages, "count": len(messages), "error": ""}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass
