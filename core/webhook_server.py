"""
Operon Webhook / REST API Server.

Exposes the Operon agent as an HTTP endpoint so external services, scripts,
and automations can trigger it without running the interactive REPL.

Endpoints
---------
  POST /chat          — send a message, get a response
  POST /batch         — send a list of prompts, get a list of responses
  GET  /status        — health check + session info
  GET  /tools         — list available tools
  DELETE /session     — clear the current session

Authentication
--------------
  Set OPERON_WEBHOOK_TOKEN in the environment.
  If set, every request must include:
    Authorization: Bearer <token>
  If not set, the server runs without auth (useful for local automation).

Usage
-----
  # From code
  from core.webhook_server import WebhookServer
  srv = WebhookServer(agent_runner=my_runner, host="127.0.0.1", port=7271)
  srv.start()
  srv.stop()

  # Via CLI:  operon --webhook [--port 7271] [--host 127.0.0.1]
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — parses JSON body, calls agent runner, returns JSON."""

    # Injected by WebhookServer
    agent_runner:   Callable[[str], str] = lambda p: "(no runner)"
    auth_token:     str                  = ""
    session_clear:  Callable             = lambda: None
    tool_list:      Callable             = lambda: []
    session_info:   Callable             = lambda: {}

    # ------------------------------------------------------------------
    def log_message(self, fmt, *args):  # suppress default access log
        pass

    # ------------------------------------------------------------------
    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str) -> None:
        self._send_json(code, {"error": message, "success": False})

    # ------------------------------------------------------------------
    def _check_auth(self) -> bool:
        if not self.auth_token:
            return True
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.auth_token}"
        return header == expected

    # ------------------------------------------------------------------
    def _read_body(self) -> Optional[dict]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            return json.loads(raw)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    # ------------------------------------------------------------------
    def do_GET(self):
        if not self._check_auth():
            return self._send_error(401, "Unauthorized")

        path = urlparse(self.path).path.rstrip("/")

        if path == "/status":
            info = self.session_info()
            self._send_json(200, {
                "success": True,
                "status":  "running",
                "session": info,
            })

        elif path == "/tools":
            tools = self.tool_list()
            self._send_json(200, {"success": True, "tools": tools, "count": len(tools)})

        else:
            self._send_error(404, f"Not found: {path}")

    # ------------------------------------------------------------------
    def do_POST(self):
        if not self._check_auth():
            return self._send_error(401, "Unauthorized")

        path = urlparse(self.path).path.rstrip("/")
        body = self._read_body()
        if body is None:
            return self._send_error(400, "Invalid JSON body")

        if path == "/chat":
            message = body.get("message", "").strip()
            if not message:
                return self._send_error(400, "'message' field is required")

            t0       = time.monotonic()
            response = self.agent_runner(message)
            elapsed  = round((time.monotonic() - t0) * 1000)
            self._send_json(200, {
                "success":    True,
                "response":   response,
                "elapsed_ms": elapsed,
            })

        elif path == "/batch":
            prompts = body.get("prompts", [])
            if not isinstance(prompts, list) or not prompts:
                return self._send_error(400, "'prompts' must be a non-empty list of strings")

            results = []
            for prompt in prompts:
                prompt = str(prompt).strip()
                t0     = time.monotonic()
                try:
                    resp = self.agent_runner(prompt)
                    results.append({
                        "prompt":     prompt,
                        "response":   resp,
                        "elapsed_ms": round((time.monotonic() - t0) * 1000),
                        "success":    True,
                    })
                except Exception as e:
                    results.append({
                        "prompt":  prompt,
                        "error":   str(e),
                        "success": False,
                    })

            self._send_json(200, {
                "success": True,
                "results": results,
                "count":   len(results),
            })

        else:
            self._send_error(404, f"Not found: {path}")

    # ------------------------------------------------------------------
    def do_DELETE(self):
        if not self._check_auth():
            return self._send_error(401, "Unauthorized")

        path = urlparse(self.path).path.rstrip("/")
        if path == "/session":
            self.session_clear()
            self._send_json(200, {"success": True, "message": "Session cleared"})
        else:
            self._send_error(404, f"Not found: {path}")


# ---------------------------------------------------------------------------
# WebhookServer
# ---------------------------------------------------------------------------

class WebhookServer:
    """
    Lightweight HTTP server that exposes Operon as a REST API.

    Parameters
    ----------
    agent_runner  : callable(prompt: str) → str
        Function that runs the agent and returns a text response.
    host          : str, default "127.0.0.1"
    port          : int, default 7271
    auth_token    : str, optional — if set, requests must supply Bearer token.
    session_clear : callable, optional — clears the active session.
    tool_list     : callable, optional — returns list of tool name strings.
    session_info  : callable, optional — returns session status dict.
    """

    def __init__(
        self,
        agent_runner:   Callable[[str], str],
        host:           str = "127.0.0.1",
        port:           int = 7271,
        auth_token:     str = "",
        session_clear:  Callable = None,
        tool_list:      Callable = None,
        session_info:   Callable = None,
    ) -> None:
        self.host         = host
        self.port         = port
        self._auth_token  = auth_token or os.environ.get("OPERON_WEBHOOK_TOKEN", "")
        self._server:     Optional[HTTPServer] = None
        self._thread:     Optional[threading.Thread] = None
        self.running      = False

        # Inject dependencies into the handler class (class-level so all requests share them)
        _Handler.agent_runner  = staticmethod(agent_runner)
        _Handler.auth_token    = self._auth_token
        _Handler.session_clear = staticmethod(session_clear or (lambda: None))
        _Handler.tool_list     = staticmethod(tool_list or (lambda: []))
        _Handler.session_info  = staticmethod(session_info or (lambda: {}))

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> str:
        """Start the server in a daemon background thread. Returns the URL."""
        if self.running:
            return self.url
        try:
            self._server = HTTPServer((self.host, self.port), _Handler)
        except OSError as e:
            raise RuntimeError(f"Could not bind to {self.host}:{self.port} — {e}") from e

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="operon-webhook",
        )
        self._thread.start()
        self.running = True
        return self.url

    def stop(self) -> None:
        """Shut down the server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        self.running = False

    def status(self) -> Dict[str, Any]:
        return {
            "running":    self.running,
            "url":        self.url if self.running else None,
            "auth":       bool(self._auth_token),
            "endpoints": [
                "POST /chat      — run a prompt, get response",
                "POST /batch     — run a list of prompts",
                "GET  /status    — health + session info",
                "GET  /tools     — list available tools",
                "DELETE /session — clear active session",
            ],
        }
