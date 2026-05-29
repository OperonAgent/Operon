"""
Operon MCP Server — expose Operon tools to any MCP client.

Adapted from Hermes Agent mcp_serve.py architecture.

Starts a stdio or HTTP/SSE MCP server so any MCP-capable client
(Claude Code, Cursor, Codex, VS Code, etc.) can discover and call
Operon's full tool suite without running Operon interactively.

Usage (stdio, for Claude Code / Cursor):
    python -m operon.mcp_server

Usage (HTTP, for network clients):
    python -m operon.mcp_server --http --port 3456

Claude Code config (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "operon": {
          "command": "python3",
          "args": ["-m", "operon.mcp_server"],
          "cwd": "/Users/you/operon"
        }
      }
    }

Protocol: JSON-RPC 2.0 over stdio (newline-delimited) or HTTP/SSE.
Supports: initialize, tools/list, tools/call, notifications/cancelled.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("operon.mcp_server")

_OPERON_DIR = Path(__file__).resolve().parent.parent
if str(_OPERON_DIR) not in sys.path:
    sys.path.insert(0, str(_OPERON_DIR))

# ---------------------------------------------------------------------------
# FastMCP path (preferred when `mcp` package is installed)
# ---------------------------------------------------------------------------

_HAS_FASTMCP = False
try:
    from mcp.server.fastmcp import FastMCP as _FastMCP
    _HAS_FASTMCP = True
except ImportError:
    _FastMCP = None   # type: ignore


# ---------------------------------------------------------------------------
# Tool registry bridge
# ---------------------------------------------------------------------------

def _load_registry():
    """Import the tool registry, returning (registry, dispatch_dict)."""
    from tools.registry import ToolRegistry
    reg = ToolRegistry()
    # Build dispatch dict: tool_name → callable
    try:
        from tools.registry import _DISPATCH  # noqa: private
        dispatch = dict(_DISPATCH)
    except Exception:
        dispatch = {}
    return reg, dispatch


def _get_tool_defs(registry) -> List[Dict]:
    """Return a list of MCP-compatible tool definitions."""
    tools = []
    try:
        raw = registry.get_definitions()   # returns list of {name, description, parameters}
    except Exception:
        raw = []
    for t in raw:
        name = t.get("name", "")
        if not name:
            continue
        desc  = t.get("description", "No description")
        params = t.get("parameters", {"type": "object", "properties": {}})
        tools.append({
            "name":        name,
            "description": desc,
            "inputSchema": params,
        })
    # Add built-in Operon meta tools
    tools.append({
        "name": "operon_status",
        "description": "Return Operon server status, version, and available tool count.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    })
    tools.append({
        "name": "operon_help",
        "description": "Return the full list of available Operon tools with descriptions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filter": {"type": "string",
                           "description": "Optional substring filter on tool name"}
            },
        },
    })
    return tools


def _call_tool(name: str, args: Dict, registry, dispatch: Dict) -> Any:
    """Call an Operon tool and return a serialisable result."""
    if name == "operon_status":
        return {
            "status": "ok", "server": "operon-mcp",
            "tool_count": len(dispatch),
            "ts": time.time(),
        }
    if name == "operon_help":
        flt = args.get("filter", "").lower()
        tools = _get_tool_defs(registry)
        if flt:
            tools = [t for t in tools if flt in t["name"].lower()
                     or flt in t["description"].lower()]
        return {"tools": tools, "count": len(tools)}

    fn = dispatch.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name!r}"}
    try:
        result = fn(**args)
        return result
    except Exception as e:
        return {"error": str(e), "tool": name}


# ---------------------------------------------------------------------------
# FastMCP server (preferred)
# ---------------------------------------------------------------------------

def _run_fastmcp(registry, dispatch: Dict) -> None:
    """Run using the `mcp` package's FastMCP (preferred)."""
    log.info("Starting Operon MCP server via FastMCP (stdio)")
    mcp = _FastMCP("operon")

    tool_defs = _get_tool_defs(registry)

    # Dynamically register each tool
    for td in tool_defs:
        name   = td["name"]
        desc   = td["description"]
        schema = td["inputSchema"]

        # Capture name in closure
        def _make_handler(_name, _dispatch, _registry):
            def _handler(**kwargs):
                result = _call_tool(_name, kwargs, _registry, _dispatch)
                if isinstance(result, (dict, list)):
                    return json.dumps(result, ensure_ascii=False, indent=2)
                return str(result)
            _handler.__name__ = _name
            _handler.__doc__  = desc
            return _handler

        handler = _make_handler(name, dispatch, registry)
        # Register with mcp
        try:
            mcp.tool(name=name, description=desc)(handler)
        except Exception as e:
            log.debug("FastMCP tool registration failed for %s: %s", name, e)

    mcp.run(transport="stdio")


# ---------------------------------------------------------------------------
# Minimal JSON-RPC 2.0 server (fallback when `mcp` not installed)
# ---------------------------------------------------------------------------

class _MinimalMCPServer:
    """
    Minimal stdio JSON-RPC 2.0 MCP server implementation.
    Handles: initialize, tools/list, tools/call.
    Line-delimited JSON over stdin/stdout.
    """

    PROTOCOL_VERSION = "2024-11-05"
    SERVER_INFO = {"name": "operon", "version": "1.0.0"}

    def __init__(self, registry, dispatch: Dict) -> None:
        self._registry = registry
        self._dispatch = dispatch
        self._tools    = _get_tool_defs(registry)
        self._initialized = False
        self._req_id_ctr  = 0

    def run(self) -> None:
        log.info("Starting Operon MCP server (minimal JSON-RPC 2.0, stdio)")
        # Force binary-safe stdin/stdout
        stdin  = sys.stdin.buffer
        stdout = sys.stdout.buffer

        while True:
            try:
                line = stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as e:
                    self._send_error(stdout, None, -32700, f"Parse error: {e}")
                    continue
                self._handle(msg, stdout)
            except KeyboardInterrupt:
                break
            except Exception as e:
                log.error("MCP server error: %s", e)

    def _handle(self, msg: Dict, stdout) -> None:
        req_id  = msg.get("id")
        method  = msg.get("method", "")
        params  = msg.get("params", {})

        if method == "initialize":
            self._initialized = True
            self._send_result(stdout, req_id, {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities":    {"tools": {}},
                "serverInfo":      self.SERVER_INFO,
            })
            # Send initialized notification
            self._send_notification(stdout, "notifications/initialized", {})

        elif method == "tools/list":
            cursor = params.get("cursor")
            # Pagination stub (return all)
            self._send_result(stdout, req_id, {"tools": self._tools})

        elif method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments", {}) or {}
            result = _call_tool(name, args, self._registry, self._dispatch)
            # MCP tools/call result format
            if isinstance(result, dict) and "error" in result and len(result) == 1:
                self._send_result(stdout, req_id, {
                    "content": [{"type": "text", "text": str(result["error"])}],
                    "isError":  True,
                })
            else:
                text = (json.dumps(result, ensure_ascii=False, indent=2)
                        if isinstance(result, (dict, list)) else str(result))
                self._send_result(stdout, req_id, {
                    "content": [{"type": "text", "text": text}],
                    "isError":  False,
                })

        elif method == "ping":
            self._send_result(stdout, req_id, {})

        elif method.startswith("notifications/"):
            pass  # Client notifications — no response needed

        else:
            if req_id is not None:
                self._send_error(stdout, req_id, -32601, f"Method not found: {method!r}")

    def _send_result(self, stdout, req_id, result: Any) -> None:
        self._write(stdout, {"jsonrpc": "2.0", "id": req_id, "result": result})

    def _send_error(self, stdout, req_id, code: int, message: str) -> None:
        self._write(stdout, {"jsonrpc": "2.0", "id": req_id,
                             "error": {"code": code, "message": message}})

    def _send_notification(self, stdout, method: str, params: Any) -> None:
        self._write(stdout, {"jsonrpc": "2.0", "method": method, "params": params})

    @staticmethod
    def _write(stdout, obj: Dict) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        stdout.write(line.encode("utf-8"))
        stdout.flush()


# ---------------------------------------------------------------------------
# HTTP / SSE server (for network MCP clients)
# ---------------------------------------------------------------------------

def _run_http_server(registry, dispatch: Dict, host: str = "127.0.0.1",
                     port: int = 3456) -> None:
    """Run a minimal HTTP/SSE MCP server."""
    try:
        from http.server import BaseHTTPRequestHandler, HTTPServer
    except ImportError:
        print("HTTP server unavailable", file=sys.stderr)
        return

    tools = _get_tool_defs(registry)

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass  # suppress request log

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                msg = json.loads(body)
            except Exception:
                self._respond({"error": "bad JSON"}, 400)
                return

            method = msg.get("method", "")
            params = msg.get("params", {}) or {}
            req_id = msg.get("id")

            if method == "initialize":
                result = {
                    "protocolVersion": _MinimalMCPServer.PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": _MinimalMCPServer.SERVER_INFO,
                }
            elif method == "tools/list":
                result = {"tools": tools}
            elif method == "tools/call":
                name = params.get("name", "")
                args = params.get("arguments", {}) or {}
                res  = _call_tool(name, args, registry, dispatch)
                text = (json.dumps(res, ensure_ascii=False, indent=2)
                        if isinstance(res, (dict, list)) else str(res))
                result = {"content": [{"type": "text", "text": text}]}
            else:
                result = {"error": f"unknown method: {method}"}

            self._respond({"jsonrpc": "2.0", "id": req_id, "result": result})

        def _respond(self, payload: Dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer((host, port), _Handler)
    print(f"[Operon MCP] HTTP server listening on http://{host}:{port}", flush=True)
    print(f"[Operon MCP] {len(tools)} tools available", flush=True)
    server.serve_forever()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_server(mode: str = "stdio", host: str = "127.0.0.1",
                 port: int = 3456) -> None:
    """Start the MCP server.

    mode: "stdio" (default) or "http"
    """
    os.chdir(_OPERON_DIR)
    try:
        registry, dispatch = _load_registry()
        n_tools = len(dispatch)
        log.info("Operon MCP server: %d tools loaded", n_tools)
    except Exception as e:
        print(f"[Operon MCP] Failed to load tool registry: {e}", file=sys.stderr)
        sys.exit(1)

    if mode == "http":
        _run_http_server(registry, dispatch, host=host, port=port)
        return

    # stdio mode
    if _HAS_FASTMCP:
        _run_fastmcp(registry, dispatch)
    else:
        server = _MinimalMCPServer(registry, dispatch)
        server.run()


def generate_client_config(mode: str = "stdio", port: int = 3456) -> str:
    """Return a JSON config snippet for Claude Desktop / Claude Code."""
    operon_path = str(_OPERON_DIR)
    py = sys.executable

    if mode == "http":
        cfg = {
            "mcpServers": {
                "operon": {
                    "url": f"http://127.0.0.1:{port}",
                    "type": "http",
                }
            }
        }
    else:
        cfg = {
            "mcpServers": {
                "operon": {
                    "command": py,
                    "args": ["-m", "core.mcp_server"],
                    "cwd": operon_path,
                }
            }
        }
    return json.dumps(cfg, indent=2)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    parser = argparse.ArgumentParser(description="Operon MCP Server")
    parser.add_argument("--http",   action="store_true", help="Run HTTP server instead of stdio")
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   type=int, default=3456)
    parser.add_argument("--config", action="store_true",
                        help="Print client config and exit")
    args = parser.parse_args()

    if args.config:
        mode = "http" if args.http else "stdio"
        print(generate_client_config(mode=mode, port=args.port))
        sys.exit(0)

    mode = "http" if args.http else "stdio"
    start_server(mode=mode, host=args.host, port=args.port)
