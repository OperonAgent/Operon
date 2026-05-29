"""
Operon MCP (Model Context Protocol) Client.

Connects Operon to any external MCP server and dynamically registers
its tools into Operon's live tool registry at runtime.

Supports two transport types:
  • stdio  — launches a local process, communicates over stdin/stdout
  • http   — connects to a running HTTP/SSE MCP server

JSON-RPC 2.0 wire format per the MCP spec.

Usage:
    from core.mcp import MCPManager
    mcp = MCPManager()
    mcp.connect("filesystem", "stdio", command=["npx", "@modelcontextprotocol/server-filesystem", "/tmp"])
    mcp.connect("my-api",     "http",  url="http://localhost:3000/mcp")

    tools = mcp.list_all_tools()        # [{name, description, server, params}, ...]
    result = mcp.call_tool("server_name", "tool_name", {"arg": "value"})
"""

import json
import subprocess
import threading
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Optional

try:
    import requests as _requests
    _HTTP_OK = True
except ImportError:
    _HTTP_OK = False

log = logging.getLogger("operon.mcp")

MCP_CONFIG_PATH = Path.home() / ".operon" / "mcp_servers.json"


# ── JSON-RPC helpers ──────────────────────────────────────────────────────────

def _make_request(method: str, params: Any = None, req_id: Any = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "id":      req_id or str(uuid.uuid4())[:8],
        "method":  method,
        **({"params": params} if params is not None else {}),
    }


# ── Stdio transport ───────────────────────────────────────────────────────────

class _StdioTransport:
    """
    Launches a subprocess MCP server and communicates via stdin/stdout.
    Each JSON-RPC message is sent as a single newline-terminated line.
    """

    def __init__(self, command: list[str], env: dict = None):
        import os
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proc_env,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        log.debug("Stdio MCP process started: PID %d", self._proc.pid)

    def _read_loop(self):
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    msg_id = str(msg.get("id", ""))
                    with self._lock:
                        if msg_id in self._pending:
                            self._pending[msg_id] = msg
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

    def send_request(self, method: str, params: Any = None, timeout: float = 15.0) -> dict:
        req = _make_request(method, params)
        req_id = req["id"]
        with self._lock:
            self._pending[req_id] = None

        line = json.dumps(req) + "\n"
        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except Exception as e:
            with self._lock:
                self._pending.pop(req_id, None)
            return {"error": {"code": -1, "message": f"Write failed: {e}"}}

        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                resp = self._pending.get(req_id)
            if resp is not None:
                with self._lock:
                    del self._pending[req_id]
                return resp
            time.sleep(0.05)

        with self._lock:
            self._pending.pop(req_id, None)
        return {"error": {"code": -32000, "message": f"Timeout waiting for {method}"}}

    def close(self):
        try:
            self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=3)
        except Exception:
            pass


# ── HTTP transport ────────────────────────────────────────────────────────────

class _HttpTransport:
    """
    Calls a running HTTP MCP server via POST /mcp (standard Streamable HTTP transport).
    Falls back to classic JSON-RPC POST if the server doesn't respond to /mcp.
    """

    def __init__(self, url: str, headers: dict = None):
        if not _HTTP_OK:
            raise ImportError("requests library required for HTTP MCP transport.")
        self._base = url.rstrip("/")
        self._headers = {"Content-Type": "application/json", **(headers or {})}

    def send_request(self, method: str, params: Any = None, timeout: float = 15.0) -> dict:
        req = _make_request(method, params)
        # Try /mcp endpoint first (Streamable HTTP), fall back to base URL
        for endpoint in (f"{self._base}/mcp", self._base):
            try:
                resp = _requests.post(
                    endpoint,
                    json=req,
                    headers=self._headers,
                    timeout=timeout,
                )
                if resp.status_code == 404 and endpoint.endswith("/mcp"):
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if endpoint == self._base:
                    return {"error": {"code": -1, "message": str(e)}}
        return {"error": {"code": -1, "message": "All endpoints failed"}}

    def close(self):
        pass  # stateless


# ── MCP Server connection ─────────────────────────────────────────────────────

class _MCPServer:
    """Represents one connected MCP server with its discovered tools."""

    def __init__(self, name: str, transport):
        self.name       = name
        self._transport = transport
        self.tools: list[dict] = []
        self._initialized = False

    def initialize(self) -> bool:
        """Perform the MCP initialize handshake and discover tools."""
        resp = self._transport.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities":    {"tools": {}},
            "clientInfo":      {"name": "operon", "version": "2.0"},
        })
        if "error" in resp:
            log.warning("MCP init error for %s: %s", self.name, resp["error"])
            return False

        # Send initialized notification (no response expected)
        try:
            notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            if isinstance(self._transport, _StdioTransport):
                self._transport._proc.stdin.write(json.dumps(notif) + "\n")
                self._transport._proc.stdin.flush()
        except Exception:
            pass

        self._initialized = True
        self._discover_tools()
        return True

    def _discover_tools(self):
        resp = self._transport.send_request("tools/list")
        if "error" in resp:
            log.warning("tools/list failed for %s: %s", self.name, resp["error"])
            return
        result = resp.get("result", {})
        raw_tools = result.get("tools", [])
        self.tools = []
        for t in raw_tools:
            self.tools.append({
                "name":        t.get("name", ""),
                "description": t.get("description", ""),
                "server":      self.name,
                "input_schema": t.get("inputSchema", {}),
            })
        log.info("MCP server '%s' registered %d tools.", self.name, len(self.tools))

    def call_tool(self, tool_name: str, arguments: dict, timeout: float = 30.0) -> dict:
        resp = self._transport.send_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout=timeout,
        )
        if "error" in resp:
            return {"success": False, "output": None, "error": str(resp["error"])}
        result = resp.get("result", {})
        # MCP tools/call returns a "content" array
        content = result.get("content", [])
        if content:
            text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return {
                "success": True,
                "output":  "\n".join(text_parts) if text_parts else json.dumps(content),
                "error":   "",
            }
        return {"success": True, "output": json.dumps(result), "error": ""}

    def close(self):
        self._transport.close()


# ── MCP Manager ───────────────────────────────────────────────────────────────

class MCPManager:
    """
    Manages multiple MCP server connections. Integrates with ToolRegistry
    by providing a callable per discovered tool and patching the dispatch map.
    """

    def __init__(self):
        self._servers: dict[str, _MCPServer] = {}
        self._lock = threading.Lock()
        self._auto_load()

    # ── Connection management ─────────────────────────────────────────────────

    def connect(
        self,
        name:    str,
        transport: str = "stdio",
        *,
        command: list[str]  = None,
        url:     str        = None,
        headers: dict       = None,
        env:     dict       = None,
    ) -> bool:
        """
        Connect to an MCP server.

        For stdio: pass command=["npx", "my-mcp-server", ...]
        For http:  pass url="http://localhost:3000"
        """
        try:
            if transport == "stdio":
                if not command:
                    raise ValueError("'command' is required for stdio transport")
                tr = _StdioTransport(command, env=env)
            elif transport == "http":
                if not url:
                    raise ValueError("'url' is required for http transport")
                tr = _HttpTransport(url, headers=headers)
            else:
                raise ValueError(f"Unknown transport: {transport!r}")

            server = _MCPServer(name, tr)
            if not server.initialize():
                tr.close()
                return False

            with self._lock:
                # Disconnect existing server with same name
                if name in self._servers:
                    self._servers[name].close()
                self._servers[name] = server

            log.info("MCP server '%s' connected (%d tools).", name, len(server.tools))
            self._persist_server(name, transport, command=command, url=url,
                                 headers=headers, env=env)
            return True

        except Exception as e:
            log.error("Failed to connect MCP server '%s': %s", name, e)
            return False

    def disconnect(self, name: str) -> bool:
        with self._lock:
            server = self._servers.pop(name, None)
        if server:
            server.close()
            self._remove_persisted(name)
            return True
        return False

    # ── Tool access ───────────────────────────────────────────────────────────

    def list_all_tools(self) -> list[dict]:
        """Return metadata for every tool across all connected servers."""
        tools = []
        with self._lock:
            for server in self._servers.values():
                tools.extend(server.tools)
        return tools

    def call_tool(self, server_name: str, tool_name: str,
                  arguments: dict, timeout: float = 30.0) -> dict:
        with self._lock:
            server = self._servers.get(server_name)
        if server is None:
            return {"success": False, "output": None,
                    "error": f"MCP server '{server_name}' not connected."}
        return server.call_tool(tool_name, arguments, timeout=timeout)

    def call_tool_auto(self, tool_name: str, arguments: dict) -> dict:
        """
        Call a tool by name alone — searches all connected servers for it.
        Useful when Operon's dispatcher calls an MCP tool without knowing
        which server owns it.
        """
        with self._lock:
            servers = dict(self._servers)
        for server in servers.values():
            for t in server.tools:
                if t["name"] == tool_name:
                    return server.call_tool(tool_name, arguments)
        return {"success": False, "output": None,
                "error": f"MCP tool '{tool_name}' not found on any connected server."}

    # ── Operon registry integration ───────────────────────────────────────────

    def inject_into_registry(self, dispatch: dict, definitions: list) -> int:
        """
        Inject all MCP tools into Operon's live dispatch map and definitions list.
        Skips tools already present in either structure (idempotent).
        Returns the number of tools newly added.
        """
        added = 0
        existing_def_names = {d["name"] for d in definitions}

        for tool in self.list_all_tools():
            name   = f"mcp__{tool['server']}__{tool['name']}"
            schema = tool.get("input_schema", {})
            props  = schema.get("properties", {})

            # Build params description from JSON schema properties
            params_desc = {
                k: f"{v.get('type', 'any')} — {v.get('description', '')}"
                for k, v in props.items()
            }

            is_new = name not in existing_def_names and name not in dispatch

            if name not in existing_def_names:
                definitions.append({
                    "name":        name,
                    "description": f"[MCP:{tool['server']}] {tool['description']}",
                    "params":      params_desc,
                })
                existing_def_names.add(name)  # prevent re-adding on next server

            # Capture loop variable in closure (always update dispatch)
            server_name = tool["server"]
            tool_name   = tool["name"]

            def _make_caller(sn, tn):
                def _call(**kwargs) -> dict:
                    return self.call_tool(sn, tn, kwargs)
                return _call

            dispatch[name] = _make_caller(server_name, tool_name)
            if is_new:
                added += 1

        return added

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "name":       name,
                    "tools":      len(server.tools),
                    "tool_names": [t["name"] for t in server.tools],
                }
                for name, server in self._servers.items()
            ]

    def close_all(self):
        with self._lock:
            servers = list(self._servers.values())
            self._servers.clear()
        for s in servers:
            s.close()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        try:
            if MCP_CONFIG_PATH.exists():
                return json.loads(MCP_CONFIG_PATH.read_text())
        except Exception:
            pass
        return {}

    def _save_config(self, cfg: dict):
        try:
            MCP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            MCP_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        except Exception:
            pass

    def _persist_server(self, name, transport, *, command, url, headers, env):
        cfg = self._load_config()
        cfg[name] = {
            "transport": transport,
            "command":   command,
            "url":       url,
            "headers":   headers,
            "env":       env,
        }
        self._save_config(cfg)

    def _remove_persisted(self, name):
        cfg = self._load_config()
        cfg.pop(name, None)
        self._save_config(cfg)

    def _auto_load(self):
        """On startup, reconnect any previously saved MCP servers."""
        cfg = self._load_config()
        for name, opts in cfg.items():
            try:
                self.connect(
                    name,
                    opts.get("transport", "stdio"),
                    command=opts.get("command"),
                    url=opts.get("url"),
                    headers=opts.get("headers"),
                    env=opts.get("env"),
                )
            except Exception as e:
                log.warning("Auto-reconnect failed for MCP server '%s': %s", name, e)

    def __repr__(self):
        return f"<MCPManager servers={list(self._servers.keys())}>"
