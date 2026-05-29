"""
Operon Web Dashboard.

A lightweight, zero-dependency (uses Python stdlib only) local web dashboard
that gives you a browser-based view of:
  • Live session history
  • Long-term memory contents
  • Tool call log (last 200 calls)
  • System status (model, provider, CPU/RAM)
  • One-click memory management

Runs on http://localhost:7270 by default.
Uses a self-contained single-page app served from a single HTTP handler.
Auto-refreshes every 5 seconds via EventSource (SSE).
"""

import json
import os
import threading
import time
import webbrowser
from http.server   import HTTPServer, BaseHTTPRequestHandler
from pathlib       import Path
from urllib.parse  import urlparse, parse_qs
from typing        import Optional, Callable

_DEFAULT_PORT   = 7270
_DEFAULT_HOST   = "127.0.0.1"
_TOOL_LOG_LIMIT = 200  # keep last N tool calls in memory

# ── Global state shared with the main process ─────────────────────────────────

_tool_log:       list[dict] = []
_tool_log_lock   = threading.Lock()

# These are set by DashboardServer.start() via callbacks:
_get_session:    Optional[Callable] = None   # () → list[dict] messages
_get_memory:     Optional[Callable] = None   # () → list[dict]
_get_status:     Optional[Callable] = None   # () → dict
_delete_memory:  Optional[Callable] = None   # (int id) → None
_clear_memory:   Optional[Callable] = None   # () → None


def log_tool_call(tool_name: str, params: dict, result: dict) -> None:
    """Called from the agent loop to record tool usage."""
    entry = {
        "ts":        time.strftime("%H:%M:%S"),
        "tool":      tool_name,
        "params":    {k: str(v)[:80] for k, v in list(params.items())[:5]},
        "success":   result.get("success", False),
        "preview":   str(result.get("output", "") or "")[:120],
    }
    with _tool_log_lock:
        _tool_log.append(entry)
        if len(_tool_log) > _TOOL_LOG_LIMIT:
            del _tool_log[0]


# ── HTML / CSS / JS (self-contained SPA) ─────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Operon Dashboard</title>
<style>
  :root{--bg:#0e0e14;--surface:#16161f;--border:#2a2a3a;--accent:#9b59ff;
        --green:#39d98a;--red:#ff5454;--text:#e0e0f0;--dim:#7070a0;--mono:'JetBrains Mono',monospace}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px}
  header{background:var(--surface);border-bottom:1px solid var(--border);
         padding:12px 24px;display:flex;align-items:center;gap:16px}
  header h1{font-size:20px;color:var(--accent);letter-spacing:2px;font-weight:700}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .tabs{display:flex;gap:0;border-bottom:1px solid var(--border);padding:0 24px;background:var(--surface)}
  .tab{padding:10px 20px;cursor:pointer;color:var(--dim);border-bottom:2px solid transparent;font-weight:500}
  .tab.active{color:var(--accent);border-bottom-color:var(--accent)}
  .pane{display:none;padding:20px 24px;height:calc(100vh - 100px);overflow-y:auto}
  .pane.active{display:block}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:12px}
  .card h3{color:var(--accent);margin-bottom:8px;font-size:12px;text-transform:uppercase;letter-spacing:1px}
  .status-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
  .stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px}
  .stat .val{font-size:24px;font-weight:700;color:var(--accent)}
  .stat .lbl{color:var(--dim);font-size:11px;margin-top:4px;text-transform:uppercase}
  .msg{padding:8px 12px;border-radius:6px;margin-bottom:6px;font-family:var(--mono);font-size:12px;line-height:1.5}
  .msg.user{background:#1a1a2e;border-left:3px solid #4a9eff}
  .msg.assistant{background:#1a1e1a;border-left:3px solid var(--green)}
  .msg .role{font-weight:700;margin-bottom:4px;font-size:10px;text-transform:uppercase;color:var(--dim)}
  .mem-row{display:flex;align-items:start;gap:10px;padding:8px 0;border-bottom:1px solid var(--border)}
  .mem-row:last-child{border-bottom:none}
  .badge{padding:2px 8px;border-radius:12px;font-size:10px;font-weight:600;text-transform:uppercase}
  .badge-pref{background:#2a1f4a;color:#b09fff}
  .badge-fact{background:#1a2a1a;color:#80ff80}
  .badge-manual{background:#2a2000;color:#ffcc00}
  .badge-legacy{background:#2a1a1a;color:#ff8080}
  .mem-content{flex:1;font-size:13px;line-height:1.4}
  .mem-del{cursor:pointer;color:var(--red);font-size:16px;padding:0 6px;line-height:1}
  .tool-row{font-family:var(--mono);font-size:12px;padding:6px 0;border-bottom:1px solid var(--border);display:flex;gap:10px;align-items:start}
  .tool-row .ts{color:var(--dim);min-width:60px}
  .tool-row .name{color:var(--accent);min-width:180px}
  .tool-row .ok{color:var(--green)}
  .tool-row .fail{color:var(--red)}
  .tool-row .preview{color:var(--dim);flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
  .btn{padding:6px 14px;border-radius:6px;border:1px solid var(--border);background:var(--surface);
       color:var(--text);cursor:pointer;font-size:12px}
  .btn:hover{border-color:var(--accent);color:var(--accent)}
  .btn-danger{border-color:var(--red);color:var(--red)}
  .btn-danger:hover{background:#2a0000}
  .toolbar{margin-bottom:14px;display:flex;gap:8px;align-items:center}
  #refresh-ts{color:var(--dim);font-size:11px;margin-left:auto}
</style>
</head>
<body>
<header>
  <div class="dot"></div>
  <h1>OPERON</h1>
  <span style="color:var(--dim);font-size:12px">Dashboard</span>
  <span id="model-badge" style="margin-left:auto;color:var(--dim);font-size:12px"></span>
</header>

<div class="tabs">
  <div class="tab active" onclick="showTab('status')">Status</div>
  <div class="tab" onclick="showTab('session')">Session</div>
  <div class="tab" onclick="showTab('memory')">Memory</div>
  <div class="tab" onclick="showTab('tools')">Tool Log</div>
</div>

<div id="pane-status" class="pane active">
  <div id="status-grid" class="status-grid"></div>
</div>

<div id="pane-session" class="pane">
  <div class="toolbar">
    <span id="msg-count" style="color:var(--dim)"></span>
    <span id="refresh-ts"></span>
  </div>
  <div id="session-msgs"></div>
</div>

<div id="pane-memory" class="pane">
  <div class="toolbar">
    <button class="btn btn-danger" onclick="clearMemory()">Clear All Memory</button>
    <span id="mem-count" style="color:var(--dim)"></span>
  </div>
  <div id="memory-list"></div>
</div>

<div id="pane-tools" class="pane">
  <div class="toolbar">
    <span id="tool-count" style="color:var(--dim)"></span>
  </div>
  <div id="tool-log"></div>
</div>

<script>
function showTab(name){
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',['status','session','memory','tools'][i]===name));
  document.querySelectorAll('.pane').forEach(p=>p.classList.toggle('active',p.id==='pane-'+name));
  refresh();
}

async function apiFetch(path){
  try{const r=await fetch(path);return await r.json();}catch{return null;}
}

function escHtml(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

function badgeClass(type){return{preference:'badge-pref',fact:'badge-fact',manual:'badge-manual'}[type]||'badge-legacy';}

async function refresh(){
  const now=new Date().toLocaleTimeString();
  document.getElementById('refresh-ts').textContent='Updated '+now;

  // Status
  const st=await apiFetch('/api/status');
  if(st){
    document.getElementById('model-badge').textContent=st.model||'';
    const grid=document.getElementById('status-grid');
    const stats=[
      ['Model',st.model||'—'],['Provider',st.provider||'—'],
      ['Turns',st.turns||0],['Messages',st.messages||0],
      ['Memory Items',st.memory_items||0],['Skills',st.skills||0],
      ['Tools',st.tools||0],['CPU',st.cpu||'—'],['RAM',st.ram||'—'],
    ];
    grid.innerHTML=stats.map(([l,v])=>`<div class="stat"><div class="val">${escHtml(v)}</div><div class="lbl">${l}</div></div>`).join('');
  }

  // Session
  const sess=await apiFetch('/api/session');
  if(sess){
    const msgs=sess.messages||[];
    document.getElementById('msg-count').textContent=msgs.length+' messages';
    document.getElementById('session-msgs').innerHTML=msgs.slice(-60).map(m=>`
      <div class="msg ${m.role}">
        <div class="role">${m.role}</div>
        <div>${escHtml(String(m.content||'').slice(0,400))}</div>
      </div>`).join('');
  }

  // Memory
  const mem=await apiFetch('/api/memory');
  if(mem){
    const items=mem.items||[];
    document.getElementById('mem-count').textContent=items.length+' items';
    document.getElementById('memory-list').innerHTML=items.map(m=>`
      <div class="mem-row">
        <span class="badge ${badgeClass(m.type)}">${m.type||'?'}</span>
        <span class="mem-content">${escHtml(m.content||'')}</span>
        <span class="mem-del" onclick="deleteMem(${m.id})" title="Delete">✕</span>
      </div>`).join('');
  }

  // Tool log
  const tl=await apiFetch('/api/tools');
  if(tl){
    const calls=tl.calls||[];
    document.getElementById('tool-count').textContent=calls.length+' calls';
    document.getElementById('tool-log').innerHTML=[...calls].reverse().map(c=>`
      <div class="tool-row">
        <span class="ts">${escHtml(c.ts)}</span>
        <span class="name">${escHtml(c.tool)}</span>
        <span class="${c.success?'ok':'fail'}">${c.success?'✓':'✗'}</span>
        <span class="preview">${escHtml(c.preview||'')}</span>
      </div>`).join('');
  }
}

async function deleteMem(id){
  await fetch('/api/memory/delete?id='+id,{method:'POST'});
  refresh();
}

async function clearMemory(){
  if(!confirm('Clear all long-term memories?')) return;
  await fetch('/api/memory/clear',{method:'POST'});
  refresh();
}

// Auto-refresh every 5 seconds
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


# ── HTTP request handler ──────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        pass  # Suppress access log noise

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        qs     = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._send(200, "text/html", _HTML.encode())
        elif path == "/api/status":
            self._json(_get_status() if _get_status else {})
        elif path == "/api/session":
            msgs = _get_session() if _get_session else []
            self._json({"messages": [{"role": m.get("role"), "content": str(m.get("content", ""))[:500]}
                                     for m in msgs[-60:]]})
        elif path == "/api/memory":
            items = _get_memory() if _get_memory else []
            self._json({"items": items})
        elif path == "/api/tools":
            with _tool_log_lock:
                calls = list(_tool_log)
            self._json({"calls": calls})
        else:
            self._send(404, "text/plain", b"Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        qs     = parse_qs(parsed.query)

        if path == "/api/memory/delete":
            try:
                mem_id = int(qs.get("id", [0])[0])
                if _delete_memory:
                    _delete_memory(mem_id)
            except Exception:
                pass
            self._json({"ok": True})
        elif path == "/api/memory/clear":
            if _clear_memory:
                _clear_memory()
            self._json({"ok": True})
        else:
            self._send(404, "text/plain", b"Not found")

    def _json(self, data):
        body = json.dumps(data, default=str).encode()
        self._send(200, "application/json", body)

    def _send(self, status: int, content_type: str, body: bytes):
        # Determine request origin for CORS — only allow localhost origins.
        origin = self.headers.get("Origin", "")
        allowed_origin = (
            origin if (origin.startswith("http://localhost") or
                       origin.startswith("http://127.0.0.1"))
            else "http://localhost:7270"
        )

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Restrict CORS to localhost only — no wildcard in production dashboard
        self.send_header("Access-Control-Allow-Origin", allowed_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        # Content-Security-Policy: allow only self and inline scripts (for the embedded UI)
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'"
        )
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.end_headers()
        self.wfile.write(body)


# ── Dashboard server ──────────────────────────────────────────────────────────

class DashboardServer:

    def __init__(self, host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT):
        self._host   = host
        self._port   = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(
        self,
        get_session:   Callable,
        get_memory:    Callable,
        get_status:    Callable,
        delete_memory: Callable,
        clear_memory:  Callable,
        open_browser:  bool = False,
    ) -> str:
        """
        Start the dashboard HTTP server.
        Pass in callables that the handler will invoke to fetch live data.
        Returns the dashboard URL.
        """
        global _get_session, _get_memory, _get_status, _delete_memory, _clear_memory

        _get_session   = get_session
        _get_memory    = get_memory
        _get_status    = get_status
        _delete_memory = delete_memory
        _clear_memory  = clear_memory

        if self._running:
            return self.url

        self._server = HTTPServer((self._host, self._port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="operon-dashboard",
        )
        self._thread.start()
        self._running = True

        url = self.url
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        return url

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None
        self._running = False

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def running(self) -> bool:
        return self._running

    def status(self) -> dict:
        return {
            "running": self._running,
            "url":     self.url if self._running else "",
            "host":    self._host,
            "port":    self._port,
        }
