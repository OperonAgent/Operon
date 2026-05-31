"""
Operon Cloud Execution Backends.

Inspired by Hermes Agent's Modal / Daytona / Singularity support.
Run code or commands on cloud workers without provisioning servers.

Backends
--------
  Modal   — serverless Python functions (modal.com)
            pip install modal && modal setup
  Daytona — managed dev environments (daytona.io)
            pip install daytona-sdk OR use REST API
            export DAYTONA_API_KEY=dtn_xxxxxxxxxx
            export DAYTONA_SERVER_URL=https://app.daytona.io

All functions accept **_ for registry compatibility.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.request   # noqa: F401 — used as urllib.request.* in Daytona helpers
import urllib.error     # noqa: F401 — used as urllib.error.* in Daytona helpers
from pathlib import Path
from typing import Any, Dict, Optional


# ── Modal ─────────────────────────────────────────────────────────────────────

def modal_run(
    code:         str   = "",
    requirements: list  = None,
    image:        str   = "python:3.12",
    timeout:      int   = 60,
    gpu:          str   = "",
    **_,
) -> dict:
    """
    Run Python code on a Modal serverless worker.
    Requires: pip install modal && modal setup (one-time auth)

    Args:
        code         — Python code to execute (required)
        requirements — list of pip packages to install e.g. ['numpy', 'pandas'] (optional)
        image        — base Docker image (optional, default 'python:3.12')
        timeout      — seconds before timeout (optional, default 60)
        gpu          — GPU type e.g. 'T4', 'A10G', 'A100' — leave empty for CPU (optional)

    Returns:
        {success, stdout, stderr, elapsed_ms, error}
    """
    if not code:
        return {"success": False, "error": "code is required."}

    try:
        import modal
    except ImportError:
        return {
            "success": False,
            "error": "modal not installed. Run: pip install modal && modal setup",
        }

    reqs = requirements or []
    t0   = time.monotonic()

    try:
        # Build the Modal image
        base_image = modal.Image.from_registry(image)
        if reqs:
            base_image = base_image.pip_install(*reqs)

        # Write code to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp = f.name

        app  = modal.App()
        kwargs: Dict[str, Any] = {"image": base_image, "timeout": timeout}
        if gpu:
            kwargs["gpu"] = gpu

        @app.function(**kwargs)
        def _run():
            import subprocess as _sp
            r = _sp.run(["python3", "/tmp/operon_code.py"],
                        capture_output=True, text=True)
            return r.stdout, r.stderr, r.returncode

        # Mount and run
        with modal.runner.deploy_stub(app):
            stdout, stderr, rc = _run.remote()

        elapsed = round((time.monotonic() - t0) * 1000)
        return {
            "success":    rc == 0,
            "stdout":     stdout,
            "stderr":     stderr,
            "returncode": rc,
            "elapsed_ms": elapsed,
            "backend":    "modal",
            "error":      stderr if rc != 0 else "",
        }
    except Exception as e:
        return {
            "success":    False,
            "stdout":     "",
            "stderr":     str(e),
            "elapsed_ms": round((time.monotonic() - t0) * 1000),
            "backend":    "modal",
            "error":      str(e),
        }
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


def modal_status(**_) -> dict:
    """Check whether Modal is installed and authenticated."""
    try:
        import modal
        result = subprocess.run(
            ["modal", "profile", "current"],
            capture_output=True, text=True, timeout=10,
        )
        return {
            "success":       True,
            "installed":     True,
            "authenticated": result.returncode == 0,
            "profile":       result.stdout.strip(),
            "error":         "",
        }
    except ImportError:
        return {"success": False, "installed": False, "error": "pip install modal"}
    except Exception as e:
        return {"success": False, "installed": True, "error": str(e)}


# ── Daytona ───────────────────────────────────────────────────────────────────

def _daytona_headers() -> dict:
    key = os.environ.get("DAYTONA_API_KEY", "").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _daytona_url(path: str) -> str:
    base = os.environ.get("DAYTONA_SERVER_URL", "https://app.daytona.io").rstrip("/")
    return f"{base}/api{path}"


def daytona_run(
    command:      str  = "",
    workspace_id: str  = "",
    image:        str  = "ubuntu:22.04",
    timeout:      int  = 60,
    **_,
) -> dict:
    """
    Run a shell command in a Daytona managed workspace.
    Requires DAYTONA_API_KEY env var.

    Args:
        command      — shell command to execute (required)
        workspace_id — existing workspace ID (optional — creates ephemeral one if not given)
        image        — Docker image for new workspace (optional, default 'ubuntu:22.04')
        timeout      — seconds before timeout (optional, default 60)

    Returns:
        {success, stdout, stderr, returncode, elapsed_ms, workspace_id, error}
    """
    if not command:
        return {"success": False, "error": "command is required."}

    api_key = os.environ.get("DAYTONA_API_KEY", "").strip()
    if not api_key:
        return {
            "success": False,
            "error": (
                "DAYTONA_API_KEY not set.\n"
                "  1. Sign up at https://daytona.io\n"
                "  2. export DAYTONA_API_KEY=dtn_xxxxxxxxxx"
            ),
        }

    def _post(path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        r    = urllib.request.Request(
            _daytona_url(path), data=data, headers=_daytona_headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read())
            except Exception:
                return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

    t0 = time.monotonic()

    # Create or reuse workspace
    ws_id = workspace_id
    created = False
    if not ws_id:
        ws_resp = _post("/workspaces", {
            "name":  f"operon-{int(time.time())}",
            "image": image,
        })
        ws_id   = ws_resp.get("id", "")
        created = bool(ws_id)
        if not ws_id:
            return {
                "success": False,
                "error":   ws_resp.get("error", "Failed to create Daytona workspace"),
            }
        # Wait for workspace to be ready
        time.sleep(3)

    # Execute command
    exec_resp = _post(f"/workspaces/{ws_id}/exec", {"command": command, "timeout": timeout})
    elapsed   = round((time.monotonic() - t0) * 1000)

    result = {
        "success":      exec_resp.get("exit_code", -1) == 0,
        "stdout":       exec_resp.get("stdout", ""),
        "stderr":       exec_resp.get("stderr", ""),
        "returncode":   exec_resp.get("exit_code", -1),
        "elapsed_ms":   elapsed,
        "workspace_id": ws_id,
        "backend":      "daytona",
        "error":        exec_resp.get("error", "") or (exec_resp.get("stderr", "") if exec_resp.get("exit_code", 0) != 0 else ""),
    }

    # Delete ephemeral workspace
    if created:
        try:
            _req_del = urllib.request.Request(
                _daytona_url(f"/workspaces/{ws_id}"),
                headers=_daytona_headers(),
                method="DELETE",
            )
            urllib.request.urlopen(_req_del, timeout=10)
        except Exception:
            pass

    return result


def daytona_list_workspaces(**_) -> dict:
    """
    List Daytona workspaces. Requires DAYTONA_API_KEY.

    Returns:
        {success, workspaces: [{id, name, status, image}], count, error}
    """
    api_key = os.environ.get("DAYTONA_API_KEY", "").strip()
    if not api_key:
        return {"success": False, "error": "DAYTONA_API_KEY not set."}

    r = urllib.request.Request(
        _daytona_url("/workspaces"), headers=_daytona_headers(), method="GET"
    )
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            data = json.loads(resp.read())
            wss  = data if isinstance(data, list) else data.get("workspaces", [])
            return {
                "success":    True,
                "workspaces": [
                    {
                        "id":     w.get("id", ""),
                        "name":   w.get("name", ""),
                        "status": w.get("status", ""),
                        "image":  w.get("image", ""),
                    }
                    for w in wss
                ],
                "count": len(wss),
                "error": "",
            }
    except Exception as e:
        return {"success": False, "error": str(e)}
