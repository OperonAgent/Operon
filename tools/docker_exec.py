"""
Operon Docker Execution Backend.

Runs code or shell commands inside isolated Docker containers for safe,
sandboxed execution without affecting the host filesystem or processes.

Requirements
------------
  Docker Desktop or Docker Engine must be running.
  pip install docker     # optional — falls back to `docker` CLI subprocess

Usage
-----
  from tools.docker_exec import docker_run, docker_run_code, docker_list_containers

All functions accept **_ for registry compatibility.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_IMAGE   = "python:3.12-slim"
_DEFAULT_TIMEOUT = 30   # seconds
_MEMORY_LIMIT    = "256m"
_CPU_LIMIT       = "0.5"


# ── Docker SDK helper ─────────────────────────────────────────────────────────

def _docker_sdk_available() -> bool:
    try:
        import docker as _d
        _d.from_env().ping()
        return True
    except Exception:
        return False


def _run_via_sdk(
    image:   str,
    command: str,
    env:     dict,
    timeout: int,
    workdir: str,
    volumes: dict,
) -> dict:
    import docker as _d
    client = _d.from_env()
    t0 = time.monotonic()
    try:
        logs = client.containers.run(
            image=image,
            command=["sh", "-c", command],
            environment=env,
            working_dir=workdir,
            volumes=volumes,
            mem_limit=_MEMORY_LIMIT,
            nano_cpus=int(float(_CPU_LIMIT) * 1e9),
            network_mode="bridge",
            remove=True,
            stdout=True,
            stderr=True,
            timeout=timeout,
        )
        output = logs.decode("utf-8", errors="replace") if isinstance(logs, bytes) else str(logs)
        return {
            "success":    True,
            "stdout":     output,
            "stderr":     "",
            "returncode": 0,
            "elapsed_ms": round((time.monotonic() - t0) * 1000),
            "image":      image,
            "error":      "",
        }
    except _d.errors.ContainerError as e:
        return {
            "success":    False,
            "stdout":     e.stderr.decode("utf-8", errors="replace") if e.stderr else "",
            "stderr":     str(e),
            "returncode": e.exit_status,
            "elapsed_ms": round((time.monotonic() - t0) * 1000),
            "image":      image,
            "error":      str(e),
        }
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e),
                "returncode": -1, "elapsed_ms": 0, "image": image, "error": str(e)}


def _run_via_cli(
    image:   str,
    command: str,
    env:     dict,
    timeout: int,
    workdir: str,
    volumes: dict,
) -> dict:
    """Fallback: use `docker run` subprocess."""
    cmd = [
        "docker", "run", "--rm",
        f"--memory={_MEMORY_LIMIT}",
        f"--cpus={_CPU_LIMIT}",
        "--network=bridge",
        "-w", workdir,
    ]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    for host_path, bind_info in volumes.items():
        mode = bind_info.get("mode", "rw")
        cmd += ["-v", f"{host_path}:{bind_info['bind']}:{mode}"]
    cmd += [image, "sh", "-c", command]

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        return {
            "success":    result.returncode == 0,
            "stdout":     result.stdout,
            "stderr":     result.stderr,
            "returncode": result.returncode,
            "elapsed_ms": round((time.monotonic() - t0) * 1000),
            "image":      image,
            "error":      result.stderr if result.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "Timed out",
                "returncode": -1, "elapsed_ms": timeout * 1000, "image": image,
                "error": f"Container timed out after {timeout}s"}
    except FileNotFoundError:
        return {"success": False, "stdout": "", "stderr": "docker not found",
                "returncode": -1, "elapsed_ms": 0, "image": image,
                "error": "Docker not installed or not in PATH. Install from https://docker.com"}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e),
                "returncode": -1, "elapsed_ms": 0, "image": image, "error": str(e)}


def _run_container(
    image:   str,
    command: str,
    env:     dict   = None,
    timeout: int    = _DEFAULT_TIMEOUT,
    workdir: str    = "/workspace",
    volumes: dict   = None,
) -> dict:
    env     = env     or {}
    volumes = volumes or {}
    if _docker_sdk_available():
        return _run_via_sdk(image, command, env, timeout, workdir, volumes)
    return _run_via_cli(image, command, env, timeout, workdir, volumes)


# ── Public tool functions ─────────────────────────────────────────────────────

def docker_run(
    command:  str  = "",
    image:    str  = _DEFAULT_IMAGE,
    env:      dict = None,
    timeout:  int  = _DEFAULT_TIMEOUT,
    workdir:  str  = "/workspace",
    **_,
) -> dict:
    """
    Run a shell command inside a Docker container and return its output.

    Args:
        command  — shell command to run (required)
        image    — Docker image to use (optional, default 'python:3.12-slim')
        env      — environment variables dict (optional)
        timeout  — seconds before killing the container (optional, default 30)
        workdir  — working directory inside container (optional, default '/workspace')

    Returns:
        {success, stdout, stderr, returncode, elapsed_ms, image, error}
    """
    if not command:
        return {"success": False, "error": "command is required."}

    return _run_container(
        image=image,
        command=command,
        env=env or {},
        timeout=timeout,
        workdir=workdir,
    )


def docker_run_code(
    code:     str  = "",
    language: str  = "python",
    image:    str  = "",
    timeout:  int  = _DEFAULT_TIMEOUT,
    **_,
) -> dict:
    """
    Execute code in a sandboxed Docker container.
    The code is written to a temp file and executed inside the container.

    Args:
        code      — source code to execute (required)
        language  — 'python' | 'node' | 'bash' | 'ruby' (optional, default 'python')
        image     — override Docker image (optional — auto-selected from language)
        timeout   — seconds before killing (optional, default 30)

    Returns:
        {success, stdout, stderr, returncode, elapsed_ms, image, language, error}
    """
    if not code:
        return {"success": False, "error": "code is required."}

    lang = language.lower().strip()
    _LANG_CONFIG: dict[str, dict] = {
        "python":     {"image": "python:3.12-slim", "ext": "py",  "cmd": "python3 /tmp/code.py"},
        "python3":    {"image": "python:3.12-slim", "ext": "py",  "cmd": "python3 /tmp/code.py"},
        "node":       {"image": "node:20-slim",     "ext": "js",  "cmd": "node /tmp/code.js"},
        "javascript": {"image": "node:20-slim",     "ext": "js",  "cmd": "node /tmp/code.js"},
        "bash":       {"image": "bash:5",            "ext": "sh",  "cmd": "bash /tmp/code.sh"},
        "shell":      {"image": "alpine:latest",     "ext": "sh",  "cmd": "sh /tmp/code.sh"},
        "ruby":       {"image": "ruby:3.3-slim",     "ext": "rb",  "cmd": "ruby /tmp/code.rb"},
    }
    if lang not in _LANG_CONFIG:
        return {
            "success":  False,
            "error":    f"Unsupported language '{lang}'. Supported: {list(_LANG_CONFIG)}",
        }

    cfg          = _LANG_CONFIG[lang]
    use_image    = image or cfg["image"]
    ext          = cfg["ext"]
    run_cmd      = cfg["cmd"]

    # Write code to a temp file that we'll mount into the container
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f".{ext}", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        volumes = {tmp_path: {"bind": f"/tmp/code.{ext}", "mode": "ro"}}
        result  = _run_container(
            image=use_image,
            command=run_cmd,
            timeout=timeout,
            volumes=volumes,
        )
        result["language"] = lang
        return result
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def docker_list_containers(running_only: bool = False, **_) -> dict:
    """
    List Docker containers on the host.

    Args:
        running_only — only show running containers (optional, default False)

    Returns:
        {success, containers: [{id, name, image, status, ports}], count, error}
    """
    try:
        flag = [] if running_only else ["-a"]
        result = subprocess.run(
            ["docker", "ps", "--format", "{{json .}}"] + flag,
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {"success": False, "error": result.stderr or "docker ps failed"}

        containers = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                containers.append({
                    "id":     row.get("ID", "")[:12],
                    "name":   row.get("Names", ""),
                    "image":  row.get("Image", ""),
                    "status": row.get("Status", ""),
                    "ports":  row.get("Ports", ""),
                })
            except Exception:
                pass

        return {
            "success":    True,
            "containers": containers,
            "count":      len(containers),
            "error":      "",
        }
    except FileNotFoundError:
        return {"success": False, "error": "Docker not installed or not in PATH."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def docker_pull(image: str = "", **_) -> dict:
    """
    Pull a Docker image from Docker Hub.

    Args:
        image — image name e.g. 'python:3.12-slim' (required)

    Returns:
        {success, image, elapsed_ms, error}
    """
    if not image:
        return {"success": False, "error": "image is required."}

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True, text=True, timeout=300,
        )
        return {
            "success":    result.returncode == 0,
            "image":      image,
            "elapsed_ms": round((time.monotonic() - t0) * 1000),
            "error":      result.stderr if result.returncode != 0 else "",
        }
    except FileNotFoundError:
        return {"success": False, "error": "Docker not installed."}
    except Exception as e:
        return {"success": False, "error": str(e)}
