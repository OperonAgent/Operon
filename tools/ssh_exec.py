"""
Operon SSH Remote Execution Tool.

Runs commands on remote machines over SSH.
Strategy (tried in order):
  1. paramiko  — pure-Python SSH, no system ssh binary required
  2. subprocess ssh — falls back to the system `ssh` binary

Supports:
  • Password and key-based auth
  • SCP file upload / download
  • Port tunnelling awareness
  • Persistent connections (one connection per host reused within a session)
"""

import os
import subprocess
import threading
import tempfile
from pathlib import Path
from typing import Optional

try:
    import paramiko
    _PARAMIKO = True
except ImportError:
    _PARAMIKO = False

# ── Connection pool (one per session, keyed by host:port:user) ────────────────

_pool: dict[str, "paramiko.SSHClient"] = {}
_pool_lock = threading.Lock()


def _pool_key(host: str, port: int, user: str) -> str:
    return f"{user}@{host}:{port}"


def _get_or_create_client(
    host: str,
    port: int,
    user: str,
    password: Optional[str],
    key_path: Optional[str],
) -> "paramiko.SSHClient":
    key = _pool_key(host, port, user)
    with _pool_lock:
        client = _pool.get(key)
        if client:
            try:
                client.get_transport().send_ignore()  # keepalive ping
                return client
            except Exception:
                _pool.pop(key, None)

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs: dict = {"hostname": host, "port": port, "username": user, "timeout": 15}
        if password:
            connect_kwargs["password"] = password
        if key_path:
            key_file = os.path.expanduser(key_path)
            if os.path.exists(key_file):
                connect_kwargs["key_filename"] = key_file
        if not password and not key_path:
            # Let paramiko try the SSH agent + default keys
            connect_kwargs["look_for_keys"] = True
            connect_kwargs["allow_agent"]   = True

        client.connect(**connect_kwargs)
        _pool[key] = client
        return client


def _close_client(host: str, port: int, user: str):
    key = _pool_key(host, port, user)
    with _pool_lock:
        client = _pool.pop(key, None)
    if client:
        try:
            client.close()
        except Exception:
            pass


# ── Paramiko backend ──────────────────────────────────────────────────────────

def _exec_paramiko(
    host: str,
    command: str,
    port: int,
    user: str,
    password: Optional[str],
    key_path: Optional[str],
    timeout: int,
    cwd: Optional[str],
) -> dict:
    try:
        client = _get_or_create_client(host, port, user, password, key_path)
        full_cmd = f"cd {cwd} && {command}" if cwd else command
        _, stdout, stderr = client.exec_command(full_cmd, timeout=timeout, get_pty=False)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return {
            "success":   exit_code == 0,
            "output":    out,
            "stderr":    err,
            "exit_code": exit_code,
            "host":      f"{user}@{host}:{port}",
            "error":     err if exit_code != 0 else "",
        }
    except Exception as e:
        # Try to clean up potentially broken connection
        _close_client(host, port, user)
        return {
            "success": False, "output": "", "stderr": "",
            "exit_code": -1, "host": f"{user}@{host}:{port}",
            "error": f"paramiko: {type(e).__name__}: {e}",
        }


# ── Subprocess ssh backend ────────────────────────────────────────────────────

def _exec_subprocess(
    host: str,
    command: str,
    port: int,
    user: str,
    key_path: Optional[str],
    timeout: int,
    cwd: Optional[str],
) -> dict:
    ssh_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-p", str(port),
    ]
    if key_path:
        ssh_cmd += ["-i", os.path.expanduser(key_path)]
    ssh_cmd.append(f"{user}@{host}")

    full_cmd = f"cd {cwd} && {command}" if cwd else command
    ssh_cmd.append(full_cmd)

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success":   result.returncode == 0,
            "output":    result.stdout,
            "stderr":    result.stderr,
            "exit_code": result.returncode,
            "host":      f"{user}@{host}:{port}",
            "error":     result.stderr if result.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False, "output": "", "stderr": "",
            "exit_code": -1, "host": f"{user}@{host}:{port}",
            "error": f"Command timed out after {timeout}s",
        }
    except FileNotFoundError:
        return {
            "success": False, "output": "", "stderr": "",
            "exit_code": -1, "host": f"{user}@{host}:{port}",
            "error": "ssh binary not found. Install OpenSSH or: pip install paramiko",
        }


# ── Public API ────────────────────────────────────────────────────────────────

def ssh_exec(
    host:     str,
    command:  str,
    port:     int        = 22,
    user:     str        = None,
    password: str        = None,
    key_path: str        = None,
    timeout:  int        = 30,
    cwd:      str        = None,
) -> dict:
    """
    Execute a command on a remote host over SSH.

    Returns:
        {success, output, stderr, exit_code, host, error}
    """
    if not host:
        return {"success": False, "output": "", "stderr": "",
                "exit_code": -1, "host": "", "error": "host is required"}
    if not command:
        return {"success": False, "output": "", "stderr": "",
                "exit_code": -1, "host": host, "error": "command is required"}

    # Default to current system user
    if not user:
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "root"

    # Prefer paramiko (richer API, reusable connections)
    if _PARAMIKO:
        return _exec_paramiko(host, command, port, user, password, key_path, timeout, cwd)
    else:
        return _exec_subprocess(host, command, port, user, key_path, timeout, cwd)


def ssh_upload(
    host:      str,
    local_path: str,
    remote_path: str,
    port:      int  = 22,
    user:      str  = None,
    password:  str  = None,
    key_path:  str  = None,
) -> dict:
    """
    Upload a file to a remote host via SCP/SFTP.

    Returns:
        {success, local_path, remote_path, host, error}
    """
    if not user:
        user = os.environ.get("USER") or "root"
    if not os.path.exists(local_path):
        return {"success": False, "error": f"Local file not found: {local_path}",
                "local_path": local_path, "remote_path": remote_path, "host": host}

    if _PARAMIKO:
        try:
            client = _get_or_create_client(host, port, user, password, key_path)
            sftp   = client.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
            return {
                "success":     True,
                "local_path":  local_path,
                "remote_path": remote_path,
                "host":        f"{user}@{host}:{port}",
                "error":       "",
            }
        except Exception as e:
            _close_client(host, port, user)
            return {"success": False, "error": str(e),
                    "local_path": local_path, "remote_path": remote_path,
                    "host": f"{user}@{host}:{port}"}
    else:
        # subprocess scp
        cmd = ["scp", "-P", str(port), "-o", "StrictHostKeyChecking=no"]
        if key_path:
            cmd += ["-i", os.path.expanduser(key_path)]
        cmd += [local_path, f"{user}@{host}:{remote_path}"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return {
                "success":     result.returncode == 0,
                "local_path":  local_path,
                "remote_path": remote_path,
                "host":        f"{user}@{host}:{port}",
                "error":       result.stderr if result.returncode != 0 else "",
            }
        except Exception as e:
            return {"success": False, "error": str(e),
                    "local_path": local_path, "remote_path": remote_path, "host": host}


def ssh_download(
    host:       str,
    remote_path: str,
    local_path:  str,
    port:        int = 22,
    user:        str = None,
    password:    str = None,
    key_path:    str = None,
) -> dict:
    """
    Download a file from a remote host via SFTP/SCP.

    Returns:
        {success, local_path, remote_path, host, error}
    """
    if not user:
        user = os.environ.get("USER") or "root"

    if _PARAMIKO:
        try:
            client = _get_or_create_client(host, port, user, password, key_path)
            sftp   = client.open_sftp()
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_path, local_path)
            sftp.close()
            return {
                "success":     True,
                "local_path":  local_path,
                "remote_path": remote_path,
                "host":        f"{user}@{host}:{port}",
                "error":       "",
            }
        except Exception as e:
            _close_client(host, port, user)
            return {"success": False, "error": str(e),
                    "local_path": local_path, "remote_path": remote_path,
                    "host": f"{user}@{host}:{port}"}
    else:
        cmd = ["scp", "-P", str(port), "-o", "StrictHostKeyChecking=no"]
        if key_path:
            cmd += ["-i", os.path.expanduser(key_path)]
        cmd += [f"{user}@{host}:{remote_path}", local_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return {
                "success":     result.returncode == 0,
                "local_path":  local_path,
                "remote_path": remote_path,
                "host":        f"{user}@{host}:{port}",
                "error":       result.stderr if result.returncode != 0 else "",
            }
        except Exception as e:
            return {"success": False, "error": str(e),
                    "local_path": local_path, "remote_path": remote_path, "host": host}
