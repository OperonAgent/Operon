"""
Operon Shell Execution Tool.

Executes bash/shell commands via subprocess with:
  • Non-blocking stdout/stderr capture
  • Configurable timeout
  • Working directory support
  • Sanitised output (no control character injection)
  • Command risk analysis (blocks CRITICAL, warns on HIGH)
"""

import os
import subprocess
import sys
import time
from typing import Optional

from core.command_risk import analyse_command, RiskLevel


_DEFAULT_TIMEOUT = 30   # seconds
_MAX_OUTPUT_BYTES = 64 * 1024   # 64 KB cap per run


def shell_exec(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT,
    env_extra: Optional[dict] = None,
    allow_high_risk: bool = False,
) -> dict:
    """
    Execute a shell command and capture its output.

    Returns:
        {
            "success":    bool,
            "stdout":     str,
            "stderr":     str,
            "returncode": int,
            "truncated":  bool,
            "risk_level": str,   # "SAFE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
        }
    """
    if not command or not command.strip():
        return {"success": False, "stdout": "", "stderr": "Empty command", "returncode": -1, "truncated": False, "risk_level": "SAFE"}

    # ── Command risk analysis ──────────────────────────────────────────────────
    risk = analyse_command(command, block_critical=True, warn_high=True)
    if risk.blocked:
        return {
            "success":    False,
            "stdout":     "",
            "stderr":     f"[BLOCKED] {risk.reason}",
            "returncode": -1,
            "truncated":  False,
            "risk_level": risk.level.name,
        }
    if risk.level >= RiskLevel.HIGH and not allow_high_risk:
        findings_text = "; ".join(f"{f.rule}: {f.description}" for f in risk.findings)
        return {
            "success":    False,
            "stdout":     "",
            "stderr":     (
                f"[HIGH RISK] Command flagged as {risk.level.name} — execution blocked.\n"
                f"Findings: {findings_text}\n"
                f"Set allow_high_risk=True to override."
            ),
            "returncode": -1,
            "truncated":  False,
            "risk_level": risk.level.name,
        }

    # Build environment
    env = os.environ.copy()
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})

    # Resolve working directory
    work_dir = cwd or os.getcwd()
    if not os.path.isdir(work_dir):
        return {
            "success": False,
            "stdout":  "",
            "stderr":  f"Working directory does not exist: {work_dir}",
            "returncode": -1,
            "truncated": False,
        }

    try:
        # Use Popen so output streams line-by-line to the terminal in real time
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=work_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        total_out = 0
        truncated = False
        timed_out = False

        import threading as _threading

        # Watchdog: kill the process if it runs past the deadline.
        # This is the primary timeout mechanism — it fires even for commands
        # that produce no output (e.g. `sleep 10`), unlike the per-line check.
        def _watchdog():
            if proc.poll() is None:
                proc.kill()

        watchdog = _threading.Timer(timeout, _watchdog)
        watchdog.daemon = True
        watchdog.start()

        # Collect stderr in background thread
        def _drain_stderr():
            for chunk in iter(lambda: proc.stderr.read(256), b""):
                stderr_chunks.append(chunk)

        stderr_thread = _threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        deadline = time.time() + timeout
        for line in iter(proc.stdout.readline, b""):
            # Stream to terminal so user sees output in real time
            sys.stdout.write(line.decode("utf-8", errors="replace"))
            sys.stdout.flush()
            total_out += len(line)
            if total_out < _MAX_OUTPUT_BYTES:
                stdout_chunks.append(line)
            else:
                truncated = True
            if time.time() > deadline:
                proc.kill()
                truncated = True
                break

        proc.wait(timeout=5)
        watchdog.cancel()
        stderr_thread.join(timeout=3)

        # If the watchdog fired (process was killed) → timed_out
        if proc.returncode in (-9, -15):
            timed_out = True

        stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace").rstrip()
        stderr_raw = b"".join(stderr_chunks)
        if len(stderr_raw) > _MAX_OUTPUT_BYTES:
            stderr_raw = stderr_raw[:_MAX_OUTPUT_BYTES]
        stderr = stderr_raw.decode("utf-8", errors="replace").rstrip()

        if timed_out or truncated:
            stderr = (stderr + f"\n[timed out after {timeout}s]").lstrip()
            return {
                "success":    False,
                "stdout":     stdout,
                "stderr":     stderr,
                "returncode": proc.returncode,
                "truncated":  True,
                "risk_level": risk.level.name,
            }

        return {
            "success":    proc.returncode == 0,
            "stdout":     stdout,
            "stderr":     stderr,
            "returncode": proc.returncode,
            "truncated":  truncated,
            "risk_level": risk.level.name,
        }

    except subprocess.TimeoutExpired:
        return {
            "success":    False,
            "stdout":     "",
            "stderr":     f"Command timed out after {timeout}s",
            "returncode": -1,
            "truncated":  False,
            "risk_level": risk.level.name,
        }
    except Exception as e:
        return {
            "success":    False,
            "stdout":     "",
            "stderr":     str(e),
            "returncode": -1,
            "truncated":  False,
            "risk_level": risk.level.name,
        }
