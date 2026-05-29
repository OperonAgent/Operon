"""
Operon Python Code Execution Sandbox.

Runs arbitrary Python code in a subprocess with a timeout.
Captures stdout, stderr, and the return code.
Uses a temp file so multi-line code works cleanly.
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path


def python_exec(code: str, timeout: int = 30, cwd: str = None) -> dict:
    """
    Execute Python code and return captured output.

    Returns:
        {success, stdout, stderr, returncode, error}
    """
    if not code or not code.strip():
        return {"success": False, "stdout": "", "stderr": "", "returncode": -1,
                "error": "No code provided."}

    work_dir = cwd or str(Path.home())
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
        )

        return {
            "success":    result.returncode == 0,
            "stdout":     result.stdout,
            "stderr":     result.stderr,
            "returncode": result.returncode,
            "error":      result.stderr if result.returncode != 0 else "",
        }

    except subprocess.TimeoutExpired:
        return {
            "success":    False,
            "stdout":     "",
            "stderr":     "",
            "returncode": -1,
            "error":      f"Code execution timed out after {timeout}s.",
        }
    except Exception as e:
        return {
            "success":    False,
            "stdout":     "",
            "stderr":     "",
            "returncode": -1,
            "error":      f"{type(e).__name__}: {e}",
        }
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
