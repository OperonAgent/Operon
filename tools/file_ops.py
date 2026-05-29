"""
Operon File Operations Tool.

Full CRUD: create, read, write, update, append, patch, delete, list.
All operations return a standard {success, output, error} dict.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Any

from core.security_checks import validate_workspace_path, check_path_traversal


def _result(success: bool, output: Any = None, error: str = "") -> dict:
    return {"success": success, "output": output, "error": error}


def _check_path(path: str) -> tuple[bool, str]:
    """Gate every file operation through path traversal + sensitive-dir checks."""
    if check_path_traversal(path):
        return False, f"Path traversal detected: {path}"
    ok, reason = validate_workspace_path(path)
    if not ok:
        return False, reason
    return True, ""


def file_read(path: str, encoding: str = "utf-8") -> dict:
    """Read and return the full contents of a file."""
    ok, err = _check_path(path)
    if not ok:
        return _result(False, error=err)
    try:
        with open(path, "r", encoding=encoding) as f:
            content = f.read()
        return _result(True, content)
    except FileNotFoundError:
        return _result(False, error=f"File not found: {path}")
    except Exception as e:
        return _result(False, error=str(e))


def file_write(path: str, content: str, encoding: str = "utf-8") -> dict:
    """Create or overwrite a file with the given content."""
    ok, err = _check_path(path)
    if not ok:
        return _result(False, error=err)
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
        size = os.path.getsize(path)
        return _result(True, f"Written {size} bytes to {path}")
    except Exception as e:
        return _result(False, error=str(e))


def file_append(path: str, content: str, encoding: str = "utf-8") -> dict:
    """Append content to a file (creates the file if it does not exist)."""
    ok, err = _check_path(path)
    if not ok:
        return _result(False, error=err)
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding=encoding) as f:
            f.write(content)
        return _result(True, f"Appended to {path}")
    except Exception as e:
        return _result(False, error=str(e))


def file_patch(path: str, old_text: str, new_text: str, encoding: str = "utf-8") -> dict:
    """Replace the first occurrence of old_text with new_text inside the file."""
    ok, err = _check_path(path)
    if not ok:
        return _result(False, error=err)
    try:
        with open(path, "r", encoding=encoding) as f:
            original = f.read()
        if old_text not in original:
            return _result(False, error=f"Target text not found in {path}")
        patched = original.replace(old_text, new_text, 1)
        with open(path, "w", encoding=encoding) as f:
            f.write(patched)
        return _result(True, f"Patched {path} successfully")
    except FileNotFoundError:
        return _result(False, error=f"File not found: {path}")
    except Exception as e:
        return _result(False, error=str(e))


def file_delete(path: str) -> dict:
    """Delete a file."""
    ok, err = _check_path(path)
    if not ok:
        return _result(False, error=err)
    try:
        if not os.path.exists(path):
            return _result(False, error=f"Path does not exist: {path}")
        if os.path.isdir(path):
            shutil.rmtree(path)
            return _result(True, f"Directory deleted: {path}")
        os.remove(path)
        return _result(True, f"File deleted: {path}")
    except Exception as e:
        return _result(False, error=str(e))


def dir_list(path: str = ".", max_depth: int = 3) -> dict:
    """
    Recursively list the directory tree up to max_depth.
    Returns a formatted string tree and a flat list of paths.
    """
    try:
        base = Path(path).resolve()
        if not base.exists():
            return _result(False, error=f"Path does not exist: {path}")

        lines = []
        flat  = []

        def _walk(p: Path, prefix: str, depth: int):
            if depth > max_depth:
                return
            try:
                entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            except PermissionError:
                lines.append(f"{prefix}[permission denied]")
                return

            for i, entry in enumerate(entries):
                connector = "└── " if i == len(entries) - 1 else "├── "
                lines.append(f"{prefix}{connector}{entry.name}")
                flat.append(str(entry))
                if entry.is_dir():
                    extension = "    " if i == len(entries) - 1 else "│   "
                    _walk(entry, prefix + extension, depth + 1)

        lines.append(str(base))
        _walk(base, "", 1)
        return _result(True, {"tree": "\n".join(lines), "paths": flat})
    except Exception as e:
        return _result(False, error=str(e))


def file_exists(path: str) -> dict:
    """Check whether a file or directory exists."""
    exists = os.path.exists(path)
    kind   = "directory" if os.path.isdir(path) else ("file" if os.path.isfile(path) else "unknown")
    return _result(True, {"exists": exists, "kind": kind if exists else None, "path": path})


def file_info(path: str) -> dict:
    """Return metadata (size, modified time, permissions) for a file."""
    try:
        stat = os.stat(path)
        return _result(True, {
            "path":     path,
            "size":     stat.st_size,
            "modified": stat.st_mtime,
            "is_dir":   os.path.isdir(path),
            "mode":     oct(stat.st_mode),
        })
    except FileNotFoundError:
        return _result(False, error=f"File not found: {path}")
    except Exception as e:
        return _result(False, error=str(e))
