"""
Operon File Search Tool.

Recursive grep-like search over directory trees.
Returns file paths and matching line numbers / snippets.
"""

import os
import re
from pathlib import Path


def file_search(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    case_sensitive: bool = False,
    file_pattern: str = "*",
    max_results: int = 50,
    context_lines: int = 0,
) -> dict:
    """
    Search file contents for a pattern.

    Args:
        pattern        — Regex or plain-text string to search for (required)
        path           — Directory or file to search in, default '.' (optional)
        recursive      — Recurse into subdirectories, default True (optional)
        case_sensitive — Case-sensitive match, default False (optional)
        file_pattern   — Glob pattern to filter filenames e.g. '*.py' (optional)
        max_results    — Max matching lines to return, default 50 (optional)
        context_lines  — Extra lines of context before/after match, default 0 (optional)

    Returns:
        {success, matches: [{file, line_no, line, context_before, context_after}], total, error}
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return {"success": False, "matches": [], "total": 0,
                "error": f"Invalid regex: {e}"}

    target = Path(path).expanduser().resolve()
    if not target.exists():
        return {"success": False, "matches": [], "total": 0,
                "error": f"Path not found: {path}"}

    # Collect candidate files
    if target.is_file():
        candidates = [target]
    elif recursive:
        candidates = list(target.rglob(file_pattern))
    else:
        candidates = list(target.glob(file_pattern))

    # Filter to readable text files
    candidates = [f for f in candidates if f.is_file()]

    matches = []
    total = 0

    for filepath in candidates:
        try:
            lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines):
            if regex.search(line):
                total += 1
                if len(matches) < max_results:
                    ctx_before = lines[max(0, i - context_lines):i] if context_lines else []
                    ctx_after  = lines[i + 1:i + 1 + context_lines] if context_lines else []
                    matches.append({
                        "file":           str(filepath),
                        "line_no":        i + 1,
                        "line":           line.rstrip(),
                        "context_before": ctx_before,
                        "context_after":  ctx_after,
                    })

    return {
        "success": True,
        "matches": matches,
        "total":   total,
        "error":   "",
    }
