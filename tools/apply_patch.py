"""
Operon Apply Patch Tool.

Adapted from OpenClaw src/agents/apply-patch.ts.

Applies structured patches (unified diff, search-replace, or JSON-patch format)
to files.  Unlike raw shell `patch`, this tool:
  - Validates the patch before applying
  - Supports fuzzy matching for context lines
  - Returns precise error messages on failure
  - Creates backups before applying
  - Supports dry-run mode

Patch formats supported:
  1. unified_diff  — standard `--- a/file +++ b/file @@ ...` format
  2. search_replace — list of {"search": str, "replace": str} operations
  3. json_patch     — RFC 6902 JSON Patch operations
"""

from __future__ import annotations

import copy
import difflib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union


# ── Unified diff application ───────────────────────────────────────────────────

def _parse_unified_diff(diff_text: str) -> list[dict]:
    """
    Parse a unified diff into a list of hunks.
    Returns [{"file": str, "hunks": [{"old_start", "old_len", "new_start", "new_len", "lines"}]}]
    """
    files: list[dict] = []
    current_file: Optional[dict] = None
    current_hunk: Optional[dict] = None

    for line in diff_text.splitlines():
        if line.startswith("--- "):
            # Ignore header "---" lines (they indicate old file name)
            pass
        elif line.startswith("+++ "):
            fname = re.sub(r"^b/", "", line[4:].split("\t")[0].strip())
            current_file = {"file": fname, "hunks": []}
            files.append(current_file)
        elif line.startswith("@@ ") and current_file is not None:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                current_hunk = {
                    "old_start": int(m.group(1)),
                    "old_len":   int(m.group(2)) if m.group(2) is not None else 1,
                    "new_start": int(m.group(3)),
                    "new_len":   int(m.group(4)) if m.group(4) is not None else 1,
                    "lines":     [],
                }
                current_file["hunks"].append(current_hunk)
        elif current_hunk is not None:
            if line.startswith(("-", "+", " ")):
                current_hunk["lines"].append(line)

    return files


def _apply_hunk(lines: list[str], hunk: dict, fuzzy: bool = True) -> tuple[list[str], bool]:
    """
    Apply a single diff hunk to lines.
    Returns (new_lines, success).
    """
    old_start  = hunk["old_start"] - 1   # 0-indexed
    old_lines  = [l[1:] for l in hunk["lines"] if l.startswith((" ", "-"))]
    new_lines  = [l[1:] for l in hunk["lines"] if l.startswith((" ", "+"))]

    # Find old_start in target (fuzzy search if needed)
    search_from = max(0, old_start - 3) if fuzzy else old_start
    search_to   = min(len(lines), old_start + len(old_lines) + 3) if fuzzy else old_start + len(old_lines)

    best_start = old_start
    if fuzzy and old_lines:
        # Use difflib to find the best matching position
        matcher = difflib.SequenceMatcher(None,
                                          "\n".join(old_lines),
                                          "\n".join(lines[search_from:search_to]))
        best_ratio = 0.0
        for i in range(search_from, min(search_to, len(lines))):
            chunk = lines[i:i + len(old_lines)]
            ratio = difflib.SequenceMatcher(None, old_lines, chunk).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
        if best_ratio < 0.5:
            return lines, False   # no good match found

    end = best_start + len(old_lines)
    result = lines[:best_start] + [l + "\n" for l in new_lines if not l.endswith("\n")] + lines[end:]
    # Re-add newlines correctly
    new_with_newlines = []
    for i, nl in enumerate(new_lines):
        if nl.endswith("\n"):
            new_with_newlines.append(nl)
        else:
            new_with_newlines.append(nl + "\n")
    result = lines[:best_start] + new_with_newlines + lines[end:]
    return result, True


def apply_unified_diff(
    diff_text:    str,
    workspace:    str  = ".",
    dry_run:      bool = False,
    fuzzy:        bool = True,
    backup:       bool = True,
) -> dict:
    """
    Apply a unified diff to files in `workspace`.

    Returns::

        {
            "success": bool,
            "applied": [{"file": str, "hunks_applied": int, "hunks_failed": int}],
            "errors":  [str],
        }
    """
    ws     = Path(workspace).resolve()
    parsed = _parse_unified_diff(diff_text)
    if not parsed:
        return {"success": False, "applied": [], "errors": ["No valid hunks found in diff"]}

    applied: list[dict] = []
    errors:  list[str]  = []

    for file_patch in parsed:
        rel_path = file_patch["file"]
        full_path = ws / rel_path

        if not full_path.exists():
            # New file — build from additions
            if not dry_run:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                new_content = "\n".join(
                    l[1:] for h in file_patch["hunks"] for l in h["lines"]
                    if l.startswith("+")
                )
                full_path.write_text(new_content, encoding="utf-8")
            applied.append({"file": rel_path, "hunks_applied": len(file_patch["hunks"]), "hunks_failed": 0})
            continue

        content = full_path.read_text(encoding="utf-8", errors="replace")
        lines   = content.splitlines(keepends=True)

        ok_count  = 0
        fail_count = 0

        for hunk in file_patch["hunks"]:
            lines, success = _apply_hunk(lines, hunk, fuzzy=fuzzy)
            if success:
                ok_count += 1
            else:
                fail_count += 1
                errors.append(f"{rel_path}: hunk @@ -{hunk['old_start']} failed to apply")

        if not dry_run and ok_count > 0:
            if backup:
                shutil.copy2(str(full_path), str(full_path) + ".bak")
            full_path.write_text("".join(lines), encoding="utf-8")

        applied.append({
            "file": rel_path,
            "hunks_applied": ok_count,
            "hunks_failed": fail_count,
        })

    all_ok = all(r["hunks_failed"] == 0 for r in applied)
    return {"success": all_ok, "applied": applied, "errors": errors}


# ── Search-replace patch ───────────────────────────────────────────────────────

def apply_search_replace(
    file_path:   str,
    operations:  list[dict],
    dry_run:     bool = False,
    backup:      bool = True,
) -> dict:
    """
    Apply a list of search-replace operations to a file.

    Each operation: {"search": str, "replace": str, "count": int (optional)}

    Returns::

        {"success": bool, "changes": int, "errors": [str]}
    """
    path = Path(file_path)
    if not path.exists():
        return {"success": False, "changes": 0, "errors": [f"File not found: {file_path}"]}

    content = path.read_text(encoding="utf-8", errors="replace")
    original = content
    errors: list[str] = []
    total_changes = 0

    for op in operations:
        search  = op.get("search", "")
        replace = op.get("replace", "")
        count   = op.get("count", 0)   # 0 = replace all

        if not search:
            errors.append("Operation missing 'search' field")
            continue

        if search not in content:
            errors.append(f"Search string not found: {search[:60]!r}")
            continue

        if count:
            new_content = content.replace(search, replace, count)
        else:
            new_content = content.replace(search, replace)

        changes = content.count(search)
        content = new_content
        total_changes += changes

    if not dry_run and content != original:
        if backup:
            shutil.copy2(str(path), str(path) + ".bak")
        path.write_text(content, encoding="utf-8")

    return {
        "success":  len(errors) == 0,
        "changes":  total_changes,
        "errors":   errors,
    }


# ── JSON patch (RFC 6902) ──────────────────────────────────────────────────────

def apply_json_patch(
    file_path:   str,
    operations:  list[dict],
    dry_run:     bool = False,
    backup:      bool = True,
) -> dict:
    """
    Apply RFC 6902 JSON Patch operations to a JSON file.

    Supported operations: add, remove, replace, copy, move, test

    Returns::

        {"success": bool, "result": dict | None, "errors": [str]}
    """
    path = Path(file_path)
    if not path.exists():
        return {"success": False, "result": None, "errors": [f"File not found: {file_path}"]}

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"success": False, "result": None, "errors": [f"Invalid JSON: {e}"]}

    result = copy.deepcopy(doc)
    errors: list[str] = []

    def _get_pointer(obj, path_parts: list[str]):
        """Follow a JSON Pointer path."""
        current = obj
        for part in path_parts:
            if isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError) as e:
                    raise KeyError(f"Index error at /{'/'.join(path_parts)}: {e}")
            elif isinstance(current, dict):
                if part not in current:
                    raise KeyError(f"Key not found: {part!r}")
                current = current[part]
            else:
                raise KeyError(f"Cannot index into {type(current)}")
        return current

    def _set_pointer(obj, path_parts: list[str], value):
        """Set a value at a JSON Pointer path."""
        if not path_parts:
            return value
        parent = _get_pointer(obj, path_parts[:-1]) if len(path_parts) > 1 else obj
        key = path_parts[-1]
        if isinstance(parent, list):
            if key == "-":
                parent.append(value)
            else:
                parent.insert(int(key), value)
        elif isinstance(parent, dict):
            parent[key] = value
        return obj

    def _del_pointer(obj, path_parts: list[str]):
        """Delete at a JSON Pointer path."""
        parent = _get_pointer(obj, path_parts[:-1]) if len(path_parts) > 1 else obj
        key = path_parts[-1]
        if isinstance(parent, list):
            del parent[int(key)]
        elif isinstance(parent, dict):
            del parent[key]

    for op in operations:
        try:
            op_type = op.get("op", "")
            path_str = op.get("path", "")
            path_parts = [p for p in path_str.split("/") if p]

            if op_type == "add":
                _set_pointer(result, path_parts, op["value"])
            elif op_type == "remove":
                _del_pointer(result, path_parts)
            elif op_type == "replace":
                _del_pointer(result, path_parts)
                _set_pointer(result, path_parts, op["value"])
            elif op_type == "copy":
                from_parts = [p for p in op["from"].split("/") if p]
                val = copy.deepcopy(_get_pointer(result, from_parts))
                _set_pointer(result, path_parts, val)
            elif op_type == "move":
                from_parts = [p for p in op["from"].split("/") if p]
                val = copy.deepcopy(_get_pointer(result, from_parts))
                _del_pointer(result, from_parts)
                _set_pointer(result, path_parts, val)
            elif op_type == "test":
                actual = _get_pointer(result, path_parts)
                if actual != op["value"]:
                    errors.append(f"test failed at {path_str}: expected {op['value']!r}, got {actual!r}")
            else:
                errors.append(f"Unknown op: {op_type!r}")
        except Exception as e:
            errors.append(f"Operation {op} failed: {e}")

    if not errors and not dry_run:
        if backup:
            shutil.copy2(str(path), str(path) + ".bak")
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "success": len(errors) == 0,
        "result":  result if not errors else None,
        "errors":  errors,
    }


# ── Unified entry point ────────────────────────────────────────────────────────

def apply_patch(
    patch:        Union[str, list, dict],
    workspace:    str  = ".",
    file_path:    str  = "",
    format:       str  = "auto",
    dry_run:      bool = False,
    backup:       bool = True,
) -> dict:
    """
    Apply a patch to files.

    Parameters
    ----------
    patch      : the patch content (str for unified/search-replace, list for JSON-patch)
    workspace  : root directory for unified diffs
    file_path  : target file for search-replace and JSON-patch
    format     : "auto" | "unified_diff" | "search_replace" | "json_patch"
    dry_run    : if True, validate but don't write changes
    backup     : if True, create .bak files before modifying

    Returns::

        {"success": bool, ...format-specific fields..., "errors": [str]}
    """
    # Auto-detect format
    if format == "auto":
        if isinstance(patch, list):
            # List of operations → JSON patch or search-replace
            first = patch[0] if patch else {}
            if isinstance(first, dict) and "op" in first:
                format = "json_patch"
            else:
                format = "search_replace"
        elif isinstance(patch, str) and patch.strip().startswith("---"):
            format = "unified_diff"
        elif isinstance(patch, dict) and "search" in patch:
            format = "search_replace"
            patch  = [patch]
        else:
            format = "unified_diff"

    if format == "unified_diff":
        return apply_unified_diff(str(patch), workspace=workspace, dry_run=dry_run, backup=backup)
    elif format == "search_replace":
        ops = patch if isinstance(patch, list) else [patch]
        return apply_search_replace(file_path, ops, dry_run=dry_run, backup=backup)
    elif format == "json_patch":
        ops = patch if isinstance(patch, list) else [patch]
        return apply_json_patch(file_path, ops, dry_run=dry_run, backup=backup)
    else:
        return {"success": False, "errors": [f"Unknown format: {format!r}"]}
