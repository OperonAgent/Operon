"""
ui/banner.py — Operon startup banner

📏 MATHEMATICALLY CALIBRATED GRID — 78 Character Fixed Ceiling Width
Every printed line: 1 leading space + 78 chars = 79 visible cols total.

Layout (exact match to original code structure):
  _IW = 76  (inner width between ╭/│ and ╮/│)
  _LW = 27  (left column between │ chars — holds mascot)
  _RW = 48  (right column between │ chars — holds tool matrix)
  _LW + 1(inner│) + _RW = 76 = _IW ✓

  ASCII art OPERON title (52 cols, centered in 79) above the box.
"""

from __future__ import annotations

import os
import re
import unicodedata
import datetime
import subprocess
from typing import Dict, List, Optional

# ── Optional psutil for telemetry ────────────────────────────────────────────
try:
    import psutil as _psutil
    def _cpu_pct() -> str:
        return str(int(_psutil.cpu_percent(interval=0.1)))
    def _ram_gb() -> str:
        return str(round(_psutil.virtual_memory().total / (1024 ** 3)))
except ImportError:
    def _cpu_pct() -> str: return "N/A"
    def _ram_gb() -> str:  return "N/A"

# ── Theme imports ─────────────────────────────────────────────────────────────
from ui.theme import (
    RESET, BOLD,
    PURPLE_BASE, PURPLE_LIGHT, PURPLE_DIM,
    CYAN_GLOW, WHITE_BRIGHT, GRAY_TEXT,
)

# ── Extra color not in theme ──────────────────────────────────────────────────
_BLUE_SOFT  = "\033[38;5;111m"   # soft sky-blue for subtitle text
_CYAN_TITLE = "\033[1;38;5;51m"  # bright cyan for OPERON ASCII art

# ── ANSI strip helper ─────────────────────────────────────────────────────────
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _vlen(s: str) -> int:
    """Visual width: strip ANSI, count display columns (W/F=2, else 1)."""
    plain = _ANSI_RE.sub("", s)
    w = 0
    for ch in plain:
        eaw = unicodedata.east_asian_width(ch)
        w += 2 if eaw in ("W", "F") else 1
    return w


def _rpad(s: str, width: int) -> str:
    """Right-pad to `width` visual cols; truncate with … if overlong."""
    vis = _vlen(s)
    if vis > width:
        return _ANSI_RE.sub("", s)[: width - 1] + "…"
    return s + " " * (width - vis)


def _center_line(text: str, total_width: int = 79) -> str:
    """Center plain text within total_width, return with no trailing newline."""
    vw  = _vlen(text)
    pad = max(0, (total_width - vw) // 2)
    return " " * pad + text


# ── Fixed grid constants ──────────────────────────────────────────────────────
_IW = 76   # inner width  (╭─{76}─╮ and │{76}│)
_LW = 27   # left column  (mascot, between │ chars)
_RW = 48   # right column (tools,  between │ chars)  — _LW+1+_RW == _IW ✓

# ── OPERON ASCII-art title (52 visible cols each row, in _CYAN_TITLE) ─────────
# Centered: (79-52)//2 = 13 leading spaces → start at col 14 (aligns over box)
_ART_PAD = " " * 13
_OPERON_ART: List[str] = [
    f"{_ART_PAD}{_CYAN_TITLE} ██████╗ ██████╗ ███████╗██████╗  ██████╗ ███╗   ██╗{RESET}",
    f"{_ART_PAD}{_CYAN_TITLE}██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔═══██╗████╗  ██║{RESET}",
    f"{_ART_PAD}{_CYAN_TITLE}██║   ██║██████╔╝█████╗  ██████╔╝██║   ██║██╔██╗ ██║{RESET}",
    f"{_ART_PAD}{_CYAN_TITLE}██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██║   ██║██║╚██╗██║{RESET}",
    f"{_ART_PAD}{_CYAN_TITLE}╚██████╔╝██║     ███████╗██║  ██║╚██████╔╝██║ ╚████║{RESET}",
    f"{_ART_PAD}{_CYAN_TITLE} ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝{RESET}",
]
_SUBTITLE = (
    f"{_BLUE_SOFT}AI Terminal Cockpit  ·  Autonomous Agent Framework  ·  Phase 11{RESET}"
)


# ── Robot mascot (original block-character design, 27 cols each row) ─────────
_MASCOT: List[str] = [
    f"        {PURPLE_DIM}▄▄███████▄▄{RESET}        ",
    f"      {PURPLE_DIM}▄██▀▀  {PURPLE_LIGHT}▲{PURPLE_DIM}  ▀▀██▄{RESET}      ",
    f"      {PURPLE_DIM}██   {PURPLE_LIGHT}▮   ▮{PURPLE_DIM}   ██{RESET}      ",
    f"      {PURPLE_DIM}██▄         ▄██{RESET}      ",
    f"      {PURPLE_DIM}▀▀███████████▀▀{RESET}      ",
    f"     {PURPLE_DIM}▄▄██▀▀  {PURPLE_LIGHT}║{PURPLE_DIM}  ▀▀██▄▄{RESET}     ",
    f"    {PURPLE_DIM}███▀  {PURPLE_LIGHT}┃  ║  ┃{PURPLE_DIM}  ▀███{RESET}    ",
    f"    {PURPLE_DIM}▀█████████████████▀{RESET}    ",
]


# ── Tool-set display map ──────────────────────────────────────────────────────
_TOOLSET_CAPS: Dict[str, tuple] = {
    "shell":       ("shell/python_exec",       "Shell & Python sandbox"),
    "code":        ("python/shell_exec",        "Shell & Python sandbox"),
    "filesystem":  ("file_read/write",           "Full filesystem access"),
    "web":         ("duckduckgo_search …",      "Search + web access"),
    "email":       ("email_draft",              "Email composition"),
    "browser":     ("browser_navigate +10",     "Headless browser auto."),
    "computer":    ("computer_use",             "Desktop automation"),
    "delegation":  ("delegate_task",            "Multi-agent delegation"),
    "vision":      ("vision_analyze",           "Image understanding"),
    "messaging":   ("telegram/mcp/http",        "Messaging + MCP + APIs"),
    "database":    ("db_query/list_tables",     "Database operations"),
    "voice":       ("voice_record",             "Voice & transcription"),
    "ssh":         ("ssh_exec/upload/dl",       "Remote SSH execution"),
    "image":       ("image_generate/tts",       "DALL-E 3 + speech"),
}


def _build_left_rows(
    model_name: str,
    cwd: Optional[str],
    session_id: Optional[str],
    tool_count: int,
    skill_count: int,
) -> List[str]:
    """
    Left column rows (_LW=27 visible cols each).
    Mascot (8 rows) + blank + model + cwd + session + counts.
    """
    rows: List[str] = list(_MASCOT)  # 8 rows

    rows.append("")  # blank separator

    # model name  — cyan, truncated to _LW-1 (1 leading space)
    m_trunc = model_name[:_LW - 1]
    rows.append(f" {CYAN_GLOW}{m_trunc}{RESET}")

    # cwd
    if cwd:
        cwd_short = cwd.replace(os.path.expanduser("~"), "~")
        rows.append(f" {GRAY_TEXT}{cwd_short}{RESET}")

    # session id
    if session_id:
        sid = session_id[:_LW - 11]  # "Session: " = 9 chars
        rows.append(f" {GRAY_TEXT}Session: {PURPLE_LIGHT}{sid}{RESET}")

    # tool / skill count  (only if non-zero)
    if tool_count or skill_count:
        rows.append(
            f" {GRAY_TEXT}{tool_count} tools · {skill_count} skills{RESET}"
        )

    return rows


def _build_right_rows(toolsets: Dict[str, List[str]]) -> List[str]:
    """
    Build right-column rows, each exactly _RW=48 visible cols.
    Format: "  {name:<20} {desc:<25}"  =  2+20+1+25 = 48 ✓
    Blank rows are 48 spaces.
    """
    rows: List[str] = [""]   # first row blank (aligns with mascot top)

    shown = 0
    for ts_name, ts_tools in toolsets.items():
        if shown >= 12:
            remaining = len(toolsets) - shown
            entry = f"  {'(and ' + str(remaining) + ' more toolsets…)':<46}"
            rows.append(entry)
            break

        if ts_name in _TOOLSET_CAPS:
            label, desc = _TOOLSET_CAPS[ts_name]
        else:
            top2  = ts_tools[:2]
            label = "/".join(top2) + (f" +{len(ts_tools)-2}" if len(ts_tools) > 2 else "")
            desc  = f"{ts_name.title()} tools"

        # Exactly 48 visible chars: 2sp + name(20) + 1sp + desc(25)
        label = (label[:19] + "…") if len(label) > 20 else label  # prevent overflow
        name_part = f"  {WHITE_BRIGHT}{BOLD}{label:<20}{RESET}"
        desc_part = f" {_BLUE_SOFT}{desc:<25}{RESET}"
        rows.append(name_part + desc_part)
        shown += 1

    if not toolsets:
        rows.append(f"  {GRAY_TEXT}{'No tools loaded':<46}{RESET}")
        rows.append(f"  {GRAY_TEXT}{'... type /help for commands':<46}{RESET}")

    rows.append("")  # trailing blank row
    return rows


def _get_git_short_sha() -> str:
    if os.environ.get("OPERON_NO_GIT"):
        return "Phase11"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except Exception:
        return "Phase11"


# ── Main render ───────────────────────────────────────────────────────────────

def render(
    model_name:  str  = "operon",
    tool_count:  int  = 0,
    skill_count: int  = 0,
    toolsets:    Optional[Dict[str, List[str]]] = None,
    skills:      Optional[List[dict]] = None,
    session_id:  Optional[str] = None,
    cwd:         Optional[str] = None,
    version:     str  = "3.1.0",
) -> str:
    toolsets = toolsets or {}

    lines: List[str] = []

    # ── OPERON ASCII-art title (above the box) ────────────────────────────────
    lines.append("")
    lines.extend(_OPERON_ART)
    lines.append(_center_line(_ANSI_RE.sub("", _SUBTITLE) and _SUBTITLE, 79))
    lines.append("")

    # ── Header (╭ ... ╮) — PURPLE_BASE active for every char including dashes ──
    sha      = _get_git_short_sha()
    ver_text = f" Operon v{version} ({datetime.date.today().strftime('%Y.%-m.%-d')}) · build {sha} "
    ld = (_IW - len(ver_text)) // 2
    rd = _IW - len(ver_text) - ld
    lines.append(
        f" {PURPLE_BASE}╭{'─'*ld}{ver_text}{'─'*rd}╮{RESET}"
    )

    # ── Column-header row ─────────────────────────────────────────────────────
    # Both inner │ chars are PURPLE_BASE; text styled with WHITE_BRIGHT / CYAN_GLOW
    lines.append(
        f" {PURPLE_BASE}│{RESET}"
        f"  {WHITE_BRIGHT}{BOLD}{'SYSTEM STATUS':<25}{RESET}"
        f"{PURPLE_BASE}│{RESET}"
        f" {CYAN_GLOW}{BOLD}{'AVAILABLE TOOLS':<46}{RESET}"
        f" {PURPLE_BASE}│{RESET}"
    )

    # ── Separator ─────────────────────────────────────────────────────────────
    lines.append(f" {PURPLE_BASE}├{'─'*_LW}┼{'─'*_RW}┤{RESET}")

    # ── Body rows: left column (mascot+info) + right column (tools) ──────────
    left_rows  = _build_left_rows(model_name, cwd, session_id, tool_count, skill_count)
    right_rows = _build_right_rows(toolsets)
    n = max(len(left_rows), len(right_rows))

    for i in range(n):
        lc = left_rows[i]  if i < len(left_rows)  else ""
        rc = right_rows[i] if i < len(right_rows) else ""
        lines.append(
            f" {PURPLE_BASE}│{RESET}"
            f"{_rpad(lc, _LW)}"
            f"{PURPLE_BASE}│{RESET}"
            f"{_rpad(rc, _RW)}"
            f"{PURPLE_BASE}│{RESET}"
        )

    # ── Footer separator (┴ merges the two columns) ──────────────────────────
    lines.append(f" {PURPLE_BASE}├{'─'*_LW}┴{'─'*_RW}┤{RESET}")

    # ── Telemetry row ─────────────────────────────────────────────────────────
    # Visible budget: 1(sp) + 10(TELEMETRY:) + 1+16(date) + 3+14(CPU) + 3+14(RAM)
    #               = 62 fixed.  Remaining = 76-62-1 = 13 for model name.
    date_s    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cpu_s     = f"CPU: {_cpu_pct()}%"
    ram_s     = f"RAM: {_ram_gb()}GB"
    model_tel = model_name[:_IW - 63]   # 76-63 = 13 chars max → total exactly 76
    tel_inner = (
        f" {GRAY_TEXT}TELEMETRY:{RESET}"
        f" {CYAN_GLOW}{date_s[:16]:<16}{RESET}"
        f" {PURPLE_BASE}│{RESET} {CYAN_GLOW}{cpu_s[:14]:<14}{RESET}"
        f" {PURPLE_BASE}│{RESET} {CYAN_GLOW}{ram_s[:14]:<14}{RESET}"
        f" {WHITE_BRIGHT}{model_tel}{RESET}"
    )
    lines.append(
        f" {PURPLE_BASE}│{RESET}{_rpad(tel_inner, _IW)}{PURPLE_BASE}│{RESET}"
    )

    # ── Bottom border ─────────────────────────────────────────────────────────
    lines.append(f" {PURPLE_BASE}╰{'─'*_IW}╯{RESET}")

    # ── Below-box welcome (matches Hermes Agent / Claude Code style) ──────────
    lines.append("")
    lines.append(
        f"{WHITE_BRIGHT}Welcome to Operon!{RESET} "
        f"{GRAY_TEXT}Type your message or "
        f"{PURPLE_LIGHT}/help{RESET}{GRAY_TEXT} for commands.{RESET}"
    )
    lines.append(
        f"{GRAY_TEXT}  ? for shortcuts  ·  ← /delegate  ·  "
        f"hint:code / hint:fast / hint:reasoning  ·  Ctrl+D to exit{RESET}"
    )

    return "\n".join(lines)


# ── Banner class ──────────────────────────────────────────────────────────────

class Banner:
    def display(
        self,
        model_name:  str  = "operon",
        tool_count:  int  = 0,
        skill_count: int  = 0,
        toolsets:    Optional[Dict[str, List[str]]] = None,
        skills:      Optional[List[dict]] = None,
        session_id:  Optional[str] = None,
        cwd:         Optional[str] = None,
        version:     str  = "3.1.0",
    ) -> None:
        os.system("clear" if os.name != "nt" else "cls")
        print(render(
            model_name  = model_name,
            tool_count  = tool_count,
            skill_count = skill_count,
            toolsets    = toolsets,
            skills      = skills,
            session_id  = session_id,
            cwd         = cwd,
            version     = version,
        ))
        print()
