"""
Operon Terminal Theme — V2.
Matches Operon Engine V1.0.0 color palette + prompt style.
Adds: ThinkingSpinner, streaming chat output, emoji scratchpad rows.
"""

import re
import sys
import time
import threading

# ── Optional syntax highlighting (pygments) ───────────────────────────────────
try:
    from pygments import highlight as _pyg_highlight
    from pygments.lexers import get_lexer_by_name as _get_lexer, TextLexer as _TextLexer
    from pygments.formatters import Terminal256Formatter as _TermFmt
    _PYGMENTS = True
except ImportError:
    _PYGMENTS = False

_CODE_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def _highlight_code_fences(text: str) -> str:
    """
    Replace ```lang\\n...``` blocks with pygments-highlighted output.
    Falls back to plain text if pygments is not installed.
    """
    if not _PYGMENTS:
        return text

    def _replace(m: re.Match) -> str:
        lang = m.group(1).strip() or "text"
        code = m.group(2)
        try:
            lexer = _get_lexer(lang, stripall=False)
        except Exception:
            lexer = _TextLexer()
        highlighted = _pyg_highlight(code, lexer, _TermFmt(style="monokai"))
        # Wrap in a subtle dim border so it stands out from prose
        return f"\033[2m───\033[0m\n{highlighted.rstrip()}\n\033[2m───\033[0m"

    return _CODE_FENCE_RE.sub(_replace, text)

# ── Raw escape helpers ────────────────────────────────────────────────────────
def _fg(n: int) -> str:
    return f"\033[38;5;{n}m"

def _bg(n: int) -> str:
    return f"\033[48;5;{n}m"

# ── Named palette (matching V1.0.0 variable names) ────────────────────────────
RESET         = "\033[0m"
BOLD          = "\033[1m"
DIM           = "\033[2m"
ITALIC        = "\033[3m"
UNDERLINE     = "\033[4m"

# V1.0.0 primary names
PURPLE_BASE   = "\033[1;38;5;99m"    # bold purple-blue
PURPLE_LIGHT  = "\033[1;38;5;141m"   # bold light purple
PURPLE_DIM    = "\033[38;5;61m"      # muted purple
CYAN_GLOW     = "\033[1;38;5;81m"    # bold neon cyan
WHITE_BRIGHT  = "\033[1;38;5;255m"   # bold white
GRAY_TEXT     = "\033[38;5;244m"     # mid gray

# Extended palette
PURPLE_DEEP   = _fg(55)
PURPLE        = _fg(93)
CYAN_NEON     = _fg(51)
CYAN          = _fg(87)
CYAN_DIM      = _fg(38)
WHITE         = _fg(231)
WHITE_DIM     = _fg(252)
GRAY          = _fg(244)
GRAY_DARK     = _fg(238)
AMBER         = _fg(214)
GREEN_NEON    = _fg(82)
GREEN         = _fg(46)
RED           = _fg(196)
RED_SOFT      = _fg(203)
ORANGE        = _fg(208)
YELLOW        = _fg(226)
BLUE          = _fg(33)
MAGENTA       = _fg(201)
PINK          = _fg(213)

BG_DARK       = _bg(234)
BG_MID        = _bg(236)
BG_PURPLE     = _bg(54)


# ── Thinking Spinner ─────────────────────────────────────────────────────────

class ThinkingSpinner:
    """Braille-character animation that runs in a daemon thread."""

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self):
        self.active    = False
        self._thread   = None

    def _spin(self):
        idx = 0
        while self.active:
            sys.stdout.write(
                f"\r {PURPLE_LIGHT}{self._FRAMES[idx]} Operon is analyzing data streams...{RESET}"
            )
            sys.stdout.flush()
            idx = (idx + 1) % len(self._FRAMES)
            time.sleep(0.08)
        # ANSI: \r = move to col 0, \033[2K = erase entire current line
        # This fully removes the spinner line instead of leaving blank space
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()

    def start(self):
        self.active  = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self.active = False
        if self._thread:
            self._thread.join(timeout=1)


# ── Theme class ───────────────────────────────────────────────────────────────

class Theme:
    """Centralised styling for all Operon terminal output."""

    WIDTH = 78

    # ── Prompts ───────────────────────────────────────────────────────────────

    def prompt(self) -> str:
        """User input prompt — matches V1.0.0 style."""
        return f"{PURPLE_LIGHT} YOU{PURPLE_BASE} ❯{RESET} "

    # ── Status indicators ─────────────────────────────────────────────────────

    def thinking(self, text: str) -> str:
        return f"{DIM}{GRAY_TEXT}{text}{RESET}"

    def info(self, text: str) -> str:
        return f"{CYAN_DIM}  {text}{RESET}"

    def success(self, text: str) -> str:
        return f"{GREEN_NEON}  ✓ {text}{RESET}"

    def warning(self, text: str) -> str:
        return f"{AMBER}  ! {text}{RESET}"

    def error(self, text: str) -> str:
        return f"{RED}  ✗ {text}{RESET}"

    def dim(self, text: str) -> str:
        return f"{DIM}{GRAY_TEXT}{text}{RESET}"

    # ── Agent response — V1.0.0 style with optional streaming ────────────────

    def assistant_response(self, text: str, stream: bool = True) -> None:
        """
        Print the agent's final response.
        - Code fences are syntax-highlighted (requires pygments)
        - First line is prefixed with   OPERON ❯
        - Remaining lines are indented
        - stream=True: character-by-character typing effect
        """
        text = _highlight_code_fences(text)
        text_lines = text.strip().split("\n")

        # Build prefix for line 0
        text_lines[0] = (
            f"\n{PURPLE_LIGHT} OPERON ❯ {WHITE_BRIGHT}" + text_lines[0]
        )
        for i in range(1, len(text_lines)):
            text_lines[i] = f"{WHITE_BRIGHT}{text_lines[i]}"

        for line in text_lines:
            if stream:
                for ch in line:
                    sys.stdout.write(ch)
                    sys.stdout.flush()
                    time.sleep(0.006)
                sys.stdout.write(f"{RESET}\n")
            else:
                print(f"{line}{RESET}")
        print()

    # ── Tool display ──────────────────────────────────────────────────────────

    def tool_call(self, text: str) -> str:
        return f"{CYAN_GLOW}    {text}{RESET}"

    def tool_result(self, text: str) -> str:
        return f" {PURPLE_DIM} [OBSERVATION]: {text}{RESET}"

    # ── Generic box (for /help, /config, etc.) ────────────────────────────────

    def box(self, lines: list[str], color: str = None) -> str:
        color    = color or PURPLE_BASE
        inner_w  = self.WIDTH - 2
        top      = f"{color}╭{'─' * inner_w}╮{RESET}"
        bottom   = f"{color}╰{'─' * inner_w}╯{RESET}"
        sep      = f"{color}├{'─' * inner_w}┤{RESET}"

        result = [top]
        for line in lines:
            if line == "---":
                result.append(sep)
            else:
                padded = line[:inner_w].ljust(inner_w)
                result.append(f"{color}│{RESET}{WHITE_DIM}{padded}{RESET}{color}│{RESET}")
        result.append(bottom)
        return "\n".join(result)

    # ── Planner / scratchpad box ──────────────────────────────────────────────

    def planner_box(self, rows: list[tuple[str, str, str]]) -> str:
        """
        rows: list of (emoji_label, label_text, value)
        Renders the V1.0.0-style scratchpad frame with rounded corners.
        Total width = 78 chars, inner = 76.
        """
        inner_w  = self.WIDTH - 2  # 76
        content_w = inner_w - 2   # 74 (1 space each side)

        header_tag = " [OPERON CORE SCRATCHPAD FRAME] "
        dashes_left  = 2
        dashes_right = inner_w - dashes_left - len(header_tag)
        top    = f"{PURPLE_BASE}┌{'─' * dashes_left}{WHITE_BRIGHT}{header_tag}{RESET}{PURPLE_BASE}{'─' * dashes_right}┐{RESET}"
        bottom = f"{PURPLE_BASE}└{'─' * inner_w}┘{RESET}"

        result = [top]
        for emoji, label, value in rows:
            # Combined label: " OBJECTIVE : " = fixed 15-char visual slot
            label_str = f"{emoji} {AMBER}{BOLD}{label:<11}{RESET}{GRAY_TEXT}:{RESET}"
            # value fills the rest, truncated to fit
            # label visual width ≈ 2(emoji) + 1 + 11 + 1 = 15, then ": " = 2 → ~17
            # But emoji = 2 cols, so real = 2 + 1 + 11 + 1 = 15 visible non-ANSI chars + emoji width
            # Use a fixed val_w that looks right
            val_w = content_w - 16
            value_clean = str(value).replace("\n", " ↵ ")
            if len(value_clean) > val_w:
                value_clean = value_clean[:val_w - 1] + "…"
            val_str = f"{WHITE_BRIGHT}{value_clean:<{val_w}}{RESET}"

            inner = f" {label_str} {val_str} "
            result.append(
                f"{PURPLE_BASE}│{RESET}{inner}{PURPLE_BASE}│{RESET}"
            )
        result.append(bottom)
        return "\n".join(result)
