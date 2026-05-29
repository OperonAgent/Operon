"""
ui/tui.py — Operon Terminal UI (Hermes Agent + Claude Code style)

  • Persistent input bar at the bottom (like Claude Code / Fish shell)
  • Full command history saved to ~/.operon/history
  • Tab completion for all /commands
  • Ctrl+C  — interrupt current input (not exit)
  • Ctrl+D  — graceful exit
  • Multi-line input: Shift+Enter or trailing \\ for continuation
  • Status bar (bottom toolbar) — matches Hermes Agent layout:
      model | ctx [██████░░░░░░] N% | turn #N | $cost/local | ⊙ Ns
  • Clean ❯  prompt (no emoji noise)
  • Inline syntax highlighting for code fences (requires pygments)

Graceful degradation: if prompt_toolkit is not installed, falls back to
the original bare input() path.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

# ── prompt_toolkit optional import ───────────────────────────────────────────
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import HTML, ANSI
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style
    _PT_AVAILABLE = True
except ImportError:
    _PT_AVAILABLE = False

# ── History file ─────────────────────────────────────────────────────────────
_HISTORY_PATH = Path.home() / ".operon" / "history"
_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── All /commands for tab completion ─────────────────────────────────────────
_SLASH_COMMANDS: List[str] = [
    "/help", "/exit", "/quit", "/clear", "/model", "/config",
    "/memory", "/memory search", "/memory list", "/memory clear",
    "/context", "/context clear", "/context save",
    "/tools", "/tools list",
    "/shell", "/python",
    "/kanban", "/kanban add", "/kanban list", "/kanban done",
    "/checkpoint", "/checkpoint save", "/checkpoint list", "/checkpoint restore",
    "/plugin", "/plugin search", "/plugin install", "/plugin list",
    "/conv", "/conv status", "/conv compress", "/conv reset",
    "/obsidian", "/obsidian sync", "/obsidian status",
    "/slack", "/slack send", "/slack list",
    "/desktop", "/desktop screenshot", "/desktop click", "/desktop type",
    "/voice", "/voice start", "/voice stop",
    "/swe", "/swe solve",
    "/delegate", "/delegate spawn",
    "/skill", "/skill list", "/skill create",
    "/mesh", "/mesh auto",
    "/local", "/local use", "/local list",
    "/cost", "/session", "/stats", "/approve",
    "/multiline", "/paste",
    "/doctor", "/dashboard",
    "/vector", "/vector search",
    "/router", "/router status",
]

# ── Context progress bar ──────────────────────────────────────────────────────
_CTX_FILL  = "█"
_CTX_EMPTY = "░"
_CTX_WIDTH = 12   # chars in the bar


def _ctx_bar(used: int, total: int) -> str:
    """Render a 12-char block progress bar for context usage."""
    if total <= 0:
        return _CTX_EMPTY * _CTX_WIDTH
    frac   = min(1.0, used / total)
    filled = round(frac * _CTX_WIDTH)
    bar    = _CTX_FILL * filled + _CTX_EMPTY * (_CTX_WIDTH - filled)
    pct    = int(frac * 100)
    return f"{bar} {pct}%"


# ── Tab completer ─────────────────────────────────────────────────────────────
class OperonCompleter(Completer if _PT_AVAILABLE else object):   # type: ignore
    """Tab-completes /commands."""

    def get_completions(self, document: Any, complete_event: Any):  # type: ignore
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd in _SLASH_COMMANDS:
            if cmd.startswith(text) and cmd != text:
                suffix = cmd[len(text):]
                yield Completion(
                    suffix,
                    start_position=0,
                    display=cmd,
                    display_meta="operon",
                )


# ── prompt_toolkit Style — Operon purple/dark palette ────────────────────────
_PT_STYLE = Style.from_dict({
    # Bottom toolbar — dark navy background, soft lavender text
    "bottom-toolbar":           "bg:#0d0d1a #bf7fff",
    "bottom-toolbar.text":      "bg:#0d0d1a #bf7fff",
    "bottom-toolbar.model":     "bg:#0d0d1a #d7afff bold",
    "bottom-toolbar.ctx":       "bg:#0d0d1a #9b59ff",
    "bottom-toolbar.turn":      "bg:#0d0d1a #888888",
    "bottom-toolbar.cost":      "bg:#0d0d1a #4ecca3",
    "bottom-toolbar.sep":       "bg:#0d0d1a #333355",
    "bottom-toolbar.ok":        "bg:#0d0d1a #4ECC7A",
    "bottom-toolbar.warn":      "bg:#0d0d1a #d787ff",
    "bottom-toolbar.dim":       "bg:#0d0d1a #555577",

    # Input area — purple prompt arrow
    "prompt":                   "#d7afff bold",

    # Completion menu — purple highlight
    "completion-menu.completion":               "bg:#1a1a2e #e0e0e0",
    "completion-menu.completion.current":       "bg:#9b59ff #ffffff bold",
    "completion-menu.meta.completion":          "bg:#0d0d1a #888888",
    "completion-menu.meta.completion.current":  "bg:#7b3fff #ffffff",

    # Scrollbar
    "scrollbar.background":     "bg:#1a1a2e",
    "scrollbar.button":         "bg:#9b59ff",

    # Auto-suggest (ghost text)
    "auto-suggestion":          "#333355 italic",
}) if _PT_AVAILABLE else None


def _make_toolbar(
    model_name:   str   = "operon",
    turn:         int   = 0,
    cost_usd:     float = 0.0,
    ctx_used:     int   = 0,
    ctx_total:    int   = 0,
    elapsed_s:    float = 0.0,
    mem_facts:    int   = 0,
    extra:        str   = "",
) -> "HTML":
    """
    Build bottom status bar matching Hermes Agent style:
      model  │  ctx [████████░░░░] 68%  │  turn #3  │  local  │  ⊙ 0s
    """
    model_short = model_name[:22]
    cost_str    = f"${cost_usd:.4f}" if cost_usd > 0 else "local"
    bar_str     = _ctx_bar(ctx_used, ctx_total)
    turn_str    = f"#{turn}"
    elapsed_str = f"⊙ {elapsed_s:.0f}s" if elapsed_s > 0 else "⊙ 0s"
    mem_str     = f"  <sep>│</sep>  <dim>{mem_facts} facts</dim>" if mem_facts else ""
    extra_str   = f"  <sep>│</sep>  <warn>{extra}</warn>" if extra else ""

    return HTML(
        f'<bottom-toolbar>'
        f'  <model>{model_short}</model>'
        f'  <sep> │ </sep>'
        f'ctx <ctx>[{bar_str}]</ctx>'
        f'  <sep> │ </sep>'
        f'<turn>turn {turn_str}</turn>'
        f'  <sep> │ </sep>'
        f'<cost>{cost_str}</cost>'
        f'  <sep> │ </sep>'
        f'<dim>{elapsed_str}</dim>'
        + mem_str
        + extra_str
        + f'</bottom-toolbar>'
    )


# ── Key bindings ──────────────────────────────────────────────────────────────
def _make_keybindings() -> "KeyBindings":
    kb = KeyBindings()

    @kb.add("escape", eager=True)
    def _escape(event):
        event.current_buffer.reset()

    @kb.add("c-c")
    def _ctrl_c(event):
        raise KeyboardInterrupt()

    @kb.add("c-d")
    def _ctrl_d(event):
        raise EOFError()

    def _shift_enter(event):
        event.current_buffer.insert_text("\n")

    try:
        kb.add("s-enter")(_shift_enter)
    except (ValueError, KeyError):
        pass

    return kb


# ── OperonTUI ─────────────────────────────────────────────────────────────────
class OperonTUI:
    """
    Drop-in replacement for Python's built-in input().

    Usage:
        tui = OperonTUI(model_name="hermes3:8b")
        try:
            text = tui.prompt()       # blocks, returns stripped string
        except KeyboardInterrupt:
            ...                       # Ctrl+C — don't exit
        except EOFError:
            break                     # Ctrl+D — exit
    """

    def __init__(
        self,
        model_name:      str   = "operon",
        history_path:    Path  = _HISTORY_PATH,
        enable_autosugg: bool  = True,
        ctx_total:       int   = 4096,
    ) -> None:
        self.model_name   = model_name
        self._turn        = 0
        self._cost_usd    = 0.0
        self._mem_facts   = 0
        self._ctx_used    = 0
        self._ctx_total   = ctx_total
        self._extra       = ""
        self._turn_start  = time.monotonic()

        if not _PT_AVAILABLE:
            self._session = None
            return

        kb = _make_keybindings()

        self._session: Optional[PromptSession] = PromptSession(  # type: ignore
            history               = FileHistory(str(history_path)),
            completer             = OperonCompleter(),
            auto_suggest          = AutoSuggestFromHistory() if enable_autosugg else None,
            key_bindings          = kb,
            style                 = _PT_STYLE,
            vi_mode               = False,
            mouse_support         = False,
            bottom_toolbar        = self._toolbar,
            multiline             = False,
            wrap_lines            = True,
            complete_while_typing = True,
            reserve_space_for_menu = 4,
        )

    # ── Status setters (called from main loop) ────────────────────────────────

    def set_model(self, name: str) -> None:
        self.model_name = name

    def set_turn(self, n: int) -> None:
        self._turn       = n
        self._turn_start = time.monotonic()

    def add_cost(self, usd: float) -> None:
        self._cost_usd += usd

    def set_cost(self, usd: float) -> None:
        self._cost_usd = usd

    def set_mem_facts(self, n: int) -> None:
        self._mem_facts = n

    def set_ctx(self, used: int, total: int = 0) -> None:
        """Update context window usage for the progress bar."""
        self._ctx_used  = used
        if total > 0:
            self._ctx_total = total

    def set_status(self, text: str) -> None:
        self._extra = text

    def clear_status(self) -> None:
        self._extra = ""

    # ── Back-compat alias used by tests and older call-sites ──────────────────
    @property
    def _extra_status(self) -> str:
        return self._extra

    @_extra_status.setter
    def _extra_status(self, value: str) -> None:
        self._extra = value

    # ── Toolbar callback ──────────────────────────────────────────────────────

    def _toolbar(self) -> "HTML":
        elapsed = time.monotonic() - self._turn_start
        return _make_toolbar(
            model_name = self.model_name,
            turn       = self._turn,
            cost_usd   = self._cost_usd,
            ctx_used   = self._ctx_used,
            ctx_total  = self._ctx_total,
            elapsed_s  = elapsed,
            mem_facts  = self._mem_facts,
            extra      = self._extra,
        )

    # ── Primary prompt ────────────────────────────────────────────────────────

    def prompt(self, placeholder: str = "") -> str:
        """
        Display the input bar and return the user's text.
        Raises KeyboardInterrupt or EOFError.
        """
        if self._session is None:
            try:
                return input(_fallback_prompt()).strip()
            except KeyboardInterrupt:
                return ""
            except EOFError:
                return "/exit"

        # Clean purple prompt arrow: ❯
        prompt_msg = HTML('<prompt><b>❯</b></prompt> ')

        text = self._session.prompt(
            prompt_msg,
            placeholder=(
                HTML(f'<ansigray>{placeholder}</ansigray>')
                if placeholder else None
            ),
        )
        text = text.strip()

        # Multi-line continuation with trailing backslash
        while text.endswith("\\"):
            text = text[:-1] + "\n"
            cont = self._session.prompt(
                HTML('<ansigray>  … </ansigray>'),
            )
            text += cont.strip()

        self._turn_start = time.monotonic()
        return text

    # ── Inline yes/no ─────────────────────────────────────────────────────────

    def ask_yn(self, question: str, default: bool = False) -> bool:
        suffix = " [Y/n]" if default else " [y/N]"
        if self._session is None:
            _ans = input(f"  {question}{suffix} ").strip().lower()
            return _ans in ("y", "yes") or (default and _ans == "")
        try:
            ans = self._session.prompt(
                HTML(f'<ansiyellow>  ⚠ {question}{suffix} </ansiyellow>'),
            ).strip().lower()
            return ans in ("y", "yes") or (default and ans == "")
        except (KeyboardInterrupt, EOFError):
            return default

    # ── Status print above the bar ────────────────────────────────────────────

    def print_status(self, text: str, colour: str = "cyan") -> None:
        _colour_map = {
            "cyan":   "\033[1;38;5;81m",
            "green":  "\033[1;38;5;82m",
            "yellow": "\033[38;5;214m",
            "amber":  "\033[1;38;5;214m",
            "red":    "\033[1;38;5;196m",
            "gray":   "\033[38;5;244m",
            "purple": "\033[1;38;5;141m",
        }
        c = _colour_map.get(colour, "")
        sys.stdout.write(f"\n{c}  {text}\033[0m\n")
        sys.stdout.flush()


# ── Fallback prompt (no prompt_toolkit) ──────────────────────────────────────
def _fallback_prompt() -> str:
    return "\033[1;38;5;141m❯\033[0m "


# ── Module-level singleton ────────────────────────────────────────────────────
_tui_instance: Optional[OperonTUI] = None


def get_tui(model_name: str = "operon", ctx_total: int = 4096) -> OperonTUI:
    global _tui_instance
    if _tui_instance is None:
        _tui_instance = OperonTUI(model_name=model_name, ctx_total=ctx_total)
    return _tui_instance


def reset_tui(model_name: str = "operon", ctx_total: int = 4096) -> OperonTUI:
    global _tui_instance
    _tui_instance = OperonTUI(model_name=model_name, ctx_total=ctx_total)
    return _tui_instance


# ── Availability flag ─────────────────────────────────────────────────────────
TUI_AVAILABLE = _PT_AVAILABLE
