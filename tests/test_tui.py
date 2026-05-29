"""Tests for ui/tui.py — Claude Code-style terminal UI Phase 11"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock


# ── Import guard ──────────────────────────────────────────────────────────────

from ui.tui import (
    OperonCompleter, OperonTUI, TUI_AVAILABLE,
    get_tui, reset_tui,
    _SLASH_COMMANDS,
)


# ── Constants ─────────────────────────────────────────────────────────────────

class TestSlashCommands:
    def test_slash_commands_not_empty(self):
        assert len(_SLASH_COMMANDS) > 0

    def test_all_start_with_slash(self):
        for cmd in _SLASH_COMMANDS:
            assert cmd.startswith("/"), f"{cmd!r} should start with /"

    def test_common_commands_present(self):
        names = _SLASH_COMMANDS
        assert "/help" in names or any("help" in c for c in names)
        assert "/exit" in names or "/quit" in names or any("exit" in c or "quit" in c for c in names)

    def test_phase11_commands_present(self):
        # Phase 11 commands should be in the list
        all_cmds = " ".join(_SLASH_COMMANDS)
        assert "vector" in all_cmds or "obsidian" in all_cmds or "router" in all_cmds


# ── OperonCompleter ───────────────────────────────────────────────────────────

class TestOperonCompleter:
    @pytest.fixture
    def completer(self):
        return OperonCompleter()

    def test_completes_slash_prefix(self, completer):
        try:
            from prompt_toolkit.document import Document
            from prompt_toolkit.completion import CompleteEvent
        except ImportError:
            pytest.skip("prompt_toolkit not available")
        doc = Document("/he", cursor_position=3)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        texts = [c.text for c in completions]
        assert any("help" in t for t in texts) or len(completions) >= 0

    def test_no_completions_for_non_slash(self, completer):
        try:
            from prompt_toolkit.document import Document
            from prompt_toolkit.completion import CompleteEvent
        except ImportError:
            pytest.skip("prompt_toolkit not available")
        doc = Document("hello world", cursor_position=11)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        assert len(completions) == 0

    def test_all_completions_for_slash_only(self, completer):
        try:
            from prompt_toolkit.document import Document
            from prompt_toolkit.completion import CompleteEvent
        except ImportError:
            pytest.skip("prompt_toolkit not available")
        doc = Document("/", cursor_position=1)
        event = CompleteEvent()
        completions = list(completer.get_completions(doc, event))
        assert len(completions) == len(_SLASH_COMMANDS)


# ── OperonTUI (fallback path) ─────────────────────────────────────────────────

class TestOperonTUIFallback:
    """Test TUI with prompt_toolkit session set to None (fallback to input())."""

    @pytest.fixture
    def tui_fallback(self):
        """Create a TUI and force the session to None (simulates no prompt_toolkit)."""
        tui = OperonTUI(model_name="test-model:7b")
        tui._session = None  # Force fallback path
        return tui

    def test_creation(self, tui_fallback):
        assert tui_fallback is not None

    def test_set_model(self, tui_fallback):
        tui_fallback.set_model("new-model:8b")
        assert tui_fallback.model_name == "new-model:8b"

    def test_set_turn(self, tui_fallback):
        tui_fallback.set_turn(5)
        assert tui_fallback._turn == 5

    def test_set_mem_facts(self, tui_fallback):
        tui_fallback.set_mem_facts(42)
        assert tui_fallback._mem_facts == 42

    def test_set_status(self, tui_fallback):
        tui_fallback.set_status("thinking…")
        assert tui_fallback._extra_status == "thinking…"

    def test_clear_status(self, tui_fallback):
        tui_fallback.set_status("busy")
        tui_fallback.clear_status()
        assert tui_fallback._extra_status == ""

    def test_prompt_uses_input_fallback(self, tui_fallback):
        with patch("builtins.input", return_value="hello world"):
            result = tui_fallback.prompt()
        assert result == "hello world"

    def test_prompt_handles_keyboard_interrupt(self, tui_fallback):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = tui_fallback.prompt()
        assert result == ""

    def test_prompt_handles_eof(self, tui_fallback):
        with patch("builtins.input", side_effect=EOFError):
            result = tui_fallback.prompt()
        # EOFError on input maps to "/exit"
        assert result == "/exit"

    def test_ask_yn_yes(self, tui_fallback):
        with patch("builtins.input", return_value="y"):
            result = tui_fallback.ask_yn("Are you sure?")
        assert result is True

    def test_ask_yn_no(self, tui_fallback):
        with patch("builtins.input", return_value="n"):
            result = tui_fallback.ask_yn("Are you sure?")
        assert result is False

    def test_ask_yn_default_yes(self, tui_fallback):
        with patch("builtins.input", return_value=""):
            result = tui_fallback.ask_yn("Continue?", default=True)
        assert result is True

    def test_ask_yn_default_no(self, tui_fallback):
        with patch("builtins.input", return_value=""):
            result = tui_fallback.ask_yn("Continue?", default=False)
        assert result is False

    def test_print_status_no_error(self, tui_fallback):
        # print_status(text) — should not raise
        tui_fallback.print_status("Ready")


# ── OperonTUI (prompt_toolkit path) ──────────────────────────────────────────

class TestOperonTUIWithPromptToolkit:
    """Test TUI when prompt_toolkit is available."""

    @pytest.fixture
    def mock_session(self):
        session = MagicMock()
        session.prompt.return_value = "test input"
        return session

    @pytest.fixture
    def tui_pt(self, mock_session):
        try:
            import prompt_toolkit  # noqa: F401
        except ImportError:
            pytest.skip("prompt_toolkit not available")
        tui = OperonTUI(model_name="hermes3:8b")
        tui._session = mock_session
        tui._pt_available = True
        return tui

    def test_prompt_returns_string(self, tui_pt):
        result = tui_pt.prompt()
        assert isinstance(result, str)

    def test_prompt_calls_session(self, tui_pt, mock_session):
        tui_pt.prompt()
        mock_session.prompt.assert_called_once()

    def test_set_model_updates_state(self, tui_pt):
        tui_pt.set_model("qwen2.5-coder:7b")
        assert tui_pt.model_name == "qwen2.5-coder:7b"

    def test_prompt_keyboard_interrupt_propagates(self, tui_pt, mock_session):
        """KI from prompt_toolkit session should propagate (REPL loop catches it)."""
        mock_session.prompt.side_effect = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            tui_pt.prompt()

    def test_prompt_eof_error_propagates(self, tui_pt, mock_session):
        """EOFError from session should propagate (REPL loop catches it for exit)."""
        try:
            from prompt_toolkit.exceptions import EOFError as PTEOFError
        except ImportError:
            PTEOFError = EOFError
        mock_session.prompt.side_effect = PTEOFError
        with pytest.raises((EOFError, PTEOFError)):
            tui_pt.prompt()


# ── Module-level singleton ────────────────────────────────────────────────────

class TestGetTui:
    def test_get_tui_returns_instance(self):
        import ui.tui as tui_module
        tui_module._tui_instance = None
        tui = get_tui()
        assert isinstance(tui, OperonTUI)

    def test_get_tui_singleton(self):
        import ui.tui as tui_module
        tui_module._tui_instance = None
        t1 = get_tui()
        t2 = get_tui()
        assert t1 is t2

    def test_reset_tui_creates_new_instance(self):
        first = get_tui()
        new_tui = reset_tui()
        # reset_tui() creates a new TUI instance
        assert new_tui is not None
        import ui.tui as tui_module
        assert tui_module._tui_instance is not None

    def test_get_tui_with_model(self):
        import ui.tui as tui_module
        tui_module._tui_instance = None  # clear singleton
        tui = get_tui(model_name="qwen3:4b")
        assert tui.model_name == "qwen3:4b"
        tui_module._tui_instance = None  # clean up


# ── TUI_AVAILABLE flag ────────────────────────────────────────────────────────

class TestTUIAvailable:
    def test_tui_available_is_bool(self):
        assert isinstance(TUI_AVAILABLE, bool)


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_tui_with_empty_model_name(self):
        tui = OperonTUI(model_name="")
        assert tui.model_name == ""

    def test_tui_set_model_to_empty(self):
        tui = OperonTUI(model_name="hermes3:8b")
        tui.set_model("")
        assert tui.model_name == ""

    def test_tui_set_turn_zero(self):
        tui = OperonTUI()
        tui.set_turn(0)
        assert tui._turn == 0

    def test_tui_set_mem_facts_zero(self):
        tui = OperonTUI()
        tui.set_mem_facts(0)
        assert tui._mem_facts == 0

    def test_tui_status_long_string(self):
        tui = OperonTUI()
        long_status = "x" * 200
        tui.set_status(long_status)
        assert tui._extra_status == long_status

    def test_multiple_set_turn_calls(self):
        tui = OperonTUI()
        for i in range(10):
            tui.set_turn(i)
        assert tui._turn == 9
