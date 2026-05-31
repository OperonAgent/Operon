"""tests/test_macro_cmds.py — migrated /macro modular handler."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cmd_handlers as ch
import main


@pytest.fixture(autouse=True)
def _reset_macros():
    saved = main._macros
    main._macros = None
    yield
    main._macros = saved


def _ctx(cmd, mgr=None):
    parts = cmd.split()
    theme = MagicMock()
    theme.box = lambda l, **k: "BOX"
    theme.info = theme.success = theme.error = theme.warning = lambda x: str(x)
    ctx = ch.CommandContext(command=cmd, parts=parts, cmd=parts[0], args=parts[1:],
                            theme=theme, tool_registry=MagicMock())
    if mgr is not None:
        main._macros = mgr
    return ctx


class TestRegistration:
    def test_macro_registered(self):
        assert "/macro" in ch.DISPATCH

    def test_not_in_legacy_chain(self):
        # main.py legacy elif no longer handles /macro inline
        src = (Path(__file__).resolve().parent.parent / "main.py").read_text()
        assert 'elif cmd == "/macro":' not in src


class TestLazyInit:
    def test_creates_manager_on_first_use(self):
        fake = MagicMock(); fake.list_macros.return_value = []
        ctx = _ctx("/macro list")
        with patch("main.MacroManager", return_value=fake), patch("builtins.print"):
            ch.dispatch(ctx)
        assert main._macros is fake

    def test_reuses_existing_manager(self):
        fake = MagicMock(); fake.list_macros.return_value = []
        ctx = _ctx("/macro list", mgr=fake)
        with patch("main.MacroManager") as ctor, patch("builtins.print"):
            ch.dispatch(ctx)
        ctor.assert_not_called()


class TestSubcommands:
    def test_list_empty(self):
        fake = MagicMock(); fake.list_macros.return_value = []
        ctx = _ctx("/macro list", mgr=fake)
        with patch("builtins.print") as p:
            ch.dispatch(ctx)
        assert p.called

    def test_list_with_macros(self):
        fake = MagicMock()
        fake.list_macros.return_value = [{"name": "daily", "steps": [1, 2], "description": "d"}]
        ctx = _ctx("/macro list", mgr=fake)
        with patch("builtins.print"):
            ch.dispatch(ctx)
        fake.list_macros.assert_called_once()

    def test_run_parses_vars(self):
        fake = MagicMock()
        fake.run.return_value = {"success": True, "steps": [1], "output": "ok"}
        ctx = _ctx("/macro run job a=1 b=2", mgr=fake)
        with patch("builtins.print"):
            ch.dispatch(ctx)
        fake.run.assert_called_once_with("job", vars={"a": "1", "b": "2"})

    def test_run_no_name_warns(self):
        fake = MagicMock()
        ctx = _ctx("/macro run", mgr=fake)
        with patch("builtins.print"):
            ch.dispatch(ctx)
        fake.run.assert_not_called()

    def test_run_failure(self):
        fake = MagicMock()
        fake.run.return_value = {"success": False, "error": "boom"}
        ctx = _ctx("/macro run job", mgr=fake)
        with patch("builtins.print"):
            ch.dispatch(ctx)
        fake.run.assert_called_once()

    def test_delete(self):
        fake = MagicMock()
        fake.delete.return_value = {"success": True}
        ctx = _ctx("/macro delete job", mgr=fake)
        with patch("builtins.print"):
            ch.dispatch(ctx)
        fake.delete.assert_called_once_with("job")

    def test_delete_no_name_warns(self):
        fake = MagicMock()
        ctx = _ctx("/macro delete", mgr=fake)
        with patch("builtins.print"):
            ch.dispatch(ctx)
        fake.delete.assert_not_called()

    def test_define_hint(self):
        fake = MagicMock()
        ctx = _ctx("/macro define", mgr=fake)
        with patch("builtins.print") as p:
            ch.dispatch(ctx)
        assert p.called

    def test_unknown_sub_warns(self):
        fake = MagicMock(); fake.list_macros.return_value = []
        ctx = _ctx("/macro frobnicate", mgr=fake)
        with patch("builtins.print") as p:
            ch.dispatch(ctx)
        assert p.called

    def test_bare_macro_defaults_to_list(self):
        fake = MagicMock(); fake.list_macros.return_value = []
        ctx = _ctx("/macro", mgr=fake)
        with patch("builtins.print"):
            ch.dispatch(ctx)
        fake.list_macros.assert_called_once()
