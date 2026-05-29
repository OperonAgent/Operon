"""Tests for core/computer_use.py — Desktop automation Phase 11

Actual API (from inspection):
  _DISPATCH keys: desktop_screenshot, desktop_mouse_click, desktop_keyboard_type,
                  desktop_keyboard_hotkey, desktop_mouse_scroll, desktop_find_on_screen,
                  desktop_open_app, desktop_clipboard_get, desktop_clipboard_set, desktop_status
  Module functions: screenshot, mouse_click, mouse_move, mouse_drag, mouse_scroll,
                    keyboard_type, keyboard_hotkey, keyboard_press, keyboard_hold,
                    clipboard_get, clipboard_set, list_windows, open_application,
                    find_on_screen, click_image, wait_for_image, computer_use_status
  ComputerUse class: methods mirror module functions (static wrappers)
"""
import base64
import pytest
from unittest.mock import patch, MagicMock, call


# Import module with pyautogui/mss mocked
import sys
_mock_pyautogui = MagicMock()
_mock_pyautogui.size.return_value = (1920, 1080)
_mock_pyautogui.position.return_value = (500, 300)
_mock_pyautogui.FAILSAFE = True
_mock_pyautogui.PAUSE = 0.05

_mock_mss = MagicMock()

# Patch at sys.modules level before import
sys.modules.setdefault("pyautogui", _mock_pyautogui)
sys.modules.setdefault("mss", _mock_mss)
sys.modules.setdefault("mss.tools", MagicMock())

import core.computer_use as cu


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset mock call counts between tests."""
    _mock_pyautogui.reset_mock()
    _mock_mss.reset_mock()
    yield


# ── Tool definitions ──────────────────────────────────────────────────────────

class TestToolDefinitions:
    def test_tool_definitions_exist(self):
        assert len(cu._TOOL_DEFINITIONS) >= 8

    def test_dispatch_covers_definitions(self):
        for td in cu._TOOL_DEFINITIONS:
            name = td["name"]
            assert name in cu._DISPATCH, f"{name} missing from _DISPATCH"

    def test_tool_definitions_have_required_keys(self):
        for td in cu._TOOL_DEFINITIONS:
            assert "name" in td
            assert "description" in td
            assert "input_schema" in td or "parameters" in td or "params" in td

    def test_desktop_screenshot_in_dispatch(self):
        assert "desktop_screenshot" in cu._DISPATCH

    def test_desktop_mouse_click_in_dispatch(self):
        assert "desktop_mouse_click" in cu._DISPATCH

    def test_desktop_keyboard_type_in_dispatch(self):
        assert "desktop_keyboard_type" in cu._DISPATCH

    def test_desktop_status_in_dispatch(self):
        assert "desktop_status" in cu._DISPATCH

    def test_all_dispatch_values_callable(self):
        for k, v in cu._DISPATCH.items():
            assert callable(v), f"_DISPATCH[{k!r}] is not callable"


# ── screenshot ────────────────────────────────────────────────────────────────

class TestScreenshot:
    def test_screenshot_returns_dict(self):
        fake_b64 = base64.b64encode(b"fakepngdata").decode()
        with patch.object(cu, "_take_screenshot", return_value=fake_b64, create=True):
            result = cu.screenshot()
        assert isinstance(result, dict)

    def test_screenshot_no_args(self):
        # Should not raise
        with patch.object(cu, "_take_screenshot", return_value="abc", create=True):
            result = cu.screenshot()
        assert isinstance(result, dict)

    def test_screenshot_with_save_path(self, tmp_path):
        fake_b64 = base64.b64encode(b"fakepng").decode()
        with patch.object(cu, "_take_screenshot", return_value=fake_b64, create=True):
            result = cu.screenshot(save_path=str(tmp_path / "screen.png"))
        assert isinstance(result, dict)

    def test_screenshot_error_returns_dict(self):
        # Force both _MSS and _PYAUTOGUI to False so screenshot() returns error dict
        original_mss = cu._MSS
        original_pag = cu._PYAUTOGUI
        try:
            cu._MSS = False
            cu._PYAUTOGUI = False
            result = cu.screenshot()
        finally:
            cu._MSS = original_mss
            cu._PYAUTOGUI = original_pag
        assert isinstance(result, dict)
        assert "error" in result


# ── mouse_click ───────────────────────────────────────────────────────────────

class TestMouseClick:
    def test_mouse_click_returns_dict(self):
        result = cu.mouse_click(x=500, y=300)
        assert isinstance(result, dict)

    def test_mouse_click_calls_pyautogui_click(self):
        cu.mouse_click(x=100, y=200)
        _mock_pyautogui.click.assert_called()

    def test_mouse_click_right_button(self):
        cu.mouse_click(x=100, y=200, button="right")
        _mock_pyautogui.click.assert_called()

    def test_mouse_click_double_click(self):
        cu.mouse_click(x=100, y=200, clicks=2)
        _mock_pyautogui.click.assert_called()

    def test_mouse_double_click_function(self):
        result = cu.mouse_double_click(x=300, y=400)
        assert isinstance(result, dict)

    def test_mouse_right_click_function(self):
        result = cu.mouse_right_click(x=300, y=400)
        assert isinstance(result, dict)


# ── mouse_move ────────────────────────────────────────────────────────────────

class TestMouseMove:
    def test_mouse_move_returns_dict(self):
        result = cu.mouse_move(x=100, y=200)
        assert isinstance(result, dict)

    def test_mouse_move_calls_moveTo(self):
        cu.mouse_move(x=800, y=600)
        _mock_pyautogui.moveTo.assert_called()


# ── mouse_drag ────────────────────────────────────────────────────────────────

class TestMouseDrag:
    def test_mouse_drag_returns_dict(self):
        result = cu.mouse_drag(from_x=0, from_y=0, to_x=100, to_y=100)
        assert isinstance(result, dict)


# ── mouse_scroll ──────────────────────────────────────────────────────────────

class TestMouseScroll:
    def test_mouse_scroll_returns_dict(self):
        result = cu.mouse_scroll(x=500, y=300, clicks=3)
        assert isinstance(result, dict)

    def test_mouse_scroll_down(self):
        result = cu.mouse_scroll(x=500, y=300, clicks=3, direction="down")
        assert isinstance(result, dict)

    def test_mouse_scroll_calls_scroll(self):
        cu.mouse_scroll(x=500, y=300, clicks=2)
        _mock_pyautogui.scroll.assert_called()


# ── keyboard_type ─────────────────────────────────────────────────────────────

class TestKeyboardType:
    def test_keyboard_type_returns_dict(self):
        result = cu.keyboard_type(text="hello world")
        assert isinstance(result, dict)

    def test_keyboard_type_called(self):
        cu.keyboard_type(text="test string")
        # Either typewrite or write called
        called = _mock_pyautogui.typewrite.called or _mock_pyautogui.write.called
        assert called


# ── keyboard_hotkey ───────────────────────────────────────────────────────────

class TestKeyboardHotkey:
    def test_hotkey_returns_dict(self):
        result = cu.keyboard_hotkey(keys=["ctrl", "c"])
        assert isinstance(result, dict)

    def test_hotkey_calls_hotkey(self):
        cu.keyboard_hotkey(keys=["cmd", "v"])
        _mock_pyautogui.hotkey.assert_called()


# ── keyboard_press ────────────────────────────────────────────────────────────

class TestKeyboardPress:
    def test_press_returns_dict(self):
        result = cu.keyboard_press(key="enter")
        assert isinstance(result, dict)

    def test_press_calls_press(self):
        cu.keyboard_press(key="tab")
        _mock_pyautogui.press.assert_called()


# ── keyboard_hold ─────────────────────────────────────────────────────────────

class TestKeyboardHold:
    def test_hold_returns_dict(self):
        result = cu.keyboard_hold(key="shift", duration=0.05)
        assert isinstance(result, dict)

    def test_hold_calls_keyDown_keyUp(self):
        cu.keyboard_hold(key="ctrl", duration=0.05)
        _mock_pyautogui.keyDown.assert_called()
        _mock_pyautogui.keyUp.assert_called()


# ── get_screen_size / get_mouse_position ─────────────────────────────────────

class TestScreenInfo:
    def test_get_screen_size_returns_dict(self):
        result = cu.get_screen_size()
        assert isinstance(result, dict)

    def test_get_mouse_position_returns_dict(self):
        result = cu.get_mouse_position()
        assert isinstance(result, dict)


# ── clipboard ─────────────────────────────────────────────────────────────────

class TestClipboard:
    def test_clipboard_get_returns_dict(self):
        with patch("subprocess.check_output", return_value=b"clipboard content"):
            result = cu.clipboard_get()
        assert isinstance(result, dict)

    def test_clipboard_set_returns_dict(self):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            result = cu.clipboard_set(text="hello clipboard")
        assert isinstance(result, dict)


# ── list_windows ──────────────────────────────────────────────────────────────

class TestListWindows:
    def test_list_windows_returns_dict(self):
        with patch("subprocess.check_output", return_value=b"Terminal\nSafari\nFinder\n"):
            result = cu.list_windows()
        assert isinstance(result, dict)

    def test_list_windows_error_graceful(self):
        with patch("subprocess.check_output", side_effect=Exception("osascript error")):
            result = cu.list_windows()
        assert isinstance(result, dict)


# ── open_application ──────────────────────────────────────────────────────────

class TestOpenApplication:
    def test_open_app_returns_dict(self):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            result = cu.open_application(app_name="Terminal")
        assert isinstance(result, dict)

    def test_open_app_error_graceful(self):
        with patch("subprocess.run", side_effect=Exception("not found")):
            result = cu.open_application(app_name="NonExistentApp")
        assert isinstance(result, dict)


# ── find_on_screen ────────────────────────────────────────────────────────────

class TestFindOnScreen:
    def test_find_on_screen_returns_dict(self):
        _mock_pyautogui.locateOnScreen.return_value = None
        result = cu.find_on_screen(image_path="/fake/path.png")
        assert isinstance(result, dict)

    def test_find_on_screen_not_found(self):
        _mock_pyautogui.locateOnScreen.return_value = None
        result = cu.find_on_screen(image_path="/fake/button.png")
        assert "found" in result or "error" in result or "x" in result


# ── computer_use_status ───────────────────────────────────────────────────────

class TestComputerUseStatus:
    def test_status_returns_dict(self):
        result = cu.computer_use_status()
        assert isinstance(result, dict)

    def test_status_has_content(self):
        result = cu.computer_use_status()
        assert len(result) > 0


# ── ComputerUse class ─────────────────────────────────────────────────────────

class TestComputerUseClass:
    def test_screenshot_method_returns_dict(self):
        with patch.object(cu, "_take_screenshot", return_value="abc", create=True):
            result = cu.ComputerUse.screenshot()
        assert isinstance(result, dict)

    def test_mouse_click_method_returns_dict(self):
        result = cu.ComputerUse.mouse_click(x=100, y=200)
        assert isinstance(result, dict)

    def test_keyboard_type_method_returns_dict(self):
        result = cu.ComputerUse.keyboard_type(text="hello")
        assert isinstance(result, dict)

    def test_status_method_returns_dict(self):
        result = cu.ComputerUse.status()
        assert isinstance(result, dict)

    def test_open_app_method_returns_dict(self):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            result = cu.ComputerUse.open_app(app_name="Terminal")
        assert isinstance(result, dict)


# ── Dispatch callable tests ────────────────────────────────────────────────────

class TestDispatchCallable:
    def test_desktop_screenshot_dispatch(self):
        fn = cu._DISPATCH.get("desktop_screenshot")
        assert fn is not None
        # screenshot() uses mss/_PYAUTOGUI internally; just check it returns dict
        result = fn()
        assert isinstance(result, dict)

    def test_desktop_mouse_click_dispatch(self):
        fn = cu._DISPATCH.get("desktop_mouse_click")
        assert fn is not None
        result = fn(x=100, y=200)
        assert isinstance(result, dict)

    def test_desktop_keyboard_type_dispatch(self):
        fn = cu._DISPATCH.get("desktop_keyboard_type")
        assert fn is not None
        result = fn(text="hello from dispatch")
        assert isinstance(result, dict)

    def test_desktop_status_dispatch(self):
        fn = cu._DISPATCH.get("desktop_status")
        assert fn is not None
        result = fn()
        assert isinstance(result, dict)

    def test_desktop_clipboard_get_dispatch(self):
        fn = cu._DISPATCH.get("desktop_clipboard_get")
        assert fn is not None
        with patch("subprocess.check_output", return_value=b"clipboard text"):
            result = fn()
        assert isinstance(result, dict)
