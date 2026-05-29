"""Tests for tools/shell_exec.py
Result shape: {success, stdout, stderr, returncode, truncated, risk_level}
"""
import os
import sys
import time
import pytest

from tools.shell_exec import shell_exec


class TestShellExecBasic:
    def test_echo_stdout(self):
        r = shell_exec("echo hello")
        assert r["success"]
        assert "hello" in r["stdout"]

    def test_zero_returncode_on_success(self):
        r = shell_exec("true")
        assert r["returncode"] == 0

    def test_nonzero_returncode_on_failure(self):
        r = shell_exec("false")
        assert r["returncode"] != 0
        assert not r["success"]

    def test_stderr_captured(self):
        r = shell_exec("echo err >&2")
        assert "err" in r["stderr"]

    def test_multiline_output(self):
        r = shell_exec("printf 'line1\\nline2\\nline3\\n'")
        assert r["success"]
        lines = r["stdout"].strip().split("\n")
        assert len(lines) == 3

    def test_result_keys_present(self):
        r = shell_exec("echo hi")
        for key in ("success", "stdout", "stderr", "returncode"):
            assert key in r, f"Missing key: {key}"


class TestShellExecEnvironment:
    def test_cwd_respected(self, tmp_path):
        r = shell_exec("pwd", cwd=str(tmp_path))
        assert r["success"]
        assert str(tmp_path.resolve()) in r["stdout"].strip()

    def test_invalid_cwd_error(self):
        r = shell_exec("echo hi", cwd="/no/such/directory")
        assert not r["success"]

    def test_env_variable_readable(self):
        r = shell_exec("echo $HOME")
        assert r["success"]
        assert r["stdout"].strip() != ""


class TestShellExecTimeout:
    def test_timeout_kills_process(self):
        start = time.time()
        r = shell_exec("sleep 10", timeout=1)
        elapsed = time.time() - start
        assert elapsed < 5
        assert not r["success"] or r["returncode"] != 0


class TestShellExecComposition:
    def test_pipe_commands(self):
        r = shell_exec("echo 'alpha beta gamma' | wc -w")
        assert r["success"]
        assert "3" in r["stdout"]

    def test_semicolon_chaining(self):
        r = shell_exec("echo first; echo second")
        assert r["success"]
        assert "first" in r["stdout"]
        assert "second" in r["stdout"]

    def test_python_subprocess(self):
        # inline -c code is flagged HIGH risk by shell_exec; use allow_high_risk
        r = shell_exec(f"{sys.executable} -c \"print('py_ok')\"",
                       allow_high_risk=True)
        assert r["success"]
        assert "py_ok" in r["stdout"]

    def test_write_and_read_file(self, tmp_path):
        p = str(tmp_path / "out.txt")
        r = shell_exec(f"echo written > {p}")
        assert r["success"]
        r2 = shell_exec(f"cat {p}")
        assert "written" in r2["stdout"]


class TestShellExecErrors:
    def test_nonexistent_command(self):
        r = shell_exec("this_command_xyz_does_not_exist")
        assert not r["success"]
        assert r["returncode"] != 0

    def test_empty_command(self):
        r = shell_exec("")
        assert isinstance(r, dict)

    def test_syntax_error_in_shell(self):
        r = shell_exec("if; then; fi")
        assert not r["success"] or r["returncode"] != 0
