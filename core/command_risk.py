"""
Operon Command Risk Analysis.

Adapted from OpenClaw src/infra/command-analysis/risks.ts.

Analyses shell commands before execution and classifies them by risk level.
Detects:
  • Interpreter eval / command injection  (bash -c "...", python -c "...")
  • Carrier chain attacks (|| and && chains with injected payloads)
  • Shell wrapper payloads (sub-shells, process substitution)
  • Data exfiltration patterns (curl | sh, wget | bash, etc.)
  • Privilege escalation (sudo, su, chmod +s, setuid)
  • File system destruction (rm -rf /, truncate, dd if=/dev/zero)
  • Network bind-shells (nc -l, ncat --listen)
  • Sensitive file access (reading /etc/passwd, SSH keys, etc.)
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ── Risk levels ────────────────────────────────────────────────────────────────

class RiskLevel(IntEnum):
    SAFE     = 0   # routine read/list operations
    LOW      = 1   # potentially noteworthy but harmless
    MEDIUM   = 2   # may have side effects; flag for review
    HIGH     = 3   # dangerous; should require confirmation
    CRITICAL = 4   # should be blocked outright by default


# ── Risk finding dataclass ─────────────────────────────────────────────────────

@dataclass
class RiskFinding:
    rule:        str
    level:       RiskLevel
    description: str
    snippet:     str = ""


@dataclass
class CommandRiskResult:
    command:  str
    level:    RiskLevel
    findings: list[RiskFinding] = field(default_factory=list)
    blocked:  bool = False
    reason:   str  = ""

    @property
    def safe(self) -> bool:
        return not self.blocked and self.level < RiskLevel.HIGH

    def summary(self) -> str:
        if not self.findings:
            return f"[{self.level.name}] OK"
        lines = [f"[{self.level.name}]"]
        for f_ in self.findings[:3]:
            lines.append(f"  • {f_.rule}: {f_.description}")
        if len(self.findings) > 3:
            lines.append(f"  … and {len(self.findings) - 3} more")
        return "\n".join(lines)


# ── Rule definitions ───────────────────────────────────────────────────────────

@dataclass
class _Rule:
    name:     str
    level:    RiskLevel
    pattern:  re.Pattern
    message:  str


def _r(name: str, level: RiskLevel, pattern: str, message: str) -> _Rule:
    return _Rule(name, level, re.compile(pattern, re.IGNORECASE | re.MULTILINE), message)


_RULES: list[_Rule] = [
    # ── Critical ──────────────────────────────────────────────────────────────
    _r("fs_destroy_root",    RiskLevel.CRITICAL,
       r"\brm\s+(?:-[rfRF]+\s+)?(?:/|\$HOME|~)\s*$|rm\s+-[rfRF]+\s+/",
       "Recursive deletion of root or home directory"),
    _r("dd_overwrite",       RiskLevel.CRITICAL,
       r"\bdd\b.*\bof=/dev/(?:sda|hda|nvme|disk|null)\b",
       "dd writing to block device — potential drive wipe"),
    _r("fork_bomb",          RiskLevel.CRITICAL,
       r":\(\)\s*\{.*:\|:&\s*\}|:\s*\(\s*\)\s*\{",
       "Fork bomb pattern detected"),
    _r("bind_shell",         RiskLevel.CRITICAL,
       r"\b(?:nc|ncat|netcat)\b.*(?:-l|-lp|--listen).*\b(?:bash|sh|/bin/sh)\b"
       r"|\b(?:bash|sh)\b.*-i\s*[>&]+\s*/dev/tcp",
       "Reverse/bind shell detected"),
    _r("pipe_to_shell",      RiskLevel.CRITICAL,
       r"(?:curl|wget|fetch)\s+['\"]?https?://[^\s]+['\"]?\s*\|+\s*(?:bash|sh|zsh|fish|python|ruby|perl|node)",
       "Pipe-from-internet-to-shell — remote code execution"),

    # ── High ──────────────────────────────────────────────────────────────────
    _r("interpreter_eval",   RiskLevel.HIGH,
       r'\b(?:bash|sh|zsh|fish)\s+-c\s+["\']'
       r'|python(?:3)?\s+-c\s+["\']'
       r'|ruby\s+-e\s+["\']'
       r'|perl\s+-e\s+["\']'
       r'|node\s+-e\s+["\']'
       r'|php\s+-r\s+["\']',
       "Interpreter eval — inline code execution"),
    _r("privesc_sudo",       RiskLevel.HIGH,
       r"\bsudo\s+(?:-[Sisu]+\s+)?(?:bash|sh|su|visudo|passwd|chsh)",
       "Privilege escalation via sudo to shell/passwd"),
    _r("setuid_chmod",       RiskLevel.HIGH,
       r"\bchmod\s+(?:[0-9]*[46][0-9][0-9]|[ugo]+[=+]s)\b"
       r"|\bchown\s+root\b",
       "Setting setuid/setgid bit or chown to root"),
    _r("cron_install",       RiskLevel.HIGH,
       r"\b(?:crontab\s+-[ei]|echo\s+.*>>\s*/etc/cron)",
       "Installing cron job"),
    _r("process_subst",      RiskLevel.HIGH,
       r"<\(|>\(",
       "Process substitution — potential shell execution"),
    _r("sensitive_read",     RiskLevel.HIGH,
       r"(?:cat|less|head|tail|cp|scp|rsync)\s+.*(?:/etc/shadow|/etc/passwd|"
       r"\.ssh/(?:id_rsa|id_ed25519|authorized_keys)|"
       r"\.aws/credentials|\.gnupg/secring)",
       "Reading sensitive credential/key file"),

    # ── Medium ────────────────────────────────────────────────────────────────
    _r("carrier_chain_or",   RiskLevel.MEDIUM,
       r"[^\|]\|\|[^\|]",
       "OR carrier chain (cmd1 || cmd2) — may execute on failure"),
    _r("carrier_chain_semi", RiskLevel.MEDIUM,
       r";\s*(?:rm|curl|wget|sudo|bash|python)\b",
       "Semicolon-chained dangerous command"),
    _r("env_overwrite",      RiskLevel.MEDIUM,
       r"(?:export|set)\s+(?:PATH|LD_PRELOAD|LD_LIBRARY_PATH|PYTHONPATH)\s*=",
       "Overwriting critical environment variable"),
    _r("sys_file_write",     RiskLevel.MEDIUM,
       r"(?:echo|tee|printf|cat)\s+.*>\s*/(?:etc|usr|lib|bin|sbin)/",
       "Writing to system directory"),
    _r("network_scan",       RiskLevel.MEDIUM,
       r"\b(?:nmap|masscan|zmap)\b",
       "Network scanning tool"),
    _r("history_clear",      RiskLevel.MEDIUM,
       r"\bhistory\s+-[cCw]\b|\bunset\s+HISTFILE\b",
       "Clearing shell history — potential evidence removal"),

    # ── Low ───────────────────────────────────────────────────────────────────
    _r("eval_var",           RiskLevel.LOW,
       r"\beval\s+[\$\`]",
       "eval with variable/subshell"),
    _r("wget_download",      RiskLevel.LOW,
       r"\b(?:wget|curl)\s+['\"]?https?://",
       "Downloading file from internet"),
    _r("background_job",     RiskLevel.LOW,
       r"&\s*$",
       "Command runs in background — output hidden from agent"),
    _r("redirect_stderr",    RiskLevel.LOW,
       r"2>/dev/null|2>&1\s*>\s*/dev/null",
       "Stderr suppressed — errors will be invisible"),
]

# Rules that should be blocked outright (never executed)
_BLOCK_LEVELS = frozenset({RiskLevel.CRITICAL})
# Rules that require confirmation before execution
_CONFIRM_LEVELS = frozenset({RiskLevel.HIGH})


# ── Analyser ───────────────────────────────────────────────────────────────────

def analyse_command(
    command:         str,
    block_critical:  bool = True,
    warn_high:       bool = True,
) -> CommandRiskResult:
    """
    Analyse a shell command and return a CommandRiskResult.

    Parameters
    ----------
    command         : the shell command string to analyse
    block_critical  : if True, CRITICAL-level commands are marked blocked=True
    warn_high       : if True, HIGH-level commands are included in findings

    Returns a CommandRiskResult with the highest risk level found, all
    matching findings, and whether the command should be blocked.
    """
    if not command or not command.strip():
        return CommandRiskResult(command=command, level=RiskLevel.SAFE)

    findings: list[RiskFinding] = []
    max_level = RiskLevel.SAFE

    # Try to expand aliases / $(...) for scanning
    check_cmd = _expand_for_scan(command)

    for rule in _RULES:
        m = rule.pattern.search(check_cmd)
        if m:
            snippet = m.group(0)[:80]
            findings.append(RiskFinding(
                rule=rule.name,
                level=rule.level,
                description=rule.message,
                snippet=snippet,
            ))
            if rule.level > max_level:
                max_level = rule.level

    blocked = False
    reason  = ""
    if block_critical and max_level in _BLOCK_LEVELS:
        blocked = True
        critical_findings = [f for f in findings if f.level == RiskLevel.CRITICAL]
        reason = "; ".join(f.rule + ": " + f.description for f in critical_findings)

    return CommandRiskResult(
        command=command,
        level=max_level,
        findings=findings,
        blocked=blocked,
        reason=reason,
    )


def _expand_for_scan(command: str) -> str:
    """
    Lightly expand a command string for scanning without executing it.
    Splits arg0 from its arguments so patterns match embedded payloads.
    """
    # Unescape common backslash sequences for scanning only
    expanded = command
    expanded = re.sub(r"\\n", "\n", expanded)
    expanded = re.sub(r"\\t", "\t", expanded)
    # Decode obvious base64-looking blobs for scanning
    # (only if they appear to be passed as arguments)
    b64_match = re.search(
        r'(?:echo|printf)\s+["\']?([A-Za-z0-9+/]{30,}={0,2})["\']?\s*\|\s*base64\s+-[dD]',
        expanded,
    )
    if b64_match:
        import base64
        try:
            decoded = base64.b64decode(b64_match.group(1)).decode("utf-8", errors="replace")
            expanded += " " + decoded
        except Exception:
            pass
    return expanded


# ── Convenience helpers ────────────────────────────────────────────────────────

def is_safe(command: str) -> bool:
    """Return True if the command is below HIGH risk and not blocked."""
    return analyse_command(command).safe


def risk_summary(command: str) -> str:
    """Return a human-readable risk summary for a command."""
    return analyse_command(command).summary()
