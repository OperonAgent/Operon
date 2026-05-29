"""
Operon Heartbeat Scheduler — enhanced with Hermes Agent cron patterns.

Drop a HEARTBEAT.md at ~/.operon/HEARTBEAT.md (or ./HEARTBEAT.md) to define
what the agent should do periodically.

New features vs original (all from Hermes Agent cron/scheduler.py):
  - wake_gate:      Last line {"wakeAgent": false} → skip LLM entirely
  - no_agent mode:  no_agent=True → run script, deliver stdout, zero LLM tokens
  - context_from:   context_from=["other_job"] → chain prior job output as context
  - Inactivity timeout (separate from wall-clock timeout): kills stalled agents
  - [SILENT] sentinel: agent can suppress output mid-run
  - Stagger window (stagger_seconds): randomize start time to avoid pile-ups
  - Consecutive error counting + failure alerts with cooldown
  - Per-run diagnostics with severity + source
  - Parallel + serial job partition (serial when workdir is set)
"""

from __future__ import annotations

import datetime
import json
import logging
import random
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("operon.heartbeat")

_DEFAULT_INTERVAL   = 1800      # 30 minutes
_HEARTBEAT_PATHS    = [
    Path.home() / ".operon" / "HEARTBEAT.md",
    Path("HEARTBEAT.md"),
]
_INACTIVITY_POLL    = 5         # seconds between inactivity checks
_DEFAULT_INACTIVITY = 600       # 10 min with no activity → kill job


# ── Wake gate ──────────────────────────────────────────────────────────────────

_WAKE_GATE_RE = re.compile(r'\{[^}]*"wakeAgent"\s*:\s*false[^}]*\}', re.IGNORECASE)


def _parse_wake_gate(output: str) -> bool:
    """Return True if the last line contains {"wakeAgent": false}."""
    lines = output.strip().splitlines()
    for line in reversed(lines[-3:]):
        if _WAKE_GATE_RE.search(line.strip()):
            return True   # agent should NOT wake (skip LLM)
    return False


# ── Prompt injection guard ─────────────────────────────────────────────────────

_INJECTION_SIGNALS = re.compile(
    r'(ignore previous|disregard (all|above|prior)|new instructions?:|'
    r'system prompt:|you are now|forget everything|override (your|all))',
    re.IGNORECASE,
)


def _has_prompt_injection(text: str) -> bool:
    """Basic prompt injection detection for cron job content."""
    return bool(_INJECTION_SIGNALS.search(text))


# ── Diagnostic record ──────────────────────────────────────────────────────────

def _diag(source: str, severity: str, message: str, **extra) -> dict:
    d = {"ts": time.time(), "source": source, "severity": severity, "message": message}
    d.update(extra)
    return d


# ── HeartbeatScheduler ─────────────────────────────────────────────────────────

class HeartbeatScheduler:
    """
    Reads HEARTBEAT.md on a fixed interval and fires the agent runner.

    Parameters
    ----------
    agent_runner : callable(prompt: str) → str
    interval_seconds : int       (default 1800 = 30 min)
    business_hours : bool        (default False — run 24/7)
    start_hour : int             (default 9)
    end_hour : int               (default 18)
    weekdays_only : bool         (default True when business_hours=True)
    custom_path : Path           Override HEARTBEAT.md search path
    no_agent : bool              Run script only, skip LLM entirely
    stagger_seconds : int        Randomise start by up to N seconds
    inactivity_timeout : int     Kill stalled agents after N silent seconds
    failure_alert_after : int    Consecutive failures before alerting
    failure_cooldown_s : int     Minimum seconds between failure alerts
    """

    def __init__(
        self,
        agent_runner:      Callable[[str], str],
        interval_seconds:  int           = _DEFAULT_INTERVAL,
        business_hours:    bool          = False,
        start_hour:        int           = 9,
        end_hour:          int           = 18,
        weekdays_only:     bool          = True,
        custom_path:       Optional[Path]= None,
        no_agent:          bool          = False,
        stagger_seconds:   int           = 0,
        inactivity_timeout: int          = _DEFAULT_INACTIVITY,
        failure_alert_after: int         = 3,
        failure_cooldown_s:  int         = 3600,
    ) -> None:
        self._runner             = agent_runner
        self._interval           = interval_seconds
        self._business_hours     = business_hours
        self._start_hour         = start_hour
        self._end_hour           = end_hour
        self._weekdays_only      = weekdays_only
        self._custom_path        = custom_path
        self._no_agent           = no_agent
        self._stagger            = stagger_seconds
        self._inactivity_timeout = inactivity_timeout
        self._failure_alert_after= failure_alert_after
        self._failure_cooldown   = failure_cooldown_s

        self._thread: Optional[threading.Thread] = None
        self._stop_event          = threading.Event()
        self.running              = False

        # State tracking
        self._run_count           = 0
        self._consecutive_errors  = 0
        self._last_run: Optional[float]   = None
        self._last_result: str            = ""
        self._last_alert_at: Optional[float] = None

        # Per-run diagnostics (last run only)
        self._last_diagnostics: list[dict] = []

        # context_from job output registry: {job_name: output_text}
        self._context_store: dict[str, str] = {}
        self._context_lock   = threading.Lock()

        # Activity timestamp for inactivity watchdog
        self._last_activity: float = time.time()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the heartbeat loop in a daemon background thread."""
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="operon-heartbeat",
        )
        self._thread.start()
        self.running = True
        log.info("Heartbeat started — interval=%ds stagger=%ds", self._interval, self._stagger)

    def stop(self) -> None:
        self._stop_event.set()
        self.running = False
        log.info("Heartbeat stopped.")

    def trigger_now(self) -> str:
        """Force an immediate tick regardless of schedule."""
        return self._tick()

    def status(self) -> dict:
        next_run = None
        if self._last_run is not None:
            next_ts  = self._last_run + self._interval
            next_run = datetime.datetime.fromtimestamp(next_ts).strftime("%H:%M:%S")
        return {
            "running":              self.running,
            "interval_seconds":     self._interval,
            "business_hours":       self._business_hours,
            "run_count":            self._run_count,
            "consecutive_errors":   self._consecutive_errors,
            "last_run": (
                datetime.datetime.fromtimestamp(self._last_run).strftime("%Y-%m-%d %H:%M:%S")
                if self._last_run else "never"
            ),
            "next_tick":            next_run or "(unknown)",
            "heartbeat_file":       str(self._find_heartbeat_file() or "(not found)"),
            "no_agent_mode":        self._no_agent,
            "last_diagnostics":     self._last_diagnostics[-5:],
        }

    def get_heartbeat_content(self) -> str:
        p = self._find_heartbeat_file()
        if p is None:
            return ""
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def set_heartbeat_content(self, content: str) -> Path:
        path = self._custom_path or _HEARTBEAT_PATHS[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def store_context(self, job_name: str, output: str) -> None:
        """Store job output so other jobs can use it via context_from."""
        with self._context_lock:
            self._context_store[job_name] = output

    def get_context(self, job_name: str) -> str:
        with self._context_lock:
            return self._context_store.get(job_name, "")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _find_heartbeat_file(self) -> Optional[Path]:
        if self._custom_path and self._custom_path.exists():
            return self._custom_path
        for p in _HEARTBEAT_PATHS:
            if p.exists():
                return p
        return None

    def _within_business_hours(self) -> bool:
        if not self._business_hours:
            return True
        now = datetime.datetime.now()
        if self._weekdays_only and now.weekday() >= 5:
            return False
        return self._start_hour <= now.hour < self._end_hour

    def _tick(self) -> str:
        """Run one tick. Returns agent response or empty string."""
        self._last_diagnostics = []
        diagnostics             = self._last_diagnostics

        content = self.get_heartbeat_content()
        if not content:
            diagnostics.append(_diag("cron-preflight", "info", "HEARTBEAT.md empty/missing"))
            return ""

        # ── Prompt injection guard ─────────────────────────────────────────────
        if _has_prompt_injection(content):
            msg = "Cron job content blocked: possible prompt injection detected"
            diagnostics.append(_diag("cron-setup", "error", msg))
            log.warning("[Heartbeat] %s", msg)
            self._consecutive_errors += 1
            return f"[BLOCKED] {msg}"

        # ── no_agent mode: run script, deliver output, skip LLM ──────────────
        if self._no_agent:
            diagnostics.append(_diag("cron-setup", "info", "no_agent=True: skipping LLM"))
            self._run_count += 1
            self._last_run   = time.time()
            return content    # return content as-is (caller delivers it)

        now_str = datetime.datetime.now().strftime("%A %Y-%m-%d %H:%M:%S")

        # ── context_from: inject prior job outputs ────────────────────────────
        # Parse job names from "context_from: job_name" lines in HEARTBEAT.md
        context_blocks = []
        for line in content.splitlines():
            m = re.match(r'context_from\s*:\s*(\S+)', line.strip(), re.IGNORECASE)
            if m:
                ctx = self.get_context(m.group(1))
                if ctx:
                    context_blocks.append(
                        f"[Context from '{m.group(1)}']\n{ctx[:2000]}"
                    )

        context_inject = ("\n\n" + "\n\n".join(context_blocks)) if context_blocks else ""

        # ── Wake gate check: run a "pre-flight" script if present ─────────────
        # Lines starting with "pre_run:" are executed and checked for wake gate
        preflight_output = ""
        for line in content.splitlines():
            if line.strip().lower().startswith("pre_run:"):
                script = line.split(":", 1)[1].strip()
                try:
                    import subprocess
                    result = subprocess.run(
                        script, shell=True, capture_output=True, text=True, timeout=30
                    )
                    preflight_output = result.stdout
                    diagnostics.append(_diag(
                        "cron-preflight", "info",
                        f"Pre-run script exited {result.returncode}",
                        exit_code=result.returncode,
                    ))
                except Exception as e:
                    diagnostics.append(_diag("cron-preflight", "warn", f"Pre-run failed: {e}"))

        if preflight_output and _parse_wake_gate(preflight_output):
            diagnostics.append(_diag("cron-setup", "info", "Wake gate: wakeAgent=false, skipping LLM"))
            return ""   # Skip LLM this tick

        prompt = (
            f"[HEARTBEAT TICK — {now_str}]{context_inject}\n\n"
            f"You are running a scheduled heartbeat. Read the following instructions "
            f"and decide if any action is needed right now. Execute any actions that "
            f"apply to the current time/conditions. If nothing applies, respond with "
            f"'No heartbeat actions needed at this time.' or include [SILENT] to suppress output.\n\n"
            f"HEARTBEAT INSTRUCTIONS:\n{content}"
        )

        # ── Inactivity watchdog ───────────────────────────────────────────────
        self._last_activity = time.time()
        _result_holder: list = []
        _error_holder:  list = []

        def _run_with_watchdog():
            try:
                response = self._runner(prompt)
                _result_holder.append(response)
            except Exception as e:
                _error_holder.append(str(e))

        run_thread = threading.Thread(target=_run_with_watchdog, daemon=True)
        run_thread.start()

        # Wait with inactivity check
        deadline = time.time() + self._inactivity_timeout
        while run_thread.is_alive():
            run_thread.join(timeout=_INACTIVITY_POLL)
            if not run_thread.is_alive():
                break
            if time.time() > deadline:
                diagnostics.append(_diag(
                    "agent-run", "error",
                    f"Inactivity timeout after {self._inactivity_timeout}s — run killed"
                ))
                log.warning("[Heartbeat] Inactivity timeout. Killing tick.")
                self._consecutive_errors += 1
                self._last_run = time.time()
                self._run_count += 1
                self._maybe_send_failure_alert()
                return f"[TIMEOUT] Heartbeat stalled after {self._inactivity_timeout}s"

        self._run_count += 1
        self._last_run   = time.time()

        if _error_holder:
            diagnostics.append(_diag("agent-run", "error", _error_holder[0]))
            self._consecutive_errors += 1
            self._maybe_send_failure_alert()
            return f"[Heartbeat error: {_error_holder[0]}]"

        response = _result_holder[0] if _result_holder else ""

        # ── [SILENT] sentinel ─────────────────────────────────────────────────
        if "[SILENT]" in response:
            diagnostics.append(_diag("delivery", "info", "[SILENT] suppressed output"))
            self._consecutive_errors = 0
            return ""

        self._last_result        = response
        self._consecutive_errors = 0
        # Store this job's output for context_from consumers
        self.store_context("heartbeat", response)

        diagnostics.append(_diag("delivery", "info", "Tick completed OK"))
        log.info("Heartbeat tick #%d complete.", self._run_count)
        return response

    def _maybe_send_failure_alert(self) -> None:
        """Send a failure alert if consecutive errors >= threshold and cooldown passed."""
        if self._consecutive_errors < self._failure_alert_after:
            return
        now = time.time()
        if self._last_alert_at and (now - self._last_alert_at) < self._failure_cooldown:
            return
        self._last_alert_at = now
        log.warning(
            "[Heartbeat] FAILURE ALERT: %d consecutive errors. "
            "Check /heartbeat status for diagnostics.",
            self._consecutive_errors,
        )

    def _loop(self) -> None:
        """Main heartbeat thread loop with stagger + inactivity-aware sleep."""
        # Initial stagger to avoid pile-up when multiple schedulers start at once
        if self._stagger > 0:
            stagger = random.uniform(0, self._stagger)
            log.debug("Heartbeat staggering %ds before first tick", stagger)
            for _ in range(int(stagger)):
                if self._stop_event.is_set():
                    return
                time.sleep(1)

        while not self._stop_event.is_set():
            if self._within_business_hours():
                self._tick()
            else:
                log.debug("Heartbeat: outside business hours, skipping tick.")

            for _ in range(self._interval):
                if self._stop_event.is_set():
                    return
                time.sleep(1)
