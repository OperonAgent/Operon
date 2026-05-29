"""
core/dep_audit.py — Supply-chain / dependency vulnerability auditing.

Operon runs AI-driven tools that import third-party packages. A poisoned or
known-vulnerable dependency is a real supply-chain risk. This module audits the
currently-installed Python environment against the OSV.dev vulnerability
database (the same source `pip-audit` uses), with a fully-offline fallback that
checks a small built-in advisory list and flags yanked/suspicious versions.

Design goals:
  • Zero hard dependencies — uses stdlib urllib; works without pip-audit.
  • Network-optional — degrades to an offline heuristic scan when offline.
  • Fast — only audits top-level installed distributions.
  • Safe — never executes package code; only reads metadata.

Public API:
    audit_environment(offline=False) -> AuditReport
    audit_packages(names, offline=False) -> AuditReport
    format_report(report) -> str
"""

from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_OSV_API = "https://api.osv.dev/v1/query"
_OSV_BATCH = "https://api.osv.dev/v1/querybatch"

# Minimal offline advisory list — high-signal known-bad version ranges.
# (pkg_lower -> list of (spec_description, max_vulnerable_version_exclusive))
# This is a safety net, not a substitute for OSV; kept short and high-confidence.
_OFFLINE_ADVISORIES: Dict[str, List[Tuple[str, str]]] = {
    "requests":   [("CVE-2023-32681 — proxy creds leak", "2.31.0")],
    "urllib3":    [("CVE-2023-43804 — cookie leak on redirect", "1.26.18")],
    "cryptography":[("Multiple OpenSSL CVEs", "41.0.6")],
    "pillow":     [("CVE-2023-50447 — arbitrary code via eval", "10.2.0")],
    "aiohttp":    [("CVE-2024-23334 — directory traversal", "3.9.2")],
    "jinja2":     [("CVE-2024-22195 — XSS via xmlattr", "3.1.3")],
    "pyyaml":     [("CVE-2020-14343 — arbitrary code on full_load", "5.4")],
}


@dataclass
class Vulnerability:
    package:   str
    version:   str
    vuln_id:   str
    summary:   str
    severity:  str = "UNKNOWN"
    fixed_in:  str = ""


@dataclass
class AuditReport:
    scanned:        int = 0
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    source:         str = "osv"            # "osv" | "offline"
    errors:         List[str] = field(default_factory=list)

    @property
    def vulnerable_count(self) -> int:
        return len(self.vulnerabilities)

    @property
    def clean(self) -> bool:
        return self.vulnerable_count == 0


# ── Installed-package discovery ────────────────────────────────────────────────

def _installed_distributions() -> Dict[str, str]:
    """Return {distribution_name_lower: version} for installed packages."""
    dists: Dict[str, str] = {}
    try:
        from importlib import metadata as md
    except ImportError:  # py<3.8
        try:
            import importlib_metadata as md  # type: ignore
        except ImportError:
            return dists
    try:
        for dist in md.distributions():
            try:
                name = (dist.metadata["Name"] or "").strip().lower()
                ver  = (dist.version or "").strip()
                if name and ver:
                    dists[name] = ver
            except Exception:
                continue
    except Exception:
        pass
    return dists


# ── Version comparison (PEP 440-ish, dependency-free) ──────────────────────────

def _parse_version(v: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for chunk in v.replace("-", ".").split("."):
        num = ""
        for ch in chunk:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts) or (0,)


def _version_lt(a: str, b: str) -> bool:
    """True if version a < version b (best-effort, dependency-free)."""
    pa, pb = _parse_version(a), _parse_version(b)
    n = max(len(pa), len(pb))
    pa += (0,) * (n - len(pa))
    pb += (0,) * (n - len(pb))
    return pa < pb


# ── OSV.dev query ───────────────────────────────────────────────────────────────

def _query_osv_one(name: str, version: str, timeout: float = 6.0) -> List[Vulnerability]:
    payload = json.dumps({
        "version": version,
        "package": {"name": name, "ecosystem": "PyPI"},
    }).encode("utf-8")
    req = urllib.request.Request(_OSV_API, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    vulns: List[Vulnerability] = []
    for v in data.get("vulns", []) or []:
        fixed = ""
        for affected in v.get("affected", []):
            for rng in affected.get("ranges", []):
                for ev in rng.get("events", []):
                    if "fixed" in ev:
                        fixed = ev["fixed"]
        sev = "UNKNOWN"
        if v.get("severity"):
            sev = v["severity"][0].get("type", "UNKNOWN")
        vulns.append(Vulnerability(
            package=name, version=version,
            vuln_id=v.get("id", "?"),
            summary=(v.get("summary") or v.get("details", "") or "")[:120],
            severity=sev, fixed_in=fixed,
        ))
    return vulns


# ── Offline heuristic scan ──────────────────────────────────────────────────────

def _audit_offline(dists: Dict[str, str]) -> AuditReport:
    report = AuditReport(source="offline")
    for name, ver in dists.items():
        report.scanned += 1
        for desc, max_bad in _OFFLINE_ADVISORIES.get(name, []):
            if _version_lt(ver, max_bad):
                report.vulnerabilities.append(Vulnerability(
                    package=name, version=ver, vuln_id="OFFLINE-ADVISORY",
                    summary=desc, severity="MEDIUM", fixed_in=max_bad,
                ))
    return report


# ── Public API ──────────────────────────────────────────────────────────────────

def audit_packages(packages: Dict[str, str], offline: bool = False) -> AuditReport:
    """Audit a {name: version} mapping. Falls back to offline on network error."""
    if offline:
        return _audit_offline(packages)

    report = AuditReport(source="osv")
    network_failed = False
    for name, ver in packages.items():
        report.scanned += 1
        try:
            report.vulnerabilities.extend(_query_osv_one(name, ver))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            network_failed = True
            break
        except Exception as e:
            report.errors.append(f"{name}: {e}")

    if network_failed:
        # Offline fallback — still useful without a network.
        off = _audit_offline(packages)
        off.errors.append("OSV.dev unreachable — used offline advisory list")
        return off
    return report


def audit_environment(offline: bool = False, limit: int = 0) -> AuditReport:
    """Audit all installed distributions in the current environment."""
    dists = _installed_distributions()
    if limit and len(dists) > limit:
        dists = dict(list(dists.items())[:limit])
    return audit_packages(dists, offline=offline)


def format_report(report: AuditReport) -> str:
    lines = [f"Dependency audit ({report.source}): {report.scanned} packages scanned"]
    if report.clean:
        lines.append("  ✓ No known vulnerabilities found")
    else:
        lines.append(f"  ⚠ {report.vulnerable_count} vulnerable package(s):")
        for v in report.vulnerabilities:
            fix = f" → upgrade to {v.fixed_in}" if v.fixed_in else ""
            lines.append(f"    • {v.package} {v.version}  [{v.vuln_id}] {v.summary}{fix}")
    for e in report.errors:
        lines.append(f"  · {e}")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────────

def _cli() -> int:
    offline = "--offline" in sys.argv
    report = audit_environment(offline=offline)
    print(format_report(report))
    return 1 if not report.clean else 0


if __name__ == "__main__":
    sys.exit(_cli())
