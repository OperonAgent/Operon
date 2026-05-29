"""tests/test_dep_audit.py — supply-chain dependency auditing."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import dep_audit


class TestVersionCompare:
    def test_parse_simple(self):
        assert dep_audit._parse_version("1.2.3") == (1, 2, 3)

    def test_parse_with_suffix(self):
        assert dep_audit._parse_version("2.31.0")[:3] == (2, 31, 0)

    def test_lt_true(self):
        assert dep_audit._version_lt("2.30.0", "2.31.0") is True

    def test_lt_false_equal(self):
        assert dep_audit._version_lt("2.31.0", "2.31.0") is False

    def test_lt_false_greater(self):
        assert dep_audit._version_lt("3.0.0", "2.31.0") is False

    def test_lt_different_lengths(self):
        assert dep_audit._version_lt("1.26", "1.26.18") is True


class TestOfflineAudit:
    def test_flags_old_requests(self):
        rep = dep_audit.audit_packages({"requests": "2.30.0"}, offline=True)
        assert not rep.clean
        assert rep.vulnerabilities[0].package == "requests"

    def test_clean_for_new_requests(self):
        rep = dep_audit.audit_packages({"requests": "2.32.0"}, offline=True)
        assert rep.clean

    def test_unknown_package_clean(self):
        rep = dep_audit.audit_packages({"some_random_pkg": "1.0.0"}, offline=True)
        assert rep.clean

    def test_scanned_count(self):
        rep = dep_audit.audit_packages({"a": "1.0", "b": "2.0"}, offline=True)
        assert rep.scanned == 2

    def test_source_is_offline(self):
        rep = dep_audit.audit_packages({"a": "1.0"}, offline=True)
        assert rep.source == "offline"

    def test_multiple_vulns(self):
        rep = dep_audit.audit_packages(
            {"requests": "2.0.0", "pyyaml": "5.0", "pillow": "9.0.0"}, offline=True)
        assert rep.vulnerable_count >= 3


class TestOSVFallback:
    def test_network_error_falls_back_offline(self):
        import urllib.error
        with patch("core.dep_audit._query_osv_one",
                   side_effect=urllib.error.URLError("no net")):
            rep = dep_audit.audit_packages({"requests": "2.0.0"}, offline=False)
        assert rep.source == "offline"
        assert any("OSV.dev unreachable" in e for e in rep.errors)

    def test_osv_success_path(self):
        fake_vuln = dep_audit.Vulnerability(
            package="requests", version="2.0.0", vuln_id="GHSA-xxx",
            summary="test", severity="HIGH", fixed_in="2.31.0")
        with patch("core.dep_audit._query_osv_one", return_value=[fake_vuln]):
            rep = dep_audit.audit_packages({"requests": "2.0.0"}, offline=False)
        assert rep.source == "osv"
        assert rep.vulnerable_count == 1


class TestReport:
    def test_format_clean(self):
        rep = dep_audit.AuditReport(scanned=5, source="offline")
        out = dep_audit.format_report(rep)
        assert "No known vulnerabilities" in out

    def test_format_with_vulns(self):
        rep = dep_audit.AuditReport(scanned=1, source="offline")
        rep.vulnerabilities.append(dep_audit.Vulnerability(
            "requests", "2.0.0", "CVE-x", "leak", "HIGH", "2.31.0"))
        out = dep_audit.format_report(rep)
        assert "requests" in out and "CVE-x" in out

    def test_clean_property(self):
        assert dep_audit.AuditReport(scanned=1).clean is True


class TestEnvironment:
    def test_audit_environment_offline_runs(self):
        rep = dep_audit.audit_environment(offline=True, limit=10)
        assert rep.scanned >= 1
        assert isinstance(rep.clean, bool)

    def test_installed_distributions_nonempty(self):
        dists = dep_audit._installed_distributions()
        assert len(dists) > 0
        assert all(isinstance(k, str) for k in dists)


class TestDoctorIntegration:
    def test_doctor_check_runs(self):
        from core.doctor import check_dependency_vulnerabilities, CheckStatus
        r = check_dependency_vulnerabilities()
        assert r.status in (CheckStatus.PASS, CheckStatus.WARN, CheckStatus.SKIP)

    def test_doctor_check_in_all_checks(self):
        from core.doctor import _ALL_CHECKS, check_dependency_vulnerabilities
        assert check_dependency_vulnerabilities in _ALL_CHECKS
