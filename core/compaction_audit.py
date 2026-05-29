"""
Operon Compaction Quality Audit.

Adapted from OpenClaw src/agents/pi-hooks/compaction-safeguard.ts.

When the conversation history is compressed ("compacted"), this module
validates that the resulting summary preserves all the critical information
that the agent needs to continue working safely.

Required sections in a valid compaction summary:
  • ## Decisions        — choices made during the session
  • ## Open TODOs       — pending work items
  • ## Constraints/Rules — rules and constraints the agent must follow
  • ## Pending user asks — questions or requests from the user not yet answered
  • ## Exact identifiers — literal names, paths, IDs referenced in conversation

Quality checks:
  1. Required sections present
  2. No critical identifiers missing (compared against recent messages)
  3. Summary not suspiciously short (< 200 chars)
  4. Summary not obviously truncated (no mid-sentence ending)
  5. Security rules not stripped (email_send restriction, etc.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Required sections ─────────────────────────────────────────────────────────

REQUIRED_SECTIONS: list[str] = [
    "## Decisions",
    "## Open TODOs",
    "## Constraints/Rules",
    "## Pending user asks",
    "## Exact identifiers",
]

# Security rules that MUST appear in every compaction summary
REQUIRED_SECURITY_RULES: list[str] = [
    "email_send",        # The email_send security restriction must be preserved
]

# Minimum length for a valid summary (characters)
MIN_SUMMARY_LENGTH = 200

# Patterns that suggest truncation
_TRUNCATION_PATTERNS = [
    r"[a-z]\s*$",                       # ends mid-sentence
    r"\.\.\.$",                          # ellipsis ending
    r"(?:and|or|but|the|a|an)\s*$",     # ends with conjunction/article
]


# ── Audit result ──────────────────────────────────────────────────────────────

@dataclass
class CompactionAuditResult:
    valid:            bool
    missing_sections: list[str]   = field(default_factory=list)
    missing_ids:      list[str]   = field(default_factory=list)
    security_issues:  list[str]   = field(default_factory=list)
    quality_issues:   list[str]   = field(default_factory=list)
    score:            float       = 1.0   # 0.0 – 1.0

    def summary(self) -> str:
        if self.valid:
            return f"✅ Compaction valid (score={self.score:.2f})"
        issues: list[str] = []
        if self.missing_sections:
            issues.append(f"Missing sections: {', '.join(self.missing_sections)}")
        if self.missing_ids:
            issues.append(f"Missing identifiers: {', '.join(self.missing_ids[:5])}")
        if self.security_issues:
            issues.append(f"Security: {'; '.join(self.security_issues)}")
        if self.quality_issues:
            issues.append(f"Quality: {'; '.join(self.quality_issues)}")
        return f"❌ Compaction issues — {'; '.join(issues)}"


# ── Identifier extraction ─────────────────────────────────────────────────────

_IDENTIFIER_PATTERNS: list[re.Pattern] = [
    # File paths
    re.compile(r'(?:^|[ "\'])(/(?:[\w./]+)+\.\w{1,6})(?:[ "\']|$)', re.MULTILINE),
    # Variable-like identifiers (snake_case, camelCase, SCREAMING_SNAKE)
    re.compile(r'\b([a-z][a-z0-9_]{2,}(?:_[a-z0-9]+)+)\b'),   # snake_case
    re.compile(r'\b([A-Z][A-Z0-9_]{2,}(?:_[A-Z0-9]+)+)\b'),   # SCREAMING_SNAKE
    # Python class/function names that look important
    re.compile(r'\b((?:class|def)\s+([A-Za-z_]\w+))'),
    # Quoted strings that look like names
    re.compile(r'"([a-zA-Z][a-zA-Z0-9_\-\.]{2,30})"'),
    # git commit hashes
    re.compile(r'\b([0-9a-f]{7,40})\b'),
    # UUIDs
    re.compile(r'\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b'),
]


def _extract_identifiers(text: str) -> set[str]:
    """Extract critical identifiers from text."""
    ids: set[str] = set()
    for pattern in _IDENTIFIER_PATTERNS:
        for m in pattern.finditer(text):
            # Use last group if it's a named capture, else group(1)
            val = m.group(m.lastindex or 1).strip()
            if len(val) >= 3:
                ids.add(val)
    return ids


# ── Core auditor ──────────────────────────────────────────────────────────────

def audit_compaction(
    summary:         str,
    recent_messages: Optional[list[dict]] = None,
    strict:          bool                 = False,
) -> CompactionAuditResult:
    """
    Audit a compaction summary for quality and completeness.

    Parameters
    ----------
    summary         : the compaction summary text to audit
    recent_messages : original messages that were compacted (for ID comparison)
    strict          : if True, fail on any quality issue (not just missing sections)

    Returns a CompactionAuditResult.
    """
    missing_sections: list[str] = []
    security_issues:  list[str] = []
    quality_issues:   list[str] = []
    missing_ids:      list[str] = []

    if not summary:
        return CompactionAuditResult(
            valid=False,
            quality_issues=["Summary is empty"],
            score=0.0,
        )

    # 1. Required sections
    for section in REQUIRED_SECTIONS:
        if section not in summary:
            missing_sections.append(section)

    # 2. Security rules
    for rule in REQUIRED_SECURITY_RULES:
        if rule not in summary:
            security_issues.append(
                f"Security rule '{rule}' not preserved in summary"
            )

    # 3. Length check
    if len(summary.strip()) < MIN_SUMMARY_LENGTH:
        quality_issues.append(
            f"Summary too short ({len(summary.strip())} chars < {MIN_SUMMARY_LENGTH})"
        )

    # 4. Truncation check
    last_line = summary.rstrip().split("\n")[-1].strip()
    for pat in _TRUNCATION_PATTERNS:
        if re.search(pat, last_line):
            quality_issues.append(f"Summary may be truncated (ends: {last_line!r})")
            break

    # 5. Identifier preservation (if original messages provided)
    if recent_messages:
        # Extract IDs from original messages (last 20)
        original_text = "\n".join(
            str(m.get("content", ""))
            for m in recent_messages[-20:]
        )
        original_ids  = _extract_identifiers(original_text)
        summary_ids   = _extract_identifiers(summary)
        # Check which important IDs are missing from the summary
        # Focus on long identifiers (more likely to be important)
        important_ids = {i for i in original_ids if len(i) >= 6}
        missing       = important_ids - summary_ids
        # Only flag the most significant ones (paths, SCREAMING_SNAKE)
        critical_missing = [
            i for i in missing
            if "/" in i or "_" in i or i.upper() == i
        ]
        missing_ids = critical_missing[:10]

    # Calculate score
    deductions = 0.0
    deductions += len(missing_sections) * 0.15
    deductions += len(security_issues) * 0.25
    deductions += len(quality_issues) * 0.10
    deductions += min(len(missing_ids) * 0.05, 0.20)
    score = max(0.0, 1.0 - deductions)

    # Validity: must have no missing sections, no security issues, and score > 0.4
    valid = (
        len(missing_sections) == 0
        and len(security_issues) == 0
        and score > 0.4
        and (not strict or len(quality_issues) == 0)
    )

    return CompactionAuditResult(
        valid            = valid,
        missing_sections = missing_sections,
        missing_ids      = missing_ids,
        security_issues  = security_issues,
        quality_issues   = quality_issues,
        score            = score,
    )


def generate_compaction_template() -> str:
    """
    Return a template for a valid compaction summary.
    Agents should use this as a starting point when compacting.
    """
    return """## Decisions
- (List key decisions made during this session)

## Open TODOs
- (List pending work items that have not been completed)

## Constraints/Rules
- email_send must never be exposed to the model as a callable tool
- Credentials must never be typed directly into the chat
- (Other constraints from the user or system)

## Pending user asks
- (Questions or requests from the user that haven't been answered yet)

## Exact identifiers
- (File paths, variable names, IDs, hashes referenced in the conversation)
"""


def check_and_repair_compaction(
    summary:         str,
    recent_messages: Optional[list[dict]] = None,
) -> tuple[str, CompactionAuditResult]:
    """
    Audit a compaction summary and attempt basic repairs.

    Returns (repaired_summary, audit_result).
    If no repairs are needed, returns the original summary unchanged.
    """
    result = audit_compaction(summary, recent_messages)
    if result.valid:
        return summary, result

    repaired = summary

    # Add missing sections at the end
    for section in result.missing_sections:
        repaired += f"\n\n{section}\n- (Preserved from original session)"

    # Add security rules if missing
    if result.security_issues:
        constraints_section = "## Constraints/Rules"
        if constraints_section in repaired:
            # Insert after the header
            repaired = repaired.replace(
                constraints_section + "\n",
                constraints_section + "\n- email_send must never be model-callable\n",
                1,
            )
        else:
            repaired += f"\n\n{constraints_section}\n- email_send must never be model-callable\n"

    # Re-audit the repaired version
    final_result = audit_compaction(repaired, recent_messages)
    return repaired, final_result
