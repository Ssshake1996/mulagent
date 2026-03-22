"""Security tools: output sanitization, sensitive data detection, injection prevention.

Provides pre/post hooks for the ReAct orchestrator and a standalone scan tool.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from tools.base import ToolDef

logger = logging.getLogger(__name__)

# ── Sensitive data patterns ──────────────────────────────────────────

_SENSITIVE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("API Key", re.compile(
        r'(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?key)'
        r'\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
        re.IGNORECASE,
    )),
    ("Bearer Token", re.compile(
        r'Bearer\s+([a-zA-Z0-9_\-\.]{20,})',
    )),
    ("Password", re.compile(
        r'(?:password|passwd|pwd)\s*[=:]\s*["\']?(\S{6,})["\']?',
        re.IGNORECASE,
    )),
    ("Private Key", re.compile(
        r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----',
    )),
    ("AWS Key", re.compile(
        r'(?:AKIA|ASIA)[A-Z0-9]{16}',
    )),
    ("JWT Token", re.compile(
        r'eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}',
    )),
    ("Connection String", re.compile(
        r'(?:mongodb|mysql|postgres|redis)://\S+:\S+@\S+',
        re.IGNORECASE,
    )),
    ("Phone Number (CN)", re.compile(
        r'(?<!\d)1[3-9]\d{9}(?!\d)',
    )),
    ("ID Card (CN)", re.compile(
        r'(?<!\d)\d{17}[\dXx](?!\d)',
    )),
    ("Email (in sensitive context)", re.compile(
        r'(?:email|邮箱)\s*[=:：]\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',
        re.IGNORECASE,
    )),
]

# ── Prompt injection patterns ────────────────────────────────────────

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r'ignore\s+(all\s+)?previous\s+instructions', re.IGNORECASE),
    re.compile(r'forget\s+(all\s+)?your\s+(previous\s+)?instructions', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+(?:a|an)\s+', re.IGNORECASE),
    re.compile(r'system\s*:\s*you\s+are', re.IGNORECASE),
    re.compile(r'<\|(?:im_start|im_end|system|endoftext)\|>', re.IGNORECASE),
    re.compile(r'```\s*system\b', re.IGNORECASE),
    re.compile(r'IMPORTANT:\s*(?:ignore|disregard|override)', re.IGNORECASE),
]


def scan_sensitive(text: str) -> list[dict[str, str]]:
    """Scan text for sensitive data patterns.

    Returns list of {type, match, position} for each finding.
    """
    findings = []
    for label, pattern in _SENSITIVE_PATTERNS:
        for m in pattern.finditer(text):
            findings.append({
                "type": label,
                "match": m.group()[:30] + "..." if len(m.group()) > 30 else m.group(),
                "position": str(m.start()),
            })
    return findings


def redact_sensitive(text: str) -> str:
    """Replace sensitive data with redacted placeholders."""
    result = text
    for label, pattern in _SENSITIVE_PATTERNS:
        def _replacer(m):
            return f"[REDACTED:{label}]"
        result = pattern.sub(_replacer, result)
    return result


def detect_injection(text: str) -> list[str]:
    """Detect potential prompt injection attempts in user input.

    Returns list of matched pattern descriptions.
    """
    detections = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            detections.append(pattern.pattern[:60])
    return detections


# ── Tool Hooks (Pre/Post) ────────────────────────────────────────────

def pre_tool_hook(tool_name: str, args: dict, directives: list[str]) -> str | None:
    """Pre-execution hook: check if tool call violates any directive.

    Returns an error message if blocked, None if allowed.
    """
    args_str = str(args).lower()

    for directive in directives:
        d_lower = directive.lower()

        # Block destructive operations if directive requires confirmation
        if any(kw in d_lower for kw in ["确认", "同意", "审批", "批准"]):
            destructive_patterns = ["rm ", "rm -", "delete", "drop ", "truncate ",
                                    "rmdir", "del ", "remove"]
            if tool_name in ("execute_shell", "write_file"):
                if any(p in args_str for p in destructive_patterns):
                    return (
                        f"[BLOCKED] Directive requires user confirmation for destructive operations: "
                        f"'{directive}'. Skipping {tool_name} with args that contain destructive commands."
                    )

        # Block scope violations
        if "只" in d_lower or "仅" in d_lower or "不要" in d_lower:
            # These are context-dependent, just log a warning
            logger.debug("Directive scope check for '%s': %s", tool_name, directive)

    return None


def post_tool_hook(tool_name: str, result: str) -> str:
    """Post-execution hook: sanitize tool output.

    Redacts any sensitive data that appears in tool results.
    """
    findings = scan_sensitive(result)
    if findings:
        logger.warning("Sensitive data found in %s output: %s",
                       tool_name, [f["type"] for f in findings])
        result = redact_sensitive(result)
    return result


# ── Standalone security scan tool ─────────────────────────────────────

async def _security_scan(params: dict[str, Any], **deps: Any) -> str:
    """Scan text for security issues."""
    text = params.get("text", "")
    if not text:
        return "Error: text is required"

    results = []

    # Check sensitive data
    sensitive = scan_sensitive(text)
    if sensitive:
        results.append("**Sensitive Data Found:**")
        for s in sensitive:
            results.append(f"  - {s['type']}: {s['match']} (pos: {s['position']})")
    else:
        results.append("No sensitive data detected.")

    # Check injection
    injections = detect_injection(text)
    if injections:
        results.append("\n**Potential Prompt Injection Detected:**")
        for i in injections:
            results.append(f"  - Pattern: {i}")
    else:
        results.append("No prompt injection patterns detected.")

    return "\n".join(results)


SECURITY_SCAN = ToolDef(
    name="security_scan",
    description="Scan text for sensitive data (API keys, passwords, PII) and prompt injection attempts.",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to scan for security issues",
            },
        },
        "required": ["text"],
    },
    fn=_security_scan,
)
