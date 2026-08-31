"""Structured audit log for the Review 1 policy gate.

Privacy / redaction invariants
--------------------------------
- Raw tool arguments and tool results are NEVER stored.
- The ``tool`` field stores only the tool name (a short identifier), not any argument value.
- The ``reason`` field stores only the gate's own reason code, not model-generated content.
- Sensitive-looking strings (email addresses, bearer/API-key shapes, IP addresses) are
  scrubbed from any free-text field before storage using ``_redact``.
"""

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

# ── Redaction ────────────────────────────────────────────────────────────────

# Patterns that indicate sensitive material that must never appear in audit logs.
_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Email addresses
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "[REDACTED_EMAIL]"),
    # Bearer tokens / JWT-like strings (three Base64url segments separated by dots)
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), "[REDACTED_TOKEN]"),
    # API-key shapes: alphanumeric strings of 20+ chars with no spaces (heuristic)
    (re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"), "[REDACTED_KEY]"),
    # IPv4 addresses
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
    # Credential-like key=value pairs
    (re.compile(r"(?i)(password|secret|token|api_key|credential)\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
]


def _redact(text: str) -> str:
    """Scrub recognizable sensitive patterns from ``text``."""
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ── Audit event ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuditEvent:
    """Immutable record of a single gate decision.

    Fields deliberately excluded: raw arguments, raw tool results, any user content.
    """
    timestamp: str   # ISO-8601 UTC
    tool: str        # allowlisted tool name only
    outcome: str     # "allowed" | "blocked"
    reason: str      # gate reason code (redacted free text if any)


# ── Audit log ────────────────────────────────────────────────────────────────

class AuditLog:
    """Append-only store for AuditEvents.

    Arguments and tool results are NEVER passed to this class;
    callers must pass only gate-generated metadata.
    """

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(self, tool: str, outcome: str, reason: str) -> None:
        """Append an event.  ``reason`` is redacted before storage."""
        self._events.append(
            AuditEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                tool=_redact(str(tool)),
                outcome=outcome,
                reason=_redact(str(reason)),
            )
        )

    def events(self) -> list[dict[str, Any]]:
        """Return all events as plain dicts (safe to serialize)."""
        return [asdict(event) for event in self._events]
