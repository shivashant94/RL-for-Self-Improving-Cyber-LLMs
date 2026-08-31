"""Policy gate: the only permitted path from a model-proposed tool call to execution.

Design invariants
-----------------
- The LLM may PROPOSE a tool call; this gate AUTHORIZES and EXECUTES it.
- Unknown, malformed, oversized, secret-access, external-URL, code-exec, side-effecting,
  traversal-like, and unallowlisted calls are BLOCKED and AUDITED before any execution.
- Tool results never bypass the allowlist; every failure produces a safe denial.
"""

import re
from dataclasses import dataclass
from typing import Any, Optional

from .audit import AuditLog
from .tools import ToolRegistry

# ── Constants ────────────────────────────────────────────────────────────────

_MAX_PURPOSE_LEN: int = 200
_MAX_ARGUMENT_VALUE_LEN: int = 200  # per field – also enforced inside tools.py

# Patterns that indicate a purpose string is requesting secret/credential access.
_SECRET_PATTERNS: tuple[str, ...] = (
    "password",
    "credential",
    "api_key",
    "api key",
    "secret",
    "token",
    "private key",
    "access key",
    "auth key",
)

# Patterns that indicate an external URL fetch or code/shell execution request.
_UNSAFE_CONTENT_PATTERNS: tuple[str, ...] = (
    "http://",
    "https://",
    "exec(",
    "subprocess",
    "shell",
    "os.system",
    "eval(",
    "import os",
    "import subprocess",
)


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolCall:
    """Structured tool proposal from the model.  Validated before any execution."""
    tool: str
    arguments: dict[str, Any]
    purpose: str


@dataclass(frozen=True)
class PolicyDecision:
    """Machine-readable enforcement decision recorded alongside every gate outcome."""
    rule_id: str
    reason_code: str
    safe_message: str


@dataclass(frozen=True)
class GateResult:
    """Outcome of a gate evaluation.  ``data`` is only present when ``allowed`` is True."""
    allowed: bool
    message: str
    data: Optional[dict[str, Any]] = None
    decision: Optional[PolicyDecision] = None


# ── Gate ─────────────────────────────────────────────────────────────────────

class PolicyGate:
    """The only permitted execution path from a model-proposed call to a tool."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        audit_log: Optional[AuditLog] = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.audit_log = audit_log or AuditLog()

    # ── Public API ────────────────────────────────────────────────────────

    def execute(self, call: ToolCall) -> GateResult:
        """Validate, authorize, and (if allowed) execute ``call``."""

        # 1. Structural checks — tool name
        if not isinstance(call.tool, str) or not call.tool:
            return self._deny(
                "invalid_tool",
                rule_id="GATE-001",
                reason_code="invalid_tool_name",
                safe_message="Tool name must be a non-empty string.",
            )

        # 2. Structural checks — arguments type
        if not isinstance(call.arguments, dict):
            return self._deny(
                call.tool,
                rule_id="GATE-002",
                reason_code="malformed_arguments",
                safe_message="Tool arguments must be a JSON object.",
            )

        # 3. Purpose must be present and not oversized
        if not isinstance(call.purpose, str) or not call.purpose.strip():
            return self._deny(
                call.tool,
                rule_id="GATE-003",
                reason_code="missing_purpose",
                safe_message="A concise, non-empty purpose is required.",
            )
        if len(call.purpose) > _MAX_PURPOSE_LEN:
            return self._deny(
                call.tool,
                rule_id="GATE-004",
                reason_code="oversized_purpose",
                safe_message="Purpose exceeds the maximum permitted length.",
            )

        # 4. Secret-access guard — purpose must not request credentials/keys
        purpose_lower = call.purpose.lower()
        if any(pat in purpose_lower for pat in _SECRET_PATTERNS):
            return self._deny(
                call.tool,
                rule_id="GATE-005",
                reason_code="secret_access_attempt",
                safe_message=(
                    "Access to credentials, tokens, or secrets is not permitted. "
                    "I can help with an approved read-only information task."
                ),
            )

        # 5. External-URL / code-exec guard in purpose
        if any(pat in purpose_lower for pat in _UNSAFE_CONTENT_PATTERNS):
            return self._deny(
                call.tool,
                rule_id="GATE-006",
                reason_code="unsafe_content_in_purpose",
                safe_message=(
                    "Requests that reference external URLs or code execution are not permitted."
                ),
            )

        # 6. Tool must be allowlisted
        tool = self.registry.get(call.tool)
        if tool is None:
            return self._deny(
                call.tool,
                rule_id="GATE-007",
                reason_code="tool_not_allowlisted",
                safe_message="That tool is not available. I can help with an approved alternative.",
            )

        # 7. Per-tool argument validation (schema, traversal, allowlist)
        valid, reason = tool.validate(call.arguments)
        if not valid:
            return self._deny(
                call.tool,
                rule_id="GATE-008",
                reason_code="invalid_arguments",
                safe_message=reason,
            )

        # 8. Execute — any exception produces a safe failure; result is never leaked
        try:
            result = tool.run(call.arguments)
        except Exception:
            return self._deny(
                call.tool,
                rule_id="GATE-009",
                reason_code="tool_execution_failure",
                safe_message="The tool could not complete safely. No partial result was released.",
            )

        decision = PolicyDecision(
            rule_id="GATE-000",
            reason_code="allowed",
            safe_message="Approved read-only tool call.",
        )
        self.audit_log.record(call.tool, "allowed", "Read-only tool executed.")
        return GateResult(True, "Approved read-only tool call.", result, decision)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _deny(
        self,
        tool: str,
        *,
        rule_id: str,
        reason_code: str,
        safe_message: str,
    ) -> GateResult:
        """Record a block event and return a fail-closed GateResult."""
        self.audit_log.record(tool, "blocked", reason_code)
        decision = PolicyDecision(
            rule_id=rule_id,
            reason_code=reason_code,
            safe_message=safe_message,
        )
        return GateResult(False, safe_message, decision=decision)
