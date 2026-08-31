from dataclasses import dataclass
from typing import Any, Callable, Optional


def _single_string_argument(arguments: dict[str, Any], field: str) -> tuple[bool, str]:
    if set(arguments) != {field}:
        return False, f"Only the required '{field}' argument is permitted."
    value = arguments[field]
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        return False, f"'{field}' must be a non-empty string of at most 200 characters."
    if ".." in value or value.startswith(("/", "~")):
        return False, "Path-like or traversal values are not permitted."
    return True, ""


@dataclass(frozen=True)
class SafeTool:
    name: str
    validate: Callable[[dict[str, Any]], tuple[bool, str]]
    run: Callable[[dict[str, Any]], dict[str, Any]]


class ToolRegistry:
    """Review 1 fixture-only tools. All are read-only and deterministic."""

    def __init__(self) -> None:
        self._documents = {
            "doc_102": "Incident report: suspicious login was blocked and the account was reset.",
        }
        self._records = {
            "case_42": {"status": "open", "summary": "Phishing report awaiting triage."},
        }
        self._tools = {
            "get_document": SafeTool("get_document", self._validate_document, self._get_document),
            "lookup_record": SafeTool("lookup_record", self._validate_record, self._lookup_record),
            "search_mail": SafeTool("search_mail", self._validate_query, self._search_mail),
            "sandbox_text": SafeTool("sandbox_text", self._validate_text, self._sandbox_text),
        }

    def get(self, name: str) -> Optional[SafeTool]:
        return self._tools.get(name)

    def _validate_document(self, arguments: dict[str, Any]) -> tuple[bool, str]:
        valid, reason = _single_string_argument(arguments, "document_id")
        if valid and arguments["document_id"] not in self._documents:
            return False, "Document ID is outside the fixture allowlist."
        return valid, reason

    def _validate_record(self, arguments: dict[str, Any]) -> tuple[bool, str]:
        valid, reason = _single_string_argument(arguments, "record_id")
        if valid and arguments["record_id"] not in self._records:
            return False, "Record ID is outside the fixture allowlist."
        return valid, reason

    def _validate_query(self, arguments: dict[str, Any]) -> tuple[bool, str]:
        return _single_string_argument(arguments, "query")

    def _validate_text(self, arguments: dict[str, Any]) -> tuple[bool, str]:
        return _single_string_argument(arguments, "text")

    def _get_document(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"document_id": arguments["document_id"], "content": self._documents[arguments["document_id"]]}

    def _lookup_record(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"record_id": arguments["record_id"], "record": self._records[arguments["record_id"]]}

    def _search_mail(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments["query"].lower()
        messages = [
            {"id": "mail_1", "subject": "Suspicious login", "body": "Please review the alert."},
            {"id": "mail_2", "subject": "Weekly status", "body": "All services are healthy."},
        ]
        return {"matches": [m for m in messages if query in (m["subject"] + " " + m["body"]).lower()]}

    def _sandbox_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        text = arguments["text"]
        lowered = text.lower()
        signals = [term for term in ("ignore previous", "system prompt", "password", "http://", "https://") if term in lowered]
        return {"classification": "suspicious" if signals else "neutral", "signals": signals, "executed": False}
