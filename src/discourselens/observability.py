"""Privacy-preserving structured events for local CLI and MCP diagnostics."""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TextIO

_EVENT_FIELDS: dict[str, frozenset[str]] = {
    "document.import.started": frozenset({"byte_count"}),
    "document.import.completed": frozenset({"document_id", "revision_id", "status", "byte_count"}),
    "analysis.run.created": frozenset({"analysis_run_id", "scope_id", "status"}),
    "analysis.run.started": frozenset({"analysis_run_id", "scope_id", "status"}),
    "analysis.run.completed": frozenset({"analysis_run_id", "scope_id", "status"}),
    "analysis.run.failed": frozenset({"analysis_run_id", "error_type", "status"}),
    "analysis.run.resumed": frozenset({"analysis_run_id", "status"}),
}
_IDENTIFIER_FIELDS = frozenset({"document_id", "revision_id", "analysis_run_id", "scope_id"})
_IMPORT_STATUSES = frozenset(
    {"SUCCESS", "SUCCESS_WITH_WARNINGS", "OCR_REQUIRED", "PASSWORD_REQUIRED", "DUPLICATE"}
)
_EVENT_STATUSES: dict[str, frozenset[str]] = {
    "document.import.completed": _IMPORT_STATUSES,
    "analysis.run.created": frozenset({"PENDING"}),
    "analysis.run.started": frozenset({"RUNNING"}),
    "analysis.run.completed": frozenset({"COMPLETED"}),
    "analysis.run.failed": frozenset({"FAILED"}),
    "analysis.run.resumed": frozenset({"RUNNING"}),
}
_MAX_BYTE_COUNT = (1 << 63) - 1
_SAFE_LEVEL_NAMES = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}
# Lifecycle identifiers are generated UUIDs; free-form strings can carry document text.
_SAFE_IDENTIFIER = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[47][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SAFE_EXCEPTION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,79}$")
_CREDENTIAL_SHAPED = re.compile(
    r"(?:sk|pk|rk)[_-]|github_pat_|gh[pousr]_|xox[baprs]-|"
    r"(?:AKIA|ASIA)[A-Z0-9]|AIza|ya29\.|eyJ[A-Za-z0-9_-]+\.",
    re.IGNORECASE,
)


def _safe_fields(event: str, fields: Mapping[str, object]) -> dict[str, str | int]:
    allowed = _EVENT_FIELDS.get(event, frozenset())
    safe: dict[str, str | int] = {}
    for key, value in fields.items():
        if key not in allowed:
            continue
        if (
            key in _IDENTIFIER_FIELDS
            and isinstance(value, str)
            and _SAFE_IDENTIFIER.fullmatch(value)
            and not _CREDENTIAL_SHAPED.search(value)
        ):
            safe[key] = value
        elif key == "status" and isinstance(value, str):
            if value in _EVENT_STATUSES.get(event, frozenset()):
                safe[key] = value
        elif key == "byte_count":
            if type(value) is int and 0 <= value <= _MAX_BYTE_COUNT:
                safe[key] = value
        elif (
            key == "error_type"
            and isinstance(value, str)
            and _SAFE_EXCEPTION_NAME.fullmatch(value)
            and not _CREDENTIAL_SHAPED.search(value)
        ):
            safe[key] = value
    return safe


def log_event(
    logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: object
) -> None:
    """Emit only allowlisted metadata; unknown fields and event names are suppressed."""
    if event not in _EVENT_FIELDS:
        return
    logger.log(
        level,
        event,
        extra={
            "discourselens_event": event,
            "discourselens_fields": _safe_fields(event, fields),
        },
    )


class JsonEventFormatter(logging.Formatter):
    """Render approved lifecycle events as one JSON object per stderr line."""

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "discourselens_event", None)
        if not isinstance(event, str) or event not in _EVENT_FIELDS:
            event = "unstructured_log_suppressed"
            fields: dict[str, str | int] = {}
        else:
            try:
                candidate = getattr(record, "discourselens_fields", {})
                fields = _safe_fields(event, candidate) if isinstance(candidate, Mapping) else {}
            except Exception:  # noqa: BLE001 - custom Mapping methods may raise arbitrary errors.
                # A malformed/custom mapping must not make logging's fallback write its
                # exception and potentially sensitive details to process stderr.
                fields = {}
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": _SAFE_LEVEL_NAMES.get(record.levelno, "OTHER"),
            "event": event,
        }
        payload.update(fields)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def configure_cli_logging(
    *, verbose: bool, debug: bool, stream: TextIO | None = None
) -> Callable[[], None]:
    """Temporarily configure package event logging and return a state restorer."""
    logger = logging.getLogger("discourselens")
    previous_level = logger.level
    previous_propagate = logger.propagate
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonEventFormatter())
    handler.setLevel(logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING)
    logger.addHandler(handler)
    logger.setLevel(handler.level)
    logger.propagate = False

    def restore() -> None:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    return restore
