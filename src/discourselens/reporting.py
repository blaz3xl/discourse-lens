"""Deterministic report rendering from persisted, validated records."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from functools import partial
from html import escape
from itertools import islice
from pathlib import Path
from textwrap import wrap
from typing import Any, Literal, TypedDict, cast
from urllib.parse import parse_qsl, urlsplit
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile

import pymupdf as fitz

from .errors import DocumentError, EvidenceValidationError, ModelProviderError
from .models import ModelPrice
from .secrets import has_credential_like_endpoint_host, is_credential_like_alias
from .validation import require_valid_finding

_MARKDOWN_ESCAPABLE = frozenset(r"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_REPORT_SECRET_METADATA_KEYS = {
    "api_key",
    "api_token",
    "apikey",
    "authorization",
    "credentials",
    "credential_ref",
    "credential_reference",
    "credential_value",
    "password",
    "secret",
    "secret_ref",
    "secret_reference",
    "secret_value",
    "token",
    "access_token",
    "auth_token",
}
_REPORT_SECRET_KEY_CONTEXTS = {
    "access",
    "api",
    "auth",
    "authorization",
    "client",
    "credential",
    "encryption",
    "master",
    "private",
    "secret",
    "service",
    "signing",
    "ssh",
    "vendor",
}
_REPORT_SECRET_TOKEN_CONTEXTS = {
    "access",
    "api",
    "auth",
    "authorization",
    "bearer",
    "client",
    "credential",
    "oauth",
    "private",
    "refresh",
    "secret",
    "service",
    "session",
    "vendor",
}
_REPORT_SAFE_TOKEN_USAGE_KEYS = {
    "accepted_prediction_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cached_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "reasoning_tokens",
    "rejected_prediction_tokens",
    "total_tokens",
    "uncached_tokens",
}
_REPORT_HOST_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_./-])/(?!/)[^\s\"'<>|?*]+"),
    re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+\\[^\\/\s]+)"),
    re.compile(r"(?i)\bfile:(?://)?(?:/[A-Za-z]:)?[/\\]"),
)
_CREDENTIAL_ALIAS_METADATA_KEYS = {
    "credential_alias",
    "reranking_credential_alias",
}
_PRICE_SOURCE_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|"
    r"access[-_ ]?key|authorization|password|secret|signature|sig|token|credential)"
    r"(\s*[:=]\s*)"
)
_PRICE_SOURCE_BEARER_CREDENTIAL = re.compile(r"(?i)\bbearer\s+[^\s,;&]+")
_PRICE_SOURCE_SECRET_QUERY_KEYS = {
    "access_key",
    "access_token",
    "aws_access_key_id",
    "awsaccesskeyid",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "client_secret",
    "credential",
    "credentials",
    "key",
    "password",
    "secret",
    "signature",
    "sig",
    "token",
    "x_amz_credential",
    "x_amz_signature",
    "x_amz_security_token",
}
_PROMPT_PROVENANCE_FRAMEWORK_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_PROMPT_PROVENANCE_SKILL_ID = re.compile(r"[a-z0-9][a-z0-9.-]{0,95}\Z")
_PROMPT_PROVENANCE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}\Z")
_PROMPT_PROVENANCE_REF = re.compile(r"[a-z0-9][a-z0-9-]{0,127}\Z")
_PROMPT_PROVENANCE_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_PROMPT_PROVENANCE_FIELDS = (
    "framework_id",
    "framework_version",
    "skill_id",
    "skill_version",
    "prompt_ref",
    "prompt_template_sha256",
)


def _normalize_report_metadata_key(key: object) -> str:
    text = str(key)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _is_secret_report_metadata_key(key: object) -> bool:
    normalized = _normalize_report_metadata_key(key)
    if normalized in _REPORT_SECRET_METADATA_KEYS:
        return True
    parts = set(normalized.split("_"))
    if parts.intersection({"password", "secret", "credentials", "authorization"}):
        return True
    if "key" in parts and parts.intersection(_REPORT_SECRET_KEY_CONTEXTS):
        return True
    if "tokens" in parts and parts.intersection(_REPORT_SECRET_TOKEN_CONTEXTS):
        return True
    if "token" in parts:
        if parts.intersection(_REPORT_SECRET_TOKEN_CONTEXTS):
            return True
        return normalized not in _REPORT_SAFE_TOKEN_USAGE_KEYS and not parts.intersection(
            {"count", "counts"}
        )
    return False


def _contains_sensitive_report_metadata(
    value: object, *, allow_counterevidence_query_labels: bool = False
) -> tuple[bool, bool]:
    """Find secret-like keys and machine-local paths in report metadata only."""
    pending = [value]
    secret_found = False
    path_found = False
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                if (
                    allow_counterevidence_query_labels
                    and _normalize_report_metadata_key(key) == "counterevidence_queries"
                    and isinstance(child, dict)
                ):
                    # These keys are query identifiers, not metadata field names.
                    # Continue walking their values so nested config objects still
                    # receive the normal secret/path checks.
                    pending.extend(child.values())
                    continue
                secret_found = secret_found or _is_secret_report_metadata_key(key)
                pending.append(child)
        elif isinstance(current, list):
            pending.extend(current)
        elif isinstance(current, str):
            path_found = path_found or any(
                pattern.search(current) for pattern in _REPORT_HOST_PATH_PATTERNS
            )
        if secret_found and path_found:
            break
    return secret_found, path_found


def _contains_suspicious_credential_alias(value: object) -> bool:
    """Find contaminated aliases in structured and JSON-encoded export metadata."""
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                normalized = _normalize_report_metadata_key(key)
                if (
                    normalized in _CREDENTIAL_ALIAS_METADATA_KEYS
                    and child is not None
                    and is_credential_like_alias(child)
                ):
                    return True
                pending.append(child)
                if normalized.endswith("_json") and isinstance(child, str):
                    try:
                        encoded = json.loads(child)
                    except (ValueError, RecursionError):
                        continue
                    if isinstance(encoded, (dict, list)):
                        pending.append(encoded)
        elif isinstance(current, list):
            pending.extend(current)
    return False


def _contains_suspicious_provider_endpoint(value: object) -> bool:
    """Find credential-shaped hostnames in serialized provider endpoint metadata."""
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                normalized = _normalize_report_metadata_key(key)
                if normalized in {"endpoint", "provider_endpoint"} and (
                    has_credential_like_endpoint_host(child)
                ):
                    return True
                pending.append(child)
                if normalized.endswith("_json") and isinstance(child, str):
                    try:
                        encoded = json.loads(child)
                    except (ValueError, RecursionError):
                        continue
                    if isinstance(encoded, (dict, list)):
                        pending.append(encoded)
        elif isinstance(current, list):
            pending.extend(current)
    return False


def _validate_report_metadata(
    value: object, label: str, *, allow_counterevidence_query_labels: bool = False
) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, float) and not math.isfinite(current):
            raise EvidenceValidationError(f"{label} contains non-finite numeric metadata")
        if isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    has_secret, has_host_path = _contains_sensitive_report_metadata(
        value, allow_counterevidence_query_labels=allow_counterevidence_query_labels
    )
    if has_secret:
        raise EvidenceValidationError(f"{label} contains secret-like metadata")
    if _contains_suspicious_credential_alias(value):
        raise EvidenceValidationError(f"{label} contains a suspicious credential alias")
    if _contains_suspicious_provider_endpoint(value):
        raise EvidenceValidationError(f"{label} contains credential-like provider endpoint metadata")
    if has_host_path:
        raise EvidenceValidationError(f"{label} contains an absolute host path")


def _controlled_cda_prompt_provenance(config: object) -> dict[str, object] | None:
    """Select bounded built-in prompt identifiers and digests from a controlled CDA config."""
    if not isinstance(config, dict) or config.get("profile") != "general-cda-controlled-v3":
        return None
    if (
        type(config.get("framework_prompt_snapshot_version")) is not int
        or config["framework_prompt_snapshot_version"] != 2
        or type(config.get("synthesis_prompt_snapshot_version")) is not int
        or config["synthesis_prompt_snapshot_version"] != 2
    ):
        return None

    framework_ids = config.get("frameworks")
    raw_frameworks = config.get("framework_prompt_snapshots")
    if (
        not isinstance(framework_ids, list)
        or not framework_ids
        or len(framework_ids) > 64
        or not all(
            isinstance(item, str) and _PROMPT_PROVENANCE_FRAMEWORK_ID.fullmatch(item)
            for item in framework_ids
        )
        or len(set(framework_ids)) != len(framework_ids)
        or not isinstance(raw_frameworks, list)
        or len(raw_frameworks) != len(framework_ids)
    ):
        return None

    def select_snapshot(
        value: object, *, include_template_provenance: bool = False
    ) -> dict[str, str] | None:
        if not isinstance(value, dict):
            return None
        fields = _PROMPT_PROVENANCE_FIELDS
        if include_template_provenance:
            fields += ("prompt_template_ref", "prompt_template_version")
        snapshot = {field: value.get(field) for field in fields}
        if (
            not isinstance(snapshot["framework_id"], str)
            or not _PROMPT_PROVENANCE_FRAMEWORK_ID.fullmatch(snapshot["framework_id"])
            or not isinstance(snapshot["framework_version"], str)
            or not _PROMPT_PROVENANCE_VERSION.fullmatch(snapshot["framework_version"])
            or not isinstance(snapshot["skill_id"], str)
            or not _PROMPT_PROVENANCE_SKILL_ID.fullmatch(snapshot["skill_id"])
            or not isinstance(snapshot["skill_version"], str)
            or not _PROMPT_PROVENANCE_VERSION.fullmatch(snapshot["skill_version"])
            or not isinstance(snapshot["prompt_ref"], str)
            or not _PROMPT_PROVENANCE_REF.fullmatch(snapshot["prompt_ref"])
            or not isinstance(snapshot["prompt_template_sha256"], str)
            or not _PROMPT_PROVENANCE_SHA256.fullmatch(snapshot["prompt_template_sha256"])
        ):
            return None
        if include_template_provenance and (
            not isinstance(snapshot["prompt_template_ref"], str)
            or not _PROMPT_PROVENANCE_REF.fullmatch(snapshot["prompt_template_ref"])
            or not isinstance(snapshot["prompt_template_version"], str)
            or not _PROMPT_PROVENANCE_VERSION.fullmatch(snapshot["prompt_template_version"])
        ):
            return None
        return cast(dict[str, str], snapshot)

    selected_frameworks: list[dict[str, str]] = []
    for framework_id, raw_snapshot in zip(framework_ids, raw_frameworks, strict=True):
        snapshot = select_snapshot(raw_snapshot, include_template_provenance=True)
        if snapshot is None or snapshot["framework_id"] != framework_id:
            return None
        selected_frameworks.append(snapshot)

    synthesis = select_snapshot(config.get("synthesis_prompt_snapshot"))
    if (
        synthesis is None
        or synthesis["framework_id"] != "evidence-constrained-synthesis"
        or synthesis["skill_id"] != "builtin.synthesis"
    ):
        return None

    return {
        "framework_prompt_snapshot_version": 2,
        "framework_prompt_snapshots": selected_frameworks,
        "synthesis_prompt_snapshot_version": 2,
        "synthesis_prompt_snapshot": synthesis,
    }


def _report_scope_batches(project, revision_ids: list[str]) -> Iterator[list[str]]:
    """Keep report scope filters below both SQLite's active and portable bind limits."""
    active_limit = project.db.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    batch_size = max(1, min(900, active_limit))
    for offset in range(0, len(revision_ids), batch_size):
        yield revision_ids[offset : offset + batch_size]


def _report_scope_query(
    project,
    revision_ids: list[str],
    *,
    select_sql: str,
    count_sql: str | None = None,
    order_key: Callable[[sqlite3.Row], tuple[str | int, ...]],
    offset: int = 0,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Query scope batches and merge their sorted rows before applying a page."""
    cursors: list[sqlite3.Cursor] = []
    total = 0
    for batch in _report_scope_batches(project, revision_ids):
        placeholders = ",".join("?" for _ in batch)
        if count_sql is not None:
            total += int(
                project.db.execute(count_sql.format(scope_ids=placeholders), batch).fetchone()[0]
            )
        cursors.append(project.db.execute(select_sql.format(scope_ids=placeholders), batch))

    def ordered_rows(
        cursor: sqlite3.Cursor, batch_index: int
    ) -> Iterator[tuple[tuple[str | int, ...], int, int, dict[str, Any]]]:
        for row_index, row in enumerate(cursor):
            yield order_key(row), batch_index, row_index, dict(row)

    merged = heapq.merge(*(ordered_rows(cursor, index) for index, cursor in enumerate(cursors)))
    if limit is None:
        rows = [item[3] for item in merged]
        total = len(rows)
    else:
        rows = [item[3] for item in islice(merged, offset, offset + limit)]
    return rows, total


def _render_report_citation(project, evidence_id: str) -> str:
    """Keep Core-rendered citations from exposing machine-local document paths."""
    citation = str(project.render_citation(evidence_id))
    _validate_report_metadata(citation, "report citation")
    return citation


def _render_report_span_location(project, span_id: str) -> str:
    """Validate canonical annotation locations before they cross the report boundary."""
    citation = str(project.render_span_location(span_id))
    _validate_report_metadata(citation, "report citation")
    return citation


def _unique_report_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate metadata keys rather than validating last-key-wins JSON."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_non_finite_report_json_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not supported")


def _parse_model_price_snapshot(raw: object) -> dict[str, Any] | None:
    """Validate a persisted price snapshot while leaving its encoded bytes untouched."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError("model price snapshot must be encoded JSON")
    try:
        snapshot = json.loads(
            raw,
            object_pairs_hook=_unique_report_json_object,
            parse_constant=_reject_non_finite_report_json_number,
        )
    except (ValueError, RecursionError) as exc:
        raise ValueError("model price snapshot is invalid") from exc
    if not isinstance(snapshot, dict):
        raise TypeError("model price snapshot must be a JSON object")
    try:
        price = ModelPrice.from_snapshot(snapshot)
    except (ModelProviderError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("model price snapshot is invalid") from exc
    if set(snapshot) != set(price.to_snapshot()):
        raise ValueError("model price snapshot contains unsupported fields")
    return snapshot


def _validate_model_execution_metadata(executions: list[dict[str, Any]]) -> None:
    for execution in executions:
        _validate_report_metadata(execution, "model execution report metadata")
        for field in ("parameters_json", "capabilities_json"):
            raw = execution.get(field)
            if not isinstance(raw, str):
                raise EvidenceValidationError("model execution report metadata is invalid")
            try:
                metadata = json.loads(raw, object_pairs_hook=_unique_report_json_object)
            except (ValueError, RecursionError) as exc:
                raise EvidenceValidationError("model execution report metadata is invalid") from exc
            if not isinstance(metadata, dict):
                raise EvidenceValidationError("model execution report metadata is invalid")
            _validate_report_metadata(metadata, "model execution report metadata")
        try:
            price = _parse_model_price_snapshot(execution.get("model_price_json"))
        except (TypeError, ValueError) as exc:
            raise EvidenceValidationError("model execution report metadata is invalid") from exc
        if price is None:
            continue
        if price["provider_id"] != execution.get("provider_id") or price["model"] != execution.get(
            "model"
        ):
            raise EvidenceValidationError(
                "model execution price snapshot does not match executed provider and model"
            )
        _validate_report_metadata(price, "model execution report metadata")
        source_reference = price.get("source_reference")
        if source_reference is not None and _price_source_contains_secret(source_reference):
            raise EvidenceValidationError(
                "model execution report metadata contains secret-like source reference"
            )


def _price_source_contains_secret(value: object) -> bool:
    """Detect credentials and signed URL parameters in a price snapshot source."""
    if not isinstance(value, str):
        return True
    if _PRICE_SOURCE_SECRET_ASSIGNMENT.search(value) or _PRICE_SOURCE_BEARER_CREDENTIAL.search(
        value
    ):
        return True
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if parsed.username is not None or parsed.password is not None:
        return True
    for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
        normalized = _normalize_report_metadata_key(key)
        if normalized in _PRICE_SOURCE_SECRET_QUERY_KEYS or _is_secret_report_metadata_key(key):
            return True
        # Common cloud signed URL parameters bind a request to a credential or
        # signature even when the provider-specific names are unfamiliar.
        if any(part in normalized for part in ("signature", "sig", "credential")):
            return True
    return False


def _markdown_text(value: object) -> str:
    """Keep untrusted values inline so they cannot add Markdown structure."""
    text = str(value).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return "".join(f"\\{char}" if char in _MARKDOWN_ESCAPABLE else char for char in text)


def _unescape_markdown_text(value: str) -> str:
    """Recover literal text when the simple PDF/DOCX writers consume Markdown."""
    return re.sub(r"\\([!\"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~])", r"\1", value)


def _evidence_role_label(value: str) -> str:
    """Render controlled evidence roles explicitly instead of flattening lineage."""
    return value.replace("_", " ").lower().capitalize()


class EvidencePayload(TypedDict):
    evidence_id: str
    role: str
    kind: str
    text: str | None
    citation: str | None
    source_span_id: str | None
    rationale: str | None


class ObservationCandidatePayload(TypedDict):
    id: str
    evidence_id: str
    excerpt: str
    description: str
    status: Literal["UNVERIFIED"]
    citation: str


class HumanReviewPayload(TypedDict):
    id: str
    status: str
    note: str | None
    created_at: str


class FindingPayload(TypedDict):
    finding: dict[str, Any]
    coverage: dict[str, Any]
    evidence: list[EvidencePayload]
    observation_candidates: list[ObservationCandidatePayload]
    observation_candidate_total: int
    observation_candidates_truncated: bool
    evidence_total: int
    evidence_truncated: bool
    human_reviews: list[HumanReviewPayload]
    human_review_total: int
    human_reviews_truncated: bool


def _report_coverage_disclosure(project, coverage_id: object) -> dict[str, Any]:
    """Resolve a referenced descriptor without turning a missing row into a claim."""
    if coverage_id is None:
        return {"coverage_id": None, "status": "NOT_RECORDED", "descriptor": None}
    if not isinstance(coverage_id, str) or not coverage_id:
        raise EvidenceValidationError("coverage descriptor reference is invalid")
    try:
        descriptor = project.coverage_descriptor(coverage_id)
    except EvidenceValidationError as exc:
        exists = project.db.execute(
            "SELECT 1 FROM coverage_descriptors WHERE id=?", (coverage_id,)
        ).fetchone()
        if exists:
            raise EvidenceValidationError("coverage descriptor report metadata is invalid") from exc
        return {"coverage_id": coverage_id, "status": "MISSING", "descriptor": None}
    descriptor_payload = descriptor.model_dump(mode="json")
    _validate_report_metadata(descriptor_payload, "coverage descriptor report metadata")
    return {
        "coverage_id": coverage_id,
        "status": "RECORDED",
        "descriptor": descriptor_payload,
    }


def _report_coverage_disclosures(project, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve the unique coverage references present in these report records."""
    coverage_ids: set[str] = set()
    for record in records:
        coverage_id = record.get("coverage_id")
        if coverage_id is None:
            continue
        if not isinstance(coverage_id, str) or not coverage_id:
            raise EvidenceValidationError("coverage descriptor reference is invalid")
        coverage_ids.add(coverage_id)
    return [
        _report_coverage_disclosure(project, coverage_id) for coverage_id in sorted(coverage_ids)
    ]


def _coverage_disclosure_values(coverage: dict[str, Any]) -> list[tuple[str, str]]:
    status = coverage["status"]
    if status == "NOT_RECORDED":
        return [("Coverage descriptor", "not recorded")]
    if status == "MISSING":
        return [("Coverage descriptor", f"missing for id {coverage['coverage_id']}")]
    descriptor = coverage["descriptor"]
    fields = (
        ("Coverage mode", "coverage_mode"),
        ("Scope snapshot id", "scope_snapshot_id"),
        ("Population definition", "population_definition"),
        ("Population count", "population_count"),
        ("Selection policy", "selection_policy"),
        ("Inspection count", "inspection_count"),
        ("Completion criterion", "completion_criterion"),
        ("Termination reason", "termination_reason"),
        ("Termination state", "termination_state"),
        ("Negative result status", "negative_result_status"),
    )
    values = [("Coverage id", str(coverage["coverage_id"]))]
    for label, field in fields:
        value = descriptor.get(field)
        values.append((label, "not recorded" if value is None else str(value)))
    return values


def _append_coverage_markdown(lines: list[str], coverage: dict[str, Any]) -> None:
    lines.append("**Coverage:**")
    lines.extend(
        f"- **{_markdown_text(label)}:** {_markdown_text(value)}"
        for label, value in _coverage_disclosure_values(coverage)
    )


def _append_coverage_html(parts: list[str], coverage: dict[str, Any]) -> None:
    parts.append("<p><strong>Coverage:</strong></p><ul>")
    parts.extend(
        f"<li><strong>{escape(label)}:</strong> {escape(value)}</li>"
        for label, value in _coverage_disclosure_values(coverage)
    )
    parts.append("</ul>")


def _cap_report_strings(value: Any, *, max_chars: int = 2_000) -> tuple[Any, bool]:
    """Recursively cap untrusted/persisted text in bounded MCP report pages."""
    if isinstance(value, str):
        return (value[:max_chars], True) if len(value) > max_chars else (value, False)
    if isinstance(value, list):
        output_items: list[Any] = []
        truncated = False
        for item in value:
            bounded, item_truncated = _cap_report_strings(item, max_chars=max_chars)
            output_items.append(bounded)
            truncated = truncated or item_truncated
        return output_items, truncated
    if isinstance(value, dict):
        output_mapping: dict[str, Any] = {}
        fields: list[str] = []
        for key, item in value.items():
            bounded, item_truncated = _cap_report_strings(item, max_chars=max_chars)
            output_mapping[key] = bounded
            if item_truncated:
                fields.append(str(key))
        if fields:
            output_mapping["_truncated_fields"] = fields
        return output_mapping, bool(fields)
    return value, False


def _report_field_truncated(record: dict[str, Any], field: str) -> bool:
    truncated_fields = record.get("_truncated_fields", [])
    return bool(record.get(f"{field}_truncated")) or (
        isinstance(truncated_fields, list) and field in truncated_fields
    )


def _markdown_report_field(record: dict[str, Any], field: str) -> str:
    value = _markdown_text(record.get(field, ""))
    return value + (" [truncated]" if _report_field_truncated(record, field) else "")


def _html_report_field(record: dict[str, Any], field: str) -> str:
    value = escape(str(record.get(field, "")))
    return value + (" [truncated]" if _report_field_truncated(record, field) else "")


def finding_payload(
    project,
    finding_id: str,
    *,
    evidence_limit: int | None = None,
    review_limit: int | None = None,
    observation_candidate_limit: int | None = None,
) -> FindingPayload:
    """Return a renderer-neutral, citation-derived representation of one finding."""
    require_valid_finding(project, finding_id)
    finding = project.get_finding(finding_id)
    links = project.finding_evidence(finding_id, limit=evidence_limit)
    evidence_total = (
        len(links)
        if evidence_limit is None
        else int(
            project.db.execute(
                "SELECT COUNT(*) FROM finding_evidence WHERE finding_id=?", (finding_id,)
            ).fetchone()[0]
        )
    )
    if not links:
        raise EvidenceValidationError("finding has no evidence")
    evidence: list[EvidencePayload] = []
    for item in links:
        payload: EvidencePayload = {
            "evidence_id": item["id"],
            "role": item["role"],
            "kind": item["kind"],
            "text": None,
            "citation": None,
            "source_span_id": None,
            "rationale": item["rationale"],
        }
        if item["kind"] == "TEXTUAL":
            payload["text"] = project.span_text(item["span_id"])
            payload["source_span_id"] = item["span_id"]
        elif item["kind"] == "CORPUS_STATISTIC":
            content = project.evidence_content(item["id"])
            provenance = content.get("provenance")
            if isinstance(provenance, dict) and provenance.get("comparison_run_id"):
                comparison_record = {
                    "label": content.get("label"),
                    "metric": provenance.get("comparison_metric"),
                    "value": content.get("value"),
                    "selected_row": provenance.get("selected_row"),
                    "comparison_run_id": provenance.get("comparison_run_id"),
                    "comparison_result_hash": provenance.get("comparison_result_hash"),
                    "calculation_version": provenance.get("calculation_version"),
                    "left_scope_snapshot_id": provenance.get("left_scope_snapshot_id"),
                    "left_revision_ids": provenance.get("left_revision_ids"),
                    "left_population": provenance.get("left_population"),
                    "right_scope_snapshot_id": provenance.get("right_scope_snapshot_id"),
                    "right_revision_ids": provenance.get("right_revision_ids"),
                    "right_population": provenance.get("right_population"),
                    "comparability_basis": provenance.get("comparability_basis"),
                    "comparability_basis_note": (
                        "Researcher-supplied; not independently assessed by DiscourseLens."
                    ),
                    "population_definition": content.get("population_definition"),
                    "coverage_id": provenance.get("coverage_id"),
                }
                payload["text"] = "Verified comparison evidence: " + json.dumps(
                    comparison_record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            else:
                payload["text"] = (
                    f"{content.get('label', 'corpus statistic')}: {content.get('value')} "
                    f"({content.get('population_definition', 'population not specified')})"
                )
        else:
            content = project.evidence_content(item["id"])
            payload["text"] = str(content.get("excerpt") or content.get("content") or "")
        payload["citation"] = _render_report_citation(project, item["id"])
        evidence.append(payload)
    observation_candidate_total = (
        len(project.finding_observation_candidates(finding_id))
        if observation_candidate_limit is None
        else int(
            project.db.execute(
                "SELECT COUNT(*) FROM finding_observation_candidates WHERE finding_id=?",
                (finding_id,),
            ).fetchone()[0]
        )
    )
    observation_candidates: list[ObservationCandidatePayload] = []
    linked_evidence_ids = {
        str(item["id"]) for item in project.finding_evidence(finding_id)
    }
    for item in project.finding_observation_candidates(
        finding_id, limit=observation_candidate_limit
    ):
        if item["status"] != "UNVERIFIED":
            raise EvidenceValidationError("observation candidate has an unsupported status")
        if str(item["evidence_id"]) not in linked_evidence_ids:
            raise EvidenceValidationError("observation candidate is not linked to finding evidence")
        observation_candidates.append(
            {
                "id": str(item["id"]),
                "evidence_id": str(item["evidence_id"]),
                "excerpt": str(item["excerpt"]),
                "description": str(item["description"]),
                "status": "UNVERIFIED",
                "citation": _render_report_citation(project, str(item["evidence_id"])),
            }
        )
    reviews = [
        HumanReviewPayload(
            id=item["id"],
            status=item["status"],
            note=item["note"],
            created_at=item["created_at"],
        )
        for item in project.finding_reviews(finding_id, limit=review_limit)
    ]
    review_total = (
        len(reviews)
        if review_limit is None
        else int(
            project.db.execute(
                "SELECT COUNT(*) FROM finding_reviews WHERE finding_id=?", (finding_id,)
            ).fetchone()[0]
        )
    )
    return {
        "finding": dict(finding),
        "coverage": _report_coverage_disclosure(project, finding["coverage_id"]),
        "evidence": evidence,
        "observation_candidates": observation_candidates,
        "observation_candidate_total": observation_candidate_total,
        "observation_candidates_truncated": (
            len(observation_candidates) < observation_candidate_total
        ),
        "evidence_total": evidence_total,
        "evidence_truncated": len(evidence) < evidence_total,
        "human_reviews": reviews,
        "human_review_total": review_total,
        "human_reviews_truncated": len(reviews) < review_total,
    }


def render_finding_markdown(project, finding_id: str) -> str:
    payload = finding_payload(project, finding_id)
    finding = payload["finding"]
    lines = [
        f"## {_markdown_text(finding['phenomenon'])}",
        "",
        f"**Observation:** {_markdown_text(finding['observation'])}",
    ]
    if finding["interpretation"]:
        lines.extend(["", f"**Interpretation:** {_markdown_text(finding['interpretation'])}"])
    if finding["alternative_interpretation"]:
        lines.extend(
            [
                "",
                f"**Alternative interpretation:** {_markdown_text(finding['alternative_interpretation'])}",
            ]
        )
    if payload["observation_candidates"]:
        lines.extend(["", "### Unverified observation candidates"])
        for candidate in payload["observation_candidates"]:
            lines.append(
                f"- **{_markdown_text(candidate['status'])} (evidence ID: "
                f"{_markdown_text(candidate['evidence_id'])}):** "
                f"{_markdown_text(candidate['description'])} Exact excerpt: "
                f"“{_markdown_text(candidate['excerpt'])}” — "
                f"{_markdown_text(candidate['citation'])}"
            )
    if payload["observation_candidates_truncated"]:
        lines.append(
            f"- Additional unverified observation candidates omitted; "
            f"{payload['observation_candidate_total']} total."
        )
    if finding["claim_types_json"] != "[]":
        lines.extend(["", f"**Claim types:** {_markdown_text(finding['claim_types_json'])}"])
    lines.extend([""])
    _append_coverage_markdown(lines, payload["coverage"])
    lines.extend(
        [
            "",
            f"**Evidence sufficiency:** {_markdown_text(finding['evidence_sufficiency'])}",
            f"**Interpretation confidence:** {_markdown_text(finding['interpretation_confidence'])}",
        ]
    )
    if finding["limitations_json"] != "[]":
        lines.extend(["", f"**Limitations:** {_markdown_text(finding['limitations_json'])}"])
    lines.extend(["", f"**Human review status:** {_markdown_text(finding['review_status'])}"])
    if payload["human_reviews"]:
        for review in payload["human_reviews"]:
            review_text = (
                f"**{_markdown_text(review['status'])}** ({_markdown_text(review['created_at'])})"
            )
            if review["note"]:
                review_text += f": {_markdown_text(review['note'])}"
            lines.append(f"- {review_text}")
    lines.extend(["", "### Evidence"])
    for item in payload["evidence"]:
        if item["text"]:
            role = _markdown_text(_evidence_role_label(item["role"]))
            lines.append(
                f"- **{role} (evidence ID: {_markdown_text(item['evidence_id'])}):** "
                f"{_markdown_text(item['text'])} — {_markdown_text(item['citation'])}"
                + (
                    f" — Link rationale: {_markdown_text(item['rationale'])}"
                    if item["rationale"]
                    else ""
                )
            )
    return "\n".join(lines) + "\n"


def render_finding_html(project, finding_id: str) -> str:
    """Render static, escaped HTML with model descriptions visibly UNVERIFIED."""
    payload = finding_payload(project, finding_id)
    finding = payload["finding"]
    parts = ["<!doctype html>", "<html><body>", f"<h2>{escape(str(finding['phenomenon']))}</h2>"]
    parts.append(f"<p><strong>Observation:</strong> {escape(str(finding['observation']))}</p>")
    if finding["interpretation"]:
        parts.append(
            f"<p><strong>Interpretation:</strong> {escape(str(finding['interpretation']))}</p>"
        )
    if finding["alternative_interpretation"]:
        parts.append(
            f"<p><strong>Alternative interpretation:</strong> {escape(str(finding['alternative_interpretation']))}</p>"
        )
    if payload["observation_candidates"]:
        parts.append("<h3>Unverified observation candidates</h3><ul>")
        for candidate in payload["observation_candidates"]:
            parts.append(
                f"<li><strong>{escape(candidate['status'])} (evidence ID: "
                f"{escape(candidate['evidence_id'])}):</strong> "
                f"{escape(candidate['description'])} Exact excerpt: “"
                f"{escape(candidate['excerpt'])}” — {escape(candidate['citation'])}</li>"
            )
        parts.append("</ul>")
    if payload["observation_candidates_truncated"]:
        parts.append(
            f"<p>Additional unverified observation candidates omitted; "
            f"{payload['observation_candidate_total']} total.</p>"
        )
    if finding["claim_types_json"] != "[]":
        parts.append(
            f"<p><strong>Claim types:</strong> {escape(str(finding['claim_types_json']))}</p>"
        )
    _append_coverage_html(parts, payload["coverage"])
    parts.append(
        f"<p><strong>Evidence sufficiency:</strong> {escape(str(finding['evidence_sufficiency']))}</p>"
    )
    parts.append(
        f"<p><strong>Interpretation confidence:</strong> {escape(str(finding['interpretation_confidence']))}</p>"
    )
    if finding["limitations_json"] != "[]":
        parts.append(
            f"<p><strong>Limitations:</strong> {escape(str(finding['limitations_json']))}</p>"
        )
    parts.append(
        f"<p><strong>Human review status:</strong> {escape(str(finding['review_status']))}</p>"
    )
    if payload["human_reviews"]:
        parts.append("<ul>")
        for review in payload["human_reviews"]:
            review_text = (
                f"<strong>{escape(str(review['status']))}</strong> "
                f"({escape(str(review['created_at']))})"
            )
            if review["note"]:
                review_text += f": {escape(str(review['note']))}"
            parts.append(f"<li>{review_text}</li>")
        parts.append("</ul>")
    parts.append("<h3>Evidence</h3><ul>")
    for item in payload["evidence"]:
        if item["text"]:
            role = escape(_evidence_role_label(item["role"]))
            parts.append(
                f"<li><strong>{role} (evidence ID: {escape(str(item['evidence_id']))}):</strong> "
                f"{escape(str(item['text']))} — "
                f"{escape(str(item['citation']))}"
                + (
                    f" — <strong>Link rationale:</strong> {escape(str(item['rationale']))}"
                    if item["rationale"]
                    else ""
                )
                + "</li>"
            )
    parts.extend(["</ul>", "</body></html>"])
    return "\n".join(parts) + "\n"


def analysis_payload(project, analysis_run_id: str) -> dict[str, Any]:
    """Return a secret-free, evidence-validated report payload for one immutable run."""
    try:
        summary = project.analysis_run_summary(analysis_run_id)
        run = summary["analysis_run"]
    except Exception as exc:
        if isinstance(exc, EvidenceValidationError):
            raise
        if isinstance(exc, DocumentError) and str(exc) == "stored model execution metadata is unsafe":
            raise EvidenceValidationError("model execution report metadata is invalid") from None
        raise EvidenceValidationError("analysis run provenance is invalid") from exc
    _validate_report_metadata(
        run.get("config"),
        "analysis run report configuration",
        allow_counterevidence_query_labels=True,
    )
    model_executions = [dict(item) for item in summary["model_executions"]]
    _validate_model_execution_metadata(model_executions)
    findings = project.db.execute(
        "SELECT id FROM findings WHERE analysis_run_id=? ORDER BY created_at,id", (analysis_run_id,)
    ).fetchall()
    finding_payloads = [finding_payload(project, item["id"]) for item in findings]
    skill_runs = [
        dict(item)
        for item in project.db.execute(
            "SELECT * FROM skill_runs WHERE analysis_run_id=? ORDER BY id", (analysis_run_id,)
        ).fetchall()
    ]
    retrieval_runs = [
        dict(item)
        for item in project.db.execute(
            "SELECT * FROM retrieval_runs WHERE analysis_run_id=? ORDER BY created_at,id",
            (analysis_run_id,),
        ).fetchall()
    ]
    coverage_records = (
        [{"coverage_id": item["finding"].get("coverage_id")} for item in finding_payloads]
        + skill_runs
        + retrieval_runs
    )
    coverage_disclosures = _report_coverage_disclosures(project, coverage_records)
    scopes = project.scope_revisions(run["scope_id"])
    revisions, _ = _report_scope_query(
        project,
        scopes,
        select_sql=(
            "SELECT id,document_id,source_hash,canonical_hash,extractor,mime_type,"
            "ocr_language,ocr_dpi FROM document_revisions "
            "WHERE id IN ({scope_ids}) ORDER BY id"
        ),
        order_key=lambda item: (str(item["id"]),),
    )
    annotations, _ = _report_scope_query(
        project,
        scopes,
        select_sql=(
            "SELECT annotations.id,annotations.memo,annotations.created_at,codes.label,"
            "codes.definition,codebooks.name AS codebook_name,source_spans.id AS span_id,"
            "source_spans.revision_id,source_spans.start_offset,source_spans.end_offset "
            "FROM annotations JOIN codes ON codes.id=annotations.code_id "
            "JOIN codebooks ON codebooks.id=codes.codebook_id "
            "JOIN source_spans ON source_spans.id=annotations.span_id "
            "WHERE source_spans.revision_id IN ({scope_ids}) "
            "ORDER BY source_spans.revision_id,source_spans.start_offset,"
            "annotations.created_at,annotations.id"
        ),
        order_key=lambda item: (
            str(item["revision_id"]),
            int(item["start_offset"]),
            str(item["created_at"]),
            str(item["id"]),
        ),
    )
    reranking_runs = [
        dict(item)
        for item in project.db.execute(
            "SELECT rr.* FROM reranking_runs rr JOIN retrieval_runs r "
            "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=? ORDER BY rr.created_at,rr.id",
            (analysis_run_id,),
        ).fetchall()
    ]
    reranking_hits = [
        dict(item)
        for item in project.db.execute(
            "SELECT rh.* FROM reranking_hits rh JOIN reranking_runs rr "
            "ON rr.id=rh.reranking_run_id JOIN retrieval_runs r "
            "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=? "
            "ORDER BY rr.created_at,rr.id,rh.rank",
            (analysis_run_id,),
        ).fetchall()
    ]
    _validate_report_metadata(reranking_runs, "reranking report metadata")
    _validate_report_metadata(reranking_hits, "reranking report metadata")
    return {
        "analysis_run": dict(run),
        "document_revisions": revisions,
        "findings": finding_payloads,
        "coverage_descriptors": coverage_disclosures,
        "finding_lineage": [
            dict(item)
            for item in project.db.execute(
                "SELECT l.* FROM finding_lineage l JOIN findings f ON f.id=l.finding_id "
                "WHERE f.analysis_run_id=? ORDER BY l.finding_id,l.source_finding_id",
                (analysis_run_id,),
            ).fetchall()
        ],
        "skill_runs": skill_runs,
        "model_executions": model_executions,
        "retrieval_runs": retrieval_runs,
        "retrieval_hits": [
            dict(item)
            for item in project.db.execute(
                "SELECT h.* FROM retrieval_hits h JOIN retrieval_runs r "
                "ON r.id=h.retrieval_run_id WHERE r.analysis_run_id=? "
                "ORDER BY r.created_at,r.id,h.rank,h.id",
                (analysis_run_id,),
            ).fetchall()
        ],
        "reranking_runs": reranking_runs,
        "reranking_hits": reranking_hits,
        "qualitative_coding": {
            "annotations": [
                {
                    **dict(item),
                    "text": project.span_text(item["span_id"]),
                    "citation": _render_report_span_location(project, item["span_id"]),
                }
                for item in annotations
            ],
            "memos": [dict(item) for item in project.research_memos(run["scope_id"])],
        },
    }


def analysis_payload_page(
    project, analysis_run_id: str, *, offset: int = 0, limit: int = 25
) -> dict[str, Any]:
    """Build a report page without loading unbounded finding/retrieval collections."""
    if offset < 0 or not 1 <= limit <= 100:
        raise EvidenceValidationError("report page requires non-negative offset and limit 1..100")
    metadata = project.analysis_run_metadata(analysis_run_id)
    run = metadata["analysis_run"]
    revision_ids = metadata["revision_ids"]
    if not isinstance(run, dict) or not isinstance(revision_ids, list) or not revision_ids:
        raise EvidenceValidationError("analysis run metadata is invalid")
    _validate_report_metadata(
        run.get("config"),
        "analysis run report configuration",
        allow_counterevidence_query_labels=True,
    )
    pagination: dict[str, dict[str, int | bool]] = {}

    def page_rows(
        key: str, count_sql: str, page_sql: str, params: tuple[object, ...]
    ) -> list[dict[str, Any]]:
        total = int(project.db.execute(count_sql, params).fetchone()[0])
        rows = project.db.execute(page_sql, (*params, limit, offset)).fetchall()
        pagination[key] = {
            "offset": offset,
            "limit": limit,
            "total_count": total,
            "has_more": offset + len(rows) < total,
        }
        return [dict(row) for row in rows]

    run_params = (analysis_run_id,)
    revisions, revision_total = _report_scope_query(
        project,
        revision_ids,
        count_sql="SELECT COUNT(*) FROM document_revisions WHERE id IN ({scope_ids})",
        select_sql=(
            "SELECT id,document_id,source_hash,canonical_hash,extractor,mime_type,"
            "ocr_language,ocr_dpi FROM document_revisions "
            "WHERE id IN ({scope_ids}) ORDER BY id"
        ),
        order_key=lambda item: (str(item["id"]),),
        offset=offset,
        limit=limit,
    )
    pagination["document_revisions"] = {
        "offset": offset,
        "limit": limit,
        "total_count": revision_total,
        "has_more": offset + len(revisions) < revision_total,
    }
    finding_count = int(
        project.db.execute(
            "SELECT COUNT(*) FROM findings WHERE analysis_run_id=?", run_params
        ).fetchone()[0]
    )
    finding_ids = project.db.execute(
        "SELECT id FROM findings WHERE analysis_run_id=? ORDER BY created_at,id LIMIT ? OFFSET ?",
        (*run_params, limit, offset),
    ).fetchall()
    pagination["findings"] = {
        "offset": offset,
        "limit": limit,
        "total_count": finding_count,
        "has_more": offset + len(finding_ids) < finding_count,
    }
    findings: list[dict[str, Any]] = []
    for item in finding_ids:
        finding_id = str(item["id"])
        finding_source = finding_payload(
            project,
            finding_id,
            evidence_limit=10,
            review_limit=10,
            observation_candidate_limit=20,
        )
        finding = dict(finding_source["finding"])
        evidence_rows = [dict(item) for item in finding_source["evidence"]]
        review_rows = [dict(item) for item in finding_source["human_reviews"]]
        payload: dict[str, Any] = {
            "finding": finding,
            "coverage": finding_source["coverage"],
            "evidence": evidence_rows,
            "observation_candidates": finding_source["observation_candidates"],
            "human_reviews": review_rows,
        }
        for field in (
            "observation",
            "interpretation",
            "alternative_interpretation",
            "limitations_json",
        ):
            value = finding.get(field)
            if isinstance(value, str) and len(value) > 2_000:
                finding[field] = value[:2_000]
                finding[f"{field}_truncated"] = True
        for evidence in evidence_rows:
            value = evidence.get("text")
            if isinstance(value, str) and len(value) > 1_000:
                evidence["text"] = value[:1_000]
                evidence["text_truncated"] = True
            else:
                evidence["text_truncated"] = False
        for review in review_rows:
            note = review.get("note")
            if isinstance(note, str) and len(note) > 1_000:
                review["note"] = note[:1_000]
                review["note_truncated"] = True
            else:
                review["note_truncated"] = False
        payload["evidence_total"] = finding_source["evidence_total"]
        payload["evidence_truncated"] = finding_source["evidence_truncated"]
        payload["human_review_total"] = finding_source["human_review_total"]
        payload["human_reviews_truncated"] = finding_source["human_reviews_truncated"]
        payload["observation_candidate_total"] = finding_source[
            "observation_candidate_total"
        ]
        payload["observation_candidates_truncated"] = finding_source[
            "observation_candidates_truncated"
        ]
        findings.append(payload)

    lineage = page_rows(
        "finding_lineage",
        "SELECT COUNT(*) FROM finding_lineage l JOIN findings f ON f.id=l.finding_id "
        "WHERE f.analysis_run_id=?",
        "SELECT l.* FROM finding_lineage l JOIN findings f ON f.id=l.finding_id "
        "WHERE f.analysis_run_id=? ORDER BY l.finding_id,l.source_finding_id LIMIT ? OFFSET ?",
        run_params,
    )
    skill_runs = page_rows(
        "skill_runs",
        "SELECT COUNT(*) FROM skill_runs WHERE analysis_run_id=?",
        "SELECT * FROM skill_runs WHERE analysis_run_id=? ORDER BY id LIMIT ? OFFSET ?",
        run_params,
    )
    model_executions = page_rows(
        "model_executions",
        "SELECT COUNT(*) FROM model_executions WHERE analysis_run_id=?",
        "SELECT * FROM model_executions WHERE analysis_run_id=? ORDER BY id LIMIT ? OFFSET ?",
        run_params,
    )
    _validate_model_execution_metadata(model_executions)
    for execution in model_executions:
        try:
            project.validate_model_execution_skill_run(
                analysis_run_id, execution.get("skill_run_id")
            )
        except DocumentError:
            raise EvidenceValidationError(
                "model execution report provenance is invalid"
            ) from None
    retrieval_runs = page_rows(
        "retrieval_runs",
        "SELECT COUNT(*) FROM retrieval_runs WHERE analysis_run_id=?",
        "SELECT * FROM retrieval_runs WHERE analysis_run_id=? ORDER BY created_at,id LIMIT ? OFFSET ?",
        run_params,
    )
    coverage_records = (
        [{"coverage_id": item["finding"].get("coverage_id")} for item in findings]
        + skill_runs
        + retrieval_runs
    )
    coverage_disclosures = _report_coverage_disclosures(project, coverage_records)
    retrieval_hits = page_rows(
        "retrieval_hits",
        "SELECT COUNT(*) FROM retrieval_hits h JOIN retrieval_runs r "
        "ON r.id=h.retrieval_run_id WHERE r.analysis_run_id=?",
        "SELECT h.* FROM retrieval_hits h JOIN retrieval_runs r "
        "ON r.id=h.retrieval_run_id WHERE r.analysis_run_id=? "
        "ORDER BY r.created_at,r.id,h.rank,h.id LIMIT ? OFFSET ?",
        run_params,
    )
    reranking_runs = page_rows(
        "reranking_runs",
        "SELECT COUNT(*) FROM reranking_runs rr JOIN retrieval_runs r "
        "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=?",
        "SELECT rr.* FROM reranking_runs rr JOIN retrieval_runs r "
        "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=? "
        "ORDER BY rr.created_at,rr.id LIMIT ? OFFSET ?",
        run_params,
    )
    _validate_report_metadata(reranking_runs, "reranking report metadata")
    reranking_hits = page_rows(
        "reranking_hits",
        "SELECT COUNT(*) FROM reranking_hits rh JOIN reranking_runs rr "
        "ON rr.id=rh.reranking_run_id JOIN retrieval_runs r "
        "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=?",
        "SELECT rh.* FROM reranking_hits rh JOIN reranking_runs rr "
        "ON rr.id=rh.reranking_run_id JOIN retrieval_runs r "
        "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=? "
        "ORDER BY rr.created_at,rr.id,rh.rank LIMIT ? OFFSET ?",
        run_params,
    )
    _validate_report_metadata(reranking_hits, "reranking report metadata")
    annotations, annotation_total = _report_scope_query(
        project,
        revision_ids,
        count_sql=(
            "SELECT COUNT(*) FROM annotations JOIN source_spans "
            "ON source_spans.id=annotations.span_id "
            "WHERE source_spans.revision_id IN ({scope_ids})"
        ),
        select_sql=(
            "SELECT annotations.id,annotations.memo,annotations.created_at,codes.label,"
            "codes.definition,codebooks.name AS codebook_name,source_spans.id AS span_id,"
            "source_spans.revision_id,source_spans.start_offset,source_spans.end_offset "
            "FROM annotations JOIN codes ON codes.id=annotations.code_id "
            "JOIN codebooks ON codebooks.id=codes.codebook_id "
            "JOIN source_spans ON source_spans.id=annotations.span_id "
            "WHERE source_spans.revision_id IN ({scope_ids}) "
            "ORDER BY source_spans.revision_id,source_spans.start_offset,"
            "annotations.created_at,annotations.id"
        ),
        order_key=lambda item: (
            str(item["revision_id"]),
            int(item["start_offset"]),
            str(item["created_at"]),
            str(item["id"]),
        ),
        offset=offset,
        limit=limit,
    )
    pagination["annotations"] = {
        "offset": offset,
        "limit": limit,
        "total_count": annotation_total,
        "has_more": offset + len(annotations) < annotation_total,
    }
    for annotation in annotations:
        annotation["text"] = project.span_text(annotation["span_id"])
        annotation["citation"] = _render_report_span_location(project, annotation["span_id"])
    scope_id = str(run["scope_id"])
    memos = page_rows(
        "memos",
        "SELECT COUNT(*) FROM research_memos WHERE scope_id=?",
        "SELECT * FROM research_memos WHERE scope_id=? ORDER BY created_at,id LIMIT ? OFFSET ?",
        (scope_id,),
    )
    for memo in memos:
        content = memo.get("content")
        if isinstance(content, str) and len(content) > 2_000:
            memo["content"] = content[:2_000]
            memo["content_truncated"] = True
        else:
            memo["content_truncated"] = False

    page = {
        "analysis_run": run,
        "document_revisions": revisions,
        "findings": findings,
        "coverage_descriptors": coverage_disclosures,
        "finding_lineage": lineage,
        "skill_runs": skill_runs,
        "model_executions": model_executions,
        "retrieval_runs": retrieval_runs,
        "retrieval_hits": retrieval_hits,
        "reranking_runs": reranking_runs,
        "reranking_hits": reranking_hits,
        "qualitative_coding": {"annotations": annotations, "memos": memos},
        "pagination": pagination,
    }
    bounded_page, _ = _cap_report_strings(page)
    return bounded_page


def render_analysis_markdown(project, analysis_run_id: str) -> str:
    return render_analysis_markdown_payload(analysis_payload(project, analysis_run_id))


def render_analysis_markdown_payload(payload: dict[str, Any]) -> str:
    """Render the standard Markdown view from a full or bounded report payload."""
    if _contains_suspicious_credential_alias(payload):
        raise EvidenceValidationError(
            "analysis report metadata contains a suspicious credential alias"
        )
    if _contains_suspicious_provider_endpoint(payload):
        raise EvidenceValidationError(
            "analysis report metadata contains credential-like provider endpoint metadata"
        )
    run = payload["analysis_run"]
    lines = ["# DiscourseLens analysis report", "", "## Method", ""]
    lines.append(f"- Run: {_markdown_text(run['id'])}")
    lines.append(f"- Status: {_markdown_text(run['status'])}")
    lines.append(
        f"- Configuration: {_markdown_text(json.dumps(run['config'], ensure_ascii=False, sort_keys=True))}"
    )
    lines.extend(["", "## Findings"])
    for item in payload["findings"]:
        finding = item["finding"]
        lines.extend(
            [
                "",
                f"### {_markdown_text(finding['phenomenon'])}",
                "",
                f"Observation: {_markdown_report_field(finding, 'observation')}",
            ]
        )
        if finding["interpretation"]:
            lines.extend(
                ["", f"Interpretation: {_markdown_report_field(finding, 'interpretation')}"]
            )
        if finding["alternative_interpretation"]:
            lines.extend(
                [
                    "",
                    f"Alternative: {_markdown_report_field(finding, 'alternative_interpretation')}",
                ]
            )
        observation_candidates = item.get("observation_candidates", [])
        if observation_candidates:
            lines.extend(["", "Unverified observation candidates:"])
            for candidate in observation_candidates:
                lines.append(
                    f"- {_markdown_text(candidate['status'])} (evidence ID: "
                    f"{_markdown_text(candidate['evidence_id'])}): "
                    f"{_markdown_report_field(candidate, 'description')} Exact excerpt: “"
                    f"{_markdown_report_field(candidate, 'excerpt')}” — "
                    f"{_markdown_report_field(candidate, 'citation')}"
                )
        if item.get("observation_candidates_truncated") is True:
            lines.append(
                "- Additional unverified observation candidates omitted; "
                f"{_markdown_text(str(item.get('observation_candidate_total', 'unknown')))} total."
            )
        if finding["claim_types_json"] != "[]":
            lines.extend(
                ["", f"Claim types: {_markdown_report_field(finding, 'claim_types_json')}"]
            )
        coverage = item.get("coverage")
        if isinstance(coverage, dict):
            lines.append("")
            _append_coverage_markdown(lines, coverage)
        lines.extend(
            [
                "",
                f"Evidence sufficiency: {_markdown_report_field(finding, 'evidence_sufficiency')}",
                f"Interpretation confidence: {_markdown_report_field(finding, 'interpretation_confidence')}",
            ]
        )
        if finding["limitations_json"] != "[]":
            lines.append(f"Limitations: {_markdown_report_field(finding, 'limitations_json')}")
        lines.append(f"Human review status: {_markdown_text(finding['review_status'])}")
        if item["human_reviews"]:
            for review in item["human_reviews"]:
                review_text = (
                    f"**{_markdown_text(review['status'])}** "
                    f"({_markdown_text(review['created_at'])})"
                )
                if review["note"]:
                    review_text += f": {_markdown_report_field(review, 'note')}"
                lines.append(f"- {review_text}")
        if item.get("human_reviews_truncated"):
            lines.append(
                "- Human review list is partial: showing "
                f"{_markdown_text(str(len(item['human_reviews'])))} of "
                f"{_markdown_text(str(item.get('human_review_total', len(item['human_reviews']))))}; "
                "additional reviews omitted."
            )
        for evidence in item["evidence"]:
            if evidence["text"]:
                role = _markdown_text(_evidence_role_label(evidence["role"]))
                lines.append(
                    f"- **{role} (evidence ID: "
                    f"{_markdown_text(evidence['evidence_id'])}):** "
                    f"{_markdown_report_field(evidence, 'text')} — "
                    f"{_markdown_report_field(evidence, 'citation')}"
                    + (
                        f" — Link rationale: {_markdown_report_field(evidence, 'rationale')}"
                        if evidence.get("rationale")
                        else ""
                    )
                )
        if item.get("evidence_truncated"):
            lines.append(
                "- Evidence list is partial: showing "
                f"{_markdown_text(str(len(item['evidence'])))} of "
                f"{_markdown_text(str(item.get('evidence_total', len(item['evidence']))))}; "
                "additional evidence omitted."
            )
    finding_coverage_ids = {
        item["coverage"].get("coverage_id")
        for item in payload["findings"]
        if isinstance(item.get("coverage"), dict)
        and item["coverage"].get("coverage_id") is not None
    }
    other_coverage = [
        coverage
        for coverage in payload.get("coverage_descriptors", [])
        if isinstance(coverage, dict) and coverage.get("coverage_id") not in finding_coverage_ids
    ]
    if other_coverage:
        lines.extend(["", "## Other coverage references", ""])
        for coverage in other_coverage:
            _append_coverage_markdown(lines, coverage)
    coding = payload["qualitative_coding"]
    if coding["annotations"] or coding["memos"]:
        lines.extend(["", "## Qualitative coding"])
        for annotation in coding["annotations"]:
            lines.append(
                f"- {_markdown_report_field(annotation, 'codebook_name')} / "
                f"{_markdown_report_field(annotation, 'label')}: "
                f"{_markdown_report_field(annotation, 'text')} — "
                f"{_markdown_report_field(annotation, 'citation')}"
            )
        for memo in coding["memos"]:
            lines.append(
                f"- Memo: {_markdown_report_field(memo, 'title')} — "
                f"{_markdown_report_field(memo, 'content')}"
            )
    lines.extend(["", "## Technical appendix", ""])
    lines.append(f"- DiscourseLens version: {_markdown_text(run['software_version'])}")
    lines.append(f"- Project format version: {_markdown_text(run['project_format_version'])}")
    lines.append(f"- Database schema version: {_markdown_text(run['database_schema_version'])}")
    lines.append(f"- Configuration SHA-256: {_markdown_text(run['config_hash'])}")
    prompt_provenance = _controlled_cda_prompt_provenance(run.get("config"))
    if prompt_provenance is not None:
        lines.extend(
            [
                "",
                "- Built-in prompt provenance (refs, versions, and SHA-256 digests; prompt bodies omitted):",
                "",
                "```json",
                json.dumps(prompt_provenance, ensure_ascii=False, sort_keys=True, indent=2),
                "```",
            ]
        )
    lines.append(
        f"- Revisions: {_markdown_text(json.dumps(payload['document_revisions'], ensure_ascii=False))}"
    )
    lines.append(
        f"- Skill runs: {_markdown_text(json.dumps(payload['skill_runs'], ensure_ascii=False))}"
    )
    lines.append(
        "- Model executions (provider/model, credential alias, reasoning, usage, and hashes; "
        "no secret values): "
        f"{_markdown_text(json.dumps(payload['model_executions'], ensure_ascii=False))}"
    )
    lines.append(
        "- Retrieval runs and hits: "
        f"{_markdown_text(json.dumps({'runs': payload['retrieval_runs'], 'hits': payload['retrieval_hits']}, ensure_ascii=False))}"
    )
    lines.append(
        "- Reranking runs and hits: "
        f"{_markdown_text(json.dumps({'runs': payload['reranking_runs'], 'hits': payload['reranking_hits']}, ensure_ascii=False))}"
    )
    lines.append(
        "- Finding lineage: "
        f"{_markdown_text(json.dumps(payload['finding_lineage'], ensure_ascii=False))}"
    )
    pagination = payload.get("pagination")
    if isinstance(pagination, dict):
        partial_collections: list[str] = []
        for collection, page in pagination.items():
            if not isinstance(collection, str) or not isinstance(page, dict):
                continue
            offset = page.get("offset")
            limit = page.get("limit")
            total = page.get("total_count")
            has_more = page.get("has_more") is True
            if type(offset) is not int or type(limit) is not int or type(total) is not int:
                continue
            if offset == 0 and not has_more:
                continue
            omitted = []
            if offset > 0:
                omitted.append("earlier entries omitted")
            if has_more:
                omitted.append("additional entries omitted")
            partial_collections.append(
                f"- {_markdown_text(collection.replace('_', ' '))}: page offset "
                f"{offset}, limit {limit}, total {total}; {'; '.join(omitted)}."
            )
        if partial_collections:
            lines.extend(["", "## Page completeness", "", *partial_collections])
    return "\n".join(lines) + "\n"


def render_analysis_html(project, analysis_run_id: str) -> str:
    """Render an immutable run with escaped, separately marked candidate descriptions."""
    payload = analysis_payload(project, analysis_run_id)
    run = payload["analysis_run"]
    parts = ["<!doctype html>", "<html><body>", "<h1>DiscourseLens analysis report</h1>"]
    parts.extend(
        [
            "<h2>Method</h2>",
            f"<p>Run: <code>{escape(str(run['id']))}</code></p>",
            f"<p>Status: {escape(str(run['status']))}</p>",
            f"<pre>{escape(json.dumps(run['config'], ensure_ascii=False, sort_keys=True))}</pre>",
            "<h2>Findings</h2>",
        ]
    )
    other_coverage = [
        coverage
        for coverage in payload.get("coverage_descriptors", [])
        if isinstance(coverage, dict)
        and coverage.get("coverage_id")
        not in {
            item["coverage"].get("coverage_id")
            for item in payload["findings"]
            if isinstance(item.get("coverage"), dict)
        }
    ]
    if other_coverage:
        parts.append("<h2>Other coverage references</h2>")
        for coverage in other_coverage:
            _append_coverage_html(parts, coverage)
    for item in payload["findings"]:
        finding = item["finding"]
        parts.append(f"<h3>{escape(str(finding['phenomenon']))}</h3>")
        parts.append(
            f"<p><strong>Observation:</strong> {_html_report_field(finding, 'observation')}</p>"
        )
        if finding["interpretation"]:
            parts.append(
                f"<p><strong>Interpretation:</strong> {_html_report_field(finding, 'interpretation')}</p>"
            )
        if finding["alternative_interpretation"]:
            parts.append(
                "<p><strong>Alternative interpretation:</strong> "
                f"{_html_report_field(finding, 'alternative_interpretation')}</p>"
            )
        observation_candidates = item.get("observation_candidates", [])
        if observation_candidates:
            parts.append("<h4>Unverified observation candidates</h4><ul>")
            for candidate in observation_candidates:
                parts.append(
                    f"<li>{_html_report_field(candidate, 'status')} (evidence ID: "
                    f"{escape(str(candidate['evidence_id']))}): "
                    f"{_html_report_field(candidate, 'description')} Exact excerpt: “"
                    f"{_html_report_field(candidate, 'excerpt')}” — "
                    f"{_html_report_field(candidate, 'citation')}</li>"
                )
            parts.append("</ul>")
        if item.get("observation_candidates_truncated") is True:
            parts.append(
                "<p>Additional unverified observation candidates omitted; "
                f"{escape(str(item.get('observation_candidate_total', 'unknown')))} total.</p>"
            )
        if finding["claim_types_json"] != "[]":
            parts.append(
                f"<p><strong>Claim types:</strong> {_html_report_field(finding, 'claim_types_json')}</p>"
            )
        coverage = item.get("coverage")
        if isinstance(coverage, dict):
            _append_coverage_html(parts, coverage)
        parts.append(
            f"<p><strong>Evidence sufficiency:</strong> {_html_report_field(finding, 'evidence_sufficiency')}</p>"
        )
        parts.append(
            f"<p><strong>Interpretation confidence:</strong> {_html_report_field(finding, 'interpretation_confidence')}</p>"
        )
        if finding["limitations_json"] != "[]":
            parts.append(
                f"<p><strong>Limitations:</strong> {_html_report_field(finding, 'limitations_json')}</p>"
            )
        parts.append(
            f"<p><strong>Human review status:</strong> {escape(str(finding['review_status']))}</p>"
        )
        if item["human_reviews"]:
            parts.append("<ul>")
            for review in item["human_reviews"]:
                review_text = (
                    f"<strong>{escape(str(review['status']))}</strong> "
                    f"({escape(str(review['created_at']))})"
                )
                if review["note"]:
                    review_text += f": {_html_report_field(review, 'note')}"
                parts.append(f"<li>{review_text}</li>")
            parts.append("</ul>")
        parts.append("<ul>")
        for evidence in item["evidence"]:
            if evidence["text"]:
                role = escape(_evidence_role_label(evidence["role"]))
                parts.append(
                    f"<li><strong>{role} (evidence ID: "
                    f"{escape(str(evidence['evidence_id']))}):</strong> "
                    f"{_html_report_field(evidence, 'text')} — "
                    f"{_html_report_field(evidence, 'citation')}"
                    + (
                        f" — <strong>Link rationale:</strong> "
                        f"{_html_report_field(evidence, 'rationale')}"
                        if evidence.get("rationale")
                        else ""
                    )
                    + "</li>"
                )
        parts.append("</ul>")
    coding = payload["qualitative_coding"]
    if coding["annotations"] or coding["memos"]:
        parts.append("<h2>Qualitative coding</h2><ul>")
        for annotation in coding["annotations"]:
            parts.append(
                f"<li>{_html_report_field(annotation, 'codebook_name')} / "
                f"{_html_report_field(annotation, 'label')}: "
                f"{_html_report_field(annotation, 'text')} — "
                f"{_html_report_field(annotation, 'citation')}</li>"
            )
        for memo in coding["memos"]:
            parts.append(
                f"<li>Memo: <strong>{_html_report_field(memo, 'title')}</strong> — "
                f"{_html_report_field(memo, 'content')}</li>"
            )
        parts.append("</ul>")
    parts.extend(
        [
            "<h2>Technical appendix</h2>",
            f"<p>DiscourseLens version: {escape(str(run['software_version']))}</p>",
            f"<p>Project format version: {escape(str(run['project_format_version']))}</p>",
            f"<p>Database schema version: {escape(str(run['database_schema_version']))}</p>",
            f"<p>Configuration SHA-256: {escape(str(run['config_hash']))}</p>",
            f"<pre>{escape(json.dumps(payload['document_revisions'], ensure_ascii=False))}</pre>",
            f"<pre>{escape(json.dumps(payload['skill_runs'], ensure_ascii=False))}</pre>",
            f"<pre>{escape(json.dumps(payload['model_executions'], ensure_ascii=False))}</pre>",
            f"<pre>{escape(json.dumps({'retrieval_runs': payload['retrieval_runs'], 'retrieval_hits': payload['retrieval_hits']}, ensure_ascii=False))}</pre>",
            f"<pre>{escape(json.dumps({'reranking_runs': payload['reranking_runs'], 'reranking_hits': payload['reranking_hits']}, ensure_ascii=False))}</pre>",
            f"<pre>{escape(json.dumps(payload['finding_lineage'], ensure_ascii=False))}</pre>",
        ]
    )
    prompt_provenance = _controlled_cda_prompt_provenance(run.get("config"))
    if prompt_provenance is not None:
        parts.extend(
            [
                "<h3>Built-in prompt provenance</h3>",
                "<p>Prompt references, versions, and SHA-256 digests; prompt bodies omitted.</p>",
                f"<pre>{escape(json.dumps(prompt_provenance, ensure_ascii=False, sort_keys=True, indent=2))}</pre>",
            ]
        )
    parts.extend(["</body></html>"])
    return "\n".join(parts) + "\n"


def write_analysis_report(
    project, analysis_run_id: str, destination: str | Path, output_format: str
) -> Path:
    """Write a deterministic, local report from evidence-validated run records.

    PDF and DOCX exports intentionally render the same Core-generated Markdown
    body rather than accepting template content from an LLM or an arbitrary
    external document. The caller controls the destination; no network or secret
    value is involved.
    """
    destination = Path(destination)
    if destination.exists() and destination.is_dir():
        raise EvidenceValidationError("report destination must be a file")
    if output_format == "markdown":
        content = render_analysis_markdown(project, analysis_run_id)
        writer = partial(_write_report_text, content=content)
    elif output_format == "html":
        content = render_analysis_html(project, analysis_run_id)
        writer = partial(_write_report_text, content=content)
    elif output_format == "json":
        content = json.dumps(
            analysis_payload(project, analysis_run_id), ensure_ascii=False, indent=2
        )
        writer = partial(_write_report_text, content=content)
    elif output_format == "pdf":
        markdown = render_analysis_markdown(project, analysis_run_id)
        writer = partial(_write_pdf, markdown=markdown)
    elif output_format == "docx":
        markdown = render_analysis_markdown(project, analysis_run_id)
        writer = partial(_write_docx, markdown=markdown)
    else:
        raise EvidenceValidationError("report format must be markdown, html, json, pdf, or docx")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_report_atomically(destination, writer)
    return destination


def _write_report_atomically(destination: Path, writer: Callable[[Path], object]) -> None:
    """Stage a report privately, then replace the destination without following symlinks."""
    with tempfile.TemporaryDirectory(
        prefix=".discourselens-report-", dir=destination.parent
    ) as temporary_directory:
        staged = Path(temporary_directory) / destination.name
        writer(staged)
        os.chmod(staged, 0o600)
        os.replace(staged, destination)


def _write_report_text(destination: Path, *, content: str) -> int:
    return destination.write_text(content, encoding="utf-8")


def _write_pdf(destination: Path, markdown: str) -> None:
    """Create a simple paginated PDF without invoking browser or shell tooling."""
    document = fitz.open()
    try:
        page = document.new_page()
        width, height = page.rect.width, page.rect.height
        margin = 42.0
        y = margin
        for raw in markdown.splitlines():
            heading_level = len(raw) - len(raw.lstrip("#")) if raw.startswith("#") else 0
            text = _unescape_markdown_text(raw[heading_level:].strip() if heading_level else raw)
            size = 16.0 if heading_level == 1 else 13.0 if heading_level == 2 else 10.0
            leading = size + 4.0
            lines = wrap(text, width=98, break_long_words=True, break_on_hyphens=False) or [""]
            for line in lines:
                if y + leading > height - margin:
                    page = document.new_page(width=width, height=height)
                    y = margin
                page.insert_text((margin, y), line, fontsize=size, fontname="helv")
                y += leading
            if not text:
                y += 4.0
        document.save(destination)
    finally:
        document.close()


def _xml_safe_text(value: str) -> str:
    return "".join(
        character
        for character in value
        if (
            character in "\t\n\r"
            or 0x20 <= ord(character) <= 0xD7FF
            or 0xE000 <= ord(character) <= 0xFFFD
            or 0x10000 <= ord(character) <= 0x10FFFF
        )
    )


def _write_docx(destination: Path, markdown: str) -> None:
    """Write a minimal standards-based DOCX with escaped, local report text."""
    paragraphs = []
    for line in markdown.splitlines():
        text = _unescape_markdown_text(line.lstrip("#").strip() if line.startswith("#") else line)
        escaped = xml_escape(_xml_safe_text(text))
        paragraphs.append(f'<w:p><w:r><w:t xml:space="preserve">{escaped}</w:t></w:r></w:p>')
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(paragraphs)}<w:sectPr/></w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    with ZipFile(destination, "w", ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", content_types)
        package.writestr("_rels/.rels", relationships)
        package.writestr("word/document.xml", document_xml)


def comparison_payload(project, comparison_id: str) -> dict[str, Any]:
    """Return an integrity-checked, renderer-neutral corpus comparison record."""
    row = project.comparison_run(comparison_id)
    try:
        config = json.loads(row["config_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError("comparison result is corrupt") from exc
    if not isinstance(config, dict):
        raise EvidenceValidationError("comparison result has invalid structure")
    _validate_report_metadata(config, "comparison report configuration")
    if row["storage_kind"] == "rows-v1":
        stored = project.comparison_rows_all(comparison_id)
        rows = cast(list[dict[str, Any]], stored["rows"])
        result_hash_verified = bool(stored["result_hash_verified"])
    else:
        try:
            rows = json.loads(row["result_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise EvidenceValidationError("comparison result is corrupt") from exc
        if not isinstance(rows, list):
            raise EvidenceValidationError("comparison result has invalid structure")
        result_hash = hashlib.sha256(row["result_json"].encode("utf-8")).hexdigest()
        if result_hash != row["result_hash"]:
            raise EvidenceValidationError("comparison result hash mismatch")
        result_hash_verified = True
    left_total = sum(
        int(item["left_count"])
        for item in rows
        if isinstance(item, dict) and isinstance(item.get("left_count"), int)
    )
    right_total = sum(
        int(item["right_count"])
        for item in rows
        if isinstance(item, dict) and isinstance(item.get("right_count"), int)
    )
    return {
        "comparison_run": {
            **dict(row),
            "config": config,
            "result_hash_verified": result_hash_verified,
        },
        "left_scope": {
            "id": row["left_scope_id"],
            "revision_ids": project.scope_revisions(row["left_scope_id"]),
            "surface_token_population": left_total,
        },
        "right_scope": {
            "id": row["right_scope_id"],
            "revision_ids": project.scope_revisions(row["right_scope_id"]),
            "surface_token_population": right_total,
        },
        "rows": rows,
        "limitations": [
            "The comparison is descriptive deterministic surface-token statistics.",
            "It does not by itself establish causation, intent, ideology, or semantic equivalence.",
        ],
    }


def comparison_payload_page(
    project, comparison_id: str, *, offset: int = 0, limit: int = 100
) -> dict[str, Any]:
    """Verify and page normalized comparison rows or a legacy JSON result."""
    if offset < 0 or not 1 <= limit <= 1_000:
        raise EvidenceValidationError(
            "comparison page requires non-negative offset and limit 1..1000"
        )
    row = project.db.execute(
        "SELECT id,left_scope_id,right_scope_id,metric,calculation_version,config_json,"
        "result_hash,created_at,storage_kind,length(result_json) AS result_char_count "
        "FROM comparison_runs WHERE id=?",
        (comparison_id,),
    ).fetchone()
    if not row:
        raise EvidenceValidationError("unknown comparison run")
    try:
        config = json.loads(row["config_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError("comparison configuration is corrupt") from exc
    if not isinstance(config, dict):
        raise EvidenceValidationError("comparison configuration has invalid structure")
    _validate_report_metadata(config, "comparison report configuration")

    if row["storage_kind"] == "rows-v1":
        stored = project.comparison_rows_page(comparison_id, offset=offset, limit=limit)
        rows = cast(list[dict[str, Any]], stored["rows"])
        left_revision_ids = project.scope_revisions(row["left_scope_id"])
        right_revision_ids = project.scope_revisions(row["right_scope_id"])
        total_rows = int(stored["total_rows"])
        return {
            "comparison_run": {
                **dict(row),
                "config": config,
                "result_hash_verified": stored["result_hash_verified"],
            },
            "left_scope": {
                "id": row["left_scope_id"],
                "revision_ids": left_revision_ids[offset : offset + limit],
                "revision_count": len(left_revision_ids),
                "surface_token_population": stored["left_population"],
            },
            "right_scope": {
                "id": row["right_scope_id"],
                "revision_ids": right_revision_ids[offset : offset + limit],
                "revision_count": len(right_revision_ids),
                "surface_token_population": stored["right_population"],
            },
            "rows": rows,
            "pagination": {
                "rows": {
                    "offset": offset,
                    "limit": limit,
                    "total_count": total_rows,
                    "has_more": offset + len(rows) < total_rows,
                },
                "left_revisions": {
                    "offset": offset,
                    "limit": limit,
                    "total_count": len(left_revision_ids),
                    "has_more": offset + limit < len(left_revision_ids),
                },
                "right_revisions": {
                    "offset": offset,
                    "limit": limit,
                    "total_count": len(right_revision_ids),
                    "has_more": offset + limit < len(right_revision_ids),
                },
            },
            "limitations": [
                "The comparison is descriptive deterministic surface-token statistics.",
                "It does not by itself establish causation, intent, ideology, or semantic equivalence.",
            ],
        }

    validation = project.db.execute(
        "SELECT json_valid(result_json), "
        "CASE WHEN json_valid(result_json) THEN json_type(result_json) END "
        "FROM comparison_runs WHERE id=?",
        (comparison_id,),
    ).fetchone()
    if not validation or validation[0] != 1 or validation[1] != "array":
        raise EvidenceValidationError("comparison result has invalid structure")

    result_length = int(row["result_char_count"])
    result_digest = hashlib.sha256()
    chunk_size = 64_000
    for start in range(1, result_length + 1, chunk_size):
        chunk = project.db.execute(
            "SELECT substr(result_json,?,?) FROM comparison_runs WHERE id=?",
            (start, chunk_size, comparison_id),
        ).fetchone()[0]
        if not isinstance(chunk, str):
            raise EvidenceValidationError("comparison result is unreadable")
        result_digest.update(chunk.encode("utf-8"))
    if result_digest.hexdigest() != row["result_hash"]:
        raise EvidenceValidationError("comparison result hash mismatch")

    try:
        totals = project.db.execute(
            "SELECT COUNT(*), "
            "COALESCE(SUM(CASE WHEN json_type(j.value,'$.left_count')='integer' "
            "THEN CAST(json_extract(j.value,'$.left_count') AS INTEGER) ELSE 0 END),0), "
            "COALESCE(SUM(CASE WHEN json_type(j.value,'$.right_count')='integer' "
            "THEN CAST(json_extract(j.value,'$.right_count') AS INTEGER) ELSE 0 END),0) "
            "FROM comparison_runs c,json_each(c.result_json) j WHERE c.id=?",
            (comparison_id,),
        ).fetchone()
        page_rows = project.db.execute(
            "SELECT j.value FROM comparison_runs c,json_each(c.result_json) j "
            "WHERE c.id=? ORDER BY CAST(j.key AS INTEGER) LIMIT ? OFFSET ?",
            (comparison_id, limit, offset),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise EvidenceValidationError(
            "comparison result cannot be read as ranked JSON rows"
        ) from exc

    total_rows, left_population, right_population = map(int, totals)
    rows: list[dict[str, Any]] = []
    for item in page_rows:
        try:
            parsed = json.loads(item[0])
        except (TypeError, json.JSONDecodeError) as exc:
            raise EvidenceValidationError("comparison result row is corrupt") from exc
        if not isinstance(parsed, dict):
            raise EvidenceValidationError("comparison result row has invalid structure")
        token = parsed.get("token")
        if isinstance(token, str) and len(token) > 2_000:
            parsed["token"] = token[:2_000]
            parsed["token_truncated"] = True
        rows.append(parsed)

    left_revision_ids = project.scope_revisions(row["left_scope_id"])
    right_revision_ids = project.scope_revisions(row["right_scope_id"])
    return {
        "comparison_run": {
            **dict(row),
            "config": config,
            "result_hash_verified": True,
        },
        "left_scope": {
            "id": row["left_scope_id"],
            "revision_ids": left_revision_ids[offset : offset + limit],
            "revision_count": len(left_revision_ids),
            "surface_token_population": left_population,
        },
        "right_scope": {
            "id": row["right_scope_id"],
            "revision_ids": right_revision_ids[offset : offset + limit],
            "revision_count": len(right_revision_ids),
            "surface_token_population": right_population,
        },
        "rows": rows,
        "pagination": {
            "rows": {
                "offset": offset,
                "limit": limit,
                "total_count": total_rows,
                "has_more": offset + len(rows) < total_rows,
            },
            "left_revisions": {
                "offset": offset,
                "limit": limit,
                "total_count": len(left_revision_ids),
                "has_more": offset + limit < len(left_revision_ids),
            },
            "right_revisions": {
                "offset": offset,
                "limit": limit,
                "total_count": len(right_revision_ids),
                "has_more": offset + limit < len(right_revision_ids),
            },
        },
        "limitations": [
            "The comparison is descriptive deterministic surface-token statistics.",
            "It does not by itself establish causation, intent, ideology, or semantic equivalence.",
        ],
    }


def render_comparison_markdown(
    project, comparison_id: str, *, limit: int = 25, offset: int = 0
) -> str:
    """Render one integrity-checked comparison page without loading all ranked rows."""
    if offset < 0 or not 1 <= limit <= 1_000:
        raise EvidenceValidationError(
            "comparison report requires non-negative offset and limit 1..1000"
        )
    payload = comparison_payload_page(project, comparison_id, offset=offset, limit=limit)
    comparison = payload["comparison_run"]
    left = payload["left_scope"]
    right = payload["right_scope"]
    lines = ["# DiscourseLens corpus comparison", "", "## Method", ""]
    lines.extend(
        [
            f"- Comparison run: {_markdown_text(comparison['id'])}",
            f"- Metric: {_markdown_text(comparison['metric'])}",
            f"- Calculation version: {_markdown_text(comparison['calculation_version'])}",
            (
                f"- Left scope: {_markdown_text(left['id'])} ({left['revision_count']} revisions; "
                f"{left['surface_token_population']} surface-token positions)"
            ),
            (
                f"- Right scope: {_markdown_text(right['id'])} ({right['revision_count']} revisions; "
                f"{right['surface_token_population']} surface-token positions)"
            ),
            f"- Result hash: {_markdown_text(comparison['result_hash'])}",
            "",
            "## Ranked descriptive results",
            "",
        ]
    )
    page = payload["pagination"]["rows"]
    total_rows = int(page["total_count"])
    if payload["rows"]:
        first = offset + 1
        last = offset + len(payload["rows"])
        lines.extend([f"Showing rows {first}–{last} of {total_rows}.", ""])
    else:
        lines.extend([f"No rows at offset {offset}; {total_rows} rows total.", ""])
    for item in payload["rows"]:
        if not isinstance(item, dict):
            continue
        statistic = (
            _markdown_text(f"log-likelihood={item.get('log_likelihood')}")
            if comparison["metric"] == "keyness"
            else _markdown_text(f"rate difference={item.get('rate_difference')}")
        )
        lines.append(
            f"- {_markdown_text(item.get('token', '?'))}: left={_markdown_text(item.get('left_count'))}, "
            f"right={_markdown_text(item.get('right_count'))}; {statistic}"
        )
    if page["has_more"]:
        lines.extend(
            ["", f"More rows are available; request offset {offset + len(payload['rows'])}."]
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    return "\n".join(lines) + "\n"
