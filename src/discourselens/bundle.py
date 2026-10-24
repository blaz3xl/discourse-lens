"""Portable, secret-free research-bundle export."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .errors import DocumentError, ProjectError
from .json_safety import validate_json_nesting
from .reporting import (
    _contains_suspicious_credential_alias,
    _contains_suspicious_provider_endpoint,
    _parse_model_price_snapshot,
    _price_source_contains_secret,
    finding_payload,
)
from .secrets import _validate_provider_endpoint_metadata
from .types import CoverageDescriptor

BUNDLE_VERSION = 16
MAX_BUNDLE_MANIFEST_BYTES = 64 * 1024 * 1024
_SUPPORTED_BUNDLE_VERSIONS = {10, 11, 12, 13, 14, 15, BUNDLE_VERSION}
_INTEGRITY_BUNDLE_VERSIONS = {11, 12, 13, 14, 15, BUNDLE_VERSION}
_REVIEW_HISTORY_BUNDLE_VERSIONS = {12, 13, 14, 15, BUNDLE_VERSION}
_FINDING_SPAN_BUNDLE_VERSIONS = {14, 15, BUNDLE_VERSION}
_CANONICAL_TEXT_BUNDLE_VERSIONS = {15, BUNDLE_VERSION}
_SECRET_METADATA_KEYS = {
    "api_token",
    "api_key",
    "apikey",
    "password",
    "secret",
    "secret_reference",
    "secret_ref",
    "secret_value",
    "token",
    "access_token",
    "auth_token",
    "authorization",
    "credential",
    "credential_ref",
    "credential_reference",
    "credential_value",
    "credentials",
}
_SAFE_TOKEN_USAGE_KEYS = {
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
_SECRET_KEY_CONTEXTS = {
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
_SECRET_TOKEN_CONTEXTS = {
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
_ABSOLUTE_HOST_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_./-])/(?!/)[^\s\"'<>|?*]*"),
    re.compile(r"(?<![A-Za-z0-9_./:-])//(?!/)[^\s\"'<>|?*]+"),
    re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+\\[^\\/\s]+)"),
    re.compile(r"(?i)\bfile:(?://)?(?:/[A-Za-z]:)?[/\\]"),
)


def _canonical_json(payload: dict[str, Any]) -> bytes:
    """Serialize bundle content deterministically for accidental-corruption checks."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_text_preflight_size(
    payload: dict[str, Any], revision_ids: list[str]
) -> int:
    """Return a serialized lower bound with canonical text strings left empty."""
    preflight_payload = dict(payload)
    preflight_payload["canonical_texts"] = [
        {"revision_id": revision_id, "sha256": "0" * 64, "text": ""}
        for revision_id in revision_ids
    ]
    preflight_payload["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "digest": "0" * 64,
    }
    return len(
        json.dumps(
            preflight_payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def _contains_secret_metadata_key(
    value: object, *, allow_counterevidence_query_labels: bool = False
) -> bool:
    """Defend the export boundary against secret-bearing historical rows."""
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                if (
                    allow_counterevidence_query_labels
                    and _normalize_metadata_key(key) == "counterevidence_queries"
                    and isinstance(child, dict)
                ):
                    # Query-map labels are identifiers, while each value remains
                    # subject to recursive metadata screening.
                    pending.extend(child.values())
                    continue
                if _is_secret_metadata_key(_normalize_metadata_key(key)):
                    return True
                pending.append(child)
        elif isinstance(current, list):
            pending.extend(current)
    return False


def _contains_absolute_host_path(value: object) -> bool:
    """Reject machine-local filesystem paths in portable run metadata."""
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
        elif isinstance(current, str) and any(
            pattern.search(current) for pattern in _ABSOLUTE_HOST_PATH_PATTERNS
        ):
            return True
    return False


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys instead of inheriting parser-specific last-key-wins behavior."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_non_finite_json_number(value: str) -> None:
    """Reject NaN and Infinity, which are accepted by Python but are not JSON numbers."""
    raise ValueError(f"non-finite JSON number {value!r} is not supported")


def _normalize_metadata_key(key: object) -> str:
    """Normalize snake, kebab, camel, and Pascal-case metadata spellings."""
    text = str(key)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _is_secret_metadata_key(normalized: str) -> bool:
    if normalized in _SECRET_METADATA_KEYS:
        return True

    parts = set(normalized.split("_"))
    if parts.intersection({"password", "secret", "credentials", "authorization"}):
        return True

    if "key" in parts and parts.intersection(_SECRET_KEY_CONTEXTS):
        return True

    if "tokens" in parts and parts.intersection(_SECRET_TOKEN_CONTEXTS):
        return True

    if "token" in parts:
        if parts.intersection(_SECRET_TOKEN_CONTEXTS):
            return True
        # Provider usage metadata commonly reports plural token totals or an
        # explicit token count. Singular, otherwise-unqualified `*_token`
        # fields are much more likely to carry a credential value.
        if normalized in _SAFE_TOKEN_USAGE_KEYS or "count" in parts or "counts" in parts:
            return False
        return "tokens" not in parts

    return False


def _validate_model_execution_metadata(raw: object, *, boundary: str) -> None:
    """Validate encoded model metadata at export and bundle-verification boundaries."""
    if not isinstance(raw, str):
        raise ProjectError(f"{boundary} model execution metadata is invalid")
    try:
        metadata = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_non_finite_json_number,
        )
    except (ValueError, RecursionError) as exc:
        raise ProjectError(f"{boundary} model execution metadata is invalid") from exc
    if not isinstance(metadata, dict) or _contains_secret_metadata_key(metadata):
        raise ProjectError(f"{boundary} contains secret-bearing model execution metadata")
    if _contains_absolute_host_path(metadata):
        raise ProjectError(f"{boundary} contains absolute host paths in model execution metadata")
    source_reference = metadata.get("source_reference")
    if source_reference is not None and _price_source_contains_secret(source_reference):
        raise ProjectError(f"{boundary} contains secret-like model price source reference")


def _validate_document_revision_provenance(
    revision: object, *, boundary: str
) -> None:
    """Validate the portable extraction identity added in bundle v16."""
    if not isinstance(revision, dict):
        raise ProjectError(f"{boundary} document revision provenance is invalid")
    raw_manifest = revision.get("extraction_manifest_json")
    structure_hash = revision.get("structure_hash")
    if not isinstance(raw_manifest, str):
        raise ProjectError(f"{boundary} document revision extraction manifest is invalid")
    try:
        if len(raw_manifest.encode("utf-8")) > 64 * 1024:
            raise ProjectError(
                f"{boundary} document revision extraction manifest is invalid"
            )
    except UnicodeEncodeError:
        raise ProjectError(
            f"{boundary} document revision extraction manifest is invalid"
        ) from None
    if not isinstance(structure_hash, str) or re.fullmatch(r"[a-f0-9]{64}", structure_hash) is None:
        raise ProjectError(f"{boundary} document revision structure hash is malformed")
    try:
        validate_json_nesting(raw_manifest)
        manifest = json.loads(
            raw_manifest,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_non_finite_json_number,
        )
        canonical_manifest = json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        raise ProjectError(f"{boundary} document revision extraction manifest is invalid") from None
    if not isinstance(manifest, dict) or canonical_manifest != raw_manifest:
        raise ProjectError(f"{boundary} document revision extraction manifest is invalid")
    if (
        _contains_secret_metadata_key(manifest)
        or _contains_suspicious_credential_alias(manifest)
        or _contains_suspicious_provider_endpoint(manifest)
    ):
        raise ProjectError(f"{boundary} contains secret-bearing document extraction metadata")
    if _contains_absolute_host_path(manifest):
        raise ProjectError(
            f"{boundary} contains absolute host paths in document extraction metadata"
        )


def _validate_model_execution_endpoint(endpoint: object, *, boundary: str) -> None:
    """Reject credential-bearing endpoint data without echoing it into errors."""
    if endpoint is not None and not isinstance(endpoint, str):
        raise ProjectError(f"{boundary} contains credential-like provider endpoint metadata")
    try:
        _validate_provider_endpoint_metadata(endpoint)
    except (ProjectError, TypeError, ValueError):
        raise ProjectError(
            f"{boundary} contains credential-like provider endpoint metadata"
        ) from None


def _validate_coverage_descriptor_metadata(
    raw: object, *, boundary: str, expected_scope_id: str | None = None
) -> CoverageDescriptor:
    """Validate portable coverage provenance at export and verification boundaries."""
    if not isinstance(raw, str):
        raise ProjectError(f"{boundary} coverage descriptor is invalid")
    try:
        metadata = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_non_finite_json_number,
        )
    except (ValueError, RecursionError) as exc:
        raise ProjectError(f"{boundary} coverage descriptor is invalid") from exc
    if not isinstance(metadata, dict):
        raise ProjectError(f"{boundary} coverage descriptor is invalid")
    if _contains_secret_metadata_key(metadata):
        raise ProjectError(f"{boundary} contains secret-bearing coverage descriptor metadata")
    if _contains_absolute_host_path(metadata):
        raise ProjectError(
            f"{boundary} contains absolute host paths in coverage descriptor metadata"
        )
    try:
        descriptor = CoverageDescriptor.model_validate(metadata)
    except (ValidationError, ValueError, TypeError) as exc:
        raise ProjectError(f"{boundary} coverage descriptor is invalid") from exc
    if expected_scope_id is not None and descriptor.scope_snapshot_id != expected_scope_id:
        raise ProjectError(f"{boundary} coverage descriptor has a mismatched scope reference")
    return descriptor


def _validate_model_price_snapshot(
    raw: object,
    *,
    boundary: str,
    execution_provider: object,
    execution_model: object,
) -> None:
    try:
        metadata = _parse_model_price_snapshot(raw)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"{boundary} model price snapshot is invalid") from exc
    if metadata is None:
        return
    if _contains_secret_metadata_key(metadata):
        raise ProjectError(f"{boundary} contains secret-bearing model price metadata")
    if _contains_absolute_host_path(metadata):
        raise ProjectError(f"{boundary} contains absolute host paths in model price metadata")
    if _price_source_contains_secret(metadata.get("source_reference")):
        raise ProjectError(f"{boundary} contains secret-like model price source reference")
    if metadata["provider_id"] != execution_provider or metadata["model"] != execution_model:
        raise ProjectError(f"{boundary} model price snapshot does not match model execution")


def _model_execution_rows(project, analysis_run_id: str) -> list[dict[str, Any]]:
    rows = project.db.execute(
        "SELECT * FROM model_executions WHERE analysis_run_id=? ORDER BY id", (analysis_run_id,)
    ).fetchall()
    for row in rows:
        try:
            project.validate_model_execution_skill_run(analysis_run_id, row["skill_run_id"])
        except DocumentError:
            raise ProjectError(
                "research bundle model execution references an invalid skill run"
            ) from None
        _validate_model_execution_endpoint(row["endpoint"], boundary="analysis run")
        for field in ("parameters_json", "capabilities_json"):
            _validate_model_execution_metadata(row[field], boundary="analysis run")
        _validate_model_price_snapshot(
            row["model_price_json"],
            boundary="analysis run",
            execution_provider=row["provider_id"],
            execution_model=row["model"],
        )
    return [dict(row) for row in rows]


def _id_batches(project, ids: list[str]) -> Iterator[list[str]]:
    """Yield ID batches that fit this SQLite connection's bound-variable limit."""
    limit = project.db.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    if limit < 1:
        raise ProjectError("SQLite connection cannot bind research bundle selection IDs")
    for offset in range(0, len(ids), limit):
        yield ids[offset : offset + limit]


def _placeholders(ids: list[str]) -> str:
    return ",".join("?" for _ in ids)


def _directory_fd_write_supported() -> bool:
    """Whether this platform can pin a bundle destination for atomic replacement."""
    return (
        os.name == "posix"
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.rename in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
    )


def supports_descriptor_relative_bundle_export() -> bool:
    """Whether this platform can create and atomically write a pinned bundle directory."""
    return _directory_fd_write_supported() and os.mkdir in os.supports_dir_fd


def _verify_pinned_bundle_directory(destination: Path, directory_fd: int) -> None:
    try:
        entry = os.stat(destination, follow_symlinks=False)
        opened = os.fstat(directory_fd)
    except OSError as exc:
        raise ProjectError("research bundle destination changed during export") from exc
    if not stat.S_ISDIR(entry.st_mode) or (entry.st_dev, entry.st_ino) != (
        opened.st_dev,
        opened.st_ino,
    ):
        raise ProjectError("research bundle destination changed during export")


def _write_bundle_manifest_at(directory_fd: int, serialized: bytes) -> None:
    """Atomically write a manifest relative to an already pinned directory FD."""
    if not _directory_fd_write_supported():
        raise ProjectError("descriptor-relative research bundle writes are unavailable")

    directory_stat = os.fstat(directory_fd)
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise ProjectError("research bundle destination must be an open directory")

    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    temporary_fd = -1
    temporary_name: str | None = None
    try:
        for _ in range(10):
            candidate = f".manifest-{secrets.token_hex(16)}"
            try:
                temporary_fd = os.open(candidate, file_flags, 0o600, dir_fd=directory_fd)
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if temporary_fd < 0 or temporary_name is None:
            raise ProjectError("could not reserve a temporary research bundle manifest")

        descriptor = temporary_fd
        temporary_fd = -1
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(serialized)
            temporary.flush()
        os.rename(
            temporary_name,
            "manifest.json",
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = None
    except ProjectError:
        raise
    except OSError as exc:
        raise ProjectError("research bundle destination is unavailable or unsafe") from exc
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except OSError:
                pass


def _write_bundle_manifest(destination: Path, serialized: bytes) -> None:
    """Atomically write the manifest without following a swapped destination path."""
    if not _directory_fd_write_supported():
        temporary_path: Path | None = None
        output = destination / "manifest.json"
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination,
                prefix=".manifest-",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(serialized)
            os.replace(temporary_path, output)
        except OSError as exc:
            raise ProjectError("research bundle destination is unavailable or unsafe") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = -1
    try:
        directory_fd = os.open(destination, directory_flags)
        _verify_pinned_bundle_directory(destination, directory_fd)
        _write_bundle_manifest_at(directory_fd, serialized)
        _verify_pinned_bundle_directory(destination, directory_fd)
    except ProjectError:
        raise
    except OSError as exc:
        raise ProjectError("research bundle destination is unavailable or unsafe") from exc
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


def export_bundle(
    project,
    analysis_run_id: str,
    destination: str | Path,
    *,
    include_canonical_text: bool = False,
) -> Path:
    result = _export_bundle_impl(
        project, analysis_run_id, Path(destination), None, include_canonical_text
    )
    if not isinstance(result, Path):
        raise TypeError("path-based bundle export returned no destination")
    return result


def export_bundle_to_directory_fd(
    project,
    analysis_run_id: str,
    directory_fd: int,
    *,
    include_canonical_text: bool = False,
) -> bytes:
    """Export a bundle through a pinned directory FD and return the exact manifest bytes."""
    if not _directory_fd_write_supported():
        raise ProjectError("descriptor-relative research bundle writes are unavailable")
    result = _export_bundle_impl(
        project, analysis_run_id, None, directory_fd, include_canonical_text
    )
    if not isinstance(result, bytes):
        raise TypeError("descriptor-relative bundle export returned no manifest")
    return result


def _export_bundle_impl(
    project,
    analysis_run_id: str,
    destination: Path | None,
    directory_fd: int | None,
    include_canonical_text: bool,
) -> Path | bytes:
    if type(include_canonical_text) is not bool:
        raise ProjectError("include_canonical_text must be a boolean")
    try:
        run = project.analysis_run_metadata(analysis_run_id)["analysis_run"]
    except Exception as exc:
        if isinstance(exc, ProjectError):
            if str(exc) == "analysis run configuration contains secret data":
                stored = project.db.execute(
                    "SELECT config_json FROM analysis_runs WHERE id=?", (analysis_run_id,)
                ).fetchone()
                if stored is not None:
                    try:
                        config = json.loads(stored["config_json"])
                    except (TypeError, ValueError, RecursionError):
                        config = None
                    if _contains_secret_metadata_key(
                        config, allow_counterevidence_query_labels=True
                    ):
                        raise DocumentError(
                            "analysis run configuration contains secret data "
                            "(secret-bearing configuration metadata)"
                        ) from None
            raise
        raise ProjectError("analysis run provenance is invalid") from exc
    if _contains_secret_metadata_key(run.get("config"), allow_counterevidence_query_labels=True):
        raise ProjectError("analysis run contains secret-bearing configuration metadata")
    if _contains_suspicious_credential_alias(run.get("config")):
        raise ProjectError("analysis run contains suspicious credential alias metadata")
    if _contains_suspicious_provider_endpoint(run.get("config")):
        raise ProjectError("analysis run contains credential-like provider endpoint metadata")
    if _contains_absolute_host_path(run.get("config")):
        raise ProjectError("analysis run contains absolute host paths in configuration metadata")
    if destination is not None and destination.is_symlink():
        raise ProjectError("research bundle destination must not be a symlink")
    findings = project.db.execute(
        "SELECT * FROM findings WHERE analysis_run_id=? ORDER BY created_at,id",
        (analysis_run_id,),
    ).fetchall()
    revision_ids = set(project.scope_revisions(run["scope_id"]))
    payload = {
        "bundle_version": BUNDLE_VERSION,
        "project_format_version": 1,
        "analysis_run": dict(run),
        "findings": [dict(item) for item in findings],
        "finding_lineage": [],
        "evidence": {},
        "finding_evidence_spans": [],
        "canonical_texts": [],
        "document_revisions": [],
        "skill_runs": [],
        "coverage_descriptors": [],
        "model_executions": [],
        "retrieval_runs": [],
        "retrieval_hits": [],
        "reranking_runs": [],
        "reranking_hits": [],
        "human_reviews": {},
        "qualitative_coding": {
            "codebooks": [],
            "codes": [],
            "annotations": [],
            "memos": [],
            "source_spans": [],
        },
    }
    if include_canonical_text:
        # finding_payload validates textual evidence by loading its source text, so
        # reject bundles whose cited source bytes cannot fit before building it.
        cited_revision_rows = project.db.execute(
            "SELECT DISTINCT s.revision_id FROM findings f "
            "JOIN finding_evidence fe ON fe.finding_id=f.id "
            "JOIN evidence_artifacts e ON e.id=fe.evidence_id "
            "JOIN source_spans s ON s.id=e.span_id "
            "WHERE f.analysis_run_id=? AND e.kind='TEXTUAL' ORDER BY s.revision_id",
            (analysis_run_id,),
        ).fetchall()
        cited_revision_ids = [str(item["revision_id"]) for item in cited_revision_rows]
        for revision_id in cited_revision_ids:
            if revision_id not in revision_ids:
                raise ProjectError(
                    "research bundle textual evidence source span belongs to an out-of-scope revision"
                )
        cited_revision_sizes = {
            revision_id: project.revision_text_size_bytes(revision_id)
            for revision_id in cited_revision_ids
        }
        preflight_size = _canonical_text_preflight_size(payload, cited_revision_ids)
        remaining_lower_bound = MAX_BUNDLE_MANIFEST_BYTES - preflight_size
        if remaining_lower_bound < 0 or sum(cited_revision_sizes.values()) > remaining_lower_bound:
            raise ProjectError(
                "research bundle canonical text would exceed the configured size limit"
            )
    for finding in findings:
        finding_data = finding_payload(project, finding["id"])
        payload["evidence"][finding["id"]] = finding_data["evidence"]
        payload["human_reviews"][finding["id"]] = [
            {
                "id": item["id"],
                "status": item["status"],
                "note": item["note"],
                "created_at": item["created_at"],
                "review_order": item["review_order"],
            }
            for item in project.finding_reviews(finding["id"])
        ]
    # Keep canonical span coordinates for textual finding evidence distinct from
    # qualitative-coding spans. The latter are limited to annotation provenance;
    # this inventory is limited to spans actually cited by this run's findings.
    evidence_span_rows = project.db.execute(
        "SELECT DISTINCT s.id,s.revision_id,s.start_offset,s.end_offset,s.text_hash "
        "FROM findings f JOIN finding_evidence fe ON fe.finding_id=f.id "
        "JOIN evidence_artifacts e ON e.id=fe.evidence_id "
        "JOIN source_spans s ON s.id=e.span_id "
        "WHERE f.analysis_run_id=? AND e.kind='TEXTUAL' "
        "ORDER BY s.revision_id,s.start_offset,s.end_offset,s.id",
        (analysis_run_id,),
    ).fetchall()
    text_lengths: dict[str, int] = {}
    for item in evidence_span_rows:
        revision_id = str(item["revision_id"])
        if revision_id not in revision_ids:
            raise ProjectError(
                "research bundle textual evidence source span belongs to an out-of-scope revision"
            )
        if revision_id not in text_lengths:
            text_lengths[revision_id] = (
                0 if include_canonical_text else len(project.revision_text(revision_id))
            )
        payload["finding_evidence_spans"].append(
            {
                "id": str(item["id"]),
                "revision_id": revision_id,
                "start_offset": int(item["start_offset"]),
                "end_offset": int(item["end_offset"]),
                "text_hash": str(item["text_hash"]),
                "canonical_text_length": text_lengths[revision_id],
            }
        )
    evidence_span_revisions = {
        item["id"]: item["revision_id"] for item in payload["finding_evidence_spans"]
    }
    for evidence_rows in payload["evidence"].values():
        for evidence in evidence_rows:
            span_id = evidence.get("source_span_id")
            if evidence.get("kind") == "TEXTUAL":
                revision_id = evidence_span_revisions.get(span_id)
                if revision_id is None:
                    raise ProjectError(
                        "research bundle textual evidence has no canonical source span"
                    )
                evidence["source_revision_id"] = revision_id
            else:
                evidence["source_revision_id"] = None
    payload["finding_lineage"] = [
        dict(item)
        for item in project.db.execute(
            "SELECT l.* FROM finding_lineage l JOIN findings f ON f.id=l.finding_id "
            "WHERE f.analysis_run_id=? ORDER BY l.finding_id,l.source_finding_id",
            (analysis_run_id,),
        ).fetchall()
    ]
    # A run's ScopeSnapshot is authoritative even if a finding currently cites
    # only external/user context. Deriving revisions solely from evidence spans
    # would make a valid contextual analysis bundle lose its corpus provenance.
    for revision_id in sorted(revision_ids):
        revision = project.db.execute(
            """
            SELECT id,document_id,source_hash,canonical_hash,extractor,coordinate_scheme,mime_type,
                   source_size,created_at,ocr_language,ocr_dpi,extraction_manifest_json,structure_hash
            FROM document_revisions WHERE id=?
            """,
            (revision_id,),
        ).fetchone()
        if revision:
            revision_payload = dict(revision)
            _validate_document_revision_provenance(
                revision_payload, boundary="research bundle export"
            )
            payload["document_revisions"].append(revision_payload)
    payload["skill_runs"] = [
        dict(item)
        for item in project.db.execute(
            "SELECT * FROM skill_runs WHERE analysis_run_id=? ORDER BY id", (analysis_run_id,)
        ).fetchall()
    ]
    payload["retrieval_runs"] = [
        dict(item)
        for item in project.db.execute(
            "SELECT * FROM retrieval_runs WHERE analysis_run_id=? ORDER BY created_at,id",
            (analysis_run_id,),
        ).fetchall()
    ]
    coverage_ids = sorted(
        {item["coverage_id"] for item in payload["skill_runs"] if item["coverage_id"]}
        | {item["coverage_id"] for item in findings if item["coverage_id"]}
        | {item["coverage_id"] for item in payload["retrieval_runs"] if item["coverage_id"]}
    )
    if coverage_ids:
        coverage_rows = []
        for id_batch in _id_batches(project, coverage_ids):
            placeholders = _placeholders(id_batch)
            coverage_rows.extend(
                project.db.execute(
                    f"SELECT * FROM coverage_descriptors WHERE id IN ({placeholders}) ORDER BY id",
                    id_batch,
                ).fetchall()
            )
        payload["coverage_descriptors"] = [
            dict(item) for item in sorted(coverage_rows, key=lambda item: item["id"])
        ]
        for descriptor in payload["coverage_descriptors"]:
            _validate_coverage_descriptor_metadata(
                descriptor.get("payload_json"),
                boundary="research bundle export",
                expected_scope_id=run["scope_id"],
            )
    # Rows from older projects or manually repaired databases may predate the
    # write-time secret checks. Revalidate at export, which is a trust boundary.
    payload["model_executions"] = _model_execution_rows(project, analysis_run_id)
    payload["retrieval_hits"] = [
        dict(item)
        for item in project.db.execute(
            "SELECT h.* FROM retrieval_hits h JOIN retrieval_runs r "
            "ON r.id=h.retrieval_run_id WHERE r.analysis_run_id=? "
            "ORDER BY r.created_at,r.id,h.rank",
            (analysis_run_id,),
        ).fetchall()
    ]
    payload["reranking_runs"] = [
        dict(item)
        for item in project.db.execute(
            "SELECT rr.* FROM reranking_runs rr JOIN retrieval_runs r "
            "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=? ORDER BY rr.created_at,rr.id",
            (analysis_run_id,),
        ).fetchall()
    ]
    payload["reranking_hits"] = [
        dict(item)
        for item in project.db.execute(
            "SELECT rh.* FROM reranking_hits rh JOIN reranking_runs rr "
            "ON rr.id=rh.reranking_run_id JOIN retrieval_runs r "
            "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=? "
            "ORDER BY rr.created_at,rr.id,rh.rank",
            (analysis_run_id,),
        ).fetchall()
    ]
    if revision_ids:
        annotation_rows = []
        source_span_rows = []
        codes_by_id = {}
        for id_batch in _id_batches(project, sorted(revision_ids)):
            placeholders = _placeholders(id_batch)
            annotation_rows.extend(
                project.db.execute(
                    "SELECT annotations.* FROM annotations JOIN source_spans "
                    "ON source_spans.id=annotations.span_id "
                    f"WHERE source_spans.revision_id IN ({placeholders}) "
                    "ORDER BY annotations.created_at,annotations.id",
                    id_batch,
                ).fetchall()
            )
            source_span_rows.extend(
                project.db.execute(
                    "SELECT DISTINCT source_spans.id,source_spans.revision_id,"
                    "source_spans.start_offset,source_spans.end_offset "
                    "FROM annotations JOIN source_spans ON source_spans.id=annotations.span_id "
                    f"WHERE source_spans.revision_id IN ({placeholders}) "
                    "ORDER BY source_spans.revision_id,source_spans.start_offset,source_spans.id",
                    id_batch,
                ).fetchall()
            )
            code_rows = project.db.execute(
                "WITH RECURSIVE scoped_codes(id) AS ("
                "SELECT DISTINCT annotations.code_id FROM annotations "
                "JOIN source_spans ON source_spans.id=annotations.span_id "
                f"WHERE source_spans.revision_id IN ({placeholders}) "
                "UNION "
                "SELECT codes.parent_code_id FROM codes "
                "JOIN scoped_codes ON scoped_codes.id=codes.id "
                "WHERE codes.parent_code_id IS NOT NULL"
                ") SELECT codes.* FROM codes "
                "JOIN scoped_codes ON scoped_codes.id=codes.id ORDER BY codes.id",
                id_batch,
            ).fetchall()
            codes_by_id.update((item["id"], item) for item in code_rows)
        payload["qualitative_coding"]["annotations"] = [
            dict(item)
            for item in sorted(annotation_rows, key=lambda item: (item["created_at"], item["id"]))
        ]
        payload["qualitative_coding"]["source_spans"] = [
            dict(item)
            for item in sorted(
                {item["id"]: item for item in source_span_rows}.values(),
                key=lambda item: (
                    item["revision_id"],
                    item["start_offset"],
                    item["id"],
                ),
            )
        ]
        codes = [codes_by_id[code_id] for code_id in sorted(codes_by_id)]
        payload["qualitative_coding"]["codes"] = [dict(item) for item in codes]
        codebook_ids = sorted({item["codebook_id"] for item in codes})
        if codebook_ids:
            codebook_rows = []
            for id_batch in _id_batches(project, codebook_ids):
                placeholders = _placeholders(id_batch)
                codebook_rows.extend(
                    project.db.execute(
                        f"SELECT * FROM codebooks WHERE id IN ({placeholders}) ORDER BY id",
                        id_batch,
                    ).fetchall()
                )
            payload["qualitative_coding"]["codebooks"] = [
                dict(item) for item in sorted(codebook_rows, key=lambda item: item["id"])
            ]
    payload["qualitative_coding"]["memos"] = [
        dict(item) for item in project.research_memos(run["scope_id"])
    ]
    if include_canonical_text:
        revisions_by_id = {item["id"]: item for item in payload["document_revisions"]}
        referenced_revision_ids = sorted(
            {item["revision_id"] for item in payload["finding_evidence_spans"]}
        )
        for revision_id in referenced_revision_ids:
            if revision_id not in revisions_by_id:
                raise ProjectError(
                    "research bundle textual evidence references a missing document revision"
                )

        revision_sizes = {
            revision_id: project.revision_text_size_bytes(revision_id)
            for revision_id in referenced_revision_ids
        }
        preflight_size = _canonical_text_preflight_size(payload, referenced_revision_ids)
        remaining_lower_bound = MAX_BUNDLE_MANIFEST_BYTES - preflight_size
        if remaining_lower_bound < 0 or sum(revision_sizes.values()) > remaining_lower_bound:
            raise ProjectError(
                "research bundle canonical text would exceed the configured size limit"
            )

        text_lengths: dict[str, int] = {}
        for revision_id in referenced_revision_ids:
            revision = revisions_by_id[revision_id]
            canonical_text = project.revision_text(revision_id)
            canonical_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
            if not hmac.compare_digest(canonical_hash, revision["canonical_hash"]):
                raise ProjectError(
                    "research bundle canonical source text does not match its revision hash"
                )
            text_lengths[revision_id] = len(canonical_text)
            payload["canonical_texts"].append(
                {"revision_id": revision_id, "sha256": canonical_hash, "text": canonical_text}
            )
        for source_span in payload["finding_evidence_spans"]:
            source_span["canonical_text_length"] = text_lengths[source_span["revision_id"]]
    if _contains_suspicious_credential_alias(payload):
        raise ProjectError("research bundle contains suspicious credential alias metadata")
    if _contains_suspicious_provider_endpoint(payload):
        raise ProjectError("research bundle contains credential-like provider endpoint metadata")
    payload["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "digest": hashlib.sha256(_canonical_json(payload)).hexdigest(),
    }
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(serialized) > MAX_BUNDLE_MANIFEST_BYTES:
        raise ProjectError("research bundle manifest exceeds the configured size limit")
    if directory_fd is not None:
        _write_bundle_manifest_at(directory_fd, serialized)
        return serialized
    if destination is None:
        raise AssertionError("path-based bundle export requires a destination")
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or not destination.is_dir():
        raise ProjectError("research bundle destination must be a real directory")
    _write_bundle_manifest(destination, serialized)
    return destination / "manifest.json"


def verify_bundle(manifest_path: str | Path | bytes) -> dict[str, object]:
    """Verify a portable research bundle without opening a project or using network.

    Manifests are limited to ``MAX_BUNDLE_MANIFEST_BYTES`` for bounded parsing.
    Version 11 and newer bundles carry a canonical JSON SHA-256 digest. Version
    10 bundles remain structurally inspectable but cannot provide an integrity result.
    The digest detects accidental corruption; it is not a signature or proof of
    authorship because an editor can recompute it after changing a manifest.
    """
    descriptor = -1
    try:
        if isinstance(manifest_path, bytes):
            # MCP reads pass bytes copied from an already-pinned, no-follow
            # descriptor so validation covers the exact managed file checked
            # by that boundary. Path callers retain the established behavior.
            raw = manifest_path
        else:
            path = Path(manifest_path)
            # O_NONBLOCK prevents a named pipe supplied as the manifest path from
            # hanging verification before we can reject it. fstat the opened file
            # descriptor so a path swap cannot turn this check into a FIFO read;
            # symlinks to regular manifest files remain supported.
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0),
            )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ProjectError("research bundle manifest must be a regular file")
            with os.fdopen(descriptor, "rb") as manifest:
                descriptor = -1
                raw = manifest.read(MAX_BUNDLE_MANIFEST_BYTES + 1)
        if len(raw) > MAX_BUNDLE_MANIFEST_BYTES:
            raise ProjectError("research bundle manifest exceeds the configured size limit")
        manifest_text = raw.decode("utf-8")
        validate_json_nesting(manifest_text)
        payload = json.loads(
            manifest_text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_non_finite_json_number,
        )
    except ProjectError:
        raise
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ProjectError("research bundle manifest is unreadable or invalid JSON") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise ProjectError("research bundle manifest must be a JSON object")
    if _contains_suspicious_credential_alias(payload):
        raise ProjectError("research bundle contains suspicious credential alias metadata")
    if _contains_suspicious_provider_endpoint(payload):
        raise ProjectError("research bundle contains credential-like provider endpoint metadata")

    version = payload.get("bundle_version")
    if type(version) is not int or version not in _SUPPORTED_BUNDLE_VERSIONS:
        raise ProjectError("unsupported research bundle version")
    if version in _INTEGRITY_BUNDLE_VERSIONS:
        allowed_fields = {
            "bundle_version",
            "project_format_version",
            "analysis_run",
            "findings",
            "finding_lineage",
            "evidence",
            "document_revisions",
            "skill_runs",
            "coverage_descriptors",
            "model_executions",
            "retrieval_runs",
            "retrieval_hits",
            "reranking_runs",
            "reranking_hits",
            "qualitative_coding",
            "integrity",
        }
        if version in _FINDING_SPAN_BUNDLE_VERSIONS:
            allowed_fields.add("finding_evidence_spans")
        if version in _CANONICAL_TEXT_BUNDLE_VERSIONS:
            allowed_fields.add("canonical_texts")
        if version in _REVIEW_HISTORY_BUNDLE_VERSIONS:
            allowed_fields.add("human_reviews")
        if set(payload) - allowed_fields:
            raise ProjectError("research bundle contains unsupported top-level fields")
    project_format_version = payload.get("project_format_version")
    if type(project_format_version) is not int or project_format_version < 1:
        raise ProjectError("research bundle has an invalid project format version")
    required_collections = (
        "findings",
        "finding_lineage",
        "document_revisions",
        "skill_runs",
        "coverage_descriptors",
        "model_executions",
        "retrieval_runs",
        "retrieval_hits",
        "reranking_runs",
        "reranking_hits",
    )
    if (
        not isinstance(payload.get("analysis_run"), dict)
        or not isinstance(payload["analysis_run"].get("id"), str)
        or not payload["analysis_run"]["id"]
    ):
        raise ProjectError("research bundle lacks valid analysis-run provenance")
    for key in required_collections:
        if not isinstance(payload.get(key), list):
            raise ProjectError(f"research bundle field {key!r} must be an array")
    if version in _FINDING_SPAN_BUNDLE_VERSIONS and not isinstance(
        payload.get("finding_evidence_spans"), list
    ):
        raise ProjectError("research bundle field 'finding_evidence_spans' must be an array")
    if version in _CANONICAL_TEXT_BUNDLE_VERSIONS and not isinstance(
        payload.get("canonical_texts"), list
    ):
        raise ProjectError("research bundle field 'canonical_texts' must be an array")
    coverage_descriptor_ids: set[str] = set()
    for descriptor in payload["coverage_descriptors"]:
        if (
            not isinstance(descriptor, dict)
            or not isinstance(descriptor.get("id"), str)
            or not descriptor["id"]
            or descriptor["id"] in coverage_descriptor_ids
        ):
            raise ProjectError("research bundle contains an invalid coverage descriptor ID")
        _validate_coverage_descriptor_metadata(
            descriptor.get("payload_json"),
            boundary="research bundle",
            expected_scope_id=payload["analysis_run"].get("scope_id"),
        )
        coverage_descriptor_ids.add(descriptor["id"])
    skill_runs_by_id: dict[str, dict[str, Any]] = {}
    for skill_run in payload["skill_runs"]:
        if isinstance(skill_run, dict) and isinstance(skill_run.get("id"), str) and skill_run["id"]:
            if skill_run["id"] in skill_runs_by_id:
                raise ProjectError("research bundle contains duplicate skill run IDs")
            skill_runs_by_id[skill_run["id"]] = skill_run
    for collection_name in ("findings", "skill_runs", "retrieval_runs"):
        for record in payload[collection_name]:
            if not isinstance(record, dict) or "coverage_id" not in record:
                raise ProjectError(
                    f"research bundle field {collection_name!r} contains an invalid coverage reference"
                )
            coverage_id = record["coverage_id"]
            if coverage_id is None:
                if collection_name == "retrieval_runs":
                    raise ProjectError(
                        "research bundle retrieval run contains an invalid coverage reference"
                    )
                continue
            if not isinstance(coverage_id, str) or not coverage_id:
                raise ProjectError(
                    f"research bundle field {collection_name!r} contains an invalid coverage reference"
                )
            if coverage_id not in coverage_descriptor_ids:
                raise ProjectError(
                    f"research bundle field {collection_name!r} references a missing coverage descriptor"
                )
    if not isinstance(payload.get("evidence"), dict):
        raise ProjectError("research bundle evidence must be an object")
    if version in _INTEGRITY_BUNDLE_VERSIONS:
        qualitative_coding = payload.get("qualitative_coding")
        expected_coding_fields = {"codebooks", "codes", "annotations", "memos"}
        if version >= 13:
            expected_coding_fields.add("source_spans")
        if (
            not isinstance(qualitative_coding, dict)
            or set(qualitative_coding) != expected_coding_fields
        ):
            raise ProjectError("research bundle qualitative_coding has an invalid structure")
        for key in expected_coding_fields:
            if not isinstance(qualitative_coding[key], list):
                raise ProjectError(
                    f"research bundle qualitative_coding field {key!r} must be an array"
                )
        analysis_scope_id = payload["analysis_run"].get("scope_id")
        if not isinstance(analysis_scope_id, str) or not analysis_scope_id:
            raise ProjectError("research bundle analysis-run scope provenance is invalid")

        codebook_ids: set[str] = set()
        for codebook in qualitative_coding["codebooks"]:
            if (
                not isinstance(codebook, dict)
                or not isinstance(codebook.get("id"), str)
                or not codebook["id"]
                or codebook["id"] in codebook_ids
            ):
                raise ProjectError(
                    "research bundle qualitative coding contains an invalid codebook"
                )
            codebook_ids.add(codebook["id"])

        codes_by_id: dict[str, dict[str, Any]] = {}
        for code in qualitative_coding["codes"]:
            if (
                not isinstance(code, dict)
                or not isinstance(code.get("id"), str)
                or not code["id"]
                or code["id"] in codes_by_id
            ):
                raise ProjectError("research bundle qualitative coding contains an invalid code")
            if (
                not isinstance(code.get("codebook_id"), str)
                or code["codebook_id"] not in codebook_ids
            ):
                raise ProjectError("research bundle qualitative coding code has a missing codebook")
            parent_code_id = code.get("parent_code_id")
            if parent_code_id is not None and (
                not isinstance(parent_code_id, str) or not parent_code_id
            ):
                raise ProjectError("research bundle qualitative coding code has an invalid parent")
            codes_by_id[code["id"]] = code
        for code in codes_by_id.values():
            parent_code_id = code["parent_code_id"]
            if parent_code_id is not None:
                parent = codes_by_id.get(parent_code_id)
                if parent is None or parent["codebook_id"] != code["codebook_id"]:
                    raise ProjectError(
                        "research bundle qualitative coding code has a missing parent"
                    )

        source_span_ids: set[str] = set()
        if version >= 13:
            for source_span in qualitative_coding["source_spans"]:
                if (
                    not isinstance(source_span, dict)
                    or set(source_span) != {"id", "revision_id", "start_offset", "end_offset"}
                    or not isinstance(source_span.get("id"), str)
                    or not source_span["id"]
                    or source_span["id"] in source_span_ids
                    or not isinstance(source_span.get("revision_id"), str)
                    or not source_span["revision_id"]
                    or type(source_span.get("start_offset")) is not int
                    or type(source_span.get("end_offset")) is not int
                    or source_span["start_offset"] < 0
                    or source_span["end_offset"] <= source_span["start_offset"]
                ):
                    raise ProjectError(
                        "research bundle qualitative coding contains an invalid source span"
                    )
                source_span_ids.add(source_span["id"])

        annotation_ids: set[str] = set()
        referenced_span_ids: set[str] = set()
        for annotation in qualitative_coding["annotations"]:
            if (
                not isinstance(annotation, dict)
                or not isinstance(annotation.get("id"), str)
                or not annotation["id"]
                or annotation["id"] in annotation_ids
            ):
                raise ProjectError(
                    "research bundle qualitative coding contains an invalid annotation"
                )
            if (
                not isinstance(annotation.get("code_id"), str)
                or annotation["code_id"] not in codes_by_id
            ):
                raise ProjectError(
                    "research bundle qualitative coding annotation references a missing code"
                )
            if not isinstance(annotation.get("span_id"), str) or not annotation["span_id"]:
                raise ProjectError(
                    "research bundle qualitative coding annotation has an invalid span reference"
                )
            if version >= 13 and annotation["span_id"] not in source_span_ids:
                raise ProjectError(
                    "research bundle qualitative coding annotation references a missing source span"
                )
            referenced_span_ids.add(annotation["span_id"])
            annotation_ids.add(annotation["id"])
        if version >= 13 and source_span_ids != referenced_span_ids:
            raise ProjectError(
                "research bundle qualitative coding source spans do not match its annotations"
            )

        memo_ids: set[str] = set()
        for memo in qualitative_coding["memos"]:
            if (
                not isinstance(memo, dict)
                or not isinstance(memo.get("id"), str)
                or not memo["id"]
                or memo["id"] in memo_ids
            ):
                raise ProjectError("research bundle qualitative coding contains an invalid memo")
            if memo.get("scope_id") != analysis_scope_id:
                raise ProjectError(
                    "research bundle qualitative coding memo has a mismatched scope reference"
                )
            memo_ids.add(memo["id"])
    if version in _REVIEW_HISTORY_BUNDLE_VERSIONS and not isinstance(
        payload.get("human_reviews"), dict
    ):
        raise ProjectError("research bundle human_reviews must be an object")
    finding_ids: set[str] = set()
    for finding in payload["findings"]:
        if (
            not isinstance(finding, dict)
            or not isinstance(finding.get("id"), str)
            or not finding["id"]
        ):
            raise ProjectError("research bundle contains a finding without a valid ID")
        finding_id = finding["id"]
        if finding_id in finding_ids:
            raise ProjectError("research bundle contains duplicate finding IDs")
        finding_ids.add(finding_id)
        if finding.get("analysis_run_id") != payload["analysis_run"]["id"]:
            raise ProjectError("research bundle finding belongs to a different analysis run")
    if set(payload["evidence"]) != finding_ids:
        raise ProjectError("research bundle evidence keys do not match finding IDs")
    if version in _REVIEW_HISTORY_BUNDLE_VERSIONS:
        human_reviews = payload["human_reviews"]
        if set(human_reviews) != finding_ids:
            raise ProjectError("research bundle human_reviews keys do not match finding IDs")
        review_ids: set[str] = set()
        for finding_id, review_rows in human_reviews.items():
            if not isinstance(review_rows, list):
                raise ProjectError(
                    f"research bundle human_reviews for {finding_id!r} must be an array"
                )
            previous_order = 0
            for review in review_rows:
                if (
                    not isinstance(review, dict)
                    or set(review) != {"id", "status", "note", "created_at", "review_order"}
                    or not isinstance(review.get("id"), str)
                    or not review["id"]
                    or not isinstance(review.get("status"), str)
                    or review["status"] not in {"ACCEPTED", "MODIFIED", "REJECTED"}
                    or (review.get("note") is not None and not isinstance(review.get("note"), str))
                    or not isinstance(review.get("created_at"), str)
                    or not review["created_at"]
                    or type(review.get("review_order")) is not int
                    or review["review_order"] != previous_order + 1
                ):
                    raise ProjectError(
                        f"research bundle human_reviews for {finding_id!r} has an invalid review"
                    )
                if review["id"] in review_ids:
                    raise ProjectError("research bundle contains duplicate human review IDs")
                review_ids.add(review["id"])
                previous_order = review["review_order"]
    for lineage in payload["finding_lineage"]:
        if (
            not isinstance(lineage, dict)
            or set(lineage) != {"finding_id", "source_finding_id", "relation"}
            or not isinstance(lineage.get("finding_id"), str)
            or not isinstance(lineage.get("source_finding_id"), str)
            or lineage.get("relation") != "SYNTHESIZES"
            or lineage["finding_id"] not in finding_ids
            or lineage["source_finding_id"] not in finding_ids
            or lineage["finding_id"] == lineage["source_finding_id"]
        ):
            raise ProjectError("research bundle finding_lineage contains an invalid reference")
    evidence_count = 0
    textual_evidence_rows: list[dict[str, Any]] = []
    for finding_id, evidence_rows in payload["evidence"].items():
        if not isinstance(evidence_rows, list):
            raise ProjectError(f"research bundle evidence for {finding_id!r} must be an array")
        for evidence in evidence_rows:
            if (
                not isinstance(evidence, dict)
                or not isinstance(evidence.get("evidence_id"), str)
                or not evidence["evidence_id"]
            ):
                raise ProjectError("research bundle contains evidence without a valid ID")
            if version in _FINDING_SPAN_BUNDLE_VERSIONS:
                kind = evidence.get("kind")
                span_id = evidence.get("source_span_id")
                source_revision_id = evidence.get("source_revision_id")
                role = evidence.get("role")
                if (
                    not isinstance(role, str)
                    or role
                    not in {
                        "SUPPORT",
                        "COUNTEREVIDENCE",
                        "EXCEPTION",
                        "EXAMPLE",
                        "CONTEXT",
                        "ALTERNATIVE",
                    }
                    or not isinstance(kind, str)
                    or kind
                    not in {
                        "TEXTUAL",
                        "CORPUS_STATISTIC",
                        "EXTERNAL_SOURCE",
                        "USER_CONTEXT",
                    }
                ):
                    raise ProjectError("research bundle contains an invalid evidence role or kind")
                if (
                    not isinstance(evidence.get("text"), str)
                    or not isinstance(evidence.get("citation"), str)
                    or (
                        evidence.get("rationale") is not None
                        and not isinstance(evidence.get("rationale"), str)
                    )
                ):
                    raise ProjectError("research bundle contains invalid evidence metadata")
                if kind == "TEXTUAL":
                    if (
                        not isinstance(span_id, str)
                        or not span_id
                        or not isinstance(source_revision_id, str)
                        or not source_revision_id
                    ):
                        raise ProjectError(
                            "research bundle textual evidence has an invalid source span reference"
                        )
                elif span_id is not None or source_revision_id is not None:
                    raise ProjectError(
                        "research bundle non-textual evidence has an invalid source span reference"
                    )
            if evidence.get("kind") == "TEXTUAL":
                textual_evidence_rows.append(evidence)
            evidence_count += 1

    finding_evidence_spans_by_id: dict[str, dict[str, Any]] = {}
    if version in _FINDING_SPAN_BUNDLE_VERSIONS:
        for source_span in payload["finding_evidence_spans"]:
            if (
                not isinstance(source_span, dict)
                or set(source_span)
                != {
                    "id",
                    "revision_id",
                    "start_offset",
                    "end_offset",
                    "text_hash",
                    "canonical_text_length",
                }
                or not isinstance(source_span.get("id"), str)
                or not source_span["id"]
                or source_span["id"] in finding_evidence_spans_by_id
                or not isinstance(source_span.get("revision_id"), str)
                or not source_span["revision_id"]
                or type(source_span.get("start_offset")) is not int
                or type(source_span.get("end_offset")) is not int
                or type(source_span.get("canonical_text_length")) is not int
                or not isinstance(source_span.get("text_hash"), str)
                or source_span["start_offset"] < 0
                or source_span["end_offset"] <= source_span["start_offset"]
                or source_span["canonical_text_length"] < source_span["end_offset"]
                or re.fullmatch(r"[a-f0-9]{64}", source_span["text_hash"]) is None
            ):
                raise ProjectError(
                    "research bundle contains an invalid finding evidence source span"
                )
            finding_evidence_spans_by_id[source_span["id"]] = source_span

        referenced_source_span_ids: set[str] = set()
        for evidence in textual_evidence_rows:
            span_id = evidence["source_span_id"]
            source_span = finding_evidence_spans_by_id.get(span_id)
            if source_span is None:
                raise ProjectError(
                    "research bundle textual evidence references a missing source span"
                )
            if evidence["source_revision_id"] != source_span["revision_id"]:
                raise ProjectError(
                    "research bundle textual evidence source revision does not match its source span"
                )
            evidence_text = evidence.get("text")
            if not isinstance(evidence_text, str) or (
                source_span["end_offset"] - source_span["start_offset"] != len(evidence_text)
            ):
                raise ProjectError(
                    "research bundle textual evidence span length does not match its excerpt"
                )
            if (
                hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()
                != source_span["text_hash"]
            ):
                raise ProjectError(
                    "research bundle textual evidence text does not match its source span"
                )
            referenced_source_span_ids.add(span_id)
        if referenced_source_span_ids != set(finding_evidence_spans_by_id):
            raise ProjectError(
                "research bundle finding evidence source spans do not match textual evidence"
            )

    revisions = payload["document_revisions"]
    revision_ids: set[str] = set()
    for revision in revisions:
        if (
            not isinstance(revision, dict)
            or not isinstance(revision.get("id"), str)
            or not revision["id"]
        ):
            raise ProjectError("research bundle contains a document revision without a valid ID")
        if revision["id"] in revision_ids:
            raise ProjectError("research bundle contains duplicate document revision IDs")
        if version in _INTEGRITY_BUNDLE_VERSIONS and any(
            not isinstance(revision.get(key), str)
            or re.fullmatch(r"[a-f0-9]{64}", revision[key]) is None
            for key in ("source_hash", "canonical_hash")
        ):
            raise ProjectError("research bundle contains a malformed document revision hash")
        if version == BUNDLE_VERSION:
            _validate_document_revision_provenance(revision, boundary="research bundle")
        revision_ids.add(revision["id"])

    if version in _FINDING_SPAN_BUNDLE_VERSIONS:
        for source_span in finding_evidence_spans_by_id.values():
            if source_span["revision_id"] not in revision_ids:
                raise ProjectError(
                    "research bundle finding evidence source span belongs to an out-of-scope revision"
                )

    canonical_texts_by_revision: dict[str, dict[str, Any]] = {}
    if version in _CANONICAL_TEXT_BUNDLE_VERSIONS:
        revisions_by_id = {revision["id"]: revision for revision in revisions}
        for canonical_text in payload["canonical_texts"]:
            if (
                not isinstance(canonical_text, dict)
                or set(canonical_text) != {"revision_id", "sha256", "text"}
                or not isinstance(canonical_text.get("revision_id"), str)
                or not canonical_text["revision_id"]
                or canonical_text["revision_id"] in canonical_texts_by_revision
                or not isinstance(canonical_text.get("sha256"), str)
                or re.fullmatch(r"[a-f0-9]{64}", canonical_text["sha256"]) is None
                or not isinstance(canonical_text.get("text"), str)
            ):
                raise ProjectError("research bundle contains malformed canonical source text")
            revision_id = canonical_text["revision_id"]
            revision = revisions_by_id.get(revision_id)
            if revision is None:
                raise ProjectError(
                    "research bundle canonical source text belongs to an out-of-scope revision"
                )
            text_hash = hashlib.sha256(canonical_text["text"].encode("utf-8")).hexdigest()
            if (
                not hmac.compare_digest(text_hash, canonical_text["sha256"])
                or not hmac.compare_digest(text_hash, revision["canonical_hash"])
            ):
                raise ProjectError(
                    "research bundle canonical source text does not match its revision hash"
                )
            canonical_texts_by_revision[revision_id] = canonical_text

        referenced_revision_ids = {
            source_span["revision_id"]
            for source_span in finding_evidence_spans_by_id.values()
        }
        included_revision_ids = set(canonical_texts_by_revision)
        if included_revision_ids and included_revision_ids != referenced_revision_ids:
            raise ProjectError(
                "research bundle canonical source texts do not match cited revisions"
            )
        if canonical_texts_by_revision:
            for evidence in textual_evidence_rows:
                source_span = finding_evidence_spans_by_id[evidence["source_span_id"]]
                canonical_text = canonical_texts_by_revision[source_span["revision_id"]]["text"]
                start_offset = source_span["start_offset"]
                end_offset = source_span["end_offset"]
                if (
                    len(canonical_text) != source_span["canonical_text_length"]
                    or end_offset > len(canonical_text)
                    or canonical_text[start_offset:end_offset] != evidence["text"]
                ):
                    raise ProjectError(
                        "research bundle canonical source text does not match cited evidence location"
                    )

    if version >= 13:
        for source_span in payload["qualitative_coding"]["source_spans"]:
            if source_span["revision_id"] not in revision_ids:
                raise ProjectError(
                    "research bundle qualitative coding source span belongs to an out-of-scope revision"
                )

    integrity_status = "NOT_AVAILABLE_LEGACY"
    if version in _INTEGRITY_BUNDLE_VERSIONS:
        integrity = payload.get("integrity")
        if (
            not isinstance(integrity, dict)
            or set(integrity) != {"algorithm", "digest"}
            or integrity.get("algorithm") != "sha256-canonical-json-v1"
        ):
            raise ProjectError("research bundle has missing or unsupported integrity metadata")
        expected = integrity.get("digest")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ProjectError("research bundle integrity digest is malformed")
        unsigned_payload = {key: value for key, value in payload.items() if key != "integrity"}
        try:
            actual = hashlib.sha256(_canonical_json(unsigned_payload)).hexdigest()
        except (TypeError, ValueError, RecursionError) as exc:
            raise ProjectError(
                "research bundle contains values that cannot be canonically encoded"
            ) from exc
        if not hmac.compare_digest(actual, expected):
            raise ProjectError("research bundle integrity check failed")
        integrity_status = "VERIFIED_SHA256"

    # Legacy v10 bundles lack an integrity digest, but the verifier still
    # returns their provenance to callers. Apply the export-boundary metadata
    # checks to every accepted version before reporting any bundle as valid.
    run_config = payload["analysis_run"].get("config")
    if _contains_secret_metadata_key(run_config, allow_counterevidence_query_labels=True):
        raise ProjectError("research bundle contains secret-bearing configuration metadata")
    if _contains_absolute_host_path(run_config):
        raise ProjectError("research bundle contains absolute host paths in portable run metadata")
    for execution in payload["model_executions"]:
        if not isinstance(execution, dict):
            raise ProjectError("research bundle contains invalid model execution metadata")
        skill_run_id = execution.get("skill_run_id")
        if skill_run_id is not None:
            linked_skill_run = (
                skill_runs_by_id.get(skill_run_id)
                if isinstance(skill_run_id, str) and skill_run_id
                else None
            )
            if (
                linked_skill_run is None
                or linked_skill_run.get("analysis_run_id") != payload["analysis_run"].get("id")
                or execution.get("analysis_run_id") != payload["analysis_run"].get("id")
            ):
                raise ProjectError(
                    "research bundle model execution references an invalid skill run"
                )
        if "endpoint" in execution:
            _validate_model_execution_endpoint(execution["endpoint"], boundary="research bundle")
        for field in ("parameters_json", "capabilities_json"):
            if field in execution:
                _validate_model_execution_metadata(execution[field], boundary="research bundle")
        if "model_price_json" in execution:
            _validate_model_price_snapshot(
                execution["model_price_json"],
                boundary="research bundle",
                execution_provider=execution.get("provider_id"),
                execution_model=execution.get("model"),
            )

    if not textual_evidence_rows:
        source_location_verification = "NOT_APPLICABLE"
    elif canonical_texts_by_revision:
        source_location_verification = "VERIFIED_CANONICAL_TEXT"
    else:
        source_location_verification = "NOT_VERIFIED_CANONICAL_TEXT_NOT_INCLUDED"
    return {
        "status": "VALID",
        "bundle_version": version,
        "analysis_run_id": payload["analysis_run"]["id"],
        "finding_count": len(finding_ids),
        "evidence_count": evidence_count,
        "document_revision_count": len(revision_ids),
        "integrity": integrity_status,
        "source_location_verification": source_location_verification,
    }
