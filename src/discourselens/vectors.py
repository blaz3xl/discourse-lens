"""Project-local, rebuildable sqlite-vec indexes with physical scope isolation."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import sqlite3
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .artifact_integrity import sha256_file, verify_project_artifact
from .errors import (
    ArtifactIntegrityError,
    OptionalDependencyError,
    ScopeViolationError,
)
from .execution import (
    build_embedding_gateway,
    effective_embedding_endpoint,
    require_embedding_model_capability,
)
from .models import EmbeddingProvider
from .network_policy import is_loopback_endpoint
from .project import Project
from .providers import (
    CohereEmbeddingProvider,
    GeminiEmbeddingProvider,
    JinaEmbeddingProvider,
    VoyageEmbeddingProvider,
    embedding_request_payload,
)
from .secrets import provider_auth_reference
from .types import (
    CoverageDescriptor,
    CoverageMode,
    CoverageTerminationState,
    NegativeResultStatus,
)

MAX_PRESET_QUERY_CHARS = 16_000
MAX_VECTOR_SEARCH_RESULTS = 1_000
_ENDPOINT_FINGERPRINT_PREFIX = "embedding-endpoint-v1:"
_SQLITE_IN_BATCH_SIZE = 900


def _is_embedding_endpoint_fingerprint(value: object) -> bool:
    if (
        not isinstance(value, str)
        or not value.startswith(_ENDPOINT_FINGERPRINT_PREFIX)
    ):
        return False
    digest = value.removeprefix(_ENDPOINT_FINGERPRINT_PREFIX)
    if len(digest) != 64 or digest.lower() != digest:
        return False
    try:
        int(digest, 16)
    except ValueError:
        return False
    return True


def _embedding_endpoint_fingerprint(
    provider_id: str,
    credential_alias: str,
    secret_reference: str,
    endpoint: str | None,
) -> str:
    """Fingerprint endpoint selection without retaining URL or secret material.

    Provider, credential alias/reference, and model remain part of the embedding
    provenance contract. Only a digest of the selected endpoint identity is stored;
    endpoint userinfo, query strings, and fragments are rejected because they can
    contain credentials or identity-bearing values that cannot be safely retained.
    """
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (provider_id, credential_alias, secret_reference)
    ):
        raise ScopeViolationError("embedding endpoint identity metadata is incomplete")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ScopeViolationError("embedding endpoint identity is unavailable")
    if (
        endpoint != endpoint.strip()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in endpoint)
        or "?" in endpoint
        or "#" in endpoint
    ):
        raise ScopeViolationError(
            "embedding endpoint must be an HTTP(S) base URL without userinfo, query, or fragment"
        )
    try:
        parsed = urlparse(endpoint)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ScopeViolationError("embedding endpoint identity is invalid") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ScopeViolationError(
            "embedding endpoint must be an HTTP(S) base URL without userinfo, query, or fragment"
        )
    scheme = parsed.scheme.lower()
    normalized_host = hostname.lower()
    if ":" in normalized_host and not normalized_host.startswith("["):
        normalized_host = f"[{normalized_host}]"
    if port is not None and port != (443 if scheme == "https" else 80):
        normalized_host = f"{normalized_host}:{port}"
    path = parsed.path.rstrip("/")
    endpoint_identity = f"{scheme}://{normalized_host}{path}"
    material = json.dumps(
        {
            "provider_id": provider_id,
            "credential_alias": credential_alias,
            "secret_reference": secret_reference,
            "endpoint": endpoint_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{_ENDPOINT_FINGERPRINT_PREFIX}{digest}"


def _require_embedding_provider_endpoint_matches_preset(
    embedding_provider: EmbeddingProvider,
    provider_id: str,
    credential_alias: str,
    secret_reference: str,
    configured_endpoint: str | None,
) -> str:
    """Bind an injected adapter's declared endpoint to the selected credential.

    An injected implementation must declare the effective endpoint so the preset
    provenance cannot silently fingerprint one deployment while sending to another.
    This is an adapter contract, not a sandbox for arbitrary caller-supplied code.
    """
    provider_endpoint = getattr(embedding_provider, "endpoint", None)
    if not isinstance(provider_endpoint, str) or not provider_endpoint.strip():
        raise ScopeViolationError(
            "embedding provider must declare an endpoint matching the selected credential"
        )
    expected_fingerprint = _embedding_endpoint_fingerprint(
        provider_id, credential_alias, secret_reference, configured_endpoint
    )
    provider_fingerprint = _embedding_endpoint_fingerprint(
        provider_id, credential_alias, secret_reference, provider_endpoint
    )
    if provider_fingerprint != expected_fingerprint:
        raise ScopeViolationError(
            "embedding provider endpoint does not match the selected credential"
        )
    return expected_fingerprint


@dataclass(frozen=True)
class VectorChunk:
    chunk_id: str
    start: int
    end: int
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class VectorHit:
    """One bounded nearest-neighbor candidate, not evidence of corpus coverage."""

    artifact_id: str
    revision_id: str
    chunk_id: str
    start: int
    end: int
    distance: float


@dataclass(frozen=True)
class HybridHit:
    """A rank-fused candidate; this value does not certify corpus coverage.

    Empty or truncated top-k results cannot support absence, universal, or exact
    corpus-count claims. Such claims need a separate CoverageDescriptor and, where
    required, deterministic exhaustive computation.
    """

    revision_id: str
    unit_id: str
    lexical_rank: int | None
    vector_rank: int | None
    fusion_score: float
    artifact_id: str | None = None


@dataclass(frozen=True)
class _HybridSearchOutcome:
    hits: list[HybridHit]
    vector_metric: str
    candidate_count: int
    vector_candidate_count: int
    lexical_candidate_count: int
    lexical_candidate_limit: int
    lexical_candidate_limit_reached: bool
    fusion_limit_reached: bool


class RetrievalSearchResult[HitT](list[HitT]):
    """Backward-compatible list envelope with provenance available for empty results."""

    __slots__ = ("retrieval_run_id", "scope_id")

    def __init__(
        self, scope_id: str, retrieval_run_id: str, hits: list[HitT]
    ) -> None:
        super().__init__(hits)
        self.scope_id = scope_id
        self.retrieval_run_id = retrieval_run_id

    @property
    def hits(self) -> RetrievalSearchResult[HitT]:
        return self


@dataclass(frozen=True)
class EmbeddingPreviewInput:
    paragraph_id: str
    start_offset: int
    end_offset: int
    character_count: int
    text_sha256: str
    prepared_character_count: int
    prepared_text_sha256: str
    prepared_text: str | None = None


@dataclass(frozen=True)
class EmbeddingPreviewBatch:
    index: int
    method: str
    path: str
    inputs: tuple[EmbeddingPreviewInput, ...]
    request_body_template: dict[str, object]
    request_body: dict[str, object] | None


@dataclass(frozen=True)
class ParagraphEmbeddingRequestPreview:
    """Local, secret-free preview of a selected paragraph embedding request window."""

    provider_id: str
    credential_alias: str
    model: str
    endpoint: str
    revision_id: str
    input_type: str
    total_paragraphs: int
    offset: int
    selected_count: int
    include_text: bool
    offline_mode: bool
    blocked_by_offline_mode: bool
    secret_values_resolved: bool
    network_request_made: bool
    batches: tuple[EmbeddingPreviewBatch, ...]


def _validate_preset_query_text(query_text: object, *, operation: str) -> str:
    """Bound preset-backed query text before any project credential lookup or egress."""
    if not isinstance(query_text, str) or not query_text.strip():
        raise ScopeViolationError(f"{operation} query must be non-empty text")
    if len(query_text) > MAX_PRESET_QUERY_CHARS:
        raise ScopeViolationError(
            f"{operation} query exceeds the {MAX_PRESET_QUERY_CHARS}-character limit"
        )
    return query_text


def _validate_vector_search_limit(limit: object, *, operation: str) -> int:
    """Keep per-index ANN fan-out and machine-readable results bounded."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_VECTOR_SEARCH_RESULTS:
        raise ScopeViolationError(
            f"{operation} limit must be between 1 and {MAX_VECTOR_SEARCH_RESULTS}"
        )
    return limit


def _validate_vector_values(values: tuple[float, ...], label: str) -> None:
    """Reject values sqlite-vec cannot preserve as finite float32 components."""
    if not values:
        raise ScopeViolationError(f"{label} must contain at least one finite numeric value")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ScopeViolationError(f"{label} must contain only finite numeric values")
        try:
            numeric = float(value)
            float32 = struct.unpack("<f", struct.pack("<f", numeric))[0]
        except (OverflowError, ValueError, struct.error):
            numeric = float("inf")
            float32 = float("inf")
        if (
            not math.isfinite(numeric)
            or not math.isfinite(float32)
            or (numeric != 0.0 and float32 == 0.0)
        ):
            raise ScopeViolationError(f"{label} must contain only finite numeric values")


def _has_nonzero_vector_component(values: tuple[float, ...] | list[float]) -> bool:
    """Check whether a validated vector has a defined cosine direction."""
    return any(float(value) != 0.0 for value in values)


def _query_vector_sha256(values: tuple[float, ...]) -> str:
    """Hash the effective float32 vector without retaining its components."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<f", float(value)))
    return digest.hexdigest()


def _vector_index_dimension(schema_version: object) -> int:
    """Read a supported vector-index dimension from registry metadata."""
    if not isinstance(schema_version, str):
        raise ScopeViolationError("vector artifact has invalid dimension metadata")
    version, separator, raw_dimension = schema_version.rpartition(":")
    if version not in {
        "vector-index-v1",
        "vector-index-v2",
        "vector-index-v3",
        "vector-index-v4",
    } or not separator:
        raise ScopeViolationError("vector artifact has invalid dimension metadata")
    try:
        dimension = int(raw_dimension)
    except ValueError as exc:
        raise ScopeViolationError("vector artifact has invalid dimension metadata") from exc
    if dimension < 1 or str(dimension) != raw_dimension:
        raise ScopeViolationError("vector artifact has invalid dimension metadata")
    return dimension


def _vector_index_metric(schema_version: object) -> str:
    """Return the metric fixed by an immutable vector-index schema version."""
    _vector_index_dimension(schema_version)
    assert isinstance(schema_version, str)
    return "cosine" if schema_version.startswith("vector-index-v4:") else "L2"


def _distance_score_definition(metric: str) -> str:
    score = "cosine distance" if metric == "cosine" else "squared-L2 score"
    return (
        f"sqlite-vec {score}; lower is nearer; score is an ordering value, "
        "not a calibrated similarity"
    )


def _vector_metric_policy_name(metric: str) -> str:
    return "cosine-distance" if metric == "cosine" else "squared-L2-score"


def _require_uniform_vector_metric(
    artifacts: list[sqlite3.Row], *, operation: str
) -> str:
    """Reject incomparable metric values before searching or fusing candidates."""
    metrics = {_vector_index_metric(artifact["schema_version"]) for artifact in artifacts}
    if len(metrics) > 1:
        raise ScopeViolationError(
            f"{operation} cannot mix L2 and cosine vector indexes"
        )
    return next(iter(metrics))


def _vector_index_provenance(
    artifact: sqlite3.Row, embedding_endpoint_fingerprint: str | None
) -> tuple[object, ...]:
    """Return a checked signature used to reject incomparable vector spaces."""
    schema_version = artifact["schema_version"]
    dimension = _vector_index_dimension(schema_version)
    version = schema_version.partition(":")[0]
    provider_id = artifact["embedding_provider_id"]
    model = artifact["embedding_model"]
    stored_dimension = artifact["embedding_dimension"]
    chunker_version = artifact["chunker_version"]
    input_type = artifact["embedding_input_type"]
    metadata = (provider_id, model, stored_dimension, chunker_version, input_type)

    if version == "vector-index-v1":
        if any(value is not None for value in metadata):
            raise ScopeViolationError("vector artifact has malformed legacy provenance metadata")
        return (version, dimension, _vector_index_metric(schema_version), None, None, None, None)
    if version == "vector-index-v4" and not any(value is not None for value in metadata):
        if embedding_endpoint_fingerprint is not None:
            raise ScopeViolationError("vector artifact has malformed embedding provenance metadata")
        return (version, dimension, _vector_index_metric(schema_version), None, None, None, None)
    if (
        not isinstance(provider_id, str)
        or not provider_id.strip()
        or not isinstance(model, str)
        or not model.strip()
        or type(stored_dimension) is not int
        or stored_dimension != dimension
        or not isinstance(chunker_version, str)
        or not chunker_version.strip()
        or not isinstance(input_type, str)
        or not input_type.strip()
    ):
        raise ScopeViolationError("vector artifact has malformed embedding provenance metadata")
    return (
        version,
        dimension,
        _vector_index_metric(schema_version),
        provider_id,
        model,
        chunker_version,
        input_type,
        embedding_endpoint_fingerprint
        if version in {"vector-index-v3", "vector-index-v4"}
        else None,
    )


def _require_uniform_vector_provenance(
    project: Project,
    artifacts: list[sqlite3.Row],
    *,
    operation: str,
    expected_embedding_endpoint_fingerprint: str | None = None,
) -> None:
    """Do not compare distances from different embedding or chunking spaces."""
    # Callers pass rows only after validating the artifact IDs against the
    # explicit scope, so endpoint metadata is never inspected out of scope.
    signatures: set[tuple[object, ...]] = set()
    for artifact in artifacts:
        _, endpoint_fingerprint = _verify_vector_index_identity(
            project,
            artifact,
            expected_embedding_endpoint_fingerprint=(
                expected_embedding_endpoint_fingerprint
            ),
        )
        signatures.add(_vector_index_provenance(artifact, endpoint_fingerprint))
    if len(signatures) > 1:
        raise ScopeViolationError(
            f"{operation} cannot mix vector indexes with incompatible embedding or chunker provenance"
        )


def _validate_vector_artifact_ids(artifact_ids: object) -> list[str]:
    if not isinstance(artifact_ids, list) or not artifact_ids or any(
        not isinstance(artifact_id, str)
        or not artifact_id
        or artifact_id != artifact_id.strip()
        for artifact_id in artifact_ids
    ):
        raise ScopeViolationError("vector search requires non-empty artifact IDs")
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ScopeViolationError("vector artifact IDs must not be duplicated")
    return artifact_ids


def _artifact_rows_for_ids(
    project: Project,
    artifact_ids: list[str],
    columns: str,
    *,
    vector_indexes_only: bool = False,
) -> list[sqlite3.Row]:
    """Resolve artifact IDs in portable SQLite-sized batches, preserving input order."""
    rows_by_id: dict[str, sqlite3.Row] = {}
    type_filter = " AND artifact_type='vector_index'" if vector_indexes_only else ""
    for offset in range(0, len(artifact_ids), _SQLITE_IN_BATCH_SIZE):
        batch = artifact_ids[offset : offset + _SQLITE_IN_BATCH_SIZE]
        placeholders = ",".join("?" for _ in batch)
        rows = project.db.execute(
            f"SELECT {columns} FROM artifact_registry "
            f"WHERE id IN ({placeholders}){type_filter}",
            batch,
        ).fetchall()
        rows_by_id.update({str(row["id"]): row for row in rows})
    return [rows_by_id[artifact_id] for artifact_id in artifact_ids if artifact_id in rows_by_id]


def _validated_scope_revisions(project: Project, scope_id: object) -> set[str]:
    if not isinstance(scope_id, str) or not scope_id or scope_id != scope_id.strip():
        raise ScopeViolationError("scope is malformed or unknown")
    try:
        revisions = project.scope_revisions(scope_id)
    except ScopeViolationError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise ScopeViolationError("scope has malformed revision metadata") from exc
    if (
        not isinstance(revisions, list)
        or not revisions
        or any(
            not isinstance(revision_id, str)
            or not revision_id
            or revision_id != revision_id.strip()
            for revision_id in revisions
        )
        or len(revisions) != len(set(revisions))
    ):
        raise ScopeViolationError("scope has malformed revision metadata")

    found: set[str] = set()
    # Keep each lookup below SQLite's conservative parameter limit so large,
    # otherwise valid corpus scopes remain searchable.
    for offset in range(0, len(revisions), 900):
        batch = revisions[offset : offset + 900]
        placeholders = ",".join("?" for _ in batch)
        found.update(
            row[0]
            for row in project.db.execute(
                f"SELECT id FROM document_revisions WHERE id IN ({placeholders})", batch
            )
            if isinstance(row[0], str)
        )
    if found != set(revisions):
        raise ScopeViolationError("scope references an unknown project revision")
    return found


def _scoped_vector_artifacts(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    *,
    operation: str,
) -> list[sqlite3.Row]:
    """Resolve one compatible vector index per selected revision before retrieval."""
    artifact_ids = _validate_vector_artifact_ids(artifact_ids)
    scope = _validated_scope_revisions(project, scope_id)
    artifacts = _artifact_rows_for_ids(
        project,
        artifact_ids,
        "id,revision_id,relative_path,schema_version,file_hash,embedding_provider_id,"
        "embedding_model,embedding_dimension,chunker_version,embedding_input_type",
        vector_indexes_only=True,
    )
    if len(artifacts) != len(artifact_ids) or any(
        not isinstance(row["id"], str)
        or not isinstance(row["revision_id"], str)
        or row["revision_id"] not in scope
        for row in artifacts
    ):
        raise ScopeViolationError(
            "vector artifact is unknown, not a vector index, or outside the explicit scope"
        )
    _require_uniform_vector_metric(artifacts, operation=operation)
    registry_signatures = {
        _vector_index_provenance(artifact, None) for artifact in artifacts
    }
    if len(registry_signatures) > 1:
        raise ScopeViolationError(
            f"{operation} cannot mix vector indexes with incompatible embedding or chunker provenance"
        )
    revision_ids = [str(row["revision_id"]) for row in artifacts]
    if len(revision_ids) != len(set(revision_ids)):
        raise ScopeViolationError(
            f"{operation} accepts at most one vector artifact per scoped revision"
        )
    _require_uniform_vector_provenance(project, artifacts, operation=operation)
    return artifacts


def _vector_index_relative_path(artifact_id: str) -> Path:
    return Path("artifacts") / "vectors" / f"{artifact_id}.sqlite"


def _verify_vector_index_identity(
    project: Project,
    artifact: sqlite3.Row,
    *,
    expected_embedding_endpoint_fingerprint: str | None = None,
) -> tuple[Path, str | None]:
    """Bind registry metadata to the project and revision stored inside the index."""
    artifact_id = artifact["id"]
    revision_id = artifact["revision_id"]
    relative_path = artifact["relative_path"]
    if (
        not isinstance(artifact_id, str)
        or not artifact_id
        or not isinstance(revision_id, str)
        or not revision_id
    ):
        raise ScopeViolationError("vector artifact has malformed identity metadata; rebuild it")
    if relative_path != str(_vector_index_relative_path(artifact_id)):
        raise ArtifactIntegrityError("vector index path is invalid or unavailable")
    if not isinstance(artifact["file_hash"], str) or not artifact["file_hash"]:
        raise ScopeViolationError("vector artifact is missing integrity metadata; rebuild it")

    artifact_path = verify_project_artifact(project.root, relative_path, artifact["file_hash"])
    connection = sqlite3.connect(artifact_path)
    connection.row_factory = sqlite3.Row
    try:
        try:
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(vector_index_identity)"
                ).fetchall()
            }
            has_endpoint_fingerprint = "embedding_endpoint_fingerprint" in columns
            identity_columns = "project_id,artifact_id,revision_id"
            if has_endpoint_fingerprint:
                identity_columns += ",embedding_endpoint_fingerprint"
            identity_rows = connection.execute(
                f"SELECT {identity_columns} FROM vector_index_identity"
            ).fetchall()
        except sqlite3.Error as exc:
            raise ScopeViolationError(
                "vector index identity metadata is missing or malformed; rebuild it"
            ) from exc
    finally:
        connection.close()

    expected = (project.project_id, artifact_id, revision_id)
    identity_values = tuple(identity_rows[0]) if len(identity_rows) == 1 else ()
    if (
        len(identity_values) not in {3, 4}
        or any(not isinstance(value, str) for value in identity_values[:3])
        or identity_values[:3] != expected
    ):
        raise ScopeViolationError(
            "vector index identity does not match its project and revision; rebuild it"
        )
    schema_version = artifact["schema_version"]
    stored_fingerprint = (
        identity_rows[0]["embedding_endpoint_fingerprint"]
        if has_endpoint_fingerprint
        else None
    )
    legacy_requires_endpoint = (
        isinstance(schema_version, str) and schema_version.startswith("vector-index-v3:")
    )
    optional_v4_endpoint_is_present = (
        isinstance(schema_version, str)
        and schema_version.startswith("vector-index-v4:")
        and stored_fingerprint is not None
    )
    if (legacy_requires_endpoint or optional_v4_endpoint_is_present) and not (
        _is_embedding_endpoint_fingerprint(stored_fingerprint)
    ):
        raise ScopeViolationError(
            "vector index endpoint provenance is missing or malformed; rebuild the index"
        )
    if expected_embedding_endpoint_fingerprint is not None and (
        not has_endpoint_fingerprint
        or stored_fingerprint != expected_embedding_endpoint_fingerprint
    ):
        raise ScopeViolationError(
            "vector index endpoint provenance is missing or incompatible; rebuild the index"
        )
    return (
        artifact_path,
        stored_fingerprint if isinstance(stored_fingerprint, str) else None,
    )


def _sqlite_vec():
    try:
        import sqlite_vec
    except ImportError as exc:
        raise OptionalDependencyError("vector indexes require discourselens[vectors]") from exc
    return sqlite_vec


def _connection(path: Path):
    import sqlite3

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        connection.enable_load_extension(True)
        try:
            _sqlite_vec().load(connection)
        finally:
            connection.enable_load_extension(False)
        return connection
    except BaseException:
        connection.close()
        raise


def _validate_vector_metric_ddl(
    connection: sqlite3.Connection, expected_metric: str, *, error_message: str
) -> None:
    """Bind an artifact's declared metric to the actual sqlite-vec table DDL."""
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors'"
    ).fetchone()
    if row is None or not isinstance(row["sql"], str):
        raise ArtifactIntegrityError(error_message)
    ddl = "".join(row["sql"].lower().split())
    has_metric = "distance_metric=" in ddl
    if expected_metric == "cosine":
        matches = "distance_metric=cosine" in ddl
    else:
        # Schema versions v1-v3 predate the explicit option and use sqlite-vec's
        # default L2 metric. Accept that historical DDL and an explicit L2 form,
        # but reject a different declared metric.
        matches = not has_metric or "distance_metric=l2" in ddl
    if not matches:
        raise ArtifactIntegrityError(error_message)


def _validate_published_vector_metric(
    project: Project, artifact: sqlite3.Row
) -> None:
    """Check a published artifact's stored metric before preset query egress."""
    artifact_path = verify_project_artifact(
        project.root, artifact["relative_path"], artifact["file_hash"]
    )
    try:
        connection = sqlite3.connect(artifact_path)
    except sqlite3.Error as exc:
        raise ArtifactIntegrityError(
            "vector artifact metric does not match its schema version"
        ) from exc
    connection.row_factory = sqlite3.Row
    try:
        _validate_vector_metric_ddl(
            connection,
            _vector_index_metric(artifact["schema_version"]),
            error_message="vector artifact metric does not match its schema version",
        )
    except sqlite3.Error as exc:
        raise ArtifactIntegrityError(
            "vector artifact metric does not match its schema version"
        ) from exc
    finally:
        connection.close()


def _validate_built_vector_index(
    path: Path,
    *,
    project_id: str,
    artifact_id: str,
    revision_id: str,
    chunks: list[VectorChunk],
    embedding_endpoint_fingerprint: str | None,
) -> None:
    """Verify a completed staged index before it is made visible to the project.

    The registry hash protects a published artifact from later modification; it
    does not establish that the file was internally complete when first
    registered. Validate both SQLite structure and the exact chunk/vector rows
    produced by the build before publication.
    """
    connection = _connection(path)
    try:
        integrity = [str(row[0]) for row in connection.execute("PRAGMA quick_check")]
        if integrity != ["ok"]:
            raise ArtifactIntegrityError("staged vector index failed SQLite integrity validation")

        identity = connection.execute(
            "SELECT project_id,artifact_id,revision_id,embedding_endpoint_fingerprint "
            "FROM vector_index_identity"
        ).fetchall()
        expected_identity = (
            project_id,
            artifact_id,
            revision_id,
            embedding_endpoint_fingerprint,
        )
        if len(identity) != 1 or tuple(identity[0]) != expected_identity:
            raise ArtifactIntegrityError("staged vector index identity validation failed")

        actual_chunks = [
            tuple(row)
            for row in connection.execute(
                "SELECT id,start_offset,end_offset FROM chunks ORDER BY id"
            )
        ]
        expected_chunks = sorted(
            ((chunk.chunk_id, chunk.start, chunk.end) for chunk in chunks),
            key=lambda row: row[0],
        )
        if actual_chunks != expected_chunks:
            raise ArtifactIntegrityError("staged vector index chunk validation failed")

        actual_vectors = [
            tuple(row)
            for row in connection.execute("SELECT id,embedding FROM vectors ORDER BY id")
        ]
        _validate_vector_metric_ddl(
            connection,
            "cosine",
            error_message="staged vector index metric validation failed",
        )
        expected_vectors = sorted(
            (
                (chunk.chunk_id, struct.pack(f"<{len(chunk.embedding)}f", *chunk.embedding))
                for chunk in chunks
            ),
            key=lambda row: row[0],
        )
        if actual_vectors != expected_vectors:
            raise ArtifactIntegrityError("staged vector index embedding validation failed")
    except sqlite3.Error as exc:
        raise ArtifactIntegrityError("staged vector index could not be validated") from exc
    finally:
        connection.close()


def _embedding_endpoint(provider_id: str, configured_endpoint: str | None) -> str | None:
    return effective_embedding_endpoint(provider_id, configured_endpoint)


def _document_embedding_input_type(provider_id: str, model: str) -> str:
    if provider_id == "cohere-embeddings":
        return "search_document"
    if provider_id == "voyage-embeddings":
        return "document"
    if provider_id == "jina-embeddings":
        return JinaEmbeddingProvider.document_input_type(model)
    if provider_id == "gemini-embeddings":
        return GeminiEmbeddingProvider.document_input_type(model)
    return "symmetric"


def _require_embedding_provider_matches_preset(
    embedding_provider: EmbeddingProvider, preset_provider_id: object
) -> None:
    if embedding_provider.provider_id != preset_provider_id:
        raise ScopeViolationError("embedding provider does not match selected embedding preset")


def _embedding_preview_endpoint(provider_id: str, configured_endpoint: str | None) -> str:
    endpoint = _embedding_endpoint(provider_id, configured_endpoint)
    if endpoint is None:
        raise ScopeViolationError(
            "embedding preview requires a supported provider or explicit compatible endpoint"
        )
    parsed = urlparse(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ScopeViolationError("embedding endpoint must be an HTTP(S) URL without credentials")
    return endpoint.rstrip("/")


_DEFAULT_EMBEDDING_BATCH_SIZE = 64
_OLLAMA_EMBEDDING_BATCH_SIZE = 8


def _embedding_batch_size(provider_id: str) -> int:
    """Return the shared conservative request size for preview and index builds."""
    if provider_id == "ollama":
        declared_limit = _OLLAMA_EMBEDDING_BATCH_SIZE
    elif provider_id == "cohere-embeddings":
        declared_limit = getattr(CohereEmbeddingProvider, "MAX_INPUTS_PER_REQUEST", None)
    elif provider_id == "voyage-embeddings":
        declared_limit = getattr(VoyageEmbeddingProvider, "MAX_INPUTS_PER_REQUEST", None)
    elif provider_id == "jina-embeddings":
        declared_limit = getattr(JinaEmbeddingProvider, "MAX_INPUTS_PER_REQUEST", None)
    elif provider_id == "gemini-embeddings":
        declared_limit = getattr(GeminiEmbeddingProvider, "MAX_INPUTS_PER_REQUEST", None)
    else:
        declared_limit = None
    if declared_limit is None:
        return _DEFAULT_EMBEDDING_BATCH_SIZE
    if type(declared_limit) is not int or declared_limit < 1:
        raise ScopeViolationError("embedding provider has an invalid request batch limit")
    return min(_DEFAULT_EMBEDDING_BATCH_SIZE, declared_limit)


def _embedding_preview_request_body(
    provider_id: str, model: str, texts: list[str]
) -> tuple[str, dict[str, object]]:
    return embedding_request_payload(provider_id, model, texts, input_role="document")


def _embedding_request_template(
    provider_id: str, model: str, paragraph_ids: list[str]
) -> tuple[str, dict[str, object]]:
    placeholders = [f"<paragraph text omitted: {paragraph_id}>" for paragraph_id in paragraph_ids]
    return _embedding_preview_request_body(provider_id, model, placeholders)


def _validate_canonical_paragraph_rows(
    project: Project, revision_id: str, rows: list[sqlite3.Row]
) -> None:
    """Reject corrupted derived paragraph text before preview or provider egress."""
    canonical_text = project.revision_text(revision_id)
    for row in rows:
        start = row["start_offset"]
        end = row["end_offset"]
        paragraph_text = row["text"]
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or end > len(canonical_text)
            or not isinstance(paragraph_text, str)
            or canonical_text[start:end] != paragraph_text
        ):
            raise ScopeViolationError(
                "persisted paragraph does not match canonical revision text; rebuild paragraph segmentation"
            )


def preview_paragraph_embedding_request(
    project: Project,
    preset_name: str,
    revision_id: str,
    *,
    offset: int = 0,
    limit: int = 20,
    include_text: bool = False,
) -> ParagraphEmbeddingRequestPreview:
    """Describe the exact provider wire shape for one local paragraph window.

    The function reads only project metadata and canonical paragraph text. It validates
    the configured secret *reference* but never resolves its value or sends a request.
    The exact request body is returned only when ``include_text`` is explicitly true.
    """
    if isinstance(offset, bool) or offset < 0:
        raise ScopeViolationError("embedding preview offset must be a non-negative integer")
    if isinstance(limit, bool) or limit < 1 or limit > 1000:
        raise ScopeViolationError("embedding preview limit must be between 1 and 1000")
    if type(include_text) is not bool:
        raise ScopeViolationError("embedding preview include_text must be boolean")

    preset = project.model_preset(preset_name)
    provider_id = str(preset["provider_id"])
    model = str(preset["model"])
    require_embedding_model_capability(project, provider_id, model)
    credential_alias = str(preset["credential_alias"])
    credential = project.credential_reference(provider_id, credential_alias)
    configured_endpoint = credential["endpoint"]
    endpoint = _embedding_preview_endpoint(provider_id, configured_endpoint)
    # Validate the selector only. The returned reference is intentionally discarded.
    provider_auth_reference(provider_id, str(credential["secret_reference"]), endpoint)
    input_type = _document_embedding_input_type(provider_id, model)
    rows = project.db.execute(
        "SELECT id,start_offset,end_offset,text FROM paragraphs "
        "WHERE revision_id=? ORDER BY start_offset,id LIMIT ? OFFSET ?",
        (revision_id, limit, offset),
    ).fetchall()
    total = int(
        project.db.execute(
            "SELECT COUNT(*) FROM paragraphs WHERE revision_id=?", (revision_id,)
        ).fetchone()[0]
    )
    if total == 0:
        raise ScopeViolationError("revision has no paragraphs to preview")
    if offset >= total:
        raise ScopeViolationError("embedding preview offset is outside the paragraph range")
    _validate_canonical_paragraph_rows(project, revision_id, rows)

    prepared: list[tuple[str, str, str]] = []
    preview_inputs: list[EmbeddingPreviewInput] = []
    for row in rows:
        text = str(row["text"])
        if provider_id == "gemini-embeddings":
            request_item = GeminiEmbeddingProvider.document_request_item(model, text)
            content = request_item["content"]
            if not isinstance(content, dict):
                raise ScopeViolationError("Gemini embedding request content is invalid")
            parts = content.get("parts")
            if not isinstance(parts, list) or not parts or not isinstance(parts[0], dict):
                raise ScopeViolationError("Gemini embedding request text is invalid")
            prepared_text = str(parts[0]["text"])
        else:
            prepared_text = text
        paragraph_id = str(row["id"])
        prepared.append((paragraph_id, text, prepared_text))
        preview_inputs.append(
            EmbeddingPreviewInput(
                paragraph_id=paragraph_id,
                start_offset=int(row["start_offset"]),
                end_offset=int(row["end_offset"]),
                character_count=len(text),
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                prepared_character_count=len(prepared_text),
                prepared_text_sha256=hashlib.sha256(prepared_text.encode("utf-8")).hexdigest(),
                prepared_text=prepared_text if include_text else None,
            )
        )

    batch_size = _embedding_batch_size(provider_id)
    batches: list[EmbeddingPreviewBatch] = []
    for batch_index, start in enumerate(range(0, len(prepared), batch_size)):
        current = prepared[start : start + batch_size]
        paragraph_ids = [paragraph_id for paragraph_id, _, _ in current]
        texts = [source_text for _, source_text, _ in current]
        path, template = _embedding_request_template(provider_id, model, paragraph_ids)
        body: dict[str, object] | None = None
        if include_text:
            actual_path, body = _embedding_preview_request_body(provider_id, model, texts)
            if path != actual_path:
                raise ScopeViolationError("embedding preview request path is inconsistent")
        batches.append(
            EmbeddingPreviewBatch(
                index=batch_index,
                method="POST",
                path=path,
                inputs=tuple(preview_inputs[start : start + batch_size]),
                request_body_template=template,
                request_body=body,
            )
        )

    blocked = project.offline_mode and not is_loopback_endpoint(endpoint)
    return ParagraphEmbeddingRequestPreview(
        provider_id=provider_id,
        credential_alias=credential_alias,
        model=model,
        endpoint=endpoint,
        revision_id=revision_id,
        input_type=input_type,
        total_paragraphs=total,
        offset=offset,
        selected_count=len(rows),
        include_text=include_text,
        offline_mode=project.offline_mode,
        blocked_by_offline_mode=blocked,
        secret_values_resolved=False,
        network_request_made=False,
        batches=tuple(batches),
    )


def build_vector_index(
    project: Project,
    revision_id: str,
    chunks: list[VectorChunk],
    *,
    embedding_provider_id: str | None = None,
    embedding_model: str | None = None,
    chunker_version: str | None = None,
    embedding_input_type: str | None = None,
    embedding_endpoint_fingerprint: str | None = None,
) -> str:
    """Persist rebuildable vectors and optional complete embedding provenance."""
    if not chunks:
        raise ScopeViolationError("vector index requires at least one chunk")
    provenance = (embedding_provider_id, embedding_model, chunker_version, embedding_input_type)
    provenance_supplied = any(value is not None for value in provenance)
    if provenance_supplied and any(
        not isinstance(value, str) or not value.strip() for value in provenance
    ):
        raise ScopeViolationError(
            "vector provenance must include non-empty provider, model, chunker and input type"
        )
    if embedding_endpoint_fingerprint is not None and (
        not provenance_supplied
        or not _is_embedding_endpoint_fingerprint(embedding_endpoint_fingerprint)
    ):
        raise ScopeViolationError(
            "vector endpoint fingerprint requires complete provenance and a valid fingerprint"
        )
    text = project.revision_text(revision_id)
    dimension = len(chunks[0].embedding)
    if dimension == 0 or any(len(chunk.embedding) != dimension for chunk in chunks):
        raise ScopeViolationError("all vector chunks require a non-empty common dimension")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks) or any(
        not chunk.chunk_id for chunk in chunks
    ):
        raise ScopeViolationError("vector chunk IDs must be non-empty and unique")
    for chunk in chunks:
        _validate_vector_values(chunk.embedding, "vector embedding")
        if not _has_nonzero_vector_component(chunk.embedding):
            raise ScopeViolationError(
                "cosine vector index embeddings must have non-zero norm"
            )
    if any(
        type(chunk.start) is not int
        or type(chunk.end) is not int
        or chunk.start < 0
        or chunk.end <= chunk.start
        or chunk.end > len(text)
        for chunk in chunks
    ):
        raise ScopeViolationError("vector chunk offsets must be integers within canonical text")
    artifact_id = project.uid()
    relative = _vector_index_relative_path(artifact_id)
    artifact_path = project.hidden / relative
    staging_path = artifact_path.with_name(f".{artifact_id}.building.sqlite")
    expected_hidden = project.root / ".discourselens"
    expected_parent = expected_hidden / "artifacts" / "vectors"
    try:
        unsafe_path = (
            project.hidden.resolve(strict=True) != expected_hidden
            or artifact_path.parent.resolve(strict=True) != expected_parent
        )
    except (OSError, RuntimeError) as exc:
        raise ArtifactIntegrityError("vector index path is invalid or unavailable") from exc
    if (
        unsafe_path
        or artifact_path.exists()
        or artifact_path.is_symlink()
        or staging_path.exists()
        or staging_path.is_symlink()
    ):
        raise ArtifactIntegrityError("vector index path is invalid or unavailable")

    staging_owned = False
    published_identity: tuple[int, int] | None = None
    registry_transaction_started = False
    try:
        # Build privately under a staging name. The final immutable artifact path
        # is not visible until the completed SQLite file has passed validation.
        # sqlite3.connect otherwise creates files with mode 0666 filtered only by
        # umask; reserve the staging file privately before SQLite opens it.
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(staging_path, flags, 0o600)
        except OSError as exc:
            raise ArtifactIntegrityError("vector index path is invalid or unavailable") from exc
        os.close(descriptor)
        staging_owned = True
        connection = _connection(staging_path)
        try:
            connection.execute(
                "CREATE TABLE vector_index_identity(project_id TEXT NOT NULL,"
                "artifact_id TEXT NOT NULL,revision_id TEXT NOT NULL,"
                "embedding_endpoint_fingerprint TEXT)"
            )
            connection.execute(
                "INSERT INTO vector_index_identity VALUES(?,?,?,?)",
                (
                    project.project_id,
                    artifact_id,
                    revision_id,
                    embedding_endpoint_fingerprint,
                ),
            )
            connection.execute(
                "CREATE TABLE chunks(id TEXT PRIMARY KEY,start_offset INTEGER,end_offset INTEGER)"
            )
            connection.execute(
                f"CREATE VIRTUAL TABLE vectors USING vec0(id TEXT PRIMARY KEY,embedding float[{dimension}] distance_metric=cosine)"
            )
            for chunk in chunks:
                connection.execute(
                    "INSERT INTO chunks VALUES(?,?,?)", (chunk.chunk_id, chunk.start, chunk.end)
                )
                connection.execute(
                    "INSERT INTO vectors VALUES(?,?)", (chunk.chunk_id, json.dumps(chunk.embedding))
                )
            connection.commit()
        finally:
            connection.close()
        _validate_built_vector_index(
            staging_path,
            project_id=project.project_id,
            artifact_id=artifact_id,
            revision_id=revision_id,
            chunks=chunks,
            embedding_endpoint_fingerprint=embedding_endpoint_fingerprint,
        )
        stage_stat = os.stat(staging_path, follow_symlinks=False)
        if not stat.S_ISREG(stage_stat.st_mode):
            raise ArtifactIntegrityError("staged vector index path is invalid or unavailable")
        stage_identity = (stage_stat.st_dev, stage_stat.st_ino)
        file_hash = sha256_file(staging_path)

        # A hard link in the same directory is an atomic create-if-absent
        # publication: unlike replace/rename, it cannot overwrite an artifact
        # already published under this immutable ID. Record the expected inode
        # before the call so cleanup can also handle an interrupt delivered
        # after the link succeeds but before Python returns from os.link.
        published_identity = stage_identity
        try:
            os.link(staging_path, artifact_path)
        except FileExistsError as exc:
            raise ArtifactIntegrityError("vector index path is invalid or unavailable") from exc
        except OSError as exc:
            raise ArtifactIntegrityError("vector index could not be atomically published") from exc
        final_stat = os.stat(artifact_path, follow_symlinks=False)
        if (final_stat.st_dev, final_stat.st_ino) != published_identity:
            raise ArtifactIntegrityError("published vector index identity changed unexpectedly")
        if sha256_file(artifact_path) != file_hash:
            raise ArtifactIntegrityError("published vector index changed during publication")
        staging_path.unlink()
        staging_owned = False

        registry_transaction_started = True
        with project.db:
            project.db.execute(
                "INSERT INTO artifact_registry(id,revision_id,artifact_type,backend,backend_version,model,"
                "model_version,schema_version,content_hash,relative_path,created_at,file_hash,"
                "embedding_provider_id,embedding_model,embedding_dimension,chunker_version,embedding_input_type) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    artifact_id,
                    revision_id,
                    "vector_index",
                    "sqlite-vec",
                    None,
                    None,
                    None,
                    f"vector-index-v4:{dimension}",
                    "derived",
                    str(relative),
                    project.db.execute("SELECT datetime('now')").fetchone()[0],
                    file_hash,
                    embedding_provider_id,
                    embedding_model,
                    dimension if provenance_supplied else None,
                    chunker_version,
                    embedding_input_type,
                ),
            )
        registry_transaction_started = False
    except BaseException:
        # A failed SQLite COMMIT may leave the connection in a transaction.
        # Roll it back before removing the file so a failed publication cannot
        # leave registry metadata visible for a missing artifact.
        try:
            if registry_transaction_started and project.db.in_transaction:
                project.db.rollback()
        except sqlite3.Error:
            pass
        cleanup_paths: list[Path] = []
        if staging_owned:
            cleanup_paths.extend(
                (staging_path, Path(f"{staging_path}-wal"), Path(f"{staging_path}-shm"))
            )
        if published_identity is not None:
            try:
                current = os.stat(artifact_path, follow_symlinks=False)
            except OSError:
                current = None
            if current is not None and (current.st_dev, current.st_ino) == published_identity:
                cleanup_paths.extend(
                    (artifact_path, Path(f"{artifact_path}-wal"), Path(f"{artifact_path}-shm"))
                )
        for candidate in cleanup_paths:
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return artifact_id


def build_paragraph_embedding_index(
    project: Project,
    preset_name: str,
    revision_id: str,
    embedding_provider: EmbeddingProvider | None = None,
) -> str:
    """Embed persisted paragraph text using exactly one user-selected model preset."""
    preset = project.model_preset(preset_name)
    provider_id = str(preset["provider_id"])
    model = str(preset["model"])
    require_embedding_model_capability(project, provider_id, model)
    gateway = build_embedding_gateway(
        project,
        provider_id,
        str(preset["credential_alias"]),
        embedding_provider=embedding_provider,
    )
    embedding_provider = gateway.embedders[provider_id]
    _require_embedding_provider_matches_preset(embedding_provider, provider_id)
    credential = project.credential_reference(preset["provider_id"], preset["credential_alias"])
    endpoint = _embedding_endpoint(preset["provider_id"], credential["endpoint"])
    endpoint_fingerprint = _require_embedding_provider_endpoint_matches_preset(
        embedding_provider,
        str(preset["provider_id"]),
        str(preset["credential_alias"]),
        credential["secret_reference"],
        endpoint,
    )
    paragraphs = project.db.execute(
        "SELECT id,start_offset,end_offset,text,segmentation_version FROM paragraphs "
        "WHERE revision_id=? ORDER BY start_offset,id",
        (revision_id,),
    ).fetchall()
    if not paragraphs:
        raise ScopeViolationError("revision has no paragraphs to embed")
    _validate_canonical_paragraph_rows(project, revision_id, paragraphs)
    chunker_versions = {row["segmentation_version"] for row in paragraphs}
    if len(chunker_versions) != 1 or None in chunker_versions:
        raise ScopeViolationError(
            "paragraph vector provenance requires one known segmentation version per revision"
        )
    batch_size = _embedding_batch_size(embedding_provider.provider_id)
    vectors: list[tuple[float, ...]] = []
    dimension: int | None = None
    for start in range(0, len(paragraphs), batch_size):
        batch = paragraphs[start : start + batch_size]
        batch_vectors = gateway.embed(
            provider_id,
            model,
            [str(row["text"]) for row in batch],
        )
        if not isinstance(batch_vectors, (list, tuple)) or len(batch_vectors) != len(batch):
            raise ScopeViolationError("embedding provider must return one vector per paragraph")
        for vector in batch_vectors:
            if not isinstance(vector, (list, tuple)):
                raise ScopeViolationError("embedding provider returned a malformed vector")
            checked_vector = tuple(vector)
            _validate_vector_values(checked_vector, "vector embedding")
            if dimension is None:
                dimension = len(checked_vector)
            elif len(checked_vector) != dimension:
                raise ScopeViolationError("all vector chunks require a non-empty common dimension")
            vectors.append(checked_vector)
    input_type = _document_embedding_input_type(embedding_provider.provider_id, model)
    return build_vector_index(
        project,
        revision_id,
        [
            VectorChunk(row["id"], row["start_offset"], row["end_offset"], vector)
            for row, vector in zip(paragraphs, vectors, strict=True)
        ],
        embedding_provider_id=embedding_provider.provider_id,
        embedding_model=model,
        chunker_version=str(next(iter(chunker_versions))),
        embedding_input_type=input_type,
        embedding_endpoint_fingerprint=endpoint_fingerprint,
    )


def semantic_search_with_preset(
    project: Project,
    preset_name: str,
    scope_id: str,
    artifact_ids: list[str],
    query_text: str,
    embedding_provider: EmbeddingProvider | None = None,
    limit: int = 10,
) -> RetrievalSearchResult[VectorHit]:
    """Embed one explicit query and query only vector artifacts in the supplied scope."""
    query_text = _validate_preset_query_text(query_text, operation="semantic search")
    _validate_vector_search_limit(limit, operation="semantic search")
    if not artifact_ids:
        raise ScopeViolationError("semantic search requires explicit indexes")
    query, embedding_config = _embed_query_with_preset(
        project,
        preset_name,
        scope_id,
        artifact_ids,
        query_text,
        embedding_provider,
        operation="semantic search",
    )
    hits, candidate_count, metric = _search_vectors_with_candidate_count(
        project, scope_id, artifact_ids, query, limit
    )
    selection_policy = (
        f"selected vector artifacts; {_vector_metric_policy_name(metric)} top {limit} per artifact; "
        f"global distance ascending; return at most {limit}"
    )
    retrieval_run_id = _persist_preset_search_retrieval(
        project,
        scope_id,
        artifact_ids,
        query_text,
        "semantic_search_with_preset",
        "retrieval/semantic-vector-preset-v1",
        selection_policy,
        {
            **embedding_config,
            "limit": limit,
            "score_definition": _distance_score_definition(metric),
        },
        [
            (
                hit.revision_id,
                hit.start,
                hit.end,
                "semantic-vector",
                float(hit.distance),
                hit.artifact_id,
                hit.chunk_id,
                None,
                rank,
            )
            for rank, hit in enumerate(hits, start=1)
        ],
        candidate_count=candidate_count,
        inspected_count=candidate_count,
        candidate_limit_reached=candidate_count >= limit,
    )
    return RetrievalSearchResult(scope_id, retrieval_run_id, hits)


def _embed_query_with_preset(
    project: Project,
    preset_name: str,
    scope_id: str,
    artifact_ids: list[str],
    query_text: str,
    embedding_provider: EmbeddingProvider | None,
    *,
    operation: str,
    require_complete_scope: bool = False,
) -> tuple[tuple[float, ...], dict[str, object]]:
    """Validate index provenance before asking a provider to embed the query."""
    preset = project.model_preset(preset_name)
    provider_id = str(preset["provider_id"])
    model = str(preset["model"])
    require_embedding_model_capability(project, provider_id, model)
    gateway = build_embedding_gateway(
        project,
        provider_id,
        str(preset["credential_alias"]),
        embedding_provider=embedding_provider,
    )
    embedding_provider = gateway.embedders[provider_id]
    _require_embedding_provider_matches_preset(embedding_provider, provider_id)
    credential = project.credential_reference(preset["provider_id"], preset["credential_alias"])
    endpoint = _embedding_endpoint(preset["provider_id"], credential["endpoint"])
    endpoint_fingerprint = _require_embedding_provider_endpoint_matches_preset(
        embedding_provider,
        str(preset["provider_id"]),
        str(preset["credential_alias"]),
        credential["secret_reference"],
        endpoint,
    )
    _validate_embedding_index_compatibility(
        project,
        scope_id,
        artifact_ids,
        embedding_provider.provider_id,
        model,
        endpoint_fingerprint,
        require_complete_scope=require_complete_scope,
    )
    query_vectors = gateway.embed_query(provider_id, model, [query_text])
    if not isinstance(query_vectors, (list, tuple)) or len(query_vectors) != 1:
        raise ScopeViolationError("embedding provider must return exactly one query vector")
    query = query_vectors[0]
    if not isinstance(query, (tuple, list)):
        raise ScopeViolationError("embedding provider returned a malformed query vector")
    _validate_vector_values(query, "vector query")
    return tuple(query), {
        "preset_name": preset_name,
        "provider_id": provider_id,
        "model": model,
        "endpoint_fingerprint": endpoint_fingerprint,
        "embedding_dimension": len(query),
    }


def _persist_preset_search_retrieval(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    query_text: str,
    operation: str,
    retrieval_plan_ref: str,
    selection_policy: str,
    embedding_config: dict[str, object],
    ranked_hits: list[tuple[str, int, int, str, float, str | None, str, int | None, int | None]],
    *,
    candidate_count: int,
    inspected_count: int,
    candidate_limit_reached: bool,
    analysis_run_id: str | None = None,
    skill_run_id: str | None = None,
) -> str:
    """Persist privacy-safe, scope-bound provenance for one preset vector query."""
    configuration: dict[str, object] = {
        **embedding_config,
        "artifact_ids": list(artifact_ids),
        "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
        "query_characters": len(query_text),
        "query_representation": "SHA-256 and character count; raw query omitted",
    }
    return _persist_search_retrieval(
        project,
        scope_id,
        operation,
        retrieval_plan_ref,
        selection_policy,
        configuration,
        ranked_hits,
        candidate_count=candidate_count,
        inspected_count=inspected_count,
        candidate_limit_reached=candidate_limit_reached,
        analysis_run_id=analysis_run_id,
        skill_run_id=skill_run_id,
    )


def _persist_search_retrieval(
    project: Project,
    scope_id: str,
    operation: str,
    retrieval_plan_ref: str,
    selection_policy: str,
    configuration: dict[str, object],
    ranked_hits: list[tuple[str, int, int, str, float, str | None, str, int | None, int | None]],
    *,
    candidate_count: int,
    inspected_count: int,
    candidate_limit_reached: bool,
    analysis_run_id: str | None = None,
    skill_run_id: str | None = None,
) -> str:
    """Persist retrieval lineage for either a standalone or linked search."""
    query_terms: list[str] = []

    scoped_revisions = set(project.scope_revisions(scope_id))
    ranges_by_revision: dict[str, list[tuple[int, int]]] = {}
    for revision_id, start, end, _, _, _, _, _, _ in ranked_hits:
        if (
            revision_id not in scoped_revisions
            or type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or end > len(project.revision_text(revision_id))
        ):
            raise ScopeViolationError("preset search result is outside its explicit scope")
        ranges_by_revision.setdefault(revision_id, []).append((start, end))

    spans_by_revision: dict[str, list[str]] = {}
    for revision_id, ranges in ranges_by_revision.items():
        spans_by_revision[revision_id] = project.create_spans(revision_id, ranges)
    span_ids_by_result: list[str] = []
    result_offsets: dict[str, int] = {revision_id: 0 for revision_id in spans_by_revision}
    for revision_id, _, _, _, _, _, _, _, _ in ranked_hits:
        offset = result_offsets[revision_id]
        span_ids_by_result.append(spans_by_revision[revision_id][offset])
        result_offsets[revision_id] = offset + 1

    if len(set(span_ids_by_result)) != len(span_ids_by_result):
        raise ScopeViolationError(
            "preset search returned duplicate canonical spans; select one vector index per revision"
        )
    returned_count = len(ranked_hits)
    termination_state = (
        CoverageTerminationState.LIMIT_REACHED
        if candidate_limit_reached
        else CoverageTerminationState.COMPLETED
    )
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=scope_id,
            population_definition=(
                f"Candidate units selected by {operation} under its recorded policy from the "
                "original ScopeSnapshot; not an exhaustive corpus population."
            ),
            selection_policy=selection_policy,
            inspection_count=inspected_count,
            termination_reason=(
                "top-k candidate limit reached" if candidate_limit_reached else "query completed"
            ),
            termination_state=termination_state,
            negative_result_status=(
                NegativeResultStatus.NO_MATCHES_UNDER_PLAN
                if not returned_count
                else NegativeResultStatus.NOT_APPLICABLE
            ),
        )
    )
    if (analysis_run_id is None) != (skill_run_id is None):
        raise ScopeViolationError("retrieval linkage requires both analysis and skill run IDs")
    linked = analysis_run_id is not None and skill_run_id is not None
    retrieval_run_id = project.create_retrieval_run(
        analysis_run_id=analysis_run_id if linked else None,
        skill_run_id=skill_run_id if linked else None,
        scope_id=scope_id,
        retrieval_plan_ref=retrieval_plan_ref,
        query_terms=query_terms,
        selection_policy=selection_policy,
        candidate_count=candidate_count,
        inspected_count=inspected_count,
        returned_count=returned_count,
        candidate_limit_reached=candidate_limit_reached,
        coverage_id=coverage_id,
        standalone_operation=None if linked else operation,
        standalone_config=None if linked else configuration,
        linked_context={"operation": operation, **configuration} if linked else None,
    )
    ranked_rows = [
        (span_id, method, score, (artifact_id, unit_id, lexical_rank, vector_rank))
        for span_id, (
            _,
            _,
            _,
            method,
            score,
            artifact_id,
            unit_id,
            lexical_rank,
            vector_rank,
        ) in zip(span_ids_by_result, ranked_hits, strict=True)
    ]
    for offset in range(0, len(ranked_rows), 1_000):
        batch = ranked_rows[offset : offset + 1_000]
        project.add_ranked_retrieval_hits(
            retrieval_run_id,
            [(span_id, method, score) for span_id, method, score, _ in batch],
            first_rank=offset + 1,
            hit_contexts=[context for _, _, _, context in batch],
        )
    return retrieval_run_id


def _validate_embedding_index_compatibility(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    provider_id: str,
    model: str,
    embedding_endpoint_fingerprint: str,
    *,
    require_complete_scope: bool = False,
) -> None:
    """Reject unknown/legacy or deployment-incompatible indexes before query egress.

    Legacy v1-v3 artifacts retain raw vector-search, listing, and rebuild support.
    Preset-backed semantic queries require v4 because older indexes do not bind the
    embedding endpoint identity and cosine-distance schema.
    """
    artifact_ids = _validate_vector_artifact_ids(artifact_ids)
    revisions = _validated_scope_revisions(project, scope_id)
    artifacts = _artifact_rows_for_ids(
        project,
        artifact_ids,
        "id,revision_id,relative_path,file_hash,schema_version,embedding_provider_id,embedding_model,"
        "embedding_dimension,chunker_version,embedding_input_type",
        vector_indexes_only=True,
    )
    if len(artifacts) != len(artifact_ids) or any(
        not isinstance(row["id"], str)
        or not isinstance(row["revision_id"], str)
        or row["revision_id"] not in revisions
        for row in artifacts
    ):
        raise ScopeViolationError("vector artifact is unknown or outside the explicit scope")
    if require_complete_scope:
        indexed_revisions = [str(row["revision_id"]) for row in artifacts]
        scoped_revisions = set(revisions)
        if (
            len(indexed_revisions) != len(scoped_revisions)
            or len(set(indexed_revisions)) != len(indexed_revisions)
            or set(indexed_revisions) != scoped_revisions
        ):
            raise ScopeViolationError(
                "hybrid ask requires exactly one vector index for every revision in its explicit scope"
            )
    dimensions = {
        row["id"]: _vector_index_dimension(row["schema_version"]) for row in artifacts
    }
    _require_uniform_vector_metric(artifacts, operation="semantic search")
    if any(
        not str(row["schema_version"]).startswith("vector-index-v4:")
        for row in artifacts
    ):
        raise ScopeViolationError(
            "semantic query requires endpoint-identified vector indexes; rebuild the index"
        )
    if any(
        row["embedding_provider_id"] != provider_id
        or row["embedding_model"] != model
        or row["embedding_dimension"] is None
        or row["embedding_dimension"] != dimensions[row["id"]]
        or not row["chunker_version"]
        or row["embedding_input_type"] != _document_embedding_input_type(provider_id, model)
        or not str(row["schema_version"]).startswith("vector-index-v4:")
        for row in artifacts
    ):
        raise ScopeViolationError(
            "semantic query provider/model/endpoint does not match vector index provenance; rebuild the index"
        )
    _require_uniform_vector_provenance(
        project,
        artifacts,
        operation="semantic search",
        expected_embedding_endpoint_fingerprint=embedding_endpoint_fingerprint,
    )
    for artifact in artifacts:
        _validate_published_vector_metric(project, artifact)
    revision_ids = [str(row["revision_id"]) for row in artifacts]
    if len(set(revision_ids)) != len(revision_ids):
        raise ScopeViolationError(
            "preset retrieval accepts at most one vector artifact per scoped revision"
        )


def hybrid_retrieval_configuration_snapshot(
    project: Project,
    preset_name: str,
    scope_id: str,
    artifact_ids: list[str],
    *,
    require_complete_scope: bool = True,
) -> dict[str, object]:
    """Return secret-free, resolved embedding/index provenance without query egress.

    This lets an AnalysisRun freeze the effective provider/model/index identity
    before the question is sent to an embedding endpoint. Compatibility and
    optional full-scope coverage are validated using local metadata and files.
    """
    preset = project.model_preset(preset_name)
    provider_id = str(preset["provider_id"])
    model = str(preset["model"])
    require_embedding_model_capability(project, provider_id, model)
    credential = project.credential_reference(provider_id, str(preset["credential_alias"]))
    endpoint = _embedding_endpoint(provider_id, credential["endpoint"])
    endpoint_fingerprint = _embedding_endpoint_fingerprint(
        provider_id,
        str(preset["credential_alias"]),
        str(credential["secret_reference"]),
        endpoint,
    )
    _validate_embedding_index_compatibility(
        project,
        scope_id,
        artifact_ids,
        provider_id,
        model,
        endpoint_fingerprint,
        require_complete_scope=require_complete_scope,
    )
    artifacts = _artifact_rows_for_ids(
        project,
        artifact_ids,
        "id,revision_id,content_hash,file_hash,schema_version,embedding_provider_id,embedding_model,"
        "embedding_dimension,chunker_version,embedding_input_type",
        vector_indexes_only=True,
    )
    ordered_artifacts = sorted(artifacts, key=lambda item: (str(item["revision_id"]), str(item["id"])))
    return {
        "preset_name": str(preset["name"]),
        "provider_id": provider_id,
        "model": model,
        "endpoint_fingerprint": endpoint_fingerprint,
        "artifacts": [
            {
                "artifact_id": str(artifact["id"]),
                "revision_id": str(artifact["revision_id"]),
                "content_hash": str(artifact["content_hash"]),
                "file_hash": str(artifact["file_hash"]),
                "schema_version": str(artifact["schema_version"]),
                "embedding_provider_id": str(artifact["embedding_provider_id"]),
                "embedding_model": str(artifact["embedding_model"]),
                "embedding_dimension": int(artifact["embedding_dimension"]),
                "chunker_version": str(artifact["chunker_version"]),
                "embedding_input_type": str(artifact["embedding_input_type"]),
            }
            for artifact in ordered_artifacts
        ],
    }


def search_vectors(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    query: tuple[float, ...],
    limit: int = 10,
) -> list[VectorHit]:
    """Search per-revision indexes inside the explicit scope, then globally rank."""
    return _search_vectors_with_candidate_count(project, scope_id, artifact_ids, query, limit)[0]


def search_vectors_with_provenance(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    query: tuple[float, ...],
    limit: int = 10,
) -> RetrievalSearchResult[VectorHit]:
    """Search caller-supplied vectors and persist scope-bound hit lineage.

    The query vector is represented in run configuration only by a SHA-256 of its
    effective float32 components and its dimension. The vector itself is never
    stored. ``search_vectors`` remains the list-returning compatibility API.
    """
    hits, candidate_count, metric = _search_vectors_with_candidate_count(
        project, scope_id, artifact_ids, query, limit
    )
    selection_policy = (
        f"caller-supplied raw vector; {_vector_metric_policy_name(metric)} top "
        f"{limit} per artifact; global distance ascending; return at most {limit}"
    )
    retrieval_run_id = _persist_search_retrieval(
        project,
        scope_id,
        "search_vectors_with_provenance",
        "retrieval/raw-vector-search-v1",
        selection_policy,
        {
            "artifact_ids": list(artifact_ids),
            "vector_provenance": "caller-supplied-raw-vector",
            "query_sha256": _query_vector_sha256(query),
            "query_dimension": len(query),
            "query_representation": (
                "caller-supplied raw vector; canonical float32 SHA-256 and dimension; "
                "vector omitted"
            ),
            "limit": limit,
            "score_definition": _distance_score_definition(metric),
        },
        [
            (
                hit.revision_id,
                hit.start,
                hit.end,
                "raw-vector",
                float(hit.distance),
                hit.artifact_id,
                hit.chunk_id,
                None,
                rank,
            )
            for rank, hit in enumerate(hits, start=1)
        ],
        candidate_count=candidate_count,
        inspected_count=candidate_count,
        candidate_limit_reached=candidate_count >= limit,
    )
    return RetrievalSearchResult(scope_id, retrieval_run_id, hits)


def _search_vectors_with_candidate_count(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    query: tuple[float, ...],
    limit: int = 10,
    *,
    validated_artifacts: list[sqlite3.Row] | None = None,
) -> tuple[list[VectorHit], int, str]:
    """Return globally ranked results and the candidate count inspected per index."""
    _validate_vector_search_limit(limit, operation="vector search")
    if not query:
        raise ScopeViolationError(
            "vector search requires artifacts and a query vector"
        )
    artifact_ids = _validate_vector_artifact_ids(artifact_ids)
    if not isinstance(query, (tuple, list)):
        raise ScopeViolationError("vector query must be a numeric vector")
    _validate_vector_values(query, "vector query")
    artifacts = validated_artifacts or _scoped_vector_artifacts(
        project, scope_id, artifact_ids, operation="vector search"
    )
    metric = _require_uniform_vector_metric(artifacts, operation="vector search")
    if metric == "cosine" and not _has_nonzero_vector_component(query):
        raise ScopeViolationError("cosine vector search requires a non-zero query vector")
    for artifact in artifacts:
        if len(query) != _vector_index_dimension(artifact["schema_version"]):
            raise ScopeViolationError("query vector dimension does not match vector artifact")
    # Merge each revision-local top-k into a bounded global top-k. Retaining
    # every local result until the end costs O(limit * revision_count) memory
    # for large corpus scopes even though the public result is capped at k.
    top_hits: list[tuple[float, int, VectorHit]] = []
    candidate_count = 0
    sequence = 0
    for artifact in artifacts:
        artifact_path = verify_project_artifact(
            project.root, artifact["relative_path"], artifact["file_hash"]
        )
        connection = _connection(artifact_path)
        try:
            _validate_vector_metric_ddl(
                connection,
                _vector_index_metric(artifact["schema_version"]),
                error_message="vector artifact metric does not match its schema version",
            )
            rows = connection.execute(
                "SELECT v.id,v.distance,c.start_offset,c.end_offset FROM vectors v JOIN chunks c ON c.id=v.id WHERE v.embedding MATCH ? AND k=?",
                (json.dumps(query), limit),
            ).fetchall()
        finally:
            connection.close()
        text_length = len(project.revision_text(artifact["revision_id"]))
        candidate_count += len(rows)
        for row in rows:
            start = row["start_offset"]
            end = row["end_offset"]
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end <= start
                or end > text_length
            ):
                raise ScopeViolationError(
                    "vector index returned malformed offsets; rebuild the index"
                )
            distance = row["distance"]
            try:
                valid_distance = (
                    not isinstance(distance, bool)
                    and isinstance(distance, (int, float))
                    and math.isfinite(float(distance))
                )
            except (OverflowError, ValueError):
                valid_distance = False
            if not valid_distance:
                raise ScopeViolationError(
                    "vector index returned a malformed distance; rebuild the index"
                )
            if not isinstance(row["id"], str) or not row["id"]:
                raise ScopeViolationError(
                    "vector index returned a malformed candidate ID; rebuild the index"
                )
            hit = VectorHit(
                artifact["id"],
                artifact["revision_id"],
                row["id"],
                row["start_offset"],
                row["end_offset"],
                float(distance),
            )
            # Reverse the first two sort fields so the max-distance, latest
            # equal-distance candidate remains at heap[0] for replacement.
            # The unique sequence value prevents comparisons reaching VectorHit.
            entry = (-hit.distance, -sequence, hit)
            sequence += 1
            if len(top_hits) < limit:
                heapq.heappush(top_hits, entry)
            elif entry[:2] > top_hits[0][:2]:
                heapq.heapreplace(top_hits, entry)

    ranked = [
        entry[2]
        for entry in sorted(top_hits, key=lambda item: (-item[0], -item[1]))
    ]
    return ranked, candidate_count, metric


def hybrid_search(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    lexical_query: str,
    vector_query: tuple[float, ...],
    *,
    limit: int = 10,
    rrf_constant: int = 60,
) -> list[HybridHit]:
    """Fuse strict lexical and vector candidate lists using reciprocal-rank fusion.

    This preserves both source ranks rather than presenting a fused result as an
    exhaustive lexical result or a semantic fact. Vector artifacts are validated
    against the immutable scope before any ranking. This is a low-level API: a raw
    caller-supplied vector carries no inferable provider/model provenance, so callers
    must ensure it belongs to the selected indexes' vector space. Use
    :func:`hybrid_search_with_preset` when the query should be embedded and checked
    against explicit preset and artifact provenance here.
    """
    return _hybrid_search_with_candidate_count(
        project,
        scope_id,
        artifact_ids,
        lexical_query,
        vector_query,
        limit=limit,
        rrf_constant=rrf_constant,
    ).hits


def hybrid_search_with_provenance(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    lexical_query: str,
    vector_query: tuple[float, ...],
    *,
    limit: int = 10,
    rrf_constant: int = 60,
) -> RetrievalSearchResult[HybridHit]:
    """Fuse lexical and caller-supplied vector candidates with durable lineage.

    The lexical query is stored only as a digest and character count. The caller's
    vector is stored only as a digest of its effective float32 components and its
    dimension. ``hybrid_search`` remains the list-returning compatibility API.
    """
    outcome = _hybrid_search_with_candidate_count(
        project,
        scope_id,
        artifact_ids,
        lexical_query,
        vector_query,
        limit=limit,
        rrf_constant=rrf_constant,
    )
    hits = outcome.hits
    selection_policy = (
        f"top {limit} exact lexical matches combined with caller-supplied raw vector "
        f"{_vector_metric_policy_name(outcome.vector_metric)} top {limit} per artifact using reciprocal-rank fusion "
        f"(constant {rrf_constant}); only lexical row retention is capped: the complete "
        f"exact lexical stream is inspected to count matches and rank vector candidates; "
        f"return at most {limit}; candidate count includes lexical/vector channel entries"
    )
    retrieval_run_id = _persist_search_retrieval(
        project,
        scope_id,
        "hybrid_search_with_provenance",
        "retrieval/raw-hybrid-rrf-v1",
        selection_policy,
        {
            "artifact_ids": list(artifact_ids),
            "vector_provenance": "caller-supplied-raw-vector",
            "query_sha256": hashlib.sha256(lexical_query.encode("utf-8")).hexdigest(),
            "query_characters": len(lexical_query),
            "query_representation": "SHA-256 and character count; raw lexical query omitted",
            "query_vector_sha256": _query_vector_sha256(vector_query),
            "query_vector_dimension": len(vector_query),
            "query_vector_representation": (
                "caller-supplied raw vector; canonical float32 SHA-256 and dimension; "
                "vector omitted"
            ),
            "limit": limit,
            "rrf_constant": rrf_constant,
            "lexical_candidate_limit": outcome.lexical_candidate_limit,
            "lexical_candidate_count": outcome.lexical_candidate_count,
            "lexical_candidates_retained": min(
                outcome.lexical_candidate_count, outcome.lexical_candidate_limit
            ),
            "lexical_candidate_limit_reached": outcome.lexical_candidate_limit_reached,
            "lexical_candidate_population_complete": True,
            "lexical_stream_fully_inspected": True,
            "lexical_retention_policy": (
                "retain at most the lexical top-k rows plus true lexical ranks for vector hits; "
                "inspect the complete exact lexical stream to count matches and assign ranks"
            ),
            "candidate_count_definition": (
                "exact lexical match count plus vector hits returned by each per-artifact "
                "top-k query; cross-channel duplicates count once per channel"
            ),
            "score_definition": (
                "sum of reciprocal ranks; 1/(rrf_constant + lexical/vector rank)"
            ),
        },
        _hybrid_hit_retrieval_rows(project, hits),
        candidate_count=outcome.candidate_count,
        inspected_count=outcome.candidate_count,
        candidate_limit_reached=(
            outcome.vector_candidate_count >= limit
            or outcome.lexical_candidate_limit_reached
            or outcome.fusion_limit_reached
        ),
    )
    return RetrievalSearchResult(scope_id, retrieval_run_id, hits)


def _hybrid_search_with_candidate_count(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    lexical_query: str,
    vector_query: tuple[float, ...],
    *,
    limit: int = 10,
    rrf_constant: int = 60,
) -> _HybridSearchOutcome:
    """Fuse bounded channel candidates while preserving exact RRF ranks.

    The complete lexical stream is inspected to count exact matches and give
    vector hits their true lexical ranks, but only its top ``limit`` rows and
    ranks for vector spans are retained. Candidate identity is the canonical
    ``(revision_id,start,end)`` span, never a caller-defined unit ID. A result
    ranked below ``limit`` in both channels cannot enter the final top-k: at least
    ``limit`` candidates precede it in each channel. Thus the union of the lexical
    top-k and vector top-k is sufficient for exact final top-k RRF. Retained rows
    are bounded; lexical scan work still scales with the exact-match population.
    """
    _validate_vector_search_limit(limit, operation="hybrid search")
    if (
        not isinstance(lexical_query, str)
        or not lexical_query.strip()
        or isinstance(rrf_constant, bool)
        or not isinstance(rrf_constant, int)
        or rrf_constant < 1
    ):
        raise ScopeViolationError(
            "hybrid search requires query text and a positive RRF constant"
        )
    artifact_ids = _validate_vector_artifact_ids(artifact_ids)
    artifacts = _scoped_vector_artifacts(
        project, scope_id, artifact_ids, operation="hybrid search"
    )
    vector_hits, vector_candidate_count, vector_metric = _search_vectors_with_candidate_count(
        project,
        scope_id,
        artifact_ids,
        vector_query,
        limit=limit,
        validated_artifacts=artifacts,
    )
    vector_spans = {(hit.revision_id, hit.start, hit.end) for hit in vector_hits}
    lexical_rows, vector_lexical_ranks, lexical_candidate_count = (
        project._search_exact_ranked_candidates(
            lexical_query,
            scope_id,
            limit=limit,
            additional_spans=vector_spans,
        )
    )
    candidates: dict[tuple[str, int, int], list[int | None]] = {}
    unit_ids_by_candidate: dict[tuple[str, int, int], str] = {}
    vector_artifacts_by_candidate: dict[tuple[str, int, int], str] = {}
    for rank, row in enumerate(lexical_rows, start=1):
        span = (row["revision_id"], int(row["start_offset"]), int(row["end_offset"]))
        candidates.setdefault(span, [None, None])[0] = rank
        unit_ids_by_candidate[span] = str(row["paragraph_id"])
    for rank, hit in enumerate(vector_hits, start=1):
        span = (hit.revision_id, hit.start, hit.end)
        ranks = candidates.setdefault(span, [vector_lexical_ranks.get(span), None])
        if ranks[1] is None or rank < ranks[1]:
            ranks[1] = rank
            unit_ids_by_candidate[span] = hit.chunk_id
            vector_artifacts_by_candidate[span] = hit.artifact_id
    results = [
        HybridHit(
            revision_id=span[0],
            unit_id=unit_ids_by_candidate[span],
            lexical_rank=ranks[0],
            vector_rank=ranks[1],
            fusion_score=sum(1 / (rrf_constant + rank) for rank in ranks if rank is not None),
            artifact_id=(
                vector_artifacts_by_candidate[span]
                if ranks[1] is not None
                else None
            ),
        )
        for span, ranks in candidates.items()
    ]
    ranked = sorted(results, key=lambda hit: (-hit.fusion_score, hit.revision_id, hit.unit_id))[
        :limit
    ]
    candidate_count = lexical_candidate_count + vector_candidate_count
    fusion_limit_reached = len(results) > len(ranked)
    return _HybridSearchOutcome(
        hits=ranked,
        vector_metric=vector_metric,
        candidate_count=candidate_count,
        vector_candidate_count=vector_candidate_count,
        lexical_candidate_count=lexical_candidate_count,
        lexical_candidate_limit=limit,
        lexical_candidate_limit_reached=lexical_candidate_count > limit,
        fusion_limit_reached=fusion_limit_reached,
    )


def _hybrid_hit_retrieval_rows(
    project: Project, hits: list[HybridHit]
) -> list[tuple[str, int, int, str, float, str | None, str, int | None, int | None]]:
    """Resolve vector candidates through their indexes and lexical hits as paragraphs."""
    vector_hits_by_artifact: dict[str, list[HybridHit]] = {}
    paragraph_hit_ids: list[str] = []
    for hit in hits:
        if hit.vector_rank is not None:
            if not isinstance(hit.artifact_id, str) or not hit.artifact_id:
                raise ScopeViolationError("hybrid vector candidate has no selected artifact")
            vector_hits_by_artifact.setdefault(hit.artifact_id, []).append(hit)
        else:
            paragraph_hit_ids.append(hit.unit_id)

    vector_ranges: dict[tuple[str, str], tuple[str, int, int]] = {}
    artifact_ids = list(vector_hits_by_artifact)
    for artifact_batch_start in range(0, len(artifact_ids), 900):
        artifact_batch = artifact_ids[artifact_batch_start : artifact_batch_start + 900]
        placeholders = ",".join("?" for _ in artifact_batch)
        artifact_rows = project.db.execute(
            "SELECT id,revision_id,relative_path,schema_version,file_hash "
            "FROM artifact_registry WHERE artifact_type='vector_index' "
            f"AND id IN ({placeholders})",
            artifact_batch,
        ).fetchall()
        artifacts_by_id = {str(row["id"]): row for row in artifact_rows}
        if len(artifacts_by_id) != len(artifact_batch):
            raise ScopeViolationError("hybrid vector candidate artifact is unavailable")
        for artifact_id in artifact_batch:
            artifact = artifacts_by_id[artifact_id]
            path, _ = _verify_vector_index_identity(project, artifact)
            connection = _connection(path)
            try:
                artifact_hits = vector_hits_by_artifact[artifact_id]
                for hit_batch_start in range(0, len(artifact_hits), 900):
                    hit_batch = artifact_hits[hit_batch_start : hit_batch_start + 900]
                    unit_ids = [hit.unit_id for hit in hit_batch]
                    chunk_placeholders = ",".join("?" for _ in unit_ids)
                    chunk_rows = connection.execute(
                        "SELECT id,start_offset,end_offset FROM chunks "
                        f"WHERE id IN ({chunk_placeholders})",
                        unit_ids,
                    ).fetchall()
                    if len(chunk_rows) != len(unit_ids):
                        raise ScopeViolationError(
                            "hybrid vector candidate is missing from its selected index"
                        )
                    for row in chunk_rows:
                        if (
                            not isinstance(row["id"], str)
                            or type(row["start_offset"]) is not int
                            or type(row["end_offset"]) is not int
                        ):
                            raise ScopeViolationError(
                                "hybrid vector candidate has malformed span metadata"
                            )
                        vector_ranges[(artifact_id, row["id"])] = (
                            str(artifact["revision_id"]),
                            row["start_offset"],
                            row["end_offset"],
                        )
            finally:
                connection.close()

    paragraphs_by_id: dict[str, tuple[str, int, int]] = {}
    for batch_start in range(0, len(paragraph_hit_ids), 900):
        paragraph_ids = paragraph_hit_ids[batch_start : batch_start + 900]
        placeholders = ",".join("?" for _ in paragraph_ids)
        rows = project.db.execute(
            "SELECT id,revision_id,start_offset,end_offset FROM paragraphs "
            f"WHERE id IN ({placeholders})",
            paragraph_ids,
        ).fetchall()
        paragraphs_by_id.update(
            {
                str(row["id"]): (
                    str(row["revision_id"]),
                    int(row["start_offset"]),
                    int(row["end_offset"]),
                )
                for row in rows
            }
        )
    ranked_hits: list[
        tuple[str, int, int, str, float, str | None, str, int | None, int | None]
    ] = []
    for hit in hits:
        span = (
            vector_ranges.get((hit.artifact_id, hit.unit_id))
            if hit.vector_rank is not None and hit.artifact_id is not None
            else paragraphs_by_id.get(hit.unit_id)
        )
        if (
            span is None
            or span[0] != hit.revision_id
            or ((hit.artifact_id is None) != (hit.vector_rank is None))
        ):
            raise ScopeViolationError("hybrid search returned a non-canonical paragraph candidate")
        ranked_hits.append(
            (
                hit.revision_id,
                span[1],
                span[2],
                "hybrid-rrf",
                float(hit.fusion_score),
                hit.artifact_id,
                hit.unit_id,
                hit.lexical_rank,
                hit.vector_rank,
            )
        )
    return ranked_hits


def hybrid_search_with_preset(
    project: Project,
    preset_name: str,
    scope_id: str,
    artifact_ids: list[str],
    query_text: str,
    embedding_provider: EmbeddingProvider | None = None,
    *,
    limit: int = 10,
    rrf_constant: int = 60,
    analysis_run_id: str | None = None,
    skill_run_id: str | None = None,
    require_complete_scope: bool = False,
) -> RetrievalSearchResult[HybridHit]:
    """Embed one query with a preset, then fuse scoped lexical and vector results.

    Every selected index is checked for scope, integrity, and matching provider,
    model, dimension, chunker, and input-type provenance before the provider is
    asked to embed the query. The same text is used for lexical and vector search.
    """
    query_text = _validate_preset_query_text(query_text, operation="hybrid search")
    _validate_vector_search_limit(limit, operation="hybrid search")
    if (
        isinstance(rrf_constant, bool)
        or not isinstance(rrf_constant, int)
        or rrf_constant < 1
    ):
        raise ScopeViolationError(
            "hybrid search requires query text and a positive RRF constant"
        )
    if (analysis_run_id is None) != (skill_run_id is None):
        raise ScopeViolationError("retrieval linkage requires both analysis and skill run IDs")
    if analysis_run_id is not None and skill_run_id is not None:
        analysis = project.db.execute(
            "SELECT scope_id FROM analysis_runs WHERE id=?", (analysis_run_id,)
        ).fetchone()
        skill = project.db.execute(
            "SELECT analysis_run_id,scope_id,status FROM skill_runs WHERE id=?", (skill_run_id,)
        ).fetchone()
        if (
            analysis is None
            or skill is None
            or analysis["scope_id"] != scope_id
            or skill["analysis_run_id"] != analysis_run_id
            or skill["scope_id"] != scope_id
            or skill["status"] != "RUNNING"
        ):
            raise ScopeViolationError(
                "linked hybrid retrieval must retain a running skill and its original ScopeSnapshot"
            )
    query, embedding_config = _embed_query_with_preset(
        project,
        preset_name,
        scope_id,
        artifact_ids,
        query_text,
        embedding_provider,
        operation="hybrid search",
        require_complete_scope=require_complete_scope,
    )
    outcome = _hybrid_search_with_candidate_count(
        project,
        scope_id,
        artifact_ids,
        query_text,
        query,
        limit=limit,
        rrf_constant=rrf_constant,
    )
    hits = outcome.hits
    selection_policy = (
        f"top {limit} exact lexical matches combined with "
        f"{_vector_metric_policy_name(outcome.vector_metric)} top "
        f"{limit} per artifact using reciprocal-rank fusion (constant {rrf_constant}); "
        f"only lexical row retention is capped: the complete exact lexical stream is "
        f"inspected to count matches and rank vector candidates; return at most {limit}; "
        f"candidate count includes lexical/vector channel entries"
    )
    ranked_hits = _hybrid_hit_retrieval_rows(project, hits)
    retrieval_run_id = _persist_preset_search_retrieval(
        project,
        scope_id,
        artifact_ids,
        query_text,
        "hybrid_search_with_preset",
        "retrieval/hybrid-rrf-preset-v1",
        selection_policy,
        {
            **embedding_config,
            "limit": limit,
            "rrf_constant": rrf_constant,
            "lexical_candidate_limit": outcome.lexical_candidate_limit,
            "lexical_candidate_count": outcome.lexical_candidate_count,
            "lexical_candidates_retained": min(
                outcome.lexical_candidate_count, outcome.lexical_candidate_limit
            ),
            "lexical_candidate_limit_reached": outcome.lexical_candidate_limit_reached,
            "lexical_candidate_population_complete": True,
            "lexical_stream_fully_inspected": True,
            "lexical_retention_policy": (
                "retain at most the lexical top-k rows plus true lexical ranks for vector hits; "
                "inspect the complete exact lexical stream to count matches and assign ranks"
            ),
            "candidate_count_definition": (
                "exact lexical match count plus vector hits returned by each per-artifact "
                "top-k query; cross-channel duplicates count once per channel"
            ),
            "score_definition": "sum of reciprocal ranks; 1/(rrf_constant + lexical/vector rank)",
        },
        ranked_hits,
        candidate_count=outcome.candidate_count,
        inspected_count=outcome.candidate_count,
        candidate_limit_reached=(
            outcome.vector_candidate_count >= limit
            or outcome.lexical_candidate_limit_reached
            or outcome.fusion_limit_reached
        ),
        analysis_run_id=analysis_run_id,
        skill_run_id=skill_run_id,
    )
    enriched_hits = [
        HybridHit(
            revision_id=hit.revision_id,
            unit_id=hit.unit_id,
            lexical_rank=hit.lexical_rank,
            vector_rank=hit.vector_rank,
            fusion_score=hit.fusion_score,
            artifact_id=hit.artifact_id,
        )
        for hit in hits
    ]
    return RetrievalSearchResult(scope_id, retrieval_run_id, enriched_hits)
