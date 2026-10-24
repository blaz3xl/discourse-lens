"""Versioned, rebuildable token artifacts stored outside canonical SQLite state."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from heapq import nsmallest
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import BinaryIO, Literal, TypedDict, cast

from .artifact_integrity import sha256_file, verify_project_artifact
from .corpus import (
    SURFACE_NORMALIZATION_V2,
    SURFACE_TOKENIZER_V2,
    _surface_form_v2,
    _tokenize_v2_chunks,
)
from .errors import (
    ArtifactIntegrityError,
    EvidenceValidationError,
    OptionalDependencyError,
    ScopeViolationError,
)
from .languages import (
    OPERATIONAL_LIMITED,
    OPERATIONAL_SUPPORTED,
    _has_valid_dependency_tree,
)
from .nlp import NlpBackend
from .project import Project
from .types import (
    CoverageDescriptor,
    CoverageMode,
    CoverageTerminationState,
    NegativeResultStatus,
)

SURFACE_TOKEN_SCHEMA_VERSION = f"surface-token-v2-ucd-{unicodedata.unidata_version}"
LEGACY_SURFACE_TOKEN_SCHEMA_VERSION = "surface-token-v1"
LEGACY_PARSED_TOKEN_SCHEMA_VERSION = "parsed-token-v1"
PARSED_TOKEN_SCHEMA_VERSION = "parsed-token-v2"
_PARSED_TOKEN_COVERAGE_METADATA_KEY = "discourselens.parsed_token_coverage"
_PARSED_TOKEN_COVERAGE_FORMAT_VERSION = 1
_SURFACE_TOKEN_BATCH_SIZE = 100_000
_PARSED_TOKEN_BATCH_SIZE = 100_000
_SURFACE_TEXT_CHUNK_CHARS = 1_048_576
_MAX_LEMMA_KWIC_WINDOW = 100
_MAX_LEMMA_KWIC_LIMIT = 1_000


def _prepare_nlp_artifact_directory(project: Project) -> Path:
    """Create/check artifact parents one level at a time without following symlinks."""
    hidden = project.hidden
    try:
        hidden_mode = hidden.lstat().st_mode
        if stat.S_ISLNK(hidden_mode) or not stat.S_ISDIR(hidden_mode):
            raise ArtifactIntegrityError("derived artifact directory is invalid or unavailable")
        if hidden.resolve(strict=True) != hidden:
            raise ArtifactIntegrityError("derived artifact directory is invalid or unavailable")

        artifacts = hidden / "artifacts"
        nlp_artifacts = artifacts / "nlp"
        for directory in (artifacts, nlp_artifacts):
            try:
                directory.mkdir()
            except FileExistsError:
                pass
            mode = directory.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ArtifactIntegrityError("derived artifact directory is invalid or unavailable")
            if directory.resolve(strict=True) != directory:
                raise ArtifactIntegrityError("derived artifact directory is invalid or unavailable")
        return nlp_artifacts
    except ArtifactIntegrityError:
        raise
    except (OSError, RuntimeError) as exc:
        raise ArtifactIntegrityError(
            "derived artifact directory is invalid or unavailable"
        ) from exc


def _open_nlp_artifact_directory(project: Project) -> tuple[int, ...] | None:
    """Pin the project artifact directory through no-follow directory handles."""
    if (
        os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.link not in os.supports_dir_fd
        or os.unlink not in os.supports_dir_fd
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
    ):
        return None

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        root_fd = os.open(project.root, directory_flags)
        descriptors.append(root_fd)
        root_entry = os.stat(project.root, follow_symlinks=False)
        root_opened = os.fstat(root_fd)
        if not stat.S_ISDIR(root_entry.st_mode) or (
            root_entry.st_dev,
            root_entry.st_ino,
        ) != (root_opened.st_dev, root_opened.st_ino):
            raise ArtifactIntegrityError("derived artifact directory is invalid or unavailable")

        parent_fd = root_fd
        for component in (".discourselens", "artifacts", "nlp"):
            entry = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(entry.st_mode):
                raise ArtifactIntegrityError(
                    "derived artifact directory is invalid or unavailable"
                )
            child_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            descriptors.append(child_fd)
            opened = os.fstat(child_fd)
            if (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino):
                raise ArtifactIntegrityError(
                    "derived artifact directory changed during access"
                )
            parent_fd = child_fd

        directory_fds = tuple(descriptors)
        _validate_nlp_artifact_directory(project, directory_fds)
        return directory_fds
    except ArtifactIntegrityError:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    except OSError as exc:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise ArtifactIntegrityError(
            "derived artifact directory is invalid or unavailable"
        ) from exc


def _validate_nlp_artifact_directory(
    project: Project, directory_fds: tuple[int, ...]
) -> None:
    """Ensure every project-relative directory entry still matches its pinned handle."""
    if len(directory_fds) != 4:
        raise ArtifactIntegrityError("derived artifact directory is invalid or unavailable")
    root_fd, hidden_fd, artifacts_fd, nlp_fd = directory_fds
    try:
        root_entry = os.stat(project.root, follow_symlinks=False)
        root_opened = os.fstat(root_fd)
        if not stat.S_ISDIR(root_entry.st_mode) or (
            root_entry.st_dev,
            root_entry.st_ino,
        ) != (root_opened.st_dev, root_opened.st_ino):
            raise ArtifactIntegrityError("derived artifact directory changed during access")
        for parent_fd, name, child_fd in (
            (root_fd, ".discourselens", hidden_fd),
            (hidden_fd, "artifacts", artifacts_fd),
            (artifacts_fd, "nlp", nlp_fd),
        ):
            entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            opened = os.fstat(child_fd)
            if not stat.S_ISDIR(entry.st_mode) or (entry.st_dev, entry.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise ArtifactIntegrityError(
                    "derived artifact directory changed during access"
                )
    except ArtifactIntegrityError:
        raise
    except OSError as exc:
        raise ArtifactIntegrityError(
            "derived artifact directory changed during access"
        ) from exc


def _close_nlp_artifact_directory(directory_fds: tuple[int, ...] | None) -> None:
    if directory_fds is not None:
        for descriptor in reversed(directory_fds):
            os.close(descriptor)


def _open_temporary_artifact(path: Path, *, directory_fd: int | None = None) -> BinaryIO:
    """Create an artifact temp file exclusively so an existing link is never followed."""
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        if directory_fd is None:
            descriptor = os.open(path, flags, 0o600)
        else:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
    except FileExistsError as exc:
        raise ArtifactIntegrityError("derived artifact temporary path already exists") from exc
    except OSError as exc:
        raise ArtifactIntegrityError("derived artifact temporary file is unavailable") from exc
    try:
        return os.fdopen(descriptor, "w+b")
    except OSError as exc:
        os.close(descriptor)
        raise ArtifactIntegrityError("derived artifact temporary file is unavailable") from exc


def _publish_temporary_artifact(
    temporary_path: Path, artifact_path: Path, *, directory_fd: int | None = None
) -> tuple[int, int]:
    """Publish atomically without replacing a destination created after preflight."""
    try:
        source_stat = (
            os.stat(temporary_path, follow_symlinks=False)
            if directory_fd is None
            else os.stat(temporary_path.name, dir_fd=directory_fd, follow_symlinks=False)
        )
        if not stat.S_ISREG(source_stat.st_mode):
            raise ArtifactIntegrityError("derived artifact temporary file is invalid")
        # A same-directory hard link is an atomic create-if-absent operation.
        # Path.replace would silently overwrite a concurrent artifact at this path.
        if directory_fd is None:
            os.link(temporary_path, artifact_path, follow_symlinks=False)
        else:
            os.link(
                temporary_path.name,
                artifact_path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
    except FileExistsError as exc:
        raise ArtifactIntegrityError("derived artifact identity already exists") from exc
    try:
        destination_stat = (
            artifact_path.lstat()
            if directory_fd is None
            else os.stat(artifact_path.name, dir_fd=directory_fd, follow_symlinks=False)
        )
        if (
            not stat.S_ISREG(destination_stat.st_mode)
            or (source_stat.st_dev, source_stat.st_ino)
            != (destination_stat.st_dev, destination_stat.st_ino)
        ):
            _unlink_entry_if_identity(
                artifact_path,
                (destination_stat.st_dev, destination_stat.st_ino),
                directory_fd=directory_fd,
            )
            raise ArtifactIntegrityError("derived artifact changed during publication")
    except OSError as exc:
        raise ArtifactIntegrityError("derived artifact publication could not be verified") from exc
    try:
        if directory_fd is None:
            temporary_path.unlink()
        else:
            os.unlink(temporary_path.name, dir_fd=directory_fd)
    except OSError:
        # Remove only the link we just created; another writer may have replaced
        # the destination between publication and cleanup.
        try:
            _unlink_entry_if_identity(
                artifact_path,
                (source_stat.st_dev, source_stat.st_ino),
                directory_fd=directory_fd,
            )
        except OSError:
            pass
        raise
    return source_stat.st_dev, source_stat.st_ino


def _unlink_entry_if_identity(
    path: Path,
    identity: tuple[int, int],
    *,
    directory_fd: int | None = None,
) -> bool:
    """Unlink only a regular file whose device/inode still belongs to this build."""
    try:
        entry = (
            path.lstat()
            if directory_fd is None
            else os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        )
        if (
            not stat.S_ISREG(entry.st_mode)
            or (entry.st_dev, entry.st_ino) != identity
        ):
            return False
        if directory_fd is None:
            path.unlink()
        else:
            os.unlink(path.name, dir_fd=directory_fd)
        return True
    except FileNotFoundError:
        return False


def _sha256_artifact_entry(path: Path, *, directory_fd: int | None = None) -> str:
    """Hash an artifact through a no-follow handle anchored to its directory."""
    if directory_fd is None:
        return sha256_file(path)

    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ArtifactIntegrityError("derived artifact is unavailable") from exc
    try:
        opened_before = os.fstat(descriptor)
        path_before = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or (opened_before.st_dev, opened_before.st_ino)
            != (path_before.st_dev, path_before.st_ino)
        ):
            raise ArtifactIntegrityError("derived artifact changed while it was opened")
        with os.fdopen(descriptor, "rb") as artifact:
            descriptor = -1
            for block in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(block)
            opened_after = os.fstat(artifact.fileno())
        path_after = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
    except ArtifactIntegrityError:
        raise
    except OSError as exc:
        raise ArtifactIntegrityError("derived artifact changed while it was hashed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    fingerprint = (
        opened_before.st_dev,
        opened_before.st_ino,
        opened_before.st_size,
        opened_before.st_mtime_ns,
        opened_before.st_ctime_ns,
    )
    after_fingerprint = (
        opened_after.st_dev,
        opened_after.st_ino,
        opened_after.st_size,
        opened_after.st_mtime_ns,
        opened_after.st_ctime_ns,
    )
    path_fingerprint = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if fingerprint != after_fingerprint or fingerprint != path_fingerprint:
        raise ArtifactIntegrityError("derived artifact changed while it was hashed")
    return digest.hexdigest()


def _unlink_nlp_artifact(
    project: Project, path: Path, *, expected_identity: tuple[int, int]
) -> None:
    """Remove this build's artifact without deleting a later replacement at its path."""
    directory_fds = _open_nlp_artifact_directory(project)
    if directory_fds is None:
        artifact_directory = _prepare_nlp_artifact_directory(project)
        if path.parent != artifact_directory:
            raise ArtifactIntegrityError("derived artifact path is outside managed storage")
        _unlink_entry_if_identity(path, expected_identity)
        return
    try:
        _validate_nlp_artifact_directory(project, directory_fds)
        _unlink_entry_if_identity(path, expected_identity, directory_fd=directory_fds[-1])
    finally:
        _close_nlp_artifact_directory(directory_fds)


def _require_available_artifact_identity(project: Project, artifact_id: str, path: Path) -> None:
    """Reject obvious identity or destination collisions before expensive generation."""
    if project.db.execute(
        "SELECT 1 FROM artifact_registry WHERE id=?", (artifact_id,)
    ).fetchone():
        raise ArtifactIntegrityError("derived artifact identity already exists")
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ArtifactIntegrityError("derived artifact destination is unavailable") from exc
    raise ArtifactIntegrityError("derived artifact identity already exists")


class ParsedTokenCoverageField(TypedDict):
    status: Literal["SUPPORTED", "LIMITED"]
    limitation: str | None
    annotated_tokens: int
    total_tokens: int


class ParsedTokenizationCoverage(TypedDict):
    status: Literal["SUPPORTED", "LIMITED"]
    covered_non_whitespace_characters: int
    source_non_whitespace_characters: int
    total_tokens: int
    limitation: str | None


class ParsedArtifactCoverage(TypedDict):
    format_version: Literal[1]
    tokenization: ParsedTokenizationCoverage
    lemma: ParsedTokenCoverageField
    pos: ParsedTokenCoverageField
    dependency_parse: ParsedTokenCoverageField


@dataclass(frozen=True)
class TokenArtifact:
    artifact_id: str
    revision_id: str
    relative_path: str
    derivation_hash: str
    file_hash: str
    schema_version: str = ""
    coverage: ParsedArtifactCoverage | None = None

    @property
    def content_hash(self) -> str:
        """SHA-256 of the persisted artifact bytes; retained as a familiar accessor."""
        return self.file_hash


class DependencyPatternRow(TypedDict):
    head_lemma: str
    relation: str
    dependent_lemma: str
    frequency: int
    document_frequency: int


class DependencyPatternsResult(TypedDict):
    scope_id: str
    artifact_ids: list[str]
    calculation_version: str
    filters: dict[str, str | None]
    offset: int
    limit: int
    total_rows: int
    has_more: bool
    candidate_count_lower_bound: int
    rows: list[DependencyPatternRow]
    truncated: bool
    coverage: dict[str, dict[str, object]]


class LemmaKwicRow(TypedDict):
    revision_id: str
    paragraph_id: str
    start_offset: int
    end_offset: int
    left: str
    keyword: str
    lemma: str
    right: str


class LemmaKwicPage(TypedDict):
    scope_id: str
    artifact_ids: list[str]
    lemma: str
    result_semantics: str
    window: int
    offset: int
    limit: int
    total_count: int
    has_more: bool
    rows: list[LemmaKwicRow]
    field_coverage: dict[str, object]
    tokenization_coverage: dict[str, object]


class LemmaKwicEvidenceResult(TypedDict):
    scope_id: str
    artifact_ids: list[str]
    revision_id: str
    lemma: str
    start_offset: int
    end_offset: int
    exact_text: str
    coverage_id: str
    coverage_mode: str
    backend: str
    backend_version: str | None
    model: str
    model_version: str | None
    schema_version: str
    field_coverage: dict[str, object]
    tokenization_coverage: dict[str, object]
    evidence_id: str
    citation: str


class ParsedFrequencyPageRow(TypedDict):
    value: str
    count: int


class ParsedFrequencyPage(TypedDict):
    scope_id: str
    revision_ids: list[str]
    artifact_ids: list[str]
    field: str
    coverage: dict[str, dict[str, object]]
    offset: int
    limit: int
    total_values: int
    has_more: bool
    rows: list[ParsedFrequencyPageRow]
    field_coverage: dict[str, object]


class SurfaceFrequencyPageRow(TypedDict):
    token: str
    count: int


class SurfaceFrequencyPage(TypedDict):
    scope_id: str
    revision_ids: list[str]
    artifact_ids: list[str]
    artifact_schema_version: str
    calculation_version: str
    tokenizer: str
    normalization: str
    unicode_data_version: str
    field: Literal["normalized", "document_frequency"]
    offset: int
    limit: int
    total_values: int
    has_more: bool
    rows: list[SurfaceFrequencyPageRow]


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise OptionalDependencyError("token artifacts require discourselens[corpus]") from exc
    return pa, pq


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise OptionalDependencyError("token analytics require discourselens[corpus]") from exc
    return duckdb


def _backend_version(backend_id: str, backend: NlpBackend | None = None) -> str | None:
    """Return installed implementation version when it can be identified locally."""
    declared = getattr(backend, "backend_version", None)
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    distribution = {"spacy": "spacy", "stanza": "stanza"}.get(backend_id, backend_id)
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def _model_version(backend: NlpBackend) -> str | None:
    """Use backend-declared or local model metadata; never infer a version from a name."""
    declared = getattr(backend, "model_version", None)
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    pipeline = getattr(backend, "pipeline", None)
    metadata = getattr(pipeline, "meta", None)
    if isinstance(metadata, dict):
        value = metadata.get("version")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_surface_token_artifact(project: Project, revision_id: str) -> TokenArtifact:
    """Build a deterministic token artifact for one immutable revision.

    `normalized` uses surface-token-v2 NFC-after-casefold form, never a lemma.
    """
    pa, pq = _pyarrow()
    revision = project.db.execute(
        "SELECT canonical_hash,text_path FROM document_revisions WHERE id=?", (revision_id,)
    ).fetchone()
    if not revision:
        raise ScopeViolationError("unknown revision")
    paragraph_cursor = project.db.execute(
        "SELECT id,start_offset,end_offset FROM paragraphs WHERE revision_id=? ORDER BY start_offset",
        (revision_id,),
    )
    artifact_id = project.uid()
    relative = Path("artifacts") / "nlp" / f"{artifact_id}.parquet"
    artifact_directory = _prepare_nlp_artifact_directory(project)
    artifact_path = artifact_directory / f"{artifact_id}.parquet"
    _require_available_artifact_identity(project, artifact_id, artifact_path)
    temporary_path = artifact_path.with_name(f".{artifact_id}.partial")
    revision_text_path = (project.hidden / str(revision["text_path"])).resolve()
    try:
        revision_text_path.relative_to(project.hidden.resolve())
    except ValueError as exc:
        raise ArtifactIntegrityError("canonical text path escapes project storage") from exc
    if not revision_text_path.is_file():
        raise ArtifactIntegrityError("canonical text artifact is missing")
    schema = pa.schema(
        [
            pa.field("revision_id", pa.string(), nullable=False),
            pa.field("paragraph_id", pa.string()),
            pa.field("token_index", pa.int64(), nullable=False),
            pa.field("surface", pa.string(), nullable=False),
            pa.field("normalized", pa.string(), nullable=False),
            pa.field("char_start", pa.int64(), nullable=False),
            pa.field("char_end", pa.int64(), nullable=False),
        ]
    )
    batch: dict[str, list[object]] = {
        "revision_id": [],
        "paragraph_id": [],
        "token_index": [],
        "surface": [],
        "normalized": [],
        "char_start": [],
        "char_end": [],
    }
    paragraph = paragraph_cursor.fetchone()
    token_index = 0
    canonical_hasher = hashlib.sha256()
    directory_fds = _open_nlp_artifact_directory(project)
    directory_fd = directory_fds[-1] if directory_fds is not None else None
    temporary_created = False
    published_identity: tuple[int, int] | None = None
    try:
        if directory_fds is not None:
            _validate_nlp_artifact_directory(project, directory_fds)
        temporary_file = _open_temporary_artifact(temporary_path, directory_fd=directory_fd)
        temporary_created = True
        try:
            parquet_writer = pq.ParquetWriter(temporary_file, schema, compression="zstd")
        except Exception:
            temporary_file.close()
            raise
        with temporary_file, parquet_writer as writer:
            for start, end, token in _iter_surface_matches(
                revision_text_path,
                _SURFACE_TEXT_CHUNK_CHARS,
                on_text_chunk=lambda chunk: canonical_hasher.update(chunk.encode("utf-8")),
            ):
                while paragraph is not None and start >= paragraph["end_offset"]:
                    paragraph = paragraph_cursor.fetchone()
                paragraph_id = None
                if (
                    paragraph is not None
                    and paragraph["start_offset"] <= start < paragraph["end_offset"]
                ):
                    paragraph_id = str(paragraph["id"])
                batch["revision_id"].append(revision_id)
                batch["paragraph_id"].append(paragraph_id)
                batch["token_index"].append(token_index)
                batch["surface"].append(token)
                batch["normalized"].append(_surface_form_v2(token))
                batch["char_start"].append(start)
                batch["char_end"].append(end)
                token_index += 1
                if token_index % _SURFACE_TOKEN_BATCH_SIZE == 0:
                    _write_parquet_batch(writer, pa, schema, batch)
            _write_parquet_batch(writer, pa, schema, batch)
            if canonical_hasher.hexdigest() != revision["canonical_hash"]:
                raise ArtifactIntegrityError("canonical text does not match document revision hash")
        if directory_fds is not None:
            _validate_nlp_artifact_directory(project, directory_fds)
        published_identity = _publish_temporary_artifact(
            temporary_path, artifact_path, directory_fd=directory_fd
        )
        if directory_fds is not None:
            _validate_nlp_artifact_directory(project, directory_fds)
        file_hash = _sha256_artifact_entry(artifact_path, directory_fd=directory_fd)
        if directory_fds is not None:
            _validate_nlp_artifact_directory(project, directory_fds)
        derivation_hash = hashlib.sha256(
            f"{revision['canonical_hash']}:{SURFACE_TOKEN_SCHEMA_VERSION}".encode()
        ).hexdigest()
    except Exception:
        if directory_fd is None:
            if temporary_created:
                temporary_path.unlink(missing_ok=True)
            if published_identity is not None:
                _unlink_entry_if_identity(artifact_path, published_identity)
        else:
            if temporary_created:
                try:
                    os.unlink(temporary_path.name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            if published_identity is not None:
                _unlink_entry_if_identity(
                    artifact_path, published_identity, directory_fd=directory_fd
                )
        raise
    finally:
        _close_nlp_artifact_directory(directory_fds)
    try:
        project.db.execute(
            "INSERT INTO artifact_registry(id,revision_id,artifact_type,backend,backend_version,"
            "model,model_version,schema_version,content_hash,relative_path,created_at,file_hash) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                artifact_id,
                revision_id,
                "surface_tokens",
                "pyarrow",
                _backend_version("pyarrow"),
                None,
                None,
                SURFACE_TOKEN_SCHEMA_VERSION,
                derivation_hash,
                str(relative),
                project.db.execute("SELECT datetime('now')").fetchone()[0],
                file_hash,
            ),
        )
        project.db.commit()
    except Exception:
        project.db.rollback()
        try:
            assert published_identity is not None
            _unlink_nlp_artifact(
                project, artifact_path, expected_identity=published_identity
            )
        except ArtifactIntegrityError:
            pass
        raise
    return TokenArtifact(
        artifact_id,
        revision_id,
        str(relative),
        derivation_hash,
        file_hash,
        SURFACE_TOKEN_SCHEMA_VERSION,
    )


def _iter_surface_matches(
    path: Path,
    chunk_chars: int,
    *,
    on_text_chunk: Callable[[str], None] | None = None,
) -> Iterator[tuple[int, int, str]]:
    """Tokenize canonical text with v2 while retaining code-point offsets."""
    if chunk_chars < 1:
        raise ValueError("text chunk size must be positive")

    def text_chunks() -> Iterator[str]:
        # Preserve canonical CRLF/CR code points for offsets and source hashing.
        with path.open("r", encoding="utf-8", newline="") as stream:
            while chunk := stream.read(chunk_chars):
                if on_text_chunk is not None:
                    on_text_chunk(chunk)
                yield chunk

    for token in _tokenize_v2_chunks(text_chunks()):
        yield token.start_offset, token.end_offset, token.surface


def _write_parquet_batch(writer, pa, schema, rows: dict[str, list[object]]) -> None:
    row_count = len(rows["token_index"])
    if not row_count:
        return
    writer.write_table(pa.Table.from_pydict(rows, schema=schema), row_group_size=row_count)
    for values in rows.values():
        values.clear()


def _count_non_whitespace(text: str) -> int:
    return sum(not character.isspace() for character in text)


def _token_span_coverage(text: str, tokens) -> tuple[int, bool]:
    """Count uniquely covered source characters and reject incomplete tokenization.

    Counts are exact Unicode code points, not a quality percentage. Whitespace
    need not be tokenized, but every non-whitespace source character must be
    covered exactly once by an in-order token span.
    """
    covered = 0
    previous_end = 0
    complete = True
    for token in tokens:
        start = token.char_start
        end = token.char_end
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or end > len(text)
        ):
            return 0, False
        if start < previous_end:
            complete = False
        if any(character.isspace() for character in text[start:end]):
            complete = False
        unique_start = max(start, previous_end)
        if unique_start < end:
            covered += _count_non_whitespace(text[unique_start:end])
        previous_end = max(previous_end, end)
    source_count = _count_non_whitespace(text)
    return covered, complete and covered == source_count


def _parsed_coverage_payload(
    *,
    token_count: int,
    lemma_count: int,
    pos_count: int,
    dependency_count: int,
    dependency_trees_complete: bool,
    source_characters: int,
    covered_characters: int,
    token_spans_complete: bool,
) -> ParsedArtifactCoverage:
    tokenization_supported = token_spans_complete and covered_characters == source_characters
    tokenization_status = (
        OPERATIONAL_SUPPORTED if tokenization_supported else OPERATIONAL_LIMITED
    )

    def annotation_coverage(count: int, name: str) -> dict[str, object]:
        supported = tokenization_supported and count == token_count
        limitation = None if supported else (
            f"The artifact has incomplete {name} annotations or token coverage."
        )
        return {
            "status": OPERATIONAL_SUPPORTED if supported else OPERATIONAL_LIMITED,
            "annotated_tokens": count,
            "total_tokens": token_count,
            "limitation": limitation,
        }

    dependency_supported = (
        tokenization_supported and dependency_trees_complete and dependency_count == token_count
    )
    return cast(ParsedArtifactCoverage, {
        "format_version": _PARSED_TOKEN_COVERAGE_FORMAT_VERSION,
        "tokenization": {
            "status": tokenization_status,
            "covered_non_whitespace_characters": covered_characters,
            "source_non_whitespace_characters": source_characters,
            "total_tokens": token_count,
            "limitation": (
                None if tokenization_supported else
                "The artifact token spans did not cover every non-whitespace source character exactly once."
            ),
        },
        "lemma": annotation_coverage(lemma_count, "lemma"),
        "pos": annotation_coverage(pos_count, "POS"),
        "dependency_parse": {
            "status": OPERATIONAL_SUPPORTED if dependency_supported else OPERATIONAL_LIMITED,
            "annotated_tokens": dependency_count,
            "total_tokens": token_count,
            "limitation": (
                None if dependency_supported else
                "The artifact has incomplete dependency annotations or token coverage."
            ),
        },
    })


def build_parsed_token_artifact(
    project: Project, revision_id: str, backend: NlpBackend
) -> TokenArtifact:
    """Persist a versioned local NLP parse as a derived Parquet/Zstd artifact.

    This operation never changes a DocumentRevision: parser/model upgrades simply
    create a new artifact with their own declared provenance.
    """
    pa, pq = _pyarrow()
    revision = project.db.execute(
        "SELECT canonical_hash,text_path FROM document_revisions WHERE id=?", (revision_id,)
    ).fetchone()
    if not revision:
        raise ScopeViolationError("unknown revision")
    text_path = (project.hidden / str(revision["text_path"])).resolve()
    try:
        text_path.relative_to(project.hidden.resolve())
    except ValueError as exc:
        raise ArtifactIntegrityError("canonical text path escapes project storage") from exc
    if not text_path.is_file():
        raise ArtifactIntegrityError("canonical text artifact is missing")

    artifact_id = project.uid()
    relative = Path("artifacts") / "nlp" / f"{artifact_id}.parquet"
    artifact_directory = _prepare_nlp_artifact_directory(project)
    artifact_path = artifact_directory / f"{artifact_id}.parquet"
    _require_available_artifact_identity(project, artifact_id, artifact_path)
    temporary_path = artifact_path.with_name(f".{artifact_id}.partial")
    schema = pa.schema(
        [
            pa.field("revision_id", pa.string(), nullable=False),
            pa.field("paragraph_id", pa.string()),
            pa.field("token_index", pa.int64(), nullable=False),
            pa.field("surface", pa.string(), nullable=False),
            pa.field("lemma", pa.string(), nullable=False),
            pa.field("pos", pa.string(), nullable=False),
            pa.field("morphology", pa.string()),
            pa.field("dependency_head", pa.int64()),
            pa.field("dependency_relation", pa.string()),
            pa.field("char_start", pa.int64(), nullable=False),
            pa.field("char_end", pa.int64(), nullable=False),
        ]
    )
    rows: dict[str, list[object]] = {field.name: [] for field in schema}
    token_index = 0
    directory_fds = _open_nlp_artifact_directory(project)
    directory_fd = directory_fds[-1] if directory_fds is not None else None
    temporary_created = False
    published_identity: tuple[int, int] | None = None
    try:
        if directory_fds is not None:
            _validate_nlp_artifact_directory(project, directory_fds)
        temporary_file = _open_temporary_artifact(temporary_path, directory_fd=directory_fd)
        temporary_created = True
        try:
            parquet_writer = pq.ParquetWriter(temporary_file, schema, compression="zstd")
        except Exception:
            temporary_file.close()
            raise
        with temporary_file, parquet_writer as writer:
            token_count = 0
            lemma_count = 0
            pos_count = 0
            dependency_count = 0
            source_characters = 0
            covered_characters = 0
            token_spans_complete = True
            dependency_trees_complete = True
            paragraphs = project.db.execute(
                "SELECT id,start_offset,end_offset,text FROM paragraphs "
                "WHERE revision_id=? ORDER BY start_offset",
                (revision_id,),
            )
            for paragraph_id, start, end, paragraph_text in _iter_canonical_paragraphs(
                text_path,
                paragraphs,
                str(revision["canonical_hash"]),
            ):
                paragraph_token_base = token_index
                parsed = backend.parse(paragraph_text)
                paragraph_covered, paragraph_tokenization_complete = _token_span_coverage(
                    paragraph_text, parsed
                )
                source_characters += _count_non_whitespace(paragraph_text)
                covered_characters += paragraph_covered
                token_spans_complete = token_spans_complete and paragraph_tokenization_complete
                if (parsed or _count_non_whitespace(paragraph_text)) and not _has_valid_dependency_tree(parsed):
                    dependency_trees_complete = False
                token_count += len(parsed)
                lemma_count += sum(
                    isinstance(token.lemma, str) and bool(token.lemma.strip())
                    for token in parsed
                )
                pos_count += sum(
                    isinstance(token.pos, str) and bool(token.pos.strip())
                    for token in parsed
                )
                dependency_count += sum(
                    isinstance(token.dependency_relation, str)
                    and bool(token.dependency_relation.strip())
                    and (
                        token.dependency_relation.casefold() == "root"
                        and token.dependency_head is None
                        or type(token.dependency_head) is int
                        and 0 <= token.dependency_head < len(parsed)
                        and token.dependency_head != index
                    )
                    for index, token in enumerate(parsed)
                )
                for token in parsed:
                    if (
                        type(token.char_start) is not int
                        or type(token.char_end) is not int
                        or token.char_start < 0
                        or token.char_end <= token.char_start
                        or token.char_end > len(paragraph_text)
                    ):
                        raise ScopeViolationError(
                            "NLP backend returned invalid canonical offsets"
                        )
                    if paragraph_text[token.char_start : token.char_end] != token.surface:
                        raise ScopeViolationError(
                            "NLP backend token text does not match canonical text"
                        )
                    local_head = token.dependency_head
                    if local_head is not None and (
                        type(local_head) is not int or not 0 <= local_head < len(parsed)
                    ):
                        raise ScopeViolationError("NLP backend returned an invalid dependency head")
                    rows["revision_id"].append(revision_id)
                    rows["paragraph_id"].append(paragraph_id)
                    rows["token_index"].append(token_index)
                    rows["surface"].append(token.surface)
                    rows["lemma"].append(token.lemma)
                    rows["pos"].append(token.pos)
                    rows["morphology"].append(token.morphology)
                    rows["dependency_head"].append(
                        paragraph_token_base + local_head if local_head is not None else None
                    )
                    rows["dependency_relation"].append(token.dependency_relation)
                    rows["char_start"].append(start + token.char_start)
                    rows["char_end"].append(start + token.char_end)
                    token_index += 1
                    if token_index % _PARSED_TOKEN_BATCH_SIZE == 0:
                        _write_parquet_batch(writer, pa, schema, rows)
            _write_parquet_batch(writer, pa, schema, rows)
            coverage = _parsed_coverage_payload(
                token_count=token_count,
                lemma_count=lemma_count,
                pos_count=pos_count,
                dependency_count=dependency_count,
                dependency_trees_complete=dependency_trees_complete,
                source_characters=source_characters,
                covered_characters=covered_characters,
                token_spans_complete=token_spans_complete,
            )
            writer.add_key_value_metadata(
                {
                    _PARSED_TOKEN_COVERAGE_METADATA_KEY: json.dumps(
                        coverage, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                    )
                }
            )
        if directory_fds is not None:
            _validate_nlp_artifact_directory(project, directory_fds)
        published_identity = _publish_temporary_artifact(
            temporary_path, artifact_path, directory_fd=directory_fd
        )
        if directory_fds is not None:
            _validate_nlp_artifact_directory(project, directory_fds)
        file_hash = _sha256_artifact_entry(artifact_path, directory_fd=directory_fd)
        if directory_fds is not None:
            _validate_nlp_artifact_directory(project, directory_fds)
        backend_version = _backend_version(backend.backend_id, backend)
        model_version = _model_version(backend)
        backend_language = backend.language
        if not isinstance(backend_language, str) or not backend_language.strip():
            raise ScopeViolationError("NLP backend must declare a non-empty language")
        # Language is an input to parsing just like backend/model identity. Use
        # structured serialization rather than delimiter joining so custom
        # identifiers cannot make distinct derivations hash to the same payload.
        provenance = json.dumps(
            [
                "parsed-token-derivation-v2",
                str(revision["canonical_hash"]),
                backend.backend_id,
                backend_version or "unknown-backend-version",
                backend.model_id,
                model_version or "unknown-model-version",
                backend_language,
                PARSED_TOKEN_SCHEMA_VERSION,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        derivation_hash = hashlib.sha256(provenance.encode()).hexdigest()
    except Exception:
        if directory_fd is None:
            if temporary_created:
                temporary_path.unlink(missing_ok=True)
            if published_identity is not None:
                _unlink_entry_if_identity(artifact_path, published_identity)
        else:
            if temporary_created:
                try:
                    os.unlink(temporary_path.name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            if published_identity is not None:
                _unlink_entry_if_identity(
                    artifact_path, published_identity, directory_fd=directory_fd
                )
        raise
    finally:
        _close_nlp_artifact_directory(directory_fds)
    try:
        project.db.execute(
            "INSERT INTO artifact_registry(id,revision_id,artifact_type,backend,backend_version,"
            "model,model_version,schema_version,content_hash,relative_path,created_at,file_hash) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                artifact_id,
                revision_id,
                "parsed_tokens",
                backend.backend_id,
                backend_version,
                backend.model_id,
                model_version,
                PARSED_TOKEN_SCHEMA_VERSION,
                derivation_hash,
                str(relative),
                project.db.execute("SELECT datetime('now')").fetchone()[0],
                file_hash,
            ),
        )
        project.db.commit()
    except Exception:
        project.db.rollback()
        try:
            assert published_identity is not None
            _unlink_nlp_artifact(
                project, artifact_path, expected_identity=published_identity
            )
        except ArtifactIntegrityError:
            pass
        raise
    return TokenArtifact(
        artifact_id,
        revision_id,
        str(relative),
        derivation_hash,
        file_hash,
        PARSED_TOKEN_SCHEMA_VERSION,
        coverage,
    )


def _iter_canonical_paragraphs(text_path, paragraphs, expected_hash: str):
    """Yield paragraph slices while hashing canonical text without materializing it."""
    canonical_hasher = hashlib.sha256()
    canonical_offset = 0
    # Preserve the canonical file's original newline code points for offset checks
    # and for the revision hash, which was computed from the unnormalized text.
    with text_path.open("r", encoding="utf-8", newline="") as source:
        for row in paragraphs:
            start = int(row["start_offset"])
            end = int(row["end_offset"])
            if start < canonical_offset or end < start:
                raise ArtifactIntegrityError("paragraph offsets are invalid for canonical text")
            while canonical_offset < start:
                chunk = source.read(min(_SURFACE_TEXT_CHUNK_CHARS, start - canonical_offset))
                if not chunk:
                    raise ArtifactIntegrityError("paragraph offset exceeds canonical text length")
                if chunk.strip():
                    raise ArtifactIntegrityError(
                        "canonical text is missing corresponding paragraph records"
                    )
                canonical_hasher.update(chunk.encode("utf-8"))
                canonical_offset += len(chunk)
            if end == start:
                continue
            pieces: list[str] = []
            remaining = end - start
            while remaining:
                chunk = source.read(min(_SURFACE_TEXT_CHUNK_CHARS, remaining))
                if not chunk:
                    raise ArtifactIntegrityError("paragraph offset exceeds canonical text length")
                pieces.append(chunk)
                canonical_hasher.update(chunk.encode("utf-8"))
                canonical_offset += len(chunk)
                remaining -= len(chunk)
            paragraph_text = "".join(pieces)
            stored_text = row["text"]
            if stored_text is not None and stored_text != paragraph_text:
                raise ArtifactIntegrityError("paragraph text does not match canonical offsets")
            yield str(row["id"]), start, end, paragraph_text
        while True:
            chunk = source.read(_SURFACE_TEXT_CHUNK_CHARS)
            if not chunk:
                break
            if chunk.strip():
                raise ArtifactIntegrityError(
                    "canonical text is missing corresponding paragraph records"
                )
            canonical_hasher.update(chunk.encode("utf-8"))
            canonical_offset += len(chunk)
    if canonical_hasher.hexdigest() != expected_hash:
        raise ArtifactIntegrityError("canonical text does not match document revision hash")


def normalized_frequency(project: Project, scope_id: str, artifact_ids: list[str]) -> Counter[str]:
    """Aggregate selected derived artifacts after validating their immutable scope."""
    artifacts = _scoped_artifacts(project, scope_id, artifact_ids, "surface_tokens")
    _require_compatible_surface_artifacts(artifacts)
    duckdb = _duckdb()
    paths = [
        str(verify_project_artifact(project.root, item["relative_path"], item["file_hash"]))
        for item in artifacts
    ]
    connection = duckdb.connect()
    try:
        result = connection.execute(
            "SELECT normalized,count(*) FROM read_parquet(?) GROUP BY normalized ORDER BY normalized",
            [paths],
        ).fetchall()
    finally:
        connection.close()
    return Counter({normalized: count for normalized, count in result})


def document_frequency(project: Project, scope_id: str, artifact_ids: list[str]) -> Counter[str]:
    """Count immutable document revisions containing each normalized surface token."""
    artifacts = _scoped_artifacts(project, scope_id, artifact_ids, "surface_tokens")
    _require_compatible_surface_artifacts(artifacts)
    duckdb = _duckdb()
    paths = [
        str(verify_project_artifact(project.root, item["relative_path"], item["file_hash"]))
        for item in artifacts
    ]
    connection = duckdb.connect()
    try:
        result = connection.execute(
            "SELECT normalized,count(DISTINCT revision_id) FROM read_parquet(?) "
            "GROUP BY normalized ORDER BY normalized",
            [paths],
        ).fetchall()
    finally:
        connection.close()
    return Counter({normalized: count for normalized, count in result})


def surface_frequency_page(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    *,
    field: Literal["normalized", "document_frequency"],
    offset: int = 0,
    limit: int = 100,
) -> SurfaceFrequencyPage:
    """Return a bounded, deterministic surface-frequency page for a full scope.

    ``normalized`` ranks token occurrence counts; ``document_frequency`` ranks
    the number of scoped revisions containing each normalized token.
    """
    if not isinstance(field, str) or field not in {"normalized", "document_frequency"}:
        raise ScopeViolationError("field must be normalized or document_frequency")
    if (
        type(offset) is not int
        or type(limit) is not int
        or offset < 0
        or not 1 <= limit <= 1_000
    ):
        raise ScopeViolationError(
            "surface frequency offset must be non-negative and limit must be 1..1000"
        )

    revision_ids = project.scope_revisions(scope_id)
    artifacts = _complete_scope_artifacts(
        project, scope_id, artifact_ids, "surface_tokens"
    )
    _require_compatible_surface_artifacts(artifacts)
    paths = [
        str(verify_project_artifact(project.root, item["relative_path"], item["file_hash"]))
        for item in artifacts
    ]
    duckdb = _duckdb()
    connection = duckdb.connect()
    try:
        count_expression = (
            "count(*)" if field == "normalized" else "count(DISTINCT revision_id)"
        )
        grouped = (
            f"SELECT normalized AS token,{count_expression} AS count "
            "FROM read_parquet(?) GROUP BY normalized"
        )
        total_row = connection.execute(
            f"SELECT count(*) FROM ({grouped})", [paths]
        ).fetchone()
        if total_row is None:
            raise ScopeViolationError("surface frequency query returned no aggregate row")
        total_values = int(total_row[0])
        rows = connection.execute(
            f"{grouped} ORDER BY count DESC,token ASC LIMIT ? OFFSET ?",
            [paths, limit, offset],
        ).fetchall()
    finally:
        connection.close()

    page_rows: list[SurfaceFrequencyPageRow] = [
        {"token": str(token), "count": int(count)} for token, count in rows
    ]
    return {
        "scope_id": scope_id,
        "revision_ids": revision_ids,
        "artifact_ids": [str(item["id"]) for item in artifacts],
        "artifact_schema_version": SURFACE_TOKEN_SCHEMA_VERSION,
        "calculation_version": "surface-token-artifact-frequency-v2",
        "tokenizer": SURFACE_TOKENIZER_V2,
        "normalization": SURFACE_NORMALIZATION_V2,
        "unicode_data_version": unicodedata.unidata_version,
        "field": field,
        "offset": offset,
        "limit": limit,
        "total_values": total_values,
        "has_more": offset + len(page_rows) < total_values,
        "rows": page_rows,
    }


def lemma_frequency(project: Project, scope_id: str, artifact_ids: list[str]) -> Counter[str]:
    """Aggregate parser-derived lemmas from explicit parsed-token artifacts."""
    return _column_frequency(project, scope_id, artifact_ids, "lemma")


def pos_frequency(project: Project, scope_id: str, artifact_ids: list[str]) -> Counter[str]:
    """Aggregate universal/POS tags supplied by the explicitly selected local backend."""
    return _column_frequency(project, scope_id, artifact_ids, "pos")


def dependency_relation_frequency(
    project: Project, scope_id: str, artifact_ids: list[str]
) -> Counter[str]:
    """Count parser-derived dependency relations; this is a descriptive preprocessor."""
    return _column_frequency(project, scope_id, artifact_ids, "dependency_relation")


def parsed_frequency_page(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    *,
    field: str,
    offset: int = 0,
    limit: int = 100,
) -> ParsedFrequencyPage:
    """Return a deterministic page of parser-derived values in a complete scope."""
    if not isinstance(field, str) or field not in {
        "lemma",
        "pos",
        "dependency_relation",
    }:
        raise ScopeViolationError("field must be lemma, pos, or dependency_relation")
    if (
        type(offset) is not int
        or type(limit) is not int
        or offset < 0
        or not 1 <= limit <= 1_000
    ):
        raise ScopeViolationError(
            "parsed frequency offset must be non-negative and limit must be 1..1000"
        )

    revision_ids = project.scope_revisions(scope_id)
    artifacts = _complete_compatible_parsed_scope_artifacts(
        project, scope_id, artifact_ids
    )
    coverage = _scope_parsed_coverage(project, artifacts)
    if field == "lemma":
        counts = lemma_frequency(project, scope_id, artifact_ids)
    elif field == "pos":
        counts = pos_frequency(project, scope_id, artifact_ids)
    else:
        counts = dependency_relation_frequency(project, scope_id, artifact_ids)
    field_coverage = coverage[
        "dependency_parse" if field == "dependency_relation" else field
    ]

    total_values = len(counts)
    selected = nsmallest(
        offset + limit,
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    rows: list[ParsedFrequencyPageRow] = [
        {"value": value, "count": count}
        for value, count in selected[offset : offset + limit]
    ]
    return {
        "scope_id": scope_id,
        "revision_ids": revision_ids,
        "artifact_ids": [str(item["id"]) for item in artifacts],
        "field": field,
        "coverage": coverage,
        "field_coverage": field_coverage,
        "offset": offset,
        "limit": limit,
        "total_values": total_values,
        "has_more": offset + len(rows) < total_values,
        "rows": rows,
    }


def dependency_patterns(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    *,
    relation: str | None = None,
    head_lemma: str | None = None,
    dependent_lemma: str | None = None,
    offset: int = 0,
    limit: int = 100,
) -> DependencyPatternsResult:
    """Aggregate parser-derived head→dependent patterns in a complete explicit scope.

    These are syntactic candidates, not discourse interpretations. Requiring
    exactly one parsed artifact per scoped revision avoids accidental mixtures
    of partial builds or multiple parser/model versions.
    """
    if (
        type(offset) is not int
        or offset < 0
        or offset > 0x7FFF_FFFF_FFFF_FFFF
        or type(limit) is not int
        or not 1 <= limit <= 1_000
    ):
        raise ScopeViolationError(
            "dependency pattern offset must be in the signed 64-bit range and limit must be between 1 and 1000"
        )
    relation_filter = relation.strip() if relation and relation.strip() else None
    head_filter = head_lemma.casefold().strip() if head_lemma and head_lemma.strip() else None
    dependent_filter = (
        dependent_lemma.casefold().strip() if dependent_lemma and dependent_lemma.strip() else None
    )
    artifacts = _complete_compatible_parsed_scope_artifacts(
        project, scope_id, artifact_ids
    )
    coverage = _scope_parsed_coverage(project, artifacts)
    pattern_supported = all(
        coverage[field]["status"] == OPERATIONAL_SUPPORTED
        for field in ("lemma", "dependency_parse")
    )
    coverage["patterns"] = {
        "status": OPERATIONAL_SUPPORTED if pattern_supported else OPERATIONAL_LIMITED,
        "limitation": (
            None if pattern_supported else
            "Dependency patterns have incomplete lemma, dependency, or token coverage."
        ),
    }
    paths = [
        str(verify_project_artifact(project.root, item["relative_path"], item["file_hash"]))
        for item in artifacts
    ]
    grouped_query = (
        "WITH tokens AS (SELECT * FROM read_parquet(?)), "
        "arcs AS (SELECT head.revision_id AS revision_id, "
        "head.lemma AS head_lemma, dep.dependency_relation AS relation, "
        "dep.lemma AS dependent_lemma FROM tokens AS dep "
        "JOIN tokens AS head ON dep.revision_id=head.revision_id "
        "AND dep.dependency_head=head.token_index "
        "WHERE dep.dependency_relation IS NOT NULL AND dep.dependency_relation <> '' "
        "AND head.lemma IS NOT NULL AND head.lemma <> '' "
        "AND dep.lemma IS NOT NULL AND dep.lemma <> ''), "
        "grouped AS (SELECT head_lemma,relation,dependent_lemma,count(*) AS frequency, "
        "count(DISTINCT revision_id) AS document_frequency FROM arcs "
        "WHERE (? IS NULL OR relation = ?) "
        "AND (? IS NULL OR lower(head_lemma) = ?) "
        "AND (? IS NULL OR lower(dependent_lemma) = ?) "
        "GROUP BY head_lemma,relation,dependent_lemma) "
    )
    filter_parameters = [
        paths,
        relation_filter,
        relation_filter,
        head_filter,
        head_filter,
        dependent_filter,
        dependent_filter,
    ]
    duckdb = _duckdb()
    connection = duckdb.connect()
    try:
        total_row = connection.execute(
            f"{grouped_query} SELECT count(*) FROM grouped", filter_parameters
        ).fetchone()
        if total_row is None:
            raise ScopeViolationError("dependency pattern count query returned no aggregate row")
        total_rows = int(total_row[0])
        rows = connection.execute(
            f"{grouped_query} SELECT head_lemma,relation,dependent_lemma,frequency,"
            "document_frequency FROM grouped "
            "ORDER BY frequency DESC,head_lemma,relation,dependent_lemma LIMIT ? OFFSET ?",
            [*filter_parameters, limit, offset],
        ).fetchall()
    finally:
        connection.close()
    has_more = offset + len(rows) < total_rows
    # Keep the historical lower-bound field stable for the default first page.
    candidate_count_lower_bound = min(total_rows, limit + 1)
    return {
        "scope_id": scope_id,
        "artifact_ids": [str(item["id"]) for item in artifacts],
        "calculation_version": "parsed-dependency-patterns-v2",
        "coverage": coverage,
        "filters": {
            "relation": relation_filter,
            "head_lemma": head_filter,
            "dependent_lemma": dependent_filter,
        },
        "offset": offset,
        "limit": limit,
        "total_rows": total_rows,
        "has_more": has_more,
        "candidate_count_lower_bound": candidate_count_lower_bound,
        "rows": [
            {
                "head_lemma": str(head),
                "relation": str(dep_relation),
                "dependent_lemma": str(dependent),
                "frequency": int(count),
                "document_frequency": int(document_count),
            }
            for head, dep_relation, dependent, count, document_count in rows
        ],
        "truncated": has_more,
    }


def lemma_kwic_page(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    lemma: str,
    *,
    window: int = 5,
    offset: int = 0,
    limit: int = 100,
) -> LemmaKwicPage:
    """Return exact parser-lemma occurrences with canonical token context.

    The supplied parsed-token artifacts must cover every revision in the
    immutable scope and share one parser/model/schema identity. Occurrences are
    ordered by snapshot revision order, then canonical offset and token index.
    Context counts parsed tokens within the same paragraph and preserves the
    exact spacing and punctuation between their canonical text offsets. Results
    describe stored parser annotations; zero hits do not establish absence from
    raw text or absence of the underlying concept.
    """
    if not isinstance(lemma, str) or not lemma.strip() or len(lemma) > 256:
        raise ScopeViolationError("lemma KWIC requires a non-empty lemma of at most 256 characters")
    if type(window) is not int or not 0 <= window <= _MAX_LEMMA_KWIC_WINDOW:
        raise ScopeViolationError("lemma KWIC window must be between 0 and 100")
    if (
        type(offset) is not int
        or type(limit) is not int
        or offset < 0
        or not 1 <= limit <= _MAX_LEMMA_KWIC_LIMIT
    ):
        raise ScopeViolationError("lemma KWIC offset must be non-negative and limit must be 1..1000")

    artifacts = _complete_compatible_parsed_scope_artifacts(project, scope_id, artifact_ids)
    coverage = _scope_parsed_coverage(project, artifacts)
    field_coverage = coverage["lemma"]
    revision_ids = project.scope_revisions(scope_id)
    paths = [
        str(verify_project_artifact(project.root, item["relative_path"], item["file_hash"]))
        for item in artifacts
    ]

    duckdb = _duckdb()
    connection = duckdb.connect()
    try:
        count_row = connection.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE lemma = ?",
            [paths, lemma],
        ).fetchone()
        total_count = int(count_row[0]) if count_row is not None else 0
        result_rows = connection.execute(
            "WITH tokens AS MATERIALIZED (SELECT * FROM read_parquet(?)), "
            "page AS (SELECT revision_id,paragraph_id,token_index,surface,lemma,"
            "char_start,char_end FROM tokens WHERE lemma = ? "
            "ORDER BY list_position(?, revision_id),char_start,token_index "
            "LIMIT ? OFFSET ?) "
            "SELECT p.revision_id,p.paragraph_id,p.token_index,p.surface,p.lemma,"
            "p.char_start,p.char_end,c.token_index,c.surface,c.char_start,c.char_end "
            "FROM page AS p LEFT JOIN tokens AS c "
            "ON c.revision_id=p.revision_id AND c.paragraph_id=p.paragraph_id "
            "AND c.token_index BETWEEN greatest(0,p.token_index-?) AND p.token_index+? "
            "ORDER BY list_position(?,p.revision_id),p.char_start,p.token_index,c.token_index",
            [paths, lemma, revision_ids, limit, offset, window, window, revision_ids],
        ).fetchall()
    finally:
        connection.close()

    # The SQL result contains one row per selected occurrence and its bounded
    # same-paragraph token neighborhood. Keep page occurrences in SQL order.
    occurrence_order: list[tuple[str, int]] = []
    occurrences: dict[tuple[str, int], dict[str, object]] = {}
    context_tokens: dict[tuple[str, int], list[tuple[int, str, int, int]]] = {}
    for result in result_rows:
        revision_id = str(result[0])
        paragraph_value = result[1]
        token_index_value = result[2]
        if paragraph_value is None or type(token_index_value) is not int:
            raise ArtifactIntegrityError("parsed token artifact has an invalid lemma occurrence")
        paragraph_id = str(paragraph_value)
        token_index = token_index_value
        key = (revision_id, token_index)
        if key not in occurrences:
            occurrence_order.append(key)
            occurrences[key] = {
                "revision_id": revision_id,
                "paragraph_id": paragraph_id,
                "token_index": token_index,
                "surface": str(result[3]),
                "lemma": str(result[4]),
                "start_offset": int(result[5]),
                "end_offset": int(result[6]),
            }
            context_tokens[key] = []
        elif occurrences[key]["paragraph_id"] != paragraph_id:
            raise ArtifactIntegrityError("parsed token artifact has duplicate token identities")

        context_index, context_surface = result[7], result[8]
        context_start, context_end = result[9], result[10]
        if context_index is None:
            raise ArtifactIntegrityError("parsed token artifact is missing lemma context tokens")
        if (
            type(context_index) is not int
            or type(context_start) is not int
            or type(context_end) is not int
            or not isinstance(context_surface, str)
        ):
            raise ArtifactIntegrityError("parsed token artifact has invalid context token offsets")
        context_tokens[key].append(
            (context_index, context_surface, context_start, context_end)
        )

    canonical_revision_id: str | None = None
    canonical_text = ""
    paragraph_ranges: dict[str, tuple[str, int, int, str | None]] = {}
    rows: list[LemmaKwicRow] = []
    for key in occurrence_order:
        occurrence = occurrences[key]
        revision_id = str(occurrence["revision_id"])
        paragraph_id = str(occurrence["paragraph_id"])
        raw_token_index = occurrence["token_index"]
        raw_start = occurrence["start_offset"]
        raw_end = occurrence["end_offset"]
        if (
            type(raw_token_index) is not int
            or type(raw_start) is not int
            or type(raw_end) is not int
        ):
            raise ArtifactIntegrityError("parsed token artifact has invalid lemma occurrence offsets")
        token_index = raw_token_index
        start = raw_start
        end = raw_end
        surface = str(occurrence["surface"])
        stored_lemma = str(occurrence["lemma"])
        if stored_lemma != lemma:
            raise ArtifactIntegrityError("parsed token artifact returned a non-matching lemma")

        if canonical_revision_id != revision_id:
            canonical_text = project.revision_text(revision_id)
            canonical_revision_id = revision_id
        paragraph = paragraph_ranges.get(paragraph_id)
        if paragraph is None:
            paragraph_row = project.db.execute(
                "SELECT revision_id,start_offset,end_offset,text FROM paragraphs WHERE id=?",
                (paragraph_id,),
            ).fetchone()
            if paragraph_row is None:
                raise ArtifactIntegrityError("parsed token artifact references an unknown paragraph")
            paragraph = (
                str(paragraph_row["revision_id"]),
                int(paragraph_row["start_offset"]),
                int(paragraph_row["end_offset"]),
                paragraph_row["text"],
            )
            paragraph_ranges[paragraph_id] = paragraph
        paragraph_revision, paragraph_start, paragraph_end, stored_paragraph_text = paragraph
        if (
            paragraph_revision != revision_id
            or not 0 <= paragraph_start <= start < end <= paragraph_end <= len(canonical_text)
            or canonical_text[start:end] != surface
            or (
                stored_paragraph_text is not None
                and canonical_text[paragraph_start:paragraph_end] != stored_paragraph_text
            )
        ):
            raise ArtifactIntegrityError("parsed token artifact offsets do not match canonical text")

        context = context_tokens[key]
        context.sort(key=lambda item: item[0])
        if len({item[0] for item in context}) != len(context):
            raise ArtifactIntegrityError("parsed token artifact has duplicate token identities")
        previous_end = -1
        for _, context_token_surface, context_token_start, context_token_end in context:
            if (
                not paragraph_start <= context_token_start < context_token_end <= paragraph_end
                or context_token_start < previous_end
                or canonical_text[context_token_start:context_token_end] != context_token_surface
            ):
                raise ArtifactIntegrityError("parsed token context does not match canonical text")
            previous_end = context_token_end

        target_tokens = [item for item in context if item[0] == token_index]
        if len(target_tokens) != 1 or (
            target_tokens[0][2], target_tokens[0][3], target_tokens[0][1]
        ) != (start, end, surface):
            raise ArtifactIntegrityError("lemma occurrence is absent from its parsed token context")
        preceding = [item for item in context if item[0] < token_index]
        following = [item for item in context if item[0] > token_index]
        if window == 0:
            left_start = start
            right_end = end
        else:
            left_start = preceding[0][2] if len(preceding) >= window else paragraph_start
            right_end = following[-1][3] if len(following) >= window else paragraph_end
        rows.append(
            {
                "revision_id": revision_id,
                "paragraph_id": paragraph_id,
                "start_offset": start,
                "end_offset": end,
                "left": canonical_text[left_start:start],
                "keyword": canonical_text[start:end],
                "lemma": stored_lemma,
                "right": canonical_text[end:right_end],
            }
        )

    return {
        "scope_id": scope_id,
        "artifact_ids": [str(item["id"]) for item in artifacts],
        "lemma": lemma,
        "field_coverage": field_coverage,
        "tokenization_coverage": coverage["tokenization"],
        "result_semantics": (
            "exact stored lemma annotations in the complete selected parsed-token artifacts; "
            "field coverage status is reported separately; zero hits do not establish raw-text "
            "or conceptual absence"
        ),
        "window": window,
        "offset": offset,
        "limit": limit,
        "total_count": total_count,
        "has_more": offset + limit < total_count,
        "rows": rows,
    }


def lemma_kwic_evidence(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    revision_id: str,
    lemma: str,
    start_offset: int,
    end_offset: int,
) -> LemmaKwicEvidenceResult:
    """Persist one exact parser-lemma occurrence as canonical textual evidence.

    The selected artifacts must completely cover the explicit snapshot and use
    one compatible parser/model/schema identity. This records one verified
    example only; it does not estimate prevalence or support an absence claim.
    """
    if not isinstance(lemma, str) or not lemma.strip() or len(lemma) > 256:
        raise EvidenceValidationError(
            "lemma KWIC evidence requires a non-empty lemma of at most 256 characters"
        )
    if (
        type(start_offset) is not int
        or type(end_offset) is not int
        or start_offset < 0
        or end_offset <= start_offset
    ):
        raise EvidenceValidationError("selected offsets are not a lemma KWIC hit in this scope")

    artifacts = _complete_compatible_parsed_scope_artifacts(project, scope_id, artifact_ids)
    coverage = _scope_parsed_coverage(project, artifacts)
    field_coverage = coverage["lemma"]
    scope_revisions = project.scope_revisions(scope_id)
    if revision_id not in scope_revisions:
        raise EvidenceValidationError("selected offsets are not a lemma KWIC hit in this scope")
    selected_artifacts = [item for item in artifacts if item["revision_id"] == revision_id]
    if len(selected_artifacts) != 1:
        raise EvidenceValidationError("selected revision has no unique parsed artifact in this scope")
    selected_artifact = selected_artifacts[0]
    canonical_text = project.revision_text(revision_id)
    if end_offset > len(canonical_text):
        raise EvidenceValidationError("selected offsets are not a lemma KWIC hit in this scope")

    paths = [
        str(verify_project_artifact(project.root, item["relative_path"], item["file_hash"]))
        for item in artifacts
    ]
    duckdb = _duckdb()
    connection = duckdb.connect()
    try:
        occurrences = connection.execute(
            "SELECT revision_id,paragraph_id,lemma,surface,char_start,char_end "
            "FROM read_parquet(?) WHERE revision_id=? AND lemma=? "
            "AND char_start=? AND char_end=?",
            [paths, revision_id, lemma, start_offset, end_offset],
        ).fetchall()
    finally:
        connection.close()
    if len(occurrences) != 1:
        raise EvidenceValidationError("selected offsets are not a lemma KWIC hit in this scope")

    token_revision, paragraph_id, stored_lemma, surface, token_start, token_end = occurrences[0]
    if (
        token_revision != revision_id
        or stored_lemma != lemma
        or type(token_start) is not int
        or type(token_end) is not int
        or token_start != start_offset
        or token_end != end_offset
    ):
        raise ArtifactIntegrityError("parsed token artifact returned an invalid lemma occurrence")
    if not isinstance(paragraph_id, str) or not isinstance(surface, str):
        raise ArtifactIntegrityError("parsed token artifact has invalid occurrence metadata")
    if canonical_text[start_offset:end_offset] != surface:
        raise ArtifactIntegrityError("parsed token artifact offsets do not match canonical text")

    paragraph = project.db.execute(
        "SELECT revision_id,start_offset,end_offset,text FROM paragraphs WHERE id=?",
        (paragraph_id,),
    ).fetchone()
    if paragraph is None:
        raise ArtifactIntegrityError("parsed token artifact references an unknown paragraph")
    paragraph_revision = str(paragraph["revision_id"])
    paragraph_start = int(paragraph["start_offset"])
    paragraph_end = int(paragraph["end_offset"])
    stored_paragraph_text = paragraph["text"]
    if (
        paragraph_revision != revision_id
        or not 0 <= paragraph_start <= start_offset < end_offset <= paragraph_end <= len(canonical_text)
        or (
            stored_paragraph_text is not None
            and canonical_text[paragraph_start:paragraph_end] != stored_paragraph_text
        )
    ):
        raise ArtifactIntegrityError("parsed token artifact offsets do not match canonical paragraph")

    backend = selected_artifact["backend"]
    backend_version = selected_artifact["backend_version"]
    model = selected_artifact["model"]
    model_version = selected_artifact["model_version"]
    schema_version = selected_artifact["schema_version"]
    if (
        not isinstance(backend, str)
        or not backend
        or not isinstance(model, str)
        or not model
        or not isinstance(schema_version, str)
        or not schema_version
        or (backend_version is not None and not isinstance(backend_version, str))
        or (model_version is not None and not isinstance(model_version, str))
    ):
        raise ArtifactIntegrityError("parsed token artifact has invalid parser provenance")

    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXAMPLE_BASED,
            scope_snapshot_id=scope_id,
            population_definition=(
                "one caller-selected parser-annotated lemma occurrence in a complete parsed-token "
                "artifact set for the immutable ScopeSnapshot"
            ),
            selection_policy=(
                "exact revision, stored lemma, and canonical half-open offsets selected by caller"
            ),
            inspection_count=1,
            completion_criterion=(
                "the selected parsed token exactly matched its canonical revision text slice"
            ),
            termination_reason="selected example verified",
            termination_state=CoverageTerminationState.COMPLETED,
        )
    )
    selected_artifact_ids = [str(item["id"]) for item in artifacts]
    span_id = project.create_span(revision_id, start_offset, end_offset)
    evidence_id = project.textual_evidence(
        span_id,
        provenance={
            "acquisition_method": "corpus_lemma_kwic",
            "calculation_version": "parsed-lemma-kwic-evidence-v1",
            "scope_snapshot_id": scope_id,
            "artifact_ids": selected_artifact_ids,
            "coverage_descriptor_id": coverage_id,
            "coverage_mode": CoverageMode.EXAMPLE_BASED.value,
            "revision_id": revision_id,
            "lemma": lemma,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "backend": backend,
            "backend_version": backend_version,
            "model": model,
            "model_version": model_version,
            "schema_version": schema_version,
            "result_semantics": (
                "one parser-annotated example; this evidence does not establish corpus "
                "prevalence or absence"
            ),
            "field_coverage": field_coverage,
            "tokenization_coverage": coverage["tokenization"],
        },
    )
    exact_text = project.span_text(span_id)
    return {
        "scope_id": scope_id,
        "artifact_ids": selected_artifact_ids,
        "revision_id": revision_id,
        "lemma": lemma,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "exact_text": exact_text,
        "coverage_id": coverage_id,
        "coverage_mode": CoverageMode.EXAMPLE_BASED.value,
        "backend": backend,
        "backend_version": backend_version,
        "model": model,
        "model_version": model_version,
        "schema_version": schema_version,
        "field_coverage": field_coverage,
        "tokenization_coverage": coverage["tokenization"],
        "evidence_id": evidence_id,
        "citation": project.render_citation(evidence_id),
    }


def _complete_compatible_parsed_scope_artifacts(
    project: Project, scope_id: str, artifact_ids: list[str]
):
    artifacts = _complete_scope_artifacts(project, scope_id, artifact_ids, "parsed_tokens")
    _require_compatible_parsed_artifacts(artifacts)
    return artifacts


def _require_compatible_surface_artifacts(artifacts) -> None:
    """Reject surface artifacts whose registered schema is not understood here."""
    if any(item["schema_version"] != SURFACE_TOKEN_SCHEMA_VERSION for item in artifacts):
        raise ScopeViolationError(
            f"surface token artifacts require {SURFACE_TOKEN_SCHEMA_VERSION}; rebuild "
            f"legacy {LEGACY_SURFACE_TOKEN_SCHEMA_VERSION} artifacts"
        )


def _require_compatible_parsed_artifacts(artifacts) -> None:
    """Require every parsed artifact to share the same declared parser identity."""
    compatibility_fields = (
        "backend",
        "backend_version",
        "model",
        "model_version",
        "schema_version",
    )
    signatures = {tuple(item[field] for field in compatibility_fields) for item in artifacts}
    if (
        any(
            item["schema_version"]
            not in {LEGACY_PARSED_TOKEN_SCHEMA_VERSION, PARSED_TOKEN_SCHEMA_VERSION}
            for item in artifacts
        )
        or len(signatures) != 1
        or any(
            not isinstance(item["backend"], str)
            or not item["backend"].strip()
            or not isinstance(item["model"], str)
            or not item["model"].strip()
            or any(
                value is not None and (not isinstance(value, str) or not value.strip())
                for value in (item["backend_version"], item["model_version"])
            )
            for item in artifacts
        )
    ):
        raise ScopeViolationError(
            "complete parsed-token artifacts require one compatible parser and model"
        )


def _validate_parsed_coverage_entry(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactIntegrityError("parsed token artifact has invalid coverage metadata")
    status = value.get("status")
    if not isinstance(status, str) or status not in {
        OPERATIONAL_SUPPORTED,
        OPERATIONAL_LIMITED,
    }:
        raise ArtifactIntegrityError("parsed token artifact has invalid coverage metadata")
    for count_name in ("annotated_tokens", "total_tokens"):
        if count_name in value and (
            type(value[count_name]) is not int or value[count_name] < 0
        ):
            raise ArtifactIntegrityError("parsed token artifact has invalid coverage metadata")
    if (
        "annotated_tokens" in value
        and "total_tokens" in value
        and value["annotated_tokens"] > value["total_tokens"]
    ):
        raise ArtifactIntegrityError("parsed token artifact has invalid coverage metadata")
    if field == "tokenization":
        for count_name in (
            "covered_non_whitespace_characters",
            "source_non_whitespace_characters",
        ):
            if type(value.get(count_name)) is not int or value[count_name] < 0:
                raise ArtifactIntegrityError("parsed token artifact has invalid coverage metadata")
        if value["covered_non_whitespace_characters"] > value["source_non_whitespace_characters"]:
            raise ArtifactIntegrityError("parsed token artifact has invalid coverage metadata")
    limitation = value.get("limitation")
    if limitation is not None and not isinstance(limitation, str):
        raise ArtifactIntegrityError("parsed token artifact has invalid coverage metadata")
    return dict(value)


def _parsed_artifact_coverage(project: Project, artifact) -> dict[str, dict[str, object]]:
    """Read verified artifact coverage, conservatively downgrading legacy formats."""
    if artifact["schema_version"] == LEGACY_PARSED_TOKEN_SCHEMA_VERSION:
        return {
            field: {
                "status": OPERATIONAL_LIMITED,
                "annotated_tokens": None,
                "total_tokens": None,
                "limitation": (
                    "Legacy parsed-token artifacts predate field-level coverage metadata."
                ),
            }
            for field in ("lemma", "pos", "dependency_parse")
        } | {
            "tokenization": {
                "status": OPERATIONAL_LIMITED,
                "covered_non_whitespace_characters": None,
                "source_non_whitespace_characters": None,
                "total_tokens": None,
                "limitation": (
                    "Legacy parsed-token artifacts predate source token-coverage metadata."
                ),
            }
        }

    path = verify_project_artifact(
        project.root, artifact["relative_path"], artifact["file_hash"]
    )
    _, pq = _pyarrow()
    raw_metadata = pq.read_metadata(path).metadata or {}
    raw_coverage = raw_metadata.get(_PARSED_TOKEN_COVERAGE_METADATA_KEY.encode())
    if not isinstance(raw_coverage, bytes):
        raise ArtifactIntegrityError("parsed token artifact is missing coverage metadata")
    try:
        payload = json.loads(raw_coverage.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError("parsed token artifact has invalid coverage metadata") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != _PARSED_TOKEN_COVERAGE_FORMAT_VERSION
        or type(payload.get("format_version")) is not int
    ):
        raise ArtifactIntegrityError("parsed token artifact has invalid coverage metadata")
    fields = payload.get("fields")
    if fields is None:
        # Coverage fields are top-level in format version 1.
        fields = payload
    if not isinstance(fields, dict):
        raise ArtifactIntegrityError("parsed token artifact has invalid coverage metadata")
    return {
        field: _validate_parsed_coverage_entry(fields.get(field), field=field)
        for field in ("tokenization", "lemma", "pos", "dependency_parse")
    }


def _aggregate_parsed_coverage(
    project: Project, artifacts, field: str
) -> dict[str, object]:
    entries = [_parsed_artifact_coverage(project, artifact)[field] for artifact in artifacts]
    return _combine_parsed_coverage_entries(entries, field)


def _combine_parsed_coverage_entries(
    entries: list[dict[str, object]], field: str
) -> dict[str, object]:
    status = (
        OPERATIONAL_SUPPORTED
        if entries and all(item["status"] == OPERATIONAL_SUPPORTED for item in entries)
        else OPERATIONAL_LIMITED
    )
    result: dict[str, object] = {"status": status}
    count_names = (
        ("covered_non_whitespace_characters", "source_non_whitespace_characters", "total_tokens")
        if field == "tokenization"
        else ("annotated_tokens", "total_tokens")
    )
    for name in count_names:
        values = [item.get(name) for item in entries]
        result[name] = (
            sum(value for value in values if type(value) is int)
            if all(type(value) is int for value in values)
            else None
        )
    limitations: list[str] = []
    for item in entries:
        limitation = item.get("limitation")
        if isinstance(limitation, str) and limitation not in limitations:
            limitations.append(limitation)
    result["limitation"] = "; ".join(limitations) if limitations else None
    return result


def _scope_parsed_coverage(project: Project, artifacts) -> dict[str, dict[str, object]]:
    per_artifact = [_parsed_artifact_coverage(project, artifact) for artifact in artifacts]
    return {
        field: _combine_parsed_coverage_entries(
            [coverage[field] for coverage in per_artifact], field
        )
        for field in ("tokenization", "lemma", "pos", "dependency_parse")
    }


def artifact_frequency_statistic_evidence(
    project: Project,
    scope_id: str,
    artifact_ids: list[str],
    *,
    field: str,
    value: str,
) -> dict[str, object]:
    """Persist a reproducible token-artifact statistic with annotation coverage.

    This deliberately accepts an explicit *complete* artifact set only: exactly
    one compatible derived artifact for every revision in the immutable scope.
    That prevents a partial NLP build from being described as a whole-corpus
    lemma, POS, or dependency count.
    """
    field_spec = {
        "normalized": ("surface_tokens", "normalized", SURFACE_TOKEN_SCHEMA_VERSION),
        "lemma": ("parsed_tokens", "lemma", None),
        "pos": ("parsed_tokens", "pos", None),
        "dependency_relation": ("parsed_tokens", "dependency_relation", None),
    }
    if field not in field_spec or not value.strip():
        raise ScopeViolationError("unsupported artifact statistic field or empty value")
    artifact_type, column, schema_version = field_spec[field]
    if artifact_type == "parsed_tokens":
        artifacts = _complete_compatible_parsed_scope_artifacts(
            project, scope_id, artifact_ids
        )
        coverage = _scope_parsed_coverage(project, artifacts)
        field_coverage = coverage[
            "dependency_parse" if field == "dependency_relation" else field
        ]
        schema_version = str(artifacts[0]["schema_version"])
    else:
        artifacts = _complete_scope_artifacts(
            project, scope_id, artifact_ids, artifact_type
        )
        _require_compatible_surface_artifacts(artifacts)
        coverage = None
        field_coverage = None
    query_value = _surface_form_v2(value) if field == "normalized" else value
    duckdb = _duckdb()
    paths = [
        str(verify_project_artifact(project.root, item["relative_path"], item["file_hash"]))
        for item in artifacts
    ]
    connection = duckdb.connect()
    try:
        row = connection.execute(
            f"SELECT count(NULLIF({column}, '')), "
            f"count(*) FILTER (WHERE {column} = ?) FROM read_parquet(?)",
            [query_value, paths],
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ScopeViolationError("artifact statistics query returned no aggregate row")
    population_count, count = cast(tuple[int, int], row)
    population_definition = (
        f"all non-null, non-empty {field} values in one {artifact_type} artifact per revision "
        "of the immutable ScopeSnapshot"
    )
    is_complete_field_population = (
        field_coverage is None or field_coverage["status"] == OPERATIONAL_SUPPORTED
    )
    coverage_mode = (
        CoverageMode.EXHAUSTIVE if is_complete_field_population else CoverageMode.BROAD
    )
    if not is_complete_field_population:
        population_definition += (
            "; parser annotation coverage is LIMITED and counts are over available annotations only"
        )
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=coverage_mode,
            scope_snapshot_id=scope_id,
            population_definition=population_definition,
            population_count=int(population_count),
            selection_policy=(
                f"exact deterministic {field} enumeration for {query_value!r} in every selected scope artifact"
            ),
            inspection_count=int(population_count),
            completion_criterion=(
                "all selected, complete scope artifacts read through DuckDB and enumerated"
            ),
            termination_reason="completed deterministic artifact enumeration",
            negative_result_status=(
                NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION
                if int(count) == 0 and is_complete_field_population
                else NegativeResultStatus.NOT_APPLICABLE
            ),
        )
    )
    calculation_version = (
        "derived-surface-token-frequency-v2"
        if field == "normalized"
        else "derived-token-artifact-frequency-v2"
    )
    provenance = {
        "calculation_version": calculation_version,
        "artifact_ids": [item["id"] for item in artifacts],
        "artifact_type": artifact_type,
        "artifact_schema_version": schema_version,
        "field": field,
        "query_value": query_value,
        "query_engine": "duckdb",
        "coverage_id": coverage_id,
        "population_count": int(population_count),
    }
    if field_coverage is not None and coverage is not None:
        provenance["field_coverage"] = field_coverage
        provenance["tokenization_coverage"] = coverage["tokenization"]
    if field == "normalized":
        provenance.update(
            {
                "tokenizer": SURFACE_TOKENIZER_V2,
                "normalization": SURFACE_NORMALIZATION_V2,
                "unicode_data_version": unicodedata.unidata_version,
            }
        )
    if artifact_type == "parsed_tokens":
        parser_identity = artifacts[0]
        provenance.update(
            {
                "backend": parser_identity["backend"],
                "backend_version": parser_identity["backend_version"],
                "model": parser_identity["model"],
                "model_version": parser_identity["model_version"],
                "schema_version": parser_identity["schema_version"],
            }
        )
    evidence_id = project.corpus_statistic_evidence(
        f"{field} frequency for {query_value!r}",
        int(count),
        scope_snapshot_id=scope_id,
        population_definition=population_definition,
        provenance=provenance,
    )
    result: dict[str, object] = {
        "field": field,
        "value": query_value,
        "count": int(count),
        "population_count": int(population_count),
        "scope_id": scope_id,
        "artifact_ids": provenance["artifact_ids"],
        "coverage_id": coverage_id,
        "evidence_id": evidence_id,
    }
    if field_coverage is not None and coverage is not None:
        result["field_coverage"] = field_coverage
        result["tokenization_coverage"] = coverage["tokenization"]
    return result


def lemma_dispersion(project: Project, scope_id: str, artifact_ids: list[str]) -> dict[str, int]:
    """Return number of immutable revisions containing each parser-derived lemma."""
    artifacts = _scoped_artifacts(project, scope_id, artifact_ids, "parsed_tokens")
    return _query_counts(
        project,
        artifacts,
        "SELECT lemma,count(DISTINCT revision_id) FROM read_parquet(?) "
        "WHERE lemma IS NOT NULL AND lemma <> '' GROUP BY lemma ORDER BY lemma",
    )


def _column_frequency(
    project: Project, scope_id: str, artifact_ids: list[str], column: str
) -> Counter[str]:
    artifacts = _scoped_artifacts(project, scope_id, artifact_ids, "parsed_tokens")
    if column not in {"lemma", "pos", "dependency_relation"}:
        raise ScopeViolationError("unsupported parsed-token column")
    return Counter(
        _query_counts(
            project,
            artifacts,
            f"SELECT {column},count(*) FROM read_parquet(?) "
            f"WHERE {column} IS NOT NULL AND {column} <> '' "
            f"GROUP BY {column} ORDER BY {column}",
        )
    )


def _query_counts(project: Project, artifacts, query: str) -> dict[str, int]:
    duckdb = _duckdb()
    paths = [
        str(verify_project_artifact(project.root, item["relative_path"], item["file_hash"]))
        for item in artifacts
    ]
    connection = duckdb.connect()
    try:
        rows = connection.execute(query, [paths]).fetchall()
    finally:
        connection.close()
    return {str(value): int(count) for value, count in rows}


def _scoped_artifacts(
    project: Project, scope_id: str, artifact_ids: list[str], expected_type: str | None = None
):
    """Resolve derived artifacts only after checking their canonical revision scope."""
    if not artifact_ids:
        raise ScopeViolationError("token analytics requires explicit artifact IDs")
    scope_revisions = set(project.scope_revisions(scope_id))
    placeholders = ",".join("?" * len(artifact_ids))
    artifacts = project.db.execute(
        f"SELECT id,revision_id,relative_path,artifact_type,file_hash,backend,backend_version,"
        f"model,model_version,schema_version FROM artifact_registry WHERE id IN ({placeholders})",
        artifact_ids,
    ).fetchall()
    if (
        len(artifacts) != len(set(artifact_ids))
        or any(item["revision_id"] not in scope_revisions for item in artifacts)
        or (
            expected_type is not None
            and any(item["artifact_type"] != expected_type for item in artifacts)
        )
    ):
        raise ScopeViolationError("artifact set is unknown or outside the explicit scope")
    # Token artifact builders produce a complete derived view of one revision,
    # not disjoint shards. Combining two versions for the same revision would
    # silently double-count tokens (or mix incompatible parser outputs).
    revision_ids = [str(item["revision_id"]) for item in artifacts]
    if len(revision_ids) != len(set(revision_ids)):
        raise ScopeViolationError(
            "select at most one token artifact per document revision; artifact versions cannot be combined"
        )
    for artifact in artifacts:
        verify_project_artifact(project.root, artifact["relative_path"], artifact["file_hash"])
    return artifacts


def _complete_scope_artifacts(
    project: Project, scope_id: str, artifact_ids: list[str], expected_type: str
):
    """Require a non-duplicated, one-to-one revision/artifact population."""
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ScopeViolationError("artifact IDs must not be duplicated")
    artifacts = _scoped_artifacts(project, scope_id, artifact_ids, expected_type)
    scope_revisions = set(project.scope_revisions(scope_id))
    by_revision = {item["revision_id"] for item in artifacts}
    if by_revision != scope_revisions or len(artifacts) != len(scope_revisions):
        raise ScopeViolationError(
            "exhaustive artifact statistics require one compatible artifact per scoped revision"
        )
    return sorted(artifacts, key=lambda item: str(item["id"]))
