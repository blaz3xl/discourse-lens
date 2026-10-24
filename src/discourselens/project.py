from __future__ import annotations

import base64
import hashlib
import heapq
import hmac
import json
import logging
import math
import os
import re
import sqlite3
import stat
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path, PureWindowsPath
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Self, overload

from .config import ProjectConfig, resolve_project_config, write_project_config
from .errors import (
    DocumentError,
    EvidenceValidationError,
    ModelProviderError,
    ProjectError,
    ScopeViolationError,
    SourceSpanError,
)
from .evidence_limits import (
    MAX_CONTEXT_EVIDENCE_CHARS,
    MAX_CONTEXT_LABEL_CHARS,
    MAX_EXTERNAL_SOURCE_TIMESTAMP_CHARS,
    MAX_EXTERNAL_SOURCE_URL_CHARS,
)
from .migrations import SCHEMA_VERSION, migrate
from .models import (
    CredentialReference,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelPreset,
    ModelPrice,
    TokenUsage,
    decimal_text,
    estimate_model_cost,
)
from .observability import log_event
from .policy import claim_allowed, is_scope_revision_population_definition
from .reasoning_policy import validate_cohere_manual_budget
from .revision_provenance import basic_extraction_manifest, canonical_json
from .secrets import (
    _validate_provider_endpoint_metadata,
    has_credential_like_endpoint_host,
    provider_auth_reference,
    validate_credential_alias,
)
from .types import (
    ClaimType,
    CoverageDescriptor,
    CoverageMode,
    CoverageTerminationState,
    EvidenceKind,
    EvidenceRole,
    NegativeResultStatus,
    RunStatus,
)
from .versioning import software_version

_LOGGER = logging.getLogger(__name__)
_SQLITE_IN_BATCH_SIZE = 900
MAX_LEXICAL_SEARCH_CANDIDATES = 5_000
MAX_LEXICAL_SEARCH_OFFSET = MAX_LEXICAL_SEARCH_CANDIDATES - 1
MAX_LEXICAL_SEARCH_LIMIT = 100
MAX_SCOPE_REVISION_PAGE_SIZE = 100
MAX_RESEARCH_MEMO_PAGE_SIZE = 100
MAX_CODEBOOK_PAGE_SIZE = 100
MAX_RESEARCH_MEMO_TITLE_CHARS = 512
MAX_RESEARCH_MEMO_TEXT_CHARS = 4_000
_DEFAULT_CHAT_MODEL_PRESET_KEY = "default_chat_model_preset"
_CHAT_REASONING_PREFERENCES_KEY = "chat_reasoning_preferences"
_SUPPORTS_DIR_FD_OPEN = os.open in os.supports_dir_fd
_SUPPORTS_DIR_FD_DIRECTORY_CREATION = (
    _SUPPORTS_DIR_FD_OPEN and os.mkdir in os.supports_dir_fd and os.stat in os.supports_dir_fd
)


def _chat_title_from_prompt(prompt: str) -> str:
    """Create a short, readable chat title from its opening question."""
    title = " ".join(prompt.split()).strip(" \t\r\n-—")
    if not title:
        return "Conversación"
    if len(title) > 88:
        shortened = title[:85].rsplit(" ", 1)[0].rstrip(" .,;:!?—-")
        title = f"{shortened or title[:85].rstrip()}…"
    return title


if TYPE_CHECKING:
    from .analysis import (
        AnalysisResult,
        AskQuestionResult,
        ControlledCdaResult,
        CounterEvidenceAnalysisResult,
        EvidenceExtractionAnalysisResult,
        QuoteAttributionAnalysisResult,
        SavedProfileBatchItem,
        ScopeQuestionAnalysisResult,
        TextualDescriptorAnalysisResult,
        VanDijkGroupComparisonAnalysisResult,
    )
    from .corpus import (
        CollocationAnalysisResult,
        CollocationEvidenceResult,
        CollocationsPage,
        ComparisonPage,
        DocumentDispersionEvidenceResult,
        DocumentDispersionPage,
        DocumentDispersionResult,
        FrequencyPage,
        KwicEvidenceResult,
        KwicPage,
        KwicRow,
        LexicalBundleEvidenceResult,
        LexicalBundlesResult,
        NgramPage,
    )
    from .documents import BatchImportResult, ImportResult
    from .execution import PresetExecutionResult, PresetRequestPreview
    from .interpretation import InterpretationResult
    from .languages import LanguageCapability
    from .models import EmbeddingProvider, GenerationImage
    from .nlp import NlpBackend
    from .nlp_artifacts import (
        DependencyPatternsResult,
        LemmaKwicEvidenceResult,
        LemmaKwicPage,
        ParsedFrequencyPage,
        SurfaceFrequencyPage,
        TokenArtifact,
    )
    from .skills import VanDijkGroupComparisonGroupInput
    from .validation import FindingValidation
    from .vectors import (
        HybridHit,
        ParagraphEmbeddingRequestPreview,
        RetrievalSearchResult,
        VectorHit,
    )


def uid() -> str:
    uuid7 = getattr(uuid, "uuid7", None)
    return str(uuid7()) if callable(uuid7) else str(uuid.uuid4())


def _in_parameter_batches(values: list[str]) -> Iterator[list[str]]:
    """Keep variable-length IN queries below SQLite's portable parameter limit."""
    for offset in range(0, len(values), _SQLITE_IN_BATCH_SIZE):
        yield values[offset : offset + _SQLITE_IN_BATCH_SIZE]


def _path_safe_display_name(value: object) -> str | None:
    """Return an import-label basename without either platform's path components."""
    if not isinstance(value, str) or "\x00" in value:
        return None
    basename = PureWindowsPath(value).name
    if not basename.strip() or basename in {".", ".."}:
        return None
    return basename


def _analysis_profile_with_defaults(payload: dict[str, object]) -> dict[str, object]:
    """Materialize the profile metadata defaults used by the Pydantic contract."""
    from .profiles import ControlledCdaProfile

    defaults = ControlledCdaProfile.model_construct(frameworks=[])
    normalized = dict(payload)
    normalized.setdefault("version", defaults.version)
    normalized.setdefault("kind", defaults.kind)
    return normalized


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _normalize_secret_config_key(key: object) -> str:
    """Normalize snake, kebab, camel, and Pascal-case config key spellings."""
    text = str(key)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _contains_secret_config_key(
    value: object, *, allow_counterevidence_query_labels: bool = False
) -> bool:
    """Reject secret-bearing configuration before it reaches run history or exports.

    Credential aliases and model names are reproducibility metadata; secret
    references and values are not.  This is intentionally key-based so benign
    analytical settings such as ``tokenizer`` remain valid.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalize_secret_config_key(key)
            if normalized in {"credential_alias", "reranking_credential_alias"}:
                if key not in {"credential_alias", "reranking_credential_alias"}:
                    return True
                if child is not None:
                    try:
                        validate_credential_alias(child)
                    except ProjectError:
                        return True
                continue
            if normalized == "credential_reference_hash":
                if key != "credential_reference_hash":
                    return True
                if child is None:
                    continue
                if not isinstance(child, str) or re.fullmatch(r"[0-9a-fA-F]{64}", child) is None:
                    return True
                continue
            if normalized in {"endpoint", "provider_endpoint"} and (
                has_credential_like_endpoint_host(child)
            ):
                return True
            if (
                allow_counterevidence_query_labels
                and normalized == "counterevidence_queries"
                and isinstance(child, dict)
            ):
                if any(
                    _contains_secret_config_key(item, allow_counterevidence_query_labels=True)
                    for item in child.values()
                ):
                    return True
                continue
            if normalized.replace("_", "").endswith(
                ("apikey", "apitoken", "clientsecret", "privatekey", "sessiontoken")
            ):
                return True
            parts = set(normalized.split("_"))
            if parts.intersection(
                {"password", "secret", "credential", "credentials", "authorization"}
            ):
                return True
            if "key" in parts and parts.intersection(
                {
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
            ):
                return True
            if "token" in parts:
                if parts.intersection(
                    {
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
                ):
                    return True
                if (
                    normalized
                    not in {
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
                    and "count" not in parts
                    and "counts" not in parts
                    and "tokens" not in parts
                ):
                    return True
            if _contains_secret_config_key(
                child, allow_counterevidence_query_labels=allow_counterevidence_query_labels
            ):
                return True
    elif isinstance(value, (list, tuple)):
        return any(
            _contains_secret_config_key(
                item, allow_counterevidence_query_labels=allow_counterevidence_query_labels
            )
            for item in value
        )
    return False


def _validate_stored_credential_alias(value: object, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    try:
        validate_credential_alias(value)
    except ProjectError:
        raise DocumentError("stored credential metadata is unsafe") from None


def _validated_credential_ledger_rows(
    rows: Iterable[sqlite3.Row], *, allow_none: bool = False
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in rows:
        record = dict(row)
        if "credential_alias" in record:
            _validate_stored_credential_alias(record["credential_alias"], allow_none=allow_none)
        records.append(record)
    return records


def _validated_model_execution_rows(rows: Iterable[sqlite3.Row]) -> list[dict[str, object]]:
    """Return model ledger rows only when serialized metadata is secret-free."""
    records = _validated_credential_ledger_rows(rows)
    for record in records:
        endpoint = record.get("endpoint")
        if endpoint is not None:
            if not isinstance(endpoint, str):
                raise DocumentError("stored model execution metadata is unsafe")
            try:
                _validate_provider_endpoint_metadata(endpoint)
            except ProjectError:
                raise DocumentError("stored model execution metadata is unsafe") from None
        for field in ("parameters_json", "capabilities_json", "model_price_json"):
            raw = record.get(field)
            if raw is None and field == "model_price_json":
                continue
            if not isinstance(raw, str):
                raise DocumentError("stored model execution metadata is unsafe")
            try:
                metadata = json.loads(raw)
            except (TypeError, ValueError, RecursionError):
                raise DocumentError("stored model execution metadata is unsafe") from None
            if not isinstance(metadata, dict) or _contains_secret_config_key(metadata):
                raise DocumentError("stored model execution metadata is unsafe")
            if field == "model_price_json":
                try:
                    ModelPrice.from_snapshot(metadata)
                except (ProjectError, TypeError, ValueError, RecursionError):
                    raise DocumentError("stored model execution metadata is unsafe") from None
    return records


def _validate_model_preset_metadata(row: sqlite3.Row) -> None:
    _validate_stored_credential_alias(row["credential_alias"])
    try:
        parameters = json.loads(row["parameters_json"])
    except (TypeError, ValueError):
        raise DocumentError("stored model preset metadata is unsafe") from None
    if not isinstance(parameters, dict) or _contains_secret_config_key(parameters):
        raise DocumentError("stored model preset metadata is unsafe")


_MANAGED_STORAGE_PATHS = (
    ".discourselens",
    ".discourselens/project.sqlite",
    ".discourselens/source",
    ".discourselens/artifacts",
    ".discourselens/artifacts/extraction",
    ".discourselens/artifacts/nlp",
    ".discourselens/artifacts/chunks",
    ".discourselens/artifacts/vectors",
    ".discourselens/artifacts/analytics",
    ".discourselens/manifests",
)


def _validate_managed_storage_paths(root: Path) -> None:
    """Reject symlinked project storage roots and managed artifact directories."""
    canonical_root = root.resolve()
    try:
        for relative in _MANAGED_STORAGE_PATHS:
            path = canonical_root / relative
            resolved = path.resolve()
            if path.is_symlink() or not resolved.is_relative_to(canonical_root):
                raise ProjectError(
                    f"managed project storage path must remain inside the project: {relative}"
                )
            if path.is_file() and path.stat().st_nlink > 1:
                raise ProjectError(
                    f"managed project storage file must not be hard-linked: {relative}"
                )
    except (OSError, RuntimeError) as exc:
        raise ProjectError("managed project storage is unavailable or unsafe") from exc


def _opened_database_identity(
    connection: sqlite3.Connection,
) -> tuple[int, int, int]:
    """Return the identity of SQLite's reported main database file.

    SQLite reports the VFS ``xFullPathname`` result here. This detects transient
    redirects when that VFS canonicalizes links; VFSes that only absolutize paths
    do not expose the opened OS file identity through Python's sqlite3 API.
    """
    try:
        databases = connection.execute("PRAGMA database_list").fetchall()
        main_database = next(row for row in databases if row[1] == "main")
        filename = main_database[2]
        if not filename:
            raise ProjectError("managed project storage changed during database open")
        file_stat = Path(filename).stat()
    except (OSError, StopIteration, sqlite3.Error, ValueError) as exc:
        raise ProjectError("managed project storage changed during database open") from exc
    return file_stat.st_dev, file_stat.st_ino, file_stat.st_nlink


def _project_root_identity(root: Path) -> tuple[int, int]:
    """Return the filesystem identity of an opened project root directory."""
    try:
        root_stat = root.stat()
    except OSError as exc:
        raise ProjectError("project root is unavailable or unsafe") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ProjectError("project root is unavailable or unsafe")
    return root_stat.st_dev, root_stat.st_ino


def _create_project_directories(root: Path) -> None:
    """Create project directories without following swapped managed-path symlinks."""
    directories = (
        "corpus",
        "metadata",
        "analyses",
        "reports",
        ".discourselens/source",
        ".discourselens/artifacts/extraction",
        ".discourselens/artifacts/nlp",
        ".discourselens/artifacts/chunks",
        ".discourselens/artifacts/vectors",
        ".discourselens/artifacts/analytics",
        ".discourselens/manifests",
    )
    if not _SUPPORTS_DIR_FD_DIRECTORY_CREATION:
        for relative in directories:
            _validate_managed_storage_paths(root)
            (root / relative).mkdir(parents=True, exist_ok=True)
            _validate_managed_storage_paths(root)
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root, flags)
    except OSError as exc:
        raise ProjectError("project root is unavailable or unsafe") from exc
    try:
        root_entry = os.stat(root, follow_symlinks=False)
        opened_root = os.fstat(root_fd)
        if not stat.S_ISDIR(root_entry.st_mode) or (root_entry.st_dev, root_entry.st_ino) != (
            opened_root.st_dev,
            opened_root.st_ino,
        ):
            raise ProjectError("project root changed during initialization")

        for relative in directories:
            parent_fd = os.dup(root_fd)
            try:
                for component in relative.split("/"):
                    try:
                        os.mkdir(component, dir_fd=parent_fd)
                    except FileExistsError:
                        pass
                    entry = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                    if not stat.S_ISDIR(entry.st_mode):
                        raise ProjectError("managed project storage path is unsafe")
                    child_fd = os.open(component, flags, dir_fd=parent_fd)
                    opened = os.fstat(child_fd)
                    current_entry = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                    if not stat.S_ISDIR(current_entry.st_mode) or (
                        current_entry.st_dev,
                        current_entry.st_ino,
                    ) != (opened.st_dev, opened.st_ino):
                        os.close(child_fd)
                        raise ProjectError("managed project storage changed during initialization")
                    os.close(parent_fd)
                    parent_fd = child_fd
            finally:
                os.close(parent_fd)
    except ProjectError:
        raise
    except OSError as exc:
        raise ProjectError("managed project storage is unavailable or unsafe") from exc
    finally:
        os.close(root_fd)


def _create_project_database_file(root: Path) -> None:
    """Reserve the SQLite file before opening it, pinning the directory when possible.

    Platforms without directory-descriptor support use exclusive path creation after
    validation; that fallback cannot pin the parent directory against a concurrent swap.
    """
    database_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    if not _SUPPORTS_DIR_FD_DIRECTORY_CREATION:
        _validate_managed_storage_paths(root)
        try:
            descriptor = os.open(root / ".discourselens" / "project.sqlite", database_flags, 0o600)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ProjectError("managed project storage is unavailable or unsafe") from exc
        else:
            os.close(descriptor)
        _validate_managed_storage_paths(root)
        return

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_fd = -1
    hidden_fd = -1
    try:
        root_fd = os.open(root, directory_flags)
        root_entry = os.stat(root, follow_symlinks=False)
        opened_root = os.fstat(root_fd)
        if not stat.S_ISDIR(root_entry.st_mode) or (root_entry.st_dev, root_entry.st_ino) != (
            opened_root.st_dev,
            opened_root.st_ino,
        ):
            raise ProjectError("project root changed during database creation")

        hidden_entry = os.stat(".discourselens", dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(hidden_entry.st_mode):
            raise ProjectError("managed project storage path is unsafe")
        hidden_fd = os.open(".discourselens", directory_flags, dir_fd=root_fd)
        opened_hidden = os.fstat(hidden_fd)
        if (hidden_entry.st_dev, hidden_entry.st_ino) != (
            opened_hidden.st_dev,
            opened_hidden.st_ino,
        ):
            raise ProjectError("managed project storage changed during database creation")

        try:
            descriptor = os.open("project.sqlite", database_flags, 0o600, dir_fd=hidden_fd)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
    except ProjectError:
        raise
    except OSError as exc:
        raise ProjectError("managed project storage is unavailable or unsafe") from exc
    finally:
        if hidden_fd >= 0:
            os.close(hidden_fd)
        if root_fd >= 0:
            os.close(root_fd)
    _validate_managed_storage_paths(root)


class Project:
    uid = staticmethod(uid)

    def __init__(self, root: Path, *, read_only: bool = False, create_database: bool = True):
        self.root = root.resolve()
        self._root_identity = _project_root_identity(self.root)
        self.read_only = read_only
        if not read_only:
            # Older or partially initialized workspaces may have a database but
            # be missing an empty managed directory such as ``source``. Restore
            # the required directory layout before opening the writable project.
            _create_project_directories(self.root)
        self.hidden = self.root / ".discourselens"
        database_path = self.hidden / "project.sqlite"
        try:
            database_before_open = database_path.stat(follow_symlinks=False)
        except FileNotFoundError:
            database_before_open = None
        except OSError as exc:
            raise ProjectError("managed project storage is unavailable or unsafe") from exc
        _validate_managed_storage_paths(self.root)
        try:
            if read_only:
                if not database_path.is_file():
                    raise ProjectError("read-only project database is unavailable")
                self.db = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)
            elif create_database:
                self.db = sqlite3.connect(database_path)
            else:
                self.db = sqlite3.connect(f"{database_path.as_uri()}?mode=rw", uri=True)
        except sqlite3.Error as exc:
            raise ProjectError("managed project storage is unavailable or unsafe") from exc
        try:
            opened_database_identity = _opened_database_identity(self.db)
            database_after_open = database_path.stat(follow_symlinks=False)
            path_identity_after_open = (
                database_after_open.st_dev,
                database_after_open.st_ino,
                database_after_open.st_nlink,
            )
            if opened_database_identity != path_identity_after_open:
                raise ProjectError("managed project storage changed during database open")
            if (
                database_before_open is not None
                and (
                    database_before_open.st_dev,
                    database_before_open.st_ino,
                    database_before_open.st_nlink,
                )
                != opened_database_identity
            ):
                raise ProjectError("managed project storage changed during database open")
            if _project_root_identity(self.root) != self._root_identity:
                raise ProjectError("project root changed during database open")
            _validate_managed_storage_paths(self.root)
        except BaseException as exc:
            self.db.close()
            if isinstance(exc, ProjectError):
                raise
            if isinstance(exc, OSError):
                raise ProjectError("managed project storage changed during database open") from exc
            raise
        self.db.row_factory = sqlite3.Row
        try:
            if read_only:
                self.db.execute("PRAGMA query_only=ON")
                self._validate_read_only_database()
            else:
                migrate(self.db)
                self._ensure_project_metadata()
            if _project_root_identity(self.root) != self._root_identity:
                raise ProjectError("project root changed during database open")
        except BaseException:
            self.db.close()
            raise

    @property
    def root_identity(self) -> tuple[int, int]:
        """Immutable device/inode identity captured for the opened project root."""
        return self._root_identity

    @classmethod
    def init(cls, root: str | Path) -> Project:
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        _validate_managed_storage_paths(root)
        _create_project_directories(root)
        _validate_managed_storage_paths(root)
        if not (root / "discourselens.yaml").exists():
            write_project_config(root, ProjectConfig())
        _create_project_database_file(root)
        return cls(root, create_database=False)

    @classmethod
    def open(cls, root: str | Path, *, read_only: bool = False) -> Project:
        return cls(Path(root), read_only=read_only, create_database=False)

    def close(self):
        self.db.close()

    def __enter__(self) -> Self:
        """Allow SDK callers to scope the SQLite connection with ``with``."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def import_document(
        self,
        path: str | Path,
        *,
        ocr_language: str | None = None,
        ocr_dpi: int = 300,
        auto_ocr: bool = False,
        max_bytes: int | None = None,
    ) -> ImportResult:
        """Import a supported local document through the shared Document Engine."""
        self._require_writable()
        from .documents import import_document

        return import_document(
            self,
            path,
            max_bytes=max_bytes,
            ocr_language=ocr_language,
            ocr_dpi=ocr_dpi,
            auto_ocr=auto_ocr,
        )

    def reprocess_document(
        self,
        document_id: str,
        path: str | Path,
        *,
        extractor_suffix: str = ".reprocessed",
        max_bytes: int | None = None,
        ocr_language: str | None = None,
        ocr_dpi: int = 300,
    ) -> ImportResult:
        """Re-extract a PDF as a new revision of an existing document."""
        self._require_writable()
        from .documents import reprocess_pdf

        return reprocess_pdf(
            self,
            document_id,
            path,
            extractor_suffix=extractor_suffix,
            max_bytes=max_bytes,
            ocr_language=ocr_language,
            ocr_dpi=ocr_dpi,
        )

    def import_documents(
        self,
        paths: Iterable[str | Path],
        *,
        max_bytes: int | None = None,
        ocr_language: str | None = None,
        ocr_dpi: int = 300,
    ) -> BatchImportResult:
        """Import paths in iteration order, retaining successful items on failure.

        Each input is passed through :meth:`import_document` exactly once. Expected
        document and filesystem errors become per-item failures; unexpected errors
        propagate so programming defects are not hidden by batch processing.
        """
        self._require_writable()

        def operations() -> Iterator[Callable[[], ImportResult]]:
            for path in paths:
                yield lambda path=path: self.import_document(
                    path,
                    max_bytes=max_bytes,
                    ocr_language=ocr_language,
                    ocr_dpi=ocr_dpi,
                )

        return self._import_batch_operations(operations())

    def import_document_payloads(
        self,
        payloads: Iterable[tuple[str, bytes]],
        *,
        max_bytes: int | None = None,
        ocr_language: str | None = None,
        ocr_dpi: int = 300,
    ) -> BatchImportResult:
        """Batch-import caller-authorized immutable bytes through Core dispatch.

        Adapters that securely bind authorization to an already-open file
        descriptor can supply a lazy iterable of ``(filename, bytes)`` pairs.
        Payloads are consumed sequentially; batch aggregation and expected
        per-document failure handling are shared with :meth:`import_documents`.
        """
        self._require_writable()
        from .documents import import_document_bytes

        effective_max_bytes = self.config.import_max_bytes if max_bytes is None else max_bytes

        def operations() -> Iterator[Callable[[], ImportResult]]:
            for filename, raw in payloads:
                yield lambda filename=filename, raw=raw: import_document_bytes(
                    self,
                    filename,
                    raw,
                    max_bytes=effective_max_bytes,
                    ocr_language=ocr_language,
                    ocr_dpi=ocr_dpi,
                )

        return self._import_batch_operations(operations())

    def _import_batch_operations(
        self,
        operations: Iterable[Callable[[], ImportResult]],
    ) -> BatchImportResult:
        """Apply the shared ordered outcome policy for all batch source adapters."""
        from .documents import BatchImportItemResult, BatchImportResult

        items: list[BatchImportItemResult] = []
        for index, operation in enumerate(operations):
            try:
                imported = operation()
            except DocumentError:
                items.append(
                    BatchImportItemResult(
                        index=index, status="FAILED", failure_code="DOCUMENT_ERROR"
                    )
                )
            except OSError:
                items.append(
                    BatchImportItemResult(index=index, status="FAILED", failure_code="OS_ERROR")
                )
            else:
                if imported.status in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}:
                    status = "SUCCESS"
                    failure_code = None
                elif imported.status == "DUPLICATE":
                    status = "DUPLICATE"
                    failure_code = None
                else:
                    status = "FAILED"
                    failure_code = (
                        imported.status
                        if imported.status in {"OCR_REQUIRED", "PASSWORD_REQUIRED"}
                        else "IMPORT_FAILED"
                    )
                items.append(
                    BatchImportItemResult(
                        index=index,
                        status=status,
                        document_id=imported.document_id,
                        revision_id=imported.revision_id,
                        failure_code=failure_code,
                        warnings=imported.warnings,
                    )
                )

        item_results = tuple(items)
        return BatchImportResult(
            items=item_results,
            total_count=len(item_results),
            success_count=sum(item.status == "SUCCESS" for item in item_results),
            duplicate_count=sum(item.status == "DUPLICATE" for item in item_results),
            failed_count=sum(item.status == "FAILED" for item in item_results),
        )

    def preview_paragraph_embedding_request(
        self,
        preset_name: str,
        revision_id: str,
        *,
        offset: int = 0,
        limit: int = 20,
        include_text: bool = False,
    ) -> ParagraphEmbeddingRequestPreview:
        """Inspect a paginated embedding request without resolving secrets or using the network."""
        from .vectors import preview_paragraph_embedding_request

        return preview_paragraph_embedding_request(
            self,
            preset_name,
            revision_id,
            offset=offset,
            limit=limit,
            include_text=include_text,
        )

    def build_paragraph_embedding_index(
        self,
        preset_name: str,
        revision_id: str,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> str:
        """Build a derived local paragraph index using one explicit embedding preset.

        When no provider implementation is injected, the selected preset may send
        the revision's paragraph text to its configured remote endpoint.
        """
        from .vectors import build_paragraph_embedding_index

        return build_paragraph_embedding_index(
            self, preset_name, revision_id, embedding_provider=embedding_provider
        )

    def vector_index_artifacts(self, revision_id: str) -> list[dict[str, object]]:
        """List verified local vector indexes for one immutable revision.

        Paths and file hashes remain internal. Returned metadata is sufficient
        for a caller to select an index without exposing project filesystem
        details.
        """
        if not isinstance(revision_id, str) or not revision_id.strip():
            raise ScopeViolationError("vector-index listing requires a revision ID")
        if self.db.execute(
            "SELECT 1 FROM document_revisions WHERE id=?", (revision_id,)
        ).fetchone() is None:
            raise ScopeViolationError("vector-index listing references an unknown revision")
        rows = self.db.execute(
            "SELECT id,revision_id,embedding_provider_id,embedding_model,"
            "embedding_dimension,chunker_version,created_at,relative_path,file_hash "
            "FROM artifact_registry WHERE artifact_type='vector_index' AND revision_id=? "
            "ORDER BY created_at DESC,id",
            (revision_id,),
        ).fetchall()
        from .artifact_integrity import verify_project_artifact

        artifacts: list[dict[str, object]] = []
        for row in rows:
            verify_project_artifact(self.root, row["relative_path"], row["file_hash"])
            artifacts.append(
                {
                    "artifact_id": str(row["id"]),
                    "revision_id": str(row["revision_id"]),
                    "provider_id": row["embedding_provider_id"],
                    "model": row["embedding_model"],
                    "dimension": row["embedding_dimension"],
                    "chunker_version": row["chunker_version"],
                    "created_at": str(row["created_at"]),
                }
            )
        return artifacts

    def semantic_search_with_preset(
        self,
        preset_name: str,
        scope_id: str,
        artifact_ids: list[str],
        query_text: str,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        limit: int = 10,
    ) -> RetrievalSearchResult[VectorHit]:
        """Search only explicit vector artifacts in a snapshot using a model preset.

        If no local provider is injected, the query text may be sent to the
        endpoint configured by the selected preset.
        """
        from .vectors import semantic_search_with_preset

        return semantic_search_with_preset(
            self,
            preset_name,
            scope_id,
            artifact_ids,
            query_text,
            embedding_provider=embedding_provider,
            limit=limit,
        )

    def search_vectors_with_provenance(
        self,
        scope_id: str,
        artifact_ids: list[str],
        query: tuple[float, ...],
        *,
        limit: int = 10,
    ) -> RetrievalSearchResult[VectorHit]:
        """Search caller-supplied vectors and persist scope-bound hit lineage."""
        from .vectors import search_vectors_with_provenance

        return search_vectors_with_provenance(self, scope_id, artifact_ids, query, limit=limit)

    def hybrid_search_with_provenance(
        self,
        scope_id: str,
        artifact_ids: list[str],
        lexical_query: str,
        vector_query: tuple[float, ...],
        *,
        limit: int = 10,
        rrf_constant: int = 60,
    ) -> RetrievalSearchResult[HybridHit]:
        """Fuse lexical and caller-supplied vectors with durable hit lineage."""
        from .vectors import hybrid_search_with_provenance

        return hybrid_search_with_provenance(
            self,
            scope_id,
            artifact_ids,
            lexical_query,
            vector_query,
            limit=limit,
            rrf_constant=rrf_constant,
        )

    def hybrid_search_with_preset(
        self,
        preset_name: str,
        scope_id: str,
        artifact_ids: list[str],
        query_text: str,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        limit: int = 10,
        rrf_constant: int = 60,
    ) -> RetrievalSearchResult[HybridHit]:
        """Fuse lexical and vector results in an explicit scope using one preset.

        If no local provider is injected, the query text may be sent to the
        endpoint configured by the selected preset.
        """
        from .vectors import hybrid_search_with_preset

        return hybrid_search_with_preset(
            self,
            preset_name,
            scope_id,
            artifact_ids,
            query_text,
            embedding_provider=embedding_provider,
            limit=limit,
            rrf_constant=rrf_constant,
        )

    def preview_model_preset_request(
        self,
        preset_name: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> PresetRequestPreview:
        """Inspect a generation destination and capability mapping without network or secrets."""
        from .execution import preview_model_preset_request

        return preview_model_preset_request(self, preset_name, prompt, system_prompt=system_prompt)

    def generate_with_preset(
        self,
        preset_name: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        image: GenerationImage | None = None,
        reasoning_override: str | None = None,
        reasoning_mode_override: str | None = None,
    ) -> PresetExecutionResult:
        """Generate through one saved preset and persist Core's execution ledger entry.

        Core validates the selected provider/model, offline policy, image support,
        and declared capabilities. Prompts and response text are not persisted.
        """
        self._require_writable()
        from .execution import execute_model_preset

        return execute_model_preset(
            self,
            preset_name,
            prompt,
            system_prompt=system_prompt,
            image=image,
            reasoning_override=reasoning_override,
            reasoning_mode_override=reasoning_mode_override,
        )

    def stream_with_preset(
        self,
        preset_name: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        reasoning_override: str | None = None,
        reasoning_mode_override: str | None = None,
    ) -> Iterator[str]:
        """Stream text from one saved preset using its registered provider adapter."""
        self._require_writable()
        from .execution import execute_model_preset_stream

        return execute_model_preset_stream(
            self,
            preset_name,
            prompt,
            system_prompt=system_prompt,
            reasoning_override=reasoning_override,
            reasoning_mode_override=reasoning_mode_override,
        )

    def corpus_frequency(self, scope_id: str) -> Counter[str]:
        """Count deterministic surface tokens in one explicit immutable scope."""
        from .corpus import frequency

        return frequency(self, scope_id)

    def corpus_frequency_page(
        self, scope_id: str, *, offset: int = 0, limit: int = 100
    ) -> FrequencyPage:
        """Return a deterministic bounded frequency page for interface adapters."""
        from .corpus import frequency_page

        return frequency_page(self, scope_id, offset=offset, limit=limit)

    def corpus_compare_frequency_page(
        self,
        left_scope_id: str,
        right_scope_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> ComparisonPage:
        """Return a bounded surface-frequency comparison for two frozen scopes."""
        from .corpus import compare_frequency_page

        return compare_frequency_page(
            self, left_scope_id, right_scope_id, offset=offset, limit=limit
        )

    def corpus_keyness_page(
        self,
        left_scope_id: str,
        right_scope_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> ComparisonPage:
        """Return a bounded log-likelihood comparison for two frozen scopes."""
        from .corpus import keyness_page

        return keyness_page(self, left_scope_id, right_scope_id, offset=offset, limit=limit)

    def compare_named_corpora(
        self,
        left_corpus_id: str,
        right_corpus_id: str,
        *,
        metric: str = "frequency",
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, object]:
        """Persist a comparison of two named corpora through membership snapshots."""
        self._require_writable()
        from .corpus import compare_named_corpora

        return compare_named_corpora(
            self,
            left_corpus_id,
            right_corpus_id,
            metric=metric,
            offset=offset,
            limit=limit,
        )

    def corpus_frequency_evidence(self, scope_id: str, term: str) -> dict[str, object]:
        """Persist a scope-exhaustive surface-token count as corpus evidence."""
        from .corpus import frequency_statistic_evidence

        return frequency_statistic_evidence(self, scope_id, term)

    def corpus_dispersion(self, scope_id: str, term: str) -> DocumentDispersionResult:
        """Describe a surface token's distribution across every revision in a scope."""
        from .corpus import document_dispersion

        return document_dispersion(self, scope_id, term)

    def corpus_dispersion_page(
        self, scope_id: str, term: str, *, offset: int = 0, limit: int = 100
    ) -> DocumentDispersionPage:
        """Return exhaustive dispersion totals with a bounded document page."""
        from .corpus import document_dispersion_page

        return document_dispersion_page(self, scope_id, term, offset=offset, limit=limit)

    def corpus_dispersion_evidence(
        self, scope_id: str, term: str
    ) -> DocumentDispersionEvidenceResult:
        """Persist a scope-exhaustive per-revision distribution as corpus evidence."""
        from .corpus import document_dispersion_evidence

        return document_dispersion_evidence(self, scope_id, term)

    def corpus_comparison_evidence(
        self,
        analysis_run_id: str,
        comparison_id: str,
        token: str,
        *,
        comparability_basis: str,
    ) -> dict[str, object]:
        """Link one verified comparison row to an exhaustive finding scope."""
        from .corpus import comparison_statistic_evidence

        return comparison_statistic_evidence(
            self,
            analysis_run_id,
            comparison_id,
            token,
            comparability_basis=comparability_basis,
        )

    def create_comparative_finding(
        self, comparison_id: str, token: str, *, comparability_basis: str
    ) -> dict[str, object]:
        """Create and validate a reproducible deterministic comparison finding."""
        from .corpus import create_comparative_finding

        return create_comparative_finding(
            self, comparison_id, token, comparability_basis=comparability_basis
        )

    def corpus_kwic(self, scope_id: str, query: str, *, window: int = 5) -> list[KwicRow]:
        """Return KWIC rows with canonical half-open offsets in an explicit scope."""
        from .corpus import kwic

        return kwic(self, scope_id, query, window)

    def corpus_kwic_page(
        self,
        scope_id: str,
        query: str,
        *,
        window: int = 5,
        offset: int = 0,
        limit: int = 100,
    ) -> KwicPage:
        """Return a bounded KWIC page with total count and continuation metadata."""
        from .corpus import kwic_page

        return kwic_page(self, scope_id, query, window=window, offset=offset, limit=limit)

    def corpus_lemma_kwic(
        self,
        scope_id: str,
        artifact_ids: list[str],
        query: str,
        *,
        window: int = 5,
        offset: int = 0,
        limit: int = 100,
    ) -> LemmaKwicPage:
        """Return a bounded exact-lemma KWIC page from complete parsed artifacts."""
        from .nlp_artifacts import lemma_kwic_page

        return lemma_kwic_page(
            self,
            scope_id,
            artifact_ids,
            query,
            window=window,
            offset=offset,
            limit=limit,
        )

    def corpus_lemma_kwic_evidence(
        self,
        scope_id: str,
        artifact_ids: list[str],
        revision_id: str,
        lemma: str,
        start_offset: int,
        end_offset: int,
    ) -> LemmaKwicEvidenceResult:
        """Persist one verified parser-lemma occurrence as textual evidence."""
        from .nlp_artifacts import lemma_kwic_evidence

        return lemma_kwic_evidence(
            self,
            scope_id,
            artifact_ids,
            revision_id,
            lemma,
            start_offset,
            end_offset,
        )

    def corpus_kwic_evidence(
        self,
        scope_id: str,
        revision_id: str,
        query: str,
        start_offset: int,
        end_offset: int,
    ) -> KwicEvidenceResult:
        """Create citation-backed evidence for one returned, scope-bound KWIC hit."""
        from .corpus import kwic_evidence

        return kwic_evidence(self, scope_id, revision_id, query, start_offset, end_offset)

    def corpus_collocations(self, scope_id: str, query: str, *, window: int = 5) -> Counter[str]:
        """Count co-occurring surface tokens in exactly one explicit scope."""
        from .corpus import collocations

        return collocations(self, scope_id, query, window)

    def corpus_collocations_page(
        self,
        scope_id: str,
        query: str,
        *,
        window: int = 5,
        offset: int = 0,
        limit: int = 100,
    ) -> CollocationsPage:
        """Return a bounded collocate-count page for one explicit scope."""
        from .corpus import collocations_page

        return collocations_page(
            self,
            scope_id,
            query,
            window=window,
            offset=offset,
            limit=limit,
        )

    def corpus_collocation_analysis(
        self,
        scope_id: str,
        query: str,
        *,
        window: int = 5,
        min_frequency: int = 2,
        offset: int = 0,
        limit: int = 100,
    ) -> CollocationAnalysisResult:
        """Return one ranked page of scoped collocation association measures."""
        from .corpus import collocation_analysis

        return collocation_analysis(
            self,
            scope_id,
            query,
            window=window,
            min_frequency=min_frequency,
            offset=offset,
            limit=limit,
        )

    def corpus_collocation_evidence(
        self,
        scope_id: str,
        query: str,
        collocate: str,
        *,
        window: int = 5,
        min_frequency: int = 2,
    ) -> CollocationEvidenceResult:
        """Persist a selected collocation association with exhaustive scope provenance."""
        from .corpus import collocation_association_evidence

        return collocation_association_evidence(
            self,
            scope_id,
            query,
            collocate,
            window=window,
            min_frequency=min_frequency,
        )

    def corpus_ngrams(self, scope_id: str, size: int) -> Counter[tuple[str, ...]]:
        """Count deterministic surface n-grams in exactly one explicit scope."""
        from .corpus import ngrams

        return ngrams(self, scope_id, size)

    def corpus_ngram_page(
        self, scope_id: str, size: int, *, offset: int = 0, limit: int = 100
    ) -> NgramPage:
        """Return a bounded, deterministic surface n-gram page for an explicit scope."""
        from .corpus import ngram_page

        return ngram_page(self, scope_id, size, offset=offset, limit=limit)

    def corpus_lexical_bundles(
        self,
        scope_id: str,
        *,
        size: int = 3,
        min_frequency: int = 2,
        min_document_frequency: int = 1,
        offset: int = 0,
        limit: int = 100,
    ) -> LexicalBundlesResult:
        """Describe recurring surface bundles in one immutable explicit scope."""
        from .corpus import lexical_bundles

        return lexical_bundles(
            self,
            scope_id,
            size=size,
            min_frequency=min_frequency,
            min_document_frequency=min_document_frequency,
            offset=offset,
            limit=limit,
        )

    def corpus_lexical_bundle_evidence(
        self, scope_id: str, bundle: str, *, size: int
    ) -> LexicalBundleEvidenceResult:
        """Persist an exhaustive citable occurrence count for one surface bundle."""
        from .corpus import lexical_bundle_statistic_evidence

        return lexical_bundle_statistic_evidence(self, scope_id, bundle, size=size)

    def corpus_dependency_patterns(
        self,
        scope_id: str,
        artifact_ids: list[str],
        *,
        relation: str | None = None,
        head_lemma: str | None = None,
        dependent_lemma: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> DependencyPatternsResult:
        """Aggregate parser-derived dependency arcs in a complete explicit scope."""
        from .nlp_artifacts import dependency_patterns

        return dependency_patterns(
            self,
            scope_id,
            artifact_ids,
            relation=relation,
            head_lemma=head_lemma,
            dependent_lemma=dependent_lemma,
            offset=offset,
            limit=limit,
        )

    def corpus_parsed_frequency_page(
        self,
        scope_id: str,
        artifact_ids: list[str],
        *,
        field: str,
        offset: int = 0,
        limit: int = 100,
    ) -> ParsedFrequencyPage:
        """Return a bounded parser-derived frequency page for a complete scope."""
        from .nlp_artifacts import parsed_frequency_page

        return parsed_frequency_page(
            self,
            scope_id,
            artifact_ids,
            field=field,
            offset=offset,
            limit=limit,
        )

    def corpus_surface_frequency_page(
        self,
        scope_id: str,
        artifact_ids: list[str],
        *,
        field: Literal["normalized", "document_frequency"],
        offset: int = 0,
        limit: int = 100,
    ) -> SurfaceFrequencyPage:
        """Return a bounded v2 normalized or document-frequency page for a full scope."""
        from .nlp_artifacts import surface_frequency_page

        return surface_frequency_page(
            self,
            scope_id,
            artifact_ids,
            field=field,
            offset=offset,
            limit=limit,
        )

    def inspect_language_capabilities(
        self,
        language: str,
        *,
        backend: str,
        model: str,
        model_dir: str | Path | None = None,
    ) -> tuple[LanguageCapability, ...]:
        """Probe one explicitly selected local NLP model without downloading it."""
        from .languages import inspect_local_backend

        return inspect_local_backend(
            language,
            backend=backend,
            model=model,
            model_dir=str(model_dir) if model_dir is not None else None,
        )

    def build_surface_token_artifact(self, revision_id: str) -> TokenArtifact:
        """Build and register the deterministic surface-token Parquet artifact."""
        from .nlp_artifacts import build_surface_token_artifact

        return build_surface_token_artifact(self, revision_id)

    def build_parsed_token_artifact(self, revision_id: str, backend: NlpBackend) -> TokenArtifact:
        """Build a derived parse artifact with an explicitly selected local backend."""
        from .nlp_artifacts import build_parsed_token_artifact

        return build_parsed_token_artifact(self, revision_id, backend)

    def analyze_textual_descriptors(
        self,
        revision_ids: list[str],
        *,
        languages: tuple[str, ...] = ("en", "es"),
    ) -> TextualDescriptorAnalysisResult:
        """Run deterministic textual descriptors as an immutable AnalysisRun."""
        from .analysis import run_textual_descriptors_analysis

        return run_textual_descriptors_analysis(self, revision_ids, languages)

    def extract_evidence(
        self,
        revision_ids: list[str],
        *,
        languages: tuple[str, ...] = ("en", "es"),
        max_examples: int = 48,
    ) -> EvidenceExtractionAnalysisResult:
        """Run descriptive evidence extraction without an LLM or interpretation claims."""
        from .analysis import run_evidence_extraction_analysis

        return run_evidence_extraction_analysis(
            self, revision_ids, languages, max_examples=max_examples
        )

    def analyze_quote_attribution_candidates(
        self,
        revision_ids: list[str],
        *,
        revision_languages: Mapping[str, str] | None = None,
    ) -> QuoteAttributionAnalysisResult:
        """Inventory quotation and attribution candidates in an explicit revision scope."""
        from .analysis import run_quote_attribution_candidate_analysis

        return run_quote_attribution_candidate_analysis(
            self, revision_ids, revision_languages=revision_languages
        )

    def _validate_parser_candidate_selection(
        self, revision_ids: list[str], artifact_ids: list[str]
    ) -> None:
        """Validate explicit parser artifacts against the selected revision scope."""
        from .artifact_integrity import verify_project_artifact

        if (
            not isinstance(revision_ids, list)
            or not revision_ids
            or any(not isinstance(item, str) or not item.strip() for item in revision_ids)
        ):
            raise ScopeViolationError("scope requires explicit document revision IDs")
        if len(revision_ids) != len(set(revision_ids)):
            raise ScopeViolationError("scope cannot contain duplicate document revisions")
        if (
            not isinstance(artifact_ids, list)
            or not artifact_ids
            or any(not isinstance(item, str) or not item.strip() for item in artifact_ids)
        ):
            raise ScopeViolationError("token analytics requires explicit artifact IDs")
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ScopeViolationError("artifact IDs must not be duplicated")

        selected_revisions = set(revision_ids)
        existing_revisions: set[str] = set()
        for batch in _in_parameter_batches(revision_ids):
            placeholders = ",".join("?" for _ in batch)
            existing_revisions.update(
                str(row[0])
                for row in self.db.execute(
                    f"SELECT id FROM document_revisions WHERE id IN ({placeholders})", batch
                )
            )
        if existing_revisions != selected_revisions:
            raise ScopeViolationError("scope references an unknown document revision")

        artifact_rows: list[sqlite3.Row] = []
        for batch in _in_parameter_batches(artifact_ids):
            placeholders = ",".join("?" for _ in batch)
            artifact_rows.extend(
                self.db.execute(
                    "SELECT id,revision_id,artifact_type,relative_path,file_hash "
                    f"FROM artifact_registry WHERE id IN ({placeholders})",
                    batch,
                ).fetchall()
            )
        if len(artifact_rows) != len(artifact_ids):
            raise ScopeViolationError("artifact set contains an unknown artifact ID")
        if any(
            row["artifact_type"] != "parsed_tokens" or row["revision_id"] not in selected_revisions
            for row in artifact_rows
        ):
            raise ScopeViolationError("artifact set is unknown or outside the explicit scope")
        artifact_revisions = [str(row["revision_id"]) for row in artifact_rows]
        if len(artifact_revisions) != len(set(artifact_revisions)):
            raise ScopeViolationError(
                "select at most one token artifact per document revision; "
                "artifact versions cannot be combined"
            )
        for row in artifact_rows:
            verify_project_artifact(self.root, row["relative_path"], row["file_hash"])

    def analyze_social_actor_candidates(
        self, revision_ids: list[str], artifact_ids: list[str]
    ) -> AnalysisResult:
        """Run deterministic social-actor parser candidate extraction."""
        from .analysis import run_social_actor_candidate_analysis

        self._validate_parser_candidate_selection(revision_ids, artifact_ids)
        return run_social_actor_candidate_analysis(self, revision_ids, artifact_ids)

    def analyze_agency_candidates(
        self, revision_ids: list[str], artifact_ids: list[str]
    ) -> AnalysisResult:
        """Run dependency-based agency candidates with parser capability preflight."""
        from .analysis import run_agency_candidate_analysis

        self._validate_parser_candidate_selection(revision_ids, artifact_ids)
        return run_agency_candidate_analysis(self, revision_ids, artifact_ids)

    def run_scope_question_candidates(
        self,
        revision_ids: list[str],
        framework_id: str,
        question: str,
        *,
        limit: int = 12,
    ) -> ScopeQuestionAnalysisResult:
        """Retrieve bounded lexical examples for one question in a frozen revision scope."""
        from .analysis import run_scope_question_candidates_analysis

        return run_scope_question_candidates_analysis(
            self,
            revision_ids,
            framework_id,
            question,
            limit=limit,
        )

    def ask_question(
        self,
        revision_ids: list[str],
        question: str,
        framework_id: str,
        preset_name: str,
        *,
        limit: int = 8,
        retrieval_mode: Literal["lexical", "hybrid"] = "lexical",
        embedding_preset_name: str | None = None,
        embedding_artifact_ids: list[str] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> AskQuestionResult:
        """Answer a scoped question with targeted lexical or opt-in hybrid evidence."""
        from .analysis import ask_question_analysis

        arguments = (
            self,
            revision_ids,
            question,
            framework_id,
            preset_name,
        )
        if conversation_context is None:
            return ask_question_analysis(
                *arguments,
                limit=limit,
                retrieval_mode=retrieval_mode,
                embedding_preset_name=embedding_preset_name,
                embedding_artifact_ids=embedding_artifact_ids,
                embedding_provider=embedding_provider,
            )
        return ask_question_analysis(
            *arguments,
            limit=limit,
            retrieval_mode=retrieval_mode,
            embedding_preset_name=embedding_preset_name,
            embedding_artifact_ids=embedding_artifact_ids,
            embedding_provider=embedding_provider,
            conversation_context=conversation_context,
        )

    def prepare_van_dijk_group_comparison(
        self,
        revision_ids: list[str],
        *,
        groups: list[VanDijkGroupComparisonGroupInput],
        shared_topic_event: str,
        comparability_basis: str,
        codebook_id: str,
    ) -> VanDijkGroupComparisonAnalysisResult:
        """Persist a descriptive comparison of researcher-selected annotation sets."""
        from .analysis import run_van_dijk_group_comparison_analysis

        return run_van_dijk_group_comparison_analysis(
            self,
            revision_ids,
            groups=groups,
            shared_topic_event=shared_topic_event,
            comparability_basis=comparability_basis,
            codebook_id=codebook_id,
        )

    def counterevidence_review(
        self,
        finding_id: str,
        phrase: str,
        *,
        offset: int = 0,
        limit: int = 100,
        retrieval_run_id: str | None = None,
    ) -> CounterEvidenceAnalysisResult:
        """Exhaustively scan a Finding's original scope or page its saved result."""
        from .analysis import run_counterevidence_review

        return run_counterevidence_review(
            self,
            finding_id,
            phrase,
            offset=offset,
            limit=limit,
            retrieval_run_id=retrieval_run_id,
        )

    def analyze_controlled_cda(
        self,
        revision_ids: list[str],
        preset_name: str,
        *,
        frameworks: tuple[str, ...] | None = None,
        languages: tuple[str, ...] = ("en", "es"),
        counterevidence_queries: dict[str, str] | None = None,
        routing_profile: str | None = None,
    ) -> ControlledCdaResult:
        """Run the controlled Core pipeline with an explicitly selected model preset."""
        from .analysis import DEFAULT_CDA_FRAMEWORKS, run_controlled_cda_review

        return run_controlled_cda_review(
            self,
            revision_ids,
            preset_name,
            frameworks=DEFAULT_CDA_FRAMEWORKS if frameworks is None else frameworks,
            languages=languages,
            counterevidence_queries=counterevidence_queries,
            routing_profile=routing_profile,
        )

    def analyze_profile(
        self,
        revision_ids: list[str],
        profile_name: str,
        *,
        preset_name: str | None = None,
    ) -> ControlledCdaResult:
        """Execute a saved, validated analysis profile through the trusted pipeline."""
        from .analysis import run_saved_controlled_profile

        return run_saved_controlled_profile(
            self, revision_ids, profile_name, preset_name=preset_name
        )

    @overload
    def run_batch_saved_analysis_profile(
        self,
        revision_ids: list[str],
        profile_name: str,
        *,
        preset_name: str | None = None,
        continue_on_error: Literal[False] = False,
    ) -> list[ControlledCdaResult]: ...

    @overload
    def run_batch_saved_analysis_profile(
        self,
        revision_ids: list[str],
        profile_name: str,
        *,
        preset_name: str | None = None,
        continue_on_error: Literal[True],
    ) -> list[SavedProfileBatchItem]: ...

    @overload
    def run_batch_saved_analysis_profile(
        self,
        revision_ids: list[str],
        profile_name: str,
        *,
        preset_name: str | None = None,
        continue_on_error: bool,
    ) -> list[ControlledCdaResult] | list[SavedProfileBatchItem]: ...

    def run_batch_saved_analysis_profile(
        self,
        revision_ids: list[str],
        profile_name: str,
        *,
        preset_name: str | None = None,
        continue_on_error: bool = False,
    ) -> list[ControlledCdaResult] | list[SavedProfileBatchItem]:
        """Run one independent saved-profile analysis per selected revision.

        With ``continue_on_error=True``, return a status item for each revision;
        resume a failed item with an ``analysis_run_id`` via
        :meth:`resume_controlled_cda`.
        """
        from .analysis import run_batch_saved_controlled_profile

        return run_batch_saved_controlled_profile(
            self,
            revision_ids,
            profile_name,
            preset_name=preset_name,
            continue_on_error=continue_on_error,
        )

    def resume_controlled_cda(self, analysis_run_id: str) -> ControlledCdaResult:
        """Resume a failed controlled run without replaying completed stages."""
        from .analysis import resume_controlled_cda_review

        return resume_controlled_cda_review(self, analysis_run_id)

    def validate_finding(self, finding_id: str) -> FindingValidation:
        """Validate canonical evidence lineage before a finding is presented or exported."""
        from .validation import validate_finding

        return validate_finding(self, finding_id)

    def review_finding_with_model(
        self,
        finding_id: str,
        preset_name: str,
        *,
        framework: str | None = None,
        context_evidence_ids: tuple[str, ...] = (),
        conversation_context: list[dict[str, str]] | None = None,
    ) -> InterpretationResult:
        """Create an immutable model-review run using optional supplied context evidence."""
        from .interpretation import review_finding_with_model

        arguments = (self, finding_id, preset_name)
        if conversation_context is None:
            return review_finding_with_model(
                *arguments,
                framework=framework,
                context_evidence_ids=context_evidence_ids,
            )
        return review_finding_with_model(
            *arguments,
            framework=framework,
            context_evidence_ids=context_evidence_ids,
            conversation_context=conversation_context,
        )

    def export_analysis_report(
        self,
        analysis_run_id: str,
        destination: str | Path,
        *,
        format: Literal["markdown", "html", "json", "pdf", "docx"] = "markdown",
    ) -> Path:
        """Write a deterministic report generated only from persisted run records."""
        from .reporting import write_analysis_report

        return write_analysis_report(self, analysis_run_id, destination, format)

    def export_research_bundle(
        self,
        analysis_run_id: str,
        destination: str | Path,
        *,
        include_canonical_text: bool = False,
    ) -> Path:
        """Export a bundle; full canonical text is opt-in, cited excerpts remain included."""
        from .bundle import export_bundle

        return export_bundle(
            self,
            analysis_run_id,
            destination,
            include_canonical_text=include_canonical_text,
        )

    @property
    def config(self) -> ProjectConfig:
        return resolve_project_config(self.root)

    def set_project_config(self, config: ProjectConfig) -> None:
        self._require_writable()
        write_project_config(self.root, config)

    def _require_writable(self) -> None:
        if self.read_only:
            raise ProjectError("project is open in read-only mode")

    def _validate_read_only_database(self) -> None:
        """Open only a fully initialized current schema; never migrate in read-only mode."""
        try:
            row = self.db.execute("SELECT max(version) FROM schema_migrations").fetchone()
        except sqlite3.Error as exc:
            raise ProjectError("read-only project has no readable DiscourseLens schema") from exc
        version = row[0] if row is not None else None
        if version != SCHEMA_VERSION:
            raise ProjectError(
                f"read-only mode requires schema version {SCHEMA_VERSION}; found {version!r}; "
                "open the project once in writable mode to apply migrations"
            )
        if not self.db.execute("SELECT 1 FROM project_meta WHERE key='project_id'").fetchone():
            raise ProjectError("read-only project identity is missing")

    def _ensure_project_metadata(self) -> None:
        """Create portable project identity/settings once, never from a filesystem path."""
        # INSERT OR IGNORE makes initialization safe when two local processes open
        # a project before its metadata has been created. A read-then-insert here
        # allowed both processes to observe missing keys and race on the primary key.
        self.db.executemany(
            "INSERT OR IGNORE INTO project_meta(key,value) VALUES(?,?)",
            [("project_id", uid()), ("offline_mode", "false")],
        )
        self.db.commit()

    @property
    def project_id(self) -> str:
        row = self.db.execute("SELECT value FROM project_meta WHERE key='project_id'").fetchone()
        if not row:
            raise DocumentError("project identity is missing")
        return str(row["value"])

    @property
    def offline_mode(self) -> bool:
        row = self.db.execute("SELECT value FROM project_meta WHERE key='offline_mode'").fetchone()
        return row is not None and row["value"] == "true"

    def set_offline_mode(self, enabled: bool) -> None:
        """Persist a project-local network policy without storing any credentials."""
        self._require_writable()
        self.db.execute(
            "INSERT INTO project_meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("offline_mode", "true" if enabled else "false"),
        )
        self.db.commit()

    def import_text(
        self,
        name: str,
        text: str,
        source_bytes: bytes | None = None,
        extractor: str = "plain-text@1",
        document_id: str | None = None,
        mime_type: str = "text/plain",
        create_paragraphs: bool = True,
        extraction_status: str = "SUCCESS",
        ocr_language: str | None = None,
        ocr_dpi: int | None = None,
        _extraction_manifest_json: str | None = None,
        _structure_hash: str | None = None,
        _defer_commit: bool = False,
    ) -> tuple[str, str]:
        self._require_writable()
        if _defer_commit and not self.db.in_transaction:
            raise ProjectError("deferred document import requires an active transaction")
        display_name = _path_safe_display_name(name)
        if display_name is None:
            raise DocumentError("document name must be a non-empty filename")
        if not text:
            raise DocumentError("empty canonical text")
        source = source_bytes if source_bytes is not None else text.encode()
        did = document_id or uid()
        rid = uid()
        sh = digest(source)
        ch = digest(text.encode())
        extraction_manifest_json = (
            _extraction_manifest_json
            if _extraction_manifest_json is not None
            else basic_extraction_manifest(extractor)
        )
        structure_hash = _structure_hash if _structure_hash is not None else ch
        if not re.fullmatch(r"[0-9a-f]{64}", structure_hash):
            raise DocumentError("document structure fingerprint is invalid")
        try:
            manifest_payload = json.loads(extraction_manifest_json)
        except (TypeError, ValueError, RecursionError) as exc:
            raise DocumentError("document extraction manifest is invalid") from exc
        if (
            not isinstance(manifest_payload, dict)
            or len(extraction_manifest_json) > 64 * 1024
            or canonical_json(manifest_payload) != extraction_manifest_json
        ):
            raise DocumentError("document extraction manifest is invalid")
        text_path = Path("source") / f"{rid}.txt"
        original_path = Path("source") / f"{rid}.original"
        text_file = self._resolve_managed_source_file(str(text_path), must_exist=False)
        original_file = self._resolve_managed_source_file(str(original_path), must_exist=False)
        source_directory_fds = self._open_managed_source_directory()
        created_paths: list[Path] = []
        created_entries: list[str] = []
        try:
            if source_directory_fds is None:
                # Windows does not expose dir_fd for os.open; retain its portable
                # path-based implementation with component validation immediately
                # before each exclusive create.
                self._resolve_managed_source_file(str(text_path), must_exist=False)
                text_descriptor = os.open(
                    text_file,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_BINARY", 0),
                    0o600,
                )
                created_paths.append(text_file)
                self._write_source_content(text_descriptor, text)
                self._resolve_managed_source_file(str(original_path), must_exist=False)
                original_descriptor = os.open(
                    original_file,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_BINARY", 0),
                    0o600,
                )
                created_paths.append(original_file)
                self._write_source_content(original_descriptor, source)
            else:
                root_fd, hidden_fd, source_fd = source_directory_fds
                self._verify_source_directory_handles(root_fd, hidden_fd, source_fd)
                self._write_source_entry(source_fd, text_path.name, text, created_entries)
                self._verify_source_directory_handles(root_fd, hidden_fd, source_fd)
                self._write_source_entry(source_fd, original_path.name, source, created_entries)
                self._verify_source_directory_handles(root_fd, hidden_fd, source_fd)
        except DocumentError:
            self._remove_uncommitted_source_files(created_paths)
            if source_directory_fds is not None:
                self._remove_uncommitted_source_entries(source_directory_fds[2], created_entries)
                self._close_managed_source_directory(source_directory_fds)
            raise
        except OSError as exc:
            self._remove_uncommitted_source_files(created_paths)
            if source_directory_fds is not None:
                self._remove_uncommitted_source_entries(source_directory_fds[2], created_entries)
                self._close_managed_source_directory(source_directory_fds)
            raise DocumentError("project source storage is unavailable") from exc
        try:
            if document_id is None:
                self.db.execute("INSERT INTO documents VALUES(?,?,?)", (did, display_name, now()))
            self.db.execute(
                """
                INSERT INTO document_revisions(
                    id,document_id,source_hash,canonical_hash,extractor,coordinate_scheme,
                    text_path,original_path,mime_type,source_size,extraction_status,created_at,
                    ocr_language,ocr_dpi,extraction_manifest_json,structure_hash
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    rid,
                    did,
                    sh,
                    ch,
                    extractor,
                    "unicode-codepoint-half-open-v1",
                    str(text_path),
                    str(original_path),
                    mime_type,
                    len(source),
                    extraction_status,
                    now(),
                    ocr_language,
                    ocr_dpi,
                    extraction_manifest_json,
                    structure_hash,
                ),
            )
            if create_paragraphs:
                for ordinal, (start, end, p) in enumerate(self._paragraphs(text)):
                    pid = uid()
                    self.db.execute(
                        "INSERT INTO paragraphs(id,revision_id,ordinal,start_offset,end_offset,page_number,text,segmentation_version) VALUES(?,?,?,?,?,?,?,?)",
                        (pid, rid, ordinal, start, end, None, p, "plain-text-lines-v2"),
                    )
                    self.db.execute("INSERT INTO revision_fts VALUES(?,?,?)", (rid, pid, p))
                self._persist_sentences(rid)
            if source_directory_fds is not None:
                self._verify_source_directory_handles(*source_directory_fds)
            if not _defer_commit:
                self.db.commit()
        except BaseException:
            if self.db.in_transaction:
                self.db.rollback()
            self._remove_uncommitted_source_files(created_paths)
            if source_directory_fds is not None:
                self._remove_uncommitted_source_entries(source_directory_fds[2], created_entries)
            raise
        finally:
            if source_directory_fds is not None:
                self._close_managed_source_directory(source_directory_fds)
        return did, rid

    def list_research_projects(self) -> list[dict[str, object]]:
        """List private, user-owned research workspaces, creating a default for existing data."""
        rows = self.db.execute(
            "SELECT id,name,created_at,updated_at FROM research_projects ORDER BY created_at,id"
        ).fetchall()
        if not rows:
            project_id = uid()
            timestamp = now()
            self.db.execute(
                "INSERT INTO research_projects(id,name,created_at,updated_at) VALUES(?,?,?,?)",
                (project_id, "Mi primer proyecto", timestamp, timestamp),
            )
            self.db.execute(
                "INSERT OR IGNORE INTO research_project_documents(project_id,document_id,added_at) "
                "SELECT ?,id,? FROM documents",
                (project_id, timestamp),
            )
            self.db.commit()
            rows = self.db.execute(
                "SELECT id,name,created_at,updated_at FROM research_projects ORDER BY created_at,id"
            ).fetchall()
        return [dict(row) for row in rows]

    def create_research_project(self, name: str) -> str:
        self._require_writable()
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 120:
            raise DocumentError("project name must contain 1..120 characters")
        project_id = uid()
        timestamp = now()
        self.db.execute(
            "INSERT INTO research_projects(id,name,created_at,updated_at) VALUES(?,?,?,?)",
            (project_id, name.strip(), timestamp, timestamp),
        )
        self.db.commit()
        return project_id

    def rename_research_project(self, project_id: str, name: str) -> dict[str, str]:
        """Rename one private research project without moving its contents."""
        self._require_writable()
        clean_name = name.strip() if isinstance(name, str) else ""
        if not clean_name or len(clean_name) > 120:
            raise DocumentError("project name must contain 1..120 characters")
        if not isinstance(project_id, str) or not project_id.strip():
            raise DocumentError("research project ID is invalid")
        cursor = self.db.execute(
            "UPDATE research_projects SET name=?,updated_at=? WHERE id=?",
            (clean_name, now(), project_id),
        )
        if cursor.rowcount != 1:
            raise DocumentError("unknown research project")
        self.db.commit()
        return {"id": project_id, "name": clean_name}

    def delete_research_project(self, project_id: str) -> dict[str, object]:
        """Remove a research project while preserving its sources and conversations."""
        self._require_writable()
        if not isinstance(project_id, str) or not project_id.strip():
            raise DocumentError("research project ID is invalid")

        try:
            self.db.execute("BEGIN IMMEDIATE")
            project = self.db.execute(
                "SELECT id,name FROM research_projects WHERE id=?", (project_id,)
            ).fetchone()
            if not project:
                raise DocumentError("unknown research project")

            fallback = self.db.execute(
                "SELECT id,name FROM research_projects WHERE id<>? ORDER BY created_at,id LIMIT 1",
                (project_id,),
            ).fetchone()
            if fallback is None:
                fallback_id = uid()
                fallback_name = "Proyecto principal"
                timestamp = now()
                self.db.execute(
                    "INSERT INTO research_projects(id,name,created_at,updated_at) VALUES(?,?,?,?)",
                    (fallback_id, fallback_name, timestamp, timestamp),
                )
            else:
                fallback_id = str(fallback["id"])
                fallback_name = str(fallback["name"])

            moved_documents = int(
                self.db.execute(
                    "SELECT COUNT(*) FROM research_project_documents WHERE project_id=?",
                    (project_id,),
                ).fetchone()[0]
            )
            moved_conversations = int(
                self.db.execute(
                    "SELECT COUNT(*) FROM chat_conversations WHERE research_project_id=?",
                    (project_id,),
                ).fetchone()[0]
            )
            timestamp = now()
            self.db.execute(
                "INSERT OR IGNORE INTO research_project_documents(project_id,document_id,added_at) "
                "SELECT ?,document_id,? FROM research_project_documents WHERE project_id=?",
                (fallback_id, timestamp, project_id),
            )
            self.db.execute(
                "UPDATE chat_conversations SET research_project_id=? WHERE research_project_id=?",
                (fallback_id, project_id),
            )
            self.db.execute(
                "UPDATE research_projects SET updated_at=? WHERE id=?",
                (timestamp, fallback_id),
            )
            self.db.execute("DELETE FROM research_projects WHERE id=?", (project_id,))
            self.db.commit()
        except BaseException:
            if self.db.in_transaction:
                self.db.rollback()
            raise

        return {
            "id": str(project["id"]),
            "name": str(project["name"]),
            "moved_to_id": fallback_id,
            "moved_to_name": fallback_name,
            "moved_documents": moved_documents,
            "moved_conversations": moved_conversations,
        }

    def add_document_to_research_project(self, project_id: str, document_id: str) -> None:
        self._require_writable()
        if not self.db.execute("SELECT 1 FROM research_projects WHERE id=?", (project_id,)).fetchone():
            raise DocumentError("unknown research project")
        if not self.db.execute("SELECT 1 FROM documents WHERE id=?", (document_id,)).fetchone():
            raise DocumentError("unknown document")
        self.db.execute(
            "INSERT OR IGNORE INTO research_project_documents(project_id,document_id,added_at) "
            "VALUES(?,?,?)",
            (project_id, document_id, now()),
        )
        self.db.execute(
            "UPDATE research_projects SET updated_at=? WHERE id=?", (now(), project_id)
        )
        self.db.commit()

    def delete_document_from_research_project(self, project_id: str, document_id: str) -> None:
        """Permanently delete a document and all project data derived from it.

        Conversations that include one of the document's revisions are deleted
        because their prompts and answers can contain source text and citations.
        Files are removed only after their database references have been purged.
        """
        self._require_writable()
        if not self.db.execute(
            "SELECT 1 FROM research_project_documents WHERE project_id=? AND document_id=?",
            (project_id, document_id),
        ).fetchone():
            raise DocumentError("document is not in this research project")

        revision_rows = self.db.execute(
            "SELECT id,text_path,original_path FROM document_revisions WHERE document_id=?",
            (document_id,),
        ).fetchall()
        revision_ids = {str(row["id"]) for row in revision_rows}
        revision_paths = [
            path
            for row in revision_rows
            for path in (row["text_path"], row["original_path"])
            if isinstance(path, str) and path
        ]
        artifact_rows = self.db.execute(
            "SELECT relative_path,file_hash FROM artifact_registry WHERE revision_id IN "
            "(SELECT id FROM document_revisions WHERE document_id=?)",
            (document_id,),
        ).fetchall()

        span_ids = {
            str(row[0])
            for row in self.db.execute(
                "SELECT id FROM source_spans WHERE revision_id IN "
                "(SELECT id FROM document_revisions WHERE document_id=?)",
                (document_id,),
            )
        }
        paragraph_ids = {
            str(row[0])
            for row in self.db.execute(
                "SELECT id FROM paragraphs WHERE revision_id IN "
                "(SELECT id FROM document_revisions WHERE document_id=?)",
                (document_id,),
            )
        }
        scope_ids: set[str] = set()
        for row in self.db.execute("SELECT id,revision_ids_json FROM scope_snapshots"):
            try:
                scope_revision_ids = json.loads(row["revision_ids_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(scope_revision_ids, list) and any(
                isinstance(revision_id, str) and revision_id in revision_ids
                for revision_id in scope_revision_ids
            ):
                scope_ids.add(str(row["id"]))

        conversation_ids: set[str] = set()
        for row in self.db.execute("SELECT id,revision_ids_json FROM chat_conversations"):
            try:
                conversation_revision_ids = json.loads(row["revision_ids_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(conversation_revision_ids, list) and any(
                isinstance(revision_id, str) and revision_id in revision_ids
                for revision_id in conversation_revision_ids
            ):
                conversation_ids.add(str(row["id"]))

        def select_column_values(
            table: str, selected_column: str, filter_column: str, values: set[str]
        ) -> set[str]:
            result: set[str] = set()
            ordered = sorted(values)
            for offset in range(0, len(ordered), _SQLITE_IN_BATCH_SIZE):
                batch = ordered[offset : offset + _SQLITE_IN_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                result.update(
                    str(row[0])
                    for row in self.db.execute(
                        f"SELECT {selected_column} FROM {table} "
                        f"WHERE {filter_column} IN ({placeholders})",
                        batch,
                    )
                )
            return result

        analysis_ids = select_column_values("analysis_runs", "id", "scope_id", scope_ids)
        skill_ids = select_column_values("skill_runs", "id", "scope_id", scope_ids)
        skill_ids.update(
            select_column_values("skill_runs", "id", "analysis_run_id", analysis_ids)
        )
        retrieval_ids = select_column_values("retrieval_runs", "id", "scope_id", scope_ids)
        retrieval_ids.update(
            select_column_values("retrieval_runs", "id", "analysis_run_id", analysis_ids)
        )
        retrieval_ids.update(
            select_column_values("retrieval_runs", "id", "skill_run_id", skill_ids)
        )
        retrieval_ids.update(
            str(row[0])
            for row in self.db.execute(
                "SELECT DISTINCT h.retrieval_run_id FROM retrieval_hits h "
                "JOIN source_spans s ON s.id=h.source_span_id "
                "JOIN document_revisions r ON r.id=s.revision_id WHERE r.document_id=?",
                (document_id,),
            )
        )
        reranking_ids = select_column_values(
            "reranking_runs", "id", "retrieval_run_id", retrieval_ids
        )
        finding_ids = select_column_values("findings", "id", "analysis_run_id", analysis_ids)
        finding_ids.update(
            select_column_values("findings", "id", "skill_run_id", skill_ids)
        )
        evidence_ids = select_column_values("evidence_artifacts", "id", "span_id", span_ids)
        if evidence_ids:
            finding_ids.update(
                select_column_values("finding_evidence", "finding_id", "evidence_id", evidence_ids)
            )
        coverage_ids = select_column_values(
            "skill_runs", "coverage_id", "id", skill_ids
        )
        coverage_ids.update(
            select_column_values("retrieval_runs", "coverage_id", "id", retrieval_ids)
        )
        coverage_ids.discard("None")

        # Artifact paths are verified before their database rows are removed.
        from .artifact_integrity import verify_project_artifact

        artifact_paths: list[Path] = []
        for row in artifact_rows:
            try:
                artifact_paths.append(
                    verify_project_artifact(self.root, row["relative_path"], row["file_hash"])
                )
            except (OSError, ProjectError, ValueError):
                # An invalid or modified registry entry must never redirect deletion
                # outside the managed project directory.
                continue
        source_paths: list[Path] = []
        for stored_path in revision_paths:
            try:
                source_paths.append(self._resolve_managed_source_file(stored_path))
            except DocumentError:
                continue

        def delete_values(table: str, column: str, values: set[str]) -> None:
            ordered = sorted(values)
            for offset in range(0, len(ordered), _SQLITE_IN_BATCH_SIZE):
                batch = ordered[offset : offset + _SQLITE_IN_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                self.db.execute(f"DELETE FROM {table} WHERE {column} IN ({placeholders})", batch)

        comparison_ids = select_column_values(
            "comparison_runs", "id", "left_scope_id", scope_ids
        )
        comparison_ids.update(
            select_column_values("comparison_runs", "id", "right_scope_id", scope_ids)
        )
        retrieval_hit_ids = select_column_values(
            "retrieval_hits", "id", "retrieval_run_id", retrieval_ids
        )

        self.db.execute("BEGIN IMMEDIATE")
        triggers = self.db.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger' AND sql IS NOT NULL"
        ).fetchall()
        try:
            for trigger in triggers:
                escaped_name = str(trigger["name"]).replace('"', '""')
                self.db.execute(f'DROP TRIGGER "{escaped_name}"')

            delete_values("chat_messages", "conversation_id", conversation_ids)
            delete_values("chat_conversations", "id", conversation_ids)
            delete_values("comparison_result_rows", "comparison_id", comparison_ids)
            delete_values("comparison_runs", "id", comparison_ids)
            delete_values("research_memos", "scope_id", scope_ids)

            delete_values("reranking_hits", "reranking_run_id", reranking_ids)
            delete_values("reranking_runs", "id", reranking_ids)
            delete_values("retrieval_hit_context", "retrieval_hit_id", retrieval_hit_ids)
            delete_values("retrieval_hits", "retrieval_run_id", retrieval_ids)
            delete_values("retrieval_run_context", "retrieval_run_id", retrieval_ids)
            delete_values("retrieval_runs", "id", retrieval_ids)

            delete_values("finding_reviews", "finding_id", finding_ids)
            delete_values("finding_observation_candidates", "finding_id", finding_ids)
            delete_values("finding_evidence", "finding_id", finding_ids)
            delete_values("finding_evidence", "evidence_id", evidence_ids)
            delete_values("finding_lineage", "finding_id", finding_ids)
            delete_values("finding_lineage", "source_finding_id", finding_ids)
            delete_values("findings", "id", finding_ids)
            delete_values("model_executions", "analysis_run_id", analysis_ids)
            delete_values("model_executions", "skill_run_id", skill_ids)
            delete_values("skill_runs", "id", skill_ids)
            delete_values("analysis_runs", "id", analysis_ids)
            delete_values("coverage_descriptors", "id", coverage_ids)
            delete_values("scope_snapshots", "id", scope_ids)

            delete_values("annotations", "span_id", span_ids)
            delete_values("evidence_artifacts", "id", evidence_ids)
            delete_values("source_span_segments", "span_id", span_ids)
            delete_values("source_spans", "id", span_ids)
            delete_values("sentences", "revision_id", revision_ids)
            delete_values("paragraphs", "id", paragraph_ids)
            delete_values("pages", "revision_id", revision_ids)
            delete_values("revision_fts", "revision_id", revision_ids)
            delete_values("extraction_warnings", "revision_id", revision_ids)
            delete_values("artifact_registry", "revision_id", revision_ids)
            delete_values("corpus_memberships", "revision_id", revision_ids)
            delete_values("subcorpus_memberships", "revision_id", revision_ids)
            delete_values("research_project_documents", "document_id", {document_id})
            delete_values("document_metadata", "document_id", {document_id})
            delete_values("document_revisions", "id", revision_ids)
            delete_values("documents", "id", {document_id})

            for trigger in triggers:
                self.db.execute(str(trigger["sql"]))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

        for path in [*source_paths, *artifact_paths]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                _LOGGER.warning("could not remove deleted document storage entry", exc_info=True)

    def research_project_document_ids(self, project_id: str) -> list[str]:
        if not self.db.execute("SELECT 1 FROM research_projects WHERE id=?", (project_id,)).fetchone():
            raise DocumentError("unknown research project")
        return [
            str(row[0])
            for row in self.db.execute(
                "SELECT document_id FROM research_project_documents WHERE project_id=? "
                "ORDER BY added_at,document_id",
                (project_id,),
            ).fetchall()
        ]

    def document_ids_for_revisions(self, revision_ids: list[str]) -> set[str]:
        document_ids: set[str] = set()
        for revision_id in revision_ids:
            row = self.db.execute(
                "SELECT document_id FROM document_revisions WHERE id=?", (revision_id,)
            ).fetchone()
            if row is not None:
                document_ids.add(str(row[0]))
        return document_ids

    def documents_page(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        research_project_id: str | None = None,
    ) -> dict[str, object]:
        """Return a bounded, path-free page of document and latest-revision metadata."""
        if (
            type(offset) is not int
            or not 0 <= offset <= 0x7FFF_FFFF_FFFF_FFFF
            or type(limit) is not int
            or not 1 <= limit <= 1_000
        ):
            raise DocumentError(
                "documents page requires an integer offset in SQLite range "
                "and integer limit 1..1000"
            )

        if research_project_id is None:
            total = int(self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            page_from = "FROM documents d ORDER BY d.display_name,d.id LIMIT ? OFFSET ?"
            page_parameters: tuple[object, ...] = (limit, offset)
        else:
            if not self.db.execute(
                "SELECT 1 FROM research_projects WHERE id=?", (research_project_id,)
            ).fetchone():
                raise DocumentError("unknown research project")
            total = int(
                self.db.execute(
                    "SELECT COUNT(*) FROM research_project_documents WHERE project_id=?",
                    (research_project_id,),
                ).fetchone()[0]
            )
            page_from = (
                "FROM documents d JOIN research_project_documents m ON m.document_id=d.id "
                "WHERE m.project_id=? ORDER BY d.display_name,d.id LIMIT ? OFFSET ?"
            )
            page_parameters = (research_project_id, limit, offset)
        records = self.db.execute(
            "WITH page_documents AS (SELECT d.id,d.display_name,d.created_at "
            + page_from
            + ") "
            "SELECT d.id AS document_id,d.display_name,d.created_at,"
            "(SELECT COUNT(*) FROM document_revisions revision_count "
            "WHERE revision_count.document_id=d.id) AS revision_count,"
            "latest.id AS latest_revision_id,latest.created_at AS latest_revision_created_at,"
            "latest.extraction_status AS latest_extraction_status,"
            "latest.mime_type AS latest_mime_type "
            "FROM page_documents d LEFT JOIN document_revisions latest ON latest.id=("
            "SELECT candidate.id FROM document_revisions candidate "
            "WHERE candidate.document_id=d.id "
            "ORDER BY candidate.created_at DESC,candidate.id DESC LIMIT 1) "
            "ORDER BY d.display_name,d.id",
            page_parameters,
        ).fetchall()
        items = [dict(row) for row in records]
        for item in items:
            display_name = _path_safe_display_name(item["display_name"])
            item["display_name"] = display_name or "Unnamed document"
        return {
            "items": items,
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + len(items) < total,
        }

    def document_revision_summary(self, revision_id: str) -> dict[str, object]:
        """Return path-free metadata and revision-local counts for one revision."""
        row = self.db.execute(
            "SELECT r.id AS revision_id,r.document_id,d.display_name,"
            "r.source_hash,r.canonical_hash,"
            "r.extractor,r.mime_type,r.source_size,r.created_at,r.ocr_language,r.ocr_dpi,"
            "(SELECT COUNT(*) FROM pages p WHERE p.revision_id=r.id) AS page_count,"
            "(SELECT COUNT(*) FROM paragraphs p WHERE p.revision_id=r.id) AS paragraph_count "
            "FROM document_revisions r JOIN documents d ON d.id=r.document_id "
            "WHERE r.id=?",
            (revision_id,),
        ).fetchone()
        if row is None:
            raise DocumentError("unknown revision")
        result = dict(row)
        display_name = _path_safe_display_name(result["display_name"])
        result["display_name"] = display_name or "Unnamed document"
        return result

    @staticmethod
    def _remove_uncommitted_source_files(files: Iterable[Path]) -> None:
        """Remove only source files created by an import whose DB write failed."""
        for path in reversed(list(files)):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                _LOGGER.warning("could not remove uncommitted project source file", exc_info=True)

    @staticmethod
    def _verify_directory_handle(parent_fd: int, name: str, directory_fd: int) -> None:
        """Ensure a managed directory entry still names the opened directory."""
        try:
            entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            opened = os.fstat(directory_fd)
        except OSError as exc:
            raise DocumentError("stored document path is unsafe") from exc
        if not stat.S_ISDIR(entry.st_mode) or (entry.st_dev, entry.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise DocumentError("stored document path is unsafe")

    def _verify_source_directory_handles(
        self, root_fd: int, hidden_fd: int, source_fd: int
    ) -> None:
        """Verify root-relative managed directory entries against pinned handles."""
        try:
            root_entry = self.root.stat(follow_symlinks=False)
            opened_root = os.fstat(root_fd)
        except OSError as exc:
            raise DocumentError("stored document path is unsafe") from exc
        if not stat.S_ISDIR(root_entry.st_mode) or (root_entry.st_dev, root_entry.st_ino) != (
            opened_root.st_dev,
            opened_root.st_ino,
        ):
            raise DocumentError("stored document path is unsafe")
        self._verify_directory_handle(root_fd, ".discourselens", hidden_fd)
        self._verify_directory_handle(hidden_fd, "source", source_fd)

    def _open_managed_source_directory(self) -> tuple[int, int, int] | None:
        """Pin the managed source directory for race-safe source-file creation.

        Platforms without ``openat``-style directory descriptors retain the
        path-based fallback, with the existing component validation.
        """
        if not _SUPPORTS_DIR_FD_OPEN or not hasattr(os, "O_DIRECTORY"):
            return None
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        opened: list[int] = []
        try:
            root_fd = os.open(self.root, directory_flags)
            opened.append(root_fd)
            hidden_fd = os.open(".discourselens", directory_flags, dir_fd=root_fd)
            opened.append(hidden_fd)
            source_fd = os.open("source", directory_flags, dir_fd=hidden_fd)
            opened.append(source_fd)
            self._verify_source_directory_handles(root_fd, hidden_fd, source_fd)
            return root_fd, hidden_fd, source_fd
        except DocumentError:
            for descriptor in reversed(opened):
                os.close(descriptor)
            raise
        except OSError as exc:
            for descriptor in reversed(opened):
                os.close(descriptor)
            raise DocumentError("stored document path is unsafe") from exc

    @staticmethod
    def _write_source_entry(
        source_fd: int,
        filename: str,
        content: str | bytes,
        created_files: list[str],
    ) -> None:
        """Create one source file relative to an already-validated directory handle."""
        if not filename or Path(filename).name != filename or filename in {".", ".."}:
            raise DocumentError("stored document path is invalid")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(filename, flags, 0o600, dir_fd=source_fd)
        created_files.append(filename)
        Project._write_source_content(descriptor, content)

    @staticmethod
    def _write_source_content(descriptor: int, content: str | bytes) -> None:
        """Write content to an exclusively created source file and close it."""
        try:
            # Binary writes preserve the exact UTF-8 bytes that the database hashes;
            # Windows text mode would otherwise translate LF into CRLF.
            encoded = content if isinstance(content, bytes) else content.encode("utf-8")
            stream = os.fdopen(descriptor, "wb")
            descriptor = -1
            with stream:
                stream.write(encoded)
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _remove_uncommitted_source_entries(source_fd: int, filenames: Iterable[str]) -> None:
        """Remove only files created through a pinned source directory handle."""
        for filename in reversed(list(filenames)):
            try:
                os.unlink(filename, dir_fd=source_fd)
            except FileNotFoundError:
                continue
            except OSError:
                _LOGGER.warning("could not remove uncommitted project source file", exc_info=True)

    @staticmethod
    def _close_managed_source_directory(directory_fds: tuple[int, int, int]) -> None:
        for descriptor in reversed(directory_fds):
            try:
                os.close(descriptor)
            except OSError:
                _LOGGER.warning("could not close project source directory", exc_info=True)

    def original_bytes(self, revision_id: str) -> bytes:
        row = self.db.execute(
            "SELECT original_path,source_hash FROM document_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        if not row:
            raise DocumentError("unknown revision")
        if not row[0]:
            raise DocumentError("original source is unavailable for this revision")
        content = self._read_managed_source_file(row[0])
        if digest(content) != row["source_hash"]:
            raise DocumentError("stored original source integrity check failed")
        return content

    def _resolve_managed_source_file(self, stored_path: object, *, must_exist: bool = True) -> Path:
        """Resolve a revision source file without trusting paths from the database.

        Project databases are portable user data and may have been edited or
        tampered with. Revision paths are therefore constrained to one filename
        directly inside the managed ``source`` directory; traversal and symlink
        escapes must not turn document APIs into arbitrary file reads.
        """
        if not isinstance(stored_path, str) or not stored_path or "\x00" in stored_path:
            raise DocumentError("stored document path is invalid")
        # Older projects created on Windows may contain backslash separators.
        # Normalize both separator spellings before validating the logical path.
        normalized = stored_path.replace("\\", "/")
        parts = normalized.split("/")
        if (
            normalized.startswith("/")
            or len(parts) != 2
            or parts[0] != "source"
            or parts[1] in {"", ".", ".."}
            or ":" in parts[1]
        ):
            raise DocumentError("stored document path is invalid")

        hidden = self.hidden
        source_dir = hidden / "source"
        source_file = source_dir / parts[1]
        try:
            if hidden.is_symlink() or source_dir.is_symlink() or source_file.is_symlink():
                raise DocumentError("stored document path is unsafe")
            resolved_hidden = hidden.resolve(strict=True)
            resolved_source = source_dir.resolve(strict=True)
            resolved_file = source_file.resolve(strict=must_exist)
            if (
                not resolved_source.is_relative_to(resolved_hidden)
                or not resolved_file.is_relative_to(resolved_source)
                or (must_exist and not resolved_file.is_file())
            ):
                raise DocumentError("stored document path is unsafe")
        except DocumentError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise DocumentError("stored document source is unavailable") from exc
        return resolved_file

    def _read_managed_source_file(self, stored_path: object) -> bytes:
        """Read one managed source through a checked, non-following file descriptor.

        Path validation followed by ``Path.read_bytes()`` leaves a race in which a
        replaced source entry (or parent directory) can redirect the read. Open the
        validated path without following a final symlink, reject non-regular files
        (including FIFOs), and compare the opened inode with the path before reading.
        The descriptor then remains pinned to that file even if its directory entry
        changes during the read.
        """
        source_file = self._resolve_managed_source_file(stored_path)
        descriptor = -1
        try:
            before = source_file.stat(follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):
                raise DocumentError("stored document path is unsafe")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            descriptor = os.open(source_file, flags)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise DocumentError("stored document path changed during read")

            # Revalidate after open too: this detects a parent-directory swap that
            # happened between path resolution and open, including one restored
            # before this check (the descriptor identity above must still match).
            current = source_file.stat(follow_symlinks=False)
            self._resolve_managed_source_file(stored_path)
            if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise DocumentError("stored document path changed during read")

            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                content = stream.read()
                after = os.fstat(stream.fileno())
            if (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
                raise DocumentError("stored document source changed during read")
            return content
        except DocumentError:
            raise
        except OSError as exc:
            raise DocumentError("stored document source is unavailable or unsafe") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _managed_source_file_size(self, stored_path: object) -> int:
        """Inspect a managed source size without reading its contents."""
        source_file = self._resolve_managed_source_file(stored_path)
        descriptor = -1
        try:
            before = source_file.stat(follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):
                raise DocumentError("stored document path is unsafe")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            descriptor = os.open(source_file, flags)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise DocumentError("stored document path changed during inspection")

            current = source_file.stat(follow_symlinks=False)
            self._resolve_managed_source_file(stored_path)
            if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise DocumentError("stored document path changed during inspection")
            return opened.st_size
        except DocumentError:
            raise
        except OSError as exc:
            raise DocumentError("stored document source is unavailable or unsafe") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def record_extraction_warning(
        self,
        revision_id: str,
        code: str,
        detail: str | None = None,
        *,
        commit: bool = True,
    ) -> str:
        if not self.db.execute(
            "SELECT 1 FROM document_revisions WHERE id=?", (revision_id,)
        ).fetchone():
            raise DocumentError("unknown revision")
        warning_id = uid()
        self.db.execute(
            "INSERT INTO extraction_warnings VALUES(?,?,?,?)",
            (warning_id, revision_id, code, detail),
        )
        if commit:
            self.db.commit()
        return warning_id

    def extraction_warnings(self, revision_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT code,detail FROM extraction_warnings WHERE revision_id=? ORDER BY id",
            (revision_id,),
        ).fetchall()

    def _paragraphs(self, text):
        offset = 0
        line_breaks = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
        for raw_line in text.splitlines(keepends=True):
            content = raw_line
            if content.endswith("\r\n"):
                content = content[:-2]
            elif content.endswith(tuple(line_breaks)):
                content = content[:-1]
            if content.strip():
                yield offset, offset + len(content), content
            offset += len(raw_line)

    def _persist_sentences(self, revision_id: str) -> None:
        """Persist conservative sentence boundaries without changing canonical text."""
        ordinal = 0
        for paragraph in self.db.execute(
            "SELECT id,start_offset,text FROM paragraphs WHERE revision_id=? ORDER BY start_offset",
            (revision_id,),
        ):
            text = paragraph["text"] or ""
            for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text, re.UNICODE):
                sentence = match.group(0).strip()
                if not sentence:
                    continue
                local_start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
                start = paragraph["start_offset"] + local_start
                end = start + len(sentence)
                self.db.execute(
                    "INSERT INTO sentences VALUES(?,?,?,?,?,?,?,?)",
                    (
                        uid(),
                        revision_id,
                        paragraph["id"],
                        ordinal,
                        start,
                        end,
                        sentence,
                        "regex-sentence-v1",
                    ),
                )
                ordinal += 1

    def create_span(self, revision_id: str, start: int, end: int) -> str:
        return self.create_spans(revision_id, [(start, end)])[0]

    def create_spans(
        self,
        revision_id: str,
        ranges: Iterable[tuple[int, int]],
        *,
        canonical_text: str | None = None,
        project_segments: bool = True,
    ) -> list[str]:
        """Create or reuse canonical spans while reading one revision only once.

        SourceSpan identity is the revision plus its half-open canonical range, so
        repeated retrieval runs reuse the same span instead of creating duplicate
        evidence identity. This bulk form also avoids rereading a large revision
        once per match during exhaustive deterministic retrieval.
        """
        selected = list(ranges)
        if not selected:
            raise SourceSpanError("at least one span range is required")
        text = self._validated_revision_text(revision_id, canonical_text)
        return self._persist_span_batch(
            revision_id, selected, text, project_segments=project_segments
        )

    def create_span_batches(
        self,
        revision_id: str,
        ranges: Iterable[tuple[int, int]],
        *,
        canonical_text: str,
        batch_size: int = 512,
    ) -> Iterator[list[str]]:
        """Persist a potentially large span stream in bounded batches.

        The revision text hash is verified once, then only ``batch_size`` ranges
        and IDs are retained at a time. Structural page projection is deferred
        until a returned candidate is inspected or cited.
        """
        if not 1 <= batch_size <= 1_000:
            raise SourceSpanError("span batch size must be between 1 and 1000")
        text = self._validated_revision_text(revision_id, canonical_text)
        iterator = iter(ranges)
        while selected := list(islice(iterator, batch_size)):
            yield self._persist_span_batch(revision_id, selected, text, project_segments=False)

    def _validated_revision_text(self, revision_id: str, canonical_text: str | None) -> str:
        revision = self.db.execute(
            "SELECT canonical_hash FROM document_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        if not revision:
            raise SourceSpanError("unknown document revision")
        text = canonical_text if canonical_text is not None else self.revision_text(revision_id)
        if canonical_text is not None and digest(text.encode()) != revision["canonical_hash"]:
            raise SourceSpanError("provided text does not match the canonical document revision")
        return text

    def _persist_span_batch(
        self,
        revision_id: str,
        selected: list[tuple[int, int]],
        text: str,
        *,
        project_segments: bool,
    ) -> list[str]:
        for start, end in selected:
            # ``bool`` is an ``int`` subclass in Python, and SQLite can also
            # persist non-integral numeric values in INTEGER-affinity columns.
            # Reject both at the SDK boundary so source coordinates remain
            # unambiguously Unicode code-point offsets and failures stay typed.
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end <= start
                or end > len(text)
            ):
                raise SourceSpanError("invalid half-open offsets")
        # Validate before deduplication: Python considers (False, end) equal to
        # (0, end), which would otherwise silently turn an invalid input into a
        # valid span whenever the canonical range appeared first.
        unique_ranges = list(dict.fromkeys(selected))
        self.db.execute("SAVEPOINT create_span_batch")
        try:
            self.db.executemany(
                "INSERT OR IGNORE INTO source_spans VALUES(?,?,?,?,?)",
                [
                    (uid(), revision_id, start, end, digest(text[start:end].encode()))
                    for start, end in unique_ranges
                ],
            )
            ids_by_range: dict[tuple[int, int], str] = {}
            # Keep SQLite parameter usage below older builds' default 999 limit.
            lookup_size = 400
            for index in range(0, len(unique_ranges), lookup_size):
                chunk = unique_ranges[index : index + lookup_size]
                predicate = " OR ".join("(start_offset=? AND end_offset=?)" for _ in chunk)
                parameters: list[int | str] = [revision_id]
                for start, end in chunk:
                    parameters.extend((start, end))
                rows = self.db.execute(
                    "SELECT id,start_offset,end_offset FROM source_spans "
                    f"WHERE revision_id=? AND ({predicate})",
                    parameters,
                ).fetchall()
                ids_by_range.update(
                    {
                        (int(row["start_offset"]), int(row["end_offset"])): str(row["id"])
                        for row in rows
                    }
                )
            if len(ids_by_range) != len(unique_ranges):
                raise SourceSpanError("span persistence failed")
            ids = [ids_by_range[item] for item in selected]
            if (
                project_segments
                and self.db.execute(
                    "SELECT 1 FROM pages WHERE revision_id=?", (revision_id,)
                ).fetchone()
            ):
                for span_id in dict.fromkeys(ids):
                    self.project_span_to_segments(span_id, commit=False)
            self.db.commit()
        except BaseException:
            if self.db.in_transaction:
                try:
                    self.db.execute("ROLLBACK TO SAVEPOINT create_span_batch")
                    self.db.execute("RELEASE SAVEPOINT create_span_batch")
                except sqlite3.Error:
                    self.db.rollback()
            raise
        # Structural projection is derived only from immutable revision records. It
        # remains outside identity and is materialized only when page records exist.
        return ids

    def revision_text(self, rid: str) -> str:
        row = self.db.execute(
            "SELECT text_path,canonical_hash FROM document_revisions WHERE id=?", (rid,)
        ).fetchone()
        if not row:
            raise DocumentError("unknown revision")
        content = self._read_managed_source_file(row["text_path"])
        if digest(content) != row["canonical_hash"]:
            raise DocumentError("stored canonical text integrity check failed")
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentError("stored canonical text is not valid UTF-8") from exc

    def revision_text_size_bytes(self, rid: str) -> int:
        """Return a validated canonical text file size without loading its contents."""
        row = self.db.execute(
            "SELECT text_path FROM document_revisions WHERE id=?", (rid,)
        ).fetchone()
        if not row:
            raise DocumentError("unknown revision")
        return self._managed_source_file_size(row["text_path"])

    def span_text(self, sid: str) -> str:
        row = self.db.execute(
            "SELECT revision_id,start_offset,end_offset FROM source_spans WHERE id=?", (sid,)
        ).fetchone()
        if not row:
            raise SourceSpanError("unknown span")
        return self.revision_text(row[0])[row[1] : row[2]]

    def page(self, revision_id: str, page_index: int) -> sqlite3.Row:
        """Return one persisted page from an immutable revision.

        Page indices are internal zero-based coordinates. Presentation adapters
        must use the stored page label rather than asking callers to supply a
        citation location.
        """
        row = self.db.execute(
            "SELECT * FROM pages WHERE revision_id=? AND page_index=?", (revision_id, page_index)
        ).fetchone()
        if not row:
            raise DocumentError("unknown page in document revision")
        return row

    def paragraph(self, paragraph_id: str) -> sqlite3.Row:
        """Return one immutable structural paragraph by its canonical ID."""
        row = self.db.execute("SELECT * FROM paragraphs WHERE id=?", (paragraph_id,)).fetchone()
        if not row:
            raise DocumentError("unknown paragraph")
        return row

    def span_context(
        self, span_id: str, *, before: int = 200, after: int = 200
    ) -> dict[str, object]:
        """Resolve bounded local context around a canonical evidence span."""
        if before < 0 or after < 0 or before > 10_000 or after > 10_000:
            raise SourceSpanError("context bounds must be between 0 and 10000 characters")
        row = self.db.execute(
            "SELECT revision_id,start_offset,end_offset FROM source_spans WHERE id=?", (span_id,)
        ).fetchone()
        if not row:
            raise SourceSpanError("unknown span")
        text = self.revision_text(row["revision_id"])
        start = max(0, row["start_offset"] - before)
        end = min(len(text), row["end_offset"] + after)
        return {
            "span_id": span_id,
            "revision_id": row["revision_id"],
            "context_start": start,
            "context_end": end,
            "before": text[start : row["start_offset"]],
            "exact_text": text[row["start_offset"] : row["end_offset"]],
            "after": text[row["end_offset"] : end],
        }

    def textual_evidence(self, span_id: str, *, provenance: dict[str, object] | None = None) -> str:
        """Persist a textual evidence artifact with optional acquisition provenance."""
        if not self.db.execute("SELECT 1 FROM source_spans WHERE id=?", (span_id,)).fetchone():
            raise EvidenceValidationError("unknown span")
        eid = uid()
        self.db.execute(
            "INSERT INTO evidence_artifacts VALUES(?,?,?,?,?)",
            (
                eid,
                EvidenceKind.TEXTUAL,
                span_id,
                json.dumps(provenance or {}, ensure_ascii=False, sort_keys=True),
                now(),
            ),
        )
        self.db.commit()
        return eid

    def external_source_evidence(
        self,
        source: str,
        excerpt: str,
        *,
        retrieved_at: str,
        url: str | None = None,
        title: str | None = None,
    ) -> str:
        """Persist user-supplied external context without treating it as corpus text.

        This operation never fetches a URL. The caller supplies the exact excerpt
        and provenance, allowing a later report to distinguish supplied external
        context from the project corpus.
        """
        self._require_writable()
        if not source.strip() or not excerpt.strip() or not retrieved_at.strip():
            raise EvidenceValidationError(
                "external evidence requires source, excerpt, and retrieval timestamp"
            )
        if len(source) > MAX_CONTEXT_LABEL_CHARS:
            raise EvidenceValidationError(
                f"external source label exceeds the {MAX_CONTEXT_LABEL_CHARS}-character limit"
            )
        if len(excerpt) > MAX_CONTEXT_EVIDENCE_CHARS:
            raise EvidenceValidationError(
                f"external source excerpt exceeds the {MAX_CONTEXT_EVIDENCE_CHARS}-character limit"
            )
        if len(retrieved_at) > MAX_EXTERNAL_SOURCE_TIMESTAMP_CHARS:
            raise EvidenceValidationError(
                "external source retrieval timestamp exceeds the supported length"
            )
        if title is not None and len(title) > MAX_CONTEXT_LABEL_CHARS:
            raise EvidenceValidationError(
                f"external source title exceeds the {MAX_CONTEXT_LABEL_CHARS}-character limit"
            )
        if url is not None and (not url.strip() or len(url) > MAX_EXTERNAL_SOURCE_URL_CHARS):
            raise EvidenceValidationError(
                f"external evidence URL must be nonblank and at most {MAX_EXTERNAL_SOURCE_URL_CHARS} characters"
            )
        evidence_id = uid()
        payload = {
            "source": source.strip(),
            "title": title.strip() if title else None,
            "url": url.strip() if url else None,
            "retrieved_at": retrieved_at.strip(),
            "excerpt": excerpt,
        }
        self.db.execute(
            "INSERT INTO evidence_artifacts VALUES(?,?,?,?,?)",
            (
                evidence_id,
                EvidenceKind.EXTERNAL_SOURCE,
                None,
                json.dumps(payload, ensure_ascii=False),
                now(),
            ),
        )
        self.db.commit()
        return evidence_id

    def user_context_evidence(self, label: str, content: str) -> str:
        """Persist researcher-provided context with an explicit, non-corpus label."""
        self._require_writable()
        if not label.strip() or len(label) > MAX_CONTEXT_LABEL_CHARS:
            raise EvidenceValidationError(
                f"user context label must be nonblank and at most {MAX_CONTEXT_LABEL_CHARS} characters"
            )
        if not content.strip():
            raise EvidenceValidationError("user context evidence requires nonblank content")
        if len(content) > MAX_CONTEXT_EVIDENCE_CHARS:
            raise EvidenceValidationError(
                f"user context content exceeds the {MAX_CONTEXT_EVIDENCE_CHARS}-character limit"
            )
        evidence_id = uid()
        payload = {"label": label.strip(), "content": content}
        self.db.execute(
            "INSERT INTO evidence_artifacts VALUES(?,?,?,?,?)",
            (
                evidence_id,
                EvidenceKind.USER_CONTEXT,
                None,
                json.dumps(payload, ensure_ascii=False),
                now(),
            ),
        )
        self.db.commit()
        return evidence_id

    def evidence_content(self, evidence_id: str) -> dict[str, object]:
        row = self.db.execute(
            "SELECT content_json FROM evidence_artifacts WHERE id=?", (evidence_id,)
        ).fetchone()
        if not row:
            raise EvidenceValidationError("unknown evidence")
        try:
            payload = json.loads(row["content_json"])
        except json.JSONDecodeError as exc:
            raise EvidenceValidationError("evidence metadata is corrupt") from exc
        if not isinstance(payload, dict):
            raise EvidenceValidationError("evidence metadata must be an object")
        return payload

    def render_citation(self, eid: str) -> str:
        evidence = self.db.execute(
            "SELECT kind FROM evidence_artifacts WHERE id=?", (eid,)
        ).fetchone()
        if not evidence:
            raise EvidenceValidationError("unknown evidence")
        if evidence["kind"] == EvidenceKind.EXTERNAL_SOURCE:
            payload = self.evidence_content(eid)
            source = str(payload.get("title") or payload.get("source") or "external source")
            retrieved = str(payload.get("retrieved_at") or "unknown retrieval date")
            return f"[{source} · retrieved {retrieved}]"
        if evidence["kind"] == EvidenceKind.USER_CONTEXT:
            payload = self.evidence_content(eid)
            return f"[user context · {payload.get('label', 'unlabelled')}]"
        if evidence["kind"] == EvidenceKind.CORPUS_STATISTIC:
            payload = self.evidence_content(eid)
            provenance = payload.get("provenance")
            if isinstance(provenance, dict) and provenance.get("comparison_run_id"):
                selected_row = provenance.get("selected_row")
                comparison_id = provenance.get("comparison_run_id")
                metric = provenance.get("comparison_metric")
                if (
                    not isinstance(selected_row, dict)
                    or not isinstance(selected_row.get("token"), str)
                    or not isinstance(comparison_id, str)
                    or metric not in {"frequency", "keyness"}
                ):
                    raise EvidenceValidationError("comparison citation provenance is invalid")
                return f"[comparison · {metric} · {selected_row['token']} · {comparison_id}]"
            return f"[corpus statistic · {payload.get('label', 'unlabelled')}]"
        span_id = self.db.execute(
            "SELECT span_id FROM evidence_artifacts WHERE id=?", (eid,)
        ).fetchone()
        if not span_id or not span_id[0]:
            raise EvidenceValidationError("textual evidence required")
        return self.render_span_location(span_id[0])

    def render_span_location(self, span_id: str) -> str:
        """Render a canonical source location from a SourceSpan, never caller text."""
        row = self.db.execute(
            "SELECT d.display_name,p.ordinal FROM source_spans ss "
            "JOIN document_revisions r ON ss.revision_id=r.id "
            "JOIN documents d ON r.document_id=d.id "
            "LEFT JOIN paragraphs p ON p.revision_id=r.id "
            "AND ss.start_offset>=p.start_offset AND ss.start_offset<p.end_offset "
            "WHERE ss.id=?",
            (span_id,),
        ).fetchone()
        if not row:
            raise SourceSpanError("unknown span")
        self.project_span_to_segments(span_id)
        pages = self.db.execute(
            "SELECT DISTINCT p.page_label,p.page_index FROM source_span_segments s "
            "JOIN pages p ON s.page_number=p.id WHERE s.span_id=? ORDER BY p.page_index",
            (span_id,),
        ).fetchall()
        if pages:
            labels = [item[0] for item in pages]
            location = f"p. {labels[0]}" if len(labels) == 1 else f"pp. {labels[0]}–{labels[-1]}"
            return f"[{row[0]} · {location}]"
        return f"[{row[0]} · ¶{(row[1] or 0) + 1}]"

    def snapshot(self, revisions: list[str]) -> str:
        if not revisions:
            raise ScopeViolationError("scope is required")
        if len(set(revisions)) != len(revisions):
            raise ScopeViolationError("scope cannot contain duplicate document revisions")
        found: set[str] = set()
        for batch in _in_parameter_batches(revisions):
            placeholders = ",".join("?" for _ in batch)
            found.update(
                str(row[0])
                for row in self.db.execute(
                    f"SELECT id FROM document_revisions WHERE id IN ({placeholders})", batch
                )
            )
        missing = [revision_id for revision_id in revisions if revision_id not in found]
        if missing:
            raise ScopeViolationError("scope references an unknown document revision")
        sid = uid()
        self.db.execute("INSERT INTO scope_snapshots VALUES(?,?)", (sid, json.dumps(revisions)))
        self.db.commit()
        return sid

    def create_comparison_run(
        self,
        left_scope_id: str,
        right_scope_id: str,
        *,
        metric: str,
        calculation_version: str,
        config: dict[str, object],
        rows: list[dict[str, float | int | str]],
    ) -> str:
        """Persist an immutable deterministic comparison of two frozen scopes.

        Comparison provenance is intentionally separate from an AnalysisRun: a
        single-run Finding has one original scope, while corpus comparison has
        two independent populations that must remain visible to researchers.
        """
        if left_scope_id == right_scope_id:
            raise ScopeViolationError("comparison requires two distinct scope snapshots")
        self.scope_revisions(left_scope_id)
        self.scope_revisions(right_scope_id)
        if not metric or not calculation_version:
            raise DocumentError("comparison metric and calculation version are required")
        config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
        result_json = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        comparison_id = uid()
        self.db.execute(
            "INSERT INTO comparison_runs "
            "(id,left_scope_id,right_scope_id,metric,calculation_version,config_json,"
            "result_json,result_hash,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                comparison_id,
                left_scope_id,
                right_scope_id,
                metric,
                calculation_version,
                config_json,
                result_json,
                digest(result_json.encode("utf-8")),
                now(),
            ),
        )
        self.db.commit()
        return comparison_id

    def create_comparison_run_stream(
        self,
        left_scope_id: str,
        right_scope_id: str,
        *,
        metric: str,
        calculation_version: str,
        config: dict[str, object],
        rows: Iterable[dict[str, float | int | str]],
    ) -> str:
        """Persist ranked comparison rows incrementally without a corpus-sized JSON value.

        The caller may yield rows in any order. SQLite stores typed row records and
        ranks them by absolute metric value with a stable lexical tie-break. The
        integrity hash covers the canonical ordered JSON-array representation used
        by legacy comparison runs.
        """
        if left_scope_id == right_scope_id:
            raise ScopeViolationError("comparison requires two distinct scope snapshots")
        self.scope_revisions(left_scope_id)
        self.scope_revisions(right_scope_id)
        if metric not in {"frequency", "keyness"} or not calculation_version:
            raise DocumentError("comparison metric and calculation version are required")
        config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
        comparison_id = uid()
        savepoint_open = False
        try:
            self.db.execute("SAVEPOINT comparison_stream")
            savepoint_open = True
            self.db.execute(
                "INSERT INTO comparison_runs "
                "(id,left_scope_id,right_scope_id,metric,calculation_version,config_json,"
                "result_json,result_hash,created_at,storage_kind) "
                "VALUES(?,?,?,?,?,?, '[]','',?,'rows-v1')",
                (
                    comparison_id,
                    left_scope_id,
                    right_scope_id,
                    metric,
                    calculation_version,
                    config_json,
                    now(),
                ),
            )
            self.db.executemany(
                "INSERT INTO comparison_result_rows(comparison_id,token,rank_value,row_json) "
                "VALUES(?,?,?,?)",
                self._comparison_storage_values(comparison_id, metric, rows),
            )
            stored = self.db.execute(
                "SELECT token,rank_value,row_json FROM comparison_result_rows "
                "WHERE comparison_id=? ORDER BY rank_value DESC,token COLLATE BINARY ASC",
                (comparison_id,),
            )
            hasher = hashlib.sha256()
            hasher.update(b"[")
            count = 0
            for stored_row in stored:
                if count:
                    hasher.update(b",")
                hasher.update(str(stored_row["row_json"]).encode("utf-8"))
                count += 1
            hasher.update(b"]")
            if count == 0:
                raise DocumentError("comparison result cannot be empty")
            self.db.execute(
                "UPDATE comparison_runs SET result_hash=? WHERE id=?",
                (hasher.hexdigest(), comparison_id),
            )
            self.db.execute("RELEASE SAVEPOINT comparison_stream")
            savepoint_open = False
            self.db.commit()
            return comparison_id
        except Exception:
            if savepoint_open:
                self.db.execute("ROLLBACK TO SAVEPOINT comparison_stream")
                self.db.execute("RELEASE SAVEPOINT comparison_stream")
            self.db.rollback()
            raise

    @staticmethod
    def _comparison_storage_values(
        comparison_id: str,
        metric: str,
        rows: Iterable[dict[str, float | int | str]],
    ) -> Iterable[tuple[str, str, float, str]]:
        expected = (
            {"token", "left_count", "right_count", "left_rate", "right_rate", "rate_difference"}
            if metric == "frequency"
            else {"token", "left_count", "right_count", "log_likelihood"}
        )
        rank_key = "rate_difference" if metric == "frequency" else "log_likelihood"
        for row in rows:
            if set(row) != expected:
                raise DocumentError("comparison row has an invalid schema")
            token = row.get("token")
            left_count = row.get("left_count")
            right_count = row.get("right_count")
            score = row.get(rank_key)
            if (
                not isinstance(token, str)
                or not token
                or not isinstance(left_count, int)
                or isinstance(left_count, bool)
                or left_count < 0
                or not isinstance(right_count, int)
                or isinstance(right_count, bool)
                or right_count < 0
                or not isinstance(score, (float, int))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
            ):
                raise DocumentError("comparison row contains invalid values")
            try:
                row_json = json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise DocumentError("comparison row cannot be serialized") from exc
            yield comparison_id, token, abs(float(score)), row_json

    @staticmethod
    def _validate_stored_comparison_row(row: sqlite3.Row, metric: str) -> dict[str, object]:
        try:
            parsed = json.loads(row["row_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise EvidenceValidationError("comparison result row is corrupt") from exc
        expected = (
            {"token", "left_count", "right_count", "left_rate", "right_rate", "rate_difference"}
            if metric == "frequency"
            else {"token", "left_count", "right_count", "log_likelihood"}
        )
        if (
            not isinstance(parsed, dict)
            or set(parsed) != expected
            or parsed.get("token") != row["token"]
        ):
            raise EvidenceValidationError("comparison result row has invalid structure")
        for count_key in ("left_count", "right_count"):
            count_value = parsed.get(count_key)
            if not isinstance(count_value, int) or isinstance(count_value, bool) or count_value < 0:
                raise EvidenceValidationError("comparison population counts are invalid")
        score_key = "rate_difference" if metric == "frequency" else "log_likelihood"
        score = parsed.get(score_key)
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or abs(float(score)) != float(row["rank_value"])
        ):
            raise EvidenceValidationError("comparison result ranking is corrupt")
        if metric == "frequency":
            for rate_key in ("left_rate", "right_rate"):
                rate = parsed.get(rate_key)
                if (
                    not isinstance(rate, (int, float))
                    or isinstance(rate, bool)
                    or not math.isfinite(float(rate))
                ):
                    raise EvidenceValidationError("comparison result rate is corrupt")
            return {
                "token": parsed["token"],
                "left_count": parsed["left_count"],
                "right_count": parsed["right_count"],
                "left_rate": parsed["left_rate"],
                "right_rate": parsed["right_rate"],
                "rate_difference": parsed["rate_difference"],
            }
        return {
            "token": parsed["token"],
            "left_count": parsed["left_count"],
            "right_count": parsed["right_count"],
            "log_likelihood": parsed["log_likelihood"],
        }

    def comparison_rows_page(
        self, comparison_id: str, *, offset: int = 0, limit: int = 100
    ) -> dict[str, object]:
        """Integrity-check and return one bounded page of normalized comparison rows."""
        if offset < 0 or not 1 <= limit <= 1_000:
            raise EvidenceValidationError(
                "comparison page requires non-negative offset and limit 1..1000"
            )
        comparison = self.comparison_run(comparison_id)
        if comparison["storage_kind"] != "rows-v1":
            raise DocumentError("comparison does not use normalized row storage")
        cursor = self.db.execute(
            "SELECT token,rank_value,row_json FROM comparison_result_rows WHERE comparison_id=? "
            "ORDER BY rank_value DESC,token COLLATE BINARY ASC",
            (comparison_id,),
        )
        hasher = hashlib.sha256()
        hasher.update(b"[")
        total = 0
        rows: list[dict[str, object]] = []
        for item in cursor:
            if total:
                hasher.update(b",")
            hasher.update(str(item["row_json"]).encode("utf-8"))
            if offset <= total < offset + limit:
                rows.append(self._validate_stored_comparison_row(item, str(comparison["metric"])))
            total += 1
        hasher.update(b"]")
        if hasher.hexdigest() != comparison["result_hash"]:
            raise EvidenceValidationError("comparison result hash mismatch")
        try:
            populations = self.db.execute(
                "SELECT COUNT(*),"
                "COALESCE(SUM(CAST(json_extract(row_json,'$.left_count') AS INTEGER)),0),"
                "COALESCE(SUM(CAST(json_extract(row_json,'$.right_count') AS INTEGER)),0) "
                "FROM comparison_result_rows WHERE comparison_id=?",
                (comparison_id,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            raise EvidenceValidationError("comparison result rows are corrupt") from exc
        if not populations or int(populations[0]) != total:
            raise EvidenceValidationError("comparison result row count is inconsistent")
        left_population, right_population = int(populations[1]), int(populations[2])
        return {
            "rows": rows,
            "total_rows": total,
            "left_population": left_population,
            "right_population": right_population,
            "result_hash_verified": True,
        }

    def comparison_rows_all(self, comparison_id: str) -> dict[str, object]:
        """Explicitly materialize a verified full comparison for local archival use."""
        comparison = self.comparison_run(comparison_id)
        if comparison["storage_kind"] != "rows-v1":
            raise DocumentError("comparison does not use normalized row storage")
        cursor = self.db.execute(
            "SELECT token,rank_value,row_json FROM comparison_result_rows WHERE comparison_id=? "
            "ORDER BY rank_value DESC,token COLLATE BINARY ASC",
            (comparison_id,),
        )
        rows: list[dict[str, object]] = []
        hasher = hashlib.sha256()
        hasher.update(b"[")
        for item in cursor:
            parsed = self._validate_stored_comparison_row(item, str(comparison["metric"]))
            if rows:
                hasher.update(b",")
            hasher.update(str(item["row_json"]).encode("utf-8"))
            rows.append(parsed)
        hasher.update(b"]")
        if hasher.hexdigest() != comparison["result_hash"]:
            raise EvidenceValidationError("comparison result hash mismatch")
        return {"rows": rows, "result_hash_verified": True}

    def comparison_run(self, comparison_id: str) -> sqlite3.Row:
        row = self.db.execute(
            "SELECT * FROM comparison_runs WHERE id=?", (comparison_id,)
        ).fetchone()
        if not row:
            raise DocumentError("unknown comparison run")
        return row

    def create_codebook(self, name: str, description: str | None = None) -> str:
        """Create a local qualitative codebook; names remain unique per project."""
        if not name.strip():
            raise DocumentError("codebook name is required")
        codebook_id = uid()
        try:
            self.db.execute(
                "INSERT INTO codebooks VALUES(?,?,?,?)",
                (codebook_id, name.strip(), description, now()),
            )
        except sqlite3.IntegrityError as exc:
            raise DocumentError("codebook name already exists") from exc
        self.db.commit()
        return codebook_id

    def codebook_pages(self, *, offset: int = 0, limit: int = MAX_CODEBOOK_PAGE_SIZE) -> dict[str, object]:
        """Return one bounded, deterministic page of project codebooks."""
        if (
            type(offset) is not int
            or not 0 <= offset <= 0x7FFF_FFFF_FFFF_FFFF
            or type(limit) is not int
            or not 1 <= limit <= MAX_CODEBOOK_PAGE_SIZE
        ):
            raise DocumentError(
                "codebook page requires an integer offset in SQLite range "
                f"and integer limit 1..{MAX_CODEBOOK_PAGE_SIZE}"
            )

        total = int(self.db.execute("SELECT COUNT(*) FROM codebooks").fetchone()[0])
        records = self.db.execute(
            "SELECT id,name,description,created_at FROM codebooks "
            "ORDER BY name COLLATE BINARY,id COLLATE BINARY LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        items = [dict(row) for row in records]
        return {
            "items": items,
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + len(items) < total,
        }

    def codes_for_codebook_page(
        self,
        codebook_id: str,
        *,
        offset: int = 0,
        limit: int = MAX_CODEBOOK_PAGE_SIZE,
    ) -> dict[str, object]:
        """Return all codes for one codebook, retaining unused codes and parent links."""
        if (
            type(offset) is not int
            or not 0 <= offset <= 0x7FFF_FFFF_FFFF_FFFF
            or type(limit) is not int
            or not 1 <= limit <= MAX_CODEBOOK_PAGE_SIZE
        ):
            raise DocumentError(
                "code page requires an integer offset in SQLite range "
                f"and integer limit 1..{MAX_CODEBOOK_PAGE_SIZE}"
            )
        if not self.db.execute("SELECT 1 FROM codebooks WHERE id=?", (codebook_id,)).fetchone():
            raise DocumentError("unknown codebook")

        total = int(
            self.db.execute("SELECT COUNT(*) FROM codes WHERE codebook_id=?", (codebook_id,)).fetchone()[0]
        )
        records = self.db.execute(
            "SELECT id,codebook_id,parent_code_id,label,definition,created_at FROM codes "
            "WHERE codebook_id=? ORDER BY label COLLATE BINARY,id COLLATE BINARY LIMIT ? OFFSET ?",
            (codebook_id, limit, offset),
        ).fetchall()
        items = [dict(row) for row in records]
        return {
            "items": items,
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + len(items) < total,
        }

    def create_code(
        self,
        codebook_id: str,
        label: str,
        definition: str | None = None,
        parent_code_id: str | None = None,
    ) -> str:
        """Create a controlled code, optionally nested under a code in its codebook."""
        if not label.strip():
            raise DocumentError("code label is required")
        if not self.db.execute("SELECT 1 FROM codebooks WHERE id=?", (codebook_id,)).fetchone():
            raise DocumentError("unknown codebook")
        if parent_code_id is not None:
            parent = self.db.execute(
                "SELECT codebook_id FROM codes WHERE id=?", (parent_code_id,)
            ).fetchone()
            if not parent or parent["codebook_id"] != codebook_id:
                raise DocumentError("parent code must belong to the same codebook")
        code_id = uid()
        try:
            self.db.execute(
                "INSERT INTO codes VALUES(?,?,?,?,?,?)",
                (code_id, codebook_id, parent_code_id, label.strip(), definition, now()),
            )
        except sqlite3.IntegrityError as exc:
            raise DocumentError("code label already exists in this codebook") from exc
        self.db.commit()
        return code_id

    def annotate_span(self, code_id: str, span_id: str, memo: str | None = None) -> str:
        """Attach a human code to canonical text without copying or rewriting it."""
        if not self.db.execute("SELECT 1 FROM codes WHERE id=?", (code_id,)).fetchone():
            raise DocumentError("unknown code")
        if not self.db.execute(
            "SELECT 1 FROM source_spans JOIN document_revisions "
            "ON document_revisions.id=source_spans.revision_id WHERE source_spans.id=?",
            (span_id,),
        ).fetchone():
            raise SourceSpanError("unknown span")
        annotation_id = uid()
        try:
            self.db.execute(
                "INSERT INTO annotations VALUES(?,?,?,?,?)",
                (annotation_id, code_id, span_id, memo, now()),
            )
        except sqlite3.IntegrityError as exc:
            raise DocumentError("this code is already attached to the source span") from exc
        self.db.commit()
        return annotation_id

    def _coding_scope_revisions(self, scope_id: str) -> list[str]:
        """Resolve a coding scope only while every selected revision still exists."""
        revision_ids = self.scope_revisions(scope_id)
        known: set[str] = set()
        for batch in _in_parameter_batches(revision_ids):
            placeholders = ",".join("?" for _ in batch)
            known.update(
                str(row[0])
                for row in self.db.execute(
                    f"SELECT id FROM document_revisions WHERE id IN ({placeholders})", batch
                )
            )
        if len(known) != len(revision_ids):
            raise ScopeViolationError("scope references an unknown document revision")
        return revision_ids

    def annotations_for_scope(self, scope_id: str) -> list[sqlite3.Row]:
        """Read annotations only through an explicit immutable document scope."""
        revision_ids = self._coding_scope_revisions(scope_id)
        rows: list[sqlite3.Row] = []
        for batch in _in_parameter_batches(revision_ids):
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                self.db.execute(
                    "SELECT annotations.id,annotations.memo,annotations.created_at,codes.id AS code_id,"
                    "codes.label,codes.definition,codes.codebook_id,source_spans.id AS span_id,"
                    "source_spans.revision_id,source_spans.start_offset,source_spans.end_offset "
                    "FROM annotations JOIN codes ON codes.id=annotations.code_id "
                    "JOIN source_spans ON source_spans.id=annotations.span_id "
                    f"WHERE source_spans.revision_id IN ({placeholders}) "
                    "ORDER BY source_spans.revision_id,source_spans.start_offset,"
                    "annotations.created_at,annotations.id",
                    batch,
                ).fetchall()
            )
        rows.sort(
            key=lambda row: (
                row["revision_id"],
                row["start_offset"],
                row["created_at"],
                row["id"],
            )
        )
        return rows

    def annotation_pages_for_revisions(
        self, revision_ids: list[str], *, offset: int = 0, limit: int = 100
    ) -> dict[str, object]:
        """Read one bounded annotation page for explicit immutable revisions.

        Unlike :meth:`annotations_for_scope`, this adapter-facing form does not
        create and persist a new ScopeSnapshot merely to read a page.
        """
        if (
            type(offset) is not int
            or not 0 <= offset <= 0x7FFF_FFFF_FFFF_FFFF
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise DocumentError(
                "annotation page requires an integer offset in SQLite range "
                "and integer limit 1..100"
            )
        if (
            not isinstance(revision_ids, list)
            or not revision_ids
            or any(type(revision_id) is not str or not revision_id for revision_id in revision_ids)
        ):
            raise ScopeViolationError("scope requires a non-empty list of document revision IDs")
        if len(set(revision_ids)) != len(revision_ids):
            raise ScopeViolationError("scope cannot contain duplicate document revisions")

        revisions_json = json.dumps(revision_ids)
        known_count = int(
            self.db.execute(
                "SELECT COUNT(*) FROM document_revisions "
                "WHERE id IN (SELECT value FROM json_each(?))",
                (revisions_json,),
            ).fetchone()[0]
        )
        if known_count != len(revision_ids):
            raise ScopeViolationError("scope references an unknown document revision")

        revision_filter = "source_spans.revision_id IN (SELECT value FROM json_each(?))"
        joins = (
            "FROM annotations JOIN codes ON codes.id=annotations.code_id "
            "JOIN source_spans ON source_spans.id=annotations.span_id "
        )
        total = int(
            self.db.execute(
                f"SELECT COUNT(*) {joins} WHERE {revision_filter}", (revisions_json,)
            ).fetchone()[0]
        )
        records = self.db.execute(
            "SELECT annotations.id,annotations.memo,annotations.created_at,codes.id AS code_id,"
            "codes.label,codes.definition,codes.codebook_id,source_spans.id AS span_id,"
            "source_spans.revision_id,source_spans.start_offset,source_spans.end_offset "
            f"{joins}WHERE {revision_filter} "
            "ORDER BY source_spans.revision_id,source_spans.start_offset,"
            "annotations.created_at,annotations.id LIMIT ? OFFSET ?",
            (revisions_json, limit, offset),
        ).fetchall()
        items = [dict(row) for row in records]
        return {
            "items": items,
            "revision_count": len(revision_ids),
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + len(items) < total,
        }

    def create_research_memo(self, scope_id: str, title: str, content: str) -> str:
        """Persist a human memo against an immutable scope, not a mutable project view."""
        self._coding_scope_revisions(scope_id)
        if not title.strip() or not content.strip():
            raise DocumentError("research memo requires a title and content")
        memo_id = uid()
        self.db.execute(
            "INSERT INTO research_memos VALUES(?,?,?,?,?)",
            (memo_id, scope_id, title.strip(), content, now()),
        )
        self.db.commit()
        return memo_id

    def research_memos(self, scope_id: str) -> list[sqlite3.Row]:
        """Read only memos that explicitly name the requested historical scope."""
        self._coding_scope_revisions(scope_id)
        return self.db.execute(
            "SELECT id,scope_id,title,content,created_at FROM research_memos "
            "WHERE scope_id=? ORDER BY created_at,id",
            (scope_id,),
        ).fetchall()

    def research_memos_page(
        self, scope_id: str, *, offset: int = 0, limit: int = MAX_RESEARCH_MEMO_PAGE_SIZE
    ) -> dict[str, object]:
        """Read a bounded page of memos attached to one valid frozen scope."""
        if (
            type(offset) is not int
            or not 0 <= offset <= 0x7FFF_FFFF_FFFF_FFFF
            or type(limit) is not int
            or not 1 <= limit <= MAX_RESEARCH_MEMO_PAGE_SIZE
        ):
            raise DocumentError(
                "research memo page requires an integer offset in SQLite range "
                f"and integer limit 1..{MAX_RESEARCH_MEMO_PAGE_SIZE}"
            )

        self._coding_scope_revisions(scope_id)
        total_count = int(
            self.db.execute(
                "SELECT COUNT(*) FROM research_memos WHERE scope_id=?", (scope_id,)
            ).fetchone()[0]
        )
        records = self.db.execute(
            "SELECT id,scope_id,title,content,created_at FROM research_memos "
            "WHERE scope_id=? ORDER BY created_at,id LIMIT ? OFFSET ?",
            (scope_id, limit, offset),
        ).fetchall()
        rows: list[dict[str, object]] = []
        for record in records:
            title = str(record["title"])
            content = str(record["content"])
            title_characters = len(title)
            content_characters = len(content)
            rows.append(
                {
                    "id": str(record["id"]),
                    "scope_id": str(record["scope_id"]),
                    "title": title[:MAX_RESEARCH_MEMO_TITLE_CHARS],
                    "content": content[:MAX_RESEARCH_MEMO_TEXT_CHARS],
                    "created_at": str(record["created_at"]),
                    "title_characters": title_characters,
                    "content_characters": content_characters,
                    "title_truncated": title_characters > MAX_RESEARCH_MEMO_TITLE_CHARS,
                    "content_truncated": content_characters > MAX_RESEARCH_MEMO_TEXT_CHARS,
                }
            )
        return {
            "scope_id": scope_id,
            "offset": offset,
            "limit": limit,
            "total_count": total_count,
            "has_more": offset + len(rows) < total_count,
            "rows": rows,
        }

    def create_corpus(self, name: str, description: str | None = None) -> str:
        if not name.strip():
            raise DocumentError("corpus name is required")
        corpus_id = uid()
        try:
            self.db.execute(
                "INSERT INTO corpora VALUES(?,?,?,?)", (corpus_id, name.strip(), description, now())
            )
        except sqlite3.IntegrityError as exc:
            raise DocumentError("corpus name already exists") from exc
        self.db.commit()
        return corpus_id

    def add_revision_to_corpus(self, corpus_id: str, revision_id: str) -> None:
        if not self.db.execute("SELECT 1 FROM corpora WHERE id=?", (corpus_id,)).fetchone():
            raise DocumentError("unknown corpus")
        if not self.db.execute(
            "SELECT 1 FROM document_revisions WHERE id=?", (revision_id,)
        ).fetchone():
            raise DocumentError("unknown revision")
        self.db.execute(
            "INSERT OR IGNORE INTO corpus_memberships VALUES(?,?,?)",
            (corpus_id, revision_id, now()),
        )
        self.db.commit()

    def corpus_snapshot(self, corpus_id: str) -> str:
        rows = self.db.execute(
            "SELECT revision_id FROM corpus_memberships WHERE corpus_id=? ORDER BY revision_id",
            (corpus_id,),
        ).fetchall()
        if not rows:
            raise ScopeViolationError("corpus is empty or unknown")
        return self.snapshot([row[0] for row in rows])

    def create_subcorpus(
        self, corpus_id: str, name: str, revision_ids: list[str], description: str | None = None
    ) -> str:
        """Freeze an explicit subset of a named corpus for reproducible comparison.

        A subcorpus stores membership rather than a dynamic metadata predicate, so
        subsequent corpus additions or metadata changes cannot silently alter a
        historic selection.
        """
        if not name.strip() or not revision_ids:
            raise DocumentError("subcorpus requires a name and at least one revision")
        if not self.db.execute("SELECT 1 FROM corpora WHERE id=?", (corpus_id,)).fetchone():
            raise DocumentError("unknown corpus")
        unique_revisions = list(dict.fromkeys(revision_ids))
        rows: list[sqlite3.Row] = []
        for batch in _in_parameter_batches(unique_revisions):
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                self.db.execute(
                    "SELECT revision_id FROM corpus_memberships WHERE corpus_id=? "
                    f"AND revision_id IN ({placeholders})",
                    [corpus_id, *batch],
                ).fetchall()
            )
        if {row["revision_id"] for row in rows} != set(unique_revisions):
            raise ScopeViolationError(
                "subcorpus revisions must already belong to its parent corpus"
            )
        subcorpus_id = uid()
        try:
            self.db.execute(
                "INSERT INTO subcorpora VALUES(?,?,?,?,?)",
                (subcorpus_id, corpus_id, name.strip(), description, now()),
            )
        except sqlite3.IntegrityError as exc:
            raise DocumentError("subcorpus name already exists in this corpus") from exc
        self.db.executemany(
            "INSERT INTO subcorpus_memberships VALUES(?,?)",
            [(subcorpus_id, revision_id) for revision_id in unique_revisions],
        )
        self.db.commit()
        return subcorpus_id

    def subcorpus_snapshot(self, subcorpus_id: str) -> str:
        rows = self.db.execute(
            "SELECT revision_id FROM subcorpus_memberships WHERE subcorpus_id=? ORDER BY revision_id",
            (subcorpus_id,),
        ).fetchall()
        if not rows:
            raise ScopeViolationError("subcorpus is empty or unknown")
        return self.snapshot([row["revision_id"] for row in rows])

    def list_corpora(self) -> list[sqlite3.Row]:
        return self.db.execute(
            """
            SELECT c.id,c.name,c.description,c.created_at,COUNT(m.revision_id) AS revision_count
            FROM corpora c LEFT JOIN corpus_memberships m ON m.corpus_id=c.id
            GROUP BY c.id ORDER BY c.name
            """
        ).fetchall()

    def corpora_page(self, *, offset: int = 0, limit: int = 100) -> dict[str, object]:
        """Return a bounded, deterministically ordered page of corpus metadata."""
        if (
            type(offset) is not int
            or not 0 <= offset <= 0x7FFF_FFFF_FFFF_FFFF
            or type(limit) is not int
            or not 1 <= limit <= 1_000
        ):
            raise DocumentError(
                "corpora page requires an integer offset in SQLite range and integer limit 1..1000"
            )

        total = int(self.db.execute("SELECT COUNT(*) FROM corpora").fetchone()[0])
        records = self.db.execute(
            "SELECT c.id,c.name,c.description,c.created_at,COUNT(m.revision_id) AS revision_count "
            "FROM corpora c LEFT JOIN corpus_memberships m ON m.corpus_id=c.id "
            "GROUP BY c.id ORDER BY c.name,c.id LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        items = [dict(row) for row in records]
        return {
            "items": items,
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + len(items) < total,
        }

    def list_subcorpora(self, corpus_id: str | None = None) -> list[sqlite3.Row]:
        where = "WHERE s.corpus_id=?" if corpus_id else ""
        parameters: list[object] = [corpus_id] if corpus_id else []
        return self.db.execute(
            "SELECT s.id,s.corpus_id,s.name,s.description,s.created_at,COUNT(m.revision_id) AS revision_count "
            "FROM subcorpora s LEFT JOIN subcorpus_memberships m ON m.subcorpus_id=s.id "
            f"{where} GROUP BY s.id ORDER BY s.name",
            parameters,
        ).fetchall()

    def subcorpora_page(
        self, corpus_id: str | None = None, *, offset: int = 0, limit: int = 100
    ) -> dict[str, object]:
        """Return a bounded page of subcorpora, optionally filtered by parent."""
        if (
            type(offset) is not int
            or not 0 <= offset <= 0x7FFF_FFFF_FFFF_FFFF
            or type(limit) is not int
            or not 1 <= limit <= 1_000
        ):
            raise DocumentError(
                "subcorpora page requires an integer offset in SQLite range "
                "and integer limit 1..1000"
            )

        where = "WHERE s.corpus_id=?" if corpus_id else ""
        parameters: list[object] = [corpus_id] if corpus_id else []
        total = int(
            self.db.execute(f"SELECT COUNT(*) FROM subcorpora s {where}", parameters).fetchone()[0]
        )
        records = self.db.execute(
            "SELECT s.id,s.corpus_id,s.name,s.description,s.created_at,"
            "COUNT(m.revision_id) AS revision_count "
            "FROM subcorpora s LEFT JOIN subcorpus_memberships m ON m.subcorpus_id=s.id "
            f"{where} GROUP BY s.id ORDER BY s.name,s.id LIMIT ? OFFSET ?",
            [*parameters, limit, offset],
        ).fetchall()
        items = [dict(row) for row in records]
        return {
            "items": items,
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + len(items) < total,
        }

    def set_document_metadata(self, document_id: str, key: str, value: object) -> None:
        if not key.strip():
            raise DocumentError("metadata key is required")
        if not self.db.execute("SELECT 1 FROM documents WHERE id=?", (document_id,)).fetchone():
            raise DocumentError("unknown document")
        self.db.execute(
            """
            INSERT INTO document_metadata VALUES(?,?,?,?)
            ON CONFLICT(document_id,key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at
            """,
            (
                document_id,
                key.strip(),
                json.dumps(value, ensure_ascii=False, sort_keys=True),
                now(),
            ),
        )
        self.db.commit()

    def document_metadata(self, document_id: str) -> dict[str, object]:
        rows = self.db.execute(
            "SELECT key,value_json FROM document_metadata WHERE document_id=? ORDER BY key",
            (document_id,),
        ).fetchall()
        return {row["key"]: json.loads(row["value_json"]) for row in rows}

    def snapshot_by_document_metadata(self, filters: dict[str, object]) -> tuple[str, list[str]]:
        """Freeze latest revisions for documents matching every exact metadata filter.

        Metadata remains mutable project organisation, but the resulting
        ScopeSnapshot is immutable. Later metadata edits, imports or document
        revisions therefore cannot expand an already-created retrieval/analysis
        scope. JSON values are compared canonically, preserving type distinctions
        such as ``2025`` versus ``"2025"``.
        """
        if not filters or any(not isinstance(key, str) or not key.strip() for key in filters):
            raise ScopeViolationError("metadata selection requires one or more non-empty filters")
        clauses = " OR ".join("(key=? AND value_json=?)" for _ in filters)
        parameters: list[object] = []
        for key, value in sorted(filters.items()):
            parameters.extend([key.strip(), json.dumps(value, ensure_ascii=False, sort_keys=True)])
        documents = self.db.execute(
            f"SELECT document_id FROM document_metadata WHERE {clauses} "
            "GROUP BY document_id HAVING COUNT(*)=? ORDER BY document_id",
            [*parameters, len(filters)],
        ).fetchall()
        revisions: list[str] = []
        for document in documents:
            latest = self.db.execute(
                "SELECT id FROM document_revisions WHERE document_id=? "
                "ORDER BY created_at DESC,rowid DESC LIMIT 1",
                (document["document_id"],),
            ).fetchone()
            if latest:
                revisions.append(str(latest["id"]))
        if not revisions:
            raise ScopeViolationError("metadata selection matched no document revisions")
        return self.snapshot(revisions), revisions

    def scope_revisions(self, scope_id: str) -> list[str]:
        scope = self.db.execute(
            "SELECT revision_ids_json FROM scope_snapshots WHERE id=?", (scope_id,)
        ).fetchone()
        if not scope:
            raise ScopeViolationError("unknown scope")
        revisions = json.loads(scope[0])
        if not revisions:
            raise ScopeViolationError("scope is required")
        return revisions

    def scope_revisions_page(
        self,
        scope_id: str,
        *,
        offset: int = 0,
        limit: int = MAX_SCOPE_REVISION_PAGE_SIZE,
    ) -> dict[str, object]:
        """Read one bounded page from the complete membership of a ScopeSnapshot."""
        if (
            not isinstance(scope_id, str)
            or not scope_id.strip()
            or type(offset) is not int
            or not 0 <= offset <= 0x7FFF_FFFF_FFFF_FFFF
            or type(limit) is not int
            or not 1 <= limit <= MAX_SCOPE_REVISION_PAGE_SIZE
        ):
            raise ProjectError(
                "scope revisions page requires a scope ID, integer offset in SQLite range, "
                f"and integer limit 1..{MAX_SCOPE_REVISION_PAGE_SIZE}"
            )

        revisions = self.scope_revisions(scope_id)
        page = revisions[offset : offset + limit]
        return {
            "scope_id": scope_id,
            "offset": offset,
            "limit": limit,
            "revision_count": len(revisions),
            "revision_ids": page,
            "has_more": offset + len(page) < len(revisions),
        }

    def search(self, query: str, scope_id: str) -> list[sqlite3.Row]:
        revisions = self.scope_revisions(scope_id)
        rows: list[sqlite3.Row] = []
        for batch in _in_parameter_batches(revisions):
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                self.db.execute(
                    "SELECT revision_id,paragraph_id,text FROM revision_fts "
                    f"WHERE revision_fts MATCH ? AND revision_id IN ({placeholders}) "
                    "ORDER BY revision_id,paragraph_id",
                    [query, *batch],
                ).fetchall()
            )
        rows.sort(key=lambda row: row["revision_id"])
        return rows

    def search_paragraphs(self, query: str, scope_id: str, *, limit: int) -> list[sqlite3.Row]:
        """Return bounded FTS paragraph candidates with canonical offsets in explicit scope."""
        if not query.strip() or not 1 <= limit <= 5_000:
            raise ScopeViolationError("paragraph retrieval requires a query and a 1..5000 limit")
        ids = self.scope_revisions(scope_id)
        rows: list[sqlite3.Row] = []
        for batch in _in_parameter_batches(ids):
            placeholders = ",".join("?" for _ in batch)
            rows.extend(
                self.db.execute(
                    "SELECT f.revision_id,f.paragraph_id,f.text,p.start_offset,p.end_offset "
                    "FROM revision_fts f JOIN paragraphs p ON p.id=f.paragraph_id "
                    f"WHERE revision_fts MATCH ? AND f.revision_id IN ({placeholders}) "
                    "ORDER BY f.revision_id,p.start_offset,p.id LIMIT ?",
                    [query, *batch, limit],
                ).fetchall()
            )
        rows.sort(key=lambda row: (row["revision_id"], row["start_offset"], row["paragraph_id"]))
        return rows[:limit]

    def search_lexical_page(
        self,
        query: str,
        scope_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, object]:
        """Return one stable, bounded FTS page from an existing immutable scope.

        ``has_more`` describes only the retrieved candidate window. When that
        window reaches the configured cap, ``candidate_window_full`` warns that
        the complete result population may extend beyond the window.
        """
        if (
            offset < 0
            or offset > MAX_LEXICAL_SEARCH_OFFSET
            or not 1 <= limit <= MAX_LEXICAL_SEARCH_LIMIT
        ):
            raise ScopeViolationError(
                "lexical-search offset must be 0..4999 and limit must be 1..100"
            )
        candidates = self.search_paragraphs(query, scope_id, limit=MAX_LEXICAL_SEARCH_CANDIDATES)
        selected = candidates[offset : offset + limit]
        revision_ids = sorted({str(row["revision_id"]) for row in selected})
        document_ids = (
            {
                str(row["id"]): str(row["document_id"])
                for row in self.db.execute(
                    "SELECT id,document_id FROM document_revisions WHERE id IN ("
                    + ",".join("?" for _ in revision_ids)
                    + ")",
                    revision_ids,
                ).fetchall()
            }
            if revision_ids
            else {}
        )
        return {
            "scope_id": scope_id,
            "offset": offset,
            "limit": limit,
            "has_more": len(candidates) > offset + limit,
            "candidate_window_full": len(candidates) == MAX_LEXICAL_SEARCH_CANDIDATES,
            "rows": [
                {
                    "revision_id": str(row["revision_id"]),
                    "document_id": document_ids[str(row["revision_id"])],
                    "paragraph_id": str(row["paragraph_id"]),
                    "start_offset": int(row["start_offset"]),
                    "end_offset": int(row["end_offset"]),
                    "text": str(row["text"]),
                }
                for row in selected
            ],
        }

    def create_retrieval_run(
        self,
        *,
        analysis_run_id: str | None,
        skill_run_id: str | None,
        scope_id: str,
        retrieval_plan_ref: str,
        query_terms: list[str],
        selection_policy: str,
        candidate_count: int,
        inspected_count: int,
        returned_count: int,
        candidate_limit_reached: bool,
        coverage_id: str,
        standalone_operation: str | None = None,
        standalone_config: dict[str, object] | None = None,
        linked_context: dict[str, object] | None = None,
    ) -> str:
        """Persist one scoped retrieval attempt and its coverage/selection metadata.

        Analysis-backed retrieval requires both its AnalysisRun and SkillRun. A
        standalone direct search supplies neither and records its method/config
        in ``retrieval_run_context``. Linked runs may record privacy-safe search
        parameters there without changing their analysis/skill lineage.
        """
        if min(candidate_count, inspected_count, returned_count) < 0:
            raise ScopeViolationError("retrieval counts cannot be negative")
        if inspected_count > candidate_count or returned_count > inspected_count:
            raise ScopeViolationError("retrieval counts are inconsistent")
        direct_search = analysis_run_id is None and skill_run_id is None
        linked_context_json: str | None = None
        if direct_search:
            if (
                not isinstance(standalone_operation, str)
                or not standalone_operation.strip()
                or not isinstance(standalone_config, dict)
            ):
                raise ScopeViolationError(
                    "standalone retrieval requires an explicit operation and configuration"
                )
            if linked_context is not None:
                raise ScopeViolationError("standalone retrieval cannot also supply linked context")
            try:
                standalone_config_json = json.dumps(
                    standalone_config,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise ScopeViolationError("standalone retrieval configuration is invalid") from exc
            if _contains_secret_config_key(standalone_config):
                raise ScopeViolationError(
                    "standalone retrieval configuration contains secret-bearing metadata"
                )
            run = skill = None
        elif analysis_run_id is None or skill_run_id is None or standalone_operation is not None:
            raise ScopeViolationError(
                "retrieval must link both analysis and skill runs or be standalone"
            )
        else:
            if standalone_config is not None:
                raise ScopeViolationError(
                    "analysis-linked retrieval cannot include standalone configuration"
                )
            if linked_context is not None:
                if not isinstance(linked_context, dict) or _contains_secret_config_key(
                    linked_context
                ):
                    raise ScopeViolationError(
                        "linked retrieval context is invalid or secret-bearing"
                    )
                try:
                    linked_context_json = json.dumps(
                        linked_context,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                except (TypeError, ValueError) as exc:
                    raise ScopeViolationError("linked retrieval context is invalid") from exc
            else:
                linked_context_json = None
            run = self.db.execute(
                "SELECT scope_id FROM analysis_runs WHERE id=?", (analysis_run_id,)
            ).fetchone()
            skill = self.db.execute(
                "SELECT analysis_run_id,scope_id FROM skill_runs WHERE id=?", (skill_run_id,)
            ).fetchone()
            standalone_config_json = None
        coverage = self.coverage_descriptor(coverage_id)
        if coverage.scope_snapshot_id != scope_id:
            raise ScopeViolationError("retrieval must retain its original ScopeSnapshot")
        if not direct_search and (
            not run
            or not skill
            or run["scope_id"] != scope_id
            or skill["analysis_run_id"] != analysis_run_id
            or skill["scope_id"] != scope_id
        ):
            raise ScopeViolationError("retrieval must retain its skill and original ScopeSnapshot")
        retrieval_run_id = uid()
        self.db.execute(
            "INSERT INTO retrieval_runs(id,analysis_run_id,skill_run_id,scope_id,retrieval_plan_ref,"
            "query_terms_json,selection_policy,candidate_count,inspected_count,returned_count,"
            "candidate_limit_reached,coverage_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                retrieval_run_id,
                analysis_run_id,
                skill_run_id,
                scope_id,
                retrieval_plan_ref,
                json.dumps(query_terms, ensure_ascii=False),
                selection_policy,
                candidate_count,
                inspected_count,
                returned_count,
                int(candidate_limit_reached),
                coverage_id,
                now(),
            ),
        )
        if direct_search:
            self.db.execute(
                "INSERT INTO retrieval_run_context VALUES(?,?,?)",
                (retrieval_run_id, standalone_operation, standalone_config_json),
            )
        elif linked_context is not None:
            self.db.execute(
                "INSERT INTO retrieval_run_context VALUES(?,?,?)",
                (retrieval_run_id, "linked_retrieval", linked_context_json),
            )
        self.db.commit()
        return retrieval_run_id

    def add_retrieval_hit(
        self,
        retrieval_run_id: str,
        source_span_id: str,
        retrieval_method: str,
        rank: int,
        score: float,
    ) -> str:
        if rank < 1 or not retrieval_method.strip():
            raise ScopeViolationError("retrieval hit requires a positive rank and method")
        retrieval = self.db.execute(
            "SELECT scope_id FROM retrieval_runs WHERE id=?", (retrieval_run_id,)
        ).fetchone()
        span = self.db.execute(
            "SELECT revision_id FROM source_spans WHERE id=?", (source_span_id,)
        ).fetchone()
        if (
            not retrieval
            or not span
            or span["revision_id"] not in self.scope_revisions(retrieval["scope_id"])
        ):
            raise ScopeViolationError("retrieval hit source span is outside its explicit scope")
        hit_id = uid()
        self.db.execute(
            "INSERT INTO retrieval_hits VALUES(?,?,?,?,?,?)",
            (hit_id, retrieval_run_id, source_span_id, retrieval_method, rank, score),
        )
        self.db.commit()
        return hit_id

    def add_ranked_retrieval_hits(
        self,
        retrieval_run_id: str,
        hits: Iterable[tuple[str, str, float]],
        *,
        first_rank: int = 1,
        hit_contexts: Iterable[tuple[str | None, str, int | None, int | None]] | None = None,
    ) -> list[str]:
        """Persist a ranked batch with scoped spans and optional component lineage."""
        selected = list(hits)
        if not selected:
            return []
        contexts = list(hit_contexts) if hit_contexts is not None else None
        source_span_ids = [hit[0] for hit in selected]
        if (
            len(selected) > 1_000
            or len(set(source_span_ids)) != len(source_span_ids)
            or first_rank < 1
            or (contexts is not None and len(contexts) != len(selected))
            or any(
                not isinstance(method, str)
                or not method.strip()
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                for _, method, score in selected
            )
            or (
                contexts is not None
                and any(
                    (
                        artifact_id is not None
                        and (not isinstance(artifact_id, str) or not artifact_id)
                    )
                    or not isinstance(unit_id, str)
                    or not unit_id
                    or (
                        lexical_rank is not None
                        and (type(lexical_rank) is not int or lexical_rank < 1)
                    )
                    or (
                        vector_rank is not None
                        and (type(vector_rank) is not int or vector_rank < 1)
                    )
                    or ((artifact_id is None) != (vector_rank is None))
                    for artifact_id, unit_id, lexical_rank, vector_rank in contexts
                )
            )
        ):
            raise ScopeViolationError("ranked retrieval-hit batch metadata is invalid")
        retrieval = self.db.execute(
            "SELECT scope_id FROM retrieval_runs WHERE id=?", (retrieval_run_id,)
        ).fetchone()
        if retrieval is None:
            raise ScopeViolationError("unknown retrieval run")
        allowed_revisions = set(self.scope_revisions(retrieval["scope_id"]))
        placeholders = ",".join("?" for _ in source_span_ids)
        spans = self.db.execute(
            f"SELECT id,revision_id FROM source_spans WHERE id IN ({placeholders})",
            source_span_ids,
        ).fetchall()
        if len(spans) != len(source_span_ids) or any(
            span["revision_id"] not in allowed_revisions for span in spans
        ):
            raise ScopeViolationError("retrieval hit source span is outside its explicit scope")

        hit_ids = [uid() for _ in selected]
        self.db.execute("SAVEPOINT add_ranked_retrieval_hits")
        try:
            self.db.executemany(
                "INSERT INTO retrieval_hits VALUES(?,?,?,?,?,?)",
                [
                    (
                        hit_id,
                        retrieval_run_id,
                        source_span_id,
                        method,
                        first_rank + index,
                        float(score),
                    )
                    for index, (hit_id, (source_span_id, method, score)) in enumerate(
                        zip(hit_ids, selected, strict=True)
                    )
                ],
            )
            if contexts is not None:
                self.db.executemany(
                    "INSERT INTO retrieval_hit_context VALUES(?,?,?,?,?)",
                    [
                        (hit_id, artifact_id, unit_id, lexical_rank, vector_rank)
                        for hit_id, (artifact_id, unit_id, lexical_rank, vector_rank) in zip(
                            hit_ids, contexts, strict=True
                        )
                    ],
                )
            self.db.execute("RELEASE SAVEPOINT add_ranked_retrieval_hits")
        except BaseException:
            if self.db.in_transaction:
                try:
                    self.db.execute("ROLLBACK TO SAVEPOINT add_ranked_retrieval_hits")
                    self.db.execute("RELEASE SAVEPOINT add_ranked_retrieval_hits")
                except sqlite3.Error:
                    self.db.rollback()
            raise
        self.db.commit()
        return hit_ids

    def add_retrieval_hits(
        self,
        retrieval_run_id: str,
        source_span_ids: Iterable[str],
        retrieval_method: str,
        first_rank: int,
        score: float,
    ) -> list[str]:
        """Validate and persist a bounded, contiguous batch of scoped retrieval hits."""
        selected = list(source_span_ids)
        if not selected:
            return []
        if (
            len(selected) > 1_000
            or len(set(selected)) != len(selected)
            or first_rank < 1
            or not retrieval_method.strip()
            or not math.isfinite(score)
        ):
            raise ScopeViolationError("retrieval-hit batch metadata is invalid")
        retrieval = self.db.execute(
            "SELECT scope_id FROM retrieval_runs WHERE id=?", (retrieval_run_id,)
        ).fetchone()
        if not retrieval:
            raise ScopeViolationError("unknown retrieval run")
        allowed_revisions = set(self.scope_revisions(retrieval["scope_id"]))
        placeholders = ",".join("?" for _ in selected)
        rows = self.db.execute(
            f"SELECT id,revision_id FROM source_spans WHERE id IN ({placeholders})", selected
        ).fetchall()
        if len(rows) != len(selected) or any(
            row["revision_id"] not in allowed_revisions for row in rows
        ):
            raise ScopeViolationError("retrieval hit source span is outside its explicit scope")
        hit_ids = [uid() for _ in selected]
        self.db.execute("SAVEPOINT add_retrieval_hits")
        try:
            self.db.executemany(
                "INSERT INTO retrieval_hits VALUES(?,?,?,?,?,?)",
                [
                    (
                        hit_id,
                        retrieval_run_id,
                        span_id,
                        retrieval_method,
                        first_rank + index,
                        score,
                    )
                    for index, (hit_id, span_id) in enumerate(zip(hit_ids, selected, strict=True))
                ],
            )
            self.db.execute("RELEASE SAVEPOINT add_retrieval_hits")
        except BaseException:
            if self.db.in_transaction:
                try:
                    self.db.execute("ROLLBACK TO SAVEPOINT add_retrieval_hits")
                    self.db.execute("RELEASE SAVEPOINT add_retrieval_hits")
                except sqlite3.Error:
                    self.db.rollback()
            raise
        self.db.commit()
        return hit_ids

    def record_reranking(
        self,
        retrieval_run_id: str,
        *,
        provider_id: str,
        credential_alias: str | None,
        model: str,
        query: str,
        top_n: int | None,
        effective_top_n: int,
        candidate_count: int,
        ranked_hits: list[tuple[str, int, float]],
    ) -> str:
        """Persist reranker provenance separately from the original lexical hits."""
        if credential_alias is not None:
            try:
                validate_credential_alias(credential_alias)
            except ProjectError:
                raise ScopeViolationError("reranking credential alias is unsafe metadata") from None
        if not provider_id.strip() or not model.strip() or not query.strip():
            raise ScopeViolationError("reranking provenance requires provider, model, and query")
        if (
            candidate_count < 1
            or len(ranked_hits) < 1
            or len(ranked_hits) > candidate_count
            or (top_n is not None and top_n < 1)
            or effective_top_n != min(candidate_count, top_n or candidate_count)
            or len(ranked_hits) > effective_top_n
        ):
            raise ScopeViolationError("reranking result count is inconsistent with top_n")
        ranks = [rank for _, rank, _ in ranked_hits]
        hit_ids = [hit_id for hit_id, _, _ in ranked_hits]
        if sorted(ranks) != list(range(1, len(ranked_hits) + 1)) or len(set(hit_ids)) != len(
            hit_ids
        ):
            raise ScopeViolationError(
                "reranking ranks and retrieval hits must be unique and contiguous"
            )
        if any(not math.isfinite(score) for _, _, score in ranked_hits):
            raise ScopeViolationError("reranking relevance scores must be finite")
        retrieval = self.db.execute(
            "SELECT id FROM retrieval_runs WHERE id=?", (retrieval_run_id,)
        ).fetchone()
        if not retrieval:
            raise ScopeViolationError("reranking requires an existing RetrievalRun")
        placeholders = ",".join("?" for _ in hit_ids)
        rows = self.db.execute(
            f"SELECT id FROM retrieval_hits WHERE retrieval_run_id=? AND id IN ({placeholders})",
            [retrieval_run_id, *hit_ids],
        ).fetchall()
        if len(rows) != len(hit_ids):
            raise ScopeViolationError("reranking can reference only hits from its RetrievalRun")
        retrieved_count = self.db.execute(
            "SELECT count(*) FROM retrieval_hits WHERE retrieval_run_id=?", (retrieval_run_id,)
        ).fetchone()[0]
        if candidate_count != retrieved_count:
            raise ScopeViolationError(
                "reranking candidate count must match its submitted RetrievalRun hits"
            )

        reranking_run_id = uid()
        self.db.execute("SAVEPOINT record_reranking")
        try:
            self.db.execute(
                "INSERT INTO reranking_runs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    reranking_run_id,
                    retrieval_run_id,
                    provider_id,
                    credential_alias,
                    model,
                    query,
                    top_n,
                    effective_top_n,
                    candidate_count,
                    len(hit_ids),
                    now(),
                ),
            )
            self.db.executemany(
                "INSERT INTO reranking_hits VALUES(?,?,?,?,?)",
                [
                    (uid(), reranking_run_id, hit_id, rank, score)
                    for hit_id, rank, score in ranked_hits
                ],
            )
            self.db.execute("RELEASE SAVEPOINT record_reranking")
        except BaseException:
            if self.db.in_transaction:
                try:
                    self.db.execute("ROLLBACK TO SAVEPOINT record_reranking")
                    self.db.execute("RELEASE SAVEPOINT record_reranking")
                except sqlite3.Error:
                    self.db.rollback()
            raise
        self.db.commit()
        return reranking_run_id

    def reranking_hits(self, reranking_run_id: str) -> list[sqlite3.Row]:
        """Return one rerank's provider scores mapped to its original RetrievalHits."""
        return self.db.execute(
            "SELECT rh.rank,rh.relevance_score,h.id AS retrieval_hit_id,h.rank AS lexical_rank,"
            "h.score AS lexical_score,h.source_span_id FROM reranking_hits rh "
            "JOIN retrieval_hits h ON h.id=rh.retrieval_hit_id "
            "WHERE rh.reranking_run_id=? ORDER BY rh.rank",
            (reranking_run_id,),
        ).fetchall()

    def search_exact(self, phrase: str, scope_id: str) -> list[sqlite3.Row]:
        """Find a literal phrase only inside an immutable explicit scope.

        FTS5 narrows the candidate set; a Unicode-aware Python comparison verifies the
        literal phrase so FTS query syntax can never broaden or reinterpret user text.
        """
        terms = re.findall(r"[\w’'-]+", phrase, re.UNICODE)
        if not phrase or not terms:
            raise ScopeViolationError("exact search requires a non-empty text phrase")
        candidate_query = " AND ".join(f'"{term.replace('"', '""')}"' for term in terms)
        return [
            row
            for row in self.search(candidate_query, scope_id)
            if phrase.casefold() in row[2].casefold()
        ]

    def search_exact_ranked(self, phrase: str, scope_id: str) -> list[sqlite3.Row]:
        """Find literal phrase matches in explicit scope ordered by FTS5 BM25 relevance.

        FTS5 BM25 returns lower (typically more-negative) scores for more relevant
        candidates. The score orders matching paragraphs; revision, offset, and ID
        provide stable tie-breaking. A final Unicode-aware substring check preserves
        literal-exact semantics rather than trusting FTS token matching alone.
        """
        return list(self._iter_exact_ranked_matches(phrase, scope_id))

    def _iter_exact_ranked_matches(
        self, phrase: str, scope_id: str, *, include_end_offset: bool = False
    ) -> Iterator[sqlite3.Row]:
        """Yield exact BM25 matches in deterministic order without retaining them."""
        terms = re.findall(r"[\w’'-]+", phrase, re.UNICODE)
        if not phrase or not terms:
            raise ScopeViolationError("exact search requires a non-empty text phrase")
        revisions = self.scope_revisions(scope_id)
        candidate_query = " AND ".join('"' + term.replace('"', '""') + '"' for term in terms)
        folded_phrase = phrase.casefold()
        end_offset_column = ",p.end_offset" if include_end_offset else ""

        def exact_matches(
            batch: list[str],
        ) -> Iterator[tuple[tuple[float, str, int, str], sqlite3.Row]]:
            placeholders = ",".join("?" for _ in batch)
            cursor = self.db.execute(
                "SELECT f.revision_id,f.paragraph_id,f.text,bm25(revision_fts) AS bm25_score,"
                "p.start_offset" + end_offset_column + " "
                "FROM revision_fts f JOIN paragraphs p ON p.id=f.paragraph_id "
                f"WHERE revision_fts MATCH ? AND f.revision_id IN ({placeholders}) "
                "ORDER BY bm25(revision_fts),f.revision_id,p.start_offset,p.id",
                [candidate_query, *batch],
            )
            for row in cursor:
                if folded_phrase not in row["text"].casefold():
                    continue
                sort_key = (
                    float(row["bm25_score"]),
                    str(row["revision_id"]),
                    int(row["start_offset"]),
                    str(row["paragraph_id"]),
                )
                yield sort_key, row

        ranked = heapq.merge(
            *(exact_matches(batch) for batch in _in_parameter_batches(revisions)),
            key=lambda item: item[0],
        )
        return (row for _, row in ranked)

    def _search_exact_ranked_candidates(
        self,
        phrase: str,
        scope_id: str,
        *,
        limit: int,
        additional_spans: set[tuple[str, int, int]],
    ) -> tuple[list[sqlite3.Row], dict[tuple[str, int, int], int], int]:
        """Bound retained lexical rows while preserving exact ranks for other units.

        The complete ordered result stream is inspected, so query work scales with
        the exact-match population. Vector candidates are matched to lexical rows
        only by canonical ``(revision_id,start_offset,end_offset)`` span, not by
        caller-defined unit ID. Only the first ``limit`` lexical rows and ranks for
        ``additional_spans`` are retained, bounding memory independently of the
        number of exact matches. The returned count is the exact lexical population,
        not an estimate or a coverage claim.
        """
        if type(limit) is not int or limit < 1:
            raise ScopeViolationError("exact-search candidate limit must be a positive integer")
        if not isinstance(additional_spans, set) or any(
            not isinstance(span, tuple)
            or len(span) != 3
            or not isinstance(span[0], str)
            or not span[0]
            or type(span[1]) is not int
            or type(span[2]) is not int
            or span[1] < 0
            or span[2] <= span[1]
            for span in additional_spans
        ):
            raise ScopeViolationError("exact-search additional candidate spans are malformed")

        lexical_candidates: list[sqlite3.Row] = []
        additional_ranks: dict[tuple[str, int, int], int] = {}
        population_count = 0
        for rank, row in enumerate(
            self._iter_exact_ranked_matches(phrase, scope_id, include_end_offset=True),
            start=1,
        ):
            population_count = rank
            if rank <= limit:
                lexical_candidates.append(row)
            span = (
                str(row["revision_id"]),
                int(row["start_offset"]),
                int(row["end_offset"]),
            )
            if span in additional_spans:
                additional_ranks[span] = rank
        return lexical_candidates, additional_ranks, population_count

    def search_exact_page(
        self, phrase: str, scope_id: str, *, offset: int = 0, limit: int = 100
    ) -> dict[str, object]:
        """Return a bounded literal-search page without materializing all FTS candidates."""
        if offset < 0 or limit < 1 or limit > 1_000:
            raise ScopeViolationError("exact-search offset must be non-negative and limit 1..1000")
        terms = re.findall(r"[\w’'-]+", phrase, re.UNICODE)
        if not phrase or not terms:
            raise ScopeViolationError("exact search requires a non-empty text phrase")
        revisions = self.scope_revisions(scope_id)
        candidate_query = " AND ".join('"' + term.replace('"', '""') + '"' for term in terms)

        def exact_matches(
            batch: list[str],
        ) -> Iterator[tuple[str, sqlite3.Row]]:
            placeholders = ",".join("?" for _ in batch)
            cursor = self.db.execute(
                "SELECT f.revision_id,f.paragraph_id,f.text "
                "FROM revision_fts f JOIN paragraphs p ON p.id=f.paragraph_id "
                f"WHERE revision_fts MATCH ? AND f.revision_id IN ({placeholders}) "
                "ORDER BY f.revision_id,p.start_offset,p.id",
                [candidate_query, *batch],
            )
            for row in cursor:
                if folded_phrase in row["text"].casefold():
                    yield str(row["revision_id"]), row

        rows: list[sqlite3.Row] = []
        total_count = 0
        folded_phrase = phrase.casefold()
        matching_rows = heapq.merge(
            *(exact_matches(batch) for batch in _in_parameter_batches(revisions)),
            key=lambda item: item[0],
        )
        for total_count, (_, row) in enumerate(matching_rows, start=1):
            if offset < total_count <= offset + limit:
                rows.append(row)
        return {
            "offset": offset,
            "limit": limit,
            "total_count": total_count,
            "has_more": offset + len(rows) < total_count,
            "rows": rows,
        }

    def coverage(self, payload: CoverageDescriptor) -> str:
        cid = self._insert_coverage_descriptor(payload)
        self.db.commit()
        return cid

    def _insert_coverage_descriptor(self, payload: CoverageDescriptor) -> str:
        self.scope_revisions(payload.scope_snapshot_id)
        cid = uid()
        self.db.execute(
            "INSERT INTO coverage_descriptors VALUES(?,?)", (cid, payload.model_dump_json())
        )
        return cid

    def coverage_descriptor(self, coverage_id: str) -> CoverageDescriptor:
        row = self.db.execute(
            "SELECT payload_json FROM coverage_descriptors WHERE id=?", (coverage_id,)
        ).fetchone()
        if not row:
            raise EvidenceValidationError("unknown coverage descriptor")
        try:
            return CoverageDescriptor.model_validate_json(row["payload_json"])
        except ValueError as exc:
            raise EvidenceValidationError("coverage descriptor is corrupt") from exc

    def corpus_statistic_evidence(
        self,
        label: str,
        value: float,
        *,
        scope_snapshot_id: str,
        population_definition: str,
        provenance: dict[str, object] | None = None,
    ) -> str:
        """Persist a deterministic corpus result separately from source spans."""
        evidence_id = self._insert_corpus_statistic_evidence(
            label,
            value,
            scope_snapshot_id=scope_snapshot_id,
            population_definition=population_definition,
            provenance=provenance,
        )
        self.db.commit()
        return evidence_id

    def _insert_corpus_statistic_evidence(
        self,
        label: str,
        value: float,
        *,
        scope_snapshot_id: str,
        population_definition: str,
        provenance: dict[str, object] | None = None,
    ) -> str:
        if not label.strip() or not population_definition.strip():
            raise EvidenceValidationError(
                "corpus statistic evidence requires label and population definition"
            )
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            raise EvidenceValidationError("corpus statistic value must be finite and numeric")
        self.scope_revisions(scope_snapshot_id)
        evidence_id = uid()
        payload = {
            "label": label.strip(),
            "value": value,
            "scope_snapshot_id": scope_snapshot_id,
            "population_definition": population_definition.strip(),
        }
        if provenance:
            payload["provenance"] = provenance
        self.db.execute(
            "INSERT INTO evidence_artifacts VALUES(?,?,?,?,?)",
            (evidence_id, EvidenceKind.CORPUS_STATISTIC, None, json.dumps(payload), now()),
        )
        return evidence_id

    def corpus_statistic_evidence_with_coverage(
        self,
        label: str,
        value: float,
        *,
        coverage: CoverageDescriptor,
        population_definition: str,
        provenance: dict[str, object] | None = None,
    ) -> tuple[str, str]:
        """Atomically persist a corpus statistic with its coverage record.

        The coverage ID is added to statistic provenance so callers cannot persist
        a statistic that points at a different or missing coverage descriptor.
        """
        coverage_id: str | None = None
        self.db.execute("SAVEPOINT corpus_statistic_with_coverage")
        try:
            coverage_id = self._insert_coverage_descriptor(coverage)
            statistic_provenance = dict(provenance or {})
            statistic_provenance["coverage_id"] = coverage_id
            evidence_id = self._insert_corpus_statistic_evidence(
                label,
                value,
                scope_snapshot_id=coverage.scope_snapshot_id,
                population_definition=population_definition,
                provenance=statistic_provenance,
            )
            self.db.execute("RELEASE SAVEPOINT corpus_statistic_with_coverage")
            self.db.commit()
        except BaseException:
            if self.db.in_transaction:
                try:
                    self.db.execute("ROLLBACK TO SAVEPOINT corpus_statistic_with_coverage")
                    self.db.execute("RELEASE SAVEPOINT corpus_statistic_with_coverage")
                except sqlite3.Error:
                    self.db.rollback()
            raise
        assert coverage_id is not None
        return coverage_id, evidence_id

    def create_run(self, scope_id: str, config: dict) -> str:
        self.scope_revisions(scope_id)
        if _contains_secret_config_key(config, allow_counterevidence_query_labels=True):
            raise DocumentError(
                "analysis run configuration must not contain secrets or secret references"
            )
        rid = uid()
        data = json.dumps(config, sort_keys=True)
        self.db.execute(
            "INSERT INTO analysis_runs(id,scope_id,status,config_json,config_hash,started_at,"
            "completed_at,software_version,project_format_version,database_schema_version) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                rid,
                scope_id,
                RunStatus.PENDING,
                data,
                digest(data.encode()),
                now(),
                None,
                software_version(),
                self.config.project_format,
                SCHEMA_VERSION,
            ),
        )
        self.db.commit()
        log_event(
            _LOGGER,
            "analysis.run.created",
            analysis_run_id=rid,
            scope_id=scope_id,
            status=RunStatus.PENDING,
        )
        return rid

    def start_run(self, run_id: str) -> None:
        """Transition an immutable analysis configuration into active execution."""
        cur = self.db.execute(
            "UPDATE analysis_runs SET status=? WHERE id=? AND status=?",
            (RunStatus.RUNNING, run_id, RunStatus.PENDING),
        )
        if not cur.rowcount:
            raise DocumentError("run is not pending or is unknown")
        self.db.commit()
        row = self.db.execute("SELECT scope_id FROM analysis_runs WHERE id=?", (run_id,)).fetchone()
        log_event(
            _LOGGER,
            "analysis.run.started",
            analysis_run_id=run_id,
            scope_id=row["scope_id"] if row else None,
            status=RunStatus.RUNNING,
        )

    def fail_run(self, run_id: str, error: BaseException) -> None:
        """Persist terminal failure without overwriting configuration or prior results."""
        error_type = type(error).__name__
        self.db.execute("SAVEPOINT fail_analysis_run")
        try:
            failed_at = now()
            self.db.execute(
                "UPDATE skill_runs SET status=?,completed_at=?,error_type=? "
                "WHERE analysis_run_id=? AND status=?",
                (RunStatus.FAILED, failed_at, error_type, run_id, RunStatus.RUNNING),
            )
            cur = self.db.execute(
                "UPDATE analysis_runs SET status=?,completed_at=? WHERE id=? AND status IN (?,?)",
                (RunStatus.FAILED, failed_at, run_id, RunStatus.PENDING, RunStatus.RUNNING),
            )
            if not cur.rowcount:
                raise DocumentError("run is immutable or unknown")
            self.db.execute("RELEASE SAVEPOINT fail_analysis_run")
        except BaseException:
            if self.db.in_transaction:
                try:
                    self.db.execute("ROLLBACK TO SAVEPOINT fail_analysis_run")
                    self.db.execute("RELEASE SAVEPOINT fail_analysis_run")
                except sqlite3.Error:
                    self.db.rollback()
            raise
        self.db.commit()
        log_event(
            _LOGGER,
            "analysis.run.failed",
            analysis_run_id=run_id,
            error_type=error_type,
            status=RunStatus.FAILED,
        )

    def resume_run(self, run_id: str) -> None:
        """Resume only a failed, still-configured analysis run in its original scope."""
        cur = self.db.execute(
            "UPDATE analysis_runs SET status=?,completed_at=NULL WHERE id=? AND status=?",
            (RunStatus.RUNNING, run_id, RunStatus.FAILED),
        )
        if not cur.rowcount:
            raise DocumentError("only a failed analysis run can be resumed")
        self.db.commit()
        log_event(
            _LOGGER,
            "analysis.run.resumed",
            analysis_run_id=run_id,
            status=RunStatus.RUNNING,
        )

    def list_analysis_runs(self, limit: int = 100) -> list[dict[str, object]]:
        """List immutable run metadata without materializing findings or report text."""
        if limit < 1 or limit > 1_000:
            raise DocumentError("run list limit must be between 1 and 1000")
        rows = self.db.execute(
            """
            SELECT ar.id,ar.scope_id,ar.status,ar.config_hash,ar.started_at,ar.completed_at,
                   COUNT(DISTINCT sr.id) AS skill_run_count,COUNT(DISTINCT f.id) AS finding_count
            FROM analysis_runs ar
            LEFT JOIN skill_runs sr ON sr.analysis_run_id=ar.id
            LEFT JOIN findings f ON f.analysis_run_id=ar.id
            GROUP BY ar.id ORDER BY ar.started_at DESC,ar.id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def analysis_runs_page(self, *, offset: int = 0, limit: int = 100) -> dict[str, object]:
        """Return a bounded, deterministic page of immutable run metadata."""
        if (
            type(offset) is not int
            or not 0 <= offset <= 0x7FFF_FFFF_FFFF_FFFF
            or type(limit) is not int
            or not 1 <= limit <= 1_000
        ):
            raise DocumentError(
                "analysis runs page requires an integer offset in SQLite range "
                "and integer limit 1..1000"
            )

        total = int(self.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0])
        rows = self.db.execute(
            "SELECT ar.id,ar.scope_id,ar.status,ar.config_hash,ar.started_at,ar.completed_at,"
            "COUNT(DISTINCT sr.id) AS skill_run_count,COUNT(DISTINCT f.id) AS finding_count "
            "FROM analysis_runs ar "
            "LEFT JOIN skill_runs sr ON sr.analysis_run_id=ar.id "
            "LEFT JOIN findings f ON f.analysis_run_id=ar.id "
            "GROUP BY ar.id ORDER BY ar.started_at DESC,ar.id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        items = [dict(row) for row in rows]
        return {
            "items": items,
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + len(items) < total,
        }

    def analysis_run_summary(self, run_id: str) -> dict[str, object]:
        """Return secret-free reproducibility metadata for one historical run."""
        metadata = self.analysis_run_metadata(run_id)
        model_executions = self.db.execute(
            "SELECT * FROM model_executions WHERE analysis_run_id=? ORDER BY started_at,id",
            (run_id,),
        ).fetchall()
        reranking_runs = self.db.execute(
            "SELECT rr.* FROM reranking_runs rr JOIN retrieval_runs r "
            "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=? "
            "ORDER BY rr.created_at,rr.id",
            (run_id,),
        ).fetchall()
        safe_model_executions = self._validated_analysis_run_model_execution_rows(
            run_id, model_executions
        )
        safe_reranking_runs = _validated_credential_ledger_rows(reranking_runs, allow_none=True)
        return {
            **metadata,
            "skill_runs": [
                dict(row)
                for row in self.db.execute(
                    "SELECT * FROM skill_runs WHERE analysis_run_id=? ORDER BY started_at,id",
                    (run_id,),
                ).fetchall()
            ],
            "model_executions": safe_model_executions,
            "retrieval_runs": [
                dict(row)
                for row in self.db.execute(
                    "SELECT * FROM retrieval_runs WHERE analysis_run_id=? ORDER BY created_at,id",
                    (run_id,),
                ).fetchall()
            ],
            "retrieval_hits": [
                dict(row)
                for row in self.db.execute(
                    "SELECT h.* FROM retrieval_hits h JOIN retrieval_runs r "
                    "ON r.id=h.retrieval_run_id WHERE r.analysis_run_id=? "
                    "ORDER BY r.created_at,r.id,h.rank",
                    (run_id,),
                ).fetchall()
            ],
            "reranking_runs": safe_reranking_runs,
            "reranking_hits": [
                dict(row)
                for row in self.db.execute(
                    "SELECT rh.* FROM reranking_hits rh JOIN reranking_runs rr "
                    "ON rr.id=rh.reranking_run_id JOIN retrieval_runs r "
                    "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=? "
                    "ORDER BY rr.created_at,rr.id,rh.rank",
                    (run_id,),
                ).fetchall()
            ],
        }

    def _validated_analysis_run_model_execution_rows(
        self, run_id: str, rows: Iterable[sqlite3.Row]
    ) -> list[dict[str, object]]:
        """Validate secret-free execution metadata and any structural SkillRun links."""
        records = _validated_model_execution_rows(rows)
        for record in records:
            skill_run_id = record.get("skill_run_id")
            if skill_run_id is None:
                continue
            if not isinstance(skill_run_id, str):
                raise DocumentError("stored model execution skill run reference is invalid")
            try:
                self.validate_model_execution_skill_run(run_id, skill_run_id)
            except DocumentError:
                raise DocumentError(
                    "stored model execution skill run reference is invalid"
                ) from None
        return records

    def analysis_run_metadata(self, run_id: str) -> dict[str, object]:
        """Validate one run and return only its bounded metadata and frozen scope."""
        run = self.db.execute("SELECT * FROM analysis_runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            raise DocumentError("unknown analysis run")
        try:
            config = json.loads(run["config_json"])
        except json.JSONDecodeError as exc:
            raise DocumentError("analysis run configuration is corrupt") from exc
        if not isinstance(config, dict):
            raise DocumentError("analysis run configuration is corrupt")
        if digest(run["config_json"].encode("utf-8")) != run["config_hash"]:
            raise DocumentError("analysis run configuration hash mismatch")
        if _contains_secret_config_key(config, allow_counterevidence_query_labels=True):
            raise DocumentError("analysis run configuration contains secret data")
        return {
            "analysis_run": {**dict(run), "config": config, "config_json": None},
            "revision_ids": self.scope_revisions(run["scope_id"]),
        }

    def analysis_run_summary_page(
        self, run_id: str, *, offset: int = 0, limit: int = 100
    ) -> dict[str, object]:
        """Return a bounded provenance page; use ``analysis_run_summary`` for local archives."""
        if (
            type(offset) is not int
            or offset < 0
            or offset > 0x7FFF_FFFF_FFFF_FFFF
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise DocumentError(
                "analysis run page requires an integer offset in SQLite range and integer limit 1..100"
            )
        metadata = self.analysis_run_metadata(run_id)
        run = metadata["analysis_run"]
        revision_ids = metadata["revision_ids"]
        if not isinstance(run, dict) or not isinstance(revision_ids, list):
            raise DocumentError("analysis run metadata is invalid")
        params = (run_id,)
        pagination: dict[str, dict[str, int | bool]] = {}

        def page_rows(
            key: str, count_sql: str, page_sql: str, query_params: tuple[object, ...]
        ) -> list[dict[str, object]]:
            total = int(self.db.execute(count_sql, query_params).fetchone()[0])
            rows = self.db.execute(page_sql, (*query_params, limit, offset)).fetchall()
            pagination[key] = {
                "offset": offset,
                "limit": limit,
                "total_count": total,
                "has_more": offset + len(rows) < total,
            }
            if key == "model_executions":
                return self._validated_analysis_run_model_execution_rows(run_id, rows)
            if key == "reranking_runs":
                return _validated_credential_ledger_rows(rows, allow_none=True)
            return [dict(row) for row in rows]

        revision_total = len(revision_ids)
        paged_revision_ids = revision_ids[offset : offset + limit]
        pagination["revision_ids"] = {
            "offset": offset,
            "limit": limit,
            "total_count": revision_total,
            "has_more": offset + len(paged_revision_ids) < revision_total,
        }
        skill_runs = page_rows(
            "skill_runs",
            "SELECT COUNT(*) FROM skill_runs WHERE analysis_run_id=?",
            "SELECT * FROM skill_runs WHERE analysis_run_id=? ORDER BY started_at,id LIMIT ? OFFSET ?",
            params,
        )
        model_executions = page_rows(
            "model_executions",
            "SELECT COUNT(*) FROM model_executions WHERE analysis_run_id=?",
            "SELECT * FROM model_executions WHERE analysis_run_id=? ORDER BY started_at,id LIMIT ? OFFSET ?",
            params,
        )
        retrieval_runs = page_rows(
            "retrieval_runs",
            "SELECT COUNT(*) FROM retrieval_runs WHERE analysis_run_id=?",
            "SELECT * FROM retrieval_runs WHERE analysis_run_id=? ORDER BY created_at,id LIMIT ? OFFSET ?",
            params,
        )
        retrieval_hits = page_rows(
            "retrieval_hits",
            "SELECT COUNT(*) FROM retrieval_hits h JOIN retrieval_runs r "
            "ON r.id=h.retrieval_run_id WHERE r.analysis_run_id=?",
            "SELECT h.* FROM retrieval_hits h JOIN retrieval_runs r "
            "ON r.id=h.retrieval_run_id WHERE r.analysis_run_id=? "
            "ORDER BY r.created_at,r.id,h.rank LIMIT ? OFFSET ?",
            params,
        )
        reranking_runs = page_rows(
            "reranking_runs",
            "SELECT COUNT(*) FROM reranking_runs rr JOIN retrieval_runs r "
            "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=?",
            "SELECT rr.* FROM reranking_runs rr JOIN retrieval_runs r "
            "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=? "
            "ORDER BY rr.created_at,rr.id LIMIT ? OFFSET ?",
            params,
        )
        reranking_hits = page_rows(
            "reranking_hits",
            "SELECT COUNT(*) FROM reranking_hits rh JOIN reranking_runs rr "
            "ON rr.id=rh.reranking_run_id JOIN retrieval_runs r "
            "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=?",
            "SELECT rh.* FROM reranking_hits rh JOIN reranking_runs rr "
            "ON rr.id=rh.reranking_run_id JOIN retrieval_runs r "
            "ON r.id=rr.retrieval_run_id WHERE r.analysis_run_id=? "
            "ORDER BY rr.created_at,rr.id,rh.rank LIMIT ? OFFSET ?",
            params,
        )
        return {
            "analysis_run": run,
            "revision_ids": paged_revision_ids,
            "skill_runs": skill_runs,
            "model_executions": model_executions,
            "retrieval_runs": retrieval_runs,
            "retrieval_hits": retrieval_hits,
            "reranking_runs": reranking_runs,
            "reranking_hits": reranking_hits,
            "pagination": pagination,
        }

    def save_credential_reference(self, credential: CredentialReference) -> str:
        """Persist an alias and secret reference only; the secret value never enters SQLite."""
        if not all((credential.provider_id, credential.alias, credential.secret_reference)):
            raise DocumentError("provider ID, credential alias, and secret reference are required")
        try:
            validate_credential_alias(credential.alias)
        except ProjectError:
            raise DocumentError("credential alias must be a short human-readable label") from None
        provider_auth_reference(
            credential.provider_id, credential.secret_reference, credential.endpoint
        )
        existing = self.db.execute(
            "SELECT id,secret_reference FROM provider_credentials "
            "WHERE provider_id=? AND credential_alias=?",
            (credential.provider_id, credential.alias),
        ).fetchone()
        credential_id = existing[0] if existing else uid()
        self.db.execute(
            """
            INSERT INTO provider_credentials(id,provider_id,credential_alias,secret_reference,endpoint,created_at)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(provider_id,credential_alias) DO UPDATE SET
                secret_reference=excluded.secret_reference,endpoint=excluded.endpoint
            """,
            (
                credential_id,
                credential.provider_id,
                credential.alias,
                credential.secret_reference,
                credential.endpoint,
                now(),
            ),
        )
        self.db.commit()
        if (
            existing is not None
            and existing["secret_reference"] != credential.secret_reference
            and str(existing["secret_reference"]).startswith("project-secret:")
        ):
            still_referenced = self.db.execute(
                "SELECT 1 FROM provider_credentials WHERE secret_reference=? LIMIT 1",
                (existing["secret_reference"],),
            ).fetchone()
            if still_referenced is None:
                self.delete_encrypted_secret(str(existing["secret_reference"]))
        return credential_id

    def save_encrypted_provider_secret(
        self, provider_id: str, alias: str, secret_value: str
    ) -> str:
        """Encrypt a provider key in this project DB and return its opaque reference.

        The deployment key is supplied out of band through
        ``DISCOURSELENS_CREDENTIALS_ENCRYPTION_KEY`` and is never stored in the
        project. A distinct key is derived for each immutable project identity.
        """
        import re

        from cryptography.fernet import Fernet

        if not isinstance(provider_id, str) or not re.fullmatch(
            r"[a-z][a-z0-9-]{0,63}", provider_id
        ):
            raise DocumentError("provider ID is invalid")
        try:
            validate_credential_alias(alias)
        except ProjectError:
            raise DocumentError("credential alias must be a short human-readable label") from None
        if (
            not isinstance(secret_value, str)
            or not secret_value
            or len(secret_value.encode("utf-8")) > 16_384
            or any(ord(character) < 32 and character not in "\t\n\r" for character in secret_value)
        ):
            raise DocumentError("provider credential value is invalid")
        reference = f"project-secret:{uid()}"
        cipher = Fernet(self._provider_secret_fernet_key())
        ciphertext = cipher.encrypt(secret_value.encode("utf-8"))
        timestamp = now()
        self.db.execute(
            "INSERT INTO encrypted_secrets(secret_reference,ciphertext,created_at,updated_at) "
            "VALUES(?,?,?,?)",
            (reference, ciphertext, timestamp, timestamp),
        )
        self.db.commit()
        return reference

    def resolve_encrypted_secret(self, reference: str) -> str:
        """Decrypt a credential only from this project's database."""
        from cryptography.fernet import Fernet, InvalidToken

        if not isinstance(reference, str) or not reference.startswith("project-secret:"):
            raise ProjectError("encrypted project secret reference is invalid")
        row = self.db.execute(
            "SELECT ciphertext FROM encrypted_secrets WHERE secret_reference=?", (reference,)
        ).fetchone()
        if row is None:
            raise ProjectError("encrypted project secret is unavailable")
        try:
            return Fernet(self._provider_secret_fernet_key()).decrypt(row["ciphertext"]).decode(
                "utf-8"
            )
        except (InvalidToken, UnicodeDecodeError, TypeError):
            raise ProjectError("encrypted project secret could not be decrypted") from None

    def delete_encrypted_secret(self, reference: str) -> None:
        """Remove a project-encrypted secret by its opaque reference."""
        if not isinstance(reference, str) or not reference.startswith("project-secret:"):
            raise DocumentError("encrypted project secret reference is invalid")
        self.db.execute(
            "DELETE FROM encrypted_secrets WHERE secret_reference=?", (reference,)
        )
        self.db.commit()

    def delete_provider_credential(
        self, provider_id: str, alias: str, *, delete_presets: bool = False
    ) -> None:
        """Remove a provider credential and its project-encrypted secret safely."""
        self._require_writable()
        try:
            validate_credential_alias(alias)
        except ProjectError:
            raise DocumentError("credential alias must be a short human-readable label") from None
        if type(delete_presets) is not bool:
            raise DocumentError("delete_presets must be a boolean")
        presets = self.db.execute(
            "SELECT name FROM model_presets WHERE provider_id=? AND credential_alias=? ORDER BY name",
            (provider_id, alias),
        ).fetchall()
        if presets and not delete_presets:
            raise DocumentError(
                "Esta credencial está asociada a modelos guardados. Elimínalos primero "
                "o confirma también la eliminación de esos modelos."
            )
        credential = self.db.execute(
            "SELECT secret_reference FROM provider_credentials "
            "WHERE provider_id=? AND credential_alias=?",
            (provider_id, alias),
        ).fetchone()
        if credential is None:
            raise DocumentError("unknown provider credential")
        if presets:
            self.db.execute(
                "DELETE FROM model_presets WHERE provider_id=? AND credential_alias=?",
                (provider_id, alias),
            )
        self.db.execute(
            "DELETE FROM provider_credentials WHERE provider_id=? AND credential_alias=?",
            (provider_id, alias),
        )
        secret_reference = str(credential["secret_reference"])
        if secret_reference.startswith("project-secret:"):
            still_referenced = self.db.execute(
                "SELECT 1 FROM provider_credentials WHERE secret_reference=? LIMIT 1",
                (secret_reference,),
            ).fetchone()
            if still_referenced is None:
                self.db.execute(
                    "DELETE FROM encrypted_secrets WHERE secret_reference=?",
                    (secret_reference,),
                )
        self._repair_default_chat_model_preset()
        self.db.commit()

    def _provider_secret_fernet_key(self) -> bytes:
        configured_key = os.environ.get("DISCOURSELENS_CREDENTIALS_ENCRYPTION_KEY", "")
        try:
            master_key = base64.urlsafe_b64decode(configured_key.encode("ascii"))
        except (ValueError, UnicodeEncodeError):
            raise ProjectError("provider-key encryption is not configured") from None
        if len(master_key) != 32:
            raise ProjectError("provider-key encryption is not configured")
        project_key = hmac.digest(
            master_key,
            f"discourselens/project-secrets/v1/{self.project_id}".encode(),
            "sha256",
        )
        return base64.urlsafe_b64encode(project_key)

    def list_credential_references(self) -> list[sqlite3.Row]:
        rows = self.db.execute(
            "SELECT id,provider_id,credential_alias,secret_reference,endpoint,created_at "
            "FROM provider_credentials ORDER BY provider_id,credential_alias"
        ).fetchall()
        for row in rows:
            _validate_stored_credential_alias(row["credential_alias"])
            try:
                provider_auth_reference(
                    row["provider_id"], row["secret_reference"], row["endpoint"]
                )
            except ProjectError:
                raise DocumentError("stored credential metadata is unsafe") from None
        return rows

    def save_model_preset(self, preset: ModelPreset) -> str:
        if not all((preset.name, preset.provider_id, preset.credential_alias, preset.model)):
            raise DocumentError("model preset requires name, provider, credential alias, and model")
        try:
            validate_credential_alias(preset.credential_alias)
        except ProjectError:
            raise DocumentError("credential alias must be a short human-readable label") from None
        parameters = preset.parameters or {}
        if not isinstance(parameters, dict):
            raise DocumentError("model preset parameters must be a JSON object")
        if _contains_secret_config_key(parameters):
            raise DocumentError("model preset parameters must not contain secrets")
        allowed_reasoning_modes = {"adaptive", "manual"}
        if preset.provider_id == "cohere":
            allowed_reasoning_modes = {"enabled", "disabled", "manual"}
        if (
            preset.reasoning_mode is not None
            and preset.reasoning_mode not in allowed_reasoning_modes
        ):
            raise DocumentError("model preset reasoning mode is unsupported for this provider")
        if preset.reasoning_budget_tokens is not None and (
            not isinstance(preset.reasoning_budget_tokens, int)
            or isinstance(preset.reasoning_budget_tokens, bool)
            or preset.reasoning_budget_tokens < 1
        ):
            raise DocumentError("model preset reasoning budget must be a positive integer")
        if preset.reasoning_budget_tokens is not None and preset.reasoning_mode != "manual":
            raise DocumentError("reasoning budget requires manual reasoning mode")
        if preset.reasoning_mode == "manual":
            if preset.provider_id == "cohere":
                try:
                    validate_cohere_manual_budget(preset.reasoning_budget_tokens, parameters)
                except ModelProviderError as exc:
                    raise DocumentError(str(exc)) from None
            elif preset.reasoning_budget_tokens is None or preset.reasoning_budget_tokens < 1024:
                raise DocumentError(
                    "manual reasoning mode requires a budget of at least 1024 tokens"
                )
        try:
            parameters_json = json.dumps(parameters, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DocumentError("model preset parameters must be JSON-compatible") from exc
        reference = self.db.execute(
            "SELECT 1 FROM provider_credentials WHERE provider_id=? AND credential_alias=?",
            (preset.provider_id, preset.credential_alias),
        ).fetchone()
        if not reference:
            raise DocumentError("model preset references an unknown provider credential alias")
        existing = self.db.execute(
            "SELECT id FROM model_presets WHERE name=?", (preset.name,)
        ).fetchone()
        preset_id = existing[0] if existing else uid()
        self.db.execute(
            """
            INSERT INTO model_presets(id,name,provider_id,credential_alias,model,reasoning,reasoning_mode,reasoning_budget_tokens,parameters_json,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET provider_id=excluded.provider_id,
                credential_alias=excluded.credential_alias,model=excluded.model,
                reasoning=excluded.reasoning,reasoning_mode=excluded.reasoning_mode,
                reasoning_budget_tokens=excluded.reasoning_budget_tokens,
                parameters_json=excluded.parameters_json
            """,
            (
                preset_id,
                preset.name,
                preset.provider_id,
                preset.credential_alias,
                preset.model,
                preset.reasoning,
                preset.reasoning_mode,
                preset.reasoning_budget_tokens,
                parameters_json,
                now(),
            ),
        )
        self.db.commit()
        return preset_id

    def list_model_presets(self) -> list[sqlite3.Row]:
        rows = self.db.execute(
            "SELECT id,name,provider_id,credential_alias,model,reasoning,reasoning_mode,reasoning_budget_tokens,parameters_json,created_at "
            "FROM model_presets ORDER BY name"
        ).fetchall()
        for row in rows:
            _validate_model_preset_metadata(row)
        return rows

    def _first_chat_model_preset_name(self) -> str | None:
        from .provider_catalog import supports_generation

        rows = self.db.execute(
            "SELECT name,provider_id FROM model_presets ORDER BY name"
        ).fetchall()
        return next(
            (str(row["name"]) for row in rows if supports_generation(str(row["provider_id"]))),
            None,
        )

    def default_chat_model_preset(self) -> str | None:
        """Return the saved default chat model, falling back to the first chat preset."""
        from .provider_catalog import supports_generation

        setting = self.db.execute(
            "SELECT value FROM project_meta WHERE key=?", (_DEFAULT_CHAT_MODEL_PRESET_KEY,)
        ).fetchone()
        if setting is not None:
            preset = self.db.execute(
                "SELECT provider_id FROM model_presets WHERE name=?", (setting["value"],)
            ).fetchone()
            if preset is not None and supports_generation(str(preset["provider_id"])):
                return str(setting["value"])
        return self._first_chat_model_preset_name()

    def set_default_chat_model_preset(self, name: str) -> None:
        """Set the saved default used for new chat conversations in this project."""
        self._require_writable()
        from .provider_catalog import supports_generation

        preset = self.model_preset(name)
        if not supports_generation(str(preset["provider_id"])):
            raise DocumentError("El modelo seleccionado no permite generar conversaciones.")
        self.db.execute(
            "INSERT INTO project_meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (_DEFAULT_CHAT_MODEL_PRESET_KEY, name),
        )
        self.db.commit()

    def _chat_reasoning_preferences(self) -> dict[str, str]:
        setting = self.db.execute(
            "SELECT value FROM project_meta WHERE key=?",
            (_CHAT_REASONING_PREFERENCES_KEY,),
        ).fetchone()
        if setting is None:
            return {}
        try:
            payload = json.loads(str(setting["value"]))
        except (TypeError, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            name: value
            for name, value in payload.items()
            if isinstance(name, str) and isinstance(value, str)
        }

    def chat_reasoning_preference(self, preset_name: str) -> str | None:
        """Return the saved chat reasoning override for one model preset."""
        if not isinstance(preset_name, str) or not preset_name.strip():
            return None
        return self._chat_reasoning_preferences().get(preset_name)

    def set_chat_reasoning_preference(
        self, preset_name: str, reasoning: str | None
    ) -> None:
        """Save or clear a per-preset chat reasoning override in project metadata."""
        self._require_writable()
        if not isinstance(preset_name, str) or not preset_name.strip():
            raise DocumentError("model preset name is invalid")
        self.model_preset(preset_name)
        if reasoning is not None and (
            not isinstance(reasoning, str) or not reasoning.strip() or len(reasoning) > 64
        ):
            raise DocumentError("chat reasoning preference is invalid")
        preferences = self._chat_reasoning_preferences()
        if reasoning is None:
            preferences.pop(preset_name, None)
        else:
            preferences[preset_name] = reasoning
        if preferences:
            self.db.execute(
                "INSERT INTO project_meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (
                    _CHAT_REASONING_PREFERENCES_KEY,
                    json.dumps(
                        preferences,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
        else:
            self.db.execute(
                "DELETE FROM project_meta WHERE key=?",
                (_CHAT_REASONING_PREFERENCES_KEY,),
            )
        self.db.commit()

    def _repair_default_chat_model_preset(self) -> None:
        """Choose another available chat preset if the saved default was removed."""
        from .provider_catalog import supports_generation

        setting = self.db.execute(
            "SELECT value FROM project_meta WHERE key=?", (_DEFAULT_CHAT_MODEL_PRESET_KEY,)
        ).fetchone()
        if setting is None:
            return
        current = self.db.execute(
            "SELECT provider_id FROM model_presets WHERE name=?", (setting["value"],)
        ).fetchone()
        if current is not None and supports_generation(str(current["provider_id"])):
            return
        fallback = self._first_chat_model_preset_name()
        if fallback is None:
            self.db.execute(
                "DELETE FROM project_meta WHERE key=?", (_DEFAULT_CHAT_MODEL_PRESET_KEY,)
            )
        else:
            self.db.execute(
                "UPDATE project_meta SET value=? WHERE key=?",
                (fallback, _DEFAULT_CHAT_MODEL_PRESET_KEY),
            )

    def delete_model_preset(self, preset_id: str) -> str:
        """Delete one saved provider/model configuration without touching chat history."""
        self._require_writable()
        if not isinstance(preset_id, str) or not preset_id.strip():
            raise DocumentError("model preset ID is invalid")
        row = self.db.execute(
            "SELECT name FROM model_presets WHERE id=?", (preset_id,)
        ).fetchone()
        if not row:
            raise DocumentError("unknown model preset")
        preferences = self._chat_reasoning_preferences()
        if preferences.pop(str(row["name"]), None) is not None:
            if preferences:
                self.db.execute(
                    "INSERT INTO project_meta(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (
                        _CHAT_REASONING_PREFERENCES_KEY,
                        json.dumps(
                            preferences,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                )
            else:
                self.db.execute(
                    "DELETE FROM project_meta WHERE key=?",
                    (_CHAT_REASONING_PREFERENCES_KEY,),
                )
        self.db.execute("DELETE FROM model_presets WHERE id=?", (preset_id,))
        self._repair_default_chat_model_preset()
        self.db.commit()
        return str(row["name"])

    def model_preset(self, name: str) -> sqlite3.Row:
        preset = self.db.execute(
            "SELECT id,name,provider_id,credential_alias,model,reasoning,reasoning_mode,reasoning_budget_tokens,parameters_json "
            "FROM model_presets WHERE name=?",
            (name,),
        ).fetchone()
        if not preset:
            raise DocumentError("unknown model preset")
        _validate_model_preset_metadata(preset)
        return preset

    def create_chat_conversation(
        self,
        revision_ids: list[str],
        framework_id: str,
        preset_name: str,
        *,
        retrieval_mode: Literal["lexical", "hybrid"] = "lexical",
        embedding_preset_name: str | None = None,
        embedding_artifact_ids: list[str] | None = None,
        research_project_id: str | None = None,
    ) -> str:
        """Create a saved evidence-bounded conversation over an immutable revision scope."""
        self._require_writable()
        from .frameworks import FRAMEWORK_REVIEWS

        if (
            not isinstance(revision_ids, list)
            or any(not isinstance(item, str) or not item.strip() for item in revision_ids)
            or len(set(revision_ids)) != len(revision_ids)
        ):
            raise DocumentError("chat requires a list of unique revision IDs")
        if (
            not isinstance(framework_id, str)
            or (framework_id != "auto" and framework_id not in FRAMEWORK_REVIEWS)
            or framework_id == "critical-reviewer"
        ):
            raise DocumentError("chat requires an automatic or registered interpretive framework")
        if not isinstance(preset_name, str) or not preset_name.strip():
            raise DocumentError("chat requires a saved generation preset")
        self.model_preset(preset_name.strip())
        normalized_revisions = [item.strip() for item in revision_ids]
        if len(set(normalized_revisions)) != len(normalized_revisions):
            raise DocumentError("chat revision IDs must be unique")
        for revision_id in normalized_revisions:
            if not self.db.execute(
                "SELECT 1 FROM document_revisions WHERE id=?", (revision_id,)
            ).fetchone():
                raise DocumentError("chat scope contains an unknown revision")
        if research_project_id is not None:
            project_document_ids = set(self.research_project_document_ids(research_project_id))
            if not self.document_ids_for_revisions(normalized_revisions).issubset(project_document_ids):
                raise DocumentError("chat sources must belong to the selected research project")
        if retrieval_mode == "lexical":
            if embedding_preset_name is not None or embedding_artifact_ids is not None:
                raise DocumentError("lexical chat cannot include embedding configuration")
        elif retrieval_mode == "hybrid":
            if not isinstance(embedding_preset_name, str) or not embedding_preset_name.strip():
                raise DocumentError("hybrid chat requires an embedding preset")
            if (
                not isinstance(embedding_artifact_ids, list)
                or not embedding_artifact_ids
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in embedding_artifact_ids
                )
                or len(set(embedding_artifact_ids)) != len(embedding_artifact_ids)
            ):
                raise DocumentError("hybrid chat requires unique embedding artifact IDs")
        else:
            raise DocumentError("chat retrieval mode must be lexical or hybrid")
        conversation_id = uid()
        timestamp = now()
        self.db.execute(
            "INSERT INTO chat_conversations(id,title,revision_ids_json,framework_id,preset_name,"
            "retrieval_mode,embedding_preset_name,embedding_artifact_ids_json,created_at,updated_at,research_project_id) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                conversation_id,
                "Nueva conversación",
                json.dumps(normalized_revisions, ensure_ascii=False),
                framework_id,
                preset_name.strip(),
                retrieval_mode,
                embedding_preset_name,
                json.dumps(embedding_artifact_ids or [], ensure_ascii=False),
                timestamp,
                timestamp,
                research_project_id,
            ),
        )
        self.db.commit()
        return conversation_id

    def chat_conversation(self, conversation_id: str) -> dict[str, object]:
        """Return immutable scope and model configuration for one saved thread."""
        row = self.db.execute(
            "SELECT * FROM chat_conversations WHERE id=?", (conversation_id,)
        ).fetchone()
        if row is None:
            raise DocumentError("unknown chat conversation")
        try:
            revision_ids = json.loads(row["revision_ids_json"])
            artifact_ids = json.loads(row["embedding_artifact_ids_json"])
        except (TypeError, json.JSONDecodeError):
            raise DocumentError("stored chat conversation is corrupt") from None
        if (
            not isinstance(revision_ids, list)
            or any(not isinstance(item, str) for item in revision_ids)
            or not isinstance(artifact_ids, list)
            or any(not isinstance(item, str) for item in artifact_ids)
        ):
            raise DocumentError("stored chat conversation is corrupt")
        return {
            "id": str(row["id"]),
            "title": str(row["title"]),
            "revision_ids": revision_ids,
            "framework_id": str(row["framework_id"]),
            "preset_name": str(row["preset_name"]),
            "retrieval_mode": str(row["retrieval_mode"]),
            "embedding_preset_name": row["embedding_preset_name"],
            "embedding_artifact_ids": artifact_ids,
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "research_project_id": row["research_project_id"],
        }

    def touch_chat_conversation(
        self, conversation_id: str, *, title_from_prompt: str | None = None
    ) -> None:
        """Persist last activity and set a concise title for a new conversation."""
        self._require_writable()
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise DocumentError("chat conversation ID is invalid")
        if title_from_prompt is not None and (
            not isinstance(title_from_prompt, str) or len(title_from_prompt) > 8_000
        ):
            raise DocumentError("chat conversation title source is invalid")
        row = self.db.execute(
            "SELECT title FROM chat_conversations WHERE id=?", (conversation_id,)
        ).fetchone()
        if row is None:
            raise DocumentError("unknown chat conversation")
        title = str(row["title"])
        if title == "Nueva conversación" and title_from_prompt is not None:
            title = _chat_title_from_prompt(title_from_prompt)
        self.db.execute(
            "UPDATE chat_conversations SET title=?,updated_at=? WHERE id=?",
            (title, now(), conversation_id),
        )
        self.db.commit()

    def list_chat_conversations(self, *, limit: int = 100) -> list[dict[str, object]]:
        if type(limit) is not int or not 1 <= limit <= 500:
            raise DocumentError("chat listing limit must be between 1 and 500")
        rows = self.db.execute(
            "SELECT id,title,framework_id,preset_name,retrieval_mode,research_project_id,created_at,updated_at "
            "FROM chat_conversations ORDER BY updated_at DESC,id LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def chat_messages(self, conversation_id: str, *, limit: int = 100) -> list[dict[str, object]]:
        self.chat_conversation(conversation_id)
        if type(limit) is not int or not 1 <= limit <= 500:
            raise DocumentError("chat message limit must be between 1 and 500")
        rows = self.db.execute(
            "SELECT id,ordinal,role,content_json,analysis_run_id,finding_id,created_at "
            "FROM chat_messages WHERE conversation_id=? ORDER BY ordinal LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        messages: list[dict[str, object]] = []
        for row in rows:
            try:
                content = json.loads(row["content_json"])
            except (TypeError, json.JSONDecodeError):
                raise DocumentError("stored chat message is corrupt") from None
            messages.append(
                {
                    "id": str(row["id"]),
                    "ordinal": int(row["ordinal"]),
                    "role": str(row["role"]),
                    "content": content,
                    "analysis_run_id": row["analysis_run_id"],
                    "finding_id": row["finding_id"],
                    "created_at": str(row["created_at"]),
                }
            )
        return messages

    def append_chat_turn(
        self,
        conversation_id: str,
        question: str,
        answer: dict[str, object],
    ) -> tuple[str, str]:
        """Persist one complete user/model turn and its analysis lineage atomically."""
        self._require_writable()
        self.chat_conversation(conversation_id)
        if not isinstance(question, str) or not question.strip() or len(question) > 8_000:
            raise DocumentError("chat question must contain 1..8000 characters")
        if not isinstance(answer, dict):
            raise DocumentError("chat answer must be a result object")
        serialized_user = json.dumps({"text": question.strip()}, ensure_ascii=False, allow_nan=False)
        serialized_answer = json.dumps(answer, ensure_ascii=False, allow_nan=False)
        analysis_run_id = answer.get("answer_analysis_run_id") or answer.get(
            "candidate_analysis_run_id"
        )
        finding_id = answer.get("answer_finding_id") or answer.get("candidate_finding_id")
        if analysis_run_id is not None and not isinstance(analysis_run_id, str):
            raise DocumentError("chat answer analysis run ID is invalid")
        if finding_id is not None and not isinstance(finding_id, str):
            raise DocumentError("chat answer Finding ID is invalid")
        first_message_id, second_message_id = uid(), uid()
        created = now()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            first_ordinal = int(
                self.db.execute(
                    "SELECT COALESCE(MAX(ordinal),-1)+1 FROM chat_messages "
                    "WHERE conversation_id=?",
                    (conversation_id,),
                ).fetchone()[0]
            )
            self.db.execute(
                "INSERT INTO chat_messages(id,conversation_id,ordinal,role,content_json,created_at) "
                "VALUES(?,?,?,'user',?,?)",
                (first_message_id, conversation_id, first_ordinal, serialized_user, created),
            )
            self.db.execute(
                "INSERT INTO chat_messages(id,conversation_id,ordinal,role,content_json,"
                "analysis_run_id,finding_id,created_at) VALUES(?,?,?,'assistant',?,?,?,?)",
                (
                    second_message_id,
                    conversation_id,
                    first_ordinal + 1,
                    serialized_answer,
                    analysis_run_id,
                    finding_id,
                    created,
                ),
            )
            title_row = self.db.execute(
                "SELECT title FROM chat_conversations WHERE id=?", (conversation_id,)
            ).fetchone()
            title = _chat_title_from_prompt(question)
            self.db.execute(
                "UPDATE chat_conversations SET title=?,updated_at=? WHERE id=?",
                (title if title_row["title"] == "Nueva conversación" else title_row["title"], created, conversation_id),
            )
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        return first_message_id, second_message_id

    def delete_chat_conversation(self, conversation_id: str) -> None:
        """Delete one thread and its stored turns without changing analysis evidence."""
        self._require_writable()
        cursor = self.db.execute("DELETE FROM chat_conversations WHERE id=?", (conversation_id,))
        if cursor.rowcount != 1:
            raise DocumentError("unknown chat conversation")
        self.db.commit()

    def save_model_price(self, price: ModelPrice) -> str:
        """Add one immutable version of a user-supplied exact-model price record."""
        self._require_writable()
        if not isinstance(price, ModelPrice):
            raise DocumentError("model price must be a validated ModelPrice")
        snapshot = price.to_snapshot()
        existing = self.db.execute(
            "SELECT * FROM model_prices WHERE provider_id=? AND model=? AND rate_version=?",
            (price.provider_id, price.model, price.rate_version),
        ).fetchone()
        if existing:
            existing_price = ModelPrice(
                provider_id=existing["provider_id"],
                model=existing["model"],
                input_per_million_tokens=existing["input_per_million_tokens"],
                output_per_million_tokens=existing["output_per_million_tokens"],
                reasoning_per_million_tokens=existing["reasoning_per_million_tokens"],
                currency=existing["currency"],
                source_reference=existing["source_reference"],
                rate_version=existing["rate_version"],
                rate_date=existing["rate_date"],
            )
            if existing_price != price:
                raise DocumentError(
                    "model price version already exists; add a new rate_version to change rates"
                )
            return str(existing["id"])

        price_id = uid()
        self.db.execute(
            "INSERT INTO model_prices(id,provider_id,model,input_per_million_tokens,"
            "output_per_million_tokens,reasoning_per_million_tokens,currency,source_reference,"
            "rate_version,rate_date,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                price_id,
                price.provider_id,
                price.model,
                snapshot["input_per_million_tokens"],
                snapshot["output_per_million_tokens"],
                snapshot["reasoning_per_million_tokens"],
                price.currency,
                price.source_reference,
                price.rate_version,
                price.rate_date,
                now(),
            ),
        )
        self.db.commit()
        return price_id

    def model_price(
        self, provider_id: str, model: str, rate_version: str | None = None
    ) -> ModelPrice | None:
        """Return the latest dated rate, or a specific immutable rate version."""
        if rate_version is None:
            row = self.db.execute(
                "SELECT * FROM model_prices WHERE provider_id=? AND model=? "
                "ORDER BY rate_date DESC,created_at DESC,id DESC LIMIT 1",
                (provider_id, model),
            ).fetchone()
        else:
            row = self.db.execute(
                "SELECT * FROM model_prices WHERE provider_id=? AND model=? AND rate_version=?",
                (provider_id, model, rate_version),
            ).fetchone()
        if row is None:
            return None
        return ModelPrice(
            provider_id=row["provider_id"],
            model=row["model"],
            input_per_million_tokens=row["input_per_million_tokens"],
            output_per_million_tokens=row["output_per_million_tokens"],
            reasoning_per_million_tokens=row["reasoning_per_million_tokens"],
            currency=row["currency"],
            source_reference=row["source_reference"],
            rate_version=row["rate_version"],
            rate_date=row["rate_date"],
        )

    def model_costs_page(self, *, offset: int = 0, limit: int = 100) -> dict[str, object]:
        """Return a bounded, secret-free page of stored model execution costs."""
        sqlite_int64_max = 0x7FFF_FFFF_FFFF_FFFF
        if (
            type(offset) is not int
            or not 0 <= offset <= sqlite_int64_max
            or type(limit) is not int
            or not 1 <= limit <= 1_000
        ):
            raise DocumentError(
                "model cost page requires an integer offset in SQLite range and integer limit 1..1000"
            )

        total_count = int(self.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0])
        stored_rows = self.db.execute(
            "SELECT id,analysis_run_id,provider_id,model,input_tokens,output_tokens,"
            "reasoning_tokens,model_price_json,estimated_cost,estimated_cost_currency,"
            "error_type,started_at,completed_at FROM model_executions "
            "ORDER BY started_at,id LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        rows: list[dict[str, object]] = []
        for row in stored_rows:
            raw_snapshot = row["model_price_json"]
            if raw_snapshot is None:
                price_snapshot: dict[str, str | None] | None = None
            else:
                try:
                    price = ModelPrice.from_snapshot(json.loads(raw_snapshot))
                    if price.provider_id != row["provider_id"] or price.model != row["model"]:
                        raise ValueError("model price snapshot does not match execution")
                    price_snapshot = price.to_snapshot()
                except (ProjectError, KeyError, TypeError, ValueError, RecursionError):
                    raise ProjectError("stored model price snapshot is invalid") from None

            rows.append(
                {
                    "execution_id": row["id"],
                    "analysis_run_id": row["analysis_run_id"],
                    "provider_id": row["provider_id"],
                    "model": row["model"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "reasoning_tokens": row["reasoning_tokens"],
                    "estimated_cost": row["estimated_cost"],
                    "estimated_cost_currency": row["estimated_cost_currency"],
                    "model_price_snapshot": price_snapshot,
                    "error_type": row["error_type"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                }
            )

        return {
            "offset": offset,
            "limit": limit,
            "total_count": total_count,
            "has_more": offset + len(rows) < total_count,
            "rows": rows,
        }

    def save_model_capability_profile(self, profile: ModelCapabilityProfile) -> str:
        if not profile.provider_id.strip() or not profile.model.strip():
            raise DocumentError("model capability profile requires provider and exact model")
        if profile.assurance not in {"UNASSESSED", "TESTED", "VALIDATED"}:
            raise DocumentError("unknown model capability assurance")
        capabilities = profile.capabilities
        for name in (
            "text",
            "system_prompt",
            "vision",
            "structured_output",
            "streaming",
            "embeddings",
        ):
            if type(getattr(capabilities, name)) is not bool:
                raise DocumentError(f"{name} capability must be boolean")
        for name, value in (
            ("reasoning_efforts", capabilities.reasoning_efforts),
            ("reasoning_modes", capabilities.reasoning_modes),
            ("generation_parameters", capabilities.generation_parameters),
        ):
            if not isinstance(value, tuple) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise DocumentError(f"{name} must contain non-empty strings")
        allowed_reasoning_modes = {"adaptive", "manual"}
        if profile.provider_id == "cohere":
            allowed_reasoning_modes = {"enabled", "disabled", "manual"}
        if not set(capabilities.reasoning_modes) <= allowed_reasoning_modes:
            raise DocumentError("unknown reasoning mode capability")
        payload = {
            "text": capabilities.text,
            "system_prompt": capabilities.system_prompt,
            "vision": capabilities.vision,
            "structured_output": capabilities.structured_output,
            "streaming": capabilities.streaming,
            "embeddings": capabilities.embeddings,
            "reasoning_efforts": list(capabilities.reasoning_efforts),
            "reasoning_modes": list(capabilities.reasoning_modes),
            "generation_parameters": list(capabilities.generation_parameters),
        }
        profile_id = uid()
        self.db.execute(
            """
            INSERT INTO model_capability_profiles(
                id,provider_id,model,capabilities_json,assurance,source_reference,updated_at
            ) VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(provider_id,model) DO UPDATE SET
                capabilities_json=excluded.capabilities_json,
                assurance=excluded.assurance,
                source_reference=excluded.source_reference,
                updated_at=excluded.updated_at
            """,
            (
                profile_id,
                profile.provider_id,
                profile.model,
                json.dumps(payload, sort_keys=True),
                profile.assurance,
                profile.source_reference,
                now(),
            ),
        )
        self.db.commit()
        return profile_id

    def model_capability_profile(
        self, provider_id: str, model: str
    ) -> ModelCapabilityProfile | None:
        row = self.db.execute(
            "SELECT * FROM model_capability_profiles WHERE provider_id=? AND model=?",
            (provider_id, model),
        ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row["capabilities_json"])
            if not isinstance(payload, dict):
                raise TypeError("not an object")
            capabilities = ModelCapabilities(
                text=payload["text"],
                system_prompt=payload.get("system_prompt", True),
                vision=payload["vision"],
                structured_output=payload["structured_output"],
                streaming=payload["streaming"],
                embeddings=payload["embeddings"],
                reasoning_efforts=tuple(payload["reasoning_efforts"]),
                reasoning_modes=tuple(payload.get("reasoning_modes", ())),
                generation_parameters=tuple(payload["generation_parameters"]),
            )
            if any(
                type(getattr(capabilities, name)) is not bool
                for name in (
                    "text",
                    "system_prompt",
                    "vision",
                    "structured_output",
                    "streaming",
                    "embeddings",
                )
            ):
                raise ValueError("invalid boolean capability")
            if any(
                not all(isinstance(item, str) and item for item in values)
                for values in (
                    capabilities.reasoning_efforts,
                    capabilities.reasoning_modes,
                    capabilities.generation_parameters,
                )
            ):
                raise ValueError("invalid capability names")
        except (KeyError, TypeError, ValueError) as exc:
            raise DocumentError("model capability profile is corrupt") from exc
        return ModelCapabilityProfile(
            row["provider_id"],
            row["model"],
            capabilities,
            row["assurance"],
            row["source_reference"],
        )

    def list_model_capability_profiles(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT id,provider_id,model,capabilities_json,assurance,source_reference,updated_at "
            "FROM model_capability_profiles ORDER BY provider_id,model"
        ).fetchall()

    def save_model_routing_profile(
        self, name: str, routes: dict[str, str], fallback_preset: str | None = None
    ) -> str:
        """Persist explicit per-stage model selection without fallback guessing."""
        if not name.strip() or not routes:
            raise DocumentError("routing profile requires a name and at least one route")
        if any(not stage.strip() or not preset.strip() for stage, preset in routes.items()):
            raise DocumentError("routing profile routes must have non-empty stage and preset names")
        presets = set(routes.values()) | ({fallback_preset} if fallback_preset else set())
        for preset_name in presets:
            self.model_preset(preset_name)
        routing_id = uid()
        try:
            self.db.execute(
                "INSERT INTO model_routing_profiles VALUES(?,?,?,?,?)",
                (
                    routing_id,
                    name.strip(),
                    json.dumps(routes, sort_keys=True),
                    fallback_preset,
                    now(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise DocumentError("model routing profile name already exists") from exc
        self.db.commit()
        return routing_id

    def model_routing_profile(self, name: str) -> tuple[dict[str, str], str | None]:
        row = self.db.execute(
            "SELECT routes_json,fallback_preset FROM model_routing_profiles WHERE name=?", (name,)
        ).fetchone()
        if not row:
            raise DocumentError("unknown model routing profile")
        try:
            routes = json.loads(row["routes_json"])
        except json.JSONDecodeError as exc:
            raise DocumentError("model routing profile is corrupt") from exc
        if not isinstance(routes, dict) or not all(
            isinstance(stage, str) and stage and isinstance(preset, str) and preset
            for stage, preset in routes.items()
        ):
            raise DocumentError("model routing profile is corrupt")
        return routes, row["fallback_preset"]

    def list_model_routing_profiles(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT id,name,routes_json,fallback_preset,created_at "
            "FROM model_routing_profiles ORDER BY name"
        ).fetchall()

    def save_analysis_profile(self, name: str, payload: dict[str, object]) -> str:
        if not name.strip():
            raise DocumentError("analysis profile name is required")
        normalized_payload = _analysis_profile_with_defaults(payload)
        if _contains_secret_config_key(normalized_payload, allow_counterevidence_query_labels=True):
            raise DocumentError("analysis profile must not contain secret data")
        profile_id = uid()
        try:
            self.db.execute(
                "INSERT INTO analysis_profiles VALUES(?,?,?,?)",
                (
                    profile_id,
                    name.strip(),
                    json.dumps(normalized_payload, sort_keys=True),
                    now(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise DocumentError("analysis profile name already exists") from exc
        self.db.commit()
        return profile_id

    def analysis_profile(self, name: str) -> dict[str, object]:
        row = self.db.execute(
            "SELECT payload_json FROM analysis_profiles WHERE name=?", (name,)
        ).fetchone()
        if not row:
            raise DocumentError("unknown analysis profile")
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError as exc:
            raise DocumentError("analysis profile is corrupt") from exc
        if not isinstance(payload, dict):
            raise DocumentError("analysis profile is corrupt")
        if _contains_secret_config_key(payload, allow_counterevidence_query_labels=True):
            raise DocumentError("analysis profile contains unsafe credential metadata")
        return _analysis_profile_with_defaults(payload)

    def list_analysis_profiles(self) -> list[dict[str, str]]:
        """List declarative profile metadata without executing or loading plugins."""
        rows = self.db.execute(
            "SELECT id,name,payload_json,created_at FROM analysis_profiles ORDER BY name"
        ).fetchall()
        profiles: list[dict[str, str]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError as exc:
                raise DocumentError("analysis profile is corrupt") from exc
            if not isinstance(payload, dict):
                raise DocumentError("analysis profile is corrupt")
            if _contains_secret_config_key(payload, allow_counterevidence_query_labels=True):
                raise DocumentError("analysis profile contains unsafe credential metadata")
            payload = _analysis_profile_with_defaults(payload)
            version = payload.get("version")
            kind = payload.get("kind")
            if not isinstance(version, str) or not isinstance(kind, str):
                raise DocumentError("analysis profile is corrupt")
            profiles.append(
                {
                    "profile_id": str(row["id"]),
                    "name": str(row["name"]),
                    "version": version,
                    "kind": kind,
                    "created_at": str(row["created_at"]),
                }
            )
        return profiles

    def credential_reference(self, provider_id: str, alias: str) -> sqlite3.Row:
        if not isinstance(alias, str) or not alias:
            raise DocumentError("unknown provider credential alias")
        try:
            validate_credential_alias(alias)
        except ProjectError:
            raise DocumentError("credential alias must be a short human-readable label") from None
        credential = self.db.execute(
            "SELECT provider_id,credential_alias,secret_reference,endpoint "
            "FROM provider_credentials WHERE provider_id=? AND credential_alias=?",
            (provider_id, alias),
        ).fetchone()
        if not credential:
            raise DocumentError("unknown provider credential alias")
        _validate_stored_credential_alias(credential["credential_alias"])
        try:
            provider_auth_reference(
                credential["provider_id"],
                credential["secret_reference"],
                credential["endpoint"],
            )
        except ProjectError:
            raise DocumentError("stored credential metadata is unsafe") from None
        return credential

    def validate_model_execution_skill_run(
        self, analysis_run_id: str | None, skill_run_id: str | None
    ) -> None:
        """Validate an optional model-call link against the same analysis run."""
        if skill_run_id is None:
            return
        if not isinstance(skill_run_id, str) or not skill_run_id.strip():
            raise DocumentError("model execution skill run reference is invalid")
        if analysis_run_id is None:
            raise DocumentError("model execution skill run requires an analysis run")
        skill_run = self.db.execute(
            "SELECT analysis_run_id FROM skill_runs WHERE id=?", (skill_run_id,)
        ).fetchone()
        if skill_run is None or skill_run["analysis_run_id"] != analysis_run_id:
            raise DocumentError("model execution skill run must belong to the same analysis run")

    def record_model_execution(
        self,
        provider_id: str,
        credential_alias: str,
        model: str,
        reasoning: str | None,
        request_hash: str,
        response_hash: str | None,
        error_type: str | None = None,
        analysis_run_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        latency_ms: int | None = None,
        parameters: dict[str, object] | None = None,
        capabilities: dict[str, object] | None = None,
        reasoning_mode: str | None = None,
        reasoning_budget_tokens: int | None = None,
        model_price: ModelPrice | None = None,
        endpoint: str | None = None,
        skill_run_id: str | None = None,
    ) -> str:
        try:
            validate_credential_alias(credential_alias)
        except ProjectError:
            raise DocumentError("credential alias must be a short human-readable label") from None
        if (
            analysis_run_id is not None
            and not self.db.execute(
                "SELECT 1 FROM analysis_runs WHERE id=?", (analysis_run_id,)
            ).fetchone()
        ):
            raise DocumentError("model execution references unknown analysis run")
        self.validate_model_execution_skill_run(analysis_run_id, skill_run_id)
        try:
            _validate_provider_endpoint_metadata(endpoint)
        except ProjectError:
            raise DocumentError(
                "model execution endpoint must not contain credential material"
            ) from None
        parameters = parameters or {}
        if _contains_secret_config_key(parameters):
            raise DocumentError("model execution parameters must not contain secrets")
        try:
            parameters_json = json.dumps(parameters, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DocumentError("model execution parameters must be JSON-compatible") from exc
        capabilities = capabilities or {}
        if _contains_secret_config_key(capabilities):
            raise DocumentError("model execution capabilities must not contain secrets")
        try:
            capabilities_json = json.dumps(capabilities, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DocumentError("model execution capabilities must be JSON-compatible") from exc
        model_price_json = None
        estimated_cost_text = None
        estimated_cost_currency = None
        if model_price is not None:
            if not isinstance(model_price, ModelPrice):
                raise DocumentError("model execution price must be a validated ModelPrice")
            if model_price.provider_id != provider_id or model_price.model != model:
                raise DocumentError(
                    "model execution price must match the executed provider and model"
                )
            model_price_json = json.dumps(
                model_price.to_snapshot(), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
            )
            estimated_cost = estimate_model_cost(model_price, usage)
            if estimated_cost is not None:
                estimated_cost_text = decimal_text(estimated_cost)
                estimated_cost_currency = model_price.currency
        execution_id = uid()
        self.db.execute(
            "INSERT INTO model_executions(id,analysis_run_id,provider_id,credential_alias,model,reasoning,reasoning_mode,reasoning_budget_tokens,request_hash,response_hash,error_type,input_tokens,output_tokens,reasoning_tokens,latency_ms,started_at,completed_at,parameters_json,capabilities_json,model_price_json,estimated_cost,estimated_cost_currency,endpoint,skill_run_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                execution_id,
                analysis_run_id,
                provider_id,
                credential_alias,
                model,
                reasoning,
                reasoning_mode,
                reasoning_budget_tokens,
                request_hash,
                response_hash,
                error_type,
                input_tokens,
                output_tokens,
                reasoning_tokens,
                latency_ms,
                now(),
                now(),
                parameters_json,
                capabilities_json,
                model_price_json,
                estimated_cost_text,
                estimated_cost_currency,
                endpoint,
                skill_run_id,
            ),
        )
        self.db.commit()
        return execution_id

    def create_skill_run(
        self,
        analysis_run_id: str,
        scope_id: str,
        skill_id: str,
        skill_version: str,
        executor: str,
        config_hash: str,
        input_finding_id: str | None = None,
        coverage_id: str | None = None,
    ) -> str:
        parent = self.db.execute(
            "SELECT scope_id,status FROM analysis_runs WHERE id=?", (analysis_run_id,)
        ).fetchone()
        if not parent:
            raise EvidenceValidationError("skill run references unknown analysis run")
        if parent["scope_id"] != scope_id:
            raise ScopeViolationError("skill run must retain the analysis run ScopeSnapshot")
        if coverage_id is not None:
            coverage = self.coverage_descriptor(coverage_id)
            if coverage.scope_snapshot_id != scope_id:
                raise ScopeViolationError(
                    "skill run coverage must retain its original ScopeSnapshot"
                )
        if parent["status"] not in {RunStatus.PENDING, RunStatus.RUNNING}:
            raise DocumentError("cannot start a skill in a finalized analysis run")
        if input_finding_id is not None:
            input_finding = self.db.execute(
                """
                SELECT ar.scope_id FROM findings f
                JOIN analysis_runs ar ON ar.id=f.analysis_run_id
                WHERE f.id=?
                """,
                (input_finding_id,),
            ).fetchone()
            if not input_finding:
                raise EvidenceValidationError("skill run references unknown input finding")
            if input_finding["scope_id"] != scope_id:
                raise ScopeViolationError("input finding must retain the skill run ScopeSnapshot")
        skill_run_id = uid()
        self.db.execute(
            """
            INSERT INTO skill_runs(
                id,analysis_run_id,skill_id,skill_version,scope_id,status,executor,config_hash,
                coverage_id,input_finding_id,started_at,completed_at,error_type
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                skill_run_id,
                analysis_run_id,
                skill_id,
                skill_version,
                scope_id,
                RunStatus.RUNNING,
                executor,
                config_hash,
                coverage_id,
                input_finding_id,
                now(),
                None,
                None,
            ),
        )
        self.db.commit()
        return skill_run_id

    def complete_skill_run(self, skill_run_id: str) -> None:
        cur = self.db.execute(
            "UPDATE skill_runs SET status=?,completed_at=? WHERE id=? AND status=?",
            (RunStatus.COMPLETED, now(), skill_run_id, RunStatus.RUNNING),
        )
        if not cur.rowcount:
            raise DocumentError("skill run is immutable or unknown")
        self.db.commit()

    def set_running_skill_coverage(self, skill_run_id: str, coverage_id: str) -> None:
        """Attach retrieval coverage once while a skill run is still active."""
        skill = self.db.execute(
            "SELECT scope_id,status,coverage_id FROM skill_runs WHERE id=?", (skill_run_id,)
        ).fetchone()
        if skill is None or skill["status"] != RunStatus.RUNNING or skill["coverage_id"] is not None:
            raise DocumentError("coverage can only be assigned once to a running skill")
        coverage = self.coverage_descriptor(coverage_id)
        if coverage.scope_snapshot_id != skill["scope_id"]:
            raise ScopeViolationError("skill coverage must retain its original ScopeSnapshot")
        cur = self.db.execute(
            "UPDATE skill_runs SET coverage_id=? WHERE id=? AND status=? AND coverage_id IS NULL",
            (coverage_id, skill_run_id, RunStatus.RUNNING),
        )
        if not cur.rowcount:
            raise DocumentError("skill run coverage is immutable or unknown")
        self.db.commit()

    def skip_skill_run(self, skill_run_id: str, reason: str) -> None:
        """Persist an intentionally omitted stage without misreporting it as completed."""
        if not reason.strip():
            raise DocumentError("skipped skill run requires a reason")
        cur = self.db.execute(
            "UPDATE skill_runs SET status=?,completed_at=?,skip_reason=? WHERE id=? AND status=?",
            (RunStatus.SKIPPED, now(), reason, skill_run_id, RunStatus.RUNNING),
        )
        if not cur.rowcount:
            raise DocumentError("skill run is immutable or unknown")
        self.db.commit()

    def finalize_run(self, rid: str):
        run = self.db.execute("SELECT status FROM analysis_runs WHERE id=?", (rid,)).fetchone()
        if run is None or run["status"] not in {RunStatus.PENDING, RunStatus.RUNNING}:
            raise DocumentError("run is immutable or unknown")
        incomplete = self.db.execute(
            "SELECT 1 FROM skill_runs WHERE analysis_run_id=? AND status NOT IN (?,?,?) LIMIT 1",
            (rid, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.SKIPPED),
        ).fetchone()
        if incomplete:
            raise DocumentError("cannot finalize an analysis run with incomplete skill runs")
        completed_at = now()
        self.db.execute("SAVEPOINT finalize_analysis_run")
        try:
            self.db.execute(
                "UPDATE findings SET finalized_at=? WHERE analysis_run_id=? AND finalized_at IS NULL",
                (completed_at, rid),
            )
            cur = self.db.execute(
                "UPDATE analysis_runs SET status=?,completed_at=? WHERE id=? AND status IN (?,?)",
                (RunStatus.COMPLETED, completed_at, rid, RunStatus.PENDING, RunStatus.RUNNING),
            )
            if not cur.rowcount:
                raise DocumentError("run is immutable or unknown")
            self.db.execute("RELEASE SAVEPOINT finalize_analysis_run")
        except BaseException:
            if self.db.in_transaction:
                try:
                    self.db.execute("ROLLBACK TO SAVEPOINT finalize_analysis_run")
                    self.db.execute("RELEASE SAVEPOINT finalize_analysis_run")
                except sqlite3.Error:
                    self.db.rollback()
            raise
        self.db.commit()
        row = self.db.execute("SELECT scope_id FROM analysis_runs WHERE id=?", (rid,)).fetchone()
        log_event(
            _LOGGER,
            "analysis.run.completed",
            analysis_run_id=rid,
            scope_id=row["scope_id"] if row else None,
            status=RunStatus.COMPLETED,
        )

    def _new_page(self, revision_id, index, width, height, rotation, start, end, text):
        page_id = uid()
        self.db.execute(
            "INSERT INTO pages VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                page_id,
                revision_id,
                index,
                str(index + 1),
                width,
                height,
                rotation,
                start,
                end,
                text,
            ),
        )
        return page_id

    def _segment_for_range(self, revision_id, page_id, start, end, boxes):
        row = self.db.execute(
            "SELECT id FROM source_spans WHERE revision_id=? AND start_offset=? AND end_offset=?",
            (revision_id, start, end),
        ).fetchone()
        span = row[0] if row else self.create_span(revision_id, start, end)
        self.db.execute(
            "INSERT INTO source_span_segments("
            "id,span_id,paragraph_id,page_number,start_offset,end_offset,bounding_boxes_json,"
            "bbox_status,bbox_projection_version) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                uid(),
                span,
                None,
                page_id,
                start,
                end,
                json.dumps(boxes),
                "APPROXIMATE" if boxes else "UNAVAILABLE",
                "paragraph-block-v1",
            ),
        )
        return span

    def span_segments(self, span_id):
        return self.db.execute(
            "SELECT * FROM source_span_segments WHERE span_id=? ORDER BY start_offset,end_offset,id",
            (span_id,),
        ).fetchall()

    def project_span_to_segments(self, span_id: str, *, commit: bool = True):
        from .documents import BBOX_PROJECTION_VERSION, project_pdf_text_boxes

        row = self.db.execute(
            "SELECT revision_id,start_offset,end_offset FROM source_spans WHERE id=?", (span_id,)
        ).fetchone()
        if not row:
            raise SourceSpanError("unknown span")
        existing = self.span_segments(span_id)
        if existing and all(
            segment["bbox_projection_version"] == BBOX_PROJECTION_VERSION for segment in existing
        ):
            return existing

        revision_id, span_start, span_end = row[0], row[1], row[2]
        pages = self.db.execute(
            "SELECT id,page_index,canonical_start,canonical_end FROM pages "
            "WHERE revision_id=? AND canonical_end>? AND canonical_start<? ORDER BY page_index",
            (row[0], row[1], row[2]),
        ).fetchall()
        revision = self.db.execute(
            "SELECT mime_type,ocr_language,ocr_dpi FROM document_revisions WHERE id=?",
            (revision_id,),
        ).fetchone()
        source_bytes = None
        if revision and revision["mime_type"] == "application/pdf":
            try:
                source_bytes = self.original_bytes(revision_id)
            except DocumentError:
                source_bytes = None

        retained_page_ids: set[str] = set()
        for page in pages:
            start = max(span_start, page["canonical_start"])
            end = min(span_end, page["canonical_end"])
            paragraph_rows = self.db.execute(
                "SELECT id,start_offset,end_offset,text FROM paragraphs "
                "WHERE revision_id=? AND page_id=? AND end_offset>? AND start_offset<? "
                "ORDER BY start_offset",
                (revision_id, page["id"], start, end),
            ).fetchall()
            if source_bytes is not None:
                boxes, bbox_status, paragraph_id = project_pdf_text_boxes(
                    source_bytes,
                    page["page_index"],
                    [dict(paragraph) for paragraph in paragraph_rows],
                    start,
                    end,
                    ocr_language=revision["ocr_language"] if revision else None,
                    ocr_dpi=revision["ocr_dpi"] if revision else None,
                )
            else:
                boxes = []
                bbox_status = "UNAVAILABLE"
                paragraph_id = paragraph_rows[0]["id"] if len(paragraph_rows) == 1 else None

            retained_page_ids.add(page["id"])
            current = next(
                (segment for segment in existing if segment["page_number"] == page["id"]),
                None,
            )
            if current:
                self.db.execute(
                    "UPDATE source_span_segments SET paragraph_id=?,start_offset=?,end_offset=?,"
                    "bounding_boxes_json=?,bbox_status=?,bbox_projection_version=? WHERE id=?",
                    (
                        paragraph_id,
                        start,
                        end,
                        json.dumps(boxes),
                        bbox_status,
                        BBOX_PROJECTION_VERSION,
                        current["id"],
                    ),
                )
            else:
                self.db.execute(
                    "INSERT INTO source_span_segments("
                    "id,span_id,paragraph_id,page_number,start_offset,end_offset,bounding_boxes_json,"
                    "bbox_status,bbox_projection_version) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        uid(),
                        span_id,
                        paragraph_id,
                        page["id"],
                        start,
                        end,
                        json.dumps(boxes),
                        bbox_status,
                        BBOX_PROJECTION_VERSION,
                    ),
                )
        for segment in existing:
            if segment["page_number"] not in retained_page_ids:
                self.db.execute("DELETE FROM source_span_segments WHERE id=?", (segment["id"],))
        if commit:
            self.db.commit()
        return self.span_segments(span_id)

    def create_finding(
        self,
        analysis_run_id: str,
        skill_id: str,
        skill_version: str,
        phenomenon: str,
        observation: str,
        skill_run_id: str | None = None,
        interpretation: str | None = None,
        alternative_interpretation: str | None = None,
        claim_types: list[ClaimType] | None = None,
        evidence_sufficiency: str = "LIMITED",
        interpretation_confidence: str = "LOW",
        limitations: list[str] | None = None,
        coverage_id: str | None = None,
    ) -> str:
        """Persist a finding only after its claim/coverage metadata passes structural checks.

        UNIVERSAL claims are intentionally bounded to one declared population item per
        revision in the analysis run's immutable ScopeSnapshot. Full citation coverage is
        checked once evidence has been attached and the finding is validated.
        """
        run = self.db.execute(
            "SELECT scope_id,status FROM analysis_runs WHERE id=?", (analysis_run_id,)
        ).fetchone()
        if not run:
            raise EvidenceValidationError("unknown analysis run")
        if run["status"] not in {RunStatus.PENDING, RunStatus.RUNNING}:
            raise EvidenceValidationError("cannot add a finding to an immutable analysis run")
        if skill_run_id is not None:
            skill_run = self.db.execute(
                "SELECT analysis_run_id,scope_id,skill_id,skill_version FROM skill_runs WHERE id=?",
                (skill_run_id,),
            ).fetchone()
            if (
                not skill_run
                or skill_run["analysis_run_id"] != analysis_run_id
                or skill_run["scope_id"] != run["scope_id"]
                or skill_run["skill_id"] != skill_id
                or skill_run["skill_version"] != skill_version
            ):
                raise EvidenceValidationError("skill run provenance does not match finding")
        parsed_claims = [ClaimType(item) for item in (claim_types or [])]
        self._validate_claim_contract(
            run["scope_id"],
            parsed_claims,
            coverage_id,
            limitations or [],
            interpretation=interpretation,
            alternative_interpretation=alternative_interpretation,
        )
        finding_id = uid()
        self.db.execute(
            """
            INSERT INTO findings(
                id,analysis_run_id,skill_run_id,skill_id,skill_version,phenomenon,observation,
                interpretation,alternative_interpretation,claim_types_json,coverage_id,
                evidence_sufficiency,interpretation_confidence,limitations_json,review_status,
                created_at,finalized_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                finding_id,
                analysis_run_id,
                skill_run_id,
                skill_id,
                skill_version,
                phenomenon,
                observation,
                interpretation,
                alternative_interpretation,
                json.dumps([str(item) for item in parsed_claims]),
                coverage_id,
                evidence_sufficiency,
                interpretation_confidence,
                json.dumps(limitations or []),
                "GENERATED",
                now(),
                None,
            ),
        )
        self.db.commit()
        return finding_id

    def _validate_claim_contract(
        self,
        scope_id: str,
        claim_types: list[ClaimType],
        coverage_id: str | None,
        limitations: list[str],
        *,
        interpretation: str | None,
        alternative_interpretation: str | None,
    ) -> None:
        """Reject structural overclaims before a finding becomes persistent."""
        coverage = self.coverage_descriptor(coverage_id) if coverage_id else None
        if coverage and coverage.scope_snapshot_id != scope_id:
            raise EvidenceValidationError(
                "coverage descriptor must use the finding's original ScopeSnapshot"
            )
        requires_coverage = {
            ClaimType.ABSENCE,
            ClaimType.UNIVERSAL,
            ClaimType.QUANTIFIED,
            ClaimType.INTERPRETIVE,
            ClaimType.COMPARATIVE,
        }
        if any(claim in requires_coverage for claim in claim_types) and not coverage:
            raise EvidenceValidationError("claim type requires a coverage descriptor")
        if coverage:
            for claim in claim_types:
                if claim is ClaimType.QUANTIFIED:
                    if (
                        coverage.coverage_mode is not CoverageMode.EXHAUSTIVE
                        or not coverage.population_definition
                        or coverage.population_count is None
                    ):
                        raise EvidenceValidationError(
                            "quantified claim requires exhaustive defined coverage"
                        )
                elif claim is ClaimType.COMPARATIVE:
                    if (
                        coverage.coverage_mode is not CoverageMode.EXHAUSTIVE
                        or not coverage.population_definition
                        or coverage.population_count is None
                    ):
                        raise EvidenceValidationError(
                            "comparative claim requires exhaustive coverage over two defined populations"
                        )
                elif claim is ClaimType.UNIVERSAL:
                    scoped_revision_count = len(self.scope_revisions(scope_id))
                    if (
                        not claim_allowed(claim, coverage.coverage_mode)
                        or not coverage.population_definition
                        or coverage.population_count is None
                        or coverage.population_count != scoped_revision_count
                        or coverage.inspection_count != coverage.population_count
                        or not is_scope_revision_population_definition(
                            coverage.population_definition
                        )
                    ):
                        raise EvidenceValidationError(
                            "universal claim requires exhaustive coverage with one population item "
                            "per scoped revision and complete inspection"
                        )
                elif claim is ClaimType.ABSENCE:
                    if (
                        not claim_allowed(
                            claim,
                            coverage.coverage_mode,
                            negative_result_status=coverage.negative_result_status,
                        )
                        or coverage.termination_state is not CoverageTerminationState.COMPLETED
                        or not coverage.population_definition
                        or coverage.population_count is None
                        or coverage.inspection_count != coverage.population_count
                        or coverage.negative_result_status
                        is not NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION
                    ):
                        raise EvidenceValidationError(
                            "absence requires completed exhaustive coverage with full inspection and "
                            "ABSENCE_IN_EXHAUSTIVE_POPULATION status"
                        )
                elif claim is ClaimType.INTERPRETIVE and not claim_allowed(
                    claim,
                    coverage.coverage_mode,
                    limitations=any(isinstance(item, str) and item.strip() for item in limitations),
                    interpretation=bool(interpretation and interpretation.strip()),
                    alternative_reading=bool(
                        alternative_interpretation and alternative_interpretation.strip()
                    ),
                ):
                    raise EvidenceValidationError(
                        "interpretive claim requires an interpretation, an alternative reading, and stated limitations"
                    )

    def attach_evidence(
        self,
        finding_id: str,
        evidence_id: str,
        role: EvidenceRole,
        rationale: str | None = None,
    ) -> None:
        try:
            role = EvidenceRole(role)
        except ValueError as exc:
            raise EvidenceValidationError("unknown finding evidence role") from exc
        self._require_active_finding_run(finding_id)
        if not self.db.execute(
            "SELECT 1 FROM evidence_artifacts WHERE id=?", (evidence_id,)
        ).fetchone():
            raise EvidenceValidationError("unknown evidence")
        self.db.execute(
            "INSERT INTO finding_evidence VALUES(?,?,?,?)",
            (finding_id, evidence_id, role, rationale),
        )
        self.db.commit()

    def _require_active_finding_run(self, finding_id: str) -> sqlite3.Row:
        row = self.db.execute(
            "SELECT ar.status FROM findings f JOIN analysis_runs ar ON ar.id=f.analysis_run_id "
            "WHERE f.id=?",
            (finding_id,),
        ).fetchone()
        if row is None:
            raise EvidenceValidationError("unknown finding")
        if row["status"] not in {RunStatus.PENDING, RunStatus.RUNNING}:
            raise EvidenceValidationError("finding belongs to an immutable analysis run")
        return row

    def review_finding(self, finding_id: str, status: str, note: str | None = None) -> str:
        """Append a human review decision without mutating a finding's analysis content."""
        allowed = {"ACCEPTED", "MODIFIED", "REJECTED"}
        normalized_status = status.upper()
        if normalized_status not in allowed:
            raise EvidenceValidationError(
                "finding review status must be ACCEPTED, MODIFIED, or REJECTED"
            )
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if not self.db.execute("SELECT 1 FROM findings WHERE id=?", (finding_id,)).fetchone():
                raise EvidenceValidationError("unknown finding")
            review_order = self.db.execute(
                "SELECT COALESCE(MAX(review_order),0)+1 FROM finding_reviews WHERE finding_id=?",
                (finding_id,),
            ).fetchone()[0]
            review_id = uid()
            self.db.execute(
                "INSERT INTO finding_reviews"
                "(id,finding_id,status,note,created_at,review_order) VALUES(?,?,?,?,?,?)",
                (review_id, finding_id, normalized_status, note, now(), review_order),
            )
            self.db.commit()
            return review_id
        except BaseException:
            self.db.rollback()
            raise

    def finding_reviews(
        self, finding_id: str, limit: int | None = None, offset: int = 0
    ) -> list[sqlite3.Row]:
        if not self.db.execute("SELECT 1 FROM findings WHERE id=?", (finding_id,)).fetchone():
            raise EvidenceValidationError("unknown finding")
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 1_000):
            raise EvidenceValidationError("finding review limit must be between 1 and 1000")
        if type(offset) is not int or not 0 <= offset <= 0x7FFF_FFFF_FFFF_FFFF:
            raise EvidenceValidationError("finding review offset must be a non-negative integer")
        query = (
            "SELECT id,status,note,created_at,review_order FROM finding_reviews "
            "WHERE finding_id=? ORDER BY review_order"
        )
        params: list[str | int] = [finding_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        elif offset:
            query += " LIMIT -1"
        if offset:
            query += " OFFSET ?"
            params.append(offset)
        return self.db.execute(query, params).fetchall()

    def finding_evidence(self, finding_id: str, limit: int | None = None) -> list[sqlite3.Row]:
        if limit is not None and not 1 <= limit <= 1_000:
            raise EvidenceValidationError("finding evidence limit must be between 1 and 1000")
        query = (
            "SELECT e.*,fe.role,fe.rationale FROM finding_evidence fe "
            "JOIN evidence_artifacts e ON e.id=fe.evidence_id "
            "LEFT JOIN source_spans s ON s.id=e.span_id WHERE fe.finding_id=? "
            "ORDER BY s.revision_id,s.start_offset,s.end_offset,e.created_at,e.id"
        )
        if limit is not None:
            query += " LIMIT ?"
            return self.db.execute(query, (finding_id, limit)).fetchall()
        return self.db.execute(query, (finding_id,)).fetchall()

    def create_finding_observation_candidates(
        self, finding_id: str, candidates: Iterable[Mapping[str, str]]
    ) -> list[str]:
        """Persist model descriptions as explicitly unverified, evidence-linked candidates."""
        self._require_active_finding_run(finding_id)
        finding = self.db.execute(
            "SELECT ar.scope_id FROM findings f JOIN analysis_runs ar "
            "ON ar.id=f.analysis_run_id WHERE f.id=?",
            (finding_id,),
        ).fetchone()
        if finding is None:
            raise EvidenceValidationError("unknown finding")

        candidate_items = list(candidates)
        if not candidate_items:
            return []
        savepoint = f"finding_observation_candidates_{uuid.uuid4().hex}"
        self.db.execute(f"SAVEPOINT {savepoint}")
        candidate_ids: list[str] = []
        try:
            ordinal = int(
                self.db.execute(
                    "SELECT COALESCE(MAX(ordinal),-1)+1 FROM finding_observation_candidates "
                    "WHERE finding_id=?",
                    (finding_id,),
                ).fetchone()[0]
            )
            scoped_revisions = set(self.scope_revisions(str(finding["scope_id"])))
            for candidate in candidate_items:
                evidence_id = candidate.get("evidence_id")
                excerpt = candidate.get("excerpt")
                description = candidate.get("description")
                if (
                    not isinstance(evidence_id, str)
                    or not evidence_id.strip()
                    or not isinstance(excerpt, str)
                    or not excerpt.strip()
                    or not isinstance(description, str)
                    or not description.strip()
                ):
                    raise EvidenceValidationError(
                        "observation candidates require an evidence ID, excerpt, and description"
                    )
                if excerpt != excerpt.strip():
                    raise EvidenceValidationError(
                        "observation candidate excerpt must be exact and trimmed"
                    )
                evidence = self.db.execute(
                    "SELECT e.kind,e.span_id,s.revision_id FROM finding_evidence fe "
                    "JOIN evidence_artifacts e ON e.id=fe.evidence_id "
                    "LEFT JOIN source_spans s ON s.id=e.span_id "
                    "WHERE fe.finding_id=? AND fe.evidence_id=?",
                    (finding_id, evidence_id),
                ).fetchone()
                if (
                    evidence is None
                    or evidence["kind"] != EvidenceKind.TEXTUAL
                    or not evidence["span_id"]
                    or evidence["revision_id"] not in scoped_revisions
                ):
                    raise EvidenceValidationError(
                        "observation candidate must link to cited textual evidence in the finding scope"
                    )
                if excerpt not in self.span_text(str(evidence["span_id"])):
                    raise EvidenceValidationError(
                        "observation candidate excerpt does not match its cited textual evidence"
                    )
                candidate_id = uid()
                self.db.execute(
                    "INSERT INTO finding_observation_candidates"
                    "(id,finding_id,ordinal,evidence_id,excerpt,description,status,created_at) "
                    "VALUES(?,?,?,?,?,?,'UNVERIFIED',?)",
                    (
                        candidate_id,
                        finding_id,
                        ordinal,
                        evidence_id,
                        excerpt,
                        description,
                        now(),
                    ),
                )
                candidate_ids.append(candidate_id)
                ordinal += 1
            self.db.execute(f"RELEASE SAVEPOINT {savepoint}")
            return candidate_ids
        except BaseException:
            self.db.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            self.db.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise

    def finding_observation_candidates(
        self, finding_id: str, limit: int | None = None
    ) -> list[sqlite3.Row]:
        """Return persisted model descriptions without upgrading their UNVERIFIED status."""
        if not self.db.execute("SELECT 1 FROM findings WHERE id=?", (finding_id,)).fetchone():
            raise EvidenceValidationError("unknown finding")
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 1_000):
            raise EvidenceValidationError("observation candidate limit must be between 1 and 1000")
        query = (
            "SELECT id,finding_id,ordinal,evidence_id,excerpt,description,status,created_at "
            "FROM finding_observation_candidates WHERE finding_id=? ORDER BY ordinal"
        )
        params: tuple[str | int, ...] = (finding_id,)
        if limit is not None:
            query += " LIMIT ?"
            params += (limit,)
        return self.db.execute(query, params).fetchall()

    def get_finding(self, finding_id: str) -> sqlite3.Row:
        finding = self.db.execute(
            "SELECT id,analysis_run_id,skill_run_id,skill_id,skill_version,phenomenon,observation,"
            "interpretation,alternative_interpretation,claim_types_json,coverage_id,"
            "evidence_sufficiency,interpretation_confidence,limitations_json,"
            "COALESCE((SELECT status FROM finding_reviews WHERE finding_id=findings.id "
            "ORDER BY review_order DESC LIMIT 1),'GENERATED') AS review_status,"
            "created_at,finalized_at FROM findings WHERE id=?",
            (finding_id,),
        ).fetchone()
        if not finding:
            raise EvidenceValidationError("unknown finding")
        return finding

    def link_finding_source(self, finding_id: str, source_finding_id: str, relation: str) -> None:
        """Persist explicit same-run lineage for a derived/synthesized finding."""
        if finding_id == source_finding_id or relation != "SYNTHESIZES":
            raise EvidenceValidationError("finding lineage relation must be SYNTHESIZES")
        self._require_active_finding_run(finding_id)
        target = self.db.execute(
            "SELECT analysis_run_id FROM findings WHERE id=?", (finding_id,)
        ).fetchone()
        source = self.db.execute(
            "SELECT analysis_run_id FROM findings WHERE id=?", (source_finding_id,)
        ).fetchone()
        if not target or not source:
            raise EvidenceValidationError("finding lineage references an unknown finding")
        if target["analysis_run_id"] != source["analysis_run_id"]:
            raise ScopeViolationError("derived findings must retain their source analysis run")
        self.db.execute(
            "INSERT INTO finding_lineage VALUES(?,?,?)",
            (finding_id, source_finding_id, relation),
        )
        self.db.commit()

    def finding_sources(self, finding_id: str) -> list[sqlite3.Row]:
        if not self.db.execute("SELECT 1 FROM findings WHERE id=?", (finding_id,)).fetchone():
            raise EvidenceValidationError("unknown finding")
        return self.db.execute(
            "SELECT source_finding_id,relation FROM finding_lineage "
            "WHERE finding_id=? ORDER BY source_finding_id,relation",
            (finding_id,),
        ).fetchall()
