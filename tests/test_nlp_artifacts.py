import os
import re
import shutil
import sqlite3
import unicodedata
from hashlib import sha256
from pathlib import Path

import pytest

from discourselens import Project, nlp_artifacts
from discourselens.artifact_integrity import verify_project_artifact
from discourselens.corpus import (
    SURFACE_NORMALIZATION_V2,
    SURFACE_TOKENIZER_V2,
    _surface_form_v2,
    _tokenize_v2,
)
from discourselens.errors import ArtifactIntegrityError, ScopeViolationError
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import (
    LEGACY_PARSED_TOKEN_SCHEMA_VERSION,
    PARSED_TOKEN_SCHEMA_VERSION,
    SURFACE_TOKEN_SCHEMA_VERSION,
    _iter_surface_matches,
    artifact_frequency_statistic_evidence,
    build_parsed_token_artifact,
    build_surface_token_artifact,
    dependency_patterns,
    dependency_relation_frequency,
    document_frequency,
    lemma_dispersion,
    lemma_frequency,
    normalized_frequency,
    parsed_frequency_page,
    pos_frequency,
)


class _ParsedBackend:
    language = "en"
    backend_id = "fixture"
    model_id = "fixture-v1"

    def parse(self, text: str) -> list[ParsedToken]:
        return [
            ParsedToken("Café", "café", "NOUN", None, None, "ROOT", 0, 4),
            ParsedToken("migración", "migración", "NOUN", None, 0, "obj", 5, 14),
        ]


class _VersionedParsedBackend(_ParsedBackend):
    backend_version = "fixture-backend-1"
    model_version = "fixture-model-1"


class _UpdatedParsedModelBackend(_VersionedParsedBackend):
    model_version = "fixture-model-2"


class _CommitFailingConnection:
    def __init__(self, connection):
        self.connection = connection

    def commit(self):
        raise RuntimeError("injected commit failure")

    def __getattr__(self, name):
        return getattr(self.connection, name)


def _build_artifact(project, revision, artifact_kind, backend=None):
    if artifact_kind == "surface":
        return build_surface_token_artifact(project, revision)
    return build_parsed_token_artifact(project, revision, backend or _ParsedBackend())


def _assert_artifact_build_left_no_files_or_registry_rows(project, revision, artifact_kind):
    artifact_type = f"{artifact_kind}_tokens"
    assert list((project.hidden / "artifacts" / "nlp").glob("*.parquet")) == []
    assert list((project.hidden / "artifacts" / "nlp").glob("*.partial")) == []
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE revision_id=? AND artifact_type=?",
            (revision, artifact_type),
        ).fetchone()[0]
        == 0
    )


def test_surface_token_artifact_is_derived_and_scope_checked(tmp_path):
    project = Project.init(tmp_path / "p")
    _, selected = project.import_text("a.txt", "Café migration. Café.")
    _, excluded = project.import_text("b.txt", "Elephant migration.")
    selected_artifact = build_surface_token_artifact(project, selected)
    excluded_artifact = build_surface_token_artifact(project, excluded)

    assert (project.hidden / selected_artifact.relative_path).is_file()
    registry = project.db.execute(
        "SELECT file_hash FROM artifact_registry WHERE id=?", (selected_artifact.artifact_id,)
    ).fetchone()
    assert (
        registry["file_hash"]
        == sha256((project.hidden / selected_artifact.relative_path).read_bytes()).hexdigest()
    )
    assert selected_artifact.content_hash == registry["file_hash"]
    counts = normalized_frequency(
        project, project.snapshot([selected]), [selected_artifact.artifact_id]
    )
    assert counts == {"café": 2, "migration": 1}

    with pytest.raises(ScopeViolationError):
        normalized_frequency(project, project.snapshot([selected]), [excluded_artifact.artifact_id])

    (project.hidden / selected_artifact.relative_path).write_bytes(b"corrupted artifact")
    with pytest.raises(ArtifactIntegrityError, match="integrity check"):
        normalized_frequency(project, project.snapshot([selected]), [selected_artifact.artifact_id])

    project.db.execute(
        "UPDATE artifact_registry SET relative_path=? WHERE id=?",
        ("../../../../etc/passwd", selected_artifact.artifact_id),
    )
    project.db.commit()
    with pytest.raises(ArtifactIntegrityError, match="path is invalid"):
        normalized_frequency(project, project.snapshot([selected]), [selected_artifact.artifact_id])


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 19])
def test_streaming_surface_tokenizer_matches_shared_v2_contract_across_chunk_boundaries(
    tmp_path, chunk_size
):
    text = "Café cafe\u0301 migración—don't re-enter 007 one_two O’Neil\r\núltimo"
    path = tmp_path / "canonical.txt"
    path.write_bytes(text.encode("utf-8"))

    streamed = list(_iter_surface_matches(path, chunk_size))
    with path.open("r", encoding="utf-8", newline="") as source:
        canonical = source.read()
    expected = [
        (token.start_offset, token.end_offset, token.surface) for token in _tokenize_v2(canonical)
    ]

    assert streamed == expected


@pytest.mark.parametrize("artifact_kind", ["surface", "parsed"])
def test_nlp_artifacts_preserve_canonical_crlf_offsets_and_hash(tmp_path, artifact_kind):
    pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    project = Project.init(tmp_path / f"crlf-{artifact_kind}")
    canonical = "Café migración\r\nCafé migración"
    _, revision = project.import_text("crlf.txt", canonical)

    artifact = (
        build_surface_token_artifact(project, revision)
        if artifact_kind == "surface"
        else build_parsed_token_artifact(project, revision, _ParsedBackend())
    )
    rows = pq.read_table(project.hidden / artifact.relative_path).to_pylist()

    assert [row["surface"] for row in rows] == ["Café", "migración"] * 2
    assert all(canonical[row["char_start"] : row["char_end"]] == row["surface"] for row in rows)
    assert [row["char_start"] for row in rows] == [0, 5, 16, 21]


def test_surface_token_artifact_writes_bounded_parquet_row_groups(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    monkeypatch.setattr(nlp_artifacts, "_SURFACE_TOKEN_BATCH_SIZE", 2)
    monkeypatch.setattr(nlp_artifacts, "_SURFACE_TEXT_CHUNK_CHARS", 7)
    project = Project.init(tmp_path / "streamed-parquet")
    canonical = "Café cafe\u0301 migración.\n\nDon't re-enter política."
    _, revision = project.import_text("unicode.txt", canonical)

    artifact = build_surface_token_artifact(project, revision)
    path = project.hidden / artifact.relative_path
    table = pq.read_table(path)
    metadata = pq.read_metadata(path)
    rows = table.to_pylist()
    expected_matches = [
        (token.start_offset, token.end_offset, token.surface) for token in _tokenize_v2(canonical)
    ]
    paragraphs = project.db.execute(
        "SELECT id,start_offset,end_offset FROM paragraphs WHERE revision_id=? ORDER BY start_offset",
        (revision,),
    ).fetchall()

    assert metadata.num_row_groups >= 3
    assert [
        (row["char_start"], row["char_end"], row["surface"]) for row in rows
    ] == expected_matches
    assert [row["token_index"] for row in rows] == list(range(len(expected_matches)))
    assert [row["normalized"] for row in rows] == [
        _surface_form_v2(token) for _, _, token in expected_matches
    ]
    assert all(row["revision_id"] == revision for row in rows)
    assert all(
        row["paragraph_id"]
        == next(
            (
                paragraph["id"]
                for paragraph in paragraphs
                if paragraph["start_offset"] <= start < paragraph["end_offset"]
            ),
            None,
        )
        for row, (start, _, _) in zip(rows, expected_matches, strict=True)
    )


def test_surface_token_artifact_supports_nonempty_revisions_with_no_tokens(tmp_path):
    pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    project = Project.init(tmp_path / "empty-token-artifact")
    _, revision = project.import_text("punctuation.txt", "— …\n\n!!!")

    artifact = build_surface_token_artifact(project, revision)

    table = pq.read_table(project.hidden / artifact.relative_path)
    assert table.num_rows == 0
    assert table.column_names == [
        "revision_id",
        "paragraph_id",
        "token_index",
        "surface",
        "normalized",
        "char_start",
        "char_end",
    ]


@pytest.mark.parametrize("artifact_kind", ["surface", "parsed"])
@pytest.mark.parametrize("failure_point", ["temporary_write", "atomic_publish"])
def test_nlp_artifact_builder_cleans_temporary_file_on_publication_failure(
    tmp_path, monkeypatch, artifact_kind, failure_point
):
    pytest.importorskip("pyarrow")
    project = Project.init(tmp_path / f"publication-{artifact_kind}-{failure_point}")
    _, revision = project.import_text("source.txt", "Café migración.")
    backend = None

    if failure_point == "temporary_write":
        if artifact_kind == "surface":
            def fail_token_iteration(*args, **kwargs):
                raise RuntimeError("injected temporary write failure")

            monkeypatch.setattr(nlp_artifacts, "_iter_surface_matches", fail_token_iteration)
        else:
            class FailingBackend(_ParsedBackend):
                def parse(self, text):
                    raise RuntimeError("injected temporary write failure")

            backend = FailingBackend()
    else:
        backend = None
        original_link = os.link

        def fail_partial_publish(source, target, *args, **kwargs):
            if Path(source).name.endswith(".partial"):
                raise OSError("injected atomic publication failure")
            return original_link(source, target, *args, **kwargs)

        monkeypatch.setattr(nlp_artifacts.os, "link", fail_partial_publish)

    expected_message = (
        "temporary write failure"
        if failure_point == "temporary_write"
        else "atomic publication failure"
    )
    expected_exception = RuntimeError if failure_point == "temporary_write" else OSError
    with pytest.raises(expected_exception, match=expected_message):
        _build_artifact(project, revision, artifact_kind, backend)
    _assert_artifact_build_left_no_files_or_registry_rows(project, revision, artifact_kind)


@pytest.mark.parametrize("artifact_kind", ["surface", "parsed"])
def test_nlp_artifact_builder_does_not_overwrite_destination_created_during_build(
    tmp_path, monkeypatch, artifact_kind
):
    pytest.importorskip("pyarrow")
    project = Project.init(tmp_path / f"late-destination-{artifact_kind}")
    _, revision = project.import_text("source.txt", "Café migración.")
    artifact_id = "00000000-0000-4000-8000-000000000100"
    monkeypatch.setattr(project, "uid", lambda: artifact_id)
    artifact_directory = project.hidden / "artifacts" / "nlp"
    destination = artifact_directory / f"{artifact_id}.parquet"
    sentinel = b"historical artifact at the claimed destination"

    if artifact_kind == "surface":
        original_iterator = nlp_artifacts._iter_surface_matches

        def create_destination_after_tokenization(path, chunk_chars, *, on_text_chunk=None):
            yield from original_iterator(
                path, chunk_chars, on_text_chunk=on_text_chunk
            )
            destination.write_bytes(sentinel)

        monkeypatch.setattr(
            nlp_artifacts, "_iter_surface_matches", create_destination_after_tokenization
        )
        build = lambda: build_surface_token_artifact(project, revision)
    else:
        class CollidingBackend(_ParsedBackend):
            def parse(self, text):
                destination.write_bytes(sentinel)
                return super().parse(text)

        build = lambda: build_parsed_token_artifact(project, revision, CollidingBackend())

    with pytest.raises(ArtifactIntegrityError, match="identity already exists"):
        build()

    assert destination.read_bytes() == sentinel
    assert not list(artifact_directory.glob("*.partial"))
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE id=?", (artifact_id,)
        ).fetchone()[0]
        == 0
    )


@pytest.mark.parametrize("artifact_kind", ["surface", "parsed"])
@pytest.mark.parametrize("failure_point", ["hash", "registry_insert", "commit"])
def test_nlp_artifact_builder_cleans_published_file_on_registration_failure(
    tmp_path, monkeypatch, artifact_kind, failure_point
):
    pytest.importorskip("pyarrow")
    project = Project.init(tmp_path / f"registration-{artifact_kind}-{failure_point}")
    _, revision = project.import_text("source.txt", "Café migración.")

    if failure_point == "hash":
        def fail_hash(path, *, directory_fd=None):
            del path, directory_fd
            raise RuntimeError("injected artifact hash failure")

        monkeypatch.setattr(nlp_artifacts, "_sha256_artifact_entry", fail_hash)
        expected_message = "artifact hash failure"
        expected_exception = RuntimeError
    elif failure_point == "registry_insert":
        artifact_type = f"{artifact_kind}_tokens"
        project.db.execute(
            "CREATE TRIGGER reject_nlp_artifact BEFORE INSERT ON artifact_registry "
            f"WHEN NEW.artifact_type='{artifact_type}' BEGIN "
            "SELECT RAISE(ABORT, 'injected registry failure'); END"
        )
        project.db.commit()
        expected_message = "injected registry failure"
        expected_exception = sqlite3.IntegrityError
    else:
        monkeypatch.setattr(project, "db", _CommitFailingConnection(project.db))
        expected_message = "injected commit failure"
        expected_exception = RuntimeError

    with pytest.raises(expected_exception, match=expected_message):
        _build_artifact(project, revision, artifact_kind)
    _assert_artifact_build_left_no_files_or_registry_rows(project, revision, artifact_kind)


@pytest.mark.parametrize("artifact_kind", ["surface", "parsed"])
def test_registration_failure_does_not_delete_replacement_artifact(
    tmp_path, monkeypatch, artifact_kind
):
    pytest.importorskip("pyarrow")
    project = Project.init(tmp_path / f"registration-replacement-{artifact_kind}")
    _, revision = project.import_text("source.txt", "Café migración.")
    artifact_directory = project.hidden / "artifacts" / "nlp"
    sentinel = b"replacement owned by another writer"
    original_hash = nlp_artifacts._sha256_artifact_entry

    project.db.execute(
        "CREATE TRIGGER reject_nlp_artifact BEFORE INSERT ON artifact_registry "
        f"WHEN NEW.artifact_type='{artifact_kind}_tokens' BEGIN "
        "SELECT RAISE(ABORT, 'injected registry failure'); END"
    )
    project.db.commit()

    def replace_destination_after_hash(path, *, directory_fd=None):
        digest = original_hash(path, directory_fd=directory_fd)
        replacement = artifact_directory / "concurrent-writer.tmp"
        replacement.write_bytes(sentinel)
        replacement.replace(path)
        return digest

    monkeypatch.setattr(nlp_artifacts, "_sha256_artifact_entry", replace_destination_after_hash)

    with pytest.raises(sqlite3.IntegrityError, match="injected registry failure"):
        _build_artifact(project, revision, artifact_kind)

    replacement_path = next(artifact_directory.glob("*.parquet"))
    assert replacement_path.read_bytes() == sentinel
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE revision_id=? AND artifact_type=?",
            (revision, f"{artifact_kind}_tokens"),
        ).fetchone()[0]
        == 0
    )


@pytest.mark.parametrize("artifact_kind", ["surface", "parsed"])
def test_duplicate_artifact_identity_does_not_destroy_registered_file(
    tmp_path, monkeypatch, artifact_kind
):
    pytest.importorskip("pyarrow")
    project = Project.init(tmp_path / f"duplicate-identity-{artifact_kind}")
    _, revision = project.import_text("source.txt", "Café migración.")
    artifact = _build_artifact(project, revision, artifact_kind)
    artifact_path = project.hidden / artifact.relative_path
    original_hash = artifact.file_hash

    monkeypatch.setattr(project, "uid", lambda: artifact.artifact_id)
    with pytest.raises(ArtifactIntegrityError, match="identity already exists"):
        _build_artifact(project, revision, artifact_kind)

    assert artifact_path.is_file()
    assert verify_project_artifact(project.root, artifact.relative_path, original_hash) == artifact_path
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE id=?", (artifact.artifact_id,)
        ).fetchone()[0]
        == 1
    )


@pytest.mark.parametrize("artifact_kind", ["surface", "parsed"])
def test_nlp_artifact_builder_never_follows_preexisting_temporary_symlink(
    tmp_path, monkeypatch, artifact_kind
):
    pytest.importorskip("pyarrow")
    project = Project.init(tmp_path / f"symlinked-temporary-{artifact_kind}")
    _, revision = project.import_text("source.txt", "Café migration.")
    artifact_id = "00000000-0000-4000-8000-000000000099"
    monkeypatch.setattr(project, "uid", lambda: artifact_id)
    artifact_directory = project.hidden / "artifacts" / "nlp"
    temporary_path = artifact_directory / f".{artifact_id}.partial"
    outside = tmp_path / f"outside-{artifact_kind}.bin"
    sentinel = b"outside file must remain unchanged"
    outside.write_bytes(sentinel)
    try:
        temporary_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ArtifactIntegrityError, match="temporary path already exists"):
        _build_artifact(project, revision, artifact_kind)

    assert outside.read_bytes() == sentinel
    assert temporary_path.is_symlink()
    assert not (artifact_directory / f"{artifact_id}.parquet").exists()
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE revision_id=?", (revision,)
        ).fetchone()[0]
        == 0
    )


def test_surface_token_artifact_rejects_canonical_text_path_escape(tmp_path):
    project = Project.init(tmp_path / "unsafe-canonical-path")
    _, revision = project.import_text("valid.txt", "Café.")
    (tmp_path / "outside.txt").write_text("not project content", encoding="utf-8")
    project.db.execute(
        "UPDATE document_revisions SET text_path=? WHERE id=?", ("../../outside.txt", revision)
    )
    project.db.commit()

    with pytest.raises(ArtifactIntegrityError, match="escapes project storage"):
        build_surface_token_artifact(project, revision)
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE revision_id=? AND artifact_type='surface_tokens'",
            (revision,),
        ).fetchone()[0]
        == 0
    )


def test_surface_token_artifact_rejects_tampered_canonical_text(tmp_path):
    project = Project.init(tmp_path / "tampered-canonical-text")
    _, revision = project.import_text("source.txt", "Café migration.")
    text_path = project.db.execute(
        "SELECT text_path FROM document_revisions WHERE id=?", (revision,)
    ).fetchone()[0]
    (project.hidden / text_path).write_text("Altered migration.", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="revision hash"):
        build_surface_token_artifact(project, revision)
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE revision_id=? AND artifact_type='surface_tokens'",
            (revision,),
        ).fetchone()[0]
        == 0
    )
    assert not list((project.hidden / "artifacts" / "nlp").glob("*.partial"))


def test_derived_artifact_symlink_cannot_escape_project(tmp_path):
    project = Project.init(tmp_path / "project")
    artifact_dir = project.hidden / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not project data")
    link = artifact_dir / "escape.bin"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ArtifactIntegrityError, match="path is invalid"):
        verify_project_artifact(project.root, "artifacts/escape.bin", None)


@pytest.mark.parametrize("artifact_kind", ["surface", "parsed"])
def test_nlp_artifact_builder_rejects_directory_symlink_after_project_open(tmp_path, artifact_kind):
    pytest.importorskip("pyarrow")
    project = Project.init(tmp_path / f"symlinked-{artifact_kind}-artifacts")
    _, revision = project.import_text("source.txt", "Café migration.")
    outside = tmp_path / f"outside-{artifact_kind}"
    outside.mkdir()
    nlp_directory = project.hidden / "artifacts" / "nlp"
    nlp_directory.rmdir()
    try:
        nlp_directory.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ArtifactIntegrityError, match="artifact directory is invalid"):
        if artifact_kind == "surface":
            build_surface_token_artifact(project, revision)
        else:
            build_parsed_token_artifact(project, revision, _ParsedBackend())

    assert list(outside.iterdir()) == []
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE revision_id=?", (revision,)
        ).fetchone()[0]
        == 0
    )


def test_nlp_artifact_builder_does_not_follow_directory_swapped_after_preflight(
    tmp_path, monkeypatch
):
    pytest.importorskip("pyarrow")
    project = Project.init(tmp_path / "swapped-nlp-directory")
    _, revision = project.import_text("source.txt", "Café migration.")
    artifact_id = "00000000-0000-4000-8000-000000000101"
    monkeypatch.setattr(project, "uid", lambda: artifact_id)
    artifact_directory = project.hidden / "artifacts" / "nlp"
    displaced_directory = project.hidden / "artifacts" / "nlp-original"
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    original_open_temporary = nlp_artifacts._open_temporary_artifact
    swapped = False

    def swap_directory_before_temporary_open(path_arg, *, directory_fd=None):
        nonlocal swapped
        if not swapped:
            artifact_directory.rename(displaced_directory)
            try:
                artifact_directory.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                pytest.skip(f"directory symlinks unavailable: {exc}")
            swapped = True
        return original_open_temporary(path_arg, directory_fd=directory_fd)

    monkeypatch.setattr(
        nlp_artifacts, "_open_temporary_artifact", swap_directory_before_temporary_open
    )

    with pytest.raises(ArtifactIntegrityError):
        build_surface_token_artifact(project, revision)

    assert swapped
    assert list(outside.iterdir()) == []
    assert list(displaced_directory.iterdir()) == []
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE id=?", (artifact_id,)
        ).fetchone()[0]
        == 0
    )


def test_derived_artifact_rejects_project_storage_root_symlink(tmp_path):
    root = tmp_path / "project"
    project = Project.init(root)
    project.close()

    external_storage = tmp_path / "external-storage"
    hidden = root / ".discourselens"
    hidden.replace(external_storage)
    try:
        hidden.symlink_to(external_storage, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    (external_storage / "artifacts" / "outside.bin").write_bytes(b"outside project storage")

    with pytest.raises(ArtifactIntegrityError, match="path is invalid"):
        verify_project_artifact(root, "artifacts/outside.bin", None)


@pytest.mark.parametrize("relative_path", ["C:\\outside.bin", "..\\outside.bin", "/etc/passwd"])
def test_derived_artifact_rejects_absolute_and_windows_paths(tmp_path, relative_path):
    project = Project.init(tmp_path / "project")
    with pytest.raises(ArtifactIntegrityError, match="path is invalid"):
        verify_project_artifact(project.root, relative_path, None)


def test_hashed_nlp_artifact_remains_valid_after_project_copy(tmp_path):
    project = Project.init(tmp_path / "source")
    _, revision = project.import_text("a.txt", "Café migration.")
    artifact = build_surface_token_artifact(project, revision)
    project.close()

    shutil.copytree(tmp_path / "source", tmp_path / "copied")
    copied = Project.open(tmp_path / "copied")
    try:
        counts = normalized_frequency(copied, copied.snapshot([revision]), [artifact.artifact_id])
        assert counts == {"café": 1, "migration": 1}
        row = copied.db.execute(
            "SELECT file_hash FROM artifact_registry WHERE id=?", (artifact.artifact_id,)
        ).fetchone()
        assert row["file_hash"] == artifact.file_hash
    finally:
        copied.close()


def test_same_size_artifact_tampering_with_restored_mtime_invalidates_hash_cache(tmp_path):
    project = Project.init(tmp_path / "tamper-cache")
    _, revision = project.import_text("a.txt", "Café migration.")
    artifact = build_surface_token_artifact(project, revision)
    path = project.hidden / artifact.relative_path
    assert verify_project_artifact(project.root, artifact.relative_path, artifact.file_hash) == path
    original_stat = path.stat()
    content = bytearray(path.read_bytes())
    content[0] = (content[0] + 1) % 256
    path.write_bytes(content)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    changed_stat = path.stat()
    assert changed_stat.st_size == original_stat.st_size
    assert changed_stat.st_mtime_ns == original_stat.st_mtime_ns
    with pytest.raises(ArtifactIntegrityError, match="integrity check"):
        verify_project_artifact(project.root, artifact.relative_path, artifact.file_hash)


def test_document_frequency_uses_derived_artifacts_without_scope_expansion(tmp_path):
    project = Project.init(tmp_path / "p")
    _, first = project.import_text("a.txt", "Migration is discussed.")
    _, second = project.import_text("b.txt", "Migration and migration.")
    first_artifact = build_surface_token_artifact(project, first)
    second_artifact = build_surface_token_artifact(project, second)

    counts = document_frequency(
        project,
        project.snapshot([first, second]),
        [first_artifact.artifact_id, second_artifact.artifact_id],
    )
    assert counts["migration"] == 2
    assert counts["and"] == 1
    with pytest.raises(ScopeViolationError):
        document_frequency(project, project.snapshot([first]), [second_artifact.artifact_id])


@pytest.mark.parametrize("operation", ["normalized_frequency", "document_frequency", "statistic"])
@pytest.mark.parametrize("schema_version", ["surface-token-v1", "surface-token-v999"])
def test_surface_artifact_analytics_reject_unknown_schema_versions(
    tmp_path, operation, schema_version
):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / operation)
    _, revision = project.import_text("source.txt", "alpha alpha")
    artifact = build_surface_token_artifact(project, revision)
    scope = project.snapshot([revision])
    project.db.execute(
        "UPDATE artifact_registry SET schema_version=? WHERE id=?",
        (schema_version, artifact.artifact_id),
    )
    project.db.commit()

    with pytest.raises(ScopeViolationError, match="surface token artifacts require"):
        if operation == "normalized_frequency":
            normalized_frequency(project, scope, [artifact.artifact_id])
        elif operation == "document_frequency":
            document_frequency(project, scope, [artifact.artifact_id])
        else:
            artifact_frequency_statistic_evidence(
                project,
                scope,
                [artifact.artifact_id],
                field="normalized",
                value="alpha",
            )


@pytest.mark.parametrize("artifact_kind", ["surface", "parsed"])
def test_token_analytics_reject_multiple_artifact_versions_for_one_revision(
    tmp_path, artifact_kind
):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / f"duplicate-{artifact_kind}-artifacts")
    _, revision = project.import_text("source.txt", "Café migración")
    first = _build_artifact(project, revision, artifact_kind)
    second = _build_artifact(project, revision, artifact_kind)

    with pytest.raises(ScopeViolationError, match="one token artifact per document revision"):
        if artifact_kind == "surface":
            normalized_frequency(
                project,
                project.snapshot([revision]),
                [first.artifact_id, second.artifact_id],
            )
        else:
            lemma_frequency(
                project,
                project.snapshot([revision]),
                [first.artifact_id, second.artifact_id],
            )


def test_dependency_patterns_join_heads_inside_complete_explicit_scope(tmp_path):
    project = Project.init(tmp_path / "p")
    _, first = project.import_text("a.txt", "Café migración")
    _, second = project.import_text("b.txt", "Café migración")
    _, outside = project.import_text("outside.txt", "Café migración")
    first_artifact = build_parsed_token_artifact(project, first, _ParsedBackend())
    second_artifact = build_parsed_token_artifact(project, second, _ParsedBackend())
    outside_artifact = build_parsed_token_artifact(project, outside, _ParsedBackend())
    scope = project.snapshot([first, second])

    result = dependency_patterns(
        project,
        scope,
        [first_artifact.artifact_id, second_artifact.artifact_id],
        relation="obj",
        head_lemma="CAFÉ",
    )

    assert result["scope_id"] == scope
    assert result["calculation_version"] == "parsed-dependency-patterns-v2"
    assert result["candidate_count_lower_bound"] == 1
    assert result["truncated"] is False
    assert result["rows"] == [
        {
            "head_lemma": "café",
            "relation": "obj",
            "dependent_lemma": "migración",
            "frequency": 2,
            "document_frequency": 2,
        }
    ]
    with pytest.raises(ScopeViolationError, match="outside the explicit scope"):
        dependency_patterns(
            project, scope, [first_artifact.artifact_id, outside_artifact.artifact_id]
        )


def test_dependency_patterns_reject_partial_scope_artifact_population(tmp_path):
    project = Project.init(tmp_path / "p")
    _, first = project.import_text("a.txt", "Café migración")
    _, second = project.import_text("b.txt", "Café migración")
    artifact = build_parsed_token_artifact(project, first, _ParsedBackend())

    with pytest.raises(ScopeViolationError, match="one compatible artifact"):
        dependency_patterns(project, project.snapshot([first, second]), [artifact.artifact_id])


def test_dependency_patterns_bound_output_and_reject_invalid_limit(tmp_path):
    class MultiArcBackend:
        language = "en"
        backend_id = "fixture"
        model_id = "fixture-v1"

        def parse(self, text: str) -> list[ParsedToken]:
            return [
                ParsedToken("They", "they", "PRON", None, 1, "nsubj", 0, 4),
                ParsedToken("support", "support", "VERB", None, 1, "ROOT", 5, 12),
                ParsedToken("policy", "policy", "NOUN", None, 1, "obj", 13, 19),
            ]

    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "They support policy")
    artifact = build_parsed_token_artifact(project, revision, MultiArcBackend())
    scope = project.snapshot([revision])

    result = dependency_patterns(project, scope, [artifact.artifact_id], limit=1)

    assert len(result["rows"]) == 1
    assert result["truncated"] is True
    assert result["candidate_count_lower_bound"] == 2
    with pytest.raises(ScopeViolationError, match="limit"):
        dependency_patterns(project, scope, [artifact.artifact_id], limit=0)


def test_parsed_artifact_records_available_backend_and_model_versions(tmp_path):
    class VersionedBackend(_ParsedBackend):
        backend_version = "4.5.6"
        model_version = "7.8.9"

    class UpdatedModelBackend(VersionedBackend):
        model_version = "7.9.0"

    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Café migración")
    first = build_parsed_token_artifact(project, revision, VersionedBackend())
    second = build_parsed_token_artifact(project, revision, UpdatedModelBackend())
    first_row = project.db.execute(
        "SELECT backend_version,model_version FROM artifact_registry WHERE id=?",
        (first.artifact_id,),
    ).fetchone()

    assert first_row["backend_version"] == "4.5.6"
    assert first_row["model_version"] == "7.8.9"
    assert first.derivation_hash != second.derivation_hash


def test_parsed_artifact_derivation_identity_includes_backend_language(tmp_path):
    class SpanishBackend(_VersionedParsedBackend):
        language = "es"

    project = Project.init(tmp_path / "language-derivation")
    _, revision = project.import_text("source.txt", "Café migración")

    english = build_parsed_token_artifact(project, revision, _VersionedParsedBackend())
    spanish = build_parsed_token_artifact(project, revision, SpanishBackend())

    assert english.derivation_hash != spanish.derivation_hash


def test_parsed_nlp_artifact_is_versioned_and_keeps_canonical_offsets(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Café migración")
    artifact = build_parsed_token_artifact(project, revision, _ParsedBackend())
    row = project.db.execute(
        "SELECT artifact_type,backend,model,schema_version FROM artifact_registry WHERE id=?",
        (artifact.artifact_id,),
    ).fetchone()
    assert tuple(row) == (
        "parsed_tokens",
        "fixture",
        "fixture-v1",
        PARSED_TOKEN_SCHEMA_VERSION,
    )
    scope = project.snapshot([revision])
    assert lemma_frequency(project, scope, [artifact.artifact_id]) == {"café": 1, "migración": 1}
    assert pos_frequency(project, scope, [artifact.artifact_id]) == {"NOUN": 2}
    assert dependency_relation_frequency(project, scope, [artifact.artifact_id]) == {
        "ROOT": 1,
        "obj": 1,
    }
    assert lemma_dispersion(project, scope, [artifact.artifact_id]) == {"café": 1, "migración": 1}


def test_missing_parser_annotations_are_not_counted_as_empty_categories(tmp_path):
    class IncompleteBackend:
        language = "en"
        backend_id = "fixture"
        model_id = "incomplete-v1"

        def parse(self, text: str) -> list[ParsedToken]:
            return [
                ParsedToken("They", "", "", None, None, "ROOT", 0, 4),
                ParsedToken("speak", "speak", "VERB", None, 0, "dep", 5, 10),
            ]

    project = Project.init(tmp_path / "missing-parser-annotations")
    _, revision = project.import_text("a.txt", "They speak")
    artifact = build_parsed_token_artifact(project, revision, IncompleteBackend())
    scope = project.snapshot([revision])

    assert lemma_frequency(project, scope, [artifact.artifact_id]) == {"speak": 1}
    assert pos_frequency(project, scope, [artifact.artifact_id]) == {"VERB": 1}
    assert lemma_dispersion(project, scope, [artifact.artifact_id]) == {"speak": 1}
    assert dependency_patterns(project, scope, [artifact.artifact_id])["rows"] == []
    statistic = artifact_frequency_statistic_evidence(
        project,
        scope,
        [artifact.artifact_id],
        field="lemma",
        value="speak",
    )
    assert statistic["count"] == 1
    assert statistic["population_count"] == 1


def test_partial_parser_annotations_are_exposed_in_field_results_and_evidence(tmp_path):
    class PartialAnnotationsBackend:
        language = "en"
        backend_id = "fixture"
        model_id = "partial-fields-v1"

        def parse(self, text: str) -> list[ParsedToken]:
            return [
                ParsedToken("They", "", "", None, None, "", 0, 4),
                ParsedToken("speak", "speak", "VERB", None, 0, "dep", 5, 10),
                ParsedToken(".", ".", "PUNCT", None, 1, "punct", 10, 11),
            ]

    project = Project.init(tmp_path / "partial-parser-annotations")
    _, revision = project.import_text("a.txt", "They speak.")
    artifact = build_parsed_token_artifact(project, revision, PartialAnnotationsBackend())
    scope = project.snapshot([revision])

    lemma = parsed_frequency_page(project, scope, [artifact.artifact_id], field="lemma")
    pos = parsed_frequency_page(project, scope, [artifact.artifact_id], field="pos")
    dependencies = parsed_frequency_page(
        project, scope, [artifact.artifact_id], field="dependency_relation"
    )
    patterns = dependency_patterns(project, scope, [artifact.artifact_id])
    statistic = artifact_frequency_statistic_evidence(
        project, scope, [artifact.artifact_id], field="lemma", value="speak"
    )
    evidence = project.evidence_content(str(statistic["evidence_id"]))
    provenance = evidence.get("provenance")
    assert isinstance(provenance, dict)
    evidence_coverage = provenance.get("field_coverage")
    assert isinstance(evidence_coverage, dict)
    descriptor = project.coverage_descriptor(str(statistic["coverage_id"]))

    assert lemma["rows"] == [{"value": ".", "count": 1}, {"value": "speak", "count": 1}]
    assert lemma["field_coverage"]["status"] == "LIMITED"
    assert lemma["field_coverage"]["annotated_tokens"] == 2
    assert lemma["field_coverage"]["total_tokens"] == 3
    assert pos["field_coverage"]["status"] == "LIMITED"
    assert dependencies["field_coverage"]["status"] == "LIMITED"
    assert lemma["coverage"]["tokenization"]["status"] == "SUPPORTED"
    assert lemma["coverage"]["tokenization"]["covered_non_whitespace_characters"] == 10
    assert lemma["coverage"]["tokenization"]["source_non_whitespace_characters"] == 10
    assert artifact.coverage is not None
    assert artifact.coverage["pos"]["status"] == "LIMITED"
    assert artifact.__dict__["coverage"]["dependency_parse"]["status"] == "LIMITED"
    assert patterns["coverage"]["patterns"]["status"] == "LIMITED"
    assert statistic["count"] == 1
    statistic_coverage = statistic.get("field_coverage")
    assert isinstance(statistic_coverage, dict)
    assert statistic_coverage["status"] == "LIMITED"
    assert evidence_coverage["status"] == "LIMITED"
    assert descriptor.coverage_mode.value == "BROAD"
    assert descriptor.negative_result_status.value == "NOT_APPLICABLE"


def test_dropped_non_whitespace_material_marks_all_parsed_fields_limited(tmp_path):
    class TokenDroppingBackend:
        language = "en"
        backend_id = "fixture"
        model_id = "token-drop-v1"

        def parse(self, text: str) -> list[ParsedToken]:
            return [ParsedToken("Apples", "apple", "NOUN", None, None, "ROOT", 0, 6)]

    project = Project.init(tmp_path / "dropped-parser-material")
    _, revision = project.import_text("a.txt", "Apples are green.")
    artifact = build_parsed_token_artifact(project, revision, TokenDroppingBackend())
    scope = project.snapshot([revision])

    result = parsed_frequency_page(project, scope, [artifact.artifact_id], field="lemma")

    assert result["rows"] == [{"value": "apple", "count": 1}]
    assert result["coverage"]["tokenization"]["status"] == "LIMITED"
    assert result["coverage"]["tokenization"]["covered_non_whitespace_characters"] == 6
    assert result["coverage"]["tokenization"]["source_non_whitespace_characters"] == 15
    assert result["field_coverage"]["status"] == "LIMITED"
    assert result["field_coverage"]["annotated_tokens"] == 1
    assert result["field_coverage"]["total_tokens"] == 1
    assert artifact.coverage is not None
    assert artifact.coverage["tokenization"]["status"] == "LIMITED"


def test_legacy_parsed_schema_remains_readable_but_coverage_is_unverified(tmp_path):
    project = Project.init(tmp_path / "legacy-parsed-schema")
    _, revision = project.import_text("a.txt", "Café migración")
    artifact = build_parsed_token_artifact(project, revision, _ParsedBackend())
    project.db.execute(
        "UPDATE artifact_registry SET schema_version=? WHERE id=?",
        (LEGACY_PARSED_TOKEN_SCHEMA_VERSION, artifact.artifact_id),
    )
    project.db.commit()
    scope = project.snapshot([revision])

    result = parsed_frequency_page(project, scope, [artifact.artifact_id], field="lemma")

    assert result["rows"] == [
        {"value": "café", "count": 1},
        {"value": "migración", "count": 1},
    ]
    assert result["coverage"]["tokenization"]["status"] == "LIMITED"
    assert result["field_coverage"]["status"] == "LIMITED"
    limitation = result["field_coverage"]["limitation"]
    assert isinstance(limitation, str)
    assert "predate field-level coverage" in limitation


@pytest.mark.parametrize("malformed_field", ["char_start", "dependency_head"])
def test_parsed_artifact_rejects_boolean_integer_fields(tmp_path, malformed_field):
    pytest.importorskip("pyarrow")

    class BooleanFieldBackend:
        language = "en"
        backend_id = "fixture"
        model_id = f"boolean-{malformed_field}-v1"

        def parse(self, text: str) -> list[ParsedToken]:
            tokens = [
                ParsedToken("They", "they", "PRON", None, None, "ROOT", 0, 4),
                ParsedToken("may", "may", "AUX", None, 0, "dep", 5, 8),
                ParsedToken("act", "act", "VERB", None, 1, "dep", 9, 12),
            ]
            if malformed_field == "char_start":
                token = tokens[0]
                tokens[0] = ParsedToken(
                    token.surface,
                    token.lemma,
                    token.pos,
                    token.morphology,
                    token.dependency_head,
                    token.dependency_relation,
                    False,
                    token.char_end,
                )
            else:
                token = tokens[2]
                tokens[2] = ParsedToken(
                    token.surface,
                    token.lemma,
                    token.pos,
                    token.morphology,
                    True,
                    token.dependency_relation,
                    token.char_start,
                    token.char_end,
                )
            return tokens

    project = Project.init(tmp_path / f"boolean-{malformed_field}")
    _, revision = project.import_text("source.txt", "They may act")

    with pytest.raises(ScopeViolationError, match="invalid canonical offsets|invalid dependency head"):
        build_parsed_token_artifact(project, revision, BooleanFieldBackend())

    _assert_artifact_build_left_no_files_or_registry_rows(project, revision, "parsed")


def test_parsed_artifact_parses_and_writes_bounded_paragraph_batches(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    class ParagraphBackend:
        language = "es"
        backend_id = "fixture"
        model_id = "paragraph-v1"

        def __init__(self):
            self.inputs = []

        def parse(self, text: str) -> list[ParsedToken]:
            self.inputs.append(text)
            matches = list(re.finditer(r"\S+", text))
            return [
                ParsedToken(
                    match.group(),
                    match.group().casefold(),
                    "NOUN",
                    None,
                    0,
                    "ROOT" if index == 0 else "obj",
                    match.start(),
                    match.end(),
                )
                for index, match in enumerate(matches)
            ]

    monkeypatch.setattr(nlp_artifacts, "_PARSED_TOKEN_BATCH_SIZE", 2)
    project = Project.init(tmp_path / "paragraph-streaming")
    canonical = "Café migración.\n\n“Quotes” — policy."
    _, revision = project.import_text("unicode.txt", canonical)
    backend = ParagraphBackend()

    artifact = build_parsed_token_artifact(project, revision, backend)

    assert backend.inputs == ["Café migración.", "“Quotes” — policy."]
    table = pq.read_table(project.hidden / artifact.relative_path)
    rows = table.to_pylist()
    assert pq.read_metadata(project.hidden / artifact.relative_path).num_row_groups >= 2
    assert [row["token_index"] for row in rows] == list(range(len(rows)))
    assert [row["surface"] for row in rows] == [
        "Café",
        "migración.",
        "“Quotes”",
        "—",
        "policy.",
    ]
    for row in rows:
        assert canonical[row["char_start"] : row["char_end"]] == row["surface"]
        paragraph = project.paragraph(row["paragraph_id"])
        assert paragraph["start_offset"] <= row["char_start"] < row["char_end"]
    assert [row["dependency_head"] for row in rows] == [0, 0, 2, 2, 2]


def test_parsed_artifact_rejects_tampered_canonical_text_and_cleans_partial(tmp_path):
    project = Project.init(tmp_path / "tampered-parsed-source")
    _, revision = project.import_text("source.txt", "Café migration.\n\nSecond paragraph.")
    text_path = project.db.execute(
        "SELECT text_path FROM document_revisions WHERE id=?", (revision,)
    ).fetchone()[0]
    (project.hidden / text_path).write_text(
        "Altered migration.\n\nSecond paragraph.", encoding="utf-8"
    )

    with pytest.raises(ArtifactIntegrityError, match="paragraph text does not match canonical"):
        build_parsed_token_artifact(project, revision, _ParsedBackend())

    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE revision_id=? AND artifact_type='parsed_tokens'",
            (revision,),
        ).fetchone()[0]
        == 0
    )
    assert not list((project.hidden / "artifacts" / "nlp").glob("*.partial"))


def test_parsed_artifact_rejects_unrepresented_canonical_text_instead_of_zero_counts(tmp_path):
    project = Project.init(tmp_path / "missing-paragraph-record")
    _, revision = project.import_text("source.txt", "They speak")
    project.db.execute("DELETE FROM paragraphs WHERE revision_id=?", (revision,))
    project.db.commit()

    with pytest.raises(
        ArtifactIntegrityError, match="canonical text is missing corresponding paragraph records"
    ):
        build_parsed_token_artifact(project, revision, _ParsedBackend())

    _assert_artifact_build_left_no_files_or_registry_rows(project, revision, "parsed")
    assert project.revision_text(revision) == "They speak"


def test_parsed_artifact_rejects_canonical_text_path_escape(tmp_path):
    project = Project.init(tmp_path / "unsafe-parsed-source")
    _, revision = project.import_text("source.txt", "Café migration.")
    project.db.execute(
        "UPDATE document_revisions SET text_path=? WHERE id=?",
        ("../../outside.txt", revision),
    )
    project.db.commit()

    with pytest.raises(ArtifactIntegrityError, match="escapes project storage"):
        build_parsed_token_artifact(project, revision, _ParsedBackend())

    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE revision_id=? AND artifact_type='parsed_tokens'",
            (revision,),
        ).fetchone()[0]
        == 0
    )


def test_artifact_statistic_evidence_requires_complete_scope_and_preserves_provenance(tmp_path):
    project = Project.init(tmp_path / "p")
    _, first = project.import_text("one.txt", "Café migration")
    _, second = project.import_text("two.txt", "Café elephant")
    first_artifact = build_surface_token_artifact(project, first)
    second_artifact = build_surface_token_artifact(project, second)
    scope = project.snapshot([first, second])

    with pytest.raises(ScopeViolationError, match="one compatible artifact"):
        artifact_frequency_statistic_evidence(
            project, scope, [first_artifact.artifact_id], field="normalized", value="café"
        )

    result = artifact_frequency_statistic_evidence(
        project,
        scope,
        [first_artifact.artifact_id, second_artifact.artifact_id],
        field="normalized",
        value="Café",
    )

    coverage = project.coverage_descriptor(str(result["coverage_id"]))
    evidence = project.evidence_content(str(result["evidence_id"]))
    assert result["count"] == 2
    assert coverage.coverage_mode.value == "EXHAUSTIVE"
    assert coverage.population_count == 4
    provenance = evidence["provenance"]
    assert isinstance(provenance, dict)
    assert set(provenance["artifact_ids"]) == {
        first_artifact.artifact_id,
        second_artifact.artifact_id,
    }
    assert provenance["calculation_version"] == "derived-surface-token-frequency-v2"
    assert provenance["artifact_schema_version"] == SURFACE_TOKEN_SCHEMA_VERSION
    assert provenance["tokenizer"] == SURFACE_TOKENIZER_V2
    assert provenance["normalization"] == SURFACE_NORMALIZATION_V2
    assert provenance["unicode_data_version"] == unicodedata.unidata_version
    assert "backend" not in provenance
    assert "model" not in provenance
    assert "model_version" not in provenance


@pytest.mark.parametrize(
    "operation",
    ["parsed_frequency_page", "dependency_patterns", "parsed_statistic_evidence"],
)
def test_complete_parsed_aggregates_reject_mixed_parser_identity_before_aggregation(
    tmp_path, monkeypatch, operation
):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / operation)
    _, first = project.import_text("one.txt", "Café migración")
    _, second = project.import_text("two.txt", "Café migración")
    first_artifact = build_parsed_token_artifact(project, first, _VersionedParsedBackend())
    second_artifact = build_parsed_token_artifact(
        project, second, _UpdatedParsedModelBackend()
    )
    scope = project.snapshot([first, second])
    artifact_ids = [first_artifact.artifact_id, second_artifact.artifact_id]
    evidence_count = project.db.execute(
        "SELECT COUNT(*) FROM evidence_artifacts"
    ).fetchone()[0]

    def unexpected_aggregate():
        pytest.fail("mixed parser identity must be rejected before aggregation")

    monkeypatch.setattr(nlp_artifacts, "_duckdb", unexpected_aggregate)
    with pytest.raises(ScopeViolationError, match="one compatible parser and model"):
        if operation == "parsed_frequency_page":
            parsed_frequency_page(project, scope, artifact_ids, field="lemma")
        elif operation == "dependency_patterns":
            dependency_patterns(project, scope, artifact_ids)
        else:
            artifact_frequency_statistic_evidence(
                project, scope, artifact_ids, field="lemma", value="café"
            )

    assert project.db.execute(
        "SELECT COUNT(*) FROM evidence_artifacts"
    ).fetchone()[0] == evidence_count


def test_compatible_parsed_aggregates_succeed_and_evidence_records_parser_identity(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / "compatible-parsed-aggregates")
    _, first = project.import_text("one.txt", "Café migración")
    _, second = project.import_text("two.txt", "Café migración")
    first_artifact = build_parsed_token_artifact(project, first, _VersionedParsedBackend())
    second_artifact = build_parsed_token_artifact(project, second, _VersionedParsedBackend())
    scope = project.snapshot([first, second])
    artifact_ids = [first_artifact.artifact_id, second_artifact.artifact_id]

    frequency_page = parsed_frequency_page(project, scope, artifact_ids, field="lemma")
    patterns = dependency_patterns(project, scope, artifact_ids)
    statistic = artifact_frequency_statistic_evidence(
        project, scope, artifact_ids, field="lemma", value="café"
    )

    assert frequency_page["rows"] == [
        {"value": "café", "count": 2},
        {"value": "migración", "count": 2},
    ]
    assert patterns["rows"] == [
        {
            "head_lemma": "café",
            "relation": "obj",
            "dependent_lemma": "migración",
            "frequency": 2,
            "document_frequency": 2,
        }
    ]
    assert statistic["count"] == 2
    evidence = project.evidence_content(str(statistic["evidence_id"]))
    provenance = evidence["provenance"]
    assert isinstance(provenance, dict)
    assert set(provenance["artifact_ids"]) == set(artifact_ids)
    assert provenance["backend"] == "fixture"
    assert provenance["backend_version"] == "fixture-backend-1"
    assert provenance["model"] == "fixture-v1"
    assert provenance["model_version"] == "fixture-model-1"
    assert provenance["schema_version"] == PARSED_TOKEN_SCHEMA_VERSION
