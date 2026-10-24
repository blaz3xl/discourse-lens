import os
import sqlite3
import stat
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

import discourselens.project as project_module
from discourselens import Project
from discourselens.errors import DocumentError, ProjectError


def _symlink_or_skip(link, target, *, target_is_directory=False):
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def _is_database_connect_path(value, database: Path) -> bool:
    if isinstance(value, str) and value.startswith("file:"):
        return Path(unquote(urlparse(value).path)) == database
    return Path(value) == database


def test_project_init_rejects_external_managed_storage_symlink(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(root / ".discourselens", outside, target_is_directory=True)

    with pytest.raises(ProjectError, match="managed project storage"):
        Project.init(root)

    assert not (outside / "source").exists()


def test_project_init_does_not_create_directories_through_swapped_storage_symlink(
    tmp_path, monkeypatch
):
    if (
        os.open not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
    ):
        pytest.skip("directory-descriptor creation is unavailable")
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    hidden = root / ".discourselens"
    hidden.mkdir()
    saved_hidden = root / ".discourselens-original"
    original_validate = project_module._validate_managed_storage_paths
    swapped = False

    def replace_storage_after_validation(project_root):
        nonlocal swapped
        original_validate(project_root)
        if not swapped:
            hidden.rename(saved_hidden)
            _symlink_or_skip(hidden, outside, target_is_directory=True)
            swapped = True

    monkeypatch.setattr(
        project_module, "_validate_managed_storage_paths", replace_storage_after_validation
    )

    with pytest.raises(ProjectError, match="managed project storage"):
        Project.init(root)

    assert swapped
    assert list(outside.iterdir()) == []


def test_project_open_rejects_managed_artifact_directory_symlink(tmp_path):
    source = Project.init(tmp_path / "source-project")
    source_root = source.root
    source.close()
    project_root = tmp_path / "project"
    project = Project.init(project_root)
    project.close()

    nlp_dir = project_root / ".discourselens" / "artifacts" / "nlp"
    nlp_dir.rmdir()
    _symlink_or_skip(
        nlp_dir, source_root / ".discourselens" / "artifacts" / "nlp", target_is_directory=True
    )

    with pytest.raises(ProjectError, match="managed project storage"):
        Project.open(project_root)


def test_project_open_rejects_symlinked_database(tmp_path):
    source = Project.init(tmp_path / "source-project")
    source_database = source.hidden / "project.sqlite"
    source.close()
    project_root = tmp_path / "project"
    project = Project.init(project_root)
    project.close()

    database = project_root / ".discourselens" / "project.sqlite"
    database.unlink()
    _symlink_or_skip(database, source_database)

    with pytest.raises(ProjectError, match="managed project storage"):
        Project.open(project_root)


@pytest.mark.parametrize("operation", ["open", "init"])
def test_project_rejects_hardlinked_database_without_mutating_external_project(
    tmp_path, operation
):
    external = Project.init(tmp_path / "external-project")
    external_database = external.hidden / "project.sqlite"
    external.close()

    project_root = tmp_path / "selected-project"
    project = Project.init(project_root)
    project_database = project.hidden / "project.sqlite"
    project.close()
    project_database.unlink()
    try:
        project_database.hardlink_to(external_database)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")

    with pytest.raises(ProjectError, match="managed project storage file must not be hard-linked"):
        if operation == "open":
            Project.open(project_root)
        else:
            Project.init(project_root)

    project_database.unlink()
    with Project.open(tmp_path / "external-project", read_only=True) as unchanged:
        assert unchanged.list_corpora() == []


@pytest.mark.parametrize("component", ["database", "hidden_directory"])
def test_project_open_rejects_storage_symlink_swapped_during_sqlite_open(
    tmp_path, monkeypatch, component
):
    external = Project.init(tmp_path / "external-project")
    external_root = external.root
    external_database = external.hidden / "project.sqlite"
    external_hidden = external.hidden
    external.close()

    project_root = tmp_path / "selected-project"
    project = Project.init(project_root)
    project.close()
    database = project_root / ".discourselens" / "project.sqlite"
    hidden = project_root / ".discourselens"
    saved_component = tmp_path / f"selected-{component}-original"
    original_connect = project_module.sqlite3.connect
    swapped = False

    def replace_storage_before_connect(path, *args, **kwargs):
        nonlocal swapped
        if not swapped and _is_database_connect_path(path, database):
            if component == "database":
                database.rename(saved_component)
                _symlink_or_skip(database, external_database)
            else:
                hidden.rename(saved_component)
                _symlink_or_skip(hidden, external_hidden, target_is_directory=True)
            swapped = True
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(project_module.sqlite3, "connect", replace_storage_before_connect)

    with pytest.raises(ProjectError, match="managed project storage"):
        Project.open(project_root)

    assert swapped
    if component == "database":
        database.unlink()
    else:
        hidden.unlink()
    saved_component.rename(database if component == "database" else hidden)
    with Project.open(external_root, read_only=True) as unchanged:
        assert unchanged.list_corpora() == []


@pytest.mark.parametrize("component", ["database", "hidden_directory"])
def test_project_open_rejects_sqlite_target_redirect_restored_before_validation(
    tmp_path, monkeypatch, component
):
    project_root = tmp_path / "selected-project"
    project = Project.init(project_root)
    project.close()
    database = project_root / ".discourselens" / "project.sqlite"
    hidden = project_root / ".discourselens"
    displaced = tmp_path / f"selected-{component}-original"
    external_hidden = tmp_path / "outside-hidden"

    if component == "database":
        external_database = tmp_path / "outside.sqlite"
    else:
        external_hidden.mkdir()
        external_database = external_hidden / "project.sqlite"
    sqlite3.connect(external_database).close()

    original_connect = project_module.sqlite3.connect
    redirected = False
    redirected_connections: list[sqlite3.Connection] = []

    def transient_redirect(path, *args, **kwargs):
        nonlocal redirected
        if not redirected and _is_database_connect_path(path, database):
            redirected = True
            if component == "database":
                database.rename(displaced)
                _symlink_or_skip(database, external_database)
            else:
                hidden.rename(displaced)
                _symlink_or_skip(hidden, external_hidden, target_is_directory=True)
            try:
                connection = original_connect(path, *args, **kwargs)
                redirected_connections.append(connection)
                return connection
            finally:
                if component == "database":
                    database.unlink()
                    displaced.rename(database)
                else:
                    hidden.unlink()
                    displaced.rename(hidden)
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(project_module.sqlite3, "connect", transient_redirect)

    with pytest.raises(ProjectError, match="managed project storage changed during database open"):
        Project.open(project_root)

    assert redirected
    assert len(redirected_connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        redirected_connections[0].execute("SELECT 1")
    external_connection = original_connect(external_database)
    try:
        external_tables = {
            row[0]
            for row in external_connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        external_connection.close()
    assert "schema_migrations" not in external_tables
    with Project.open(project_root, read_only=True) as unchanged:
        assert unchanged.list_corpora() == []


@pytest.mark.parametrize("component", ["database", "hidden_directory"])
def test_project_open_does_not_create_a_database_through_transient_symlink(
    tmp_path, monkeypatch, component
):
    project_root = tmp_path / "selected-project"
    project = Project.init(project_root)
    project.close()
    database = project_root / ".discourselens" / "project.sqlite"
    hidden = project_root / ".discourselens"
    displaced = tmp_path / f"selected-{component}-original"
    external_hidden = tmp_path / "outside-hidden"

    if component == "database":
        external_database = tmp_path / "outside.sqlite"
    else:
        external_hidden.mkdir()
        external_database = external_hidden / "project.sqlite"

    original_connect = project_module.sqlite3.connect
    redirected = False

    def transient_redirect(path, *args, **kwargs):
        nonlocal redirected
        if not redirected and _is_database_connect_path(path, database):
            redirected = True
            if component == "database":
                database.rename(displaced)
                _symlink_or_skip(database, external_database)
            else:
                hidden.rename(displaced)
                _symlink_or_skip(hidden, external_hidden, target_is_directory=True)
            try:
                return original_connect(path, *args, **kwargs)
            finally:
                if component == "database":
                    database.unlink()
                    displaced.rename(database)
                else:
                    hidden.unlink()
                    displaced.rename(hidden)
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(project_module.sqlite3, "connect", transient_redirect)

    with pytest.raises(ProjectError, match="managed project storage is unavailable or unsafe"):
        Project.open(project_root)

    assert redirected
    assert not external_database.exists()


def test_project_init_reserves_database_via_pinned_directory_descriptor(tmp_path, monkeypatch):
    if (
        os.open not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
    ):
        pytest.skip("directory-descriptor storage creation is unavailable")

    root = tmp_path / "project"
    root.mkdir()
    hidden = root / ".discourselens"
    displaced = root / ".discourselens-original"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_open = project_module.os.open
    swapped = False

    def redirect_hidden_before_database_create(path, flags, *args, **kwargs):
        nonlocal swapped
        if (
            not swapped
            and path == "project.sqlite"
            and kwargs.get("dir_fd") is not None
        ):
            hidden.rename(displaced)
            _symlink_or_skip(hidden, outside, target_is_directory=True)
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(project_module.os, "open", redirect_hidden_before_database_create)

    with pytest.raises(ProjectError, match="managed project storage"):
        Project.init(root)

    assert swapped
    assert list(outside.iterdir()) == []
    assert (displaced / "project.sqlite").is_file()
    hidden.unlink()
    displaced.rename(hidden)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not Windows ACLs")
def test_project_init_creates_database_without_group_or_other_permissions(tmp_path):
    project = Project.init(tmp_path / "project")
    database = project.hidden / "project.sqlite"
    project.close()

    assert stat.S_IMODE(database.stat().st_mode) & 0o077 == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not Windows ACLs")
def test_import_text_creates_source_files_without_group_or_other_permissions(tmp_path):
    project = Project.init(tmp_path / "project")
    previous_umask = os.umask(0o022)
    try:
        _, revision_id = project.import_text(
            "notes.txt", "Private project text.", source_bytes=b"original source bytes"
        )
    finally:
        os.umask(previous_umask)

    paths = project.db.execute(
        "SELECT text_path,original_path FROM document_revisions WHERE id=?", (revision_id,)
    ).fetchone()
    assert paths is not None
    text_file = project.hidden / paths["text_path"]
    original_file = project.hidden / paths["original_path"]

    assert stat.S_IMODE(text_file.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(original_file.stat().st_mode) & 0o077 == 0


@pytest.mark.parametrize(
    ("column", "reader"),
    [("text_path", "revision_text"), ("original_path", "original_bytes")],
)
def test_revision_source_reads_reject_database_path_traversal(tmp_path, column, reader):
    project = Project.init(tmp_path / "project")
    _, revision_id = project.import_text("notes.txt", "Private project text.")
    outside = tmp_path / "outside.txt"
    outside.write_text("must not be readable through the project API", encoding="utf-8")
    project.db.execute(
        f"UPDATE document_revisions SET {column}=? WHERE id=?",
        ("../../outside.txt", revision_id),
    )
    project.db.commit()

    with pytest.raises(DocumentError, match="stored document path is invalid"):
        getattr(project, reader)(revision_id)


@pytest.mark.parametrize(
    ("column", "reader", "suffix"),
    [("text_path", "revision_text", ".txt"), ("original_path", "original_bytes", ".original")],
)
def test_revision_source_reads_reject_symlink_replaced_after_open(
    tmp_path, column, reader, suffix
):
    project = Project.init(tmp_path / "project")
    _, revision_id = project.import_text("notes.txt", "Private project text.")
    relative_path = project.db.execute(
        f"SELECT {column} FROM document_revisions WHERE id=?", (revision_id,)
    ).fetchone()[0]
    source_file = project.hidden / relative_path
    source_file.unlink()
    outside = tmp_path / f"outside{suffix}"
    outside.write_text("must not be readable through the project API", encoding="utf-8")
    _symlink_or_skip(source_file, outside)

    with pytest.raises(DocumentError, match="stored document path is unsafe"):
        getattr(project, reader)(revision_id)


@pytest.mark.parametrize(
    ("column", "reader", "hash_column"),
    [
        ("text_path", "revision_text", "canonical_hash"),
        ("original_path", "original_bytes", "source_hash"),
    ],
)
def test_revision_source_reads_reject_symlink_retargeted_after_validation(
    tmp_path, monkeypatch, column, reader, hash_column
):
    project = Project.init(tmp_path / "project")
    _, revision_id = project.import_text(
        "notes.txt", "Private project text.", source_bytes=b"private original bytes"
    )
    revision = project.db.execute(
        f"SELECT {column} FROM document_revisions WHERE id=?", (revision_id,)
    ).fetchone()
    source_file = project.hidden / revision[column]
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"external private payload")
    # Model a portable/tampered project whose recorded hash matches the file an
    # attacker is trying to redirect the path-backed reader to. The confinement
    # guard, rather than only the content hash, must prevent the read.
    project.db.execute(
        f"UPDATE document_revisions SET {hash_column}=? WHERE id=?",
        (project_module.digest(outside.read_bytes()), revision_id),
    )
    project.db.commit()

    original_resolver = project._resolve_managed_source_file
    retargeted = False

    def validate_then_retarget(stored_path, *, must_exist=True):
        nonlocal retargeted
        resolved = original_resolver(stored_path, must_exist=must_exist)
        if not retargeted:
            source_file.unlink()
            _symlink_or_skip(source_file, outside)
            retargeted = True
        return resolved

    monkeypatch.setattr(project, "_resolve_managed_source_file", validate_then_retarget)

    with pytest.raises(DocumentError, match="unsafe|changed during read"):
        getattr(project, reader)(revision_id)
    assert retargeted


def test_import_text_rejects_source_directory_symlink_replaced_after_open(tmp_path):
    root = tmp_path / "project"
    project = Project.init(root)
    source_dir = project.hidden / "source"
    original_dir = project.hidden / "source-original"
    source_dir.rename(original_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(source_dir, outside, target_is_directory=True)

    with pytest.raises(DocumentError, match="stored document path is unsafe"):
        project.import_text("notes.txt", "Must remain inside the project.")

    assert list(outside.iterdir()) == []
    assert not project.db.execute("SELECT 1 FROM document_revisions").fetchone()


def test_import_text_persists_sentences_once_across_multiple_paragraphs(tmp_path):
    project = Project.init(tmp_path / "project")

    _, revision_id = project.import_text(
        "notes.txt", "First sentence. Second sentence.\nThird sentence. Fourth sentence."
    )

    rows = project.db.execute(
        "SELECT ordinal,text FROM sentences WHERE revision_id=? ORDER BY ordinal",
        (revision_id,),
    ).fetchall()
    assert [(row["ordinal"], row["text"]) for row in rows] == [
        (0, "First sentence."),
        (1, "Second sentence."),
        (2, "Third sentence."),
        (3, "Fourth sentence."),
    ]
    project.close()


def test_import_text_pins_source_directory_when_it_is_swapped_before_file_open(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "project")
    source_dir = project.hidden / "source"
    original_source_dir = project.hidden / "source-original"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_open = project_module.os.open
    swapped = False

    def replace_source_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if (
            not swapped
            and isinstance(path, str)
            and path.endswith(".txt")
            and kwargs.get("dir_fd") is not None
        ):
            source_dir.rename(original_source_dir)
            _symlink_or_skip(source_dir, outside, target_is_directory=True)
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(project_module.os, "open", replace_source_before_open)

    with pytest.raises(DocumentError, match="stored document path is unsafe"):
        project.import_text("notes.txt", "Must stay inside the project.")

    assert swapped
    assert list(outside.iterdir()) == []
    assert list(original_source_dir.iterdir()) == []
    assert not project.db.execute("SELECT 1 FROM document_revisions").fetchone()
    source_dir.unlink()
    original_source_dir.rename(source_dir)
    project.close()
