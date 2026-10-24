from __future__ import annotations

import os
from pathlib import Path

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError


def _create_project_and_roots(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    Project.init(project_root).close()
    import_root = tmp_path / "inbox"
    import_root.mkdir()
    return project_root, import_root


@pytest.mark.parametrize(
    ("entrypoint", "swap_phase"),
    [
        ("single", "before_parent_open"),
        ("batch", "before_parent_open"),
        ("single", "after_parent_open"),
        ("batch", "after_parent_open"),
    ],
)
def test_mcp_import_ancestor_symlink_swap_never_imports_outside_bytes(
    tmp_path, monkeypatch, entrypoint, swap_phase
):
    if not mcp_module._supports_secure_import_open():
        pytest.skip("descriptor-relative no-follow opens unavailable")

    project_root, import_root = _create_project_and_roots(tmp_path)
    source_directory = import_root / "sub"
    source_directory.mkdir()
    source_file = source_directory / "input.txt"
    source_file.write_text("allowed initial file", encoding="utf-8")
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    outside_marker = "SYNTHETIC_OUTSIDE_MARKER_MUST_NOT_IMPORT"
    (outside_directory / source_file.name).write_text(outside_marker, encoding="utf-8")
    saved_directory = import_root / "sub-original"

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(project_root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(import_root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
    real_open = os.open
    swapped = False
    # The monkeypatch changes the shared os module object; report the platform
    # capabilities captured above rather than making the test disable itself.
    monkeypatch.setattr(mcp_module, "_supports_secure_import_open", lambda: True)

    def replace_ancestor_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        target_component = "sub" if swap_phase == "before_parent_open" else source_file.name
        if path == target_component and kwargs.get("dir_fd") is not None and not swapped:
            source_directory.rename(saved_directory)
            try:
                source_directory.symlink_to(outside_directory, target_is_directory=True)
            except OSError as exc:
                saved_directory.rename(source_directory)
                pytest.skip(f"directory symlinks unavailable: {exc}")
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(mcp_module.os, "open", replace_ancestor_before_open)

    if swap_phase == "before_parent_open":
        if entrypoint == "single":
            with pytest.raises(ProjectError):
                mcp_module.import_document(str(source_file))
        else:
            result = mcp_module.import_documents([str(source_file)])
            assert result["failed_count"] == 1
            items = result["items"]
            assert isinstance(items, list)
            item = items[0]
            assert isinstance(item, dict)
            assert item["status"] == "FAILED"
    elif entrypoint == "single":
        result = mcp_module.import_document(str(source_file))
        assert result["status"] == "SUCCESS"
    else:
        result = mcp_module.import_documents([str(source_file)])
        assert result["success_count"] == 1

    assert swapped
    monkeypatch.setattr(mcp_module.os, "open", real_open)
    with Project.open(project_root, read_only=True) as project:
        expected_count = int(swap_phase == "after_parent_open")
        assert project.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == expected_count
        source_files = list((project.hidden / "source").glob("*"))
        assert all(outside_marker.encode() not in path.read_bytes() for path in source_files)
        if expected_count:
            imported = project.db.execute("SELECT id FROM document_revisions").fetchone()
            assert imported is not None
            text = project.revision_text(str(imported["id"]))
            assert text == "allowed initial file"
            assert outside_marker not in text


def test_mcp_single_import_still_accepts_an_in_root_regular_file(tmp_path, monkeypatch):
    project_root, import_root = _create_project_and_roots(tmp_path)
    source = import_root / "ordinary.txt"
    source.write_text("ordinary in-root content", encoding="utf-8")
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(project_root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(import_root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)

    result = mcp_module.import_document(str(source))

    assert result["status"] == "SUCCESS"
    with Project.open(project_root, read_only=True) as project:
        revision_id = str(result["revision_id"])
        assert project.revision_text(revision_id) == "ordinary in-root content"


def test_mcp_batch_import_preserves_order_and_partial_success(tmp_path, monkeypatch):
    project_root, import_root = _create_project_and_roots(tmp_path)
    first = import_root / "first.txt"
    first.write_text("first in-root document", encoding="utf-8")
    second = import_root / "second.txt"
    second.write_text("second in-root document", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside document", encoding="utf-8")
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(project_root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(import_root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)

    result = mcp_module.import_documents([str(first), str(outside), str(second)])

    assert result["total_count"] == 3
    assert result["success_count"] == 2
    assert result["failed_count"] == 1
    items = result["items"]
    assert isinstance(items, list)
    assert [item["index"] for item in items] == [0, 1, 2]
    assert [item["status"] for item in items] == ["SUCCESS", "FAILED", "SUCCESS"]
    assert items[1]["failure_code"] == "OUTSIDE_IMPORT_ROOT"


@pytest.mark.parametrize("entrypoint", ["single", "batch"])
def test_mcp_import_fails_closed_when_secure_descriptor_open_is_unavailable(
    tmp_path, monkeypatch, entrypoint
):
    project_root, import_root = _create_project_and_roots(tmp_path)
    source = import_root / "ordinary.txt"
    source.write_text("local content", encoding="utf-8")
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(project_root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(import_root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
    monkeypatch.setattr(mcp_module, "_supports_secure_import_open", lambda: False)

    if entrypoint == "single":
        with pytest.raises(ProjectError, match="unsupported on this platform"):
            mcp_module.import_document(str(source))
    else:
        result = mcp_module.import_documents([str(source)])
        assert result["failed_count"] == 1
        items = result["items"]
        assert isinstance(items, list)
        item = items[0]
        assert isinstance(item, dict)
        assert item["failure_code"] == "SECURE_IMPORT_UNAVAILABLE"
    with Project.open(project_root, read_only=True) as project:
        assert project.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_mcp_import_reader_enforces_size_limit_before_core_import(tmp_path):
    import_root = tmp_path / "inbox"
    import_root.mkdir()
    source = import_root / "large.txt"
    source.write_bytes(b"123456")

    filename, raw, failure = mcp_module._read_mcp_import_source(
        str(source), import_root, max_bytes=5
    )

    assert filename is None
    assert raw is None
    assert failure == "DOCUMENT_ERROR"
