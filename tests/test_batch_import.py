from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from discourselens import Project
from discourselens.documents import ImportResult


def test_sdk_batch_import_preserves_order_counts_and_input(tmp_path: Path):
    first = tmp_path / "first.txt"
    invalid = tmp_path / "invalid.txt"
    last = tmp_path / "last.txt"
    first.write_text("First document.", encoding="utf-8")
    invalid.write_bytes(b"\xff")
    last.write_text("Last document.", encoding="utf-8")
    paths = [first, invalid, first, last]
    original_paths = paths.copy()

    with Project.init(tmp_path / "project") as project:
        result = project.import_documents(paths)

    assert paths == original_paths
    assert [item.index for item in result.items] == [0, 1, 2, 3]
    assert [item.status for item in result.items] == [
        "SUCCESS",
        "FAILED",
        "DUPLICATE",
        "SUCCESS",
    ]
    assert result.total_count == 4
    assert result.success_count == 2
    assert result.duplicate_count == 1
    assert result.failed_count == 1
    assert result.items[0].document_id is not None
    assert result.items[0].revision_id is not None
    assert result.items[2].document_id == result.items[0].document_id
    assert result.items[2].revision_id == result.items[0].revision_id
    assert result.items[1].failure_code == "DOCUMENT_ERROR"
    assert result.items[1].document_id is None
    assert result.items[1].revision_id is None

    payload = json.dumps(asdict(result))
    assert str(tmp_path) not in payload
    assert "invalid.txt" not in payload


def test_sdk_batch_import_handles_os_errors_and_empty_iterable(tmp_path: Path, monkeypatch):
    with Project.init(tmp_path / "project") as project:
        empty = project.import_documents(iter(()))
        assert asdict(empty) == {
            "items": (),
            "total_count": 0,
            "success_count": 0,
            "duplicate_count": 0,
            "failed_count": 0,
        }

        calls: list[Path] = []
        successful_path = tmp_path / "successful.txt"
        failing_path = tmp_path / "private-secret-name.txt"

        def import_one(path, **kwargs):
            del kwargs
            calls.append(path)
            if path == failing_path:
                raise OSError(f"cannot read {path}")
            return ImportResult("document-id", "revision-id", "SUCCESS")

        monkeypatch.setattr(project, "import_document", import_one)
        result = project.import_documents([successful_path, failing_path])

        assert calls == [successful_path, failing_path]
        assert [item.status for item in result.items] == ["SUCCESS", "FAILED"]
        assert [item.failure_code for item in result.items] == [None, "OS_ERROR"]
        assert result.success_count == 1
        assert result.failed_count == 1
        assert str(tmp_path) not in json.dumps(asdict(result))
        assert "private-secret-name.txt" not in json.dumps(asdict(result))


def test_sdk_batch_import_does_not_swallow_unexpected_errors(tmp_path: Path, monkeypatch):
    with Project.init(tmp_path / "project") as project:
        def broken_import(path, **kwargs):
            del path, kwargs
            raise RuntimeError("unexpected implementation failure")

        monkeypatch.setattr(project, "import_document", broken_import)
        with pytest.raises(RuntimeError, match="unexpected implementation failure"):
            project.import_documents([tmp_path / "source.txt"])


def test_sdk_batch_import_preserves_success_with_warnings(tmp_path: Path, monkeypatch):
    with Project.init(tmp_path / "project") as project:
        monkeypatch.setattr(
            project,
            "import_document",
            lambda path, **kwargs: ImportResult(
                "document-id", "revision-id", "SUCCESS_WITH_WARNINGS", ("OCR_PARTIAL",)
            ),
        )

        result = project.import_documents([tmp_path / "scan.pdf"])

    assert result.success_count == 1
    assert result.failed_count == 0
    assert result.items[0].status == "SUCCESS"
    assert result.items[0].revision_id == "revision-id"
    assert result.items[0].warnings == ("OCR_PARTIAL",)


@pytest.mark.parametrize("status", ["OCR_REQUIRED", "PASSWORD_REQUIRED"])
def test_sdk_batch_import_preserves_actionable_document_statuses(
    tmp_path: Path, monkeypatch, status: str
):
    with Project.init(tmp_path / "project") as project:
        monkeypatch.setattr(
            project,
            "import_document",
            lambda path, **kwargs: ImportResult(None, None, status),
        )

        result = project.import_documents([tmp_path / "source.pdf"])

    assert result.failed_count == 1
    assert result.items[0].status == "FAILED"
    assert result.items[0].failure_code == status
