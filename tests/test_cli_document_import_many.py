from __future__ import annotations

import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def test_document_import_many_help_advertises_ordered_inputs_and_options():
    result = CliRunner().invoke(app, ["document", "import-many", "--help"])

    assert result.exit_code == 0, result.output
    assert "processed in this order" in result.output
    assert "--max-bytes" in result.output
    assert "--ocr-language" in result.output
    assert "--ocr-dpi" in result.output


def test_document_import_many_preserves_order_and_reports_duplicates(tmp_path):
    project_root = tmp_path / "batch-project"
    Project.init(project_root).close()
    first = tmp_path / "first.txt"
    duplicate = tmp_path / "first-copy.txt"
    last = tmp_path / "last.md"
    first.write_text("First source text.", encoding="utf-8")
    duplicate.write_text("First source text.", encoding="utf-8")
    last.write_text("Last source text.", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "document",
            "import-many",
            str(project_root),
            str(first),
            str(duplicate),
            str(last),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total_count"] == 3
    assert payload["success_count"] == 2
    assert payload["duplicate_count"] == 1
    assert payload["failed_count"] == 0
    items = payload["items"]
    assert [item["index"] for item in items] == [0, 1, 2]
    assert [item["source_name"] for item in items] == [
        "first.txt",
        "first-copy.txt",
        "last.md",
    ]
    assert [item["status"] for item in items] == ["SUCCESS", "DUPLICATE", "SUCCESS"]
    assert items[0]["document_id"] == items[1]["document_id"]
    assert items[0]["revision_id"] == items[1]["revision_id"]
    assert items[2]["document_id"] != items[0]["document_id"]
    assert str(project_root) not in result.output


def test_document_import_many_returns_all_outcomes_and_nonzero_on_failures(tmp_path):
    project_root = tmp_path / "partial-batch-project"
    Project.init(project_root).close()
    valid = tmp_path / "valid.txt"
    too_large = tmp_path / "too-large.txt"
    unsupported = tmp_path / "unsupported.bin"
    directory = tmp_path / "folder"
    directory.mkdir()
    (directory / "nested.txt").write_text("Must not be imported recursively.", encoding="utf-8")
    valid.write_text("x", encoding="utf-8")
    too_large.write_text("This input is over the byte limit.", encoding="utf-8")
    unsupported.write_bytes(b"not a supported format")

    result = CliRunner().invoke(
        app,
        [
            "document",
            "import-many",
            str(project_root),
            str(valid),
            str(too_large),
            str(unsupported),
            str(directory),
            "--max-bytes",
            "8",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["total_count"] == 4
    assert payload["success_count"] == 1
    assert payload["duplicate_count"] == 0
    assert payload["failed_count"] == 3
    items = payload["items"]
    assert [item["index"] for item in items] == [0, 1, 2, 3]
    assert [item["source_name"] for item in items] == [
        "valid.txt",
        "too-large.txt",
        "unsupported.bin",
        "folder",
    ]
    assert [item["status"] for item in items] == ["SUCCESS", "FAILED", "FAILED", "FAILED"]
    assert all(item["failure_code"] for item in items[1:])
    assert all("error" not in item and "path" not in item for item in items)
    assert str(tmp_path) not in result.output

    with Project.open(project_root, read_only=True) as project:
        assert project.documents_page()["total"] == 1
