from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def _write_pdf(path: Path, text: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    if text is not None:
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def _import_initial_pdf(project_root: Path, source: Path) -> tuple[str, str]:
    with Project.init(project_root) as project:
        imported = project.import_document(source)
    assert imported.document_id is not None
    assert imported.revision_id is not None
    return imported.document_id, imported.revision_id


def test_document_reprocess_help_exposes_pdf_options() -> None:
    result = CliRunner().invoke(app, ["document", "reprocess", "--help"])

    assert result.exit_code == 0, result.output
    assert "document_id" in result.output
    assert "--extractor-suffix" in result.output
    assert "--max-bytes" in result.output
    assert "--ocr-language" in result.output
    assert "--ocr-dpi" in result.output


def test_document_reprocess_creates_a_new_revision_without_leaking_path(tmp_path: Path) -> None:
    project_root = tmp_path / "reprocess-project"
    source = tmp_path / "private-inbox" / "updated scan.pdf"
    _write_pdf(source, "Revision A is archived.")
    document_id, first_revision = _import_initial_pdf(project_root, source)
    _write_pdf(source, "Revision B is preserved.")

    result = CliRunner().invoke(
        app,
        ["document", "reprocess", str(project_root), document_id, str(source)],
    )

    assert result.exit_code == 0, result.output
    payload: dict[str, Any] = json.loads(result.stdout)
    assert payload["document_id"] == document_id
    assert payload["status"] == "SUCCESS"
    assert isinstance(payload["revision_id"], str)
    assert payload["revision_id"] != first_revision
    assert str(tmp_path) not in result.output
    assert source.name not in result.output

    with Project.open(project_root, read_only=True) as project:
        extractor = project.document_revision_summary(payload["revision_id"])["extractor"]
        assert isinstance(extractor, str)
        assert extractor.endswith(".reprocessed")


def test_document_reprocess_reports_ocr_required_and_exits_nonzero(tmp_path: Path) -> None:
    project_root = tmp_path / "ocr-project"
    original = tmp_path / "original.pdf"
    blank_scan = tmp_path / "private-inbox" / "blank scan.pdf"
    _write_pdf(original, "Initial text exists.")
    document_id, _ = _import_initial_pdf(project_root, original)
    _write_pdf(blank_scan, None)

    result = CliRunner().invoke(
        app,
        ["document", "reprocess", str(project_root), document_id, str(blank_scan)],
    )

    assert result.exit_code == 1
    payload: dict[str, Any] = json.loads(result.stdout)
    assert payload["status"] == "OCR_REQUIRED"
    assert payload["document_id"] is None
    assert "NO_TEXT_FOUND" in payload["warnings"]
    assert "--ocr-language" in result.stderr
    assert str(tmp_path) not in result.output


def test_document_reprocess_invalid_document_and_path_errors_are_safe(tmp_path: Path) -> None:
    project_root = tmp_path / "error-project"
    source = tmp_path / "source.pdf"
    _write_pdf(source, "Source text.")
    document_id, _ = _import_initial_pdf(project_root, source)

    unknown_document = CliRunner().invoke(
        app,
        ["document", "reprocess", str(project_root), "missing-document", str(source)],
    )
    missing_path = tmp_path / "private-inbox" / "missing.pdf"
    inaccessible_file = CliRunner().invoke(
        app,
        ["document", "reprocess", str(project_root), document_id, str(missing_path)],
    )

    assert unknown_document.exit_code == 1
    assert "unknown document" in unknown_document.output
    assert str(tmp_path) not in unknown_document.output
    assert inaccessible_file.exit_code == 1
    assert "document path cannot be opened safely" in inaccessible_file.output
    assert str(tmp_path) not in inaccessible_file.output
