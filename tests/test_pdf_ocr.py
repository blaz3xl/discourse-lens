import json
from types import SimpleNamespace

import pymupdf as fitz
import pytest
from typer.testing import CliRunner

from discourselens import Project, documents
from discourselens.cli import app
from discourselens.documents import project_pdf_text_boxes
from discourselens.errors import DocumentError


def _image_only_pdf(path):
    pdf = fitz.open()
    pdf.new_page()
    pdf.save(path)


def _blank_pdf(path, *, pages=1, width=612, height=792):
    pdf = fitz.open()
    for _ in range(pages):
        pdf.new_page(width=width, height=height)
    pdf.save(path)


def test_scanned_pdf_stays_ocr_required_without_explicit_local_ocr(tmp_path, monkeypatch):
    source = tmp_path / "scan.pdf"
    _image_only_pdf(source)
    project = Project.init(tmp_path / "project")

    def forbidden_ocr(*args, **kwargs):
        raise AssertionError("OCR must not run without explicit opt-in")

    monkeypatch.setattr(fitz.Page, "get_textpage_ocr", forbidden_ocr)

    result = documents.import_pdf(project, source)

    assert result.status == "OCR_REQUIRED"
    assert result.revision_id is None
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_automatic_ocr_fills_scanned_pdf_pages_without_user_options(tmp_path, monkeypatch):
    source = tmp_path / "scan.pdf"
    _image_only_pdf(source)
    project = Project.init(tmp_path / "project")
    calls = []

    def mocked_extraction(page, *, ocr_language, ocr_dpi):
        calls.append((page.number, ocr_language, ocr_dpi))
        if ocr_language is None:
            return documents._PdfBlockExtraction([], reading_order_ambiguous=False)
        return documents._PdfBlockExtraction(
            [("Café migración reconocida.", (20.0, 20.0, 220.0, 50.0))],
            reading_order_ambiguous=False,
        )

    monkeypatch.setattr(documents, "_extract_pdf_blocks", mocked_extraction)
    result = project.import_document(source, auto_ocr=True)

    assert result.status == "SUCCESS_WITH_WARNINGS"
    assert result.warnings == ("AUTOMATIC_OCR_APPLIED",)
    assert result.revision_id is not None
    assert calls == [(0, None, 300), (0, "eng+spa+nld", 300)]
    assert project.revision_text(result.revision_id) == "Café migración reconocida."
    revision = project.db.execute(
        "SELECT ocr_language,ocr_dpi FROM document_revisions WHERE id=?",
        (result.revision_id,),
    ).fetchone()
    assert revision["ocr_language"] == "eng+spa+nld"
    assert revision["ocr_dpi"] == 300


def test_explicit_local_ocr_persists_text_configuration_and_cli_provenance(tmp_path, monkeypatch):
    root = tmp_path / "project"
    Project.init(root).close()
    source = tmp_path / "scan.pdf"
    _image_only_pdf(source)
    calls = []

    def mocked_local_ocr(page, *, ocr_language, ocr_dpi):
        calls.append((page.number, ocr_language, ocr_dpi))
        return [("Café migración reconocida.", (20.0, 20.0, 220.0, 50.0))]

    monkeypatch.setattr(documents, "_extract_pdf_blocks", mocked_local_ocr)
    result = CliRunner().invoke(
        app,
        [
            "document",
            "import",
            str(root),
            str(source),
            "--ocr-language",
            "spa+eng",
            "--ocr-dpi",
            "300",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "SUCCESS_WITH_WARNINGS"
    assert payload["warnings"] == ["LOCAL_OCR_ENABLED"]
    assert calls == [(0, "spa+eng", 300)]
    project = Project.open(root)
    try:
        revision = project.db.execute(
            "SELECT extractor,ocr_language,ocr_dpi,extraction_status FROM document_revisions "
            "WHERE id=?",
            (payload["revision_id"],),
        ).fetchone()
        assert revision["ocr_language"] == "spa+eng"
        assert revision["ocr_dpi"] == 300
        assert "tesseract-ocr[lang=spa+eng,dpi=300,full=false]" in revision["extractor"]
        assert revision["extraction_status"] == "SUCCESS_WITH_WARNINGS"
        assert project.revision_text(payload["revision_id"]) == "Café migración reconocida."
        assert [item["code"] for item in project.extraction_warnings(payload["revision_id"])] == [
            "LOCAL_OCR_ENABLED"
        ]
    finally:
        project.close()


def test_explicit_ocr_reimport_keeps_extractor_provenance_in_duplicate_identity(
    tmp_path, monkeypatch
):
    source = tmp_path / "digital.pdf"
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "Same canonical text.")
    pdf.save(source)
    pdf.close()
    project = Project.init(tmp_path / "project")
    calls = []

    def same_text_with_selected_extractor(page, *, ocr_language, ocr_dpi):
        calls.append((page.number, ocr_language, ocr_dpi))
        return [("Same canonical text.", (72.0, 72.0, 200.0, 90.0))]

    monkeypatch.setattr(documents, "_extract_pdf_blocks", same_text_with_selected_extractor)

    digital = documents.import_pdf(project, source)
    ocr = documents.import_pdf(project, source, ocr_language="eng", ocr_dpi=240)
    repeated_ocr = documents.import_pdf(project, source, ocr_language="eng", ocr_dpi=240)

    assert digital.status == "SUCCESS"
    assert ocr.status == "SUCCESS_WITH_WARNINGS"
    assert ocr.document_id == digital.document_id
    assert ocr.revision_id != digital.revision_id
    assert ocr.warnings == ("LOCAL_OCR_ENABLED",)
    assert repeated_ocr.status == "DUPLICATE"
    assert repeated_ocr.revision_id == ocr.revision_id
    assert calls == [(0, None, 300), (0, "eng", 240), (0, "eng", 240)]

    revisions = project.db.execute(
        "SELECT extractor,ocr_language,ocr_dpi,extraction_status FROM document_revisions "
        "ORDER BY id"
    ).fetchall()
    assert len(revisions) == 2
    native_revision = next(revision for revision in revisions if revision["ocr_language"] is None)
    ocr_revision = next(revision for revision in revisions if revision["ocr_language"] == "eng")
    assert ocr_revision["ocr_dpi"] == 240
    assert "tesseract-ocr[lang=eng,dpi=240,full=false]" in ocr_revision["extractor"]
    assert ocr_revision["extraction_status"] == "SUCCESS_WITH_WARNINGS"
    assert native_revision["extraction_status"] == "SUCCESS"


def test_local_ocr_failure_does_not_create_partial_revision(tmp_path, monkeypatch):
    source = tmp_path / "scan.pdf"
    _image_only_pdf(source)
    project = Project.init(tmp_path / "project")

    def unavailable_backend(page, *, ocr_language, ocr_dpi):
        raise RuntimeError("Tesseract unavailable")

    monkeypatch.setattr(documents, "_extract_pdf_blocks", unavailable_backend)
    with pytest.raises(DocumentError, match="local OCR failed"):
        documents.import_pdf(project, source, ocr_language="eng")
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_unexpected_local_ocr_errors_are_normalized_and_leave_no_revision(tmp_path, monkeypatch):
    source = tmp_path / "scan.pdf"
    _image_only_pdf(source)
    project = Project.init(tmp_path / "project")

    def backend_failure(page, *, ocr_language, ocr_dpi):
        del page, ocr_language, ocr_dpi
        raise LookupError("private tessdata path /research/secret/tessdata missing")

    monkeypatch.setattr(documents, "_extract_pdf_blocks", backend_failure)
    with pytest.raises(DocumentError, match="local OCR failed") as error:
        documents.import_pdf(project, source, ocr_language="eng")

    assert "/research/secret/tessdata" not in str(error.value)
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_ocr_text_and_bbox_paths_reuse_the_explicit_local_settings(tmp_path, monkeypatch):
    source = tmp_path / "digital.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Local OCR settings.")
    pdf.save(source)
    calls = []

    def local_text_page(self, *, language, dpi, full):
        calls.append((language, dpi, full))
        return self.get_textpage()

    monkeypatch.setattr(fitz.Page, "get_textpage_ocr", local_text_page)
    with fitz.open(source) as opened:
        blocks = documents._extract_pdf_blocks(opened[0], ocr_language="eng", ocr_dpi=240)
    text = blocks[0][0]
    boxes, status, _ = project_pdf_text_boxes(
        source.read_bytes(),
        0,
        [{"id": "paragraph", "text": text, "start_offset": 0, "end_offset": len(text)}],
        0,
        len(text),
        ocr_language="eng",
        ocr_dpi=240,
    )

    assert text == "Local OCR settings."
    assert calls == [("eng", 240, False), ("eng", 240, False)]
    assert status == "AVAILABLE"
    assert boxes


@pytest.mark.parametrize(
    ("language", "dpi"),
    [
        ("", 300),
        ("eng;rm -rf", 300),
        ("eng+", 300),
        ("eng", 71),
        ("eng", 601),
        ("eng", 300.0),
        ("eng", True),
        (123, 300),
    ],
)
def test_invalid_local_ocr_configuration_is_rejected(tmp_path, language, dpi):
    source = tmp_path / "scan.pdf"
    _image_only_pdf(source)
    project = Project.init(tmp_path / "project")

    with pytest.raises(DocumentError, match="OCR"):
        documents.import_pdf(project, source, ocr_language=language, ocr_dpi=dpi)


@pytest.mark.parametrize(
    ("language", "dpi"),
    [("eng", None), (None, 300)],
)
def test_pdf_geometry_rejects_incomplete_local_ocr_configuration(language, dpi):
    with pytest.raises(DocumentError, match="provided together"):
        project_pdf_text_boxes(b"", 0, [], 0, 1, ocr_language=language, ocr_dpi=dpi)


def test_local_ocr_rejects_huge_finite_page_dimensions_without_float_overflow():
    class FakePdf:
        page_count = 1

        def __getitem__(self, index):
            assert index == 0
            return SimpleNamespace(rect=SimpleNamespace(width=1e308, height=1.0))

    with pytest.raises(DocumentError, match="pixel safety limit"):
        documents._validate_pdf_ocr_workload(FakePdf(), 600)


def test_text_files_reject_pdf_ocr_options(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("Café", encoding="utf-8")
    project = Project.init(tmp_path / "project")

    with pytest.raises(DocumentError, match="only for PDF"):
        documents.import_document(project, source, ocr_language="eng")


def test_local_ocr_rejects_excessive_page_count_before_rendering(tmp_path, monkeypatch):
    source = tmp_path / "many-pages.pdf"
    _blank_pdf(source, pages=2)
    project = Project.init(tmp_path / "project")
    calls = []

    def forbidden_ocr(page, *, ocr_language, ocr_dpi):
        del page, ocr_language, ocr_dpi
        calls.append("called")
        return []

    monkeypatch.setattr(documents, "MAX_OCR_PAGES", 1)
    monkeypatch.setattr(documents, "_extract_pdf_blocks", forbidden_ocr)

    with pytest.raises(DocumentError, match="page-count safety limit"):
        documents.import_pdf(project, source, ocr_language="eng")

    assert calls == []
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_local_ocr_rejects_oversized_page_render_before_rendering(tmp_path, monkeypatch):
    source = tmp_path / "oversized-page.pdf"
    _blank_pdf(source, width=1000, height=1000)
    project = Project.init(tmp_path / "project")
    calls = []

    def forbidden_ocr(page, *, ocr_language, ocr_dpi):
        del page, ocr_language, ocr_dpi
        calls.append("called")
        return []

    monkeypatch.setattr(documents, "_extract_pdf_blocks", forbidden_ocr)

    with pytest.raises(DocumentError, match="page render.*pixel safety limit"):
        documents.import_pdf(project, source, ocr_language="eng", ocr_dpi=600)

    assert calls == []
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_local_ocr_rejects_excessive_aggregate_pixels_before_rendering(tmp_path, monkeypatch):
    source = tmp_path / "large-workload.pdf"
    _blank_pdf(source, pages=2)
    project = Project.init(tmp_path / "project")
    calls = []

    def forbidden_ocr(page, *, ocr_language, ocr_dpi):
        del page, ocr_language, ocr_dpi
        calls.append("called")
        return []

    # Each Letter page at 300 DPI is below the per-page cap, but their combined
    # estimated render workload exceeds this deterministic test budget.
    monkeypatch.setattr(documents, "MAX_OCR_TOTAL_PIXELS", 10_000_000)
    monkeypatch.setattr(documents, "_extract_pdf_blocks", forbidden_ocr)

    with pytest.raises(DocumentError, match="pixel workload limit"):
        documents.import_pdf(project, source, ocr_language="eng")

    assert calls == []
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0
