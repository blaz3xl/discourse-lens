import json
import os
import secrets
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, cast

import fitz
import pytest

from discourselens import Project, documents
from discourselens.config import ProjectConfig
from discourselens.documents import (
    PdfExtractionLimits,
    import_document,
    import_pdf,
    reprocess_pdf,
)
from discourselens.errors import DocumentError


def pdf(path, pages):
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)


def test_pdf_pages_segments_citation_and_duplicate(tmp_path):
    source = tmp_path / "paper.pdf"
    pdf(source, ["Café migración.", "Second page evidence."])
    project = Project.init(tmp_path / "p")
    result = import_pdf(project, source)
    assert result.status == "SUCCESS"
    assert result.revision_id is not None and result.document_id is not None
    revision = project.db.execute(
        "SELECT mime_type,source_size,original_path FROM document_revisions WHERE id=?",
        (result.revision_id,),
    ).fetchone()
    assert revision["mime_type"] == "application/pdf"
    assert revision["source_size"] == source.stat().st_size
    assert project.original_bytes(result.revision_id) == source.read_bytes()
    assert project.db.execute("select count(*) from pages").fetchone()[0] == 2
    assert project.db.execute(
        "SELECT COUNT(*) FROM paragraphs WHERE revision_id=?", (result.revision_id,)
    ).fetchone()[0] == 2
    assert project.db.execute(
        "SELECT COUNT(*) FROM sentences WHERE revision_id=?", (result.revision_id,)
    ).fetchone()[0] == 2
    assert project.db.execute(
        "SELECT COUNT(*) FROM revision_fts WHERE revision_id=?", (result.revision_id,)
    ).fetchone()[0] == 2
    assert import_pdf(project, source).status == "DUPLICATE"
    revision_id = result.revision_id
    start = project.revision_text(revision_id).index("Café")
    span = project.create_span(revision_id, start, start + 4)
    evidence = project.textual_evidence(span)
    project._segment_for_range(
        revision_id,
        project.db.execute("select id from pages order by page_index").fetchone()[0],
        start,
        start + 4,
        [(72, 72, 120, 90)],
    )
    assert "p. 1" in project.render_citation(evidence)


def test_pdf_structure_change_with_same_text_creates_a_distinct_revision(tmp_path, monkeypatch):
    source = tmp_path / "stable-source.pdf"
    pdf(source, ["same stable text"])
    project = Project.init(tmp_path / "project")

    first_blocks = documents._PdfBlockExtraction(
        [("same stable text", (72.0, 72.0, 180.0, 90.0))],
        reading_order_ambiguous=False,
    )
    monkeypatch.setattr(
        documents,
        "_extract_pdf_blocks",
        lambda page, *, ocr_language, ocr_dpi: first_blocks,
    )
    first = import_pdf(project, source)
    assert first.revision_id is not None

    second_blocks = documents._PdfBlockExtraction(
        [("same stable text", (82.0, 82.0, 190.0, 100.0))],
        reading_order_ambiguous=False,
    )
    monkeypatch.setattr(
        documents,
        "_extract_pdf_blocks",
        lambda page, *, ocr_language, ocr_dpi: second_blocks,
    )
    second = import_pdf(project, source)
    assert second.status == "SUCCESS"
    assert second.document_id == first.document_id
    assert second.revision_id is not None and second.revision_id != first.revision_id

    rows = project.db.execute(
        "SELECT id,extraction_manifest_json,structure_hash FROM document_revisions "
        "ORDER BY created_at,id"
    ).fetchall()
    assert len(rows) == 2
    first_manifest = json.loads(rows[0]["extraction_manifest_json"])
    assert first_manifest["normalization_version"] == documents.PDF_NORMALIZATION_VERSION
    assert first_manifest["reading_order_version"] == documents.PDF_READING_ORDER_VERSION
    assert first_manifest["paragraph_segmentation_version"] == (
        documents.PDF_PARAGRAPH_SEGMENTATION_VERSION
    )
    assert rows[0]["structure_hash"] != rows[1]["structure_hash"]
    assert import_pdf(project, source).status == "DUPLICATE"


def test_no_text_and_invalid_pdf(tmp_path):
    image = tmp_path / "image.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(image)
    project = Project.init(tmp_path / "p")
    assert import_pdf(project, image).status == "OCR_REQUIRED"
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not pdf")
    with pytest.raises(DocumentError):
        import_pdf(project, bad)


def test_pdf_page_limit_rejects_before_extraction_and_project_remains_usable(
    tmp_path, monkeypatch
):
    oversized = tmp_path / "too-many-pages.pdf"
    pdf(oversized, ["page one", "page two", "page three"])
    project = Project.init(tmp_path / "p")
    monkeypatch.setattr(documents, "MAX_PDF_PAGES", 2)
    extracted_pages = []
    original_extract = documents._extract_pdf_blocks

    def track_extraction(page, *, ocr_language, ocr_dpi):
        extracted_pages.append(page.number)
        return original_extract(page, ocr_language=ocr_language, ocr_dpi=ocr_dpi)

    monkeypatch.setattr(documents, "_extract_pdf_blocks", track_extraction)
    with pytest.raises(DocumentError, match="PDF exceeds configured page-count safety limit"):
        import_document(project, oversized)

    assert extracted_pages == []
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0

    accepted = tmp_path / "within-limit.pdf"
    pdf(accepted, ["project remains usable"])
    result = import_document(project, accepted)

    assert result.status == "SUCCESS"
    assert result.revision_id is not None
    assert project.revision_text(result.revision_id) == "project remains usable"
    assert extracted_pages == [0]


@pytest.mark.parametrize(
    ("blocks", "limits", "message", "expected_text"),
    [
        (
            [("expanded canonical text", (10.0, 10.0, 200.0, 30.0))],
            PdfExtractionLimits(max_canonical_chars=10, max_layout_blocks=10),
            "character limit",
            "expanded canonical text",
        ),
        (
            [
                ("A", (10.0, 10.0, 20.0, 30.0)),
                ("B", (30.0, 10.0, 40.0, 30.0)),
            ],
            PdfExtractionLimits(max_canonical_chars=10, max_layout_blocks=1),
            "block limit",
            "A\nB",
        ),
    ],
)
def test_pdf_extraction_limits_are_configurable_and_reject_before_persistence(
    tmp_path, monkeypatch, blocks, limits, message, expected_text
):
    source = tmp_path / "bounded-output.pdf"
    pdf(source, ["fixture text"])
    project = Project.init(tmp_path / "project")
    monkeypatch.setattr(
        documents,
        "_extract_pdf_blocks",
        lambda page, *, ocr_language, ocr_dpi: blocks,
    )

    with pytest.raises(DocumentError, match=message):
        import_document(project, source, pdf_extraction_limits=limits)

    for table in (
        "documents",
        "document_revisions",
        "pages",
        "paragraphs",
        "sentences",
        "extraction_warnings",
        "revision_fts",
    ):
        assert project.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert list((project.hidden / "source").iterdir()) == []
    assert not project.db.in_transaction

    accepted_limits = PdfExtractionLimits(max_canonical_chars=100, max_layout_blocks=10)
    accepted = import_document(project, source, pdf_extraction_limits=accepted_limits)
    assert accepted.status == "SUCCESS"
    assert accepted.revision_id is not None
    assert project.revision_text(accepted.revision_id) == expected_text


def test_compact_pdf_text_expansion_is_bounded_before_project_mutation(tmp_path):
    source = tmp_path / "compact-expanded-text.pdf"
    source_text = "A" * 10_000
    document = fitz.open()
    document.new_page(width=595, height=842).insert_text((2, 2), source_text, fontsize=0.1)
    document.save(source, deflate=True, garbage=4)
    document.close()
    assert source.stat().st_size < len(source_text)
    project = Project.init(tmp_path / "project")

    with pytest.raises(DocumentError, match="character limit"):
        import_document(
            project,
            source,
            pdf_extraction_limits=PdfExtractionLimits(
                max_canonical_chars=100, max_layout_blocks=10
            ),
        )

    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0] == 0
    assert list((project.hidden / "source").iterdir()) == []
    assert not project.db.in_transaction

    accepted = import_document(
        project,
        source,
        pdf_extraction_limits=PdfExtractionLimits(
            max_canonical_chars=20_000, max_layout_blocks=10
        ),
    )
    assert accepted.revision_id is not None
    assert len(project.revision_text(accepted.revision_id)) > 100


def test_encrypted_pdf_requires_password_without_attempting_extraction(tmp_path):
    source = tmp_path / "encrypted.pdf"
    protected = fitz.open()
    protected.new_page().insert_text((72, 72), "Must stay encrypted")
    user_password = secrets.token_urlsafe(24)
    owner_password = secrets.token_urlsafe(24)
    protected.save(
        source,
        encryption=cast(Any, fitz).PDF_ENCRYPT_AES_256,
        user_pw=user_password,
        owner_pw=owner_password,
    )
    protected.close()
    project = Project.init(tmp_path / "p")

    result = import_pdf(project, source)

    assert result.status == "PASSWORD_REQUIRED"
    assert "ENCRYPTED_PDF" in result.warnings
    assert result.document_id is None and result.revision_id is None
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_truncated_pdf_is_a_controlled_error_without_partial_revision(tmp_path):
    valid = tmp_path / "valid.pdf"
    pdf(valid, ["Truncation fixture."])
    truncated = tmp_path / "truncated.pdf"
    truncated.write_bytes(valid.read_bytes()[:64])
    project = Project.init(tmp_path / "p")

    with pytest.raises(DocumentError, match="PDF"):
        import_pdf(project, truncated)
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_pdf_page_persistence_failure_removes_new_revision_and_project_files(
    tmp_path, monkeypatch
):
    source = tmp_path / "page-write-failure.pdf"
    pdf(source, ["First page.", "Second page."])
    project = Project.init(tmp_path / "p")
    original_new_page = project._new_page
    created_pages = []

    def fail_on_second_page(*args, **kwargs):
        assert project.db.in_transaction
        created_pages.append(args[1])
        if len(created_pages) == 2:
            raise RuntimeError("injected page persistence failure")
        return original_new_page(*args, **kwargs)

    monkeypatch.setattr(project, "_new_page", fail_on_second_page)
    with pytest.raises(RuntimeError, match="injected page persistence failure"):
        import_pdf(project, source)

    for table in (
        "documents",
        "document_revisions",
        "pages",
        "paragraphs",
        "sentences",
        "extraction_warnings",
        "revision_fts",
    ):
        assert project.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert list((project.hidden / "source").iterdir()) == []
    assert not project.db.in_transaction

    monkeypatch.setattr(project, "_new_page", original_new_page)
    result = import_pdf(project, source)
    assert result.status == "SUCCESS"
    assert result.revision_id is not None
    assert import_pdf(project, source).status == "DUPLICATE"
    assert project.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 1
    assert len(list((project.hidden / "source").iterdir())) == 2


def test_pdf_reprocess_persistence_failure_keeps_existing_document_and_revision(
    tmp_path, monkeypatch
):
    original_source = tmp_path / "original.pdf"
    updated_source = tmp_path / "updated.pdf"
    pdf(original_source, ["Original revision."])
    pdf(updated_source, ["Updated first page.", "Updated second page."])
    project = Project.init(tmp_path / "p")
    original_result = import_pdf(project, original_source)
    assert original_result.document_id is not None
    assert original_result.revision_id is not None
    original_revision = original_result.revision_id
    original_new_page = project._new_page
    created_pages = []

    def fail_on_second_page(*args, **kwargs):
        created_pages.append(args[1])
        if len(created_pages) == 2:
            raise RuntimeError("injected reprocess persistence failure")
        return original_new_page(*args, **kwargs)

    monkeypatch.setattr(project, "_new_page", fail_on_second_page)
    with pytest.raises(RuntimeError, match="injected reprocess persistence failure"):
        reprocess_pdf(project, original_result.document_id, updated_source)

    assert project.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 1
    assert project.db.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 1
    assert project.revision_text(original_revision) == "Original revision."
    assert len(list((project.hidden / "source").iterdir())) == 2

    monkeypatch.setattr(project, "_new_page", original_new_page)
    reprocessed = reprocess_pdf(project, original_result.document_id, updated_source)
    assert reprocessed.status == "SUCCESS"
    assert reprocessed.revision_id is not None
    assert project.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 2
    assert import_pdf(project, original_source).status == "DUPLICATE"


def test_pdf_actions_attachments_and_links_are_not_executed_or_fetched(tmp_path, monkeypatch):
    source = tmp_path / "active-content.pdf"
    active = fitz.open()
    page = active.new_page()
    page.insert_text((72, 72), "Visible source text.")
    page.insert_link(
        {
            "kind": fitz.LINK_URI,
            "from": fitz.Rect(50, 50, 150, 80),
            "uri": "https://example.invalid/research",
        }
    )
    active.embfile_add("payload.txt", b"ATTACHMENT_SENTINEL")
    active.xref_set_key(
        active.pdf_catalog(),
        "OpenAction",
        "<< /S /JavaScript /JS (app.alert\\(1\\)) >>",
    )
    active.save(source)
    active.close()

    def forbidden(*args, **kwargs):
        raise AssertionError("document extraction attempted external execution or networking")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    project = Project.init(tmp_path / "p")

    result = import_pdf(project, source)

    assert result.status == "SUCCESS"
    assert result.revision_id is not None
    canonical = project.revision_text(result.revision_id)
    assert canonical == "Visible source text."
    assert "ATTACHMENT_SENTINEL" not in canonical
    assert "app.alert" not in canonical


def test_pdf_extension_with_non_pdf_bytes_is_not_silently_imported_as_plain_text(tmp_path):
    masquerading = tmp_path / "not-really.pdf"
    masquerading.write_text("This is UTF-8 text, not a PDF.", encoding="utf-8")
    project = Project.init(tmp_path / "p")
    with pytest.raises(DocumentError, match="PDF extension"):
        import_document(project, masquerading)
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_pdf_partial_extraction_records_a_structured_warning(tmp_path, monkeypatch):
    source = tmp_path / "partial.pdf"
    doc = fitz.open()
    doc.new_page()
    page = doc.new_page()
    page.insert_text((72, 72), "Usable second page.")
    doc.save(source)
    project = Project.init(tmp_path / "p")
    original_record_warning = project.record_extraction_warning
    warning_commits = []

    def record_warning_in_import_transaction(revision_id, code, detail=None, *, commit=True):
        assert project.db.in_transaction
        warning_commits.append(commit)
        return original_record_warning(revision_id, code, detail, commit=commit)

    monkeypatch.setattr(project, "record_extraction_warning", record_warning_in_import_transaction)
    result = import_pdf(project, source)
    assert result.status == "SUCCESS_WITH_WARNINGS"
    assert result.revision_id is not None
    assert "PAGE_EXTRACTION_FAILED" in result.warnings
    assert [row["code"] for row in project.extraction_warnings(result.revision_id)] == [
        "PAGE_EXTRACTION_FAILED"
    ]
    assert warning_commits == [False]


def test_two_column_pdf_records_sorted_reading_order_ambiguity(tmp_path):
    source = tmp_path / "two-column.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    column_rows = [
        ("LEFT COLUMN ONE", "RIGHT COLUMN ONE"),
        ("LEFT COLUMN TWO", "RIGHT COLUMN TWO"),
        ("LEFT COLUMN THREE", "RIGHT COLUMN THREE"),
    ]
    for row_index, (left, right) in enumerate(column_rows):
        baseline = 72 + row_index * 24
        page.insert_text((55, baseline), left, fontsize=10)
        page.insert_text((330, baseline), right, fontsize=10)
    doc.save(source)
    doc.close()

    project = Project.init(tmp_path / "p")
    result = import_pdf(project, source)

    assert result.status == "SUCCESS_WITH_WARNINGS"
    assert result.revision_id is not None
    assert result.warnings == ("AMBIGUOUS_READING_ORDER",)
    canonical = project.revision_text(result.revision_id)
    assert canonical.index("RIGHT COLUMN ONE") < canonical.index("LEFT COLUMN TWO")
    assert [row["code"] for row in project.extraction_warnings(result.revision_id)] == [
        "AMBIGUOUS_READING_ORDER"
    ]
    assert project.db.execute(
        "SELECT extraction_status FROM document_revisions WHERE id=?",
        (result.revision_id,),
    ).fetchone()[0] == "SUCCESS_WITH_WARNINGS"


def test_two_column_pdf_with_rows_six_points_apart_records_warning(tmp_path):
    source = tmp_path / "compact-two-column.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for row_index, (left, right) in enumerate(
        (("LEFT ROW A", "RIGHT ROW A"), ("LEFT ROW B", "RIGHT ROW B"))
    ):
        baseline = 72 + row_index * 6
        page.insert_text((55, baseline), left, fontsize=4)
        page.insert_text((330, baseline), right, fontsize=4)
    doc.save(source)
    doc.close()

    project = Project.init(tmp_path / "p")
    result = import_pdf(project, source)

    assert result.status == "SUCCESS_WITH_WARNINGS"
    assert result.revision_id is not None
    assert "AMBIGUOUS_READING_ORDER" in result.warnings
    canonical = project.revision_text(result.revision_id)
    assert canonical.index("RIGHT ROW A") < canonical.index("LEFT ROW B")


def test_indented_single_column_pdf_does_not_get_reading_order_warning(tmp_path):
    source = tmp_path / "indented-single-column.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((55, 72), "A single-column paragraph starts here.", fontsize=10)
    page.insert_text((82, 96), "An indented continuation keeps reading downward.", fontsize=10)
    page.insert_text((55, 120), "The next line returns to the original margin.", fontsize=10)
    doc.save(source)
    doc.close()

    project = Project.init(tmp_path / "p")
    result = import_pdf(project, source)

    assert result.status == "SUCCESS"
    assert result.warnings == ()


def test_import_size_policy_rejects_before_pdf_parsing(tmp_path):
    source = tmp_path / "large.pdf"
    pdf(source, ["Text."])
    project = Project.init(tmp_path / "p")
    with pytest.raises(DocumentError, match="size limit"):
        import_document(project, source, max_bytes=1)


def test_import_size_limit_uses_open_file_metadata_not_a_stale_path_stat(
    tmp_path, monkeypatch
):
    source = tmp_path / "large.pdf"
    pdf(source, ["A PDF larger than the configured limit."])
    project = Project.init(tmp_path / "p")
    original_stat = Path.stat

    def stale_stat(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == source:
            values = list(result)
            values[6] = 0  # Simulate a stale/small pathname size observation.
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "stat", stale_stat)
    with pytest.raises(DocumentError, match="size limit"):
        import_document(project, source, max_bytes=1)
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_document_import_rejects_source_symlinks(tmp_path):
    source = tmp_path / "paper.pdf"
    pdf(source, ["Selected source."])
    link = tmp_path / "linked.pdf"
    try:
        link.symlink_to(source)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    project = Project.init(tmp_path / "p")

    with pytest.raises(DocumentError, match="must not be a symlink"):
        import_document(project, link)
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_document_import_rejects_non_regular_source_paths(tmp_path):
    source_directory = tmp_path / "not-a-file"
    source_directory.mkdir()
    project = Project.init(tmp_path / "p")

    with pytest.raises(DocumentError, match="regular file"):
        import_document(project, source_directory)
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_direct_pdf_and_docx_imports_apply_positive_bounded_file_policy(tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf(pdf_path, ["More than one byte."])
    project = Project.init(tmp_path / "p")
    with pytest.raises(DocumentError, match="size limit"):
        import_pdf(project, pdf_path, max_bytes=1)

    docx_path = tmp_path / "source.docx"
    docx_path.write_bytes(b"PK\x03\x04" + b"x" * 100)
    with pytest.raises(DocumentError, match="size limit"):
        import_document(project, docx_path, max_bytes=1)
    with pytest.raises(DocumentError, match="positive integer"):
        import_pdf(project, pdf_path, max_bytes=0)
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_direct_pdf_import_honors_project_import_size_limit(tmp_path):
    source = tmp_path / "bounded.pdf"
    pdf(source, ["The project policy must cover direct PDF imports."])
    project = Project.init(tmp_path / "p")
    project.set_project_config(ProjectConfig(import_max_bytes=source.stat().st_size - 1))

    with pytest.raises(DocumentError, match="size limit"):
        import_pdf(project, source)

    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 0


def test_pdf_reprocessing_honors_project_import_size_limit(tmp_path):
    source = tmp_path / "bounded-reprocess.pdf"
    pdf(source, ["Reprocessing must not bypass the project import limit."])
    project = Project.init(tmp_path / "p")
    initial = import_pdf(project, source, max_bytes=source.stat().st_size)
    assert initial.document_id is not None and initial.revision_id is not None
    project.set_project_config(ProjectConfig(import_max_bytes=source.stat().st_size - 1))

    with pytest.raises(DocumentError, match="size limit"):
        reprocess_pdf(project, initial.document_id, source)

    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 1


def test_page_crossing_projection_survives_reopen(tmp_path):
    source = tmp_path / "cross.pdf"
    pdf(source, ["End of first page.", "Start of second page."])
    project = Project.init(tmp_path / "p")
    result = import_pdf(project, source)
    assert result.revision_id is not None
    text = project.revision_text(result.revision_id)
    span = project.create_span(
        result.revision_id, text.index("first"), text.index("second") + len("second")
    )
    segments = project.project_span_to_segments(span)
    assert len(segments) == 2
    assert [segment["bbox_status"] for segment in segments] == ["AVAILABLE", "AVAILABLE"]
    assert all(json.loads(segment["bounding_boxes_json"]) for segment in segments)
    evidence = project.textual_evidence(span)
    citation = project.render_citation(evidence)
    assert "pp. 1–2" in citation
    project.close()
    reopened = Project.open(tmp_path / "p")
    assert (
        reopened.span_text(span) == text[text.index("first") : text.index("second") + len("second")]
    )
    assert reopened.render_citation(evidence) == citation


def test_reprocess_keeps_historical_evidence(tmp_path):
    source = tmp_path / "r.pdf"
    pdf(source, ["Historical evidence."])
    project = Project.init(tmp_path / "p")
    first = import_pdf(project, source)
    assert first.revision_id is not None and first.document_id is not None
    span = project.create_span(first.revision_id, 0, 10)
    evidence = project.textual_evidence(span)
    citation = project.render_citation(evidence)
    second = reprocess_pdf(project, first.document_id, source)
    assert second.document_id == first.document_id and second.revision_id != first.revision_id
    assert project.span_text(span) == "Historical" and project.render_citation(evidence) == citation

    repeated = reprocess_pdf(project, first.document_id, source)
    assert repeated.status == "DUPLICATE"
    assert repeated.document_id == second.document_id
    assert repeated.revision_id == second.revision_id
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 2


def test_pdf_paragraphs_and_project_copy(tmp_path):
    source = tmp_path / "paragraphs.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "First paragraph Café.")
    page.insert_text((72, 120), "Second paragraph migración.")
    doc.save(source)
    project = Project.init(tmp_path / "original")
    imported = import_pdf(project, source)
    assert imported.revision_id is not None and imported.document_id is not None
    paragraphs = project.db.execute(
        "SELECT id,start_offset,end_offset,text,page_id FROM paragraphs WHERE revision_id=? ORDER BY start_offset",
        (imported.revision_id,),
    ).fetchall()
    text = project.revision_text(imported.revision_id)
    assert len(paragraphs) == 2
    assert [row["text"] for row in paragraphs] == [
        "First paragraph Café.",
        "Second paragraph migración.",
    ]
    assert all(
        text[row["start_offset"] : row["end_offset"]] == row["text"] and row["page_id"]
        for row in paragraphs
    )
    span = project.create_span(
        imported.revision_id, paragraphs[0]["start_offset"], paragraphs[1]["end_offset"]
    )
    evidence = project.textual_evidence(span)
    segments = project.project_span_to_segments(span)
    citation = project.render_citation(evidence)
    identifiers = (imported.document_id, imported.revision_id, paragraphs[0]["id"], span, evidence)
    project.close()
    shutil.copytree(tmp_path / "original", tmp_path / "copied")
    copied = Project.open(tmp_path / "copied")
    assert copied.revision_text(identifiers[1]) == text
    assert (
        copied.span_text(identifiers[3])
        == text[paragraphs[0]["start_offset"] : paragraphs[1]["end_offset"]]
    )
    assert copied.render_citation(identifiers[4]) == citation and len(
        copied.span_segments(identifiers[3])
    ) == len(segments)


def test_pdf_projection_uses_persisted_paragraph_boxes_and_offsets(tmp_path):
    source = tmp_path / "projection.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((72, 72), "First paragraph Café.")
    page.insert_text((72, 128), "Second paragraph migración.")
    doc.save(source)
    project = Project.init(tmp_path / "project")
    result = import_pdf(project, source)
    assert result.revision_id is not None
    text = project.revision_text(result.revision_id)
    paragraphs = project.db.execute(
        "SELECT * FROM paragraphs WHERE revision_id=? ORDER BY ordinal", (result.revision_id,)
    ).fetchall()
    assert len(paragraphs) == 2
    assert all(text[row["start_offset"] : row["end_offset"]] == row["text"] for row in paragraphs)
    span = project.create_span(
        result.revision_id, paragraphs[0]["start_offset"], paragraphs[1]["end_offset"]
    )
    segments = project.span_segments(span)
    assert len(segments) == 1
    boxes = json.loads(segments[0]["bounding_boxes_json"])
    assert len(boxes) >= 2
    assert segments[0]["bbox_status"] == "AVAILABLE"
    assert segments[0]["bbox_projection_version"] == "pymupdf-char-range-v1"
    page_row = project.db.execute(
        "SELECT width,height FROM pages WHERE id=?", (segments[0]["page_number"],)
    ).fetchone()
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= page_row["width"]
        assert 0 <= y0 < y1 <= page_row["height"]
    evidence = project.textual_evidence(span)
    assert "p. 1" in project.render_citation(evidence)


def test_pdf_span_bboxes_cover_selected_characters_not_entire_paragraph(tmp_path):
    source = tmp_path / "selected-word.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((72, 72), "First paragraph Café.")
    doc.save(source)
    project = Project.init(tmp_path / "project")
    result = import_pdf(project, source)
    assert result.revision_id is not None
    text = project.revision_text(result.revision_id)
    start = text.index("paragraph")
    span = project.create_span(result.revision_id, start, start + len("paragraph"))
    paragraph = project.db.execute(
        "SELECT id,bounding_boxes_json FROM paragraphs WHERE revision_id=?", (result.revision_id,)
    ).fetchone()
    page_id = project.db.execute(
        "SELECT id FROM pages WHERE revision_id=?", (result.revision_id,)
    ).fetchone()["id"]
    project._segment_for_range(
        result.revision_id,
        page_id,
        start,
        start + len("paragraph"),
        json.loads(paragraph["bounding_boxes_json"]),
    )

    segment = project.project_span_to_segments(span)[0]
    boxes = json.loads(segment["bounding_boxes_json"])
    paragraph_box = json.loads(paragraph["bounding_boxes_json"])[0]

    assert segment["bbox_status"] == "AVAILABLE"
    assert len(boxes) == 1
    assert boxes[0][0] > paragraph_box[0]
    assert boxes[0][2] < paragraph_box[2]


def test_pdf_char_geometry_preserves_wrapped_line_locations(tmp_path):
    source = tmp_path / "wrapped-lines.pdf"
    doc = fitz.open()
    page = doc.new_page(width=220, height=300)
    page.insert_textbox(
        fitz.Rect(50, 50, 170, 130),
        "A long research paragraph with precise character geometry across wrapped lines and a final marker.",
        fontsize=11,
    )
    doc.save(source)
    project = Project.init(tmp_path / "project")
    result = import_pdf(project, source)
    assert result.revision_id is not None
    assert result.warnings == ()
    text = project.revision_text(result.revision_id)
    selected = "final marker"
    start = text.index(selected)
    span = project.create_span(result.revision_id, start, start + len(selected))

    segment = project.project_span_to_segments(span)[0]
    boxes = json.loads(segment["bounding_boxes_json"])
    paragraph = project.db.execute(
        "SELECT bounding_boxes_json FROM paragraphs WHERE revision_id=?", (result.revision_id,)
    ).fetchone()
    paragraph_box = json.loads(paragraph["bounding_boxes_json"])[0]

    assert segment["bbox_status"] == "AVAILABLE"
    assert boxes
    assert all(box[1] > paragraph_box[1] for box in boxes)
    page_record = project.db.execute(
        "SELECT width,height FROM pages WHERE revision_id=?", (result.revision_id,)
    ).fetchone()
    assert all(0 <= x0 < x1 <= page_record["width"] for x0, _, x1, _ in boxes)
    assert all(0 <= y0 < y1 <= page_record["height"] for _, y0, _, y1 in boxes)
