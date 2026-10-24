import hashlib

import fitz
import pytest

from discourselens import Project
from discourselens.errors import DocumentError


def test_document_revision_summary_returns_path_free_revision_metadata(tmp_path):
    source_path = tmp_path / "private-document.pdf"
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "revision one private text")
    pdf.new_page().insert_text((72, 72), "revision one second page")
    pdf.save(source_path)
    pdf.close()

    with Project.init(tmp_path / "project") as project:
        imported = project.import_document(source_path)
        assert imported.revision_id is not None
        assert imported.document_id is not None
        first_revision_id = imported.revision_id

        second_text = "revision two private text"
        _, second_revision_id = project.import_text(
            "ignored later filename.txt",
            second_text,
            document_id=imported.document_id,
            ocr_language="eng",
            ocr_dpi=200,
        )

        changes_before = project.db.total_changes
        first = project.document_revision_summary(first_revision_id)
        second = project.document_revision_summary(second_revision_id)
        changes_after = project.db.total_changes

        expected_keys = {
            "revision_id",
            "document_id",
            "display_name",
            "source_hash",
            "canonical_hash",
            "extractor",
            "mime_type",
            "source_size",
            "created_at",
            "ocr_language",
            "ocr_dpi",
            "page_count",
            "paragraph_count",
        }
        assert set(first) == expected_keys
        assert first["revision_id"] == first_revision_id
        assert first["document_id"] == imported.document_id
        assert first["display_name"] == "private-document.pdf"
        assert first["source_hash"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert first["canonical_hash"] == hashlib.sha256(
            project.revision_text(first_revision_id).encode("utf-8")
        ).hexdigest()
        assert str(first["extractor"]).startswith("pymupdf@")
        assert first["mime_type"] == "application/pdf"
        assert first["source_size"] == source_path.stat().st_size
        assert first["created_at"]
        assert first["ocr_language"] is None
        assert first["ocr_dpi"] is None
        assert first["page_count"] == 2
        assert first["paragraph_count"] == project.db.execute(
            "SELECT COUNT(*) FROM paragraphs WHERE revision_id=?", (first_revision_id,)
        ).fetchone()[0]
        assert isinstance(first["paragraph_count"], int) and first["paragraph_count"] > 0

        assert set(second) == expected_keys
        assert second["revision_id"] == second_revision_id
        assert second["page_count"] == 0
        assert second["paragraph_count"] == 1
        assert second["ocr_language"] == "eng"
        assert second["ocr_dpi"] == 200
        assert changes_after == changes_before

        result_text = repr(first) + repr(second)
        assert str(source_path) not in result_text
        assert "revision one private text" not in result_text
        assert "revision one second page" not in result_text
        assert second_text not in result_text
        assert "original_path" not in result_text
        assert "text_path" not in result_text


def test_document_revision_summary_rejects_unknown_revision(tmp_path):
    with (
        Project.init(tmp_path / "project") as project,
        pytest.raises(DocumentError, match="unknown revision"),
    ):
        project.document_revision_summary("missing-revision")


@pytest.mark.parametrize(
    ("name", "expected_display_name"),
    [
        ("/private/research/notes.txt", "notes.txt"),
        (r"C:\Users\private\report.txt", "report.txt"),
    ],
)
def test_path_like_import_names_are_reduced_to_basenames(tmp_path, name, expected_display_name):
    with Project.init(tmp_path / "project") as project:
        document_id, revision_id = project.import_text(name, "Private body text.")
        stored = project.db.execute(
            "SELECT display_name FROM documents WHERE id=?", (document_id,)
        ).fetchone()
        assert stored["display_name"] == expected_display_name

        page = project.documents_page()
        items = page["items"]
        assert isinstance(items, list)
        item = next(row for row in items if row["document_id"] == document_id)
        summary = project.document_revision_summary(revision_id)

        assert item["display_name"] == expected_display_name
        assert summary["display_name"] == expected_display_name
        assert name not in repr(item)
        assert name not in repr(summary)
        assert str(tmp_path) not in repr(item)
        assert str(tmp_path) not in repr(summary)

        # Older project files may contain path-like labels, so public readers
        # also strip paths when presenting stored values.
        project.db.execute(
            "UPDATE documents SET display_name=? WHERE id=?", (name, document_id)
        )
        project.db.commit()
        legacy_page = project.documents_page()
        legacy_items = legacy_page["items"]
        assert isinstance(legacy_items, list)
        legacy_item = legacy_items[0]
        legacy_summary = project.document_revision_summary(revision_id)
        assert legacy_item["display_name"] == expected_display_name
        assert legacy_summary["display_name"] == expected_display_name
        assert name not in repr(legacy_item)
        assert name not in repr(legacy_summary)


@pytest.mark.parametrize("name", ["", ".", "..", "/", "C:"])
def test_import_text_rejects_empty_or_dot_names(tmp_path, name):
    with (
        Project.init(tmp_path / "project") as project,
        pytest.raises(DocumentError, match="non-empty filename"),
    ):
        project.import_text(name, "A body.")
