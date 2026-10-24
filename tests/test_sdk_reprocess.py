import fitz
import pytest

from discourselens import Project
from discourselens.errors import ProjectError


def _pdf(path):
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Revision extraction fixture.")
    document.save(path)
    document.close()


def test_sdk_reprocess_creates_equivalent_new_revision(tmp_path):
    source = tmp_path / "paper.pdf"
    _pdf(source)

    with Project.init(tmp_path / "project") as project:
        imported = project.import_document(source)
        assert imported.status == "SUCCESS"
        assert imported.document_id is not None
        assert imported.revision_id is not None

        reprocessed = project.reprocess_document(imported.document_id, source)

        assert reprocessed.status == "SUCCESS"
        assert reprocessed.document_id == imported.document_id
        assert reprocessed.revision_id is not None
        assert reprocessed.revision_id != imported.revision_id
        revisions = project.db.execute(
            "SELECT id,source_hash,canonical_hash,extractor FROM document_revisions "
            "WHERE document_id=? ORDER BY created_at,id",
            (imported.document_id,),
        ).fetchall()

    assert len(revisions) == 2
    assert revisions[0]["source_hash"] == revisions[1]["source_hash"]
    assert revisions[0]["canonical_hash"] == revisions[1]["canonical_hash"]
    assert revisions[0]["extractor"] != revisions[1]["extractor"]
    assert revisions[1]["extractor"].endswith(".reprocessed")


def test_sdk_reprocess_rejects_read_only_project(tmp_path):
    source = tmp_path / "paper.pdf"
    _pdf(source)
    root = tmp_path / "project"

    with Project.init(root) as project:
        imported = project.import_document(source)
        assert imported.document_id is not None

    with (
        Project.open(root, read_only=True) as project,
        pytest.raises(ProjectError, match="read-only"),
    ):
        project.reprocess_document(imported.document_id, source)
