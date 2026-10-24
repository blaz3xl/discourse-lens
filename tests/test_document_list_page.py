from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import DocumentError


def test_documents_page_orders_rows_and_reports_latest_revision_without_paths(
    tmp_path, monkeypatch
):
    with Project.init(tmp_path / "document-pages") as project:
        ids = iter(
            (
                "doc-zeta",
                "rev-first",
                "rev-later-a",
                "rev-later-z",
                "doc-alpha-z",
                "rev-alpha-z",
                "doc-alpha-a",
                "rev-alpha-a",
            )
        )
        timestamps = iter(
            (
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T01:00:00+00:00",
                "2026-01-02T00:00:00+00:00",
                "2026-01-02T00:00:00+00:00",
                "2026-01-03T00:00:00+00:00",
                "2026-01-03T01:00:00+00:00",
                "2026-01-03T00:00:00+00:00",
                "2026-01-03T01:00:00+00:00",
            )
        )
        with monkeypatch.context() as patch:
            patch.setattr("discourselens.project.uid", lambda: next(ids))
            patch.setattr("discourselens.project.now", lambda: next(timestamps))
            document_id, first_revision = project.import_text(
                "Zulu", "first version", create_paragraphs=False
            )
            project.import_text(
                "unused display name",
                "later revision",
                document_id=document_id,
                create_paragraphs=False,
                extraction_status="PARTIAL",
                mime_type="application/pdf",
            )
            _, latest_revision = project.import_text(
                "unused display name",
                "latest revision",
                document_id=document_id,
                create_paragraphs=False,
                extraction_status="FAILED",
                mime_type="application/octet-stream",
            )
            project.import_text("Alpha", "alpha z", create_paragraphs=False)
            project.import_text("Alpha", "alpha a", create_paragraphs=False)

        project.db.execute(
            "INSERT INTO documents(id,display_name,created_at) VALUES(?,?,?)",
            ("doc-empty", "Empty", "2026-01-04T00:00:00+00:00"),
        )
        project.db.commit()
        expected_keys = {
            "document_id",
            "display_name",
            "created_at",
            "revision_count",
            "latest_revision_id",
            "latest_revision_created_at",
            "latest_extraction_status",
            "latest_mime_type",
        }
        changes_before = project.db.total_changes
        full_page = project.documents_page(limit=10)
        first_page = project.documents_page(offset=0, limit=2)
        second_page = project.documents_page(offset=2, limit=2)
        changes_after = project.db.total_changes

        full_items = cast(list[dict[str, object]], full_page["items"])
        assert [item["display_name"] for item in full_items] == [
            "Alpha",
            "Alpha",
            "Empty",
            "Zulu",
        ]
        assert [item["document_id"] for item in full_items[:2]] == [
            "doc-alpha-a",
            "doc-alpha-z",
        ]
        assert all(set(item) == expected_keys for item in full_items)
        zulu = full_items[-1]
        assert zulu == {
            "document_id": document_id,
            "display_name": "Zulu",
            "created_at": "2026-01-01T00:00:00+00:00",
            "revision_count": 3,
            "latest_revision_id": latest_revision,
            "latest_revision_created_at": "2026-01-02T00:00:00+00:00",
            "latest_extraction_status": "FAILED",
            "latest_mime_type": "application/octet-stream",
        }
        empty_revision = full_items[2]
        assert empty_revision["revision_count"] == 0
        assert empty_revision["latest_revision_id"] is None
        assert empty_revision["latest_revision_created_at"] is None
        assert empty_revision["latest_extraction_status"] is None
        assert empty_revision["latest_mime_type"] is None
        assert first_page == {
            "items": full_items[:2],
            "offset": 0,
            "limit": 2,
            "total": 4,
            "has_more": True,
        }
        assert second_page == {
            "items": full_items[2:],
            "offset": 2,
            "limit": 2,
            "total": 4,
            "has_more": False,
        }
        assert first_revision != latest_revision
        assert changes_after == changes_before


def test_documents_page_handles_empty_and_out_of_range_offsets(tmp_path):
    with Project.init(tmp_path / "empty-document-pages") as project:
        assert project.documents_page(offset=0, limit=2) == {
            "items": [],
            "offset": 0,
            "limit": 2,
            "total": 0,
            "has_more": False,
        }
        assert project.documents_page(offset=0x7FFF_FFFF_FFFF_FFFF, limit=1) == {
            "items": [],
            "offset": 0x7FFF_FFFF_FFFF_FFFF,
            "limit": 1,
            "total": 0,
            "has_more": False,
        }

        project.import_text("Only document", "Only revision.", create_paragraphs=False)
        assert project.documents_page(offset=1, limit=1) == {
            "items": [],
            "offset": 1,
            "limit": 1,
            "total": 1,
            "has_more": False,
        }


@pytest.mark.parametrize(
    ("offset", "limit"),
    [
        (-1, 1),
        (True, 1),
        (1.5, 1),
        (0x8000_0000_0000_0000, 1),
        (0, 0),
        (0, 1_001),
        (0, True),
        (0, 1.5),
    ],
)
def test_documents_page_rejects_invalid_integer_bounds(tmp_path, offset, limit):
    with (
        Project.init(tmp_path / f"invalid-document-page-{offset}-{limit}") as project,
        pytest.raises(DocumentError, match="documents page"),
    ):
        project.documents_page(offset=offset, limit=limit)
