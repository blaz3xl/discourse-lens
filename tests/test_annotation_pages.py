from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import DocumentError, ScopeViolationError


def _coded_project(project):
    _, first_revision = project.import_text("first.txt", "Alpha bravo charlie.")
    _, second_revision = project.import_text("second.txt", "Delta echo foxtrot.")
    codebook = project.create_codebook("analysis")
    code = project.create_code(codebook, "observation")
    for revision_id, text, phrases in (
        (first_revision, "Alpha bravo charlie.", ("Alpha", "bravo", "charlie")),
        (second_revision, "Delta echo foxtrot.", ("Delta", "echo")),
    ):
        for phrase in phrases:
            start = text.index(phrase)
            span_id = project.create_span(revision_id, start, start + len(phrase))
            project.annotate_span(code, span_id, f"Code for {phrase}.")
    return first_revision, second_revision


def test_annotation_pages_are_bounded_stable_and_read_only(tmp_path):
    with Project.init(tmp_path / "project") as project:
        first_revision, second_revision = _coded_project(project)
        revision_ids = [second_revision, first_revision]
        legacy_scope = project.snapshot(revision_ids)
        expected = [dict(row) for row in project.annotations_for_scope(legacy_scope)]
        before_snapshot_count = project.db.execute(
            "SELECT COUNT(*) FROM scope_snapshots"
        ).fetchone()[0]
        before_changes = project.db.total_changes

        first_page = project.annotation_pages_for_revisions(revision_ids, offset=0, limit=2)
        middle_page = project.annotation_pages_for_revisions(revision_ids, offset=2, limit=2)
        last_page = project.annotation_pages_for_revisions(revision_ids, offset=4, limit=2)

        assert first_page == {
            "items": expected[:2],
            "revision_count": len(revision_ids),
            "offset": 0,
            "limit": 2,
            "total": 5,
            "has_more": True,
        }
        assert middle_page == {
            "items": expected[2:4],
            "revision_count": len(revision_ids),
            "offset": 2,
            "limit": 2,
            "total": 5,
            "has_more": True,
        }
        assert last_page == {
            "items": expected[4:],
            "revision_count": len(revision_ids),
            "offset": 4,
            "limit": 2,
            "total": 5,
            "has_more": False,
        }
        first_items = cast(list[dict[str, object]], first_page["items"])
        middle_items = cast(list[dict[str, object]], middle_page["items"])
        last_items = cast(list[dict[str, object]], last_page["items"])
        assert [item["id"] for item in first_items + middle_items + last_items] == [
            item["id"] for item in expected
        ]
        assert set(first_items[0]) == set(expected[0])
        assert project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == (
            before_snapshot_count
        )
        assert project.db.total_changes == before_changes


@pytest.mark.parametrize(
    ("revision_ids", "message"),
    [([], "non-empty"), (["missing-revision"], "unknown document revision")],
)
def test_annotation_pages_reject_empty_or_unknown_revision_ids(tmp_path, revision_ids, message):
    with Project.init(tmp_path / f"invalid-scope-{message.split()[0]}") as project:
        project.import_text("source.txt", "A source.")
        before_snapshots = project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
        before_changes = project.db.total_changes

        with pytest.raises(ScopeViolationError, match=message):
            project.annotation_pages_for_revisions(revision_ids)

        assert project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == (
            before_snapshots
        )
        assert project.db.total_changes == before_changes


def test_annotation_pages_reject_duplicate_revisions(tmp_path):
    with Project.init(tmp_path / "duplicate-scope") as project:
        _, revision = project.import_text("source.txt", "A source.")
        with pytest.raises(ScopeViolationError, match="duplicate document revisions"):
            project.annotation_pages_for_revisions([revision, revision])


@pytest.mark.parametrize(
    ("offset", "limit"),
    [
        (-1, 1),
        (True, 1),
        (0, 0),
        (0, 101),
        (0, 1.5),
        (0x8000_0000_0000_0000, 1),
    ],
)
def test_annotation_pages_reject_invalid_bounds(tmp_path, offset, limit):
    with (
        Project.init(tmp_path / f"invalid-bounds-{offset}-{limit}") as project,
        pytest.raises(DocumentError, match="annotation page"),
    ):
        project.annotation_pages_for_revisions(["revision"], offset=offset, limit=limit)


def test_annotation_pages_accept_max_sqlite_offset(tmp_path):
    with Project.init(tmp_path / "maximum-offset") as project:
        _, revision = project.import_text("source.txt", "A source.")
        page = project.annotation_pages_for_revisions(
            [revision], offset=0x7FFF_FFFF_FFFF_FFFF, limit=100
        )

        assert page == {
            "items": [],
            "revision_count": 1,
            "offset": 0x7FFF_FFFF_FFFF_FFFF,
            "limit": 100,
            "total": 0,
            "has_more": False,
        }
