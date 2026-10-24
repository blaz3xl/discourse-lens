from __future__ import annotations

import pytest

from discourselens import Project
from discourselens.errors import ProjectError, ScopeViolationError


def test_scope_revisions_page_returns_complete_ordered_pages_read_only(tmp_path):
    root = tmp_path / "scope-page-project"
    with Project.init(root) as project:
        revisions = [
            project.import_text(f"source-{index}.txt", f"Source {index}.")[1]
            for index in range(3)
        ]
        scope_id = project.snapshot(revisions)

    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    project = Project.open(root, read_only=True)
    try:
        first = project.scope_revisions_page(scope_id, offset=0, limit=2)
        second = project.scope_revisions_page(scope_id, offset=2, limit=2)
        beyond_end = project.scope_revisions_page(scope_id, offset=10, limit=2)
    finally:
        project.close()

    assert first == {
        "scope_id": scope_id,
        "offset": 0,
        "limit": 2,
        "revision_count": 3,
        "revision_ids": revisions[:2],
        "has_more": True,
    }
    assert second["revision_ids"] == revisions[2:]
    assert second["revision_count"] == 3
    assert second["has_more"] is False
    assert beyond_end["revision_ids"] == []
    assert beyond_end["revision_count"] == 3
    assert beyond_end["has_more"] is False
    assert database.read_bytes() == before


@pytest.mark.parametrize(
    ("scope_id", "offset", "limit"),
    [
        ("scope", -1, 1),
        ("scope", 0, 0),
        ("scope", 0, 101),
        ("scope", True, 1),
        ("scope", 0, True),
        ("scope", 0x7FFF_FFFF_FFFF_FFFF + 1, 1),
        ("", 0, 1),
    ],
)
def test_scope_revisions_page_rejects_invalid_bounds(tmp_path, scope_id, offset, limit):
    root = tmp_path / "invalid-scope-page-project"
    Project.init(root).close()
    project = Project.open(root, read_only=True)
    try:
        with pytest.raises(ProjectError, match="scope revisions page"):
            project.scope_revisions_page(scope_id, offset=offset, limit=limit)
    finally:
        project.close()


def test_scope_revisions_page_preserves_unknown_scope_error(tmp_path):
    root = tmp_path / "unknown-scope-page-project"
    Project.init(root).close()
    project = Project.open(root, read_only=True)
    try:
        with pytest.raises(ScopeViolationError, match="unknown scope"):
            project.scope_revisions_page("unknown-scope")
    finally:
        project.close()
