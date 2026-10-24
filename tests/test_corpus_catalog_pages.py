from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import DocumentError


def test_corpora_pages_match_legacy_rows_and_have_bounded_metadata(tmp_path):
    with Project.init(tmp_path / "corpora-pages") as project:
        _, first_revision = project.import_text("first.txt", "First source.")
        _, second_revision = project.import_text("second.txt", "Second source.")
        project.create_corpus("Zulu")
        alpha_id = project.create_corpus("Alpha")
        middle_id = project.create_corpus("Middle")
        project.add_revision_to_corpus(alpha_id, first_revision)
        project.add_revision_to_corpus(middle_id, second_revision)

        legacy = [dict(row) for row in project.list_corpora()]
        first_page = project.corpora_page(offset=0, limit=2)
        second_page = project.corpora_page(offset=2, limit=2)

        assert first_page == {
            "items": legacy[:2],
            "offset": 0,
            "limit": 2,
            "total": 3,
            "has_more": True,
        }
        assert second_page == {
            "items": legacy[2:],
            "offset": 2,
            "limit": 2,
            "total": 3,
            "has_more": False,
        }
        assert [item["name"] for item in cast(list[dict[str, object]], first_page["items"])] == [
            "Alpha",
            "Middle",
        ]
        assert [item["revision_count"] for item in legacy] == [1, 1, 0]


def test_subcorpora_pages_filter_parent_and_use_stable_order_without_writes(tmp_path):
    with Project.init(tmp_path / "subcorpora-pages") as project:
        _, first_revision = project.import_text("first.txt", "First source.")
        _, second_revision = project.import_text("second.txt", "Second source.")
        first_parent = project.create_corpus("First parent")
        second_parent = project.create_corpus("Second parent")
        project.add_revision_to_corpus(first_parent, first_revision)
        project.add_revision_to_corpus(second_parent, second_revision)
        project.create_subcorpus(first_parent, "Shared", [first_revision])
        project.create_subcorpus(first_parent, "Alpha", [first_revision])
        project.create_subcorpus(second_parent, "Shared", [second_revision])
        project.create_subcorpus(second_parent, "Beta", [second_revision])

        expected_all = sorted(
            (dict(row) for row in project.list_subcorpora()),
            key=lambda row: (row["name"], row["id"]),
        )
        expected_first_parent = sorted(
            (dict(row) for row in project.list_subcorpora(first_parent)),
            key=lambda row: (row["name"], row["id"]),
        )
        changes_before = project.db.total_changes
        first_page = project.subcorpora_page(offset=0, limit=2)
        second_page = project.subcorpora_page(offset=2, limit=2)
        filtered_page = project.subcorpora_page(first_parent, limit=10)
        empty_filter = project.subcorpora_page("", limit=2)
        changes_after = project.db.total_changes

        assert first_page["items"] == expected_all[:2]
        assert second_page["items"] == expected_all[2:]
        assert first_page["total"] == second_page["total"] == 4
        assert first_page["has_more"] is True
        assert second_page["has_more"] is False
        assert filtered_page["items"] == expected_first_parent
        assert filtered_page["total"] == 2
        assert filtered_page["has_more"] is False
        assert empty_filter == {
            "items": expected_all[:2],
            "offset": 0,
            "limit": 2,
            "total": 4,
            "has_more": True,
        }
        assert changes_after == changes_before


@pytest.mark.parametrize("method_name", ["corpora_page", "subcorpora_page"])
def test_catalog_pages_handle_empty_and_out_of_range_offsets(tmp_path, method_name):
    with Project.init(tmp_path / f"empty-{method_name}") as project:
        method = getattr(project, method_name)
        assert method(offset=0, limit=2) == {
            "items": [],
            "offset": 0,
            "limit": 2,
            "total": 0,
            "has_more": False,
        }
        assert method(offset=1, limit=2) == {
            "items": [],
            "offset": 1,
            "limit": 2,
            "total": 0,
            "has_more": False,
        }
        assert method(offset=0x7FFF_FFFF_FFFF_FFFF, limit=1) == {
            "items": [],
            "offset": 0x7FFF_FFFF_FFFF_FFFF,
            "limit": 1,
            "total": 0,
            "has_more": False,
        }


@pytest.mark.parametrize("method_name", ["corpora_page", "subcorpora_page"])
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
def test_catalog_pages_reject_invalid_integer_bounds(tmp_path, method_name, offset, limit):
    with (
        Project.init(tmp_path / f"invalid-{method_name}-{offset}-{limit}") as project,
        pytest.raises(
            DocumentError,
            match=f"{method_name.removesuffix('_page').replace('_', ' ')} page",
        ),
    ):
        getattr(project, method_name)(offset=offset, limit=limit)
