import json
from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import DocumentError


def test_codebook_pages_handle_empty_and_out_of_range_offsets(tmp_path):
    with Project.init(tmp_path / "empty-codebook-pages") as project:
        assert project.codebook_pages(offset=0, limit=2) == {
            "items": [],
            "offset": 0,
            "limit": 2,
            "total": 0,
            "has_more": False,
        }
        assert project.codebook_pages(offset=0x7FFF_FFFF_FFFF_FFFF, limit=1) == {
            "items": [],
            "offset": 0x7FFF_FFFF_FFFF_FFFF,
            "limit": 1,
            "total": 0,
            "has_more": False,
        }

        empty_book = project.create_codebook("Empty")
        assert project.codes_for_codebook_page(empty_book, offset=0, limit=2) == {
            "items": [],
            "offset": 0,
            "limit": 2,
            "total": 0,
            "has_more": False,
        }
        assert project.codes_for_codebook_page(empty_book, offset=1, limit=1) == {
            "items": [],
            "offset": 1,
            "limit": 1,
            "total": 0,
            "has_more": False,
        }


def test_codebook_and_code_pages_are_stable_and_keep_hierarchy(tmp_path):
    with Project.init(tmp_path / "codebook-pages") as project:
        zeta_book = project.create_codebook("Zeta", "Later in sort order")
        alpha_book = project.create_codebook("Alpha", "Earlier in sort order")
        other_code = project.create_code(zeta_book, "Unrelated")
        parent = project.create_code(alpha_book, "Parent")
        child = project.create_code(alpha_book, "Child", parent_code_id=parent)
        project.create_code(alpha_book, "Sibling")

        all_books = project.codebook_pages(limit=10)
        book_items = cast(list[dict[str, object]], all_books["items"])
        assert [row["id"] for row in book_items] == [alpha_book, zeta_book]
        assert book_items == project.codebook_pages(limit=10)["items"]
        assert project.codebook_pages(offset=0, limit=1) == {
            "items": book_items[:1],
            "offset": 0,
            "limit": 1,
            "total": 2,
            "has_more": True,
        }
        assert project.codebook_pages(offset=1, limit=1) == {
            "items": book_items[1:],
            "offset": 1,
            "limit": 1,
            "total": 2,
            "has_more": False,
        }

        full_page = project.codes_for_codebook_page(alpha_book, limit=10)
        full_items = cast(list[dict[str, object]], full_page["items"])
        assert [row["label"] for row in full_items] == ["Child", "Parent", "Sibling"]
        assert next(row for row in full_items if row["id"] == child)["parent_code_id"] == parent
        assert next(row for row in full_items if row["id"] == parent)["parent_code_id"] is None
        assert all(row["codebook_id"] == alpha_book for row in full_items)
        assert other_code not in {row["id"] for row in full_items}
        assert all("created_at" in row for row in full_items)
        assert project.codes_for_codebook_page(alpha_book, limit=10) == full_page
        assert project.codes_for_codebook_page(alpha_book, offset=0, limit=2) == {
            "items": full_items[:2],
            "offset": 0,
            "limit": 2,
            "total": 3,
            "has_more": True,
        }
        assert project.codes_for_codebook_page(alpha_book, offset=2, limit=2) == {
            "items": full_items[2:],
            "offset": 2,
            "limit": 2,
            "total": 3,
            "has_more": False,
        }
        # Read pages are JSON-safe, including nullable descriptions and parent links.
        json.dumps(all_books)
        json.dumps(full_page)


def test_codes_for_codebook_page_rejects_unknown_codebook(tmp_path):
    with (
        Project.init(tmp_path / "unknown-codebook-page") as project,
        pytest.raises(DocumentError, match="unknown codebook"),
    ):
        project.codes_for_codebook_page("missing")


@pytest.mark.parametrize(
    ("offset", "limit"),
    [
        (-1, 1),
        (True, 1),
        (1.5, 1),
        (0x8000_0000_0000_0000, 1),
        (0, 0),
        (0, 101),
        (0, True),
        (0, 1.5),
    ],
)
def test_codebook_pages_reject_invalid_integer_bounds(tmp_path, offset, limit):
    with Project.init(tmp_path / f"invalid-codebook-page-{offset}-{limit}") as project:
        codebook_id = project.create_codebook("Analysis")
        with pytest.raises(DocumentError, match="page requires"):
            project.codebook_pages(offset=offset, limit=limit)
        with pytest.raises(DocumentError, match="page requires"):
            project.codes_for_codebook_page(codebook_id, offset=offset, limit=limit)
