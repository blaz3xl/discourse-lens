from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import DocumentError


def test_analysis_run_pages_match_legacy_order_and_counts(tmp_path, monkeypatch):
    with Project.init(tmp_path / "project") as project:
        _, first_revision = project.import_text("first.txt", "First source.")
        _, second_revision = project.import_text("second.txt", "Second source.")
        first_scope = project.snapshot([first_revision])
        second_scope = project.snapshot([second_revision])

        run_ids = iter(("run-z", "run-a", "run-m"))
        with monkeypatch.context() as patch:
            patch.setattr("discourselens.project.uid", lambda: next(run_ids))
            patch.setattr("discourselens.project.now", lambda: "2026-01-01T00:00:00+00:00")
            first_run = project.create_run(first_scope, {"profile": "first"})
            second_run = project.create_run(second_scope, {"profile": "second"})
            third_run = project.create_run(first_scope, {"profile": "third"})

        project.create_finding(first_run, "fixture", "1.0.0", "first", "A finding.")
        legacy_items = project.list_analysis_runs()
        first_page = project.analysis_runs_page(offset=0, limit=2)
        second_page = project.analysis_runs_page(offset=2, limit=2)

        assert [item["id"] for item in legacy_items] == ["run-z", "run-m", "run-a"]
        assert first_page == {
            "items": legacy_items[:2],
            "offset": 0,
            "limit": 2,
            "total": 3,
            "has_more": True,
        }
        assert second_page == {
            "items": legacy_items[2:],
            "offset": 2,
            "limit": 2,
            "total": 3,
            "has_more": False,
        }
        first_items = cast(list[dict[str, object]], first_page["items"])
        assert first_items[0]["finding_count"] == 1
        assert first_items[1]["finding_count"] == 0
        assert [item["scope_id"] for item in legacy_items] == [
            first_scope,
            first_scope,
            second_scope,
        ]
        assert "config_json" not in legacy_items[0]
        assert {item["id"] for item in legacy_items} == {
            first_run,
            second_run,
            third_run,
        }


def test_analysis_run_page_empty_and_maximum_offset(tmp_path):
    with Project.init(tmp_path / "empty-project") as project:
        page = project.analysis_runs_page(offset=0x7FFF_FFFF_FFFF_FFFF, limit=1)

    assert page == {
        "items": [],
        "offset": 0x7FFF_FFFF_FFFF_FFFF,
        "limit": 1,
        "total": 0,
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
def test_analysis_run_page_rejects_invalid_bounds(tmp_path, offset, limit):
    with (
        Project.init(tmp_path / f"invalid-page-{offset}-{limit}") as project,
        pytest.raises(DocumentError, match="analysis runs page"),
    ):
        project.analysis_runs_page(offset=offset, limit=limit)
