from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def test_scope_revisions_paginates_complete_snapshot_as_json_without_writes(tmp_path):
    root = tmp_path / "scope-revisions-project"
    project = Project.init(root)
    revision_ids = [
        project.import_text(f"source-{index}.txt", f"Source number {index}.")[1]
        for index in range(3)
    ]
    scope_id = project.snapshot(revision_ids)
    project.close()

    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    runner = CliRunner()
    first_result = runner.invoke(
        app,
        [
            "corpus",
            "scope-revisions",
            str(root),
            scope_id,
            "--offset",
            "0",
            "--limit",
            "2",
        ],
    )
    second_result = runner.invoke(
        app,
        [
            "corpus",
            "scope-revisions",
            str(root),
            scope_id,
            "--offset",
            "2",
            "--limit",
            "2",
        ],
    )
    beyond_end_result = runner.invoke(
        app,
        ["corpus", "scope-revisions", str(root), scope_id, "--offset", "10"],
    )

    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    assert beyond_end_result.exit_code == 0, beyond_end_result.output
    first = json.loads(first_result.output)
    second = json.loads(second_result.output)
    beyond_end = json.loads(beyond_end_result.output)
    assert first == {
        "scope_id": scope_id,
        "offset": 0,
        "limit": 2,
        "revision_count": 3,
        "revision_ids": revision_ids[:2],
        "has_more": True,
    }
    assert second["revision_ids"] == revision_ids[2:]
    assert second["revision_count"] == 3
    assert second["has_more"] is False
    assert beyond_end["revision_ids"] == []
    assert beyond_end["revision_count"] == 3
    assert beyond_end["has_more"] is False
    assert database.read_bytes() == before


@pytest.mark.parametrize(
    ("option", "value"),
    [("--offset", "-1"), ("--limit", "0"), ("--limit", "101")],
)
def test_scope_revisions_rejects_invalid_page_bounds(tmp_path, option, value):
    root = tmp_path / "invalid-scope-page-project"
    Project.init(root).close()

    result = CliRunner().invoke(
        app,
        ["corpus", "scope-revisions", str(root), "missing-scope", option, value],
    )

    assert result.exit_code == 2
    assert option in result.output


def test_scope_revisions_unknown_scope_error_does_not_disclose_project_path(tmp_path):
    root = tmp_path / "unknown-scope-revisions-project"
    Project.init(root).close()

    result = CliRunner().invoke(
        app, ["corpus", "scope-revisions", str(root), "unknown-scope"]
    )

    assert result.exit_code == 2
    assert "unknown scope" in result.output.lower()
    assert str(root) not in result.output
