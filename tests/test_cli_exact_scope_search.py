from __future__ import annotations

import json
import sqlite3
from typing import cast

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app


def test_exact_scope_search_matches_core_pages_and_keeps_snapshot_membership_read_only(
    tmp_path, monkeypatch
):
    root = tmp_path / "exact-scope-search"
    project = Project.init(root)
    _, first_revision = project.import_text(
        "first.txt", "Migration in first paragraph.\n\nMigration in second paragraph."
    )
    _, second_revision = project.import_text(
        "second.txt", "Migration in another paragraph.\n\nMigration in the last paragraph."
    )
    scope_id = project.snapshot([first_revision, second_revision])
    _, later_revision = project.import_text("later.txt", "Migration is outside this frozen scope.")
    project.close()

    with Project.open(root, read_only=True) as expected_project:
        expected_first = expected_project.search_exact_page("Migration", scope_id, limit=2)
        expected_second = expected_project.search_exact_page(
            "Migration", scope_id, offset=2, limit=2
        )

    opened_read_only: list[bool] = []
    original_open = Project.open

    def tracked_open(path, *, read_only: bool = False):
        opened_read_only.append(read_only)
        return original_open(path, read_only=read_only)

    monkeypatch.setattr(cli_module.Project, "open", staticmethod(tracked_open))
    database = root / ".discourselens" / "project.sqlite"
    database_before = database.read_bytes()
    runner = CliRunner()

    def invoke_page(offset: int):
        return runner.invoke(
            app,
            [
                "search",
                "exact-scope",
                str(root),
                scope_id,
                "Migration",
                "--format",
                "json",
                "--offset",
                str(offset),
                "--limit",
                "2",
            ],
        )

    first_result = invoke_page(0)
    second_result = invoke_page(2)

    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    first_page = json.loads(first_result.stdout)
    second_page = json.loads(second_result.stdout)
    for actual, expected in (
        (first_page, expected_first),
        (second_page, expected_second),
    ):
        assert actual == {
            "scope_id": scope_id,
            "offset": expected["offset"],
            "limit": expected["limit"],
            "total_count": expected["total_count"],
            "has_more": expected["has_more"],
            "rows": [
                dict(row) for row in cast(list[sqlite3.Row], expected["rows"])
            ],
        }

    returned_revision_ids = {
        row["revision_id"] for row in first_page["rows"] + second_page["rows"]
    }
    assert returned_revision_ids == {first_revision, second_revision}
    assert later_revision not in returned_revision_ids
    assert opened_read_only == [True, True]
    assert database.read_bytes() == database_before


def test_exact_scope_search_supports_existing_row_output_formats(tmp_path):
    root = tmp_path / "exact-scope-search-formats"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "Migration appears here.")
    scope_id = project.snapshot([revision])
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "search",
            "exact-scope",
            str(root),
            scope_id,
            "Migration",
            "--format",
            "jsonl",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["revision_id"] == revision
    assert f"scope_id={scope_id}" in result.stderr
    assert "has_more=false" in result.stderr


@pytest.mark.parametrize(
    ("option", "value"),
    [("--offset", "-1"), ("--limit", "0"), ("--limit", "1001")],
)
def test_exact_scope_search_rejects_invalid_page_bounds(tmp_path, option, value):
    root = tmp_path / "exact-scope-invalid-page"
    Project.init(root).close()

    result = CliRunner().invoke(
        app,
        [
            "search",
            "exact-scope",
            str(root),
            "unknown-scope",
            "Migration",
            option,
            value,
        ],
    )

    assert result.exit_code == 2
    assert option in result.output


@pytest.mark.parametrize("phrase", ["", "   ", "x" * 2_001])
def test_exact_scope_search_rejects_blank_or_oversized_phrases(tmp_path, phrase):
    root = tmp_path / "exact-scope-invalid-phrase"
    Project.init(root).close()

    result = CliRunner().invoke(
        app, ["search", "exact-scope", str(root), "scope", phrase]
    )

    assert result.exit_code == 2
    assert "phrase must contain 1..2000 characters" in result.output


def test_exact_scope_search_reports_unknown_scope_clearly(tmp_path):
    root = tmp_path / "exact-scope-unknown"
    Project.init(root).close()

    result = CliRunner().invoke(
        app,
        ["search", "exact-scope", str(root), "missing-scope", "Migration"],
    )

    assert result.exit_code == 2
    assert "unknown scope" in result.output.lower()
    assert str(root) not in result.output


def test_exact_scope_search_rejects_unsupported_output_format(tmp_path):
    root = tmp_path / "exact-scope-invalid-format"
    Project.init(root).close()

    result = CliRunner().invoke(
        app,
        [
            "search",
            "exact-scope",
            str(root),
            "scope",
            "Migration",
            "--format",
            "yaml",
        ],
    )

    assert result.exit_code == 2
    assert "format must be json, jsonl, csv, or table" in result.output
