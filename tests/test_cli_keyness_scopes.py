from __future__ import annotations

import csv
import io
import json
from typing import cast

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app


def _invoke(root, left_scope_id: str, right_scope_id: str, *options: str):
    return CliRunner().invoke(
        app,
        [
            "corpus",
            "keyness-scopes",
            str(root),
            left_scope_id,
            right_scope_id,
            *options,
        ],
    )


def _scope_ids(project: Project) -> list[str]:
    return [
        str(row[0])
        for row in project.db.execute("SELECT id FROM scope_snapshots ORDER BY id").fetchall()
    ]


def test_keyness_scopes_matches_core_pages_for_frozen_multirevision_scopes(
    tmp_path, monkeypatch
):
    root = tmp_path / "keyness-scopes-parity"
    project = Project.init(root)
    _, left_first = project.import_text("left-first.txt", "leftonly leftonly shared alpha")
    _, left_second = project.import_text("left-second.txt", "leftonly lower beta")
    _, right_first = project.import_text("right-first.txt", "rightonly rightonly shared gamma")
    _, right_second = project.import_text("right-second.txt", "rightonly upper delta")
    left_scope_id = project.snapshot([left_first, left_second])
    right_scope_id = project.snapshot([right_first, right_second])
    _, later_revision = project.import_text(
        "later.txt", "outsidesentinel outsidesentinel outsidesentinel"
    )
    project.close()

    original_open = Project.open
    with original_open(root, read_only=True) as expected_project:
        expected_pages = [
            expected_project.corpus_keyness_page(
                left_scope_id, right_scope_id, offset=offset, limit=3
            )
            for offset in (0, 3, 6)
        ]
        left_members_before = expected_project.scope_revisions(left_scope_id)
        right_members_before = expected_project.scope_revisions(right_scope_id)
        scope_ids_before = _scope_ids(expected_project)

    opened_read_only: list[bool] = []

    def tracked_open(path, *, read_only: bool = False):
        opened_read_only.append(read_only)
        return original_open(path, read_only=read_only)

    monkeypatch.setattr(cli_module.Project, "open", staticmethod(tracked_open))
    database = root / ".discourselens" / "project.sqlite"
    database_before = database.read_bytes()
    results = [
        _invoke(
            root,
            left_scope_id,
            right_scope_id,
            "--offset",
            str(offset),
            "--limit",
            "3",
        )
        for offset in (0, 3, 6)
    ]

    pages: list[dict[str, object]] = []
    for result, expected in zip(results, expected_pages, strict=True):
        assert result.exit_code == 0, result.output
        page = json.loads(result.stdout)
        assert page["left_scope_id"] == left_scope_id
        assert page["right_scope_id"] == right_scope_id
        assert {
            key: value
            for key, value in page.items()
            if key not in {"left_scope_id", "right_scope_id"}
        } == expected
        pages.append(page)

    rows = [row for page in pages for row in cast(list[dict[str, object]], page["rows"])]
    tokens = [str(row["token"]) for row in rows]
    assert [page["has_more"] for page in pages] == [True, True, False]
    assert len(tokens) == len(set(tokens)) == 9
    assert "outsidesentinel" not in tokens
    assert later_revision not in left_members_before + right_members_before
    assert left_members_before == [left_first, left_second]
    assert right_members_before == [right_first, right_second]
    assert opened_read_only == [True, True, True]
    assert database.read_bytes() == database_before

    with original_open(root, read_only=True) as verified:
        assert verified.scope_revisions(left_scope_id) == left_members_before
        assert verified.scope_revisions(right_scope_id) == right_members_before
        assert _scope_ids(verified) == scope_ids_before


@pytest.mark.parametrize("output_format", ["jsonl", "csv", "table"])
def test_keyness_scopes_supports_existing_row_formats(tmp_path, output_format):
    root = tmp_path / f"keyness-scopes-format-{output_format}"
    project = Project.init(root)
    _, left_revision = project.import_text("left.txt", "alpha alpha beta")
    _, right_revision = project.import_text("right.txt", "gamma gamma beta")
    left_scope_id = project.snapshot([left_revision])
    right_scope_id = project.snapshot([right_revision])
    project.close()

    result = _invoke(
        root,
        left_scope_id,
        right_scope_id,
        "--format",
        output_format,
        "--limit",
        "1",
    )

    assert result.exit_code == 0, result.output
    assert "token" in result.stdout
    assert "has_more" not in result.stdout
    assert left_scope_id not in result.stdout
    assert right_scope_id not in result.stdout
    assert "keyness rows 1–1 of 3;" in result.stderr
    assert "has_more=true" in result.stderr
    assert f"left_scope_id={left_scope_id}" in result.stderr
    assert f"right_scope_id={right_scope_id}" in result.stderr
    if output_format == "jsonl":
        assert json.loads(result.stdout.splitlines()[0])["token"]
    elif output_format == "csv":
        assert next(iter(csv.DictReader(io.StringIO(result.stdout))))["token"]


def test_keyness_scopes_csv_keeps_headers_for_empty_page(tmp_path):
    root = tmp_path / "keyness-scopes-empty-csv"
    project = Project.init(root)
    _, left_revision = project.import_text("left.txt", "alpha")
    _, right_revision = project.import_text("right.txt", "beta")
    left_scope_id = project.snapshot([left_revision])
    right_scope_id = project.snapshot([right_revision])
    project.close()

    result = _invoke(
        root,
        left_scope_id,
        right_scope_id,
        "--format",
        "csv",
        "--offset",
        "100",
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "token,left_count,right_count,log_likelihood\n"
    assert "no keyness rows at offset 100; total_rows=2" in result.stderr
    assert f"left_scope_id={left_scope_id}" in result.stderr
    assert f"right_scope_id={right_scope_id}" in result.stderr
    assert "has_more=false" in result.stderr


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--offset", "-1"),
        ("--offset", "1000001"),
        ("--limit", "0"),
        ("--limit", "1001"),
    ],
)
def test_keyness_scopes_rejects_invalid_pagination_before_project_open(
    tmp_path, monkeypatch, option, value
):
    root = tmp_path / f"keyness-scopes-invalid-{option}-{value}"
    Project.init(root).close()
    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("invalid CLI bounds must fail before opening the project")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = _invoke(root, "left-scope", "right-scope", option, value)

    assert result.exit_code == 2
    assert option in result.output
    assert "Traceback" not in result.output
    assert opened == []


def test_keyness_scopes_rejects_unsupported_format_before_project_open(tmp_path, monkeypatch):
    root = tmp_path / "keyness-scopes-invalid-format"
    Project.init(root).close()
    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("invalid output format must fail before opening the project")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = _invoke(root, "left-scope", "right-scope", "--format", "yaml")

    assert result.exit_code == 2
    assert "format must be json, jsonl, csv, or table" in result.output
    assert opened == []


@pytest.mark.parametrize("unknown_side", ["left", "right"])
def test_keyness_scopes_reports_unknown_scope_clearly(tmp_path, unknown_side):
    root = tmp_path / f"keyness-scopes-unknown-{unknown_side}"
    project = Project.init(root)
    _, revision = project.import_text("known.txt", "alpha beta")
    known_scope = project.snapshot([revision])
    project.close()

    left_scope_id = "missing-scope" if unknown_side == "left" else known_scope
    right_scope_id = "missing-scope" if unknown_side == "right" else known_scope
    result = _invoke(root, left_scope_id, right_scope_id)

    assert result.exit_code == 2
    assert "unknown scope" in result.output.lower()
    assert str(root) not in result.output


def test_keyness_scopes_rejects_empty_token_populations(tmp_path):
    root = tmp_path / "keyness-scopes-empty-populations"
    project = Project.init(root)
    _, left_revision = project.import_text("left.txt", "   \n\t")
    _, right_revision = project.import_text("right.txt", "  \n ")
    left_scope_id = project.snapshot([left_revision])
    right_scope_id = project.snapshot([right_revision])
    project.close()

    result = _invoke(root, left_scope_id, right_scope_id)

    assert result.exit_code == 2
    assert "comparison requires two non-empty explicit scopes" in result.output
