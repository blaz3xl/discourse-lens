from __future__ import annotations

import csv
import io
import json

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app


def _invoke(root, scope_id: str, query: str, *options: str):
    return CliRunner().invoke(
        app,
        ["corpus", "collocations-scope", str(root), scope_id, query, *options],
    )


def _scope_ids(project: Project) -> list[str]:
    return [
        str(row[0])
        for row in project.db.execute("SELECT id FROM scope_snapshots ORDER BY id").fetchall()
    ]


def test_collocations_scope_matches_core_pages_and_excludes_later_revision(
    tmp_path, monkeypatch
):
    root = tmp_path / "scope-collocations-parity"
    project = Project.init(root)
    _, first_revision = project.import_text(
        "first.txt", "target zulu target alpha target beta target alpha"
    )
    _, second_revision = project.import_text(
        "second.txt", "target delta target gamma target delta"
    )
    scope_id = project.snapshot([first_revision, second_revision])
    _, later_revision = project.import_text("later.txt", "outsideonly target outsideonly")
    project.close()

    original_open = Project.open
    with original_open(root, read_only=True) as expected_project:
        expected_pages = [
            expected_project.corpus_collocations_page(
                scope_id, "target", window=1, offset=offset, limit=2
            )
            for offset in (0, 2, 4)
        ]
        scope_members_before = expected_project.scope_revisions(scope_id)
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
            scope_id,
            "target",
            "--window",
            "1",
            "--offset",
            str(offset),
            "--limit",
            "2",
        )
        for offset in (0, 2, 4)
    ]

    pages = []
    for result, expected in zip(results, expected_pages, strict=True):
        assert result.exit_code == 0, result.output
        page = json.loads(result.stdout)
        assert page == expected
        pages.append(page)

    terms = [row["term"] for page in pages for row in page["rows"]]
    assert [page["has_more"] for page in pages] == [True, True, False]
    assert len(terms) == len(set(terms)) == 5
    assert "outsideonly" not in terms
    assert later_revision not in scope_members_before
    assert scope_members_before == [first_revision, second_revision]
    assert opened_read_only == [True, True, True]
    assert database.read_bytes() == database_before

    with original_open(root, read_only=True) as verified:
        assert verified.scope_revisions(scope_id) == scope_members_before
        assert _scope_ids(verified) == scope_ids_before


@pytest.mark.parametrize("output_format", ["jsonl", "csv", "table"])
def test_collocations_scope_supports_existing_row_formats(tmp_path, output_format):
    root = tmp_path / f"scope-collocations-format-{output_format}"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "target alpha target beta")
    scope_id = project.snapshot([revision])
    project.close()

    result = _invoke(
        root,
        scope_id,
        "target",
        "--window",
        "1",
        "--limit",
        "1",
        "--format",
        output_format,
    )

    assert result.exit_code == 0, result.output
    assert "term" in result.stdout
    assert f"scope_id={scope_id}" in result.stderr
    assert "has_more=true" in result.stderr
    if output_format == "jsonl":
        assert json.loads(result.stdout.splitlines()[0])["term"]
    elif output_format == "csv":
        assert next(iter(csv.DictReader(io.StringIO(result.stdout))))["term"]


def test_collocations_scope_csv_keeps_header_for_empty_result(tmp_path):
    root = tmp_path / "scope-collocations-empty-csv"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "target")
    scope_id = project.snapshot([revision])
    project.close()

    result = _invoke(root, scope_id, "target", "--format", "csv")

    assert result.exit_code == 0, result.output
    assert result.stdout == "term,count\n"
    assert "no collocation rows" in result.stderr
    assert "has_more=false" in result.stderr


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--window", "0"),
        ("--window", "101"),
        ("--offset", "-1"),
        ("--offset", "1000001"),
        ("--limit", "0"),
        ("--limit", "1001"),
    ],
)
def test_collocations_scope_rejects_invalid_bounds_before_project_open(
    tmp_path, monkeypatch, option, value
):
    root = tmp_path / f"scope-collocations-invalid-{option}-{value}"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "target alpha")
    scope_id = project.snapshot([revision])
    project.close()

    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("invalid CLI bounds must fail before opening the project")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = _invoke(root, scope_id, "target", option, value)

    assert result.exit_code == 2
    assert option in result.output
    assert "Traceback" not in result.output
    assert opened == []


@pytest.mark.parametrize(
    ("query", "expected_message"),
    [("", "query cannot be blank"), ("two tokens", "exactly one surface token")],
)
def test_collocations_scope_rejects_invalid_queries(tmp_path, query, expected_message):
    root = tmp_path / "scope-collocations-invalid-query"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "target alpha")
    scope_id = project.snapshot([revision])
    project.close()

    result = _invoke(root, scope_id, query)

    assert result.exit_code == 2
    assert expected_message in result.output


def test_collocations_scope_reports_unknown_scope_clearly(tmp_path):
    root = tmp_path / "scope-collocations-unknown"
    project = Project.init(root)
    project.close()

    result = _invoke(root, "missing-scope", "target")

    assert result.exit_code == 2
    assert "unknown scope" in result.output.lower()
    assert str(root) not in result.output


def test_collocations_scope_rejects_unsupported_output_format(tmp_path):
    root = tmp_path / "scope-collocations-format-invalid"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "target alpha")
    scope_id = project.snapshot([revision])
    project.close()

    result = _invoke(root, scope_id, "target", "--format", "yaml")

    assert result.exit_code == 2
    assert "format must be json, jsonl, csv, or table" in result.output
