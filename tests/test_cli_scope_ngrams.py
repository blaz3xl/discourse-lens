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


def _invoke(root, scope_id: str, *options: str):
    return CliRunner().invoke(
        app,
        ["corpus", "ngrams-scope", str(root), scope_id, *options],
    )


def _scope_ids(project: Project) -> list[str]:
    return [
        str(row[0])
        for row in project.db.execute("SELECT id FROM scope_snapshots ORDER BY id").fetchall()
    ]


def test_scope_ngrams_matches_core_pages_without_crossing_or_expanding_scope(
    tmp_path, monkeypatch
):
    root = tmp_path / "scope-ngrams-parity"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "red blue red")
    _, second_revision = project.import_text("second.txt", "green gold green")
    scope_id = project.snapshot([first_revision, second_revision])
    _, later_revision = project.import_text("later.txt", "outsidengram token outsidengram")
    project.close()

    original_open = Project.open
    with original_open(root, read_only=True) as expected_project:
        expected_pages = [
            expected_project.corpus_ngram_page(scope_id, 2, offset=offset, limit=2)
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
            "--size",
            "2",
            "--offset",
            str(offset),
            "--limit",
            "2",
        )
        for offset in (0, 2, 4)
    ]

    pages: list[dict[str, object]] = []
    for result, expected in zip(results, expected_pages, strict=True):
        assert result.exit_code == 0, result.output
        page = json.loads(result.stdout)
        assert page == expected
        pages.append(page)

    rows = [
        row
        for page in pages
        for row in cast(list[dict[str, object]], page["rows"])
    ]
    ngrams = [str(row["ngram"]) for row in rows]
    assert [page["has_more"] for page in pages] == [True, False, False]
    assert len(ngrams) == len(set(ngrams)) == 4
    assert "red green" not in ngrams
    assert "outsidengram token" not in ngrams
    assert "token outsidengram" not in ngrams
    assert later_revision not in scope_members_before
    assert scope_members_before == [first_revision, second_revision]
    assert opened_read_only == [True, True, True]
    assert database.read_bytes() == database_before

    with original_open(root, read_only=True) as verified:
        assert verified.scope_revisions(scope_id) == scope_members_before
        assert _scope_ids(verified) == scope_ids_before


@pytest.mark.parametrize("output_format", ["jsonl", "csv", "table"])
def test_scope_ngrams_supports_existing_row_formats(tmp_path, output_format):
    root = tmp_path / f"scope-ngrams-format-{output_format}"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "red blue red")
    scope_id = project.snapshot([revision])
    project.close()

    result = _invoke(
        root,
        scope_id,
        "--size",
        "2",
        "--limit",
        "1",
        "--format",
        output_format,
    )

    assert result.exit_code == 0, result.output
    assert "ngram" in result.stdout
    assert f"scope_id={scope_id}" in result.stderr
    assert "has_more=true" in result.stderr
    if output_format == "jsonl":
        assert json.loads(result.stdout.splitlines()[0])["ngram"]
    elif output_format == "csv":
        assert next(iter(csv.DictReader(io.StringIO(result.stdout))))["ngram"]


def test_scope_ngrams_csv_keeps_headers_for_empty_page(tmp_path):
    root = tmp_path / "scope-ngrams-empty-csv"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "single")
    scope_id = project.snapshot([revision])
    project.close()

    result = _invoke(root, scope_id, "--size", "2", "--format", "csv")

    assert result.exit_code == 0, result.output
    assert result.stdout == "ngram,count\n"
    assert "no n-gram rows at offset 0; total_values=0" in result.stderr
    assert "has_more=false" in result.stderr


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--size", "0"),
        ("--size", "-1"),
        ("--offset", "-1"),
        ("--offset", "1000001"),
        ("--limit", "0"),
        ("--limit", "1001"),
    ],
)
def test_scope_ngrams_rejects_invalid_size_and_bounds_before_project_open(
    tmp_path, monkeypatch, option, value
):
    root = tmp_path / f"scope-ngrams-invalid-{option}-{value}"
    Project.init(root).close()
    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("invalid CLI bounds must fail before opening the project")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    options = [option, value] if option == "--size" else ["--size", "2", option, value]
    result = _invoke(root, "known-scope", *options)

    assert result.exit_code == 2
    assert option in result.output
    assert "Traceback" not in result.output
    assert opened == []


def test_scope_ngrams_rejects_unsupported_output_format_before_project_open(
    tmp_path, monkeypatch
):
    root = tmp_path / "scope-ngrams-invalid-format"
    Project.init(root).close()
    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("invalid output format must fail before opening the project")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = _invoke(root, "known-scope", "--size", "2", "--format", "yaml")

    assert result.exit_code == 2
    assert "format must be json, jsonl, csv, or table" in result.output
    assert opened == []


def test_scope_ngrams_rejects_blank_and_unknown_scope(tmp_path):
    root = tmp_path / "scope-ngrams-unknown"
    project = Project.init(root)
    project.close()

    blank = _invoke(root, "", "--size", "2")
    unknown = _invoke(root, "missing-scope", "--size", "2")

    assert blank.exit_code == 2
    assert "scope ID cannot be blank" in blank.output
    assert unknown.exit_code == 2
    assert "unknown scope" in unknown.output.lower()
    assert str(root) not in unknown.output
