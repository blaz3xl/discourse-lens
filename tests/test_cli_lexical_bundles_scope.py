from __future__ import annotations

import csv
import io
import json

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app


def _invoke(root, scope_id: str, *options: str):
    return CliRunner().invoke(
        app,
        ["corpus", "lexical-bundles-scope", str(root), scope_id, *options],
    )


def test_cli_lexical_bundles_scope_matches_core_pages_without_mutating_scope(
    tmp_path, monkeypatch
):
    root = tmp_path / "cli-lexical-bundles-scope"
    project = Project.init(root)
    _, first_revision = project.import_text(
        "first.txt", "we must speak we must act"
    )
    _, second_revision = project.import_text(
        "second.txt", "we must listen we must act"
    )
    scope_id = project.snapshot([first_revision, second_revision])
    project.close()

    with Project.open(root) as writer:
        _, later_revision = writer.import_text(
            "later.txt", "we must outside we must"
        )

    with Project.open(root, read_only=True) as expected_project:
        expected_pages = [
            expected_project.corpus_lexical_bundles(
                scope_id,
                size=2,
                min_frequency=2,
                min_document_frequency=2,
                offset=offset,
                limit=1,
            )
            for offset in (0, 1)
        ]
        scope_revisions = expected_project.scope_revisions(scope_id)
        scope_count = expected_project.db.execute(
            "SELECT COUNT(*) FROM scope_snapshots"
        ).fetchone()[0]

    original_open = Project.open
    opened_read_only: list[bool] = []

    def tracked_open(path, *, read_only: bool = False):
        opened_read_only.append(read_only)
        return original_open(path, read_only=read_only)

    monkeypatch.setattr(cli_module.Project, "open", staticmethod(tracked_open))
    database = root / ".discourselens" / "project.sqlite"
    database_before = database.read_bytes()
    pages = []
    for offset in (0, 1):
        result = _invoke(
            root,
            scope_id,
            "--size",
            "2",
            "--min-frequency",
            "2",
            "--min-documents",
            "2",
            "--offset",
            str(offset),
            "--limit",
            "1",
        )
        assert result.exit_code == 0, result.output
        pages.append(json.loads(result.stdout))

    assert pages == expected_pages
    assert [row["bundle"] for page in pages for row in page["rows"]] == [
        "we must",
        "must act",
    ]
    assert [row["frequency"] for page in pages for row in page["rows"]] == [4, 2]
    assert [row["document_frequency"] for page in pages for row in page["rows"]] == [2, 2]
    assert [page["has_more"] for page in pages] == [True, False]
    assert [page["truncated"] for page in pages] == [True, False]
    assert all(page["candidate_count"] == 2 for page in pages)
    assert later_revision not in scope_revisions
    assert scope_revisions == [first_revision, second_revision]
    assert opened_read_only == [True, True]
    assert database.read_bytes() == database_before

    with original_open(root, read_only=True) as verified:
        assert verified.scope_revisions(scope_id) == scope_revisions
        assert verified.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == scope_count


@pytest.mark.parametrize("output_format", ["jsonl", "csv", "table"])
def test_cli_lexical_bundles_scope_supports_existing_row_formats(tmp_path, output_format):
    root = tmp_path / f"cli-lexical-bundles-scope-{output_format}"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "red blue red blue green")
    scope_id = project.snapshot([revision])
    project.close()

    result = _invoke(
        root,
        scope_id,
        "--size",
        "2",
        "--min-frequency",
        "1",
        "--format",
        output_format,
        "--limit",
        "1",
    )

    assert result.exit_code == 0, result.output
    assert "bundle" in result.stdout
    assert f"scope_id={scope_id}" in result.stderr
    assert "min_frequency=1" in result.stderr
    assert "min_documents=1" in result.stderr
    assert "of 3" in result.stderr
    assert "candidate_count=3" in result.stderr
    assert "has_more=true" in result.stderr
    if output_format == "jsonl":
        assert json.loads(result.stdout.splitlines()[0])["bundle"] == "red blue"
    elif output_format == "csv":
        row = next(iter(csv.DictReader(io.StringIO(result.stdout))))
        assert row["bundle"] == "red blue"


def test_cli_lexical_bundles_scope_keeps_csv_header_for_empty_page(tmp_path):
    root = tmp_path / "cli-lexical-bundles-scope-empty-csv"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "red blue")
    scope_id = project.snapshot([revision])
    project.close()

    result = _invoke(root, scope_id, "--size", "2", "--format", "csv")

    assert result.exit_code == 0, result.output
    assert result.stdout == "bundle,frequency,document_frequency\n"
    assert "candidate_count=0" in result.stderr
    assert "has_more=false" in result.stderr


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--size", "1"),
        ("--size", "11"),
        ("--min-frequency", "0"),
        ("--min-documents", "0"),
        ("--offset", "-1"),
        ("--offset", str(0x7FFF_FFFF_FFFF_FFFF + 1)),
        ("--limit", "0"),
        ("--limit", "1001"),
    ],
)
def test_cli_lexical_bundles_scope_rejects_invalid_bounds_before_open(
    tmp_path, monkeypatch, option, value
):
    root = tmp_path / f"cli-lexical-bundles-scope-invalid-{option}-{value}"
    Project.init(root).close()
    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("invalid lexical bundle options must fail before opening the project")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = _invoke(root, "known-scope", option, value)

    assert result.exit_code == 2
    assert option in result.output
    assert "Traceback" not in result.output
    assert opened == []


def test_cli_lexical_bundles_scope_reports_unknown_and_blank_scope(tmp_path):
    root = tmp_path / "cli-lexical-bundles-scope-unknown"
    Project.init(root).close()

    unknown = _invoke(root, "missing-scope")
    blank = _invoke(root, " ")

    assert unknown.exit_code == 2
    assert "unknown scope" in unknown.output.lower()
    assert str(root) not in unknown.output
    assert blank.exit_code == 2
    assert "scope ID cannot be blank" in blank.output
