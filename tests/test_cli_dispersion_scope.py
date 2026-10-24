from __future__ import annotations

import csv
import io
import json

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app


def _invoke(root, scope_id: str, term: str = "migration", *options: str):
    return CliRunner().invoke(
        app,
        ["corpus", "dispersion-scope", str(root), scope_id, term, *options],
    )


def test_cli_dispersion_scope_matches_core_pages_and_stays_read_only(tmp_path, monkeypatch):
    root = tmp_path / "cli-dispersion-scope"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "migration migration policy")
    _, second_revision = project.import_text("second.txt", "public migration policy")
    _, outside_revision = project.import_text("outside.txt", "migration outside selected scope")
    scope_id = project.snapshot([first_revision, second_revision])
    project.close()

    with Project.open(root, read_only=True) as expected_project:
        expected_pages = [
            expected_project.corpus_dispersion_page(
                scope_id, "migration", offset=offset, limit=1
            )
            for offset in (0, 1)
        ]
        scope_revisions = expected_project.scope_revisions(scope_id)
        scope_count = expected_project.db.execute(
            "SELECT COUNT(*) FROM scope_snapshots"
        ).fetchone()[0]

    with Project.open(root) as writer:
        _, later_revision = writer.import_text(
            "later.txt", "migration imported after the scope"
        )

    opened_read_only: list[bool] = []
    original_open = Project.open

    def tracked_open(path, *, read_only: bool = False):
        opened_read_only.append(read_only)
        return original_open(path, read_only=read_only)

    monkeypatch.setattr(cli_module.Project, "open", staticmethod(tracked_open))
    database = root / ".discourselens" / "project.sqlite"
    database_before = database.read_bytes()
    pages = []
    for offset in (0, 1):
        result = _invoke(root, scope_id, "migration", "--offset", str(offset), "--limit", "1")
        assert result.exit_code == 0, result.output
        pages.append(json.loads(result.stdout))

    assert pages == expected_pages
    assert [row["revision_id"] for page in pages for row in page["documents"]] == [
        first_revision,
        second_revision,
    ]
    assert [page["has_more"] for page in pages] == [True, False]
    assert all(page["total_rows"] == page["document_count"] == 2 for page in pages)
    assert all(page["total_occurrences"] == 3 for page in pages)
    assert outside_revision not in {row["revision_id"] for page in pages for row in page["documents"]}
    assert later_revision not in scope_revisions
    assert scope_revisions == [first_revision, second_revision]
    assert opened_read_only == [True, True]
    assert database.read_bytes() == database_before

    with original_open(root, read_only=True) as verified:
        assert verified.scope_revisions(scope_id) == scope_revisions
        assert verified.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == scope_count


@pytest.mark.parametrize("output_format", ["jsonl", "csv", "table"])
def test_cli_dispersion_scope_supports_existing_document_row_formats(tmp_path, output_format):
    root = tmp_path / f"cli-dispersion-scope-{output_format}"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "migration policy. Migration matters.")
    scope_id = project.snapshot([revision])
    project.close()

    result = _invoke(
        root,
        scope_id,
        "migration",
        "--format",
        output_format,
        "--limit",
        "1",
    )

    assert result.exit_code == 0, result.output
    assert "revision_id" in result.stdout
    assert f"scope_id={scope_id}" in result.stderr
    assert "document_frequency=1" in result.stderr
    assert "has_more=false" in result.stderr
    if output_format == "jsonl":
        assert json.loads(result.stdout.splitlines()[0])["revision_id"] == revision
    elif output_format == "csv":
        row = next(iter(csv.DictReader(io.StringIO(result.stdout))))
        assert row["revision_id"] == revision


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--offset", "-1"),
        ("--offset", str(0x7FFF_FFFF_FFFF_FFFF + 1)),
        ("--limit", "0"),
        ("--limit", "1001"),
    ],
)
def test_cli_dispersion_scope_rejects_invalid_bounds_before_open(
    tmp_path, monkeypatch, option, value
):
    root = tmp_path / f"cli-dispersion-scope-invalid-{option}-{value}"
    Project.init(root).close()
    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("invalid page bounds must fail before opening the project")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = _invoke(root, "known-scope", "migration", option, value)

    assert result.exit_code == 2
    assert option in result.output
    assert "Traceback" not in result.output
    assert opened == []


def test_cli_dispersion_scope_reports_unknown_scope_and_rejects_blank_values(tmp_path):
    root = tmp_path / "cli-dispersion-scope-unknown"
    Project.init(root).close()

    unknown = _invoke(root, "missing-scope")
    blank_scope = _invoke(root, " ")
    blank_term = _invoke(root, "scope-id", " ")

    assert unknown.exit_code == 2
    assert "unknown scope" in unknown.output.lower()
    assert str(root) not in unknown.output
    assert blank_scope.exit_code == 2
    assert "scope ID cannot be blank" in blank_scope.output
    assert blank_term.exit_code == 2
    assert "term cannot be blank" in blank_term.output
