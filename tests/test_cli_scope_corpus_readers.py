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


def test_scope_frequency_and_kwic_pages_match_core_and_remain_frozen_read_only(
    tmp_path, monkeypatch
):
    root = tmp_path / "scope-corpus-readers"
    project = Project.init(root)
    _, first_revision = project.import_text(
        "first.txt", "Migration policy frames debate.\n\nMigration policy shifts later."
    )
    _, second_revision = project.import_text(
        "second.txt", "Migration debates continue. Apples grow in spring."
    )
    scope_id = project.snapshot([first_revision, second_revision])
    _, later_revision = project.import_text(
        "later.txt", "outsideword outsideword Migration."
    )
    project.close()

    with Project.open(root, read_only=True) as expected_project:
        expected_frequency_first = expected_project.corpus_frequency_page(
            scope_id, offset=0, limit=3
        )
        expected_frequency_second = expected_project.corpus_frequency_page(
            scope_id, offset=3, limit=3
        )
        expected_kwic_first = expected_project.corpus_kwic_page(
            scope_id, "Migration", window=2, offset=0, limit=2
        )
        expected_kwic_second = expected_project.corpus_kwic_page(
            scope_id, "Migration", window=2, offset=2, limit=2
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

    def invoke_frequency(offset: int):
        return runner.invoke(
            app,
            [
                "corpus",
                "frequency-scope",
                str(root),
                scope_id,
                "--format",
                "json",
                "--offset",
                str(offset),
                "--limit",
                "3",
            ],
        )

    def invoke_kwic(offset: int):
        return runner.invoke(
            app,
            [
                "corpus",
                "kwic-scope",
                str(root),
                scope_id,
                "Migration",
                "--format",
                "json",
                "--window",
                "2",
                "--offset",
                str(offset),
                "--limit",
                "2",
            ],
        )

    results = [
        (invoke_frequency(0), expected_frequency_first),
        (invoke_frequency(3), expected_frequency_second),
        (invoke_kwic(0), expected_kwic_first),
        (invoke_kwic(2), expected_kwic_second),
    ]

    pages: list[dict[str, object]] = []
    for result, expected in results:
        assert result.exit_code == 0, result.output
        actual = json.loads(result.stdout)
        assert actual == expected
        pages.append(actual)

    frequency_tokens = {
        str(row["token"])
        for page in pages[:2]
        for row in cast(list[dict[str, object]], page["rows"])
    }
    kwic_revision_ids = {
        str(row["revision_id"])
        for page in pages[2:]
        for row in cast(list[dict[str, object]], page["rows"])
    }
    assert "outsideword" not in frequency_tokens
    assert kwic_revision_ids == {first_revision, second_revision}
    assert later_revision not in kwic_revision_ids
    assert opened_read_only == [True, True, True, True]
    assert database.read_bytes() == database_before


@pytest.mark.parametrize(
    ("command", "scope_arguments", "column"),
    [
        ("frequency-scope", (), "token"),
        ("kwic-scope", ("migration",), "keyword"),
    ],
)
@pytest.mark.parametrize("output_format", ["jsonl", "csv", "table"])
def test_scope_corpus_readers_follow_existing_row_output_formats(
    tmp_path, command, scope_arguments, column, output_format
):
    root = tmp_path / f"scope-reader-{command}-{output_format}"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "Migration policy. Migration matters.")
    scope_id = project.snapshot([revision])
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            command,
            str(root),
            scope_id,
            *scope_arguments,
            "--format",
            output_format,
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert column in result.stdout
    assert f"scope_id={scope_id}" in result.stderr
    assert "has_more=true" in result.stderr
    if output_format == "jsonl":
        assert json.loads(result.stdout.splitlines()[0])[column]
    elif output_format == "csv":
        assert next(iter(csv.DictReader(io.StringIO(result.stdout))))[column]


@pytest.mark.parametrize(
    ("command", "scope_arguments", "expected_header"),
    [
        ("frequency-scope", (), "token,count\n"),
        (
            "kwic-scope",
            ("migration",),
            "revision_id,start_offset,end_offset,left,keyword,right\n",
        ),
    ],
)
def test_scope_corpus_readers_keep_csv_headers_for_empty_pages(
    tmp_path, command, scope_arguments, expected_header
):
    root = tmp_path / f"scope-reader-empty-{command}"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "Apples grow here.")
    scope_id = project.snapshot([revision])
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            command,
            str(root),
            scope_id,
            *scope_arguments,
            "--format",
            "csv",
            "--offset",
            "100",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == expected_header
    assert "has_more=false" in result.stderr


@pytest.mark.parametrize(
    ("command", "scope_arguments"),
    [("frequency-scope", ()), ("kwic-scope", ("migration",))],
)
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--offset", "-1"),
        ("--limit", "0"),
        ("--limit", "1001"),
    ],
)
def test_scope_corpus_readers_reject_invalid_page_bounds(
    tmp_path, command, scope_arguments, option, value
):
    root = tmp_path / f"scope-reader-invalid-{command}-{option}-{value}"
    Project.init(root).close()

    result = CliRunner().invoke(
        app,
        ["corpus", command, str(root), "missing-scope", *scope_arguments, option, value],
    )

    assert result.exit_code == 2
    assert option in result.output


@pytest.mark.parametrize("value", ["-1", "101"])
def test_scope_kwic_reader_rejects_invalid_window_bounds(tmp_path, value):
    root = tmp_path / f"scope-kwic-invalid-window-{value}"
    Project.init(root).close()

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "kwic-scope",
            str(root),
            "missing-scope",
            "migration",
            "--window",
            value,
        ],
    )

    assert result.exit_code == 2
    assert "--window" in result.output


@pytest.mark.parametrize(
    ("command", "scope_arguments"),
    [("frequency-scope", ()), ("kwic-scope", ("migration",))],
)
def test_scope_corpus_readers_reject_blank_and_unknown_scopes(
    tmp_path, command, scope_arguments
):
    root = tmp_path / f"scope-reader-bad-scope-{command}"
    Project.init(root).close()
    runner = CliRunner()

    blank = runner.invoke(app, ["corpus", command, str(root), "", *scope_arguments])
    unknown = runner.invoke(
        app, ["corpus", command, str(root), "missing-scope", *scope_arguments]
    )

    assert blank.exit_code == 2
    assert "scope ID cannot be blank" in blank.output
    assert unknown.exit_code == 2
    assert "unknown scope" in unknown.output.lower()
    assert str(root) not in unknown.output


def test_scope_kwic_reader_reports_invalid_query_clearly(tmp_path):
    root = tmp_path / "scope-kwic-invalid-query"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "Migration policy matters.")
    scope_id = project.snapshot([revision])
    project.close()

    result = CliRunner().invoke(
        app,
        ["corpus", "kwic-scope", str(root), scope_id, "two words"],
    )

    assert result.exit_code == 2
    assert "KWIC requires exactly one surface token" in result.output
