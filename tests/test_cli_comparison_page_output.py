import csv
import io
import json

import pytest
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


@pytest.fixture
def comparison_project(tmp_path):
    root = tmp_path / "comparison-project"
    with Project.init(root) as project:
        _, left_revision = project.import_text("left.txt", "alpha beta gamma delta")
        _, right_revision = project.import_text("right.txt", "epsilon zeta eta theta")
    return root, left_revision, right_revision


def _assert_flattened_output_is_parseable(output_format, stdout):
    if output_format == "csv":
        assert list(csv.reader(io.StringIO(stdout)))
    elif output_format == "jsonl":
        assert [json.loads(line) for line in stdout.splitlines()]
    else:
        assert stdout.strip()


@pytest.mark.parametrize("command, label", [("compare-frequency", "frequency"), ("keyness", "keyness")])
@pytest.mark.parametrize("output_format", ["csv", "jsonl", "table"])
def test_comparison_flattened_pages_report_status_only_on_stderr(
    comparison_project, command, label, output_format
):
    root, left_revision, right_revision = comparison_project
    runner = CliRunner()

    first_page = runner.invoke(
        app,
        [
            "corpus",
            command,
            str(root),
            left_revision,
            right_revision,
            "--format",
            output_format,
            "--offset",
            "0",
            "--limit",
            "3",
        ],
    )
    assert first_page.exit_code == 0, first_page.output
    _assert_flattened_output_is_parseable(output_format, first_page.stdout)
    assert first_page.stderr == f"{label} rows 1–3 of 8; has_more=true\n"
    assert "has_more" not in first_page.stdout

    final_page = runner.invoke(
        app,
        [
            "corpus",
            command,
            str(root),
            left_revision,
            right_revision,
            "--format",
            output_format,
            "--offset",
            "6",
            "--limit",
            "3",
        ],
    )
    assert final_page.exit_code == 0, final_page.output
    _assert_flattened_output_is_parseable(output_format, final_page.stdout)
    assert final_page.stderr == f"{label} rows 7–8 of 8; has_more=false\n"
    assert "has_more" not in final_page.stdout


@pytest.mark.parametrize("command", ["compare-frequency", "keyness"])
def test_comparison_json_keeps_structured_page_without_status_stderr(
    comparison_project, command
):
    root, left_revision, right_revision = comparison_project
    result = CliRunner().invoke(
        app,
        [
            "corpus",
            command,
            str(root),
            left_revision,
            right_revision,
            "--format",
            "json",
            "--offset",
            "0",
            "--limit",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    page = json.loads(result.stdout)
    assert page["offset"] == 0
    assert page["limit"] == 3
    assert page["total_rows"] == 8
    assert page["has_more"] is True
    assert result.stderr == ""
