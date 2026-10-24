import json

import pytest
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def test_cli_dispersion_keeps_full_default_and_matches_sdk_pages(tmp_path):
    root = tmp_path / "cli-dispersion-pages"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "migration policy")
    _, second_revision = project.import_text("second.txt", "public migration")
    _, outside_revision = project.import_text("outside.txt", "migration outside")
    scope_id = project.snapshot([first_revision, second_revision])
    sdk_full = project.corpus_dispersion(scope_id, "migration")
    sdk_page = project.corpus_dispersion_page(scope_id, "migration", offset=1, limit=1)
    project.close()

    runner = CliRunner()
    full = runner.invoke(
        app,
        ["corpus", "dispersion", str(root), f"{first_revision},{second_revision}", "migration"],
    )
    page = runner.invoke(
        app,
        [
            "corpus",
            "dispersion",
            str(root),
            f"{first_revision},{second_revision}",
            "migration",
            "--offset",
            "1",
            "--limit",
            "1",
        ],
    )

    assert full.exit_code == 0, full.output
    assert page.exit_code == 0, page.output
    full_payload = json.loads(full.stdout)
    page_payload = json.loads(page.stdout)
    assert {key: value for key, value in full_payload.items() if key != "scope_id"} == {
        key: value for key, value in sdk_full.items() if key != "scope_id"
    }
    assert {key: value for key, value in page_payload.items() if key != "scope_id"} == {
        key: value for key, value in sdk_page.items() if key != "scope_id"
    }
    assert page_payload["total_rows"] == 2
    assert page_payload["has_more"] is False
    assert outside_revision not in {row["revision_id"] for row in page_payload["documents"]}


@pytest.mark.parametrize("option,value", [("--offset", "-1"), ("--limit", "0"), ("--limit", "1001")])
def test_cli_dispersion_rejects_invalid_page_bounds(tmp_path, option, value):
    root = tmp_path / "cli-dispersion-invalid"
    project = Project.init(root)
    _, revision = project.import_text("text.txt", "migration policy")
    project.close()

    result = CliRunner().invoke(
        app,
        ["corpus", "dispersion", str(root), revision, "migration", option, value],
    )
    assert result.exit_code != 0
