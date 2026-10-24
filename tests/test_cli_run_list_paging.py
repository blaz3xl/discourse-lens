from __future__ import annotations

import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def _create_run_history(root):
    project = Project.init(root)
    _, revision_id = project.import_text("sample.txt", "A bounded run listing fixture.")
    scope_id = project.snapshot([revision_id])
    run_ids = []
    for index in range(3):
        run_id = project.create_run(scope_id, {"index": index})
        run_ids.append(run_id)
    project.close()
    return run_ids


def test_run_list_keeps_legacy_list_and_supports_offset_pages(tmp_path):
    root = tmp_path / "run-list-project"
    run_ids = _create_run_history(root)
    runner = CliRunner()

    legacy = runner.invoke(app, ["run", "list", str(root)])
    limited_legacy = runner.invoke(app, ["run", "list", str(root), "--limit", "2"])
    page = runner.invoke(
        app,
        ["run", "list", str(root), "--offset", "1", "--limit", "1"],
    )

    assert legacy.exit_code == 0, legacy.output
    assert limited_legacy.exit_code == 0, limited_legacy.output
    assert page.exit_code == 0, page.output
    legacy_rows = json.loads(legacy.output)
    limited_rows = json.loads(limited_legacy.output)
    assert len(legacy_rows) == 3
    assert {row["id"] for row in legacy_rows} == set(run_ids)
    assert limited_rows == legacy_rows[:2]

    payload = json.loads(page.output)
    assert payload["offset"] == 1
    assert payload["limit"] == 1
    assert payload["total"] == 3
    assert payload["has_more"] is True
    assert payload["items"] == legacy_rows[1:2]


def test_run_list_offset_uses_default_limit_and_rejects_negative_values(tmp_path):
    root = tmp_path / "run-list-defaults"
    run_ids = _create_run_history(root)
    runner = CliRunner()

    legacy = runner.invoke(app, ["run", "list", str(root)])
    page = runner.invoke(app, ["run", "list", str(root), "--offset", "1"])
    invalid = runner.invoke(app, ["run", "list", str(root), "--offset", "-1"])

    assert legacy.exit_code == 0, legacy.output
    assert page.exit_code == 0, page.output
    legacy_rows = json.loads(legacy.output)
    payload = json.loads(page.output)
    assert payload["offset"] == 1
    assert payload["limit"] == 100
    assert payload["total"] == 3
    assert payload["has_more"] is False
    assert {row["id"] for row in legacy_rows} == set(run_ids)
    assert payload["items"] == legacy_rows[1:]
    assert invalid.exit_code == 2
