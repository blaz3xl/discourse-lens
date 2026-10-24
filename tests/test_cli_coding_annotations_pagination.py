from __future__ import annotations

import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def _create_annotations(root):
    project = Project.init(root)
    _, revision_id = project.import_text("sample.txt", "Alpha beta gamma.")
    codebook_id = project.create_codebook("fixture")
    code_id = project.create_code(codebook_id, "sample")
    first_span = project.create_span(revision_id, 0, 5)
    second_span = project.create_span(revision_id, 6, 10)
    project.annotate_span(code_id, first_span, "first")
    project.annotate_span(code_id, second_span, "second")
    project.close()
    return revision_id


def test_coding_annotations_keeps_legacy_list_and_supports_pages(tmp_path):
    root = tmp_path / "annotation-project"
    revision_id = _create_annotations(root)
    runner = CliRunner()
    with Project.open(root, read_only=True) as project:
        snapshots_before = project.db.execute(
            "SELECT COUNT(*) FROM scope_snapshots"
        ).fetchone()[0]

    legacy = runner.invoke(app, ["coding", "annotations", str(root), revision_id])
    page = runner.invoke(
        app,
        ["coding", "annotations", str(root), revision_id, "--offset", "1", "--limit", "1"],
    )
    limit_only = runner.invoke(
        app,
        ["coding", "annotations", str(root), revision_id, "--limit", "1"],
    )

    assert legacy.exit_code == 0, legacy.output
    assert page.exit_code == 0, page.output
    assert limit_only.exit_code == 0, limit_only.output
    legacy_rows = json.loads(legacy.output)
    assert len(legacy_rows) == 2
    with Project.open(root, read_only=True) as project:
        snapshots_after = project.db.execute(
            "SELECT COUNT(*) FROM scope_snapshots"
        ).fetchone()[0]
    assert snapshots_after == snapshots_before

    payload = json.loads(page.output)
    assert payload["items"] == legacy_rows[1:]
    assert payload["revision_count"] == 1
    assert payload["offset"] == 1
    assert payload["limit"] == 1
    assert payload["total"] == 2
    assert payload["has_more"] is False

    first_page = json.loads(limit_only.output)
    assert first_page["items"] == legacy_rows[:1]
    assert first_page["offset"] == 0
    assert first_page["limit"] == 1
    assert first_page["total"] == 2
    assert first_page["has_more"] is True


def test_coding_annotations_offset_uses_default_limit_and_rejects_invalid_bounds(tmp_path):
    root = tmp_path / "annotation-default-limit"
    revision_id = _create_annotations(root)
    runner = CliRunner()

    page = runner.invoke(
        app,
        ["coding", "annotations", str(root), revision_id, "--offset", "1"],
    )
    invalid_offset = runner.invoke(
        app,
        ["coding", "annotations", str(root), revision_id, "--offset", "-1"],
    )
    invalid_limit = runner.invoke(
        app,
        ["coding", "annotations", str(root), revision_id, "--limit", "101"],
    )

    assert page.exit_code == 0, page.output
    payload = json.loads(page.output)
    assert payload["offset"] == 1
    assert payload["limit"] == 100
    assert payload["total"] == 2
    assert payload["has_more"] is False
    assert len(payload["items"]) == 1
    assert invalid_offset.exit_code == 2
    assert invalid_limit.exit_code == 2
