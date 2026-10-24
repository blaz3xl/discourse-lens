from __future__ import annotations

import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def _project_with_revisions(tmp_path):
    project_path = tmp_path / "research"
    project = Project.init(project_path)
    _, first = project.import_text("first.txt", 'Officials said “wait”.')
    _, second = project.import_text("second.txt", 'The witness said «stay here».')
    _, outside = project.import_text("outside.txt", 'Officials said "leave now."')
    project.close()
    return project_path, first, second, outside


def test_quote_attributions_json_is_limited_to_selected_revisions_without_language(
    tmp_path,
):
    project_path, first, _second, outside = _project_with_revisions(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "skills",
            "quote-attributions",
            str(project_path),
            "--revision",
            first,
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["candidate_count"] == 1
    assert payload["attribution_candidate_count"] == 0
    assert payload["scope_snapshot_id"]
    assert {item["revision_id"] for item in payload["candidates"]} == {first}
    assert all(not item["attribution_cues"] for item in payload["candidates"])
    assert outside not in {item["revision_id"] for item in payload["candidates"]}


def test_quote_attributions_language_is_applied_to_each_selected_revision(tmp_path):
    project_path, first, second, outside = _project_with_revisions(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "skills",
            "quote-attributions",
            str(project_path),
            "--revision",
            first,
            "--revision",
            second,
            "--language",
            "en",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["candidate_count"] == 2
    assert payload["attribution_candidate_count"] == 2
    assert {item["revision_id"] for item in payload["candidates"]} == {first, second}
    assert all(item["attribution_cues"] == ["said"] for item in payload["candidates"])
    assert all(item["attribution_cue_languages"] == ["en"] for item in payload["candidates"])
    assert outside not in {item["revision_id"] for item in payload["candidates"]}


def test_quote_attributions_supports_standard_candidate_row_formats(tmp_path):
    project_path, first, _second, _outside = _project_with_revisions(tmp_path)
    runner = CliRunner()

    jsonl = runner.invoke(
        app,
        [
            "skills",
            "quote-attributions",
            str(project_path),
            "--revision",
            first,
            "--format",
            "jsonl",
        ],
    )
    csv = runner.invoke(
        app,
        [
            "skills",
            "quote-attributions",
            str(project_path),
            "--revision",
            first,
            "--format",
            "csv",
        ],
    )
    table = runner.invoke(
        app,
        [
            "skills",
            "quote-attributions",
            str(project_path),
            "--revision",
            first,
            "--format",
            "table",
        ],
    )

    assert jsonl.exit_code == csv.exit_code == table.exit_code == 0
    assert json.loads(jsonl.stdout.splitlines()[0])["revision_id"] == first
    assert "candidate_id" in csv.stdout and "revision_id" in csv.stdout
    assert first in csv.stdout and "scope_snapshot_id=" in csv.stderr
    assert "candidate_id" in table.stdout and first in table.stdout


def test_quote_attributions_rejects_missing_duplicate_and_unknown_revisions(tmp_path):
    project_path, first, _second, _outside = _project_with_revisions(tmp_path)
    runner = CliRunner()
    base = ["skills", "quote-attributions", str(project_path)]

    missing = runner.invoke(app, base)
    duplicate = runner.invoke(
        app,
        [*base, "--revision", first, "--revision", first],
    )
    unknown = runner.invoke(app, [*base, "--revision", "missing-revision"])

    assert missing.exit_code == 2
    assert "--revision" in missing.output
    assert duplicate.exit_code == 2
    assert "must not be duplicated" in duplicate.output
    assert unknown.exit_code == 2
    assert "unknown document revision" in unknown.output
    assert "Traceback" not in unknown.output


def test_quote_attributions_language_accepts_only_reviewed_choices(tmp_path):
    project_path, first, _second, _outside = _project_with_revisions(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "skills",
            "quote-attributions",
            str(project_path),
            "--revision",
            first,
            "--language",
            "fr",
        ],
    )

    assert result.exit_code == 2
    assert "en" in result.output and "es" in result.output
