from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app


def test_textual_descriptors_cli_preserves_multi_revision_scope_order(tmp_path):
    root = tmp_path / "textual-descriptors-multi-revision"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "Officials must publish records.")
    _, second_revision = project.import_text("second.txt", "Residents may request review.")
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "textual-descriptors",
            str(root),
            f"{second_revision},{first_revision}",
            "--languages",
            "en",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    reopened = Project.open(root)
    try:
        metadata = reopened.analysis_run_metadata(payload["analysis_run_id"])
        assert metadata["revision_ids"] == [second_revision, first_revision]
    finally:
        reopened.close()


def test_textual_descriptors_trims_and_forwards_revision_ids_in_order(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "textual-descriptors-trimmed-revisions"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "Officials must publish records.")
    _, second_revision = project.import_text("second.txt", "Residents may request review.")
    project.close()

    captured_revision_ids: list[str] = []

    def capture_core_revision_ids(_instance, revision_ids, _languages):
        captured_revision_ids.extend(revision_ids)
        return {"revision_ids": revision_ids}

    monkeypatch.setattr(
        cli_module, "run_textual_descriptors_analysis", capture_core_revision_ids
    )

    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "textual-descriptors",
            str(root),
            f" {first_revision} , {second_revision} ",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured_revision_ids == [first_revision, second_revision]


@pytest.mark.parametrize(
    "revision_selection",
    ["", "revision-a,", ",revision-a", "revision-a,,revision-b", "revision-a, ,revision-b"],
)
def test_textual_descriptors_rejects_empty_or_blank_revision_ids_before_open(
    tmp_path, monkeypatch: pytest.MonkeyPatch, revision_selection: str
) -> None:
    def unexpected_open(_path) -> Project:
        pytest.fail("invalid textual-descriptor revision IDs must not open a project")

    monkeypatch.setattr(cli_module.Project, "open", staticmethod(unexpected_open))

    result = CliRunner().invoke(
        app,
        ["analyze", "textual-descriptors", str(tmp_path / "unused"), revision_selection],
    )

    assert result.exit_code == 2
    error_text = " ".join(line.strip(" │") for line in result.output.splitlines())
    assert "revision IDs must be nonempty and must not be blank" in error_text


def test_textual_descriptors_rejects_duplicate_revisions_before_open(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_open(_path) -> Project:
        pytest.fail("duplicate textual-descriptor revisions must not open a project")

    monkeypatch.setattr(cli_module.Project, "open", staticmethod(unexpected_open))

    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "textual-descriptors",
            str(tmp_path / "unused"),
            "revision-a,revision-a",
        ],
    )

    assert result.exit_code == 2
    error_text = " ".join(line.strip(" │") for line in result.output.splitlines())
    assert "revision IDs must not be duplicated" in error_text
