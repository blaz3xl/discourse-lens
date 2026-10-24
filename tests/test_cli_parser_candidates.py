from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import build_parsed_token_artifact


class _ParsedFixture:
    language = "en"
    backend_id = "fixture"
    model_id = "fixture-v1"

    def parse(self, text: str) -> list[ParsedToken]:
        del text
        return [
            ParsedToken("Minister", "minister", "NOUN", None, 1, "nsubj", 0, 8),
            ParsedToken("speaks", "speak", "VERB", None, 1, "ROOT", 9, 15),
        ]


class _ProjectStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], list[str]]] = []
        self.closed = False

    def analyze_social_actor_candidates(
        self, revision_ids: list[str], artifact_ids: list[str]
    ) -> dict[str, object]:
        self.calls.append(("social", revision_ids, artifact_ids))
        return {
            "analysis_run_id": "analysis-run",
            "skill_run_id": "skill-run",
            "finding_ids": ["finding-a"],
        }

    def analyze_agency_candidates(
        self, revision_ids: list[str], artifact_ids: list[str]
    ) -> dict[str, object]:
        self.calls.append(("agency", revision_ids, artifact_ids))
        return {
            "analysis_run_id": "analysis-run",
            "skill_run_id": "skill-run",
            "finding_ids": ["finding-a"],
        }

    def close(self) -> None:
        self.closed = True


def _install_project_stub(monkeypatch: pytest.MonkeyPatch, stub: _ProjectStub) -> list[str]:
    opened: list[str] = []

    def open_project(path) -> _ProjectStub:
        opened.append(str(path))
        return stub

    monkeypatch.setattr(cli_module.Project, "open", staticmethod(open_project))
    return opened


@pytest.mark.parametrize(
    ("command", "method"),
    [
        ("social-actor-candidates", "social"),
        ("agency-candidates", "agency"),
    ],
)
def test_parser_candidate_commands_forward_repeated_ids_and_emit_json(
    monkeypatch, command, method
):
    stub = _ProjectStub()
    _install_project_stub(monkeypatch, stub)

    result = CliRunner().invoke(
        app,
        [
            "analyze",
            command,
            "research",
            "--revision",
            "revision-a",
            "--revision",
            "revision-b",
            "--artifact",
            "artifact-a",
            "--artifact",
            "artifact-b",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "analysis_run_id": "analysis-run",
        "skill_run_id": "skill-run",
        "finding_ids": ["finding-a"],
    }
    assert stub.calls == [
        (method, ["revision-a", "revision-b"], ["artifact-a", "artifact-b"])
    ]
    assert stub.closed


@pytest.mark.parametrize("command", ["social-actor-candidates", "agency-candidates"])
def test_parser_candidate_commands_require_nonempty_selections_before_open(
    monkeypatch, command
):
    stub = _ProjectStub()
    opened = _install_project_stub(monkeypatch, stub)
    runner = CliRunner()
    base = ["analyze", command, "research"]

    missing_revision = runner.invoke(
        app,
        [*base, "--artifact", "artifact-a"],
    )
    missing_artifact = runner.invoke(
        app,
        [*base, "--revision", "revision-a"],
    )
    blank_revision = runner.invoke(
        app,
        [*base, "--revision", "  ", "--artifact", "artifact-a"],
    )
    duplicate_artifact = runner.invoke(
        app,
        [
            *base,
            "--revision",
            "revision-a",
            "--artifact",
            "artifact-a",
            "--artifact",
            "artifact-a",
        ],
    )

    assert missing_revision.exit_code == 2
    assert "--revision" in missing_revision.output
    assert missing_artifact.exit_code == 2
    assert "--artifact" in missing_artifact.output
    assert blank_revision.exit_code == 2
    assert "revision IDs must be nonempty" in blank_revision.output
    assert duplicate_artifact.exit_code == 2
    assert "artifact IDs must not be duplicated" in duplicate_artifact.output
    assert opened == []


def test_parser_candidate_scope_mismatch_fails_before_creating_analysis_run(tmp_path):
    project_path = tmp_path / "scope-mismatch"
    project = Project.init(project_path)
    _, selected_revision = project.import_text("selected.txt", "Minister speaks.")
    _, outside_revision = project.import_text("outside.txt", "Minister speaks.")
    outside_artifact = build_parsed_token_artifact(project, outside_revision, _ParsedFixture())
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "social-actor-candidates",
            str(project_path),
            "--revision",
            selected_revision,
            "--artifact",
            outside_artifact.artifact_id,
        ],
    )

    assert result.exit_code == 2
    assert "outside the explicit scope" in result.output
    assert "Traceback" not in result.output

    reopened = Project.open(project_path)
    try:
        assert reopened.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == 0
        assert reopened.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
    finally:
        reopened.close()
