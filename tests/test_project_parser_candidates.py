import json

import pytest

import discourselens.analysis as analysis_module
from discourselens import Project
from discourselens.errors import ProjectError, ScopeViolationError, SkillManifestError
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import build_parsed_token_artifact, build_surface_token_artifact


class _CandidateBackend:
    language = "en"
    backend_id = "fixture"
    model_id = "fixture-v1"

    def parse(self, text: str) -> list[ParsedToken]:
        return [
            ParsedToken("Minister", "minister", "NOUN", None, 1, "nsubj", 0, 8),
            ParsedToken("speaks", "speak", "VERB", None, 1, "ROOT", 9, 15),
        ]


def _supported_dependency_parse_preflight(_languages, _required, *, routes=None):
    assert routes is not None
    return {
        language: {
            "runnable": True,
            "limitations": [],
            "required_capabilities": ["dependency_parse"],
            "statuses": {"dependency_parse": "SUPPORTED"},
            "backend": route["backend"],
            "model": route["model"],
            "model_dir_env": None,
            "route_source": "profile",
        }
        for language, route in routes.items()
    }


def test_project_social_actor_candidates_exposes_analysis_lineage(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("statement.txt", "Minister speaks.")
    artifact = build_parsed_token_artifact(project, revision, _CandidateBackend())

    result = project.analyze_social_actor_candidates([revision], [artifact.artifact_id])

    assert set(result) == {"analysis_run_id", "skill_run_id", "finding_ids"}
    metadata = project.analysis_run_summary(result["analysis_run_id"])
    run = metadata["analysis_run"]
    assert isinstance(run, dict)
    run_config = run["config"]
    assert isinstance(run_config, dict)
    assert run["status"] == "COMPLETED"
    assert project.scope_revisions(run["scope_id"]) == [revision]
    assert run_config["artifact_ids"] == [artifact.artifact_id]
    skill_run = project.db.execute(
        "SELECT analysis_run_id,scope_id,status FROM skill_runs WHERE id=?",
        (result["skill_run_id"],),
    ).fetchone()
    assert skill_run["analysis_run_id"] == result["analysis_run_id"]
    assert skill_run["scope_id"] == run["scope_id"]
    assert skill_run["status"] == "COMPLETED"
    assert result["finding_ids"]
    assert all(
        project.get_finding(finding_id)["analysis_run_id"] == result["analysis_run_id"]
        for finding_id in result["finding_ids"]
    )


def test_project_agency_candidates_preserves_dependency_parse_preflight(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "project")
    document_id, revision = project.import_text("statement.txt", "Minister speaks.")
    project.set_document_metadata(document_id, "language", "en")
    artifact = build_parsed_token_artifact(project, revision, _CandidateBackend())
    monkeypatch.setattr(
        analysis_module, "local_execution_preflight", _supported_dependency_parse_preflight
    )

    result = project.analyze_agency_candidates([revision], [artifact.artifact_id])

    metadata = project.analysis_run_summary(result["analysis_run_id"])
    run = metadata["analysis_run"]
    assert isinstance(run, dict)
    run_config = run["config"]
    assert isinstance(run_config, dict)
    preflight = run_config["agency_dependency_parse_preflight"]
    assert isinstance(preflight, dict)
    dependency_snapshot = preflight[revision]
    assert isinstance(dependency_snapshot, dict)
    assert dependency_snapshot["status"] == "SUPPORTED"
    assert dependency_snapshot["artifact_id"] == artifact.artifact_id
    assert run["status"] == "COMPLETED"
    assert result["finding_ids"]


def test_project_candidate_selection_rejects_scope_mismatch_before_snapshot_or_run(tmp_path):
    project = Project.init(tmp_path / "project")
    _, selected_revision = project.import_text("selected.txt", "Minister speaks.")
    _, outside_revision = project.import_text("outside.txt", "Minister speaks.")
    outside_artifact = build_parsed_token_artifact(
        project, outside_revision, _CandidateBackend()
    )

    with pytest.raises(ScopeViolationError, match="outside the explicit scope"):
        project.analyze_social_actor_candidates(
            [selected_revision], [outside_artifact.artifact_id]
        )

    assert project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


@pytest.mark.parametrize(
    "invalid_selection",
    [
        "empty_revisions",
        "blank_revision",
        "duplicate_revisions",
        "unknown_revision",
        "empty_artifacts",
        "blank_artifact",
        "duplicate_artifacts",
        "unknown_artifact",
        "wrong_artifact_type",
    ],
)
def test_project_candidate_selection_rejects_invalid_ids_before_snapshot_or_run(
    tmp_path, invalid_selection
):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("statement.txt", "Minister speaks.")
    artifact = build_parsed_token_artifact(project, revision, _CandidateBackend())
    selections = {
        "empty_revisions": ([], [artifact.artifact_id]),
        "blank_revision": (["  "], [artifact.artifact_id]),
        "duplicate_revisions": ([revision, revision], [artifact.artifact_id]),
        "unknown_revision": (["missing-revision"], [artifact.artifact_id]),
        "empty_artifacts": ([revision], []),
        "blank_artifact": ([revision], ["  "]),
        "duplicate_artifacts": ([revision], [artifact.artifact_id, artifact.artifact_id]),
        "unknown_artifact": ([revision], ["missing-artifact"]),
        "wrong_artifact_type": (
            [revision],
            [build_surface_token_artifact(project, revision).artifact_id],
        ),
    }

    with pytest.raises(ProjectError):
        project.analyze_social_actor_candidates(*selections[invalid_selection])

    assert project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


def test_project_agency_candidates_fails_when_dependency_parse_is_not_supported(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "project")
    document_id, revision = project.import_text("statement.txt", "Minister speaks.")
    project.set_document_metadata(document_id, "language", "en")
    artifact = build_parsed_token_artifact(project, revision, _CandidateBackend())

    def unsupported_dependency_parse_preflight(_languages, _required, *, routes=None):
        assert routes is not None
        return {
            language: {
                "runnable": False,
                "limitations": ["dependency_parse: UNSUPPORTED"],
                "required_capabilities": ["dependency_parse"],
                "statuses": {"dependency_parse": "UNSUPPORTED"},
                "backend": route["backend"],
                "model": route["model"],
                "model_dir_env": None,
                "route_source": "profile",
            }
            for language, route in routes.items()
        }

    monkeypatch.setattr(
        analysis_module, "local_execution_preflight", unsupported_dependency_parse_preflight
    )

    with pytest.raises(SkillManifestError, match="dependency_parse=UNSUPPORTED"):
        project.analyze_agency_candidates([revision], [artifact.artifact_id])

    failed_run = project.db.execute(
        "SELECT status,config_json FROM analysis_runs ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    assert failed_run["status"] == "FAILED"
    failed_config = json.loads(failed_run["config_json"])
    assert isinstance(failed_config, dict)
    failed_preflight = failed_config["agency_dependency_parse_preflight"]
    assert isinstance(failed_preflight, dict)
    failed_snapshot = failed_preflight[revision]
    assert isinstance(failed_snapshot, dict)
    assert failed_snapshot["status"] == "UNSUPPORTED"
