from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from typer.testing import CliRunner

import discourselens.analysis as analysis_module
import discourselens.cli as cli_module
import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.analysis import run_batch_saved_controlled_profile
from discourselens.cli import app
from discourselens.errors import ProjectError
from discourselens.mcp_server import run_batch_saved_analysis_profile
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
)


class _FixtureProvider:
    provider_id = "openai-compatible"

    def capabilities(self, model: str) -> ModelCapabilities:
        del model
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload = json.loads(request.prompt.split("UNTRUSTED_RESEARCH_DATA_JSON:\n", 1)[1])
        records = payload["evidence_records"]
        response = {
            "interpretation": "Evidence-bound reading.",
            "alternative_interpretation": "Alternative reading.",
            "evidence_ids": [records[0]["evidence_id"]],
        }
        if "source_findings" not in payload:
            response["observations"] = [
                {
                    "evidence_id": records[0]["evidence_id"],
                    "excerpt": records[0]["excerpt"],
                    "description": "The passage contains an observable wording feature.",
                }
            ]
        return GenerationResponse(
            json.dumps(response),
            self.provider_id,
            request.model,
        )


def _configured_project(path) -> tuple[Project, list[str]]:
    project = Project.init(path)
    revisions = [
        project.import_text("first.txt", "We must act.")[1],
        project.import_text("second.txt", "We must act.")[1],
    ]
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    project.save_analysis_profile(
        "focused",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "default_preset": "review",
        },
    )
    return project, revisions


def _side_effect_counts(project: Project) -> tuple[int, int]:
    runs = project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0]
    scopes = project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
    return int(runs), int(scopes)


def test_saved_profile_batch_creates_one_run_per_revision_scope(tmp_path) -> None:
    project, revisions = _configured_project(tmp_path / "project")
    results = run_batch_saved_controlled_profile(
        project,
        revisions,
        "focused",
        provider_factory=lambda endpoint, reference, provider_id: _FixtureProvider(),
    )

    run_ids = [result["analysis_run_id"] for result in results]
    assert len(run_ids) == len(set(run_ids)) == 2
    run_scopes = {
        run_id: project.scope_revisions(
            str(
                project.db.execute(
                    "SELECT scope_id FROM analysis_runs WHERE id=?", (run_id,)
                ).fetchone()[0]
            )
        )
        for run_id in run_ids
    }
    assert list(run_scopes.values()) == [[revisions[0]], [revisions[1]]]
    project.close()


def test_saved_profile_batch_can_return_partial_results_and_retry_failed_revision(
    monkeypatch, tmp_path
) -> None:
    project, revisions = _configured_project(tmp_path / "project")
    fail_second = True

    preexisting_scope_id = project.snapshot([revisions[1]])
    preexisting_failed_run_id = project.create_run(
        preexisting_scope_id, {"profile": "concurrent-attempt"}
    )
    project.start_run(preexisting_failed_run_id)
    project.fail_run(preexisting_failed_run_id, RuntimeError("another attempt failed"))

    execute_controlled_steps = analysis_module._execute_controlled_cda_steps

    def fail_second_revision_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal fail_second
        target_project = args[0]
        scope_id = args[2]
        if target_project.scope_revisions(scope_id) == [revisions[1]] and fail_second:
            fail_second = False
            raise RuntimeError("provider secret=do-not-return")
        return execute_controlled_steps(*args, **kwargs)

    monkeypatch.setattr(
        analysis_module, "_execute_controlled_cda_steps", fail_second_revision_once
    )
    outcomes = run_batch_saved_controlled_profile(
        project,
        revisions,
        "focused",
        provider_factory=lambda endpoint, reference, provider_id: _FixtureProvider(),
        continue_on_error=True,
    )

    assert len(outcomes) == 2
    completed = outcomes[0]
    assert completed["status"] == "completed"
    assert completed["revision_id"] == revisions[0]
    completed_run_id = completed["result"]["analysis_run_id"]
    assert project.scope_revisions(
        str(
            project.db.execute(
                "SELECT scope_id FROM analysis_runs WHERE id=?", (completed_run_id,)
            ).fetchone()[0]
        )
    ) == [revisions[0]]

    failed = outcomes[1]
    assert failed["status"] == "failed"
    assert failed["revision_id"] == revisions[1]
    assert failed["analysis_run_id"] is not None
    assert failed["analysis_run_id"] != preexisting_failed_run_id
    assert failed["error"] == {
        "type": "RuntimeError",
        "message": "Saved-profile analysis failed for this revision.",
    }
    assert "do-not-return" not in json.dumps(failed)
    failed_run = project.db.execute(
        "SELECT status,scope_id FROM analysis_runs WHERE id=?", (failed["analysis_run_id"],)
    ).fetchone()
    assert failed_run is not None
    assert failed_run["status"] == "FAILED"
    assert project.scope_revisions(str(failed_run["scope_id"])) == [revisions[1]]

    retried = run_batch_saved_controlled_profile(
        project,
        [failed["revision_id"]],
        "focused",
        provider_factory=lambda endpoint, reference, provider_id: _FixtureProvider(),
        continue_on_error=True,
    )
    assert len(retried) == 1
    assert retried[0]["status"] == "completed"
    assert retried[0]["revision_id"] == revisions[1]
    project.close()


@pytest.mark.parametrize(
    "revision_ids",
    [[], ["revision-a", "revision-a"], ["revision-a", "missing-revision"]],
)
@pytest.mark.parametrize("continue_on_error", [False, True])
def test_saved_profile_batch_rejects_invalid_selection_before_any_writes(
    tmp_path, revision_ids, continue_on_error
) -> None:
    project, revisions = _configured_project(tmp_path / "project")
    if revision_ids and revision_ids[0] == "revision-a":
        revision_ids = [revisions[0], *revision_ids[1:]]
    before = _side_effect_counts(project)

    with pytest.raises(ProjectError):
        run_batch_saved_controlled_profile(
            project,
            revision_ids,
            "focused",
            continue_on_error=continue_on_error,
        )

    assert _side_effect_counts(project) == before
    project.close()


def test_sdk_cli_and_mcp_delegate_to_the_same_batch_core(monkeypatch, tmp_path) -> None:
    project, _ = _configured_project(tmp_path / "project")
    calls: list[tuple[object, list[str], str, str | None, object, bool]] = []

    def fake_core(
        target_project: object,
        revision_ids: list[str],
        profile_name: str,
        *,
        preset_name: str | None = None,
        provider_factory: object = None,
        continue_on_error: bool = False,
    ) -> list[Any]:
        calls.append(
            (
                target_project,
                revision_ids,
                profile_name,
                preset_name,
                provider_factory,
                continue_on_error,
            )
        )
        return []

    monkeypatch.setattr(analysis_module, "run_batch_saved_controlled_profile", fake_core)
    assert project.run_batch_saved_analysis_profile(
        ["revision-a", "revision-b"], "focused", preset_name="custom"
    ) == []
    assert calls[-1][1:] == (
        ["revision-a", "revision-b"],
        "focused",
        "custom",
        None,
        False,
    )

    assert project.run_batch_saved_analysis_profile(
        ["revision-b"], "focused", continue_on_error=True
    ) == []
    assert calls[-1][1:] == (["revision-b"], "focused", None, None, True)

    monkeypatch.setattr(cli_module, "run_batch_saved_controlled_profile", fake_core)
    default_cli_result = CliRunner().invoke(
        app,
        ["profile", "batch", str(tmp_path / "project"), "focused", "revision-a,revision-b"],
    )
    assert default_cli_result.exit_code == 0, default_cli_result.output
    assert json.loads(default_cli_result.output) == []
    assert calls[-1][1:] == (
        ["revision-a", "revision-b"],
        "focused",
        None,
        None,
        False,
    )

    cli_result = CliRunner().invoke(
        app,
        [
            "profile",
            "batch",
            str(tmp_path / "project"),
            "focused",
            "revision-a,revision-b",
            "--preset",
            "custom",
            "--continue-on-error",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert json.loads(cli_result.output) == []
    assert calls[-1][1:] == (
        ["revision-a", "revision-b"],
        "focused",
        "custom",
        None,
        True,
    )

    monkeypatch.setattr(mcp_module, "_open_project", lambda: project)
    monkeypatch.setattr(mcp_module, "run_batch_saved_controlled_profile", fake_core)
    assert run_batch_saved_analysis_profile(
        ["revision-a", "revision-b"], "focused", preset_name="custom"
    ) == []
    assert calls[-1][1:] == (
        ["revision-a", "revision-b"],
        "focused",
        "custom",
        None,
        False,
    )

    assert run_batch_saved_analysis_profile(
        ["revision-a", "revision-b"],
        "focused",
        preset_name="custom",
        continue_on_error=True,
    ) == []
    assert calls[-1][1:] == (
        ["revision-a", "revision-b"],
        "focused",
        "custom",
        None,
        True,
    )
    registered_tool_names = {tool.name for tool in asyncio.run(mcp_module.server.list_tools())}
    assert "run_batch_saved_analysis_profile" in registered_tool_names
    project.close()


def test_cli_partial_batch_failure_keeps_json_and_returns_nonzero(monkeypatch, tmp_path) -> None:
    root = tmp_path / "partial-batch-cli"
    project, revisions = _configured_project(root)
    project.close()
    outcomes = [
        {
            "revision_id": revisions[0],
            "status": "completed",
            "result": {"analysis_run_id": "run-complete"},
        },
        {
            "revision_id": revisions[1],
            "status": "failed",
            "analysis_run_id": "run-failed",
            "error": {"type": "RuntimeError", "message": "analysis failed"},
        },
    ]

    def fake_core(*args: object, **kwargs: object) -> list[dict[str, object]]:
        del args, kwargs
        return outcomes

    monkeypatch.setattr(cli_module, "run_batch_saved_controlled_profile", fake_core)
    result = CliRunner().invoke(
        app,
        [
            "profile",
            "batch",
            str(root),
            "focused",
            ",".join(revisions),
            "--continue-on-error",
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.output) == outcomes
