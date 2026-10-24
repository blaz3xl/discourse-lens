"""Prompt provenance and resume checks for controlled CDA runs."""

import json
from typing import cast

import pytest

import discourselens.interpretation as interpretation_module
import discourselens.synthesis as synthesis_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.frameworks import FRAMEWORK_REVIEWS, FrameworkReview
from discourselens.interpretation import (
    FRAMEWORK_REVIEW_PROMPT_TEMPLATE_REF,
    FRAMEWORK_REVIEW_PROMPT_TEMPLATE_VERSION,
    framework_review_prompt_template_sha256,
)
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
)
from discourselens.synthesis import SYNTHESIS_SYSTEM_PROMPT, synthesis_prompt_template_sha256


class _FixtureProvider:
    provider_id = "openai-compatible"

    def __init__(self, *, fail_first_synthesis: bool = False) -> None:
        self.fail_first_synthesis = fail_first_synthesis
        self.failed_synthesis = False
        self.payloads: list[dict[str, object]] = []

    def capabilities(self, model: str) -> ModelCapabilities:
        del model
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        marker = "UNTRUSTED_RESEARCH_DATA_JSON:\n"
        payload = cast(dict[str, object], json.loads(request.prompt.split(marker, 1)[1]))
        self.payloads.append(payload)
        is_synthesis = "source_findings" in payload
        if is_synthesis and self.fail_first_synthesis and not self.failed_synthesis:
            self.failed_synthesis = True
            raise RuntimeError("fixture synthesis interruption")

        evidence_records = cast(list[dict[str, object]], payload["evidence_records"])
        evidence_id = str(evidence_records[0]["evidence_id"])
        response: dict[str, object] = {
            "interpretation": "A bounded fixture interpretation of the supplied evidence.",
            "alternative_interpretation": "The evidence may also admit a neutral reading.",
            "evidence_ids": [evidence_id],
            "counterevidence_ids": [],
            "limitations": ["Fixture output is limited to the supplied records."],
        }
        if not is_synthesis and request.system_prompt and "Each observation must contain" in request.system_prompt:
            observation_record = next(
                item
                for item in evidence_records
                if isinstance(item.get("evidence_id"), str)
                and isinstance(item.get("excerpt"), str)
            )
            response["observations"] = [
                {
                    "evidence_id": observation_record["evidence_id"],
                    "excerpt": observation_record["excerpt"],
                    "description": "The passage contains a modal marker.",
                }
            ]
        return GenerationResponse(json.dumps(response), self.provider_id, request.model)


def _configure_fake_model(project: Project, monkeypatch, provider: _FixtureProvider) -> None:
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "https://fixture.invalid/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    monkeypatch.setattr(
        "discourselens.execution.OpenAICompatibleProvider",
        lambda *_args, **_kwargs: provider,
    )


def _fail_at_synthesis(tmp_path, monkeypatch) -> tuple[Project, _FixtureProvider, str]:
    project = Project.init(tmp_path / "project")
    document_id, revision_id = project.import_text(
        "selected.txt", "Officials must respond to migration. The committee should explain exceptions."
    )
    project.set_document_metadata(document_id, "language", "en")
    provider = _FixtureProvider(fail_first_synthesis=True)
    _configure_fake_model(project, monkeypatch, provider)
    with pytest.raises(RuntimeError, match="fixture synthesis interruption"):
        project.analyze_controlled_cda([revision_id], "review", frameworks=("fairclough-text",))
    run = project.db.execute(
        "SELECT id,status FROM analysis_runs ORDER BY started_at DESC,id DESC LIMIT 1"
    ).fetchone()
    assert run is not None
    assert run["status"] == "FAILED"
    return project, provider, str(run["id"])


def test_controlled_cda_persists_digest_only_prompt_snapshot_and_rejects_changed_content(
    tmp_path, monkeypatch
):
    project, _provider, analysis_run_id = _fail_at_synthesis(tmp_path, monkeypatch)
    try:
        run_metadata = project.analysis_run_metadata(analysis_run_id)["analysis_run"]
        assert isinstance(run_metadata, dict)
        config = run_metadata["config"]
        assert isinstance(config, dict)
        raw_framework_snapshots = config["framework_prompt_snapshots"]
        assert isinstance(raw_framework_snapshots, list) and raw_framework_snapshots
        snapshot = raw_framework_snapshots[0]
        profile = FRAMEWORK_REVIEWS["fairclough-text"]

        assert config["framework_prompt_snapshot_version"] == 2
        assert snapshot == {
            "framework_id": "fairclough-text",
            "framework_version": profile.version,
            "skill_id": "builtin.fairclough-text",
            "skill_version": profile.version,
            "prompt_ref": "fairclough-text-review-v1",
            "prompt_template_ref": FRAMEWORK_REVIEW_PROMPT_TEMPLATE_REF,
            "prompt_template_version": str(FRAMEWORK_REVIEW_PROMPT_TEMPLATE_VERSION),
            "prompt_template_sha256": framework_review_prompt_template_sha256(profile),
        }
        assert config["synthesis_prompt_snapshot_version"] == 2
        assert config["synthesis_prompt_snapshot"] == {
            "framework_id": "evidence-constrained-synthesis",
            "framework_version": "1.0.0",
            "skill_id": "builtin.synthesis",
            "skill_version": "1.0.0",
            "prompt_ref": "evidence-constrained-synthesis-v1",
            "prompt_template_sha256": synthesis_prompt_template_sha256(),
        }
        assert "instruction" not in snapshot
        assert profile.instruction not in json.dumps(config)
        assert SYNTHESIS_SYSTEM_PROMPT not in json.dumps(config)

        monkeypatch.setitem(
            FRAMEWORK_REVIEWS,
            "fairclough-text",
            FrameworkReview(
                profile.id,
                profile.version,
                profile.instruction + " Changed built-in instruction.",
                profile.methodology_ref,
                profile.prompt_ref,
            ),
        )
        with pytest.raises(ProjectError, match="prompt snapshots differ from the current built-in instructions"):
            project.resume_controlled_cda(analysis_run_id)

        monkeypatch.setitem(FRAMEWORK_REVIEWS, "fairclough-text", profile)
        monkeypatch.setattr(
            interpretation_module,
            "_FRAMEWORK_REVIEW_SHARED_SYSTEM_POLICY",
            interpretation_module._FRAMEWORK_REVIEW_SHARED_SYSTEM_POLICY
            + " Changed shared safety instruction.",
        )
        with pytest.raises(ProjectError, match="prompt snapshots differ from the current built-in instructions"):
            project.resume_controlled_cda(analysis_run_id)

        monkeypatch.setattr(
            interpretation_module,
            "_FRAMEWORK_REVIEW_SHARED_SYSTEM_POLICY",
            interpretation_module._FRAMEWORK_REVIEW_SHARED_SYSTEM_POLICY.removesuffix(
                " Changed shared safety instruction."
            ),
        )
        monkeypatch.setattr(
            synthesis_module,
            "SYNTHESIS_SYSTEM_PROMPT",
            SYNTHESIS_SYSTEM_PROMPT + " Changed built-in synthesis instructions.",
        )
        with pytest.raises(
            ProjectError,
            match="synthesis prompt snapshot differs from the current built-in instructions",
        ):
            project.resume_controlled_cda(analysis_run_id)

        assert (
            project.db.execute(
                "SELECT status FROM analysis_runs WHERE id=?", (analysis_run_id,)
            ).fetchone()["status"]
            == "FAILED"
        )
    finally:
        project.close()


def test_controlled_cda_resume_accepts_unchanged_prompt_snapshot(tmp_path, monkeypatch):
    project, provider, analysis_run_id = _fail_at_synthesis(tmp_path, monkeypatch)
    try:
        result = project.resume_controlled_cda(analysis_run_id)
        assert result["analysis_run_id"] == analysis_run_id
        assert result["synthesis_finding_id"] is not None
        assert (
            project.db.execute(
                "SELECT status FROM analysis_runs WHERE id=?", (analysis_run_id,)
            ).fetchone()["status"]
            == "COMPLETED"
        )
        assert provider.failed_synthesis
    finally:
        project.close()
