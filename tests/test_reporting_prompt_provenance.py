"""Safe built-in prompt metadata in controlled CDA technical appendices."""

import json
from typing import cast

from discourselens import Project
from discourselens.frameworks import FRAMEWORK_REVIEWS
from discourselens.interpretation import framework_review_prompt_template_sha256
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
)
from discourselens.reporting import (
    analysis_payload,
    render_analysis_html,
    render_analysis_markdown,
)
from discourselens.synthesis import SYNTHESIS_SYSTEM_PROMPT, synthesis_prompt_template_sha256


class _FixtureProvider:
    provider_id = "openai-compatible"

    def capabilities(self, model: str) -> ModelCapabilities:
        del model
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        marker = "UNTRUSTED_RESEARCH_DATA_JSON:\n"
        payload = cast(dict[str, object], json.loads(request.prompt.split(marker, 1)[1]))
        evidence_records = cast(list[dict[str, object]], payload["evidence_records"])
        evidence_id = str(evidence_records[0]["evidence_id"])
        response: dict[str, object] = {
            "interpretation": "A bounded fixture interpretation of supplied evidence.",
            "alternative_interpretation": "The evidence may admit another reading.",
            "evidence_ids": [evidence_id],
            "counterevidence_ids": [],
            "limitations": ["Fixture output is limited to supplied records."],
        }
        if (
            "source_findings" not in payload
            and request.system_prompt
            and "Each observation must contain" in request.system_prompt
        ):
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


def _configure_fake_model(project: Project, monkeypatch) -> None:
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "https://fixture.invalid/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    monkeypatch.setattr(
        "discourselens.execution.OpenAICompatibleProvider",
        lambda *_args, **_kwargs: _FixtureProvider(),
    )


def test_controlled_cda_appendix_contains_only_builtin_prompt_provenance(
    tmp_path, monkeypatch
):
    profile = FRAMEWORK_REVIEWS["fairclough-text"]
    framework_digest = framework_review_prompt_template_sha256(profile)
    synthesis_digest = synthesis_prompt_template_sha256()
    corpus_marker = "CORPUS_TEXT_MUST_NOT_APPEAR_IN_PROMPT_PROVENANCE"

    with Project.init(tmp_path / "controlled-cda") as project:
        document_id, revision_id = project.import_text(
            "selected.txt", f"Officials must respond to migration. {corpus_marker}"
        )
        project.set_document_metadata(document_id, "language", "en")
        _configure_fake_model(project, monkeypatch)
        result = project.analyze_controlled_cda(
            [revision_id], "review", frameworks=("fairclough-text",)
        )
        analysis_run_id = str(result["analysis_run_id"])

        markdown = render_analysis_markdown(project, analysis_run_id)
        markdown_appendix = markdown.split("## Technical appendix\n", 1)[1]
        html = render_analysis_html(project, analysis_run_id)
        html_appendix = html.split("<h2>Technical appendix</h2>", 1)[1]

        for appendix in (markdown_appendix, html_appendix):
            assert "fairclough-text-review-v1" in appendix
            assert "1.3.0" in appendix
            assert "builtin.fairclough-text" in appendix
            assert "evidence-bound-framework-review-executor-v2" in appendix
            assert "prompt_template_version" in appendix
            assert "prompt_template_sha256" in appendix
            assert framework_digest in appendix
            assert "evidence-constrained-synthesis-v1" in appendix
            assert "builtin.synthesis" in appendix
            assert synthesis_digest in appendix
            assert profile.instruction not in appendix
            assert SYNTHESIS_SYSTEM_PROMPT not in appendix
            assert corpus_marker not in appendix


def test_custom_run_keeps_config_unchanged_without_prompt_provenance_appendix(tmp_path):
    config = {
        "profile": "custom.fixture",
        "frameworks": ["fairclough-text"],
                "framework_prompt_snapshot_version": 1,
        "framework_prompt_snapshots": [
            {
                "framework_id": "fairclough-text",
                "framework_version": "1.3.0",
                "skill_id": "builtin.fairclough-text",
                "skill_version": "1.3.0",
                "prompt_ref": "fairclough-text-review-v1",
                "prompt_template_sha256": "a" * 64,
            }
        ],
        "synthesis_prompt_snapshot_version": 1,
        "synthesis_prompt_snapshot": {
            "framework_id": "evidence-constrained-synthesis",
            "framework_version": "1.0.0",
            "skill_id": "builtin.synthesis",
            "skill_version": "1.0.0",
            "prompt_ref": "evidence-constrained-synthesis-v1",
            "prompt_template_sha256": "b" * 64,
        },
    }
    with Project.init(tmp_path / "custom-run") as project:
        _, revision_id = project.import_text("custom.txt", "A custom analysis fixture.")
        analysis_run_id = project.create_run(project.snapshot([revision_id]), config)

        payload = analysis_payload(project, analysis_run_id)
        assert payload["analysis_run"]["config"] == config
        markdown = render_analysis_markdown(project, analysis_run_id)
        html = render_analysis_html(project, analysis_run_id)

        assert "Built-in prompt provenance" not in markdown
        assert "<h3>Built-in prompt provenance</h3>" not in html
