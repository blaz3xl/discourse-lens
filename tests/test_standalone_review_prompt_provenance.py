"""Safe prompt provenance for standalone model review runs."""

import hashlib
import json
from typing import cast

import pytest

from discourselens import Project
from discourselens.frameworks import FRAMEWORK_REVIEWS, FrameworkReview
from discourselens.interpretation import (
    FRAMEWORK_REVIEW_PROMPT_TEMPLATE_REF,
    FRAMEWORK_REVIEW_PROMPT_TEMPLATE_VERSION,
    FRAMEWORK_REVIEW_UNTRUSTED_DATA_PREFIX,
    build_framework_review_system_prompt,
    framework_review_prompt_template_sha256,
    review_finding_with_model,
)
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
)
from discourselens.types import EvidenceRole


class _ReviewProvider:
    provider_id = "openai-compatible"

    def __init__(self) -> None:
        self.prompt = ""
        self.system_prompt: str | None = None

    def capabilities(self, model: str) -> ModelCapabilities:
        del model
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.prompt = request.prompt
        self.system_prompt = request.system_prompt
        data = json.loads(
            request.prompt.split(FRAMEWORK_REVIEW_UNTRUSTED_DATA_PREFIX, 1)[1]
        )
        records = cast(list[dict[str, object]], data["evidence_records"])
        record = next(
            item
            for item in records
            if isinstance(item.get("evidence_id"), str)
            and isinstance(item.get("excerpt"), str)
        )
        evidence_id = str(record["evidence_id"])
        response: dict[str, object] = {
            "interpretation": "The wording supports a bounded reading.",
            "alternative_interpretation": "The wording may also be descriptive.",
            "evidence_ids": [evidence_id],
            "counterevidence_ids": [],
            "limitations": ["One supplied passage provides limited context."],
        }
        if request.system_prompt and "Each observation must contain" in request.system_prompt:
            response["observations"] = [
                {
                    "evidence_id": evidence_id,
                    "excerpt": record["excerpt"],
                    "description": "The supplied passage contains an observable feature.",
                }
            ]
        return GenerationResponse(json.dumps(response), self.provider_id, request.model)


@pytest.mark.parametrize(
    ("framework", "expected_prompt_ref"),
    [
        (None, "evidence-bound-review-v1"),
        ("fairclough-text", "fairclough-text-review-v1"),
        ("critical-reviewer", "framework-review-instructions-v4:critical-reviewer"),
    ],
)
def test_standalone_review_run_snapshots_safe_prompt_provenance_only(
    tmp_path, framework: str | None, expected_prompt_ref: str
):
    project = Project.init(tmp_path / (framework or "generic-review"))
    corpus_text = "PRIVATE_CORPUS_SENTINEL: Officials must explain exceptions."
    observation = "PRIVATE_USER_SENTINEL: Observe the use of must."
    secret_reference = "PRIVATE_SECRET_SENTINEL"
    _, revision_id = project.import_text("source.txt", corpus_text)
    source_run = project.create_run(project.snapshot([revision_id]), {"profile": "fixture"})
    evidence_id = project.textual_evidence(
        project.create_span(revision_id, 0, len(corpus_text))
    )
    finding_id = project.create_finding(
        source_run, "fixture", "1.0.0", "modal-language", observation
    )
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(source_run)
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "fixture", secret_reference, "https://fixture.invalid/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("review", "openai-compatible", "fixture", "fixture")
    )
    provider = _ReviewProvider()

    try:
        result = review_finding_with_model(
            project,
            finding_id,
            "review",
            framework=framework,
            provider_factory=lambda *_args: provider,
        )
        run_row = project.db.execute(
            "SELECT config_json FROM analysis_runs WHERE id=?", (result.analysis_run_id,)
        ).fetchone()
        assert run_row is not None
        config_json = str(run_row["config_json"])
        config = json.loads(config_json)
        selected_framework: FrameworkReview | None = (
            FRAMEWORK_REVIEWS[framework] if framework is not None else None
        )
        expected_snapshot = {
            "prompt_template_ref": FRAMEWORK_REVIEW_PROMPT_TEMPLATE_REF,
            "prompt_template_version": str(FRAMEWORK_REVIEW_PROMPT_TEMPLATE_VERSION),
            "prompt_ref": expected_prompt_ref,
            "prompt_template_sha256": framework_review_prompt_template_sha256(
                selected_framework
            ),
        }
        if selected_framework is not None:
            expected_snapshot["framework_id"] = selected_framework.id
            expected_snapshot["framework_version"] = selected_framework.version

        assert config["review_prompt_snapshot"] == expected_snapshot
        assert provider.system_prompt is not None
        assert provider.system_prompt == build_framework_review_system_prompt(
            selected_framework
        )
        assert expected_snapshot["prompt_template_sha256"] == hashlib.sha256(
            (
                provider.system_prompt
                + "\0"
                + FRAMEWORK_REVIEW_UNTRUSTED_DATA_PREFIX
                + "{canonical-research-data-json}"
            ).encode("utf-8")
        ).hexdigest()
        assert provider.prompt not in config_json
        assert provider.system_prompt not in config_json
        assert "PRIVATE_CORPUS_SENTINEL" not in config_json
        assert "PRIVATE_USER_SENTINEL" not in config_json
        assert secret_reference not in config_json
        assert "instruction" not in config["review_prompt_snapshot"]
    finally:
        project.close()
