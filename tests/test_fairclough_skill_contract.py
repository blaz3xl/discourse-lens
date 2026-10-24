"""Focused offline contract checks for the Fairclough review profile."""

import json

import pytest

from discourselens import Project
from discourselens.errors import EvidenceValidationError
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
)
from discourselens.skills import SkillContext, run_framework_review
from discourselens.types import ClaimType, EvidenceRole


class _FaircloughFixtureProvider:
    provider_id = "openai-compatible"

    def __init__(self, *, invalid: str | None = None) -> None:
        self.invalid = invalid
        self.system_prompt = ""

    def capabilities(self, model: str) -> ModelCapabilities:
        del model
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.system_prompt = request.system_prompt or ""
        marker = "UNTRUSTED_RESEARCH_DATA_JSON:\n"
        envelope = json.loads(request.prompt.split(marker, 1)[1])
        record = next(
            item
            for item in envelope["evidence_records"]
            if item["evidence_kind"] == "TEXTUAL"
        )
        evidence_id = record["evidence_id"]
        excerpt = record["excerpt"]
        if self.invalid == "unknown-evidence":
            evidence_id = "outside-scope-evidence-id"
        elif self.invalid == "wrong-excerpt":
            excerpt = "This wording is not in the source."
        payload = {
            "observations": [
                {
                    "evidence_id": evidence_id,
                    "excerpt": excerpt,
                    "description": "The modal verb 'must' marks an obligation.",
                }
            ],
            "interpretation": "The wording presents the response as obligatory.",
            "alternative_interpretation": "The obligation may reflect a routine procedural requirement.",
            "evidence_ids": [evidence_id],
            "counterevidence_ids": [],
            "limitations": ["This is a targeted reading of one passage."],
        }
        return GenerationResponse(json.dumps(payload), self.provider_id, request.model)


def _review_fixture(tmp_path, provider: _FaircloughFixtureProvider):
    project = Project.init(tmp_path / "project")
    _, selected_revision = project.import_text(
        "selected.txt", "Officials must respond to the request."
    )
    _, outside_revision = project.import_text(
        "outside.txt", "OUTSIDE MARKER: Officials must respond to the request."
    )
    scope_id = project.snapshot([selected_revision])
    source_run = project.create_run(scope_id, {"profile": "fairclough-contract-fixture"})
    span_id = project.create_span(selected_revision, 0, len("Officials must respond to the request."))
    source_evidence_id = project.textual_evidence(span_id)
    source_finding_id = project.create_finding(
        source_run,
        "fixture",
        "1.0.0",
        "modality",
        "The passage contains a modal expression.",
        claim_types=[ClaimType.OBSERVATIONAL],
    )
    project.attach_evidence(source_finding_id, source_evidence_id, EvidenceRole.EXAMPLE)
    project.finalize_run(source_run)
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "https://fixture.invalid/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    result = run_framework_review(
        # The review run is created by the executor, retaining the source scope.
        SkillContext(
            project, project.create_run(scope_id, {"profile": "review"}), scope_id
        ),
        source_finding_id,
        "review",
        "fairclough-text",
        provider_factory=lambda *_args: provider,
    )
    return project, result, provider, selected_revision, outside_revision, scope_id


def test_fairclough_keeps_observation_interpretation_and_alternative_distinct(tmp_path):
    provider = _FaircloughFixtureProvider()
    project, result, provider, selected_revision, outside_revision, scope_id = _review_fixture(
        tmp_path, provider
    )
    finding = project.get_finding(result.finding_id)

    assert finding["observation"].startswith("Core-validated verbatim textual excerpts:")
    assert "Officials must respond to the request." in finding["observation"]
    assert "modal verb 'must' marks an obligation" not in finding["observation"]
    assert finding["interpretation"] == "The wording presents the response as obligatory."
    assert finding["alternative_interpretation"] == (
        "The obligation may reflect a routine procedural requirement."
    )
    assert "Do not infer author intent" in provider.system_prompt
    assert "audience effects" in provider.system_prompt
    assert all(
        forbidden not in finding["interpretation"].casefold()
        for forbidden in ("the author intended", "audiences will", "readers will")
    )

    evidence = project.finding_evidence(result.finding_id)
    cited_revision_ids = {
        project.db.execute(
            "SELECT revision_id FROM source_spans WHERE id=?", (item["span_id"],)
        ).fetchone()["revision_id"]
        for item in evidence
        if item["span_id"]
    }
    assert cited_revision_ids == {selected_revision}
    assert outside_revision not in project.scope_revisions(scope_id)
    claim_types = project.db.execute(
        "SELECT claim_types_json FROM findings WHERE id=?", (result.finding_id,)
    ).fetchone()["claim_types_json"]
    assert json.loads(claim_types) == ["INTERPRETIVE"]


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("unknown-evidence", "unknown evidence ID"),
        ("wrong-excerpt", "does not exactly match"),
    ],
)
def test_fairclough_rejects_observation_not_bound_to_scoped_evidence(
    tmp_path, invalid, message
):
    provider = _FaircloughFixtureProvider(invalid=invalid)
    with pytest.raises(EvidenceValidationError, match=message):
        _review_fixture(tmp_path, provider)
