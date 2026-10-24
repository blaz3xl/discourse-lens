import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource

from discourselens import Project
from discourselens.bundle import export_bundle
from discourselens.errors import EvidenceValidationError, ScopeViolationError, SkillManifestError
from discourselens.execution import snapshot_model_preset
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
)
from discourselens.reporting import analysis_payload, analysis_payload_page, finding_payload
from discourselens.skills import (
    BUILTIN_SKILLS,
    SkillContext,
    run_agency_candidates,
    run_counterevidence_lexical,
    run_evidence_extraction,
    run_framework_review,
    run_quote_attribution_candidates,
    run_scope_question_candidates,
    run_social_actor_candidates,
    run_synthesis,
    run_textual_descriptors,
    skill_resource_text,
)
from discourselens.types import ClaimType, EvidenceRole

SCHEMA_DIR = Path(__file__).parents[1] / "schemas"
PUBLIC_SCHEMA_NAMES = (
    "evidence-reference-v1.json",
    "evidence-reference-v2.json",
    "coverage-disclosure-v1.json",
    "finding-report-v1.json",
    "finding-report-v2.json",
    "analysis-run-v1.json",
    "analysis-report-v1.json",
    "analysis-report-v2.json",
    "research-bundle-v11.json",
    "research-bundle-v12.json",
    "research-bundle-v13.json",
    "research-bundle-v14.json",
    "research-bundle-v15.json",
    "research-bundle-v16.json",
    "skill-manifest-v0.1.json",
    "counterevidence-lexical-input-v1.json",
    "agency-candidates-input-v1.json",
    "evidence-bound-review-input-v1.json",
    "evidence-bound-review-executor-input-v1.json",
    "evidence-bound-review-executor-output-v1.json",
    "evidence-bound-review-model-input-v1.json",
    "evidence-bound-review-model-input-v2.json",
    "evidence-bound-review-model-output-v1.json",
    "framework-review-v1.json",
    "quote-attribution-candidates-output-v1.json",
    "evidence-extraction-input-v1.json",
    "scope-question-candidates-input-v1.json",
    "social-actor-candidates-input-v1.json",
    "synthesis-input-v1.json",
    "synthesis-executor-input-v1.json",
    "synthesis-executor-output-v1.json",
    "synthesis-model-input-v1.json",
    "textual-descriptors-input-v1.json",
    "van-dijk-group-comparison-input-v1.json",
    "van-dijk-group-comparison-output-v1.json",
)


def _public_schemas() -> tuple[dict[str, dict[str, object]], Registry]:
    schemas = {
        name: json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
        for name in PUBLIC_SCHEMA_NAMES
    }
    registry = Registry()
    for schema in schemas.values():
        schema_id = schema.get("$id")
        if isinstance(schema_id, str):
            registry = registry.with_resource(schema_id, Resource.from_contents(schema))
    return schemas, registry


def _validate(instance: object, schema: dict[str, object], registry: Registry) -> None:
    Draft202012Validator(schema, registry=registry).validate(instance)


def test_runtime_builtin_skill_manifests_conform_to_published_schema():
    schemas, registry = _public_schemas()
    schema = schemas["skill-manifest-v0.1.json"]

    for manifest in BUILTIN_SKILLS.manifests():
        _validate(manifest.model_dump(mode="json"), schema, registry)


def test_counterevidence_input_schema_matches_runtime_blank_phrase_rejection(tmp_path):
    schemas, registry = _public_schemas()
    project = Project.init(tmp_path / "project")
    try:
        context = SkillContext(project, "unused-run", "unused-scope")
        with pytest.raises(SkillManifestError):
            run_counterevidence_lexical(context, "unused-finding", " \t ")

        with pytest.raises(ValidationError):
            _validate(
                {"finding_id": "unused-finding", "phrase": " \t "},
                schemas["counterevidence-lexical-input-v1.json"],
                registry,
            )
    finally:
        project.close()


def test_textual_descriptor_input_schema_rejects_empty_language_selection(tmp_path):
    schemas, registry = _public_schemas()
    project = Project.init(tmp_path / "project")
    try:
        with pytest.raises(SkillManifestError):
            run_textual_descriptors(SkillContext(project, "unused-run", "unused-scope"), ())

        with pytest.raises(ValidationError):
            _validate(
                {"scope_id": "scope-fixture", "languages": []},
                schemas["textual-descriptors-input-v1.json"],
                registry,
            )
    finally:
        project.close()


def test_evidence_extraction_schema_allows_runtime_defaulted_options(tmp_path):
    schemas, registry = _public_schemas()
    project = Project.init(tmp_path / "project")
    try:
        _, revision_id = project.import_text("speech.txt", "A short fixture.")
        scope_id = project.snapshot([revision_id])
        run_id = project.create_run(scope_id, {"profile": "defaulted-input-schema-fixture"})
        result = run_evidence_extraction(SkillContext(project, run_id, scope_id))

        assert result["selected_count"] == 0
        _validate(
            {"scope_snapshot_id": scope_id},
            schemas["evidence-extraction-input-v1.json"],
            registry,
        )
    finally:
        project.close()


def test_quote_attribution_output_schema_accepts_runtime_straight_quote_markers(tmp_path):
    schemas, registry = _public_schemas()
    project = Project.init(tmp_path / "project")
    try:
        _, revision_id = project.import_text("speech.txt", 'Officials said "Leave now."')
        scope_id = project.snapshot([revision_id])
        run_id = project.create_run(scope_id, {"profile": "quote-schema-fixture"})
        result = run_quote_attribution_candidates(SkillContext(project, run_id, scope_id))

        assert result["candidates"][0]["opening_marker"] == '"'
        schema_name = "quote-attribution-candidates-output-v1.json"
        _validate(result, schemas[schema_name], registry)

        runtime_schema = json.loads(skill_resource_text(f"schemas/{schema_name}"))
        _validate(result, runtime_schema, registry)
    finally:
        project.close()


def test_scope_question_schema_rejects_runtime_blank_question(tmp_path):
    schemas, registry = _public_schemas()
    project = Project.init(tmp_path / "project")
    try:
        context = SkillContext(project, "unused-run", "unused-scope")
        with pytest.raises(SkillManifestError):
            run_scope_question_candidates(context, "fairclough-text", " \t ")

        with pytest.raises(ValidationError):
            _validate(
                {
                    "framework_id": "fairclough-text",
                    "research_question": " \t ",
                    "scope_snapshot_id": "scope-fixture",
                },
                schemas["scope-question-candidates-input-v1.json"],
                registry,
            )
    finally:
        project.close()


def test_evidence_bound_review_schema_rejects_runtime_blank_framework(tmp_path):
    schemas, registry = _public_schemas()
    project = Project.init(tmp_path / "project")
    try:
        with pytest.raises(SkillManifestError):
            run_framework_review(
                SkillContext(project, "unused-run", "unused-scope"),
                "unused-finding",
                "unused-preset",
                " \t ",
            )

        with pytest.raises(ValidationError):
            _validate(
                {
                    "finding_id": "unused-finding",
                    "framework_id": " \t ",
                    "preset_name": "unused-preset",
                },
                schemas["evidence-bound-review-executor-input-v1.json"],
                registry,
            )
    finally:
        project.close()


def test_synthesis_executor_input_schema_rejects_empty_finding_ids_like_runtime(tmp_path):
    schemas, registry = _public_schemas()
    project = Project.init(tmp_path / "project")
    try:
        with pytest.raises(EvidenceValidationError):
            run_synthesis(
                SkillContext(project, "unused-run", "unused-scope"), [], "unused-preset", {}
            )

        with pytest.raises(ValidationError):
            _validate(
                {"finding_ids": [], "preset_name": "unused-preset", "preset_snapshot": {}},
                schemas["synthesis-executor-input-v1.json"],
                registry,
            )
    finally:
        project.close()


def test_van_dijk_group_comparison_schemas_match_nonempty_selection_contract():
    schemas, registry = _public_schemas()
    input_schema = schemas["van-dijk-group-comparison-input-v1.json"]
    output_schema = schemas["van-dijk-group-comparison-output-v1.json"]
    input_payload = {
        "groups": [
            {"label": "Officials", "annotation_ids": ["annotation-1"]},
            {"label": "Residents", "annotation_ids": []},
        ],
        "shared_topic_event": "A shared event",
        "comparability_basis": "The sources address the same event",
        "codebook_id": "codebook-1",
    }

    _validate(input_payload, input_schema, registry)
    empty_selection = {
        **input_payload,
        "groups": [
            {"label": "Officials", "annotation_ids": []},
            {"label": "Residents", "annotation_ids": []},
        ],
    }
    with pytest.raises(ValidationError):
        _validate(empty_selection, input_schema, registry)

    valid_output = {
        "skill_run_id": "skill-run-1",
        "finding_id": "finding-1",
        "coverage_id": "coverage-1",
        "scope_snapshot_id": "scope-1",
        "artifact_type": "descriptive_qualitative_group_comparison_preparation",
        "shared_topic_event": "A shared event",
        "comparability_basis": "The sources address the same event",
        "codebook_id": "codebook-1",
        "codebook_name": "Comparison codebook",
        "count_basis": "SELECTED_ANNOTATIONS",
        "claim_semantics": "EXAMPLE_BASED",
        "selected_annotation_count": 1,
        "groups": [
            {
                "label": "Officials",
                "selected_annotation_count": 1,
                "coding_status": "CODED",
                "code_counts": [
                    {
                        "code_id": "code-1",
                        "label": "Action",
                        "selected_annotation_count": 1,
                    }
                ],
                "annotations": [
                    {
                        "annotation_id": "annotation-1",
                        "code_id": "code-1",
                        "code_label": "Action",
                        "code_definition": None,
                        "span_id": "span-1",
                        "revision_id": "revision-1",
                        "start_offset": 0,
                        "end_offset": 4,
                        "text": "text",
                        "evidence_id": "evidence-1",
                        "citation": "[source]",
                    }
                ],
            },
            {
                "label": "Residents",
                "selected_annotation_count": 0,
                "coding_status": "UNKNOWN",
                "code_counts": [],
                "annotations": [],
            },
        ],
        "evidence_ids": ["evidence-1"],
        "limitations": [],
    }

    _validate(valid_output, output_schema, registry)
    with pytest.raises(ValidationError):
        _validate(
            {**valid_output, "selected_annotation_count": 0}, output_schema, registry
        )
    with pytest.raises(ValidationError):
        _validate(
            {**valid_output, "claim_semantics": "CORPUS_WIDE"}, output_schema, registry
        )


def test_llm_manifest_schemas_match_runtime_executor_and_model_payloads(tmp_path):
    schemas, registry = _public_schemas()
    project = Project.init(tmp_path / "project")
    try:
        text = "Officials announced a new policy."
        _, revision_id = project.import_text("speech.txt", text)
        scope_id = project.snapshot([revision_id])
        source_run_id = project.create_run(scope_id, {"profile": "schema-source"})
        evidence_id = project.textual_evidence(project.create_span(revision_id, 0, len(text)))
        source_finding_id = project.create_finding(
            source_run_id,
            "fixture.source",
            "1.0.0",
            "policy announcement",
            "The source reports a policy announcement.",
        )
        project.attach_evidence(source_finding_id, evidence_id, EvidenceRole.EXAMPLE)
        project.create_finding_observation_candidates(
            source_finding_id,
            [
                {
                    "evidence_id": evidence_id,
                    "excerpt": "Officials announced a new policy.",
                    "description": "The author deliberately conceals the actor.",
                }
            ],
        )
        project.finalize_run(source_run_id)
        project.save_credential_reference(
            CredentialReference(
                "openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1"
            )
        )
        project.save_model_preset(
            ModelPreset("review", "openai-compatible", "fixture", "fixture-model")
        )
        preset_snapshot = snapshot_model_preset(project, "review")

        class SchemaFixtureProvider:
            provider_id = "openai-compatible"

            def __init__(self):
                self.model_inputs: list[dict[str, object]] = []
                self.model_outputs: list[dict[str, object]] = []

            def capabilities(self, model: str) -> ModelCapabilities:
                return ModelCapabilities()

            def generate(self, request: GenerationRequest) -> GenerationResponse:
                model_input = json.loads(
                    request.prompt.split("UNTRUSTED_RESEARCH_DATA_JSON:\n", 1)[1]
                )
                self.model_inputs.append(model_input)
                records = model_input["evidence_records"]
                cited_id = next(
                    item["evidence_id"]
                    for item in records
                    if item["evidence_kind"] == "TEXTUAL"
                )
                model_output = {
                    "interpretation": "The wording foregrounds a policy announcement.",
                    "alternative_interpretation": "The sentence may be a neutral report.",
                    "evidence_ids": [cited_id],
                    "counterevidence_ids": [],
                    "limitations": ["One selected passage was reviewed."],
                }
                if "source_findings" not in model_input:
                    model_output["observations"] = [
                        {
                            "evidence_id": cited_id,
                            "excerpt": next(
                                item["excerpt"]
                                for item in records
                                if item["evidence_id"] == cited_id
                                and isinstance(item.get("excerpt"), str)
                            ),
                            "description": "The passage contains an observable textual feature.",
                        }
                    ]
                self.model_outputs.append(model_output)
                return GenerationResponse(
                    json.dumps(model_output), self.provider_id, request.model
                )

        provider = SchemaFixtureProvider()
        provider_factory = lambda endpoint, reference, provider_id: provider
        review_run_id = project.create_run(scope_id, {"profile": "schema-review"})
        project.start_run(review_run_id)
        review_context = SkillContext(project, review_run_id, scope_id)
        framework_manifest = BUILTIN_SKILLS.get("builtin.fairclough-text").manifest
        assert framework_manifest.model_input_schema_ref is not None
        assert framework_manifest.model_output_schema_ref is not None
        framework_input = {
            "finding_id": source_finding_id,
            "preset_name": "review",
            "framework_id": "fairclough-text",
            "preset_snapshot": preset_snapshot,
        }
        _validate(
            framework_input,
            schemas[Path(framework_manifest.input_schema_ref).name],
            registry,
        )
        review_result = run_framework_review(
            review_context,
            source_finding_id,
            "review",
            "fairclough-text",
            provider_factory=provider_factory,
            preset_snapshot=preset_snapshot,
        )
        _validate(
            asdict(review_result),
            schemas[Path(framework_manifest.output_schema_ref).name],
            registry,
        )
        _validate(
            provider.model_inputs[0],
            schemas[Path(framework_manifest.model_input_schema_ref).name],
            registry,
        )
        assert provider.model_inputs[0]["unverified_observation_candidates"] == [
            {
                "candidate_id": project.finding_observation_candidates(
                    source_finding_id
                )[0]["id"],
                "evidence_id": evidence_id,
                "excerpt": "Officials announced a new policy.",
                "description": "The author deliberately conceals the actor.",
                "status": "UNVERIFIED",
            }
        ]
        _validate(
            provider.model_outputs[0],
            schemas[Path(framework_manifest.model_output_schema_ref).name],
            registry,
        )

        synthesis_manifest = BUILTIN_SKILLS.get("builtin.synthesis").manifest
        assert synthesis_manifest.model_input_schema_ref is not None
        assert synthesis_manifest.model_output_schema_ref is not None
        synthesis_input = {
            "finding_ids": [review_result.finding_id],
            "preset_name": "review",
            "preset_snapshot": preset_snapshot,
        }
        _validate(
            synthesis_input,
            schemas[Path(synthesis_manifest.input_schema_ref).name],
            registry,
        )
        synthesis_result = run_synthesis(
            review_context,
            [review_result.finding_id],
            "review",
            preset_snapshot,
            provider_factory=provider_factory,
        )
        _validate(
            synthesis_result,
            schemas[Path(synthesis_manifest.output_schema_ref).name],
            registry,
        )
        _validate(
            provider.model_inputs[1],
            schemas[Path(synthesis_manifest.model_input_schema_ref).name],
            registry,
        )
        _validate(
            provider.model_outputs[1],
            schemas[Path(synthesis_manifest.model_output_schema_ref).name],
            registry,
        )
    finally:
        project.close()


@pytest.mark.parametrize(
    ("schema_name", "executor"),
    [
        ("agency-candidates-input-v1.json", run_agency_candidates),
        ("social-actor-candidates-input-v1.json", run_social_actor_candidates),
    ],
)
def test_candidate_input_schemas_reject_runtime_blank_artifact_ids(
    tmp_path, schema_name, executor
):
    schemas, registry = _public_schemas()
    project = Project.init(tmp_path / schema_name.removesuffix(".json"))
    try:
        _, revision_id = project.import_text("speech.txt", "A short fixture.")
        scope_id = project.snapshot([revision_id])
        context = SkillContext(project, "unused-run", scope_id)

        with pytest.raises(ScopeViolationError):
            executor(context, [" \t "])

        with pytest.raises(ScopeViolationError):
            executor(context, [])

        with pytest.raises(TypeError):
            executor(context)

        with pytest.raises(ValidationError):
            _validate(
                {"artifact_ids": [" \t "]}, schemas[schema_name], registry
            )

        with pytest.raises(ValidationError):
            _validate({}, schemas[schema_name], registry)
    finally:
        project.close()


def test_partial_runtime_reports_conform_with_null_and_empty_fields(tmp_path):
    schemas, registry = _public_schemas()
    project = Project.init(tmp_path / "project")
    try:
        _, revision_id = project.import_text("speech.txt", "Café migration.")
        scope_id = project.snapshot([revision_id])
        run_id = project.create_run(scope_id, {"profile": "partial-schema-fixture"})
        project.start_run(run_id)
        skill_run_id = project.create_skill_run(
            run_id,
            scope_id,
            "fixture.partial",
            "1.0.0",
            "test-fixture",
            "a" * 64,
        )
        span_id = project.create_span(revision_id, 0, len("Café"))
        evidence_id = project.textual_evidence(span_id)
        finding_id = project.create_finding(
            run_id,
            "fixture.partial",
            "1.0.0",
            "lexical observation",
            "The selected string occurs in the document.",
            claim_types=[ClaimType.OBSERVATIONAL],
            # Null interpretation, alternative, coverage and skill-run links are
            # supported report states for descriptive/partial findings.
        )
        project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
        project.record_model_execution(
            "fixture-provider",
            "local-fixture",
            "fixture-model",
            None,
            "b" * 64,
            None,
            error_type="RuntimeError",
            analysis_run_id=run_id,
        )

        for status in ("RUNNING", "FAILED"):
            if status == "FAILED":
                project.fail_run(run_id, RuntimeError("fixture interruption"))

            summary = project.analysis_run_summary(run_id)
            report = analysis_payload(project, run_id)
            page = analysis_payload_page(project, run_id, offset=0, limit=1)
            finding = finding_payload(project, finding_id)

            _validate(summary, schemas["analysis-run-v1.json"], registry)
            _validate(report, schemas["analysis-report-v2.json"], registry)
            _validate(page, schemas["analysis-report-v2.json"], registry)
            _validate(finding, schemas["finding-report-v2.json"], registry)
            for evidence in finding["evidence"]:
                _validate(evidence, schemas["evidence-reference-v2.json"], registry)

            run_record = cast(dict[str, Any], summary["analysis_run"])
            skill_runs = cast(list[dict[str, Any]], summary["skill_runs"])
            assert run_record["status"] == status
            assert skill_runs[0]["id"] == skill_run_id
            assert finding["finding"]["interpretation"] is None
            assert finding["finding"]["alternative_interpretation"] is None
            assert finding["finding"]["coverage_id"] is None
            assert finding["finding"]["skill_run_id"] is None
            assert finding["human_reviews"] == []
            assert report["retrieval_runs"] == []
            assert report["reranking_runs"] == []
            assert report["findings"][0]["finding"]["id"] == finding_id

        bundle_path = export_bundle(project, run_id, tmp_path / "bundle")
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        bundle_schema = schemas["research-bundle-v16.json"]
        _validate(bundle, bundle_schema, registry)

        # Published report v2 is a closed record contract. Core remains the
        # authority for whether IDs exist across records; JSON Schema rejects
        # malformed local evidence shape but cannot enforce those foreign keys.
        malformed_finding = finding_payload(project, finding_id)
        malformed_finding["finding"]["unversioned_field"] = "not part of v1"
        with pytest.raises(ValidationError):
            _validate(malformed_finding, schemas["finding-report-v2.json"], registry)

        malformed_claims = finding_payload(project, finding_id)
        malformed_claims["finding"]["claim_types_json"] = '["UNSUPPORTED_CLAIM"]'
        with pytest.raises(ValidationError):
            _validate(malformed_claims, schemas["finding-report-v2.json"], registry)

        malformed_evidence_bundle = json.loads(json.dumps(bundle))
        evidence_ref = malformed_evidence_bundle["evidence"][finding_id][0]
        evidence_ref["role"] = "UNCONTROLLED_ROLE"
        with pytest.raises(ValidationError):
            _validate(malformed_evidence_bundle, bundle_schema, registry)

        malformed_textual_reference = finding_payload(project, finding_id)
        malformed_textual_reference["evidence"][0]["source_span_id"] = None
        with pytest.raises(ValidationError):
            _validate(
                malformed_textual_reference,
                schemas["finding-report-v2.json"],
                registry,
            )
    finally:
        project.close()
