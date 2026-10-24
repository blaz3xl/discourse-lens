import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource

from discourselens import Project
from discourselens.bundle import export_bundle
from discourselens.reporting import analysis_payload, analysis_payload_page, finding_payload
from discourselens.skills import BUILTIN_SKILLS
from discourselens.types import CoverageDescriptor, CoverageMode, EvidenceRole, SkillManifest

SCHEMAS = Path(__file__).parents[1] / "schemas"


def _schema_registry() -> tuple[dict[str, dict], Registry]:
    schemas = {
        path.name: json.loads(path.read_text(encoding="utf-8")) for path in SCHEMAS.glob("*.json")
    }
    registry = Registry()
    for schema in schemas.values():
        if "$id" in schema:
            registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return schemas, registry


def _validate(instance: object, schema: dict, registry: Registry) -> None:
    Draft202012Validator(schema, registry=registry).validate(instance)


def test_published_schemas_are_valid_json_schema_documents():
    schemas, _ = _schema_registry()

    assert {
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
    } <= set(schemas)
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)


def test_published_skill_schema_accepts_every_builtin_manifest():
    schemas, registry = _schema_registry()
    schema = schemas["skill-manifest-v0.1.json"]

    for manifest in BUILTIN_SKILLS.manifests():
        _validate(manifest.model_dump(mode="json"), schema, registry)


def test_coverage_disclosure_schema_requires_state_consistent_descriptors():
    schemas, registry = _schema_registry()
    schema = schemas["coverage-disclosure-v1.json"]
    descriptor = {
        "coverage_mode": "TARGETED",
        "scope_snapshot_id": "scope-1",
        "population_definition": None,
        "population_count": None,
        "selection_policy": "inspect submitted query",
        "inspection_count": 3,
        "completion_criterion": None,
        "termination_reason": "query completed",
        "termination_state": "COMPLETED",
        "negative_result_status": "NO_MATCHES_UNDER_PLAN",
    }
    disclosures = (
        {"coverage_id": None, "status": "NOT_RECORDED", "descriptor": None},
        {"coverage_id": "coverage-missing", "status": "MISSING", "descriptor": None},
        {"coverage_id": "coverage-1", "status": "RECORDED", "descriptor": descriptor},
    )
    for disclosure in disclosures:
        _validate(disclosure, schema, registry)

    invalid_disclosures = (
        {"coverage_id": "coverage-1", "status": "NOT_RECORDED", "descriptor": None},
        {"coverage_id": None, "status": "MISSING", "descriptor": None},
        {"coverage_id": "coverage-1", "status": "RECORDED", "descriptor": None},
        {
            "coverage_id": "coverage-1",
            "status": "RECORDED",
            "descriptor": {**descriptor, "unpublished": True},
        },
    )
    for disclosure in invalid_disclosures:
        with pytest.raises(ValidationError):
            _validate(disclosure, schema, registry)


def test_core_finding_run_report_and_bundle_match_published_schemas(tmp_path):
    schemas, registry = _schema_registry()
    project = Project.init(tmp_path / "project")
    try:
        _, revision_id = project.import_text("speech.txt", "Apples are green.")
        scope_id = project.snapshot([revision_id])
        analysis_run_id = project.create_run(scope_id, {"profile": "schema-fixture"})
        coverage_id = project.coverage(
            CoverageDescriptor(
                coverage_mode=CoverageMode.TARGETED,
                scope_snapshot_id=scope_id,
                population_definition="one fixture document",
                selection_policy="explicit test fixture",
            )
        )
        skill_run_id = project.create_skill_run(
            analysis_run_id,
            scope_id,
            "fixture.exact-phrase",
            "1.0.0",
            "test-fixture",
            hashlib.sha256(b"fixture-config").hexdigest(),
            coverage_id=coverage_id,
        )
        span_id = project.create_span(revision_id, 0, 6)
        codebook_id = project.create_codebook("schema coding")
        code_id = project.create_code(codebook_id, "fruit")
        project.annotate_span(code_id, span_id, "Schema fixture coding note.")
        evidence_id = project.textual_evidence(span_id)
        statistic_id = project.corpus_statistic_evidence(
            "surface token count",
            3,
            scope_snapshot_id=scope_id,
            population_definition="all surface tokens in the fixture document",
        )
        external_id = project.external_source_evidence(
            "Fixture bibliography",
            "A researcher-supplied source excerpt.",
            retrieved_at="2026-09-29T00:00:00Z",
            url="https://example.org/source",
        )
        context_id = project.user_context_evidence("Research note", "Researcher supplied context.")
        finding_id = project.create_finding(
            analysis_run_id,
            "fixture.exact-phrase",
            "1.0.0",
            "literal phrase",
            "The selected phrase occurs in the source text.",
            skill_run_id=skill_run_id,
        )
        project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
        project.attach_evidence(finding_id, statistic_id, EvidenceRole.CONTEXT)
        project.attach_evidence(finding_id, external_id, EvidenceRole.CONTEXT)
        project.attach_evidence(finding_id, context_id, EvidenceRole.CONTEXT)
        retrieval_id = project.create_retrieval_run(
            analysis_run_id=analysis_run_id,
            skill_run_id=skill_run_id,
            scope_id=scope_id,
            retrieval_plan_ref="retrieval/schema-fixture-v1",
            query_terms=["Apples"],
            selection_policy="explicit fixture term",
            candidate_count=1,
            inspected_count=1,
            returned_count=1,
            candidate_limit_reached=False,
            coverage_id=coverage_id,
        )
        retrieval_hit_id = project.add_retrieval_hit(retrieval_id, span_id, "lexical", 1, 1.0)
        project.record_reranking(
            retrieval_id,
            provider_id="fixture-reranker",
            credential_alias="test",
            model="fixture-model",
            query="Apples",
            top_n=1,
            effective_top_n=1,
            candidate_count=1,
            ranked_hits=[(retrieval_hit_id, 1, 0.9)],
        )
        project.record_model_execution(
            "fixture-provider",
            "test",
            "fixture-model",
            None,
            hashlib.sha256(b"request").hexdigest(),
            hashlib.sha256(b"response").hexdigest(),
            analysis_run_id=analysis_run_id,
        )
        project.complete_skill_run(skill_run_id)
        project.finalize_run(analysis_run_id)

        run_summary = project.analysis_run_summary(analysis_run_id)
        report = analysis_payload(project, analysis_run_id)
        report_page = analysis_payload_page(project, analysis_run_id, offset=0, limit=1)
        finding = finding_payload(project, finding_id)
        bundle_path = export_bundle(project, analysis_run_id, tmp_path / "bundle")
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

        _validate(run_summary, schemas["analysis-run-v1.json"], registry)
        _validate(finding, schemas["finding-report-v2.json"], registry)
        assert {item["kind"] for item in finding["evidence"]} == {
            "TEXTUAL",
            "CORPUS_STATISTIC",
            "EXTERNAL_SOURCE",
            "USER_CONTEXT",
        }
        for item in finding["evidence"]:
            _validate(item, schemas["evidence-reference-v2.json"], registry)
        _validate(report, schemas["analysis-report-v2.json"], registry)
        _validate(report_page, schemas["analysis-report-v2.json"], registry)
        assert report_page["findings"][0]["coverage"] == finding["coverage"]
        assert report_page["coverage_descriptors"] == report["coverage_descriptors"]

        malformed_page = json.loads(json.dumps(report_page))
        malformed_page["pagination"]["unexpected"] = {
            "offset": 0,
            "limit": 1,
            "total_count": 0,
            "has_more": False,
        }
        with pytest.raises(ValidationError):
            _validate(malformed_page, schemas["analysis-report-v2.json"], registry)

        _validate(bundle, schemas["research-bundle-v16.json"], registry)

        malformed_reviews = json.loads(json.dumps(bundle))
        malformed_reviews["human_reviews"][finding_id] = [{"id": "review-1", "status": "UNKNOWN"}]
        with pytest.raises(ValidationError):
            _validate(malformed_reviews, schemas["research-bundle-v16.json"], registry)

        legacy_v11 = json.loads(json.dumps(bundle))
        legacy_v11["bundle_version"] = 11
        legacy_v11.pop("human_reviews")
        legacy_v11.pop("finding_evidence_spans")
        legacy_v11.pop("canonical_texts")
        for revision_record in legacy_v11["document_revisions"]:
            revision_record.pop("extraction_manifest_json")
            revision_record.pop("structure_hash")
        for evidence_rows in legacy_v11["evidence"].values():
            for evidence in evidence_rows:
                evidence.pop("rationale")
                evidence.pop("source_revision_id")
        legacy_v11["qualitative_coding"].pop("source_spans")
        _validate(legacy_v11, schemas["research-bundle-v11.json"], registry)

        empty_finding_evidence = json.loads(json.dumps(finding))
        empty_finding_evidence["evidence"] = []
        with pytest.raises(ValidationError):
            _validate(empty_finding_evidence, schemas["finding-report-v2.json"], registry)

        missing_retrieval_provenance = json.loads(json.dumps(bundle))
        del missing_retrieval_provenance["retrieval_runs"][0]["scope_id"]
        with pytest.raises(ValidationError):
            _validate(
                missing_retrieval_provenance,
                schemas["research-bundle-v16.json"],
                registry,
            )

        missing_bundle_evidence_span = json.loads(json.dumps(bundle))
        textual_reference = next(
            item
            for item in missing_bundle_evidence_span["evidence"][finding_id]
            if item["kind"] == "TEXTUAL"
        )
        textual_reference["source_span_id"] = None
        with pytest.raises(ValidationError):
            _validate(missing_bundle_evidence_span, schemas["research-bundle-v16.json"], registry)
    finally:
        project.close()


def test_evidence_schema_rejects_uncontrolled_roles():
    schemas, registry = _schema_registry()
    evidence: dict[str, object] = {
        "evidence_id": "ev-1",
        "role": "SUPPORT",
        "kind": "TEXTUAL",
        "text": "exact excerpt",
        "citation": "[speech.txt · ¶1]",
        "source_span_id": "span-1",
    }
    _validate(evidence, schemas["evidence-reference-v1.json"], registry)
    evidence["role"] = "FAVOURABLE"
    with pytest.raises(ValidationError):
        _validate(evidence, schemas["evidence-reference-v1.json"], registry)

    evidence["role"] = "SUPPORT"
    evidence["source_span_id"] = None
    with pytest.raises(ValidationError):
        _validate(evidence, schemas["evidence-reference-v1.json"], registry)


def test_evidence_schema_requires_generated_citation_and_kind_consistent_span():
    schemas, registry = _schema_registry()
    schema = schemas["evidence-reference-v1.json"]
    evidence: dict[str, object] = {
        "evidence_id": "ev-context",
        "role": "CONTEXT",
        "kind": "USER_CONTEXT",
        "text": "Researcher context.",
        "citation": "[user context · note]",
        "source_span_id": None,
    }
    _validate(evidence, schema, registry)

    for field, value in (("citation", None), ("citation", ""), ("source_span_id", "span-1")):
        malformed = {**evidence, field: value}
        with pytest.raises(ValidationError):
            _validate(malformed, schema, registry)


def test_published_schemas_reject_missing_finding_and_run_provenance(tmp_path):
    schemas, registry = _schema_registry()
    project = Project.init(tmp_path / "project")
    try:
        _, revision_id = project.import_text("speech.txt", "Apples are green.")
        scope_id = project.snapshot([revision_id])
        run_id = project.create_run(scope_id, {"profile": "schema-negative-fixture"})
        span_id = project.create_span(revision_id, 0, 6)
        evidence_id = project.textual_evidence(span_id)
        finding_id = project.create_finding(
            run_id, "fixture", "1.0.0", "literal phrase", "The phrase is present."
        )
        project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
        project.finalize_run(run_id)

        run_summary = project.analysis_run_summary(run_id)
        finding = finding_payload(project, finding_id)
        report = analysis_payload(project, run_id)
        bundle = json.loads(export_bundle(project, run_id, tmp_path / "bundle").read_text())

        malformed_run = json.loads(json.dumps(run_summary))
        del malformed_run["analysis_run"]["config_hash"]
        with pytest.raises(ValidationError):
            _validate(malformed_run, schemas["analysis-run-v1.json"], registry)

        malformed_finding = json.loads(json.dumps(finding))
        del malformed_finding["finding"]["analysis_run_id"]
        with pytest.raises(ValidationError):
            _validate(malformed_finding, schemas["finding-report-v2.json"], registry)

        malformed_report = json.loads(json.dumps(report))
        del malformed_report["analysis_run"]["scope_id"]
        with pytest.raises(ValidationError):
            _validate(malformed_report, schemas["analysis-report-v2.json"], registry)

        malformed_bundle = json.loads(json.dumps(bundle))
        del malformed_bundle["document_revisions"][0]["canonical_hash"]
        with pytest.raises(ValidationError):
            _validate(malformed_bundle, schemas["research-bundle-v16.json"], registry)
    finally:
        project.close()


def test_skill_manifest_schema_requires_methodology_references():
    schemas, registry = _schema_registry()
    schema = schemas["skill-manifest-v0.1.json"]
    manifest = BUILTIN_SKILLS.manifests()[0].model_dump(mode="json")
    _validate(manifest, schema, registry)

    missing = {key: value for key, value in manifest.items() if key != "methodology_refs"}
    with pytest.raises(ValidationError):
        _validate(missing, schema, registry)

    blank = {**manifest, "methodology_refs": ["  "]}
    with pytest.raises(ValidationError):
        _validate(blank, schema, registry)


@pytest.mark.parametrize(
    "change",
    [
        {"id": " \t "},
        {"framework": "\n "},
        {"input_schema_ref": "\t"},
        {"output_schema_ref": " "},
        {"model_input_schema_ref": " \t"},
        {"model_output_schema_ref": "\n"},
        {"executor": "\n"},
        {"documentation_ref": " \t"},
        {"methodology_refs": [" \t"]},
        {"required_capabilities": ["\n"]},
        {"prohibited_claims": [" "]},
        {"prompt_refs": ["\t"]},
        {"dependencies": [" \n"]},
        {"retrieval_plan_ref": "retrieval/ \t"},
    ],
)
def test_skill_manifest_schema_rejects_runtime_blank_text(change: dict[str, object]):
    schemas, registry = _schema_registry()
    schema = schemas["skill-manifest-v0.1.json"]
    manifest = BUILTIN_SKILLS.manifests()[0].model_dump(mode="json") | change

    with pytest.raises(ValueError):
        SkillManifest.model_validate(manifest)
    with pytest.raises(ValidationError):
        _validate(manifest, schema, registry)


def test_skill_manifest_schema_accepts_runtime_trimmed_retrieval_reference():
    schemas, registry = _schema_registry()
    schema = schemas["skill-manifest-v0.1.json"]
    manifest = BUILTIN_SKILLS.manifests()[0].model_dump(mode="json")
    manifest["retrieval_plan_ref"] = " retrieval/fixture-v1 "

    validated = SkillManifest.model_validate(manifest)
    _validate(manifest, schema, registry)
    assert validated.retrieval_plan_ref == "retrieval/fixture-v1"


def test_old_v0_1_manifests_remain_valid_without_model_schema_refs():
    schemas, registry = _schema_registry()
    schema = schemas["skill-manifest-v0.1.json"]
    manifest = BUILTIN_SKILLS.manifests()[0].model_dump(mode="json")
    manifest.pop("model_input_schema_ref")
    manifest.pop("model_output_schema_ref")

    _validate(manifest, schema, registry)
    assert SkillManifest.model_validate(manifest).model_input_schema_ref is None


@pytest.mark.parametrize(
    "version",
    [
        "1.2.3",
        "1.2.3-alpha",
        "1.2.3-alpha.1",
        "1.2.3-0",
        "1.2.3-alpha-01",
        "1.2.3+build.01",
    ],
)
def test_skill_manifest_schema_accepts_semver_2_versions(version: str):
    schemas, registry = _schema_registry()
    manifest = BUILTIN_SKILLS.manifests()[0].model_dump(mode="json")
    manifest["version"] = version

    SkillManifest.model_validate(manifest)
    _validate(manifest, schemas["skill-manifest-v0.1.json"], registry)


@pytest.mark.parametrize("version", ["01.2.3", "1.2.3-01", "1.2.3-alpha.01"])
def test_skill_manifest_schema_rejects_semver_2_leading_zeroes(version: str):
    schemas, registry = _schema_registry()
    manifest = BUILTIN_SKILLS.manifests()[0].model_dump(mode="json")
    manifest["version"] = version

    with pytest.raises(ValueError):
        SkillManifest.model_validate(manifest)
    with pytest.raises(ValidationError):
        _validate(manifest, schemas["skill-manifest-v0.1.json"], registry)
