import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource

from discourselens import Project
from discourselens.reporting import analysis_payload, finding_payload
from discourselens.skills import skill_resource_text
from discourselens.types import EvidenceRole

SCHEMAS = Path(__file__).parents[1] / "schemas"


def _load_registry():
    names = (
        "analysis-run-v1.json",
        "analysis-report-v1.json",
        "analysis-report-v2.json",
        "coverage-disclosure-v1.json",
        "evidence-reference-v1.json",
        "evidence-reference-v2.json",
        "finding-report-v1.json",
        "finding-report-v2.json",
    )
    schemas = {
        name: json.loads((SCHEMAS / name).read_text(encoding="utf-8")) for name in names
    }
    registry = Registry()
    for schema in schemas.values():
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return schemas, registry


def _validate(instance, schema, registry):
    Draft202012Validator(schema, registry=registry).validate(instance)


def test_report_v2_carries_link_rationale_and_preserves_v1_contract(tmp_path):
    schemas, registry = _load_registry()
    project = Project.init(tmp_path / "project")
    try:
        _, revision_id = project.import_text("speech.txt", "Officials approved the plan.")
        run_id = project.create_run(project.snapshot([revision_id]), {})
        finding_id = project.create_finding(
            run_id,
            "fixture",
            "1.0.0",
            "Agency construction",
            "Officials are represented as agents.",
        )
        evidence_id = project.textual_evidence(
            project.create_span(revision_id, 0, len("Officials approved the plan."))
        )
        project.attach_evidence(
            finding_id,
            evidence_id,
            EvidenceRole.SUPPORT,
            rationale="The grammatical subject is the agent of approved.",
        )

        finding = finding_payload(project, finding_id)
        report = analysis_payload(project, run_id)
        _validate(finding, schemas["finding-report-v2.json"], registry)
        _validate(report, schemas["analysis-report-v2.json"], registry)
        _validate(
            finding["evidence"][0], schemas["evidence-reference-v2.json"], registry
        )
        rationale = finding["evidence"][0]["rationale"]
        assert rationale is not None
        assert rationale.startswith("The grammatical subject")

        # v1 remains an immutable closed contract: legacy-shaped records still
        # validate, while the new rationale property is rejected there.
        legacy_finding = json.loads(json.dumps(finding))
        legacy_finding["evidence"][0].pop("rationale")
        _validate(legacy_finding, schemas["finding-report-v1.json"], registry)
        _validate(
            legacy_finding["evidence"][0],
            schemas["evidence-reference-v1.json"],
            registry,
        )
        with pytest.raises(ValidationError):
            _validate(finding, schemas["finding-report-v1.json"], registry)
        with pytest.raises(ValidationError):
            _validate(finding["evidence"][0], schemas["evidence-reference-v1.json"], registry)
        with pytest.raises(ValidationError):
            _validate(report, schemas["analysis-report-v1.json"], registry)

        # Published package resources expose the same immutable v2 contracts.
        for name in (
            "evidence-reference-v2.json",
            "finding-report-v2.json",
            "analysis-report-v2.json",
        ):
            runtime_schema = json.loads(skill_resource_text(f"schemas/{name}"))
            assert runtime_schema == schemas[name]
    finally:
        project.close()
