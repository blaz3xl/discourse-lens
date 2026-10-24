import json

import pytest

from discourselens.errors import SkillManifestError
from discourselens.skills import (
    BUILTIN_SKILLS,
    builtin_manifest,
    skill_documentation,
    skill_resource_text,
)
from discourselens.types import ExecutionKind


def test_builtin_skill_catalog_exposes_versioned_method_contracts():
    manifests = BUILTIN_SKILLS.manifests()
    ids = {manifest.id for manifest in manifests}

    assert "builtin.textual-descriptors" in ids
    assert "builtin.quote-attribution-candidates" in ids
    assert "builtin.scope-question-hybrid-candidates" in ids
    assert "builtin.van-dijk-group-comparison" in ids
    assert "builtin.fairclough-text" in ids
    assert "builtin.argumentation" in ids
    assert "builtin.metaphor" in ids
    assert "builtin.intertextuality" in ids
    assert "builtin.sociocognitive" in ids
    assert "builtin.historical-context" in ids
    assert "builtin.synthesis" in ids
    assert (
        builtin_manifest("builtin.textual-descriptors").execution_kind
        is ExecutionKind.DETERMINISTIC
    )
    assert (
        builtin_manifest("builtin.quote-attribution-candidates").execution_kind
        is ExecutionKind.DETERMINISTIC
    )
    comparison = builtin_manifest("builtin.van-dijk-group-comparison")
    assert comparison.execution_kind is ExecutionKind.DETERMINISTIC
    assert comparison.allowed_claims == ["EXAMPLE_BASED"]
    assert "ideology_score" in comparison.prohibited_claims
    assert "asymmetry_score" in comparison.prohibited_claims
    assert all(manifest.methodology_refs and manifest.documentation_ref for manifest in manifests)


def test_all_builtin_methodology_and_schema_resources_resolve():
    for manifest in BUILTIN_SKILLS.manifests():
        assert skill_documentation(manifest.id).strip()
        assert skill_resource_text(manifest.documentation_ref).strip()
        for reference in (
            *manifest.methodology_refs,
            manifest.input_schema_ref,
            manifest.output_schema_ref,
            *(
                reference
                for reference in (
                    manifest.model_input_schema_ref,
                    manifest.model_output_schema_ref,
                )
                if reference is not None
            ),
        ):
            content = skill_resource_text(reference)
            assert content.strip()
            if reference.endswith(".json"):
                json.loads(content)


def test_llm_manifests_separate_executor_and_model_schema_contracts():
    synthesis = builtin_manifest("builtin.synthesis")
    framework = builtin_manifest("builtin.fairclough-text")
    deterministic = builtin_manifest("builtin.textual-descriptors")

    assert synthesis.input_schema_ref == "schemas/synthesis-executor-input-v1.json"
    assert synthesis.output_schema_ref == "schemas/synthesis-executor-output-v1.json"
    assert synthesis.model_input_schema_ref == "schemas/synthesis-model-input-v1.json"
    assert synthesis.model_output_schema_ref == "schemas/evidence-bound-review-model-output-v1.json"
    assert framework.input_schema_ref == "schemas/evidence-bound-review-executor-input-v1.json"
    assert framework.output_schema_ref == "schemas/evidence-bound-review-executor-output-v1.json"
    assert framework.model_input_schema_ref == "schemas/evidence-bound-review-model-input-v2.json"
    assert framework.model_output_schema_ref == "schemas/framework-review-v1.json"
    assert framework.model_output_schema_ref != synthesis.model_output_schema_ref
    assert deterministic.model_input_schema_ref is None
    assert deterministic.model_output_schema_ref is None


def test_scope_question_hybrid_manifest_declares_its_hybrid_contract():
    hybrid = builtin_manifest("builtin.scope-question-hybrid-candidates")

    assert hybrid.execution_kind is ExecutionKind.HYBRID
    assert hybrid.input_schema_ref == "schemas/scope-question-hybrid-candidates-input-v1.json"
    assert hybrid.output_schema_ref == "schemas/scope-question-hybrid-candidates-output-v1.json"
    assert hybrid.required_capabilities == ["semantic_search"]


@pytest.mark.parametrize(
    "reference",
    ["../secrets", "schemas/../../secret.json", "/etc/passwd", "docs/README.md", "schemas/"],
)
def test_skill_resource_reader_rejects_unbundled_paths(reference):
    with pytest.raises(SkillManifestError):
        skill_resource_text(reference)


@pytest.mark.parametrize(
    "reference",
    [
        r"docs/methodology/..\..\README.md",
        r"schemas\..\README.md",
        r"C:\outside.json",
    ],
)
def test_skill_resource_reader_rejects_windows_traversal_before_lookup(reference):
    with pytest.raises(SkillManifestError, match="outside bundled skill resources"):
        skill_resource_text(reference)


def test_skill_documentation_requires_registered_skill():
    with pytest.raises(SkillManifestError, match="unknown built-in skill"):
        skill_documentation("../../outside")
