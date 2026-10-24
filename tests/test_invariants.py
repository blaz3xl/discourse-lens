import pytest
from pydantic import ValidationError

from discourselens import Project
from discourselens.errors import (
    EvidenceValidationError,
    ScopeViolationError,
    SkillManifestError,
    SourceSpanError,
)
from discourselens.policy import claim_allowed
from discourselens.profiles import ControlledCdaProfile
from discourselens.skills import parse_manifest
from discourselens.types import (
    ClaimType,
    CoverageDescriptor,
    CoverageMode,
    CoverageTerminationState,
    EvidenceRole,
    NegativeResultStatus,
)


def test_scope_rejects_duplicate_and_unknown_revisions(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("scope.txt", "One source.")

    with pytest.raises(ScopeViolationError, match="duplicate"):
        project.snapshot([revision, revision])
    with pytest.raises(ScopeViolationError, match="unknown document revision"):
        project.snapshot(["not-a-revision"])


def test_unicode_and_invalid_spans(tmp_path):
    p = Project.init(tmp_path / "p")
    _, r = p.import_text("u", "Café migración")
    assert p.span_text(p.create_span(r, 0, 4)) == "Café"
    for start, end in [(-1, 1), (1, -1), (8, 2), (0, 99), (3, 3)]:
        with pytest.raises(SourceSpanError):
            p.create_span(r, start, end)


def test_span_batch_deduplicates_identity_and_validates_before_writing(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("source.txt", "alpha beta")

    span_ids = project.create_spans(revision, [(0, 5), (6, 10), (0, 5)])

    assert span_ids[0] == span_ids[2]
    assert [project.span_text(span_id) for span_id in span_ids] == ["alpha", "beta", "alpha"]
    assert project.db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0] == 2

    with pytest.raises(SourceSpanError, match="invalid half-open offsets"):
        project.create_spans(revision, [(0, 5), (6, 99)])
    assert project.db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0] == 2


def test_evidence_foreign_keys_and_claim_policy(tmp_path):
    p = Project.init(tmp_path / "p")
    _, r = p.import_text("u", "Café")
    run = p.create_run(p.snapshot([r]), {})
    finding = p.create_finding(run, "fixture", "1.0.0", "x", "y")
    evidence = p.textual_evidence(p.create_span(r, 0, 4))
    p.attach_evidence(finding, evidence, EvidenceRole.ALTERNATIVE)
    with pytest.raises(EvidenceValidationError):
        p.attach_evidence("unknown", evidence, EvidenceRole.SUPPORT)
    with pytest.raises(EvidenceValidationError):
        p.attach_evidence(finding, "unknown", EvidenceRole.SUPPORT)
    assert not claim_allowed(ClaimType.ABSENCE, CoverageMode.TARGETED)
    assert not claim_allowed(ClaimType.ABSENCE, CoverageMode.EXHAUSTIVE)
    assert not claim_allowed(ClaimType.COMPARATIVE, CoverageMode.EXAMPLE_BASED)
    assert not claim_allowed(ClaimType.COMPARATIVE, CoverageMode.BROAD)
    assert not claim_allowed(
        ClaimType.COMPARATIVE, CoverageMode.EXHAUSTIVE, population_defined=True
    )
    assert claim_allowed(
        ClaimType.COMPARATIVE,
        CoverageMode.EXHAUSTIVE,
        population_defined=True,
        comparable_populations=True,
    )
    assert claim_allowed(
        ClaimType.ABSENCE,
        CoverageMode.EXHAUSTIVE,
        negative_result_status=NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION,
    )
    assert not claim_allowed(ClaimType.UNIVERSAL, CoverageMode.EXAMPLE_BASED)
    assert claim_allowed(
        ClaimType.QUANTIFIED,
        CoverageMode.EXHAUSTIVE,
        statistical_evidence=True,
        population_defined=True,
    )


def test_coverage_scope_and_manifest_matrix(tmp_path):
    CoverageDescriptor(coverage_mode=CoverageMode.EXAMPLE_BASED, scope_snapshot_id="s")
    CoverageDescriptor(coverage_mode=CoverageMode.TARGETED, scope_snapshot_id="s")
    CoverageDescriptor(coverage_mode=CoverageMode.BROAD, scope_snapshot_id="s")
    CoverageDescriptor(
        coverage_mode=CoverageMode.EXHAUSTIVE,
        scope_snapshot_id="s",
        population_definition="all",
        completion_criterion="done",
        termination_reason="completed",
    )
    completed = CoverageDescriptor(
        coverage_mode=CoverageMode.EXHAUSTIVE,
        scope_snapshot_id="s",
        population_definition="all",
        completion_criterion="done",
        termination_reason="completed deterministic enumeration",
    )
    assert completed.termination_state is CoverageTerminationState.COMPLETED
    assert completed.inspection_count is None
    assert completed.negative_result_status is NegativeResultStatus.NOT_APPLICABLE
    legacy = CoverageDescriptor.model_validate_json(
        '{"coverage_mode":"EXHAUSTIVE","scope_snapshot_id":"s",'
        '"population_definition":"all","completion_criterion":"done",'
        '"termination_reason":"completed deterministic enumeration"}'
    )
    assert legacy.termination_state is CoverageTerminationState.COMPLETED
    with pytest.raises(ValidationError):
        completed.coverage_mode = CoverageMode.BROAD
    with pytest.raises(ValueError, match="termination_reason"):
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id="s",
            population_definition="all",
            completion_criterion="done",
        )
    with pytest.raises(ValueError, match="COMPLETED termination_state"):
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id="s",
            population_definition="all",
            completion_criterion="done",
            termination_reason="scan failed early",
            termination_state=CoverageTerminationState.FAILED,
        )
    with pytest.raises(ValueError, match="requires EXHAUSTIVE coverage"):
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id="s",
            negative_result_status=NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION,
        )
    with pytest.raises(ValueError, match="population_count"):
        CoverageDescriptor(
            coverage_mode=CoverageMode.BROAD,
            scope_snapshot_id="s",
            population_count=-1,
        )
    p = Project.init(tmp_path / "p")
    _, a = p.import_text("a", "Apples are green.")
    _, b = p.import_text("b", "Elephants live in Africa.")
    assert not p.search("elephants", p.snapshot([a]))
    assert p.search("elephants", p.snapshot([b]))
    with pytest.raises(ScopeViolationError):
        p.search("elephants", "unknown")
    base = {
        "id": "fixture",
        "version": "1.0.0",
        "framework": "fixture",
        "execution_kind": "HYBRID",
        "input_schema_ref": "i",
        "output_schema_ref": "o",
        "executor": "builtin",
        "retrieval_plan_ref": "retrieval/a",
        "documentation_ref": "d",
        "methodology_refs": ["m"],
    }
    for change in [
        {"version": "bad"},
        {"execution_kind": "ARBITRARY"},
        {"methodology_refs": []},
        {"methodology_refs": [" "]},
        {"retrieval_plan_ref": "bad"},
        {"retrieval_plan_ref": "retrieval/"},
        {"required_capabilites": ["dependency_parse"]},
        {"required_capabilities": [""]},
        {"id": " "},
        {"framework": ""},
        {"executor": "  "},
    ]:
        with pytest.raises(SkillManifestError):
            parse_manifest(base | change)


def test_semver_contract_accepts_prerelease_build_and_rejects_invalid_versions():
    manifest = {
        "id": "fixture.semver",
        "version": "1.0.0",
        "framework": "fixture",
        "execution_kind": "DETERMINISTIC",
        "input_schema_ref": "input-v1.json",
        "output_schema_ref": "output-v1.json",
        "executor": "builtin.fixture",
        "methodology_refs": ["methodology-v1.md"],
        "retrieval_plan_ref": "retrieval/fixture-v1",
        "documentation_ref": "docs/methodology/fixture-v1.md",
    }

    for version in ("0.0.0", "1.2.3", "1.2.3-rc.1", "1.2.3+build.05", "1.2.3-rc.1+build.05"):
        assert parse_manifest(manifest | {"version": version}).version == version
        profile = ControlledCdaProfile.model_validate(
            {"version": version, "frameworks": ["fairclough-text"]}
        )
        assert profile.version == version

    for version in (
        "1.2",
        "01.2.3",
        "1.2.3-01",
        "1.2.3+",
        "1.2.3-rc..1",
        "v1.2.3",
    ):
        with pytest.raises(SkillManifestError):
            parse_manifest(manifest | {"version": version})
        with pytest.raises(ValidationError):
            ControlledCdaProfile.model_validate(
                {"version": version, "frameworks": ["fairclough-text"]}
            )
