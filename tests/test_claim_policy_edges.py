from __future__ import annotations

import json

import pytest

from discourselens import Project
from discourselens.corpus import collocation_association_evidence
from discourselens.errors import EvidenceValidationError
from discourselens.nlp_artifacts import (
    artifact_frequency_statistic_evidence,
    build_surface_token_artifact,
)
from discourselens.policy import claim_allowed
from discourselens.types import (
    ClaimType,
    CoverageDescriptor,
    CoverageMode,
    EvidenceRole,
    NegativeResultStatus,
)
from discourselens.validation import validate_finding


def test_absence_requires_exhaustive_zero_population_status():
    assert not claim_allowed(
        ClaimType.ABSENCE,
        CoverageMode.TARGETED,
        negative_result_status=NegativeResultStatus.NO_MATCHES_UNDER_PLAN,
    )
    assert not claim_allowed(
        ClaimType.ABSENCE,
        CoverageMode.EXHAUSTIVE,
        negative_result_status=NegativeResultStatus.NO_MATCHES_UNDER_PLAN,
    )


def test_universal_and_exact_quantified_claims_fail_closed_on_weak_coverage():
    assert not claim_allowed(ClaimType.UNIVERSAL, CoverageMode.EXAMPLE_BASED)
    assert not claim_allowed(
        ClaimType.QUANTIFIED,
        CoverageMode.EXAMPLE_BASED,
        statistical_evidence=True,
        population_defined=True,
    )
    assert not claim_allowed(
        ClaimType.QUANTIFIED,
        CoverageMode.EXHAUSTIVE,
        statistical_evidence=True,
        population_defined=False,
    )


def test_exhaustive_descriptor_rejects_zero_inspection_of_declared_population():
    with pytest.raises(ValueError, match="inspection_count must match population_count"):
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id="scope",
            population_definition="all items in scope",
            population_count=2,
            inspection_count=0,
            completion_criterion="all items inspected",
            termination_reason="completed deterministic enumeration",
        )


@pytest.mark.parametrize(
    ("inspection_count", "expected_valid"),
    [(None, False), (2, True)],
)
def test_universal_claim_requires_complete_population_inspection(
    tmp_path, inspection_count, expected_valid
):
    project = Project.init(tmp_path / f"universal-inspection-{inspection_count}")
    first_text = "The first item has a recorded example."
    second_text = "The second item has a recorded example."
    _, first_revision = project.import_text("first.txt", first_text)
    _, second_revision = project.import_text("second.txt", second_text)
    scope = project.snapshot([first_revision, second_revision])
    run = project.create_run(scope, {})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition=(
                "one population item per scoped revision in the immutable ScopeSnapshot"
            ),
            population_count=2,
            inspection_count=inspection_count,
            completion_criterion="all items in the declared population inspected",
            termination_reason="completed deterministic enumeration",
        )
    )
    if not expected_valid:
        with pytest.raises(
            EvidenceValidationError,
            match="universal claim requires exhaustive coverage with one population item",
        ):
            project.create_finding(
                run,
                "fixture",
                "1.0.0",
                "universal claim",
                "The scoped items have recorded examples.",
                claim_types=[ClaimType.UNIVERSAL],
                coverage_id=coverage_id,
            )
        assert project.db.execute(
            "SELECT COUNT(*) FROM findings WHERE analysis_run_id=?", (run,)
        ).fetchone()[0] == 0
        return

    finding_id = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "universal claim",
        "The scoped items have recorded examples.",
        claim_types=[ClaimType.UNIVERSAL],
        coverage_id=coverage_id,
    )
    first_evidence_id = project.textual_evidence(
        project.create_span(first_revision, 0, len(first_text))
    )
    project.attach_evidence(finding_id, first_evidence_id, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding_id)
    assert not validation.valid
    assert any(
        "supporting textual evidence in every scoped revision" in error
        for error in validation.errors
    )

    second_evidence_id = project.textual_evidence(
        project.create_span(second_revision, 0, len(second_text))
    )
    project.attach_evidence(finding_id, second_evidence_id, EvidenceRole.SUPPORT)

    assert validate_finding(project, finding_id).valid


def test_universal_claim_rejects_population_count_not_aligned_to_scope_before_persistence(
    tmp_path,
):
    project = Project.init(tmp_path / "universal-population-outside-scope")
    _, revision = project.import_text("source.txt", "One scoped revision.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition=(
                "one population item per scoped revision in the immutable ScopeSnapshot"
            ),
            population_count=2,
            inspection_count=2,
            completion_criterion="all items in the declared population inspected",
            termination_reason="completed deterministic enumeration",
        )
    )

    with pytest.raises(
        EvidenceValidationError,
        match="one population item per scoped revision and complete inspection",
    ):
        project.create_finding(
            run,
            "fixture",
            "1.0.0",
            "universal claim",
            "The scoped items have recorded examples.",
            claim_types=[ClaimType.UNIVERSAL],
            coverage_id=coverage_id,
        )

    assert project.db.execute(
        "SELECT COUNT(*) FROM findings WHERE analysis_run_id=?", (run,)
    ).fetchone()[0] == 0


def test_universal_claim_rejects_population_definition_outside_scope_revisions(tmp_path):
    project = Project.init(tmp_path / "universal-population-definition-outside-scope")
    source_text = "A single scoped revision contains a few words."
    _, revision = project.import_text("source.txt", source_text)
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition="all English words across every book",
            population_count=1,
            inspection_count=1,
            completion_criterion="all items in the declared population inspected",
            termination_reason="completed deterministic enumeration",
        )
    )
    with pytest.raises(
        EvidenceValidationError,
        match="universal claim requires exhaustive coverage with one population item",
    ):
        project.create_finding(
            run,
            "fixture",
            "1.0.0",
            "universal claim",
            "Every English word appears in the scoped revision.",
            claim_types=[ClaimType.UNIVERSAL],
            coverage_id=coverage_id,
        )

    assert project.db.execute(
        "SELECT COUNT(*) FROM findings WHERE analysis_run_id=?", (run,)
    ).fetchone()[0] == 0


def test_finding_validator_rejects_stored_universal_population_label_mismatch(tmp_path):
    project = Project.init(tmp_path / "stored-universal-population-mismatch")
    source_text = "A single scoped revision contains a few words."
    _, revision = project.import_text("source.txt", source_text)
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition="all English words across every book",
            population_count=1,
            inspection_count=1,
            completion_criterion="all items in the declared population inspected",
            termination_reason="completed deterministic enumeration",
        )
    )
    finding_id = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "stored universal claim",
        "The scoped revision contains a recorded example.",
        claim_types=[ClaimType.OBSERVATIONAL],
        coverage_id=coverage_id,
    )
    evidence_id = project.textual_evidence(project.create_span(revision, 0, len(source_text)))
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)

    # Simulate a legacy/corrupt persisted record that bypassed the current write API.
    project.db.execute(
        "UPDATE findings SET claim_types_json=? WHERE id=?",
        (json.dumps([ClaimType.UNIVERSAL.value]), finding_id),
    )
    project.db.commit()

    validation = validate_finding(project, finding_id)

    assert not validation.valid
    assert any(
        "universal claim requires one population item per scoped revision" in error
        for error in validation.errors
    )


def test_absence_claim_rejects_incomplete_exhaustive_coverage_before_persistence(tmp_path):
    project = Project.init(tmp_path / "absence-incomplete-inspection")
    _, revision = project.import_text("source.txt", "Apples and pears.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition="all surface-token positions in the immutable ScopeSnapshot",
            population_count=4,
            completion_criterion="every surface-token position enumerated",
            termination_reason="completed deterministic enumeration",
            negative_result_status=NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION,
        )
    )

    with pytest.raises(EvidenceValidationError, match="absence requires completed exhaustive"):
        project.create_finding(
            run,
            "fixture",
            "1.0.0",
            "absence",
            "Migrants are absent from this population.",
            claim_types=[ClaimType.ABSENCE],
            coverage_id=coverage_id,
        )

    assert project.db.execute(
        "SELECT COUNT(*) FROM findings WHERE analysis_run_id=?", (run,)
    ).fetchone()[0] == 0


def test_absence_claim_requires_explicit_exhaustive_absence_status(tmp_path):
    project = Project.init(tmp_path / "absence-status-not-applicable")
    _, revision = project.import_text("source.txt", "Apples and pears.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition="all surface-token positions in the immutable ScopeSnapshot",
            population_count=4,
            inspection_count=4,
            completion_criterion="every surface-token position enumerated",
            termination_reason="completed deterministic enumeration",
        )
    )

    with pytest.raises(EvidenceValidationError, match="ABSENCE_IN_EXHAUSTIVE_POPULATION status"):
        project.create_finding(
            run,
            "fixture",
            "1.0.0",
            "absence",
            "Migrants are absent from this population.",
            claim_types=[ClaimType.ABSENCE],
            coverage_id=coverage_id,
        )

    assert project.db.execute(
        "SELECT COUNT(*) FROM findings WHERE analysis_run_id=?", (run,)
    ).fetchone()[0] == 0


def test_incomplete_inspection_count_does_not_change_observational_claim_validation(tmp_path):
    project = Project.init(tmp_path / "observational-incomplete-inspection")
    source_text = "The item has a recorded example."
    _, revision = project.import_text("source.txt", source_text)
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition="all items in the immutable ScopeSnapshot",
            population_count=2,
            completion_criterion="all items in the declared population inspected",
            termination_reason="completed deterministic enumeration",
        )
    )
    finding_id = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "observational claim",
        "The item has a recorded example.",
        claim_types=[ClaimType.OBSERVATIONAL],
        coverage_id=coverage_id,
    )
    evidence_id = project.textual_evidence(
        project.create_span(revision, 0, len(source_text))
    )
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)

    assert validate_finding(project, finding_id).valid


def test_interpretive_claim_requires_limitations_and_both_readings():
    assert claim_allowed(
        ClaimType.INTERPRETIVE,
        CoverageMode.EXAMPLE_BASED,
        limitations=True,
        interpretation=True,
        alternative_reading=True,
    )
    assert not claim_allowed(
        ClaimType.INTERPRETIVE,
        CoverageMode.EXAMPLE_BASED,
        limitations=False,
        interpretation=True,
        alternative_reading=True,
    )


def test_quantified_claim_rejects_forged_dispersion_version_value(tmp_path):
    project = Project.init(tmp_path / "forged-dispersion")
    _, revision = project.import_text("source.txt", "Apples apples.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    population = "all surface-token positions in every revision of the immutable ScopeSnapshot"
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition=population,
            population_count=2,
            inspection_count=2,
            completion_criterion="every scoped canonical text was tokenized with corpus.TOKEN and counted by revision",
            termination_reason="completed deterministic enumeration",
        )
    )
    evidence_id = project.corpus_statistic_evidence(
        "invented document dispersion",
        999,
        scope_snapshot_id=scope,
        population_definition=population,
        provenance={
            "calculation_version": "surface-document-dispersion-v1",
            "coverage_id": coverage_id,
            "population_count": 2,
            "population_tokens": 2,
            "document_count": 1,
            "document_frequency": 1,
            "document_range": 1.0,
            "documents": [{"revision_id": revision, "occurrences": 999, "token_count": 999}],
        },
    )
    finding_id = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "surface occurrences",
        "The population contains exactly 999 occurrences.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=coverage_id,
    )
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)

    result = validate_finding(project, finding_id)

    assert not result.valid
    assert any("document dispersion evidence" in error for error in result.errors)


def test_quantified_claim_rejects_tampered_window_association_score(tmp_path):
    project = Project.init(tmp_path / "tampered-window-score")
    _, revision = project.import_text("source.txt", "apple red apple red apple blue")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    statistic = collocation_association_evidence(
        project, scope, "apple", "red", window=1, min_frequency=1
    )
    finding_id = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "association score",
        "The selected population has the recorded association score.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=str(statistic["coverage_id"]),
    )
    evidence_id = str(statistic["evidence_id"])
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    payload = project.evidence_content(evidence_id)
    assert isinstance(payload["value"], (int, float))
    payload["value"] = float(payload["value"]) + 100
    project.db.execute(
        "UPDATE evidence_artifacts SET content_json=? WHERE id=?",
        (json.dumps(payload), evidence_id),
    )

    result = validate_finding(project, finding_id)

    assert not result.valid
    assert any("window association evidence" in error for error in result.errors)


def test_quantified_claim_rejects_tampered_artifact_frequency_value(tmp_path):
    project = Project.init(tmp_path / "tampered-artifact-frequency")
    _, revision = project.import_text("source.txt", "Café café")
    artifact = build_surface_token_artifact(project, revision)
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    statistic = artifact_frequency_statistic_evidence(
        project, scope, [artifact.artifact_id], field="normalized", value="café"
    )
    finding_id = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "artifact frequency",
        "The selected population has the recorded artifact count.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=str(statistic["coverage_id"]),
    )
    evidence_id = str(statistic["evidence_id"])
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    payload = project.evidence_content(evidence_id)
    payload["value"] = 999
    project.db.execute(
        "UPDATE evidence_artifacts SET content_json=? WHERE id=?",
        (json.dumps(payload), evidence_id),
    )

    result = validate_finding(project, finding_id)

    assert not result.valid
    assert any("derived statistic value or coverage" in error for error in result.errors)


def test_zero_hit_retrieval_status_cannot_be_persisted_as_absence(tmp_path):
    project = Project.init(tmp_path / "targeted-absence")
    _, revision = project.import_text("source.txt", "Apples and pears.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    with pytest.raises(EvidenceValidationError, match="absence requires completed exhaustive"):
        project.create_finding(
            run,
            "fixture",
            "1.0.0",
            "absence",
            "No examples were found by the targeted search.",
            claim_types=[ClaimType.ABSENCE],
            coverage_id=project.coverage(
                CoverageDescriptor(
                    coverage_mode=CoverageMode.TARGETED,
                    scope_snapshot_id=scope,
                    selection_policy="targeted retrieval query",
                    inspection_count=0,
                    termination_reason="retrieval completed with zero hits",
                    negative_result_status=NegativeResultStatus.NO_MATCHES_UNDER_PLAN,
                )
            ),
        )
