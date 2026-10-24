import hashlib
import json
from typing import cast

import pytest

from discourselens import Project
from discourselens.corpus import (
    TOKEN,
    collocation_association_evidence,
    compare_documents,
    document_dispersion_evidence,
    frequency_statistic_evidence,
    kwic_evidence,
    lexical_bundle_statistic_evidence,
)
from discourselens.errors import EvidenceValidationError, ScopeViolationError
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import (
    LEGACY_PARSED_TOKEN_SCHEMA_VERSION,
    artifact_frequency_statistic_evidence,
    build_parsed_token_artifact,
    build_surface_token_artifact,
)
from discourselens.policy import claim_allowed
from discourselens.reporting import render_finding_markdown
from discourselens.types import (
    ClaimType,
    CoverageDescriptor,
    CoverageMode,
    EvidenceRole,
    NegativeResultStatus,
)
from discourselens.validation import _validate_artifact_statistic_provenance, validate_finding


def test_claim_policy_fails_closed_for_unknown_enum_values():
    assert not claim_allowed(cast(ClaimType, "UNRECOGNIZED"), CoverageMode.TARGETED)
    assert not claim_allowed(ClaimType.ABSENCE, cast(CoverageMode, "UNRECOGNIZED"))


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("evidence_sufficiency", "ABSOLUTE"),
        ("interpretation_confidence", "CERTAIN"),
    ],
)
def test_validator_rejects_unknown_epistemic_metadata(tmp_path, field, invalid_value):
    project = Project.init(tmp_path / "invalid-epistemic-metadata")
    _, revision = project.import_text("source.txt", "Officials must act.")
    run = project.create_run(project.snapshot([revision]), {})
    epistemic_metadata = {
        "evidence_sufficiency": "SUFFICIENT",
        "interpretation_confidence": "HIGH",
    }
    epistemic_metadata[field] = invalid_value
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "observation",
        "The wording is present.",
        claim_types=[ClaimType.OBSERVATIONAL],
        evidence_sufficiency=epistemic_metadata["evidence_sufficiency"],
        interpretation_confidence=epistemic_metadata["interpretation_confidence"],
    )
    evidence = project.textual_evidence(project.create_span(revision, 0, 19))
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("invalid epistemic metadata" in error for error in validation.errors)


@pytest.mark.parametrize(
    ("field", "invalid_json", "expected_error"),
    [
        ("claim_types_json", '{"OBSERVATIONAL":true}', "invalid claim type metadata"),
        ("limitations_json", '"A limitation"', "invalid limitations metadata"),
    ],
)
def test_validator_rejects_finding_metadata_with_wrong_json_shape(
    tmp_path, field, invalid_json, expected_error
):
    project = Project.init(tmp_path / f"invalid-{field}")
    _, revision = project.import_text("source.txt", "Canonical evidence.")
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(run, "fixture", "1.0.0", "example", "Observation")
    evidence = project.textual_evidence(project.create_span(revision, 0, 9))
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    project.db.execute(f"UPDATE findings SET {field}=? WHERE id=?", (invalid_json, finding))
    project.db.commit()

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any(expected_error in error for error in validation.errors)


def test_validator_rejects_evidence_from_outside_historical_scope(tmp_path):
    project = Project.init(tmp_path / "p")
    _, selected = project.import_text("a.txt", "Selected evidence.")
    _, excluded = project.import_text("b.txt", "Excluded evidence.")
    run = project.create_run(project.snapshot([selected]), {})
    finding = project.create_finding(run, "test", "1.0.0", "example", "Observation")
    outside = project.textual_evidence(project.create_span(excluded, 0, 8))
    project.attach_evidence(finding, outside, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert "outside the analysis scope" in validation.errors[0]
    with pytest.raises(EvidenceValidationError):
        render_finding_markdown(project, finding)


def test_validator_accepts_matching_canonical_evidence(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Canonical evidence.")
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(run, "test", "1.0.0", "example", "Observation")
    evidence = project.textual_evidence(project.create_span(revision, 0, 9))
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)

    assert validate_finding(project, finding).valid


def test_validator_rejects_source_span_offsets_outside_revision(tmp_path):
    project = Project.init(tmp_path / "invalid-span-offsets")
    text = "Canonical evidence."
    _, revision = project.import_text("source.txt", text)
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "observation",
        "The cited words occur in the source.",
        claim_types=[ClaimType.OBSERVATIONAL],
    )
    span_id = project.create_span(revision, 0, len("Canonical"))
    evidence_id = project.textual_evidence(span_id)
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    # Simulate a persisted row that could not be created through Project APIs.
    # Its hash matches Python's out-of-bounds slice, so hash-only validation
    # would otherwise accept an empty quote as canonical evidence.
    invalid_slice = text[-1:1]
    project.db.execute(
        "UPDATE source_spans SET start_offset=?,end_offset=?,text_hash=? WHERE id=?",
        (-1, 1, hashlib.sha256(invalid_slice.encode()).hexdigest(), span_id),
    )
    project.db.commit()

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("has invalid offsets" in error for error in validation.errors)


def test_validator_rejects_unrecognized_persisted_evidence_role(tmp_path):
    project = Project.init(tmp_path / "invalid-evidence-role")
    _, revision = project.import_text("source.txt", "Canonical evidence.")
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(run, "fixture", "1.0.0", "example", "Observation")
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 9))
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)
    project.db.execute(
        "UPDATE finding_evidence SET role=? WHERE finding_id=? AND evidence_id=?",
        ("SUPPORT/COUNTEREVIDENCE", finding, evidence_id),
    )
    project.db.commit()

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("unsupported evidence role" in error for error in validation.errors)


def test_validator_rejects_dangling_finding_evidence_row_even_with_valid_evidence(tmp_path):
    project = Project.init(tmp_path / "dangling-finding-evidence")
    _, revision = project.import_text("source.txt", "Canonical evidence.")
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(run, "fixture", "1.0.0", "example", "Observation")
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 9))
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)
    project.db.execute(
        "INSERT INTO finding_evidence VALUES(?,?,?,?)",
        (finding, "missing-evidence-artifact", EvidenceRole.EXAMPLE, None),
    )
    project.db.commit()

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("unknown evidence artifact" in error for error in validation.errors)


def test_validator_rejects_finding_linked_to_skill_run_from_another_analysis_run(tmp_path):
    project = Project.init(tmp_path / "foreign-skill-run")
    _, revision = project.import_text("a.txt", "The source supplies contextual evidence.")
    scope = project.snapshot([revision])
    source_run = project.create_run(scope, {"profile": "source"})
    finding_run = project.create_run(scope, {"profile": "finding"})
    foreign_skill_run = project.create_skill_run(
        source_run,
        scope,
        "builtin.synthesis",
        "1.0.0",
        "builtin.evidence_constrained_synthesis",
        "fixture-digest",
    )
    coverage = project.coverage(
        CoverageDescriptor(coverage_mode=CoverageMode.EXAMPLE_BASED, scope_snapshot_id=scope)
    )
    matching_skill_run = project.create_skill_run(
        finding_run,
        scope,
        "builtin.synthesis",
        "1.0.0",
        "builtin.evidence_constrained_synthesis",
        "fixture-digest",
    )
    finding = project.create_finding(
        finding_run,
        "builtin.synthesis",
        "1.0.0",
        "contextual reading",
        "The source establishes a limited contextual reading.",
        skill_run_id=matching_skill_run,
        interpretation="The statement frames the issue as a collective concern.",
        alternative_interpretation="It may be a conventional announcement without broader framing.",
        claim_types=[ClaimType.INTERPRETIVE],
        limitations=["One passage was considered."],
        coverage_id=coverage,
    )
    evidence = project.textual_evidence(
        project.create_span(revision, 0, len("The source supplies contextual evidence."))
    )
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    # Simulate a damaged/legacy database row to verify that validation detects
    # provenance corruption even though the normal write API rejects it.
    project.db.execute(
        "UPDATE findings SET skill_run_id=? WHERE id=?", (foreign_skill_run, finding)
    )
    project.db.commit()

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("skill run provenance" in error for error in validation.errors)


def test_builtin_finding_rejects_claim_type_outside_its_registered_manifest(tmp_path):
    project = Project.init(tmp_path / "builtin-claim-contract")
    _, revision = project.import_text("source.txt", "A researcher supplied this context.")
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(
        run,
        "builtin.social-actor-candidates",
        "1.0.0",
        "social actor",
        "A contextual claim was attached to a deterministic parser skill.",
        claim_types=[ClaimType.CONTEXTUAL],
    )
    context = project.user_context_evidence("research note", "Researcher supplied context.")
    project.attach_evidence(finding, context, EvidenceRole.CONTEXT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any(
        "built-in skill 'builtin.social-actor-candidates' does not allow claim type(s): CONTEXTUAL"
        in error
        for error in validation.errors
    )


def test_claim_contract_rejects_targeted_absence_before_persistence(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Canonical evidence.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=scope,
            selection_policy="targeted query with zero returned candidates",
            inspection_count=0,
            termination_reason="retrieval_completed",
            negative_result_status=NegativeResultStatus.NO_MATCHES_UNDER_PLAN,
        )
    )

    with pytest.raises(EvidenceValidationError, match="absence requires completed exhaustive"):
        project.create_finding(
            run,
            "test",
            "1.0.0",
            "absence",
            "No examples exist.",
            claim_types=[ClaimType.ABSENCE],
            coverage_id=coverage,
        )


@pytest.mark.parametrize(
    ("role", "expected_valid"),
    [(EvidenceRole.SUPPORT, True), (EvidenceRole.COUNTEREVIDENCE, False)],
)
def test_universal_claim_requires_supporting_evidence(tmp_path, role, expected_valid):
    project = Project.init(tmp_path / "universal-counterevidence-only")
    _, revision = project.import_text("source.txt", "Every document uses the term.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition="all documents in the immutable ScopeSnapshot",
            population_count=1,
            inspection_count=1,
            completion_criterion="every document inspected",
            termination_reason="completed deterministic enumeration",
        )
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "universal statement",
        "Every document uses the term.",
        claim_types=[ClaimType.UNIVERSAL],
        coverage_id=coverage_id,
    )
    evidence = project.textual_evidence(
        project.create_span(revision, 0, len("Every document uses the term."))
    )
    project.attach_evidence(finding, evidence, role)

    validation = validate_finding(project, finding)

    assert validation.valid is expected_valid
    if not expected_valid:
        assert any(
            "universal claim requires supporting evidence" in error for error in validation.errors
        )


def test_absence_claim_requires_exhaustive_zero_statistic_in_same_scope(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples and pears were discussed.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    statistic = frequency_statistic_evidence(project, scope, "migrants")
    coverage_id = statistic["coverage_id"]
    evidence_id = statistic["evidence_id"]
    assert isinstance(coverage_id, str)
    assert isinstance(evidence_id, str)
    coverage = project.coverage_descriptor(coverage_id)

    assert coverage.negative_result_status.value == "ABSENCE_IN_EXHAUSTIVE_POPULATION"
    finding = project.create_finding(
        run,
        "test",
        "1.0.0",
        "lexical absence",
        "The exact surface form is absent from the selected corpus population.",
        claim_types=[ClaimType.ABSENCE],
        coverage_id=coverage_id,
    )
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    assert validate_finding(project, finding).valid


@pytest.mark.parametrize("producer", ["dispersion", "bundle", "artifact_frequency"])
def test_core_count_statistic_producers_can_support_absence_claim(tmp_path, producer):
    project = Project.init(tmp_path / f"absence-{producer}")
    _, revision = project.import_text("source.txt", "Apples and pears.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    if producer == "dispersion":
        statistic = document_dispersion_evidence(project, scope, "migrants")
    elif producer == "bundle":
        statistic = lexical_bundle_statistic_evidence(project, scope, "migrants and", size=2)
    else:
        artifact = build_surface_token_artifact(project, revision)
        statistic = artifact_frequency_statistic_evidence(
            project, scope, [artifact.artifact_id], field="normalized", value="migrants"
        )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "absence",
        "The exact queried form is absent from the exhaustive population.",
        claim_types=[ClaimType.ABSENCE],
        coverage_id=str(statistic["coverage_id"]),
    )
    project.attach_evidence(finding, str(statistic["evidence_id"]), EvidenceRole.SUPPORT)

    assert validate_finding(project, finding).valid


@pytest.mark.parametrize(
    ("source_text", "bundle"),
    [
        ("cafe\u0301 one", "cafe one"),
        ("café one", "café one"),
    ],
)
def test_lexical_bundle_v1_evidence_revalidates_legacy_nfd_and_nfc_tokens(
    tmp_path, source_text, bundle
):
    project = Project.init(tmp_path / "bundle-v1-unicode-validation")
    _, revision = project.import_text("unicode.txt", source_text)
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    statistic = lexical_bundle_statistic_evidence(project, scope, bundle, size=2)
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "legacy lexical bundle frequency",
        "The v1 bundle count was recomputed using its recorded tokenizer.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=str(statistic["coverage_id"]),
    )
    project.attach_evidence(finding, str(statistic["evidence_id"]), EvidenceRole.SUPPORT)

    assert statistic["count"] == 1
    assert validate_finding(project, finding).valid


def test_historical_v1_document_dispersion_evidence_still_validates(tmp_path):
    project = Project.init(tmp_path / "legacy-dispersion-v1")
    _, revision = project.import_text("source.txt", "cafe\u0301 cafe")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    population = "all surface-token positions in every revision of the immutable ScopeSnapshot"
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition=population,
            population_count=2,
            selection_policy="exact, case-folded occurrence enumeration for 'cafe' over every scoped revision",
            inspection_count=2,
            completion_criterion="every scoped canonical text was tokenized with corpus.TOKEN and counted by revision",
            termination_reason="completed deterministic enumeration",
        )
    )
    evidence_id = project.corpus_statistic_evidence(
        "surface occurrences of 'cafe' across the scoped document population",
        2,
        scope_snapshot_id=scope,
        population_definition=population,
        provenance={
            "calculation_version": "surface-document-dispersion-v1",
            "coverage_id": coverage_id,
            "document_count": 1,
            "document_frequency": 1,
            "document_range": 1.0,
            "population_count": 2,
            "population_tokens": 2,
            "documents": [
                {"revision_id": revision, "occurrences": 2, "token_count": 2, "per_1000_tokens": 1000.0}
            ],
        },
    )
    finding_id = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "legacy exact count",
        "The legacy tokenizer counted two occurrences.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=coverage_id,
    )
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)

    assert validate_finding(project, finding_id).valid


def test_absence_claim_rejects_missing_or_nonzero_corpus_statistic(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples and pears were discussed.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})

    for value in (None, 1):
        coverage_id = project.coverage(
            CoverageDescriptor(
                coverage_mode=CoverageMode.EXHAUSTIVE,
                scope_snapshot_id=scope,
                population_definition="all surface-token positions in the selected revision",
                population_count=5,
                inspection_count=5,
                completion_criterion="all tokens enumerated",
                termination_reason="completed deterministic enumeration",
                negative_result_status=NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION,
            )
        )
        finding = project.create_finding(
            run,
            "test",
            "1.0.0",
            "lexical absence",
            "A lexical form is absent.",
            claim_types=[ClaimType.ABSENCE],
            coverage_id=coverage_id,
        )
        if value is None:
            evidence_id = project.textual_evidence(project.create_span(revision, 0, 6))
        else:
            evidence_id = project.corpus_statistic_evidence(
                "surface frequency",
                value,
                scope_snapshot_id=scope,
                population_definition="all surface-token positions in the selected revision",
                provenance={"coverage_id": coverage_id},
            )
        project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

        validation = validate_finding(project, finding)
        assert not validation.valid
        assert any("zero-valued corpus statistic" in error for error in validation.errors)


def test_absence_requires_zero_statistic_for_the_exact_declared_population(tmp_path):
    project = Project.init(tmp_path / "absence-population-mismatch")
    _, revision = project.import_text("source.txt", "Apples and pears.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    population = "all surface-token positions in the immutable ScopeSnapshot"
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition=population,
            population_count=3,
            inspection_count=3,
            completion_criterion="all surface-token positions enumerated",
            termination_reason="completed deterministic enumeration",
            negative_result_status=NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION,
        )
    )
    statistic = project.corpus_statistic_evidence(
        "zero character count",
        0,
        scope_snapshot_id=scope,
        population_definition="all character positions in the immutable ScopeSnapshot",
        provenance={
            "calculation_version": "fixture-count-v1",
            "coverage_id": coverage_id,
            "population_count": 3,
        },
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "absence",
        "The exact token is absent.",
        claim_types=[ClaimType.ABSENCE],
        coverage_id=coverage_id,
    )
    project.attach_evidence(finding, statistic, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("zero-valued corpus statistic" in error for error in validation.errors)


@pytest.mark.parametrize(
    ("claim", "value", "negative_result_status"),
    [
        (
            ClaimType.ABSENCE,
            0,
            NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION,
        ),
        (ClaimType.QUANTIFIED, 99, NegativeResultStatus.NOT_APPLICABLE),
    ],
)
def test_bundle_statistics_cannot_override_the_snapshot_count(
    tmp_path, claim, value, negative_result_status
):
    project = Project.init(tmp_path / f"bundle-scope-count-{claim.value.lower()}")
    _, revision = project.import_text("source.txt", "Apples and pears.")
    scope = project.snapshot([revision])
    population = "all contiguous 2-token surface n-gram positions in the immutable ScopeSnapshot"
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition=population,
            population_count=2,
            selection_policy="exact count for the bundle 'apples and'",
            inspection_count=2,
            completion_criterion="all contiguous bigrams enumerated",
            termination_reason="completed deterministic enumeration",
            negative_result_status=negative_result_status,
        )
    )
    evidence_id = project.corpus_statistic_evidence(
        "surface lexical bundle 'apples and' frequency",
        value,
        scope_snapshot_id=scope,
        population_definition=population,
        provenance={
            "calculation_version": "surface-lexical-bundles-v1-size-2",
            "coverage_id": coverage_id,
            "bundle": "apples and",
            "population_count": 2,
            "size": 2,
            "tokenizer": TOKEN.pattern,
        },
    )
    run = project.create_run(scope, {})
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "bundle result",
        "The bundle result matches the declared claim.",
        claim_types=[claim],
        coverage_id=coverage_id,
    )
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("does not match its ScopeSnapshot coverage" in error for error in validation.errors)


def test_absence_zero_statistic_in_counterevidence_role_does_not_support_claim(tmp_path):
    project = Project.init(tmp_path / "absence-statistic-role")
    _, revision = project.import_text("source.txt", "Apples and pears.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    statistic = frequency_statistic_evidence(project, scope, "migrants")
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "absence",
        "The exact token is absent.",
        claim_types=[ClaimType.ABSENCE],
        coverage_id=str(statistic["coverage_id"]),
    )
    project.attach_evidence(finding, str(statistic["evidence_id"]), EvidenceRole.COUNTEREVIDENCE)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("zero-valued corpus statistic" in error for error in validation.errors)


def test_quantified_claim_requires_persisted_exhaustive_statistic_evidence(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples apples.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    statistic = frequency_statistic_evidence(project, scope, "apples")
    coverage = str(statistic["coverage_id"])
    finding = project.create_finding(
        run,
        "test",
        "1.0.0",
        "token count",
        "The selected revision contains exactly two occurrences of 'apples'.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=coverage,
    )
    evidence_id = str(statistic["evidence_id"])
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    assert validate_finding(project, finding).valid
    assert "surface frequency" in render_finding_markdown(project, finding)


def test_quantified_statistic_in_counterevidence_role_does_not_support_claim(tmp_path):
    project = Project.init(tmp_path / "quantified-statistic-role")
    _, revision = project.import_text("source.txt", "Apples apples.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    statistic = frequency_statistic_evidence(project, scope, "apples")
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "surface frequency",
        "The selected population contains exactly two occurrences.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=str(statistic["coverage_id"]),
    )
    project.attach_evidence(finding, str(statistic["evidence_id"]), EvidenceRole.COUNTEREVIDENCE)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("requires deterministic statistic evidence" in error for error in validation.errors)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_corpus_statistic_rejects_non_finite_values_at_write_time(tmp_path, value):
    project = Project.init(tmp_path / "non-finite-statistic-write")
    _, revision = project.import_text("source.txt", "Apples apples.")
    scope = project.snapshot([revision])

    with pytest.raises(EvidenceValidationError, match="finite and numeric"):
        project.corpus_statistic_evidence(
            "invalid numeric result",
            value,
            scope_snapshot_id=scope,
            population_definition="all surface-token positions in the immutable ScopeSnapshot",
        )


def test_validator_rejects_non_finite_quantified_statistic_after_storage_tampering(tmp_path):
    import json

    project = Project.init(tmp_path / "non-finite-statistic-validation")
    _, revision = project.import_text("source.txt", "Apples apples.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    statistic = frequency_statistic_evidence(project, scope, "apples")
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "surface frequency",
        "The selected population contains exactly two occurrences.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=str(statistic["coverage_id"]),
    )
    evidence_id = str(statistic["evidence_id"])
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)
    payload = project.evidence_content(evidence_id)
    payload["value"] = float("nan")
    project.db.execute(
        "UPDATE evidence_artifacts SET content_json=? WHERE id=?",
        (json.dumps(payload), evidence_id),
    )

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("non-finite value" in error for error in validation.errors)


def test_finding_validation_explicitly_does_not_assess_semantic_entailment(tmp_path):
    project = Project.init(tmp_path / "structural-not-semantic")
    _, revision = project.import_text("source.txt", "Officials made mistakes.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage_id = project.coverage(
        CoverageDescriptor(coverage_mode=CoverageMode.EXAMPLE_BASED, scope_snapshot_id=scope)
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "agency",
        "The author deliberately concealed the government as agent.",
        interpretation="The construction backgrounds agency.",
        alternative_interpretation="The wording may be conventional or reflect unknown context.",
        claim_types=[ClaimType.INTERPRETIVE],
        limitations=["This fixture does not establish author intent."],
        coverage_id=coverage_id,
    )
    evidence = project.textual_evidence(
        project.create_span(revision, 0, len("Officials made mistakes."))
    )
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)

    result = validate_finding(project, finding)

    assert result.valid  # Structural validity is not semantic entailment.
    assert result.semantic_entailment_assessed is False


def test_quantified_claim_rejects_statistic_for_a_different_population(tmp_path):
    project = Project.init(tmp_path / "different-population")
    _, revision = project.import_text("source.txt", "Apples apples.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition="all character positions in the revision",
            population_count=len("Apples apples."),
            completion_criterion="every canonical character position enumerated",
            termination_reason="completed deterministic enumeration",
        )
    )
    evidence_id = project.corpus_statistic_evidence(
        "surface token frequency for 'apples'",
        2,
        scope_snapshot_id=scope,
        population_definition="all surface-token positions in the immutable ScopeSnapshot",
        provenance={
            "calculation_version": "surface-frequency-v1",
            "coverage_id": coverage_id,
        },
    )
    finding = project.create_finding(
        run,
        "test",
        "1.0.0",
        "surface frequency",
        "The corpus contains exactly two occurrences.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=coverage_id,
    )
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("same coverage and population" in error for error in validation.errors)


def test_quantified_claim_rejects_unlinked_or_unversioned_statistics(tmp_path):
    project = Project.init(tmp_path / "unlinked-statistic")
    _, revision = project.import_text("source.txt", "Apples apples.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    population = "all surface-token positions in the immutable ScopeSnapshot"
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition=population,
            population_count=2,
            completion_criterion="all surface-token positions enumerated",
            termination_reason="completed deterministic enumeration",
        )
    )
    evidence_ids = [
        project.corpus_statistic_evidence(
            "unversioned frequency",
            2,
            scope_snapshot_id=scope,
            population_definition=population,
        ),
        project.corpus_statistic_evidence(
            "unlinked frequency",
            2,
            scope_snapshot_id=scope,
            population_definition=population,
            provenance={"calculation_version": "surface-frequency-v1"},
        ),
    ]

    for evidence_id in evidence_ids:
        finding = project.create_finding(
            run,
            "test",
            "1.0.0",
            "surface frequency",
            "The corpus contains exactly two occurrences.",
            claim_types=[ClaimType.QUANTIFIED],
            coverage_id=coverage_id,
        )
        project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

        validation = validate_finding(project, finding)

        assert not validation.valid
        assert any("same coverage and population" in error for error in validation.errors)


def test_quantified_claim_rejects_user_supplied_calculation_version(tmp_path):
    project = Project.init(tmp_path / "unknown-calculation-version")
    _, revision = project.import_text("source.txt", "Apples apples.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    population = "all surface-token positions in the immutable ScopeSnapshot"
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition=population,
            population_count=2,
            completion_criterion="all surface-token positions enumerated",
            termination_reason="completed deterministic enumeration",
        )
    )
    evidence_id = project.corpus_statistic_evidence(
        "user-supplied statistic",
        2,
        scope_snapshot_id=scope,
        population_definition=population,
        provenance={
            "calculation_version": "researcher-script-v42",
            "coverage_id": coverage_id,
            "population_count": 2,
            "term": "apples",
            "tokenizer": r"[\\w]+",
        },
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "surface frequency",
        "The population contains exactly two occurrences.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=coverage_id,
    )
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("requires deterministic statistic evidence" in error for error in validation.errors)


def test_absence_claim_rejects_user_supplied_calculation_version(tmp_path):
    project = Project.init(tmp_path / "unknown-absence-calculation-version")
    _, revision = project.import_text("source.txt", "Apples and pears.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    statistic = frequency_statistic_evidence(project, scope, "migrants")
    evidence_id = str(statistic["evidence_id"])
    payload = project.evidence_content(evidence_id)
    provenance = cast(dict[str, object], payload["provenance"])
    provenance["calculation_version"] = "researcher-script-v42"
    project.db.execute(
        "UPDATE evidence_artifacts SET content_json=? WHERE id=?",
        (json.dumps(payload), evidence_id),
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "absence",
        "The exact token is absent.",
        claim_types=[ClaimType.ABSENCE],
        coverage_id=str(statistic["coverage_id"]),
    )
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("zero-valued corpus statistic" in error for error in validation.errors)


def test_absence_claim_rejects_frequency_provenance_for_another_term(tmp_path):
    project = Project.init(tmp_path / "mislabeled-frequency-absence")
    _, revision = project.import_text("source.txt", "Apples and pears.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    statistic = frequency_statistic_evidence(project, scope, "migrants")
    evidence_id = str(statistic["evidence_id"])

    payload = project.evidence_content(evidence_id)
    provenance = cast(dict[str, object], payload["provenance"])
    provenance["term"] = "apples"
    project.db.execute(
        "UPDATE evidence_artifacts SET content_json=? WHERE id=?",
        (json.dumps(payload), evidence_id),
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "lexical absence",
        "Apples are absent from the selected corpus.",
        claim_types=[ClaimType.ABSENCE],
        coverage_id=str(statistic["coverage_id"]),
    )
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("does not match its ScopeSnapshot coverage" in error for error in validation.errors)


def test_quantified_claim_accepts_core_lexical_bundle_statistic(tmp_path):
    project = Project.init(tmp_path / "recognized-lexical-bundle")
    _, revision = project.import_text("source.txt", "The public must act. The public must act.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    statistic = lexical_bundle_statistic_evidence(project, scope, "the public must", size=3)
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "bundle count",
        "The exact bundle occurs twice in the selected population.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=str(statistic["coverage_id"]),
    )
    project.attach_evidence(finding, str(statistic["evidence_id"]), EvidenceRole.SUPPORT)

    assert validate_finding(project, finding).valid


def test_quantified_claim_rejects_statistic_linked_to_another_coverage(tmp_path):
    project = Project.init(tmp_path / "other-coverage")
    _, revision = project.import_text("source.txt", "Apples apples.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    population = "all surface-token positions in the immutable ScopeSnapshot"
    coverage_ids = [
        project.coverage(
            CoverageDescriptor(
                coverage_mode=CoverageMode.EXHAUSTIVE,
                scope_snapshot_id=scope,
                population_definition=population,
                population_count=2,
                completion_criterion="all surface-token positions enumerated",
                termination_reason="completed deterministic enumeration",
            )
        )
        for _ in range(2)
    ]
    evidence_id = project.corpus_statistic_evidence(
        "surface frequency for 'apples'",
        2,
        scope_snapshot_id=scope,
        population_definition=population,
        provenance={
            "calculation_version": "surface-frequency-v1",
            "coverage_id": coverage_ids[0],
        },
    )
    finding = project.create_finding(
        run,
        "test",
        "1.0.0",
        "surface frequency",
        "The corpus contains exactly two occurrences.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=coverage_ids[1],
    )
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("same coverage and population" in error for error in validation.errors)


def test_general_corpus_statistic_provenance_validates_within_original_scope(tmp_path):
    project = Project.init(tmp_path / "statistic-provenance")
    _, revision = project.import_text("sample.txt", "apple red apple red apple blue")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    evidence_results = [
        frequency_statistic_evidence(project, scope, "apple"),
        document_dispersion_evidence(project, scope, "apple"),
        collocation_association_evidence(project, scope, "apple", "red", window=1, min_frequency=1),
    ]

    for result in evidence_results:
        finding = project.create_finding(
            run,
            "builtin.corpus-statistic",
            "1.0.0",
            "descriptive-corpus-statistic",
            "The deterministic calculation produced the recorded statistic.",
            claim_types=[ClaimType.QUANTIFIED],
            evidence_sufficiency="SUFFICIENT",
            interpretation_confidence="HIGH",
            coverage_id=str(result["coverage_id"]),
        )
        project.attach_evidence(finding, str(result["evidence_id"]), EvidenceRole.SUPPORT)
        assert validate_finding(project, finding).valid


def test_surface_frequency_v2_recomputes_nfd_counts_and_rejects_unsupported_ucd(tmp_path):
    project = Project.init(tmp_path / "frequency-v2-validation")
    _, revision = project.import_text("unicode.txt", "café cafe\u0301")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    statistic = frequency_statistic_evidence(project, scope, "cafe\u0301")
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "normalized frequency",
        "The selected scope contains exactly two canonically equivalent forms.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=str(statistic["coverage_id"]),
    )
    evidence_id = str(statistic["evidence_id"])
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    assert validate_finding(project, finding).valid
    assert project.evidence_content(evidence_id)["value"] == 2

    payload = project.evidence_content(evidence_id)
    provenance = cast(dict[str, object], payload["provenance"])
    provenance["unicode_data_version"] = "unsupported-ucd"
    project.db.execute(
        "UPDATE evidence_artifacts SET content_json=? WHERE id=?",
        (json.dumps(payload), evidence_id),
    )
    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("unsupported or invalid tokenizer provenance" in error for error in validation.errors)


def test_surface_window_association_v2_validates_nfd_with_shared_tokenizer(tmp_path):
    project = Project.init(tmp_path / "association-v2-tokenizer-validation")
    _, revision = project.import_text("unicode.txt", "cafe\u0301 red café blue")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    statistic = collocation_association_evidence(
        project, scope, "cafe\u0301", "red", window=1, min_frequency=1
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "normalized window association",
        "The versioned association calculation recorded this scoped result.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=str(statistic["coverage_id"]),
    )
    evidence_id = str(statistic["evidence_id"])
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    assert validate_finding(project, finding).valid
    payload = project.evidence_content(evidence_id)
    provenance = cast(dict[str, object], payload["provenance"])
    assert provenance["query"] == "café"
    assert provenance["tokenizer"] == "unicode-surface-tokenizer-v2"
    assert provenance["normalization"] == "NFC-after-casefold-v1"

    provenance["unicode_data_version"] = "unsupported-ucd"
    project.db.execute(
        "UPDATE evidence_artifacts SET content_json=? WHERE id=?",
        (json.dumps(payload), evidence_id),
    )
    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("invalid calculation provenance" in error for error in validation.errors)


def test_surface_window_association_keeps_recorded_legacy_tokenizer_verifiable(tmp_path):
    project = Project.init(tmp_path / "association-v2-legacy-tokenizer")
    _, revision = project.import_text("legacy.txt", "target neighbor")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    population_definition = (
        "all non-focus in-window co-occurrence opportunities for the query token "
        "in every revision of the immutable ScopeSnapshot"
    )
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition=population_definition,
            population_count=1,
            completion_criterion=(
                "every scope revision tokenized by corpus.TOKEN; all focus positions and "
                "in-window pairs enumerated without crossing revision boundaries"
            ),
            termination_reason="completed deterministic window-pair enumeration",
        )
    )
    evidence_id = project.corpus_statistic_evidence(
        "window mutual information for 'target' and 'neighbor'",
        0.0,
        scope_snapshot_id=scope,
        population_definition=population_definition,
        provenance={
            "calculation_version": "surface-window-association-v2",
            "coverage_id": coverage_id,
            "population_count": 1,
            "measure": "mutual_information_log2",
            "query": "target",
            "collocate": "neighbor",
            "window": 1,
            "min_frequency": 1,
            "focus_frequency": 1,
            "pair_frequency": 1,
            "token_frequency": 1,
            "population_tokens": 2,
            "pair_opportunities": 1,
            "expected_frequency": 1.0,
            "mutual_information": 0.0,
            "t_score": 0.0,
            "tokenizer": TOKEN.pattern,
        },
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "legacy window association",
        "This saved association records its original tokenizer explicitly.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=coverage_id,
    )
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    assert validate_finding(project, finding).valid


def test_historical_surface_frequency_v1_recalculates_with_legacy_tokenizer(tmp_path):
    project = Project.init(tmp_path / "frequency-v1-validation")
    _, revision = project.import_text("unicode.txt", "cafe\u0301")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    matches = list(TOKEN.finditer(project.revision_text(revision)))
    population = len(matches)
    term = "cafe"
    count = sum(match.group(0).casefold() == term for match in matches)
    population_definition = "all surface-token positions in the immutable ScopeSnapshot"
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition=population_definition,
            population_count=population,
            inspection_count=population,
            completion_criterion="all scope texts tokenized by corpus.TOKEN and enumerated",
            termination_reason="completed deterministic enumeration",
        )
    )
    evidence_id = project.corpus_statistic_evidence(
        "historical v1 surface frequency",
        count,
        scope_snapshot_id=scope,
        population_definition=population_definition,
        provenance={
            "calculation_version": "surface-frequency-v1",
            "coverage_id": coverage_id,
            "population_count": population,
            "population_tokens": population,
            "term": term,
            "tokenizer": TOKEN.pattern,
        },
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "historical frequency",
        "The historical tokenizer counted the legacy surface token.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=coverage_id,
    )
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    assert count == 1
    assert validate_finding(project, finding).valid


def test_kwic_v2_validator_recomputes_nfd_evidence_and_offsets(tmp_path):
    project = Project.init(tmp_path / "kwic-v2-validation")
    source_text = "🐦 cafe\u0301 migration"
    _, revision = project.import_text("unicode.txt", source_text)
    scope = project.snapshot([revision])
    finding_scope = project.snapshot([revision])
    assert finding_scope != scope
    run = project.create_run(finding_scope, {})
    result = kwic_evidence(project, scope, revision, "café", 2, 7)
    evidence_id = str(result["evidence_id"])
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "Unicode excerpt",
        "The cited span preserves the canonical decomposed surface form.",
    )
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    assert result["exact_text"] == "cafe\u0301"
    assert validate_finding(project, finding).valid

    provenance = project.evidence_content(evidence_id)
    provenance["candidate_count"] = 99
    project.db.execute(
        "UPDATE evidence_artifacts SET content_json=? WHERE id=?",
        (json.dumps(provenance), evidence_id),
    )
    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("does not match its exhaustive population" in error for error in validation.errors)


def test_historical_kwic_v1_evidence_remains_valid(tmp_path):
    project = Project.init(tmp_path / "kwic-v1-validation")
    _, revision = project.import_text("legacy.txt", "Migration migration policy.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    surfaces = list(TOKEN.finditer(project.revision_text(revision)))
    population = len(surfaces)
    candidate_count = sum(match.group(0).casefold() == "migration" for match in surfaces)
    population_definition = "all surface-token positions in the immutable ScopeSnapshot"
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition=population_definition,
            population_count=population,
            inspection_count=population,
            completion_criterion="all scope texts tokenized by corpus.TOKEN and enumerated",
            termination_reason="completed deterministic exact-token enumeration",
        )
    )
    span_id = project.create_span(revision, 0, len("Migration"))
    evidence_id = project.textual_evidence(
        span_id,
        provenance={
            "acquisition_method": "corpus_kwic",
            "query": "migration",
            "scope_snapshot_id": scope,
            "coverage_descriptor_id": coverage_id,
            "coverage_mode": "EXHAUSTIVE",
            "candidate_count": candidate_count,
            "inspected_count": population,
            "tokenizer": TOKEN.pattern,
            "calculation_version": "surface-kwic-v1",
        },
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "historical excerpt",
        "The historical KWIC producer cited an exact legacy token.",
        claim_types=[ClaimType.EXAMPLE_BASED],
        coverage_id=coverage_id,
    )
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    assert validate_finding(project, finding).valid


def test_claim_contract_rejects_coverage_from_another_scope(tmp_path):
    project = Project.init(tmp_path / "p")
    _, first = project.import_text("first.txt", "First.")
    _, second = project.import_text("second.txt", "Second.")
    run = project.create_run(project.snapshot([first]), {})
    other_scope = project.snapshot([second])
    coverage = project.coverage(
        CoverageDescriptor(coverage_mode=CoverageMode.EXAMPLE_BASED, scope_snapshot_id=other_scope)
    )

    with pytest.raises(EvidenceValidationError, match="original ScopeSnapshot"):
        project.create_finding(
            run,
            "test",
            "1.0.0",
            "reading",
            "An interpretation.",
            claim_types=[ClaimType.INTERPRETIVE],
            limitations=["Limited evidence."],
            coverage_id=coverage,
        )


@pytest.mark.parametrize(
    ("interpretation", "alternative_interpretation", "limitations"),
    [
        (None, "An alternative reading.", ["One textual example."]),
        ("An interpretation.", None, ["One textual example."]),
        ("An interpretation.", "An alternative reading.", []),
        (" ", "An alternative reading.", ["One textual example."]),
        ("An interpretation.", " ", ["One textual example."]),
        ("An interpretation.", "An alternative reading.", [" "]),
    ],
)
def test_interpretive_claim_requires_interpretation_alternative_and_limitations(
    tmp_path, interpretation, alternative_interpretation, limitations
):
    project = Project.init(tmp_path / "interpretive-contract")
    _, revision = project.import_text("source.txt", "Officials must act.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage = project.coverage(
        CoverageDescriptor(coverage_mode=CoverageMode.EXAMPLE_BASED, scope_snapshot_id=scope)
    )

    with pytest.raises(EvidenceValidationError, match="interpretive claim requires"):
        project.create_finding(
            run,
            "fixture",
            "1.0.0",
            "modality",
            "The wording expresses obligation.",
            interpretation=interpretation,
            alternative_interpretation=alternative_interpretation,
            claim_types=[ClaimType.INTERPRETIVE],
            limitations=limitations,
            coverage_id=coverage,
        )


def test_interpretive_claim_validates_with_alternative_and_limitations(tmp_path):
    project = Project.init(tmp_path / "complete-interpretive-contract")
    _, revision = project.import_text("source.txt", "Officials must act.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage = project.coverage(
        CoverageDescriptor(coverage_mode=CoverageMode.EXAMPLE_BASED, scope_snapshot_id=scope)
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "modality",
        "The wording expresses obligation.",
        interpretation="The modal verb constructs action as obligatory.",
        alternative_interpretation="It may be conventional procedural language.",
        claim_types=[ClaimType.INTERPRETIVE],
        limitations=["This is one sentence without broader context."],
        coverage_id=coverage,
    )
    evidence = project.textual_evidence(project.create_span(revision, 0, 19))
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)

    assert validate_finding(project, finding).valid


def test_validation_catches_interpretive_claim_missing_alternative_after_persistence(tmp_path):
    project = Project.init(tmp_path / "corrupt-interpretive-claim")
    _, revision = project.import_text("source.txt", "Officials must act.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage = project.coverage(
        CoverageDescriptor(coverage_mode=CoverageMode.EXAMPLE_BASED, scope_snapshot_id=scope)
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "modality",
        "The wording expresses obligation.",
        interpretation="The modal verb constructs action as obligatory.",
        alternative_interpretation="It may be conventional procedural language.",
        claim_types=[ClaimType.INTERPRETIVE],
        limitations=["One sentence only."],
        coverage_id=coverage,
    )
    evidence = project.textual_evidence(project.create_span(revision, 0, 19))
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    project.db.execute("UPDATE findings SET alternative_interpretation=NULL WHERE id=?", (finding,))
    project.db.commit()

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("claim type INTERPRETIVE" in error for error in validation.errors)


def test_observational_claim_requires_supporting_textual_evidence(tmp_path):
    project = Project.init(tmp_path / "observational-evidence-kind")
    _, revision = project.import_text("source.txt", "Officials must act.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "modal count",
        "A corpus statistic is recorded.",
        claim_types=[ClaimType.OBSERVATIONAL],
    )
    statistic = project.corpus_statistic_evidence(
        "surface token frequency",
        1,
        scope_snapshot_id=scope,
        population_definition="all surface-token positions in the ScopeSnapshot",
    )
    project.attach_evidence(finding, statistic, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("require supporting textual evidence" in error for error in validation.errors)


@pytest.mark.parametrize(
    ("claim", "role"),
    [
        (ClaimType.OBSERVATIONAL, EvidenceRole.SUPPORT),
        (ClaimType.EXAMPLE_BASED, EvidenceRole.EXAMPLE),
    ],
)
def test_observational_and_example_claims_validate_with_supporting_text(tmp_path, claim, role):
    project = Project.init(tmp_path / f"{claim.value.lower()}-text")
    _, revision = project.import_text("source.txt", "Officials must act.")
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "textual form",
        "The cited construction is present.",
        claim_types=[claim],
    )
    evidence = project.textual_evidence(project.create_span(revision, 0, 19))
    project.attach_evidence(finding, evidence, role)

    assert validate_finding(project, finding).valid


def test_counterevidence_only_text_does_not_support_an_observational_claim(tmp_path):
    project = Project.init(tmp_path / "counterevidence-is-not-support")
    _, revision = project.import_text("source.txt", "Officials must act.")
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "textual form",
        "The cited construction is present.",
        claim_types=[ClaimType.OBSERVATIONAL],
    )
    evidence = project.textual_evidence(project.create_span(revision, 0, 19))
    project.attach_evidence(finding, evidence, EvidenceRole.COUNTEREVIDENCE)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("require supporting textual evidence" in error for error in validation.errors)


def test_example_based_claim_requires_supporting_textual_evidence(tmp_path):
    project = Project.init(tmp_path / "example-evidence-kind")
    _, revision = project.import_text("source.txt", "Officials must act.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "example",
        "A researcher context note is attached.",
        claim_types=[ClaimType.EXAMPLE_BASED],
    )
    context = project.user_context_evidence("memo", "Researcher context.")
    project.attach_evidence(finding, context, EvidenceRole.EXAMPLE)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("require supporting textual evidence" in error for error in validation.errors)


def test_interpretive_claim_requires_supporting_textual_evidence(tmp_path):
    project = Project.init(tmp_path / "interpretive-evidence-kind")
    _, revision = project.import_text("source.txt", "Officials must act.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage = project.coverage(
        CoverageDescriptor(coverage_mode=CoverageMode.EXAMPLE_BASED, scope_snapshot_id=scope)
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "modality",
        "The wording expresses obligation.",
        interpretation="The modal verb constructs action as obligatory.",
        alternative_interpretation="It may be conventional procedural language.",
        claim_types=[ClaimType.INTERPRETIVE],
        limitations=["One sentence only."],
        coverage_id=coverage,
    )
    context = project.user_context_evidence("memo", "Researcher context.")
    project.attach_evidence(finding, context, EvidenceRole.CONTEXT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any(
        "interpretive claims require linked textual evidence" in error
        for error in validation.errors
    )


def test_contextual_claim_requires_contextual_evidence_kind(tmp_path):
    project = Project.init(tmp_path / "contextual-evidence-kind")
    _, revision = project.import_text("source.txt", "Officials must act.")
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "historical context",
        "The passage is contextualized.",
        claim_types=[ClaimType.CONTEXTUAL],
    )
    text_evidence = project.textual_evidence(project.create_span(revision, 0, 19))
    project.attach_evidence(finding, text_evidence, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("require external-source" in error for error in validation.errors)


def test_comparative_finding_requires_linked_comparison_populations(tmp_path):
    project = Project.init(tmp_path / "comparative-claim-contract")
    _, revision = project.import_text("source.txt", "Officials must act.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition="all surface-token positions in the selected revision",
            population_count=3,
            completion_criterion="every surface token was enumerated",
            termination_reason="completed deterministic enumeration",
        )
    )
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "comparative framing",
        "The representation differs across populations.",
        claim_types=[ClaimType.COMPARATIVE],
        coverage_id=coverage,
    )
    evidence = project.textual_evidence(project.create_span(revision, 0, 19))
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("linked comparison evidence" in error for error in validation.errors)


def _comparison_finding_fixture(
    project, *, comparison_token="migration", equal_rates=False, metric="frequency"
):
    left_text = "Migration apple." if equal_rates else "Migration migration public policy."
    right_text = "Migration pear." if equal_rates else "Migration public discourse."
    _, left = project.import_text("left.txt", left_text)
    _, right = project.import_text("right.txt", right_text)
    union_scope = project.snapshot([left, right])
    analysis_run = project.create_run(union_scope, {"fixture": "comparative-evidence"})
    comparison = compare_documents(project, left, right, metric=metric)
    result = project.corpus_comparison_evidence(
        analysis_run,
        str(comparison["comparison_id"]),
        comparison_token,
        comparability_basis="Both texts are policy speeches from the same period and genre.",
    )
    finding = project.create_finding(
        analysis_run,
        "fixture.comparison",
        "1.0.0",
        "surface-frequency contrast",
        "The selected token's relative frequencies differ between the two populations.",
        claim_types=[ClaimType.COMPARATIVE],
        coverage_id=str(result["coverage_id"]),
    )
    project.attach_evidence(finding, str(result["evidence_id"]), EvidenceRole.SUPPORT)
    return {
        "analysis_run": analysis_run,
        "comparison": comparison,
        "finding": finding,
        "result": result,
        "left": left,
        "right": right,
    }


def test_comparative_evidence_validates_both_scopes_row_and_basis(tmp_path):
    from discourselens.reporting import finding_payload

    project = Project.init(tmp_path / "comparison-evidence")
    fixture = _comparison_finding_fixture(project)

    assert validate_finding(project, fixture["finding"]).valid
    evidence = project.evidence_content(str(fixture["result"]["evidence_id"]))
    provenance = cast(dict[str, object], evidence["provenance"])
    assert provenance["left_revision_ids"] == [fixture["left"]]
    assert provenance["right_revision_ids"] == [fixture["right"]]
    assert str(provenance["comparability_basis"]).startswith("Both texts")
    assert (
        provenance["comparison_result_hash"]
        == project.comparison_run(str(fixture["comparison"]["comparison_id"]))["result_hash"]
    )
    assert project.render_citation(str(fixture["result"]["evidence_id"])) == (
        f"[comparison · frequency · migration · {fixture['comparison']['comparison_id']}]"
    )
    rendered = finding_payload(project, fixture["finding"])["evidence"][0]["text"]
    assert '"left_scope_snapshot_id"' in str(rendered)
    assert '"right_scope_snapshot_id"' in str(rendered)
    assert "not independently assessed" in str(rendered)


def test_keyness_comparison_evidence_is_linkable_and_validated(tmp_path):
    project = Project.init(tmp_path / "keyness-comparison-evidence")
    fixture = _comparison_finding_fixture(project, metric="keyness")

    assert fixture["result"]["metric"] == "keyness"
    assert validate_finding(project, fixture["finding"]).valid
    assert "comparison · keyness" in project.render_citation(str(fixture["result"]["evidence_id"]))


def test_comparison_evidence_requires_exact_union_scope_and_nonblank_basis(tmp_path):
    project = Project.init(tmp_path / "comparison-scope-contract")
    _, left = project.import_text("left.txt", "Migration policy.")
    _, right = project.import_text("right.txt", "Migration discourse.")
    comparison = compare_documents(project, left, right)
    wrong_scope_run = project.create_run(project.snapshot([left]), {})

    with pytest.raises(ScopeViolationError, match="scope must equal the union"):
        project.corpus_comparison_evidence(
            wrong_scope_run,
            str(comparison["comparison_id"]),
            "migration",
            comparability_basis="same genre",
        )

    correct_scope_run = project.create_run(project.snapshot([left, right]), {})
    with pytest.raises(ScopeViolationError, match="comparability basis"):
        project.corpus_comparison_evidence(
            correct_scope_run,
            str(comparison["comparison_id"]),
            "migration",
            comparability_basis="   ",
        )


def test_comparison_evidence_requires_an_existing_result_row(tmp_path):
    project = Project.init(tmp_path / "comparison-missing-row")
    _, left = project.import_text("left.txt", "Migration policy.")
    _, right = project.import_text("right.txt", "Migration discourse.")
    comparison = compare_documents(project, left, right)
    run = project.create_run(project.snapshot([left, right]), {})

    with pytest.raises(ScopeViolationError, match="not present"):
        project.corpus_comparison_evidence(
            run,
            str(comparison["comparison_id"]),
            "unlisted",
            comparability_basis="same genre",
        )


def test_comparison_evidence_rejects_result_hash_or_selected_row_tampering(tmp_path):
    import json

    project = Project.init(tmp_path / "comparison-integrity")
    fixture = _comparison_finding_fixture(project)
    comparison_id = str(fixture["comparison"]["comparison_id"])
    original_hash = project.comparison_run(comparison_id)["result_hash"]
    project.db.execute(
        "UPDATE comparison_runs SET result_hash=? WHERE id=?", ("0" * 64, comparison_id)
    )

    validation = validate_finding(project, fixture["finding"])
    assert not validation.valid
    assert any("comparison result hash mismatch" in error for error in validation.errors)

    project.db.execute(
        "UPDATE comparison_runs SET result_hash=? WHERE id=?",
        (original_hash, comparison_id),
    )
    evidence_id = str(fixture["result"]["evidence_id"])
    payload = project.evidence_content(evidence_id)
    provenance = cast(dict[str, object], payload["provenance"])
    selected_row = cast(dict[str, object], provenance["selected_row"])
    left_count = selected_row["left_count"]
    assert isinstance(left_count, int)
    selected_row["left_count"] = left_count + 100
    project.db.execute(
        "UPDATE evidence_artifacts SET content_json=? WHERE id=?",
        (json.dumps(payload, ensure_ascii=False), evidence_id),
    )

    validation = validate_finding(project, fixture["finding"])
    assert not validation.valid
    assert any("selected row does not match storage" in error for error in validation.errors)


def test_comparison_evidence_rejects_tampered_population_scope_provenance(tmp_path):
    import json

    project = Project.init(tmp_path / "comparison-scope-tamper")
    fixture = _comparison_finding_fixture(project)
    evidence_id = str(fixture["result"]["evidence_id"])
    payload = project.evidence_content(evidence_id)
    provenance = cast(dict[str, object], payload["provenance"])
    provenance["right_scope_snapshot_id"] = provenance["left_scope_snapshot_id"]
    project.db.execute(
        "UPDATE evidence_artifacts SET content_json=? WHERE id=?",
        (json.dumps(payload, ensure_ascii=False), evidence_id),
    )

    validation = validate_finding(project, fixture["finding"])

    assert not validation.valid
    assert any("comparison run provenance does not match" in error for error in validation.errors)


def test_comparison_statistic_cannot_substitute_for_single_population_count(tmp_path):
    import json

    project = Project.init(tmp_path / "comparison-not-quantified")
    fixture = _comparison_finding_fixture(project)
    finding = project.create_finding(
        str(fixture["analysis_run"]),
        "fixture.comparison",
        "1.0.0",
        "misclassified comparison statistic",
        "A comparison difference is not a single-population count.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=str(fixture["result"]["coverage_id"]),
    )
    project.attach_evidence(finding, str(fixture["result"]["evidence_id"]), EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any(
        "exact quantified claim requires deterministic statistic" in error
        for error in validation.errors
    )

    evidence_id = str(fixture["result"]["evidence_id"])
    payload = project.evidence_content(evidence_id)
    provenance = cast(dict[str, object], payload["provenance"])
    provenance.pop("comparison_run_id")
    project.db.execute(
        "UPDATE evidence_artifacts SET content_json=? WHERE id=?",
        (json.dumps(payload, ensure_ascii=False), evidence_id),
    )
    validation = validate_finding(project, finding)
    assert not validation.valid
    assert any("comparison statistic evidence" in error for error in validation.errors)


def test_zero_valued_comparison_difference_cannot_prove_corpus_absence(tmp_path):
    import json

    project = Project.init(tmp_path / "comparison-not-absence")
    fixture = _comparison_finding_fixture(project, equal_rates=True)
    evidence_id = str(fixture["result"]["evidence_id"])
    content = project.evidence_content(evidence_id)
    assert content["value"] == 0
    comparison_coverage = project.coverage_descriptor(str(fixture["result"]["coverage_id"]))
    absence_coverage_id = project.coverage(
        CoverageDescriptor(
            **{
                **comparison_coverage.model_dump(),
                "negative_result_status": NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION,
            }
        )
    )
    provenance = cast(dict[str, object], content["provenance"])
    provenance["coverage_id"] = absence_coverage_id
    project.db.execute(
        "UPDATE evidence_artifacts SET content_json=? WHERE id=?",
        (json.dumps(content, ensure_ascii=False), evidence_id),
    )
    finding = project.create_finding(
        str(fixture["analysis_run"]),
        "fixture.comparison",
        "1.0.0",
        "false absence",
        "A zero rate difference does not mean that the token is absent.",
        claim_types=[ClaimType.ABSENCE],
        coverage_id=absence_coverage_id,
    )
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any(
        "absence claim requires a zero-valued corpus statistic" in error
        for error in validation.errors
    )


@pytest.mark.parametrize("evidence_type", ["external", "user"])
def test_contextual_claim_accepts_provenanced_external_or_labeled_user_context(
    tmp_path, evidence_type
):
    project = Project.init(tmp_path / f"contextual-valid-{evidence_type}")
    _, revision = project.import_text("source.txt", "Officials must act.")
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "historical context",
        "The contextual source records the stated date.",
        claim_types=[ClaimType.CONTEXTUAL],
    )
    if evidence_type == "external":
        evidence = project.external_source_evidence(
            "Official archive",
            "The policy took effect in 2025.",
            url="https://example.test/policy",
            retrieved_at="2026-09-25",
        )
    else:
        evidence = project.user_context_evidence("research memo", "Researcher supplied context.")
    project.attach_evidence(finding, evidence, EvidenceRole.CONTEXT)

    assert validate_finding(project, finding).valid


def test_coverage_descriptor_requires_an_existing_scope_snapshot(tmp_path):
    project = Project.init(tmp_path / "p")

    with pytest.raises(ScopeViolationError, match="unknown scope"):
        project.coverage(
            CoverageDescriptor(
                coverage_mode=CoverageMode.EXAMPLE_BASED, scope_snapshot_id="missing"
            )
        )


def test_validator_recomputes_parsed_v2_field_coverage_and_broad_population(tmp_path):
    class PartialBackend:
        language = "en"
        backend_id = "validation-fixture"
        model_id = "partial-v1"

        def parse(self, text: str) -> list[ParsedToken]:
            return [ParsedToken("Apples", "", "NOUN", None, None, "ROOT", 0, 6)]

    project = Project.init(tmp_path / "parsed-v2-coverage-validation")
    _, revision = project.import_text("source.txt", "Apples")
    artifact = build_parsed_token_artifact(project, revision, PartialBackend())
    scope = project.snapshot([revision])
    statistic = artifact_frequency_statistic_evidence(
        project, scope, [artifact.artifact_id], field="lemma", value="apple"
    )
    evidence = project.evidence_content(str(statistic["evidence_id"]))
    coverage = project.coverage_descriptor(str(statistic["coverage_id"]))
    errors: list[str] = []

    _validate_artifact_statistic_provenance(
        project,
        str(statistic["evidence_id"]),
        scope,
        evidence,
        evidence["provenance"],
        errors,
    )

    assert statistic["count"] == 0
    field_coverage = cast(dict[str, object], statistic["field_coverage"])
    assert field_coverage["status"] == "LIMITED"
    assert coverage.coverage_mode is CoverageMode.BROAD
    assert coverage.negative_result_status is NegativeResultStatus.NOT_APPLICABLE
    assert errors == []

    altered_provenance = json.loads(json.dumps(evidence["provenance"]))
    altered_provenance["field_coverage"]["status"] = "SUPPORTED"
    altered_errors: list[str] = []
    _validate_artifact_statistic_provenance(
        project,
        str(statistic["evidence_id"]),
        scope,
        evidence,
        altered_provenance,
        altered_errors,
    )
    assert any("annotation coverage does not match" in item for item in altered_errors)


@pytest.mark.parametrize("claim_type", [ClaimType.ABSENCE, ClaimType.QUANTIFIED])
def test_legacy_parsed_v1_zero_lemma_count_cannot_support_exhaustive_claim(tmp_path, claim_type):
    class MissingLemmaBackend:
        language = "en"
        backend_id = "legacy-validation-fixture"
        model_id = "missing-lemma-v1"

        def parse(self, text: str) -> list[ParsedToken]:
            return [ParsedToken("Apples", "", "NOUN", None, None, "ROOT", 0, 6)]

    project = Project.init(tmp_path / f"legacy-v1-{claim_type.value.lower()}")
    _, revision = project.import_text("source.txt", "Apples")
    artifact = build_parsed_token_artifact(project, revision, MissingLemmaBackend())
    project.db.execute(
        "UPDATE artifact_registry SET schema_version=? WHERE id=?",
        (LEGACY_PARSED_TOKEN_SCHEMA_VERSION, artifact.artifact_id),
    )
    project.db.commit()
    scope = project.snapshot([revision])
    population_definition = (
        "all non-null, non-empty lemma values in one parsed_tokens artifact per revision "
        "of the immutable ScopeSnapshot"
    )
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope,
            population_definition=population_definition,
            population_count=0,
            selection_policy="exact deterministic lemma enumeration for 'migrants' in every selected scope artifact",
            inspection_count=0,
            completion_criterion="all selected, complete scope artifacts read through DuckDB and enumerated",
            termination_reason="completed deterministic artifact enumeration",
            negative_result_status=NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION,
        )
    )
    evidence_id = project.corpus_statistic_evidence(
        "legacy parsed-token lemma frequency for 'migrants'",
        0,
        scope_snapshot_id=scope,
        population_definition=population_definition,
        provenance={
            "calculation_version": "derived-token-artifact-frequency-v1",
            "artifact_ids": [artifact.artifact_id],
            "artifact_type": "parsed_tokens",
            "artifact_schema_version": LEGACY_PARSED_TOKEN_SCHEMA_VERSION,
            "field": "lemma",
            "query_value": "migrants",
            "query_engine": "duckdb",
            "coverage_id": coverage_id,
            "population_count": 0,
        },
    )
    run = project.create_run(scope, {})
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "legacy parsed-token count",
        "The legacy artifact records no non-empty lemma value for 'migrants'.",
        claim_types=[claim_type],
        coverage_id=coverage_id,
    )
    project.attach_evidence(finding, evidence_id, EvidenceRole.SUPPORT)

    validation = validate_finding(project, finding)

    assert not validation.valid
    assert any("derived statistic value or coverage" in error for error in validation.errors)
    if claim_type is ClaimType.ABSENCE:
        assert any("zero-valued corpus statistic" in error for error in validation.errors)


def test_legacy_parsed_v1_does_not_prevent_exact_text_example_claim(tmp_path):
    class MissingLemmaBackend:
        language = "en"
        backend_id = "legacy-validation-fixture"
        model_id = "missing-lemma-v1"

        def parse(self, text: str) -> list[ParsedToken]:
            return [ParsedToken("Apples", "", "NOUN", None, None, "ROOT", 0, 6)]

    project = Project.init(tmp_path / "legacy-v1-exact-example")
    _, revision = project.import_text("source.txt", "Apples")
    artifact = build_parsed_token_artifact(project, revision, MissingLemmaBackend())
    project.db.execute(
        "UPDATE artifact_registry SET schema_version=? WHERE id=?",
        (LEGACY_PARSED_TOKEN_SCHEMA_VERSION, artifact.artifact_id),
    )
    project.db.commit()
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "text example",
        "The source text contains the exact word 'Apples'.",
        claim_types=[ClaimType.EXAMPLE_BASED],
    )
    evidence = project.textual_evidence(project.create_span(revision, 0, 6))
    project.attach_evidence(finding, evidence, EvidenceRole.EXAMPLE)

    assert validate_finding(project, finding).valid


def test_validator_rejects_derived_statistic_when_its_artifact_provenance_is_incomplete(tmp_path):
    project = Project.init(tmp_path / "p")
    _, first = project.import_text("first.txt", "Café")
    _, second = project.import_text("second.txt", "Café")
    first_artifact = build_surface_token_artifact(project, first)
    second_artifact = build_surface_token_artifact(project, second)
    scope = project.snapshot([first, second])
    statistic = artifact_frequency_statistic_evidence(
        project,
        scope,
        [first_artifact.artifact_id, second_artifact.artifact_id],
        field="normalized",
        value="café",
    )
    run = project.create_run(scope, {})
    finding = project.create_finding(
        run,
        "test",
        "1.0.0",
        "count",
        "The selected scope contains two Café tokens.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=str(statistic["coverage_id"]),
    )
    project.attach_evidence(finding, str(statistic["evidence_id"]), EvidenceRole.SUPPORT)
    assert validate_finding(project, finding).valid

    project.db.execute("DELETE FROM artifact_registry WHERE id=?", (second_artifact.artifact_id,))
    project.db.commit()

    result = validate_finding(project, finding)
    assert not result.valid
    assert "complete in-scope artifact provenance" in result.errors[0]
