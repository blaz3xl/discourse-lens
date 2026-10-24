from discourselens import Project
from discourselens.reporting import (
    analysis_payload,
    analysis_payload_page,
    finding_payload,
    render_analysis_html,
    render_analysis_markdown,
    render_finding_html,
    render_finding_markdown,
)
from discourselens.types import (
    ClaimType,
    CoverageDescriptor,
    CoverageMode,
    CoverageTerminationState,
    EvidenceRole,
    NegativeResultStatus,
)


def _attach_textual_support(project, finding_id, revision_id):
    span_id = project.create_span(revision_id, 0, 5)
    evidence_id = project.textual_evidence(span_id)
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)


def test_targeted_zero_hit_disclosure_survives_finding_and_analysis_reports(tmp_path):
    project = Project.init(tmp_path / "targeted-zero-hit")
    _, revision_id = project.import_text("fixture.txt", "apple pear")
    scope_id = project.snapshot([revision_id])
    run_id = project.create_run(scope_id, {})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=scope_id,
            selection_policy="inspect submitted query in returned passages",
            inspection_count=3,
            termination_reason="query completed with no hits",
            termination_state=CoverageTerminationState.COMPLETED,
            negative_result_status=NegativeResultStatus.NO_MATCHES_UNDER_PLAN,
        )
    )
    finding_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "Targeted retrieval result",
        "The submitted targeted query returned no hits under its plan.",
        claim_types=[ClaimType.OBSERVATIONAL],
        coverage_id=coverage_id,
    )
    _attach_textual_support(project, finding_id, revision_id)

    single_payload = finding_payload(project, finding_id)
    single_markdown = render_finding_markdown(project, finding_id)
    single_html = render_finding_html(project, finding_id)
    report_payload = analysis_payload(project, run_id)
    report_markdown = render_analysis_markdown(project, run_id)
    report_html = render_analysis_html(project, run_id)

    disclosure = single_payload["coverage"]
    assert disclosure["status"] == "RECORDED"
    assert disclosure["descriptor"]["coverage_mode"] == "TARGETED"
    scope_snapshot_id = disclosure["descriptor"]["scope_snapshot_id"]
    assert disclosure["descriptor"]["selection_policy"] == (
        "inspect submitted query in returned passages"
    )
    assert disclosure["descriptor"]["inspection_count"] == 3
    assert disclosure["descriptor"]["negative_result_status"] == "NO_MATCHES_UNDER_PLAN"
    assert "TARGETED" in single_markdown
    assert r"NO\_MATCHES\_UNDER\_PLAN" in single_markdown
    assert "TARGETED" in single_html and "NO_MATCHES_UNDER_PLAN" in single_html
    assert "TARGETED" in report_markdown
    assert r"NO\_MATCHES\_UNDER\_PLAN" in report_markdown
    assert "TARGETED" in report_html and "NO_MATCHES_UNDER_PLAN" in report_html
    for output in (single_markdown, single_html, report_markdown, report_html):
        assert "Scope snapshot id" in output
    escaped_scope_snapshot_id = scope_snapshot_id.replace("-", r"\-")
    for output in (single_markdown, report_markdown):
        assert escaped_scope_snapshot_id in output
    for output in (single_html, report_html):
        assert scope_snapshot_id in output
    for output in (single_markdown, single_html, report_markdown, report_html):
        assert "ABSENCE_IN_EXHAUSTIVE_POPULATION" not in output
    assert report_payload["findings"][0]["coverage"] == disclosure
    assert report_payload["coverage_descriptors"] == [disclosure]


def test_exhaustive_quantified_report_discloses_population_and_completion(tmp_path):
    project = Project.init(tmp_path / "exhaustive-quantified")
    _, revision_id = project.import_text("fixture.txt", "apple apple pear")
    scope_id = project.snapshot([revision_id])
    run_id = project.create_run(scope_id, {})
    statistic = project.corpus_frequency_evidence(scope_id, "apple")
    finding_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "Surface frequency",
        "The exact surface-token frequency for apple is 2.",
        claim_types=[ClaimType.QUANTIFIED],
        coverage_id=str(statistic["coverage_id"]),
    )
    project.attach_evidence(
        finding_id, str(statistic["evidence_id"]), EvidenceRole.SUPPORT
    )

    payload = finding_payload(project, finding_id)
    descriptor = payload["coverage"]["descriptor"]
    markdown = render_finding_markdown(project, finding_id)
    html = render_finding_html(project, finding_id)

    assert descriptor["coverage_mode"] == "EXHAUSTIVE"
    assert descriptor["population_definition"] == (
        "all surface-token positions in the immutable ScopeSnapshot"
    )
    assert descriptor["population_count"] == 3
    assert descriptor["inspection_count"] == 3
    assert descriptor["selection_policy"]
    assert descriptor["completion_criterion"]
    assert descriptor["termination_reason"] == "completed deterministic enumeration"
    assert descriptor["termination_state"] == "COMPLETED"
    assert descriptor["negative_result_status"] == "NOT_APPLICABLE"
    for output in (markdown, html):
        assert "EXHAUSTIVE" in output
        assert "Population count" in output
        assert "Inspection count" in output
        assert "COMPLETED" in output


def test_absent_coverage_is_explicit_in_finding_and_analysis_json(tmp_path):
    project = Project.init(tmp_path / "no-coverage")
    _, revision_id = project.import_text("fixture.txt", "apple pear")
    scope_id = project.snapshot([revision_id])
    run_id = project.create_run(scope_id, {})
    finding_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "Textual observation",
        "The selected text contains an example.",
        claim_types=[ClaimType.OBSERVATIONAL],
    )
    _attach_textual_support(project, finding_id, revision_id)

    payload = finding_payload(project, finding_id)
    analysis = analysis_payload(project, run_id)
    page = analysis_payload_page(project, run_id, limit=1)

    assert payload["coverage"] == {
        "coverage_id": None,
        "status": "NOT_RECORDED",
        "descriptor": None,
    }
    assert "**Coverage descriptor:** not recorded" in render_finding_markdown(
        project, finding_id
    )
    assert "Coverage descriptor:</strong> not recorded" in render_finding_html(
        project, finding_id
    )
    assert analysis["coverage_descriptors"] == []
    assert analysis["findings"][0]["coverage"]["status"] == "NOT_RECORDED"
    assert page["findings"][0]["coverage"]["status"] == "NOT_RECORDED"
    assert page["coverage_descriptors"] == []
