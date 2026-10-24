import json
import sqlite3
from html import escape
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pymupdf as fitz
import pytest
from jsonschema import Draft202012Validator

from discourselens import Project
from discourselens.bundle import (
    _contains_secret_metadata_key,
    export_bundle,
    verify_bundle,
)
from discourselens.corpus import compare_named_corpora
from discourselens.errors import DocumentError, EvidenceValidationError, ProjectError
from discourselens.models import ModelPrice
from discourselens.reporting import (
    _contains_sensitive_report_metadata,
    _validate_report_metadata,
    analysis_payload,
    analysis_payload_page,
    comparison_payload,
    finding_payload,
    render_analysis_html,
    render_analysis_markdown,
    render_analysis_markdown_payload,
    render_comparison_markdown,
    render_finding_html,
    render_finding_markdown,
    write_analysis_report,
)
from discourselens.types import ClaimType, CoverageDescriptor, CoverageMode, EvidenceRole
from discourselens.validation import validate_finding


def test_report_uses_canonical_citation(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("speech.txt", "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {})
    span = project.create_span(revision, 0, 6)
    evidence = project.textual_evidence(span)
    finding = project.create_finding(run, "fixture", "1.0.0", "lexical choice", "Apples occurs")
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    report = render_finding_markdown(project, finding)
    assert "Apples" in report and r"\[speech\.txt · ¶1\]" in report
    payload = finding_payload(project, finding)
    assert payload["evidence"][0]["citation"] == "[speech.txt · ¶1]"
    evidence_id = payload["evidence"][0]["evidence_id"]
    assert (
        f"<li><strong>Support (evidence ID: {evidence_id}):</strong> "
        "Apples — [speech.txt · ¶1]</li>"
        in render_finding_html(project, finding)
    )
    report = render_analysis_markdown(project, run)
    payload = analysis_payload(project, run)
    assert "Technical appendix" in report and r"\[speech\.txt · ¶1\]" in report
    escaped_version = str(payload["analysis_run"]["software_version"]).replace(".", r"\.")
    assert f"DiscourseLens version: {escaped_version}" in report
    assert f"Configuration SHA-256: {payload['analysis_run']['config_hash']}" in report
    assert payload["findings"][0]["finding"]["id"] == finding
    analysis_html = render_analysis_html(project, run)
    assert "Apples occurs" in analysis_html and "[speech.txt · ¶1]" in analysis_html
    assert payload["analysis_run"]["software_version"] in analysis_html


def test_analysis_reports_expose_synthesized_finding_lineage_ids_and_relation(tmp_path):
    project = Project.init(tmp_path / "lineage-report")
    try:
        _, revision = project.import_text("speech.txt", "Apples are green.")
        run = project.create_run(project.snapshot([revision]), {})
        evidence = project.textual_evidence(project.create_span(revision, 0, 6))
        source = project.create_finding(
            run, "fixture.source", "1.0.0", "source observation", "Source finding text."
        )
        synthesized = project.create_finding(
            run, "builtin.synthesis", "1.0.0", "synthesized finding", "Synthesized text."
        )
        project.attach_evidence(source, evidence, EvidenceRole.SUPPORT)
        project.attach_evidence(synthesized, evidence, EvidenceRole.SUPPORT)
        project.link_finding_source(synthesized, source, "SYNTHESIZES")

        expected_lineage = [
            {
                "finding_id": synthesized,
                "source_finding_id": source,
                "relation": "SYNTHESIZES",
            }
        ]
        full_payload = analysis_payload(project, run)
        page_payload = analysis_payload_page(project, run, limit=100)
        assert full_payload["finding_lineage"] == expected_lineage
        assert page_payload["finding_lineage"] == expected_lineage
        assert set(full_payload["finding_lineage"][0]) == {
            "finding_id",
            "source_finding_id",
            "relation",
        }

        markdown = render_analysis_markdown_payload(full_payload)
        html = render_analysis_html(project, run)
        serialized_lineage = json.dumps(expected_lineage, ensure_ascii=False)
        assert "Finding lineage:" in markdown
        assert source.replace("-", r"\-") in markdown
        assert synthesized.replace("-", r"\-") in markdown
        assert "SYNTHESIZES" in markdown
        assert escape(serialized_lineage) in html
    finally:
        project.close()


def test_reports_preserve_support_and_counterevidence_roles(tmp_path):
    project = Project.init(tmp_path / "p")
    text = "Apples are green. Elephants are not green."
    _, revision = project.import_text("speech.txt", text)
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(
        run, "fixture", "1.0.0", "color description", "The text describes color."
    )
    support = project.textual_evidence(project.create_span(revision, 0, 17))
    counter = project.textual_evidence(project.create_span(revision, 18, len(text)))
    project.attach_evidence(finding, support, EvidenceRole.SUPPORT)
    project.attach_evidence(finding, counter, EvidenceRole.COUNTEREVIDENCE)

    finding_markdown = render_finding_markdown(project, finding)
    analysis_markdown = render_analysis_markdown(project, run)
    finding_html = render_finding_html(project, finding)
    analysis_html = render_analysis_html(project, run)
    evidence = finding_payload(project, finding)["evidence"]
    support_item = next(item for item in evidence if item["role"] == "SUPPORT")
    counter_item = next(item for item in evidence if item["role"] == "COUNTEREVIDENCE")
    support_id = support_item["evidence_id"]
    counter_id = counter_item["evidence_id"]
    escaped_support_id = support_id.replace("-", r"\-")
    escaped_counter_id = counter_id.replace("-", r"\-")

    assert (
        f"**Support (evidence ID: {escaped_support_id}):** "
        r"Apples are green\. — \[speech\.txt · ¶1\]"
    ) in finding_markdown
    assert (
        f"**Counterevidence (evidence ID: {escaped_counter_id}):** "
        r"Elephants are not green\. — \[speech\.txt · ¶1\]"
    ) in finding_markdown
    assert escaped_counter_id in analysis_markdown
    assert r"\[speech\.txt · ¶1\]" in analysis_markdown
    assert f"<strong>Support (evidence ID: {support_id}):</strong> Apples are green." in finding_html
    assert (
        f"<strong>Counterevidence (evidence ID: {counter_id}):</strong> "
        "Elephants are not green. — [speech.txt · ¶1]"
    ) in finding_html
    assert counter_id in analysis_html and "[speech.txt · ¶1]" in analysis_html


def test_analysis_reports_include_reranking_provenance(tmp_path):
    project = Project.init(tmp_path / "reranking-report")
    _, revision = project.import_text("speech.txt", "alpha beta")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=scope,
            selection_policy="reranking report fixture",
        )
    )
    skill_run = project.create_skill_run(
        run,
        scope,
        "fixture.retrieval",
        "1.0.0",
        "test-fixture",
        "fixture-config-hash",
        coverage_id=coverage,
    )
    retrieval_run = project.create_retrieval_run(
        analysis_run_id=run,
        skill_run_id=skill_run,
        scope_id=scope,
        retrieval_plan_ref="fixture-plan",
        query_terms=["alpha"],
        selection_policy="fixture selection",
        candidate_count=1,
        inspected_count=1,
        returned_count=1,
        candidate_limit_reached=False,
        coverage_id=coverage,
    )
    span = project.create_span(revision, 0, 5)
    hit_id = project.add_retrieval_hit(retrieval_run, span, "fixture", 1, 1.0)
    reranking_id = project.record_reranking(
        retrieval_run,
        provider_id="fixture-reranker",
        credential_alias="fixture-alias",
        model="fixture-model",
        query="alpha",
        top_n=1,
        effective_top_n=1,
        candidate_count=1,
        ranked_hits=[(hit_id, 1, 0.9)],
    )

    payload = analysis_payload(project, run)
    markdown = render_analysis_markdown(project, run)
    html = render_analysis_html(project, run)

    assert payload["reranking_runs"][0]["id"] == reranking_id
    assert "Reranking runs and hits:" in markdown
    assert r"fixture\-reranker" in markdown and r"0\.9" in markdown
    assert "reranking_runs" in html and "fixture-reranker" in html and "0.9" in html


def test_counterevidence_query_labels_are_not_secret_metadata_across_exports(tmp_path):
    project = Project.init(tmp_path / "counterevidence-labels")
    try:
        _, revision = project.import_text("speech.txt", "Apples are green.")
        config = {
            "counterevidence_queries": {
                "api_key": "find counterevidence",
                "sessiontoken": "look for an alternate explanation",
            }
        }
        run = project.create_run(project.snapshot([revision]), config)

        summary = project.analysis_run_summary(run)
        summary_page = project.analysis_run_summary_page(run, limit=1)
        for result in (summary, summary_page):
            run_metadata = result["analysis_run"]
            assert isinstance(run_metadata, dict)
            assert run_metadata["config"] == config

        report = analysis_payload(project, run)
        report_page = analysis_payload_page(project, run, limit=1)
        assert report["analysis_run"]["config"] == config
        assert report_page["analysis_run"]["config"] == config

        manifest = export_bundle(project, run, tmp_path / "bundle")
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert manifest_payload["analysis_run"]["config"] == config
        assert verify_bundle(manifest)["status"] == "VALID"

        nested_secret = {
            "counterevidence_queries": {
                "api_key": {"provider_options": {"api_key": "fixture-secret"}}
            }
        }
        secret_found, _ = _contains_sensitive_report_metadata(
            nested_secret, allow_counterevidence_query_labels=True
        )
        assert secret_found
        assert _contains_secret_metadata_key(
            nested_secret, allow_counterevidence_query_labels=True
        )
        with pytest.raises(EvidenceValidationError, match="secret-like metadata"):
            _validate_report_metadata(
                nested_secret,
                "analysis run report configuration",
                allow_counterevidence_query_labels=True,
            )
    finally:
        project.close()


def test_analysis_profile_allows_counterevidence_query_labels(tmp_path):
    project = Project.init(tmp_path / "counterevidence-profile-labels")
    try:
        queries = {"api_key": "find counterevidence"}
        project.save_analysis_profile(
            "fixture",
            {
                "version": "1.0.0",
                "kind": "controlled-cda",
                "frameworks": ["fairclough-text"],
                "counterevidence_queries": queries,
            },
        )
        profile = project.analysis_profile("fixture")
        assert profile["counterevidence_queries"] == queries
    finally:
        project.close()


def test_analysis_report_preserves_model_cost_snapshot_and_nullable_cost(tmp_path):
    project = Project.init(tmp_path / "model-cost-report")
    try:
        _, revision = project.import_text("speech.txt", "Apples are green.")
        run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
        price = ModelPrice(
            provider_id="fixture-provider",
            model="fixture-model",
            input_per_million_tokens="1.25",
            output_per_million_tokens="2.5",
            currency="USD",
            source_reference="https://prices.example/models/fixture-model",
            rate_version="v1",
            rate_date="2026-01-01",
        )
        priced_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="a" * 64,
            response_hash="b" * 64,
            analysis_run_id=run,
            input_tokens=1_000,
            output_tokens=500,
            model_price=price,
        )
        unpriced_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="c" * 64,
            response_hash="d" * 64,
            analysis_run_id=run,
            input_tokens=1_000,
            output_tokens=500,
        )

        executions = {
            item["id"]: item for item in analysis_payload(project, run)["model_executions"]
        }
        assert executions[priced_id]["model_price_json"] == json.dumps(
            price.to_snapshot(), sort_keys=True, separators=(",", ":")
        )
        assert executions[priced_id]["estimated_cost"] == "0.0025"
        assert executions[priced_id]["estimated_cost_currency"] == "USD"
        assert executions[unpriced_id]["model_price_json"] is None
        assert executions[unpriced_id]["estimated_cost"] is None
        assert executions[unpriced_id]["estimated_cost_currency"] is None

        schema_path = Path(__file__).parents[1] / "schemas" / "analysis-run-v1.json"
        model_execution_schema = json.loads(schema_path.read_text(encoding="utf-8"))["$defs"][
            "modelExecution"
        ]
        validator = Draft202012Validator(model_execution_schema)
        validator.validate(executions[priced_id])
        validator.validate(executions[unpriced_id])
        historical_execution = {
            key: value
            for key, value in executions[priced_id].items()
            if key not in {"model_price_json", "estimated_cost", "estimated_cost_currency"}
        }
        validator.validate(historical_execution)
    finally:
        project.close()


@pytest.mark.parametrize("output_format", ["markdown", "html", "json"])
def test_report_maps_unsafe_historical_model_execution_to_safe_metadata_error(
    tmp_path, output_format
):
    project = Project.init(tmp_path / f"unsafe-model-execution-{output_format}")
    secret = "model-execution-secret-MUST-NOT-EXPORT"
    try:
        _, revision = project.import_text("speech.txt", "Apples are green.")
        run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
        execution_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash=None,
            analysis_run_id=run,
            parameters={"provider": {"mode": "fixture"}},
        )
        # Exercise the read-side boundary for a historical row that predates
        # the current write-time secret checks.
        project.db.execute(
            "UPDATE model_executions SET parameters_json=? WHERE id=?",
            (json.dumps({"provider": {"api_key": secret}}), execution_id),
        )
        project.db.commit()

        with pytest.raises(EvidenceValidationError) as error:
            if output_format == "markdown":
                render_analysis_markdown(project, run)
            elif output_format == "html":
                render_analysis_html(project, run)
            else:
                write_analysis_report(
                    project, run, tmp_path / "unsafe-model-execution.json", "json"
                )
        assert str(error.value) == "model execution report metadata is invalid"
        assert secret not in str(error.value)
    finally:
        project.close()


def test_counterevidence_labels_remain_secret_keys_in_model_execution_metadata(tmp_path):
    project = Project.init(tmp_path / "query-label-capability-secret")
    secret = "fixture-secret-MUST-NOT-EXPORT"
    capabilities: dict[str, object] = {"counterevidence_queries": {"api_key": secret}}
    try:
        _, revision = project.import_text("speech.txt", "Apples are green.")
        run = project.create_run(project.snapshot([revision]), {})
        with pytest.raises(DocumentError, match="capabilities must not contain secrets"):
            project.record_model_execution(
                provider_id="fixture-provider",
                credential_alias="fixture",
                model="fixture-model",
                reasoning=None,
                request_hash="request-hash",
                response_hash=None,
                analysis_run_id=run,
                capabilities=capabilities,
            )

        execution_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="safe-request-hash",
            response_hash=None,
            analysis_run_id=run,
            capabilities={"provider": {"mode": "fixture"}},
        )
        project.db.execute(
            "UPDATE model_executions SET capabilities_json=? WHERE id=?",
            (json.dumps(capabilities), execution_id),
        )
        project.db.commit()

        with pytest.raises(DocumentError, match="stored model execution metadata") as summary_error:
            project.analysis_run_summary(run)
        assert secret not in str(summary_error.value)
        with pytest.raises(EvidenceValidationError) as report_error:
            analysis_payload(project, run)
        assert str(report_error.value) == "model execution report metadata is invalid"
        assert secret not in str(report_error.value)
        with pytest.raises(ProjectError, match="secret-bearing model execution metadata") as bundle_error:
            export_bundle(project, run, tmp_path / "unsafe-capability-bundle")
        assert secret not in str(bundle_error.value)
    finally:
        project.close()


@pytest.mark.parametrize(
    "snapshot_change",
    [
        {"unsupported_field": "must be rejected"},
        {"source_reference": "https://prices.example/p?X-Amz-Signature=do-not-report-this"},
    ],
)
def test_analysis_report_rejects_unsafe_or_unknown_price_snapshot_fields(tmp_path, snapshot_change):
    project = Project.init(tmp_path / "unsafe-model-cost-report")
    try:
        _, revision = project.import_text("speech.txt", "Apples are green.")
        run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
        price = ModelPrice(
            provider_id="fixture-provider",
            model="fixture-model",
            input_per_million_tokens="1",
            output_per_million_tokens="2",
            currency="USD",
            source_reference="https://prices.example/pricing-v1",
            rate_version="v1",
            rate_date="2026-01-01",
        )
        execution_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash="response-hash",
            analysis_run_id=run,
            input_tokens=1,
            output_tokens=1,
            model_price=price,
        )
        changed_snapshot = {**price.to_snapshot(), **snapshot_change}
        project.db.execute(
            "UPDATE model_executions SET model_price_json=? WHERE id=?",
            (
                json.dumps(changed_snapshot, sort_keys=True, separators=(",", ":")),
                execution_id,
            ),
        )
        project.db.commit()

        with pytest.raises(EvidenceValidationError, match="model execution report metadata"):
            analysis_payload(project, run)
    finally:
        project.close()


@pytest.mark.parametrize(
    ("snapshot_provider", "snapshot_model"),
    [("fixture-provider", "other-model"), ("other-provider", "fixture-model")],
)
def test_analysis_report_rejects_valid_price_snapshot_for_a_different_execution(
    tmp_path, snapshot_provider, snapshot_model
):
    project = Project.init(
        tmp_path / f"mismatched-cost-report-{snapshot_provider}-{snapshot_model}"
    )
    try:
        _, revision = project.import_text("speech.txt", "Apples are green.")
        run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
        execution_price = ModelPrice(
            provider_id="fixture-provider",
            model="fixture-model",
            input_per_million_tokens="1.25",
            output_per_million_tokens="2.5",
            currency="USD",
            source_reference="https://prices.example/fixture-model",
            rate_version="v1",
            rate_date="2026-01-01",
        )
        other_model_price = ModelPrice(
            provider_id=snapshot_provider,
            model=snapshot_model,
            input_per_million_tokens="1.25",
            output_per_million_tokens="2.5",
            currency="USD",
            source_reference="https://prices.example/other-model",
            rate_version="v1",
            rate_date="2026-01-01",
        )
        execution_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash="response-hash",
            analysis_run_id=run,
            input_tokens=1_000,
            output_tokens=500,
            model_price=execution_price,
        )
        stored_cost = project.db.execute(
            "SELECT estimated_cost FROM model_executions WHERE id=?", (execution_id,)
        ).fetchone()[0]
        assert stored_cost == "0.0025"
        project.db.execute(
            "UPDATE model_executions SET model_price_json=? WHERE id=?",
            (
                json.dumps(
                    other_model_price.to_snapshot(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                execution_id,
            ),
        )
        project.db.commit()
        assert project.db.execute(
            "SELECT estimated_cost FROM model_executions WHERE id=?", (execution_id,)
        ).fetchone()[0] == stored_cost

        with pytest.raises(
            EvidenceValidationError,
            match="does not match executed provider and model",
        ):
            analysis_payload(project, run)
    finally:
        project.close()


def test_markdown_report_keeps_corpus_text_from_becoming_active_markdown(tmp_path):
    project = Project.init(tmp_path / "p")
    hostile = "Quoted text.\n\n![tracking pixel](https://attacker.test/pixel)\n# Fake section"
    _, revision = project.import_text("speech.txt", hostile)
    run = project.create_run(project.snapshot([revision]), {"note": "safe"})
    evidence = project.textual_evidence(project.create_span(revision, 0, len(hostile)))
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "textual observation",
        "Passage includes an image-like string.\n# Injected heading",
    )
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)

    finding_markdown = render_finding_markdown(project, finding)
    analysis_markdown = render_analysis_markdown(project, run)

    for rendered in (finding_markdown, analysis_markdown):
        assert "![tracking pixel](https://attacker.test/pixel)" not in rendered
        assert "\n# Fake section" not in rendered
        assert "\n# Injected heading" not in rendered
        assert r"\!\[tracking pixel\]\(https\:\/\/attacker\.test\/pixel\)" in rendered
    assert "## Findings" in analysis_markdown

    pdf = write_analysis_report(project, run, tmp_path / "report.pdf", "pdf")
    opened_pdf = fitz.open(pdf)
    try:
        pdf_text = "".join(str(page.get_text()) for page in opened_pdf)
    finally:
        opened_pdf.close()
    normalized_pdf_text = " ".join(pdf_text.split())
    assert "![tracking pixel](https://attacker.test/pixel)" in normalized_pdf_text


def test_external_and_user_context_evidence_remain_distinct_from_corpus_text(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("speech.txt", "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(run, "fixture", "1.0.0", "context", "Context is supplied.")
    external = project.external_source_evidence(
        "Official archive",
        "The policy took effect in 2025.",
        title="Policy archive",
        url="https://example.test/policy",
        retrieved_at="2026-09-25",
    )
    context = project.user_context_evidence(
        "research memo", "The researcher supplied this context."
    )
    project.attach_evidence(finding, external, EvidenceRole.CONTEXT)
    project.attach_evidence(finding, context, EvidenceRole.CONTEXT)

    payload = finding_payload(project, finding)
    assert {item["kind"] for item in payload["evidence"]} == {"EXTERNAL_SOURCE", "USER_CONTEXT"}
    assert project.render_citation(external) == "[Policy archive · retrieved 2026-09-25]"
    assert project.render_citation(context) == "[user context · research memo]"
    report = render_finding_markdown(project, finding)
    assert "policy took effect" in report and "researcher supplied" in report
    assert validate_finding(project, finding).valid


def test_human_review_is_append_only_and_reported_without_mutating_evidence(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("speech.txt", "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {})
    evidence = project.textual_evidence(project.create_span(revision, 0, 6))
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "lexicon",
        "Apples occurs",
        claim_types=[ClaimType.EXAMPLE_BASED],
        evidence_sufficiency="SUFFICIENT",
        interpretation_confidence="MEDIUM",
        limitations=["Only one passage was reviewed."],
    )
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    first = project.review_finding(finding, "ACCEPTED", "Evidence is accurately cited.")
    second = project.review_finding(finding, "MODIFIED", "<script>qualify this</script>")
    assert first != second
    assert [item["status"] for item in project.finding_reviews(finding)] == ["ACCEPTED", "MODIFIED"]
    assert project.get_finding(finding)["review_status"] == "MODIFIED"
    assert (
        finding_payload(project, finding)["human_reviews"][1]["note"]
        == "<script>qualify this</script>"
    )
    finding_markdown = render_finding_markdown(project, finding)
    analysis_markdown = render_analysis_markdown(project, run)
    finding_html = render_finding_html(project, finding)
    analysis_html = render_analysis_html(project, run)
    assert "**Human review status:** MODIFIED" in finding_markdown
    for rendered in (finding_markdown, analysis_markdown):
        assert r"EXAMPLE\_BASED" in rendered
        assert "SUFFICIENT" in rendered and "MEDIUM" in rendered
        assert "Only one passage was reviewed" in rendered
        assert r"\<script\>qualify this\<\/script\>" in rendered
    for rendered in (finding_html, analysis_html):
        assert "Claim types:" in rendered and "EXAMPLE_BASED" in rendered
        assert "Evidence sufficiency:</strong> SUFFICIENT" in rendered
        assert "Interpretation confidence:</strong> MEDIUM" in rendered
        assert "Only one passage was reviewed." in rendered
        assert "&lt;script&gt;qualify this&lt;/script&gt;" in rendered
        assert "<script>qualify this</script>" not in rendered


def test_limited_finding_payload_discloses_omitted_evidence_and_reviews(tmp_path):
    project = Project.init(tmp_path / "limited-finding-payload")
    try:
        _, revision = project.import_text("speech.txt", "Apples are green. Pears are green.")
        run = project.create_run(project.snapshot([revision]), {})
        finding = project.create_finding(
            run, "fixture", "1.0.0", "color description", "The text describes color."
        )
        first = project.textual_evidence(project.create_span(revision, 0, 6))
        second = project.textual_evidence(project.create_span(revision, 18, 23))
        project.attach_evidence(finding, first, EvidenceRole.SUPPORT)
        project.attach_evidence(finding, second, EvidenceRole.COUNTEREVIDENCE)
        project.review_finding(finding, "ACCEPTED", "First review.")
        project.review_finding(finding, "MODIFIED", "Second review.")

        partial = finding_payload(project, finding, evidence_limit=1, review_limit=1)
        full = finding_payload(project, finding)

        assert len(partial["evidence"]) == len(partial["human_reviews"]) == 1
        assert partial["evidence_total"] == partial["human_review_total"] == 2
        assert partial["evidence_truncated"] is True
        assert partial["human_reviews_truncated"] is True
        assert full["evidence_total"] == full["human_review_total"] == 2
        assert full["evidence_truncated"] is False
        assert full["human_reviews_truncated"] is False
    finally:
        project.close()


def test_comparison_report_keeps_both_populations_and_verifies_historical_result(tmp_path):
    project = Project.init(tmp_path / "p")
    _, left_revision = project.import_text("left.txt", "apples apples green")
    _, right_revision = project.import_text("right.txt", "elephants green")
    left = project.create_corpus("left")
    right = project.create_corpus("right")
    project.add_revision_to_corpus(left, left_revision)
    project.add_revision_to_corpus(right, right_revision)
    result = compare_named_corpora(project, left, right, metric="keyness")

    payload = comparison_payload(project, str(result["comparison_id"]))
    report = render_comparison_markdown(project, str(result["comparison_id"]))

    assert payload["comparison_run"]["result_hash_verified"] is True
    assert payload["left_scope"]["surface_token_population"] == 3
    assert payload["right_scope"]["surface_token_population"] == 2
    assert "Left scope" in report and "Right scope" in report
    assert "does not by itself establish causation" in report

    corrupted = project.db.execute(
        "SELECT row_json FROM comparison_result_rows WHERE comparison_id=? AND token='apples'",
        (result["comparison_id"],),
    ).fetchone()[0]
    corrupt_record = json.loads(corrupted)
    corrupt_record["left_count"] += 1
    project.db.execute(
        "UPDATE comparison_result_rows SET row_json=? WHERE comparison_id=? AND token='apples'",
        (
            json.dumps(corrupt_record, sort_keys=True, separators=(",", ":")),
            result["comparison_id"],
        ),
    )
    project.db.commit()
    with pytest.raises(EvidenceValidationError, match="hash mismatch"):
        comparison_payload(project, str(result["comparison_id"]))


def test_comparison_markdown_renders_only_requested_verified_page(tmp_path, monkeypatch):
    from discourselens import reporting

    project = Project.init(tmp_path / "paged-comparison")
    _, left_revision = project.import_text("left.txt", "alpha beta")
    _, right_revision = project.import_text("right.txt", "gamma delta")
    left_scope = project.snapshot([left_revision])
    right_scope = project.snapshot([right_revision])
    comparison_id = project.create_comparison_run(
        left_scope,
        right_scope,
        metric="frequency",
        calculation_version="fixture-v1",
        config={"source": "test"},
        rows=[
            {"token": f"term-{index}", "left_count": index, "right_count": 5 - index}
            for index in range(5)
        ],
    )

    def reject_full_payload(*args, **kwargs):
        pytest.fail("bounded Markdown renderer must not load the full comparison payload")

    monkeypatch.setattr(reporting, "comparison_payload", reject_full_payload)
    rendered = render_comparison_markdown(project, comparison_id, offset=2, limit=2)

    assert "Showing rows 3–4 of 5." in rendered
    assert r"term\-2" in rendered and r"term\-3" in rendered
    assert r"term\-0" not in rendered and r"term\-1" not in rendered and r"term\-4" not in rendered
    assert "request offset 4" in rendered


def test_analysis_report_includes_only_scope_bound_qualitative_coding(tmp_path):
    project = Project.init(tmp_path / "p")
    _, selected = project.import_text("selected.txt", "Apples are green.")
    _, outside = project.import_text("outside.txt", "Elephants live elsewhere.")
    scope = project.snapshot([selected])
    run = project.create_run(scope, {"profile": "coding"})
    codebook = project.create_codebook("descriptive")
    code = project.create_code(codebook, "fruit")
    project.annotate_span(code, project.create_span(selected, 0, 6), "Human annotation.")
    project.annotate_span(code, project.create_span(outside, 0, 9), "Must not leak.")
    project.create_research_memo(scope, "Scope note", "Selected source only.")

    payload = analysis_payload(project, run)
    report = render_analysis_markdown(project, run)
    html = render_analysis_html(project, run)
    json_report = write_analysis_report(project, run, tmp_path / "report.json", "json").read_text(
        encoding="utf-8"
    )

    assert len(payload["qualitative_coding"]["annotations"]) == 1
    assert "Apples" in report and "Scope note" in report
    assert "Elephants" not in report
    assert "Elephants" not in html and "Must not leak" not in html
    assert "Elephants" not in json_report and "Must not leak" not in json_report


@pytest.mark.parametrize("output_format", ["markdown", "html", "json"])
@pytest.mark.parametrize("metadata_kind", ["absolute_path", "secret_like_provider_key"])
def test_report_refuses_sensitive_run_and_provider_metadata(
    tmp_path, output_format, metadata_kind
):
    project = Project.init(tmp_path / f"report-{output_format}-{metadata_kind}")
    _, revision = project.import_text("selected.txt", "Selected evidence.")
    config = {"profile": "privacy-fixture"}
    secret_decoy = "vendor-key-decoy-MUST-NOT-EXPORT"
    absolute_path = str(tmp_path / "private-model-cache")
    if metadata_kind == "absolute_path":
        config["cache_path"] = absolute_path
    run_id = project.create_run(project.snapshot([revision]), config)
    if metadata_kind == "secret_like_provider_key":
        execution_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture-alias",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash=None,
            analysis_run_id=run_id,
            parameters={"provider": {"mode": "fixture"}},
        )
        # Model execution writes reject this now; simulate an older contaminated row.
        project.db.execute(
            "UPDATE model_executions SET parameters_json=? WHERE id=?",
            (json.dumps({"provider": {"vendorApiKey": secret_decoy}}), execution_id),
        )
        project.db.commit()

    with pytest.raises(EvidenceValidationError, match="report metadata|report configuration"):
        if output_format == "markdown":
            render_analysis_markdown(project, run_id)
        elif output_format == "html":
            render_analysis_html(project, run_id)
        else:
            write_analysis_report(
                project, run_id, tmp_path / f"report-{metadata_kind}.json", "json"
            )

    project.close()


@pytest.mark.parametrize("output_format", ["markdown", "html", "json"])
@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_report_refuses_non_finite_provenance_numbers(tmp_path, output_format, non_finite):
    project = Project.init(tmp_path / f"non-finite-report-{output_format}")
    _, revision = project.import_text("selected.txt", "Selected evidence.")
    run_id = project.create_run(project.snapshot([revision]), {"threshold": non_finite})
    destination = tmp_path / f"report-{output_format}.json"

    with pytest.raises(EvidenceValidationError, match="non-finite numeric metadata"):
        if output_format == "markdown":
            render_analysis_markdown(project, run_id)
        elif output_format == "html":
            render_analysis_html(project, run_id)
        else:
            write_analysis_report(project, run_id, destination, "json")

    assert not destination.exists()


def test_bounded_analysis_markdown_marks_capped_finding_evidence_and_review_text(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("speech.txt", "A" * 1_100)
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(run, "fixture", "1.0.0", "long text", "O" * 2_100)
    evidence = project.textual_evidence(project.create_span(revision, 0, 1_100))
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    project.review_finding(finding, "ACCEPTED", "R" * 1_100)

    page = analysis_payload_page(project, run, limit=1)
    report = render_analysis_markdown_payload(page)

    assert "O" * 2_000 + " [truncated]" in report
    assert "A" * 1_000 + " [truncated]" in report
    assert "R" * 1_000 + " [truncated]" in report


def test_bounded_analysis_markdown_discloses_omitted_evidence_and_reviews(tmp_path):
    project = Project.init(tmp_path / "p")
    passages = [f"Passage {index:02d}." for index in range(11)]
    text = "\n".join(passages)
    _, revision = project.import_text("speech.txt", text)
    run = project.create_run(project.snapshot([revision]), {})
    finding = project.create_finding(run, "fixture", "1.0.0", "bounded finding", "Review all evidence.")
    for passage in passages:
        start = text.index(passage)
        evidence = project.textual_evidence(project.create_span(revision, start, start + len(passage)))
        project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
        project.review_finding(finding, "ACCEPTED", f"Reviewed {passage}")

    page = analysis_payload_page(project, run, limit=1)
    report = render_analysis_markdown_payload(page)

    assert "Evidence list is partial: showing 10 of 11; additional evidence omitted." in report
    assert "Human review list is partial: showing 10 of 11; additional reviews omitted." in report


def test_bounded_analysis_markdown_discloses_paged_model_provenance(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("speech.txt", "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {})
    for index in range(2):
        project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture-alias",
            model=f"fixture-model-{index}",
            reasoning=None,
            request_hash=f"request-hash-{index}",
            response_hash=f"response-hash-{index}",
            analysis_run_id=run,
        )

    page = analysis_payload_page(project, run, limit=1)
    report = render_analysis_markdown_payload(page)

    assert page["pagination"]["model_executions"]["has_more"] is True
    assert "## Page completeness" in report
    assert "model executions: page offset 0, limit 1, total 2; additional entries omitted." in report


@pytest.mark.parametrize("invalid_link", ["dangling", "cross-run"])
def test_paged_analysis_report_rejects_corrupt_model_skill_run_link(tmp_path, invalid_link):
    with Project.init(tmp_path / f"corrupt-{invalid_link}") as project:
        _, revision = project.import_text("speech.txt", "Apples are green.")
        scope_id = project.snapshot([revision])
        run_id = project.create_run(scope_id, {"profile": "execution"})
        foreign_run_id = project.create_run(scope_id, {"profile": "foreign"})
        skill_run_id = project.create_skill_run(
            run_id,
            scope_id,
            "fixture.skill",
            "1.0.0",
            "fixture.executor",
            "fixture-config-hash",
        )
        foreign_skill_run_id = project.create_skill_run(
            foreign_run_id,
            scope_id,
            "fixture.skill",
            "1.0.0",
            "fixture.executor",
            "fixture-config-hash",
        )
        execution_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash="response-hash",
            analysis_run_id=run_id,
            skill_run_id=skill_run_id,
        )
        invalid_skill_run_id = (
            "missing-skill-run" if invalid_link == "dangling" else foreign_skill_run_id
        )
        project.db.execute(
            "UPDATE model_executions SET skill_run_id=? WHERE id=?",
            (invalid_skill_run_id, execution_id),
        )
        project.db.commit()

        with pytest.raises(
            EvidenceValidationError, match="model execution report provenance is invalid"
        ):
            analysis_payload_page(project, run_id, limit=1)


def test_report_retrieval_hit_pages_use_stable_tie_breaker(tmp_path):
    project = Project.init(tmp_path / "stable-retrieval-pages")
    _, revision = project.import_text("speech.txt", "alpha beta gamma")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {})
    coverage = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=scope,
            selection_policy="report pagination fixture",
        )
    )
    skill_run = project.create_skill_run(
        run,
        scope,
        "fixture.retrieval",
        "1.0.0",
        "test-fixture",
        "fixture-config-hash",
        coverage_id=coverage,
    )
    retrieval_run = project.create_retrieval_run(
        analysis_run_id=run,
        skill_run_id=skill_run,
        scope_id=scope,
        retrieval_plan_ref="fixture-plan",
        query_terms=["alpha"],
        selection_policy="fixture selection",
        candidate_count=2,
        inspected_count=2,
        returned_count=2,
        candidate_limit_reached=False,
        coverage_id=coverage,
    )
    first_span = project.create_span(revision, 0, 5)
    second_span = project.create_span(revision, 6, 10)
    # Retrieval ranks may tie in persisted history; make insertion order differ
    # from the canonical ID order so a missing secondary key is observable.
    project.db.executemany(
        "INSERT INTO retrieval_hits VALUES(?,?,?,?,?,?)",
        [
            ("hit-z", retrieval_run, first_span, "fixture", 1, 1.0),
            ("hit-a", retrieval_run, second_span, "fixture", 1, 1.0),
        ],
    )
    project.db.commit()

    full_ids = [item["id"] for item in analysis_payload(project, run)["retrieval_hits"]]
    first_page = analysis_payload_page(project, run, limit=1, offset=0)
    second_page = analysis_payload_page(project, run, limit=1, offset=1)
    paged_ids = [
        item["id"]
        for page in (first_page, second_page)
        for item in page["retrieval_hits"]
    ]

    assert full_ids == ["hit-a", "hit-z"]
    assert paged_ids == full_ids


def test_analysis_report_scope_over_sqlite_bind_limit_is_stable_and_bounded(tmp_path):
    project = Project.init(tmp_path / "large-report-scope")
    _, revision = project.import_text("speech.txt", "alpha")
    base = project.db.execute(
        "SELECT * FROM document_revisions WHERE id=?", (revision,)
    ).fetchone()
    synthetic_ids = [f"synthetic-revision-{index:04d}" for index in range(999)]
    project.db.executemany(
        "INSERT INTO document_revisions "
        "(id,document_id,source_hash,canonical_hash,extractor,coordinate_scheme,text_path,"
        "original_path,mime_type,source_size,extraction_status,created_at,ocr_language,ocr_dpi) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                synthetic_id,
                base["document_id"],
                f"synthetic-source-{index:04d}",
                base["canonical_hash"],
                base["extractor"],
                base["coordinate_scheme"],
                base["text_path"],
                base["original_path"],
                base["mime_type"],
                base["source_size"],
                base["extraction_status"],
                base["created_at"],
                base["ocr_language"],
                base["ocr_dpi"],
            )
            for index, synthetic_id in enumerate(synthetic_ids)
        ],
    )
    project.db.commit()

    scope = project.snapshot([revision, *synthetic_ids])
    run = project.create_run(scope, {})
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "large scope report",
        "x" * 2_100,
    )
    evidence = project.textual_evidence(project.create_span(revision, 0, 5))
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)

    limit_category = sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER
    previous_limit = project.db.setlimit(limit_category, 999)
    try:
        full = analysis_payload(project, run)
        pages = [
            analysis_payload_page(project, run, limit=100, offset=offset)
            for offset in range(0, len(synthetic_ids) + 1, 100)
        ]
    finally:
        project.db.setlimit(limit_category, previous_limit)

    paged_revisions = [
        item for page in pages for item in page["document_revisions"]
    ]
    assert paged_revisions == full["document_revisions"]
    first_finding = pages[0]["findings"][0]["finding"]
    assert first_finding["observation_truncated"] is True
    assert len(first_finding["observation"]) == 2_000


def test_analysis_report_exports_local_pdf_and_docx_from_validated_records(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("speech.txt", "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    evidence = project.textual_evidence(project.create_span(revision, 0, 6))
    finding = project.create_finding(run, "fixture", "1.0.0", "lexicon", "Apples occurs.")
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)

    pdf = write_analysis_report(project, run, tmp_path / "report.pdf", "pdf")
    docx = write_analysis_report(project, run, tmp_path / "report.docx", "docx")
    html = write_analysis_report(project, run, tmp_path / "report.html", "html")

    opened_pdf = fitz.open(pdf)
    try:
        assert "Apples occurs" in "".join(str(page.get_text()) for page in opened_pdf)
    finally:
        opened_pdf.close()
    with ZipFile(docx) as package:
        document_xml = package.read("word/document.xml").decode("utf-8")
    assert "Apples occurs." in document_xml and "[speech.txt · ¶1]" in document_xml
    assert "<h1>DiscourseLens analysis report</h1>" in html.read_text(encoding="utf-8")


def test_report_export_escapes_html_and_removes_xml_10_invalid_characters(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("speech.txt", "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    evidence = project.textual_evidence(project.create_span(revision, 0, 6))
    finding = project.create_finding(
        run,
        "fixture",
        "1.0.0",
        "Markup <script>alert(1)</script> \ufffe",
        "Observation includes <img src=x onerror=alert(1)> and \uffff.",
    )
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)

    html = write_analysis_report(project, run, tmp_path / "report.html", "html").read_text(
        encoding="utf-8"
    )
    docx = write_analysis_report(project, run, tmp_path / "report.docx", "docx")

    assert "<script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img src=x" not in html and "&lt;img src=x" in html
    with ZipFile(docx) as package:
        document_xml = package.read("word/document.xml")
    root = ElementTree.fromstring(document_xml)
    exported_text = "".join(root.itertext())
    assert "<script>alert(1)</script>" in exported_text
    assert "<img src=x onerror=alert(1)>" in exported_text
    assert "\ufffe" not in exported_text and "\uffff" not in exported_text


@pytest.mark.parametrize("output_format", ["markdown", "html", "json", "pdf", "docx"])
def test_report_export_replaces_symlink_without_overwriting_its_target(tmp_path, output_format):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("speech.txt", "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    protected_target = tmp_path / "protected.txt"
    protected_target.write_text("keep this file unchanged", encoding="utf-8")
    destination = tmp_path / f"report.{output_format}"
    destination.symlink_to(protected_target)

    result = write_analysis_report(project, run, destination, output_format)

    assert result == destination
    assert not destination.is_symlink()
    assert destination.is_file()
    assert protected_target.read_text(encoding="utf-8") == "keep this file unchanged"
