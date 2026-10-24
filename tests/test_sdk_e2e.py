import json

from discourselens import Project, verify_research_bundle
from discourselens.corpus import TOKEN
from discourselens.types import ClaimType, CoverageDescriptor, CoverageMode, EvidenceRole


def test_readme_style_offline_research_workflow_survives_reopen(tmp_path):
    root = tmp_path / "research-project"
    source = tmp_path / "speech.txt"
    source.write_text(
        "Officials must respond to migration. The public must discuss migration carefully.",
        encoding="utf-8",
    )

    with Project.init(root) as project:
        imported = project.import_document(source)
        assert imported.status == "SUCCESS"
        assert imported.document_id is not None
        assert imported.revision_id is not None
        project.import_text("outside.txt", "Elephants live in Africa.")

        scope_id = project.snapshot([imported.revision_id])
        counts = project.corpus_frequency(scope_id)
        assert counts["migration"] == 2
        assert "elephants" not in counts

        concordance = project.corpus_kwic(scope_id, "migration", window=3)
        assert len(concordance) == 2
        assert {row["revision_id"] for row in concordance} == {imported.revision_id}

        extracted = project.extract_evidence([imported.revision_id], languages=("en",))
        assert extracted["finding_ids"]

        analysis = project.analyze_textual_descriptors(
            [imported.revision_id], languages=("en",)
        )
        finding_id = analysis["finding_ids"][0]
        finding = project.get_finding(finding_id)
        evidence_links = project.finding_evidence(finding_id)
        assert finding["analysis_run_id"] == analysis["analysis_run_id"]
        assert evidence_links
        assert project.validate_finding(finding_id).valid

        textual_link = next(link for link in evidence_links if link["span_id"])
        source_excerpt = project.span_text(textual_link["span_id"])
        report_path = project.export_analysis_report(
            analysis["analysis_run_id"], tmp_path / "report.json", format="json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["analysis_run"]["id"] == analysis["analysis_run_id"]

        bundle_path = project.export_research_bundle(
            analysis["analysis_run_id"], tmp_path / "research-bundle"
        )
        verification = verify_research_bundle(bundle_path)
        assert verification["integrity"] == "VERIFIED_SHA256"

    with Project.open(root) as reopened:
        assert reopened.revision_text(imported.revision_id) == source.read_text(encoding="utf-8")
        assert reopened.get_finding(finding_id)["id"] == finding_id
        assert reopened.get_finding(finding_id)["analysis_run_id"] == analysis["analysis_run_id"]
        assert reopened.span_text(textual_link["span_id"]) == source_excerpt
        assert reopened.validate_finding(finding_id).valid
        reopened_scope_id = reopened.db.execute(
            "SELECT scope_id FROM analysis_runs WHERE id=?", (analysis["analysis_run_id"],)
        ).fetchone()["scope_id"]
        assert reopened.scope_revisions(reopened_scope_id) == [imported.revision_id]


def test_sdk_keeps_historical_v1_frequency_and_kwic_evidence_valid_after_reopen(tmp_path):
    root = tmp_path / "historical-v1-sdk"
    source_text = "cafe\u0301 migration."
    population_definition = "all surface-token positions in the immutable ScopeSnapshot"

    with Project.init(root) as project:
        _, revision = project.import_text("legacy.txt", source_text)
        scope_id = project.snapshot([revision])
        legacy_tokens = list(TOKEN.finditer(source_text))
        population_count = len(legacy_tokens)
        coverage_id = project.coverage(
            CoverageDescriptor(
                coverage_mode=CoverageMode.EXHAUSTIVE,
                scope_snapshot_id=scope_id,
                population_definition=population_definition,
                population_count=population_count,
                inspection_count=population_count,
                completion_criterion="all scope texts tokenized by corpus.TOKEN and enumerated",
                termination_reason="completed deterministic enumeration",
            )
        )

        frequency_evidence_id = project.corpus_statistic_evidence(
            "historical v1 surface frequency",
            1,
            scope_snapshot_id=scope_id,
            population_definition=population_definition,
            provenance={
                "calculation_version": "surface-frequency-v1",
                "coverage_id": coverage_id,
                "population_count": population_count,
                "population_tokens": population_count,
                "term": "cafe",
                "tokenizer": TOKEN.pattern,
            },
        )
        kwic_evidence_id = project.textual_evidence(
            project.create_span(revision, 0, 4),
            provenance={
                "acquisition_method": "corpus_kwic",
                "query": "cafe",
                "scope_snapshot_id": scope_id,
                "coverage_descriptor_id": coverage_id,
                "coverage_mode": "EXHAUSTIVE",
                "candidate_count": 1,
                "inspected_count": population_count,
                "tokenizer": TOKEN.pattern,
                "calculation_version": "surface-kwic-v1",
            },
        )

        run_id = project.create_run(scope_id, {"fixture": "historical-v1"})
        frequency_finding = project.create_finding(
            run_id,
            "fixture",
            "1.0.0",
            "historical frequency",
            "The historical tokenizer counted the legacy surface token.",
            claim_types=[ClaimType.QUANTIFIED],
            coverage_id=coverage_id,
        )
        project.attach_evidence(frequency_finding, frequency_evidence_id, EvidenceRole.SUPPORT)
        kwic_finding = project.create_finding(
            run_id,
            "fixture",
            "1.0.0",
            "historical excerpt",
            "The historical KWIC producer cited the exact legacy token span.",
            claim_types=[ClaimType.EXAMPLE_BASED],
            coverage_id=coverage_id,
        )
        project.attach_evidence(kwic_finding, kwic_evidence_id, EvidenceRole.SUPPORT)

        assert project.validate_finding(frequency_finding).valid
        assert project.validate_finding(kwic_finding).valid

    with Project.open(root, read_only=True) as reopened:
        assert reopened.validate_finding(frequency_finding).valid
        assert reopened.validate_finding(kwic_finding).valid
        frequency_metadata = reopened.evidence_content(frequency_evidence_id)
        frequency_provenance = frequency_metadata["provenance"]
        assert isinstance(frequency_provenance, dict)
        assert frequency_provenance["calculation_version"] == "surface-frequency-v1"
        kwic_metadata = reopened.evidence_content(kwic_evidence_id)
        assert kwic_metadata["calculation_version"] == "surface-kwic-v1"
