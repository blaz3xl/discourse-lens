import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def test_cli_analysis_commands_deduplicate_repeated_languages(tmp_path):
    root = tmp_path / "project"
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "We must act.")
    project.close()

    runner = CliRunner()
    descriptor_result = runner.invoke(
        app,
        [
            "analyze",
            "textual-descriptors",
            str(root),
            revision_id,
            "--languages",
            "en,en",
        ],
    )
    assert descriptor_result.exit_code == 0, descriptor_result.output
    descriptor_output = json.loads(descriptor_result.output)

    extraction_result = runner.invoke(
        app,
        [
            "analyze",
            "evidence-extraction",
            str(root),
            revision_id,
            "--languages",
            "en,en",
        ],
    )
    assert extraction_result.exit_code == 0, extraction_result.output
    extraction_output = json.loads(extraction_result.output)

    reopened = Project.open(root)
    try:
        for result in (descriptor_output, extraction_output):
            row = reopened.db.execute(
                "SELECT config_json FROM analysis_runs WHERE id=?",
                (result["analysis_run_id"],),
            ).fetchone()
            assert row is not None
            assert json.loads(row["config_json"])["languages"] == ["en"]

        descriptor_finding_ids = descriptor_output["finding_ids"]
        assert len(descriptor_finding_ids) == 1
        descriptor_evidence = reopened.finding_evidence(descriptor_finding_ids[0])
        assert len(descriptor_evidence) == 1
        assert reopened.span_text(str(descriptor_evidence[0]["span_id"])) == "must"

        assert extraction_output["candidate_count"] == 1
        extraction_finding_ids = extraction_output["finding_ids"]
        assert len(extraction_finding_ids) == 1
        extraction_evidence = reopened.finding_evidence(extraction_finding_ids[0])
        assert len(extraction_evidence) == 1
        assert extraction_evidence[0]["rationale"] == (
            "Deterministic candidate cues: en:modality, en:actor_reference"
        )
    finally:
        reopened.close()
