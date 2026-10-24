import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app
from discourselens.types import EvidenceRole


def test_offline_cli_evidence_report_and_bundle_round_trip(tmp_path):
    runner = CliRunner()
    project_root = tmp_path / "research"
    selected_source = tmp_path / "selected.txt"
    outside_source = tmp_path / "outside.txt"
    selected_source.write_text(
        "Café policy: la migración aparece en este pasaje seleccionado.\n",
        encoding="utf-8",
    )
    outside_source.write_text(
        "Elephants live in Africa. This document is deliberately out of scope.\n",
        encoding="utf-8",
    )

    initialized = runner.invoke(app, ["project", "init", str(project_root)])
    assert initialized.exit_code == 0, initialized.output

    imported_selected = runner.invoke(
        app, ["document", "import", str(project_root), str(selected_source)]
    )
    imported_outside = runner.invoke(
        app, ["document", "import", str(project_root), str(outside_source)]
    )
    assert imported_selected.exit_code == 0, imported_selected.output
    assert imported_outside.exit_code == 0, imported_outside.output
    selected_revision = json.loads(imported_selected.stdout)["revision_id"]
    outside_revision = json.loads(imported_outside.stdout)["revision_id"]
    assert selected_revision and outside_revision and selected_revision != outside_revision

    isolated_miss = runner.invoke(
        app,
        ["search", "exact", str(project_root), selected_revision, "Elephants"],
    )
    scoped_hit = runner.invoke(
        app,
        ["search", "exact", str(project_root), outside_revision, "Elephants"],
    )
    assert isolated_miss.exit_code == scoped_hit.exit_code == 0
    assert json.loads(isolated_miss.stdout)["total_count"] == 0
    assert json.loads(scoped_hit.stdout)["total_count"] == 1

    kwic = runner.invoke(
        app, ["corpus", "kwic", str(project_root), selected_revision, "migración"]
    )
    assert kwic.exit_code == 0, kwic.output
    kwic_row = json.loads(kwic.stdout)["rows"][0]
    persisted = runner.invoke(
        app,
        [
            "corpus",
            "kwic-evidence",
            str(project_root),
            selected_revision,
            "migración",
            str(kwic_row["start_offset"]),
            str(kwic_row["end_offset"]),
        ],
    )
    assert persisted.exit_code == 0, persisted.output
    evidence_result = json.loads(persisted.stdout)
    assert evidence_result["exact_text"] == "migración"
    assert evidence_result["evidence_id"]

    project = Project.open(project_root)
    try:
        analysis_scope = project.snapshot([selected_revision])
        run_id = project.create_run(analysis_scope, {"profile": "offline-cli-e2e-v1"})
        finding_id = project.create_finding(
            run_id,
            "fixture.exact-search",
            "1.0.0",
            "lexical observation",
            "The selected passage contains the surface form ‘migración’.",
        )
        project.attach_evidence(finding_id, evidence_result["evidence_id"], EvidenceRole.SUPPORT)
        project.finalize_run(run_id)
    finally:
        project.close()

    validation = runner.invoke(
        app, ["finding", "validate", str(project_root), finding_id]
    )
    assert validation.exit_code == 0, validation.output
    assert json.loads(validation.stdout)["valid"] is True

    markdown_path = tmp_path / "analysis.md"
    markdown_export = runner.invoke(
        app,
        [
            "report",
            "export",
            str(project_root),
            run_id,
            str(markdown_path),
            "--format",
            "markdown",
        ],
    )
    assert markdown_export.exit_code == 0, markdown_export.output
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "migración" in markdown and r"\[selected\.txt · ¶1\]" in markdown
    assert "Elephants" not in markdown and "outside.txt" not in markdown

    json_path = tmp_path / "analysis.json"
    json_export = runner.invoke(
        app,
        [
            "report",
            "export",
            str(project_root),
            run_id,
            str(json_path),
            "--format",
            "json",
        ],
    )
    assert json_export.exit_code == 0, json_export.output
    report_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert report_json["findings"][0]["evidence"][0]["text"] == "migración"
    assert report_json["findings"][0]["evidence"][0]["citation"] == "[selected.txt · ¶1]"
    assert "Elephants" not in json.dumps(report_json, ensure_ascii=False)
    assert "outside.txt" not in json.dumps(report_json, ensure_ascii=False)

    bundle_dir = tmp_path / "research-bundle"
    bundle_export = runner.invoke(
        app, ["export", "research-bundle", str(project_root), run_id, str(bundle_dir)]
    )
    assert bundle_export.exit_code == 0, bundle_export.output
    manifest_path = json.loads(bundle_export.stdout)["manifest"]
    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    assert [item["id"] for item in manifest["document_revisions"]] == [selected_revision]
    assert manifest["evidence"][finding_id][0]["text"] == "migración"
    assert "Elephants" not in json.dumps(manifest, ensure_ascii=False)
    assert "outside.txt" not in json.dumps(manifest, ensure_ascii=False)

    verified = runner.invoke(app, ["export", "verify-bundle", manifest_path])
    assert verified.exit_code == 0, verified.output
    verification = json.loads(verified.stdout)
    assert verification["status"] == "VALID"
    assert verification["integrity"] == "VERIFIED_SHA256"
    assert verification["analysis_run_id"] == run_id
