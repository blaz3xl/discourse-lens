from __future__ import annotations

import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def _project_with_linked_findings(project_root):
    project = Project.init(project_root)
    _, revision_id = project.import_text("source.txt", "Private source material.")
    run_id = project.create_run(project.snapshot([revision_id]), {"profile": "lineage-test"})
    project.start_run(run_id)
    source_finding_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "source-observation",
        "Sensitive source observation must not be included.",
    )
    synthesized_finding_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "synthesis",
        "Sensitive synthesis text must not be included.",
    )
    unlinked_finding_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "unlinked",
        "This finding has no source links.",
    )
    project.link_finding_source(synthesized_finding_id, source_finding_id, "SYNTHESIZES")
    project.finalize_run(run_id)
    project.close()
    return synthesized_finding_id, source_finding_id, unlinked_finding_id


def test_finding_sources_reports_only_lineage_ids_and_relation(tmp_path):
    project_root = tmp_path / "lineage-project"
    synthesized_finding_id, source_finding_id, _ = _project_with_linked_findings(project_root)

    result = CliRunner().invoke(
        app,
        ["finding", "sources", str(project_root), synthesized_finding_id],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "finding_id": synthesized_finding_id,
        "sources": [
            {"source_finding_id": source_finding_id, "relation": "SYNTHESIZES"},
        ],
    }
    assert "Sensitive source observation" not in result.stdout
    assert "Sensitive synthesis text" not in result.stdout


def test_finding_sources_returns_empty_array_when_finding_has_no_sources(tmp_path):
    project_root = tmp_path / "lineage-project"
    _, _, unlinked_finding_id = _project_with_linked_findings(project_root)

    result = CliRunner().invoke(
        app,
        ["finding", "sources", str(project_root), unlinked_finding_id],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"finding_id": unlinked_finding_id, "sources": []}


def test_finding_sources_command_is_discoverable():
    result = CliRunner().invoke(app, ["finding", "--help"])

    assert result.exit_code == 0, result.output
    assert "sources" in result.stdout
