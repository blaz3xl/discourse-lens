import json
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app
from discourselens.reporting import (
    comparison_payload,
    finding_payload,
    render_comparison_markdown,
    render_finding_markdown,
)


def test_cli_comparison_preserves_two_populations_through_evidence_and_report(tmp_path):
    root = tmp_path / "research"
    with Project.init(root) as project:
        _, left_first = project.import_text(
            "izquierda-a.txt", "migración migración café."
        )
        _, left_second = project.import_text("izquierda-b.txt", "migración debate.")
        _, right_first = project.import_text(
            "derecha-a.txt", "migración café política política."
        )
        _, right_second = project.import_text("derecha-b.txt", "política.")
        left_corpus = project.create_corpus("left-population")
        right_corpus = project.create_corpus("right-population")
        for revision in (left_first, left_second):
            project.add_revision_to_corpus(left_corpus, revision)
        for revision in (right_first, right_second):
            project.add_revision_to_corpus(right_corpus, revision)

    runner = CliRunner()
    command = [
        "corpus",
        "compare-named",
        str(root),
        left_corpus,
        right_corpus,
        "--metric",
        "frequency",
        "--format",
        "json",
    ]
    first_result = runner.invoke(app, command)
    second_result = runner.invoke(app, command)
    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    first_output = json.loads(first_result.output)
    second_output = json.loads(second_result.output)
    assert first_output["rows"] == second_output["rows"]

    comparison_id = first_output["comparison_id"]
    assert first_output["left_corpus_id"] == left_corpus
    assert first_output["right_corpus_id"] == right_corpus
    assert first_output["left_scope_id"] != first_output["right_scope_id"]
    migration_row = next(row for row in first_output["rows"] if row["token"] == "migración")
    assert migration_row["left_count"] == 3
    assert migration_row["right_count"] == 1
    assert migration_row["left_rate"] == pytest.approx(0.6)
    assert migration_row["right_rate"] == pytest.approx(0.2)
    assert migration_row["rate_difference"] == pytest.approx(0.4)

    with Project.open(root) as project:
        comparison_report = comparison_payload(project, comparison_id)
        left_scope = comparison_report["left_scope"]
        right_scope = comparison_report["right_scope"]
        assert left_scope["id"] == first_output["left_scope_id"]
        assert right_scope["id"] == first_output["right_scope_id"]
        assert left_scope["revision_ids"] == sorted([left_first, left_second])
        assert right_scope["revision_ids"] == sorted([right_first, right_second])
        assert set(left_scope["revision_ids"]).isdisjoint(right_scope["revision_ids"])
        assert left_scope["surface_token_population"] == 5
        assert right_scope["surface_token_population"] == 5
        assert comparison_report["comparison_run"]["result_hash_verified"] is True
        markdown = render_comparison_markdown(project, comparison_id).replace("\\-", "-")
        assert (
            f"Left scope: {first_output['left_scope_id']} "
            "(2 revisions; 5 surface-token positions)"
        ) in markdown
        assert (
            f"Right scope: {first_output['right_scope_id']} "
            "(2 revisions; 5 surface-token positions)"
        ) in markdown

        comparative = cast(
            dict[str, Any],
            project.create_comparative_finding(
                comparison_id,
                "migración",
                comparability_basis=(
                    "Generated fixtures use the same language and short text format."
                ),
            ),
        )
        assert comparative["validation"] == {
            "valid": True,
            "errors": [],
            "semantic_entailment_assessed": False,
        }
        evidence_id = str(comparative["evidence_id"])
        finding_id = str(comparative["finding_id"])
        evidence = cast(dict[str, Any], project.evidence_content(evidence_id))
        provenance = evidence["provenance"]
        assert provenance["comparison_run_id"] == comparison_id
        assert provenance["left_scope_snapshot_id"] == first_output["left_scope_id"]
        assert provenance["right_scope_snapshot_id"] == first_output["right_scope_id"]
        assert provenance["left_revision_ids"] == sorted([left_first, left_second])
        assert provenance["right_revision_ids"] == sorted([right_first, right_second])
        assert provenance["left_population"] == 5
        assert provenance["right_population"] == 5

        finding_report = finding_payload(project, finding_id)
        statistic_reference = next(
            item for item in finding_report["evidence"] if item["evidence_id"] == evidence_id
        )
        assert statistic_reference["kind"] == "CORPUS_STATISTIC"
        assert comparison_id in statistic_reference["citation"]
        assert comparison_id in statistic_reference["text"]
        finding_markdown = render_finding_markdown(project, finding_id)
        assert comparison_id in finding_markdown.replace("\\-", "-")

        # The run for a comparative Finding covers the union for lineage, while
        # the statistic retains the two distinct frozen populations above.
        analysis_scope_id = project.db.execute(
            "SELECT scope_id FROM analysis_runs WHERE id=?",
            (str(comparative["analysis_run_id"]),),
        ).fetchone()["scope_id"]
        assert project.scope_revisions(analysis_scope_id) == sorted(
            [left_first, left_second, right_first, right_second]
        )
