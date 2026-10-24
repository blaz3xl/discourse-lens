from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app
from discourselens.types import CoverageMode


def _artifact_counts(project: Project) -> dict[str, int]:
    return {
        table: int(project.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in (
            "scope_snapshots",
            "coverage_descriptors",
            "source_spans",
            "evidence_artifacts",
        )
    }


def _scope_ids(project: Project) -> list[str]:
    return [
        str(row[0])
        for row in project.db.execute("SELECT id FROM scope_snapshots ORDER BY id").fetchall()
    ]


def _invoke_scope_evidence(
    root, scope_id: str, revision_id: str, query: str, start_offset: int, end_offset: int
):
    arguments = [
        "corpus",
        "kwic-evidence-scope",
        str(root),
        scope_id,
        revision_id,
        query,
    ]
    if start_offset < 0:
        arguments.append("--")
    arguments.extend((str(start_offset), str(end_offset)))
    return CliRunner().invoke(
        app,
        arguments,
    )


def test_cli_kwic_evidence_scope_persists_citable_evidence_with_original_provenance(
    tmp_path,
):
    root = tmp_path / "cli-kwic-evidence-scope"
    project = Project.init(root)
    _, first_revision = project.import_text(
        "first.txt", "The Migration pattern recurs.\n\nMigration supports reform."
    )
    _, second_revision = project.import_text("second.txt", "Another migration reference.")
    scope_id = project.snapshot([first_revision, second_revision])
    _, later_revision = project.import_text("later.txt", "Migration is outside the frozen scope.")
    project.close()

    with Project.open(root, read_only=True) as before_project:
        page = before_project.corpus_kwic_page(scope_id, "migration", window=1, limit=10)
        selected_hit = next(
            row for row in page["rows"] if row["revision_id"] == first_revision
        )
        scope_members_before = before_project.scope_revisions(scope_id)
        scope_ids_before = _scope_ids(before_project)
        counts_before = _artifact_counts(before_project)

    result = _invoke_scope_evidence(
        root,
        scope_id,
        first_revision,
        "migration",
        selected_hit["start_offset"],
        selected_hit["end_offset"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["scope_id"] == scope_id
    assert payload["revision_id"] == first_revision
    assert payload["query"] == "migration"
    assert payload["start_offset"] == selected_hit["start_offset"]
    assert payload["end_offset"] == selected_hit["end_offset"]
    assert payload["exact_text"] == selected_hit["keyword"]
    assert payload["citation"] == "[first.txt · ¶1]"

    with Project.open(root) as verified:
        assert verified.scope_revisions(scope_id) == scope_members_before == [
            first_revision,
            second_revision,
        ]
        assert later_revision not in verified.scope_revisions(scope_id)
        assert _scope_ids(verified) == scope_ids_before
        counts_after = _artifact_counts(verified)
        assert counts_after == {
            "scope_snapshots": counts_before["scope_snapshots"],
            "coverage_descriptors": counts_before["coverage_descriptors"] + 1,
            "source_spans": counts_before["source_spans"] + 1,
            "evidence_artifacts": counts_before["evidence_artifacts"] + 1,
        }

        metadata = verified.evidence_content(payload["evidence_id"])
        assert metadata["acquisition_method"] == "corpus_kwic"
        assert metadata["scope_snapshot_id"] == scope_id
        assert metadata["coverage_descriptor_id"] == payload["coverage_id"]
        assert metadata["coverage_mode"] == "EXHAUSTIVE"
        assert metadata["candidate_count"] == 3
        assert metadata["inspected_count"] == 10

        coverage = verified.coverage_descriptor(payload["coverage_id"])
        assert coverage.scope_snapshot_id == scope_id
        assert coverage.coverage_mode is CoverageMode.EXHAUSTIVE
        assert coverage.population_count == coverage.inspection_count == 10

        evidence_row = verified.db.execute(
            "SELECT span_id FROM evidence_artifacts WHERE id=?", (payload["evidence_id"],)
        ).fetchone()
        assert evidence_row is not None
        span = verified.db.execute(
            "SELECT revision_id,start_offset,end_offset FROM source_spans WHERE id=?",
            (evidence_row["span_id"],),
        ).fetchone()
        assert span is not None
        assert (span["revision_id"], span["start_offset"], span["end_offset"]) == (
            first_revision,
            selected_hit["start_offset"],
            selected_hit["end_offset"],
        )
        assert verified.span_text(evidence_row["span_id"]) == payload["exact_text"]


@pytest.mark.parametrize(
    ("case", "revision_kind", "query", "offset_kind", "expected_exit", "expected_message"),
    [
        (
            "outside-revision",
            "outside",
            "migration",
            "outside-hit",
            1,
            "selected offsets are not a KWIC hit in this scope",
        ),
        (
            "negative-start",
            "included",
            "migration",
            "negative",
            2,
            "offsets require START_OFFSET >= 0",
        ),
        (
            "reversed-range",
            "included",
            "migration",
            "reversed",
            2,
            "END_OFFSET > START_OFFSET",
        ),
        (
            "non-hit-range",
            "included",
            "migration",
            "inside-token",
            1,
            "selected offsets are not a KWIC hit in this scope",
        ),
        (
            "wrong-query",
            "included",
            "policy",
            "valid-hit",
            1,
            "selected offsets are not a KWIC hit in this scope",
        ),
    ],
)
def test_cli_kwic_evidence_scope_rejects_nonmember_and_invalid_hits_without_writes(
    tmp_path, case, revision_kind, query, offset_kind, expected_exit, expected_message
):
    root = tmp_path / f"cli-kwic-evidence-reject-{case}"
    project = Project.init(root)
    _, included_revision = project.import_text("included.txt", "The Migration supports policy.")
    scope_id = project.snapshot([included_revision])
    _, outside_revision = project.import_text("outside.txt", "Migration outside the scope.")
    project.close()

    with Project.open(root, read_only=True) as before_project:
        hit = before_project.corpus_kwic_page(scope_id, "migration", limit=10)["rows"][0]
        scope_members_before = before_project.scope_revisions(scope_id)
        scope_ids_before = _scope_ids(before_project)
        counts_before = _artifact_counts(before_project)

    revision_id = outside_revision if revision_kind == "outside" else included_revision
    if offset_kind == "outside-hit":
        start_offset = 0
        end_offset = len("Migration")
    elif offset_kind == "negative":
        start_offset, end_offset = -1, hit["end_offset"]
    elif offset_kind == "reversed":
        start_offset, end_offset = hit["end_offset"], hit["start_offset"]
    elif offset_kind == "inside-token":
        start_offset, end_offset = hit["start_offset"] + 1, hit["end_offset"]
    else:
        start_offset, end_offset = hit["start_offset"], hit["end_offset"]

    result = _invoke_scope_evidence(
        root, scope_id, revision_id, query, start_offset, end_offset
    )

    assert result.exit_code == expected_exit
    assert expected_message in result.output
    with Project.open(root, read_only=True) as after_project:
        assert after_project.scope_revisions(scope_id) == scope_members_before
        assert _scope_ids(after_project) == scope_ids_before
        assert _artifact_counts(after_project) == counts_before
