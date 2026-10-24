from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app
from discourselens.types import (
    CoverageMode,
    CoverageTerminationState,
    NegativeResultStatus,
)


def _invoke(root, scope_id: str, bundle: str, size: int = 2):
    return CliRunner().invoke(
        app,
        [
            "corpus",
            "lexical-bundle-evidence-scope",
            str(root),
            scope_id,
            bundle,
            "--size",
            str(size),
        ],
    )


def _artifact_counts(project: Project) -> dict[str, int]:
    return {
        table: int(project.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("scope_snapshots", "coverage_descriptors", "evidence_artifacts")
    }


def test_cli_lexical_bundle_evidence_scope_uses_only_frozen_document_populations(tmp_path):
    root = tmp_path / "cli-lexical-bundle-evidence-scope"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "alpha beta gamma")
    _, second_revision = project.import_text("second.txt", "delta epsilon zeta")
    scope_id = project.snapshot([first_revision, second_revision])
    project.close()

    with Project.open(root) as writer:
        _, later_revision = writer.import_text(
            "later.txt", "alpha beta alpha beta outsider"
        )

    with Project.open(root, read_only=True) as before_project:
        scope_revisions = before_project.scope_revisions(scope_id)
        counts_before = _artifact_counts(before_project)

    present = _invoke(root, scope_id, "Alpha beta")
    boundary_only = _invoke(root, scope_id, "gamma delta")

    assert present.exit_code == 0, present.output
    assert boundary_only.exit_code == 0, boundary_only.output
    present_payload = json.loads(present.stdout)
    absent_payload = json.loads(boundary_only.stdout)
    assert set(present_payload) == {
        "bundle",
        "size",
        "count",
        "population_count",
        "scope_id",
        "coverage_id",
        "evidence_id",
    }
    assert present_payload["bundle"] == "alpha beta"
    assert present_payload["size"] == 2
    assert present_payload["count"] == 1
    assert present_payload["population_count"] == 4
    assert present_payload["scope_id"] == scope_id
    assert absent_payload["bundle"] == "gamma delta"
    assert absent_payload["count"] == 0
    assert absent_payload["population_count"] == 4
    assert absent_payload["scope_id"] == scope_id

    with Project.open(root, read_only=True) as verified:
        assert verified.scope_revisions(scope_id) == scope_revisions == [
            first_revision,
            second_revision,
        ]
        assert later_revision not in verified.scope_revisions(scope_id)
        assert _artifact_counts(verified) == {
            "scope_snapshots": counts_before["scope_snapshots"],
            "coverage_descriptors": counts_before["coverage_descriptors"] + 2,
            "evidence_artifacts": counts_before["evidence_artifacts"] + 2,
        }

        present_coverage = verified.coverage_descriptor(present_payload["coverage_id"])
        assert present_coverage.coverage_mode is CoverageMode.EXHAUSTIVE
        assert present_coverage.scope_snapshot_id == scope_id
        assert present_coverage.population_count == present_coverage.inspection_count == 4
        assert present_coverage.termination_state is CoverageTerminationState.COMPLETED
        assert present_coverage.negative_result_status is NegativeResultStatus.NOT_APPLICABLE

        absent_coverage = verified.coverage_descriptor(absent_payload["coverage_id"])
        assert absent_coverage.scope_snapshot_id == scope_id
        assert absent_coverage.population_count == absent_coverage.inspection_count == 4
        assert absent_coverage.negative_result_status is (
            NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION
        )

        evidence = verified.evidence_content(present_payload["evidence_id"])
        assert evidence["value"] == present_payload["count"] == 1
        assert evidence["scope_snapshot_id"] == scope_id
        assert evidence["population_definition"] == (
            "all contiguous 2-token surface n-gram positions in the immutable ScopeSnapshot"
        )
        provenance = evidence.get("provenance")
        assert isinstance(provenance, dict)
        assert provenance["coverage_id"] == present_payload["coverage_id"]
        assert provenance["bundle"] == "alpha beta"
        assert provenance["population_count"] == 4
        assert provenance["size"] == 2
        assert verified.evidence_content(absent_payload["evidence_id"])["value"] == 0


@pytest.mark.parametrize(
    ("scope_id", "bundle", "size", "message"),
    [
        ("known-scope", "alpha beta", 1, "--size"),
        ("known-scope", "alpha beta", 11, "--size"),
        ("known-scope", "alpha", 2, "exactly 2 surface tokens"),
        ("known-scope", "alpha beta gamma", 2, "exactly 2 surface tokens"),
        ("known-scope", "", 2, "bundle cannot be blank"),
        ("known-scope", "   ", 2, "bundle cannot be blank"),
        ("known-scope", "x" * 4_097, 2, "bundle must be at most 4096 characters"),
        ("", "alpha beta", 2, "scope ID cannot be blank"),
    ],
)
def test_cli_lexical_bundle_evidence_scope_rejects_invalid_input_before_open(
    tmp_path, monkeypatch, scope_id, bundle, size, message
):
    root = tmp_path / "cli-lexical-bundle-evidence-scope-invalid"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "alpha beta gamma")
    project.snapshot([revision])
    project.close()

    original_open = Project.open
    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("invalid bundle arguments must fail before opening the project")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = _invoke(root, scope_id, bundle, size)

    assert result.exit_code == 2
    assert message in result.output
    assert "Traceback" not in result.output
    assert opened == []

    with original_open(root, read_only=True) as verified:
        assert _artifact_counts(verified) == {
            "scope_snapshots": 1,
            "coverage_descriptors": 0,
            "evidence_artifacts": 0,
        }


def test_cli_lexical_bundle_evidence_scope_unknown_scope_creates_no_orphans(tmp_path):
    root = tmp_path / "cli-lexical-bundle-evidence-scope-unknown"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "alpha beta gamma")
    existing_scope = project.snapshot([revision])
    counts_before = _artifact_counts(project)
    project.close()

    result = _invoke(root, "missing-scope", "alpha beta")

    assert result.exit_code == 2
    assert "unknown scope" in result.output.lower()
    assert str(root) not in result.output

    with Project.open(root, read_only=True) as verified:
        assert _artifact_counts(verified) == counts_before
        assert verified.scope_revisions(existing_scope) == [revision]


def test_cli_lexical_bundle_evidence_scope_empty_ngram_population_creates_no_orphans(
    tmp_path,
):
    root = tmp_path / "cli-lexical-bundle-evidence-scope-empty-population"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "alpha")
    _, second_revision = project.import_text("second.txt", "beta")
    scope_id = project.snapshot([first_revision, second_revision])
    counts_before = _artifact_counts(project)
    project.close()

    result = _invoke(root, scope_id, "alpha beta")

    assert result.exit_code == 2
    assert "non-empty n-gram population" in result.output
    assert str(root) not in result.output

    with Project.open(root, read_only=True) as verified:
        assert _artifact_counts(verified) == counts_before
        assert verified.scope_revisions(scope_id) == [first_revision, second_revision]
