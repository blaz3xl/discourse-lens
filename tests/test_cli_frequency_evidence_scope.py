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


def _invoke(root, scope_id: str, term: str):
    return CliRunner().invoke(
        app,
        ["corpus", "frequency-evidence-scope", str(root), scope_id, term],
    )


def _artifact_counts(project: Project) -> dict[str, int]:
    return {
        table: int(project.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("scope_snapshots", "coverage_descriptors", "evidence_artifacts")
    }


def test_cli_frequency_evidence_scope_persists_canonical_evidence_for_frozen_scope(
    tmp_path,
):
    root = tmp_path / "cli-frequency-evidence-scope"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "Evidence evidence alpha")
    _, second_revision = project.import_text("second.txt", "evidence beta gamma")
    scope_id = project.snapshot([first_revision, second_revision])
    project.close()

    with Project.open(root) as writer:
        _, later_revision = writer.import_text("later.txt", "evidence outsideonly outsideonly")

    with Project.open(root, read_only=True) as before_project:
        scope_revisions = before_project.scope_revisions(scope_id)
        counts_before = _artifact_counts(before_project)

    result = _invoke(root, scope_id, "Evidence")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert set(payload) == {
        "term",
        "count",
        "population_count",
        "scope_id",
        "coverage_id",
        "evidence_id",
    }
    assert payload["term"] == "evidence"
    assert payload["count"] == 3
    assert payload["population_count"] == 6
    assert payload["scope_id"] == scope_id
    assert payload["coverage_id"]
    assert payload["evidence_id"]

    with Project.open(root, read_only=True) as verified:
        assert verified.scope_revisions(scope_id) == scope_revisions == [
            first_revision,
            second_revision,
        ]
        assert later_revision not in verified.scope_revisions(scope_id)
        assert _artifact_counts(verified) == {
            "scope_snapshots": counts_before["scope_snapshots"],
            "coverage_descriptors": counts_before["coverage_descriptors"] + 1,
            "evidence_artifacts": counts_before["evidence_artifacts"] + 1,
        }

        coverage = verified.coverage_descriptor(payload["coverage_id"])
        assert coverage.coverage_mode is CoverageMode.EXHAUSTIVE
        assert coverage.scope_snapshot_id == scope_id
        assert coverage.population_count == coverage.inspection_count == 6
        assert coverage.termination_state is CoverageTerminationState.COMPLETED
        assert coverage.negative_result_status is NegativeResultStatus.NOT_APPLICABLE

        evidence = verified.evidence_content(payload["evidence_id"])
        assert evidence["value"] == payload["count"] == 3
        assert evidence["scope_snapshot_id"] == scope_id
        assert evidence["population_definition"] == (
            "all surface-token positions in the immutable ScopeSnapshot"
        )
        provenance = evidence.get("provenance")
        assert isinstance(provenance, dict)
        assert provenance["coverage_id"] == payload["coverage_id"]
        assert provenance["population_count"] == provenance["population_tokens"] == 6
        assert provenance["term"] == "evidence"
        assert verified.render_citation(payload["evidence_id"]) == (
            "[corpus statistic · surface frequency for 'evidence']"
        )


def test_cli_frequency_evidence_scope_marks_zero_count_as_exhaustive_absence(tmp_path):
    root = tmp_path / "cli-frequency-evidence-scope-zero"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "alpha beta gamma")
    scope_id = project.snapshot([revision])
    project.close()

    result = _invoke(root, scope_id, "missing")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["term"] == "missing"
    assert payload["count"] == 0
    assert payload["population_count"] == 3
    assert payload["scope_id"] == scope_id

    with Project.open(root, read_only=True) as verified:
        coverage = verified.coverage_descriptor(payload["coverage_id"])
        assert coverage.scope_snapshot_id == scope_id
        assert coverage.population_count == coverage.inspection_count == 3
        assert coverage.negative_result_status is (
            NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION
        )
        assert verified.evidence_content(payload["evidence_id"])["value"] == 0


@pytest.mark.parametrize(
    ("term", "message"),
    [
        ("", "term cannot be blank"),
        ("   ", "term cannot be blank"),
        ("evidence must", "term must contain exactly one surface token"),
        ("x" * 257, "term must be at most 256 characters"),
    ],
)
def test_cli_frequency_evidence_scope_rejects_invalid_terms_before_open(
    tmp_path, monkeypatch, term, message
):
    root = tmp_path / "cli-frequency-evidence-scope-invalid-term"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "evidence words here")
    scope_id = project.snapshot([revision])
    project.close()

    original_open = Project.open
    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("invalid terms must fail before opening the project")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = _invoke(root, scope_id, term)

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


def test_cli_frequency_evidence_scope_unknown_scope_creates_no_orphan_artifacts(tmp_path):
    root = tmp_path / "cli-frequency-evidence-scope-unknown"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "evidence words here")
    scope_id = project.snapshot([revision])
    counts_before = _artifact_counts(project)
    project.close()

    result = _invoke(root, "missing-scope", "evidence")

    assert result.exit_code == 2
    assert "unknown scope" in result.output.lower()
    assert str(root) not in result.output

    with Project.open(root, read_only=True) as verified:
        assert _artifact_counts(verified) == counts_before
        assert verified.scope_revisions(scope_id) == [revision]
