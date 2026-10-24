from __future__ import annotations

import json
import math

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app
from discourselens.types import CoverageMode, CoverageTerminationState


def _invoke(
    root,
    scope_id: str,
    query: str,
    collocate: str,
    *,
    window: int = 1,
    min_frequency: int = 1,
):
    return CliRunner().invoke(
        app,
        [
            "corpus",
            "collocation-evidence-scope",
            str(root),
            scope_id,
            query,
            collocate,
            "--window",
            str(window),
            "--min-frequency",
            str(min_frequency),
        ],
    )


def _artifact_counts(project: Project) -> dict[str, int]:
    return {
        table: int(project.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("scope_snapshots", "coverage_descriptors", "evidence_artifacts")
    }


def test_cli_collocation_evidence_scope_persists_exhaustive_scores_for_frozen_scope(
    tmp_path,
):
    root = tmp_path / "cli-collocation-evidence-scope"
    project = Project.init(root)
    _, first_revision = project.import_text(
        "first.txt", "apple red apple red apple blue"
    )
    _, second_revision = project.import_text("second.txt", "green apple red blue")
    scope_id = project.snapshot([first_revision, second_revision])
    project.close()

    with Project.open(root) as writer:
        _, later_revision = writer.import_text("later.txt", "apple red apple red")

    with Project.open(root, read_only=True) as before_project:
        scope_revisions = before_project.scope_revisions(scope_id)
        counts_before = _artifact_counts(before_project)

    result = _invoke(root, scope_id, "Apple", "RED")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["scope_id"] == scope_id
    assert payload["query"] == "apple"
    assert payload["collocate"] == "red"
    assert payload["window"] == 1
    assert payload["focus_frequency"] == 4
    assert payload["pair_frequency"] == 5
    assert payload["token_frequency"] == 3
    assert payload["population_tokens"] == 10
    assert payload["pair_opportunities"] == 7
    assert payload["expected_frequency"] == pytest.approx(3.5)
    assert payload["mutual_information"] == pytest.approx(math.log2(5 / 3.5))
    assert payload["t_score"] == pytest.approx(1.5 / math.sqrt(5))
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
        assert coverage.population_count == payload["pair_opportunities"] == 7
        assert coverage.termination_state is CoverageTerminationState.COMPLETED

        evidence = verified.evidence_content(payload["evidence_id"])
        assert evidence["value"] == pytest.approx(payload["mutual_information"])
        assert evidence["scope_snapshot_id"] == scope_id
        provenance = evidence.get("provenance")
        assert isinstance(provenance, dict)
        assert provenance["coverage_id"] == payload["coverage_id"]
        assert provenance["query"] == "apple"
        assert provenance["collocate"] == "red"
        assert provenance["pair_frequency"] == 5
        assert provenance["token_frequency"] == 3
        assert provenance["population_tokens"] == 10
        assert provenance["pair_opportunities"] == 7
        assert "not significance tests" in str(provenance["interpretation_note"])
        assert "p_value" not in provenance


@pytest.mark.parametrize(
    ("scope_id", "query", "collocate", "window", "min_frequency", "message"),
    [
        ("known-scope", "", "red", 1, 1, "query cannot be blank"),
        ("known-scope", "   ", "red", 1, 1, "query cannot be blank"),
        ("known-scope", "apple", "", 1, 1, "collocate cannot be blank"),
        ("known-scope", "apple", "two words", 1, 1, "exactly one surface"),
        ("known-scope", "two words", "red", 1, 1, "exactly one surface"),
        ("known-scope", "x" * 257, "red", 1, 1, "at most 256 characters"),
        ("known-scope", "apple", "x" * 257, 1, 1, "at most 256 characters"),
        ("", "apple", "red", 1, 1, "scope ID cannot be blank"),
        ("known-scope", "apple", "red", 0, 1, "--window"),
        ("known-scope", "apple", "red", 101, 1, "--window"),
        ("known-scope", "apple", "red", 1, 0, "--min-frequency"),
    ],
)
def test_cli_collocation_evidence_scope_rejects_invalid_inputs_before_open(
    tmp_path, monkeypatch, scope_id, query, collocate, window, min_frequency, message
):
    root = tmp_path / "cli-collocation-evidence-scope-invalid"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "apple red apple blue")
    project.snapshot([revision])
    project.close()

    original_open = Project.open
    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("invalid collocation evidence input must fail before opening")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = _invoke(
        root,
        scope_id,
        query,
        collocate,
        window=window,
        min_frequency=min_frequency,
    )

    assert result.exit_code == 2
    assert message in " ".join(result.output.split())
    assert "Traceback" not in result.output
    assert opened == []

    with original_open(root, read_only=True) as verified:
        assert _artifact_counts(verified) == {
            "scope_snapshots": 1,
            "coverage_descriptors": 0,
            "evidence_artifacts": 0,
        }


@pytest.mark.parametrize(
    ("scope_id", "query", "collocate", "message"),
    [
        ("missing-scope", "apple", "red", "unknown scope"),
        ("existing-scope", "apple", "APPLE", "self-collocation evidence is not supported"),
        (
            "existing-scope",
            "apple",
            "absent",
            "pair-frequency",
        ),
    ],
)
def test_cli_collocation_evidence_scope_core_failures_create_no_orphans(
    tmp_path, scope_id, query, collocate, message
):
    root = tmp_path / "cli-collocation-evidence-scope-core-errors"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "apple red apple blue")
    existing_scope = project.snapshot([revision])
    counts_before = _artifact_counts(project)
    project.close()
    requested_scope = existing_scope if scope_id == "existing-scope" else scope_id

    result = _invoke(root, requested_scope, query, collocate)

    assert result.exit_code == 2
    assert message in " ".join(result.output.split())
    with Project.open(root, read_only=True) as verified:
        assert _artifact_counts(verified) == counts_before
        assert verified.scope_revisions(existing_scope) == [revision]


def test_cli_collocation_evidence_scope_empty_population_creates_no_orphans(tmp_path):
    root = tmp_path / "cli-collocation-evidence-scope-empty"
    project = Project.init(root)
    _, revision = project.import_text("empty.txt", " \n")
    scope_id = project.snapshot([revision])
    counts_before = _artifact_counts(project)
    project.close()

    result = _invoke(root, scope_id, "apple", "red")

    assert result.exit_code == 2
    assert "pair-frequency" in " ".join(result.output.split())
    with Project.open(root, read_only=True) as verified:
        assert _artifact_counts(verified) == counts_before
        assert verified.scope_revisions(scope_id) == [revision]
