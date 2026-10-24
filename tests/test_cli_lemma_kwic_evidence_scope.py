from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import build_parsed_token_artifact
from discourselens.types import CoverageMode


class _LemmaBackend:
    language = "en"
    backend_id = "cli-lemma-evidence-fixture"
    backend_version = "1.0"
    model_id = "fixture-model"
    model_version = "1.0"

    def parse(self, text: str) -> list[ParsedToken]:
        tokens = []
        for index, match in enumerate(re.finditer(r"\w+", text)):
            surface = match.group()
            lemma = "migrant" if surface.casefold() in {"migrants", "migrant"} else surface.casefold()
            tokens.append(
                ParsedToken(
                    surface,
                    lemma,
                    "NOUN",
                    None,
                    None,
                    "ROOT" if index == 0 else "dep",
                    match.start(),
                    match.end(),
                )
            )
        return tokens


def _invoke_scope_evidence(
    root,
    scope_id: str,
    lemma: str,
    artifact_ids: list[str],
    revision_id: str,
    start_offset: int,
    end_offset: int,
):
    return CliRunner().invoke(
        app,
        [
            "corpus",
            "lemma-kwic-evidence-scope",
            str(root),
            scope_id,
            lemma,
            "--artifacts",
            ",".join(artifact_ids),
            "--revision-id",
            revision_id,
            "--start-offset",
            str(start_offset),
            "--end-offset",
            str(end_offset),
        ],
    )


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


def test_cli_lemma_kwic_evidence_scope_persists_selected_occurrence_in_frozen_scope(
    tmp_path,
):
    root = tmp_path / "cli-lemma-kwic-evidence-scope"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "Migrants support migration.")
    _, second_revision = project.import_text("second.txt", "A migrant speaks.")
    backend = _LemmaBackend()
    artifacts = [
        build_parsed_token_artifact(project, first_revision, backend),
        build_parsed_token_artifact(project, second_revision, backend),
    ]
    artifact_ids = [artifact.artifact_id for artifact in artifacts]
    scope_id = project.snapshot([first_revision, second_revision])
    hit = next(
        row
        for row in project.corpus_lemma_kwic(scope_id, artifact_ids, "migrant", window=0)["rows"]
        if row["revision_id"] == first_revision
    )
    project.close()

    with Project.open(root) as writer:
        _, later_revision = writer.import_text("later.txt", "Migrant outside the scope.")
        later_artifact = build_parsed_token_artifact(writer, later_revision, backend)

    with Project.open(root, read_only=True) as before_project:
        scope_members_before = before_project.scope_revisions(scope_id)
        scope_ids_before = _scope_ids(before_project)
        counts_before = _artifact_counts(before_project)

    result = _invoke_scope_evidence(
        root,
        scope_id,
        "migrant",
        artifact_ids,
        first_revision,
        hit["start_offset"],
        hit["end_offset"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["scope_id"] == scope_id
    assert payload["artifact_ids"] == sorted(artifact_ids)
    assert payload["revision_id"] == first_revision
    assert payload["lemma"] == "migrant"
    assert (payload["start_offset"], payload["end_offset"]) == (
        hit["start_offset"],
        hit["end_offset"],
    )
    assert payload["exact_text"] == "Migrants"
    assert payload["coverage_mode"] == CoverageMode.EXAMPLE_BASED.value
    assert payload["coverage_id"]
    assert payload["evidence_id"]
    assert payload["citation"] == "[first.txt · ¶1]"
    assert later_artifact.artifact_id not in payload["artifact_ids"]

    with Project.open(root) as verified:
        assert verified.scope_revisions(scope_id) == scope_members_before == [
            first_revision,
            second_revision,
        ]
        assert later_revision not in verified.scope_revisions(scope_id)
        assert _scope_ids(verified) == scope_ids_before
        assert _artifact_counts(verified) == {
            "scope_snapshots": counts_before["scope_snapshots"],
            "coverage_descriptors": counts_before["coverage_descriptors"] + 1,
            "source_spans": counts_before["source_spans"] + 1,
            "evidence_artifacts": counts_before["evidence_artifacts"] + 1,
        }

        evidence = verified.evidence_content(payload["evidence_id"])
        assert evidence["acquisition_method"] == "corpus_lemma_kwic"
        assert evidence["scope_snapshot_id"] == scope_id
        assert evidence["artifact_ids"] == sorted(artifact_ids)
        assert evidence["lemma"] == "migrant"
        assert evidence["coverage_descriptor_id"] == payload["coverage_id"]
        coverage = verified.coverage_descriptor(payload["coverage_id"])
        assert coverage.scope_snapshot_id == scope_id
        assert coverage.coverage_mode is CoverageMode.EXAMPLE_BASED
        assert coverage.population_count is None
        assert coverage.inspection_count == 1
        assert verified.render_citation(payload["evidence_id"]) == payload["citation"]


def test_cli_lemma_kwic_evidence_scope_rejects_out_of_scope_target_without_writes(tmp_path):
    root = tmp_path / "cli-lemma-kwic-evidence-scope-outside"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "Migrant included.")
    _, second_revision = project.import_text("second.txt", "Another migrant.")
    backend = _LemmaBackend()
    artifacts = [
        build_parsed_token_artifact(project, first_revision, backend),
        build_parsed_token_artifact(project, second_revision, backend),
    ]
    scope_id = project.snapshot([first_revision, second_revision])
    _, later_revision = project.import_text("later.txt", "Migrant outside.")
    project.close()

    with Project.open(root, read_only=True) as before_project:
        counts_before = _artifact_counts(before_project)
        scope_members = before_project.scope_revisions(scope_id)

    result = _invoke_scope_evidence(
        root,
        scope_id,
        "migrant",
        [artifact.artifact_id for artifact in artifacts],
        later_revision,
        0,
        7,
    )

    assert result.exit_code == 1
    assert "not a lemma KWIC hit in this scope" in " ".join(result.output.split())
    with Project.open(root, read_only=True) as verified:
        assert _artifact_counts(verified) == counts_before
        assert verified.scope_revisions(scope_id) == scope_members == [
            first_revision,
            second_revision,
        ]


@pytest.mark.parametrize(
    ("scope_id", "lemma", "artifacts", "revision_id", "start", "end", "message"),
    [
        (" ", "migrant", ["artifact"], "revision", 0, 1, "scope ID cannot be blank"),
        ("scope", " ", ["artifact"], "revision", 0, 1, "lemma must not be empty"),
        ("scope", "x" * 257, ["artifact"], "revision", 0, 1, "at most 256 characters"),
        ("scope", "migrant", [], "revision", 0, 1, "must include at least one"),
        ("scope", "migrant", ["artifact"], " ", 0, 1, "revision ID cannot be blank"),
        ("scope", "migrant", ["artifact"], "revision", 2, 1, "greater than"),
        ("scope", "migrant", ["artifact"], "revision", -1, 1, "--start-offset"),
        ("scope", "migrant", ["artifact"], "revision", 0, 0, "--end-offset"),
    ],
)
def test_cli_lemma_kwic_evidence_scope_rejects_invalid_args_before_open(
    tmp_path, monkeypatch, scope_id, lemma, artifacts, revision_id, start, end, message
):
    root = tmp_path / "cli-lemma-kwic-evidence-scope-invalid"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "Migrant.")
    scope = project.snapshot([revision])
    counts_before = _artifact_counts(project)
    project.close()

    original_open = Project.open
    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("invalid lemma-KWIC evidence input must fail before opening")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = _invoke_scope_evidence(root, scope_id, lemma, artifacts, revision_id, start, end)

    assert result.exit_code == 2
    assert message in " ".join(result.output.split())
    assert "Traceback" not in result.output
    assert opened == []

    with original_open(root, read_only=True) as verified:
        assert _artifact_counts(verified) == counts_before
        assert verified.scope_revisions(scope) == [revision]


def test_cli_lemma_kwic_evidence_scope_unknown_scope_creates_no_orphans(tmp_path):
    root = tmp_path / "cli-lemma-kwic-evidence-scope-unknown"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "Migrant.")
    artifact = build_parsed_token_artifact(project, revision, _LemmaBackend())
    scope_id = project.snapshot([revision])
    counts_before = _artifact_counts(project)
    project.close()

    result = _invoke_scope_evidence(
        root, "missing-scope", "migrant", [artifact.artifact_id], revision, 0, 7
    )

    assert result.exit_code == 2
    assert "unknown scope" in " ".join(result.output.split())
    with Project.open(root, read_only=True) as verified:
        assert _artifact_counts(verified) == counts_before
        assert verified.scope_revisions(scope_id) == [revision]


def test_cli_lemma_kwic_evidence_scope_nonmatching_occurrence_creates_no_orphans(tmp_path):
    root = tmp_path / "cli-lemma-kwic-evidence-scope-invalid-occurrence"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "Migrant.")
    artifact = build_parsed_token_artifact(project, revision, _LemmaBackend())
    scope_id = project.snapshot([revision])
    counts_before = _artifact_counts(project)
    project.close()

    result = _invoke_scope_evidence(
        root, scope_id, "migrant", [artifact.artifact_id], revision, 0, 1
    )

    assert result.exit_code == 1
    assert "not a lemma KWIC hit in this scope" in " ".join(result.output.split())
    with Project.open(root, read_only=True) as verified:
        assert _artifact_counts(verified) == counts_before
        assert verified.scope_revisions(scope_id) == [revision]
