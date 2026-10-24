from __future__ import annotations

import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def test_corpus_snapshot_freezes_membership_and_emits_json(tmp_path):
    root = tmp_path / "corpus-snapshot-project"
    project = Project.init(root)
    corpus_id = project.create_corpus("Research corpus")
    _, first_revision = project.import_text("first.txt", "First source.")
    _, second_revision = project.import_text("second.txt", "Second source.")
    project.add_revision_to_corpus(corpus_id, first_revision)
    project.close()

    result = CliRunner().invoke(app, ["corpus", "snapshot", str(root), corpus_id])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["corpus_id"] == corpus_id
    assert payload["revision_count"] == 1
    assert payload["revision_ids"] == [first_revision]

    reopened = Project.open(root)
    try:
        assert reopened.scope_revisions(payload["scope_id"]) == [first_revision]
        reopened.add_revision_to_corpus(corpus_id, second_revision)
        assert reopened.scope_revisions(payload["scope_id"]) == [first_revision]
    finally:
        reopened.close()


def test_corpus_snapshot_supports_human_text_output(tmp_path):
    root = tmp_path / "corpus-snapshot-text-project"
    project = Project.init(root)
    _, revision = project.import_text("source.txt", "Source text.")
    corpus_id = project.create_corpus("Research corpus")
    project.add_revision_to_corpus(corpus_id, revision)
    project.close()

    result = CliRunner().invoke(
        app, ["corpus", "snapshot", str(root), corpus_id, "--format", "text"]
    )

    assert result.exit_code == 0, result.output
    assert "Scope snapshot: " in result.output
    assert f"Corpus: {corpus_id}" in result.output
    assert "Revisions: 1" in result.output
    assert f"  {revision}" in result.output


def test_corpus_snapshot_rejects_unsupported_format(tmp_path):
    root = tmp_path / "corpus-snapshot-format-project"
    Project.init(root).close()

    result = CliRunner().invoke(
        app, ["corpus", "snapshot", str(root), "unknown", "--format", "csv"]
    )

    assert result.exit_code == 2
    assert "format must be json or text" in result.output
