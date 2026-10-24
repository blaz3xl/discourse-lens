from __future__ import annotations

import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def test_snapshot_subcorpus_preserves_saved_membership_for_scoped_cli_use(tmp_path):
    root = tmp_path / "snapshot-subcorpus-project"
    project = Project.init(root)
    _, selected_revision = project.import_text("selected.txt", "Selected source.")
    _, later_parent_revision = project.import_text("later-parent.txt", "Later parent source.")
    _, after_snapshot_revision = project.import_text("after-snapshot.txt", "After snapshot source.")

    corpus_id = project.create_corpus("research corpus")
    project.add_revision_to_corpus(corpus_id, selected_revision)
    subcorpus_id = project.create_subcorpus(
        corpus_id, "selected subset", [selected_revision]
    )
    # New parent members do not become members of the saved subcorpus.
    project.add_revision_to_corpus(corpus_id, later_parent_revision)
    project.close()

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["corpus", "snapshot-subcorpus", str(root), subcorpus_id],
    )

    assert result.exit_code == 0, result.output
    snapshot = json.loads(result.output)
    assert snapshot == {
        "subcorpus_id": subcorpus_id,
        "scope_id": snapshot["scope_id"],
        "revision_count": 1,
        "revision_ids": [selected_revision],
    }
    assert snapshot["scope_id"]

    # The ordinary bounded scope reader can consume the returned ID, and later
    # additions to the parent corpus cannot change that frozen ScopeSnapshot.
    project = Project.open(root)
    project.add_revision_to_corpus(corpus_id, after_snapshot_revision)
    project.close()
    scope_page = runner.invoke(
        app,
        [
            "corpus",
            "scope-revisions",
            str(root),
            snapshot["scope_id"],
            "--offset",
            "0",
            "--limit",
            "10",
        ],
    )

    assert scope_page.exit_code == 0, scope_page.output
    assert json.loads(scope_page.output) == {
        "scope_id": snapshot["scope_id"],
        "offset": 0,
        "limit": 10,
        "revision_count": 1,
        "revision_ids": [selected_revision],
        "has_more": False,
    }


def test_snapshot_subcorpus_rejects_empty_and_whitespace_ids(tmp_path):
    root = tmp_path / "snapshot-subcorpus-invalid-project"
    Project.init(root).close()
    runner = CliRunner()

    for subcorpus_id in ("", "   ", " id "):
        result = runner.invoke(
            app,
            ["corpus", "snapshot-subcorpus", str(root), subcorpus_id],
        )

        assert result.exit_code == 2
        assert "subcorpus ID" in result.output


def test_snapshot_subcorpus_rejects_unknown_id(tmp_path):
    root = tmp_path / "snapshot-subcorpus-unknown-project"
    Project.init(root).close()

    result = CliRunner().invoke(
        app,
        ["corpus", "snapshot-subcorpus", str(root), "unknown-subcorpus"],
    )

    assert result.exit_code == 2
    assert "empty or unknown" in result.output.lower()
    assert str(root) not in result.output
