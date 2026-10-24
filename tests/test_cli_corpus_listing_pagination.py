from __future__ import annotations

import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def _create_corpora(root):
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "First corpus document.")
    _, second_revision = project.import_text("second.txt", "Second corpus document.")
    _, third_revision = project.import_text("third.txt", "Other corpus document.")

    first_corpus = project.create_corpus("Corpus A")
    second_corpus = project.create_corpus("Corpus B")
    project.add_revision_to_corpus(first_corpus, first_revision)
    project.add_revision_to_corpus(first_corpus, second_revision)
    project.add_revision_to_corpus(second_corpus, third_revision)
    project.create_subcorpus(first_corpus, "Subcorpus A", [first_revision])
    project.create_subcorpus(first_corpus, "Subcorpus B", [second_revision])
    project.create_subcorpus(second_corpus, "Subcorpus C", [third_revision])
    project.close()
    return first_corpus, second_corpus


def test_corpus_list_keeps_legacy_array_and_supports_pages(tmp_path):
    root = tmp_path / "corpus-list-project"
    _create_corpora(root)
    runner = CliRunner()

    legacy = runner.invoke(app, ["corpus", "list", str(root)])
    page = runner.invoke(
        app,
        ["corpus", "list", str(root), "--offset", "1", "--limit", "1"],
    )
    limit_only = runner.invoke(app, ["corpus", "list", str(root), "--limit", "1"])
    offset_only = runner.invoke(app, ["corpus", "list", str(root), "--offset", "1"])

    assert legacy.exit_code == 0, legacy.output
    assert page.exit_code == 0, page.output
    assert limit_only.exit_code == 0, limit_only.output
    assert offset_only.exit_code == 0, offset_only.output
    legacy_rows = json.loads(legacy.output)
    assert len(legacy_rows) == 2

    payload = json.loads(page.output)
    assert payload["items"] == legacy_rows[1:2]
    assert payload["offset"] == 1
    assert payload["limit"] == 1
    assert payload["total"] == 2
    assert payload["has_more"] is False

    first_page = json.loads(limit_only.output)
    assert first_page["items"] == legacy_rows[:1]
    assert first_page["offset"] == 0
    assert first_page["limit"] == 1
    assert first_page["total"] == 2
    assert first_page["has_more"] is True

    default_limit_page = json.loads(offset_only.output)
    assert default_limit_page["offset"] == 1
    assert default_limit_page["limit"] == 100
    assert default_limit_page["total"] == 2
    assert default_limit_page["items"] == legacy_rows[1:]


def test_list_subcorpora_preserves_filter_for_legacy_and_paged_output(tmp_path):
    root = tmp_path / "subcorpus-list-project"
    first_corpus, _ = _create_corpora(root)
    runner = CliRunner()

    legacy = runner.invoke(app, ["corpus", "list-subcorpora", str(root)])
    filtered = runner.invoke(
        app,
        ["corpus", "list-subcorpora", str(root), "--corpus", first_corpus],
    )
    page = runner.invoke(
        app,
        [
            "corpus",
            "list-subcorpora",
            str(root),
            "--corpus",
            first_corpus,
            "--offset",
            "1",
            "--limit",
            "1",
        ],
    )
    offset_only = runner.invoke(
        app,
        ["corpus", "list-subcorpora", str(root), "--corpus", first_corpus, "--offset", "1"],
    )

    assert legacy.exit_code == 0, legacy.output
    assert filtered.exit_code == 0, filtered.output
    assert page.exit_code == 0, page.output
    assert offset_only.exit_code == 0, offset_only.output
    assert len(json.loads(legacy.output)) == 3
    filtered_rows = json.loads(filtered.output)
    assert len(filtered_rows) == 2
    assert all(row["corpus_id"] == first_corpus for row in filtered_rows)

    payload = json.loads(page.output)
    assert payload["items"] == filtered_rows[1:2]
    assert payload["offset"] == 1
    assert payload["limit"] == 1
    assert payload["total"] == 2
    assert payload["has_more"] is False

    default_limit_page = json.loads(offset_only.output)
    assert default_limit_page["items"] == filtered_rows[1:]
    assert default_limit_page["offset"] == 1
    assert default_limit_page["limit"] == 100
    assert default_limit_page["total"] == 2


def test_corpus_listing_pagination_rejects_invalid_cli_bounds(tmp_path):
    root = tmp_path / "corpus-list-invalid-bounds"
    _create_corpora(root)
    runner = CliRunner()

    invalid_offset = runner.invoke(app, ["corpus", "list", str(root), "--offset", "-1"])
    zero_limit = runner.invoke(app, ["corpus", "list", str(root), "--limit", "0"])
    excessive_limit = runner.invoke(
        app, ["corpus", "list-subcorpora", str(root), "--limit", "1001"]
    )

    assert invalid_offset.exit_code == 2
    assert zero_limit.exit_code == 2
    assert excessive_limit.exit_code == 2
