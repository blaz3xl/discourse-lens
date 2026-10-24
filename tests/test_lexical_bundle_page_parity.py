import json

from typer.testing import CliRunner

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.cli import app


def _without_scope_id(result):
    return {key: value for key, value in result.items() if key != "scope_id"}


def test_lexical_bundle_pages_match_sdk_cli_and_mcp(tmp_path, monkeypatch):
    root = tmp_path / "lexical-bundle-page-parity"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "we must speak we must listen")
    _, second_revision = project.import_text("second.txt", "they may act we must speak")
    project.import_text("outside.txt", "outsider phrase outsider phrase")
    revision_ids = [first_revision, second_revision]
    scope_id = project.snapshot(revision_ids)

    sdk_pages = [
        project.corpus_lexical_bundles(
            scope_id,
            size=2,
            min_frequency=1,
            min_document_frequency=1,
            offset=offset,
            limit=2,
        )
        for offset in (0, 2)
    ]
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    for offset, sdk_page in zip((0, 2), sdk_pages, strict=True):
        cli_result = CliRunner().invoke(
            app,
            [
                "corpus",
                "lexical-bundles",
                str(root),
                ",".join(revision_ids),
                "--size",
                "2",
                "--min-frequency",
                "1",
                "--min-documents",
                "1",
                "--offset",
                str(offset),
                "--limit",
                "2",
                "--format",
                "json",
            ],
        )
        assert cli_result.exit_code == 0, cli_result.output
        cli_page = json.loads(cli_result.output)
        mcp_page = mcp_module.corpus_lexical_bundles(
            revision_ids,
            size=2,
            min_frequency=1,
            min_document_frequency=1,
            offset=offset,
            limit=2,
        )

        assert _without_scope_id(cli_page) == _without_scope_id(sdk_page)
        assert _without_scope_id(mcp_page) == _without_scope_id(sdk_page)
        assert sdk_page["offset"] == offset
        assert sdk_page["limit"] == 2
        assert len(sdk_page["rows"]) <= 2
        assert sdk_page["has_more"] is (
            offset + len(sdk_page["rows"]) < sdk_page["candidate_count"]
        )
        assert sdk_page["truncated"] is sdk_page["has_more"]
        assert all("outsider" not in row["bundle"] for row in sdk_page["rows"])
