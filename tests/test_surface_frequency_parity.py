import json

import pytest
from typer.testing import CliRunner

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.cli import app


def _page_without_scope_identity(page):
    return {key: value for key, value in page.items() if key != "scope_id"}


@pytest.mark.parametrize(
    "field,command",
    [
        ("normalized", "normalized-frequency"),
        ("document_frequency", "document-frequency"),
    ],
)
@pytest.mark.parametrize("offset", [0, 2])
def test_surface_frequency_page_matches_sdk_cli_and_mcp(
    tmp_path, monkeypatch, field, command, offset
):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    root = tmp_path / "surface-frequency-parity"
    project = Project.init(root)
    _, revision_id = project.import_text(
        "sample.txt", "alpha beta alpha gamma cafe\u0301 café"
    )
    _, outside_revision_id = project.import_text("outside.txt", "outside outside")
    artifact = project.build_surface_token_artifact(revision_id)
    project.build_surface_token_artifact(outside_revision_id)
    scope_id = project.snapshot([revision_id])
    sdk_page = project.corpus_surface_frequency_page(
        scope_id,
        [artifact.artifact_id],
        field=field,
        offset=offset,
        limit=2,
    )
    inline_counts = project.corpus_frequency(scope_id)
    project.close()

    cli_result = CliRunner().invoke(
        app,
        [
            "corpus",
            command,
            str(root),
            revision_id,
            artifact.artifact_id,
            "--format",
            "json",
            "--offset",
            str(offset),
            "--limit",
            "2",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_page = json.loads(cli_result.output)

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    mcp_page = (
        mcp_module.corpus_normalized_frequency(
            [revision_id], [artifact.artifact_id], offset=offset, limit=2
        )
        if field == "normalized"
        else mcp_module.corpus_document_frequency(
            [revision_id], [artifact.artifact_id], offset=offset, limit=2
        )
    )

    assert _page_without_scope_identity(cli_page) == _page_without_scope_identity(sdk_page)
    assert _page_without_scope_identity(mcp_page) == _page_without_scope_identity(sdk_page)
    assert sdk_page["revision_ids"] == [revision_id]
    assert sdk_page["artifact_ids"] == [artifact.artifact_id]
    assert sdk_page["field"] == field
    assert sdk_page["offset"] == offset
    assert sdk_page["limit"] == 2
    assert sdk_page["total_values"] == 4
    assert sdk_page["has_more"] is (offset == 0)
    assert sdk_page["tokenizer"] == "unicode-surface-tokenizer-v2"
    assert sdk_page["normalization"] == "NFC-after-casefold-v1"
    all_rows = sorted(
        (
            (token, count if field == "normalized" else 1)
            for token, count in inline_counts.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if field == "document_frequency":
        all_rows = sorted(((token, 1) for token in inline_counts), key=lambda item: item[0])
    expected_rows = all_rows[offset : offset + 2]
    assert [(row["token"], row["count"]) for row in sdk_page["rows"]] == expected_rows
    assert inline_counts["café"] == 2
