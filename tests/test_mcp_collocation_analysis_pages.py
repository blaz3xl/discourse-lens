from __future__ import annotations

import asyncio

import pytest

from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.mcp_server import corpus_collocation_analysis, server


def test_mcp_collocation_analysis_schema_exposes_offset_paging():
    tool = next(
        item
        for item in asyncio.run(server.list_tools())
        if item.name == "corpus_collocation_analysis"
    )
    tool_schema = tool.model_dump(by_alias=True)
    schema = tool_schema["inputSchema"]
    response_schema = tool_schema["outputSchema"]

    assert {"revision_ids", "offset", "limit"} <= set(schema["properties"])
    assert schema["properties"]["offset"]["default"] == 0
    assert {"offset", "limit", "total", "has_more"} <= set(response_schema["properties"])


def test_mcp_collocation_analysis_pages_match_sdk_and_keep_scope(tmp_path, monkeypatch):
    root = tmp_path / "mcp-collocation-analysis-pages"
    project = Project.init(root)
    _, selected = project.import_text(
        "selected.txt", "one target two three target four five target"
    )
    project.import_text("outside.txt", "target outsider")
    scope_id = project.snapshot([selected])
    core_pages = [
        project.corpus_collocation_analysis(
            scope_id,
            "target",
            window=1,
            min_frequency=1,
            offset=offset,
            limit=2,
        )
        for offset in (0, 2, 4)
    ]
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    mcp_pages = [
        corpus_collocation_analysis(
            [selected],
            "target",
            window=1,
            min_frequency=1,
            limit=2,
            offset=offset,
        )
        for offset in (0, 2, 4)
    ]

    assert [
        {key: value for key, value in page.items() if key != "scope_id"} for page in mcp_pages
    ] == [{key: value for key, value in page.items() if key != "scope_id"} for page in core_pages]
    assert [page["offset"] for page in mcp_pages] == [0, 2, 4]
    assert all(page["limit"] == 2 and page["total"] == 5 for page in mcp_pages)
    assert [page["has_more"] for page in mcp_pages] == [True, True, False]
    assert [page["truncated"] for page in mcp_pages] == [True, True, False]
    assert [row["term"] for page in mcp_pages for row in page["rows"]] == [
        "five",
        "four",
        "one",
        "three",
        "two",
    ]
    assert all(row["term"] != "outsider" for page in mcp_pages for row in page["rows"])
    with Project.open(root) as reopened:
        assert reopened.scope_revisions(str(mcp_pages[0]["scope_id"])) == [selected]


@pytest.mark.parametrize("offset", [-1, True])
def test_mcp_collocation_analysis_rejects_invalid_offset(tmp_path, monkeypatch, offset):
    root = tmp_path / f"mcp-collocation-analysis-invalid-offset-{offset}"
    project = Project.init(root)
    _, revision = project.import_text("text.txt", "target alpha")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ProjectError, match="offset"):
        corpus_collocation_analysis([revision], "target", min_frequency=1, offset=offset)
