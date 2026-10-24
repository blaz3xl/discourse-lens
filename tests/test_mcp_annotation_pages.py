from __future__ import annotations

import asyncio
from typing import cast

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.mcp_server import list_annotations


def _create_annotations(root):
    project = Project.init(root)
    _, revision = project.import_text("selected.txt", "alpha beta gamma delta")
    _, outside_revision = project.import_text("outside.txt", "outside annotation")
    codebook = project.create_codebook("analysis")
    code = project.create_code(codebook, "observation")
    annotation_ids = []
    for start, end in ((0, 5), (6, 10), (11, 16)):
        span = project.create_span(revision, start, end)
        memo = "M" * 4_500 if start == 0 else f"Selected note {start}."
        annotation_ids.append(project.annotate_span(code, span, memo))
    outside_span = project.create_span(outside_revision, 0, 7)
    project.annotate_span(code, outside_span, "Out-of-scope note.")
    first_page = project.annotation_pages_for_revisions([revision], offset=0, limit=2)
    second_page = project.annotation_pages_for_revisions([revision], offset=2, limit=2)
    project.close()
    return revision, annotation_ids, first_page, second_page


def test_mcp_annotations_delegate_bounded_pages_without_writes(tmp_path, monkeypatch):
    root = tmp_path / "mcp-annotation-pages"
    revision, annotation_ids, expected_first, expected_second = _create_annotations(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    first_page = list_annotations([revision], offset=0, limit=2)
    second_page = list_annotations([revision], offset=2, limit=2)

    for page, expected in ((first_page, expected_first), (second_page, expected_second)):
        for key in ("revision_count", "offset", "limit", "total", "has_more"):
            assert page[key] == expected[key]
    assert first_page["revision_count"] == 1
    assert first_page["offset"] == 0
    assert first_page["limit"] == 2
    assert first_page["total"] == second_page["total"] == 3
    assert first_page["has_more"] is True
    assert second_page["has_more"] is False
    first_rows = cast(list[dict[str, object]], first_page["items"])
    second_rows = cast(list[dict[str, object]], second_page["items"])
    assert [row["id"] for row in first_rows + second_rows] == annotation_ids
    assert len(first_rows) == 2
    assert len(second_rows) == 1
    expected_first_rows = cast(list[dict[str, object]], expected_first["items"])
    expected_second_rows = cast(list[dict[str, object]], expected_second["items"])
    for actual_rows, expected_rows in (
        (first_rows, expected_first_rows),
        (second_rows, expected_second_rows),
    ):
        for actual, expected in zip(actual_rows, expected_rows, strict=True):
            for key, value in expected.items():
                if key == "memo":
                    expected_memo = cast(str, value)
                    assert actual[key] == expected_memo[: mcp_module.MAX_FINDING_RESOURCE_TEXT_CHARS]
                else:
                    assert actual[key] == value
    assert len(cast(str, first_rows[0]["memo"])) == mcp_module.MAX_FINDING_RESOURCE_TEXT_CHARS
    assert first_page["truncated_fields"] == ["items[0].memo"]
    assert database.read_bytes() == before


def test_mcp_annotations_are_discoverable_in_read_only_mode():
    tool = next(
        item
        for item in asyncio.run(mcp_module.server.list_tools())
        if item.name == "list_annotations"
    )
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    assert {"revision_ids", "offset", "limit"} <= set(schema["properties"])

    asyncio.run(mcp_module._configure_read_only_server())
    try:
        read_only_tools = {
            item.name for item in asyncio.run(mcp_module.server.list_tools())
        }
    finally:
        mcp_module._restore_full_server_registry()
    assert "list_annotations" in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    assert "list_annotations" in read_only_tools


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(-1, 1), (0, 0), (0, 101), (9_223_372_036_854_775_808, 1)],
)
def test_mcp_annotations_reject_invalid_page_bounds(offset, limit):
    with pytest.raises(ProjectError, match="annotation page"):
        list_annotations(["revision-id"], offset=offset, limit=limit)
