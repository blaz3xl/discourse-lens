from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.mcp_server import list_documents


def _create_document_pages(root):
    project = Project.init(root)
    first_document, _ = project.import_text("alpha.txt", "Private canonical text one.")
    project.import_text("beta.txt", "Private canonical text two.")
    project.import_text("gamma.txt", "Private canonical text three.")
    project.import_text(
        "alpha-revised.txt",
        "Private canonical text four.",
        document_id=first_document,
    )
    expected_first = project.documents_page(offset=0, limit=2)
    expected_second = project.documents_page(offset=2, limit=2)
    project.close()
    return expected_first, expected_second


def _assert_metadata_only(value):
    forbidden_keys = {
        "text",
        "canonical_text",
        "text_path",
        "original_path",
        "absolute_path",
        "path",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            assert str(key).casefold() not in forbidden_keys
            _assert_metadata_only(item)
    elif isinstance(value, list):
        for item in value:
            _assert_metadata_only(item)
    elif isinstance(value, str):
        assert not Path(value).is_absolute()


def test_mcp_documents_delegate_bounded_metadata_pages_read_only(tmp_path, monkeypatch):
    root = tmp_path / "mcp-document-pages"
    expected_first, expected_second = _create_document_pages(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    first_page = list_documents(offset=0, limit=2)
    second_page = list_documents(offset=2, limit=2)
    default_page = list_documents()

    assert first_page == expected_first
    assert second_page == expected_second
    assert first_page["offset"] == 0
    assert first_page["limit"] == 2
    assert first_page["total"] == second_page["total"] == 3
    assert first_page["has_more"] is True
    assert second_page["has_more"] is False
    assert default_page["offset"] == 0
    assert default_page["limit"] == 100
    assert default_page["total"] == 3
    assert default_page["has_more"] is False
    for page in (first_page, second_page, default_page):
        _assert_metadata_only(page)
    first_items = cast(list[dict[str, object]], first_page["items"])
    assert set(first_items[0]) == {
        "document_id",
        "display_name",
        "created_at",
        "revision_count",
        "latest_revision_id",
        "latest_revision_created_at",
        "latest_extraction_status",
        "latest_mime_type",
    }
    assert first_items[0]["revision_count"] == 2
    assert database.read_bytes() == before

    tool = next(
        item
        for item in asyncio.run(mcp_module.server.list_tools())
        if item.name == "list_documents"
    )
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    assert {"offset", "limit"} <= set(schema["properties"])

    asyncio.run(mcp_module._configure_read_only_server())
    try:
        read_only_tools = {
            item.name for item in asyncio.run(mcp_module.server.list_tools())
        }
    finally:
        mcp_module._restore_full_server_registry()
    assert "list_documents" in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    assert "list_documents" in read_only_tools


@pytest.mark.parametrize(
    ("offset", "limit"),
    [
        (-1, 1),
        (True, 1),
        (0, 0),
        (0, True),
        (0, 101),
        (9_223_372_036_854_775_808, 1),
    ],
)
def test_mcp_documents_reject_invalid_page_bounds(offset, limit):
    with pytest.raises(ProjectError, match="documents page"):
        list_documents(offset=offset, limit=limit)
