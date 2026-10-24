from __future__ import annotations

import asyncio
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.mcp_server import list_codebooks, list_codes


class CodebookItem(TypedDict):
    id: str
    name: str
    description: str | None
    created_at: str


class CodeItem(TypedDict):
    id: str
    codebook_id: str
    parent_code_id: str | None
    label: str
    definition: str | None
    created_at: str


class CodebookPage(TypedDict):
    items: list[CodebookItem]
    offset: int
    limit: int
    total: int
    has_more: bool
    truncated_fields: NotRequired[list[str]]


class CodePage(TypedDict):
    items: list[CodeItem]
    offset: int
    limit: int
    total: int
    has_more: bool
    truncated_fields: NotRequired[list[str]]


def _create_codebook_pages(
    root: Path,
) -> tuple[str, str, str, CodebookPage, CodebookPage, CodePage, CodePage]:
    project = Project.init(root)
    description = "d" * (mcp_module.MAX_FINDING_RESOURCE_TEXT_CHARS + 25)
    definition = "f" * (mcp_module.MAX_FINDING_RESOURCE_TEXT_CHARS + 25)
    first_codebook = project.create_codebook("Alpha", description)
    project.create_codebook("Beta", "Second codebook.")
    parent_code = project.create_code(first_codebook, "A parent", definition=definition)
    child_code = project.create_code(
        first_codebook, "B child", definition="Child definition.", parent_code_id=parent_code
    )
    unused_code = project.create_code(first_codebook, "C unused")
    first_codebooks = cast(CodebookPage, project.codebook_pages(offset=0, limit=1))
    second_codebooks = cast(CodebookPage, project.codebook_pages(offset=1, limit=1))
    first_codes = cast(CodePage, project.codes_for_codebook_page(first_codebook, offset=0, limit=2))
    second_codes = cast(
        CodePage, project.codes_for_codebook_page(first_codebook, offset=2, limit=2)
    )
    project.close()
    return (
        first_codebook,
        child_code,
        unused_code,
        first_codebooks,
        second_codebooks,
        first_codes,
        second_codes,
    )


def test_mcp_codebook_tools_delegate_bounded_pages_and_are_discoverable_read_only(
    tmp_path, monkeypatch
):
    root = tmp_path / "mcp-codebook-pages"
    (
        first_codebook,
        child_code,
        unused_code,
        expected_codebooks_first,
        expected_codebooks_second,
        expected_codes_first,
        expected_codes_second,
    ) = _create_codebook_pages(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    codebooks_first = cast(CodebookPage, list_codebooks(offset=0, limit=1))
    codebooks_second = cast(CodebookPage, list_codebooks(offset=1, limit=1))
    codes_first = cast(CodePage, list_codes(first_codebook, offset=0, limit=2))
    codes_second = cast(CodePage, list_codes(first_codebook, offset=2, limit=2))
    default_codebooks = cast(CodebookPage, list_codebooks())
    default_codes = cast(CodePage, list_codes(first_codebook))

    assert codebooks_first["items"][0]["id"] == expected_codebooks_first["items"][0]["id"]
    assert codebooks_first["offset"] == 0
    assert codebooks_first["limit"] == 1
    assert codebooks_first["total"] == codebooks_second["total"] == 2
    assert codebooks_first["has_more"] is True
    assert codebooks_second["has_more"] is False
    assert codebooks_first["items"][0]["id"] == expected_codebooks_first["items"][0]["id"]
    assert codebooks_second["items"][0]["id"] == expected_codebooks_second["items"][0]["id"]
    assert "truncated_fields" in codebooks_first
    description = codebooks_first["items"][0]["description"]
    assert isinstance(description, str)
    assert len(description) == mcp_module.MAX_FINDING_RESOURCE_TEXT_CHARS
    assert codes_first["offset"] == 0
    assert codes_first["limit"] == 2
    assert codes_first["total"] == codes_second["total"] == 3
    assert codes_first["has_more"] is True
    assert codes_second["has_more"] is False
    assert [item["id"] for item in codes_first["items"]] == [
        item["id"] for item in expected_codes_first["items"]
    ]
    assert [item["id"] for item in codes_second["items"]] == [
        item["id"] for item in expected_codes_second["items"]
    ]
    listed_codes = codes_first["items"] + codes_second["items"]
    child = next(item for item in listed_codes if item["id"] == child_code)
    assert child["parent_code_id"] is not None
    assert any(item["id"] == unused_code for item in listed_codes)
    assert all("parent_code_id" in item for item in listed_codes)
    assert "truncated_fields" in codes_first
    parent = next(item for item in codes_first["items"] if item["label"] == "A parent")
    parent_definition = parent["definition"]
    assert isinstance(parent_definition, str)
    assert len(parent_definition) == mcp_module.MAX_FINDING_RESOURCE_TEXT_CHARS
    assert default_codebooks["offset"] == default_codes["offset"] == 0
    assert default_codebooks["limit"] == default_codes["limit"] == 100
    assert database.read_bytes() == before

    tools = asyncio.run(mcp_module.server.list_tools())
    schemas = {
        tool.name: tool.model_dump(by_alias=True)["inputSchema"]
        for tool in tools
        if tool.name in {"list_codebooks", "list_codes"}
    }
    assert set(schemas) == {"list_codebooks", "list_codes"}
    for schema in schemas.values():
        properties = schema["properties"]
        assert properties["offset"]["minimum"] == 0
        assert properties["offset"]["maximum"] == 0x7FFF_FFFF_FFFF_FFFF
        assert properties["limit"]["minimum"] == 1
        assert properties["limit"]["maximum"] == 100
    assert schemas["list_codes"]["required"] == ["codebook_id"]
    assert schemas["list_codes"]["properties"]["codebook_id"]["maxLength"] == 128

    asyncio.run(mcp_module._configure_read_only_server())
    try:
        read_only_tools = {tool.name for tool in asyncio.run(mcp_module.server.list_tools())}
    finally:
        mcp_module._restore_full_server_registry()
    assert {"list_codebooks", "list_codes"} <= mcp_module.READ_ONLY_MCP_TOOL_NAMES
    assert {"list_codebooks", "list_codes"} <= read_only_tools


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
def test_mcp_codebook_page_rejects_invalid_bounds(offset, limit):
    with pytest.raises(ProjectError, match="codebooks page"):
        list_codebooks(offset=offset, limit=limit)


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
def test_mcp_code_page_rejects_invalid_bounds(offset, limit):
    with pytest.raises(ProjectError, match="codes page"):
        list_codes("existing-but-unopened", offset=offset, limit=limit)


def test_mcp_code_page_rejects_unknown_codebook(tmp_path, monkeypatch):
    root = tmp_path / "mcp-codebook-unknown"
    Project.init(root).close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    with pytest.raises(ProjectError, match="unknown codebook"):
        list_codes("missing-codebook")
