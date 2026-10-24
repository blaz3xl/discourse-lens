from __future__ import annotations

import asyncio

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.mcp_server import list_corpora, list_subcorpora


def _create_collection_pages(root):
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "First source text.")
    _, second_revision = project.import_text("second.txt", "Second source text.")
    _, third_revision = project.import_text("third.txt", "Third source text.")

    corpora = []
    for name, revision in (
        ("Alpha", first_revision),
        ("Beta", second_revision),
        ("Gamma", third_revision),
    ):
        corpus_id = project.create_corpus(name, f"Description for {name}.")
        project.add_revision_to_corpus(corpus_id, revision)
        corpora.append(corpus_id)

    for parent, name, revision in (
        (corpora[0], "Alpha one", first_revision),
        (corpora[0], "Alpha two", first_revision),
        (corpora[1], "Beta one", second_revision),
    ):
        project.create_subcorpus(parent, name, [revision])

    expected_corpora_first = project.corpora_page(offset=0, limit=2)
    expected_corpora_second = project.corpora_page(offset=2, limit=2)
    expected_subcorpora_first = project.subcorpora_page(offset=0, limit=2)
    expected_subcorpora_second = project.subcorpora_page(offset=2, limit=2)
    expected_filtered_subcorpora = project.subcorpora_page(
        corpora[0], offset=0, limit=2
    )
    project.close()
    return (
        corpora[0],
        expected_corpora_first,
        expected_corpora_second,
        expected_subcorpora_first,
        expected_subcorpora_second,
        expected_filtered_subcorpora,
    )


def test_mcp_collection_tools_delegate_bounded_read_only_pages(tmp_path, monkeypatch):
    root = tmp_path / "mcp-collection-pages"
    (
        parent_corpus_id,
        expected_corpora_first,
        expected_corpora_second,
        expected_subcorpora_first,
        expected_subcorpora_second,
        expected_filtered_subcorpora,
    ) = _create_collection_pages(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    corpora_first = list_corpora(offset=0, limit=2)
    corpora_second = list_corpora(offset=2, limit=2)
    corpora_default = list_corpora()
    subcorpora_first = list_subcorpora(offset=0, limit=2)
    subcorpora_second = list_subcorpora(offset=2, limit=2)
    subcorpora_default = list_subcorpora()
    filtered_subcorpora = list_subcorpora(parent_corpus_id, offset=0, limit=2)

    assert corpora_first == expected_corpora_first
    assert corpora_second == expected_corpora_second
    assert subcorpora_first == expected_subcorpora_first
    assert subcorpora_second == expected_subcorpora_second
    assert filtered_subcorpora == expected_filtered_subcorpora
    assert corpora_first["offset"] == subcorpora_first["offset"] == 0
    assert corpora_first["limit"] == subcorpora_first["limit"] == 2
    assert corpora_first["total"] == corpora_second["total"] == 3
    assert subcorpora_first["total"] == subcorpora_second["total"] == 3
    assert filtered_subcorpora["total"] == 2
    assert corpora_first["has_more"] is True
    assert corpora_second["has_more"] is False
    assert corpora_default["offset"] == subcorpora_default["offset"] == 0
    assert corpora_default["limit"] == subcorpora_default["limit"] == 100
    assert corpora_default["has_more"] is subcorpora_default["has_more"] is False
    assert database.read_bytes() == before

    tools = asyncio.run(mcp_module.server.list_tools())
    schemas = {
        tool.name: tool.model_dump(by_alias=True)["inputSchema"]
        for tool in tools
        if tool.name in {"list_corpora", "list_subcorpora"}
    }
    assert {"offset", "limit"} <= set(schemas["list_corpora"]["properties"])
    assert {"corpus_id", "offset", "limit"} <= set(
        schemas["list_subcorpora"]["properties"]
    )

    asyncio.run(mcp_module._configure_read_only_server())
    try:
        read_only_tools = {
            tool.name for tool in asyncio.run(mcp_module.server.list_tools())
        }
    finally:
        mcp_module._restore_full_server_registry()
    assert {"list_corpora", "list_subcorpora"} <= mcp_module.READ_ONLY_MCP_TOOL_NAMES
    assert {"list_corpora", "list_subcorpora"} <= read_only_tools


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
def test_mcp_corpora_page_rejects_invalid_bounds(offset, limit):
    with pytest.raises(ProjectError, match="corpora page"):
        list_corpora(offset=offset, limit=limit)


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
def test_mcp_subcorpora_page_rejects_invalid_bounds(offset, limit):
    with pytest.raises(ProjectError, match="subcorpora page"):
        list_subcorpora(offset=offset, limit=limit)
