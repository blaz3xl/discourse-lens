from __future__ import annotations

import asyncio

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.mcp_server import corpus_lexical_bundles, server


def test_mcp_lexical_bundle_schema_discovers_offset_paging() -> None:
    tools = asyncio.run(server.list_tools())
    tool = next(item for item in tools if item.name == "corpus_lexical_bundles")
    schema = tool.model_dump(by_alias=True)["inputSchema"]

    assert {"revision_ids", "offset", "limit"} <= set(schema["properties"])
    assert schema["properties"]["offset"]["default"] == 0
    assert "corpus_lexical_bundles" not in mcp_module.READ_ONLY_MCP_TOOL_NAMES


def test_mcp_lexical_bundles_pages_only_the_explicit_revision_scope(tmp_path, monkeypatch):
    root = tmp_path / "research"
    project = Project.init(root)
    _, selected = project.import_text("selected.txt", "alpha beta gamma delta epsilon")
    project.import_text("outside.txt", "zeta eta theta")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    first_page = corpus_lexical_bundles(
        [selected],
        size=2,
        min_frequency=1,
        min_document_frequency=1,
        offset=0,
        limit=2,
    )
    second_page = corpus_lexical_bundles(
        [selected],
        size=2,
        min_frequency=1,
        min_document_frequency=1,
        offset=2,
        limit=2,
    )

    first_rows = first_page["rows"]
    second_rows = second_page["rows"]
    assert first_page["candidate_count"] == second_page["candidate_count"] == 4
    assert first_page["offset"] == 0
    assert first_page["limit"] == 2
    assert first_page["has_more"] is first_page["truncated"] is True
    assert second_page["offset"] == 2
    assert second_page["limit"] == 2
    assert second_page["has_more"] is second_page["truncated"] is False
    assert len(first_rows) == len(second_rows) == 2
    assert {row["bundle"] for row in first_rows + second_rows} == {
        "alpha beta",
        "beta gamma",
        "gamma delta",
        "delta epsilon",
    }
    assert not ({row["bundle"] for row in first_rows} & {row["bundle"] for row in second_rows})

    with Project.open(root) as reopened:
        assert reopened.scope_revisions(first_page["scope_id"]) == [selected]


@pytest.mark.parametrize(
    "arguments",
    [
        {"offset": True},
        {"offset": -1},
        {"limit": True},
        {"limit": 0},
        {"limit": 1_001},
    ],
)
def test_mcp_lexical_bundle_pagination_rejects_invalid_values_before_opening_project(
    monkeypatch, arguments
) -> None:
    def unexpected_open():
        raise AssertionError("invalid paging must be rejected before opening the project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_open)

    with pytest.raises(ProjectError):
        corpus_lexical_bundles(["revision-id"], **arguments)


def test_mcp_lexical_bundle_page_rejects_oversized_serialized_response(monkeypatch) -> None:
    class OversizedProject:
        def snapshot(self, revision_ids: list[str]) -> str:
            assert revision_ids == ["revision-id"]
            return "scope-id"

        def corpus_lexical_bundles(self, scope_id: str, **kwargs: object):
            assert scope_id == "scope-id"
            return {
                "scope_id": scope_id,
                "calculation_version": "surface-lexical-bundles-v1",
                "size": 2,
                "min_frequency": 1,
                "min_document_frequency": 1,
                "candidate_count": 1,
                "offset": kwargs["offset"],
                "limit": kwargs["limit"],
                "has_more": False,
                "rows": [
                    {
                        "bundle": "x" * mcp_module.MAX_MCP_LEXICAL_BUNDLE_PAGE_BYTES,
                        "frequency": 1,
                        "document_frequency": 1,
                    }
                ],
                "truncated": False,
            }

        def close(self) -> None:
            pass

    monkeypatch.setattr(mcp_module, "_open_project", OversizedProject)

    with pytest.raises(ProjectError, match="64 KiB"):
        corpus_lexical_bundles(["revision-id"], size=2, min_frequency=1)
