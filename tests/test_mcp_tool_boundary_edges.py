from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.types import CallToolRequestParams

import discourselens.mcp_server as mcp_module
from discourselens import Project


def test_controlled_profile_rejects_boolean_reranking_count_at_tool_boundary(monkeypatch):
    monkeypatch.delenv("DISCOURSELENS_PROJECT", raising=False)
    profile = {
        "frameworks": ["critical-reviewer"],
        "reranking_provider_id": "fixture-provider",
        "reranking_credential_alias": "fixture-credential",
        "reranking_model": "fixture-model",
        "reranking_top_n": True,
    }

    async def invoke() -> None:
        with pytest.raises(ToolError) as captured:
            await mcp_module.server.call_tool(
                "save_controlled_analysis_profile",
                {"name": "fixture-profile", "profile": profile},
            )
        assert not isinstance(captured.value, UnexpectedToolError)

    asyncio.run(invoke())


def test_research_memo_listing_rejects_overlarge_page_before_project_open(monkeypatch):
    monkeypatch.delenv("DISCOURSELENS_PROJECT", raising=False)

    async def invoke() -> None:
        with pytest.raises(ToolError) as captured:
            await mcp_module.server.call_tool(
                "list_research_memos",
                {"scope_id": "fixture-scope", "limit": mcp_module.MAX_RESEARCH_MEMO_PAGE_SIZE + 1},
            )
        assert not isinstance(captured.value, UnexpectedToolError)

    asyncio.run(invoke())


def test_corpus_kwic_query_schema_and_direct_call_are_bounded(monkeypatch):
    schema = next(
        tool.model_dump(by_alias=True)["inputSchema"]
        for tool in asyncio.run(mcp_module.server.list_tools())
        if tool.name == "corpus_kwic"
    )
    query_schema = schema["properties"]["query"]
    assert query_schema["maxLength"] == mcp_module.MCP_KWIC_MAX_QUERY_CHARS
    assert query_schema["minLength"] == 1

    def unexpected_project_open():
        pytest.fail("oversized KWIC queries must be rejected before project access")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)
    oversized_query = "private-query-" + "x" * mcp_module.MCP_KWIC_MAX_QUERY_CHARS
    with pytest.raises(ToolError) as captured:
        mcp_module.corpus_kwic("fixture-revision", oversized_query)
    assert str(captured.value) == (
        f"KWIC query must contain 1..{mcp_module.MCP_KWIC_MAX_QUERY_CHARS} characters"
    )
    assert oversized_query not in str(captured.value)


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "search_vector",
            {
                "revision_id": "fixture-revision",
                "artifact_ids": ["fixture-artifact"],
                "vector_query": [0.1, 0.2],
                "limit": mcp_module.MCP_SEMANTIC_SEARCH_MAX_RESULTS + 1,
            },
        ),
        (
            "search_hybrid",
            {
                "revision_id": "fixture-revision",
                "artifact_ids": ["fixture-artifact"],
                "lexical_query": "fixture",
                "vector_query": [0.1, 0.2],
                "limit": mcp_module.MCP_SEMANTIC_SEARCH_MAX_RESULTS + 1,
            },
        ),
    ],
)
def test_raw_semantic_tools_reject_unbounded_result_limits_before_project_open(
    monkeypatch, tool_name: str, arguments: dict[str, object]
):
    def unexpected_project_open():
        pytest.fail("invalid result limits must be rejected before opening a project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)

    async def invoke() -> None:
        with pytest.raises(ToolError) as captured:
            await mcp_module.server.call_tool(tool_name, arguments)
        assert not isinstance(captured.value, UnexpectedToolError)

    asyncio.run(invoke())


def test_mcp_schema_errors_do_not_echo_sensitive_argument_values():
    secret = "sk-live-mcp-error-sentinel"
    private_path = "/home/private/corpus/credentials.txt"

    async def invoke() -> None:
        result = await mcp_module.server._handle_call_tool(
            cast(Any, None),
            CallToolRequestParams(
                name="search_vector",
                arguments={
                    "revision_id": "fixture-revision",
                    "artifact_ids": [],
                    "vector_query": [{"token": secret, "source": private_path}],
                },
            ),
        )
        payload = result.model_dump(by_alias=True)
        assert payload["isError"] is True
        assert payload["structuredContent"] is None
        content = payload["content"]
        assert isinstance(content, list) and len(content) == 1
        message = content[0]["text"]
        assert message == "invalid MCP tool arguments"
        assert len(message) < 128
        assert secret not in message
        assert private_path not in message
        assert "input_value" not in message

    asyncio.run(invoke())


def test_fulltext_search_pages_cap_paragraph_utf8_bytes_and_preserve_scope_provenance(
    tmp_path, monkeypatch
):
    root = tmp_path / "mcp-search-output-bound"
    project = Project.init(root)
    _, revision_id = project.import_text("long.txt", f"needle {'界' * 30_000}")
    scope_id = project.snapshot([revision_id])
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    lexical = mcp_module.search_lexical(scope_id, "needle", limit=1)
    exact = mcp_module.search_exact(revision_id, "needle", limit=1)

    lexical_rows = cast(list[dict[str, object]], lexical["rows"])
    exact_rows = cast(list[dict[str, object]], exact["rows"])
    assert len(lexical_rows) == len(exact_rows) == 1
    assert lexical["scope_id"] == scope_id
    assert lexical_rows[0]["revision_id"] == revision_id
    assert exact["revision_id"] == revision_id
    assert exact["scope_id"]
    for row in (lexical_rows[0], exact_rows[0]):
        text = cast(str, row["text"])
        assert text.startswith("needle ")
        assert len(text.encode("utf-8")) <= mcp_module.MCP_SEARCH_PAGE_TEXT_MAX_BYTES
        assert row.get("text_truncated") is True


def test_corpus_kwic_caps_total_returned_context_text_and_discloses_truncation(
    tmp_path, monkeypatch
):
    root = tmp_path / "mcp-kwic-output-bound"
    project = Project.init(root)
    _, revision_id = project.import_text("long-token.txt", f"needle {'x' * 100_000} needle")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    page = mcp_module.corpus_kwic(revision_id, "needle", window=1, limit=1)

    rows = page["rows"]
    assert len(rows) == 1
    assert page["total_count"] == 2
    assert page["has_more"] is True
    assert page["text_truncated"] is True
    assert rows[0]["revision_id"] == revision_id
    assert rows[0]["start_offset"] == 0
    assert rows[0]["keyword"] == "needle"
    assert rows[0]["text_truncated"] is True
    text_bytes = sum(len(rows[0][field].encode("utf-8")) for field in ("left", "keyword", "right"))
    assert text_bytes <= mcp_module.MCP_KWIC_PAGE_TEXT_MAX_BYTES
    assert len(rows[0]["right"]) < 100_000
