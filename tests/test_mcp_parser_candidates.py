from __future__ import annotations

import asyncio
from typing import Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import discourselens.mcp_server as mcp_module

TOOL_METHODS = (
    ("run_social_actor_candidates", "analyze_social_actor_candidates"),
    ("run_agency_candidates", "analyze_agency_candidates"),
)


def _call_tool(tool_name: str, arguments: dict[str, object]):
    async def invoke():
        return await mcp_module.server.call_tool(tool_name, arguments)

    return asyncio.run(invoke())


@pytest.mark.parametrize("tool_name,method_name", TOOL_METHODS)
def test_parser_candidate_tool_requires_explicit_bounded_unique_scopes(
    tool_name: str, method_name: str
) -> None:
    tools = {tool.name: tool for tool in asyncio.run(mcp_module.server.list_tools())}
    assert tool_name in tools

    schema = tools[tool_name].model_dump(by_alias=True)["inputSchema"]
    assert schema["required"] == ["revision_ids", "artifact_ids"]
    properties = schema["properties"]
    assert properties["revision_ids"]["minItems"] == 1
    assert properties["revision_ids"]["maxItems"] == (
        mcp_module.MCP_PARSER_CANDIDATE_MAX_REVISIONS
    )
    assert properties["revision_ids"]["uniqueItems"] is True
    assert properties["artifact_ids"]["minItems"] == 1
    assert properties["artifact_ids"]["maxItems"] == (
        mcp_module.MCP_PARSER_CANDIDATE_MAX_ARTIFACTS
    )
    assert properties["artifact_ids"]["uniqueItems"] is True
    assert (tool_name, method_name) in TOOL_METHODS


@pytest.mark.parametrize("tool_name,_method_name", TOOL_METHODS)
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"revision_ids": ["revision-1"]},
        {"artifact_ids": ["artifact-1"]},
        {"revision_ids": [], "artifact_ids": ["artifact-1"]},
        {"revision_ids": [" \t"], "artifact_ids": ["artifact-1"]},
        {
            "revision_ids": ["revision-1", "revision-1"],
            "artifact_ids": ["artifact-1"],
        },
        {"revision_ids": ["revision-1"], "artifact_ids": []},
        {"revision_ids": ["revision-1"], "artifact_ids": [" \t"]},
        {
            "revision_ids": ["revision-1"],
            "artifact_ids": ["artifact-1", "artifact-1"],
        },
        {
            "revision_ids": ["revision-1"] * 101,
            "artifact_ids": ["artifact-1"],
        },
        {
            "revision_ids": ["revision-1"],
            "artifact_ids": ["artifact-1"] * 101,
        },
        {"revision_ids": ["r" * 129], "artifact_ids": ["artifact-1"]},
        {"revision_ids": [7], "artifact_ids": ["artifact-1"]},
    ],
)
def test_parser_candidate_tools_reject_invalid_scopes_before_open(
    monkeypatch, tool_name: str, _method_name: str, arguments: dict[str, object]
) -> None:
    def unexpected_project_open():
        pytest.fail("invalid parser-candidate scopes must not open a project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)

    with pytest.raises(ToolError, match="invalid MCP tool arguments"):
        _call_tool(tool_name, arguments)


@pytest.mark.parametrize("tool_name,method_name", TOOL_METHODS)
def test_parser_candidate_tools_forward_explicit_scopes_and_close(
    monkeypatch, tool_name: str, method_name: str
) -> None:
    observed: dict[str, Any] = {}
    expected = {
        "analysis_run_id": "analysis-run",
        "skill_run_id": "skill-run",
        "finding_ids": ["finding-1"],
    }

    class StubProject:
        def _analyze(self, revision_ids, artifact_ids):
            observed["revision_ids"] = revision_ids
            observed["artifact_ids"] = artifact_ids
            return expected

        def analyze_social_actor_candidates(self, revision_ids, artifact_ids):
            observed["method"] = "analyze_social_actor_candidates"
            return self._analyze(revision_ids, artifact_ids)

        def analyze_agency_candidates(self, revision_ids, artifact_ids):
            observed["method"] = "analyze_agency_candidates"
            return self._analyze(revision_ids, artifact_ids)

        def close(self) -> None:
            observed["closed"] = True

    stub = StubProject()
    monkeypatch.setattr(mcp_module, "_open_project", lambda: stub)

    result = getattr(mcp_module, tool_name)(["revision-a", "revision-b"], ["artifact-a"])

    assert result == expected
    assert observed == {
        "method": method_name,
        "revision_ids": ["revision-a", "revision-b"],
        "artifact_ids": ["artifact-a"],
        "closed": True,
    }


@pytest.mark.parametrize("tool_name,method_name", TOOL_METHODS)
def test_parser_candidate_tools_close_project_when_core_fails(
    monkeypatch, tool_name: str, method_name: str
) -> None:
    observed: dict[str, Any] = {}

    class StubProject:
        def _fail(self, method: str):
            observed["method"] = method
            raise RuntimeError("Core analysis failed")

        def analyze_social_actor_candidates(self, revision_ids, artifact_ids):
            return self._fail("analyze_social_actor_candidates")

        def analyze_agency_candidates(self, revision_ids, artifact_ids):
            return self._fail("analyze_agency_candidates")

        def close(self) -> None:
            observed["closed"] = True

    monkeypatch.setattr(mcp_module, "_open_project", StubProject)

    with pytest.raises(RuntimeError, match="Core analysis failed"):
        getattr(mcp_module, tool_name)(["revision-1"], ["artifact-1"])

    assert observed == {"method": method_name, "closed": True}
