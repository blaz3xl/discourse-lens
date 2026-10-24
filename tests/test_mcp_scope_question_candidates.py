from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import discourselens.mcp_server as mcp_module
from discourselens import Project


def _call_tool(arguments: dict[str, object]):
    async def invoke():
        return await mcp_module.server.call_tool("run_scope_question_candidates", arguments)

    return asyncio.run(invoke())


def _result_payload(result) -> dict[str, object]:
    assert not result.is_error
    assert len(result.content) == 1
    payload = json.loads(result.content[0].text)
    assert isinstance(payload, dict)
    return payload


def test_scope_question_mcp_tool_requires_bounded_explicit_revision_scope():
    tools = {tool.name: tool for tool in asyncio.run(mcp_module.server.list_tools())}
    assert "run_scope_question_candidates" in tools

    schema = tools["run_scope_question_candidates"].model_dump(by_alias=True)["inputSchema"]
    assert schema["required"] == ["revision_ids", "framework_id", "question"]
    properties = schema["properties"]
    assert properties["revision_ids"]["minItems"] == 1
    assert properties["revision_ids"]["maxItems"] == mcp_module.MCP_SCOPE_QUESTION_MAX_REVISIONS
    assert properties["revision_ids"]["uniqueItems"] is True
    assert properties["framework_id"]["enum"] == list(
        mcp_module._MCP_INTERPRETIVE_FRAMEWORK_IDS
    )
    assert "critical-reviewer" not in properties["framework_id"]["enum"]
    assert properties["question"]["minLength"] == 1
    assert properties["question"]["maxLength"] == mcp_module.MCP_SCOPE_QUESTION_MAX_CHARS
    assert properties["limit"]["minimum"] == 1
    assert properties["limit"]["maximum"] == mcp_module.MCP_SCOPE_QUESTION_MAX_RESULTS


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"framework_id": "fairclough-text", "question": "Describe agency."},
        {"revision_ids": [], "framework_id": "fairclough-text", "question": "Describe agency."},
        {
            "revision_ids": [" \t"],
            "framework_id": "fairclough-text",
            "question": "Describe agency.",
        },
        {
            "revision_ids": ["revision-1", "revision-1"],
            "framework_id": "fairclough-text",
            "question": "Describe agency.",
        },
        {
            "revision_ids": ["revision-1"],
            "framework_id": "critical-reviewer",
            "question": "Describe agency.",
        },
        {
            "revision_ids": ["revision-1"],
            "framework_id": "unknown-framework",
            "question": "Describe agency.",
        },
        {
            "revision_ids": ["revision-1"],
            "framework_id": "fairclough-text",
            "question": " \t ",
        },
        {
            "revision_ids": ["revision-1"],
            "framework_id": "fairclough-text",
            "question": "Describe agency.",
            "limit": 0,
        },
        {
            "revision_ids": ["revision-1"],
            "framework_id": "fairclough-text",
            "question": "Describe agency.",
            "limit": 13,
        },
    ],
)
def test_scope_question_mcp_rejects_invalid_arguments_before_open(
    monkeypatch, arguments: dict[str, object]
) -> None:
    def unexpected_project_open():
        pytest.fail("invalid scope-question arguments must not open a project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)

    with pytest.raises(ToolError, match="invalid MCP tool arguments"):
        _call_tool(arguments)


def test_scope_question_mcp_forwards_explicit_scope_and_returns_provenance(monkeypatch):
    observed: dict[str, Any] = {}
    expected: dict[str, object] = {
        "analysis_run_id": "analysis-run",
        "scope_snapshot_id": "scope-snapshot",
        "skill_run_id": "skill-run",
        "retrieval_run_id": "retrieval-run",
        "finding_id": None,
        "evidence_ids": [],
        "coverage_id": "coverage-1",
        "candidate_count": 0,
        "coverage": {
            "coverage_mode": "TARGETED",
            "scope_snapshot_id": "scope-snapshot",
            "negative_result_status": "NO_MATCHES_UNDER_PLAN",
        },
    }

    class StubProject:
        def run_scope_question_candidates(self, revision_ids, framework_id, question, *, limit):
            observed["revision_ids"] = revision_ids
            observed["framework_id"] = framework_id
            observed["question"] = question
            observed["limit"] = limit
            return expected

        def close(self) -> None:
            observed["closed"] = True

    stub = StubProject()
    monkeypatch.setattr(mcp_module, "_open_project", lambda: stub)

    result = mcp_module.run_scope_question_candidates(
        ["revision-a", "revision-b"], "fairclough-text", "Describe agency", limit=7
    )

    assert result == expected
    assert observed == {
        "revision_ids": ["revision-a", "revision-b"],
        "framework_id": "fairclough-text",
        "question": "Describe agency",
        "limit": 7,
        "closed": True,
    }


def test_scope_question_mcp_only_searches_explicit_revisions_and_keeps_zero_targeted(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    project = Project.init(root)
    _, selected = project.import_text("selected.txt", "Local reporting covered the budget.")
    project.import_text("outside.txt", "The minister shifted blame to the committee.")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    result = _result_payload(
        _call_tool(
            {
                "revision_ids": [selected],
                "framework_id": "fairclough-text",
                "question": "How does the minister shift blame?",
            }
        )
    )

    assert result["candidate_count"] == 0
    assert result["evidence_ids"] == []
    coverage = result["coverage"]
    assert isinstance(coverage, Mapping)
    assert coverage["coverage_mode"] == "TARGETED"
    assert coverage["scope_snapshot_id"] == result["scope_snapshot_id"]
    assert coverage["negative_result_status"] != "ABSENCE_IN_EXHAUSTIVE_POPULATION"

    reopened = Project.open(root)
    try:
        summary = reopened.analysis_run_summary(str(result["analysis_run_id"]))
        run = summary["analysis_run"]
        assert isinstance(run, Mapping)
        assert run["scope_id"] == result["scope_snapshot_id"]
        assert reopened.scope_revisions(str(run["scope_id"])) == [selected]
    finally:
        reopened.close()
