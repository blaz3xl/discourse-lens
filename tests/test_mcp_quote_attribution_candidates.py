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
        return await mcp_module.server.call_tool("run_quote_attribution_candidates", arguments)

    return asyncio.run(invoke())


def _result_payload(result) -> dict[str, object]:
    assert not result.is_error
    assert len(result.content) == 1
    payload = json.loads(result.content[0].text)
    assert isinstance(payload, dict)
    return payload


def test_quote_attribution_mcp_tool_requires_bounded_explicit_revision_scope():
    tools = {tool.name: tool for tool in asyncio.run(mcp_module.server.list_tools())}
    assert "run_quote_attribution_candidates" in tools

    schema = tools["run_quote_attribution_candidates"].model_dump(by_alias=True)["inputSchema"]
    assert schema["required"] == ["revision_ids"]
    properties = schema["properties"]
    assert properties["revision_ids"]["minItems"] == 1
    assert properties["revision_ids"]["maxItems"] == (
        mcp_module.MCP_QUOTE_ATTRIBUTION_MAX_REVISIONS
    )
    assert properties["revision_ids"]["uniqueItems"] is True
    assert properties["language"]["anyOf"][0]["enum"] == ["en", "es"]


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"revision_ids": []},
        {"revision_ids": [""]},
        {"revision_ids": ["revision-1", "revision-1"]},
        {"revision_ids": ["revision-1"], "language": "fr"},
    ],
)
def test_quote_attribution_mcp_rejects_invalid_scope_or_language_before_open(
    monkeypatch, arguments: dict[str, object]
) -> None:
    def unexpected_project_open():
        pytest.fail("invalid quote-attribution arguments must not open a project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)

    with pytest.raises(ToolError, match="invalid MCP tool arguments"):
        _call_tool(arguments)


def test_quote_attribution_mcp_forwards_scope_language_provenance_and_closes(monkeypatch):
    observed: dict[str, Any] = {}
    expected = {
        "analysis_run_id": "analysis-run",
        "scope_snapshot_id": "scope-snapshot",
        "skill_run_id": "skill-run",
        "finding_ids": ["finding-1"],
        "evidence_ids": ["evidence-1"],
        "coverage_id": "coverage-1",
        "candidates": [],
    }

    class StubProject:
        def close(self) -> None:
            observed["closed"] = True

    def analyze(project, revision_ids, *, revision_languages):
        observed["project"] = project
        observed["revision_ids"] = revision_ids
        observed["revision_languages"] = revision_languages
        return expected

    stub = StubProject()
    monkeypatch.setattr(mcp_module, "_open_project", lambda: stub)
    monkeypatch.setattr(mcp_module, "run_quote_attribution_candidate_analysis", analyze)

    result = mcp_module.run_quote_attribution_candidates(
        ["revision-a", "revision-b"], language="es"
    )

    assert result == expected
    assert observed == {
        "project": stub,
        "revision_ids": ["revision-a", "revision-b"],
        "revision_languages": {"revision-a": "es", "revision-b": "es"},
        "closed": True,
    }


def test_quote_attribution_mcp_runs_only_explicit_revisions_and_optional_cue_scan(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    project = Project.init(root)
    _, selected = project.import_text("selected.txt", 'The minister said, “We will stay.”')
    project.import_text("outside.txt", 'Officials said "Leave now."')
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with_language = _result_payload(
        _call_tool({"revision_ids": [selected], "language": "en"})
    )
    assert with_language["candidate_count"] == 1
    assert with_language["attribution_candidate_count"] == 1
    candidates = with_language["candidates"]
    assert isinstance(candidates, list)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert isinstance(candidate, Mapping)
    assert candidate["revision_id"] == selected
    assert candidate["attribution_cues"] == ["said"]
    assert candidate["attribution_cue_languages"] == ["en"]
    assert with_language["analysis_run_id"]
    assert with_language["scope_snapshot_id"]
    assert with_language["skill_run_id"]
    assert with_language["finding_ids"]
    assert with_language["evidence_ids"]
    assert with_language["coverage_id"]

    without_language = _result_payload(_call_tool({"revision_ids": [selected]}))
    assert without_language["candidate_count"] == 1
    assert without_language["attribution_candidate_count"] == 0
    without_language_candidates = without_language["candidates"]
    assert isinstance(without_language_candidates, list)
    assert len(without_language_candidates) == 1
    without_language_candidate = without_language_candidates[0]
    assert isinstance(without_language_candidate, Mapping)
    assert without_language_candidate["attribution_cues"] == []

    reopened = Project.open(root)
    try:
        run = reopened.analysis_run_summary(str(with_language["analysis_run_id"]))[
            "analysis_run"
        ]
        assert isinstance(run, Mapping)
        assert run["scope_id"] == with_language["scope_snapshot_id"]
        run_config = run["config"]
        assert isinstance(run_config, Mapping)
        assert run_config["revision_languages"] == {selected: "en"}
        evidence_ids = with_language["evidence_ids"]
        assert isinstance(evidence_ids, list)
        for evidence_id in evidence_ids:
            evidence = reopened.db.execute(
                "SELECT content_json FROM evidence_artifacts WHERE id=?", (evidence_id,)
            ).fetchone()
            assert evidence is not None
            provenance = json.loads(evidence["content_json"])
            assert provenance["scope_snapshot_id"] == with_language["scope_snapshot_id"]
    finally:
        reopened.close()
