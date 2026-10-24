from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.models import CredentialReference, ModelPreset


def _call_tool(arguments: dict[str, object]):
    async def invoke():
        return await mcp_module.server.call_tool("ask_question", arguments)

    return asyncio.run(invoke())


def test_ask_question_schema_is_bounded_scoped_and_model_explicit() -> None:
    tools = {tool.name: tool for tool in asyncio.run(mcp_module.server.list_tools())}
    tool = tools["ask_question"]
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    properties = schema["properties"]

    assert set(schema["required"]) == {
        "revision_ids",
        "question",
        "framework_id",
        "preset_name",
    }
    assert properties["revision_ids"]["minItems"] == 1
    assert properties["revision_ids"]["maxItems"] == 100
    assert properties["revision_ids"]["uniqueItems"] is True
    assert properties["question"]["minLength"] == 1
    assert properties["question"]["maxLength"] == 2_000
    assert properties["framework_id"]["enum"] == list(
        mcp_module._MCP_INTERPRETIVE_FRAMEWORK_IDS
    )
    assert "critical-reviewer" not in properties["framework_id"]["enum"]
    assert properties["preset_name"]["minLength"] == 1
    assert properties["preset_name"]["maxLength"] == 128
    assert properties["limit"]["minimum"] == 1
    assert properties["limit"]["maximum"] == 12
    assert properties["retrieval_mode"]["enum"] == ["lexical", "hybrid"]
    assert properties["retrieval_mode"]["default"] == "lexical"
    assert "retrieval_mode" not in schema["required"]
    assert "embedding_preset_name" not in schema["required"]
    assert "embedding_artifact_ids" not in schema["required"]
    embedding_preset_schema = properties["embedding_preset_name"]["anyOf"][0]
    embedding_artifact_ids_schema = properties["embedding_artifact_ids"]["anyOf"][0]
    assert embedding_preset_schema["maxLength"] == 128
    assert embedding_artifact_ids_schema["minItems"] == 1
    assert embedding_artifact_ids_schema["maxItems"] == (
        mcp_module.MCP_HYBRID_SEARCH_MAX_ARTIFACTS
    )
    assert embedding_artifact_ids_schema["uniqueItems"] is True
    assert schema["additionalProperties"] is False
    description = " ".join((tool.description or "").split())
    assert "configured by the selected saved preset" in description
    assert "only retrieved passages" in description
    assert "explicit revisions" in description
    assert "question may be sent to the embedding provider" in description
    assert "retrieved passages may be sent to the answer provider" in description


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"revision_ids": [], "question": "Q", "framework_id": "fairclough-text", "preset_name": "p"},
        {
            "revision_ids": ["revision-a", "revision-a"],
            "question": "Q",
            "framework_id": "fairclough-text",
            "preset_name": "p",
        },
        {
            "revision_ids": [" \t"],
            "question": "Q",
            "framework_id": "fairclough-text",
            "preset_name": "p",
        },
        {
            "revision_ids": ["revision-a"],
            "question": " \t ",
            "framework_id": "fairclough-text",
            "preset_name": "p",
        },
        {
            "revision_ids": ["revision-a"],
            "question": "q" * 2_001,
            "framework_id": "fairclough-text",
            "preset_name": "p",
        },
        {
            "revision_ids": ["revision-a"],
            "question": "Q",
            "framework_id": "critical-reviewer",
            "preset_name": "p",
        },
        {
            "revision_ids": ["revision-a"],
            "question": "Q",
            "framework_id": "not-registered",
            "preset_name": "p",
        },
        {
            "revision_ids": ["revision-a"],
            "question": "Q",
            "framework_id": "fairclough-text",
            "preset_name": " \t ",
        },
        {
            "revision_ids": ["revision-a"],
            "question": "Q",
            "framework_id": "fairclough-text",
            "preset_name": "p" * 129,
        },
        {
            "revision_ids": ["revision-a"],
            "question": "Q",
            "framework_id": "fairclough-text",
            "preset_name": "p",
            "limit": 0,
        },
        {
            "revision_ids": ["revision-a"],
            "question": "Q",
            "framework_id": "fairclough-text",
            "preset_name": "p",
            "limit": 13,
        },
        {
            "revision_ids": ["revision-a"],
            "question": "Q",
            "framework_id": "fairclough-text",
            "preset_name": "p",
            "api_key": "fixture",
        },
        {
            "revision_ids": ["revision-a"],
            "question": "Q",
            "framework_id": "fairclough-text",
            "preset_name": "p",
            "retrieval_mode": "semantic",
        },
        {
            "revision_ids": ["revision-a"],
            "question": "Q",
            "framework_id": "fairclough-text",
            "preset_name": "p",
            "embedding_preset_name": "embedding-preset",
        },
        {
            "revision_ids": ["revision-a"],
            "question": "Q",
            "framework_id": "fairclough-text",
            "preset_name": "p",
            "embedding_artifact_ids": ["artifact-a"],
        },
        {
            "revision_ids": ["revision-a"],
            "question": "Q",
            "framework_id": "fairclough-text",
            "preset_name": "p",
            "retrieval_mode": "hybrid",
        },
        {
            "revision_ids": ["revision-a"],
            "question": "Q",
            "framework_id": "fairclough-text",
            "preset_name": "p",
            "retrieval_mode": "hybrid",
            "embedding_preset_name": "embedding-preset",
        },
        {
            "revision_ids": ["revision-a"],
            "question": "Q",
            "framework_id": "fairclough-text",
            "preset_name": "p",
            "retrieval_mode": "hybrid",
            "embedding_preset_name": "embedding-preset",
            "embedding_artifact_ids": ["artifact-a", "artifact-a"],
        },
    ],
)
def test_ask_question_rejects_invalid_arguments_before_opening_project(
    monkeypatch, arguments: dict[str, object]
) -> None:
    def unexpected_project_open():
        pytest.fail("invalid ask-question arguments must be rejected before project access")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)
    with pytest.raises(
        ToolError,
        match=(
            "invalid MCP tool arguments|embedding options require hybrid retrieval mode|"
            "hybrid retrieval requires"
        ),
    ) as error:
        _call_tool(arguments)
    assert "must-not-be-accepted-or-echoed" not in str(error.value)


def test_ask_question_forwards_exact_scope_and_preserves_targeted_zero_result(monkeypatch) -> None:
    observed: dict[str, Any] = {}
    expected: dict[str, object] = {
        "analysis_run_id": "run-1",
        "scope_snapshot_id": "scope-original",
        "question": "How is agency represented?",
        "framework_id": "fairclough-text",
        "candidate_count": 0,
        "answer": None,
        "coverage": {
            "coverage_mode": "TARGETED",
            "negative_result_status": "NO_MATCHES_UNDER_PLAN",
            "scope_snapshot_id": "scope-original",
        },
        "limitations": ["No retrieved passages support an answer under this plan."],
    }

    class StubProject:
        def ask_question(
            self,
            revision_ids: list[str],
            question: str,
            framework_id: str,
            preset_name: str,
            *,
            limit: int = 8,
            retrieval_mode: str = "lexical",
            embedding_preset_name: str | None = None,
            embedding_artifact_ids: list[str] | None = None,
        ) -> dict[str, object]:
            observed.update(
                revision_ids=revision_ids,
                question=question,
                framework_id=framework_id,
                preset_name=preset_name,
                limit=limit,
                retrieval_mode=retrieval_mode,
                embedding_preset_name=embedding_preset_name,
                embedding_artifact_ids=embedding_artifact_ids,
            )
            return expected

        def close(self) -> None:
            observed["closed"] = True

    stub = StubProject()
    monkeypatch.setattr(mcp_module, "_open_project", lambda: stub)
    result = mcp_module.ask_question(
        ["revision-a", "revision-b"],
        "How is agency represented?",
        "fairclough-text",
        "research-preset",
        limit=6,
    )

    assert result == expected
    assert observed == {
        "revision_ids": ["revision-a", "revision-b"],
        "question": "How is agency represented?",
        "framework_id": "fairclough-text",
        "preset_name": "research-preset",
        "limit": 6,
        "retrieval_mode": "lexical",
        "embedding_preset_name": None,
        "embedding_artifact_ids": None,
        "closed": True,
    }
    assert expected["candidate_count"] == 0
    assert expected["answer"] is None
    coverage = expected["coverage"]
    assert isinstance(coverage, dict)
    assert coverage["negative_result_status"] == "NO_MATCHES_UNDER_PLAN"


def test_ask_question_forwards_exact_hybrid_core_kwargs(monkeypatch) -> None:
    observed: dict[str, Any] = {}
    expected = {"answer": "bounded answer"}

    class StubProject:
        def ask_question(self, *args, **kwargs):
            observed["args"] = args
            observed["kwargs"] = kwargs
            return expected

        def close(self) -> None:
            observed["closed"] = True

    stub = StubProject()
    monkeypatch.setattr(mcp_module, "_open_project", lambda: stub)
    result = mcp_module.ask_question(
        ["revision-a"],
        "How is agency represented?",
        "fairclough-text",
        "answer-preset",
        limit=5,
        retrieval_mode="hybrid",
        embedding_preset_name="embedding-preset",
        embedding_artifact_ids=["embedding-artifact-a", "embedding-artifact-b"],
    )

    assert result == expected
    assert observed == {
        "args": (
            ["revision-a"],
            "How is agency represented?",
            "fairclough-text",
            "answer-preset",
        ),
        "kwargs": {
            "limit": 5,
            "retrieval_mode": "hybrid",
            "embedding_preset_name": "embedding-preset",
            "embedding_artifact_ids": ["embedding-artifact-a", "embedding-artifact-b"],
        },
        "closed": True,
    }


def test_ask_question_rejects_hybrid_options_before_opening_project(monkeypatch) -> None:
    def unexpected_project_open():
        pytest.fail("invalid retrieval options must be rejected before project access")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)
    with pytest.raises(ToolError, match="hybrid retrieval requires"):
        mcp_module.ask_question(
            ["revision-a"],
            "How is agency represented?",
            "fairclough-text",
            "answer-preset",
            retrieval_mode="hybrid",
            embedding_preset_name="embedding-preset",
            embedding_artifact_ids=None,
        )


def test_ask_question_hybrid_is_blocked_in_read_only_mode(monkeypatch) -> None:
    def unexpected_project_open():
        pytest.fail("read-only hybrid question answering must be rejected before project access")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    with pytest.raises(mcp_module.ProjectError, match="read-only mode"):
        mcp_module.ask_question(
            ["revision-a"],
            "How is agency represented?",
            "fairclough-text",
            "answer-preset",
            retrieval_mode="hybrid",
            embedding_preset_name="embedding-preset",
            embedding_artifact_ids=["embedding-artifact-a"],
        )


def test_ask_question_zero_candidates_stay_inside_selected_revisions(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    project = Project.init(root)
    _, selected_revision = project.import_text("selected.txt", "Local budget reporting continued.")
    _, outside_revision = project.import_text("outside.txt", "Xylophone zebra framing appeared.")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DISCOURSELENS_UNUSED_ASK_KEY")
    )
    project.save_model_preset(
        ModelPreset("fixture-preset", "openai-compatible", "fixture", "fixture-model")
    )
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    def model_must_not_run(*args, **kwargs):
        pytest.fail("zero scoped candidates must not invoke the configured model")

    monkeypatch.setattr(Project, "review_finding_with_model", model_must_not_run)
    result = _call_tool(
        {
            "revision_ids": [selected_revision],
            "question": "How does xylophone zebra framing work?",
            "framework_id": "fairclough-text",
            "preset_name": "fixture-preset",
        }
    )
    assert not result.is_error
    payload = result.content[0].text
    parsed = json.loads(payload)
    assert parsed["status"] == "no_candidates"
    assert parsed["evidence_total"] == 0
    assert parsed["answer_interpretation"] is None
    assert "does not establish" in parsed["limitations"][0]

    reopened = Project.open(root, read_only=True)
    try:
        scope_id = parsed["scope_snapshot_id"]
        assert reopened.scope_revisions(scope_id) == [selected_revision]
        assert outside_revision not in reopened.scope_revisions(scope_id)
    finally:
        reopened.close()


def test_ask_question_is_not_exposed_in_read_only_mode(monkeypatch) -> None:
    async def configure_and_check() -> None:
        await mcp_module._configure_read_only_server()
        try:
            exposed = {tool.name for tool in await mcp_module.server.list_tools()}
            assert "ask_question" not in exposed
            assert "ask_question" not in mcp_module.READ_ONLY_MCP_TOOL_NAMES
            with pytest.raises(Exception, match="Unknown tool: ask_question"):
                await mcp_module.server.call_tool("ask_question", {})
        finally:
            mcp_module._restore_full_server_registry()

    asyncio.run(configure_and_check())
