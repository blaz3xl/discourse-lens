from __future__ import annotations

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.models import CredentialReference, ModelPreset
from discourselens.vectors import build_paragraph_embedding_index


class _QueryEmbeddingFixture:
    provider_id = "cohere-embeddings"
    endpoint = "https://api.cohere.com"

    def __init__(self) -> None:
        self.document_queries: list[tuple[str, list[str]]] = []
        self.query_queries: list[tuple[str, list[str]]] = []

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self.document_queries.append((model, texts))
        return [(1.0, 0.0) for _ in texts]

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self.query_queries.append((model, texts))
        return [(1.0, 0.0) for _ in texts]


def _create_hybrid_project(root, *, offline: bool = False):
    project = Project.init(root)
    selected_revision: str
    outside_revision: str
    try:
        _, selected_revision = project.import_text("selected.txt", "Apples are green.")
        _, outside_revision = project.import_text("outside.txt", "Apples are not local.")
        project.save_credential_reference(
            CredentialReference("cohere-embeddings", "fixture-alias", "DL_MCP_HYBRID_SECRET")
        )
        project.save_model_preset(
            ModelPreset("hybrid-fixture", "cohere-embeddings", "fixture-alias", "embedding-model")
        )
        provider = _QueryEmbeddingFixture()
        selected_artifact = build_paragraph_embedding_index(
            project, "hybrid-fixture", selected_revision, provider
        )
        outside_artifact = build_paragraph_embedding_index(
            project, "hybrid-fixture", outside_revision, provider
        )
        scope_id = project.snapshot([selected_revision])
        if offline:
            project.set_offline_mode(True)
        return (
            selected_revision,
            outside_revision,
            selected_artifact,
            outside_artifact,
            scope_id,
            provider,
        )
    finally:
        project.close()


def _call_tool(arguments: dict[str, object]):
    async def invoke():
        return await mcp_module.server.call_tool("search_hybrid_with_preset", arguments)

    return asyncio.run(invoke())


def _call_semantic_tool(arguments: dict[str, object]):
    async def invoke():
        return await mcp_module.server.call_tool("search_semantic_with_preset", arguments)

    return asyncio.run(invoke())


def _result_json(result) -> dict[str, object]:
    assert not result.is_error
    assert len(result.content) == 1
    value = json.loads(result.content[0].text)
    assert isinstance(value, dict)
    return value


def _inject_provider(monkeypatch, provider: _QueryEmbeddingFixture) -> None:
    core_method = Project.hybrid_search_with_preset

    def with_fixture(
        project,
        preset_name,
        scope_id,
        artifact_ids,
        query_text,
        *,
        limit=10,
        rrf_constant=60,
    ):
        return core_method(
            project,
            preset_name,
            scope_id,
            artifact_ids,
            query_text,
            embedding_provider=provider,
            limit=limit,
            rrf_constant=rrf_constant,
        )

    monkeypatch.setattr(Project, "hybrid_search_with_preset", with_fixture)


def _inject_semantic_provider(monkeypatch, provider: _QueryEmbeddingFixture) -> None:
    core_method = Project.semantic_search_with_preset

    def with_fixture(
        project,
        preset_name,
        scope_id,
        artifact_ids,
        query_text,
        *,
        embedding_provider=None,
        limit=10,
    ):
        return core_method(
            project,
            preset_name,
            scope_id,
            artifact_ids,
            query_text,
            embedding_provider=provider,
            limit=limit,
        )

    monkeypatch.setattr(Project, "semantic_search_with_preset", with_fixture)


def test_mcp_hybrid_preset_schema_warns_query_may_egress_and_is_not_read_only():
    tools = asyncio.run(mcp_module.server.list_tools())
    tool = next(item for item in tools if item.name == "search_hybrid_with_preset")
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    properties = schema["properties"]

    assert {"scope_id", "artifact_ids", "preset_name", "query_text"} <= set(schema["required"])
    assert "limit" in properties
    assert properties["query_text"]["maxLength"] == mcp_module.MCP_HYBRID_SEARCH_MAX_QUERY_CHARS
    assert properties["artifact_ids"]["minItems"] == 1
    assert properties["artifact_ids"]["maxItems"] == mcp_module.MCP_HYBRID_SEARCH_MAX_ARTIFACTS
    assert properties["limit"]["minimum"] == 1
    assert properties["limit"]["maximum"] == mcp_module.MCP_HYBRID_SEARCH_MAX_RESULTS
    assert "retrieval_run_id" in tool.model_dump(by_alias=True)["outputSchema"]["properties"]
    assert "may be sent" in properties["query_text"]["description"]
    assert "offline mode" in properties["query_text"]["description"]
    assert "search_hybrid_with_preset" not in mcp_module.READ_ONLY_MCP_TOOL_NAMES

    async def inspect_read_only_surface():
        await mcp_module._configure_read_only_server()
        return {tool.name for tool in await mcp_module.server.list_tools()}

    try:
        read_only_tools = asyncio.run(inspect_read_only_surface())
    finally:
        mcp_module._restore_full_server_registry()
    assert "search_hybrid_with_preset" not in read_only_tools


def test_mcp_semantic_preset_schema_warns_query_may_egress_and_is_not_read_only():
    tools = asyncio.run(mcp_module.server.list_tools())
    tool = next(item for item in tools if item.name == "search_semantic_with_preset")
    descriptor = tool.model_dump(by_alias=True)
    schema = descriptor["inputSchema"]
    properties = schema["properties"]

    assert {"scope_id", "artifact_ids", "preset_name", "query_text"} <= set(schema["required"])
    assert {"scope_id", "artifact_ids", "preset_name", "query_characters", "limit", "hits"} <= set(
        descriptor["outputSchema"]["properties"]
    )
    assert "retrieval_run_id" in descriptor["outputSchema"]["properties"]
    assert properties["query_text"]["maxLength"] == (
        mcp_module.MCP_SEMANTIC_SEARCH_MAX_QUERY_CHARS
    )
    assert properties["artifact_ids"]["minItems"] == 1
    assert properties["artifact_ids"]["maxItems"] == mcp_module.MCP_HYBRID_SEARCH_MAX_ARTIFACTS
    assert properties["limit"]["minimum"] == 1
    assert properties["limit"]["maximum"] == mcp_module.MCP_SEMANTIC_SEARCH_MAX_RESULTS
    assert "may be sent" in properties["query_text"]["description"]
    assert "offline mode" in properties["query_text"]["description"]
    assert "search_semantic_with_preset" not in mcp_module.READ_ONLY_MCP_TOOL_NAMES

    async def inspect_read_only_surface():
        await mcp_module._configure_read_only_server()
        names = {item.name for item in await mcp_module.server.list_tools()}
        with pytest.raises(Exception, match="Unknown tool: search_semantic_with_preset"):
            await mcp_module.server.call_tool("search_semantic_with_preset", {})
        return names

    try:
        read_only_tools = asyncio.run(inspect_read_only_surface())
    finally:
        mcp_module._restore_full_server_registry()
    assert "search_semantic_with_preset" not in read_only_tools


@pytest.mark.parametrize(
    ("artifact_ids", "query_text", "limit"),
    [
        ([], "query", 10),
        (["a", "a"], "query", 10),
        (["a"], "query", 101),
        (["a"], "x" * 4_001, 10),
        (["a"], "query", False),
    ],
)
def test_mcp_semantic_preset_rejects_invalid_scope_inputs_before_project_access(
    artifact_ids, query_text, limit
):
    with pytest.raises(ToolError):
        mcp_module.search_semantic_with_preset(
            "scope-id", artifact_ids, "preset", query_text, limit=limit
        )


def test_mcp_semantic_preset_respects_project_offline_mode(tmp_path, monkeypatch):
    root = tmp_path / "offline-semantic-project"
    _, _, selected_artifact, _, scope_id, provider = _create_hybrid_project(root, offline=True)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    _inject_semantic_provider(monkeypatch, provider)

    with pytest.raises(ToolError) as error:
        _call_semantic_tool(
            {
                "scope_id": scope_id,
                "artifact_ids": [selected_artifact],
                "preset_name": "hybrid-fixture",
                "query_text": "offline semantic query",
            }
        )

    assert "offline semantic query" not in str(error.value)
    assert provider.query_queries == []


def test_mcp_hybrid_preset_is_scoped_bounded_and_secret_free(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (
        selected_revision,
        outside_revision,
        selected_artifact,
        outside_artifact,
        scope_id,
        provider,
    ) = _create_hybrid_project(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setenv("DL_MCP_HYBRID_SECRET", "never-return-this-provider-secret")
    _inject_provider(monkeypatch, provider)

    def forbidden(*args, **kwargs):
        raise AssertionError("fixture search must not resolve secrets or use the network")

    monkeypatch.setattr("discourselens.secrets.EnvironmentSecretStore.get", forbidden)
    monkeypatch.setattr("discourselens.secrets.KeyringSecretStore.get", forbidden)
    monkeypatch.setattr("discourselens.providers._open_provider_request", forbidden)

    query = "private query from the researcher"
    result = _result_json(
        _call_tool(
            {
                "scope_id": scope_id,
                "artifact_ids": [selected_artifact],
                "preset_name": "hybrid-fixture",
                "query_text": query,
                "limit": 5,
            }
        )
    )

    assert result["scope_id"] == scope_id
    assert isinstance(result["retrieval_run_id"], str) and result["retrieval_run_id"]
    assert result["artifact_ids"] == [selected_artifact]
    assert result["preset_name"] == "hybrid-fixture"
    assert result["query_characters"] == len(query)
    assert result["limit"] == 5
    hits = result["hits"]
    assert isinstance(hits, list) and len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, dict)
    assert hit["revision_id"] == selected_revision
    assert hit["artifact_id"] == selected_artifact
    assert outside_revision not in {item["revision_id"] for item in hits if isinstance(item, dict)}
    with Project.open(root, read_only=True) as reopened:
        run = reopened.db.execute(
            "SELECT scope_id FROM retrieval_runs WHERE id=?",
            (result["retrieval_run_id"],),
        ).fetchone()
        assert run is not None and run["scope_id"] == scope_id
    assert provider.query_queries == [("embedding-model", [query])]
    serialized = json.dumps(result)
    for sensitive_value in (
        query,
        "never-return-this-provider-secret",
        "fixture-alias",
        "DL_MCP_HYBRID_SECRET",
        outside_artifact,
    ):
        assert sensitive_value not in serialized

    with pytest.raises(ToolError) as error:
        _call_tool(
            {
                "scope_id": scope_id,
                "artifact_ids": [outside_artifact],
                "preset_name": "hybrid-fixture",
                "query_text": query,
            }
        )
    assert query not in str(error.value)
    assert len(provider.query_queries) == 1


def test_mcp_semantic_preset_returns_scope_run_and_hit_lineage(tmp_path, monkeypatch):
    root = tmp_path / "semantic-project"
    (
        selected_revision,
        outside_revision,
        selected_artifact,
        outside_artifact,
        scope_id,
        provider,
    ) = _create_hybrid_project(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    _inject_semantic_provider(monkeypatch, provider)

    result = _result_json(
        _call_semantic_tool(
            {
                "scope_id": scope_id,
                "artifact_ids": [selected_artifact],
                "preset_name": "hybrid-fixture",
                "query_text": "private semantic query",
                "limit": 5,
            }
        )
    )

    assert result["scope_id"] == scope_id
    assert isinstance(result["retrieval_run_id"], str) and result["retrieval_run_id"]
    assert result["artifact_ids"] == [selected_artifact]
    hits = result["hits"]
    assert isinstance(hits, list) and len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, dict)
    assert hit["revision_id"] == selected_revision
    assert hit["artifact_id"] == selected_artifact
    assert hit["chunk_id"]
    assert outside_revision not in {
        item["revision_id"] for item in hits if isinstance(item, dict)
    }
    assert outside_artifact not in {
        item["artifact_id"] for item in hits if isinstance(item, dict)
    }
    with Project.open(root, read_only=True) as reopened:
        run = reopened.db.execute(
            "SELECT scope_id FROM retrieval_runs WHERE id=?",
            (result["retrieval_run_id"],),
        ).fetchone()
        assert run is not None and run["scope_id"] == scope_id
    assert provider.query_queries == [
        ("embedding-model", ["private semantic query"])
    ]


def test_mcp_hybrid_preset_respects_project_offline_mode(tmp_path, monkeypatch):
    root = tmp_path / "offline-project"
    _, _, selected_artifact, _, scope_id, provider = _create_hybrid_project(root, offline=True)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    _inject_provider(monkeypatch, provider)

    with pytest.raises(ToolError) as error:
        _call_tool(
            {
                "scope_id": scope_id,
                "artifact_ids": [selected_artifact],
                "preset_name": "hybrid-fixture",
                "query_text": "offline query",
            }
        )

    assert "offline query" not in str(error.value)
    assert provider.query_queries == []


def test_mcp_hybrid_preset_rejects_oversized_query_without_echo(monkeypatch):
    marker = "private-overlong-query-marker"
    with pytest.raises(ToolError) as error:
        _call_tool(
            {
                "scope_id": "fixture-scope",
                "artifact_ids": ["fixture-artifact"],
                "preset_name": "fixture-preset",
                "query_text": marker + "x" * mcp_module.MCP_HYBRID_SEARCH_MAX_QUERY_CHARS,
            }
        )
    assert marker not in str(error.value)
    assert "length limit" in str(error.value)
