from __future__ import annotations

import asyncio
from typing import Any, cast

from discourselens import Project, mcp_server
from discourselens.models import CredentialReference, ModelPreset


class _FixtureEmbeddingProvider:
    provider_id = "openai-compatible"
    endpoint = "https://embed.example/v1"

    def __init__(self) -> None:
        self.query_inputs: list[str] = []

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        return [(1.0, 0.0) for _ in texts]

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self.query_inputs.extend(texts)
        return [(1.0, 0.0) for _ in texts]


class _FixtureEmbeddingGateway:
    def __init__(self, provider: _FixtureEmbeddingProvider) -> None:
        self.embedders = {provider.provider_id: provider}
        self._provider = provider

    def embed_query(self, provider_id: str, model: str, texts: list[str]):
        assert provider_id == self._provider.provider_id
        return self._provider.embed_query(model, texts)


def test_readonly_mcp_semantic_search_returns_local_index_passages_without_writes(
    tmp_path, monkeypatch
):
    root = tmp_path / "readonly-semantic-project"
    project = Project.init(root)
    document_text = "The government framed migration as a security question."
    _, revision_id = project.import_text("speech.txt", document_text)
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "semantic-alias", "DL_MCP_SEMANTIC_KEY", "https://embed.example/v1"
        )
    )
    project.save_model_preset(
        ModelPreset(
            "semantic-preset",
            "openai-compatible",
            "semantic-alias",
            "fixture-embedding-model",
        )
    )
    provider = _FixtureEmbeddingProvider()
    artifact_id = project.build_paragraph_embedding_index(
        "semantic-preset", revision_id, embedding_provider=provider
    )
    scope_id = project.snapshot([revision_id])
    project.close()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setenv("DL_MCP_SEMANTIC_KEY", "fixture-secret")
    monkeypatch.setattr(
        "discourselens.vectors.build_embedding_gateway",
        lambda project, provider_id, credential_alias, *, embedding_provider=None: (
            _FixtureEmbeddingGateway(provider)
        ),
    )
    with Project.open(root, read_only=True) as before:
        counts_before = {
            table: before.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("scope_snapshots", "source_spans", "retrieval_runs", "retrieval_hits")
        }

    result = mcp_server.search_semantic_passages(
        scope_id,
        [artifact_id],
        "semantic-preset",
        "How is migration framed?",
    )
    indexes = mcp_server.list_semantic_indexes(scope_id)

    assert provider.query_inputs == ["How is migration framed?"]
    assert result["scope_id"] == scope_id
    assert result["coverage"] == "TOP_K_RETRIEVAL_NOT_EXHAUSTIVE"
    assert result["persistence"] == "NOT_PERSISTED_READ_ONLY"
    assert result["provider_id"] == "openai-compatible"
    rows = result["hits"]
    assert isinstance(rows, list) and len(rows) == 1
    hit = rows[0]
    assert hit["revision_id"] == revision_id
    assert hit["text"] == document_text
    assert hit["citation"] == "[speech.txt · ¶1]"
    assert hit["start"] == 0 and hit["end"] == len(document_text)
    assert "fixture-secret" not in repr(result)
    assert indexes["revision_ids"] == [revision_id]
    assert indexes["total"] == 1
    assert cast(list[dict[str, object]], indexes["items"])[0]["artifact_id"] == artifact_id

    with Project.open(root, read_only=True) as after:
        counts_after = {
            table: after.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in counts_before
        }
    assert counts_after == counts_before


def test_readonly_semantic_search_is_discovered_in_readonly_tool_surface():
    async def inspect() -> tuple[dict[str, Any], set[str]]:
        full_tools = {tool.name: tool for tool in await mcp_server.server.list_tools()}
        await mcp_server._configure_read_only_server()
        exposed = {tool.name for tool in await mcp_server.server.list_tools()}
        return full_tools["search_semantic_passages"].model_dump(by_alias=True), exposed

    try:
        schema, exposed = asyncio.run(inspect())
    finally:
        mcp_server._restore_full_server_registry()

    assert "search_semantic_passages" in exposed
    assert "search_semantic_with_preset" not in exposed
    input_schema = schema["inputSchema"]
    assert input_schema["properties"]["limit"]["maximum"] == 5
    assert "sent to the embedding provider" in schema["description"]
