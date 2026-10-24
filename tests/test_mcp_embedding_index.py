from __future__ import annotations

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from discourselens import Project, mcp_server
from discourselens.errors import ProjectError
from discourselens.models import CredentialReference, ModelPreset


class _EmbeddingFixture:
    provider_id = "openai-compatible"

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.inputs: list[tuple[str, list[str]]] = []

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self.inputs.append((model, list(texts)))
        return [(1.0, float(index)) for index, _ in enumerate(texts)]


def _create_index_project(root, *, endpoint: str, offline: bool = False):
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "First paragraph.\nSecond paragraph.")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "mcp-index-alias", "DL_MCP_INDEX_SECRET", endpoint)
    )
    preset_id = project.save_model_preset(
        ModelPreset(
            "mcp-index-preset",
            "openai-compatible",
            "mcp-index-alias",
            "fixture-embedding-model",
        )
    )
    if offline:
        project.set_offline_mode(True)
    project.close()
    return revision_id, preset_id


def _inject_fixture_provider(monkeypatch, provider: _EmbeddingFixture) -> None:
    core_method = Project.build_paragraph_embedding_index

    def with_fixture(project, preset_name, revision_id, *, embedding_provider=None):
        return core_method(
            project,
            preset_name,
            revision_id,
            embedding_provider=provider,
        )

    monkeypatch.setattr(Project, "build_paragraph_embedding_index", with_fixture)


def _call_build(arguments: dict[str, object]):
    async def invoke():
        return await mcp_server.server.call_tool("build_paragraph_embedding_index", arguments)

    return asyncio.run(invoke())


def test_mcp_embedding_index_builds_one_explicit_revision_without_returning_text_or_secrets(
    tmp_path, monkeypatch
):
    endpoint = "http://local.test/v1"
    root = tmp_path / "mcp-index-project"
    revision_id, preset_id = _create_index_project(root, endpoint=endpoint)
    provider = _EmbeddingFixture(endpoint)
    _inject_fixture_provider(monkeypatch, provider)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setenv("DL_MCP_INDEX_SECRET", "never-return-this-secret")

    def reject_network(*args, **kwargs):
        raise AssertionError("embedding fixture must not access the network")

    monkeypatch.setattr("discourselens.providers._open_provider_request", reject_network)
    tool = next(
        item
        for item in asyncio.run(mcp_server.server.list_tools())
        if item.name == "build_paragraph_embedding_index"
    )
    assert tool.description is not None
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    assert set(schema["required"]) == {"revision_id", "preset_id"}
    assert set(schema["properties"]) == {"revision_id", "preset_id"}
    assert "every persisted paragraph" in tool.description
    assert "offline mode" in tool.description

    result = _call_build({"revision_id": revision_id, "preset_id": preset_id})

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload == {
        "artifact_id": payload["artifact_id"],
        "revision_id": revision_id,
        "preset_id": preset_id,
        "status": "INDEXED",
    }
    assert provider.inputs == [
        ("fixture-embedding-model", ["First paragraph.", "Second paragraph."])
    ]
    assert "First paragraph." not in result.content[0].text
    assert endpoint not in result.content[0].text
    assert "never-return-this-secret" not in result.content[0].text

    with Project.open(root, read_only=True) as project:
        artifact = project.db.execute(
            "SELECT revision_id,artifact_type,embedding_provider_id,embedding_model "
            "FROM artifact_registry WHERE id=?",
            (payload["artifact_id"],),
        ).fetchone()
        assert tuple(artifact) == (
            revision_id,
            "vector_index",
            "openai-compatible",
            "fixture-embedding-model",
        )


def test_mcp_embedding_index_respects_project_offline_mode_before_embedding(
    tmp_path, monkeypatch
):
    endpoint = "https://embedding.example.test/v1"
    root = tmp_path / "mcp-index-offline-project"
    revision_id, preset_id = _create_index_project(root, endpoint=endpoint, offline=True)
    provider = _EmbeddingFixture(endpoint)
    _inject_fixture_provider(monkeypatch, provider)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ProjectError, match="offline mode blocks remote embeddings"):
        mcp_server.build_paragraph_embedding_index(revision_id, preset_id)

    assert provider.inputs == []
    with Project.open(root, read_only=True) as project:
        assert project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
        ).fetchone()[0] == 0


def test_mcp_embedding_index_is_not_discoverable_or_callable_read_only(tmp_path, monkeypatch):
    endpoint = "http://local.test/v1"
    root = tmp_path / "mcp-index-readonly-project"
    revision_id, preset_id = _create_index_project(root, endpoint=endpoint)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", True)

    try:
        asyncio.run(mcp_server._configure_read_only_server())
        names = {tool.name for tool in asyncio.run(mcp_server.server.list_tools())}
        assert "build_paragraph_embedding_index" not in names
        with pytest.raises(ProjectError, match="read-only mode"):
            mcp_server.build_paragraph_embedding_index(revision_id, preset_id)
    finally:
        monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", False)
        mcp_server._restore_full_server_registry()

    with Project.open(root, read_only=True) as project:
        assert project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
        ).fetchone()[0] == 0


def test_mcp_embedding_index_rejects_unknown_preset_before_build(tmp_path, monkeypatch):
    endpoint = "http://local.test/v1"
    root = tmp_path / "mcp-index-unknown-preset-project"
    revision_id, _ = _create_index_project(root, endpoint=endpoint)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ToolError, match="unknown model preset ID"):
        mcp_server.build_paragraph_embedding_index(
            revision_id,
            "00000000-0000-0000-0000-000000000000",
        )

    with Project.open(root, read_only=True) as project:
        assert project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='vector_index'"
        ).fetchone()[0] == 0
