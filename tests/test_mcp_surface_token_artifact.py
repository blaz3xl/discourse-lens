from __future__ import annotations

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from discourselens import Project, mcp_server
from discourselens.errors import ProjectError
from discourselens.nlp_artifacts import SURFACE_TOKEN_SCHEMA_VERSION


def _call_build_surface_tokens(revision_id: str):
    async def invoke():
        return await mcp_server.server.call_tool(
            "build_surface_token_artifact", {"revision_id": revision_id}
        )

    return asyncio.run(invoke())


def test_mcp_surface_token_artifact_completes_scoped_evidence_workflow(
    tmp_path, monkeypatch
):
    root = tmp_path / "surface-token-project"
    project = Project.init(root)
    _, selected_revision = project.import_text("selected.txt", "Apple apple pears.")
    _, outside_revision = project.import_text("outside.txt", "Apple outside the selection.")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    tool = next(
        item
        for item in asyncio.run(mcp_server.server.list_tools())
        if item.name == "build_surface_token_artifact"
    )
    assert tool.description is not None
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    assert schema["required"] == ["revision_id"]
    assert set(schema["properties"]) == {"revision_id"}
    assert "language-agnostic" in tool.description
    assert "No model or network" in tool.description

    response = _call_build_surface_tokens(selected_revision)
    assert not response.is_error
    artifact = json.loads(response.content[0].text)
    assert set(artifact) == {
        "artifact_id",
        "revision_id",
        "artifact_type",
        "schema_version",
        "derivation_hash",
        "content_hash",
    }
    assert artifact["revision_id"] == selected_revision
    assert artifact["artifact_type"] == "surface_tokens"
    assert artifact["schema_version"] == SURFACE_TOKEN_SCHEMA_VERSION
    assert "relative_path" not in artifact
    assert "Apple apple pears." not in response.content[0].text

    evidence = mcp_server.corpus_artifact_frequency_evidence(
        selected_revision, [artifact["artifact_id"]], "normalized", "apple"
    )
    assert evidence["count"] == 2
    assert evidence["population_count"] == 3
    scope_id = evidence["scope_id"]
    assert isinstance(scope_id, str) and scope_id
    assert evidence["artifact_ids"] == [artifact["artifact_id"]]

    with Project.open(root, read_only=True) as stored:
        assert stored.scope_revisions(scope_id) == [selected_revision]
        assert outside_revision not in stored.scope_revisions(scope_id)
        row = stored.db.execute(
            "SELECT revision_id,artifact_type,schema_version,content_hash,file_hash "
            "FROM artifact_registry WHERE id=?",
            (artifact["artifact_id"],),
        ).fetchone()
        assert tuple(row) == (
            selected_revision,
            "surface_tokens",
            SURFACE_TOKEN_SCHEMA_VERSION,
            artifact["derivation_hash"],
            artifact["content_hash"],
        )


def test_mcp_surface_token_artifact_is_not_discoverable_or_writable_read_only(
    tmp_path, monkeypatch
):
    root = tmp_path / "surface-token-readonly-project"
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "A source sentence.")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", True)

    try:
        asyncio.run(mcp_server._configure_read_only_server())
        names = {tool.name for tool in asyncio.run(mcp_server.server.list_tools())}
        assert "build_surface_token_artifact" not in names
        with pytest.raises(ProjectError, match="read-only mode"):
            mcp_server.build_surface_token_artifact(revision_id)
    finally:
        monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", False)
        mcp_server._restore_full_server_registry()

    with Project.open(root, read_only=True) as stored:
        assert stored.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='surface_tokens'"
        ).fetchone()[0] == 0


def test_mcp_surface_token_artifact_rejects_invalid_revision_before_write(
    tmp_path, monkeypatch
):
    root = tmp_path / "surface-token-invalid-project"
    Project.init(root).close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ToolError):
        mcp_server.build_surface_token_artifact(" ")

    with Project.open(root, read_only=True) as stored:
        assert stored.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='surface_tokens'"
        ).fetchone()[0] == 0
