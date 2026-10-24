from __future__ import annotations

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from typer.testing import CliRunner

from discourselens import Project, mcp_server
from discourselens.cli import app
from discourselens.errors import DocumentError, ProjectError


def test_mcp_document_metadata_completes_reproducible_selection_flow(tmp_path, monkeypatch):
    root = tmp_path / "metadata-project"
    project = Project.init(root)
    document_id, revision_id = project.import_text("source.txt", "A source sentence.")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    response = mcp_server.set_document_metadata(
        document_id,
        "language",
        {"code": "en", "reviewed": True, "labels": ["speech", "public"]},
    )
    expected_metadata = {
        "language": {"code": "en", "reviewed": True, "labels": ["speech", "public"]}
    }
    mcp_metadata = mcp_server.get_document_metadata(document_id)
    selected = mcp_server.select_documents_by_metadata(
        {"language": {"code": "en", "reviewed": True, "labels": ["speech", "public"]}}
    )
    cli_result = CliRunner().invoke(app, ["document", "metadata", str(root), document_id])

    assert response == {"document_id": document_id, "key": "language", "status": "updated"}
    assert cli_result.exit_code == 0, cli_result.output
    assert mcp_metadata == expected_metadata
    assert json.loads(cli_result.output) == expected_metadata
    assert selected["revision_ids_preview"] == [revision_id]
    assert selected["revision_count"] == 1
    assert selected["revision_ids_truncated"] is False
    with Project.open(root, read_only=True) as stored:
        assert stored.document_metadata(document_id) == expected_metadata


def test_mcp_get_document_metadata_returns_empty_mapping_for_unannotated_document(
    tmp_path, monkeypatch
):
    root = tmp_path / "empty-document-metadata-project"
    with Project.init(root) as project:
        document_id, _ = project.import_text("source.txt", "A source sentence.")
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    assert mcp_server.get_document_metadata(document_id) == {}


@pytest.mark.parametrize("document_id", ["", " ", " doc-id ", "x" * 129, 7])
def test_mcp_get_document_metadata_rejects_malformed_document_ids(document_id):
    with pytest.raises(ToolError, match="valid document ID"):
        mcp_server.get_document_metadata(document_id)


def test_mcp_document_metadata_validates_id_and_json_before_core_write(tmp_path, monkeypatch):
    root = tmp_path / "invalid-metadata-project"
    project = Project.init(root)
    document_id, _ = project.import_text("source.txt", "A source sentence.")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ToolError, match="only JSON values"):
        mcp_server.set_document_metadata(document_id, "language", object())
    nested_value: object = 0
    for _ in range(mcp_server.MAX_MCP_DOCUMENT_METADATA_JSON_DEPTH + 1):
        nested_value = [nested_value]
    with pytest.raises(ToolError, match="nesting limit"):
        mcp_server.set_document_metadata(document_id, "language", nested_value)
    with pytest.raises(DocumentError, match="unknown document"):
        mcp_server.set_document_metadata("00000000-0000-0000-0000-000000000000", "language", "en")

    with Project.open(root, read_only=True) as unchanged:
        assert unchanged.document_metadata(document_id) == {}


def test_mcp_document_metadata_is_readable_but_not_writable_in_read_only_mode(
    tmp_path, monkeypatch
):
    root = tmp_path / "readonly-metadata-project"
    expected_value = {"labels": ["speech", "public"], "reviewed": True}
    expected_metadata = {"research": expected_value}
    with Project.init(root) as project:
        document_id, _ = project.import_text("source.txt", "A source sentence.")
        project.set_document_metadata(document_id, "research", expected_value)
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", True)

    try:
        asyncio.run(mcp_server._configure_read_only_server())
        names = {tool.name for tool in asyncio.run(mcp_server.server.list_tools())}
        assert "set_document_metadata" not in names
        assert "get_document_metadata" in names
        assert mcp_server.get_document_metadata(document_id) == expected_metadata
        with pytest.raises(ProjectError, match="read-only mode"):
            mcp_server.set_document_metadata(document_id, "language", "en")
        assert database.read_bytes() == before
    finally:
        monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", False)
        mcp_server._restore_full_server_registry()
