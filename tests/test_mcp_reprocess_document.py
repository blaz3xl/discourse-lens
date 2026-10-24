from __future__ import annotations

import asyncio
import json

import fitz
import pytest
from mcp.server.mcpserver.exceptions import ToolError

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError


def _write_pdf(path, text: str) -> None:
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def test_mcp_reprocess_document_creates_revision_without_returning_source_path(
    tmp_path, monkeypatch
):
    root = tmp_path / "research"
    inbox = root / "inbox"
    inbox.mkdir(parents=True)
    original_path = inbox / "original.pdf"
    replacement_path = inbox / "replacement.pdf"
    _write_pdf(original_path, "Original source revision.")
    _write_pdf(replacement_path, "Replacement source revision.")

    project = Project.init(root)
    original = project.import_document(original_path)
    project.close()
    assert original.document_id is not None
    assert original.revision_id is not None
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(inbox))

    result = mcp_module.reprocess_document(
        original.document_id,
        str(replacement_path),
        extractor_suffix=".mcp-reprocessed",
        max_bytes=replacement_path.stat().st_size,
    )

    assert result["status"] == "SUCCESS"
    assert result["document_id"] == original.document_id
    assert result["revision_id"] != original.revision_id
    assert result["warnings"] == ()
    assert str(inbox) not in json.dumps(result)
    assert str(replacement_path) not in json.dumps(result)
    with Project.open(root, read_only=True) as reopened:
        revision = reopened.db.execute(
            "SELECT extractor FROM document_revisions WHERE id=?", (result["revision_id"],)
        ).fetchone()
        assert revision["extractor"].endswith(".mcp-reprocessed")
        assert reopened.revision_text(str(result["revision_id"])) == "Replacement source revision."


def test_mcp_reprocess_document_enforces_import_root_before_opening_project(
    tmp_path, monkeypatch
):
    root = tmp_path / "research"
    inbox = root / "inbox"
    inbox.mkdir(parents=True)
    outside = tmp_path / "private-source.pdf"
    _write_pdf(outside, "Outside import root.")
    Project.init(root).close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(inbox))

    def unexpected_project_open():
        pytest.fail("an invalid source path must be rejected before opening Core")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)
    with pytest.raises(ProjectError, match="outside the configured import root") as error:
        mcp_module.reprocess_document("document-id", str(outside))
    assert str(outside) not in str(error.value)

    missing_path = inbox / "missing.pdf"
    with pytest.raises(ProjectError, match="not a readable file"):
        mcp_module.reprocess_document("document-id", str(missing_path))

    link = inbox / "linked.pdf"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable or not permitted: {exc}")
    with pytest.raises(ProjectError, match="outside the configured import root"):
        mcp_module.reprocess_document("document-id", str(link))


def test_mcp_reprocess_document_is_hidden_and_rejected_in_read_only_mode(
    tmp_path, monkeypatch
):
    root = tmp_path / "research"
    Project.init(root).close()
    source = tmp_path / "source.pdf"
    _write_pdf(source, "Must not be reprocessed in read-only mode.")
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    def unexpected_project_open():
        pytest.fail("read-only reprocessing must be rejected before opening Core")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)
    try:
        with pytest.raises(ProjectError, match="read-only"):
            mcp_module.reprocess_document("document-id", str(source))

        async def tool_names() -> set[str]:
            await mcp_module._configure_read_only_server()
            return {tool.name for tool in await mcp_module.server.list_tools()}

        assert "reprocess_document" not in asyncio.run(tool_names())
    finally:
        monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
        mcp_module._restore_full_server_registry()

    tools = asyncio.run(mcp_module.server.list_tools())
    tool = next(item for item in tools if item.name == "reprocess_document")
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    assert {"document_id", "path"} <= set(schema["required"])
    assert "path" in schema["properties"]
    assert "reprocess_document" not in mcp_module.READ_ONLY_MCP_TOOL_NAMES


def test_mcp_reprocess_document_hides_core_errors_at_tool_boundary(tmp_path, monkeypatch):
    root = tmp_path / "research"
    inbox = root / "inbox"
    inbox.mkdir(parents=True)
    source = inbox / "source.pdf"
    _write_pdf(source, "Core error boundary.")
    Project.init(root).close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(inbox))

    async def invoke() -> None:
        with pytest.raises(ToolError) as error:
            await mcp_module.server.call_tool(
                "reprocess_document",
                {"document_id": "missing-document", "path": str(source)},
            )
        assert str(source) not in str(error.value)

    asyncio.run(invoke())
