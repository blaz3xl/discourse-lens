from __future__ import annotations

import asyncio

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError


def test_get_document_revision_matches_core_metadata_and_is_read_only(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "research"
    project = Project.init(root)
    _, revision_id = project.import_text(
        "metadata-only.txt", "PRIVATE DOCUMENT BODY must not be returned."
    )
    expected = project.document_revision_summary(revision_id)
    project.close()

    hidden = root / ".discourselens"
    files_before = {
        path.relative_to(hidden): path.read_bytes()
        for path in hidden.rglob("*")
        if path.is_file()
    }
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    try:
        asyncio.run(mcp_module._configure_read_only_server())
        tool = next(
            item for item in asyncio.run(mcp_module.server.list_tools())
            if item.name == "get_document_revision"
        )
        schema = tool.model_dump(by_alias=True)["inputSchema"]
        assert schema["required"] == ["revision_id"]
        assert set(schema["properties"]) == {"revision_id"}

        result = mcp_module.get_document_revision(revision_id)
        with pytest.raises(ProjectError):
            mcp_module.get_document_revision("unknown-revision")
    finally:
        monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
        mcp_module._restore_full_server_registry()

    assert result == expected
    assert set(result) == {
        "revision_id",
        "document_id",
        "display_name",
        "source_hash",
        "canonical_hash",
        "extractor",
        "mime_type",
        "source_size",
        "created_at",
        "ocr_language",
        "ocr_dpi",
        "page_count",
        "paragraph_count",
    }
    serialized = repr(result)
    assert "PRIVATE DOCUMENT BODY" not in serialized
    assert str(root) not in serialized
    assert not {"path", "source_path", "canonical_text", "text"} & set(result)
    files_after = {
        path.relative_to(hidden): path.read_bytes()
        for path in hidden.rglob("*")
        if path.is_file()
    }
    assert files_after == files_before
