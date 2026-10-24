from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from typing import cast

import pytest

from discourselens import Project, mcp_server
from discourselens.errors import ProjectError
from discourselens.types import EvidenceRole


def _project_with_run(root):
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "A local research bundle fixture.")
    run_id = project.create_run(project.snapshot([revision_id]), {"profile": "bundle-fixture"})
    project.start_run(run_id)
    project.finalize_run(run_id)
    project.close()
    return run_id


def test_mcp_export_bundle_chunks_and_verifies_inside_project(tmp_path, monkeypatch):
    root = tmp_path / "bundle-project"
    run_id = _project_with_run(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    exported = mcp_server.export_research_bundle(run_id)
    size_bytes = cast(int, exported["size_bytes"])
    assert set(exported) == {"bundle_id", "size_bytes", "sha256", "max_chunk_bytes"}
    assert exported["max_chunk_bytes"] == mcp_server.MAX_MCP_BUNDLE_CHUNK_BYTES
    assert "path" not in exported

    bundle_bytes = bytearray()
    offset = 0
    while True:
        page = mcp_server.get_research_bundle_chunk(
            str(exported["bundle_id"]), offset=offset, max_bytes=101
        )
        chunk = base64.b64decode(str(page["data_base64"]))
        assert len(chunk) <= 101
        assert page["offset"] == offset
        bundle_bytes.extend(chunk)
        offset = cast(int, page["next_offset"])
        if not page["has_more"]:
            break

    assert len(bundle_bytes) == size_bytes
    assert json.loads(bundle_bytes)["analysis_run"]["id"] == run_id
    verified = mcp_server.verify_research_bundle(str(exported["bundle_id"]))
    assert verified["status"] == "VALID"
    assert verified["integrity"] == "VERIFIED_SHA256"
    assert list((root / ".discourselens" / "mcp_exports").iterdir())


def test_mcp_bundle_full_text_requires_opt_in_but_keeps_cited_excerpt(
    tmp_path, monkeypatch
):
    root = tmp_path / "bundle-text-project"
    project = Project.init(root)
    source_text = "Private source corpus text."
    _, revision_id = project.import_text("source.txt", source_text)
    run_id = project.create_run(project.snapshot([revision_id]), {"profile": "bundle-text"})
    evidence_id = project.textual_evidence(
        project.create_span(revision_id, 0, len(source_text))
    )
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "text", "Source text exists."
    )
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.start_run(run_id)
    project.finalize_run(run_id)
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    default_export = mcp_server.export_research_bundle(run_id)
    opted_export = mcp_server.export_research_bundle(run_id, include_canonical_text=True)

    def read_bundle(exported):
        contents = bytearray()
        offset = 0
        while True:
            page = mcp_server.get_research_bundle_chunk(
                str(exported["bundle_id"]), offset=offset
            )
            contents.extend(base64.b64decode(str(page["data_base64"])))
            offset = cast(int, page["next_offset"])
            if not page["has_more"]:
                return json.loads(contents)

    default_payload = read_bundle(default_export)
    opted_payload = read_bundle(opted_export)
    assert default_payload["canonical_texts"] == []
    assert default_payload["evidence"][finding_id][0]["text"] == source_text
    assert opted_payload["canonical_texts"] == [
        {
            "revision_id": revision_id,
            "sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "text": source_text,
        }
    ]

    tools = {item.name: item for item in asyncio.run(mcp_server.server.list_tools())}
    export_tool = tools["export_research_bundle"]
    assert "evidence excerpts remain included" in str(export_tool.description)
    schema = export_tool.model_dump(by_alias=True)["inputSchema"]
    assert schema["properties"]["include_canonical_text"]["type"] == "boolean"
    assert schema["properties"]["include_canonical_text"]["default"] is False
    assert "include_canonical_text" not in schema["required"]


def test_mcp_bundle_ids_are_project_scoped_and_reject_traversal(tmp_path, monkeypatch):
    first_root = tmp_path / "first"
    run_id = _project_with_run(first_root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(first_root))
    first_project = mcp_server.export_research_bundle(run_id)
    second_root = tmp_path / "second"
    _project_with_run(second_root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(second_root))

    with pytest.raises(ProjectError, match="unavailable"):
        mcp_server.verify_research_bundle(str(first_project["bundle_id"]))
    with pytest.raises(ProjectError, match="ID is invalid"):
        mcp_server.get_research_bundle_chunk("../manifest.json")
    with pytest.raises(ProjectError, match="ID is invalid"):
        mcp_server.verify_research_bundle("../manifest.json")


def test_mcp_bundle_export_denied_read_only_but_existing_bundle_reads_are_allowlisted(
    tmp_path, monkeypatch
):
    root = tmp_path / "readonly"
    run_id = _project_with_run(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", True)
    try:
        with pytest.raises(ProjectError, match="read-only"):
            mcp_server.export_research_bundle(run_id)
        asyncio.run(mcp_server._configure_read_only_server())
        names = {item.name for item in asyncio.run(mcp_server.server.list_tools())}
        assert "get_research_bundle_chunk" in names
        assert "verify_research_bundle" in names
        assert "export_research_bundle" not in names
    finally:
        monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", False)
        mcp_server._restore_full_server_registry()


def test_mcp_bundle_tampering_is_rejected_and_chunk_bounds_are_enforced(tmp_path, monkeypatch):
    root = tmp_path / "tampered"
    run_id = _project_with_run(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    exported = mcp_server.export_research_bundle(run_id)
    size_bytes = cast(int, exported["size_bytes"])
    bundle_id = str(exported["bundle_id"])
    chunk = mcp_server.get_research_bundle_chunk(bundle_id, max_bytes=32 * 1024)
    assert len(base64.b64decode(str(chunk["data_base64"]))) <= 32 * 1024

    with pytest.raises(ProjectError, match="chunk size"):
        mcp_server.get_research_bundle_chunk(bundle_id, max_bytes=32 * 1024 + 1)
    with pytest.raises(ProjectError, match="offset"):
        mcp_server.get_research_bundle_chunk(bundle_id, offset=size_bytes + 1)
    monkeypatch.setattr(mcp_server, "MAX_BUNDLE_MANIFEST_BYTES", size_bytes - 1)
    with pytest.raises(ProjectError, match="size limit"):
        mcp_server.get_research_bundle_chunk(bundle_id)
    monkeypatch.setattr(mcp_server, "MAX_BUNDLE_MANIFEST_BYTES", 64 * 1024 * 1024)

    _, manifest = mcp_server._managed_bundle_paths(bundle_id)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["analysis_run"]["status"] = "ALTERED"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProjectError, match="integrity check failed"):
        mcp_server.verify_research_bundle(bundle_id)


def test_mcp_bundle_export_rejects_unknown_run_and_symlinked_managed_storage(tmp_path, monkeypatch):
    root = tmp_path / "unsafe"
    _project_with_run(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    with pytest.raises(ProjectError):
        mcp_server.export_research_bundle("not-a-run")

    managed_root = root / ".discourselens" / "mcp_exports"
    managed_root.rmdir()
    managed_root.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ProjectError, match="unsafe"):
        mcp_server.export_research_bundle("not-a-run")
