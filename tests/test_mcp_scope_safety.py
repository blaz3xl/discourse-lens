from __future__ import annotations

import asyncio
import base64
import sqlite3

import pytest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import DocumentError, ProjectError


def test_mcp_object_ids_cannot_read_across_configured_projects(tmp_path, monkeypatch):
    active_root = tmp_path / "active-project"
    active = Project.init(active_root)
    _, active_revision = active.import_text("active.txt", "Visible only in the active project.")
    active_paragraph = active.db.execute(
        "SELECT id FROM paragraphs WHERE revision_id=?", (active_revision,)
    ).fetchone()[0]
    active.close()

    foreign = Project.init(tmp_path / "foreign-project")
    _, foreign_revision = foreign.import_text("foreign.txt", "Secret from another project.")
    foreign_paragraph = foreign.db.execute(
        "SELECT id FROM paragraphs WHERE revision_id=?", (foreign_revision,)
    ).fetchone()[0]
    foreign.close()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(active_root))

    assert mcp_module.get_paragraph(active_paragraph)["text"] == (
        "Visible only in the active project."
    )
    with pytest.raises(DocumentError, match="unknown paragraph"):
        mcp_module.get_paragraph(foreign_paragraph)


@pytest.mark.parametrize("configured_import_root", ["", "   "])
def test_mcp_blank_import_root_falls_back_to_project_root(
    tmp_path, monkeypatch, configured_import_root
):
    project_root = tmp_path / "project"
    Project.init(project_root).close()
    inside = project_root / "inside.txt"
    inside.write_text("inside project", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside project", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(project_root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", configured_import_root)

    filename, content, failure_code = mcp_module._read_mcp_import_source(
        str(inside), mcp_module._configured_import_root(), max_bytes=1_024
    )
    assert filename == "inside.txt"
    assert content == b"inside project"
    assert failure_code is None

    filename, content, failure_code = mcp_module._read_mcp_import_source(
        str(outside), mcp_module._configured_import_root(), max_bytes=1_024
    )
    assert filename is None
    assert content is None
    assert failure_code == "OUTSIDE_IMPORT_ROOT"


def test_mcp_bundle_chunk_keeps_reading_the_pinned_manifest_after_path_swap(
    tmp_path, monkeypatch
):
    root = tmp_path / "bundle-project"
    Project.init(root).close()
    bundle_id = "a" * 32
    bundle_dir = root / ".discourselens" / "mcp_exports" / bundle_id
    bundle_dir.mkdir(parents=True)
    manifest = bundle_dir / "manifest.json"
    managed_bytes = b"managed bundle bytes"
    manifest.write_bytes(managed_bytes)
    outside = tmp_path / "outside-secret.txt"
    outside.write_bytes(b"outside-secret-sentinel")
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    read_manifest = mcp_module._read_managed_bundle_bytes

    def swap_path_after_read(
        bundle_id: str, *, offset: int = 0, max_bytes: int | None = None
    ) -> tuple[bytes, int]:
        result = read_manifest(bundle_id, offset=offset, max_bytes=max_bytes)
        manifest.rename(bundle_dir / "pinned-manifest.json")
        manifest.symlink_to(outside)
        return result

    monkeypatch.setattr(mcp_module, "_read_managed_bundle_bytes", swap_path_after_read)
    chunk = mcp_module.get_research_bundle_chunk(bundle_id, max_bytes=128)

    assert base64.b64decode(str(chunk["data_base64"])) == managed_bytes
    assert chunk["total_bytes"] == len(managed_bytes)


def test_mcp_bundle_verifier_uses_pinned_bytes_after_managed_path_swap(
    tmp_path, monkeypatch
):
    root = tmp_path / "verified-bundle-project"
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "A bundle verifier fixture.")
    run_id = project.create_run(project.snapshot([revision_id]), {"profile": "fixture"})
    project.start_run(run_id)
    project.finalize_run(run_id)
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    exported = mcp_module.export_research_bundle(run_id)
    bundle_id = str(exported["bundle_id"])
    _, manifest = mcp_module._managed_bundle_paths(bundle_id)
    outside = tmp_path / "outside-secret.txt"
    outside.write_bytes(b"outside-secret-sentinel")
    original_bytes = manifest.read_bytes()

    read_manifest = mcp_module._read_managed_bundle_bytes

    def swap_path_after_read(
        bundle_id: str, *, offset: int = 0, max_bytes: int | None = None
    ) -> tuple[bytes, int]:
        result = read_manifest(bundle_id, offset=offset, max_bytes=max_bytes)
        manifest.rename(manifest.with_name("pinned-manifest.json"))
        manifest.symlink_to(outside)
        return result

    monkeypatch.setattr(mcp_module, "_read_managed_bundle_bytes", swap_path_after_read)
    verified = mcp_module.verify_research_bundle(bundle_id)

    assert verified["status"] == "VALID"
    assert mcp_module.verify_bundle(original_bytes)["status"] == "VALID"


def test_mcp_managed_bundle_reads_fail_closed_without_secure_descriptor_support(
    tmp_path, monkeypatch
):
    root = tmp_path / "unsupported-bundle-project"
    Project.init(root).close()
    bundle_id = "b" * 32
    bundle_dir = root / ".discourselens" / "mcp_exports" / bundle_id
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "manifest.json").write_bytes(b"managed bundle bytes")
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_supports_secure_import_open", lambda: False)

    with pytest.raises(ProjectError, match="secure descriptor-relative file access"):
        mcp_module.get_research_bundle_chunk(bundle_id)
    with pytest.raises(ProjectError, match="secure descriptor-relative file access"):
        mcp_module.verify_research_bundle(bundle_id)


def test_mcp_read_only_hides_every_non_allowlisted_tool_and_blocks_direct_writes(
    tmp_path, monkeypatch
):
    root = tmp_path / "read-only-project"
    project = Project.init(root)
    project.create_corpus("visible corpus")
    project.close()
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    async def exercise_restricted_surface() -> None:
        await mcp_module._configure_read_only_server()
        exposed_names = {tool.name for tool in await mcp_module.server.list_tools()}
        assert exposed_names == mcp_module.READ_ONLY_MCP_TOOL_NAMES

        hidden_names = set(mcp_module._FULL_MCP_TOOLS) - exposed_names
        assert {"import_document", "create_corpus", "run_controlled_cda_review"} <= hidden_names
        for name in sorted(hidden_names):
            with pytest.raises(Exception, match=f"Unknown tool: {name}"):
                await mcp_module.server.call_tool(name, {})

    mcp_module._MCP_READ_ONLY = True
    try:
        asyncio.run(exercise_restricted_surface())
        with pytest.raises(sqlite3.OperationalError):
            mcp_module.create_corpus("blocked corpus")
    finally:
        mcp_module._MCP_READ_ONLY = False
        mcp_module._restore_full_server_registry()

    assert database.read_bytes() == before
    with Project.open(root, read_only=True) as unchanged:
        assert [row["name"] for row in unchanged.list_corpora()] == ["visible corpus"]


def test_mcp_tool_schemas_do_not_expose_project_or_access_mode_selectors():
    tools = asyncio.run(mcp_module.server.list_tools())
    forbidden_selectors = {
        "project",
        "project_root",
        "project_path",
        "import_root",
        "read_only",
        "restricted",
    }

    for tool in tools:
        schema = tool.model_dump(by_alias=True)["inputSchema"]
        properties = schema.get("properties", {})
        assert isinstance(properties, dict)
        assert not (set(properties) & forbidden_selectors), tool.name

    scoped_schema = next(
        tool.model_dump(by_alias=True)["inputSchema"]
        for tool in tools
        if tool.name == "search_exact"
    )
    assert {"revision_id", "phrase"} <= set(scoped_schema["required"])


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("corpus_frequency", {"revision_id": "fixture", "offset": True}),
        ("get_page", {"revision_id": "fixture", "page_index": True}),
        (
            "search_vector",
            {"revision_id": "fixture", "artifact_ids": [], "vector_query": [True]},
        ),
    ],
)
def test_mcp_tool_parser_rejects_booleans_for_numeric_inputs(tool_name, arguments):
    async def invoke_invalid_arguments():
        with pytest.raises(ToolError) as error:
            await mcp_module.server.call_tool(tool_name, arguments)
        assert not isinstance(error.value, UnexpectedToolError)

    asyncio.run(invoke_invalid_arguments())
