from __future__ import annotations

import asyncio
import json

from discourselens import Project, mcp_server


def test_mcp_finding_sources_is_read_only_and_returns_ids_without_text(tmp_path, monkeypatch):
    root = tmp_path / "finding-lineage-project"
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "Private corpus source material.")
    run_id = project.create_run(project.snapshot([revision_id]), {"fixture": "lineage"})
    project.start_run(run_id)
    source_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "source-observation",
        "Private source observation text.",
    )
    derived_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "derived-observation",
        "Private derived finding text.",
    )
    empty_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "unlinked-observation",
        "Private unlinked finding text.",
    )
    project.link_finding_source(derived_id, source_id, "SYNTHESIZES")
    project.finalize_run(run_id)
    project.close()
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", True)

    try:
        asyncio.run(mcp_server._configure_read_only_server())
        names = {tool.name for tool in asyncio.run(mcp_server.server.list_tools())}
        assert "get_finding_sources" in names
        linked = mcp_server.get_finding_sources(derived_id)
        unlinked = mcp_server.get_finding_sources(empty_id)
    finally:
        monkeypatch.setattr(mcp_server, "_MCP_READ_ONLY", False)
        mcp_server._restore_full_server_registry()

    assert linked == {
        "finding_id": derived_id,
        "sources": [{"source_finding_id": source_id, "relation": "SYNTHESIZES"}],
    }
    assert unlinked == {"finding_id": empty_id, "sources": []}
    serialized = json.dumps([linked, unlinked])
    assert "Private corpus source material" not in serialized
    assert "Private source observation text" not in serialized
    assert "Private derived finding text" not in serialized
    assert database.read_bytes() == before
