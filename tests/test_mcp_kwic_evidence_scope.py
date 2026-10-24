from __future__ import annotations

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import EvidenceValidationError, ProjectError


def test_mcp_scoped_kwic_evidence_persists_selected_hit_without_changing_scope(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "kwic-evidence-scope"
    project = Project.init(root)
    _, revision_a = project.import_text(
        "a.txt", "Migration begins in revision A. Migration continues."
    )
    _, revision_b = project.import_text("b.txt", "Migration appears in revision B.")
    _, outside_revision = project.import_text("outside.txt", "Migration remains outside.")
    scope_id = project.snapshot([revision_a, revision_b])
    expected_members = project.scope_revisions(scope_id)
    snapshot_count_before = project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
    page = project.corpus_kwic_page(scope_id, "migration", window=0)
    selected_hit = next(row for row in page["rows"] if row["revision_id"] == revision_a)
    project.close()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    def forbid_snapshot(*_args, **_kwargs):
        pytest.fail("scoped KWIC evidence must not create a new ScopeSnapshot")

    monkeypatch.setattr(Project, "snapshot", forbid_snapshot)
    result = mcp_module.corpus_kwic_evidence_for_scope(
        scope_id,
        revision_a,
        "MIGRATION",
        selected_hit["start_offset"],
        selected_hit["end_offset"],
    )

    assert result["scope_id"] == scope_id
    assert result["revision_id"] == revision_a
    assert result["query"] == "MIGRATION"
    assert result["exact_text"] == "Migration"
    assert result["citation"] == "[a.txt · ¶1]"
    assert result["evidence_id"]
    assert result["coverage_id"]
    assert json.loads(json.dumps(result)) == result

    with Project.open(root, read_only=True) as reopened:
        assert reopened.scope_revisions(scope_id) == expected_members
        assert outside_revision not in expected_members
        assert (
            reopened.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
            == snapshot_count_before
        )
        evidence_row = reopened.db.execute(
            "SELECT e.kind,e.span_id,s.revision_id,s.start_offset,s.end_offset "
            "FROM evidence_artifacts e JOIN source_spans s ON s.id=e.span_id WHERE e.id=?",
            (result["evidence_id"],),
        ).fetchone()
        assert evidence_row is not None
        assert evidence_row["kind"] == "TEXTUAL"
        assert evidence_row["revision_id"] == revision_a
        assert evidence_row["start_offset"] == selected_hit["start_offset"]
        assert evidence_row["end_offset"] == selected_hit["end_offset"]
        assert reopened.span_text(evidence_row["span_id"]) == "Migration"
        provenance = reopened.evidence_content(str(result["evidence_id"]))
        assert provenance["acquisition_method"] == "corpus_kwic"
        assert provenance["scope_snapshot_id"] == scope_id
        assert provenance["coverage_descriptor_id"] == result["coverage_id"]
        coverage = reopened.coverage_descriptor(str(result["coverage_id"]))
        assert coverage.scope_snapshot_id == scope_id


def test_mcp_scoped_kwic_evidence_rejects_out_of_scope_and_nonhit_offsets(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "kwic-evidence-invalid-hit"
    project = Project.init(root)
    _, included_revision = project.import_text("included.txt", "Migration starts here.")
    _, outside_revision = project.import_text("outside.txt", "Migration is excluded.")
    scope_id = project.snapshot([included_revision])
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(EvidenceValidationError, match="not a KWIC hit"):
        mcp_module.corpus_kwic_evidence_for_scope(
            scope_id, outside_revision, "Migration", 0, 9
        )
    with pytest.raises(EvidenceValidationError, match="not a KWIC hit"):
        mcp_module.corpus_kwic_evidence_for_scope(
            scope_id, included_revision, "Migration", 10, 16
        )
    with pytest.raises(EvidenceValidationError, match="not a KWIC hit"):
        mcp_module.corpus_kwic_evidence_for_scope(
            scope_id, included_revision, "Migration", 0, 8
        )

    with Project.open(root, read_only=True) as unchanged:
        assert unchanged.db.execute("SELECT COUNT(*) FROM evidence_artifacts").fetchone()[0] == 0
        assert unchanged.db.execute("SELECT COUNT(*) FROM coverage_descriptors").fetchone()[0] == 0
        assert unchanged.scope_revisions(scope_id) == [included_revision]


@pytest.mark.parametrize(
    ("scope_id", "revision_id", "query", "start_offset", "end_offset"),
    [
        (" ", "revision", "Migration", 0, 9),
        ("s" * 129, "revision", "Migration", 0, 9),
        ("scope", " ", "Migration", 0, 9),
        ("scope", "revision", " ", 0, 9),
        ("scope", "revision", "Migration outside", 0, 9),
        ("scope", "revision", "Migration", True, 9),
        ("scope", "revision", "Migration", -1, 9),
        ("scope", "revision", "Migration", 9, 9),
        ("scope", "revision", "Migration", 0, 9_223_372_036_854_775_808),
    ],
)
def test_mcp_scoped_kwic_evidence_rejects_invalid_arguments_before_opening_project(
    monkeypatch, scope_id, revision_id, query, start_offset, end_offset
) -> None:
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
    monkeypatch.setattr(
        mcp_module,
        "_open_project",
        lambda: pytest.fail("invalid scoped KWIC evidence arguments must be rejected first"),
    )
    with pytest.raises(ProjectError):
        mcp_module.corpus_kwic_evidence_for_scope(
            scope_id, revision_id, query, start_offset, end_offset
        )


def test_mcp_scoped_kwic_evidence_schema_is_mutating_and_explicit(monkeypatch) -> None:
    tools = {tool.name: tool for tool in asyncio.run(mcp_module.server.list_tools())}
    tool = tools["corpus_kwic_evidence_for_scope"]
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    assert set(schema["required"]) == {
        "scope_id",
        "revision_id",
        "query",
        "start_offset",
        "end_offset",
    }
    properties = schema["properties"]
    assert properties["scope_id"]["minLength"] == 1
    assert properties["scope_id"]["maxLength"] == 128
    assert properties["revision_id"]["minLength"] == 1
    assert properties["revision_id"]["maxLength"] == 128
    assert properties["query"]["minLength"] == 1
    assert properties["query"]["maxLength"] == mcp_module.MCP_KWIC_MAX_QUERY_CHARS
    for name in ("start_offset", "end_offset"):
        assert properties[name]["minimum"] == 0
        assert properties[name]["maximum"] == mcp_module.MCP_KWIC_EVIDENCE_MAX_OFFSET
    assert "corpus_kwic_evidence_for_scope" not in mcp_module.READ_ONLY_MCP_TOOL_NAMES

    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    monkeypatch.setattr(
        mcp_module,
        "_open_project",
        lambda: pytest.fail("read-only KWIC evidence must be rejected before project open"),
    )
    with pytest.raises(ProjectError, match="read-only mode"):
        mcp_module.corpus_kwic_evidence_for_scope("scope", "revision", "term", 0, 4)

    async def inspect_read_only_tools() -> None:
        await mcp_module._configure_read_only_server()
        exposed_names = {item.name for item in await mcp_module.server.list_tools()}
        assert "corpus_kwic_evidence_for_scope" not in exposed_names
        with pytest.raises(ToolError, match="Unknown tool: corpus_kwic_evidence_for_scope"):
            await mcp_module.server.call_tool("corpus_kwic_evidence_for_scope", {})

    try:
        asyncio.run(inspect_read_only_tools())
    finally:
        monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
        mcp_module._restore_full_server_registry()
