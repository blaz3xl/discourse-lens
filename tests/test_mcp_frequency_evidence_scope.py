from __future__ import annotations

import asyncio
from typing import cast

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError, ScopeViolationError


def _persistence_counts(root) -> tuple[int, int, int]:
    with Project.open(root, read_only=True) as project:
        coverage_count = project.db.execute(
            "SELECT COUNT(*) FROM coverage_descriptors"
        ).fetchone()[0]
        evidence_count = project.db.execute(
            "SELECT COUNT(*) FROM evidence_artifacts"
        ).fetchone()[0]
        snapshot_count = project.db.execute(
            "SELECT COUNT(*) FROM scope_snapshots"
        ).fetchone()[0]
    return coverage_count, evidence_count, snapshot_count


def test_mcp_frequency_evidence_for_scope_persists_exhaustive_original_scope(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "scope-frequency-evidence"
    project = Project.init(root)
    _, revision_a = project.import_text("a.txt", "Policy policy opening")
    _, revision_b = project.import_text("b.txt", "policy debate")
    scope_id = project.snapshot([revision_a, revision_b])
    expected_members = project.scope_revisions(scope_id)
    snapshot_count = project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
    _, later_revision = project.import_text("later.txt", "policy outsideonly")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)

    def forbid_snapshot(*_args, **_kwargs):
        pytest.fail("scope frequency evidence must not create a ScopeSnapshot")

    monkeypatch.setattr(Project, "snapshot", forbid_snapshot)
    result = mcp_module.corpus_frequency_evidence_for_scope(scope_id, "Policy")
    zero_result = mcp_module.corpus_frequency_evidence_for_scope(scope_id, "mars")

    assert set(result) == {
        "term",
        "count",
        "population_count",
        "scope_id",
        "coverage_id",
        "evidence_id",
    }
    assert result["term"] == "policy"
    assert result["count"] == 3
    assert result["population_count"] == 5
    assert result["scope_id"] == scope_id
    assert zero_result["term"] == "mars"
    assert zero_result["count"] == 0
    assert zero_result["population_count"] == 5
    assert zero_result["scope_id"] == scope_id

    with Project.open(root, read_only=True) as reopened:
        assert reopened.scope_revisions(scope_id) == expected_members
        assert later_revision not in expected_members
        assert reopened.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == (
            snapshot_count
        )
        for evidence_result in (result, zero_result):
            coverage = reopened.coverage_descriptor(str(evidence_result["coverage_id"]))
            evidence = reopened.evidence_content(str(evidence_result["evidence_id"]))
            assert coverage.coverage_mode.value == "EXHAUSTIVE"
            assert coverage.scope_snapshot_id == scope_id
            assert coverage.population_count == 5
            assert coverage.inspection_count == 5
            assert evidence["scope_snapshot_id"] == scope_id
            assert evidence["value"] == evidence_result["count"]
            provenance = cast(dict[str, object], evidence["provenance"])
            assert provenance["coverage_id"] == evidence_result["coverage_id"]
            assert provenance["population_tokens"] == 5
        zero_coverage = reopened.coverage_descriptor(str(zero_result["coverage_id"]))
        assert zero_coverage.negative_result_status.value == (
            "ABSENCE_IN_EXHAUSTIVE_POPULATION"
        )


@pytest.mark.parametrize(
    ("scope_id", "term"),
    [
        ("", "policy"),
        ("   ", "policy"),
        (" padded ", "policy"),
        ("s" * 129, "policy"),
        (True, "policy"),
        ("scope", ""),
        ("scope", "   "),
        ("scope", "two tokens"),
        ("scope", "!!!"),
        ("scope", "x" * (mcp_module.MCP_KWIC_MAX_QUERY_CHARS + 1)),
        ("scope", True),
        ("scope", None),
    ],
)
def test_mcp_scope_frequency_evidence_rejects_invalid_values_before_open(
    monkeypatch, scope_id, term
) -> None:
    monkeypatch.setattr(
        mcp_module,
        "_open_project",
        lambda: pytest.fail("invalid scope frequency evidence input opened the project"),
    )
    with pytest.raises(ProjectError):
        mcp_module.corpus_frequency_evidence_for_scope(scope_id, term)


def test_mcp_scope_frequency_evidence_failures_create_no_orphan_records(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "scope-frequency-evidence-failures"
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "policy statement")
    scope_id = project.snapshot([revision_id])
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
    before = _persistence_counts(root)

    with pytest.raises(ScopeViolationError, match="scope"):
        mcp_module.corpus_frequency_evidence_for_scope("unknown-scope", "policy")
    with pytest.raises(ProjectError, match="exactly one surface token"):
        mcp_module.corpus_frequency_evidence_for_scope(scope_id, "two tokens")

    assert _persistence_counts(root) == before


def test_mcp_scope_frequency_evidence_is_registered_but_hidden_read_only(monkeypatch) -> None:
    tools = asyncio.run(mcp_module.server.list_tools())
    names = {tool.name for tool in tools}
    assert "corpus_frequency_evidence_for_scope" in names
    assert "corpus_frequency_evidence_for_scope" not in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    tool = next(item for item in tools if item.name == "corpus_frequency_evidence_for_scope")
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    assert schema["required"] == ["scope_id", "term"]
    assert schema["properties"]["scope_id"]["minLength"] == 1
    assert schema["properties"]["scope_id"]["maxLength"] == 128
    assert schema["properties"]["term"]["minLength"] == 1
    assert schema["properties"]["term"]["maxLength"] == (
        mcp_module.MCP_KWIC_MAX_QUERY_CHARS
    )

    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    monkeypatch.setattr(
        mcp_module,
        "_open_project",
        lambda: pytest.fail("read-only frequency evidence must be rejected before opening"),
    )
    try:
        with pytest.raises(ProjectError, match="read-only"):
            mcp_module.corpus_frequency_evidence_for_scope("scope", "policy")

        asyncio.run(mcp_module._configure_read_only_server())
        exposed_names = {item.name for item in asyncio.run(mcp_module.server.list_tools())}
        assert "corpus_frequency_evidence_for_scope" not in exposed_names
        with pytest.raises(Exception, match="Unknown tool: corpus_frequency_evidence_for_scope"):
            asyncio.run(
                mcp_module.server.call_tool(
                    "corpus_frequency_evidence_for_scope",
                    {"scope_id": "scope", "term": "policy"},
                )
            )
    finally:
        monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
        mcp_module._restore_full_server_registry()
