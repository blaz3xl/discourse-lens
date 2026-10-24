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


def test_mcp_lexical_bundle_evidence_for_scope_uses_frozen_scope_and_legacy_tokenizer(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "scope-lexical-bundle-evidence"
    project = Project.init(root)
    _, revision_a = project.import_text("a.txt", "red blue red state-of-art policy")
    _, revision_b = project.import_text("b.txt", "green gold")
    scope_id = project.snapshot([revision_a, revision_b])
    expected_members = project.scope_revisions(scope_id)
    snapshot_count = project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
    _, later_revision = project.import_text(
        "later.txt", "state-of-art policy state-of-art policy"
    )
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)

    def forbid_snapshot(*_args, **_kwargs):
        pytest.fail("scoped lexical-bundle evidence must not create a ScopeSnapshot")

    monkeypatch.setattr(Project, "snapshot", forbid_snapshot)
    result = mcp_module.corpus_lexical_bundle_evidence_for_scope(
        scope_id, "state-of-art policy", 2
    )
    boundary_zero = mcp_module.corpus_lexical_bundle_evidence_for_scope(
        scope_id, "policy green", 2
    )

    assert result["bundle"] == "state-of-art policy"
    assert result["size"] == 2
    assert result["count"] == 1
    assert result["population_count"] == 5
    assert result["scope_id"] == scope_id
    assert boundary_zero["bundle"] == "policy green"
    assert boundary_zero["count"] == 0
    assert boundary_zero["population_count"] == 5
    assert boundary_zero["scope_id"] == scope_id

    with Project.open(root, read_only=True) as reopened:
        assert reopened.scope_revisions(scope_id) == expected_members
        assert later_revision not in expected_members
        assert reopened.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == (
            snapshot_count
        )
        for evidence_result in (result, boundary_zero):
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
            assert provenance["calculation_version"] == "surface-lexical-bundles-v1-size-2"
        zero_coverage = reopened.coverage_descriptor(str(boundary_zero["coverage_id"]))
        assert zero_coverage.negative_result_status.value == (
            "ABSENCE_IN_EXHAUSTIVE_POPULATION"
        )


@pytest.mark.parametrize(
    ("scope_id", "bundle", "size"),
    [
        ("", "one two", 2),
        ("   ", "one two", 2),
        (" padded ", "one two", 2),
        ("s" * 129, "one two", 2),
        (True, "one two", 2),
        ("scope", "", 2),
        ("scope", "   ", 2),
        (
            "scope",
            "x" * (mcp_module.MCP_SCOPE_LEXICAL_BUNDLE_MAX_CHARS + 1),
            2,
        ),
        ("scope", True, 2),
        ("scope", None, 2),
        ("scope", "one two", True),
        ("scope", "one two", 1),
        ("scope", "one two", 11),
        ("scope", "one two", "2"),
    ],
)
def test_mcp_scope_lexical_bundle_evidence_rejects_invalid_values_before_open(
    monkeypatch, scope_id, bundle, size
) -> None:
    monkeypatch.setattr(
        mcp_module,
        "_open_project",
        lambda: pytest.fail("invalid scoped lexical-bundle evidence opened the project"),
    )
    with pytest.raises(ProjectError):
        mcp_module.corpus_lexical_bundle_evidence_for_scope(scope_id, bundle, size)


def test_mcp_scope_lexical_bundle_evidence_failures_create_no_orphan_records(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "scope-lexical-bundle-evidence-failures"
    project = Project.init(root)
    _, normal_revision = project.import_text("normal.txt", "one two three")
    normal_scope = project.snapshot([normal_revision])
    _, empty_population_revision = project.import_text("empty.txt", "singleton")
    empty_population_scope = project.snapshot([empty_population_revision])
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
    before = _persistence_counts(root)

    with pytest.raises(ScopeViolationError, match="scope"):
        mcp_module.corpus_lexical_bundle_evidence_for_scope(
            "unknown-scope", "one two", 2
        )
    with pytest.raises(ScopeViolationError, match="exactly 2 surface tokens"):
        mcp_module.corpus_lexical_bundle_evidence_for_scope(
            normal_scope, "one two three", 2
        )
    with pytest.raises(ScopeViolationError, match="non-empty n-gram population"):
        mcp_module.corpus_lexical_bundle_evidence_for_scope(
            empty_population_scope, "one two", 2
        )

    assert _persistence_counts(root) == before


def test_mcp_scope_lexical_bundle_evidence_is_registered_but_hidden_read_only(
    monkeypatch,
) -> None:
    tools = asyncio.run(mcp_module.server.list_tools())
    names = {tool.name for tool in tools}
    tool_name = "corpus_lexical_bundle_evidence_for_scope"
    assert tool_name in names
    assert tool_name not in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    tool = next(item for item in tools if item.name == tool_name)
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    assert schema["required"] == ["scope_id", "bundle", "size"]
    properties = schema["properties"]
    assert properties["scope_id"]["minLength"] == 1
    assert properties["scope_id"]["maxLength"] == 128
    assert properties["bundle"]["minLength"] == 1
    assert properties["bundle"]["maxLength"] == mcp_module.MCP_SCOPE_LEXICAL_BUNDLE_MAX_CHARS
    assert properties["size"]["minimum"] == 2
    assert properties["size"]["maximum"] == 10

    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    monkeypatch.setattr(
        mcp_module,
        "_open_project",
        lambda: pytest.fail("read-only lexical-bundle evidence must be rejected before opening"),
    )
    try:
        with pytest.raises(ProjectError, match="read-only"):
            mcp_module.corpus_lexical_bundle_evidence_for_scope("scope", "one two", 2)

        asyncio.run(mcp_module._configure_read_only_server())
        exposed_names = {item.name for item in asyncio.run(mcp_module.server.list_tools())}
        assert tool_name not in exposed_names
        with pytest.raises(Exception, match=f"Unknown tool: {tool_name}"):
            asyncio.run(
                mcp_module.server.call_tool(
                    tool_name,
                    {"scope_id": "scope", "bundle": "one two", "size": 2},
                )
            )
    finally:
        monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
        mcp_module._restore_full_server_registry()
