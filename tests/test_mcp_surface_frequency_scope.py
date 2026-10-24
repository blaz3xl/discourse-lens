from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, TypedDict, cast

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError

SURFACE_FREQUENCY_TOOLS = (
    (mcp_module.corpus_normalized_frequency_for_scope, "normalized"),
    (mcp_module.corpus_document_frequency_for_scope, "document_frequency"),
)


class _SurfaceFrequencyPage(TypedDict):
    scope_id: str
    revision_ids: list[str]
    has_more: bool
    rows: list[dict[str, object]]


def test_scope_surface_frequency_pages_match_core_and_exclude_later_imports(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "surface-frequency-scope"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "Café café orchard")
    _, second_revision = project.import_text("second.txt", "Café river")
    first_artifact = project.build_surface_token_artifact(first_revision)
    second_artifact = project.build_surface_token_artifact(second_revision)
    scope_id = project.snapshot([first_revision, second_revision])

    _, later_revision = project.import_text("later.txt", "Café outsider")
    project.build_surface_token_artifact(later_revision)
    artifact_ids = [first_artifact.artifact_id, second_artifact.artifact_id]

    core_pages = {
        (field, offset): dict(
            project.corpus_surface_frequency_page(
                scope_id,
                artifact_ids,
                field=field,
                offset=offset,
                limit=2,
            )
        )
        for field in ("normalized", "document_frequency")
        for offset in (0, 2)
    }
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    def unexpected_snapshot(*_args, **_kwargs):
        raise AssertionError("an existing-scope surface-frequency read must not snapshot")

    monkeypatch.setattr(Project, "snapshot", unexpected_snapshot)
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    mcp_pages: dict[tuple[str, int], _SurfaceFrequencyPage] = {
        (field, offset): cast(
            _SurfaceFrequencyPage,
            tool(scope_id, artifact_ids, offset=offset, limit=2),
        )
        for tool, field in SURFACE_FREQUENCY_TOOLS
        for offset in (0, 2)
    }

    assert mcp_pages == core_pages
    for field in ("normalized", "document_frequency"):
        first_page = mcp_pages[(field, 0)]
        second_page = mcp_pages[(field, 2)]
        assert first_page["scope_id"] == second_page["scope_id"] == scope_id
        assert first_page["revision_ids"] == second_page["revision_ids"] == [
            first_revision,
            second_revision,
        ]
        assert first_page["has_more"] is True
        assert second_page["has_more"] is False
        first_tokens = {row["token"] for row in first_page["rows"]}
        second_tokens = {row["token"] for row in second_page["rows"]}
        assert first_tokens.isdisjoint(second_tokens)
        assert "outsider" not in first_tokens | second_tokens
    assert database.read_bytes() == before


@pytest.mark.parametrize(
    ("scope_id", "artifact_ids", "offset", "limit"),
    [
        (" ", ["artifact-1"], 0, 10),
        ("scope-1", [], 0, 10),
        ("scope-1", [" "], 0, 10),
        ("scope-1", ["x" * 129], 0, 10),
        ("scope-1", ["artifact-1", "artifact-1"], 0, 10),
        ("scope-1", [123], 0, 10),
        ("scope-1", ["artifact-1"], True, 10),
        ("scope-1", ["artifact-1"], 0, False),
        ("scope-1", ["artifact-1"], -1, 10),
        ("scope-1", ["artifact-1"], 0, 1_001),
    ],
)
def test_scope_surface_frequency_validates_before_open(
    monkeypatch, scope_id, artifact_ids, offset, limit
) -> None:
    def unexpected_open():
        raise AssertionError("invalid scope surface-frequency arguments must fail before open")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_open)
    for tool, _field in SURFACE_FREQUENCY_TOOLS:
        with pytest.raises(ProjectError):
            tool(scope_id, artifact_ids, offset=offset, limit=limit)


def test_scope_surface_frequency_rejects_unknown_scope_without_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "unknown-surface-frequency-scope"
    project = Project.init(root)
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ProjectError, match="scope"):
        mcp_module.corpus_normalized_frequency_for_scope("missing-scope", ["artifact-1"])


class _PageProject:
    def __init__(self, page: Mapping[str, object]) -> None:
        self.page = dict(page)
        self.page_call: tuple[str, list[str], str, int, int] | None = None
        self.closed = False

    def corpus_surface_frequency_page(
        self,
        scope_id: str,
        artifact_ids: list[str],
        *,
        field: str,
        offset: int,
        limit: int,
    ) -> Mapping[str, object]:
        self.page_call = (scope_id, artifact_ids, field, offset, limit)
        return self.page

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(("tool", "field"), SURFACE_FREQUENCY_TOOLS)
def test_scope_surface_frequency_enforces_exact_json_response_budget(
    monkeypatch, tool, field
) -> None:
    project = _PageProject(
        {
            "scope_id": "scope-1",
            "revision_ids": ["revision-1"],
            "artifact_ids": ["artifact-1"],
            "field": field,
            "offset": 0,
            "limit": 1,
            "total_values": 1,
            "has_more": False,
            # ensure_ascii=True expands each character to six bytes on the wire.
            "rows": [{"token": "é" * 11_000, "count": 1}],
        }
    )
    monkeypatch.setattr(mcp_module, "_open_project", lambda: project)

    with pytest.raises(ProjectError, match="64 KiB UTF-8 JSON response budget") as exc_info:
        tool("scope-1", ["artifact-1"])

    assert "é" * 100 not in str(exc_info.value)
    assert project.page_call == ("scope-1", ["artifact-1"], field, 0, 100)
    assert project.closed is True


def test_scope_surface_frequency_tools_are_visible_in_read_only_schema():
    async def inspect_tools() -> dict[str, Any]:
        tools = await mcp_module.server.list_tools()
        schemas: dict[str, Any] = {}
        for tool_name, _field in (
            ("corpus_normalized_frequency_for_scope", "normalized"),
            ("corpus_document_frequency_for_scope", "document_frequency"),
        ):
            tool = next(item for item in tools if item.name == tool_name)
            schemas[tool_name] = tool.model_dump(by_alias=True)["inputSchema"]
        return schemas

    assert {
        "corpus_normalized_frequency_for_scope",
        "corpus_document_frequency_for_scope",
    } <= mcp_module.READ_ONLY_MCP_TOOL_NAMES
    previous_read_only = mcp_module._MCP_READ_ONLY
    mcp_module._MCP_READ_ONLY = True
    try:
        asyncio.run(mcp_module._configure_read_only_server())
        schemas = asyncio.run(inspect_tools())
    finally:
        mcp_module._MCP_READ_ONLY = previous_read_only
        mcp_module._restore_full_server_registry()

    for schema in schemas.values():
        assert set(schema["required"]) == {"scope_id", "artifact_ids"}
        properties = schema["properties"]
        assert properties["scope_id"]["minLength"] == 1
        assert properties["scope_id"]["maxLength"] == 128
        assert properties["artifact_ids"]["minItems"] == 1
        assert (
            properties["artifact_ids"]["maxItems"]
            == mcp_module.MCP_SCOPE_SURFACE_FREQUENCY_MAX_ARTIFACTS
        )
        assert properties["artifact_ids"]["uniqueItems"] is True
        assert properties["offset"]["maximum"] == 0x7FFF_FFFF_FFFF_FFFF
        assert properties["limit"]["maximum"] == mcp_module.MCP_SCOPE_CORPUS_MAX_PAGE_SIZE
