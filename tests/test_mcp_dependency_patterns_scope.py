from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, TypedDict, cast

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.nlp import ParsedToken


class _PatternBackend:
    language = "en"
    backend_id = "mcp-dependency-pattern-scope-fixture"
    model_id = "mcp-dependency-pattern-scope-fixture-v1"

    def parse(self, text: str) -> list[ParsedToken]:
        result: list[ParsedToken] = []
        for index, part in enumerate(text.split()):
            start = text.index(part, result[-1].char_end if result else 0)
            result.append(
                ParsedToken(
                    surface=part,
                    lemma=part.casefold(),
                    pos="NOUN",
                    morphology=None,
                    dependency_head=None if index == 0 else 0,
                    dependency_relation="ROOT" if index == 0 else "obj",
                    char_start=start,
                    char_end=start + len(part),
                )
            )
        return result


class _DependencyPatternPage(TypedDict):
    scope_id: str
    artifact_ids: list[str]
    offset: int
    limit: int
    total_rows: int
    has_more: bool
    rows: list[dict[str, object]]


def test_scope_dependency_pattern_pages_match_core_and_exclude_later_imports(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    root = tmp_path / "dependency-pattern-scope"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "Rivers carry water")
    _, second_revision = project.import_text("second.txt", "Rivers carry water")
    first_artifact = project.build_parsed_token_artifact(first_revision, _PatternBackend())
    second_artifact = project.build_parsed_token_artifact(second_revision, _PatternBackend())
    scope_id = project.snapshot([first_revision, second_revision])
    _, later_revision = project.import_text("later.txt", "Rivers alter trends")
    project.build_parsed_token_artifact(later_revision, _PatternBackend())
    artifact_ids = [first_artifact.artifact_id, second_artifact.artifact_id]

    core_pages: dict[int, _DependencyPatternPage] = {
        offset: cast(
            _DependencyPatternPage,
            dict(
                project.corpus_dependency_patterns(
                    scope_id,
                    artifact_ids,
                    relation="obj",
                    head_lemma="RIVERS",
                    offset=offset,
                    limit=1,
                )
            )
        )
        for offset in (0, 1)
    }
    core_filtered = dict(
        project.corpus_dependency_patterns(
            scope_id,
            artifact_ids,
            relation="obj",
            head_lemma="rivers",
            dependent_lemma="water",
            limit=10,
        )
    )
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    def unexpected_snapshot(*_args, **_kwargs):
        raise AssertionError("existing-scope dependency-pattern reads must not snapshot")

    monkeypatch.setattr(Project, "snapshot", unexpected_snapshot)
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    first = cast(
        _DependencyPatternPage,
        mcp_module.corpus_dependency_patterns_for_scope(
            scope_id,
            artifact_ids,
            relation="obj",
            head_lemma="RIVERS",
            offset=0,
            limit=1,
        ),
    )
    second = cast(
        _DependencyPatternPage,
        mcp_module.corpus_dependency_patterns_for_scope(
            scope_id,
            artifact_ids,
            relation="obj",
            head_lemma="RIVERS",
            offset=1,
            limit=1,
        ),
    )
    filtered = mcp_module.corpus_dependency_patterns_for_scope(
        scope_id,
        artifact_ids,
        relation="obj",
        head_lemma="rivers",
        dependent_lemma="water",
        limit=10,
    )

    assert first == core_pages[0]
    assert second == core_pages[1]
    assert filtered == core_filtered
    assert first["scope_id"] == second["scope_id"] == scope_id
    assert first["artifact_ids"] == second["artifact_ids"] == sorted(artifact_ids)
    assert first["total_rows"] == second["total_rows"] == 2
    assert first["has_more"] is True
    assert second["has_more"] is False
    stitched_rows = first["rows"] + second["rows"]
    assert stitched_rows == core_pages[0]["rows"] + core_pages[1]["rows"]
    assert "alter" not in str(stitched_rows)
    assert "trends" not in str(stitched_rows)
    assert filtered["rows"] == core_filtered["rows"]
    assert filtered["filters"] == {
        "relation": "obj",
        "head_lemma": "rivers",
        "dependent_lemma": "water",
    }
    assert database.read_bytes() == before


@pytest.mark.parametrize(
    ("scope_id", "artifact_ids", "relation", "head_lemma", "dependent_lemma", "offset", "limit"),
    [
        (" ", ["artifact-1"], None, None, None, 0, 10),
        ("scope-1", [], None, None, None, 0, 10),
        ("scope-1", [" "], None, None, None, 0, 10),
        ("scope-1", ["x" * 129], None, None, None, 0, 10),
        ("scope-1", ["artifact-1", "artifact-1"], None, None, None, 0, 10),
        ("scope-1", [123], None, None, None, 0, 10),
        ("scope-1", ["artifact-1"], " ", None, None, 0, 10),
        ("scope-1", ["artifact-1"], None, None, "x" * 257, 0, 10),
        ("scope-1", ["artifact-1"], None, None, None, True, 10),
        ("scope-1", ["artifact-1"], None, None, None, 0, False),
        ("scope-1", ["artifact-1"], None, None, None, -1, 10),
        ("scope-1", ["artifact-1"], None, None, None, 0, 1_001),
    ],
)
def test_scope_dependency_patterns_validate_before_open(
    monkeypatch,
    scope_id,
    artifact_ids,
    relation,
    head_lemma,
    dependent_lemma,
    offset,
    limit,
) -> None:
    def unexpected_open():
        raise AssertionError("invalid dependency-pattern arguments must fail before open")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_open)

    with pytest.raises(ProjectError):
        mcp_module.corpus_dependency_patterns_for_scope(
            scope_id,
            artifact_ids,
            relation=relation,
            head_lemma=head_lemma,
            dependent_lemma=dependent_lemma,
            offset=offset,
            limit=limit,
        )


def test_scope_dependency_patterns_reject_unknown_incomplete_and_out_of_scope_artifacts(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    root = tmp_path / "dependency-pattern-artifacts"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "Rivers carry")
    _, second_revision = project.import_text("second.txt", "Citizens demand")
    _, outside_revision = project.import_text("outside.txt", "Visitors challenge")
    first_artifact = project.build_parsed_token_artifact(first_revision, _PatternBackend())
    second_artifact = project.build_parsed_token_artifact(second_revision, _PatternBackend())
    outside_artifact = project.build_parsed_token_artifact(outside_revision, _PatternBackend())
    scope_id = project.snapshot([first_revision, second_revision])
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ProjectError, match="scope"):
        mcp_module.corpus_dependency_patterns_for_scope(
            "unknown-scope", [first_artifact.artifact_id], limit=1
        )
    with pytest.raises(ProjectError):
        mcp_module.corpus_dependency_patterns_for_scope(
            scope_id, [first_artifact.artifact_id], limit=1
        )
    with pytest.raises(ProjectError, match="outside the explicit scope"):
        mcp_module.corpus_dependency_patterns_for_scope(
            scope_id,
            [first_artifact.artifact_id, second_artifact.artifact_id, outside_artifact.artifact_id],
            limit=1,
        )


class _PageProject:
    def __init__(self, page: Mapping[str, object]) -> None:
        self.page = dict(page)
        self.page_call: tuple[str, list[str], str | None, str | None, str | None, int, int] | None = None
        self.closed = False

    def corpus_dependency_patterns(
        self,
        scope_id: str,
        artifact_ids: list[str],
        *,
        relation: str | None,
        head_lemma: str | None,
        dependent_lemma: str | None,
        offset: int,
        limit: int,
    ) -> Mapping[str, object]:
        self.page_call = (
            scope_id,
            artifact_ids,
            relation,
            head_lemma,
            dependent_lemma,
            offset,
            limit,
        )
        return self.page

    def close(self) -> None:
        self.closed = True


def test_scope_dependency_patterns_enforce_json_response_budget(monkeypatch) -> None:
    page_project = _PageProject(
        {
            "scope_id": "scope-1",
            "artifact_ids": ["artifact-1"],
            "calculation_version": "parsed-dependency-patterns-v2",
            "coverage": {},
            "filters": {"relation": None, "head_lemma": None, "dependent_lemma": None},
            "offset": 0,
            "limit": 1,
            "total_rows": 1,
            "has_more": False,
            "candidate_count_lower_bound": 1,
            "rows": [
                {
                    "head_lemma": "é" * 11_000,
                    "relation": "obj",
                    "dependent_lemma": "water",
                    "frequency": 1,
                    "document_frequency": 1,
                }
            ],
            "truncated": False,
        }
    )
    monkeypatch.setattr(mcp_module, "_open_project", lambda: page_project)

    with pytest.raises(ProjectError, match="64 KiB UTF-8 JSON response budget") as exc_info:
        mcp_module.corpus_dependency_patterns_for_scope("scope-1", ["artifact-1"], limit=1)

    assert "é" * 100 not in str(exc_info.value)
    assert page_project.page_call == ("scope-1", ["artifact-1"], None, None, None, 0, 1)
    assert page_project.closed is True


def test_scope_dependency_patterns_are_visible_in_read_only_schema():
    async def inspect_tool() -> dict[str, Any]:
        tools = await mcp_module.server.list_tools()
        tool = next(item for item in tools if item.name == "corpus_dependency_patterns_for_scope")
        return tool.model_dump(by_alias=True)["inputSchema"]

    assert "corpus_dependency_patterns_for_scope" in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    previous_read_only = mcp_module._MCP_READ_ONLY
    mcp_module._MCP_READ_ONLY = True
    try:
        asyncio.run(mcp_module._configure_read_only_server())
        schema = asyncio.run(inspect_tool())
    finally:
        mcp_module._MCP_READ_ONLY = previous_read_only
        mcp_module._restore_full_server_registry()

    assert set(schema["required"]) == {"scope_id", "artifact_ids"}
    properties = schema["properties"]
    assert properties["scope_id"]["minLength"] == 1
    assert properties["scope_id"]["maxLength"] == 128
    assert properties["artifact_ids"]["minItems"] == 1
    assert (
        properties["artifact_ids"]["maxItems"]
        == mcp_module.MCP_SCOPE_PARSED_FREQUENCY_MAX_ARTIFACTS
    )
    assert properties["artifact_ids"]["uniqueItems"] is True
    for field_name in ("relation", "head_lemma", "dependent_lemma"):
        field_schema = properties[field_name]
        string_schema = (
            field_schema
            if field_schema.get("type") == "string"
            else next(branch for branch in field_schema["anyOf"] if branch.get("type") == "string")
        )
        assert string_schema["maxLength"] == mcp_module.MCP_SCOPE_DEPENDENCY_PATTERN_FILTER_MAX_CHARS
    assert properties["offset"]["maximum"] == 0x7FFF_FFFF_FFFF_FFFF
    assert properties["limit"]["maximum"] == mcp_module.MCP_SCOPE_CORPUS_MAX_PAGE_SIZE
