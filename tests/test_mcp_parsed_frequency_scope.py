from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, TypedDict, cast

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.nlp import ParsedToken


class _ParsedFixture:
    language = "en"
    backend_id = "mcp-parsed-frequency-scope-fixture"
    model_id = "mcp-parsed-frequency-scope-fixture-v1"

    def parse(self, text: str) -> list[ParsedToken]:
        result: list[ParsedToken] = []
        for index, part in enumerate(text.split()):
            start = text.index(part, result[-1].char_end if result else 0)
            lemma = part.casefold().rstrip(".")
            result.append(
                ParsedToken(
                    surface=part,
                    lemma=lemma,
                    pos="VERB" if lemma == "beta" else "NOUN",
                    morphology=None,
                    dependency_head=None if index == 0 else 0,
                    dependency_relation="ROOT" if index == 0 else "obj",
                    char_start=start,
                    char_end=start + len(part),
                )
            )
        return result


class _ParsedFrequencyPage(TypedDict):
    scope_id: str
    revision_ids: list[str]
    artifact_ids: list[str]
    field: str
    offset: int
    limit: int
    total_values: int
    has_more: bool
    rows: list[dict[str, object]]


def test_scope_parsed_frequency_matches_core_and_excludes_later_imports(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    root = tmp_path / "parsed-frequency-scope"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "alpha alpha beta")
    _, second_revision = project.import_text("second.txt", "beta gamma delta")
    first_artifact = project.build_parsed_token_artifact(first_revision, _ParsedFixture())
    second_artifact = project.build_parsed_token_artifact(second_revision, _ParsedFixture())
    scope_id = project.snapshot([first_revision, second_revision])
    _, later_revision = project.import_text("later.txt", "zeta outsider")
    project.build_parsed_token_artifact(later_revision, _ParsedFixture())
    artifact_ids = [first_artifact.artifact_id, second_artifact.artifact_id]
    fields = ("lemma", "pos", "dependency_relation")

    core_pages = {
        (field, offset): dict(
            project.corpus_parsed_frequency_page(
                scope_id,
                artifact_ids,
                field=field,
                offset=offset,
                limit=1,
            )
        )
        for field in fields
        for offset in (0, 1, 2, 3)
    }
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    def unexpected_snapshot(*_args, **_kwargs):
        raise AssertionError("an existing-scope parsed-frequency read must not snapshot")

    monkeypatch.setattr(Project, "snapshot", unexpected_snapshot)
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    mcp_pages: dict[tuple[str, int], _ParsedFrequencyPage] = {
        (field, offset): cast(
            _ParsedFrequencyPage,
            mcp_module.corpus_parsed_frequency_for_scope(
                scope_id,
                artifact_ids,
                field,
                offset=offset,
                limit=1,
            ),
        )
        for field in fields
        for offset in (0, 1, 2, 3)
    }

    assert mcp_pages == core_pages
    for field in fields:
        pages = [mcp_pages[(field, offset)] for offset in (0, 1, 2, 3)]
        assert all(page["scope_id"] == scope_id for page in pages)
        assert all(
            page["revision_ids"] == [first_revision, second_revision] for page in pages
        )
        assert all(page["artifact_ids"] == sorted(artifact_ids) for page in pages)
        values = [row["value"] for page in pages for row in page["rows"]]
        assert len(values) == len(set(values))
        assert "outsider" not in values
        assert "zeta" not in values
        assert pages[0]["has_more"] is True
        assert pages[-1]["has_more"] is False
    assert database.read_bytes() == before


@pytest.mark.parametrize(
    ("scope_id", "artifact_ids", "field", "offset", "limit"),
    [
        (" ", ["artifact-1"], "lemma", 0, 10),
        ("scope-1", [], "lemma", 0, 10),
        ("scope-1", [" "], "lemma", 0, 10),
        ("scope-1", ["x" * 129], "lemma", 0, 10),
        ("scope-1", ["artifact-1", "artifact-1"], "lemma", 0, 10),
        ("scope-1", [123], "lemma", 0, 10),
        ("scope-1", ["artifact-1"], "surface", 0, 10),
        ("scope-1", ["artifact-1"], "lemma", True, 10),
        ("scope-1", ["artifact-1"], "lemma", 0, False),
        ("scope-1", ["artifact-1"], "lemma", -1, 10),
        ("scope-1", ["artifact-1"], "lemma", 0, 1_001),
    ],
)
def test_scope_parsed_frequency_validates_before_open(
    monkeypatch, scope_id, artifact_ids, field, offset, limit
) -> None:
    def unexpected_open():
        raise AssertionError("invalid scope parsed-frequency arguments must fail before open")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_open)

    with pytest.raises(ProjectError):
        mcp_module.corpus_parsed_frequency_for_scope(
            scope_id,
            artifact_ids,
            field,
            offset=offset,
            limit=limit,
        )


def test_scope_parsed_frequency_rejects_unknown_and_incomplete_scope_artifacts(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    root = tmp_path / "parsed-frequency-scope-artifacts"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "alpha beta")
    _, second_revision = project.import_text("second.txt", "gamma delta")
    first_artifact = project.build_parsed_token_artifact(first_revision, _ParsedFixture())
    second_artifact = project.build_parsed_token_artifact(second_revision, _ParsedFixture())
    scope_id = project.snapshot([first_revision, second_revision])
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ProjectError, match="scope"):
        mcp_module.corpus_parsed_frequency_for_scope(
            "unknown-scope", [first_artifact.artifact_id], "lemma"
        )
    with pytest.raises(ProjectError):
        mcp_module.corpus_parsed_frequency_for_scope(
            scope_id, [first_artifact.artifact_id], "lemma"
        )
    with pytest.raises(ProjectError):
        mcp_module.corpus_parsed_frequency_for_scope(
            scope_id,
            [first_artifact.artifact_id, second_artifact.artifact_id, "outside-scope"],
            "lemma",
        )


class _PageProject:
    def __init__(self, page: Mapping[str, object]) -> None:
        self.page = dict(page)
        self.page_call: tuple[str, list[str], str, int, int] | None = None
        self.closed = False

    def corpus_parsed_frequency_page(
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


def test_scope_parsed_frequency_enforces_json_response_budget(monkeypatch) -> None:
    page_project = _PageProject(
        {
            "scope_id": "scope-1",
            "revision_ids": ["revision-1"],
            "artifact_ids": ["artifact-1"],
            "field": "lemma",
            "coverage": {},
            "field_coverage": {},
            "offset": 0,
            "limit": 1,
            "total_values": 1,
            "has_more": False,
            # ensure_ascii=True expands each character to six bytes on the wire.
            "rows": [{"value": "é" * 11_000, "count": 1}],
        }
    )
    monkeypatch.setattr(mcp_module, "_open_project", lambda: page_project)

    with pytest.raises(ProjectError, match="64 KiB UTF-8 JSON response budget") as exc_info:
        mcp_module.corpus_parsed_frequency_for_scope("scope-1", ["artifact-1"], "lemma")

    assert "é" * 100 not in str(exc_info.value)
    assert page_project.page_call == ("scope-1", ["artifact-1"], "lemma", 0, 100)
    assert page_project.closed is True


def test_scope_parsed_frequency_is_visible_in_read_only_schema():
    async def inspect_tool() -> dict[str, Any]:
        tools = await mcp_module.server.list_tools()
        tool = next(item for item in tools if item.name == "corpus_parsed_frequency_for_scope")
        return tool.model_dump(by_alias=True)["inputSchema"]

    assert "corpus_parsed_frequency_for_scope" in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    previous_read_only = mcp_module._MCP_READ_ONLY
    mcp_module._MCP_READ_ONLY = True
    try:
        asyncio.run(mcp_module._configure_read_only_server())
        schema = asyncio.run(inspect_tool())
    finally:
        mcp_module._MCP_READ_ONLY = previous_read_only
        mcp_module._restore_full_server_registry()

    assert set(schema["required"]) == {"scope_id", "artifact_ids", "field"}
    properties = schema["properties"]
    assert properties["scope_id"]["minLength"] == 1
    assert properties["scope_id"]["maxLength"] == 128
    assert properties["artifact_ids"]["minItems"] == 1
    assert (
        properties["artifact_ids"]["maxItems"]
        == mcp_module.MCP_SCOPE_PARSED_FREQUENCY_MAX_ARTIFACTS
    )
    assert properties["artifact_ids"]["uniqueItems"] is True
    assert properties["field"]["enum"] == ["lemma", "pos", "dependency_relation"]
    assert properties["offset"]["maximum"] == 0x7FFF_FFFF_FFFF_FFFF
    assert properties["limit"]["maximum"] == mcp_module.MCP_SCOPE_CORPUS_MAX_PAGE_SIZE
