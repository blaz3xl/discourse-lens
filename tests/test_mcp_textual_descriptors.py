from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator
from mcp.server.mcpserver.exceptions import ToolError

import discourselens.mcp_server as mcp_module
from discourselens import Project


def test_textual_descriptors_mcp_schema_requires_bounded_unique_revisions() -> None:
    tools = {tool.name: tool for tool in asyncio.run(mcp_module.server.list_tools())}
    schema = tools["run_textual_descriptors"].model_dump(by_alias=True)["inputSchema"]

    assert schema["required"] == ["revision_ids"]
    properties = schema["properties"]
    revisions = properties["revision_ids"]
    assert revisions["minItems"] == 1
    assert revisions["maxItems"] == 100
    assert revisions["uniqueItems"] is True
    assert revisions["items"]["pattern"] == r"\S"
    assert revisions["items"]["maxLength"] == 128
    assert "languages" not in schema["required"]
    assert "keywords" not in properties

    validator = Draft202012Validator(schema)
    assert validator.is_valid({"revision_ids": ["revision-1"]})
    assert validator.is_valid({"revision_ids": [f"revision-{index}" for index in range(100)]})
    assert not validator.is_valid({"revision_ids": []})
    assert not validator.is_valid({"revision_ids": ["revision-1", "revision-1"]})
    assert not validator.is_valid({"revision_ids": [" \t "]})
    assert not validator.is_valid({"revision_ids": [f"revision-{index}" for index in range(101)]})


def test_batch_textual_descriptors_tool_has_bounded_explicit_revision_scope() -> None:
    tools = {tool.name: tool for tool in asyncio.run(mcp_module.server.list_tools())}
    schema = tools["run_batch_textual_descriptors"].model_dump(by_alias=True)["inputSchema"]

    assert schema["required"] == ["revision_ids"]
    revisions = schema["properties"]["revision_ids"]
    assert revisions["minItems"] == 1
    assert revisions["maxItems"] == 100
    assert revisions["uniqueItems"] is True
    assert revisions["items"]["pattern"] == r"\S"
    assert revisions["items"]["maxLength"] == 128


def test_textual_descriptors_mcp_rejects_invalid_schema_before_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_project_open() -> Project:
        pytest.fail("invalid textual-descriptor MCP arguments must not open a project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)

    async def invoke():
        return await mcp_module.server.call_tool(
            "run_textual_descriptors", {"revision_ids": ["revision-1", " "]}
        )

    with pytest.raises(ToolError, match="invalid MCP tool arguments"):
        asyncio.run(invoke())


@pytest.mark.parametrize(
    "revision_ids",
    [[], ["revision-1", "revision-1"], [" \t "], [""]],
)
def test_textual_descriptors_rejects_invalid_revision_ids_before_open(
    monkeypatch: pytest.MonkeyPatch, revision_ids: list[str]
) -> None:
    def unexpected_project_open() -> Project:
        pytest.fail("invalid textual-descriptor revision IDs must not open a project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)

    with pytest.raises(ValueError):
        mcp_module.run_textual_descriptors(revision_ids=revision_ids)


def test_textual_descriptors_forwards_one_scope_and_core_language_inventory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    project = Project.init(root)
    _, english_revision = project.import_text("english.txt", "Officials must act.")
    _, spanish_revision = project.import_text("spanish.txt", "La autoridad debe actuar.")
    _, outside_revision = project.import_text("outside.txt", "Officials should wait.")
    project.close()

    opened_project = Project.open(root)
    close_calls = 0
    original_close = opened_project.close

    def close_tracked() -> None:
        nonlocal close_calls
        close_calls += 1
        original_close()

    monkeypatch.setattr(opened_project, "close", close_tracked)
    monkeypatch.setattr(mcp_module, "_open_project", lambda: opened_project)

    core_calls: list[tuple[list[str], tuple[str, ...]]] = []
    original_core = mcp_module.run_textual_descriptors_analysis

    def track_core(
        core_project: Any, revision_ids: list[str], languages: tuple[str, ...]
    ) -> dict[str, object]:
        core_calls.append((list(revision_ids), languages))
        return dict(original_core(core_project, revision_ids, languages))

    monkeypatch.setattr(mcp_module, "run_textual_descriptors_analysis", track_core)
    result = mcp_module.run_textual_descriptors(
        revision_ids=[english_revision, spanish_revision], languages=["es"]
    )

    assert result["analysis_run_id"]
    assert result["skill_run_id"]
    assert result["finding_ids"]
    assert close_calls == 1
    assert core_calls == [([english_revision, spanish_revision], ("es",))]

    inspection = Project.open(root, read_only=True)
    try:
        runs = inspection.db.execute(
            "SELECT id,scope_id,status,config_json FROM analysis_runs"
        ).fetchall()
        assert len(runs) == 1
        run = runs[0]
        assert run["id"] == result["analysis_run_id"]
        assert run["status"] == "COMPLETED"
        assert inspection.scope_revisions(run["scope_id"]) == [
            english_revision,
            spanish_revision,
        ]
        assert outside_revision not in inspection.scope_revisions(run["scope_id"])
        assert json.loads(run["config_json"])["languages"] == ["es"]

        finding_ids = cast(list[str], result["finding_ids"])
        assert len(finding_ids) == 1
        evidence = inspection.finding_evidence(finding_ids[0])
        assert [inspection.span_text(str(item["span_id"])) for item in evidence] == ["debe"]
    finally:
        inspection.close()


def test_textual_descriptors_defaults_to_both_languages(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class StubProject:
        def close(self) -> None:
            observed["closed"] = True

    stub = StubProject()

    def analyze(project: StubProject, revision_ids: list[str], languages: tuple[str, ...]):
        observed["project"] = project
        observed["revision_ids"] = revision_ids
        observed["languages"] = languages
        return {}

    monkeypatch.setattr(mcp_module, "_open_project", lambda: stub)
    monkeypatch.setattr(mcp_module, "run_textual_descriptors_analysis", analyze)

    assert mcp_module.run_textual_descriptors(revision_ids=["revision-1"]) == {}
    assert observed == {
        "project": stub,
        "revision_ids": ["revision-1"],
        "languages": ("en", "es"),
        "closed": True,
    }


def test_batch_textual_descriptors_creates_one_private_result_per_selected_revision(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "batch-project"
    project = Project.init(root)
    _, first_revision = project.import_text(
        "first.txt", "PRIVATE_BATCH_FIRST must be reviewed."
    )
    _, second_revision = project.import_text(
        "second.txt", "PRIVATE_BATCH_SECOND should be reviewed."
    )
    _, outside_revision = project.import_text(
        "outside.txt", "PRIVATE_BATCH_OUTSIDE must be ignored."
    )
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    async def invoke():
        return await mcp_module.server.call_tool(
            "run_batch_textual_descriptors",
            {"revision_ids": [second_revision, first_revision], "languages": ["en"]},
        )

    response = asyncio.run(invoke())
    assert not response.is_error
    payload = [json.loads(block.text) for block in response.content]

    assert [item["analysis_run_id"] for item in payload]
    assert [set(item) for item in payload] == [
        {"analysis_run_id", "skill_run_id", "finding_ids", "coverage_id"},
        {"analysis_run_id", "skill_run_id", "finding_ids", "coverage_id"},
    ]
    assert "PRIVATE_BATCH_" not in json.dumps(payload)

    inspection = Project.open(root, read_only=True)
    try:
        scoped_revisions = []
        for item, revision_id in zip(payload, [second_revision, first_revision], strict=True):
            run = inspection.db.execute(
                "SELECT scope_id,status,config_json FROM analysis_runs WHERE id=?",
                (item["analysis_run_id"],),
            ).fetchone()
            assert run is not None
            assert run["status"] == "COMPLETED"
            scope_revisions = inspection.scope_revisions(run["scope_id"])
            assert scope_revisions == [revision_id]
            scoped_revisions.extend(scope_revisions)
            assert json.loads(run["config_json"])["languages"] == ["en"]
        assert outside_revision not in scoped_revisions
    finally:
        inspection.close()


def test_textual_descriptors_closes_project_when_core_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, bool] = {}

    class StubProject:
        def close(self) -> None:
            observed["closed"] = True

    stub = StubProject()

    def fail_analysis(project: StubProject, revision_ids: list[str], languages: tuple[str, ...]):
        raise RuntimeError("textual descriptor Core failure")

    monkeypatch.setattr(mcp_module, "_open_project", lambda: stub)
    monkeypatch.setattr(mcp_module, "run_textual_descriptors_analysis", fail_analysis)

    with pytest.raises(RuntimeError, match="textual descriptor Core failure"):
        mcp_module.run_textual_descriptors(revision_ids=["revision-1"])

    assert observed == {"closed": True}
