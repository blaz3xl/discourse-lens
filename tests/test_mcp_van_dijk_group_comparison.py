from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ScopeViolationError

TOOL_NAME = "prepare_van_dijk_group_comparison"


def _call_tool(arguments: dict[str, object]):
    async def invoke():
        return await mcp_module.server.call_tool(TOOL_NAME, arguments)

    return asyncio.run(invoke())


def _result_payload(result) -> dict[str, Any]:
    assert not result.is_error
    assert len(result.content) == 1
    payload = json.loads(result.content[0].text)
    assert isinstance(payload, dict)
    return payload


def _base_arguments(
    revision_id: str,
    codebook_id: str,
    first_annotation_id: str,
    second_annotation_id: str,
) -> dict[str, object]:
    return {
        "revision_ids": [revision_id],
        "groups": [
            {"label": "Residents", "annotation_ids": [first_annotation_id]},
            {"label": "Officials", "annotation_ids": [second_annotation_id]},
        ],
        "shared_topic_event": "the shared road safety announcement",
        "comparability_basis": "both groups address the same announced review",
        "codebook_id": codebook_id,
    }


def test_mcp_registers_bounded_explicit_group_comparison_tool() -> None:
    tools = {tool.name: tool for tool in asyncio.run(mcp_module.server.list_tools())}
    assert TOOL_NAME in tools

    schema = tools[TOOL_NAME].model_dump(by_alias=True)["inputSchema"]
    properties = schema["properties"]
    assert set(schema["required"]) == {
        "revision_ids",
        "groups",
        "shared_topic_event",
        "comparability_basis",
        "codebook_id",
    }
    assert properties["revision_ids"]["minItems"] == 1
    assert properties["revision_ids"]["maxItems"] == 100
    assert properties["revision_ids"]["uniqueItems"] is True
    assert properties["groups"]["minItems"] == properties["groups"]["maxItems"] == 2
    group_schema = schema["$defs"]["McpVanDijkGroupInput"]
    assert group_schema["required"] == ["label", "annotation_ids"]
    assert group_schema["additionalProperties"] is False
    assert group_schema["properties"]["annotation_ids"]["maxItems"] == (
        mcp_module.MCP_VAN_DIJK_MAX_GROUP_ANNOTATIONS
    )
    assert group_schema["properties"]["annotation_ids"]["uniqueItems"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"revision_ids": []},
        {"revision_ids": ["same", "same"]},
        {"revision_ids": ["revision"] * 101},
        {"groups": [{"label": "Only one", "annotation_ids": ["annotation"]}]},
        {
            "groups": [
                {"label": "Residents", "annotation_ids": ["annotation-a"]},
                {"label": "Officials"},
            ]
        },
        {
            "groups": [
                {"label": "Same", "annotation_ids": ["annotation-a"]},
                {"label": " same ", "annotation_ids": ["annotation-b"]},
            ]
        },
        {
            "groups": [
                {"label": "Residents", "annotation_ids": ["duplicate", "duplicate"]},
                {"label": "Officials", "annotation_ids": []},
            ]
        },
        {
            "groups": [
                {"label": "Residents", "annotation_ids": ["shared"]},
                {"label": "Officials", "annotation_ids": ["shared"]},
            ]
        },
        {
            "groups": [
                {"label": "Residents", "annotation_ids": []},
                {"label": "Officials", "annotation_ids": []},
            ]
        },
        {
            "groups": [
                {"label": "Residents", "annotation_ids": ["annotation-a"], "extra": "x"},
                {"label": "Officials", "annotation_ids": []},
            ]
        },
    ],
)
def test_mcp_rejects_malformed_or_duplicate_inputs_before_opening_project(
    monkeypatch, overrides: dict[str, object]
) -> None:
    arguments: dict[str, object] = {
        "revision_ids": ["revision"],
        "groups": [
            {"label": "Residents", "annotation_ids": ["annotation-a"]},
            {"label": "Officials", "annotation_ids": ["annotation-b"]},
        ],
        "shared_topic_event": "shared event",
        "comparability_basis": "same event",
        "codebook_id": "codebook",
    }
    arguments.update(overrides)

    def unexpected_project_open():
        pytest.fail("malformed group-comparison arguments must not open a project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)

    with pytest.raises(ToolError, match="invalid MCP tool arguments"):
        _call_tool(arguments)


def test_mcp_group_comparison_returns_full_scoped_provenance_and_closes_project(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "project"
    project = Project.init(root)
    resident_text = "Residents described the road as unsafe."
    official_text = "Officials announced a safety review."
    _, resident_revision = project.import_text("residents.txt", resident_text)
    _, official_revision = project.import_text("officials.txt", official_text)
    _, unselected_revision = project.import_text("unselected.txt", "An uncoded report.")
    codebook_id = project.create_codebook("Road safety", "Researcher selected codes")
    resident_code = project.create_code(codebook_id, "Negative evaluation")
    official_code = project.create_code(codebook_id, "Institutional action")
    resident_annotation = project.annotate_span(
        resident_code, project.create_span(resident_revision, 0, len(resident_text))
    )
    official_annotation = project.annotate_span(
        official_code, project.create_span(official_revision, 0, len(official_text))
    )
    unselected_annotation = project.annotate_span(
        resident_code,
        project.create_span(unselected_revision, 0, len("An uncoded report.")),
    )
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    opened: list[Project] = []
    close_calls: list[Project] = []
    real_open = Project.open

    def open_project() -> Project:
        current = real_open(root)
        original_close = current.close

        def close() -> None:
            close_calls.append(current)
            original_close()

        current.close = close  # type: ignore[method-assign]
        opened.append(current)
        return current

    monkeypatch.setattr(mcp_module, "_open_project", open_project)
    arguments = _base_arguments(
        resident_revision,
        codebook_id,
        resident_annotation,
        official_annotation,
    )
    arguments["revision_ids"] = [resident_revision, official_revision]

    payload = _result_payload(_call_tool(arguments))

    assert len(opened) == 1
    assert close_calls == opened
    assert payload["artifact_type"] == "descriptive_qualitative_group_comparison_preparation"
    assert payload["count_basis"] == "SELECTED_ANNOTATIONS"
    assert payload["claim_semantics"] == "EXAMPLE_BASED"
    assert payload["selected_annotation_count"] == 2
    assert [group["label"] for group in payload["groups"]] == ["Residents", "Officials"]
    assert [
        group["annotations"][0]["annotation_id"] for group in payload["groups"]
    ] == [resident_annotation, official_annotation]
    assert [group["annotations"][0]["text"] for group in payload["groups"]] == [
        resident_text,
        official_text,
    ]
    assert unselected_annotation not in {
        annotation["annotation_id"]
        for group in payload["groups"]
        for annotation in group["annotations"]
    }
    assert payload["analysis_run_id"]
    assert payload["scope_snapshot_id"]
    assert payload["skill_run_id"]
    assert payload["finding_id"]
    assert payload["evidence_ids"]
    assert payload["coverage_id"]

    with Project.open(root, read_only=True) as reopened:
        summary = cast(
            dict[str, Any], reopened.analysis_run_summary(payload["analysis_run_id"])
        )
        run = summary["analysis_run"]
        assert run["scope_id"] == payload["scope_snapshot_id"]
        assert run["status"] == "COMPLETED"
        assert reopened.scope_revisions(run["scope_id"]) == [resident_revision, official_revision]
        assert run["config"]["groups"] == [
            {"label": "Residents", "annotation_ids": [resident_annotation]},
            {"label": "Officials", "annotation_ids": [official_annotation]},
        ]
        assert run["config"]["shared_topic_event"] == "the shared road safety announcement"
        assert run["config"]["comparability_basis"] == (
            "both groups address the same announced review"
        )
        assert run["config"]["codebook_id"] == codebook_id
        for group in payload["groups"]:
            annotation = group["annotations"][0]
            assert reopened.evidence_content(annotation["evidence_id"]) == {
                "annotation_id": annotation["annotation_id"],
                "codebook_id": codebook_id,
                "codebook_name": "Road safety",
                "code_id": annotation["code_id"],
                "code_label": annotation["code_label"],
                "comparability_basis": "both groups address the same announced review",
                "group_label": group["label"],
                "scope_snapshot_id": payload["scope_snapshot_id"],
                "shared_topic_event": "the shared road safety announcement",
                "skill_id": "builtin.van-dijk-group-comparison",
                "skill_version": "1.0.0",
            }


@pytest.mark.parametrize("annotation_selection", ["missing", "out_of_scope"])
def test_mcp_rejects_missing_or_out_of_scope_annotations_and_closes_project(
    tmp_path, monkeypatch, annotation_selection: str
) -> None:
    root = tmp_path / "project"
    project = Project.init(root)
    selected_text = "An included source span."
    excluded_text = "An excluded source span."
    _, selected_revision = project.import_text("selected.txt", selected_text)
    _, excluded_revision = project.import_text("excluded.txt", excluded_text)
    codebook_id = project.create_codebook("Fixture codebook")
    code_id = project.create_code(codebook_id, "Fixture code")
    selected_annotation = project.annotate_span(
        code_id, project.create_span(selected_revision, 0, len(selected_text))
    )
    excluded_annotation = project.annotate_span(
        code_id, project.create_span(excluded_revision, 0, len(excluded_text))
    )
    project.close()

    opened: list[Project] = []
    close_calls: list[Project] = []
    real_open = Project.open

    def open_project() -> Project:
        current = real_open(root)
        original_close = current.close

        def close() -> None:
            close_calls.append(current)
            original_close()

        current.close = close  # type: ignore[method-assign]
        opened.append(current)
        return current

    monkeypatch.setattr(mcp_module, "_open_project", open_project)
    second_annotation = (
        "missing-annotation"
        if annotation_selection == "missing"
        else excluded_annotation
    )
    arguments = _base_arguments(
        selected_revision,
        codebook_id,
        selected_annotation,
        second_annotation,
    )

    with pytest.raises(ToolError) as captured:
        _call_tool(arguments)
    assert isinstance(captured.value.__cause__, ScopeViolationError)

    assert len(opened) == 1
    assert close_calls == opened
    with Project.open(root, read_only=True) as reopened:
        runs = reopened.db.execute("SELECT status FROM analysis_runs").fetchall()
        assert len(runs) == 1
        assert runs[0]["status"] == "FAILED"
