from __future__ import annotations

import asyncio
from typing import cast

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.mcp_server import list_analysis_runs


def _create_analysis_runs(root):
    project = Project.init(root)
    _, revision = project.import_text("source.txt", "A local analysis fixture.")
    scope_id = project.snapshot([revision])
    run_ids = []
    for index in range(3):
        run_id = project.create_run(scope_id, {"profile": f"fixture-{index}"})
        run_ids.append(run_id)
    expected_first = project.analysis_runs_page(offset=0, limit=2)
    expected_second = project.analysis_runs_page(offset=2, limit=2)
    project.close()
    return run_ids, expected_first, expected_second


def test_mcp_analysis_run_listing_delegates_bounded_pages_in_read_only_mode(
    tmp_path, monkeypatch
):
    root = tmp_path / "mcp-analysis-runs-page"
    _, expected_first, expected_second = _create_analysis_runs(root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    first_page = list_analysis_runs(offset=0, limit=2)
    second_page = list_analysis_runs(offset=2, limit=2)

    assert first_page == expected_first
    assert second_page == expected_second
    assert first_page["offset"] == 0
    assert first_page["limit"] == 2
    assert first_page["total"] == second_page["total"] == 3
    assert first_page["has_more"] is True
    assert second_page["has_more"] is False
    assert len(cast(list[dict[str, object]], first_page["items"])) == 2
    assert len(cast(list[dict[str, object]], second_page["items"])) == 1
    assert database.read_bytes() == before

    tool = next(item for item in asyncio.run(mcp_module.server.list_tools()) if item.name == "list_analysis_runs")
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    assert {"offset", "limit"} <= set(schema["properties"])
    assert "list_analysis_runs" in mcp_module.READ_ONLY_MCP_TOOL_NAMES


@pytest.mark.parametrize(("offset", "limit"), [(-1, 1), (0, 0), (0, 1_001)])
def test_mcp_analysis_run_listing_rejects_invalid_page_bounds(offset, limit):
    with pytest.raises(ProjectError, match="analysis runs page"):
        list_analysis_runs(offset=offset, limit=limit)
