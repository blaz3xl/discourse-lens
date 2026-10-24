from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.mcp_server import list_model_costs
from discourselens.models import ModelPrice


def test_mcp_lists_bounded_model_costs_in_read_only_mode(tmp_path, monkeypatch):
    root = tmp_path / "mcp-model-costs"
    project = Project.init(root)
    price = ModelPrice(
        provider_id="openai-compatible",
        model="fixture-model",
        input_per_million_tokens="1",
        output_per_million_tokens="2",
        currency="USD",
        source_reference="https://provider.example/pricing-v1",
        rate_version="v1",
        rate_date="2026-01-01",
    )
    first_id = project.record_model_execution(
        "openai-compatible",
        "private-alias",
        "fixture-model",
        None,
        "request-one",
        "response-one",
        input_tokens=1_000,
        output_tokens=500,
        model_price=price,
    )
    second_id = project.record_model_execution(
        "openai-compatible",
        "another-private-alias",
        "fixture-model",
        None,
        "request-two",
        "response-two",
    )
    project.db.execute(
        "UPDATE model_executions SET started_at=? WHERE id=?",
        ("2026-01-01T00:00:01+00:00", first_id),
    )
    project.db.execute(
        "UPDATE model_executions SET started_at=? WHERE id=?",
        ("2026-01-01T00:00:02+00:00", second_id),
    )
    project.db.commit()
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    first_page = list_model_costs(limit=1)
    second_page = list_model_costs(offset=1, limit=1)

    assert first_page["total_count"] == second_page["total_count"] == 2
    assert first_page["has_more"] is True
    assert second_page["has_more"] is False
    first_rows = cast(list[dict[str, object]], first_page["rows"])
    second_rows = cast(list[dict[str, object]], second_page["rows"])
    assert first_rows[0]["execution_id"] == first_id
    assert second_rows[0]["execution_id"] == second_id
    serialized = json.dumps(first_page)
    assert "private-alias" not in serialized
    assert "request_hash" not in serialized
    assert "response_hash" not in serialized
    assert "credential_alias" not in serialized

    db = Project.open(root, read_only=True)
    try:
        assert db.db.total_changes == 0
    finally:
        db.close()


def test_mcp_model_cost_tool_is_discoverable_in_read_only_registry():
    tools = asyncio.run(mcp_module.server.list_tools())
    tool = next(item for item in tools if item.name == "list_model_costs")
    schema = tool.model_dump(by_alias=True)["inputSchema"]

    assert {"offset", "limit"} <= set(schema["properties"])
    assert "list_model_costs" in mcp_module.READ_ONLY_MCP_TOOL_NAMES


@pytest.mark.parametrize(("offset", "limit"), [(-1, 10), (0, 0), (0, 101)])
def test_mcp_model_cost_tool_rejects_out_of_range_pages(offset, limit):
    with pytest.raises(ProjectError, match="model cost page"):
        list_model_costs(offset=offset, limit=limit)
