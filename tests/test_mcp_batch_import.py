from __future__ import annotations

import asyncio
import json
from typing import Literal, TypedDict, cast

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError


class BatchImportItem(TypedDict):
    index: int
    status: Literal["SUCCESS", "DUPLICATE", "FAILED"]
    document_id: str | None
    revision_id: str | None
    failure_code: str | None
    warnings: list[str]


class BatchImportPayload(TypedDict):
    total_count: int
    success_count: int
    duplicate_count: int
    failed_count: int
    items: list[BatchImportItem]


def _batch_payload(value: dict[str, object]) -> BatchImportPayload:
    return cast(BatchImportPayload, value)


def test_mcp_batch_import_preserves_order_and_duplicate_status(tmp_path, monkeypatch):
    root = tmp_path / "project"
    import_root = root / "inbox"
    import_root.mkdir(parents=True)
    first_path = import_root / "first.txt"
    second_path = import_root / "second.txt"
    first_path.write_text("First source text.", encoding="utf-8")
    second_path.write_text("Second source text.", encoding="utf-8")
    Project.init(root).close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(import_root))

    result = _batch_payload(
        mcp_module.import_documents([str(second_path), str(first_path), str(second_path)])
    )

    assert result["total_count"] == 3
    assert result["success_count"] == 2
    assert result["duplicate_count"] == 1
    assert result["failed_count"] == 0
    items = result["items"]
    assert [item["index"] for item in items] == [0, 1, 2]
    assert [item["status"] for item in items] == ["SUCCESS", "SUCCESS", "DUPLICATE"]
    assert items[0]["document_id"] == items[2]["document_id"]
    assert items[0]["revision_id"] == items[2]["revision_id"]
    assert items[1]["document_id"] != items[0]["document_id"]
    assert set(items[0]) == {
        "index",
        "status",
        "document_id",
        "revision_id",
        "failure_code",
        "warnings",
    }
    with Project.open(root, read_only=True) as reopened:
        names = {
            str(row["display_name"]): str(row["id"])
            for row in reopened.db.execute("SELECT id,display_name FROM documents")
        }
        assert items[0]["document_id"] == names["second.txt"]
        assert items[1]["document_id"] == names["first.txt"]


def test_mcp_batch_import_rejects_out_of_scope_paths_before_core_and_merges_outcomes(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    import_root = root / "inbox"
    import_root.mkdir(parents=True)
    in_scope = import_root / "allowed.txt"
    in_scope.write_text("Allowed source text.", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("Private outside text.", encoding="utf-8")
    Project.init(root).close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(import_root))

    passed_to_core = []
    original_import_payloads = Project.import_document_payloads

    def capture_import_payloads(project, payloads, **kwargs):
        selected_payloads = list(payloads)
        passed_to_core.extend(selected_payloads)
        return original_import_payloads(project, selected_payloads, **kwargs)

    monkeypatch.setattr(Project, "import_document_payloads", capture_import_payloads)
    result = _batch_payload(mcp_module.import_documents([str(outside), str(in_scope)]))

    assert passed_to_core == [("allowed.txt", b"Allowed source text.")]
    assert result["total_count"] == 2
    assert result["success_count"] == 1
    assert result["duplicate_count"] == 0
    assert result["failed_count"] == 1
    items = result["items"]
    assert [item["index"] for item in items] == [0, 1]
    assert items[0] == {
        "index": 0,
        "status": "FAILED",
        "document_id": None,
        "revision_id": None,
        "failure_code": "OUTSIDE_IMPORT_ROOT",
        "warnings": [],
    }
    assert items[1]["status"] == "SUCCESS"
    serialized = json.dumps(result)
    assert str(outside) not in serialized
    assert str(import_root) not in serialized


def test_mcp_batch_import_rejects_symlink_target_outside_root_before_core(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    import_root = root / "inbox"
    import_root.mkdir(parents=True)
    outside = tmp_path / "private.txt"
    outside.write_text("Private outside text.", encoding="utf-8")
    link = import_root / "linked.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable or not permitted: {exc}")
    Project.init(root).close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(import_root))

    core_calls = []

    def reject_core_call(project, paths, **kwargs):
        core_calls.append(list(paths))
        pytest.fail("a symlink resolving outside the import root reached Core")

    monkeypatch.setattr(Project, "import_documents", reject_core_call)
    result = _batch_payload(mcp_module.import_documents([str(link)]))

    assert core_calls == []
    assert result["total_count"] == 1
    assert result["success_count"] == 0
    assert result["duplicate_count"] == 0
    assert result["failed_count"] == 1
    assert result["items"] == [
        {
            "index": 0,
            "status": "FAILED",
            "document_id": None,
            "revision_id": None,
            "failure_code": "OUTSIDE_IMPORT_ROOT",
            "warnings": [],
        }
    ]
    serialized = json.dumps(result)
    assert str(root) not in serialized
    assert str(import_root) not in serialized
    assert str(link) not in serialized
    assert str(outside) not in serialized


def test_mcp_batch_import_retains_successes_around_a_core_file_failure(tmp_path, monkeypatch):
    root = tmp_path / "project"
    import_root = root / "inbox"
    import_root.mkdir(parents=True)
    first_path = import_root / "first.txt"
    broken_path = import_root / "broken.txt"
    last_path = import_root / "last.txt"
    first_path.write_text("First source text.", encoding="utf-8")
    broken_path.write_bytes(b"not utf-8: \xff")
    last_path.write_text("Last source text.", encoding="utf-8")
    Project.init(root).close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(import_root))

    result = _batch_payload(
        mcp_module.import_documents([str(first_path), str(broken_path), str(last_path)])
    )

    assert [item["index"] for item in result["items"]] == [0, 1, 2]
    assert [item["status"] for item in result["items"]] == [
        "SUCCESS",
        "FAILED",
        "SUCCESS",
    ]
    assert result["total_count"] == 3
    assert result["success_count"] == 2
    assert result["duplicate_count"] == 0
    assert result["failed_count"] == 1
    assert result["items"][1]["failure_code"] == "DOCUMENT_ERROR"
    assert result["items"][1]["document_id"] is None
    assert result["items"][1]["revision_id"] is None
    assert str(import_root) not in json.dumps(result)


def test_mcp_batch_import_is_bounded_and_unavailable_in_read_only_mode(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    import_root = root / "inbox"
    import_root.mkdir(parents=True)
    source = import_root / "source.txt"
    source.write_text("Source text.", encoding="utf-8")
    Project.init(root).close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(import_root))

    def unexpected_project_open():
        pytest.fail("invalid or read-only import requests must not open the project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_project_open)
    with pytest.raises(ToolError, match="1..100"):
        mcp_module.import_documents([])
    with pytest.raises(ToolError, match="1..100"):
        mcp_module.import_documents([str(source)] * 101)
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    try:
        with pytest.raises(ProjectError, match="read-only"):
            mcp_module.import_documents([str(source)])
        async def readonly_tool_names() -> set[str]:
            await mcp_module._configure_read_only_server()
            return {tool.name for tool in await mcp_module.server.list_tools()}

        exposed = asyncio.run(readonly_tool_names())
        assert "import_document" not in exposed
        assert "import_documents" not in exposed
    finally:
        monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
        mcp_module._restore_full_server_registry()

    tools = asyncio.run(mcp_module.server.list_tools())
    batch_tool = next(tool for tool in tools if tool.name == "import_documents")
    schema = batch_tool.model_dump(by_alias=True)["inputSchema"]
    path_schema = schema["properties"]["paths"]
    assert path_schema["minItems"] == 1
    assert path_schema["maxItems"] == 100
    assert "paths" in schema["required"]
    assert "import_documents" not in mcp_module.READ_ONLY_MCP_TOOL_NAMES
