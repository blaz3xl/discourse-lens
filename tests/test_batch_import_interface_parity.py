from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def _project_with_batch_inputs(root: Path) -> tuple[Path, Path, Path]:
    import_root = root / "inbox"
    import_root.mkdir(parents=True)
    Project.init(root).close()

    first = import_root / "first.txt"
    duplicate = import_root / "same-content-copy.txt"
    invalid_utf8 = import_root / "invalid-utf8.txt"
    first.write_text("Shared source text.", encoding="utf-8")
    duplicate.write_text("Shared source text.", encoding="utf-8")
    invalid_utf8.write_bytes(b"Invalid UTF-8: \xff")
    return first, duplicate, invalid_utf8


def _semantic_view(
    payload: dict[str, Any],
) -> tuple[tuple[int, int, int, int], list[tuple[int, str, str | None]]]:
    counts = (
        payload["total_count"],
        payload["success_count"],
        payload["duplicate_count"],
        payload["failed_count"],
    )
    outcomes = [
        (item["index"], item["status"], item["failure_code"])
        for item in payload["items"]
    ]
    return counts, outcomes


def test_batch_import_contract_matches_sdk_cli_and_mcp(tmp_path: Path, monkeypatch):
    pytest.importorskip("mcp", reason="MCP batch-import parity requires the optional 'mcp' extra")
    import discourselens.mcp_server as mcp_module

    sdk_root = tmp_path / "sdk-project"
    sdk_paths = _project_with_batch_inputs(sdk_root)
    with Project.open(sdk_root) as project:
        sdk_result = project.import_documents(sdk_paths)
    sdk_payload = asdict(sdk_result)

    cli_root = tmp_path / "cli-project"
    cli_paths = _project_with_batch_inputs(cli_root)
    cli_result = CliRunner().invoke(
        app,
        ["document", "import-many", str(cli_root), *(str(path) for path in cli_paths)],
    )
    assert cli_result.exit_code == 1, cli_result.output
    cli_payload = json.loads(cli_result.output)

    mcp_root = tmp_path / "mcp-project"
    mcp_paths = _project_with_batch_inputs(mcp_root)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(mcp_root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(mcp_root / "inbox"))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
    mcp_payload = cast(
        dict[str, Any], mcp_module.import_documents([str(path) for path in mcp_paths])
    )

    expected = (
        (3, 1, 1, 1),
        [(0, "SUCCESS", None), (1, "DUPLICATE", None), (2, "FAILED", "DOCUMENT_ERROR")],
    )
    assert _semantic_view(sdk_payload) == expected
    assert _semantic_view(cli_payload) == expected
    assert _semantic_view(mcp_payload) == expected

    # Generated IDs are project-specific; compare duplicate relationships within
    # each independent project rather than comparing IDs between adapters.
    for payload in (sdk_payload, cli_payload, mcp_payload):
        items = payload["items"]
        assert items[0]["document_id"] is not None
        assert items[0]["revision_id"] is not None
        assert items[1]["document_id"] == items[0]["document_id"]
        assert items[1]["revision_id"] == items[0]["revision_id"]
        assert items[2]["document_id"] is None
        assert items[2]["revision_id"] is None
