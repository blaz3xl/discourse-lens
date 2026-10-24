from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import fitz
import pytest
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def _write_pdf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def _prepare(root: Path, label: str) -> tuple[Path, Path, str, str]:
    project_root = root / f"{label}-project"
    source = root / f"{label}-inbox" / "source.pdf"
    _write_pdf(source, "Revision A is archived.")
    with Project.init(project_root) as project:
        imported = project.import_document(source)
    assert imported.document_id is not None
    assert imported.revision_id is not None
    return project_root, source, imported.document_id, imported.revision_id


def _assert_reprocessed(payload: dict[str, Any], document_id: str, first_revision_id: str) -> None:
    assert payload["status"] == "SUCCESS"
    assert payload["document_id"] == document_id
    assert isinstance(payload["revision_id"], str)
    assert payload["revision_id"] != first_revision_id


def test_pdf_reprocessing_contract_matches_sdk_cli_and_mcp(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("mcp", reason="MCP reprocess parity requires the optional 'mcp' extra")
    import discourselens.mcp_server as mcp_module

    sdk_root, sdk_source, sdk_document, sdk_first_revision = _prepare(tmp_path, "sdk")
    _write_pdf(sdk_source, "Revision B is preserved.")
    with Project.open(sdk_root) as project:
        sdk_payload = asdict(project.reprocess_document(sdk_document, sdk_source))

    cli_root, cli_source, cli_document, cli_first_revision = _prepare(tmp_path, "cli")
    _write_pdf(cli_source, "Revision B is preserved.")
    cli_result = CliRunner().invoke(
        app,
        ["document", "reprocess", str(cli_root), cli_document, str(cli_source)],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_payload = json.loads(cli_result.output)

    mcp_root = tmp_path / "mcp-project"
    mcp_inbox = tmp_path / "mcp-inbox"
    mcp_source = mcp_inbox / "source.pdf"
    _write_pdf(mcp_source, "Revision A is archived.")
    with Project.init(mcp_root) as project:
        mcp_initial = project.import_document(mcp_source)
    assert mcp_initial.document_id is not None
    assert mcp_initial.revision_id is not None
    _write_pdf(mcp_source, "Revision B is preserved.")
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(mcp_root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(mcp_inbox))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
    mcp_payload = cast(
        dict[str, Any],
        mcp_module.reprocess_document(mcp_initial.document_id, str(mcp_source)),
    )

    _assert_reprocessed(sdk_payload, sdk_document, sdk_first_revision)
    _assert_reprocessed(cli_payload, cli_document, cli_first_revision)
    _assert_reprocessed(mcp_payload, mcp_initial.document_id, mcp_initial.revision_id)
    for payload in (sdk_payload, cli_payload, mcp_payload):
        assert "source.pdf" not in json.dumps(payload)
