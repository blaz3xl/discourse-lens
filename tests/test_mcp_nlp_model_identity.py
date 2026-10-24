"""MCP regression tests for spaCy model distribution identity checks."""

from __future__ import annotations

import asyncio
import re
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, InputRequiredResult

import discourselens.mcp_server as mcp_module
import discourselens.nlp as nlp_module


def _write_unrelated_spacy_model(root: Path, model_id: str, marker: Path) -> None:
    module_name = model_id.replace("-", "_")
    package = root / module_name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )

    normalized_name = re.sub(r"[-_.]+", "_", model_id)
    dist_info_name = f"{normalized_name}-1.0.dist-info"
    dist_info = root / dist_info_name
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {model_id}\nVersion: 1.0\n",
        encoding="utf-8",
    )
    record_entries = [
        f"{module_name}/__init__.py,,",
        f"{dist_info_name}/METADATA,,",
        f"{dist_info_name}/RECORD,,",
    ]
    (dist_info / "RECORD").write_text("\n".join(record_entries) + "\n", encoding="utf-8")


def _call_parsed_token_tool(
    arguments: dict[str, object],
) -> CallToolResult | InputRequiredResult:
    async def invoke() -> CallToolResult | InputRequiredResult:
        return await mcp_module.server.call_tool("build_parsed_token_artifact", arguments)

    return asyncio.run(invoke())


def test_mcp_parsed_token_route_rejects_unrelated_spacy_model_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "en_unrelated_fixture_model"
    module_name = model_id.replace("-", "_")
    shadow_root = tmp_path / "unrelated-model-site"
    marker = tmp_path / "model-initializer-executed"
    shadow_root.mkdir()
    _write_unrelated_spacy_model(shadow_root, model_id, marker)
    monkeypatch.syspath_prepend(str(shadow_root))

    # Treat the temporary root as an installed package root so this exercises
    # model identity metadata, rather than relying on a path rejection.
    discovered = distribution(model_id)
    assert Path(str(discovered.locate_file(""))).resolve() == shadow_root.resolve()
    monkeypatch.setattr(
        nlp_module,
        "_approved_install_roots",
        lambda: (shadow_root.resolve(),),
    )

    class StubProject:
        def document_revision_summary(self, revision_id: str) -> dict[str, str]:
            assert revision_id == "revision-safe"
            return {"document_id": "document-safe"}

        def document_metadata(self, document_id: str) -> dict[str, str]:
            assert document_id == "document-safe"
            return {"language": "en"}

        def close(self) -> None:
            return None

    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)
    monkeypatch.setattr(mcp_module, "_open_project", StubProject)

    with pytest.raises(
        ToolError,
        match=(
            "selected local NLP model is unavailable; "
            "no downloads or fallback were attempted"
        ),
    ) as captured:
        _call_parsed_token_tool(
            {
                "revision_id": "revision-safe",
                "backend": "spacy",
                "model": model_id,
            }
        )

    assert not marker.exists()
    assert module_name not in sys.modules
    assert str(shadow_root) not in str(captured.value)
    assert "input_value" not in str(captured.value)
