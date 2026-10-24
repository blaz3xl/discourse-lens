"""Opt-in real PDF → web API → Ollama → read-only MCP acceptance flow.

Run inside the web container with
``DISCOURSELENS_RUN_LOCAL_OLLAMA_INTEGRATION=1`` after the Ollama service is
healthy and the configured model has been pulled.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DISCOURSELENS_RUN_LOCAL_OLLAMA_INTEGRATION") != "1",
    reason="set DISCOURSELENS_RUN_LOCAL_OLLAMA_INTEGRATION=1 to run with live Ollama",
)


def test_web_pdf_import_local_embedding_search_citation_and_reopen(tmp_path: Path) -> None:
    from scripts.smoke_local_ollama import run_smoke

    result = run_smoke(tmp_path)
    assert result["web_pdf_import"] == "ok"
    assert result["real_ollama_index"] == "ok"
    assert result["project_reopen"] == "ok"
    assert result["mcp_readonly"] == "ok"
    assert isinstance(result["mcp_tools"], int) and result["mcp_tools"] >= 1
    assert isinstance(result["mcp_skills"], int) and result["mcp_skills"] >= 1
    assert result["mcp_llm_skill_manifest"] == 1
    assert isinstance(result["mcp_skill_methodology_chars"], int)
    assert result["mcp_skill_methodology_chars"] >= 100
    assert isinstance(result["mcp_semantic_hits"], int) and result["mcp_semantic_hits"] >= 1
