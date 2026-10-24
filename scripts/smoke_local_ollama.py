"""Exercise PDF import, local embeddings, semantic citations, and persistence."""

from __future__ import annotations

import asyncio
import io
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pymupdf as fitz
from fastapi.testclient import TestClient

from discourselens.models import CredentialReference, ModelPreset
from discourselens.project import Project
from discourselens.web import create_app


def _mcp_semantic_smoke(
    project_path: Path,
    scope_id: str,
    artifact_id: str,
    provider_env: str,
    token: str,
) -> dict[str, int]:
    executable = shutil.which("discourselens-mcp")
    if executable is None:
        raise RuntimeError("discourselens-mcp executable is required for MCP acceptance")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}/mcp"
    environment = os.environ.copy()
    environment["DISCOURSELENS_MCP_HTTP_TOKEN"] = token
    environment[provider_env] = os.environ[provider_env]
    with tempfile.TemporaryFile(mode="w+t") as server_log:
        server = subprocess.Popen(
            [
                executable,
                "--project",
                str(project_path),
                "--read-only",
                "--transport",
                "streamable-http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            return asyncio.run(
                _call_mcp_semantic(
                    endpoint,
                    token,
                    scope_id,
                    artifact_id,
                )
            )
        except Exception as exc:
            server_log.flush()
            server_log.seek(0)
            log = server_log.read()[-4000:]
            raise RuntimeError(f"read-only MCP acceptance failed: {exc}\n{log}") from exc
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


async def _call_mcp_semantic(
    endpoint: str,
    token: str,
    scope_id: str,
    artifact_id: str,
) -> dict[str, int]:
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.types import TextResourceContents

    headers = {"Authorization": f"Bearer {token}"}
    last_error: Exception | None = None
    for _ in range(40):
        try:
            async with httpx2.AsyncClient() as unauthenticated_client:
                unauthorized = await unauthenticated_client.get(endpoint)
            if unauthorized.status_code != 401:
                raise AssertionError(
                    "MCP endpoint must reject requests without its bearer token"
                )
            async with (
                httpx2.AsyncClient(headers=headers) as client,
                streamable_http_client(endpoint, http_client=client) as (
                    read_stream,
                    write_stream,
                ),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                tool_result = await session.list_tools()
                names = {tool.name for tool in tool_result.tools}
                if "search_semantic_passages" not in names:
                    raise AssertionError("read-only semantic search tool is missing")
                if "build_embedding_index" in names:
                    raise AssertionError("read-only MCP exposed an index write tool")
                skill_result = await session.call_tool("list_skills", {})
                skill_data = skill_result.structured_content or {}
                skill_items = skill_data.get("result", [])
                if not isinstance(skill_items, list) or not skill_items:
                    raise AssertionError("MCP skills catalog is empty")
                van_dijk_skill_id = "builtin.van-dijk-ideology"
                skill_detail = await session.call_tool(
                    "get_skill", {"skill_id": van_dijk_skill_id}
                )
                if skill_detail.is_error:
                    raise RuntimeError("MCP could not return the Van Dijk skill manifest")
                manifest = skill_detail.structured_content or {}
                if (
                    manifest.get("id") != van_dijk_skill_id
                    or manifest.get("execution_kind") != "LLM_ASSISTED"
                    or not manifest.get("prompt_refs")
                    or not manifest.get("model_input_schema_ref")
                ):
                    raise AssertionError("Van Dijk LLM skill manifest is incomplete")
                methodology = await session.read_resource(
                    f"discourselens://skills/{van_dijk_skill_id}/methodology"
                )
                methodology_chars = sum(
                    len(item.text)
                    for item in methodology.contents
                    if isinstance(item, TextResourceContents)
                )
                if methodology_chars < 100:
                    raise AssertionError("Van Dijk methodology resource is missing")
                search_result = await session.call_tool(
                    "search_semantic_passages",
                    {
                        "scope_id": scope_id,
                        "artifact_ids": [artifact_id],
                        "preset_name": "embed-local",
                            "query_text": "migration and social actors",
                        "limit": 5,
                    },
                )
                if search_result.is_error:
                    messages = [
                        item.text
                        for item in search_result.content
                        if item.type == "text"
                    ]
                    raise RuntimeError(f"MCP semantic tool failed: {' '.join(messages)}")
                result_data = search_result.structured_content or {}
                result_data = result_data.get("result", result_data)
                if not isinstance(result_data, dict):
                    raise TypeError("MCP semantic response has an invalid shape")
                hits = result_data.get("hits", [])
                if not isinstance(hits, list) or not hits:
                    raise AssertionError("MCP semantic retrieval returned no passages")
                if not all(
                    isinstance(hit, dict)
                    and isinstance(hit.get("text"), str)
                    and isinstance(hit.get("citation"), str)
                    and hit["citation"]
                    for hit in hits
                ):
                    raise AssertionError("MCP semantic hits lack cited text")
                return {
                    "tools": len(names),
                    "skills": len(skill_items),
                    "llm_skill_manifest": 1,
                    "skill_methodology_chars": methodology_chars,
                    "semantic_hits": len(hits),
                }
        except httpx2.TransportError as exc:
            last_error = exc
            await asyncio.sleep(0.25)
    raise RuntimeError(f"MCP server did not become ready: {last_error!r}")


def run_smoke(project_parent: Path) -> dict[str, object]:
    provider_env = "DL_OLLAMA_API_KEY"
    if not os.environ.get(provider_env):
        raise RuntimeError(f"{provider_env} must be available inside the web container")

    model = os.environ.get("DISCOURSELENS_LOCAL_EMBEDDING_MODEL", "embeddinggemma:300m-qat-q4_0")
    endpoint = os.environ.get("DISCOURSELENS_LOCAL_EMBEDDING_ENDPOINT", "http://ollama:11434/v1")
    project_path = project_parent / "acceptance-project"
    project = Project.init(project_path)
    project.save_credential_reference(
        CredentialReference("ollama", "local", provider_env, endpoint)
    )
    project.save_model_preset(ModelPreset("embed-local", "ollama", "local", model))
    project.close()

    pdf = fitz.open()
    for lines in (
        (
            "Political discourse is examined through public institutions.",
            "Researchers identify representation and agency in the text.",
        ),
        (
            "Social actors are described in migration policy debates.",
            "Evidence and citations support close linguistic analysis.",
        ),
    ):
        page = pdf.new_page(width=595, height=842)
        for line_number, line in enumerate(lines):
            page.insert_text((72, 100 + line_number * 35), line, fontsize=12)
    source = io.BytesIO(pdf.tobytes()).getvalue()
    pdf.close()

    token = "local-ollama-acceptance-test-token-123456"
    app = create_app(project_path, api_token=token)
    headers = {"Authorization": f"Bearer {token}"}
    with TestClient(app) as client:
        if client.get("/api/documents").status_code != 401:
            raise AssertionError("web API must reject requests without a bearer token")
        imported = client.post(
            "/api/documents",
            content=source,
            headers={**headers, "X-Document-Filename": "evaluation-fixture.pdf"},
        )
        if imported.status_code != 201:
            raise AssertionError(f"PDF import failed: {imported.text}")
        revision_id = imported.json()["revision_id"]

        paragraphs = client.get(
            f"/api/documents/{revision_id}/paragraphs", headers=headers
        )
        if paragraphs.status_code != 200 or len(paragraphs.json()["items"]) != 4:
            raise AssertionError(f"paragraph projection failed: {paragraphs.text}")

        preview = client.post(
            "/api/retrieval/preview",
            headers=headers,
            json={"revision_id": revision_id, "preset_name": "embed-local"},
        )
        if preview.status_code != 200 or preview.json()["provider_id"] != "ollama":
            raise AssertionError(f"embedding preview failed: {preview.text}")
        if preview.json()["network_request_made"] is not False:
            raise AssertionError("preview must not send document text to the provider")

        indexed = client.post(
            "/api/retrieval/index",
            headers=headers,
            json={
                "revision_id": revision_id,
                "preset_name": "embed-local",
                "provider_egress_acknowledged": True,
            },
        )
        if indexed.status_code != 200:
            raise AssertionError(f"local vector indexing failed: {indexed.text}")
        artifact_id = indexed.json()["artifact_id"]

        search = client.post(
            "/api/retrieval/search",
            headers=headers,
            json={
                "revision_id": revision_id,
                "preset_name": "embed-local",
                "artifact_id": artifact_id,
                "query": "migration and social actors",
                "query_egress_acknowledged": True,
                "limit": 5,
            },
        )
        if search.status_code != 200:
            raise AssertionError(f"semantic search failed: {search.text}")
        rows = search.json()["rows"]
        if not rows or not all(row["citation"] and row["text"] for row in rows):
            raise AssertionError("semantic results must contain cited passages")

    with Project.open(project_path) as reopened:
        if not reopened.vector_index_artifacts(revision_id):
            raise AssertionError("vector index did not persist after project reopen")
        if reopened.document_revision_summary(revision_id)["revision_id"] != revision_id:
            raise AssertionError("document revision did not persist after project reopen")
        scope_id = reopened.snapshot([revision_id])

    mcp_result = _mcp_semantic_smoke(
        project_path,
        scope_id,
        artifact_id,
        provider_env,
        token,
    )

    digital = fitz.open()
    digital_page = digital.new_page(width=595, height=842)
    digital_page.insert_text((72, 140), "Scanned test document for optical recognition.", fontsize=18)
    raster = digital_page.get_pixmap(dpi=240)
    scanned = fitz.open()
    scanned_page = scanned.new_page(width=595, height=842)
    scanned_page.insert_image(scanned_page.rect, pixmap=raster)
    scanned_source = io.BytesIO(scanned.tobytes()).getvalue()
    digital.close()
    scanned.close()

    with TestClient(app) as client:
        scanned_import = client.post(
            "/api/documents",
            content=scanned_source,
            headers={
                **headers,
                "X-Document-Filename": "scanned-evaluation.pdf",
                "X-OCR-Language": "eng",
                "X-OCR-DPI": "300",
            },
        )
        if scanned_import.status_code != 201:
            raise AssertionError(f"scanned PDF OCR import failed: {scanned_import.text}")
        scanned_revision = client.get(
            f"/api/documents/{scanned_import.json()['revision_id']}", headers=headers
        )
        if scanned_revision.status_code != 200:
            raise AssertionError(f"scanned PDF revision could not be read: {scanned_revision.text}")
        extracted = scanned_revision.json()["canonical_text_preview"].lower()
        if "optical recognition" not in extracted:
            raise AssertionError(f"OCR text was not extracted: {extracted!r}")

    return {
        "web_pdf_import": "ok",
        "paragraphs": len(paragraphs.json()["items"]),
        "real_ollama_index": "ok",
        "semantic_hits": len(rows),
        "cited": True,
        "project_reopen": "ok",
        "mcp_readonly": "ok",
        "mcp_tools": mcp_result["tools"],
        "mcp_skills": mcp_result["skills"],
        "mcp_llm_skill_manifest": mcp_result["llm_skill_manifest"],
        "mcp_skill_methodology_chars": mcp_result["skill_methodology_chars"],
        "mcp_semantic_hits": mcp_result["semantic_hits"],
        "scanned_pdf_ocr": "ok",
        "unauthorized_api": "401",
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="discourselens-acceptance-") as directory:
        print(run_smoke(Path(directory)))


if __name__ == "__main__":
    main()
