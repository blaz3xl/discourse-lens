from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi.testclient import TestClient

from discourselens.project import Project
from discourselens.vectors import RetrievalSearchResult, VectorHit
from discourselens.web import create_app

TOKEN = "d" * 64


@pytest.fixture
def web_client(tmp_path: Path):
    project_path = tmp_path / "project"
    Project.init(project_path).close()
    with TestClient(create_app(project_path, TOKEN)) as client:
        yield client


def test_web_landing_requires_token_for_project_data(web_client: TestClient):
    landing = web_client.get("/")
    assert landing.status_code == 200
    assert "Herramientas para investigar el discurso" in landing.text
    assert 'id="espacio"' in landing.text
    assert 'id="chatReasoning"' in landing.text
    assert "chat-controls{display:grid!important" in landing.text
    assert 'id="openSettingsSide"' in landing.text
    assert "Enter envía · Shift+Enter añade una línea" in landing.text
    assert "Conectar y cargar modelos" in landing.text
    assert web_client.get("/healthz").json() == {"status": "ok"}
    assert web_client.get("/api/documents").status_code == 401
    assert web_client.get(
        "/api/documents", headers={"Authorization": f"Bearer {TOKEN}"}
    ).json()["total"] == 0


def test_web_connector_status_reports_unconfigured_oauth_without_secrets(
    web_client: TestClient, monkeypatch
):
    monkeypatch.setenv("DISCOURSELENS_MCP_PUBLIC_URL", "https://research.example.test/mcp")
    for name in (
        "DISCOURSELENS_MCP_OAUTH_ISSUER_URL",
        "DISCOURSELENS_MCP_OAUTH_RESOURCE_URL",
        "DISCOURSELENS_MCP_OAUTH_AUDIENCE",
        "DISCOURSELENS_MCP_OAUTH_JWKS_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    response = web_client.get(
        "/api/connectors/status", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 200
    assert response.json()["mcp_url"] == "https://research.example.test/mcp"
    assert response.json()["endpoint_https"] is True
    assert response.json()["mcp_oauth_configured"] is False
    assert response.json()["configuration_aligned"] is False
    assert "web_login_enabled" not in response.json()


def test_web_connector_oauth_does_not_require_web_login(web_client: TestClient, monkeypatch):
    monkeypatch.setenv("DISCOURSELENS_MCP_PUBLIC_URL", "https://research.example.test/mcp")
    monkeypatch.setenv("DISCOURSELENS_MCP_OAUTH_ISSUER_URL", "https://login.example.test/issuer")
    monkeypatch.setenv(
        "DISCOURSELENS_MCP_OAUTH_RESOURCE_URL", "https://research.example.test/mcp"
    )
    monkeypatch.setenv("DISCOURSELENS_MCP_OAUTH_AUDIENCE", "https://research.example.test/mcp")
    monkeypatch.setenv("DISCOURSELENS_MCP_OAUTH_JWKS_URL", "https://login.example.test/jwks")
    response = web_client.get(
        "/api/connectors/status", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 200
    assert response.json()["configuration_aligned"] is True
    assert response.json()["mcp_oauth_configured"] is True


def test_web_connector_cloudflare_bridge_uses_shared_project_without_web_oidc(
    web_client: TestClient, monkeypatch
):
    monkeypatch.setenv("DISCOURSELENS_MCP_PUBLIC_URL", "https://research.example.test/mcp")
    monkeypatch.setenv("DISCOURSELENS_MCP_CONNECTOR_MODE", "cloudflare-oauth-proxy")
    for name in (
        "DISCOURSELENS_MCP_OAUTH_ISSUER_URL",
        "DISCOURSELENS_MCP_OAUTH_RESOURCE_URL",
        "DISCOURSELENS_MCP_OAUTH_AUDIENCE",
        "DISCOURSELENS_MCP_OAUTH_JWKS_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    response = web_client.get(
        "/api/connectors/status", headers={"Authorization": f"Bearer {TOKEN}"}
    )

    assert response.status_code == 200
    assert response.json()["configuration_aligned"] is True
    assert response.json()["connector_mode"] == "cloudflare-oauth-proxy"
    assert response.json()["access_model"] == "proyecto compartido"
    assert response.json()["tenant_routing_enabled"] is False


def test_web_saves_model_preset_without_returning_provider_secret(
    web_client: TestClient, monkeypatch
):
    from cryptography.fernet import Fernet

    monkeypatch.setenv(
        "DISCOURSELENS_CREDENTIALS_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii")
    )
    provider_secret = Fernet.generate_key().decode("ascii")
    headers = {"Authorization": f"Bearer {TOKEN}"}
    credential = web_client.post(
        "/api/provider-credentials",
        headers=headers,
        json={
            "provider_id": "openai",
            "alias": "test-chat",
            "secret": provider_secret,
        },
    )
    assert credential.status_code == 201
    assert provider_secret not in credential.text

    saved = web_client.post(
        "/api/model-presets",
        headers=headers,
        json={
            "name": "linguist-chat",
            "provider_id": "openai",
            "credential_alias": "test-chat",
            "model": "gpt-example",
            "reasoning": "high",
        },
    )
    assert saved.status_code == 201
    listed = web_client.get("/api/model-presets", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["items"] == [
        {
            "id": saved.json()["id"],
            "name": "linguist-chat",
            "provider_id": "openai",
            "credential_alias": "test-chat",
            "model": "gpt-example",
            "reasoning": "high",
            "reasoning_mode": None,
            "reasoning_budget_tokens": None,
        }
    ]
    assert "sk-test-secret-not-returned" not in listed.text


def test_web_discovers_models_using_only_saved_credential(web_client, monkeypatch):
    calls = []

    def fake_discovery(project, provider_id, credential_alias):
        assert isinstance(project, Project)
        calls.append((provider_id, credential_alias))
        return [{"model": "gpt-test", "display_name": "Test Chat"}]

    monkeypatch.setattr(
        "discourselens.model_discovery.discover_provider_models", fake_discovery
    )
    response = web_client.post(
        "/api/provider-models/discover",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"provider_id": "openai", "credential_alias": "personal"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "items": [{"model": "gpt-test", "display_name": "Test Chat"}]
    }
    assert len(calls) == 1
    assert calls == [("openai", "personal")]


def test_web_lists_declared_reasoning_options_for_chat(web_client, monkeypatch):
    def fake_preview(_project, name, prompt):
        assert name == "linguist-chat"
        assert prompt
        return SimpleNamespace(
            effective_capabilities={
                "reasoning_efforts": ("low", "high"),
                "reasoning_modes": ("adaptive",),
            },
            reasoning="low",
            reasoning_budget_tokens=None,
            allowed=True,
            block_reason=None,
        )

    monkeypatch.setattr(Project, "preview_model_preset_request", fake_preview)
    response = web_client.post(
        "/api/model-presets/capabilities",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"name": "linguist-chat"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "reasoning_efforts": ["low", "high"],
        "reasoning_modes": ["adaptive"],
        "current_reasoning": "low",
        "reasoning_budget_tokens": None,
        "allowed": True,
        "message": None,
    }


def test_web_saves_explicit_model_reasoning_capabilities(tmp_path: Path):
    project_path = tmp_path / "project"
    Project.init(project_path).close()
    with TestClient(create_app(project_path, TOKEN)) as client:
        response = client.post(
            "/api/model-capabilities",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "provider_id": "openai",
                "model": "gpt-example-reasoning",
                "reasoning_efforts": ["low", "high"],
                "reasoning_modes": [],
            },
        )
    assert response.status_code == 200
    with Project.open(project_path, read_only=True) as project:
        profile = project.model_capability_profile("openai", "gpt-example-reasoning")
    assert profile is not None
    assert profile.assurance == "UNASSESSED"
    assert profile.capabilities.reasoning_efforts == ("low", "high")


def test_web_rejects_reasoning_controls_without_provider_mapping(web_client):
    response = web_client.post(
        "/api/model-capabilities",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={
            "provider_id": "deepseek",
            "model": "deepseek-reasoner",
            "reasoning_efforts": ["high"],
            "reasoning_modes": [],
        },
    )
    assert response.status_code == 422


def test_web_model_discovery_requires_auth_and_valid_payload(web_client):
    assert web_client.post(
        "/api/provider-models/discover",
        json={"provider_id": "openai", "credential_alias": "personal"},
    ).status_code == 401
    response = web_client.post(
        "/api/provider-models/discover",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"provider_id": "openai", "credential_alias": " "},
    )
    assert response.status_code == 422


def test_web_rejects_invalid_model_reasoning_configuration(web_client: TestClient):
    response = web_client.post(
        "/api/model-presets",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={
            "name": "bad",
            "provider_id": "openai",
            "credential_alias": "missing",
            "model": "gpt-example",
            "reasoning_mode": "manual",
        },
    )
    assert response.status_code == 422
    assert "budget" in response.json()["detail"]



def test_web_chat_requires_egress_ack_and_persists_evidence_turns(tmp_path: Path, monkeypatch):
    from discourselens.models import CredentialReference, ModelPreset

    project_root = tmp_path / "project"
    project = Project.init(project_root)
    source = tmp_path / "chat.txt"
    source.write_text("A minister described the reform as necessary.\n", encoding="utf-8")
    revision_id = project.import_document(source).revision_id
    project.save_credential_reference(CredentialReference("openai", "fixture", "DL_CHAT_KEY"))
    project.save_model_preset(ModelPreset("chat-model", "openai", "fixture", "gpt-example"))
    project.close()
    calls = []

    def fake_agent(
        _project,
        project_root,
        preset_name,
        question,
        revision_ids,
        context,
        embedding_egress_acknowledged,
        reasoning_override=None,
        reasoning_mode_override=None,
    ):
        from discourselens.web import _record_chat_activity

        _record_chat_activity("Consultando la herramienta de búsqueda de prueba.")
        assert embedding_egress_acknowledged is False
        calls.append(
            {
                "root": project_root,
                "preset": preset_name,
                "question": question,
                "revision_ids": revision_ids,
                "context": context,
                "reasoning_override": reasoning_override,
                "reasoning_mode_override": reasoning_mode_override,
            }
        )
        return {
            "answer_interpretation": f"Grounded reply to: {question}",
            "alternative_interpretation": None,
            "evidence": [{"citation": "[speech.pdf · p. 2]", "excerpt": "necessary reform"}],
            "selected_framework_id": "fairclough-text",
            "tool_results": [
                {
                    "tool_name": "ask_question",
                    "arguments": {"revision_ids": revision_ids},
                    "result": '{"citation":"[speech.pdf · p. 2]","excerpt":"necessary reform"}',
                }
            ],
        }

    monkeypatch.setattr("discourselens.web._run_chat_agent", fake_agent)
    with TestClient(create_app(project_root, TOKEN)) as client:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        created = client.post(
            "/api/conversations",
            headers=headers,
            json={
                "revision_ids": [revision_id],
                "framework_id": "auto",
                "preset_name": "chat-model",
                "retrieval_mode": "lexical",
            },
        )
        assert created.status_code == 201
        conversation_id = created.json()["id"]
        denied = client.post(
            f"/api/conversations/{conversation_id}/messages",
            headers=headers,
            json={"question": "Why necessary?", "model_egress_acknowledged": False},
        )
        assert denied.status_code == 403
        assert calls == []

        for index, question in enumerate(("Why is it described as necessary?", "What is another reading?")):
            response = client.post(
                f"/api/conversations/{conversation_id}/messages",
                headers=headers,
                json={
                    "question": question,
                    "model_egress_acknowledged": True,
                    **({"reasoning": "high"} if index == 0 else {}),
                },
            )
            assert response.status_code == 201
        assert calls[0]["context"] == []
        assert [message["role"] for message in calls[1]["context"]] == [
            "user",
            "assistant",
        ]
        assert "[speech.pdf · p. 2]" in calls[1]["context"][1]["content"]
        assert "necessary reform" in calls[1]["context"][1]["content"]
        assert "ask_question" in calls[1]["context"][1]["content"]
        assert calls[0]["root"] == project_root
        assert calls[0]["preset"] == "chat-model"
        assert calls[0]["reasoning_override"] == "high"
        assert calls[0]["reasoning_mode_override"] is None
        assert calls[0]["revision_ids"] == [revision_id]
        saved = client.get(
            f"/api/conversations/{conversation_id}/messages", headers=headers
        ).json()
        assert len(saved["items"]) == 4
        assert saved["items"][1]["content"]["answer_interpretation"].startswith(
            "Grounded reply"
        )
        assert saved["items"][1]["content"]["activity"]["provider_id"] == "openai"
        assert saved["items"][1]["content"]["activity"]["model"] == "gpt-example"
        activity = client.get(
            f"/api/conversations/{conversation_id}/activity", headers=headers
        )
        assert activity.status_code == 200
        assert activity.json()["state"] == "complete"
        assert activity.json()["metadata"]["retrieval_mode"] == "lexical"
        assert any(
            "herramienta de búsqueda de prueba" in event["text"]
            for event in activity.json()["events"]
        )


def test_web_chat_accepts_pasted_text_without_documents(tmp_path: Path, monkeypatch):
    from discourselens.models import CredentialReference, ModelPreset

    project_root = tmp_path / "project"
    project = Project.init(project_root)
    project.save_credential_reference(CredentialReference("openai", "fixture", "DL_CHAT_KEY"))
    project.save_model_preset(ModelPreset("chat-model", "openai", "fixture", "gpt-example"))
    project.close()
    prompts = []

    def fake_generate(_project, preset_name, prompt, *, system_prompt=None):
        assert preset_name == "chat-model"
        assert "No corpus revisions are attached" in prompt
        prompts.append(prompt)
        return SimpleNamespace(
            response=SimpleNamespace(
                text='{"final_answer":"El fragmento presenta una evaluación."}'
            )
        )

    monkeypatch.setattr(Project, "generate_with_preset", fake_generate)
    with TestClient(create_app(project_root, TOKEN)) as client:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        created = client.post(
            "/api/conversations",
            headers=headers,
            json={
                "revision_ids": [],
                "framework_id": "auto",
                "preset_name": "chat-model",
                "retrieval_mode": "lexical",
            },
        )
        assert created.status_code == 201
        conversation_id = created.json()["id"]
        sent = client.post(
            f"/api/conversations/{conversation_id}/messages",
            headers=headers,
            json={
                "question": "Analiza este texto: ‘La reforma es necesaria.’",
                "model_egress_acknowledged": True,
            },
        )

    assert sent.status_code == 201
    assert "La reforma es necesaria" in prompts[0]
    answer = sent.json()["result"]
    assert answer["answer_interpretation"] == "El fragmento presenta una evaluación."
    assert answer["evidence"] == []
    assert answer["tool_results"] == []


def test_web_chat_agent_plans_and_executes_registered_mcp_tools(tmp_path: Path, monkeypatch):
    import json
    from types import SimpleNamespace

    from discourselens.web import _run_chat_agent

    project_root = tmp_path / "agent-project"
    project = Project.init(project_root)
    source = tmp_path / "agent-visible.txt"
    source.write_text("A local source for the agent tool catalog.\n", encoding="utf-8")
    project.import_document(source)
    project.close()
    project = Project.open(project_root)
    plans = iter(
        [
            {"tool_name": "list_documents", "arguments": {"limit": 100}},
            {"final_answer": "El proyecto contiene la fuente local agent-visible.txt."},
        ]
    )
    prompts = []

    def fake_generation(_project, _preset_name, prompt, *, system_prompt=None):
        prompts.append(prompt)
        return SimpleNamespace(
            response=SimpleNamespace(text=json.dumps(next(plans), ensure_ascii=False))
        )

    monkeypatch.setattr(Project, "generate_with_preset", fake_generation)
    try:
        result = _run_chat_agent(
            project,
            project_root,
            "frontier-chat",
            "¿Qué documentos hay en mi proyecto?",
            [],
            [],
            True,
        )
    finally:
        project.close()

    assert cast(str, result["answer_interpretation"]).startswith("El proyecto contiene")
    assert result["tool_trace"] == [{"tool_name": "list_documents"}]
    assert "agent-visible.txt" in cast(
        str, cast(list[dict[str, object]], result["tool_results"])[0]["result"]
    )
    assert "corpus_frequency_for_scope" in prompts[0]
    assert "fairclough-text" in prompts[0]
    assert "list_documents" in prompts[0]


def test_web_import_lists_and_searches_within_selected_revision(web_client: TestClient):
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/octet-stream",
        "X-Document-Filename": "public-policy.txt",
    }
    imported = web_client.post(
        "/api/documents",
        headers=headers,
        content=b"Public policy needs evidence.\nPublic trust needs care.",
    )
    assert imported.status_code == 201
    payload = imported.json()
    assert payload["display_name"] == "public-policy.txt"

    listed = web_client.get(
        "/api/documents", headers={"Authorization": f"Bearer {TOKEN}"}
    ).json()
    assert listed["total"] == 1
    assert listed["items"][0]["latest_revision_id"] == payload["revision_id"]

    result = web_client.post(
        "/api/search",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"revision_id": payload["revision_id"], "phrase": "public", "limit": 10},
    )
    assert result.status_code == 200
    assert result.json()["total_count"] == 2
    assert all("Public" in row["text"] for row in result.json()["rows"])
    assert all("public-policy.txt" in row["citation"] for row in result.json()["rows"])

    frequency = web_client.post(
        "/api/corpus/frequency",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"revision_id": payload["revision_id"], "limit": 10},
    )
    assert frequency.status_code == 200
    assert frequency.json()["total_tokens"] == 8
    assert any(
        row["token"].casefold() == "public" and row["count"] == 2
        for row in frequency.json()["rows"]
    )

    concordance = web_client.post(
        "/api/corpus/kwic",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"revision_id": payload["revision_id"], "query": "Public", "window": 3},
    )
    assert concordance.status_code == 200
    assert concordance.json()["total_count"] == 2
    assert all("public-policy.txt" in row["citation"] for row in concordance.json()["rows"])

    ngrams = web_client.post(
        "/api/corpus/ngrams",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"revision_id": payload["revision_id"], "size": 2, "limit": 20},
    )
    assert ngrams.status_code == 200
    assert any(row["ngram"] == "public policy" for row in ngrams.json()["rows"])

    collocations = web_client.post(
        "/api/corpus/collocations",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"revision_id": payload["revision_id"], "query": "public", "window": 2},
    )
    assert collocations.status_code == 200
    assert any(row["term"] == "policy" and row["count"] == 1 for row in collocations.json()["rows"])


def test_web_rejects_uploads_over_configured_limit(tmp_path: Path):
    project_path = tmp_path / "project"
    Project.init(project_path).close()
    with TestClient(create_app(project_path, TOKEN, max_upload_bytes=8)) as client:
        response = client.post(
            "/api/documents",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/octet-stream",
                "X-Document-Filename": "too-large.txt",
            },
            content=b"nine bytes",
        )
    assert response.status_code == 413


def test_web_pdf_ocr_options_are_forwarded_to_document_importer(tmp_path: Path, monkeypatch):
    project_path = tmp_path / "project"
    Project.init(project_path).close()
    captured = {}

    def capture_import(project_root, source, *, ocr_language, ocr_dpi, auto_ocr):
        captured.update(
            project_root=project_root,
            source=source,
            ocr_language=ocr_language,
            ocr_dpi=ocr_dpi,
            auto_ocr=auto_ocr,
        )
        return SimpleNamespace(
            document_id="document-1", revision_id="revision-1", status="SUCCESS_WITH_WARNINGS",
            warnings=(),
        )

    monkeypatch.setattr("discourselens.web._import_document", capture_import)
    with TestClient(create_app(project_path, TOKEN)) as client:
        response = client.post(
            "/api/documents",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "X-Document-Filename": "scanned-paper.pdf",
                "X-OCR-Language": "eng+nld",
                "X-OCR-DPI": "240",
            },
            content=b"fixture PDF bytes",
        )

    assert response.status_code == 201
    assert response.json()["status"] == "SUCCESS_WITH_WARNINGS"
    assert captured["ocr_language"] == "eng+nld"
    assert captured["ocr_dpi"] == 240
    assert captured["auto_ocr"] is False


def test_web_enables_automatic_ocr_for_pdf_uploads(tmp_path: Path, monkeypatch):
    project_path = tmp_path / "project"
    Project.init(project_path).close()
    captured = []

    def capture_import(project_root, source, *, ocr_language, ocr_dpi, auto_ocr):
        captured.append(
            {"ocr_language": ocr_language, "ocr_dpi": ocr_dpi, "auto_ocr": auto_ocr}
        )
        return SimpleNamespace(
            document_id="document-1", revision_id="revision-1", status="SUCCESS", warnings=()
        )

    monkeypatch.setattr("discourselens.web._import_document", capture_import)
    with TestClient(create_app(project_path, TOKEN)) as client:
        pdf = client.post(
            "/api/documents",
            headers={"Authorization": f"Bearer {TOKEN}", "X-Document-Filename": "paper.pdf"},
            content=b"fixture PDF bytes",
        )
        text = client.post(
            "/api/documents",
            headers={"Authorization": f"Bearer {TOKEN}", "X-Document-Filename": "notes.txt"},
            content=b"plain text",
        )
    assert pdf.status_code == text.status_code == 201
    assert captured == [
        {"ocr_language": None, "ocr_dpi": 300, "auto_ocr": True},
        {"ocr_language": None, "ocr_dpi": 300, "auto_ocr": False},
    ]


@pytest.mark.parametrize(
    ("filename", "ocr_headers", "message"),
    [
        ("notes.txt", {"X-OCR-Language": "eng"}, "OCR solo está disponible para PDF"),
        ("scan.pdf", {"X-OCR-DPI": "240"}, "Selecciona un idioma para usar el OCR"),
    ],
)
def test_web_rejects_invalid_ocr_upload_options(
    tmp_path: Path, filename: str, ocr_headers: dict[str, str], message: str
):
    project_path = tmp_path / "project"
    Project.init(project_path).close()
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "X-Document-Filename": filename,
        **ocr_headers,
    }
    with TestClient(create_app(project_path, TOKEN)) as client:
        response = client.post("/api/documents", headers=headers, content=b"source")
    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_web_compares_explicit_non_overlapping_document_groups(web_client: TestClient):
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/octet-stream",
    }
    left = web_client.post(
        "/api/documents",
        headers={**headers, "X-Document-Filename": "group-a.txt"},
        content=b"evidence evidence public trust",
    )
    right = web_client.post(
        "/api/documents",
        headers={**headers, "X-Document-Filename": "group-b.txt"},
        content=b"public debate public voice",
    )
    assert left.status_code == 201
    assert right.status_code == 201
    left_revision = left.json()["revision_id"]
    right_revision = right.json()["revision_id"]

    comparison = web_client.post(
        "/api/corpus/compare",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={
            "left_revision_ids": [left_revision],
            "right_revision_ids": [right_revision],
            "metric": "frequency",
            "limit": 20,
        },
    )
    assert comparison.status_code == 200
    payload = comparison.json()
    assert payload["metric"] == "frequency"
    assert payload["coverage"] == "EXHAUSTIVE"
    assert payload["left_revision_ids"] == [left_revision]
    assert payload["right_revision_ids"] == [right_revision]
    evidence_row = next(row for row in payload["rows"] if row["token"] == "evidence")
    assert evidence_row["left_count"] == 2
    assert evidence_row["right_count"] == 0

    keyness = web_client.post(
        "/api/corpus/compare",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={
            "left_revision_ids": [left_revision],
            "right_revision_ids": [right_revision],
            "metric": "keyness",
        },
    )
    assert keyness.status_code == 200
    assert keyness.json()["metric"] == "keyness"

    overlap = web_client.post(
        "/api/corpus/compare",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={
            "left_revision_ids": [left_revision],
            "right_revision_ids": [left_revision],
        },
    )
    assert overlap.status_code == 422


def test_web_coding_and_van_dijk_group_comparison_are_evidence_linked(
    web_client: TestClient,
):
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/octet-stream",
        "X-Document-Filename": "group-accounts.txt",
    }
    source = "Residents said the policy brought hope.\nOfficials said the policy brought risk."
    imported = web_client.post("/api/documents", headers=headers, content=source.encode())
    assert imported.status_code == 201
    revision_id = imported.json()["revision_id"]

    paragraphs = web_client.get(
        f"/api/documents/{revision_id}/paragraphs",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert paragraphs.status_code == 200
    assert paragraphs.json()["total"] == 2
    assert [row["ordinal"] for row in paragraphs.json()["items"]] == [0, 1]

    codebook = web_client.post(
        "/api/coding/codebooks",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"name": "Van Dijk evaluation", "description": "Human-coded examples"},
    )
    assert codebook.status_code == 201
    codebook_id = codebook.json()["codebook_id"]
    positive_code = web_client.post(
        f"/api/coding/codebooks/{codebook_id}/codes",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"label": "Positive consequence", "definition": "Evaluative positive wording"},
    )
    negative_code = web_client.post(
        f"/api/coding/codebooks/{codebook_id}/codes",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"label": "Negative consequence", "definition": "Evaluative negative wording"},
    )
    assert positive_code.status_code == negative_code.status_code == 201

    selected_texts = ("hope", "risk")
    created_annotations = []
    for phrase, code_id in zip(
        selected_texts,
        (positive_code.json()["code_id"], negative_code.json()["code_id"]),
        strict=True,
    ):
        start = source.index(phrase)
        annotation = web_client.post(
            "/api/coding/annotations",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "revision_id": revision_id,
                "code_id": code_id,
                "start_offset": start,
                "end_offset": start + len(phrase),
            },
        )
        assert annotation.status_code == 201
        assert annotation.json()["text"] == phrase
        assert "group-accounts.txt" in annotation.json()["citation"]
        created_annotations.append(annotation.json()["annotation_id"])

    listed = web_client.get(
        f"/api/coding/annotations?revision_id={revision_id}",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert listed.status_code == 200
    assert [row["text"] for row in listed.json()["items"]] == ["hope", "risk"]

    comparison = web_client.post(
        "/api/analysis/van-dijk/group-comparison",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={
            "revision_ids": [revision_id],
            "groups": [
                {"label": "Residents", "annotation_ids": [created_annotations[0]]},
                {"label": "Officials", "annotation_ids": [created_annotations[1]]},
            ],
            "shared_topic_event": "the public policy statement",
            "comparability_basis": "both passages report evaluations of the same policy",
            "codebook_id": codebook_id,
        },
    )
    assert comparison.status_code == 200, comparison.text
    result = comparison.json()
    assert result["claim_semantics"] == "EXAMPLE_BASED"
    assert result["selected_annotation_count"] == 2
    assert [group["label"] for group in result["groups"]] == ["Residents", "Officials"]
    assert [
        annotation["text"]
        for group in result["groups"]
        for annotation in group["annotations"]
    ] == ["hope", "risk"]
    assert web_client.get("/").status_code == 200
    assert "Codificación cualitativa" in web_client.get("/").text


def test_web_semantic_retrieval_requires_explicit_provider_egress_ack(
    web_client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/octet-stream",
        "X-Document-Filename": "retrieval.txt",
    }
    imported = web_client.post(
        "/api/documents", headers=headers, content=b"Evidence-first public policy passage."
    )
    assert imported.status_code == 201
    revision_id = imported.json()["revision_id"]
    preview = SimpleNamespace(
        provider_id="openai",
        credential_alias="university",
        model="test-embedding",
        endpoint="https://api.example.test/v1",
        revision_id=revision_id,
        input_type="document",
        total_paragraphs=1,
        offline_mode=False,
        blocked_by_offline_mode=False,
        network_request_made=False,
    )
    monkeypatch.setattr(Project, "preview_paragraph_embedding_request", lambda *a, **k: preview)
    build_calls: list[tuple[object, ...]] = []

    def build_index(self, preset_name, selected_revision, *, embedding_provider=None):
        build_calls.append((preset_name, selected_revision))
        return "vector-artifact-1"

    monkeypatch.setattr(Project, "build_paragraph_embedding_index", build_index)
    preview_response = web_client.post(
        "/api/retrieval/preview",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"revision_id": revision_id, "preset_name": "embed"},
    )
    assert preview_response.status_code == 200
    assert preview_response.json()["endpoint"] == "https://api.example.test/v1"
    assert preview_response.json()["network_request_made"] is False

    denied = web_client.post(
        "/api/retrieval/index",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"revision_id": revision_id, "preset_name": "embed"},
    )
    assert denied.status_code == 422
    assert not build_calls

    built = web_client.post(
        "/api/retrieval/index",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={
            "revision_id": revision_id,
            "preset_name": "embed",
            "provider_egress_acknowledged": True,
        },
    )
    assert built.status_code == 200
    assert built.json()["artifact_id"] == "vector-artifact-1"
    assert build_calls == [("embed", revision_id)]

    search_calls: list[str] = []

    def semantic_search(self, preset_name, scope_id, artifact_ids, query_text, *, limit=10):
        search_calls.append(query_text)
        return RetrievalSearchResult(
            scope_id,
            "retrieval-run-1",
            [VectorHit("vector-artifact-1", revision_id, "paragraph-1", 0, 14, 0.1)],
        )

    monkeypatch.setattr(Project, "semantic_search_with_preset", semantic_search)
    search_payload = {
        "revision_id": revision_id,
        "preset_name": "embed",
        "artifact_id": "vector-artifact-1",
        "query": "public policy",
    }
    search_denied = web_client.post(
        "/api/retrieval/search",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=search_payload,
    )
    assert search_denied.status_code == 422
    assert not search_calls

    searched = web_client.post(
        "/api/retrieval/search",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={**search_payload, "query_egress_acknowledged": True},
    )
    assert searched.status_code == 200
    assert searched.json()["rows"][0]["text"] == "Evidence-first"
    assert "retrieval.txt" in searched.json()["rows"][0]["citation"]
    assert searched.json()["coverage"] == "TOP_K_RETRIEVAL_NOT_EXHAUSTIVE"
    assert search_calls == ["public policy"]
