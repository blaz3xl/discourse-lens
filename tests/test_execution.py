import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from discourselens import Project
from discourselens.errors import ModelProviderError, ProjectError
from discourselens.execution import (
    build_embedding_gateway,
    build_reranking_gateway,
    execute_model_preset,
    preview_model_preset_request,
    snapshot_model_preset,
)
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
    RerankResult,
)
from discourselens.secrets import ReferenceSecretStore


class _FixtureProvider:
    provider_id = "openai-compatible"

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(generation_parameters=("max_tokens",))

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse("fixture response", self.provider_id, request.model)


def test_model_preset_snapshot_pins_alias_and_explicit_endpoint_to_local_http(
    tmp_path, monkeypatch
):
    alias_a_secret = "alias-a-test-secret"
    alias_b_secret = "alias-b-test-secret"
    monkeypatch.setenv("DL_ALIAS_A", alias_a_secret)
    monkeypatch.setenv("DL_ALIAS_B", alias_b_secret)
    for variable in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    received: list[tuple[str, str | None]] = []

    class _CaptureHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            received.append((self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "choices": [
                            {"message": {"content": "local fixture"}, "finish_reason": "stop"}
                        ]
                    }
                ).encode()
            )

        def log_message(self, format: str, *args: object) -> None:
            return

    http_server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server_thread = Thread(target=http_server.serve_forever, daemon=True)
    server_thread.start()
    port = http_server.server_address[1]
    endpoint_a = f"http://127.0.0.1:{port}/alias-a/v1"
    endpoint_b = f"http://127.0.0.1:{port}/alias-b/v1"
    project = Project.init(tmp_path / "pinned-alias")
    resolved_references: list[str] = []
    original_get = ReferenceSecretStore.get

    def record_get(store: ReferenceSecretStore, reference: str) -> str:
        resolved_references.append(reference)
        return original_get(store, reference)

    monkeypatch.setattr(ReferenceSecretStore, "get", record_get)
    try:
        project.save_credential_reference(
            CredentialReference("openai-compatible", "alias-a", "DL_ALIAS_A", endpoint_a)
        )
        project.save_credential_reference(
            CredentialReference("openai-compatible", "alias-b", "DL_ALIAS_B", endpoint_b)
        )
        project.save_model_preset(
            ModelPreset("review", "openai-compatible", "alias-a", "fixture-model")
        )
        snapshot = snapshot_model_preset(project, "review")

        # Current mutable routing changes after capture; execution must still use A.
        project.save_model_preset(
            ModelPreset("review", "openai-compatible", "alias-b", "fixture-model")
        )
        result = execute_model_preset(
            project, "review", "fixture prompt", preset_snapshot=snapshot
        )

        assert snapshot["credential_alias"] == "alias-a"
        assert snapshot["endpoint"] == endpoint_a
        assert resolved_references == ["DL_ALIAS_A"]
        assert received == [
            ("/alias-a/v1/chat/completions", f"Bearer {alias_a_secret}")
        ]
        ledger = project.db.execute("SELECT endpoint FROM model_executions").fetchone()
        assert ledger["endpoint"] == endpoint_a
        assert alias_b_secret not in str(received)
        assert result.response.text == "local fixture"
    finally:
        project.close()
        http_server.shutdown()
        server_thread.join(timeout=5)
        http_server.server_close()


def test_anthropic_compatible_preset_executes_messages_and_keeps_secret_out_of_ledger(
    tmp_path, monkeypatch
):
    secret_value = "anthropic-compatible-secret-must-not-persist"
    secret_reference = "DL_ANTHROPIC_COMPATIBLE_KEY"
    endpoint = "https://gateway.example/v1"
    monkeypatch.setenv(secret_reference, secret_value)
    project = Project.init(tmp_path / "anthropic-compatible")
    received: list[tuple[str, dict[str, str], dict[str, object]]] = []

    class _JSONResponse:
        def __init__(self) -> None:
            from email.message import Message

            self.headers = Message()
            self.body = json.dumps(
                {
                    "content": [{"type": "text", "text": "Messages fixture response"}],
                    "usage": {"input_tokens": 8, "output_tokens": 3},
                    "stop_reason": "end_turn",
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, size: int = -1) -> bytes:
            return self.body if size < 0 else self.body[:size]

    def open_request(request, *, timeout: float):
        assert timeout > 0
        received.append(
            (
                request.full_url,
                dict(request.header_items()),
                json.loads(request.data.decode("utf-8")),
            )
        )
        return _JSONResponse()

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    try:
        project.save_credential_reference(
            CredentialReference("anthropic-compatible", "university", secret_reference, endpoint)
        )
        project.save_model_preset(
            ModelPreset("analysis", "anthropic-compatible", "university", "fixture-model")
        )

        result = execute_model_preset(project, "analysis", "A local test prompt")

        assert result.response.text == "Messages fixture response"
        assert result.response.provider == "anthropic-compatible"
        assert result.response.model == "fixture-model"
        assert len(received) == 1
        request_url, headers, body = received[0]
        assert request_url == "https://gateway.example/v1/messages"
        normalized_headers = {name.casefold(): value for name, value in headers.items()}
        assert normalized_headers["x-api-key"] == secret_value
        assert normalized_headers["anthropic-version"] == "2023-06-01"
        assert body["model"] == "fixture-model"
        assert body["messages"] == [{"role": "user", "content": "A local test prompt"}]

        execution = project.db.execute("SELECT * FROM model_executions").fetchone()
        assert execution is not None
        assert execution["provider_id"] == "anthropic-compatible"
        assert execution["credential_alias"] == "university"
        assert execution["endpoint"] == endpoint
        assert execution["error_type"] is None
        assert secret_value not in json.dumps(dict(execution))
        credential = project.db.execute("SELECT * FROM provider_credentials").fetchone()
        assert credential is not None
        assert credential["secret_reference"] == secret_reference
        assert secret_value not in json.dumps(dict(credential))
    finally:
        project.close()


def test_missing_selected_secret_does_not_fall_back_to_other_alias_or_ambient_key(
    tmp_path, monkeypatch
):
    selected_reference = "DL_SELECTED_KEY"
    other_secret = "other-alias-secret-must-not-be-used"
    ambient_secret = "ambient-openai-secret-must-not-be-used"
    monkeypatch.delenv(selected_reference, raising=False)
    monkeypatch.setenv("DL_OTHER_KEY", other_secret)
    monkeypatch.setenv("OPENAI_API_KEY", ambient_secret)
    for variable in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    received_authorization: list[str | None] = []

    class _CaptureHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            received_authorization.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "choices": [
                    {"message": {"content": "unexpected fallback"}, "finish_reason": "stop"}
                ]
            }
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, format: str, *args: object) -> None:
            return

    http_server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server_thread = Thread(target=http_server.serve_forever, daemon=True)
    server_thread.start()
    endpoint = f"http://127.0.0.1:{http_server.server_port}/v1"
    project = Project.init(tmp_path / "missing-selected-secret")
    try:
        project.save_credential_reference(
            CredentialReference("openai-compatible", "selected", selected_reference, endpoint)
        )
        project.save_credential_reference(
            CredentialReference("openai-compatible", "other", "DL_OTHER_KEY", endpoint)
        )
        project.save_model_preset(
            ModelPreset("review", "openai-compatible", "selected", "fixture-model")
        )

        with pytest.raises(ProjectError) as captured:
            execute_model_preset(project, "review", "fixture prompt")

        assert str(captured.value) == "provider credential could not be resolved"
        assert selected_reference not in str(captured.value)
        assert other_secret not in str(captured.value)
        assert ambient_secret not in str(captured.value)
        assert received_authorization == []
        execution = project.db.execute("SELECT * FROM model_executions").fetchone()
        assert execution["credential_alias"] == "selected"
        assert execution["endpoint"] == endpoint
        assert execution["error_type"] == "ModelProviderError"
    finally:
        project.close()
        http_server.shutdown()
        server_thread.join(timeout=5)
        http_server.server_close()


@pytest.mark.parametrize("preview", [False, True])
def test_whitespace_prompt_is_rejected_before_provider_creation(tmp_path, preview):
    project = Project.init(tmp_path / "project")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "remote", "DL_REMOTE", "https://api.example.test/v1"
        )
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "remote", "model-a"))
    provider_calls = []

    def provider_factory(endpoint, reference, provider_id):
        provider_calls.append((endpoint, reference, provider_id))
        raise AssertionError("invalid requests must be rejected before provider creation")

    try:
        with pytest.raises(ModelProviderError, match="generation prompt must be non-empty text"):
            if preview:
                preview_model_preset_request(
                    project, "review", " \t ", provider_factory=provider_factory
                )
            else:
                execute_model_preset(
                    project, "review", " \t ", provider_factory=provider_factory
                )

        assert provider_calls == []
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0
    finally:
        project.close()


@pytest.mark.parametrize("preview", [False, True])
def test_boolean_max_tokens_is_rejected_before_provider_creation(tmp_path, preview):
    project = Project.init(tmp_path / "project")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "remote", "DL_REMOTE", "https://api.example.test/v1"
        )
    )
    project.save_model_preset(
        ModelPreset(
            "review",
            "openai-compatible",
            "remote",
            "model-a",
            parameters={"max_tokens": True},
        )
    )
    provider_calls = []

    def provider_factory(endpoint, reference, provider_id):
        provider_calls.append((endpoint, reference, provider_id))
        return _FixtureProvider()

    try:
        with pytest.raises(ModelProviderError, match="max_tokens.*boolean"):
            if preview:
                preview_model_preset_request(
                    project, "review", "prompt", provider_factory=provider_factory
                )
            else:
                execute_model_preset(
                    project, "review", "prompt", provider_factory=provider_factory
                )

        assert provider_calls == []
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0
    finally:
        project.close()


def test_boolean_reasoning_budget_snapshot_is_rejected_before_provider_creation(tmp_path):
    project = Project.init(tmp_path / "project")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "remote", "DL_REMOTE", "https://api.example.test/v1"
        )
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "remote", "model-a"))
    snapshot = snapshot_model_preset(project, "review")
    snapshot["reasoning_budget_tokens"] = True
    provider_calls = []

    def provider_factory(endpoint, reference, provider_id):
        provider_calls.append((endpoint, reference, provider_id))
        return _FixtureProvider()

    try:
        with pytest.raises(ModelProviderError, match="reasoning budget snapshot is invalid"):
            execute_model_preset(
                project,
                "review",
                "prompt",
                provider_factory=provider_factory,
                preset_snapshot=snapshot,
            )

        assert provider_calls == []
    finally:
        project.close()


@pytest.mark.parametrize("preview", [False, True])
def test_cataloged_non_generation_provider_is_rejected_before_provider_creation(
    tmp_path, preview
):
    project = Project.init(tmp_path / "project")
    project.save_credential_reference(
        CredentialReference(
            "cohere-reranker", "fixture", "DL_RERANKER", "https://api.example.test/v2"
        )
    )
    project.save_model_preset(
        ModelPreset("review", "cohere-reranker", "fixture", "rerank-model")
    )
    provider_calls = []

    class _CatalogedNonGenerationProvider(_FixtureProvider):
        provider_id = "cohere-reranker"

    def provider_factory(endpoint, reference, provider_id):
        provider_calls.append((endpoint, reference, provider_id))
        return _CatalogedNonGenerationProvider()

    try:
        if preview:
            request_preview = preview_model_preset_request(
                project, "review", "prompt", provider_factory=provider_factory
            )
            assert request_preview.allowed is False
            assert request_preview.block_reason == "provider does not support text generation"
        else:
            with pytest.raises(ModelProviderError, match="does not support text generation"):
                execute_model_preset(
                    project, "review", "prompt", provider_factory=provider_factory
                )

        assert provider_calls == []
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0
    finally:
        project.close()


@pytest.mark.parametrize(
    ("provider_id", "endpoint"),
    [
        ("jina-reranker", None),
        ("jina-reranker", "https://rerank.example.test/v1"),
        ("cohere-reranker", None),
        ("cohere-reranker", "https://rerank.example.test/v2"),
    ],
)
def test_offline_reranking_gateway_blocks_remote_before_adapter_construction(
    tmp_path, monkeypatch, provider_id, endpoint
):
    project = Project.init(tmp_path / "project")
    project.save_credential_reference(
        CredentialReference(provider_id, "selected", "DL_SELECTED", endpoint)
    )
    project.set_offline_mode(True)

    def adapter_must_not_be_constructed(*args, **kwargs):
        pytest.fail("offline policy must run before constructing a remote reranker")

    monkeypatch.setattr(
        "discourselens.execution.JinaRerankingProvider", adapter_must_not_be_constructed
    )
    monkeypatch.setattr(
        "discourselens.execution.CohereRerankingProvider", adapter_must_not_be_constructed
    )
    try:
        with pytest.raises(ProjectError, match="offline mode blocks remote reranking"):
            build_reranking_gateway(project, provider_id, "selected")
    finally:
        project.close()


@pytest.mark.parametrize("provider_id", ["jina-reranker", "cohere-reranker"])
def test_offline_reranking_gateway_allows_explicit_loopback_endpoint(
    tmp_path, monkeypatch, provider_id
):
    for name in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    endpoint = "http://127.0.0.1:11434/rerank"
    project = Project.init(tmp_path / "project")
    project.save_credential_reference(
        CredentialReference(provider_id, "selected", "DL_SELECTED", endpoint)
    )
    project.set_offline_mode(True)
    try:
        gateway = build_reranking_gateway(project, provider_id, "selected")
        assert vars(gateway.rerankers[provider_id]).get("endpoint") == endpoint
    finally:
        project.close()


def test_offline_generation_blocks_before_provider_creation_or_secret_resolution(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "offline-generation")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "selected", "DL_SELECTED", "https://provider.example/v1")
    )
    project.save_model_preset(
        ModelPreset("review", "openai-compatible", "selected", "fixture-model")
    )
    project.set_offline_mode(True)

    def unexpected_secret_read(*_args, **_kwargs):
        pytest.fail("offline generation must block before resolving a secret")

    def unexpected_provider_creation(*_args, **_kwargs):
        pytest.fail("offline generation must block before creating a provider")

    monkeypatch.setattr(ReferenceSecretStore, "get", unexpected_secret_read)
    try:
        with pytest.raises(ProjectError, match="offline mode blocks remote model execution"):
            execute_model_preset(
                project, "review", "prompt", provider_factory=unexpected_provider_creation
            )
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0
    finally:
        project.close()


def test_generation_rechecks_offline_mode_before_adapter_secret_resolution(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "offline-generation-transition")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "selected", "DL_SELECTED", "https://provider.example/v1")
    )
    project.save_model_preset(
        ModelPreset("review", "openai-compatible", "selected", "fixture-model")
    )
    secret_reads: list[str] = []
    generate_calls: list[bool] = []

    def record_secret_read(_store: ReferenceSecretStore, reference: str) -> str:
        secret_reads.append(reference)
        return "fixture-secret"

    class _SecretResolvingProvider(_FixtureProvider):
        def generate(self, request: GenerationRequest) -> GenerationResponse:
            generate_calls.append(True)
            ReferenceSecretStore().get("DL_SELECTED")
            return super().generate(request)

    def turn_offline_and_create_provider(endpoint, reference, provider_id):
        assert (endpoint, reference, provider_id) == (
            "https://provider.example/v1",
            "DL_SELECTED",
            "openai-compatible",
        )
        project.set_offline_mode(True)
        return _SecretResolvingProvider()

    monkeypatch.setattr(ReferenceSecretStore, "get", record_secret_read)
    try:
        with pytest.raises(ProjectError, match="offline mode blocks remote model execution"):
            execute_model_preset(
                project,
                "review",
                "prompt",
                provider_factory=turn_offline_and_create_provider,
            )
        assert generate_calls == []
        assert secret_reads == []
    finally:
        project.close()


@pytest.mark.parametrize(
    ("endpoint", "allowed"),
    [
        ("https://embeddings.example/v1", False),
        ("http://127.0.0.1:11434/v1", True),
    ],
)
def test_embedding_gateway_rechecks_current_offline_policy(
    tmp_path, monkeypatch, endpoint, allowed
):
    for name in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    class _FixtureEmbedder:
        provider_id = "openai-compatible"

        def __init__(self, selected_endpoint: str) -> None:
            self.endpoint = selected_endpoint
            self.calls = 0

        def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            self.calls += 1
            return [(0.2, 0.8)]

    project = Project.init(tmp_path / "offline-embedding")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "selected", "DL_SELECTED", endpoint)
    )
    provider = _FixtureEmbedder(endpoint)
    gateway = build_embedding_gateway(
        project, "openai-compatible", "selected", embedding_provider=provider
    )
    project.set_offline_mode(True)
    try:
        if allowed:
            assert gateway.embed("openai-compatible", "fixture-model", ["text"]) == [
                (0.2, 0.8)
            ]
            assert provider.calls == 1
        else:
            with pytest.raises(ProjectError, match="offline mode blocks remote embeddings"):
                gateway.embed("openai-compatible", "fixture-model", ["text"])
            assert provider.calls == 0
    finally:
        project.close()


@pytest.mark.parametrize(
    ("endpoint", "allowed"),
    [
        ("https://rerank.example/v1", False),
        ("http://127.0.0.1:11434/v1", True),
    ],
)
def test_reranking_gateway_rechecks_current_offline_policy(
    tmp_path, monkeypatch, endpoint, allowed
):
    for name in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    rerank_calls: list[bool] = []

    class _FixtureReranker:
        provider_id = "jina-reranker"

        def __init__(self, _reference, _store, *, endpoint: str) -> None:
            self.endpoint = endpoint

        def rerank(self, model, query, documents, *, top_n=None):
            rerank_calls.append(True)
            return [RerankResult(0, 1.0)]

    monkeypatch.setattr(
        "discourselens.execution.JinaRerankingProvider", _FixtureReranker
    )
    project = Project.init(tmp_path / "offline-reranking")
    project.save_credential_reference(
        CredentialReference("jina-reranker", "selected", "DL_SELECTED", endpoint)
    )
    gateway = build_reranking_gateway(project, "jina-reranker", "selected")
    project.set_offline_mode(True)
    try:
        if allowed:
            assert gateway.rerank("jina-reranker", "fixture-model", "query", ["text"]) == [
                RerankResult(0, 1.0)
            ]
            assert rerank_calls == [True]
        else:
            with pytest.raises(ProjectError, match="offline mode blocks remote reranking"):
                gateway.rerank("jina-reranker", "fixture-model", "query", ["text"])
            assert rerank_calls == []
    finally:
        project.close()
