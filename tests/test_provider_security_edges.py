import http.client
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest

from discourselens.errors import ModelProviderError
from discourselens.models import GenerationRequest
from discourselens.providers import (
    AnthropicCompatibleProvider,
    AnthropicProvider,
    CohereEmbeddingProvider,
    CohereProvider,
    CohereRerankingProvider,
    GeminiEmbeddingProvider,
    GeminiProvider,
    JinaEmbeddingProvider,
    JinaRerankingProvider,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleProvider,
    OpenAIEmbeddingProvider,
    OpenAIProvider,
    VoyageEmbeddingProvider,
)


@pytest.fixture
def local_provider_endpoint(monkeypatch):
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
    received_requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received_requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"results":[{"index":0,"relevance_score":1.0}]}')

        def log_message(self, format: str, *args: object) -> None:
            return

        def log_error(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received_requests
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class _NewlineSecret:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self, reference: str) -> str:
        return self.value


class _FailingSecretStore:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def get(self, reference: str) -> str:
        raise RuntimeError(f"credential backend failed while reading {self.secret}")


@dataclass(frozen=True)
class _SecretStoreWithSensitiveRepr:
    value: str

    def get(self, reference: str) -> str:
        return self.value


@pytest.mark.parametrize(
    "provider_factory",
    [
        lambda store: OpenAICompatibleProvider("https://provider.example/v1", "fixture", store),
        lambda store: OpenAIProvider("fixture", store),
        lambda store: OpenAICompatibleEmbeddingProvider(
            "https://provider.example/v1", "fixture", store
        ),
        lambda store: OpenAIEmbeddingProvider("fixture", store),
        lambda store: CohereEmbeddingProvider("fixture", store),
        lambda store: CohereProvider("fixture", store),
        lambda store: VoyageEmbeddingProvider("fixture", store),
        lambda store: JinaEmbeddingProvider("fixture", store),
        lambda store: GeminiEmbeddingProvider("fixture", store),
        lambda store: JinaRerankingProvider("fixture", store),
        lambda store: CohereRerankingProvider("fixture", store),
        lambda store: AnthropicProvider("fixture", store),
        lambda store: AnthropicCompatibleProvider(
            "https://provider.example/v1", "fixture", store
        ),
        lambda store: GeminiProvider("fixture", store),
    ],
    ids=[
        "openai-compatible-generation",
        "openai-generation",
        "openai-compatible-embedding",
        "openai-embedding",
        "cohere-embedding",
        "cohere-generation",
        "voyage-embedding",
        "jina-embedding",
        "gemini-embedding",
        "jina-reranking",
        "cohere-reranking",
        "anthropic-generation",
        "anthropic-compatible-generation",
        "gemini-generation",
    ],
)
def test_provider_repr_never_includes_secret_store_repr(provider_factory) -> None:
    secret = "sk-provider-repr-fixture-secret"
    provider = provider_factory(_SecretStoreWithSensitiveRepr(secret))

    rendered = repr(provider)

    assert secret not in rendered
    assert "secret_store" not in rendered


@pytest.mark.parametrize(
    "provider_factory",
    [
        lambda secret_store: OpenAICompatibleProvider(
            "https://provider.example/v1", "fixture", secret_store
        ),
        lambda secret_store: CohereProvider("fixture", secret_store),
        lambda secret_store: AnthropicProvider("fixture", secret_store),
        lambda secret_store: AnthropicCompatibleProvider(
            "https://provider.example/v1", "fixture", secret_store
        ),
        lambda secret_store: GeminiProvider("fixture", secret_store),
    ],
    ids=["openai-compatible", "cohere", "anthropic", "anthropic-compatible", "gemini"],
)
@pytest.mark.parametrize("method_name", ["generate", "stream_generate"])
def test_generation_secret_store_errors_do_not_expose_credentials(
    provider_factory, method_name: str
) -> None:
    secret = "provider-secret-store-error-sentinel"
    provider = provider_factory(_FailingSecretStore(secret))
    invoke = getattr(provider, method_name)

    with pytest.raises(ModelProviderError) as captured:
        result = invoke(GenerationRequest("private corpus prompt", "fixture-model"))
        if method_name == "stream_generate":
            next(result)

    formatted_error = "".join(
        traceback.format_exception(captured.type, captured.value, captured.value.__traceback__)
    )
    assert str(captured.value) == "provider credential could not be resolved"
    assert secret not in formatted_error
    assert captured.value.__cause__ is None


@pytest.mark.parametrize(
    "adapter_name",
    [
        "openai-compatible-embeddings",
        "cohere-embeddings",
        "voyage-embeddings",
        "jina-embeddings",
        "gemini-embeddings",
        "jina-reranker",
        "cohere-reranker",
    ],
)
def test_embedding_and_reranking_secret_store_errors_do_not_expose_credentials(
    adapter_name: str,
) -> None:
    secret = "provider-secret-store-error-sentinel"
    secret_store = _FailingSecretStore(secret)
    adapters = {
        "openai-compatible-embeddings": (
            OpenAICompatibleEmbeddingProvider(
                "https://provider.example/v1", "fixture", secret_store
            ),
            lambda provider: provider.embed("fixture-model", ["private corpus text"]),
        ),
        "cohere-embeddings": (
            CohereEmbeddingProvider("fixture", secret_store),
            lambda provider: provider.embed("fixture-model", ["private corpus text"]),
        ),
        "voyage-embeddings": (
            VoyageEmbeddingProvider("fixture", secret_store),
            lambda provider: provider.embed("fixture-model", ["private corpus text"]),
        ),
        "jina-embeddings": (
            JinaEmbeddingProvider("fixture", secret_store),
            lambda provider: provider.embed("fixture-model", ["private corpus text"]),
        ),
        "gemini-embeddings": (
            GeminiEmbeddingProvider("fixture", secret_store),
            lambda provider: provider.embed("gemini-embedding-001", ["private corpus text"]),
        ),
        "jina-reranker": (
            JinaRerankingProvider("fixture", secret_store),
            lambda provider: provider.rerank("fixture-model", "query", ["candidate"]),
        ),
        "cohere-reranker": (
            CohereRerankingProvider("fixture", secret_store),
            lambda provider: provider.rerank("fixture-model", "query", ["candidate"]),
        ),
    }
    _provider, invoke = adapters[adapter_name]

    with pytest.raises(ModelProviderError) as captured:
        invoke(_provider)

    formatted_error = "".join(
        traceback.format_exception(captured.type, captured.value, captured.value.__traceback__)
    )
    assert str(captured.value) == "provider credential could not be resolved"
    assert secret not in formatted_error
    assert captured.value.__cause__ is None


@pytest.mark.parametrize(
    ("endpoint", "marker"),
    [
        (
            "https://provider.example/v1/sk-proj-endpoint-credential-sentinel",
            "sk-proj-endpoint-credential-sentinel",
        ),
        (
            "https://provider.example/v1/%2573k%252Dproj%252Dencoded%252Dcredential",
            "sk-proj-encoded-credential",
        ),
        (
            "https://sk-proj-host-credential-sentinel.example/v1",
            "sk-proj-host-credential-sentinel",
        ),
        (
            "https://sk%2Dproj%2Dencoded%2Dhost%2Dcredential.example/v1",
            "sk-proj-encoded-host-credential",
        ),
    ],
)
def test_provider_adapters_reject_credential_like_endpoint_metadata_without_echoing(
    endpoint: str, marker: str
) -> None:
    secrets = _NewlineSecret("fixture-secret")
    constructors = (
        lambda: OpenAICompatibleProvider(endpoint, "fixture", secrets),
        lambda: OpenAIProvider("fixture", secrets, endpoint=endpoint),
        lambda: AnthropicProvider("fixture", secrets, endpoint=endpoint),
        lambda: AnthropicCompatibleProvider(endpoint, "fixture", secrets),
        lambda: GeminiProvider("fixture", secrets, endpoint=endpoint),
        lambda: OpenAICompatibleEmbeddingProvider(endpoint, "fixture", secrets),
        lambda: OpenAIEmbeddingProvider("fixture", secrets, endpoint=endpoint),
        lambda: CohereEmbeddingProvider("fixture", secrets, endpoint=endpoint),
        lambda: VoyageEmbeddingProvider("fixture", secrets, endpoint=endpoint),
        lambda: JinaEmbeddingProvider("fixture", secrets, endpoint=endpoint),
        lambda: GeminiEmbeddingProvider("fixture", secrets, endpoint=endpoint),
        lambda: JinaRerankingProvider("fixture", secrets, endpoint=endpoint),
        lambda: CohereRerankingProvider("fixture", secrets, endpoint=endpoint),
    )

    for construct in constructors:
        with pytest.raises(ModelProviderError) as captured:
            construct()
        formatted_error = "".join(traceback.format_exception(captured.value))
        assert marker not in str(captured.value)
        assert endpoint not in str(captured.value)
        assert marker not in formatted_error
        assert endpoint not in formatted_error


@pytest.mark.parametrize(
    "adapter_name",
    [
        "openai-compatible-embeddings",
        "cohere-embeddings",
        "voyage-embeddings",
        "jina-embeddings",
        "gemini-embeddings",
        "jina-reranker",
        "cohere-reranker",
    ],
)
def test_invalid_credential_header_is_normalized_without_leaking_secret(
    adapter_name, local_provider_endpoint
):
    secret = "provider-fixture-secret-do-not-disclose"
    secret_store = _NewlineSecret(f"{secret}\nInjected: {secret}")
    endpoint, received_requests = local_provider_endpoint
    adapters = {
        "openai-compatible-embeddings": (
            OpenAICompatibleEmbeddingProvider(endpoint, "fixture", secret_store),
            lambda provider: provider.embed("model", ["text"]),
        ),
        "cohere-embeddings": (
            CohereEmbeddingProvider("fixture", secret_store, endpoint=endpoint),
            lambda provider: provider.embed("model", ["text"]),
        ),
        "voyage-embeddings": (
            VoyageEmbeddingProvider("fixture", secret_store, endpoint=endpoint),
            lambda provider: provider.embed("model", ["text"]),
        ),
        "jina-embeddings": (
            JinaEmbeddingProvider("fixture", secret_store, endpoint=endpoint),
            lambda provider: provider.embed("model", ["text"]),
        ),
        "gemini-embeddings": (
            GeminiEmbeddingProvider("fixture", secret_store, endpoint=endpoint),
            lambda provider: provider.embed("gemini-embedding-001", ["text"]),
        ),
        "jina-reranker": (
            JinaRerankingProvider("fixture", secret_store, endpoint=endpoint),
            lambda provider: provider.rerank("model", "query", ["candidate"]),
        ),
        "cohere-reranker": (
            CohereRerankingProvider("fixture", secret_store, endpoint=endpoint),
            lambda provider: provider.rerank("model", "query", ["candidate"]),
        ),
    }
    provider, invoke = adapters[adapter_name]

    with pytest.raises(ModelProviderError) as captured:
        invoke(provider)

    formatted_error = "".join(
        traceback.format_exception(captured.type, captured.value, captured.value.__traceback__)
    )
    assert secret not in str(captured.value)
    assert secret not in formatted_error
    assert captured.value.__cause__ is None
    assert received_requests == []


def test_provider_rejects_cross_origin_redirect_without_forwarding_auth(monkeypatch):
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

    secret = "provider-redirect-secret-sentinel"
    redirect_body_marker = "redirect-body-must-not-leak"
    destination_headers: list[str | None] = []
    source_authorization: list[str | None] = []

    class DestinationHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            destination_headers.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"unexpected redirect follow"}}]}')

        def do_GET(self) -> None:
            destination_headers.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    destination = ThreadingHTTPServer(("127.0.0.1", 0), DestinationHandler)
    destination.daemon_threads = True
    destination_thread = Thread(target=destination.serve_forever, daemon=True)
    destination_thread.start()

    location = f"http://127.0.0.1:{destination.server_port}/different-origin"

    class SourceHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            source_authorization.append(self.headers.get("Authorization"))
            self.send_response(302)
            self.send_header("Location", location)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(redirect_body_marker.encode("ascii"))

        def log_message(self, format: str, *args: object) -> None:
            return

    source = ThreadingHTTPServer(("127.0.0.1", 0), SourceHandler)
    source.daemon_threads = True
    source_thread = Thread(target=source.serve_forever, daemon=True)
    source_thread.start()
    try:
        provider = OpenAICompatibleProvider(
            f"http://127.0.0.1:{source.server_port}/v1",
            "fixture-secret",
            _NewlineSecret(secret),
        )
        with pytest.raises(ModelProviderError) as captured:
            provider.generate(GenerationRequest("corpus text", "fixture-model"))

        formatted_error = "".join(
            traceback.format_exception(captured.type, captured.value, captured.value.__traceback__)
        )
        assert str(captured.value) == "openai-compatible generation HTTP error: 302"
        assert secret not in formatted_error
        assert redirect_body_marker not in formatted_error
        assert location not in formatted_error
        assert captured.value.__cause__ is None
        assert source_authorization == [f"Bearer {secret}"]
        assert destination_headers == []
    finally:
        source.shutdown()
        source_thread.join(timeout=5)
        source.server_close()
        destination.shutdown()
        destination_thread.join(timeout=5)
        destination.server_close()


@pytest.mark.parametrize(
    ("proxy_configured_at_construction", "credential_reference"),
    [(False, None), (True, "fixture-key")],
)
def test_loopback_provider_rechecks_proxy_policy_when_sending(
    monkeypatch, proxy_configured_at_construction, credential_reference
):
    for name in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)

    received_requests: list[tuple[str, str, str | None]] = []

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            received_requests.append(
                (self.path, body.decode("utf-8"), self.headers.get("Authorization"))
            )
            response = b'{"choices":[{"message":{"content":"proxied"},"finish_reason":"stop"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            return

    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
    proxy.daemon_threads = True
    proxy_thread = Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        endpoint = "http://127.0.0.1:9/v1"
        if proxy_configured_at_construction:
            monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.server_port}")
            monkeypatch.setenv("NO_PROXY", "")
        provider = OpenAICompatibleProvider(
            endpoint,
            credential_reference,
            _NewlineSecret("provider-loopback-proxy-secret"),
            provider_id="ollama",
        )

        if not proxy_configured_at_construction:
            # Simulate an environment proxy being added after the unauthenticated
            # provider passes its constructor-time local endpoint check.
            monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.server_port}")
            monkeypatch.setenv("NO_PROXY", "")

        with pytest.raises(ModelProviderError, match="endpoint is unavailable"):
            provider.generate(GenerationRequest("corpus proxy sentinel", "local-model"))

        assert received_requests == []
    finally:
        proxy.shutdown()
        proxy_thread.join(timeout=5)
        proxy.server_close()


class _RedirectResponse:
    def __init__(self, status: int, location: str) -> None:
        self.code = status
        self.status = status
        self.reason = "Found" if status == 302 else "OK"
        self.headers = Message()
        if status == 302:
            self.headers["Location"] = location
        self.body = b"{}"

    def info(self) -> Message:
        return self.headers

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@pytest.mark.parametrize(
    "adapter_name",
    [
        "openai-compatible-generation",
        "openai-generation",
        "anthropic-generation",
        "gemini-generation",
        "openai-compatible-embedding",
        "openai-embedding",
        "cohere-embedding",
        "voyage-embedding",
        "jina-embedding",
        "gemini-embedding",
    ],
)
@pytest.mark.parametrize("redirect_status", [301, 302, 303, 307, 308])
def test_provider_requests_do_not_follow_cross_origin_redirects(
    monkeypatch, adapter_name: str, redirect_status: int
) -> None:
    """Exercise every generation/embedding transport through a mocked HTTP connection."""
    for name in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)

    secret = "provider-redirect-fixture-secret"
    project_content = "provider-redirect-project-content-sentinel"
    source_host = "provider.example.test"
    destination_host = "unreviewed.example.test"
    destination = f"http://{destination_host}/collect"
    requests: list[tuple[str, str, bytes | None, dict[str, str]]] = []

    class MockHTTPConnection(http.client.HTTPConnection):
        def __init__(self, host: str, **_kwargs: object) -> None:
            self.host = host
            self.sock = None

        def set_debuglevel(self, level: int) -> None:
            return None

        def request(
            self,
            method: str,
            url: str,
            body: Any = None,
            headers: Mapping[str, Any] | None = None,
            *,
            encode_chunked: bool = False,
        ) -> None:
            request_body = body if isinstance(body, bytes) else None
            request_headers = {
                str(key): str(value) for key, value in (headers or {}).items()
            }
            requests.append((self.host, url, request_body, request_headers))

        def getresponse(self) -> Any:
            status = redirect_status if self.host == source_host else 200
            return _RedirectResponse(status, destination)

        def close(self) -> None:
            return None

    # Keep urllib's normal opener and redirect processing; only replace the
    # socket-owning connection, so this test performs no network I/O.
    monkeypatch.setattr(http.client, "HTTPConnection", MockHTTPConnection)
    endpoint = f"http://{source_host}/v1"
    secret_store = _NewlineSecret(secret)

    with pytest.raises(ModelProviderError) as captured:
        if adapter_name == "openai-compatible-generation":
            OpenAICompatibleProvider(endpoint, "fixture", secret_store).generate(
                GenerationRequest(project_content, "fixture-model")
            )
        elif adapter_name == "openai-generation":
            OpenAIProvider("fixture", secret_store, endpoint=endpoint).generate(
                GenerationRequest(project_content, "fixture-model")
            )
        elif adapter_name == "anthropic-generation":
            AnthropicProvider("fixture", secret_store, endpoint=endpoint).generate(
                GenerationRequest(project_content, "fixture-model")
            )
        elif adapter_name == "gemini-generation":
            GeminiProvider("fixture", secret_store, endpoint=endpoint).generate(
                GenerationRequest(project_content, "fixture-model")
            )
        elif adapter_name == "openai-compatible-embedding":
            OpenAICompatibleEmbeddingProvider(endpoint, "fixture", secret_store).embed(
                "fixture-model", [project_content]
            )
        elif adapter_name == "openai-embedding":
            OpenAIEmbeddingProvider("fixture", secret_store, endpoint=endpoint).embed(
                "fixture-model", [project_content]
            )
        elif adapter_name == "cohere-embedding":
            CohereEmbeddingProvider("fixture", secret_store, endpoint=endpoint).embed(
                "fixture-model", [project_content]
            )
        elif adapter_name == "voyage-embedding":
            VoyageEmbeddingProvider("fixture", secret_store, endpoint=endpoint).embed(
                "fixture-model", [project_content]
            )
        elif adapter_name == "jina-embedding":
            JinaEmbeddingProvider("fixture", secret_store, endpoint=endpoint).embed(
                "fixture-model", [project_content]
            )
        elif adapter_name == "gemini-embedding":
            GeminiEmbeddingProvider("fixture", secret_store, endpoint=endpoint).embed(
                "gemini-embedding-001", [project_content]
            )
        else:
            raise AssertionError(f"unknown adapter: {adapter_name}")

    formatted_error = "".join(
        traceback.format_exception(captured.type, captured.value, captured.value.__traceback__)
    )
    assert secret not in formatted_error
    assert project_content not in formatted_error
    assert destination not in formatted_error
    assert len(requests) == 1
    host, _path, body, headers = requests[0]
    assert host == source_host
    assert body is not None and project_content.encode("utf-8") in body
    assert secret in " ".join(headers.values())
    assert destination_host not in [request[0] for request in requests]
