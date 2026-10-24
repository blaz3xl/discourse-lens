import json
import traceback
from email.message import Message
from io import BytesIO
from typing import Any, Self, cast
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from discourselens.errors import (
    DocumentError,
    ModelProviderError,
    ModelResponseError,
    ProjectError,
    QuotaError,
    RateLimitError,
)
from discourselens.execution import (
    _default_provider,
    execute_model_preset,
    preview_model_preset_request,
    snapshot_model_preset,
)
from discourselens.models import (
    CredentialReference,
    EchoProvider,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelGateway,
    ModelPreset,
    RerankResult,
    TokenUsage,
)
from discourselens.provider_catalog import supported_providers
from discourselens.providers import (
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
    _open_provider_request,
    _RejectRedirects,
)
from discourselens.secrets import NO_AUTH_REFERENCE, ReferenceSecretStore


def test_gateway_is_provider_neutral_and_validates_reasoning():
    gateway = ModelGateway({"fixture.echo": EchoProvider()})
    assert (
        gateway.generate("fixture.echo", GenerationRequest("evidence", "test", "low")).text
        == "evidence"
    )
    with pytest.raises(ModelProviderError):
        gateway.generate("fixture.echo", GenerationRequest("evidence", "test", "max"))


def test_gateway_dispatches_document_and_query_embeddings_explicitly():
    class EmbeddingFixture:
        provider_id = "fixture.embedding"
        endpoint = "https://fixture.example/v1"

        def __init__(self) -> None:
            self.document_requests: list[list[str]] = []
            self.query_requests: list[list[str]] = []

        def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            self.document_requests.append(list(texts))
            return [(1.0, 0.0) for _ in texts]

        def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            self.query_requests.append(list(texts))
            return [(0.0, 1.0) for _ in texts]

    provider = EmbeddingFixture()
    gateway = ModelGateway({}, embedders={provider.provider_id: provider})

    assert gateway.embed(provider.provider_id, "fixture-model", ["passage"]) == [
        (1.0, 0.0)
    ]
    assert gateway.embed_query(provider.provider_id, "fixture-model", ["question"]) == [
        (0.0, 1.0)
    ]
    assert provider.document_requests == [["passage"]]
    assert provider.query_requests == [["question"]]


def test_gateway_normalizes_batched_document_and_query_embedding_vectors():
    class BatchedEmbeddingFixture:
        provider_id = "fixture.batched"
        endpoint = "https://fixture.example/v1"

        def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            assert model == "fixture-model"
            assert texts == ["passage one", "passage two"]
            return cast(Any, [[1, 0], (0, 1)])

        def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            assert model == "fixture-model"
            assert texts == ["question one", "question two"]
            return cast(Any, [[2, 0], (0, 2)])

    provider = BatchedEmbeddingFixture()
    gateway = ModelGateway({}, embedders={provider.provider_id: provider})

    document_vectors = gateway.embed(
        provider.provider_id, "fixture-model", ["passage one", "passage two"]
    )
    query_vectors = gateway.embed_query(
        provider.provider_id, "fixture-model", ["question one", "question two"]
    )

    assert document_vectors == [(1.0, 0.0), (0.0, 1.0)]
    assert query_vectors == [(2.0, 0.0), (0.0, 2.0)]
    assert all(type(value) is float for vector in document_vectors + query_vectors for value in vector)


def test_gateway_validates_batched_symmetric_query_embedding_fallback():
    class SymmetricBatchEmbeddingFixture:
        provider_id = "fixture.symmetric-batch"
        endpoint = "https://fixture.example/v1"

        def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            assert model == "fixture-model"
            assert texts == ["question one", "question two"]
            return cast(Any, [[0, 3], (0, 4)])

    provider = SymmetricBatchEmbeddingFixture()
    gateway = ModelGateway({}, embedders={provider.provider_id: provider})

    assert gateway.embed_query(
        provider.provider_id, "fixture-model", ["question one", "question two"]
    ) == [(0.0, 3.0), (0.0, 4.0)]


@pytest.mark.parametrize("operation", ["embed", "embed_query"])
@pytest.mark.parametrize(
    ("results", "message", "text_marker"),
    [
        ([], "invalid vector count", None),
        ([(1.0,)], "invalid vector count", None),
        ([(), (1.0,)], "invalid vector", None),
        ([(True,), (1.0,)], "invalid vector value", None),
        (
            [('provider-secret-marker',), (1.0,)],
            "invalid vector value",
            "provider-secret-marker",
        ),
        ([(float("nan"),), (1.0,)], "non-finite vector value", None),
        ([(float("inf"),), (1.0,)], "non-finite vector value", None),
        ([(float("-inf"),), (1.0,)], "non-finite vector value", None),
        ([(10**400,), (1.0,)], "non-finite vector value", None),
        ([(1.0,), (1.0, 2.0)], "inconsistent vector dimensions", None),
    ],
)
def test_gateway_rejects_malformed_embedding_results_without_echoing_payload(
    operation, results, message, text_marker
):
    class MalformedEmbeddingFixture:
        provider_id = "fixture.malformed"
        endpoint = "https://fixture.example/v1"

        def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            return cast(Any, results)

        def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            return cast(Any, results)

    provider = MalformedEmbeddingFixture()
    gateway = ModelGateway({}, embedders={provider.provider_id: provider})
    request = getattr(gateway, operation)

    with pytest.raises(ModelResponseError, match=message) as captured:
        request(provider.provider_id, "fixture-model", ["passage", "another passage"])

    if text_marker is not None:
        assert text_marker not in str(captured.value)


def test_gateway_query_embedding_falls_back_to_symmetric_document_operation():
    class SymmetricEmbeddingFixture:
        provider_id = "fixture.symmetric"
        endpoint = "https://fixture.example/v1"

        def __init__(self) -> None:
            self.requests: list[list[str]] = []

        def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            self.requests.append(list(texts))
            return [(1.0, 0.0) for _ in texts]

    provider = SymmetricEmbeddingFixture()
    gateway = ModelGateway({}, embedders={provider.provider_id: provider})

    assert gateway.embed_query(provider.provider_id, "fixture-model", ["question"]) == [
        (1.0, 0.0)
    ]
    assert provider.requests == [["question"]]


def test_gateway_rejects_unregistered_or_misidentified_embedding_provider():
    class MisidentifiedEmbeddingFixture:
        provider_id = "fixture.actual"
        endpoint = "https://fixture.example/v1"

        def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            raise AssertionError("mismatched embedder must not receive a request")

    gateway = ModelGateway({}, embedders={"fixture.selected": MisidentifiedEmbeddingFixture()})
    with pytest.raises(ModelProviderError, match="no embedding provider registered"):
        gateway.embed("fixture.missing", "model", ["text"])
    with pytest.raises(ModelProviderError, match="identity does not match selected provider"):
        gateway.embed("fixture.selected", "model", ["text"])


def test_model_capability_declarations_snapshot_mutable_inputs_before_gateway_checks():
    mutable_efforts: list[str] = []
    mutable_parameters: list[str] = []
    capabilities = ModelCapabilities(
        reasoning_efforts=cast(Any, mutable_efforts),
        generation_parameters=cast(Any, mutable_parameters),
    )

    class DeclaredProvider(EchoProvider):
        def capabilities(self, model: str) -> ModelCapabilities:
            return capabilities

    mutable_efforts.append("high")
    mutable_parameters.append("unlisted_parameter")
    assert capabilities.reasoning_efforts == ()
    assert capabilities.generation_parameters == ()

    gateway = ModelGateway({"fixture.echo": DeclaredProvider()})
    with pytest.raises(ModelProviderError, match="unsupported reasoning configuration"):
        gateway.generate("fixture.echo", GenerationRequest("prompt", "model", "high"))
    with pytest.raises(ModelProviderError, match="unsupported generation parameter"):
        gateway.generate(
            "fixture.echo",
            GenerationRequest("prompt", "model", parameters={"unlisted_parameter": 1}),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"text": 1}, "capability text must be a boolean"),
        ({"reasoning_efforts": ("",)}, "reasoning_efforts must contain non-empty text"),
        ({"generation_parameters": "temperature"}, "generation_parameters must contain"),
    ],
)
def test_model_capability_declarations_reject_malformed_values(kwargs, message):
    with pytest.raises(ModelProviderError, match=message):
        ModelCapabilities(**kwargs)


def test_gateway_normalizes_unknown_provider_to_domain_error():
    gateway = ModelGateway({"fixture.echo": EchoProvider()})
    with pytest.raises(ModelProviderError, match="no generation provider registered"):
        gateway.generate("missing-provider", GenerationRequest("evidence", "test"))


def test_gateway_rejects_provider_identity_mismatch_before_capability_lookup():
    class MisidentifiedProvider:
        provider_id = "actual-provider"

        def capabilities(self, model: str) -> ModelCapabilities:
            raise AssertionError("mismatched provider must be rejected before use")

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            raise AssertionError("mismatched provider must not receive the request")

    gateway = ModelGateway({"selected-provider": MisidentifiedProvider()})

    with pytest.raises(ModelProviderError, match="identity does not match selected provider"):
        gateway.generate("selected-provider", GenerationRequest("evidence", "test"))


def test_gateway_rejects_reranker_identity_mismatch_before_reranking():
    class MisidentifiedReranker:
        provider_id = "actual-reranker"

        def rerank(
            self,
            model: str,
            query: str,
            documents: list[str],
            *,
            top_n: int | None = None,
        ) -> list[Any]:
            raise AssertionError("mismatched reranker must not receive the request")

    gateway = ModelGateway({}, rerankers={"selected-reranker": MisidentifiedReranker()})

    with pytest.raises(ModelProviderError, match="identity does not match selected provider"):
        gateway.rerank("selected-reranker", "model", "query", ["candidate"])


@pytest.mark.parametrize(
    ("results", "top_n"),
    [
        ([RerankResult(2, 0.5)], None),
        ([RerankResult(0, 0.8), RerankResult(0, 0.7)], None),
        ([RerankResult(0, float("nan"))], None),
        ([object()], None),
        ((RerankResult(0, 0.8),), None),
        ([RerankResult(0, 0.8), RerankResult(1, 0.7)], 1),
    ],
)
def test_gateway_rejects_malformed_reranker_results(results, top_n):
    class MalformedReranker:
        provider_id = "fixture.reranker"

        def rerank(
            self,
            model: str,
            query: str,
            documents: list[str],
            *,
            top_n: int | None = None,
        ) -> list[Any]:
            return cast(list[Any], results)

    gateway = ModelGateway({}, rerankers={"fixture.reranker": MalformedReranker()})

    with pytest.raises(ModelResponseError, match="reranker returned"):
        gateway.rerank("fixture.reranker", "model", "query", ["one", "two"], top_n=top_n)


def test_gateway_rejects_model_without_text_generation_capability():
    class NonTextProvider:
        provider_id = "fixture.non-text"

        def capabilities(self, model: str) -> ModelCapabilities:
            return ModelCapabilities(text=False)

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            raise AssertionError("generation must not reach a non-text model")

    gateway = ModelGateway({"fixture.non-text": NonTextProvider()})
    with pytest.raises(ModelProviderError, match="text-generation support"):
        gateway.generate("fixture.non-text", GenerationRequest("evidence", "image-only"))


@pytest.mark.parametrize("mismatch", ["provider", "model"])
def test_gateway_rejects_generation_response_for_different_selected_identity(mismatch):
    class MismatchedResponseProvider:
        provider_id = "fixture.selected"

        def capabilities(self, model: str) -> ModelCapabilities:
            return ModelCapabilities()

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(
                "must not be accepted",
                "fixture.other" if mismatch == "provider" else self.provider_id,
                "other-model" if mismatch == "model" else request.model,
            )

    gateway = ModelGateway({"fixture.selected": MismatchedResponseProvider()})

    with pytest.raises(ModelResponseError, match="identity does not match the selected preset"):
        gateway.generate("fixture.selected", GenerationRequest("prompt", "selected-model"))


def test_generation_response_rejects_malformed_runtime_fields():
    with pytest.raises(ModelResponseError, match="response text must be text"):
        GenerationResponse(cast(Any, None), "fixture.provider", "model")
    with pytest.raises(ModelResponseError, match="identity must be non-empty text"):
        GenerationResponse("text", "  ", "model")
    with pytest.raises(ModelResponseError, match="validated token usage"):
        GenerationResponse("text", "fixture.provider", "model", cast(Any, {"output": 1}))


def test_gateway_rejects_provider_returning_non_response_object():
    class MalformedProvider:
        provider_id = "fixture.malformed"

        def capabilities(self, model: str) -> ModelCapabilities:
            return ModelCapabilities()

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            return cast(Any, {"text": "forged", "provider": self.provider_id})

    gateway = ModelGateway({"fixture.malformed": MalformedProvider()})
    with pytest.raises(ModelResponseError, match="malformed generation response"):
        gateway.generate("fixture.malformed", GenerationRequest("prompt", "model"))


def test_gateway_returns_consistent_provider_error_for_unsupported_parameters():
    gateway = ModelGateway({"fixture.echo": EchoProvider()})
    with pytest.raises(ModelProviderError, match="unsupported generation parameter"):
        gateway.generate(
            "fixture.echo",
            GenerationRequest("evidence", "test", parameters={"temperature": 0.2}),
        )


def test_gateway_rejects_structured_output_without_exact_model_capability():
    gateway = ModelGateway({"fixture.echo": EchoProvider()})
    with pytest.raises(ModelProviderError, match="structured-output support"):
        gateway.generate(
            "fixture.echo",
            GenerationRequest(
                "prompt",
                "fixture",
                response_schema_name="finding_review",
                response_schema={"type": "object"},
            ),
        )


def test_generation_request_validates_and_bounds_output_schema():
    with pytest.raises(ModelProviderError, match="both a schema name and schema"):
        GenerationRequest("prompt", "model", response_schema={"type": "object"})
    with pytest.raises(ModelProviderError, match="schema name is invalid"):
        GenerationRequest(
            "prompt",
            "model",
            response_schema_name="bad name",
            response_schema={"type": "object"},
        )
    with pytest.raises(ModelProviderError, match="JSON values"):
        GenerationRequest(
            "prompt",
            "model",
            response_schema_name="schema",
            response_schema={"value": object()},
        )
    with pytest.raises(ModelProviderError, match="64 KiB"):
        GenerationRequest(
            "prompt",
            "model",
            response_schema_name="schema",
            response_schema={"description": "x" * 65_537},
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("reasoning", "reasoning effort must be non-empty"),
        ("reasoning_mode", "reasoning mode must be non-empty"),
    ],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_generation_request_rejects_empty_reasoning_options(field, message, value):
    with pytest.raises(ModelProviderError, match=message):
        GenerationRequest("prompt", "model", **{field: value})


@pytest.mark.parametrize(
    ("prompt", "model", "message"),
    [
        ("", "model", "generation prompt must be non-empty text"),
        ("   ", "model", "generation prompt must be non-empty text"),
        ("prompt", "", "generation model must be non-empty text"),
        ("prompt", "   ", "generation model must be non-empty text"),
        (None, "model", "generation prompt must be non-empty text"),
        ("prompt", None, "generation model must be non-empty text"),
    ],
)
def test_generation_request_rejects_empty_or_non_text_prompt_and_model(prompt, model, message):
    with pytest.raises(ModelProviderError, match=message):
        GenerationRequest(prompt, model)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reasoning", 1, "reasoning effort must be non-empty text"),
        ("reasoning_mode", 1, "reasoning mode must be non-empty text"),
        ("system_prompt", 1, "system prompt must be non-empty text"),
        ("parameters", [], "generation parameters must be a JSON object"),
        ("parameters", {1: "value"}, "generation parameter names must be non-empty text"),
        ("reasoning_budget_tokens", True, "reasoning budget must be a positive integer"),
        ("reasoning_budget_tokens", 0, "reasoning budget must be a positive integer"),
        ("response_schema_name", 1, "schema name is invalid"),
    ],
)
def test_generation_request_rejects_malformed_runtime_option_types(field, value, message):
    extra = {field: value}
    if field == "response_schema_name":
        extra["response_schema"] = {"type": "object"}
    with pytest.raises(ModelProviderError, match=message):
        GenerationRequest("prompt", "model", **extra)


@pytest.mark.parametrize("provider_kind", ["openai-compatible", "openai"])
def test_openai_protocol_adapters_map_declared_json_schema(provider_kind):
    response = MagicMock()
    response.read.return_value = b'{"choices":[{"message":{"content":"{\\"ok\\":true}"},"finish_reason":"stop"}]}'
    manager = MagicMock()
    manager.__enter__.return_value = response
    capabilities = {"model-a": ModelCapabilities(structured_output=True)}
    if provider_kind == "openai":
        provider = OpenAIProvider("OPENAI_KEY", _Secrets(), capabilities_by_model=capabilities)
    else:
        provider = OpenAICompatibleProvider(
            "https://example.test/v1",
            "OPENAI_KEY",
            _Secrets(),
            capabilities_by_model=capabilities,
        )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        provider.generate(
            GenerationRequest(
                "prompt",
                "model-a",
                response_schema_name="finding_review",
                response_schema=schema,
            )
        )
    body = json.loads(opened.call_args.args[0].data.decode())
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "finding_review", "strict": True, "schema": schema},
    }


def test_anthropic_maps_declared_json_schema_to_output_config():
    provider = AnthropicProvider(
        "OPENAI_KEY",
        _Secrets(),
        capabilities_by_model={"model-a": ModelCapabilities(structured_output=True)},
    )
    schema = {"type": "object", "additionalProperties": False, "properties": {}}
    with patch(
        "discourselens.providers._post_json",
        return_value={"content": [{"type": "text", "text": "{}"}], "stop_reason": "end_turn"},
    ) as posted:
        provider.generate(
            GenerationRequest(
                "prompt",
                "model-a",
                response_schema_name="finding_review",
                response_schema=schema,
            )
        )
    payload = posted.call_args.args[1]
    assert payload["output_config"] == {"format": {"type": "json_schema", "schema": schema}}


@pytest.mark.parametrize(
    ("reasoning_mode", "expected_thinking"),
    [
        ("adaptive", {"type": "adaptive"}),
        ("manual", {"type": "enabled", "budget_tokens": 2048}),
    ],
)
def test_anthropic_combines_structured_output_and_reasoning(
    reasoning_mode, expected_thinking
):
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    provider = AnthropicProvider(
        "OPENAI_KEY",
        _Secrets(),
        capabilities_by_model={
            "model-a": ModelCapabilities(
                structured_output=True,
                reasoning_efforts=("high",),
                reasoning_modes=("adaptive", "manual"),
                generation_parameters=("max_tokens",),
            )
        },
    )
    with patch(
        "discourselens.providers._post_json",
        return_value={"content": [{"type": "text", "text": '{"ok":true}'}], "stop_reason": "end_turn"},
    ) as posted:
        provider.generate(
            GenerationRequest(
                "prompt",
                "model-a",
                "high",
                response_schema_name="result",
                response_schema=schema,
                parameters={"max_tokens": 4096} if reasoning_mode == "manual" else {},
                reasoning_mode=reasoning_mode,
                reasoning_budget_tokens=2048 if reasoning_mode == "manual" else None,
            )
        )

    payload = posted.call_args.args[1]
    assert payload["thinking"] == expected_thinking
    assert payload["output_config"] == {
        "format": {"type": "json_schema", "schema": schema},
        "effort": "high",
    }


def test_anthropic_structured_output_requires_exact_model_capability():
    provider = AnthropicProvider("OPENAI_KEY", _Secrets())
    request = GenerationRequest(
        "prompt",
        "unprofiled-model",
        response_schema_name="finding_review",
        response_schema={"type": "object", "properties": {}},
    )

    assert not provider.capabilities("unprofiled-model").structured_output
    with (
        patch("discourselens.providers._post_json") as posted,
        pytest.raises(ModelProviderError, match="structured-output support"),
    ):
        provider.generate(request)

    posted.assert_not_called()


def test_gemini_maps_declared_json_schema_to_response_format():
    provider = GeminiProvider(
        "OPENAI_KEY",
        _Secrets(),
        capabilities_by_model={"model-a": ModelCapabilities(structured_output=True)},
    )
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    with patch(
        "discourselens.providers._post_json",
        return_value={"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}]},
    ) as posted:
        provider.generate(
            GenerationRequest(
                "prompt",
                "model-a",
                response_schema_name="finding_review",
                response_schema=schema,
            )
        )
    payload = posted.call_args.args[1]
    assert payload["generationConfig"]["responseFormat"] == {
        "text": {"mimeType": "APPLICATION_JSON", "schema": schema}
    }


def test_provider_catalog_distinguishes_native_from_explicit_compatible_transports():
    entries = {item.provider_id: item for item in supported_providers()}
    assert entries["openai"].transport == "openai-chat-completions"
    assert entries["openai"].endpoint_policy == "optional_override"
    assert entries["anthropic"].transport == "anthropic-messages"
    assert entries["anthropic"].native_reasoning_mapping
    assert entries["gemini"].transport == "gemini-generate-content"
    assert entries["gemini-embeddings"].transport == "gemini-batch-embed-content-v1beta"
    assert not entries["gemini-embeddings"].generation
    assert entries["gemini-embeddings"].embeddings
    assert entries["ollama"].endpoint_policy == "required"
    assert entries["ollama"].transport == "openai-compatible"
    assert not entries["openrouter"].native_reasoning_mapping
    assert entries["cohere-embeddings"].transport == "cohere-embed-v2"
    assert not entries["cohere-embeddings"].generation
    assert entries["cohere-embeddings"].embeddings
    assert entries["voyage-embeddings"].transport == "voyage-text-embeddings-v1"
    assert not entries["voyage-embeddings"].generation
    assert entries["voyage-embeddings"].embeddings
    assert entries["jina-embeddings"].transport == "jina-embeddings-api-v1"
    assert not entries["jina-embeddings"].generation
    assert entries["jina-embeddings"].embeddings


@pytest.mark.parametrize(
    "provider_id",
    [
        item.provider_id
        for item in supported_providers()
        if item.transport == "openai-compatible"
    ],
)
def test_default_provider_preserves_each_cataloged_compatible_provider_id(provider_id):
    provider = _default_provider("https://provider.example/v1", "DL_TEST_KEY", provider_id)

    assert provider.provider_id == provider_id


def test_provider_http_transport_rejects_redirects(monkeypatch):
    opener = MagicMock()
    expected_response = MagicMock()
    opener.open.return_value = expected_response
    seen_handlers = []

    def build_test_opener(*handlers):
        seen_handlers.extend(handlers)
        return opener

    monkeypatch.setattr("discourselens.providers.build_opener", build_test_opener)
    request = Request("https://provider.example/v1/chat/completions")

    assert _open_provider_request(request, timeout=3) is expected_response
    assert len(seen_handlers) == 1
    assert isinstance(seen_handlers[0], _RejectRedirects)
    assert (
        seen_handlers[0].redirect_request(
            request,
            cast(Any, None),
            307,
            "Temporary Redirect",
            cast(Any, {}),
            "https://other.example/collect",
        )
        is None
    )


class _Secrets:
    def get(self, reference: str) -> str:
        assert reference == "OPENAI_KEY"
        return "not-persisted"


class _AnySecrets:
    def get(self, reference: str) -> str:
        assert reference
        return "not-persisted"


class _NoSecrets:
    def get(self, reference: str) -> str:
        raise AssertionError(f"no secret resolution expected for {reference!r}")


def _with_provider_id(provider: Any, provider_id: str) -> Any:
    """Give injected test adapters the identity selected by their project preset."""
    provider.provider_id = provider_id
    return provider


class _BoundedResponse:
    def __init__(
        self,
        body: bytes,
        *,
        content_length: str | None = None,
        transfer_encoding: str | None = None,
    ) -> None:
        self.body = body
        self.headers = Message()
        if content_length is not None:
            self.headers["Content-Length"] = content_length
        if transfer_encoding is not None:
            self.headers["Transfer-Encoding"] = transfer_encoding
        self.read_limit: int | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        self.read_limit = size
        if size < 0:
            raise AssertionError("provider response reader must not perform unbounded reads")
        return self.body[:size]


def _provider_path_payload(path: str) -> dict[str, object]:
    return {
        "openai-generation": {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        },
        "anthropic-generation": {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
        },
        "embedding": {"data": [{"embedding": [0.1, 0.2]}]},
        "reranking": {"results": [{"index": 0, "relevance_score": 0.9}]},
    }[path]


def _invoke_provider_path(path: str):
    if path == "openai-generation":
        return OpenAICompatibleProvider(
            "https://example.test/v1", "OPENAI_KEY", _Secrets()
        ).generate(GenerationRequest("prompt", "model"))
    if path == "anthropic-generation":
        return AnthropicProvider("OPENAI_KEY", _Secrets()).generate(
            GenerationRequest("prompt", "model")
        )
    if path == "embedding":
        return OpenAICompatibleEmbeddingProvider(
            "https://example.test/v1", "OPENAI_KEY", _Secrets()
        ).embed("embedding-model", ["text"])
    if path == "reranking":
        return JinaRerankingProvider("OPENAI_KEY", _Secrets()).rerank(
            "reranker-model", "query", ["candidate"]
        )
    raise AssertionError(f"unknown provider response path: {path}")


def _invoke_deep_json_provider_path(path: str) -> None:
    secret_store = _AnySecrets()
    if path == "openai-compatible-embedding":
        OpenAICompatibleEmbeddingProvider(
            "https://example.test/v1", "OPENAI_KEY", secret_store
        ).embed("embedding-model", ["text"])
    elif path == "cohere-embedding":
        CohereEmbeddingProvider("OPENAI_KEY", secret_store).embed("embedding-model", ["text"])
    elif path == "voyage-embedding":
        VoyageEmbeddingProvider("OPENAI_KEY", secret_store).embed("embedding-model", ["text"])
    elif path == "jina-embedding":
        JinaEmbeddingProvider("OPENAI_KEY", secret_store).embed("embedding-model", ["text"])
    elif path == "gemini-embedding":
        GeminiEmbeddingProvider("OPENAI_KEY", secret_store).embed("gemini-embedding-001", ["text"])
    elif path == "jina-reranking":
        JinaRerankingProvider("OPENAI_KEY", secret_store).rerank("model", "query", ["candidate"])
    elif path == "cohere-reranking":
        CohereRerankingProvider("OPENAI_KEY", secret_store).rerank(
            "model", "query", ["candidate"]
        )
    else:
        raise AssertionError(f"unknown deep JSON provider path: {path}")


@pytest.mark.parametrize(
    "path",
    [
        "openai-compatible-embedding",
        "cohere-embedding",
        "voyage-embedding",
        "jina-embedding",
        "gemini-embedding",
        "jina-reranking",
        "cohere-reranking",
    ],
)
def test_provider_normalizes_excessively_nested_json_response(monkeypatch, path):
    body = b"[" * 10_000 + b"0" + b"]" * 10_000
    response = _BoundedResponse(body)

    with (
        patch("discourselens.providers._open_provider_request", return_value=response),
        pytest.raises(ModelProviderError) as captured,
    ):
        _invoke_deep_json_provider_path(path)

    assert not isinstance(captured.value, RecursionError)
    assert len(str(captured.value)) < 200
    assert response.read_limit is not None


@pytest.mark.parametrize(
    "path",
    ["openai-generation", "anthropic-generation", "embedding", "reranking"],
)
def test_provider_paths_accept_response_at_configured_byte_limit(monkeypatch, path):
    limit = 128
    monkeypatch.setattr("discourselens.providers.MAX_PROVIDER_RESPONSE_BYTES", limit)
    encoded = json.dumps(_provider_path_payload(path), separators=(",", ":")).encode("utf-8")
    assert len(encoded) < limit
    body = encoded + b" " * (limit - len(encoded))
    response = _BoundedResponse(body, content_length=str(limit))

    with patch("discourselens.providers._open_provider_request", return_value=response):
        _invoke_provider_path(path)

    assert response.read_limit == limit + 1


@pytest.mark.parametrize(
    "path",
    ["openai-generation", "anthropic-generation", "embedding", "reranking"],
)
def test_provider_paths_reject_oversized_content_length_before_read(monkeypatch, path):
    limit = 128
    monkeypatch.setattr("discourselens.providers.MAX_PROVIDER_RESPONSE_BYTES", limit)
    encoded = json.dumps(_provider_path_payload(path), separators=(",", ":")).encode("utf-8")
    body = encoded + b" " * (limit + 1 - len(encoded))
    response = _BoundedResponse(body, content_length=str(limit + 1))

    with (
        patch("discourselens.providers._open_provider_request", return_value=response),
        pytest.raises(ModelResponseError, match="configured byte limit"),
    ):
        _invoke_provider_path(path)

    assert response.read_limit is None


@pytest.mark.parametrize(
    ("path", "content_length", "transfer_encoding"),
    [
        ("openai-generation", None, None),
        ("anthropic-generation", None, None),
        ("embedding", None, None),
        ("reranking", None, "chunked"),
        # A dishonest short length cannot bypass the bounded body read.
        ("openai-generation", "1", None),
    ],
)
def test_provider_paths_bound_oversized_body_without_trusting_content_length(
    monkeypatch, path, content_length, transfer_encoding
):
    limit = 128
    monkeypatch.setattr("discourselens.providers.MAX_PROVIDER_RESPONSE_BYTES", limit)
    encoded = json.dumps(_provider_path_payload(path), separators=(",", ":")).encode("utf-8")
    body = encoded + b" " * (limit + 1 - len(encoded))
    response = _BoundedResponse(
        body, content_length=content_length, transfer_encoding=transfer_encoding
    )

    with (
        patch("discourselens.providers._open_provider_request", return_value=response),
        pytest.raises(ModelResponseError, match="configured byte limit"),
    ):
        _invoke_provider_path(path)

    assert response.read_limit == limit + 1


def test_openai_compatible_provider_uses_a_secret_reference_and_parses_response():
    response = MagicMock()
    response.read.return_value = b'{"choices":[{"message":{"content":"grounded"},"finish_reason":"stop"}]}'
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = OpenAICompatibleProvider(
        endpoint="https://example.test/v1/",
        credential_reference="OPENAI_KEY",
        secret_store=_Secrets(),
    )

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        result = provider.generate(
            GenerationRequest("Use only evidence.", "model-a", parameters={"temperature": 0.2})
        )

    assert result.text == "grounded"
    request = opened.call_args.args[0]
    assert request.full_url == "https://example.test/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer not-persisted"
    assert json.loads(request.data.decode())["temperature"] == 0.2


@pytest.mark.parametrize("invalid_count", [True, False, -1, 1.5, "3"])
def test_provider_usage_ignores_malformed_token_counts(invalid_count):
    response = MagicMock()
    response.read.return_value = json.dumps(
        {
            "choices": [{"message": {"content": "grounded"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": invalid_count,
                "completion_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }
    ).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = OpenAICompatibleProvider(
        endpoint="https://example.test/v1",
        credential_reference="OPENAI_KEY",
        secret_store=_Secrets(),
    )

    with patch("discourselens.providers._open_provider_request", return_value=manager):
        result = provider.generate(GenerationRequest("prompt", "model-a"))

    assert result.usage == TokenUsage(input_tokens=None, output_tokens=7, reasoning_tokens=2)


def test_openai_reasoning_count_is_a_breakdown_of_total_completion_tokens():
    response = MagicMock()
    response.read.return_value = json.dumps(
        {
            "choices": [{"message": {"content": "grounded"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }
    ).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = OpenAICompatibleProvider(
        endpoint="https://example.test/v1",
        credential_reference="OPENAI_KEY",
        secret_store=_Secrets(),
    )

    with patch("discourselens.providers._open_provider_request", return_value=manager):
        result = provider.generate(GenerationRequest("prompt", "model-a"))

    assert result.usage == TokenUsage(input_tokens=5, output_tokens=7, reasoning_tokens=2)


def test_openai_inconsistent_reasoning_breakdown_keeps_only_safe_total_usage():
    response = MagicMock()
    response.read.return_value = json.dumps(
        {
            "choices": [{"message": {"content": "grounded"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 9},
            },
        }
    ).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = OpenAICompatibleProvider(
        endpoint="https://example.test/v1",
        credential_reference="OPENAI_KEY",
        secret_store=_Secrets(),
    )

    with patch("discourselens.providers._open_provider_request", return_value=manager):
        result = provider.generate(GenerationRequest("prompt", "model-a"))

    assert result.usage == TokenUsage(input_tokens=5, output_tokens=7)


@pytest.mark.parametrize("invalid_count", [True, -1, 1.5, "3"])
def test_token_usage_rejects_invalid_direct_counts(invalid_count):
    from discourselens.errors import ModelResponseError

    with pytest.raises(ModelResponseError, match="non-negative integer"):
        TokenUsage(input_tokens=invalid_count)


def test_openai_compatible_loopback_provider_can_send_without_authorization():
    response = MagicMock()
    response.read.return_value = b'{"choices":[{"message":{"content":"local"},"finish_reason":"stop"}]}'
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = OpenAICompatibleProvider(
        endpoint="http://127.0.0.1:11434/v1",
        credential_reference=None,
        secret_store=_NoSecrets(),
        provider_id="ollama",
    )

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        result = provider.generate(GenerationRequest("prompt", "local-model"))

    request = opened.call_args.args[0]
    assert result.text == "local"
    assert request.get_header("Authorization") is None


def test_openai_compatible_provider_rejects_no_auth_for_non_loopback_endpoint():
    with pytest.raises(ModelProviderError, match="loopback"):
        OpenAICompatibleProvider(
            "https://api.example.test/v1", None, _NoSecrets(), provider_id="openai-compatible"
        )


def test_openai_compatible_embedding_omits_authorization_for_loopback():
    response = MagicMock()
    response.read.return_value = b'{"embeddings":[[0.1,0.2]]}'
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = OpenAICompatibleEmbeddingProvider(
        "http://localhost:11434/v1", None, _NoSecrets(), provider_id="ollama"
    )

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        assert provider.embed("embedding-model", ["local text"]) == [(0.1, 0.2)]

    request = opened.call_args.args[0]
    assert request.get_header("Authorization") is None
    assert request.full_url == "http://localhost:11434/api/embed"
    assert json.loads(request.data) == {
        "model": "embedding-model",
        "input": ["local text"],
        "truncate": False,
    }


def test_no_auth_credential_is_saved_only_for_compatible_loopback(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    credential_id = project.save_credential_reference(
        CredentialReference("ollama", "local", NO_AUTH_REFERENCE, "http://localhost:11434/v1")
    )
    assert credential_id
    assert project.credential_reference("ollama", "local")["secret_reference"] == NO_AUTH_REFERENCE

    with pytest.raises(ProjectError, match="loopback"):
        project.save_credential_reference(
            CredentialReference("ollama", "remote", NO_AUTH_REFERENCE, "https://remote.example/v1")
        )
    with pytest.raises(ProjectError, match="loopback"):
        project.save_credential_reference(
            CredentialReference(
                "ollama", "private-lan", NO_AUTH_REFERENCE, "http://192.168.1.20:11434/v1"
            )
        )
    with pytest.raises(ProjectError, match="loopback"):
        project.save_credential_reference(CredentialReference("openai", "none", NO_AUTH_REFERENCE))


def test_saved_no_auth_preset_executes_via_compatible_adapter_without_auth_header(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference("ollama", "local", NO_AUTH_REFERENCE, "http://localhost:11434/v1")
    )
    project.save_model_preset(ModelPreset("local", "ollama", "local", "fixture-model"))
    response = MagicMock()
    response.read.return_value = b'{"choices":[{"message":{"content":"local answer"},"finish_reason":"stop"}]}'
    manager = MagicMock()
    manager.__enter__.return_value = response

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        result = execute_model_preset(project, "local", "prompt")

    assert result.response.text == "local answer"
    assert opened.call_args.args[0].get_header("Authorization") is None


def test_provider_rate_limit_is_typed_and_preserves_only_numeric_retry_after():
    remote_message = "sk-fake-provider-body-must-not-escape"
    headers = Message()
    headers["Retry-After"] = "3.5"
    error = HTTPError(
        "https://example.test/v1/chat/completions",
        429,
        "Too Many Requests",
        headers,
        BytesIO(
            json.dumps({"error": {"type": "rate_limit_error", "message": remote_message}}).encode()
        ),
    )
    provider = OpenAICompatibleProvider("https://example.test/v1", "OPENAI_KEY", _Secrets())

    with (
        patch("discourselens.providers._open_provider_request", side_effect=error) as opened,
        pytest.raises(RateLimitError) as captured,
    ):
        provider.generate(GenerationRequest("prompt", "model-a"))

    assert captured.value.retry_after_seconds == 3.5
    assert remote_message not in str(captured.value)
    assert opened.call_count == 1


def test_provider_quota_error_body_is_classified_without_echoing_provider_text():
    remote_message = "billing detail with fake-token-do-not-return"
    error = HTTPError(
        "https://example.test/v1/chat/completions",
        429,
        "Too Many Requests",
        Message(),
        BytesIO(
            json.dumps(
                {
                    "error": {
                        "code": "api_error",
                        "type": "insufficient_quota",
                        "message": remote_message,
                    }
                }
            ).encode()
        ),
    )
    provider = OpenAICompatibleProvider("https://example.test/v1", "OPENAI_KEY", _Secrets())

    with (
        patch("discourselens.providers._open_provider_request", side_effect=error),
        pytest.raises(QuotaError) as captured,
    ):
        provider.generate(GenerationRequest("prompt", "model-a"))

    assert remote_message not in str(captured.value)
    assert isinstance(captured.value, ModelProviderError)


def test_provider_quota_code_on_403_is_not_misclassified_as_authentication():
    error = HTTPError(
        "https://example.test/v1/chat/completions",
        403,
        "Forbidden",
        Message(),
        BytesIO(
            json.dumps(
                {"error": {"code": "insufficient_balance", "message": "no credits"}}
            ).encode()
        ),
    )
    provider = OpenAICompatibleProvider("https://example.test/v1", "OPENAI_KEY", _Secrets())

    with (
        patch("discourselens.providers._open_provider_request", side_effect=error),
        pytest.raises(QuotaError),
    ):
        provider.generate(GenerationRequest("prompt", "model-a"))


def test_openai_provider_uses_the_official_default_endpoint_without_an_endpoint_setting():
    response = MagicMock()
    response.read.return_value = b'{"choices":[{"message":{"content":"grounded"},"finish_reason":"stop"}]}'
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = OpenAIProvider("OPENAI_KEY", _Secrets())

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        result = provider.generate(GenerationRequest("Use only evidence.", "gpt-fixture"))

    assert result.provider == "openai"
    assert result.text == "grounded"
    assert opened.call_args.args[0].full_url == "https://api.openai.com/v1/chat/completions"


def test_openai_reasoning_effort_maps_only_for_exact_model_capability():
    response = MagicMock()
    response.read.return_value = b'{"choices":[{"message":{"content":"grounded"},"finish_reason":"stop"}]}'
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = OpenAIProvider(
        "OPENAI_KEY",
        _Secrets(),
        capabilities_by_model={"reasoning-model": ModelCapabilities(reasoning_efforts=("high",))},
    )

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        provider.generate(GenerationRequest("prompt", "reasoning-model", "high"))

    body = json.loads(opened.call_args.args[0].data.decode())
    assert body["reasoning_effort"] == "high"
    with pytest.raises(ModelProviderError, match="does not declare reasoning effort"):
        OpenAIProvider("OPENAI_KEY", _Secrets()).generate(
            GenerationRequest("prompt", "unknown-model", "high")
        )


@pytest.mark.parametrize("provider_kind", ["openai", "anthropic", "gemini"])
def test_low_level_reasoning_capabilities_cannot_expand_native_wire_allowlist(provider_kind):
    """Caller-supplied model capabilities cannot invent native wire values."""
    capabilities = {
        "fixture-model": ModelCapabilities(
            reasoning_efforts=("ultra-private-effort",),
            reasoning_modes=("adaptive",),
        )
    }
    providers = {
        "openai": OpenAIProvider("OPENAI_KEY", _Secrets(), capabilities_by_model=capabilities),
        "anthropic": AnthropicProvider(
            "OPENAI_KEY", _Secrets(), capabilities_by_model=capabilities
        ),
        "gemini": GeminiProvider("OPENAI_KEY", _Secrets(), capabilities_by_model=capabilities),
    }
    provider = providers[provider_kind]
    request = GenerationRequest(
        "prompt",
        "fixture-model",
        "ultra-private-effort",
        reasoning_mode="adaptive" if provider_kind == "anthropic" else None,
    )

    with (
        patch("discourselens.providers._open_provider_request") as opened,
        patch("discourselens.providers._post_json") as posted,
        pytest.raises(ModelProviderError, match="native mapping does not support reasoning effort"),
    ):
        provider.generate(request)

    opened.assert_not_called()
    posted.assert_not_called()


@pytest.mark.parametrize("provider_kind", ["anthropic", "cohere"])
def test_low_level_reasoning_modes_cannot_expand_native_wire_mapping(provider_kind):
    mode = "unmapped-native-mode"
    capabilities = {"fixture-model": ModelCapabilities(reasoning_modes=(mode,))}
    providers = {
        "anthropic": AnthropicProvider(
            "OPENAI_KEY", _Secrets(), capabilities_by_model=capabilities
        ),
        "cohere": CohereProvider("OPENAI_KEY", _Secrets(), capabilities_by_model=capabilities),
    }

    with (
        patch("discourselens.providers._open_provider_request") as opened,
        patch("discourselens.providers._post_json") as posted,
        pytest.raises(ModelProviderError),
    ):
        providers[provider_kind].generate(
            GenerationRequest("prompt", "fixture-model", reasoning_mode=mode)
        )

    opened.assert_not_called()
    posted.assert_not_called()


def test_openai_embedding_provider_uses_the_official_default_endpoint():
    response = MagicMock()
    response.read.return_value = b'{"data":[{"embedding":[0.25,0.75]}]}'
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = OpenAIEmbeddingProvider("OPENAI_KEY", _Secrets())

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        result = provider.embed("embedding-fixture", ["evidence"])

    assert result == [(0.25, 0.75)]
    assert opened.call_args.args[0].full_url == "https://api.openai.com/v1/embeddings"


def test_openai_embedding_provider_orders_indexed_response_rows():
    response = MagicMock()
    response.read.return_value = json.dumps(
        {
            "data": [
                {"index": 1, "embedding": [0.8, 0.9]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
    ).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = OpenAIEmbeddingProvider("OPENAI_KEY", _Secrets())

    with patch("discourselens.providers._open_provider_request", return_value=manager):
        result = provider.embed("embedding-fixture", ["first", "second"])

    assert result == [(0.1, 0.2), (0.8, 0.9)]


def test_openai_embedding_provider_rejects_duplicate_response_indices():
    response = MagicMock()
    response.read.return_value = json.dumps(
        {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 0, "embedding": [0.8, 0.9]},
            ]
        }
    ).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = OpenAIEmbeddingProvider("OPENAI_KEY", _Secrets())

    with (
        patch("discourselens.providers._open_provider_request", return_value=manager),
        pytest.raises(ModelResponseError, match="ordered embeddings"),
    ):
        provider.embed("embedding-fixture", ["first", "second"])


@pytest.mark.parametrize(
    ("operation", "input_type"),
    [("embed", "search_document"), ("embed_query", "search_query")],
)
def test_cohere_embedding_adapter_maps_search_input_types(operation, input_type):
    response = MagicMock()
    response.read.return_value = b'{"embeddings":{"float":[[0.25,0.75]]}}'
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = CohereEmbeddingProvider("OPENAI_KEY", _Secrets())

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        vectors = getattr(provider, operation)("embed-v4.0", ["research text"])

    assert vectors == [(0.25, 0.75)]
    request = opened.call_args.args[0]
    assert request.full_url == "https://api.cohere.com/v2/embed"
    assert request.get_header("Authorization") == "Bearer not-persisted"
    payload = json.loads(request.data.decode())
    assert payload == {
        "model": "embed-v4.0",
        "texts": ["research text"],
        "input_type": input_type,
        "embedding_types": ["float"],
        "truncate": "NONE",
    }


def test_cohere_embedding_adapter_batches_at_documented_limit():
    managers = []
    for count in (96, 1):
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"embeddings": {"float": [[0.1, 0.2] for _ in range(count)]}}
        ).encode()
        manager = MagicMock()
        manager.__enter__.return_value = response
        managers.append(manager)
    provider = CohereEmbeddingProvider("OPENAI_KEY", _Secrets())

    with patch("discourselens.providers._open_provider_request", side_effect=managers) as opened:
        vectors = provider.embed("embed-v4.0", [f"text {index}" for index in range(97)])

    assert len(vectors) == 97
    assert opened.call_count == 2
    payloads = [json.loads(call.args[0].data.decode()) for call in opened.call_args_list]
    assert [len(payload["texts"]) for payload in payloads] == [96, 1]


def test_cohere_embedding_adapter_rejects_mismatched_response_count():
    response = MagicMock()
    response.read.return_value = b'{"embeddings":{"float":[[0.1,0.2]]}}'
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = CohereEmbeddingProvider("OPENAI_KEY", _Secrets())

    with (
        patch("discourselens.providers._open_provider_request", return_value=manager),
        pytest.raises(ModelResponseError, match="dimensions do not match"),
    ):
        provider.embed("embed-v4.0", ["first", "second"])


@pytest.mark.parametrize(
    ("operation", "input_type"),
    [("embed", "document"), ("embed_query", "query")],
)
def test_voyage_embedding_adapter_maps_search_input_types(operation, input_type):
    response = MagicMock()
    response.read.return_value = b'{"data":[{"index":0,"embedding":[0.25,0.75]}]}'
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = VoyageEmbeddingProvider("OPENAI_KEY", _Secrets())

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        vectors = getattr(provider, operation)("voyage-4-large", ["research text"])

    assert vectors == [(0.25, 0.75)]
    request = opened.call_args.args[0]
    assert request.full_url == "https://api.voyageai.com/v1/embeddings"
    assert request.get_header("Authorization") == "Bearer not-persisted"
    payload = json.loads(request.data.decode())
    assert payload == {
        "input": ["research text"],
        "model": "voyage-4-large",
        "input_type": input_type,
        "truncation": False,
        "output_dtype": "float",
    }


def test_voyage_embedding_adapter_orders_responses_and_batches_requests():
    managers = []
    for count in (128, 1):
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "data": [
                    {"index": index, "embedding": [float(index), 1.0]}
                    for index in reversed(range(count))
                ]
            }
        ).encode()
        manager = MagicMock()
        manager.__enter__.return_value = response
        managers.append(manager)
    provider = VoyageEmbeddingProvider("OPENAI_KEY", _Secrets())

    with patch("discourselens.providers._open_provider_request", side_effect=managers) as opened:
        vectors = provider.embed("voyage-4", [f"text {index}" for index in range(129)])

    assert len(vectors) == 129
    assert vectors[0] == (0.0, 1.0)
    assert vectors[127] == (127.0, 1.0)
    assert opened.call_count == 2
    payloads = [json.loads(call.args[0].data.decode()) for call in opened.call_args_list]
    assert [len(payload["input"]) for payload in payloads] == [128, 1]


@pytest.mark.parametrize(
    "malformed_index",
    [pytest.param(True, id="boolean"), pytest.param(0.0, id="float"), pytest.param("0", id="string")],
)
def test_voyage_embedding_adapter_rejects_non_integer_response_indices(malformed_index):
    response = MagicMock()
    response.read.return_value = json.dumps(
        {"data": [{"index": malformed_index, "embedding": [0.25, 0.75]}]}
    ).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = VoyageEmbeddingProvider("OPENAI_KEY", _Secrets())

    with (
        patch("discourselens.providers._open_provider_request", return_value=manager),
        pytest.raises(ModelResponseError, match="ordered embeddings"),
    ):
        provider.embed("voyage-4-large", ["research text"])


@pytest.mark.parametrize("provider_kind", ["openai", "cohere", "voyage"])
@pytest.mark.parametrize(
    "invalid_vector",
    [
        pytest.param([], id="empty"),
        pytest.param([0.1, True], id="boolean"),
        pytest.param([0.1, "0.2"], id="numeric-string"),
        pytest.param([0.1, float("nan")], id="nan"),
        pytest.param([0.1, float("inf")], id="infinity"),
        pytest.param([0.1, 10**400], id="integer-overflow"),
    ],
)
def test_embedding_adapters_reject_malformed_or_non_finite_vectors(provider_kind, invalid_vector):
    if provider_kind == "openai":
        provider = OpenAICompatibleEmbeddingProvider(
            "https://example.test/v1", "OPENAI_KEY", _Secrets()
        )
        payload = {"data": [{"embedding": invalid_vector}]}
    elif provider_kind == "cohere":
        provider = CohereEmbeddingProvider("OPENAI_KEY", _Secrets())
        payload = {"embeddings": {"float": [invalid_vector]}}
    else:
        provider = VoyageEmbeddingProvider("OPENAI_KEY", _Secrets())
        payload = {"data": [{"index": 0, "embedding": invalid_vector}]}

    response = MagicMock()
    # Python's JSON decoder accepts NaN and Infinity extensions; the adapter
    # must still reject them before vectors reach a persisted index.
    response.read.return_value = json.dumps(payload, allow_nan=True).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response

    with (
        patch("discourselens.providers._open_provider_request", return_value=manager),
        pytest.raises(ModelResponseError),
    ):
        provider.embed("embedding-fixture", ["evidence"])


def test_provider_direct_calls_reject_unsupported_reasoning_without_network():
    with pytest.raises(ModelProviderError, match="does not declare reasoning effort"):
        OpenAICompatibleProvider("https://example.test/v1", "OPENAI_KEY", _Secrets()).generate(
            GenerationRequest("Use only evidence.", "model-a", "high")
        )


@pytest.mark.parametrize(
    "provider",
    [
        OpenAICompatibleProvider("https://example.test/v1", "OPENAI_KEY", _Secrets()),
        OpenAIProvider("OPENAI_KEY", _Secrets()),
        GeminiProvider("OPENAI_KEY", _Secrets()),
    ],
)
def test_provider_direct_calls_reject_unmapped_reasoning_budget_without_network(provider):
    with (
        patch("discourselens.providers._open_provider_request") as opened,
        patch("discourselens.providers._post_json") as posted,
        pytest.raises(ModelProviderError, match="does not map reasoning token budgets"),
    ):
        provider.generate(GenerationRequest("prompt", "model-a", reasoning_budget_tokens=2048))

    opened.assert_not_called()
    posted.assert_not_called()


def test_provider_direct_calls_reject_unsupported_generation_parameters_without_network():
    provider = OpenAICompatibleProvider("https://example.test/v1", "OPENAI_KEY", _Secrets())
    with pytest.raises(ModelProviderError, match="does not declare generation parameter"):
        provider.generate(
            GenerationRequest("Use only evidence.", "model-a", parameters={"top_k": 3})
        )
    with pytest.raises(ModelProviderError, match="does not declare reasoning effort"):
        GeminiProvider("OPENAI_KEY", _Secrets()).generate(
            GenerationRequest("Use only evidence.", "model-a", "high")
        )


def test_generic_compatible_adapter_never_silently_ignores_declared_reasoning():
    provider = OpenAICompatibleProvider(
        "https://example.test/v1",
        "OPENAI_KEY",
        _Secrets(),
        capabilities_by_model={"model-a": ModelCapabilities(reasoning_efforts=("high",))},
    )
    with pytest.raises(ModelProviderError, match="no native reasoning mapping"):
        provider.generate(GenerationRequest("Use only evidence.", "model-a", "high"))


@pytest.mark.parametrize(
    ("provider", "payload", "expected_url", "header"),
    [
        (
            AnthropicProvider("OPENAI_KEY", _Secrets()),
            b'{"content":[{"type":"text","text":"grounded"}],"stop_reason":"end_turn"}',
            "https://api.anthropic.com/v1/messages",
            "X-api-key",
        ),
        (
            GeminiProvider("OPENAI_KEY", _Secrets()),
            b'{"candidates":[{"content":{"parts":[{"text":"grounded"}]},"finishReason":"STOP"}]}',
            "https://generativelanguage.googleapis.com/v1beta/models/model-a:generateContent",
            "X-goog-api-key",
        ),
    ],
)
def test_native_adapters_use_explicit_provider_protocols(provider, payload, expected_url, header):
    response = MagicMock()
    response.read.return_value = payload
    manager = MagicMock()
    manager.__enter__.return_value = response
    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        result = provider.generate(GenerationRequest("Use only evidence.", "model-a"))
    assert result.text == "grounded"
    request = opened.call_args.args[0]
    assert request.full_url == expected_url
    assert request.get_header(header) == "not-persisted"


@pytest.mark.parametrize("provider_class", [AnthropicProvider, GeminiProvider])
@pytest.mark.parametrize("credential_reference", ["", "   "])
def test_native_generation_adapters_reject_empty_credential_references(
    provider_class, credential_reference
):
    with pytest.raises(ModelProviderError, match="non-empty credential reference"):
        provider_class(credential_reference, _Secrets())


@pytest.mark.parametrize(
    ("provider_name", "provider_factory"),
    [
        (
            "OpenAI-compatible generation",
            lambda timeout: OpenAICompatibleProvider(
                "https://example.test/v1", "KEY_REFERENCE", _NoSecrets(), timeout_seconds=timeout
            ),
        ),
        (
            "OpenAI generation",
            lambda timeout: OpenAIProvider(
                "KEY_REFERENCE", _NoSecrets(), timeout_seconds=timeout
            ),
        ),
        (
            "OpenAI-compatible embeddings",
            lambda timeout: OpenAICompatibleEmbeddingProvider(
                "https://example.test/v1", "KEY_REFERENCE", _NoSecrets(), timeout_seconds=timeout
            ),
        ),
        (
            "OpenAI embeddings",
            lambda timeout: OpenAIEmbeddingProvider(
                "KEY_REFERENCE", _NoSecrets(), timeout_seconds=timeout
            ),
        ),
        (
            "Cohere embeddings",
            lambda timeout: CohereEmbeddingProvider(
                "KEY_REFERENCE", _NoSecrets(), timeout_seconds=timeout
            ),
        ),
        (
            "Voyage embeddings",
            lambda timeout: VoyageEmbeddingProvider(
                "KEY_REFERENCE", _NoSecrets(), timeout_seconds=timeout
            ),
        ),
        (
            "Jina embeddings",
            lambda timeout: JinaEmbeddingProvider(
                "KEY_REFERENCE", _NoSecrets(), timeout_seconds=timeout
            ),
        ),
        (
            "Gemini embeddings",
            lambda timeout: GeminiEmbeddingProvider(
                "KEY_REFERENCE", _NoSecrets(), timeout_seconds=timeout
            ),
        ),
        (
            "Jina reranking",
            lambda timeout: JinaRerankingProvider(
                "KEY_REFERENCE", _NoSecrets(), timeout_seconds=timeout
            ),
        ),
        (
            "Cohere reranking",
            lambda timeout: CohereRerankingProvider(
                "KEY_REFERENCE", _NoSecrets(), timeout_seconds=timeout
            ),
        ),
        (
            "Anthropic generation",
            lambda timeout: AnthropicProvider(
                "KEY_REFERENCE", _NoSecrets(), timeout_seconds=timeout
            ),
        ),
        (
            "Gemini generation",
            lambda timeout: GeminiProvider("KEY_REFERENCE", _NoSecrets(), timeout_seconds=timeout),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
@pytest.mark.parametrize(
    "timeout",
    [
        0,
        -1,
        False,
        True,
        float("nan"),
        float("inf"),
        float("-inf"),
        "1",
        None,
        10**400,
    ],
    ids=[
        "zero",
        "negative",
        "false",
        "true",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "string",
        "none",
        "out-of-float-range",
    ],
)
def test_all_http_adapters_reject_invalid_timeouts(provider_name, provider_factory, timeout):
    with pytest.raises(ModelProviderError, match="timeout must be numeric, finite, and positive"):
        provider_factory(timeout)


def test_openai_compatible_adapter_uses_separate_system_and_user_messages():
    response = MagicMock()
    response.read.return_value = b'{"choices":[{"message":{"content":"grounded"},"finish_reason":"stop"}]}'
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = OpenAICompatibleProvider("https://example.test/v1", "OPENAI_KEY", _Secrets())

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        provider.generate(
            GenerationRequest("corpus data", "model-a", system_prompt="trusted policy")
        )

    payload = json.loads(opened.call_args.args[0].data.decode())
    assert payload["messages"] == [
        {"role": "system", "content": "trusted policy"},
        {"role": "user", "content": "corpus data"},
    ]


def test_anthropic_and_gemini_use_native_system_instruction_fields():
    anthropic = AnthropicProvider("OPENAI_KEY", _Secrets())
    with patch(
        "discourselens.providers._post_json",
        return_value={"content": [{"type": "text", "text": "grounded"}], "stop_reason": "end_turn"},
    ) as posted:
        anthropic.generate(
            GenerationRequest("corpus data", "claude-fixture", system_prompt="trusted policy")
        )
    anthropic_payload = posted.call_args.args[1]
    assert anthropic_payload["system"] == "trusted policy"
    assert anthropic_payload["messages"] == [{"role": "user", "content": "corpus data"}]

    gemini = GeminiProvider("OPENAI_KEY", _Secrets())
    with patch(
        "discourselens.providers._post_json",
        return_value={"candidates": [{"content": {"parts": [{"text": "grounded"}]}, "finishReason": "STOP"}]},
    ) as posted:
        gemini.generate(
            GenerationRequest("corpus data", "gemini-fixture", system_prompt="trusted policy")
        )
    gemini_payload = posted.call_args.args[1]
    assert gemini_payload["systemInstruction"] == {"parts": [{"text": "trusted policy"}]}
    assert gemini_payload["contents"] == [{"role": "user", "parts": [{"text": "corpus data"}]}]


def test_provider_and_gateway_reject_undeclared_system_prompt_support():
    provider = OpenAICompatibleProvider(
        "https://example.test/v1",
        "OPENAI_KEY",
        _Secrets(),
        capabilities_by_model={"model-a": ModelCapabilities(system_prompt=False)},
    )
    request = GenerationRequest("corpus data", "model-a", system_prompt="trusted policy")
    with (
        patch("discourselens.providers._open_provider_request") as opened,
        pytest.raises(ModelProviderError, match="system-prompt support"),
    ):
        provider.generate(request)
    opened.assert_not_called()
    with pytest.raises(ModelProviderError, match="system-prompt support"):
        ModelGateway({provider.provider_id: provider}).generate(provider.provider_id, request)


@pytest.mark.parametrize(
    ("provider", "parameters", "payload_path", "payload_key", "expected"),
    [
        (
            AnthropicProvider("OPENAI_KEY", _Secrets()),
            {"temperature": 0.2, "max_tokens": 123},
            ("temperature",),
            "temperature",
            0.2,
        ),
        (
            GeminiProvider("OPENAI_KEY", _Secrets()),
            {"temperature": 0.2, "max_output_tokens": 123},
            ("generationConfig", "maxOutputTokens"),
            "maxOutputTokens",
            123,
        ),
    ],
)
def test_native_adapters_map_supported_parameters_to_provider_wire_format(
    provider, parameters, payload_path, payload_key, expected
):
    response = MagicMock()
    response.read.return_value = b'{"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}'
    manager = MagicMock()
    manager.__enter__.return_value = response
    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        provider.generate(GenerationRequest("prompt", "model-a", parameters=parameters))
    sent = json.loads(opened.call_args.args[0].data.decode())
    cursor = sent
    for key in payload_path:
        cursor = cursor[key]
    assert cursor == expected
    if isinstance(provider, AnthropicProvider):
        assert sent["max_tokens"] == 123


def test_gemini_reasoning_level_maps_to_thinking_config_for_declared_model():
    response = MagicMock()
    response.read.return_value = b'{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}'
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = GeminiProvider(
        "OPENAI_KEY",
        _Secrets(),
        capabilities_by_model={
            "gemini-fixture": ModelCapabilities(reasoning_efforts=("low", "high"))
        },
    )

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        provider.generate(GenerationRequest("prompt", "gemini-fixture", "low"))

    body = json.loads(opened.call_args.args[0].data.decode())
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    with pytest.raises(ModelProviderError, match="does not declare reasoning effort"):
        GeminiProvider("OPENAI_KEY", _Secrets()).generate(
            GenerationRequest("prompt", "unconfigured-gemini", "low")
        )


def test_gemini_generation_skips_thought_parts_and_joins_answer_parts():
    provider = GeminiProvider("OPENAI_KEY", _Secrets())
    with patch(
        "discourselens.providers._post_json",
        return_value={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "private thought", "thought": True},
                            {"text": "Grounded "},
                            {"text": "answer."},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        },
    ):
        result = provider.generate(GenerationRequest("prompt", "gemini-thinking-fixture"))

    assert result.text == "Grounded answer."


def test_gemini_usage_combines_candidate_and_thought_tokens_into_total_output():
    provider = GeminiProvider("OPENAI_KEY", _Secrets())
    with patch(
        "discourselens.providers._post_json",
        return_value={
            "candidates": [
                {"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 7,
                "thoughtsTokenCount": 3,
                "totalTokenCount": 21,
            },
        },
    ):
        result = provider.generate(GenerationRequest("prompt", "gemini-fixture"))

    assert result.usage == TokenUsage(input_tokens=11, output_tokens=10, reasoning_tokens=3)


@pytest.mark.parametrize(
    "usage_metadata",
    [
        {
            "promptTokenCount": 11,
            "candidatesTokenCount": 7,
        },
        {
            "promptTokenCount": 11,
            "candidatesTokenCount": 7,
            "thoughtsTokenCount": "private-provider-payload",
        },
        {
            "promptTokenCount": 11,
            "candidatesTokenCount": 7,
            "thoughtsTokenCount": 3,
            "totalTokenCount": 22,
        },
    ],
)
def test_gemini_incomplete_or_inconsistent_usage_cannot_make_output_estimate(
    usage_metadata,
):
    provider = GeminiProvider("OPENAI_KEY", _Secrets())
    with patch(
        "discourselens.providers._post_json",
        return_value={
            "candidates": [
                {"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": usage_metadata,
        },
    ):
        result = provider.generate(GenerationRequest("prompt", "gemini-fixture"))

    assert result.usage == TokenUsage(input_tokens=11)
    assert "private-provider-payload" not in repr(result.usage)


@pytest.mark.parametrize("endpoint", ["example.test", "file:///tmp/model", "https://u:p@x.test"])
def test_openai_compatible_provider_rejects_unsafe_endpoint_shapes(endpoint):
    with pytest.raises(ModelProviderError):
        OpenAICompatibleProvider(endpoint, "KEY", _Secrets())


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://@provider.example/v1",
        "https://provider.example/v1?",
        "https://provider.example/v1#",
        "https://[::1]external/v1",
        "https://provider.example\n/v1",
        "https://provider.example:bad/v1",
    ],
)
def test_openai_compatible_provider_rejects_ambiguous_endpoint_urls(endpoint):
    with pytest.raises(ModelProviderError):
        OpenAICompatibleProvider(endpoint, "KEY", _Secrets())


@pytest.mark.parametrize("top_n", [True, 1.5, "1"])
def test_jina_reranker_rejects_non_integer_top_n_before_network_call(top_n):
    provider = JinaRerankingProvider("OPENAI_KEY", _Secrets())

    with (
        patch("discourselens.providers._open_provider_request") as opened,
        pytest.raises(ModelProviderError, match="top_n"),
    ):
        provider.rerank("model", "query", ["candidate"], top_n=top_n)

    opened.assert_not_called()


@pytest.mark.parametrize(
    "provider_type",
    [JinaRerankingProvider, CohereRerankingProvider],
    ids=["jina", "cohere"],
)
def test_rerankers_reject_scores_that_disagree_with_returned_rank_order(provider_type):
    provider = provider_type("OPENAI_KEY", _Secrets())
    response = _BoundedResponse(
        json.dumps(
            {
                "results": [
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.9},
                ]
            }
        ).encode("utf-8")
    )

    with (
        patch("discourselens.providers._open_provider_request", return_value=response),
        pytest.raises(ModelResponseError, match="lacks valid reranking indices and scores"),
    ):
        provider.rerank("model", "query", ["first", "second"])


def test_project_stores_multiple_credential_aliases_without_secret_values(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "personal", "DL_PERSONAL_KEY", "https://local.test/v1"
        )
    )
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "university", "DL_UNIVERSITY_KEY", "https://local.test/v1"
        )
    )
    preset_id = project.save_model_preset(
        ModelPreset("research", "openai-compatible", "university", "local-model", "high")
    )
    assert preset_id
    aliases = project.list_credential_references()
    assert [row["credential_alias"] for row in aliases] == ["personal", "university"]
    assert project.list_model_presets()[0]["credential_alias"] == "university"
    stored = [dict(row) for row in aliases]
    assert stored[1]["secret_reference"] == "DL_UNIVERSITY_KEY"
    assert "actual-secret-value" not in str(stored)


def test_model_routing_profile_requires_saved_presets_and_round_trips(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "local", "DL_LOCAL_KEY", "http://localhost:11434/v1"
        )
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "local", "review-model"))
    with pytest.raises(ProjectError, match="unknown model preset"):
        project.save_model_routing_profile("invalid", {"fairclough-text": "missing"})

    profile_id = project.save_model_routing_profile(
        "method-routing", {"fairclough-text": "review"}, "review"
    )
    routes, fallback = project.model_routing_profile("method-routing")

    assert profile_id
    assert routes == {"fairclough-text": "review"}
    assert fallback == "review"


def test_model_preset_execution_records_hashes_not_prompt_or_response(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "local", "DL_LOCAL", "http://local.test/v1")
    )
    project.save_model_preset(ModelPreset("safe-local", "openai-compatible", "local", "fixture"))

    result = execute_model_preset(
        project,
        "safe-local",
        "private corpus excerpt",
        provider_factory=lambda endpoint, reference, provider_id: _with_provider_id(
            EchoProvider(), provider_id
        ),
    )

    assert result.response.text == "private corpus excerpt"
    execution = project.db.execute(
        "SELECT * FROM model_executions WHERE id=?", (result.execution_id,)
    ).fetchone()
    assert execution["request_hash"] and execution["response_hash"]
    assert "private corpus excerpt" not in str(dict(execution))


@pytest.mark.parametrize("mismatch", ["provider", "model"])
def test_model_preset_rejects_response_identity_mismatch(tmp_path, mismatch):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "university", "DL_KEY", "http://local.test/v1")
    )
    project.save_model_preset(
        ModelPreset("review", "openai-compatible", "university", "selected-model")
    )

    class MismatchedIdentityProvider(EchoProvider):
        def generate(self, request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(
                "must not be accepted",
                "different-provider" if mismatch == "provider" else self.provider_id,
                "different-model" if mismatch == "model" else request.model,
            )

    with pytest.raises(ModelResponseError, match="does not match the selected preset"):
        execute_model_preset(
            project,
            "review",
            "prompt",
            provider_factory=lambda endpoint, reference, provider_id: _with_provider_id(
                MismatchedIdentityProvider(), provider_id
            ),
        )

    execution = project.db.execute("SELECT * FROM model_executions").fetchone()
    assert execution["provider_id"] == "openai-compatible"
    assert execution["credential_alias"] == "university"
    assert execution["model"] == "selected-model"
    assert execution["error_type"] == "ModelResponseError"
    assert execution["response_hash"] is None


def test_model_preset_pins_selected_credential_alias_and_reference(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "personal", "DL_PERSONAL_KEY", "http://local.test/v1"
        )
    )
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "university", "DL_UNIVERSITY_KEY", "http://local.test/v1"
        )
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "university", "fixture"))

    snapshot = snapshot_model_preset(project, "review")
    project.save_model_preset(ModelPreset("review", "openai-compatible", "personal", "fixture"))
    references: list[str | None] = []
    execute_model_preset(
        project,
        "review",
        "fixture prompt",
        preset_snapshot=snapshot,
        provider_factory=lambda endpoint, reference, provider_id: (
            references.append(reference) or _with_provider_id(EchoProvider(), provider_id)
        ),
    )

    assert snapshot["credential_alias"] == "university"
    assert project.model_preset("review")["credential_alias"] == "personal"
    assert snapshot["credential_reference_hash"]
    assert references == ["DL_UNIVERSITY_KEY"]
    assert "DL_UNIVERSITY_KEY" not in json.dumps(snapshot)


def test_model_preset_rejects_credential_alias_remap_before_provider_creation(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "personal", "DL_PERSONAL_KEY", "http://local.test/v1"
        )
    )
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "university", "DL_UNIVERSITY_KEY", "http://local.test/v1"
        )
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "university", "fixture"))
    snapshot = snapshot_model_preset(project, "review")
    project.save_model_preset(ModelPreset("review", "openai-compatible", "personal", "fixture"))
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "university", "DL_REPLACEMENT_KEY", "http://local.test/v1"
        )
    )
    provider_references: list[str | None] = []

    with pytest.raises(ModelProviderError, match="refusing implicit credential substitution"):
        execute_model_preset(
            project,
            "review",
            "fixture prompt",
            preset_snapshot=snapshot,
            provider_factory=lambda endpoint, reference, provider_id: (
                provider_references.append(reference)
                or _with_provider_id(EchoProvider(), provider_id)
            ),
        )

    assert provider_references == []


def test_model_preset_rejects_missing_snapshot_alias_without_falling_back(
    tmp_path, monkeypatch
):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "personal", "DL_PERSONAL_KEY", "http://local.test/v1"
        )
    )
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "university", "DL_UNIVERSITY_KEY", "http://local.test/v1"
        )
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "university", "fixture"))
    snapshot = snapshot_model_preset(project, "review")
    project.save_model_preset(ModelPreset("review", "openai-compatible", "personal", "fixture"))
    project.db.execute(
        "DELETE FROM provider_credentials WHERE provider_id=? AND credential_alias=?",
        ("openai-compatible", "university"),
    )
    monkeypatch.setenv("DL_PERSONAL_KEY", "personal-test-secret-must-not-be-used")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-test-secret-must-not-be-used")
    provider_references: list[str | None] = []

    with pytest.raises(ModelProviderError, match="credential alias captured.*unavailable"):
        execute_model_preset(
            project,
            "review",
            "fixture prompt",
            preset_snapshot=snapshot,
            provider_factory=lambda endpoint, reference, provider_id: (
                provider_references.append(reference)
                or _with_provider_id(EchoProvider(), provider_id)
            ),
        )

    assert provider_references == []


def test_selected_credential_alias_is_only_secret_resolved_for_request(
    tmp_path, monkeypatch
):
    from discourselens import Project

    personal_secret = "personal-test-secret-do-not-send"
    university_secret = "university-test-secret-selected"
    monkeypatch.setenv("DL_PERSONAL_KEY", personal_secret)
    monkeypatch.setenv("DL_UNIVERSITY_KEY", university_secret)
    project = Project.init(tmp_path / "selected-credential")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "personal", "DL_PERSONAL_KEY", "https://local.test/v1"
        )
    )
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "university", "DL_UNIVERSITY_KEY", "https://local.test/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("review", "openai-compatible", "university", "fixture")
    )
    references: list[str] = []
    original_get = ReferenceSecretStore.get

    def record_get(store: ReferenceSecretStore, reference: str) -> str:
        references.append(reference)
        return original_get(store, reference)

    monkeypatch.setattr(ReferenceSecretStore, "get", record_get)
    response = MagicMock()
    response.read.return_value = b'{"choices":[{"message":{"content":"grounded"},"finish_reason":"stop"}]}'
    manager = MagicMock()
    manager.__enter__.return_value = response

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        result = execute_model_preset(project, "review", "fixture prompt")

    request = opened.call_args.args[0]
    assert references == ["DL_UNIVERSITY_KEY"]
    assert request.get_header("Authorization") == f"Bearer {university_secret}"
    assert personal_secret not in request.get_header("Authorization")
    assert result.response.text == "grounded"


def test_selected_alias_secret_is_redacted_from_normalized_transport_error(
    tmp_path, monkeypatch
):
    from discourselens import Project

    personal_secret = "personal-test-secret-do-not-send"
    selected_secret = "university-test-secret-must-not-leak"
    monkeypatch.setenv("DL_PERSONAL_KEY", personal_secret)
    monkeypatch.setenv("DL_UNIVERSITY_KEY", selected_secret)
    project = Project.init(tmp_path / "selected-credential-error")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "personal", "DL_PERSONAL_KEY", "https://local.test/v1"
        )
    )
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "university", "DL_UNIVERSITY_KEY", "https://local.test/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("review", "openai-compatible", "university", "fixture")
    )
    references: list[str] = []
    original_get = ReferenceSecretStore.get

    def record_get(store: ReferenceSecretStore, reference: str) -> str:
        references.append(reference)
        return original_get(store, reference)

    monkeypatch.setattr(ReferenceSecretStore, "get", record_get)
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        MagicMock(side_effect=URLError(selected_secret)),
    )

    with pytest.raises(ModelProviderError) as captured:
        execute_model_preset(project, "review", "fixture prompt")

    formatted_error = "".join(
        traceback.format_exception(captured.type, captured.value, captured.value.__traceback__)
    )
    assert references == ["DL_UNIVERSITY_KEY"]
    assert selected_secret not in str(captured.value)
    assert selected_secret not in formatted_error
    assert selected_secret not in str(
        [dict(row) for row in project.db.execute("SELECT * FROM model_executions")]
    )
    assert captured.value.__cause__ is None


def test_model_request_hash_covers_system_policy_without_storing_prompt_text(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "local", "DL_LOCAL", "http://local.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "local", "fixture"))

    first = execute_model_preset(
        project,
        "review",
        "same corpus payload",
        system_prompt="policy version one",
        provider_factory=lambda endpoint, reference, provider_id: _with_provider_id(
            EchoProvider(), provider_id
        ),
    )
    second = execute_model_preset(
        project,
        "review",
        "same corpus payload",
        system_prompt="policy version two",
        provider_factory=lambda endpoint, reference, provider_id: _with_provider_id(
            EchoProvider(), provider_id
        ),
    )
    rows = project.db.execute(
        "SELECT * FROM model_executions WHERE id IN (?,?)",
        (first.execution_id, second.execution_id),
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["request_hash"] != rows[1]["request_hash"]
    assert "policy version" not in str([dict(row) for row in rows])


def test_preview_blocks_system_prompt_when_model_profile_disallows_it(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "local", "DL_LOCAL", "http://local.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "local", "fixture"))
    project.save_model_capability_profile(
        ModelCapabilityProfile(
            "openai-compatible",
            "fixture",
            ModelCapabilities(system_prompt=False),
        )
    )

    preview = preview_model_preset_request(
        project, "review", "untrusted corpus", system_prompt="trusted policy"
    )

    assert not preview.allowed
    assert preview.unsupported_system_prompt
    assert preview.block_reason == "provider does not declare system-prompt support"
    assert preview.system_prompt_hash
    assert preview.system_prompt_characters == len("trusted policy")
    assert "trusted policy" not in str(preview)


def test_model_preset_parameters_reach_provider_and_are_snapshotted(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "local", "DL_LOCAL", "http://local.test/v1")
    )
    parameters = {"temperature": 0.25, "max_tokens": 300}
    project.save_model_preset(
        ModelPreset("configured", "openai-compatible", "local", "fixture", parameters=parameters)
    )

    class ParameterProvider(EchoProvider):
        def capabilities(self, model: str) -> ModelCapabilities:
            return ModelCapabilities(generation_parameters=("temperature", "max_tokens"))

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            assert request.parameters == parameters
            return GenerationResponse("ok", self.provider_id, request.model)

    result = execute_model_preset(
        project,
        "configured",
        "prompt",
        provider_factory=lambda endpoint, reference, provider_id: _with_provider_id(
            ParameterProvider(), provider_id
        ),
    )
    row = project.db.execute(
        "SELECT parameters_json FROM model_executions WHERE id=?", (result.execution_id,)
    ).fetchone()
    assert json.loads(row["parameters_json"]) == parameters
    capability_snapshot = json.loads(
        project.db.execute(
            "SELECT capabilities_json FROM model_executions WHERE id=?", (result.execution_id,)
        ).fetchone()[0]
    )
    assert capability_snapshot["source"] == "adapter-default"
    assert capability_snapshot["effective"]["generation_parameters"] == [
        "temperature",
        "max_tokens",
    ]
    project.save_model_preset(
        ModelPreset(
            "configured",
            "openai-compatible",
            "local",
            "fixture",
            parameters={"temperature": 0.8, "max_tokens": 300},
        )
    )
    assert (
        json.loads(
            project.db.execute(
                "SELECT parameters_json FROM model_executions WHERE id=?", (result.execution_id,)
            ).fetchone()[0]
        )
        == parameters
    )


def test_model_preset_rejects_secret_like_parameters(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "local", "DL_LOCAL", "http://local.test/v1")
    )
    with pytest.raises(DocumentError, match="must not contain secrets"):
        project.save_model_preset(
            ModelPreset(
                "unsafe", "openai-compatible", "local", "fixture", parameters={"api_key": "x"}
            )
        )


def test_exact_model_capability_profile_restricts_request_and_is_snapshotted(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "local", "DL_LOCAL", "http://local.test/v1")
    )
    project.save_model_preset(
        ModelPreset("review", "openai-compatible", "local", "fixture", parameters={"top_k": 3})
    )
    project.save_model_capability_profile(
        ModelCapabilityProfile(
            "openai-compatible",
            "fixture",
            ModelCapabilities(generation_parameters=("temperature", "nonexistent_parameter")),
            "TESTED",
            "local contract test",
        )
    )

    preview = preview_model_preset_request(project, "review", "prompt")
    assert not preview.allowed
    assert preview.unsupported_parameters == ("top_k",)
    assert preview.effective_capabilities is not None
    assert preview.effective_capabilities["generation_parameters"] == ("temperature",)

    class ParameterProvider(EchoProvider):
        def capabilities(self, model: str) -> ModelCapabilities:
            return ModelCapabilities(generation_parameters=("temperature", "top_k"))

    with pytest.raises(ModelProviderError, match="unsupported generation parameter.*top_k"):
        execute_model_preset(
            project,
            "review",
            "prompt",
            provider_factory=lambda endpoint, reference, provider_id: _with_provider_id(
                ParameterProvider(), provider_id
            ),
        )

    row = project.db.execute("SELECT * FROM model_executions").fetchone()
    snapshot = json.loads(row["capabilities_json"])
    assert row["error_type"] == "ModelProviderError"
    assert snapshot["source"] == "project-profile"
    assert snapshot["assurance"] == "TESTED"
    assert snapshot["effective"]["generation_parameters"] == ["temperature"]
    assert json.loads(row["parameters_json"]) == {"top_k": 3}


def test_model_capability_profile_round_trips_by_exact_provider_and_model(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    profile = ModelCapabilityProfile(
        "gemini",
        "gemini-fixture",
        ModelCapabilities(
            system_prompt=False,
            structured_output=True,
            generation_parameters=("temperature", "max_output_tokens"),
        ),
        "UNASSESSED",
        "https://docs.example.test/model",
    )
    profile_id = project.save_model_capability_profile(profile)

    loaded = project.model_capability_profile("gemini", "gemini-fixture")

    assert profile_id
    assert loaded == profile
    assert loaded is not None and not loaded.capabilities.system_prompt
    assert project.model_capability_profile("gemini", "other-model") is None


@pytest.mark.parametrize("effort", ["high", "max"])
def test_openai_run_uses_model_profile_to_gate_and_snapshot_reasoning_effort(tmp_path, effort):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(CredentialReference("openai", "lab", "DL_OPENAI"))
    project.save_model_preset(
        ModelPreset("reasoning", "openai", "lab", "gpt-fixture", effort)
    )
    project.save_model_capability_profile(
        ModelCapabilityProfile(
            "openai",
            "gpt-fixture",
            ModelCapabilities(reasoning_efforts=(effort,)),
            "TESTED",
            "local adapter contract test",
        )
    )
    response = MagicMock()
    response.read.return_value = b'{"choices":[{"message":{"content":"grounded"},"finish_reason":"stop"}]}'
    manager = MagicMock()
    manager.__enter__.return_value = response

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        result = execute_model_preset(
            project,
            "reasoning",
            "prompt",
            provider_factory=lambda endpoint, reference, provider_id: OpenAIProvider(
                "OPENAI_KEY", _Secrets(), endpoint=endpoint
            ),
        )

    request_payload = json.loads(opened.call_args.args[0].data.decode())
    assert request_payload["reasoning_effort"] == effort
    execution = project.db.execute(
        "SELECT capabilities_json FROM model_executions WHERE id=?", (result.execution_id,)
    ).fetchone()
    snapshot = json.loads(execution["capabilities_json"])
    assert snapshot["source"] == "project-profile"
    assert snapshot["effective"]["reasoning_efforts"] == [effort]


def test_execution_records_provider_reported_usage_and_latency_without_prices(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "local", "DL_LOCAL", "http://localhost:11434/v1")
    )
    project.save_model_preset(ModelPreset("usage", "openai-compatible", "local", "fixture"))

    class UsageProvider(EchoProvider):
        def generate(self, request: GenerationRequest) -> GenerationResponse:
            return GenerationResponse(
                "grounded", self.provider_id, request.model, TokenUsage(11, 7, 3)
            )

    execute_model_preset(
        project,
        "usage",
        "private corpus excerpt",
        provider_factory=lambda endpoint, reference, provider_id: _with_provider_id(
            UsageProvider(), provider_id
        ),
    )
    execution = project.db.execute(
        "SELECT input_tokens,output_tokens,reasoning_tokens,latency_ms FROM model_executions"
    ).fetchone()
    assert tuple(execution)[:3] == (11, 7, 3)
    assert isinstance(execution["latency_ms"], int) and execution["latency_ms"] >= 0


def test_offline_project_blocks_model_execution_before_provider_creation(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "remote", "DL_REMOTE", "https://api.example.test/v1"
        )
    )
    project.save_model_preset(ModelPreset("safe-remote", "openai-compatible", "remote", "fixture"))
    project.set_offline_mode(True)
    with pytest.raises(ProjectError, match="offline mode"):
        execute_model_preset(project, "safe-remote", "private corpus excerpt")


def test_offline_project_permits_explicit_loopback_model_execution(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "local", "DL_LOCAL", "http://127.0.0.1:11434/v1")
    )
    project.save_model_preset(ModelPreset("safe-local", "openai-compatible", "local", "fixture"))
    project.set_offline_mode(True)
    result = execute_model_preset(
        project,
        "safe-local",
        "local corpus excerpt",
        provider_factory=lambda endpoint, reference, provider_id: _with_provider_id(
            EchoProvider(), provider_id
        ),
    )
    assert result.response.text == "local corpus excerpt"


def test_request_preview_is_secret_free_and_has_no_execution_side_effect(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
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
            "high",
            {"temperature": 0.4},
        )
    )
    project.set_offline_mode(True)
    preview = preview_model_preset_request(project, "review", "private source excerpt")
    assert not preview.allowed and preview.block_reason
    assert preview.parameters == {"temperature": 0.4}
    assert preview.prompt_characters == len("private source excerpt")
    assert "private source excerpt" not in str(preview)
    assert "DL_REMOTE" not in str(preview)
    assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0


def test_provider_failure_never_exposes_secret_in_exception_or_execution_ledger(
    tmp_path, monkeypatch
):
    from discourselens import Project

    secret = "sk-discourselens-fake-never-log"
    monkeypatch.setenv("DL_TEST_SECRET", secret)
    project = Project.init(tmp_path / "secret-failure")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "lab", "DL_TEST_SECRET", "https://provider.example/v1"
        )
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "lab", "fixture"))
    _, revision = project.import_text("source.txt", "Evidence remains local.")
    run_id = project.create_run(project.snapshot([revision]), {"profile": "fixture.secret-test"})
    project.start_run(run_id)
    monkeypatch.setattr(
        "discourselens.providers._open_provider_request",
        MagicMock(side_effect=URLError(secret)),
    )

    with pytest.raises(ModelProviderError) as captured:
        execute_model_preset(
            project,
            "review",
            "private research prompt",
            analysis_run_id=run_id,
        )

    formatted_error = "".join(
        traceback.format_exception(captured.type, captured.value, captured.value.__traceback__)
    )
    assert secret not in str(captured.value)
    assert secret not in formatted_error
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__
    rows = project.db.execute(
        "SELECT * FROM model_executions WHERE analysis_run_id=?", (run_id,)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["error_type"] == "ModelProviderError"
    assert secret not in str([dict(row) for row in rows])


def test_native_openai_preview_discloses_its_effective_default_endpoint(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(CredentialReference("openai", "lab", "DL_OPENAI"))
    project.save_model_preset(ModelPreset("review", "openai", "lab", "gpt-fixture"))

    preview = preview_model_preset_request(project, "review", "private source excerpt")

    assert preview.endpoint == "https://api.openai.com/v1"
    assert preview.allowed
    assert not preview.endpoint_is_loopback


def test_native_anthropic_reasoning_requires_exact_model_declaration(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(CredentialReference("anthropic", "lab", "DL_ANTHROPIC"))
    project.save_model_preset(
        ModelPreset("reasoning", "anthropic", "lab", "claude-fixture", "high")
    )

    preview = preview_model_preset_request(project, "reasoning", "prompt")

    assert not preview.allowed
    assert preview.unsupported_reasoning
    assert (
        preview.block_reason == "reasoning effort is unsupported by the declared model capabilities"
    )


def test_anthropic_adaptive_reasoning_uses_model_declared_native_mapping():
    provider = AnthropicProvider(
        "OPENAI_KEY",
        _Secrets(),
        capabilities_by_model={
            "claude-adaptive": ModelCapabilities(
                reasoning_efforts=("low", "high"), reasoning_modes=("adaptive",)
            )
        },
    )
    with patch(
        "discourselens.providers._post_json",
        return_value={"content": [{"type": "text", "text": "grounded"}], "stop_reason": "end_turn"},
    ) as posted:
        provider.generate(
            GenerationRequest("prompt", "claude-adaptive", "high", reasoning_mode="adaptive")
        )

    payload = posted.call_args.args[1]
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "high"}

    with pytest.raises(ModelProviderError, match="does not declare reasoning mode"):
        AnthropicProvider("OPENAI_KEY", _Secrets()).generate(
            GenerationRequest("prompt", "unprofiled", reasoning_mode="adaptive")
        )


def test_anthropic_manual_reasoning_requires_valid_explicit_budget():
    provider = AnthropicProvider(
        "OPENAI_KEY",
        _Secrets(),
        capabilities_by_model={
            "claude-manual": ModelCapabilities(
                reasoning_efforts=("medium",),
                reasoning_modes=("manual",),
                generation_parameters=("max_tokens",),
            )
        },
    )
    with patch(
        "discourselens.providers._post_json",
        return_value={"content": [{"type": "text", "text": "grounded"}], "stop_reason": "end_turn"},
    ) as posted:
        provider.generate(
            GenerationRequest(
                "prompt",
                "claude-manual",
                "medium",
                parameters={"max_tokens": 8192},
                reasoning_mode="manual",
                reasoning_budget_tokens=2048,
            )
        )
    payload = posted.call_args.args[1]
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert payload["output_config"] == {"effort": "medium"}

    with pytest.raises(ModelProviderError, match="less than max_tokens"):
        provider.generate(
            GenerationRequest(
                "prompt",
                "claude-manual",
                "medium",
                parameters={"max_tokens": 2048},
                reasoning_mode="manual",
                reasoning_budget_tokens=2048,
            )
        )


def test_other_native_adapters_do_not_accept_anthropic_reasoning_modes():
    with pytest.raises(ModelProviderError, match="does not map Anthropic reasoning modes"):
        OpenAIProvider(
            "OPENAI_KEY",
            _Secrets(),
            capabilities_by_model={
                "openai-model": ModelCapabilities(reasoning_modes=("adaptive",))
            },
        ).generate(GenerationRequest("prompt", "openai-model", reasoning_mode="adaptive"))

    with pytest.raises(ModelProviderError, match="separate reasoning mode"):
        GeminiProvider(
            "OPENAI_KEY",
            _Secrets(),
            capabilities_by_model={
                "gemini-model": ModelCapabilities(reasoning_modes=("adaptive",))
            },
        ).generate(GenerationRequest("prompt", "gemini-model", reasoning_mode="adaptive"))


def test_anthropic_reasoning_mode_is_snapshotted_and_audited(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(CredentialReference("anthropic", "lab", "DL_ANTHROPIC"))
    project.save_model_capability_profile(
        ModelCapabilityProfile(
            "anthropic",
            "claude-adaptive",
            ModelCapabilities(reasoning_efforts=("high",), reasoning_modes=("adaptive",)),
            "TESTED",
            "fixture profile",
        )
    )
    project.save_model_preset(
        ModelPreset(
            "review",
            "anthropic",
            "lab",
            "claude-adaptive",
            "high",
            reasoning_mode="adaptive",
        )
    )
    with patch(
        "discourselens.providers._post_json",
        return_value={"content": [{"type": "text", "text": "grounded"}], "stop_reason": "end_turn"},
    ) as posted:
        result = execute_model_preset(
            project,
            "review",
            "prompt",
            provider_factory=lambda endpoint, reference, provider_id: AnthropicProvider(
                cast(str, reference), _AnySecrets(), endpoint=endpoint
            ),
        )

    assert posted.call_args.args[1]["thinking"] == {"type": "adaptive"}
    row = project.db.execute(
        "SELECT reasoning,reasoning_mode,reasoning_budget_tokens,capabilities_json "
        "FROM model_executions WHERE id=?",
        (result.execution_id,),
    ).fetchone()
    assert (row["reasoning"], row["reasoning_mode"], row["reasoning_budget_tokens"]) == (
        "high",
        "adaptive",
        None,
    )
    assert json.loads(row["capabilities_json"])["effective"]["reasoning_modes"] == ["adaptive"]


def test_compatible_provider_without_endpoint_is_rejected_during_preview_and_execution(tmp_path):
    from discourselens import Project

    project = Project.init(tmp_path / "p")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "missing", "DL_COMPATIBLE")
    )
    project.save_model_preset(
        ModelPreset("invalid", "openai-compatible", "missing", "model-fixture")
    )

    preview = preview_model_preset_request(project, "invalid", "private source excerpt")

    assert not preview.allowed
    assert preview.block_reason == "provider requires an explicit compatible endpoint"
    with pytest.raises(ModelProviderError, match="requires an explicit compatible endpoint"):
        execute_model_preset(project, "invalid", "private source excerpt")
