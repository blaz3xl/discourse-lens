from typing import cast
from unittest.mock import patch

import pytest

from discourselens import Project
from discourselens.errors import ModelProviderError
from discourselens.execution import (
    _apply_project_capabilities,
    _default_provider,
    build_embedding_gateway,
    build_reranking_gateway,
    execute_model_preset,
    preview_model_preset_request,
)
from discourselens.models import (
    CredentialReference,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelPreset,
    QueryEmbeddingProvider,
)
from discourselens.provider_catalog import (
    is_openai_compatible_provider,
    supported_providers,
    supports_generation,
)


def test_product_spec_provider_coverage_is_explicit_and_uses_implemented_transports():
    entries = {item.provider_id: item for item in supported_providers()}
    expected_transports = {
        "openai": "openai-chat-completions",
        "anthropic": "anthropic-messages",
        "gemini": "gemini-generate-content",
        "cohere": "cohere-chat-v2",
        "deepseek": "openai-compatible",
        "zai": "openai-compatible",
        "xai": "openai-compatible",
        "mistral": "openai-compatible",
        "openrouter": "openai-compatible",
        "together": "openai-compatible",
        "groq": "openai-compatible",
        "fireworks": "openai-compatible",
        "voyage-embeddings": "voyage-text-embeddings-v1",
        "jina-embeddings": "jina-embeddings-api-v1",
        "ollama": "openai-compatible",
        "lm-studio": "openai-compatible",
        "vllm": "openai-compatible",
        "openai-compatible": "openai-compatible",
        "anthropic-compatible": "anthropic-messages-compatible",
    }

    assert {provider_id: entries[provider_id].transport for provider_id in expected_transports} == (
        expected_transports
    )

    native_generation_ids = {"openai", "anthropic", "gemini", "cohere"}
    openai_compatible_ids = {
        "openai-compatible",
        "deepseek",
        "zai",
        "xai",
        "mistral",
        "openrouter",
        "together",
        "groq",
        "fireworks",
        "ollama",
        "lm-studio",
        "vllm",
    }
    for provider_id in native_generation_ids:
        assert entries[provider_id].generation
        assert entries[provider_id].endpoint_policy == "optional_override"
    for provider_id in openai_compatible_ids:
        assert entries[provider_id].generation
        assert entries[provider_id].endpoint_policy == "required"
        assert is_openai_compatible_provider(provider_id)

    assert entries["voyage-embeddings"].embeddings
    assert not entries["voyage-embeddings"].generation
    assert entries["jina-embeddings"].embeddings
    assert not entries["jina-embeddings"].generation
    assert entries["anthropic-compatible"].generation
    assert entries["anthropic-compatible"].endpoint_policy == "required"
    assert not is_openai_compatible_provider("anthropic-compatible")


def test_every_cataloged_generation_provider_resolves_to_its_transport():
    from discourselens.providers import (
        AnthropicCompatibleProvider,
        AnthropicProvider,
        CohereProvider,
        GeminiProvider,
        OpenAICompatibleProvider,
        OpenAIProvider,
    )

    provider_types = {
        "openai-chat-completions": OpenAIProvider,
        "anthropic-messages": AnthropicProvider,
        "anthropic-messages-compatible": AnthropicCompatibleProvider,
        "gemini-generate-content": GeminiProvider,
        "cohere-chat-v2": CohereProvider,
        "openai-compatible": OpenAICompatibleProvider,
    }
    descriptors = [item for item in supported_providers() if item.generation]

    assert descriptors
    for descriptor in descriptors:
        provider = _default_provider(
            "https://provider.example/v1", "DL_TEST_KEY", descriptor.provider_id
        )
        assert isinstance(provider, provider_types[descriptor.transport])
        assert provider.provider_id == descriptor.provider_id


def test_unknown_provider_ids_are_not_implicitly_treated_as_compatible():
    provider_id = "unregistered-compatible-provider"

    assert not supports_generation(provider_id)
    assert not is_openai_compatible_provider(provider_id)
    with pytest.raises(
        ModelProviderError,
        match="provider 'unregistered-compatible-provider' does not support text generation",
    ):
        _default_provider("https://provider.example/v1", "DL_TEST_KEY", provider_id)


@pytest.mark.parametrize(
    ("provider_id", "has_native_reasoning_mapping"),
    [
        ("openai", True),
        ("anthropic", True),
        ("gemini", True),
        ("cohere", True),
        ("anthropic-compatible", False),
        ("openai-compatible", False),
        ("openrouter", False),
    ],
)
def test_catalog_reasoning_mapping_flag_matches_adapter_transport_contract(
    provider_id: str, has_native_reasoning_mapping: bool
):
    descriptors = {item.provider_id: item for item in supported_providers()}

    # This reports whether the selected transport has a documented reasoning
    # wire mapping. It does not assert support for any particular model; exact
    # model capabilities still gate reasoning at execution time.
    assert descriptors[provider_id].native_reasoning_mapping is has_native_reasoning_mapping


def test_anthropic_compatible_catalog_contract_is_distinct_and_generation_only():
    descriptors = {item.provider_id: item for item in supported_providers()}
    descriptor = descriptors["anthropic-compatible"]

    assert descriptor.transport == "anthropic-messages-compatible"
    assert descriptor.endpoint_policy == "required"
    assert descriptor.generation
    assert not descriptor.embeddings
    assert not descriptor.reranking
    assert not descriptor.native_reasoning_mapping
    assert not is_openai_compatible_provider("anthropic-compatible")


@pytest.mark.parametrize(
    ("endpoint", "credential_reference", "message"),
    [
        (None, "DL_TEST_KEY", "requires an explicit endpoint"),
        ("https://provider.example/v1", None, "requires a credential reference"),
        ("https://provider.example/v1", "no-auth", "requires a credential reference"),
    ],
)
def test_anthropic_compatible_requires_explicit_endpoint_and_credential(
    endpoint: str | None, credential_reference: str | None, message: str
):
    with pytest.raises(ModelProviderError, match=message):
        _default_provider(endpoint, credential_reference, "anthropic-compatible")


def test_anthropic_compatible_optional_capabilities_require_an_exact_model_profile():
    provider = _default_provider(
        "https://provider.example/v1", "DL_TEST_KEY", "anthropic-compatible"
    )
    _, defaults, _ = _apply_project_capabilities(
        None, "anthropic-compatible", "fixture-model", provider
    )
    assert not defaults.reasoning_efforts
    assert not defaults.reasoning_modes

    profile = ModelCapabilityProfile(
        "anthropic-compatible",
        "fixture-model",
        ModelCapabilities(reasoning_efforts=("low",), reasoning_modes=("adaptive",)),
    )
    _, declared, _ = _apply_project_capabilities(
        None, "anthropic-compatible", "fixture-model", provider, profile
    )
    assert declared.reasoning_efforts == ("low",)
    assert declared.reasoning_modes == ("adaptive",)


def test_every_cataloged_embedding_provider_resolves_to_its_adapter(tmp_path, monkeypatch):
    from discourselens import execution

    adapter_types = {
        "openai": "OpenAIEmbeddingProvider",
        "gemini-embeddings": "GeminiEmbeddingProvider",
        "cohere-embeddings": "CohereEmbeddingProvider",
        "voyage-embeddings": "VoyageEmbeddingProvider",
        "jina-embeddings": "JinaEmbeddingProvider",
    }
    factory_names = (
        "OpenAICompatibleEmbeddingProvider",
        "OpenAIEmbeddingProvider",
        "GeminiEmbeddingProvider",
        "CohereEmbeddingProvider",
        "VoyageEmbeddingProvider",
        "JinaEmbeddingProvider",
    )
    descriptors = [item for item in supported_providers() if item.embeddings]
    assert descriptors
    fixture_endpoint = "https://provider.example/v1"

    def make_factory(name, provider_id, constructed, endpoint):
        class StubEmbeddingProvider:
            def __init__(self):
                self.provider_id = provider_id
                self.endpoint = endpoint

            def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
                assert model == "fixture-model"
                return [(0.25, 0.75) for _ in texts]

        class Factory:
            @staticmethod
            def document_input_type(model: str) -> str:
                return "document"

            def __call__(self, *args: object, **kwargs: object):
                constructed.append((name, args))
                return StubEmbeddingProvider()

        return Factory()

    project = Project.init(tmp_path / "project")
    try:
        for descriptor in descriptors:
            provider_id = descriptor.provider_id
            expected_factory = adapter_types.get(provider_id, "OpenAICompatibleEmbeddingProvider")
            constructed: list[tuple[str, tuple[object, ...]]] = []

            for factory_name in factory_names:
                monkeypatch.setattr(
                    execution,
                    factory_name,
                    make_factory(factory_name, provider_id, constructed, fixture_endpoint),
                )

            project.save_credential_reference(
                CredentialReference(
                    provider_id,
                    "fixture",
                    "DL_TEST_KEY",
                    fixture_endpoint,
                )
            )
            gateway = build_embedding_gateway(project, provider_id, "fixture")

            assert [name for name, _ in constructed] == [expected_factory]
            assert gateway.embedders[provider_id].provider_id == provider_id
            if expected_factory == "OpenAICompatibleEmbeddingProvider":
                assert constructed[0][1][-1] == provider_id
    finally:
        project.close()


def test_every_cataloged_reranker_resolves_to_its_adapter(tmp_path, monkeypatch):
    from discourselens import execution

    adapter_types = {
        "cohere-rerank-v2": "CohereRerankingProvider",
        "jina-rerank-v1": "JinaRerankingProvider",
    }
    descriptors = [item for item in supported_providers() if item.reranking]
    assert descriptors

    project = Project.init(tmp_path / "project")
    constructed: list[str] = []

    class StubRerankingProvider:
        def __init__(self, provider_id: str):
            self.provider_id = provider_id

    def make_factory(name: str, provider_id: str):
        def factory(*args: object, **kwargs: object):
            constructed.append(name)
            return StubRerankingProvider(provider_id)

        return factory

    monkeypatch.setattr(
        execution,
        "CohereRerankingProvider",
        make_factory("CohereRerankingProvider", "cohere-reranker"),
    )
    monkeypatch.setattr(
        execution, "JinaRerankingProvider", make_factory("JinaRerankingProvider", "jina-reranker")
    )
    try:
        for descriptor in descriptors:
            project.save_credential_reference(
                CredentialReference(descriptor.provider_id, "fixture", "DL_TEST_KEY")
            )
            gateway = build_reranking_gateway(project, descriptor.provider_id, "fixture")
            assert gateway.rerankers[descriptor.provider_id].provider_id == descriptor.provider_id
            assert constructed[-1] == adapter_types[descriptor.transport]
    finally:
        project.close()


def test_cataloged_non_generation_providers_are_rejected_before_provider_creation(tmp_path):
    descriptors = [item for item in supported_providers() if not item.generation]
    assert descriptors

    project = Project.init(tmp_path / "project")

    def unexpected_provider_factory(*args: object, **kwargs: object):
        pytest.fail("unsupported generation provider was constructed")

    try:
        for descriptor in descriptors:
            provider_id = descriptor.provider_id
            project.save_credential_reference(
                CredentialReference(
                    provider_id, "fixture", "DL_TEST_KEY", "https://provider.example"
                )
            )
            preset_name = f"model-{provider_id}"
            project.save_model_preset(
                ModelPreset(preset_name, provider_id, "fixture", "fixture-model")
            )
            preview = preview_model_preset_request(
                project, preset_name, "fixture prompt", provider_factory=unexpected_provider_factory
            )
            assert not preview.allowed
            assert preview.block_reason == "provider does not support text generation"
            with pytest.raises(ModelProviderError, match="does not support text generation"):
                execute_model_preset(
                    project,
                    preset_name,
                    "fixture prompt",
                    provider_factory=unexpected_provider_factory,
                )
    finally:
        project.close()


def test_cataloged_non_embedding_providers_are_rejected_before_request(tmp_path):
    from discourselens.vectors import build_paragraph_embedding_index

    descriptors = [item for item in supported_providers() if not item.embeddings]
    assert descriptors
    project = Project.init(tmp_path / "project")
    try:
        _, revision_id = project.import_text("source.txt", "A fixture paragraph.")
        for descriptor in descriptors:
            provider_id = descriptor.provider_id
            project.save_credential_reference(
                CredentialReference(
                    provider_id, "fixture", "DL_TEST_KEY", "https://provider.example/v1"
                )
            )
            preset_name = f"embedding-{provider_id}"
            project.save_model_preset(
                ModelPreset(preset_name, provider_id, "fixture", "fixture-model")
            )
            with patch(
                "discourselens.providers._open_provider_request",
                side_effect=AssertionError("unsupported embeddings must not use the network"),
            ) as opened:
                with pytest.raises(ModelProviderError, match="no reviewed embedding adapter"):
                    build_paragraph_embedding_index(project, preset_name, revision_id)
                opened.assert_not_called()
    finally:
        project.close()


def test_cataloged_non_embedding_providers_are_rejected_before_credential_lookup():
    descriptors = [item for item in supported_providers() if not item.embeddings]
    assert descriptors

    class NoCredentialResolution:
        def credential_reference(self, provider_id: str, credential_alias: str):
            pytest.fail("unsupported embedder reached credential resolution")

    for descriptor in descriptors:
        with pytest.raises(ModelProviderError, match="no reviewed embedding adapter"):
            build_embedding_gateway(NoCredentialResolution(), descriptor.provider_id, "fixture")


def test_embedding_gateway_uses_the_explicit_credential_alias(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        project.save_credential_reference(
            CredentialReference(
                "openai-compatible",
                "personal",
                "DL_PERSONAL_KEY",
                "http://localhost:11434/v1",
            )
        )
        project.save_credential_reference(
            CredentialReference(
                "openai-compatible",
                "university",
                "DL_UNIVERSITY_KEY",
                "http://127.0.0.1:11434/v1",
            )
        )

        gateway = build_embedding_gateway(project, "openai-compatible", "university")
        selected = gateway.embedders["openai-compatible"]

        assert getattr(selected, "credential_reference", None) == "DL_UNIVERSITY_KEY"
        assert selected.endpoint == "http://127.0.0.1:11434/v1"
    finally:
        project.close()


def test_embedding_gateway_enforces_capability_for_the_exact_model(tmp_path):
    project = Project.init(tmp_path / "embedding-capability-gate")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "fixture", "DL_TEST_KEY", "https://provider.example/v1"
        )
    )

    class StubEmbeddingProvider:
        provider_id = "openai-compatible"
        endpoint = "https://provider.example/v1"

        def __init__(self):
            self.models: list[str] = []

        def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
            self.models.append(model)
            return [(0.25, 0.75) for _ in texts]

    provider = StubEmbeddingProvider()
    gateway = build_embedding_gateway(
        project, "openai-compatible", "fixture", embedding_provider=provider
    )
    try:
        # Existing unprofiled endpoint/model pairs remain usable.
        assert gateway.embed("openai-compatible", "unprofiled-model", ["text"])

        project.save_model_capability_profile(
            ModelCapabilityProfile(
                "openai-compatible", "restricted-model", ModelCapabilities(embeddings=False)
            )
        )
        with pytest.raises(
            ModelProviderError, match="disabled by the exact-model capability profile"
        ):
            gateway.embed("openai-compatible", "restricted-model", ["text"])
        with pytest.raises(
            ModelProviderError, match="disabled by the exact-model capability profile"
        ):
            gateway.embedders["openai-compatible"].embed("restricted-model", ["text"])
        with pytest.raises(
            ModelProviderError, match="disabled by the exact-model capability profile"
        ):
            cast(
                QueryEmbeddingProvider,
                gateway.embedders["openai-compatible"],
            ).embed_query("restricted-model", ["text"])
        assert provider.models == ["unprofiled-model"]

        project.save_model_capability_profile(
            ModelCapabilityProfile(
                "openai-compatible", "permitted-model", ModelCapabilities(embeddings=True)
            )
        )
        assert gateway.embed("openai-compatible", "permitted-model", ["text"])
        assert provider.models == ["unprofiled-model", "permitted-model"]
    finally:
        project.close()


def test_cataloged_non_rerankers_are_rejected_before_credential_resolution():
    descriptors = [item for item in supported_providers() if not item.reranking]
    assert descriptors

    class NoCredentialResolution:
        def credential_reference(self, provider_id: str, credential_alias: str):
            pytest.fail("unsupported reranker reached credential resolution")

    for descriptor in descriptors:
        with pytest.raises(ModelProviderError, match="no reviewed reranking adapter"):
            build_reranking_gateway(NoCredentialResolution(), descriptor.provider_id, "fixture")
