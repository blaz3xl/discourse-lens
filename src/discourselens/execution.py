"""Explicit model-preset execution with secret-free, reproducible audit metadata."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from typing import Protocol, cast

from .errors import DocumentError, ModelProviderError, ModelResponseError, ProjectError
from .models import (
    EmbeddingProvider,
    GenerationImage,
    GenerationProvider,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelGateway,
    ModelPrice,
    QueryEmbeddingProvider,
    RerankingProvider,
    RerankResult,
)
from .network_policy import is_loopback_endpoint
from .provider_catalog import (
    is_openai_compatible_provider,
    supported_providers,
    supports_generation,
)
from .providers import (
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
from .reasoning_policy import validate_cohere_manual_budget
from .secrets import NO_AUTH_REFERENCE, ReferenceSecretStore, provider_auth_reference


@dataclass(frozen=True)
class PresetExecutionResult:
    execution_id: str
    response: GenerationResponse


@dataclass(frozen=True)
class PresetRequestPreview:
    """Secret-free description of a pending model request, with no network side effect."""

    provider_id: str
    credential_alias: str
    endpoint: str | None
    model: str
    reasoning: str | None
    reasoning_mode: str | None
    reasoning_budget_tokens: int | None
    parameters: dict[str, object]
    effective_capabilities: dict[str, object] | None
    unsupported_parameters: tuple[str, ...]
    unsupported_reasoning: bool
    unsupported_reasoning_mode: bool
    unsupported_system_prompt: bool
    prompt_hash: str
    prompt_characters: int
    system_prompt_hash: str | None
    system_prompt_characters: int
    offline_mode: bool
    endpoint_is_loopback: bool
    allowed: bool
    block_reason: str | None


ProviderFactory = Callable[[str, str | None, str], GenerationProvider]


class _OfflineModeProject(Protocol):
    @property
    def offline_mode(self) -> bool: ...

    def model_capability_profile(
        self, provider_id: str, model: str
    ) -> ModelCapabilityProfile | None: ...


def require_embedding_model_capability(
    project: _OfflineModeProject, provider_id: str, model: str
) -> None:
    """Apply an exact-model embedding restriction when the project has one.

    Embedding profiles are optional so existing presets remain usable, but an
    explicit profile that disables embeddings must be honored by every route.
    """
    profile = project.model_capability_profile(provider_id, model)
    if profile is not None and not profile.capabilities.embeddings:
        raise ModelProviderError(
            "embedding is disabled by the exact-model capability profile"
        )


_DEFAULT_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com/v1",
    "cohere": "https://api.cohere.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "openai": "https://api.openai.com/v1",
}

_DEFAULT_EMBEDDING_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "cohere-embeddings": "https://api.cohere.com",
    "voyage-embeddings": "https://api.voyageai.com/v1",
    "jina-embeddings": "https://api.jina.ai/v1",
    "gemini-embeddings": "https://generativelanguage.googleapis.com/v1beta",
}


def effective_endpoint(provider_id: str, configured_endpoint: str | None) -> str | None:
    """Resolve only reviewed native defaults; compatible transports stay explicit."""
    return configured_endpoint or _DEFAULT_ENDPOINTS.get(provider_id)


def effective_embedding_endpoint(provider_id: str, configured_endpoint: str | None) -> str | None:
    """Resolve reviewed embedding defaults while keeping compatible endpoints explicit."""
    return configured_endpoint or _DEFAULT_EMBEDDING_ENDPOINTS.get(provider_id)


def _require_online_endpoint(project: _OfflineModeProject, endpoint: str, *, message: str) -> None:
    if project.offline_mode and not is_loopback_endpoint(endpoint):
        raise ProjectError(message)


class _OfflineCheckedGenerationProvider:
    """Recheck the project network policy immediately before provider execution."""

    def __init__(
        self,
        provider: GenerationProvider,
        project: _OfflineModeProject,
        endpoint: str,
    ) -> None:
        self.provider = provider
        self.project = project
        self.endpoint = endpoint

    @property
    def provider_id(self) -> str:
        return self.provider.provider_id

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.provider.capabilities(model)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        _require_online_endpoint(
            self.project,
            self.endpoint,
            message="project offline mode blocks remote model execution",
        )
        return self.provider.generate(request)

    def stream_generate(self, request: GenerationRequest) -> Iterator[str]:
        """Recheck offline policy at first iteration, immediately before transport I/O."""
        stream_generate = getattr(self.provider, "stream_generate", None)
        if not callable(stream_generate):
            raise ModelProviderError("provider does not implement generation streaming")

        def checked_stream() -> Iterator[str]:
            _require_online_endpoint(
                self.project,
                self.endpoint,
                message="project offline mode blocks remote model execution",
            )
            stream = cast(Callable[[GenerationRequest], Iterator[str]], stream_generate)
            yield from stream(request)

        return checked_stream()


class _OfflineCheckedModelGateway(ModelGateway):
    """Keep a built embedding/reranking gateway bound to current project policy."""

    def __init__(
        self,
        project: _OfflineModeProject,
        endpoint: str,
        block_message: str,
        *,
        rerankers: dict[str, RerankingProvider] | None = None,
        credential_reference_fingerprints: dict[str, str] | None = None,
        embedders: dict[str, EmbeddingProvider] | None = None,
    ) -> None:
        super().__init__(
            {},
            rerankers=rerankers,
            credential_reference_fingerprints=credential_reference_fingerprints,
            embedders=embedders,
        )
        self._project = project
        self._endpoint = endpoint
        self._block_message = block_message

    def _check_current_policy(self) -> None:
        _require_online_endpoint(self._project, self._endpoint, message=self._block_message)

    def embed(self, provider_id: str, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        require_embedding_model_capability(self._project, provider_id, model)
        self._check_current_policy()
        return super().embed(provider_id, model, texts)

    def embed_query(
        self, provider_id: str, model: str, texts: list[str]
    ) -> list[tuple[float, ...]]:
        require_embedding_model_capability(self._project, provider_id, model)
        self._check_current_policy()
        return super().embed_query(provider_id, model, texts)

    def rerank(
        self,
        provider_id: str,
        model: str,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        self._check_current_policy()
        return super().rerank(provider_id, model, query, documents, top_n=top_n)


class _PolicyCheckedEmbeddingProvider:
    """Keep the gateway's public provider mapping subject to project policy."""

    def __init__(
        self,
        project: _OfflineModeProject,
        endpoint: str,
        provider: EmbeddingProvider,
    ) -> None:
        self._project = project
        self._endpoint = endpoint
        self._provider = provider

    @property
    def provider_id(self) -> str:
        return self._provider.provider_id

    @property
    def endpoint(self) -> str:
        return self._provider.endpoint

    def __getattr__(self, name: str):
        # Preserve adapter metadata exposed by the historical mapping surface.
        return getattr(self._provider, name)

    def _check_policy(self, model: str) -> None:
        require_embedding_model_capability(self._project, self.provider_id, model)
        _require_online_endpoint(
            self._project,
            self._endpoint,
            message="project offline mode blocks remote embeddings",
        )

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self._check_policy(model)
        return self._provider.embed(model, texts)

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self._check_policy(model)
        if isinstance(self._provider, QueryEmbeddingProvider):
            return self._provider.embed_query(model, texts)
        return self._provider.embed(model, texts)


def _preset_parameters(preset) -> dict[str, object]:
    try:
        parameters = json.loads(preset["parameters_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ModelProviderError("saved model preset parameters are invalid JSON") from exc
    if not isinstance(parameters, dict):
        raise ModelProviderError("saved model preset parameters must be a JSON object")
    return parameters


def _validate_execution_parameters(parameters: dict[str, object]) -> None:
    """Reject booleans where a provider execution limit requires a token count."""
    if isinstance(parameters.get("max_tokens"), bool):
        raise ModelProviderError("max_tokens must be an integer, not a boolean")


def credential_reference_fingerprint(project, provider_id: str, credential_alias: str) -> str:
    """Return a secret-free fingerprint of the configured credential reference.

    This pins which environment/keyring reference an analysis selected without
    persisting the reference name itself or reading the secret value.
    """
    credential = project.credential_reference(provider_id, credential_alias)
    reference = credential["secret_reference"]
    if not isinstance(reference, str) or not reference:
        raise ModelProviderError("provider credential has an invalid secret reference")
    return _secret_reference_fingerprint(reference)


def _secret_reference_fingerprint(reference: str) -> str:
    return hashlib.sha256(reference.encode("utf-8")).hexdigest()


def _fingerprint_matches_reference(reference: object, expected_fingerprint: object) -> bool:
    if not isinstance(reference, str) or not reference:
        return False
    if (
        not isinstance(expected_fingerprint, str)
        or len(expected_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in expected_fingerprint)
    ):
        return False
    return hmac.compare_digest(_secret_reference_fingerprint(reference), expected_fingerprint)


def _credential_for_preset_snapshot(
    project,
    provider_id: str,
    credential_alias: str,
    expected_reference_hash: object,
):
    """Resolve only the credential pair captured by a run and fail closed on drift."""
    try:
        credential = project.credential_reference(provider_id, credential_alias)
    except DocumentError:
        raise ModelProviderError(
            "model credential alias captured by this analysis snapshot is unavailable; "
            "refusing implicit credential substitution"
        ) from None
    if not _fingerprint_matches_reference(credential["secret_reference"], expected_reference_hash):
        raise ModelProviderError(
            "model credential reference changed or is not pinned in this analysis snapshot; "
            "refusing implicit credential substitution"
        )
    return credential


def snapshot_model_preset(project, preset_name: str) -> dict[str, object]:
    """Resolve mutable preset/credential metadata into a secret-free run snapshot."""
    preset = project.model_preset(preset_name)
    credential = project.credential_reference(preset["provider_id"], preset["credential_alias"])
    credential_reference = credential["secret_reference"]
    if not isinstance(credential_reference, str) or not credential_reference:
        raise ModelProviderError("provider credential has an invalid secret reference")
    profile = project.model_capability_profile(preset["provider_id"], preset["model"])
    capability_data = None
    if profile:
        capability_data = {
            "capabilities": asdict(profile.capabilities),
            "assurance": profile.assurance,
            "source_reference": profile.source_reference,
        }
    price = project.model_price(preset["provider_id"], preset["model"])
    return {
        "preset_name": preset_name,
        "provider_id": preset["provider_id"],
        "credential_alias": preset["credential_alias"],
        "credential_reference_hash": _secret_reference_fingerprint(credential_reference),
        "model": preset["model"],
        "reasoning": preset["reasoning"],
        "reasoning_mode": preset["reasoning_mode"],
        "reasoning_budget_tokens": preset["reasoning_budget_tokens"],
        "parameters": _preset_parameters(preset),
        "endpoint": effective_endpoint(preset["provider_id"], credential["endpoint"]),
        "capability_profile": capability_data,
        "model_price": price.to_snapshot() if price is not None else None,
    }


def preset_declares_structured_output(snapshot: dict[str, object]) -> bool:
    """Return true only for an explicit capability declaration on the exact model."""
    profile = snapshot.get("capability_profile")
    if not isinstance(profile, dict):
        return False
    capabilities = profile.get("capabilities")
    return isinstance(capabilities, dict) and capabilities.get("structured_output") is True


def _snapshot_to_capability_profile(
    provider_id: str, model: str, snapshot: object
) -> ModelCapabilityProfile | None:
    if snapshot is None:
        return None
    if not isinstance(snapshot, dict):
        raise ModelProviderError("analysis run capability profile snapshot is invalid")
    capabilities_data = snapshot.get("capabilities")
    if not isinstance(capabilities_data, dict):
        raise ModelProviderError("analysis run capability profile snapshot is invalid")
    try:
        capabilities = ModelCapabilities(
            text=capabilities_data["text"],
            system_prompt=capabilities_data.get("system_prompt", True),
            vision=capabilities_data["vision"],
            structured_output=capabilities_data["structured_output"],
            streaming=capabilities_data["streaming"],
            embeddings=capabilities_data["embeddings"],
            reasoning_efforts=tuple(capabilities_data["reasoning_efforts"]),
            reasoning_modes=tuple(capabilities_data.get("reasoning_modes", ())),
            generation_parameters=tuple(capabilities_data["generation_parameters"]),
        )
        return ModelCapabilityProfile(
            provider_id,
            model,
            capabilities,
            str(snapshot["assurance"]),
            snapshot.get("source_reference"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelProviderError("analysis run capability profile snapshot is invalid") from exc


def _intersect_capabilities(
    adapter: ModelCapabilities, declared: ModelCapabilities
) -> ModelCapabilities:
    """A project declaration may narrow a transport, never invent a wire mapping."""
    return ModelCapabilities(
        text=adapter.text and declared.text,
        system_prompt=adapter.system_prompt and declared.system_prompt,
        vision=adapter.vision and declared.vision,
        structured_output=adapter.structured_output and declared.structured_output,
        streaming=adapter.streaming and declared.streaming,
        embeddings=adapter.embeddings and declared.embeddings,
        reasoning_efforts=tuple(
            item for item in adapter.reasoning_efforts if item in declared.reasoning_efforts
        ),
        reasoning_modes=tuple(
            item for item in adapter.reasoning_modes if item in declared.reasoning_modes
        ),
        generation_parameters=tuple(
            item for item in adapter.generation_parameters if item in declared.generation_parameters
        ),
    )


def _require_image_capability(
    provider_id: str, profile: ModelCapabilityProfile | None, image: GenerationImage | None
) -> None:
    """Require explicit exact-model vision declaration before credential lookup."""
    if image is None:
        return
    if (
        provider_id not in {"gemini", "openai", "anthropic", "anthropic-compatible"}
        and not is_openai_compatible_provider(provider_id)
    ):
        raise ModelProviderError(
            "image input is supported only by Gemini, OpenAI Chat Completions, "
            "and Anthropic Messages generation"
        )
    if profile is None or profile.capabilities.vision is not True:
        raise ModelProviderError(
            "image input requires an exact-model capability profile declaring vision support"
        )


@dataclass(frozen=True)
class _CapabilityRestrictedProvider:
    provider: GenerationProvider
    model: str
    effective_capabilities: ModelCapabilities

    @property
    def provider_id(self) -> str:
        return self.provider.provider_id

    def capabilities(self, model: str) -> ModelCapabilities:
        if model == self.model:
            return self.effective_capabilities
        return self.provider.capabilities(model)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return self.provider.generate(request)

    def stream_generate(self, request: GenerationRequest) -> Iterator[str]:
        stream_generate = getattr(self.provider, "stream_generate", None)
        if not callable(stream_generate):
            raise ModelProviderError("provider does not implement generation streaming")
        stream = cast(Callable[[GenerationRequest], Iterator[str]], stream_generate)
        return stream(request)


def _apply_project_capabilities(project, provider_id, model, provider, profile=None):
    adapter_capabilities = provider.capabilities(model)
    if profile:
        # The reviewed generation adapters implement native JSON-schema wire
        # mappings. The exact-model profile, not the transport, opts in to use.
        adapter_capabilities = replace(adapter_capabilities, structured_output=True)
    if profile and provider_id == "openai":
        # The native OpenAI adapter implements Chat Completions reasoning_effort;
        # the exact-model declaration determines which values this model permits.
        adapter_capabilities = replace(
            adapter_capabilities,
            vision=True,
            reasoning_efforts=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        )
    elif profile and is_openai_compatible_provider(provider_id):
        # Chat Completions has an image content-part mapping. Each compatible
        # model/endpoint must still opt in through its exact-model profile.
        adapter_capabilities = replace(adapter_capabilities, vision=True)
    elif profile and provider_id == "gemini":
        # This is adapter wire support. Vision is usable only after the exact
        # model's project profile independently declares it below.
        adapter_capabilities = replace(adapter_capabilities, vision=True)
        # GenerateContent thinkingLevel is a transport mapping, while exact model
        # support remains the project profile's responsibility.
        adapter_capabilities = replace(
            adapter_capabilities,
            reasoning_efforts=("minimal", "low", "medium", "high"),
        )
    elif profile and provider_id in {"anthropic", "anthropic-compatible"}:
        # These are adapter wire mappings, not model-wide support claims. For
        # compatible endpoints, the exact project profile also opts in to using
        # the mapping because the service may implement only part of Messages.
        adapter_capabilities = replace(
            adapter_capabilities,
            vision=True,
            reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
            reasoning_modes=("adaptive", "manual"),
        )
    elif profile and provider_id == "cohere":
        # Cohere Chat maps the documented thinking toggle. The local ``manual``
        # mode is available only through an exact profile and maps to enabled
        # thinking with a token budget, matching the provider-neutral contract.
        adapter_capabilities = replace(
            adapter_capabilities,
            reasoning_modes=("enabled", "disabled", "manual"),
        )
    effective = (
        _intersect_capabilities(adapter_capabilities, profile.capabilities)
        if profile
        else adapter_capabilities
    )
    if not profile and provider_id in {"openai", "anthropic", "gemini", "cohere"}:
        # These native APIs vary reasoning support by model; without an exact
        # model declaration, no reasoning option is assumed eligible.
        effective = replace(effective, reasoning_efforts=(), reasoning_modes=())
    if profile and hasattr(provider, "capabilities_by_model"):
        model_capabilities = dict(provider.capabilities_by_model)
        model_capabilities[model] = effective
        provider = replace(provider, capabilities_by_model=model_capabilities)
    needs_exact_model_gate = provider_id in {"openai", "anthropic", "gemini", "cohere"}
    constrained = (
        _CapabilityRestrictedProvider(provider, model, effective)
        if profile or needs_exact_model_gate
        else provider
    )
    snapshot: dict[str, object] = {
        "effective": asdict(effective),
        "source": "project-profile" if profile else "adapter-default",
        "assurance": profile.assurance if profile else "UNASSESSED",
        "source_reference": profile.source_reference if profile else None,
    }
    return constrained, effective, snapshot


def _default_provider(
    endpoint: str | None,
    secret_reference: str | None,
    provider_id: str,
    *,
    project=None,
) -> GenerationProvider:
    if not supports_generation(provider_id):
        raise ModelProviderError(f"provider {provider_id!r} does not support text generation")
    store = ReferenceSecretStore(project)
    if provider_id == "anthropic-compatible":
        if not endpoint:
            raise ModelProviderError("Anthropic-compatible provider requires an explicit endpoint")
        if not secret_reference or secret_reference == NO_AUTH_REFERENCE:
            raise ModelProviderError(
                "Anthropic-compatible provider requires a credential reference"
            )
        resolved_reference = provider_auth_reference(provider_id, secret_reference, endpoint)
        if resolved_reference is None:
            raise ModelProviderError(
                "Anthropic-compatible provider requires a credential reference"
            )
        return AnthropicCompatibleProvider(
            endpoint,
            resolved_reference,
            store,
            provider_id=provider_id,
        )
    if secret_reference is None:
        resolved_reference = provider_auth_reference(provider_id, "no-auth", endpoint)
    else:
        resolved_reference = provider_auth_reference(provider_id, secret_reference, endpoint)
    if provider_id == "anthropic":
        if resolved_reference is None:
            raise ModelProviderError("Anthropic requires a credential reference")
        return AnthropicProvider(
            resolved_reference, store, endpoint or _DEFAULT_ENDPOINTS["anthropic"]
        )
    if provider_id == "gemini":
        if resolved_reference is None:
            raise ModelProviderError("Gemini requires a credential reference")
        return GeminiProvider(resolved_reference, store, endpoint or _DEFAULT_ENDPOINTS["gemini"])
    if provider_id == "cohere":
        if resolved_reference is None:
            raise ModelProviderError("Cohere requires a credential reference")
        return CohereProvider(resolved_reference, store, endpoint or _DEFAULT_ENDPOINTS["cohere"])
    if provider_id == "openai":
        if resolved_reference is None:
            raise ModelProviderError("OpenAI requires a credential reference")
        return OpenAIProvider(resolved_reference, store, endpoint or _DEFAULT_ENDPOINTS["openai"])
    # OpenAI-compatible covers OpenAI, DeepSeek, Z.AI, xAI, Mistral,
    # OpenRouter, Together, Groq, Fireworks, Ollama, LM Studio and vLLM when
    # the user explicitly supplies their compatible endpoint.
    if not endpoint:
        raise ModelProviderError("OpenAI-compatible provider requires an explicit endpoint")
    return OpenAICompatibleProvider(endpoint, resolved_reference, store, provider_id)


def _create_provider(
    project,
    provider_factory: ProviderFactory,
    endpoint: str | None,
    secret_reference: str | None,
    provider_id: str,
) -> GenerationProvider:
    """Bind encrypted project secrets without changing injected factory contracts."""
    if provider_factory is _default_provider:
        return _default_provider(endpoint, secret_reference, provider_id, project=project)
    if endpoint is None:
        raise ModelProviderError("provider endpoint is not configured")
    return provider_factory(endpoint, secret_reference, provider_id)


def build_embedding_gateway(
    project,
    provider_id: str,
    credential_alias: str,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> ModelGateway:
    """Build a gateway for the explicitly selected, catalogued embedding transport.

    Catalog validation happens before credential metadata is read. Offline mode
    is checked before validating the secret reference or constructing a provider.
    The selected alias is the sole credential source; there is no ambient
    credential or provider fallback.
    """
    descriptor = next(
        (item for item in supported_providers() if item.provider_id == provider_id), None
    )
    if descriptor is None or not descriptor.embeddings:
        raise ModelProviderError(f"no reviewed embedding adapter for {provider_id!r}")

    credential = project.credential_reference(provider_id, credential_alias)
    endpoint = effective_embedding_endpoint(provider_id, credential["endpoint"])
    if not endpoint:
        raise ModelProviderError("embedding provider requires an explicit compatible endpoint")
    _require_online_endpoint(
        project, endpoint, message="project offline mode blocks remote embeddings"
    )

    auth_reference = provider_auth_reference(
        provider_id, str(credential["secret_reference"]), endpoint
    )
    if embedding_provider is None:
        store = ReferenceSecretStore(project)
        if provider_id == "openai":
            if auth_reference is None:
                raise ModelProviderError("OpenAI embeddings require a credential reference")
            embedding_provider = OpenAIEmbeddingProvider(auth_reference, store, endpoint)
        elif is_openai_compatible_provider(provider_id):
            embedding_provider = OpenAICompatibleEmbeddingProvider(
                endpoint, auth_reference, store, provider_id
            )
        elif provider_id == "cohere-embeddings":
            if auth_reference is None:
                raise ModelProviderError("Cohere embeddings require a credential reference")
            embedding_provider = CohereEmbeddingProvider(auth_reference, store, endpoint)
        elif provider_id == "voyage-embeddings":
            if auth_reference is None:
                raise ModelProviderError("Voyage embeddings require a credential reference")
            embedding_provider = VoyageEmbeddingProvider(auth_reference, store, endpoint)
        elif provider_id == "jina-embeddings":
            if auth_reference is None:
                raise ModelProviderError("Jina embeddings require a credential reference")
            embedding_provider = JinaEmbeddingProvider(auth_reference, store, endpoint)
        elif provider_id == "gemini-embeddings":
            if auth_reference is None:
                raise ModelProviderError("Gemini embeddings require a credential reference")
            embedding_provider = GeminiEmbeddingProvider(auth_reference, store, endpoint)
        else:
            raise ModelProviderError(f"no embedding adapter registered for {provider_id!r}")
    return _OfflineCheckedModelGateway(
        project,
        endpoint,
        "project offline mode blocks remote embeddings",
        credential_reference_fingerprints={
            provider_id: _secret_reference_fingerprint(str(credential["secret_reference"]))
        },
        embedders={
            provider_id: _PolicyCheckedEmbeddingProvider(
                project, endpoint, embedding_provider
            )
        },
    )


def build_reranking_gateway(
    project,
    provider_id: str,
    credential_alias: str,
    *,
    expected_credential_reference_hash: str | None = None,
) -> ModelGateway:
    """Resolve a reranker from an explicit project credential alias.

    Only reviewed reranking transports are constructed here; generation-only
    transports cannot accidentally be routed into the reranking operation.
    """
    if provider_id not in {"jina-reranker", "cohere-reranker"}:
        raise ModelProviderError(f"no reviewed reranking adapter for {provider_id!r}")
    credential = project.credential_reference(provider_id, credential_alias)
    if expected_credential_reference_hash is not None and not _fingerprint_matches_reference(
        credential["secret_reference"], expected_credential_reference_hash
    ):
        raise ModelProviderError(
            "reranking credential reference changed since this analysis snapshot; "
            "refusing implicit credential substitution"
        )
    endpoint = str(
        credential["endpoint"]
        or (
            "https://api.jina.ai/v1" if provider_id == "jina-reranker" else "https://api.cohere.com"
        )
    )
    _require_online_endpoint(
        project, endpoint, message="project offline mode blocks remote reranking"
    )
    if provider_id == "jina-reranker":
        provider = JinaRerankingProvider(
            str(credential["secret_reference"]),
            ReferenceSecretStore(project),
            endpoint=endpoint,
        )
    else:
        provider = CohereRerankingProvider(
            str(credential["secret_reference"]),
            ReferenceSecretStore(project),
            endpoint=endpoint,
        )
    return _OfflineCheckedModelGateway(
        project,
        endpoint,
        "project offline mode blocks remote reranking",
        rerankers={provider_id: provider},
        credential_reference_fingerprints={
            provider_id: _secret_reference_fingerprint(str(credential["secret_reference"]))
        },
    )


def preview_model_preset_request(
    project,
    preset_name: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    provider_factory: ProviderFactory = _default_provider,
) -> PresetRequestPreview:
    """Inspect a request before execution without reading a secret or creating a ledger row."""
    preset = project.model_preset(preset_name)
    parameters = _preset_parameters(preset)
    _validate_execution_parameters(parameters)
    # Match GenerationRequest's validation before constructing the provider. In
    # particular, whitespace-only prompts are invalid even though they are truthy.
    GenerationRequest(
        prompt,
        preset["model"],
        preset["reasoning"],
        parameters=parameters,
        reasoning_mode=preset["reasoning_mode"],
        reasoning_budget_tokens=preset["reasoning_budget_tokens"],
        system_prompt=system_prompt,
    )
    credential = project.credential_reference(preset["provider_id"], preset["credential_alias"])
    endpoint = effective_endpoint(preset["provider_id"], credential["endpoint"])
    generation_supported = supports_generation(preset["provider_id"])
    local_endpoint = is_loopback_endpoint(endpoint)
    missing_endpoint = endpoint is None
    blocked = (
        not generation_supported
        or missing_endpoint
        or (project.offline_mode and not local_endpoint)
    )
    effective_capabilities = None
    unsupported_parameters: tuple[str, ...] = ()
    unsupported_reasoning = False
    unsupported_reasoning_mode = False
    unsupported_system_prompt = False
    reasoning_configuration_invalid = False
    if endpoint is not None and generation_supported:
        auth_reference = provider_auth_reference(
            preset["provider_id"], credential["secret_reference"], endpoint
        )
        provider = _create_provider(
            project, provider_factory, endpoint, auth_reference, preset["provider_id"]
        )
        profile = project.model_capability_profile(preset["provider_id"], preset["model"])
        _, capability_set, _ = _apply_project_capabilities(
            project, preset["provider_id"], preset["model"], provider, profile
        )
        effective_capabilities = asdict(capability_set)
        unsupported_system_prompt = system_prompt is not None and not capability_set.system_prompt
        unsupported_parameters = tuple(
            sorted(set(parameters) - set(capability_set.generation_parameters))
        )
        unsupported_reasoning = bool(
            preset["reasoning"] and preset["reasoning"] not in capability_set.reasoning_efforts
        )
        unsupported_reasoning_mode = bool(
            (
                preset["reasoning_mode"]
                and preset["reasoning_mode"] not in capability_set.reasoning_modes
            )
            or (
                preset["provider_id"] == "anthropic"
                and preset["reasoning"]
                and not preset["reasoning_mode"]
            )
        )
        if preset["reasoning_mode"] == "manual" and preset["provider_id"] == "cohere":
            try:
                validate_cohere_manual_budget(preset["reasoning_budget_tokens"], parameters)
            except ModelProviderError:
                reasoning_configuration_invalid = True
            else:
                reasoning_configuration_invalid = False
        else:
            max_tokens = parameters.get("max_tokens", 4096)
            reasoning_configuration_invalid = bool(
                preset["reasoning_mode"] == "manual"
                and (
                    preset["reasoning_budget_tokens"] is None
                    or preset["reasoning_budget_tokens"] < 1024
                    or not isinstance(max_tokens, int)
                    or isinstance(max_tokens, bool)
                    or preset["reasoning_budget_tokens"] >= max_tokens
                )
            )
        blocked = (
            blocked
            or unsupported_system_prompt
            or bool(unsupported_parameters)
            or unsupported_reasoning
            or unsupported_reasoning_mode
            or reasoning_configuration_invalid
        )
    block_reason = (
        "provider does not support text generation"
        if not generation_supported
        else "provider requires an explicit compatible endpoint"
        if missing_endpoint
        else "provider does not declare system-prompt support"
        if unsupported_system_prompt
        else "project offline mode blocks this non-loopback endpoint"
        if project.offline_mode and not local_endpoint
        else "preset contains parameters unsupported by the declared model capabilities"
        if unsupported_parameters
        else "reasoning effort is unsupported by the declared model capabilities"
        if unsupported_reasoning
        else "Anthropic reasoning requires a declared model-specific mode"
        if preset["provider_id"] == "anthropic"
        and preset["reasoning"]
        and not preset["reasoning_mode"]
        else "reasoning mode is unsupported by the declared model capabilities"
        if unsupported_reasoning_mode
        else (
            "Cohere manual reasoning requires a valid budget with at least 1000 response tokens"
            if preset["provider_id"] == "cohere"
            else "manual reasoning requires a supported budget below max_tokens"
        )
        if endpoint is not None and reasoning_configuration_invalid
        else None
    )
    return PresetRequestPreview(
        provider_id=preset["provider_id"],
        credential_alias=preset["credential_alias"],
        endpoint=endpoint,
        model=preset["model"],
        reasoning=preset["reasoning"],
        reasoning_mode=preset["reasoning_mode"],
        reasoning_budget_tokens=preset["reasoning_budget_tokens"],
        parameters=parameters,
        effective_capabilities=effective_capabilities,
        unsupported_parameters=unsupported_parameters,
        unsupported_reasoning=unsupported_reasoning,
        unsupported_reasoning_mode=unsupported_reasoning_mode,
        unsupported_system_prompt=unsupported_system_prompt,
        prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_characters=len(prompt),
        system_prompt_hash=(
            hashlib.sha256(system_prompt.encode()).hexdigest()
            if system_prompt is not None
            else None
        ),
        system_prompt_characters=len(system_prompt) if system_prompt is not None else 0,
        offline_mode=project.offline_mode,
        endpoint_is_loopback=local_endpoint,
        allowed=not blocked,
        block_reason=block_reason,
    )


def execute_model_preset(
    project,
    preset_name: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    image: GenerationImage | None = None,
    response_schema_name: str | None = None,
    response_schema: dict[str, object] | None = None,
    analysis_run_id: str | None = None,
    skill_run_id: str | None = None,
    provider_factory: ProviderFactory = _default_provider,
    preset_snapshot: dict[str, object] | None = None,
    reasoning_override: str | None = None,
    reasoning_mode_override: str | None = None,
) -> PresetExecutionResult:
    """Execute only the provider/model explicitly selected in a saved preset.

    Prompts and completion text are intentionally not stored in the model-execution
    ledger; hashes preserve audit linkage without duplicating research content.
    """
    project.validate_model_execution_skill_run(analysis_run_id, skill_run_id)
    resolved = (
        snapshot_model_preset(project, preset_name) if preset_snapshot is None else preset_snapshot
    )
    if reasoning_override is not None:
        if not isinstance(reasoning_override, str) or len(reasoning_override) > 64:
            raise ModelProviderError("reasoning override is invalid")
        resolved = dict(resolved)
        resolved["reasoning"] = reasoning_override or None
    if reasoning_mode_override is not None:
        if not isinstance(reasoning_mode_override, str) or len(reasoning_mode_override) > 32:
            raise ModelProviderError("reasoning mode override is invalid")
        if not isinstance(resolved, dict):
            resolved = dict(resolved)
        resolved["reasoning_mode"] = reasoning_mode_override or None
    if resolved.get("preset_name") != preset_name:
        raise ModelProviderError("analysis run preset snapshot does not match the selected preset")
    provider_id = resolved.get("provider_id")
    credential_alias = resolved.get("credential_alias")
    model = resolved.get("model")
    reasoning = resolved.get("reasoning")
    reasoning_mode = resolved.get("reasoning_mode")
    reasoning_budget_tokens = resolved.get("reasoning_budget_tokens")
    parameters = resolved.get("parameters")
    endpoint = resolved.get("endpoint")
    if not isinstance(provider_id, str) or not provider_id:
        raise ModelProviderError("analysis run preset snapshot has an invalid provider")
    if not supports_generation(provider_id):
        raise ModelProviderError(f"provider {provider_id!r} does not support text generation")
    if not isinstance(credential_alias, str) or not credential_alias:
        raise ModelProviderError("analysis run preset snapshot has an invalid credential alias")
    if not isinstance(model, str) or not model:
        raise ModelProviderError("analysis run preset snapshot is incomplete")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ModelProviderError("analysis run reasoning snapshot is invalid")
    if reasoning_mode is not None and not isinstance(reasoning_mode, str):
        raise ModelProviderError("analysis run reasoning mode snapshot is invalid")
    if reasoning_budget_tokens is not None and (
        not isinstance(reasoning_budget_tokens, int) or isinstance(reasoning_budget_tokens, bool)
    ):
        raise ModelProviderError("analysis run reasoning budget snapshot is invalid")
    if not isinstance(parameters, dict):
        raise ModelProviderError("analysis run preset parameter snapshot is invalid")
    _validate_execution_parameters(parameters)
    if provider_id == "cohere" and reasoning_mode == "manual":
        validate_cohere_manual_budget(reasoning_budget_tokens, parameters)
    price_snapshot = resolved.get("model_price")
    model_price = ModelPrice.from_snapshot(price_snapshot) if price_snapshot is not None else None
    if model_price is not None and (
        model_price.provider_id != provider_id or model_price.model != model
    ):
        raise ModelProviderError("analysis run model price snapshot does not match its preset")
    if endpoint is None:
        raise ModelProviderError("provider requires an explicit compatible endpoint")
    if not isinstance(endpoint, str):
        raise ModelProviderError("analysis run provider endpoint snapshot is invalid")
    _require_online_endpoint(
        project, endpoint, message="project offline mode blocks remote model execution"
    )
    profile = _snapshot_to_capability_profile(
        provider_id, model, resolved.get("capability_profile")
    )
    _require_image_capability(provider_id, profile, image)
    generation_request = GenerationRequest(
        prompt,
        model,
        reasoning,
        parameters=parameters,
        reasoning_mode=reasoning_mode,
        reasoning_budget_tokens=reasoning_budget_tokens,
        system_prompt=system_prompt,
        response_schema_name=response_schema_name,
        response_schema=response_schema,
        image=image,
    )
    credential = _credential_for_preset_snapshot(
        project,
        provider_id,
        credential_alias,
        resolved.get("credential_reference_hash"),
    )
    request_payload: dict[str, object] = {"user_prompt": prompt}
    if system_prompt is not None:
        request_payload["system_prompt"] = system_prompt
    if response_schema is not None:
        request_payload["response_schema_name"] = response_schema_name or ""
        request_payload["response_schema"] = response_schema
    if generation_request.image is not None:
        request_payload["image_input"] = generation_request.image.metadata()
    request_material = (
        prompt.encode("utf-8")
        if system_prompt is None and response_schema is None and generation_request.image is None
        else json.dumps(
            request_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    request_hash = hashlib.sha256(request_material).hexdigest()
    started = time.perf_counter()
    image_metadata = generation_request.image.metadata() if generation_request.image else None
    capability_snapshot: dict[str, object] = (
        {"request_image": image_metadata} if image_metadata is not None else {}
    )
    try:
        _require_online_endpoint(
            project, endpoint, message="project offline mode blocks remote model execution"
        )
        auth_reference = provider_auth_reference(
            provider_id, str(credential["secret_reference"]), endpoint
        )
        provider = _create_provider(project, provider_factory, endpoint, auth_reference, provider_id)
        provider, _, capability_snapshot = _apply_project_capabilities(
            project, provider_id, model, provider, profile
        )
        if image_metadata is not None:
            capability_snapshot["request_image"] = image_metadata
        provider = _OfflineCheckedGenerationProvider(provider, project, endpoint)
        response = ModelGateway({provider_id: provider}).generate(
            provider_id,
            generation_request,
        )
        if response.provider != provider_id or response.model != model:
            raise ModelResponseError(
                "provider response identity does not match the selected preset"
            )
    except Exception as exc:
        project.record_model_execution(
            provider_id,
            credential_alias,
            model,
            reasoning,
            request_hash,
            None,
            type(exc).__name__,
            analysis_run_id,
            latency_ms=round((time.perf_counter() - started) * 1000),
            parameters=parameters,
            capabilities=capability_snapshot,
            reasoning_mode=reasoning_mode,
            reasoning_budget_tokens=reasoning_budget_tokens,
            model_price=model_price,
            endpoint=endpoint,
            skill_run_id=skill_run_id,
        )
        raise
    execution_id = project.record_model_execution(
        provider_id,
        credential_alias,
        model,
        reasoning,
        request_hash,
        hashlib.sha256(response.text.encode()).hexdigest(),
        None,
        analysis_run_id,
        input_tokens=response.usage.input_tokens if response.usage else None,
        output_tokens=response.usage.output_tokens if response.usage else None,
        reasoning_tokens=response.usage.reasoning_tokens if response.usage else None,
        latency_ms=round((time.perf_counter() - started) * 1000),
        parameters=parameters,
        capabilities=capability_snapshot,
        reasoning_mode=reasoning_mode,
        reasoning_budget_tokens=reasoning_budget_tokens,
        model_price=model_price,
        endpoint=endpoint,
        skill_run_id=skill_run_id,
    )
    return PresetExecutionResult(execution_id, response)


def execute_model_preset_stream(
    project,
    preset_name: str,
    prompt: str,
    *,
    system_prompt: str | None = None,
    image: GenerationImage | None = None,
    reasoning_override: str | None = None,
    reasoning_mode_override: str | None = None,
    provider_factory: ProviderFactory = _default_provider,
    preset_snapshot: dict[str, object] | None = None,
) -> Generator[str, None, None]:
    """Stream text from the explicitly selected preset without storing prompt/output.

    Streaming requires an explicit capability declaration for the selected
    model, and the provider adapter can only narrow that declaration. The
    execution ledger stores only request/response hashes or error type.
    """

    if image is not None:
        raise ModelProviderError("image input is not supported with streaming")

    def stream_chunks() -> Generator[str, None, None]:
        resolved = (
            snapshot_model_preset(project, preset_name)
            if preset_snapshot is None
            else preset_snapshot
        )
        if reasoning_override is not None or reasoning_mode_override is not None:
            resolved = dict(resolved)
            if reasoning_override is not None:
                if not isinstance(reasoning_override, str) or len(reasoning_override) > 64:
                    raise ModelProviderError("reasoning override is invalid")
                resolved["reasoning"] = reasoning_override or None
            if reasoning_mode_override is not None:
                if not isinstance(reasoning_mode_override, str) or len(reasoning_mode_override) > 32:
                    raise ModelProviderError("reasoning mode override is invalid")
                resolved["reasoning_mode"] = reasoning_mode_override or None
        if resolved.get("preset_name") != preset_name:
            raise ModelProviderError(
                "analysis run preset snapshot does not match the selected preset"
            )
        provider_id = resolved.get("provider_id")
        credential_alias = resolved.get("credential_alias")
        model = resolved.get("model")
        reasoning = resolved.get("reasoning")
        reasoning_mode = resolved.get("reasoning_mode")
        reasoning_budget_tokens = resolved.get("reasoning_budget_tokens")
        parameters = resolved.get("parameters")
        endpoint = resolved.get("endpoint")
        if not isinstance(provider_id, str) or not provider_id:
            raise ModelProviderError("analysis run preset snapshot has an invalid provider")
        if not supports_generation(provider_id):
            raise ModelProviderError(f"provider {provider_id!r} does not support text generation")
        if not isinstance(credential_alias, str) or not credential_alias:
            raise ModelProviderError("analysis run preset snapshot has an invalid credential alias")
        if not isinstance(model, str) or not model:
            raise ModelProviderError("analysis run preset snapshot is incomplete")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ModelProviderError("analysis run reasoning snapshot is invalid")
        if reasoning_mode is not None and not isinstance(reasoning_mode, str):
            raise ModelProviderError("analysis run reasoning mode snapshot is invalid")
        if reasoning_budget_tokens is not None and (
            not isinstance(reasoning_budget_tokens, int)
            or isinstance(reasoning_budget_tokens, bool)
        ):
            raise ModelProviderError("analysis run reasoning budget snapshot is invalid")
        if not isinstance(parameters, dict):
            raise ModelProviderError("analysis run preset parameter snapshot is invalid")
        _validate_execution_parameters(parameters)
        if provider_id == "cohere" and reasoning_mode == "manual":
            validate_cohere_manual_budget(reasoning_budget_tokens, parameters)
        if endpoint is None:
            raise ModelProviderError("provider requires an explicit compatible endpoint")
        if not isinstance(endpoint, str):
            raise ModelProviderError("analysis run provider endpoint snapshot is invalid")
        _require_online_endpoint(
            project, endpoint, message="project offline mode blocks remote model execution"
        )
        profile = _snapshot_to_capability_profile(
            provider_id, model, resolved.get("capability_profile")
        )
        if profile is None or not profile.capabilities.streaming:
            raise ModelProviderError(
                "streaming requires an exact-model capability profile declaring support"
            )
        credential = _credential_for_preset_snapshot(
            project,
            provider_id,
            credential_alias,
            resolved.get("credential_reference_hash"),
        )
        request = GenerationRequest(
            prompt,
            model,
            reasoning,
            parameters=parameters,
            reasoning_mode=reasoning_mode,
            reasoning_budget_tokens=reasoning_budget_tokens,
            system_prompt=system_prompt,
        )
        request_payload: dict[str, object] = {"user_prompt": prompt}
        if system_prompt is not None:
            request_payload["system_prompt"] = system_prompt
        request_material = (
            prompt.encode("utf-8")
            if system_prompt is None
            else json.dumps(
                request_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        request_hash = hashlib.sha256(request_material).hexdigest()
        price_snapshot = resolved.get("model_price")
        model_price = (
            ModelPrice.from_snapshot(price_snapshot) if price_snapshot is not None else None
        )
        if model_price is not None and (
            model_price.provider_id != provider_id or model_price.model != model
        ):
            raise ModelProviderError("analysis run model price snapshot does not match its preset")

        started = time.perf_counter()
        capability_snapshot: dict[str, object] = {}
        ledger_recorded = False
        provider_stream: Iterator[str] | None = None

        def record(response_hash: str | None, error_type: str | None) -> None:
            nonlocal ledger_recorded
            ledger_recorded = True
            project.record_model_execution(
                provider_id,
                credential_alias,
                model,
                reasoning,
                request_hash,
                response_hash,
                error_type,
                latency_ms=round((time.perf_counter() - started) * 1000),
                parameters=parameters,
                capabilities=capability_snapshot,
                reasoning_mode=reasoning_mode,
                reasoning_budget_tokens=reasoning_budget_tokens,
                model_price=model_price,
                endpoint=endpoint,
            )

        try:
            _require_online_endpoint(
                project, endpoint, message="project offline mode blocks remote model execution"
            )
            auth_reference = provider_auth_reference(
                provider_id, str(credential["secret_reference"]), endpoint
            )
            provider = _create_provider(
                project, provider_factory, endpoint, auth_reference, provider_id
            )
            provider, effective_capabilities, capability_snapshot = _apply_project_capabilities(
                project, provider_id, model, provider, profile
            )
            if not effective_capabilities.streaming:
                raise ModelProviderError(
                    "selected transport does not support streaming for this exact model"
                )
            checked_provider = _OfflineCheckedGenerationProvider(provider, project, endpoint)
            provider_stream = ModelGateway({provider_id: checked_provider}).stream(
                provider_id, request
            )
            response_digest = hashlib.sha256()
            try:
                for chunk in provider_stream:
                    if not isinstance(chunk, str):
                        raise ModelResponseError("provider returned a malformed streaming chunk")
                    response_digest.update(chunk.encode("utf-8"))
                    yield chunk
            except GeneratorExit:
                record(None, "Cancelled")
                raise
            except Exception as exc:
                record(None, type(exc).__name__)
                raise
            else:
                record(response_digest.hexdigest(), None)
        except Exception as exc:
            if not ledger_recorded:
                record(None, type(exc).__name__)
            raise
        finally:
            if provider_stream is not None:
                close = getattr(provider_stream, "close", None)
                if callable(close):
                    # Closing is best-effort and must not mask the transport/write failure.
                    with suppress(Exception):
                        close()

    return stream_chunks()
