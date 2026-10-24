"""Reviewed provider transports without model-specific capability claims.

Provider IDs must be registered here before generation can use them. Compatible
IDs share an implemented transport, but unknown IDs are not implicitly treated
as OpenAI-compatible. Exact model capabilities still require explicit review.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderDescriptor:
    provider_id: str
    transport: str
    endpoint_policy: str
    generation: bool
    embeddings: bool
    native_reasoning_mapping: bool
    note: str
    reranking: bool = False


_NATIVE = (
    ProviderDescriptor(
        "openai",
        "openai-chat-completions",
        "optional_override",
        True,
        True,
        True,
        "Direct Chat Completions adapter with OpenAI's official default endpoint; reasoning remains model-specific.",
    ),
    ProviderDescriptor(
        "anthropic",
        "anthropic-messages",
        "optional_override",
        True,
        False,
        True,
        "Native Messages adapter; exact model profiles must declare supported thinking mode and effort.",
    ),
    ProviderDescriptor(
        "gemini",
        "gemini-generate-content",
        "optional_override",
        True,
        False,
        True,
        "Native generateContent adapter; model capabilities remain explicit.",
    ),
    ProviderDescriptor(
        "cohere",
        "cohere-chat-v2",
        "optional_override",
        True,
        False,
        True,
        "Native Cohere Chat v2 adapter; exact-model profiles gate structured output and reasoning modes.",
    ),
    ProviderDescriptor(
        "gemini-embeddings",
        "gemini-batch-embed-content-v1beta",
        "optional_override",
        False,
        True,
        False,
        "Native Gemini text embeddings; known model families use explicit task configuration or documented retrieval prefixes.",
    ),
    ProviderDescriptor(
        "cohere-embeddings",
        "cohere-embed-v2",
        "optional_override",
        False,
        True,
        False,
        "Native Cohere Embed v2 transport for semantic search only; uses distinct search_document and search_query input types.",
    ),
    ProviderDescriptor(
        "cohere-reranker",
        "cohere-rerank-v2",
        "optional_override",
        False,
        False,
        False,
        "Native Cohere Rerank v2 transport; ranks at most 1000 explicitly supplied candidates and returns relevance scores, not probabilities.",
        reranking=True,
    ),
    ProviderDescriptor(
        "voyage-embeddings",
        "voyage-text-embeddings-v1",
        "optional_override",
        False,
        True,
        False,
        "Native Voyage text embeddings for semantic search only; uses distinct document/query input types and rejects silent truncation.",
    ),
    ProviderDescriptor(
        "jina-embeddings",
        "jina-embeddings-api-v1",
        "optional_override",
        False,
        True,
        False,
        "Native Jina dense embeddings; uses retrieval-side task adapters for documented v3-v5 families and records symmetric mode otherwise.",
    ),
    ProviderDescriptor(
        "jina-reranker",
        "jina-rerank-v1",
        "optional_override",
        False,
        False,
        False,
        "Jina reranking only; ranks an explicitly supplied candidate set and returns indices/scores without retrieving corpus content.",
        reranking=True,
    ),
)

_ANTHROPIC_COMPATIBLE = (
    ProviderDescriptor(
        "anthropic-compatible",
        "anthropic-messages-compatible",
        "required",
        True,
        False,
        False,
        "Requires an explicit Anthropic Messages-compatible endpoint and credential; model capabilities are not inferred.",
    ),
)

_COMPATIBLE_IDS = (
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
)


def is_openai_compatible_provider(provider_id: str) -> bool:
    """Whether an ID uses the reviewed OpenAI-compatible transport."""
    return provider_id in _COMPATIBLE_IDS


_COMPATIBLE = tuple(
    ProviderDescriptor(
        provider_id,
        "openai-compatible",
        "required",
        True,
        True,
        False,
        "Requires a researcher-supplied compatible endpoint; model capabilities and native reasoning "
        "options are never inferred from this catalog. Authentication may be omitted only for an "
        "explicit loopback endpoint.",
    )
    for provider_id in _COMPATIBLE_IDS
)


def supported_providers() -> tuple[ProviderDescriptor, ...]:
    """Return reviewed transport choices in deterministic discovery order."""
    return _NATIVE + _ANTHROPIC_COMPATIBLE + _COMPATIBLE


def supports_generation(provider_id: str) -> bool:
    """Whether a registered provider has a generation transport."""
    descriptor = next(
        (item for item in supported_providers() if item.provider_id == provider_id), None
    )
    return descriptor is not None and descriptor.generation


def is_reviewed_reranking_provider(provider_id: str) -> bool:
    """Whether the provider catalog exposes a reviewed reranking transport."""
    return any(item.provider_id == provider_id and item.reranking for item in _NATIVE)
