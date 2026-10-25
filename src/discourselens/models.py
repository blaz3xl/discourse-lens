"""Provider-neutral generation contracts; no network provider is implemented here."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Generator, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import MAX_EMAX, MIN_EMIN, Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import ClassVar, Protocol, cast, runtime_checkable
from urllib.parse import urlsplit

from .errors import ModelProviderError, ModelResponseError
from .provider_catalog import is_openai_compatible_provider
from .secrets import validate_credential_alias


@dataclass(frozen=True)
class ModelCapabilities:
    text: bool = True
    system_prompt: bool = True
    vision: bool = False
    structured_output: bool = False
    streaming: bool = False
    embeddings: bool = False
    reasoning_efforts: tuple[str, ...] = ()
    reasoning_modes: tuple[str, ...] = ()
    generation_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "text",
            "system_prompt",
            "vision",
            "structured_output",
            "streaming",
            "embeddings",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ModelProviderError(f"model capability {field_name} must be a boolean")

        for field_name in (
            "reasoning_efforts",
            "reasoning_modes",
            "generation_parameters",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, (tuple, list)):
                raise ModelProviderError(
                    f"model capability {field_name} must contain non-empty text values"
                )
            snapshot = tuple(values)
            if any(not isinstance(value, str) or not value.strip() for value in snapshot):
                raise ModelProviderError(
                    f"model capability {field_name} must contain non-empty text values"
                )
            object.__setattr__(self, field_name, snapshot)


@dataclass(frozen=True)
class ModelCapabilityProfile:
    """Exact-model capability declaration; unverified declarations stay labeled as such."""

    provider_id: str
    model: str
    capabilities: ModelCapabilities
    assurance: str = "UNASSESSED"
    source_reference: str | None = None


@dataclass(frozen=True)
class ReasoningConfig:
    """Provider-neutral intent, mapped only by adapters that declare support."""

    effort: str | None = None
    mode: str | None = None
    budget_tokens: int | None = None


MAX_GENERATION_IMAGE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, repr=False)
class GenerationImage:
    """One bounded local image, represented without retaining its source path."""

    mime_type: str
    data: bytes = field(repr=False)

    _SIGNATURES: ClassVar[dict[str, Callable[[bytes], bool]]] = {
        "image/png": lambda data: data.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": lambda data: data.startswith(b"\xff\xd8\xff"),
        "image/webp": lambda data: (
            len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP"
        ),
    }
    _SUFFIX_MIME_TYPES: ClassVar[dict[str, str]] = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    def __post_init__(self) -> None:
        if not isinstance(self.mime_type, str) or self.mime_type not in self._SIGNATURES:
            raise ModelProviderError("image MIME type is unsupported")
        if not isinstance(self.data, bytes) or not self.data:
            raise ModelProviderError("image input must contain non-empty bytes")
        if len(self.data) > MAX_GENERATION_IMAGE_BYTES:
            raise ModelProviderError(
                f"image input exceeds the {MAX_GENERATION_IMAGE_BYTES}-byte limit"
            )
        if not self._SIGNATURES[self.mime_type](self.data):
            raise ModelProviderError("image MIME type does not match its file signature")

    @classmethod
    def from_file(cls, path: str | Path) -> GenerationImage:
        """Read one local image without following a URL or loading an unbounded file."""
        raw_path = str(path)
        parsed = urlsplit(raw_path)
        is_windows_drive = bool(re.match(r"^[A-Za-z]:[\\/]", raw_path))
        if parsed.scheme and not is_windows_drive:
            raise ModelProviderError("image input must be a local file path, not a URL")
        if raw_path.startswith(("//", "\\\\")):
            raise ModelProviderError("image input must be a local file path")
        candidate = Path(path)
        mime_type = cls._SUFFIX_MIME_TYPES.get(candidate.suffix.casefold())
        if mime_type is None:
            raise ModelProviderError("image file extension is unsupported")
        try:
            with candidate.open("rb") as source:
                data = source.read(MAX_GENERATION_IMAGE_BYTES + 1)
        except OSError:
            raise ModelProviderError("could not read local image file") from None
        if len(data) > MAX_GENERATION_IMAGE_BYTES:
            raise ModelProviderError(
                f"image input exceeds the {MAX_GENERATION_IMAGE_BYTES}-byte limit"
            )
        return cls(mime_type=mime_type, data=data)

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def metadata(self) -> dict[str, str | int]:
        """Return safe, reproducible metadata without the file path or image bytes."""
        return {
            "sha256": self.sha256,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
        }

    def __repr__(self) -> str:
        return (
            f"GenerationImage(mime_type={self.mime_type!r}, size_bytes={self.size_bytes}, "
            f"sha256={self.sha256!r})"
        )


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    model: str
    reasoning: str | None = None
    parameters: dict[str, object] | None = None
    reasoning_mode: str | None = None
    reasoning_budget_tokens: int | None = None
    system_prompt: str | None = None
    response_schema_name: str | None = None
    response_schema: dict[str, object] | None = None
    image: GenerationImage | None = None
    on_usage: Callable[[TokenUsage], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.on_usage is not None and not callable(self.on_usage):
            raise ModelProviderError("generation usage callback must be callable")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ModelProviderError("generation prompt must be non-empty text")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ModelProviderError("generation model must be non-empty text")
        for field_name, value in (
            ("reasoning effort", self.reasoning),
            ("reasoning mode", self.reasoning_mode),
            ("system prompt", self.system_prompt),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ModelProviderError(f"{field_name} must be non-empty text when provided")
        if self.parameters is not None and not isinstance(self.parameters, dict):
            raise ModelProviderError("generation parameters must be a JSON object")
        if self.parameters is not None and any(
            not isinstance(key, str) or not key.strip() for key in self.parameters
        ):
            raise ModelProviderError("generation parameter names must be non-empty text")
        if self.reasoning_budget_tokens is not None and (
            isinstance(self.reasoning_budget_tokens, bool)
            or not isinstance(self.reasoning_budget_tokens, int)
            or self.reasoning_budget_tokens < 1
        ):
            raise ModelProviderError("reasoning budget must be a positive integer")
        if self.response_schema_name is not None and not isinstance(self.response_schema_name, str):
            raise ModelProviderError("structured output schema name is invalid")
        if self.image is not None and not isinstance(self.image, GenerationImage):
            raise ModelProviderError("generation image must be a validated GenerationImage")
        if (self.response_schema_name is None) != (self.response_schema is None):
            raise ModelProviderError("structured output requires both a schema name and schema")
        if self.response_schema is not None:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.response_schema_name or ""):
                raise ModelProviderError("structured output schema name is invalid")
            if not isinstance(self.response_schema, dict) or not self.response_schema:
                raise ModelProviderError("structured output schema must be a non-empty JSON object")
            try:
                serialized_schema = json.dumps(
                    self.response_schema,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise ModelProviderError(
                    "structured output schema must contain JSON values"
                ) from exc
            if len(serialized_schema.encode("utf-8")) > 65_536:
                raise ModelProviderError("structured output schema exceeds the 64 KiB limit")
            object.__setattr__(self, "response_schema", json.loads(serialized_schema))


@dataclass(frozen=True)
class TokenUsage:
    """Provider-reported counts; output is total generated tokens.

    ``reasoning_tokens`` is a subset breakdown of ``output_tokens`` when the
    provider reports a consistent breakdown. Missing values are never estimated
    as facts.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("input_tokens", "output_tokens", "reasoning_tokens"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ModelResponseError(
                    f"provider token usage {field_name} must be a non-negative integer"
                )


def _price_decimal(value: Decimal | str | int, field_name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, str, int)):
        raise ModelProviderError(f"model price {field_name} must be a decimal value")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ModelProviderError(f"model price {field_name} must be a decimal value") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ModelProviderError(f"model price {field_name} must be finite and non-negative")
    return parsed


def decimal_text(value: Decimal) -> str:
    """Serialize a finite decimal without exponent notation or insignificant zeroes."""
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


@dataclass(frozen=True)
class ModelPrice:
    """User-supplied model rates, expressed per million provider-reported tokens.

    Input and total output counts are charged at their matching rates. If a
    reasoning rate is supplied, the known reasoning subset uses that rate and
    the remaining output tokens use the output rate.
    """

    provider_id: str
    model: str
    input_per_million_tokens: Decimal | str | int
    output_per_million_tokens: Decimal | str | int
    currency: str
    source_reference: str
    rate_version: str
    rate_date: str
    reasoning_per_million_tokens: Decimal | str | int | None = None

    def __post_init__(self) -> None:
        for field_name in ("provider_id", "model", "source_reference", "rate_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ModelProviderError(f"model price {field_name} must be non-empty text")
            object.__setattr__(self, field_name, value.strip())

        if _contains_price_source_credential(self.source_reference):
            raise ModelProviderError(
                "model price source_reference must not contain secret-like values"
            )

        if not isinstance(self.currency, str) or not re.fullmatch(
            r"[A-Za-z]{3}", self.currency.strip()
        ):
            raise ModelProviderError("model price currency must be a three-letter code")
        object.__setattr__(self, "currency", self.currency.strip().upper())

        if not isinstance(self.rate_date, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", self.rate_date
        ):
            raise ModelProviderError("model price rate_date must use YYYY-MM-DD")
        try:
            parsed_date = date.fromisoformat(self.rate_date)
        except ValueError as exc:
            raise ModelProviderError("model price rate_date must be a valid date") from exc
        if parsed_date.isoformat() != self.rate_date:
            raise ModelProviderError("model price rate_date must use YYYY-MM-DD")

        object.__setattr__(
            self,
            "input_per_million_tokens",
            _price_decimal(self.input_per_million_tokens, "input_per_million_tokens"),
        )
        object.__setattr__(
            self,
            "output_per_million_tokens",
            _price_decimal(self.output_per_million_tokens, "output_per_million_tokens"),
        )
        if self.reasoning_per_million_tokens is not None:
            object.__setattr__(
                self,
                "reasoning_per_million_tokens",
                _price_decimal(self.reasoning_per_million_tokens, "reasoning_per_million_tokens"),
            )

    def to_snapshot(self) -> dict[str, str | None]:
        input_rate = cast(Decimal, self.input_per_million_tokens)
        output_rate = cast(Decimal, self.output_per_million_tokens)
        reasoning_rate = (
            cast(Decimal, self.reasoning_per_million_tokens)
            if self.reasoning_per_million_tokens is not None
            else None
        )
        return {
            "rate_unit": "per_million_tokens",
            "provider_id": self.provider_id,
            "model": self.model,
            "input_per_million_tokens": decimal_text(input_rate),
            "output_per_million_tokens": decimal_text(output_rate),
            "reasoning_per_million_tokens": (
                decimal_text(reasoning_rate) if reasoning_rate is not None else None
            ),
            "currency": self.currency,
            "source_reference": self.source_reference,
            "rate_version": self.rate_version,
            "rate_date": self.rate_date,
        }

    @classmethod
    def from_snapshot(cls, value: object) -> ModelPrice:
        if not isinstance(value, Mapping):
            raise ModelProviderError("analysis run model price snapshot is invalid")
        if value.get("rate_unit") != "per_million_tokens":
            raise ModelProviderError("analysis run model price snapshot has an unknown rate unit")

        required_text_fields = (
            "provider_id",
            "model",
            "currency",
            "source_reference",
            "rate_version",
            "rate_date",
        )
        if any(not isinstance(value.get(field), str) for field in required_text_fields):
            raise ModelProviderError("analysis run model price snapshot is invalid")
        input_rate = value.get("input_per_million_tokens")
        output_rate = value.get("output_per_million_tokens")
        reasoning_rate = value.get("reasoning_per_million_tokens")
        if not isinstance(input_rate, str) or not isinstance(output_rate, str):
            raise ModelProviderError("analysis run model price snapshot is invalid")
        if reasoning_rate is not None and not isinstance(reasoning_rate, str):
            raise ModelProviderError("analysis run model price snapshot is invalid")
        return cls(
            provider_id=value["provider_id"],
            model=value["model"],
            input_per_million_tokens=input_rate,
            output_per_million_tokens=output_rate,
            reasoning_per_million_tokens=reasoning_rate,
            currency=value["currency"],
            source_reference=value["source_reference"],
            rate_version=value["rate_version"],
            rate_date=value["rate_date"],
        )


def estimate_model_cost(price: ModelPrice | None, usage: TokenUsage | None) -> Decimal | None:
    """Estimate cost only when both provider-reported input and output usage exist."""
    if price is None or usage is None or usage.input_tokens is None or usage.output_tokens is None:
        return None
    output_rate = cast(Decimal, price.output_per_million_tokens)
    reasoning_rate = (
        cast(Decimal, price.reasoning_per_million_tokens)
        if price.reasoning_per_million_tokens is not None
        else None
    )
    terms = [
        (usage.input_tokens, cast(Decimal, price.input_per_million_tokens)),
    ]
    if reasoning_rate is not None:
        if usage.reasoning_tokens is None or usage.reasoning_tokens > usage.output_tokens:
            return None
        non_reasoning_tokens = usage.output_tokens - usage.reasoning_tokens
        terms.extend(
            (
                (non_reasoning_tokens, output_rate),
                (usage.reasoning_tokens, reasoning_rate),
            )
        )
    elif usage.reasoning_tokens is not None and usage.reasoning_tokens > usage.output_tokens:
        # A direct caller may construct usage without a provider adapter. Do not
        # estimate from a breakdown that contradicts the reported output total.
        return None
    else:
        terms.append((usage.output_tokens, output_rate))

    precision = _exact_cost_precision(terms)
    with localcontext() as context:
        context.prec = precision
        context.Emax = MAX_EMAX
        context.Emin = MIN_EMIN
        total = sum(
            (Decimal(token_count) * rate for token_count, rate in terms),
            Decimal(0),
        )
        return total / Decimal(1_000_000)


def _exact_cost_precision(terms: list[tuple[int, Decimal]]) -> int:
    """Return enough significant digits to multiply and add these terms exactly."""
    active_terms: list[tuple[int, int]] = []
    for token_count, rate in terms:
        if token_count == 0 or rate.is_zero():
            continue
        rate_tuple = rate.as_tuple()
        product_digits = len(rate_tuple.digits) + len(Decimal(token_count).as_tuple().digits)
        # ModelPrice guarantees finite rates, so the tuple exponent is an int.
        exponent = cast(int, rate_tuple.exponent)
        product_adjusted_exponent = exponent + product_digits - 1
        active_terms.append((exponent, product_adjusted_exponent))

    if not active_terms:
        return 1

    least_exponent = min(exponent for exponent, _ in active_terms)
    greatest_adjusted_exponent = max(adjusted_exponent for _, adjusted_exponent in active_terms)
    # Aligning terms to the least exponent needs every intervening digit. Add
    # enough carry room for summing the small fixed number of cost categories.
    return max(
        1,
        greatest_adjusted_exponent - least_exponent + 1 + len(str(len(active_terms))),
    )


def _contains_price_source_credential(value: str) -> bool:
    """Detect credential-like material before a price reference is persisted."""
    if re.search(
        r"(?i)\b(api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|"
        r"authorization|password|secret)(\s*[:=]\s*)([^\s,;&]+)",
        value,
    ) or re.search(r"(?i)\b(bearer\s+)([^\s,;&]+)", value):
        return True
    if re.search(
        r"(?i)[?&]\s*(?:x-amz-)?(?:api[-_ ]?key|access[-_ ]?token|"
        r"refresh[-_ ]?token|security[-_ ]?token|key|token|secret|password|"
        r"authorization|signature|sig|auth|credential)\s*=\s*[^&#]+",
        value,
    ):
        return True
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.username is not None or parsed.password is not None


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    provider: str
    model: str
    usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        """Validate the untrusted provider boundary at runtime, not only by hints."""
        if not isinstance(self.text, str):
            raise ModelResponseError("provider response text must be text")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ModelResponseError("provider response identity must be non-empty text")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ModelResponseError("provider response model must be non-empty text")
        if self.usage is not None and not isinstance(self.usage, TokenUsage):
            raise ModelResponseError("provider response usage must be validated token usage")


@dataclass(frozen=True)
class CredentialReference:
    provider_id: str
    alias: str
    secret_reference: str
    endpoint: str | None = None

    def __post_init__(self) -> None:
        validate_credential_alias(self.alias)


@dataclass(frozen=True)
class ModelPreset:
    name: str
    provider_id: str
    credential_alias: str
    model: str
    reasoning: str | None = None
    parameters: dict[str, object] | None = None
    reasoning_mode: str | None = None
    reasoning_budget_tokens: int | None = None

    def __post_init__(self) -> None:
        validate_credential_alias(self.credential_alias)


@dataclass(frozen=True)
class RerankResult:
    """Provider rank for one caller-supplied candidate; score is not a probability."""

    index: int
    relevance_score: float


@runtime_checkable
class RerankingProvider(Protocol):
    """Re-rank an already retrieved candidate set without retrieving new corpus text."""

    @property
    def provider_id(self) -> str: ...

    def rerank(
        self,
        model: str,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]: ...


class GenerationProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def capabilities(self, model: str) -> ModelCapabilities: ...
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...


class StreamingGenerationProvider(Protocol):
    """Optional text-delta interface; providers need not implement streaming."""

    def stream_generate(self, request: GenerationRequest) -> Iterator[str]: ...


class EmbeddingProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def endpoint(self) -> str: ...

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]: ...


@runtime_checkable
class QueryEmbeddingProvider(EmbeddingProvider, Protocol):
    """Embedding provider with an optional asymmetric query operation."""

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]: ...


def _validated_embedding_results(results: object, expected_count: int) -> list[tuple[float, ...]]:
    """Validate and convert untrusted vector components at the provider-neutral boundary."""
    if not isinstance(results, list) or len(results) != expected_count or not results:
        raise ModelResponseError("embedding provider returned an invalid vector count")
    vectors: list[tuple[float, ...]] = []
    expected_dimensions: int | None = None
    for result in results:
        if not isinstance(result, (list, tuple)) or not result:
            raise ModelResponseError("embedding provider returned an invalid vector")
        vector: list[float] = []
        for component in result:
            if type(component) not in {int, float}:
                raise ModelResponseError("embedding provider returned an invalid vector value")
            try:
                value = float(component)
            except (OverflowError, ValueError):
                raise ModelResponseError(
                    "embedding provider returned a non-finite vector value"
                ) from None
            if not math.isfinite(value):
                raise ModelResponseError("embedding provider returned a non-finite vector value")
            vector.append(value)
        if expected_dimensions is None:
            expected_dimensions = len(vector)
        elif len(vector) != expected_dimensions:
            raise ModelResponseError("embedding provider returned inconsistent vector dimensions")
        vectors.append(tuple(vector))
    return vectors


def _validated_reranking_results(
    results: object, candidate_count: int, top_n: int | None
) -> list[RerankResult]:
    """Validate untrusted reranker output against the submitted candidate set."""
    if not isinstance(results, list) or not results:
        raise ModelResponseError("reranker returned malformed results")
    if len(results) > candidate_count or (top_n is not None and len(results) > top_n):
        raise ModelResponseError("reranker returned more candidates than requested")

    seen: set[int] = set()
    validated: list[RerankResult] = []
    for result in results:
        if not isinstance(result, RerankResult):
            raise ModelResponseError("reranker returned a malformed result")
        index = result.index
        score = result.relevance_score
        if type(index) is not int or type(score) not in {int, float}:
            raise ModelResponseError("reranker returned an invalid candidate index or score")
        try:
            numeric_score = float(score)
        except (OverflowError, ValueError):
            raise ModelResponseError("reranker returned an invalid candidate index or score") from None
        if (
            index < 0
            or index >= candidate_count
            or index in seen
            or not math.isfinite(numeric_score)
        ):
            raise ModelResponseError("reranker returned an invalid candidate index or score")
        seen.add(index)
        validated.append(RerankResult(index, numeric_score))
    return validated


def _validate_embedding_request(model: str, texts: list[str]) -> None:
    if (
        not isinstance(model, str)
        or not model.strip()
        or not isinstance(texts, list)
        or not texts
        or any(not isinstance(text, str) or not text.strip() for text in texts)
    ):
        raise ModelProviderError("embedding requires a model and non-empty texts")


class ModelGateway:
    def __init__(
        self,
        providers: dict[str, GenerationProvider],
        rerankers: dict[str, RerankingProvider] | None = None,
        credential_reference_fingerprints: dict[str, str] | None = None,
        embedders: dict[str, EmbeddingProvider] | None = None,
    ):
        self.providers = providers
        self.rerankers = rerankers or {}
        self.credential_reference_fingerprints = credential_reference_fingerprints or {}
        self.embedders = embedders or {}

    def _provider_for_generation(
        self, provider_id: str, request: GenerationRequest
    ) -> GenerationProvider:
        provider = self.providers.get(provider_id)
        if provider is None:
            raise ModelProviderError(f"no generation provider registered for {provider_id!r}")
        if provider.provider_id != provider_id:
            raise ModelProviderError(
                "registered generation provider identity does not match selected provider"
            )
        capabilities = provider.capabilities(request.model)
        if request.image is not None:
            if (
                provider_id not in {"gemini", "openai", "anthropic", "anthropic-compatible"}
                and not is_openai_compatible_provider(provider_id)
            ):
                raise ModelProviderError(
                    "image input is supported only by Gemini, OpenAI Chat Completions, "
                    "and Anthropic Messages generation"
                )
            if not capabilities.vision:
                raise ModelProviderError("provider does not declare vision support for this model")
        if not capabilities.text:
            raise ModelProviderError(
                f"provider does not declare text-generation support for model {request.model!r}"
            )
        if request.system_prompt is not None and not capabilities.system_prompt:
            raise ModelProviderError("provider does not declare system-prompt support")
        if request.response_schema is not None and not capabilities.structured_output:
            raise ModelProviderError("provider does not declare structured-output support")
        if request.reasoning and request.reasoning not in capabilities.reasoning_efforts:
            raise ModelProviderError("unsupported reasoning configuration")
        if request.reasoning_mode and request.reasoning_mode not in capabilities.reasoning_modes:
            raise ModelProviderError("unsupported reasoning mode")
        if request.reasoning_budget_tokens is not None and request.reasoning_mode != "manual":
            raise ModelProviderError("reasoning budget requires manual reasoning mode")
        parameters = request.parameters or {}
        unsupported = sorted(set(parameters) - set(capabilities.generation_parameters))
        if unsupported:
            raise ModelProviderError(
                "unsupported generation parameter(s): " + ", ".join(unsupported)
            )
        return provider

    def generate(self, provider_id: str, request: GenerationRequest) -> GenerationResponse:
        provider = self._provider_for_generation(provider_id, request)
        response = provider.generate(request)
        if not isinstance(response, GenerationResponse):
            raise ModelResponseError("provider returned a malformed generation response")
        if response.provider != provider_id or response.model != request.model:
            raise ModelResponseError(
                "provider response identity does not match the selected preset request"
            )
        return response

    def stream(self, provider_id: str, request: GenerationRequest) -> Generator[str, None, None]:
        """Explicitly stream text deltas from an opted-in provider/model pair.

        This is a separate operation from ``generate``; buffered callers and
        analysis pipelines do not change behavior. Closing the returned iterator
        cancels consumption and lets the adapter close its underlying response.
        """
        if request.image is not None:
            raise ModelProviderError("image input is not supported with streaming")
        provider = self._provider_for_generation(provider_id, request)
        if not provider.capabilities(request.model).streaming:
            raise ModelProviderError("provider does not declare streaming support for this model")
        stream_generate = getattr(provider, "stream_generate", None)
        if not callable(stream_generate):
            raise ModelProviderError("registered provider does not implement text streaming")

        chunks = cast("Callable[[GenerationRequest], Iterator[str]]", stream_generate)(request)

        def validated_chunks() -> Generator[str, None, None]:
            try:
                for chunk in chunks:
                    if not isinstance(chunk, str):
                        raise ModelResponseError("provider stream returned a non-text delta")
                    if chunk:
                        yield chunk
            finally:
                close = getattr(chunks, "close", None)
                if callable(close):
                    close()

        return validated_chunks()

    def rerank(
        self,
        provider_id: str,
        model: str,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Use an explicit reranker on caller-provided candidates only."""
        try:
            provider = self.rerankers[provider_id]
        except KeyError as exc:
            raise ModelProviderError(
                f"no reranking provider registered for {provider_id!r}"
            ) from exc
        if provider.provider_id != provider_id:
            raise ModelProviderError(
                "registered reranking provider identity does not match selected provider"
            )
        results = provider.rerank(model, query, documents, top_n=top_n)
        return _validated_reranking_results(results, len(documents), top_n)

    def embed(self, provider_id: str, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        """Embed documents with the explicitly selected embedding provider."""
        _validate_embedding_request(model, texts)
        provider = self.embedders.get(provider_id)
        if provider is None:
            raise ModelProviderError(f"no embedding provider registered for {provider_id!r}")
        if provider.provider_id != provider_id:
            raise ModelProviderError(
                "registered embedding provider identity does not match selected provider"
            )
        return _validated_embedding_results(provider.embed(model, texts), len(texts))

    def embed_query(
        self, provider_id: str, model: str, texts: list[str]
    ) -> list[tuple[float, ...]]:
        """Embed queries using a provider's query operation when it declares one."""
        _validate_embedding_request(model, texts)
        provider = self.embedders.get(provider_id)
        if provider is None:
            raise ModelProviderError(f"no embedding provider registered for {provider_id!r}")
        if provider.provider_id != provider_id:
            raise ModelProviderError(
                "registered embedding provider identity does not match selected provider"
            )
        if isinstance(provider, QueryEmbeddingProvider):
            results = provider.embed_query(model, texts)
        else:
            results = provider.embed(model, texts)
        return _validated_embedding_results(results, len(texts))


class EchoProvider:
    provider_id = "fixture.echo"

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(reasoning_efforts=("low", "high"))

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(request.prompt, self.provider_id, request.model)
