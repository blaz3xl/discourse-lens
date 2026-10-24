"""Generic OpenAI-compatible generation adapter behind the provider-neutral gateway."""

from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, ClassVar, NoReturn, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .errors import (
    AuthenticationError,
    ModelProviderError,
    ModelResponseError,
    QuotaError,
    RateLimitError,
)
from .json_safety import validate_json_nesting
from .models import (
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    RerankResult,
    TokenUsage,
)
from .network_policy import is_explicit_loopback_endpoint, is_loopback_endpoint
from .provider_catalog import is_openai_compatible_provider
from .reasoning_policy import validate_cohere_manual_budget
from .secrets import has_credential_like_endpoint_host, is_credential_like_alias

MAX_PROVIDER_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_GEMINI_INLINE_REQUEST_BYTES = 18 * 1024 * 1024
MAX_OPENAI_CHAT_IMAGE_REQUEST_BYTES = 18 * 1024 * 1024
MAX_ANTHROPIC_MESSAGES_IMAGE_REQUEST_BYTES = 18 * 1024 * 1024
MAX_PROVIDER_STREAM_EVENT_BYTES = 1024 * 1024
_MAX_ENDPOINT_PATH_DECODING_PASSES = 8
_OPENAI_REASONING_EFFORT_WIRE_VALUES = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
_OPENAI_RESPONSES_API_MODELS = frozenset({"gpt-6-luna"})
_ANTHROPIC_REASONING_EFFORT_WIRE_VALUES = frozenset({"low", "medium", "high", "xhigh", "max"})
_GEMINI_REASONING_EFFORT_WIRE_VALUES = frozenset({"minimal", "low", "medium", "high"})
_WEEKDAYS_SHORT = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_WEEKDAYS_LONG = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_HTTP_DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?P<weekday>Mon|Tue|Wed|Thu|Fri|Sat|Sun), "
            r"[0-9]{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
            r"[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT",
            re.IGNORECASE | re.ASCII,
        ),
        "short",
    ),
    (
        re.compile(
            r"(?P<weekday>Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), "
            r"[0-9]{2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-[0-9]{2} "
            r"[0-9]{2}:[0-9]{2}:[0-9]{2} GMT",
            re.IGNORECASE | re.ASCII,
        ),
        "long",
    ),
    (
        re.compile(
            r"(?P<weekday>Mon|Tue|Wed|Thu|Fri|Sat|Sun) "
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
            r"(?: [1-9]|[12][0-9]|3[01]) [0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{4}",
            re.IGNORECASE | re.ASCII,
        ),
        "asctime",
    ),
)


class SecretStore(Protocol):
    def get(self, reference: str) -> str: ...


def _resolve_provider_credential(secret_store: SecretStore, reference: str) -> str:
    """Resolve an API key without exposing backend exception details to callers."""
    try:
        return secret_store.get(reference)
    except Exception:  # noqa: BLE001 - custom backend errors may contain credentials
        # Custom secret stores may include the credential or reference in their
        # exception text. Provider errors and tracebacks must not forward it.
        raise ModelProviderError("provider credential could not be resolved") from None


class _RejectRedirects(HTTPRedirectHandler):
    """Provider requests carry corpus text and credentials; redirects are not followed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _provider_error_codes(error: HTTPError) -> set[str]:
    """Read bounded provider codes; never copy a remote message into errors/logs."""
    try:
        payload = _parse_provider_json(error.read(16_384).decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return set()
    if not isinstance(payload, dict):
        return set()
    nested = payload.get("error")
    candidates = [nested, payload]
    codes: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        for key in ("code", "type", "status"):
            value = item.get(key)
            if isinstance(value, str):
                normalized = value.strip().casefold().replace("-", "_")
                if normalized and len(normalized) <= 80:
                    codes.add(normalized)
    return codes


def _retry_after_seconds(error: HTTPError, *, now: datetime | None = None) -> float | None:
    headers = error.headers
    value = headers.get("Retry-After") if headers is not None else None
    if not isinstance(value, str):
        return None
    normalized_value = value.strip()
    try:
        delay = float(normalized_value)
    except ValueError:
        delay = None
    if delay is not None:
        return delay if math.isfinite(delay) and delay >= 0 else None

    current = now if now is not None else datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        current = current.replace(tzinfo=UTC)
    else:
        current = current.astimezone(UTC)
    for pattern, weekday_kind in _HTTP_DATE_PATTERNS:
        match = pattern.fullmatch(normalized_value)
        if match is None:
            continue
        try:
            retry_at = parsedate_to_datetime(normalized_value)
        except (TypeError, ValueError, OverflowError):
            return None

        if weekday_kind == "asctime":
            retry_at = retry_at.replace(tzinfo=UTC)
        elif retry_at.tzinfo is None or retry_at.utcoffset() != timedelta(0):
            return None

        if weekday_kind == "long":
            two_digit_year = int(normalized_value.rsplit("-", 1)[1][:2])
            resolved_year = (current.year // 100) * 100 + two_digit_year
            if resolved_year > current.year + 50:
                resolved_year -= 100
            try:
                retry_at = retry_at.replace(year=resolved_year)
            except ValueError:
                return None

        expected_weekday = (
            _WEEKDAYS_LONG[retry_at.weekday()]
            if weekday_kind == "long"
            else _WEEKDAYS_SHORT[retry_at.weekday()]
        )
        if match.group("weekday").casefold() != expected_weekday.casefold():
            return None
        return max(0.0, (retry_at - current).total_seconds())
    return None


def _classify_provider_http_error(
    error: HTTPError,
    provider_id: str,
    operation: str,
    *,
    additional_auth_codes: tuple[int, ...] = (),
) -> NoReturn:
    """Map provider status to stable, secret-safe domain error types without retrying."""
    provider_label = {
        "jina-reranker": "Jina",
        "cohere-reranker": "Cohere",
        "cohere": "Cohere",
        "jina-embeddings": "Jina",
        "gemini-embeddings": "Gemini",
        "cohere-embeddings": "Cohere",
        "voyage-embeddings": "Voyage",
    }.get(provider_id, provider_id)
    if error.code in {401, 403, *additional_auth_codes}:
        raise AuthenticationError(f"{provider_label} rejected the configured credential") from None
    codes = _provider_error_codes(error)
    if error.code == 402 or {
        "insufficient_quota",
        "quota_exceeded",
        "billing_hard_limit_reached",
        "billing_not_active",
    }.intersection(codes):
        raise QuotaError(f"{provider_id} {operation} quota or billing limit was reached") from None
    if error.code == 429 or {
        "rate_limit_exceeded",
        "rate_limit_error",
        "too_many_requests",
    }.intersection(codes):
        raise RateLimitError(
            f"{provider_id} {operation} was rate limited",
            retry_after_seconds=_retry_after_seconds(error),
        ) from None
    raise ModelProviderError(f"{provider_id} {operation} HTTP error: {error.code}") from None


def _raise_provider_http_error(
    error: HTTPError,
    provider_id: str,
    operation: str,
    *,
    additional_auth_codes: tuple[int, ...] = (),
) -> NoReturn:
    """Classify an HTTP error, then close its response body on every path."""
    try:
        _classify_provider_http_error(
            error,
            provider_id,
            operation,
            additional_auth_codes=additional_auth_codes,
        )
    finally:
        try:
            error.close()
        except OSError:
            pass


def _open_provider_request(request: Request, timeout: float):
    if is_explicit_loopback_endpoint(request.full_url):
        # Provider objects can outlive the environment in which they were
        # constructed. Recheck the proxy policy at send time, then pin this
        # request to a direct connection so a later environment change cannot
        # route local corpus material through a newly configured proxy.
        if not is_loopback_endpoint(request.full_url):
            raise URLError("loopback endpoint would use an environment proxy")
        opener = build_opener(ProxyHandler({}), _RejectRedirects())
    else:
        opener = build_opener(_RejectRedirects())
    return opener.open(request, timeout=timeout)


def _iter_server_sent_data(response: Any) -> Iterator[str]:
    """Read bounded UTF-8 SSE events with CR, LF, or CRLF line endings."""
    data_lines: list[str] = []
    line_buffer = bytearray()
    event_bytes = 0
    skip_lf = False

    def parse_line(raw_line: bytes) -> str | None:
        nonlocal event_bytes
        if not raw_line:
            event_data = "\n".join(data_lines) if data_lines else None
            data_lines.clear()
            event_bytes = 0
            return event_data
        try:
            decoded = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            raise ModelResponseError("provider stream event is not valid UTF-8") from None
        if decoded.startswith(":"):
            return None
        field, separator, value = decoded.partition(":")
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
        return None

    reader = getattr(response, "read1", None)
    if not callable(reader):
        reader = getattr(response, "read", None)
    if not callable(reader):
        raise ModelResponseError("provider stream does not support bounded reads")

    while True:
        chunk = reader(4096)
        if not isinstance(chunk, (bytes, bytearray)):
            raise ModelResponseError("provider stream returned a malformed event")
        if not chunk:
            break
        for byte in chunk:
            if skip_lf:
                skip_lf = False
                if byte == 0x0A:
                    continue
            if byte in {0x0D, 0x0A}:
                event_bytes += 1
                if event_bytes > MAX_PROVIDER_STREAM_EVENT_BYTES:
                    raise ModelResponseError(
                        "provider stream event exceeds the configured byte limit"
                    )
                event_data = parse_line(bytes(line_buffer))
                line_buffer.clear()
                if byte == 0x0D:
                    skip_lf = True
                if event_data is not None:
                    yield event_data
            else:
                line_buffer.append(byte)
                event_bytes += 1
                if (
                    len(line_buffer) > MAX_PROVIDER_STREAM_EVENT_BYTES
                    or event_bytes > MAX_PROVIDER_STREAM_EVENT_BYTES
                ):
                    raise ModelResponseError(
                        "provider stream event exceeds the configured byte limit"
                    )

    # A final SSE event may legally end at EOF without an additional blank line.
    if line_buffer:
        event_data = parse_line(bytes(line_buffer))
        line_buffer.clear()
        if event_data is not None:
            yield event_data
    if data_lines:
        yield "\n".join(data_lines)


def _openai_stream_delta(event_data: str) -> tuple[str | None, str | None]:
    try:
        payload = _parse_provider_json(event_data)
    except (ValueError, RecursionError):
        raise ModelResponseError("provider stream contains malformed JSON") from None
    if not isinstance(payload, dict):
        raise ModelResponseError("provider stream event must be a JSON object")
    if "error" in payload:
        # Provider-controlled error text may contain sensitive request data.
        raise ModelResponseError("provider returned a streaming error event") from None
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ModelResponseError("provider stream event lacks one chat choice")
    choice = choices[0]
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        raise ModelResponseError("provider stream event lacks a text delta")
    text = delta.get("content")
    if text is not None and not isinstance(text, str):
        raise ModelResponseError("provider stream delta content must be text")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and (not isinstance(finish_reason, str) or not finish_reason):
        raise ModelResponseError("provider stream finish reason is malformed")
    return text, finish_reason


def _iter_openai_stream_text(response: Any) -> Iterator[str]:
    finished = False
    received_done = False
    for event_data in _iter_server_sent_data(response):
        if event_data.strip() == "[DONE]":
            if not finished:
                raise ModelResponseError("provider stream ended before a finish reason")
            received_done = True
            break
        text, finish_reason = _openai_stream_delta(event_data)
        if finished and text:
            raise ModelResponseError("provider stream emitted text after completion")
        if finish_reason is not None and finish_reason != "stop":
            # A length stop is partial output; filtered/tool-call results do not
            # satisfy this text-only streaming contract either.
            raise ModelResponseError("provider stream ended without a complete text response")
        if text:
            yield text
        finished = finished or finish_reason is not None
    if not received_done:
        raise ModelResponseError("provider stream ended without a completion marker")


def _iter_openai_responses_stream_text(response: Any) -> Iterator[str]:
    completed = False
    emitted_text = False
    for event_data in _iter_server_sent_data(response):
        if event_data.strip() == "[DONE]":
            if not completed:
                raise ModelResponseError("OpenAI Responses stream ended before completion")
            break
        try:
            payload = _parse_provider_json(event_data)
        except (ValueError, RecursionError):
            raise ModelResponseError("OpenAI Responses stream contains malformed JSON") from None
        if not isinstance(payload, dict):
            raise ModelResponseError("OpenAI Responses stream event must be a JSON object")
        event_type = payload.get("type")
        if not isinstance(event_type, str) or not event_type:
            raise ModelResponseError("OpenAI Responses stream event lacks a type")
        if event_type in {"error", "response.failed"}:
            raise ModelResponseError("OpenAI Responses stream failed") from None
        if event_type == "response.incomplete":
            raise ModelResponseError("OpenAI Responses stream ended with an incomplete response")
        if event_type == "response.refusal.delta":
            raise ModelResponseError("OpenAI Responses API refused the request") from None
        if event_type == "response.output_text.delta":
            if completed:
                raise ModelResponseError("OpenAI Responses stream emitted text after completion")
            delta = payload.get("delta")
            if not isinstance(delta, str):
                raise ModelResponseError("OpenAI Responses stream text delta is malformed")
            if delta:
                emitted_text = True
                yield delta
            continue
        if event_type != "response.completed":
            continue
        if completed:
            raise ModelResponseError("OpenAI Responses stream completed more than once")
        response_payload = payload.get("response")
        if not isinstance(response_payload, dict) or response_payload.get("status") != "completed":
            raise ModelResponseError("OpenAI Responses stream lacks a completed response")
        final_text = _openai_responses_output_text(response_payload)
        if not emitted_text:
            if not isinstance(final_text, str) or not final_text:
                raise ModelResponseError("OpenAI Responses stream contains no text output")
            emitted_text = True
            yield final_text
        completed = True
    if not completed:
        raise ModelResponseError("OpenAI Responses stream ended without a completion event")


def _iter_anthropic_stream_text(response: Any) -> Iterator[str]:
    """Yield only Messages text deltas and require the terminal message_stop."""
    message_started = False
    stop_reason: str | None = None
    for event_data in _iter_server_sent_data(response):
        try:
            payload = _parse_provider_json(event_data)
        except (ValueError, RecursionError):
            raise ModelResponseError("Anthropic stream contains malformed JSON") from None
        if not isinstance(payload, dict):
            raise ModelResponseError("Anthropic stream event must be a JSON object")
        event_type = payload.get("type")
        if not isinstance(event_type, str) or not event_type:
            raise ModelResponseError("Anthropic stream event lacks a type")
        if event_type == "error" or "error" in payload:
            # Provider-controlled error messages may contain sensitive input.
            raise ModelResponseError("Anthropic returned a streaming error event") from None
        if event_type == "ping":
            continue
        if event_type == "message_start":
            message = payload.get("message")
            if message_started or not isinstance(message, dict):
                raise ModelResponseError("Anthropic message_start event is malformed")
            message_started = True
            continue
        if event_type == "message_stop":
            if not message_started:
                raise ModelResponseError("Anthropic stream stopped before message_start")
            if stop_reason not in {"end_turn", "stop_sequence"}:
                raise ModelResponseError(
                    "Anthropic stream ended without a complete text response"
                )
            return
        if event_type == "message_delta":
            if not message_started or stop_reason is not None:
                raise ModelResponseError("Anthropic message_delta event is malformed")
            delta = payload.get("delta")
            if not isinstance(delta, dict):
                raise ModelResponseError("Anthropic message_delta event is malformed")
            reason = delta.get("stop_reason")
            if reason is not None and (not isinstance(reason, str) or not reason):
                raise ModelResponseError("Anthropic message_delta event is malformed")
            stop_reason = reason
            continue
        if event_type == "content_block_delta":
            if not message_started:
                raise ModelResponseError("Anthropic stream emitted a delta before message_start")
            delta = payload.get("delta")
            if not isinstance(delta, dict):
                raise ModelResponseError("Anthropic content delta is malformed")
            delta_type = delta.get("type")
            if not isinstance(delta_type, str):
                raise ModelResponseError("Anthropic content delta lacks a type")
            if delta_type == "text_delta":
                text = delta.get("text")
                if not isinstance(text, str):
                    raise ModelResponseError("Anthropic text delta must contain text")
                if text:
                    yield text
            elif delta_type == "thinking_delta":
                if not isinstance(delta.get("thinking"), str):
                    raise ModelResponseError("Anthropic thinking delta must contain text")
            elif delta_type == "signature_delta":
                if not isinstance(delta.get("signature"), str):
                    raise ModelResponseError("Anthropic signature delta must contain text")
            # Thinking, signatures, tool JSON, and future non-text deltas are
            # intentionally not exposed through this text-only interface.
    if not message_started:
        raise ModelResponseError("Anthropic stream ended before message_start")
    raise ModelResponseError("Anthropic stream ended without message_stop")


def _read_bounded_provider_response(response: Any) -> bytes:
    """Read a provider body up to 8 MiB, rejecting oversized bodies safely.

    Content-Length is only an early-rejection hint: the bounded read still asks
    for at most one byte beyond the limit so chunked, missing, or dishonest
    length headers cannot cause an unbounded allocation.
    """
    raw_length = getattr(response, "headers", None)
    raw_length = raw_length.get("Content-Length") if raw_length is not None else None
    declared_length: int | None = None
    if isinstance(raw_length, int) and not isinstance(raw_length, bool):
        declared_length = raw_length
    elif isinstance(raw_length, str):
        normalized_length = raw_length.strip()
        if normalized_length.isascii() and normalized_length.isdecimal():
            significant_digits = normalized_length.lstrip("0") or "0"
            if len(significant_digits) > len(str(MAX_PROVIDER_RESPONSE_BYTES)):
                declared_length = MAX_PROVIDER_RESPONSE_BYTES + 1
            else:
                declared_length = int(significant_digits)
    if declared_length is not None and declared_length > MAX_PROVIDER_RESPONSE_BYTES:
        raise ModelResponseError("provider response exceeds the configured byte limit")

    body = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
    if not isinstance(body, (bytes, bytearray)):
        raise ModelResponseError("provider response body must be bytes")
    if len(body) > MAX_PROVIDER_RESPONSE_BYTES:
        raise ModelResponseError("provider response exceeds the configured byte limit")
    return bytes(body)


def _parse_provider_json(text: str) -> Any:
    """Apply an explicit nesting bound before the interpreter's JSON parser."""
    validate_json_nesting(text)
    return json.loads(text)


def _encode_provider_json(payload: object) -> bytes:
    """Serialize a standards-compliant JSON request or fail before transport."""
    try:
        return json.dumps(payload, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ModelProviderError("provider request must contain valid JSON values") from None


def _validated_embedding_vector(values: object) -> tuple[float, ...]:
    """Normalize one untrusted provider vector, rejecting malformed numeric values."""
    if not isinstance(values, list) or not values:
        raise ValueError("embedding vector must be a non-empty list")
    vector: list[float] = []
    for value in values:
        # Exact type checks deliberately reject bool (an int subclass) and strings
        # that float() would otherwise accept. JSON parsers can also accept NaN/Inf.
        if type(value) not in {int, float}:
            raise ValueError("embedding vector values must be numbers")
        try:
            numeric_value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError("embedding vector value is outside the finite range") from exc
        if not math.isfinite(numeric_value):
            raise ValueError("embedding vector values must be finite")
        vector.append(numeric_value)
    return tuple(vector)


def _openai_chat_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        system_prompt=True,
        streaming=True,
        generation_parameters=(
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "seed",
        ),
    )


def _require_supported_parameters(
    request: GenerationRequest, capabilities: ModelCapabilities
) -> dict[str, object]:
    parameters = request.parameters or {}
    unsupported = sorted(set(parameters) - set(capabilities.generation_parameters))
    if unsupported:
        raise ModelProviderError(
            "provider does not declare generation parameter(s): " + ", ".join(unsupported)
        )
    return parameters


def _require_supported_system_prompt(
    request: GenerationRequest, capabilities: ModelCapabilities
) -> None:
    if request.system_prompt is not None and not capabilities.system_prompt:
        raise ModelProviderError("provider does not declare system-prompt support")


def _require_supported_structured_output(
    request: GenerationRequest, capabilities: ModelCapabilities
) -> None:
    if request.response_schema is not None and not capabilities.structured_output:
        raise ModelProviderError("provider does not declare structured-output support")


def _anthropic_user_content(request: GenerationRequest) -> str | list[dict[str, object]]:
    """Build a Messages user content value without changing the text-only wire shape."""
    if request.image is None:
        return request.prompt
    image = request.image
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.mime_type,
                "data": base64.b64encode(image.data).decode("ascii"),
            },
        },
        {"type": "text", "text": request.prompt},
    ]


def _validate_anthropic_image_request_size(payload: dict[str, object]) -> None:
    """Bound the fully serialized inline-image request before resolving credentials."""
    encoded = _encode_provider_json(payload)
    if len(encoded) > MAX_ANTHROPIC_MESSAGES_IMAGE_REQUEST_BYTES:
        raise ModelProviderError("Anthropic Messages image request exceeds the configured byte limit")


def _require_supported_reasoning(
    request: GenerationRequest,
    capabilities: ModelCapabilities,
    *,
    native_effort_values: frozenset[str] | None = None,
) -> None:
    """Reject undeclared or unmappable effort values before building a wire request.

    Exact-model capabilities can be supplied by project configuration or by a
    caller constructing a low-level adapter directly. They describe model
    support, but cannot expand the finite set of values understood by this
    adapter's native wire mapping.
    """
    if request.reasoning and request.reasoning not in capabilities.reasoning_efforts:
        raise ModelProviderError(
            f"provider does not declare reasoning effort {request.reasoning!r} for model {request.model!r}"
        )
    if (
        request.reasoning
        and native_effort_values is not None
        and request.reasoning not in native_effort_values
    ):
        raise ModelProviderError(
            f"provider native mapping does not support reasoning effort {request.reasoning!r}"
        )


def _require_supported_reasoning_mode(
    request: GenerationRequest, capabilities: ModelCapabilities
) -> None:
    if request.reasoning_mode and request.reasoning_mode not in capabilities.reasoning_modes:
        raise ModelProviderError(
            f"provider does not declare reasoning mode {request.reasoning_mode!r} "
            f"for model {request.model!r}"
        )


def _reject_unsupported_reasoning_budget(request: GenerationRequest, provider_name: str) -> None:
    """Do not silently discard token budgets on transports with no budget mapping."""
    if request.reasoning_budget_tokens is not None:
        raise ModelProviderError(f"{provider_name} does not map reasoning token budgets")


def _validate_endpoint(endpoint: str) -> str:
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or endpoint != endpoint.strip()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in endpoint)
        or "?" in endpoint
        or "#" in endpoint
    ):
        raise ModelProviderError("provider endpoint must be an absolute HTTP(S) URL")
    try:
        parsed = urlparse(endpoint)
        hostname = parsed.hostname
        # Accessing .port validates malformed and out-of-range explicit ports.
        _ = parsed.port
    except ValueError:
        raise ModelProviderError("provider endpoint must be an absolute HTTP(S) URL") from None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or hostname is None:
        raise ModelProviderError("provider endpoint must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ModelProviderError(
            "provider endpoint must not include credentials, query, or fragment"
        )
    if has_credential_like_endpoint_host(endpoint):
        raise ModelProviderError("provider endpoint must not include credential material")
    decoded_path = parsed.path
    for _ in range(_MAX_ENDPOINT_PATH_DECODING_PASSES):
        next_path = unquote(decoded_path)
        if next_path == decoded_path:
            break
        decoded_path = next_path
    else:
        raise ModelProviderError("provider endpoint path exceeds the decoding limit")
    if is_credential_like_alias(decoded_path) or any(
        is_credential_like_alias(segment) for segment in decoded_path.split("/")
    ):
        raise ModelProviderError("provider endpoint must not include credential material")
    if parsed.netloc.startswith("["):
        closing_bracket = parsed.netloc.find("]")
        if closing_bracket < 0:
            raise ModelProviderError("provider endpoint must be an absolute HTTP(S) URL")
        authority_suffix = parsed.netloc[closing_bracket + 1 :]
        if authority_suffix and not authority_suffix.startswith(":"):
            raise ModelProviderError("provider endpoint must be an absolute HTTP(S) URL")
    return endpoint.rstrip("/")


def _validate_provider_timeout(provider_name: str, timeout_seconds: object) -> None:
    """Reject invalid timeout values before constructing an HTTP provider client."""
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ModelProviderError(f"{provider_name} timeout must be numeric, finite, and positive")
    try:
        valid = math.isfinite(timeout_seconds) and timeout_seconds > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ModelProviderError(f"{provider_name} timeout must be numeric, finite, and positive")


def _validate_required_provider_auth_and_timeout(
    provider_name: str, credential_reference: str, timeout_seconds: object
) -> None:
    """Fail closed before constructing a native authenticated provider client."""
    if not isinstance(credential_reference, str) or not credential_reference.strip():
        raise ModelProviderError(f"{provider_name} requires a non-empty credential reference")
    _validate_provider_timeout(provider_name, timeout_seconds)


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    """Explicit adapter for a user-configured local or remote compatible endpoint.

    The adapter has no discovery or fallback behavior, so an analysis cannot silently
    change the endpoint to which its corpus context is sent.
    """

    endpoint: str
    credential_reference: str | None
    secret_store: SecretStore = field(repr=False)
    provider_id: str = "openai-compatible"
    capabilities_by_model: dict[str, ModelCapabilities] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    native_reasoning_mapping: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        if self.credential_reference is None and not is_loopback_endpoint(self.endpoint):
            raise ModelProviderError(
                "no-auth provider access is allowed only for loopback endpoints"
            )
        _validate_provider_timeout("provider", self.timeout_seconds)

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.capabilities_by_model.get(model, _openai_chat_capabilities())

    def _generation_http_request(self, request: GenerationRequest, *, streaming: bool) -> Request:
        if not request.prompt:
            raise ModelProviderError("generation prompt must not be empty")
        capabilities = self.capabilities(request.model)
        if request.image is not None:
            if self.provider_id != "openai" and not is_openai_compatible_provider(
                self.provider_id
            ):
                raise ModelProviderError(
                    "image input is unsupported by this provider's Chat Completions mapping"
                )
            if streaming:
                raise ModelProviderError(
                    "OpenAI Chat Completions image input is not supported with streaming"
                )
            if not capabilities.vision:
                raise ModelProviderError(
                    "OpenAI Chat Completions image input requires vision support for this exact model"
                )
        if streaming and not capabilities.streaming:
            raise ModelProviderError("provider does not declare streaming support for this model")
        if streaming and request.response_schema is not None:
            raise ModelProviderError("streaming structured output is not supported")
        _require_supported_system_prompt(request, capabilities)
        _require_supported_structured_output(request, capabilities)
        _require_supported_reasoning(
            request,
            capabilities,
            native_effort_values=(
                _OPENAI_REASONING_EFFORT_WIRE_VALUES if self.native_reasoning_mapping else None
            ),
        )
        _reject_unsupported_reasoning_budget(request, "OpenAI Chat Completions adapter")
        if request.reasoning_mode:
            raise ModelProviderError(
                "OpenAI Chat Completions adapter does not map Anthropic reasoning modes"
            )
        parameters = _require_supported_parameters(request, capabilities)
        if request.reasoning and not self.native_reasoning_mapping:
            # OpenAI-compatible APIs do not share one documented reasoning wire
            # format. A capability registry cannot create provider-native mapping.
            raise ModelProviderError(
                "generic OpenAI-compatible adapter has no native reasoning mapping; use a provider-specific adapter"
            )
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if streaming else "application/json",
        }
        messages: list[dict[str, object]] = []
        if request.system_prompt is not None:
            messages.append({"role": "system", "content": request.system_prompt})
        user_content: str | list[dict[str, object]] = request.prompt
        if request.image is not None:
            encoded_image = base64.b64encode(request.image.data).decode("ascii")
            user_content = [
                {"type": "text", "text": request.prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{request.image.mime_type};base64,{encoded_image}"
                    },
                },
            ]
        messages.append({"role": "user", "content": user_content})
        payload: dict[str, object] = {
            "model": request.model,
            "messages": messages,
            **(
                {"reasoning_effort": request.reasoning}
                if request.reasoning and self.native_reasoning_mapping
                else {}
            ),
            **parameters,
        }
        if streaming:
            payload["stream"] = True
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_schema_name,
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        body = _encode_provider_json(payload)
        if request.image is not None and len(body) > MAX_OPENAI_CHAT_IMAGE_REQUEST_BYTES:
            raise ModelProviderError(
                "OpenAI Chat Completions image request exceeds the configured byte limit"
            )
        if self.credential_reference is not None:
            headers["Authorization"] = (
                f"Bearer {_resolve_provider_credential(self.secret_store, self.credential_reference)}"
            )
        return Request(
            f"{self.endpoint}/chat/completions",
            body,
            headers=headers,
            method="POST",
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        http_request = self._generation_http_request(request, streaming=False)
        try:
            with _open_provider_request(http_request, timeout=self.timeout_seconds) as response:
                payload = _parse_provider_json(
                    _read_bounded_provider_response(response).decode("utf-8")
                )
        except HTTPError as exc:
            _raise_provider_http_error(exc, self.provider_id, "generation")
        except URLError:
            raise ModelProviderError("provider endpoint is unavailable") from None
        except (OSError, ValueError, RecursionError):
            raise ModelProviderError("provider response could not be read") from None
        if not isinstance(payload, dict):
            raise ModelResponseError("provider response must be a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ModelResponseError("provider response lacks a chat completion") from None
        finish_reason = choices[0].get("finish_reason")
        if finish_reason != "stop":
            if finish_reason == "length":
                raise ModelResponseError(
                    f"{self.provider_id} response reached its output limit before completion"
                )
            if finish_reason == "content_filter":
                raise ModelResponseError(
                    f"{self.provider_id} filtered the response before completion"
                )
            if isinstance(finish_reason, str) and finish_reason in {"function_call", "tool_calls"}:
                raise ModelResponseError(
                    f"{self.provider_id} response requires an unsupported tool handoff"
                )
            raise ModelResponseError(
                f"{self.provider_id} response has a missing or unsupported finish reason"
            )
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ModelResponseError("provider response lacks a chat completion") from None
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal:
            raise ModelResponseError(f"{self.provider_id} refused to complete the text response")
        if refusal is not None and not isinstance(refusal, str):
            raise ModelResponseError(f"{self.provider_id} response has malformed refusal metadata")
        text = message.get("content")
        if not isinstance(text, str):
            raise ModelResponseError("provider completion content must be text")
        usage = _openai_usage(payload.get("usage"))
        return GenerationResponse(text, self.provider_id, request.model, usage)

    def stream_generate(self, request: GenerationRequest) -> Iterator[str]:
        """Explicit Chat Completions SSE text streaming for opted-in models."""
        http_request = self._generation_http_request(request, streaming=True)
        try:
            with _open_provider_request(http_request, timeout=self.timeout_seconds) as response:
                yield from _iter_openai_stream_text(response)
        except HTTPError as exc:
            _raise_provider_http_error(exc, self.provider_id, "generation")
        except URLError:
            raise ModelProviderError("provider endpoint is unavailable") from None
        except OSError:
            raise ModelProviderError("provider streaming response could not be read") from None


@dataclass(frozen=True)
class OpenAIResponsesProvider:
    """OpenAI Responses API transport for models whose reasoning API is Responses-first."""

    endpoint: str
    credential_reference: str
    secret_store: SecretStore = field(repr=False)
    provider_id: str = "openai"
    capabilities_by_model: dict[str, ModelCapabilities] = field(default_factory=dict)
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        if not self.credential_reference:
            raise ModelProviderError("provider requires a credential reference")
        _validate_provider_timeout("OpenAI Responses", self.timeout_seconds)

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.capabilities_by_model.get(
            model,
            ModelCapabilities(
                system_prompt=True,
                structured_output=True,
                streaming=True,
                reasoning_efforts=tuple(_OPENAI_REASONING_EFFORT_WIRE_VALUES),
            ),
        )

    def _generation_http_request(self, request: GenerationRequest, *, streaming: bool) -> Request:
        if not request.prompt:
            raise ModelProviderError("generation prompt must not be empty")
        capabilities = self.capabilities(request.model)
        if request.image is not None and not capabilities.vision:
            raise ModelProviderError("OpenAI Responses image input requires vision support")
        if streaming and not capabilities.streaming:
            raise ModelProviderError("provider does not declare streaming support for this model")
        _require_supported_system_prompt(request, capabilities)
        _require_supported_structured_output(request, capabilities)
        _require_supported_reasoning(
            request,
            capabilities,
            native_effort_values=_OPENAI_REASONING_EFFORT_WIRE_VALUES,
        )
        _reject_unsupported_reasoning_budget(request, "OpenAI Responses API adapter")
        if request.reasoning_mode:
            raise ModelProviderError(
                "OpenAI Responses API adapter does not declare reasoning-mode support for this model"
            )
        parameters = _require_supported_parameters(request, capabilities)
        parameter_mapping = {
            "temperature": "temperature",
            "top_p": "top_p",
            "max_tokens": "max_output_tokens",
            "max_output_tokens": "max_output_tokens",
        }
        payload: dict[str, object] = {
            "model": request.model,
            "input": request.prompt,
            "store": False,
        }
        if request.system_prompt is not None:
            payload["instructions"] = request.system_prompt
        if request.reasoning is not None:
            payload["reasoning"] = {"effort": request.reasoning}
        for name, value in parameters.items():
            wire_name = parameter_mapping.get(name)
            if wire_name is None:
                raise ModelProviderError(
                    f"OpenAI Responses API adapter does not map generation parameter {name!r}"
                )
            payload[wire_name] = value
        if request.image is not None:
            image_url = (
                f"data:{request.image.mime_type};base64,"
                f"{base64.b64encode(request.image.data).decode('ascii')}"
            )
            payload["input"] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": request.prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ]
        if streaming:
            payload["stream"] = True
        if request.response_schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.response_schema_name,
                    "strict": True,
                    "schema": request.response_schema,
                }
            }
        body = _encode_provider_json(payload)
        if request.image is not None and len(body) > MAX_OPENAI_CHAT_IMAGE_REQUEST_BYTES:
            raise ModelProviderError(
                "OpenAI Responses image request exceeds the configured byte limit"
            )
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if streaming else "application/json",
            "Authorization": (
                f"Bearer {_resolve_provider_credential(self.secret_store, self.credential_reference)}"
            ),
        }
        return Request(
            f"{self.endpoint}/responses",
            body,
            headers=headers,
            method="POST",
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        http_request = self._generation_http_request(request, streaming=False)
        try:
            with _open_provider_request(http_request, timeout=self.timeout_seconds) as response:
                payload = _parse_provider_json(
                    _read_bounded_provider_response(response).decode("utf-8")
                )
        except HTTPError as exc:
            _raise_provider_http_error(exc, self.provider_id, "generation")
        except URLError:
            raise ModelProviderError("provider endpoint is unavailable") from None
        except (OSError, ValueError, RecursionError):
            raise ModelProviderError("provider response could not be read") from None
        if not isinstance(payload, dict):
            raise ModelResponseError("OpenAI Responses API response must be a JSON object")
        response_status = payload.get("status")
        if response_status == "incomplete":
            raise ModelResponseError("OpenAI Responses API returned an incomplete response")
        if response_status != "completed":
            raise ModelResponseError("OpenAI Responses API response is not complete")
        text = _openai_responses_output_text(payload)
        if not isinstance(text, str) or not text:
            raise ModelResponseError("OpenAI Responses API response contains no text output")
        return GenerationResponse(
            text,
            self.provider_id,
            request.model,
            _openai_responses_usage(payload.get("usage")),
        )

    def stream_generate(self, request: GenerationRequest) -> Iterator[str]:
        http_request = self._generation_http_request(request, streaming=True)
        try:
            with _open_provider_request(http_request, timeout=self.timeout_seconds) as response:
                yield from _iter_openai_responses_stream_text(response)
        except HTTPError as exc:
            _raise_provider_http_error(exc, self.provider_id, "generation")
        except URLError:
            raise ModelProviderError("provider endpoint is unavailable") from None
        except OSError:
            raise ModelProviderError("provider streaming response could not be read") from None


@dataclass(frozen=True)
class OpenAIProvider:
    """Direct OpenAI transport with model-specific Chat Completions and Responses routing.

    The Responses API is used for GPT-6 Luna so its documented reasoning effort
    values are sent using the Responses API shape. Other models keep the existing
    Chat Completions mapping. Both paths use the configured credential and endpoint.
    """

    credential_reference: str
    secret_store: SecretStore = field(repr=False)
    endpoint: str = "https://api.openai.com/v1"
    provider_id: str = "openai"
    capabilities_by_model: dict[str, ModelCapabilities] = field(default_factory=dict)
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        if not self.credential_reference:
            raise ModelProviderError("provider requires a credential reference")
        _validate_provider_timeout("provider", self.timeout_seconds)

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.capabilities_by_model.get(model, self._default_capabilities())

    @staticmethod
    def _default_capabilities() -> ModelCapabilities:
        return _openai_chat_capabilities()

    def _provider_for_model(self, model: str):
        if model in _OPENAI_RESPONSES_API_MODELS:
            return OpenAIResponsesProvider(
                endpoint=self.endpoint,
                credential_reference=self.credential_reference,
                secret_store=self.secret_store,
                provider_id=self.provider_id,
                capabilities_by_model=self.capabilities_by_model,
                timeout_seconds=self.timeout_seconds,
            )
        return OpenAICompatibleProvider(
            endpoint=self.endpoint,
            credential_reference=self.credential_reference,
            secret_store=self.secret_store,
            provider_id=self.provider_id,
            capabilities_by_model=self.capabilities_by_model,
            timeout_seconds=self.timeout_seconds,
            native_reasoning_mapping=True,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return self._provider_for_model(request.model).generate(request)

    def stream_generate(self, request: GenerationRequest) -> Iterator[str]:
        yield from self._provider_for_model(request.model).stream_generate(request)


@dataclass(frozen=True)
class OpenAICompatibleEmbeddingProvider:
    """Explicit OpenAI-compatible embedding adapter; no model/provider fallback."""

    endpoint: str
    credential_reference: str | None
    secret_store: SecretStore = field(repr=False)
    provider_id: str = "openai-compatible"
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        if self.credential_reference is None and not is_loopback_endpoint(self.endpoint):
            raise ModelProviderError(
                "no-auth embedding access is allowed only for loopback endpoints"
            )
        _validate_provider_timeout("embedding provider", self.timeout_seconds)

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts or any(not text for text in texts):
            raise ModelProviderError("embedding input must contain non-empty texts")
        headers = {"Content-Type": "application/json"}
        if self.credential_reference is not None:
            headers["Authorization"] = f"Bearer {_resolve_provider_credential(self.secret_store, self.credential_reference)}"
        path, body = embedding_request_payload(
            "openai-compatible", model, texts, input_role="document"
        )
        endpoint = self.endpoint
        if self.provider_id == "ollama":
            # Ollama's OpenAI-compatible endpoint truncates long inputs by
            # default. Use its native API and explicitly disable truncation so
            # an oversized paragraph fails visibly instead of embedding only
            # its leading tokens.
            endpoint = re.sub(r"/v1/?$", "", endpoint)
            path = "/api/embed"
            body = {"model": model, "input": texts, "truncate": False}
        request = Request(
            f"{endpoint}{path}",
            _encode_provider_json(body),
            headers=headers,
            method="POST",
        )
        try:
            with _open_provider_request(request, timeout=self.timeout_seconds) as response:
                payload = _parse_provider_json(
                    _read_bounded_provider_response(response).decode("utf-8")
                )
        except HTTPError as exc:
            _raise_provider_http_error(exc, self.provider_id, "embedding")
        except (
            URLError,
            OSError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
        ):
            raise ModelProviderError("embedding provider response could not be read") from None
        try:
            if self.provider_id == "ollama":
                embeddings = payload["embeddings"]
                if not isinstance(embeddings, list):
                    raise TypeError("Ollama embeddings must be a list")
                vectors = [_validated_embedding_vector(item) for item in embeddings]
            else:
                rows = payload["data"]
                if not isinstance(rows, list):
                    raise TypeError("embedding data must be a list")
                indexed = [isinstance(item, dict) and "index" in item for item in rows]
                if any(indexed):
                    if not all(indexed):
                        raise ValueError("embedding indices must be present on every result or none")
                    if any(type(item["index"]) is not int for item in rows):
                        raise ValueError("embedding indices must be integers")
                    rows = sorted(rows, key=lambda item: item["index"])
                    if [item["index"] for item in rows] != list(range(len(texts))):
                        raise ValueError("embedding indices do not match request order")
                vectors = [_validated_embedding_vector(item["embedding"]) for item in rows]
        except (KeyError, TypeError, ValueError):
            raise ModelResponseError("provider response lacks valid ordered embeddings") from None
        if (
            len(vectors) != len(texts)
            or not vectors
            or any(len(vector) != len(vectors[0]) for vector in vectors)
        ):
            raise ModelResponseError("embedding response dimensions do not match request")
        return vectors


@dataclass(frozen=True)
class OpenAIEmbeddingProvider:
    """Direct OpenAI embeddings transport with the official default endpoint."""

    credential_reference: str
    secret_store: SecretStore = field(repr=False)
    endpoint: str = "https://api.openai.com/v1"
    provider_id: str = "openai"
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        if not self.credential_reference:
            raise ModelProviderError("embedding provider requires a credential reference")
        _validate_provider_timeout("embedding provider", self.timeout_seconds)

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        return OpenAICompatibleEmbeddingProvider(
            endpoint=self.endpoint,
            credential_reference=self.credential_reference,
            secret_store=self.secret_store,
            provider_id=self.provider_id,
            timeout_seconds=self.timeout_seconds,
        ).embed(model, texts)


@dataclass(frozen=True)
class CohereEmbeddingProvider:
    """Cohere Embed v2 transport with explicit search-document/query input roles."""

    credential_reference: str
    secret_store: SecretStore = field(repr=False)
    endpoint: str = "https://api.cohere.com"
    provider_id: str = "cohere-embeddings"
    timeout_seconds: float = 60.0

    MAX_INPUTS_PER_REQUEST: ClassVar[int] = 96

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        if not self.credential_reference:
            raise ModelProviderError("Cohere embedding requires a credential reference")
        _validate_provider_timeout("Cohere embedding provider", self.timeout_seconds)

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        """Embed document/passages using Cohere's search_document input type."""
        return self._embed(model, texts, input_type="search_document")

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        """Embed user queries using Cohere's distinct search_query input type."""
        return self._embed(model, texts, input_type="search_query")

    def _embed(self, model: str, texts: list[str], *, input_type: str) -> list[tuple[float, ...]]:
        if not model or not texts or any(not text for text in texts):
            raise ModelProviderError("Cohere embedding requires a model and non-empty texts")
        api_key = _resolve_provider_credential(self.secret_store, self.credential_reference)
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self.MAX_INPUTS_PER_REQUEST):
            batch = texts[start : start + self.MAX_INPUTS_PER_REQUEST]
            path, body = embedding_request_payload(
                self.provider_id, model, batch, input_role=input_type
            )
            request = Request(
                f"{self.endpoint}{path}",
                _encode_provider_json(body),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with _open_provider_request(request, timeout=self.timeout_seconds) as response:
                    payload = _parse_provider_json(
                        _read_bounded_provider_response(response).decode("utf-8")
                    )
            except HTTPError as exc:
                _raise_provider_http_error(
                    exc,
                    self.provider_id,
                    "embedding",
                    additional_auth_codes=(498,),
                )
            except (
                URLError,
                OSError,
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RecursionError,
            ):
                raise ModelProviderError("Cohere embedding endpoint is unavailable") from None
            try:
                batch_vectors = [
                    _validated_embedding_vector(vector) for vector in payload["embeddings"]["float"]
                ]
            except (KeyError, TypeError, ValueError):
                raise ModelResponseError("Cohere response lacks valid float embeddings") from None
            if (
                len(batch_vectors) != len(batch)
                or not batch_vectors
                or any(len(vector) != len(batch_vectors[0]) for vector in batch_vectors)
                or (vectors and len(batch_vectors[0]) != len(vectors[0]))
            ):
                raise ModelResponseError("Cohere embedding dimensions do not match the request")
            vectors.extend(batch_vectors)
        return vectors


@dataclass(frozen=True)
class CohereProvider:
    """Native Cohere Chat v2 adapter with text-only streaming and JSON schema output."""

    credential_reference: str
    secret_store: SecretStore = field(repr=False)
    endpoint: str = "https://api.cohere.com"
    provider_id: str = "cohere"
    capabilities_by_model: dict[str, ModelCapabilities] = field(default_factory=dict)
    timeout_seconds: float = 60.0

    _PARAMETER_NAMES: ClassVar[dict[str, str]] = {
        "max_tokens": "max_tokens",
        "temperature": "temperature",
        "top_p": "p",
        "top_k": "k",
        "stop_sequences": "stop_sequences",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "seed": "seed",
    }

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        _validate_required_provider_auth_and_timeout(
            "Cohere provider", self.credential_reference, self.timeout_seconds
        )

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.capabilities_by_model.get(
            model,
            ModelCapabilities(
                system_prompt=True,
                streaming=True,
                generation_parameters=tuple(self._PARAMETER_NAMES),
            ),
        )

    def _chat_payload(self, request: GenerationRequest, *, streaming: bool) -> dict[str, object]:
        if not request.prompt.strip():
            raise ModelProviderError("generation prompt must be non-empty text")
        capabilities = self.capabilities(request.model)
        if streaming and not capabilities.streaming:
            raise ModelProviderError("provider does not declare streaming support for this model")
        if streaming and request.response_schema is not None:
            raise ModelProviderError("streaming structured output is not supported")
        _require_supported_system_prompt(request, capabilities)
        _require_supported_structured_output(request, capabilities)
        _require_supported_reasoning(request, capabilities)
        _require_supported_reasoning_mode(request, capabilities)
        if request.reasoning is not None:
            # Cohere documents a thinking toggle and a token budget, not effort levels.
            raise ModelProviderError(
                "Cohere Chat v2 maps reasoning modes and budgets, not reasoning effort levels"
            )

        parameters = _require_supported_parameters(request, capabilities)
        unsupported = sorted(set(parameters) - set(self._PARAMETER_NAMES))
        if unsupported:
            raise ModelProviderError(
                "Cohere Chat v2 has no wire mapping for parameter(s): " + ", ".join(unsupported)
            )
        _validate_cohere_parameters(parameters)

        messages: list[dict[str, str]] = []
        if request.system_prompt is not None:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        payload: dict[str, object] = {
            "model": request.model,
            "messages": messages,
            "stream": streaming,
        }
        payload.update({self._PARAMETER_NAMES[name]: value for name, value in parameters.items()})
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_object",
                "schema": request.response_schema,
            }
        if request.reasoning_mode is not None:
            if request.reasoning_mode not in {"enabled", "disabled", "manual"}:
                raise ModelProviderError(
                    "Cohere reasoning mode must be 'enabled', 'disabled', or budgeted 'manual'"
                )
            if request.reasoning_mode == "manual":
                validate_cohere_manual_budget(
                    request.reasoning_budget_tokens,
                    parameters,
                )
                payload["thinking"] = {
                    "type": "enabled",
                    "token_budget": request.reasoning_budget_tokens,
                }
            else:
                if request.reasoning_budget_tokens is not None:
                    raise ModelProviderError(
                        "Cohere reasoning budget requires budgeted 'manual' reasoning mode"
                    )
                payload["thinking"] = {"type": request.reasoning_mode}
        elif request.reasoning_budget_tokens is not None:
            raise ModelProviderError("Cohere reasoning budget requires budgeted 'manual' mode")
        return payload

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload = self._chat_payload(request, streaming=False)
        response = _post_json(
            f"{self.endpoint}/v2/chat",
            payload,
            {
                "Authorization": f"Bearer {_resolve_provider_credential(self.secret_store, self.credential_reference)}"
            },
            self.timeout_seconds,
            provider_id=self.provider_id,
            operation="generation",
            additional_auth_codes=(498,),
        )
        finish_reason = _cohere_finish_reason(response.get("finish_reason"))
        if finish_reason not in {"complete", "stop_sequence"}:
            raise ModelResponseError(
                "Cohere generation did not finish with a complete text response"
            )
        message = response.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ModelResponseError("Cohere response lacks an assistant message")
        content = message.get("content")
        if not isinstance(content, list):
            raise ModelResponseError("Cohere response lacks text content")
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                raise ModelResponseError("Cohere response content block is malformed")
            block_type = block.get("type")
            if block_type == "thinking":
                if not isinstance(block.get("thinking"), str):
                    raise ModelResponseError(
                        "Cohere response contains malformed thinking content"
                    )
                continue
            if block_type != "text":
                raise ModelResponseError("Cohere response contains unsupported non-text content")
            block_text = block.get("text")
            if not isinstance(block_text, str):
                raise ModelResponseError("Cohere response contains malformed text content")
            text_parts.append(block_text)
        if not any(text_parts):
            raise ModelResponseError("Cohere response lacks user-facing text")
        return GenerationResponse(
            "".join(text_parts),
            self.provider_id,
            request.model,
            _cohere_usage(response.get("usage")),
        )

    def stream_generate(self, request: GenerationRequest) -> Iterator[str]:
        payload = self._chat_payload(request, streaming=True)
        http_request = Request(
            f"{self.endpoint}/v2/chat",
            _encode_provider_json(payload),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {_resolve_provider_credential(self.secret_store, self.credential_reference)}",
            },
            method="POST",
        )
        try:
            with _open_provider_request(http_request, timeout=self.timeout_seconds) as response:
                yield from _iter_cohere_stream_text(response)
        except HTTPError as exc:
            _raise_provider_http_error(
                exc,
                self.provider_id,
                "generation",
                additional_auth_codes=(498,),
            )
        except URLError:
            raise ModelProviderError("Cohere endpoint is unavailable") from None
        except OSError:
            raise ModelProviderError("Cohere streaming response could not be read") from None


def _validate_cohere_parameters(parameters: dict[str, object]) -> None:
    for name, value in parameters.items():
        if value is None:
            raise ModelProviderError(f"Cohere {name} must not be null")

    max_tokens = parameters.get("max_tokens")
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1
    ):
        raise ModelProviderError("Cohere max_tokens must be a positive integer")

    for name in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
        value = parameters.get(name)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ModelProviderError(f"Cohere {name} must be a finite number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ModelProviderError(f"Cohere {name} must be a finite number")
        if name == "temperature" and value < 0:
            raise ModelProviderError("Cohere temperature must be non-negative")
        if name == "top_p" and not 0.01 <= value <= 0.99:
            raise ModelProviderError("Cohere top_p must be between 0.01 and 0.99")
        if name in {"frequency_penalty", "presence_penalty"} and not 0 <= value <= 1:
            raise ModelProviderError(f"Cohere {name} must be between 0 and 1")

    top_k = parameters.get("top_k")
    if top_k is not None and (
        not isinstance(top_k, int) or isinstance(top_k, bool) or not 0 <= top_k <= 500
    ):
        raise ModelProviderError("Cohere top_k must be an integer between 0 and 500")

    stop_sequences = parameters.get("stop_sequences")
    if stop_sequences is not None and (
        not isinstance(stop_sequences, list)
        or len(stop_sequences) > 5
        or any(not isinstance(item, str) or not item for item in stop_sequences)
    ):
        raise ModelProviderError(
            "Cohere stop_sequences must contain at most five non-empty strings"
        )

    seed = parameters.get("seed")
    if seed is not None and (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= 18_446_744_073_709_551_615
    ):
        raise ModelProviderError("Cohere seed must be a non-negative unsigned 64-bit integer")


def _cohere_finish_reason(value: object) -> str:
    if not isinstance(value, str):
        raise ModelResponseError("Cohere response has a malformed finish reason")
    finish_reason = value.casefold()
    if finish_reason not in {
        "complete",
        "max_tokens",
        "stop_sequence",
        "tool_call",
        "error",
        "timeout",
    }:
        raise ModelResponseError("Cohere response has an unsupported finish reason")
    return finish_reason


def _cohere_usage(payload: object) -> TokenUsage | None:
    if not isinstance(payload, dict):
        return None
    token_counts = payload.get("tokens")
    if not isinstance(token_counts, dict):
        return TokenUsage()
    return TokenUsage(
        input_tokens=_integer(token_counts, "input_tokens"),
        output_tokens=_integer(token_counts, "output_tokens"),
    )


@dataclass(frozen=True)
class VoyageEmbeddingProvider:
    """Voyage text embeddings with explicit document/query roles and no truncation."""

    credential_reference: str
    secret_store: SecretStore = field(repr=False)
    endpoint: str = "https://api.voyageai.com/v1"
    provider_id: str = "voyage-embeddings"
    timeout_seconds: float = 60.0

    MAX_INPUTS_PER_REQUEST: ClassVar[int] = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        if not self.credential_reference:
            raise ModelProviderError("Voyage embedding requires a credential reference")
        _validate_provider_timeout("Voyage embedding provider", self.timeout_seconds)

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        """Embed passages using Voyage's document retrieval input type."""
        return self._embed(model, texts, input_type="document")

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        """Embed queries using Voyage's query retrieval input type."""
        return self._embed(model, texts, input_type="query")

    def _embed(self, model: str, texts: list[str], *, input_type: str) -> list[tuple[float, ...]]:
        if not model or not texts or any(not text for text in texts):
            raise ModelProviderError("Voyage embedding requires a model and non-empty texts")
        api_key = _resolve_provider_credential(self.secret_store, self.credential_reference)
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self.MAX_INPUTS_PER_REQUEST):
            batch = texts[start : start + self.MAX_INPUTS_PER_REQUEST]
            path, body = embedding_request_payload(
                self.provider_id, model, batch, input_role=input_type
            )
            request = Request(
                f"{self.endpoint}{path}",
                _encode_provider_json(body),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with _open_provider_request(request, timeout=self.timeout_seconds) as response:
                    payload = _parse_provider_json(
                        _read_bounded_provider_response(response).decode("utf-8")
                    )
            except HTTPError as exc:
                _raise_provider_http_error(exc, self.provider_id, "embedding")
            except (
                URLError,
                OSError,
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RecursionError,
            ):
                raise ModelProviderError("Voyage embedding endpoint is unavailable") from None
            try:
                rows = payload["data"]
                if not isinstance(rows, list):
                    raise TypeError("embedding data must be a list")
                indexed = [isinstance(item, dict) and "index" in item for item in rows]
                if any(indexed):
                    if not all(indexed):
                        raise ValueError(
                            "embedding indices must be present on every result or none"
                        )
                    if any(type(item["index"]) is not int for item in rows):
                        raise ValueError("embedding indices must be integers")
                    rows = sorted(rows, key=lambda item: item["index"])
                    if [item["index"] for item in rows] != list(range(len(batch))):
                        raise ValueError("embedding indices do not match input order")
                batch_vectors = [_validated_embedding_vector(item["embedding"]) for item in rows]
            except (KeyError, TypeError, ValueError):
                raise ModelResponseError("Voyage response lacks valid ordered embeddings") from None
            if (
                len(batch_vectors) != len(batch)
                or not batch_vectors
                or any(len(vector) != len(batch_vectors[0]) for vector in batch_vectors)
                or (vectors and len(batch_vectors[0]) != len(vectors[0]))
            ):
                raise ModelResponseError("Voyage embedding dimensions do not match request")
            vectors.extend(batch_vectors)
        return vectors


@dataclass(frozen=True)
class JinaEmbeddingProvider:
    """Jina dense text embeddings with explicit retrieval-side task provenance."""

    credential_reference: str
    secret_store: SecretStore = field(repr=False)
    endpoint: str = "https://api.jina.ai/v1"
    provider_id: str = "jina-embeddings"
    timeout_seconds: float = 60.0

    MAX_INPUTS_PER_REQUEST: ClassVar[int] = 128
    _TASK_MODEL_PREFIXES: ClassVar[tuple[str, ...]] = (
        "jina-embeddings-v3",
        "jina-embeddings-v4",
        "jina-embeddings-v5",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        if not self.credential_reference:
            raise ModelProviderError("Jina embeddings require a credential reference")
        _validate_provider_timeout("Jina embedding provider", self.timeout_seconds)

    @classmethod
    def supports_retrieval_tasks(cls, model: str) -> bool:
        """Use task-specific LoRA only for Jina model families documenting that option."""
        return model.casefold().startswith(cls._TASK_MODEL_PREFIXES)

    @classmethod
    def document_input_type(cls, model: str) -> str:
        """Return the persisted input role used to construct a vector index."""
        return "retrieval.passage" if cls.supports_retrieval_tasks(model) else "symmetric"

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        task = "retrieval.passage" if self.supports_retrieval_tasks(model) else None
        return self._embed(model, texts, task=task)

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        task = "retrieval.query" if self.supports_retrieval_tasks(model) else None
        return self._embed(model, texts, task=task)

    def _embed(self, model: str, texts: list[str], *, task: str | None) -> list[tuple[float, ...]]:
        if not model or not texts or any(not text for text in texts):
            raise ModelProviderError("Jina embedding requires a model and non-empty texts")
        api_key = _resolve_provider_credential(self.secret_store, self.credential_reference)
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self.MAX_INPUTS_PER_REQUEST):
            batch = texts[start : start + self.MAX_INPUTS_PER_REQUEST]
            input_role = "document" if task in {None, "retrieval.passage"} else "query"
            path, body = embedding_request_payload(
                self.provider_id, model, batch, input_role=input_role
            )
            request = Request(
                f"{self.endpoint}{path}",
                _encode_provider_json(body),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with _open_provider_request(request, timeout=self.timeout_seconds) as response:
                    payload = _parse_provider_json(
                        _read_bounded_provider_response(response).decode("utf-8")
                    )
            except HTTPError as exc:
                _raise_provider_http_error(exc, self.provider_id, "embedding")
            except (
                URLError,
                OSError,
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RecursionError,
            ):
                raise ModelProviderError("Jina embedding endpoint is unavailable") from None
            try:
                rows = payload["data"]
                if not isinstance(rows, list):
                    raise TypeError("embedding data must be a list")
                indexed = [isinstance(item, dict) and "index" in item for item in rows]
                if any(indexed) and not all(indexed):
                    raise ValueError("embedding indices must be present on every result or none")
                if indexed and rows:
                    if any(type(item["index"]) is not int for item in rows):
                        raise ValueError("embedding indices must be integers")
                    rows = sorted(rows, key=lambda item: item["index"])
                    if [item["index"] for item in rows] != list(range(len(batch))):
                        raise ValueError("embedding indices do not match request order")
                batch_vectors: list[tuple[float, ...]] = []
                for item in rows:
                    batch_vectors.append(_validated_embedding_vector(item["embedding"]))
            except (KeyError, TypeError, ValueError):
                raise ModelResponseError("Jina response lacks valid ordered embeddings") from None
            if (
                len(batch_vectors) != len(batch)
                or not batch_vectors
                or any(not vector for vector in batch_vectors)
                or any(len(vector) != len(batch_vectors[0]) for vector in batch_vectors)
                or (vectors and len(batch_vectors[0]) != len(vectors[0]))
            ):
                raise ModelResponseError("Jina embedding dimensions do not match the request")
            vectors.extend(batch_vectors)
        return vectors


@dataclass(frozen=True)
class GeminiEmbeddingProvider:
    """Gemini text embeddings with model-specific retrieval formatting."""

    credential_reference: str
    secret_store: SecretStore = field(repr=False)
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta"
    provider_id: str = "gemini-embeddings"
    timeout_seconds: float = 60.0

    MAX_INPUTS_PER_REQUEST: ClassVar[int] = 96
    _TASK_TYPE_MODEL: ClassVar[str] = "gemini-embedding-001"
    _PREFIX_MODEL: ClassVar[str] = "gemini-embedding-2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        if not self.credential_reference:
            raise ModelProviderError("Gemini embeddings require a credential reference")
        _validate_provider_timeout("Gemini embedding provider", self.timeout_seconds)

    @classmethod
    def document_input_type(cls, model: str) -> str:
        """Identify the exact document-side embedding preparation stored in provenance."""
        if model == cls._TASK_TYPE_MODEL:
            return "RETRIEVAL_DOCUMENT"
        if model == cls._PREFIX_MODEL:
            return "title-none-text-v1"
        raise ModelProviderError(
            "unsupported Gemini embedding model; supported models are explicit"
        )

    @classmethod
    def document_request_item(cls, model: str, text: str) -> dict[str, object]:
        """Build the exact document-side batch request item without network access."""
        if model == cls._TASK_TYPE_MODEL:
            prepared_text = text
            config: dict[str, object] = {
                "autoTruncate": False,
                "taskType": "RETRIEVAL_DOCUMENT",
            }
        elif model == cls._PREFIX_MODEL:
            prepared_text = f"title: none | text: {text}"
            config = {"autoTruncate": False}
        else:
            raise ModelProviderError(
                "unsupported Gemini embedding model; supported models are explicit"
            )
        return {
            "model": f"models/{model}",
            "content": {"parts": [{"text": prepared_text}]},
            "embedContentConfig": config,
        }

    @classmethod
    def query_request_item(cls, model: str, text: str) -> dict[str, object]:
        """Build the exact query-side batch request item without network access."""
        if model == cls._TASK_TYPE_MODEL:
            prepared_text = text
            config: dict[str, object] = {
                "autoTruncate": False,
                "taskType": "RETRIEVAL_QUERY",
            }
        elif model == cls._PREFIX_MODEL:
            prepared_text = f"task: search result | query: {text}"
            config = {"autoTruncate": False}
        else:
            raise ModelProviderError(
                "unsupported Gemini embedding model; supported models are explicit"
            )
        return {
            "model": f"models/{model}",
            "content": {"parts": [{"text": prepared_text}]},
            "embedContentConfig": config,
        }

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        return self._embed(model, texts, query=False)

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        return self._embed(model, texts, query=True)

    def _embed(
        self,
        model: str,
        texts: list[str],
        *,
        query: bool,
    ) -> list[tuple[float, ...]]:
        if not texts or any(not text for text in texts):
            raise ModelProviderError("Gemini embedding input must contain non-empty texts")
        # Fail closed: task semantics changed between these model generations, and an
        # unknown model must not silently receive a guessed asymmetric/symmetric format.
        self.document_input_type(model)
        api_key = _resolve_provider_credential(self.secret_store, self.credential_reference)
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self.MAX_INPUTS_PER_REQUEST):
            batch = texts[start : start + self.MAX_INPUTS_PER_REQUEST]
            path, body = embedding_request_payload(
                self.provider_id, model, batch, input_role="query" if query else "document"
            )
            request = Request(
                f"{self.endpoint}{path}",
                _encode_provider_json(body),
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with _open_provider_request(request, timeout=self.timeout_seconds) as response:
                    payload = _parse_provider_json(
                        _read_bounded_provider_response(response).decode("utf-8")
                    )
            except HTTPError as exc:
                _raise_provider_http_error(exc, self.provider_id, "embedding")
            except (
                URLError,
                OSError,
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RecursionError,
            ):
                raise ModelProviderError("Gemini embedding endpoint is unavailable") from None
            try:
                rows = payload["embeddings"]
                if not isinstance(rows, list) or len(rows) != len(batch):
                    raise ValueError("embedding response count does not match request")
                batch_vectors: list[tuple[float, ...]] = []
                for row in rows:
                    batch_vectors.append(_validated_embedding_vector(row["values"]))
            except (KeyError, TypeError, ValueError):
                raise ModelResponseError("Gemini response lacks valid ordered embeddings") from None
            if (
                not batch_vectors
                or any(not vector for vector in batch_vectors)
                or any(len(vector) != len(batch_vectors[0]) for vector in batch_vectors)
                or (vectors and len(batch_vectors[0]) != len(vectors[0]))
            ):
                raise ModelResponseError("Gemini embedding dimensions do not match the request")
            vectors.extend(batch_vectors)
        return vectors


def embedding_request_payload(
    provider_id: str,
    model: str,
    texts: list[str],
    *,
    input_role: str,
) -> tuple[str, dict[str, object]]:
    """Shared provider wire contract for adapters and inspectable request previews."""
    if not model or not texts or any(not text for text in texts):
        raise ModelProviderError("embedding request requires a model and non-empty texts")
    if input_role not in {"document", "query", "search_document", "search_query"}:
        raise ModelProviderError("embedding request has an unsupported input role")
    is_query = input_role in {"query", "search_query"}
    if provider_id == "ollama":
        return "/api/embed", {"model": model, "input": texts, "truncate": False}
    if provider_id == "openai" or is_openai_compatible_provider(provider_id):
        return "/embeddings", {"model": model, "input": texts}
    if provider_id == "cohere-embeddings":
        return "/v2/embed", {
            "model": model,
            "texts": texts,
            "input_type": "search_query" if is_query else "search_document",
            "embedding_types": ["float"],
            "truncate": "NONE",
        }
    if provider_id == "voyage-embeddings":
        return "/embeddings", {
            "input": texts,
            "model": model,
            "input_type": "query" if is_query else "document",
            "truncation": False,
            "output_dtype": "float",
        }
    if provider_id == "jina-embeddings":
        body: dict[str, object] = {"model": model, "input": texts, "embedding_type": "float"}
        if JinaEmbeddingProvider.supports_retrieval_tasks(model):
            body["task"] = "retrieval.query" if is_query else "retrieval.passage"
        return "/embeddings", body
    if provider_id == "gemini-embeddings":
        item_builder = (
            GeminiEmbeddingProvider.query_request_item
            if is_query
            else GeminiEmbeddingProvider.document_request_item
        )
        return f"/models/{model}:batchEmbedContents", {
            "requests": [item_builder(model, text) for text in texts]
        }
    raise ModelProviderError("provider has no registered embedding request contract")


@dataclass(frozen=True)
class JinaRerankingProvider:
    """Jina hosted reranker over a caller-supplied, already scoped candidate list."""

    credential_reference: str
    secret_store: SecretStore = field(repr=False)
    endpoint: str = "https://api.jina.ai/v1"
    provider_id: str = "jina-reranker"
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        if not self.credential_reference:
            raise ModelProviderError("reranking provider requires a credential reference")
        _validate_provider_timeout("reranking provider", self.timeout_seconds)

    def rerank(
        self,
        model: str,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Return provider ranking indices; never fetches documents or follows redirects."""
        if (
            not isinstance(model, str)
            or not model.strip()
            or not isinstance(query, str)
            or not query.strip()
            or not isinstance(documents, list)
            or not documents
            or any(not isinstance(item, str) or not item.strip() for item in documents)
        ):
            raise ModelProviderError("reranking requires a model, query, and non-empty candidates")
        if top_n is not None and (not isinstance(top_n, int) or isinstance(top_n, bool)):
            raise ModelProviderError("reranking top_n must be within the candidate count")
        requested_top_n = len(documents) if top_n is None else top_n
        if not 1 <= requested_top_n <= len(documents):
            raise ModelProviderError("reranking top_n must be within the candidate count")
        api_key = _resolve_provider_credential(self.secret_store, self.credential_reference)
        request = Request(
            f"{self.endpoint}/rerank",
            _encode_provider_json(
                {
                    "model": model,
                    "query": query,
                    "documents": documents,
                    "top_n": requested_top_n,
                    "return_documents": False,
                }
            ),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with _open_provider_request(request, timeout=self.timeout_seconds) as response:
                payload = _parse_provider_json(
                    _read_bounded_provider_response(response).decode("utf-8")
                )
        except HTTPError as exc:
            _raise_provider_http_error(exc, self.provider_id, "reranking")
        except (
            URLError,
            OSError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
        ):
            raise ModelProviderError("Jina reranking endpoint is unavailable") from None
        try:
            raw_results = payload["results"]
            results: list[RerankResult] = []
            seen: set[int] = set()
            previous_score: float | None = None
            for item in raw_results:
                index = item["index"]
                score = item["relevance_score"]
                if (
                    type(index) is not int
                    or not isinstance(score, (int, float))
                    or isinstance(score, bool)
                ):
                    raise ValueError("invalid result types")
                numeric_score = float(score)
                if (
                    index < 0
                    or index >= len(documents)
                    or index in seen
                    or not math.isfinite(numeric_score)
                    or (previous_score is not None and numeric_score > previous_score)
                ):
                    raise ValueError("invalid result index or score")
                seen.add(index)
                results.append(RerankResult(index, numeric_score))
                previous_score = numeric_score
            if not results or len(results) > requested_top_n:
                raise ValueError("result count is outside requested top_n")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ModelResponseError(
                "Jina response lacks valid reranking indices and scores"
            ) from exc
        return results


@dataclass(frozen=True)
class CohereRerankingProvider:
    """Cohere Rerank v2 over a caller-supplied, already scoped candidate list."""

    credential_reference: str
    secret_store: SecretStore = field(repr=False)
    endpoint: str = "https://api.cohere.com"
    provider_id: str = "cohere-reranker"
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        if not self.credential_reference:
            raise ModelProviderError("reranking provider requires a credential reference")
        _validate_provider_timeout("reranking provider", self.timeout_seconds)

    def rerank(
        self,
        model: str,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Return Cohere's order for at most 1000 explicitly supplied passages."""
        if (
            not model.strip()
            or not query.strip()
            or not documents
            or any(not isinstance(item, str) or not item.strip() for item in documents)
        ):
            raise ModelProviderError("reranking requires a model, query, and non-empty candidates")
        if len(documents) > 1_000:
            raise ModelProviderError("Cohere reranking accepts at most 1000 candidates per request")
        requested_top_n = len(documents) if top_n is None else top_n
        if (
            not isinstance(requested_top_n, int)
            or isinstance(requested_top_n, bool)
            or not 1 <= requested_top_n <= len(documents)
        ):
            raise ModelProviderError("reranking top_n must be within the candidate count")
        api_key = _resolve_provider_credential(self.secret_store, self.credential_reference)
        request = Request(
            f"{self.endpoint}/v2/rerank",
            _encode_provider_json(
                {
                    "model": model,
                    "query": query,
                    "documents": documents,
                    "top_n": requested_top_n,
                    "max_tokens_per_doc": 4096,
                }
            ),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with _open_provider_request(request, timeout=self.timeout_seconds) as response:
                payload = _parse_provider_json(
                    _read_bounded_provider_response(response).decode("utf-8")
                )
        except HTTPError as exc:
            _raise_provider_http_error(
                exc, self.provider_id, "reranking", additional_auth_codes=(498,)
            )
        except (
            URLError,
            OSError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
        ):
            raise ModelProviderError("Cohere reranking endpoint is unavailable") from None
        try:
            raw_results = payload["results"]
            results: list[RerankResult] = []
            seen: set[int] = set()
            previous_score: float | None = None
            for item in raw_results:
                index = item["index"]
                score = item["relevance_score"]
                if (
                    type(index) is not int
                    or not isinstance(score, (int, float))
                    or isinstance(score, bool)
                ):
                    raise ValueError("invalid result types")
                numeric_score = float(score)
                if (
                    index < 0
                    or index >= len(documents)
                    or index in seen
                    or not math.isfinite(numeric_score)
                    or (previous_score is not None and numeric_score > previous_score)
                ):
                    raise ValueError("invalid result index or score")
                seen.add(index)
                results.append(RerankResult(index, numeric_score))
                previous_score = numeric_score
            if not results or len(results) > requested_top_n:
                raise ValueError("result count is outside requested top_n")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ModelResponseError(
                "Cohere response lacks valid reranking indices and scores"
            ) from exc
        return results


def _anthropic_messages_stream(
    *,
    endpoint: str,
    credential_reference: str,
    secret_store: SecretStore,
    provider_id: str,
    timeout_seconds: float,
    capabilities: ModelCapabilities,
    request: GenerationRequest,
) -> Iterator[str]:
    """Run the shared text-only Messages SSE contract for native/compatible APIs."""
    if request.image is not None:
        raise ModelProviderError("Anthropic Messages image input is not supported with streaming")
    if not request.prompt:
        raise ModelProviderError("generation prompt must not be empty")
    if not capabilities.streaming:
        raise ModelProviderError("provider does not declare streaming support for this model")
    if request.response_schema is not None:
        raise ModelProviderError("streaming structured output is not supported")
    _require_supported_system_prompt(request, capabilities)
    _require_supported_structured_output(request, capabilities)
    parameters = _require_supported_parameters(request, capabilities)
    mapped_parameters = ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences")
    unmapped_parameters = sorted(set(parameters) - set(mapped_parameters))
    if unmapped_parameters:
        raise ModelProviderError(
            "Anthropic Messages stream has no wire mapping for parameter(s): "
            + ", ".join(unmapped_parameters)
        )

    payload: dict[str, object] = {
        "model": request.model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": True,
    }
    if request.system_prompt is not None:
        payload["system"] = request.system_prompt
    for name in mapped_parameters:
        if name in parameters:
            payload[name] = parameters[name]
    output_config: dict[str, object] = {}
    _map_anthropic_thinking_configuration(
        request,
        capabilities,
        parameters,
        payload,
        output_config,
        provider_name=(
            "Anthropic-compatible" if provider_id == "anthropic-compatible" else "Anthropic"
        ),
    )
    if output_config:
        payload["output_config"] = output_config

    http_request = Request(
        f"{endpoint}/messages",
        _encode_provider_json(payload),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "x-api-key": _resolve_provider_credential(secret_store, credential_reference),
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with _open_provider_request(http_request, timeout=timeout_seconds) as response:
            yield from _iter_anthropic_stream_text(response)
    except HTTPError as exc:
        _raise_provider_http_error(exc, provider_id, "generation")
    except URLError:
        raise ModelProviderError("provider endpoint is unavailable") from None
    except (OSError, ValueError, RecursionError):
        raise ModelProviderError("provider streaming response could not be read") from None


def _map_anthropic_thinking_configuration(
    request: GenerationRequest,
    capabilities: ModelCapabilities,
    parameters: dict[str, object],
    payload: dict[str, object],
    output_config: dict[str, object],
    *,
    provider_name: str,
) -> None:
    """Validate and map exact-profiled Anthropic thinking options to Messages fields."""
    _require_supported_reasoning(
        request, capabilities, native_effort_values=_ANTHROPIC_REASONING_EFFORT_WIRE_VALUES
    )
    _require_supported_reasoning_mode(request, capabilities)
    if request.reasoning_budget_tokens is not None and request.reasoning_mode != "manual":
        raise ModelProviderError("reasoning budget requires manual reasoning mode")
    if request.reasoning and not request.reasoning_mode:
        raise ModelProviderError(
            f"{provider_name} reasoning effort requires an explicit model-specific reasoning mode"
        )
    if not request.reasoning_mode:
        return

    temperature = parameters.get("temperature")
    if temperature is not None and temperature != 1:
        raise ModelProviderError(
            f"{provider_name} thinking requires temperature 1 or an omitted temperature"
        )
    if request.reasoning_mode == "adaptive":
        payload["thinking"] = {"type": "adaptive"}
        if request.reasoning:
            output_config["effort"] = request.reasoning
    elif request.reasoning_mode == "manual":
        budget = request.reasoning_budget_tokens
        max_tokens = payload["max_tokens"]
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1024:
            raise ModelProviderError(
                f"{provider_name} manual thinking requires a budget of at least 1024 tokens"
            )
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or budget >= max_tokens:
            raise ModelProviderError("manual reasoning budget must be less than max_tokens")
        payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        if request.reasoning:
            output_config["effort"] = request.reasoning
    else:
        raise ModelProviderError(f"unsupported {provider_name} reasoning mode")


def _iter_cohere_stream_text(response: Any) -> Iterator[str]:
    """Yield Cohere text deltas and reject incomplete or non-text completions."""
    message_started = False
    finished = False
    saw_text = False
    content_blocks: dict[int, str] = {}

    for event_data in _iter_server_sent_data(response):
        try:
            payload = _parse_provider_json(event_data)
        except (ValueError, RecursionError):
            raise ModelResponseError("Cohere stream contains malformed JSON") from None
        if not isinstance(payload, dict):
            raise ModelResponseError("Cohere stream event must be a JSON object")
        if "error" in payload:
            raise ModelResponseError("Cohere returned a streaming error event") from None
        event_type = payload.get("type")
        if not isinstance(event_type, str):
            raise ModelResponseError("Cohere stream event has no valid type")

        if event_type == "message-start":
            delta = payload.get("delta")
            message = delta.get("message") if isinstance(delta, dict) else None
            if (
                message_started
                or not isinstance(message, dict)
                or message.get("role") != "assistant"
            ):
                raise ModelResponseError("Cohere stream message start is malformed")
            message_started = True
        elif event_type == "content-start":
            if not message_started or finished:
                raise ModelResponseError("Cohere stream content began outside a message")
            index = payload.get("index")
            delta = payload.get("delta")
            message = delta.get("message") if isinstance(delta, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            block_type = content.get("type") if isinstance(content, dict) else None
            if (
                type(index) is not int
                or index < 0
                or index in content_blocks
                or block_type not in {"text", "thinking"}
            ):
                raise ModelResponseError("Cohere stream content start is malformed")
            content_blocks[index] = block_type
        elif event_type == "content-delta":
            if not message_started or finished:
                raise ModelResponseError("Cohere stream content delta is out of sequence")
            index = payload.get("index")
            if type(index) is not int or index not in content_blocks:
                raise ModelResponseError("Cohere stream content delta has an unknown index")
            if content_blocks[index] == "thinking":
                continue
            delta = payload.get("delta")
            message = delta.get("message") if isinstance(delta, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            text = content.get("text") if isinstance(content, dict) else None
            if not isinstance(text, str):
                raise ModelResponseError("Cohere stream text delta is malformed")
            if text:
                saw_text = True
                yield text
        elif event_type == "content-end":
            index = payload.get("index")
            if not message_started or type(index) is not int or index not in content_blocks:
                raise ModelResponseError("Cohere stream content end is malformed")
            del content_blocks[index]
        elif event_type == "message-end":
            delta = payload.get("delta")
            if not message_started or not isinstance(delta, dict) or content_blocks:
                raise ModelResponseError("Cohere stream message end is malformed")
            finish_reason = _cohere_finish_reason(delta.get("finish_reason"))
            if finish_reason not in {"complete", "stop_sequence"}:
                raise ModelResponseError(
                    "Cohere stream did not finish with a complete text response"
                )
            finished = True
            break
        elif event_type == "debug":
            # Debug events carry metadata, never user-facing text.
            continue
        else:
            # Tool, citation, and unknown events are deliberately outside this
            # adapter's text-only contract.
            raise ModelResponseError("Cohere stream contains an unsupported non-text event")

    if not finished:
        raise ModelResponseError("Cohere stream ended without a successful message-end event")
    if not saw_text:
        raise ModelResponseError("Cohere stream completed without user-facing text")


def _iter_gemini_stream_text(response: Any) -> Iterator[str]:
    """Yield Gemini candidate text, excluding thoughts and refusing blocked output."""
    saw_text = False
    finished = False
    blocked_reasons = {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "MODEL_ARMOR",
    }

    for event_data in _iter_server_sent_data(response):
        try:
            payload = _parse_provider_json(event_data)
        except (ValueError, RecursionError):
            raise ModelResponseError("Gemini stream contains malformed JSON") from None
        if not isinstance(payload, dict):
            raise ModelResponseError("Gemini stream event must be a JSON object")
        if "error" in payload:
            # Provider-controlled messages may echo prompts or other sensitive data.
            raise ModelResponseError("Gemini returned a streaming error event") from None

        prompt_feedback = payload.get("promptFeedback")
        if prompt_feedback is not None:
            if not isinstance(prompt_feedback, dict):
                raise ModelResponseError("Gemini stream prompt feedback is malformed")
            block_reason = prompt_feedback.get("blockReason")
            if block_reason is not None and (not isinstance(block_reason, str) or not block_reason):
                raise ModelResponseError("Gemini stream prompt feedback is malformed")
            if block_reason and block_reason != "BLOCK_REASON_UNSPECIFIED":
                raise ModelResponseError("Gemini blocked the prompt under a safety policy")

        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ModelResponseError("Gemini stream event must contain one candidate")
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise ModelResponseError("Gemini stream candidate is malformed")

        safety_ratings = candidate.get("safetyRatings")
        if safety_ratings is not None:
            if not isinstance(safety_ratings, list) or any(
                not isinstance(rating, dict) for rating in safety_ratings
            ):
                raise ModelResponseError("Gemini stream safety metadata is malformed")
            for rating in safety_ratings:
                blocked = rating.get("blocked")
                if blocked is not None and type(blocked) is not bool:
                    raise ModelResponseError("Gemini stream safety metadata is malformed")
                if blocked is True:
                    raise ModelResponseError("Gemini blocked the candidate under a safety policy")

        finish_reason = candidate.get("finishReason")
        terminal = finish_reason is not None
        if terminal:
            if not isinstance(finish_reason, str) or not finish_reason:
                raise ModelResponseError("Gemini stream finish reason is malformed")
            if finish_reason in blocked_reasons:
                raise ModelResponseError("Gemini blocked the candidate under a safety policy")
            if finish_reason == "MAX_TOKENS":
                raise ModelResponseError(
                    "Gemini stream reached the configured output token limit before completion"
                )
            if finish_reason != "STOP":
                raise ModelResponseError("Gemini stream ended with an unsupported finish reason")

        content = candidate.get("content")
        text_parts: list[str] = []
        if content is not None:
            if not isinstance(content, dict):
                raise ModelResponseError("Gemini stream candidate content is malformed")
            parts = content.get("parts")
            if not isinstance(parts, list):
                raise ModelResponseError("Gemini stream candidate parts are malformed")
            for part in parts:
                if not isinstance(part, dict):
                    raise ModelResponseError("Gemini stream candidate part is malformed")
                thought = part.get("thought")
                if thought is not None and type(thought) is not bool:
                    raise ModelResponseError("Gemini stream thought marker is malformed")
                text = part.get("text")
                if thought is True or text is None:
                    # Thought parts and function-call / media parts are not user text.
                    continue
                if not isinstance(text, str):
                    raise ModelResponseError("Gemini stream text part is malformed")
                if text:
                    text_parts.append(text)

        for text in text_parts:
            saw_text = True
            yield text
        if terminal:
            finished = True
            break

    if not finished:
        raise ModelResponseError("Gemini stream ended without a finish reason")
    if not saw_text:
        raise ModelResponseError("Gemini stream completed without user-facing text")


@dataclass(frozen=True)
class AnthropicProvider:
    """Small native Anthropic Messages adapter behind the common gateway contract."""

    credential_reference: str
    secret_store: SecretStore = field(repr=False)
    endpoint: str = "https://api.anthropic.com/v1"
    provider_id: str = "anthropic"
    capabilities_by_model: dict[str, ModelCapabilities] = field(default_factory=dict)
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        _validate_required_provider_auth_and_timeout(
            "Anthropic provider", self.credential_reference, self.timeout_seconds
        )

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.capabilities_by_model.get(
            model,
            ModelCapabilities(
                # JSON-schema support varies by exact model and schema subset.
                # A transport mapping alone is not a model capability claim;
                # projects must opt in through an exact-model profile.
                structured_output=False,
                streaming=True,
                generation_parameters=(
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "stop_sequences",
                ),
            ),
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not request.prompt:
            raise ModelProviderError("generation prompt must not be empty")
        capabilities = self.capabilities(request.model)
        if request.image is not None and not capabilities.vision:
            raise ModelProviderError(
                "Anthropic image input requires vision support for this exact model"
            )
        _require_supported_system_prompt(request, capabilities)
        _require_supported_structured_output(request, capabilities)
        payload: dict[str, object] = {
            "model": request.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": _anthropic_user_content(request)}],
        }
        output_config: dict[str, object] = {}
        if request.system_prompt is not None:
            payload["system"] = request.system_prompt
        if request.response_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": request.response_schema,
            }
        parameters = _require_supported_parameters(request, self.capabilities(request.model))
        for name in ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences"):
            if name in parameters:
                payload[name] = parameters[name]
        _map_anthropic_thinking_configuration(
            request,
            capabilities,
            parameters,
            payload,
            output_config,
            provider_name="Anthropic",
        )
        if output_config:
            payload["output_config"] = output_config
        if request.image is not None:
            _validate_anthropic_image_request_size(payload)
        response = _post_json(
            f"{self.endpoint}/messages",
            payload,
            {
                "x-api-key": _resolve_provider_credential(
                    self.secret_store, self.credential_reference
                ),
                "anthropic-version": "2023-06-01",
            },
            self.timeout_seconds,
            provider_id=self.provider_id,
            operation="generation",
        )
        _require_complete_anthropic_response(response, self.provider_id)
        content = response.get("content")
        if not isinstance(content, list) or any(not isinstance(item, dict) for item in content):
            raise ModelResponseError("Anthropic response lacks text content") from None
        text_block = next((item for item in content if item.get("type") == "text"), None)
        if text_block is None:
            raise ModelResponseError("Anthropic response lacks text content") from None
        text = text_block.get("text")
        if not isinstance(text, str):
            raise ModelResponseError("Anthropic completion content must be text")
        usage_data = response.get("usage")
        usage = (
            TokenUsage(
                input_tokens=_integer(usage_data, "input_tokens"),
                output_tokens=_integer(usage_data, "output_tokens"),
            )
            if isinstance(usage_data, dict)
            else None
        )
        return GenerationResponse(text, self.provider_id, request.model, usage)

    def stream_generate(self, request: GenerationRequest) -> Iterator[str]:
        yield from _anthropic_messages_stream(
            endpoint=self.endpoint,
            credential_reference=self.credential_reference,
            secret_store=self.secret_store,
            provider_id=self.provider_id,
            timeout_seconds=self.timeout_seconds,
            capabilities=self.capabilities(request.model),
            request=request,
        )


@dataclass(frozen=True)
class AnthropicCompatibleProvider:
    """Explicit endpoint adapter for the Anthropic Messages wire format.

    Unlike :class:`AnthropicProvider`, this adapter never supplies an endpoint.
    Although it implements the Messages streaming wire mapping, saved-preset
    execution still requires an exact-model capability declaration because
    compatible services may implement only a subset of the API.
    """

    endpoint: str
    credential_reference: str
    secret_store: SecretStore = field(repr=False)
    provider_id: str = "anthropic-compatible"
    capabilities_by_model: dict[str, ModelCapabilities] = field(default_factory=dict)
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        _validate_required_provider_auth_and_timeout(
            "Anthropic-compatible provider", self.credential_reference, self.timeout_seconds
        )

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.capabilities_by_model.get(
            model,
            ModelCapabilities(
                system_prompt=True,
                structured_output=False,
                streaming=True,
                generation_parameters=(
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "stop_sequences",
                ),
            ),
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not request.prompt:
            raise ModelProviderError("generation prompt must not be empty")
        capabilities = self.capabilities(request.model)
        if request.image is not None and not capabilities.vision:
            raise ModelProviderError(
                "Anthropic-compatible image input requires vision support for this exact model"
            )
        _require_supported_system_prompt(request, capabilities)
        _require_supported_structured_output(request, capabilities)

        payload: dict[str, object] = {
            "model": request.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": _anthropic_user_content(request)}],
        }
        output_config: dict[str, object] = {}
        if request.system_prompt is not None:
            payload["system"] = request.system_prompt
        if request.response_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": request.response_schema,
            }
        parameters = _require_supported_parameters(request, capabilities)
        mapped_parameters = {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
        }
        unmapped_parameters = sorted(set(parameters) - mapped_parameters)
        if unmapped_parameters:
            raise ModelProviderError(
                "Anthropic-compatible Messages adapter has no wire mapping for parameter(s): "
                + ", ".join(unmapped_parameters)
            )
        for name in ("max_tokens", "temperature", "top_p", "top_k", "stop_sequences"):
            if name in parameters:
                payload[name] = parameters[name]
        _map_anthropic_thinking_configuration(
            request,
            capabilities,
            parameters,
            payload,
            output_config,
            provider_name="Anthropic-compatible",
        )

        if output_config:
            payload["output_config"] = output_config
        if request.image is not None:
            _validate_anthropic_image_request_size(payload)

        response = _post_json(
            f"{self.endpoint}/messages",
            payload,
            {
                "x-api-key": _resolve_provider_credential(
                    self.secret_store, self.credential_reference
                ),
                "anthropic-version": "2023-06-01",
            },
            self.timeout_seconds,
            provider_id=self.provider_id,
            operation="generation",
        )
        _require_complete_anthropic_response(response, self.provider_id)
        content = response.get("content")
        if not isinstance(content, list) or not content:
            raise ModelResponseError(
                "Anthropic-compatible response lacks valid text content"
            ) from None

        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                raise ModelResponseError(
                    "Anthropic-compatible response contains a malformed content block"
                ) from None
            block_type = item.get("type")
            if block_type == "text":
                block_text = item.get("text")
                if not isinstance(block_text, str):
                    raise ModelResponseError(
                        "Anthropic-compatible response contains malformed text content"
                    ) from None
                text_parts.append(block_text)
            elif block_type == "thinking":
                if not request.reasoning_mode or not isinstance(item.get("thinking"), str):
                    raise ModelResponseError(
                        "Anthropic-compatible response contains an unexpected reasoning block"
                    ) from None
            elif block_type == "redacted_thinking":
                if not request.reasoning_mode or not isinstance(item.get("data"), str):
                    raise ModelResponseError(
                        "Anthropic-compatible response contains an unexpected reasoning block"
                    ) from None
            else:
                raise ModelResponseError(
                    "Anthropic-compatible response contains an unsupported content block"
                ) from None
        if not text_parts:
            raise ModelResponseError(
                "Anthropic-compatible response lacks valid text content"
            ) from None

        usage_data = response.get("usage")
        usage: TokenUsage | None = None
        if usage_data is not None:
            if not isinstance(usage_data, dict):
                raise ModelResponseError(
                    "Anthropic-compatible response has malformed usage"
                ) from None
            input_tokens = usage_data.get("input_tokens")
            output_tokens = usage_data.get("output_tokens")
            if (
                type(input_tokens) is not int
                or input_tokens < 0
                or type(output_tokens) is not int
                or output_tokens < 0
            ):
                raise ModelResponseError(
                    "Anthropic-compatible response has malformed usage"
                ) from None
            usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        return GenerationResponse("".join(text_parts), self.provider_id, request.model, usage)

    def stream_generate(self, request: GenerationRequest) -> Iterator[str]:
        yield from _anthropic_messages_stream(
            endpoint=self.endpoint,
            credential_reference=self.credential_reference,
            secret_store=self.secret_store,
            provider_id=self.provider_id,
            timeout_seconds=self.timeout_seconds,
            capabilities=self.capabilities(request.model),
            request=request,
        )


@dataclass(frozen=True)
class GeminiProvider:
    """Native Gemini generateContent adapter using a secret in a request header."""

    credential_reference: str
    secret_store: SecretStore = field(repr=False)
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta"
    provider_id: str = "gemini"
    capabilities_by_model: dict[str, ModelCapabilities] = field(default_factory=dict)
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint", _validate_endpoint(self.endpoint))
        _validate_required_provider_auth_and_timeout(
            "Gemini provider", self.credential_reference, self.timeout_seconds
        )

    def capabilities(self, model: str) -> ModelCapabilities:
        return self.capabilities_by_model.get(
            model,
            ModelCapabilities(
                # The native transport implements streamGenerateContent; saved
                # preset execution still requires its exact-model profile to opt in.
                streaming=True,
                generation_parameters=(
                    "temperature",
                    "top_p",
                    "top_k",
                    "max_output_tokens",
                    "stop_sequences",
                ),
            ),
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        capabilities = self.capabilities(request.model)
        if request.image is not None and not capabilities.vision:
            raise ModelProviderError(
                "Gemini image input requires vision support for this exact model"
            )
        _require_supported_system_prompt(request, capabilities)
        _require_supported_structured_output(request, capabilities)
        _require_supported_reasoning(
            request, capabilities, native_effort_values=_GEMINI_REASONING_EFFORT_WIRE_VALUES
        )
        _reject_unsupported_reasoning_budget(request, "Gemini adapter")
        if request.reasoning_mode:
            raise ModelProviderError("Gemini adapter does not declare a separate reasoning mode")
        parameters = _require_supported_parameters(request, capabilities)
        generation_config: dict[str, object] = {}
        parameter_names = {
            "temperature": "temperature",
            "top_p": "topP",
            "top_k": "topK",
            "max_output_tokens": "maxOutputTokens",
            "stop_sequences": "stopSequences",
        }
        generation_config.update(
            {parameter_names[name]: value for name, value in parameters.items()}
        )
        if request.reasoning:
            generation_config["thinkingConfig"] = {"thinkingLevel": request.reasoning}
        content_parts: list[dict[str, object]] = [{"text": request.prompt}]
        if request.image is not None:
            content_parts.append(
                {
                    "inline_data": {
                        "mime_type": request.image.mime_type,
                        "data": base64.b64encode(request.image.data).decode("ascii"),
                    }
                }
            )
        payload: dict[str, object] = {
            "contents": [{"role": "user", "parts": content_parts}]
        }
        if request.system_prompt is not None:
            payload["systemInstruction"] = {"parts": [{"text": request.system_prompt}]}
        if generation_config:
            payload["generationConfig"] = generation_config
        if request.response_schema is not None:
            generation_config["responseFormat"] = {
                "text": {
                    "mimeType": "APPLICATION_JSON",
                    "schema": request.response_schema,
                }
            }
            payload["generationConfig"] = generation_config
        encoded_payload = _encode_provider_json(payload)
        if (
            request.image is not None
            and len(encoded_payload) > MAX_GEMINI_INLINE_REQUEST_BYTES
        ):
            raise ModelProviderError(
                "Gemini inline image request exceeds the configured byte limit"
            )
        response = _post_json(
            f"{self.endpoint}/models/{request.model}:generateContent",
            payload,
            {
                "x-goog-api-key": _resolve_provider_credential(
                    self.secret_store, self.credential_reference
                )
            },
            self.timeout_seconds,
            provider_id=self.provider_id,
            operation="generation",
        )
        _require_unblocked_gemini_prompt(response)
        try:
            candidates = response["candidates"]
            candidate = candidates[0]
            if not isinstance(candidate, dict):
                raise TypeError("Gemini candidate must be an object")
            _require_complete_gemini_candidate(candidate)
            parts = candidate["content"]["parts"]
            if not isinstance(parts, list):
                raise TypeError("Gemini candidate parts must be a list")
            text_parts = [
                part["text"]
                for part in parts
                if isinstance(part, dict)
                and part.get("thought") is not True
                and isinstance(part.get("text"), str)
            ]
            if not text_parts:
                raise ValueError("Gemini candidate lacks user-facing text")
            text = "".join(text_parts)
        except (KeyError, IndexError, TypeError, ValueError):
            raise ModelResponseError("Gemini response lacks text content") from None
        usage_data = response.get("usageMetadata")
        usage = _gemini_usage(usage_data)
        return GenerationResponse(text, self.provider_id, request.model, usage)

    def stream_generate(self, request: GenerationRequest) -> Iterator[str]:
        """Stream only user-facing Gemini text from the native SSE endpoint."""
        if request.image is not None:
            raise ModelProviderError("Gemini image input is not supported with streaming")
        capabilities = self.capabilities(request.model)
        if not capabilities.streaming:
            raise ModelProviderError("provider does not declare streaming support for this model")
        if request.response_schema is not None:
            raise ModelProviderError("streaming structured output is not supported")
        _require_supported_system_prompt(request, capabilities)
        _require_supported_structured_output(request, capabilities)
        _require_supported_reasoning(
            request, capabilities, native_effort_values=_GEMINI_REASONING_EFFORT_WIRE_VALUES
        )
        _reject_unsupported_reasoning_budget(request, "Gemini adapter")
        if request.reasoning_mode:
            raise ModelProviderError("Gemini adapter does not declare a separate reasoning mode")

        parameters = _require_supported_parameters(request, capabilities)
        parameter_names = {
            "temperature": "temperature",
            "top_p": "topP",
            "top_k": "topK",
            "max_output_tokens": "maxOutputTokens",
            "stop_sequences": "stopSequences",
        }
        generation_config: dict[str, object] = {
            parameter_names[name]: value for name, value in parameters.items()
        }
        if request.reasoning:
            generation_config["thinkingConfig"] = {"thinkingLevel": request.reasoning}
        payload: dict[str, object] = {
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}]
        }
        if request.system_prompt is not None:
            payload["systemInstruction"] = {"parts": [{"text": request.system_prompt}]}
        if generation_config:
            payload["generationConfig"] = generation_config

        encoded_model = quote(request.model, safe="-._")
        http_request = Request(
            f"{self.endpoint}/models/{encoded_model}:streamGenerateContent?alt=sse",
            _encode_provider_json(payload),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "x-goog-api-key": _resolve_provider_credential(
                    self.secret_store, self.credential_reference
                ),
            },
            method="POST",
        )
        try:
            with _open_provider_request(http_request, timeout=self.timeout_seconds) as response:
                yield from _iter_gemini_stream_text(response)
        except HTTPError as exc:
            _raise_provider_http_error(exc, self.provider_id, "generation")
        except URLError:
            raise ModelProviderError("provider endpoint is unavailable") from None
        except (OSError, ValueError, RecursionError):
            raise ModelProviderError("provider streaming response could not be read") from None


def _integer(payload: dict[str, Any] | None, key: str) -> int | None:
    value = payload.get(key) if payload else None
    # bool is a subclass of int in Python, but JSON true/false is never a token count.
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _require_complete_anthropic_response(
    response: dict[str, Any], provider_id: str
) -> None:
    """Accept only complete text-turn stop reasons from Messages APIs."""
    stop_reason = response.get("stop_reason")
    if isinstance(stop_reason, str) and stop_reason in {"end_turn", "stop_sequence"}:
        return
    if stop_reason == "max_tokens":
        raise ModelResponseError(
            f"{provider_id} response reached its output limit before completion"
        )
    if stop_reason == "refusal":
        raise ModelResponseError(f"{provider_id} refused to complete the text response")
    if isinstance(stop_reason, str) and stop_reason in {"tool_use", "pause_turn"}:
        raise ModelResponseError(
            f"{provider_id} response requires an unsupported tool handoff"
        )
    raise ModelResponseError(
        f"{provider_id} response has a missing or unsupported stop reason"
    )


def _require_unblocked_gemini_prompt(response: dict[str, Any]) -> None:
    """Reject an explicit prompt block before looking for candidate text."""
    feedback = response.get("promptFeedback")
    if feedback is None:
        return
    if not isinstance(feedback, dict):
        raise ModelResponseError("Gemini response has malformed prompt feedback")
    block_reason = feedback.get("blockReason")
    if block_reason is None or block_reason == "BLOCK_REASON_UNSPECIFIED":
        return
    if not isinstance(block_reason, str):
        raise ModelResponseError("Gemini response has malformed prompt feedback")
    raise ModelResponseError("Gemini blocked the prompt under a safety policy")


def _require_complete_gemini_candidate(candidate: dict[str, Any]) -> None:
    """Accept only STOP candidates with user-facing text rather than tool output."""
    safety_ratings = candidate.get("safetyRatings")
    if safety_ratings is not None:
        if not isinstance(safety_ratings, list) or any(
            not isinstance(rating, dict) for rating in safety_ratings
        ):
            raise ModelResponseError("Gemini response has malformed safety metadata")
        for rating in safety_ratings:
            blocked = rating.get("blocked")
            if blocked is not None and type(blocked) is not bool:
                raise ModelResponseError("Gemini response has malformed safety metadata")
            if blocked is True:
                raise ModelResponseError("Gemini blocked the candidate under a safety policy")

    finish_reason = candidate.get("finishReason")
    if finish_reason == "STOP":
        pass
    elif finish_reason == "MAX_TOKENS":
        raise ModelResponseError("Gemini response reached its output limit before completion")
    elif isinstance(finish_reason, str) and finish_reason in {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "MODEL_ARMOR",
    }:
        raise ModelResponseError("Gemini blocked the candidate under a safety policy")
    elif not isinstance(finish_reason, str) or not finish_reason:
        raise ModelResponseError("Gemini response has a missing or malformed finish reason")
    else:
        raise ModelResponseError("Gemini response has an unsupported finish reason")

    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if isinstance(parts, list) and any(
        isinstance(part, dict)
        and ("functionCall" in part or "function_call" in part)
        for part in parts
    ):
        raise ModelResponseError("Gemini response requires an unsupported tool handoff")


def _openai_usage(payload: object) -> TokenUsage | None:
    if not isinstance(payload, dict):
        return None
    details = payload.get("completion_tokens_details")
    output_tokens = _integer(payload, "completion_tokens")
    reasoning_tokens = _integer(details, "reasoning_tokens") if isinstance(details, dict) else None
    if output_tokens is None or reasoning_tokens is not None and reasoning_tokens > output_tokens:
        # OpenAI's completion_tokens already includes reasoning tokens. Keep a
        # safe total when a malformed detail breakdown cannot be trusted.
        reasoning_tokens = None
    return TokenUsage(
        input_tokens=_integer(payload, "prompt_tokens"),
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _openai_responses_output_text(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ModelResponseError("OpenAI Responses API response must be an object")
    output = payload.get("output")
    if not isinstance(output, list):
        raise ModelResponseError("OpenAI Responses API response lacks output items")
    text_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            raise ModelResponseError("OpenAI Responses API output item is malformed")
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                raise ModelResponseError("OpenAI Responses API content block is malformed")
            block_type = block.get("type")
            if block_type == "refusal":
                raise ModelResponseError("OpenAI Responses API refused the request")
            if block_type == "output_text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise ModelResponseError("OpenAI Responses API text output is malformed")
                text_parts.append(text)
    return "".join(text_parts)


def _openai_responses_usage(payload: object) -> TokenUsage | None:
    if not isinstance(payload, dict):
        return None
    details = payload.get("output_tokens_details")
    output_tokens = _integer(payload, "output_tokens")
    reasoning_tokens = (
        _integer(details, "reasoning_tokens") if isinstance(details, dict) else None
    )
    if output_tokens is None or reasoning_tokens is not None and reasoning_tokens > output_tokens:
        reasoning_tokens = None
    return TokenUsage(
        input_tokens=_integer(payload, "input_tokens"),
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _gemini_usage(payload: object) -> TokenUsage | None:
    """Normalize Gemini's separately reported candidate and thought counts."""
    if not isinstance(payload, dict):
        return None

    input_tokens = _integer(payload, "promptTokenCount")
    candidate_tokens = _integer(payload, "candidatesTokenCount")
    reasoning_tokens = _integer(payload, "thoughtsTokenCount")
    if candidate_tokens is None or reasoning_tokens is None:
        return TokenUsage(input_tokens=input_tokens)

    output_tokens = candidate_tokens + reasoning_tokens
    if "totalTokenCount" in payload:
        reported_total = _integer(payload, "totalTokenCount")
        if reported_total is None or (
            input_tokens is not None and reported_total != input_tokens + output_tokens
        ):
            # Preserve independently validated input usage, but do not produce
            # an output estimate from contradictory provider totals.
            return TokenUsage(input_tokens=input_tokens)

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _post_json(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    timeout_seconds: float,
    *,
    provider_id: str,
    operation: str,
    additional_auth_codes: tuple[int, ...] = (),
) -> dict[str, Any]:
    request = Request(
        url,
        _encode_provider_json(payload),
        headers={"Content-Type": "application/json", "Accept": "application/json", **headers},
        method="POST",
    )
    try:
        with _open_provider_request(request, timeout=timeout_seconds) as response:
            result = _parse_provider_json(
                _read_bounded_provider_response(response).decode("utf-8")
            )
    except HTTPError as exc:
        _raise_provider_http_error(
            exc,
            provider_id,
            operation,
            additional_auth_codes=additional_auth_codes,
        )
    except (URLError, OSError, ValueError, RecursionError):
        raise ModelProviderError("provider response could not be read") from None
    if not isinstance(result, dict):
        raise ModelResponseError("provider response must be a JSON object")
    return result
