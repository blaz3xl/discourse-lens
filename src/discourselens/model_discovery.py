"""Explicit, bounded model-list discovery for saved provider credentials."""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request

from .errors import ModelProviderError, ModelResponseError, ProjectError
from .execution import _DEFAULT_ENDPOINTS
from .provider_catalog import supports_generation
from .providers import (
    _open_provider_request,
    _parse_provider_json,
    _raise_provider_http_error,
    _read_bounded_provider_response,
    _validate_endpoint,
)
from .secrets import NO_AUTH_REFERENCE, ReferenceSecretStore

_MAX_DISCOVERED_MODELS = 500
_MAX_DISCOVERY_TIMEOUT_SECONDS = 12.0


def _models_url(provider_id: str, endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if provider_id == "cohere":
        base = base.removesuffix("/v2")
        return f"{base}/v1/models"
    if provider_id == "gemini":
        return f"{base}/models"
    return f"{base}/models"


def _model_rows(provider_id: str, payload: object) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        raise ModelResponseError("provider model catalog is malformed")
    rows: object
    if provider_id == "gemini" or provider_id == "cohere":
        rows = payload.get("models")
    else:
        rows = payload.get("data")
    if not isinstance(rows, list):
        raise ModelResponseError("provider model catalog is malformed")

    models: dict[str, str] = {}
    for row in rows[:_MAX_DISCOVERED_MODELS]:
        if not isinstance(row, dict):
            continue
        if provider_id == "gemini":
            supported_actions = row.get("supportedActions")
            if not isinstance(supported_actions, list) or "generateContent" not in supported_actions:
                continue
            raw_id = row.get("name")
            if isinstance(raw_id, str) and raw_id.startswith("models/"):
                raw_id = raw_id.removeprefix("models/")
        elif provider_id == "cohere":
            endpoints = row.get("endpoints")
            if isinstance(endpoints, list) and "chat" not in endpoints:
                continue
            if row.get("is_deprecated") is True:
                continue
            raw_id = row.get("name")
        else:
            raw_id = row.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip() or len(raw_id) > 256:
            continue
        model_id = raw_id.strip()
        display_name = row.get("display_name", row.get("displayName", model_id))
        if not isinstance(display_name, str) or not display_name.strip() or len(display_name) > 256:
            display_name = model_id
        models[model_id] = display_name.strip()
    return [
        {"model": model, "display_name": display_name}
        for model, display_name in sorted(models.items(), key=lambda item: item[0].casefold())
    ]


def discover_provider_models(project, provider_id: str, credential_alias: str) -> list[dict[str, str]]:
    """Fetch model IDs from a generation-capable provider on explicit request.

    Provider keys are resolved only inside the project boundary. Requests carry
    no corpus material, do not follow redirects, and responses are size bounded.
    Returned IDs are suggestions; the provider catalog may not state whether an
    individual model supports generation or reasoning.
    """
    if not supports_generation(provider_id):
        raise ModelProviderError("selected provider does not support text generation")
    credential = project.credential_reference(provider_id, credential_alias)
    reference = str(credential["secret_reference"])
    endpoint = credential["endpoint"]
    if not isinstance(endpoint, str) or not endpoint:
        endpoint = _DEFAULT_ENDPOINTS.get(provider_id)
    if not isinstance(endpoint, str) or not endpoint:
        raise ModelProviderError("provider model discovery requires a configured endpoint")
    endpoint = _validate_endpoint(endpoint)
    secret: str | None = None
    if reference != NO_AUTH_REFERENCE:
        secret = ReferenceSecretStore(project).get(reference)
        if not secret:
            raise ProjectError("provider credential is unavailable")

    headers = {"Accept": "application/json"}
    if secret:
        if provider_id == "gemini":
            headers["x-goog-api-key"] = secret
        elif provider_id == "anthropic":
            headers["x-api-key"] = secret
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {secret}"
    request = Request(_models_url(provider_id, endpoint), headers=headers, method="GET")
    try:
        with _open_provider_request(request, timeout=_MAX_DISCOVERY_TIMEOUT_SECONDS) as response:
            body = _read_bounded_provider_response(response)
        payload = _parse_provider_json(body.decode("utf-8"))
    except HTTPError as exc:
        _raise_provider_http_error(exc, provider_id, "model discovery")
    except URLError:
        raise ModelProviderError("provider model catalog is unavailable") from None
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        raise ModelProviderError("provider model catalog could not be read") from None
    return _model_rows(provider_id, payload)
