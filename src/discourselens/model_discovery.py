"""Explicit, bounded model-list discovery for saved provider credentials."""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request

from .errors import ModelProviderError, ModelResponseError, ProjectError
from .execution import _DEFAULT_EMBEDDING_ENDPOINTS, _DEFAULT_ENDPOINTS, model_role_capabilities
from .provider_catalog import supports_embeddings, supports_generation
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
    if provider_id.removesuffix("-embeddings") == "cohere":
        base = base.removesuffix("/v2")
        return f"{base}/v1/models"
    if provider_id == "gemini":
        return f"{base}/models"
    return f"{base}/models"


def _model_rows(provider_id: str, payload: object, purpose: str) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        raise ModelResponseError("provider model catalog is malformed")
    provider_family = provider_id.removesuffix("-embeddings")
    rows: object
    if provider_family in {"gemini", "cohere"}:
        rows = payload.get("models")
    else:
        rows = payload.get("data")
    if not isinstance(rows, list):
        raise ModelResponseError("provider model catalog is malformed")

    models: dict[str, dict[str, object]] = {}
    for row in rows[:_MAX_DISCOVERED_MODELS]:
        if not isinstance(row, dict):
            continue
        if provider_family == "gemini":
            supported_actions = row.get("supportedActions")
            if not isinstance(supported_actions, list):
                continue
            capabilities = {
                "chat": "generateContent" in supported_actions,
                "embeddings": "embedContent" in supported_actions,
            }
            if purpose == "both":
                if not all(capabilities.values()):
                    continue
            elif not capabilities[purpose]:
                continue
            raw_id = row.get("name")
            if isinstance(raw_id, str) and raw_id.startswith("models/"):
                raw_id = raw_id.removeprefix("models/")
        elif provider_family == "cohere":
            endpoints = row.get("endpoints")
            if not isinstance(endpoints, list):
                continue
            capabilities = {"chat": "chat" in endpoints, "embeddings": "embed" in endpoints}
            if purpose == "both":
                if not all(capabilities.values()):
                    continue
            elif not capabilities[purpose]:
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
        result: dict[str, object] = {"model": model_id, "display_name": display_name.strip()}
        if provider_family in {"gemini", "cohere"}:
            result["capabilities"] = capabilities
        models[model_id] = result
    return [
        row
        for _, row in sorted(models.items(), key=lambda item: item[0].casefold())
    ]


def _saved_embedding_profiles(project, provider_id: str) -> list[dict[str, object]]:
    discovered = []
    for profile in project.list_model_capability_profiles():
        if profile["provider_id"] != provider_id:
            continue
        roles = model_role_capabilities(project, provider_id, str(profile["model"]))
        if roles["embeddings"]:
            discovered.append(
                {
                    "model": str(profile["model"]),
                    "display_name": str(profile["model"]),
                    "capabilities": roles,
                }
            )
    return discovered


def discover_provider_models(
    project,
    provider_id: str,
    credential_alias: str,
    *,
    purpose: str = "chat",
) -> list[dict[str, object]]:
    """Fetch only model IDs suitable for the explicitly selected task.

    Provider keys are resolved only inside the project boundary. Requests carry
    no corpus material, do not follow redirects, and responses are size bounded.
    Provider discovery is only a suggestion; provider task metadata and saved
    exact-model capability profiles determine which role selectors may use each model.
    """
    if purpose not in {"chat", "embeddings", "both"}:
        raise ModelProviderError("model discovery purpose must be chat, embeddings, or both")
    provider_supports_purpose = (
        supports_generation(provider_id)
        if purpose == "chat"
        else supports_embeddings(provider_id)
        if purpose == "embeddings"
        else supports_generation(provider_id) and supports_embeddings(provider_id)
    )
    if not provider_supports_purpose:
        raise ModelProviderError(
            "selected provider does not support the requested model purpose"
        )

    # Some embedding-only vendors do not expose a catalog endpoint.
    if purpose == "embeddings" and provider_id not in {
        "cohere-embeddings",
        "gemini-embeddings",
    } and not supports_generation(provider_id):
        return _saved_embedding_profiles(project, provider_id)
    credential = project.credential_reference(provider_id, credential_alias)
    reference = str(credential["secret_reference"])
    endpoint = credential["endpoint"]
    if not isinstance(endpoint, str) or not endpoint:
        defaults = _DEFAULT_EMBEDDING_ENDPOINTS if purpose == "embeddings" else _DEFAULT_ENDPOINTS
        endpoint = defaults.get(provider_id)
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
        if provider_id in {"gemini", "gemini-embeddings"}:
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
    models = _model_rows(provider_id, payload, purpose)
    selected = []
    for item in models:
        roles = model_role_capabilities(project, provider_id, str(item["model"]))
        declared = item.get("capabilities")
        profile = project.model_capability_profile(provider_id, str(item["model"]))
        role_supported = all(roles.values()) if purpose == "both" else roles[purpose]
        declared_supported = (
            isinstance(declared, dict)
            and (all(declared.values()) if purpose == "both" else declared.get(purpose) is True)
        )
        if profile is not None and not role_supported:
            continue
        if not role_supported and not declared_supported:
            continue
        selected.append({**item, "capabilities": declared if isinstance(declared, dict) else roles})
    return selected
