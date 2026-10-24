"""Secret references only; project configuration never stores secret values."""

from __future__ import annotations

import os
import re
from urllib.parse import unquote, urlsplit

from .errors import OptionalDependencyError, ProjectError
from .network_policy import is_loopback_endpoint
from .provider_catalog import is_openai_compatible_provider

_ENVIRONMENT_REFERENCE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z", re.ASCII)
_PROJECT_SECRET_REFERENCE = re.compile(r"project-secret:[0-9a-f-]{36}\Z", re.ASCII)
_CREDENTIAL_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z", re.ASCII)
_KEYRING_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_KEYRING_ACCOUNT_PATH = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,127})*\Z",
    re.ASCII,
)
_KNOWN_SECRET_PREFIXES = (
    "sk-",
    "sk_",
    "gsk_",
    "xai-",
    "or-v1-",
    "aiza",  # Google API keys begin with AIza.
    "ya29.",
    "ghp_",
    "gho_",
    "github_pat_",
)
_CREDENTIAL_TOKEN_PATTERNS = tuple(
    re.compile(pattern, re.ASCII)
    for pattern in (
        # Common provider tokens: OpenAI and Anthropic, Groq, xAI, OpenRouter,
        # Jina, Google API/OAuth, and GitHub.
        r"(?i)(?:^|[^A-Za-z0-9])(?:sk-|gsk_|xai-|or-v1-|AIza|ya29\.)[A-Za-z0-9._-]*",
        r"(?:^|[^A-Za-z0-9])jina_[A-Za-z0-9_-]+",
        # Slack bot, user, app, and workspace tokens commonly use xox*- prefixes.
        r"(?i)(?:^|[^A-Za-z0-9])xox[a-z]-[A-Za-z0-9-]+",
        r"(?i)(?:^|[^A-Za-z0-9])(?:ghp_|gho_|ghu_|ghs_|ghr_|github_pat_)[A-Za-z0-9_-]+",
        # AWS access-key identifiers and compact signed JWTs.
        r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
    )
)

NO_AUTH_REFERENCE = "no-auth"
_MAX_ENDPOINT_PATH_DECODING_PASSES = 8


def _looks_like_secret_value(value: str) -> bool:
    # Jina tokens use a lowercase ``jina_`` prefix; checking this one case-sensitively
    # keeps the conventional uppercase JINA_API_KEY environment reference usable.
    return value.startswith("jina_") or value.casefold().startswith(_KNOWN_SECRET_PREFIXES)


def is_credential_like_alias(value: object) -> bool:
    """Return whether a proposed human alias resembles credential material.

    The check deliberately covers both recognizable provider formats and common
    places credentials get pasted by mistake. It is a detection helper, not the
    complete alias grammar: :func:`validate_credential_alias` also enforces the
    short, portable label format.
    """
    if not isinstance(value, str):
        return True
    candidate = value.strip()
    if not candidate:
        return False
    if any(pattern.search(candidate) for pattern in _CREDENTIAL_TOKEN_PATTERNS):
        return True
    if re.search(r"(?i)\bbearer\s+\S+", candidate):
        return True
    if "?" in candidate or re.search(r"[&][^=&#\s]+=[^&#\s]*", candidate):
        return True
    if re.search(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|secret|password)\s*[:=]", candidate):
        return True
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return True
    if parsed.scheme and (parsed.netloc or "://" in candidate):
        return True
    if parsed.username is not None or parsed.password is not None or "@" in candidate:
        return True

    # Catch opaque pasted values without treating ordinary compound labels as
    # secrets. Random API keys commonly have 32+ characters and varied classes.
    opaque_components = re.split(r"[_-]", candidate)
    if any(
        len(component) >= 32
        and re.fullmatch(r"[A-Za-z0-9+/=]+", component, re.ASCII)
        for component in opaque_components
    ):
        return True
    if len(candidate) >= 32 and re.fullmatch(r"[A-Za-z0-9+/=]+", candidate, re.ASCII):
        classes = sum(
            (
                any(char.islower() for char in candidate),
                any(char.isupper() for char in candidate),
                any(char.isdigit() for char in candidate),
                any(char in "+/=" for char in candidate),
            )
        )
        if classes >= 2 and len(set(candidate)) >= 10:
            return True
    return bool(re.fullmatch(r"[A-Fa-f0-9]{32,}", candidate, re.ASCII))


def has_credential_like_endpoint_host(endpoint: object) -> bool:
    """Return whether an endpoint hostname contains credential-shaped material.

    Endpoint paths and query data are checked separately. Hostnames need their
    own check because they are included in previews and provenance metadata even
    when userinfo, query, and fragment components are rejected.
    """
    if not isinstance(endpoint, str):
        return False
    try:
        hostname = urlsplit(endpoint).hostname
    except ValueError:
        return True
    if not hostname:
        return False

    decoded_hostname = hostname
    for _ in range(_MAX_ENDPOINT_PATH_DECODING_PASSES):
        next_hostname = unquote(decoded_hostname)
        if next_hostname == decoded_hostname:
            break
        decoded_hostname = next_hostname
    else:
        return True

    labels = decoded_hostname.rstrip(".").split(".")
    return is_credential_like_alias(decoded_hostname) or any(
        is_credential_like_alias(label) for label in labels
    )


def validate_credential_alias(value: object) -> None:
    """Require a short, safe human label instead of a token or secret-bearing value."""
    if (
        not isinstance(value, str)
        or not _CREDENTIAL_ALIAS.fullmatch(value)
        or is_credential_like_alias(value)
    ):
        raise ProjectError("credential alias must be a short human-readable label")


def keyring_reference_for_credential(provider_id: str, alias: str) -> str:
    """Build the deterministic keyring reference for a provider credential alias."""
    if not isinstance(provider_id, str) or not _KEYRING_IDENTIFIER.fullmatch(provider_id):
        raise ProjectError("provider ID must be a short identifier to use keyring setup")
    validate_credential_alias(alias)
    reference = f"keyring:discourselens:{provider_id}/{alias}"
    validate_secret_reference(reference)
    return reference


def validate_secret_reference(reference: str) -> None:
    """Reject values that are not references before they can enter project SQLite.

    The accepted forms are conventional environment-variable names, explicit
    OS-keyring references, and opaque references to encrypted project secrets.
    Known raw-token prefixes are rejected to catch the common mistake of pasting
    a provider key into the reference option.
    """
    if (
        not isinstance(reference, str)
        or not reference
        or reference != reference.strip()
        or len(reference) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in reference)
    ):
        raise ProjectError("secret reference must be an environment name or keyring reference")

    if _PROJECT_SECRET_REFERENCE.fullmatch(reference):
        return

    if reference.startswith("keyring:"):
        parts = reference.split(":", 2)
        if (
            len(parts) == 3
            and _KEYRING_IDENTIFIER.fullmatch(parts[1])
            and _KEYRING_ACCOUNT_PATH.fullmatch(parts[2])
            and len(parts[2]) <= 256
            and all(segment not in {".", ".."} for segment in parts[2].split("/"))
        ):
            if any(
                _looks_like_secret_value(part) or is_credential_like_alias(part)
                for part in (parts[1], *parts[2].split("/"))
            ):
                raise ProjectError("keyring reference must identify an entry, never a secret value")
            return
        raise ProjectError("keyring reference must use keyring:service:account")

    if (
        _ENVIRONMENT_REFERENCE.fullmatch(reference)
        and not _looks_like_secret_value(reference)
        and not is_credential_like_alias(reference)
    ):
        return
    raise ProjectError(
        "secret reference must be an environment-variable name or keyring:service:account; "
        "never provide the secret value"
    )


def _validate_provider_endpoint_metadata(endpoint: str | None) -> None:
    if endpoint is None:
        return
    if (
        not isinstance(endpoint, str)
        or endpoint != endpoint.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in endpoint)
    ):
        raise ProjectError("provider endpoint must not contain credential material")
    try:
        parsed_endpoint = urlsplit(endpoint)
    except ValueError:
        raise ProjectError("provider endpoint must not contain credential material") from None
    decoded_path = parsed_endpoint.path
    for _ in range(_MAX_ENDPOINT_PATH_DECODING_PASSES):
        next_path = unquote(decoded_path)
        if next_path == decoded_path:
            break
        decoded_path = next_path
    else:
        raise ProjectError("provider endpoint must not contain credential material")
    suspicious_path = is_credential_like_alias(decoded_path) or any(
        is_credential_like_alias(segment) for segment in decoded_path.split("/")
    )
    if (
        parsed_endpoint.username is not None
        or parsed_endpoint.password is not None
        or "?" in endpoint
        or "#" in endpoint
        or suspicious_path
        or has_credential_like_endpoint_host(endpoint)
    ):
        raise ProjectError("provider endpoint must not contain credential material")


def provider_auth_reference(provider_id: str, reference: str, endpoint: str | None) -> str | None:
    """Validate a saved auth selector and resolve the reserved local no-auth mode.

    No authentication is only allowed for a reviewed OpenAI-compatible provider
    with an explicit syntactic HTTP(S) loopback endpoint. DNS and private LAN
    addresses are never treated as local.
    """
    _validate_provider_endpoint_metadata(endpoint)
    if reference == NO_AUTH_REFERENCE:
        if not is_openai_compatible_provider(provider_id) or not is_loopback_endpoint(endpoint):
            raise ProjectError(
                "no-auth credentials are allowed only for OpenAI-compatible loopback endpoints"
            )
        return None
    validate_secret_reference(reference)
    return reference


class EnvironmentSecretStore:
    def get(self, reference: str) -> str:
        validate_secret_reference(reference)
        value = os.environ.get(reference)
        if not value:
            raise ProjectError(f"secret reference {reference!r} is unavailable")
        return value


class KeyringSecretStore:
    """Resolve `keyring:service:account` references without writing values to projects."""

    @staticmethod
    def _parts(reference: str) -> tuple[str, str]:
        validate_secret_reference(reference)
        if not reference.startswith("keyring:"):
            raise ProjectError("keyring reference must use keyring:service:account")
        try:
            _, service, account = reference.split(":", 2)
        except ValueError:
            raise ProjectError("keyring reference must use keyring:service:account") from None
        return service, account

    @staticmethod
    def _load_keyring() -> object:
        try:
            import keyring  # type: ignore[reportMissingImports]  # optional extra
        except ImportError:
            raise OptionalDependencyError(
                "OS keyring setup requires discourselens[secrets]; install the secrets extra"
            ) from None
        return keyring

    def ensure_absent(self, reference: str) -> None:
        """Check keyring availability and refuse an existing entry before prompting."""
        service, account = self._parts(reference)
        keyring = self._load_keyring()
        try:
            existing = keyring.get_password(service, account)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - backend messages can contain credential data
            raise ProjectError(
                "OS keyring is unavailable; configure a working keyring backend and retry"
            ) from None
        if existing is not None:
            raise ProjectError(
                "keyring entry already exists; choose a different credential alias"
            )

    def contains(self, reference: str) -> bool:
        """Return whether a keyring reference currently has a value."""
        service, account = self._parts(reference)
        keyring = self._load_keyring()
        try:
            return keyring.get_password(service, account) is not None  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - backend messages can contain credential data
            raise ProjectError(
                "OS keyring is unavailable; configure a working keyring backend and retry"
            ) from None

    def remove_if_present(self, reference: str) -> bool:
        """Remove one keyring entry, returning false when it is already absent.

        The OS keyring API has no portable compare-and-delete operation. A separate
        process can replace the value between lookup and deletion; callers should
        serialize writes/removals for a given reference.
        """
        service, account = self._parts(reference)
        keyring = self._load_keyring()
        try:
            existing = keyring.get_password(service, account)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - backend messages can contain credential data
            raise ProjectError(
                "OS keyring is unavailable; configure a working keyring backend and retry"
            ) from None
        if existing is None:
            return False

        try:
            keyring.delete_password(service, account)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - backend messages can contain credential data
            # A concurrent removal is still a successful outcome. Never expose backend
            # exception text: some implementations include account names or stored data.
            try:
                remaining = keyring.get_password(service, account)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - backend messages can contain credential data
                raise ProjectError(
                    "OS keyring is unavailable; configure a working keyring backend and retry"
                ) from None
            if remaining is None:
                return True
            raise ProjectError(
                "keyring entry could not be removed; check keyring backend access"
            ) from None

        try:
            remaining = keyring.get_password(service, account)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - backend messages can contain credential data
            raise ProjectError(
                "OS keyring is unavailable; configure a working keyring backend and retry"
            ) from None
        if remaining is not None:
            raise ProjectError("keyring entry could not be removed; check keyring backend access")
        return True

    def set_if_absent(self, reference: str, value: str) -> None:
        """Store a secret unless an entry is present at either preflight.

        This is a sequential overwrite guard, not an atomic compare-and-set: the OS
        keyring API has no portable create-if-absent operation. Callers should
        serialize writes for a given reference across processes.
        """
        if not isinstance(value, str) or not value:
            raise ProjectError("secret value must not be blank")
        service, account = self._parts(reference)
        keyring = self._load_keyring()
        try:
            existing = keyring.get_password(service, account)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - backend messages can contain credential data
            raise ProjectError(
                "OS keyring is unavailable; configure a working keyring backend and retry"
            ) from None
        if existing is not None:
            raise ProjectError(
                "keyring entry already exists; choose a different credential alias"
            )
        try:
            keyring.set_password(service, account, value)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - backend messages can contain credential data
            raise ProjectError(
                "secret could not be stored in the OS keyring; check keyring backend access"
            ) from None

    def get(self, reference: str) -> str:
        service, account = self._parts(reference)
        try:
            import keyring  # type: ignore[reportMissingImports]  # optional extra
        except ImportError as exc:
            raise OptionalDependencyError(
                "keyring references require discourselens[secrets]"
            ) from exc
        try:
            value = keyring.get_password(service, account)
        except Exception:  # noqa: BLE001 - backend messages can contain credential data
            # Keyring backends may include service/account data in their error
            # text. Those references can contain credential-bearing URLs.
            raise ProjectError("keyring secret lookup failed") from None
        if not value:
            raise ProjectError("keyring secret is unavailable")
        return value


class ReferenceSecretStore:
    """Resolve environment, keyring, or project-encrypted provider secrets."""

    def __init__(self, project: object | None = None) -> None:
        self.environment = EnvironmentSecretStore()
        self.keyring = KeyringSecretStore()
        self.project = project

    def get(self, reference: str) -> str:
        validate_secret_reference(reference)
        if reference.startswith("project-secret:"):
            if self.project is None:
                raise ProjectError("encrypted project secret is unavailable in this execution")
            resolver = getattr(self.project, "resolve_encrypted_secret", None)
            if not callable(resolver):
                raise ProjectError("encrypted project secret is unavailable in this execution")
            resolved = resolver(reference)
            if not isinstance(resolved, str):
                raise ProjectError("encrypted project secret is unavailable in this execution")
            return resolved
        return (
            self.keyring.get(reference)
            if reference.startswith("keyring:")
            else self.environment.get(reference)
        )
