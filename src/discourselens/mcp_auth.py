"""OAuth resource-server token verification for remote read-only MCP clients."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import cast

import jwt
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl

from .errors import ProjectError


@dataclass(frozen=True)
class McpOAuthConfiguration:
    """Trusted OIDC issuer and protected MCP resource configuration."""

    settings: AuthSettings
    verifier: TokenVerifier


class OidcJwtTokenVerifier:
    """Verify RS256 access tokens from a configured external OIDC issuer."""

    def __init__(
        self,
        *,
        issuer_url: str,
        resource_url: str,
        audience: str,
        jwks_url: str,
        required_scopes: list[str],
    ) -> None:
        self.issuer_url = issuer_url
        self.resource_url = resource_url
        self.audience = audience
        self.required_scopes = frozenset(required_scopes)
        self._jwks = jwt.PyJWKClient(jwks_url, cache_keys=True, timeout=5)

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return the verified user/client identity, or reject the bearer token."""
        try:
            key = await asyncio.to_thread(self._jwks.get_signing_key_from_jwt, token)
            claims = jwt.decode(
                token,
                key.key,
                algorithms=["RS256"],
                issuer=self.issuer_url,
                audience=self.audience,
                options={"require": ["iss", "sub", "exp", "iat", "aud"]},
            )
        except (jwt.PyJWTError, OSError, ValueError, TypeError):
            return None
        subject = claims.get("sub")
        client_id = claims.get("client_id", claims.get("azp"))
        scope_claim = claims.get("scope", claims.get("scp", ""))
        if not isinstance(subject, str) or not subject or not isinstance(client_id, str) or not client_id:
            return None
        if isinstance(scope_claim, str):
            scopes = scope_claim.split()
        elif isinstance(scope_claim, list) and all(isinstance(item, str) for item in scope_claim):
            scopes = scope_claim
        else:
            return None
        if not self.required_scopes.issubset(scopes):
            return None
        expires_at = claims.get("exp")
        if type(expires_at) is not int:
            return None
        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            expires_at=expires_at,
            resource=self.resource_url,
            subject=subject,
            claims={"iss": self.issuer_url},
        )


def oauth_configuration_from_environment() -> McpOAuthConfiguration | None:
    """Build protected-resource auth settings only for a complete OIDC config."""
    names = {
        "issuer": "DISCOURSELENS_MCP_OAUTH_ISSUER_URL",
        "resource": "DISCOURSELENS_MCP_OAUTH_RESOURCE_URL",
        "audience": "DISCOURSELENS_MCP_OAUTH_AUDIENCE",
        "jwks": "DISCOURSELENS_MCP_OAUTH_JWKS_URL",
    }
    values = {key: os.environ.get(name, "").strip() for key, name in names.items()}
    if not any(values.values()):
        return None
    if any(not value for value in values.values()):
        missing = [names[key] for key, value in values.items() if not value]
        raise ProjectError("incomplete MCP OAuth configuration; missing " + ", ".join(missing))
    scopes_raw = os.environ.get("DISCOURSELENS_MCP_OAUTH_REQUIRED_SCOPES", "discourselens:read")
    scopes = scopes_raw.split()
    if not scopes or any(len(scope) > 128 for scope in scopes):
        raise ProjectError("MCP OAuth required scopes are invalid")
    try:
        settings = AuthSettings(
            issuer_url=cast(AnyHttpUrl, values["issuer"]),
            resource_server_url=cast(AnyHttpUrl, values["resource"]),
            validate_token_resource=True,
            required_scopes=scopes,
        )
    except Exception as exc:
        raise ProjectError("MCP OAuth issuer/resource configuration is invalid") from exc
    verifier = OidcJwtTokenVerifier(
        issuer_url=values["issuer"],
        resource_url=values["resource"],
        audience=values["audience"],
        jwks_url=values["jwks"],
        required_scopes=scopes,
    )
    return McpOAuthConfiguration(settings=settings, verifier=verifier)
