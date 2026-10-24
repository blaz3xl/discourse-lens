import asyncio
import base64
import hashlib
import hmac
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from discourselens import mcp_server
from discourselens.errors import ProjectError
from discourselens.mcp_auth import (
    OidcJwtTokenVerifier,
    oauth_configuration_from_environment,
)


def test_oidc_verifier_accepts_only_valid_audience_and_required_scope(monkeypatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    class StubJwksClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_signing_key_from_jwt(self, _token: str):
            return SimpleNamespace(key=private_key.public_key())

    monkeypatch.setattr(jwt, "PyJWKClient", cast(object, StubJwksClient))
    verifier = OidcJwtTokenVerifier(
        issuer_url="https://login.example.test/realms/discourselens",
        resource_url="https://mcp.example.test/mcp",
        audience="https://mcp.example.test/mcp",
        jwks_url="https://login.example.test/realms/discourselens/protocol/openid-connect/certs",
        required_scopes=["discourselens:read"],
    )
    claims = {
        "iss": verifier.issuer_url,
        "sub": "researcher-1",
        "azp": "claude-connector",
        "aud": verifier.audience,
        "iat": int(time.time()),
        "exp": int(time.time()) + 300,
        "scope": "openid discourselens:read",
    }
    valid = jwt.encode(claims, private_key, algorithm="RS256")
    identity = asyncio.run(verifier.verify_token(valid))
    assert identity is not None
    assert identity.subject == "researcher-1"
    assert identity.client_id == "claude-connector"
    assert identity.resource == verifier.resource_url

    wrong_audience = jwt.encode(
        {**claims, "aud": "https://other.example.test/mcp"},
        private_key,
        algorithm="RS256",
    )
    assert asyncio.run(verifier.verify_token(wrong_audience)) is None

    missing_scope = jwt.encode(
        {**claims, "scope": "openid"}, private_key, algorithm="RS256"
    )
    assert asyncio.run(verifier.verify_token(missing_scope)) is None


def test_oidc_configuration_is_optional_but_requires_all_resource_settings(monkeypatch):
    names = (
        "DISCOURSELENS_MCP_OAUTH_ISSUER_URL",
        "DISCOURSELENS_MCP_OAUTH_RESOURCE_URL",
        "DISCOURSELENS_MCP_OAUTH_AUDIENCE",
        "DISCOURSELENS_MCP_OAUTH_JWKS_URL",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    assert oauth_configuration_from_environment() is None

    monkeypatch.setenv(names[0], "https://login.example.test/issuer")
    with pytest.raises(ProjectError, match="incomplete MCP OAuth configuration"):
        oauth_configuration_from_environment()

    monkeypatch.setenv(names[1], "https://mcp.example.test/mcp")
    monkeypatch.setenv(names[2], "https://mcp.example.test/mcp")
    monkeypatch.setenv(names[3], "https://login.example.test/jwks")
    config = oauth_configuration_from_environment()
    assert config is not None
    assert config.settings.required_scopes == ["discourselens:read"]
    assert config.settings.validate_token_resource is True


def test_authenticated_mcp_subject_opens_only_its_tenant_project(monkeypatch, tmp_path):
    from discourselens.project import Project
    from discourselens.tenant_storage import tenant_project_path

    storage_root = tmp_path / "tenant-storage"
    issuer = "https://login.example.test/realms/discourselens"
    key = "tenant-identification-key-" * 2
    project_a = tenant_project_path(storage_root, issuer, "researcher-a", key)
    project_b = tenant_project_path(storage_root, issuer, "researcher-b", key)
    Project.init(project_a).close()
    Project.init(project_b).close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(storage_root))
    monkeypatch.setenv("DISCOURSELENS_TENANT_ID_KEY", key)
    monkeypatch.setattr(mcp_server, "_MCP_OAUTH_CONFIGURATION", object())
    context_token = mcp_server.auth_context_var.set(
        AuthenticatedUser(
            AccessToken(
                token="signed-test-token",
                client_id="test-client",
                scopes=["discourselens:read"],
                subject="researcher-a",
                claims={"iss": issuer},
            )
        )
    )
    try:
        assert mcp_server._configured_project_root() == project_a
    finally:
        mcp_server.auth_context_var.reset(context_token)


def test_signed_proxy_identity_selects_only_its_private_project(monkeypatch, tmp_path):
    from discourselens.project import Project
    from discourselens.tenant_storage import tenant_project_path

    storage_root = tmp_path / "tenant-storage"
    issuer = "https://login.example.test/realms/discourselens"
    subject = "teun-oidc-subject"
    tenant_key = "tenant-identification-key-" * 2
    signing_key = "proxy-signing-key-" * 4
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(storage_root))
    monkeypatch.setenv("DISCOURSELENS_TENANT_ID_KEY", tenant_key)
    project_root = tenant_project_path(storage_root, issuer, subject, tenant_key)
    Project.init(project_root).close()
    payload = base64.urlsafe_b64encode(
        f"{issuer}\n{subject}\n{int(time.time()) + 60}".encode()
    ).decode().rstrip("=")
    signature = hmac.new(signing_key.encode(), payload.encode(), hashlib.sha256).hexdigest()
    captured: list[Path] = []

    async def downstream(_scope, _receive, send):
        captured.append(mcp_server._configured_project_root())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = mcp_server._McpBearerTokenMiddleware(
        downstream, "b" * 40, signing_key
    )

    async def call(headers):
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        await middleware(
            {
                "type": "http",
                "method": "POST",
                "headers": headers,
                "path": "/mcp",
                "query_string": b"",
                "server": ("test", 443),
                "client": ("test", 1),
                "scheme": "https",
                "http_version": "1.1",
            },
            receive,
            send,
        )
        return messages[0]["status"]

    status = asyncio.run(
        call(
            [
                (b"authorization", b"Bearer " + b"b" * 40),
                (b"x-discourselens-tenant", payload.encode()),
                (b"x-discourselens-tenant-signature", signature.encode()),
            ]
        )
    )
    assert status == 200
    assert captured == [project_root.resolve()]
    assert asyncio.run(
        call(
            [
                (b"authorization", b"Bearer " + b"b" * 40),
                (b"x-discourselens-tenant", payload.encode()),
                (b"x-discourselens-tenant-signature", b"0" * 64),
            ]
        )
    ) == 401
    assert captured == [project_root.resolve()]


def test_mcp_oidc_requires_an_authenticated_tenant_context(monkeypatch, tmp_path):
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(tmp_path.resolve()))
    monkeypatch.setenv("DISCOURSELENS_TENANT_ID_KEY", "tenant-identification-key-" * 2)
    monkeypatch.setattr(mcp_server, "_MCP_OAUTH_CONFIGURATION", object())
    with pytest.raises(ProjectError, match="identity is unavailable"):
        mcp_server._configured_project_root()


def test_writable_http_requires_explicit_oauth_write_scope(monkeypatch, tmp_path):
    monkeypatch.delenv("DISCOURSELENS_MCP_HTTP_TOKEN", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "discourselens-mcp",
            "--project",
            str(tmp_path),
            "--transport",
            "streamable-http",
        ],
    )
    monkeypatch.setattr(mcp_server, "_MCP_OAUTH_CONFIGURATION", None)
    with pytest.raises(SystemExit):
        mcp_server.main()

    monkeypatch.setattr(
        mcp_server,
        "_MCP_OAUTH_CONFIGURATION",
        SimpleNamespace(settings=SimpleNamespace(required_scopes=["discourselens:read"])),
    )
    with pytest.raises(SystemExit):
        mcp_server.main()

    calls: list[dict[str, object]] = []

    async def fake_run_server(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(mcp_server, "_run_server", fake_run_server)
    monkeypatch.setenv("DISCOURSELENS_TENANT_ID_KEY", "tenant-identification-key-" * 2)
    monkeypatch.setattr(
        mcp_server,
        "_MCP_OAUTH_CONFIGURATION",
        SimpleNamespace(settings=SimpleNamespace(required_scopes=["discourselens:write"])),
    )
    mcp_server.main()
    assert calls == [
        {
            "read_only": False,
            "transport": "streamable-http",
            "host": "127.0.0.1",
            "port": 8001,
            "bearer_token": None,
        }
    ]


def test_read_only_http_can_load_bearer_token_from_a_secret_file(
    monkeypatch, tmp_path
):
    secret_file = tmp_path / "mcp-http-token"
    secret_file.write_text("t" * 64 + "\n", encoding="utf-8")
    monkeypatch.delenv("DISCOURSELENS_MCP_HTTP_TOKEN", raising=False)
    monkeypatch.setenv("DISCOURSELENS_MCP_HTTP_TOKEN_FILE", str(secret_file))
    monkeypatch.setattr(
        "sys.argv",
        [
            "discourselens-mcp",
            "--project",
            str(tmp_path),
            "--read-only",
            "--transport",
            "streamable-http",
        ],
    )
    monkeypatch.setattr(mcp_server, "_MCP_OAUTH_CONFIGURATION", None)
    calls: list[dict[str, object]] = []

    async def fake_run_server(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(mcp_server, "_run_server", fake_run_server)
    mcp_server.main()
    assert calls[0]["bearer_token"] == "t" * 64


def test_mcp_http_rejects_ambiguous_or_unreadable_token_file(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_server, "_MCP_OAUTH_CONFIGURATION", None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "discourselens-mcp",
            "--project",
            str(tmp_path),
            "--read-only",
            "--transport",
            "streamable-http",
        ],
    )
    secret_file = tmp_path / "token"
    secret_file.write_text("v" * 64, encoding="utf-8")
    monkeypatch.setenv("DISCOURSELENS_MCP_HTTP_TOKEN", "w" * 64)
    monkeypatch.setenv("DISCOURSELENS_MCP_HTTP_TOKEN_FILE", str(secret_file))
    with pytest.raises(SystemExit):
        mcp_server.main()

    monkeypatch.delenv("DISCOURSELENS_MCP_HTTP_TOKEN", raising=False)
    monkeypatch.setenv("DISCOURSELENS_MCP_HTTP_TOKEN_FILE", str(tmp_path / "missing"))
    with pytest.raises(SystemExit):
        mcp_server.main()
