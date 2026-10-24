from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from discourselens.mcp_server import _McpBearerTokenMiddleware


def test_mcp_http_bearer_token_is_required_and_checked():
    token = "x" * 64
    app = Starlette(
        routes=[Route("/mcp", lambda request: PlainTextResponse("ok"), methods=["GET"])]
    )
    client = TestClient(_McpBearerTokenMiddleware(app, token))

    assert client.get("/mcp").status_code == 401
    assert client.get("/mcp", headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get("/mcp", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.text == "ok"
