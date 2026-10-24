from contextlib import contextmanager

from discourselens.model_discovery import _model_rows, discover_provider_models


def test_openai_model_catalog_keeps_unique_well_formed_ids():
    assert _model_rows(
        "openai",
        {
            "data": [
                {"id": "gpt-z", "display_name": "Model Z"},
                {"id": "gpt-a"},
                {"id": "gpt-z", "display_name": "Updated Z"},
                {"id": " "},
                {"id": "x" * 257},
                None,
            ]
        },
    ) == [
        {"model": "gpt-a", "display_name": "gpt-a"},
        {"model": "gpt-z", "display_name": "Updated Z"},
    ]


def test_gemini_catalog_only_includes_generation_capable_models():
    assert _model_rows(
        "gemini",
        {
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini Pro",
                    "supportedActions": ["generateContent"],
                },
                {"name": "models/embeddinggemma", "supportedActions": ["embedContent"]},
            ]
        },
    ) == [{"model": "gemini-2.5-pro", "display_name": "Gemini Pro"}]


def test_discovery_uses_saved_secret_and_bounded_request(monkeypatch):
    class ProjectStub:
        def credential_reference(self, provider_id, alias):
            assert (provider_id, alias) == ("openai", "research")
            return {
                "secret_reference": "project-secret:00000000-0000-0000-0000-000000000001",
                "endpoint": "https://api.openai.com/v1",
            }

        def resolve_encrypted_secret(self, reference):
            assert reference.endswith("000000000001")
            return "secret-key"

    class Headers(dict):
        def get(self, name, default=None):
            return super().get(name, default)

    class ResponseStub:
        headers = Headers({"Content-Length": "55"})

        def read(self, limit):
            assert limit > 55
            return b'{"data":[{"id":"gpt-research","display_name":"Research"}]}'

    calls = []

    @contextmanager
    def open_request(request, timeout):
        calls.append((request, timeout))
        yield ResponseStub()

    monkeypatch.setattr("discourselens.model_discovery._open_provider_request", open_request)
    items = discover_provider_models(ProjectStub(), "openai", "research")

    assert items == [{"model": "gpt-research", "display_name": "Research"}]
    request, timeout = calls[0]
    assert request.full_url == "https://api.openai.com/v1/models"
    assert request.get_header("Authorization") == "Bearer secret-key"
    assert request.get_header("Accept") == "application/json"
    assert timeout <= 12
