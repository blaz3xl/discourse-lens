"""Account-local spending estimates: exact snapshots, unknown costs and isolation."""

import json
import secrets
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from discourselens.errors import ProjectError
from discourselens.execution import execute_model_preset, execute_model_preset_stream
from discourselens.models import (
    CredentialReference,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelPreset,
    ModelPrice,
    TokenUsage,
)
from discourselens.project import Project
from discourselens.providers import WebResearchResponse
from discourselens.spending import clear_budget, save_preset_price, set_budget, spending_summary
from discourselens.web import create_app


def price(*, currency="USD", version="v1", input_rate="0.1"):
    return ModelPrice("openai", "gpt-6-luna", input_rate, "0.5", currency,
                      "https://example.test/pricing", version, "2026-10-25")


def record(project, *, rate=None, usage=True, calls=None):
    return project.record_model_execution(
        "openai", "fixture", "gpt-6-luna", None, "request", "response",
        input_tokens=1000 if usage else None, output_tokens=500 if usage else None,
        model_price=rate, provider_usage={"web_search_calls": calls} if calls is not None else None,
    )


def test_budget_counts_confirmed_costs_once_and_keeps_historical_snapshots(tmp_path):
    with Project.init(tmp_path / "p") as project:
        record(project)  # Unpriced history predates the explicitly entered available amount.
        assert set_budget(project, "1.0000", "usd")["remaining"] == "1"
        first = record(project, rate=price(), calls=2)
        project.save_model_price(price(version="v2", input_rate="10"))
        record(project, rate=project.model_price("openai", "gpt-6-luna"))
        with Project.open(project.root, read_only=True) as reader:
            before = reader.db.total_changes
            result = spending_summary(reader)
            assert result["spent"] == "0.0306"  # 0.00035 + 0.02 search + 0.01025
            assert result["remaining"] == "0.9694"
            assert result["web_search_calls"] == 2
            assert result["web_search_cost"] == "0.02"
            assert result["execution_count"] == 2 and result["status"] == "ready"
            assert spending_summary(reader) == result  # Polling does not charge again.
            assert reader.db.total_changes == before
        stored = project.db.execute("SELECT estimated_cost FROM model_executions WHERE id=?", (first,)).fetchone()
        assert stored[0] == "0.00035"
        assert set_budget(project, "0", "USD")["remaining"] == "0"
        record(project, rate=price())
        assert spending_summary(project)["remaining"] == "-0.00035"
        clear_budget(project)
        assert spending_summary(project)["status"] == "unconfigured"
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 4


@pytest.mark.parametrize("rate,usage,currency,calls", [
    (None, True, "USD", None), (price(), False, "USD", None),
    (price(currency="EUR"), True, "USD", None), (price(currency="EUR"), True, "EUR", 1),
])
def test_unknown_costs_or_currency_never_look_like_free_calls(tmp_path, rate, usage, currency, calls):
    with Project.init(tmp_path / "p") as project:
        set_budget(project, "20", currency)
        record(project, rate=rate, usage=usage, calls=calls)
        summary = spending_summary(project)
        assert summary["remaining"] is None and summary["status"] == "incomplete"
        assert summary["unpriced_count"] or summary["currency_mismatch_count"]


def test_zero_rates_are_valid_and_vacuum_does_not_move_the_budget_start(tmp_path):
    with Project.init(tmp_path / "p") as project:
        first = record(project, rate=price())
        baseline = record(project, rate=price())
        set_budget(project, "2", "USD")
        project.db.execute("DELETE FROM model_executions WHERE id=?", (first,))
        project.db.commit()
        project.db.execute("VACUUM")
        zero = ModelPrice("openai", "gpt-6-luna", "0", "0", "USD", "Local model", "free", "2026-10-25")
        record(project, rate=zero)
        assert spending_summary(project)["remaining"] == "2"
        project.db.execute("DELETE FROM model_executions WHERE id=?", (baseline,))
        project.db.commit()
        assert spending_summary(project)["status"] == "baseline_missing"
        assert spending_summary(project)["remaining"] is None


@pytest.mark.parametrize("amount", [None, 1, True, {}, "NaN", "Infinity", "-1", "1e100000", "1,23", "1000000000", "1.000000001"])
def test_budget_rejects_ambiguous_or_unbounded_amounts(tmp_path, amount):
    with Project.init(tmp_path / "p") as project:
        with pytest.raises(ProjectError):
            set_budget(project, amount, "USD")
        assert spending_summary(project)["status"] == "unconfigured"


def test_native_search_usage_is_saved_from_actual_provider_response(tmp_path, monkeypatch):
    class Provider:
        provider_id = "openai"

        def capabilities(self, model):
            return ModelCapabilities()

        def generate(self, request):
            return WebResearchResponse(
                "Discovery", "openai", "gpt-6-luna", TokenUsage(1000, 500),
                ({"url": "https://example.test/source", "title": "Source"},), 2,
            )

    monkeypatch.setenv("DL_TEST_KEY", "fixture")
    with Project.init(tmp_path / "p") as project:
        project.set_offline_mode(False)
        project.save_credential_reference(CredentialReference("openai", "fixture", "DL_TEST_KEY"))
        project.save_model_preset(ModelPreset("linguist", "openai", "fixture", "gpt-6-luna"))
        save_preset_price(project, {"preset_name": "linguist", "input_per_million_tokens": "0.1",
                                   "output_per_million_tokens": "0.5", "currency": "USD"})
        set_budget(project, "5", "USD")
        result = execute_model_preset(project, "linguist", "Discover sources", provider_factory=lambda *args: Provider())
        row = project.db.execute("SELECT parameters_json FROM model_executions WHERE id=?", (result.execution_id,)).fetchone()
        assert json.loads(row[0])["_provider_usage"] == {"web_search_calls": 2}
        assert spending_summary(project)["remaining"] == "4.97965"


@pytest.mark.parametrize("structured", [False, True])
def test_streaming_openai_usage_reaches_the_ledger_and_budget(tmp_path, monkeypatch, structured):
    monkeypatch.setenv("DL_TEST_KEY", "fixture")
    events = [
        {"type": "response.output_text.delta", "delta": "Answer"},
        {"type": "response.completed", "response": {
            "status": "completed", "output": [{"type": "message", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "Answer"}]}],
            "usage": {"input_tokens": 1000, "output_tokens": 500,
                      "output_tokens_details": {"reasoning_tokens": 200}},
        }},
    ]
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()
    sent = []

    def transport(request, **kwargs):
        sent.append(json.loads(request.data))
        return BytesIO(body)

    monkeypatch.setattr("discourselens.providers._open_provider_request", transport)
    with Project.init(tmp_path / "p") as project:
        project.set_offline_mode(False)
        project.save_credential_reference(CredentialReference("openai", "fixture", "DL_TEST_KEY"))
        project.save_model_preset(ModelPreset("linguist", "openai", "fixture", "gpt-6-luna"))
        project.save_model_capability_profile(ModelCapabilityProfile(
            "openai", "gpt-6-luna", ModelCapabilities(streaming=True, structured_output=True),
        ))
        project.save_model_price(price())
        set_budget(project, "2", "USD")
        chunks = execute_model_preset_stream(project, "linguist", "private test question",
            **({"response_schema_name": "answer", "response_schema": {"type": "object"}} if structured else {}))
        assert "".join(chunks) == "Answer"
        row = dict(project.db.execute("SELECT * FROM model_executions").fetchone())
        assert row["input_tokens"] == 1000 and row["output_tokens"] == 500
        assert row["reasoning_tokens"] == 200
        assert row["estimated_cost"] == "0.00035"  # reasoning is a subset, charged once
        assert spending_summary(project)["remaining"] == "1.99965"
        assert "on_usage" not in sent[0] and "private test question" not in str(row)


@pytest.mark.parametrize("usage", [{"web_search_calls": True}, {"web_search_calls": -1}, {"web_search_calls": 1001}, {"surprise": 2}])
def test_untrusted_tool_usage_cannot_change_spending(tmp_path, usage):
    with Project.init(tmp_path / "p") as project:
        with pytest.raises(ProjectError):
            project.record_model_execution("openai", "fixture", "model", None, "request", "response", provider_usage=usage)
        with pytest.raises(ProjectError, match="reserved"):
            project.record_model_execution("openai", "fixture", "model", None, "request", "response", parameters={"_provider_usage": {"web_search_calls": 0}})
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0


def test_spending_api_is_authenticated_and_readable_while_a_writer_is_busy(tmp_path):
    root = tmp_path / "p"
    Project.init(root).close()
    token = secrets.token_hex(32)
    with TestClient(create_app(root, token)) as client:
        assert client.get("/api/spending").status_code == 401
        headers = {"Authorization": f"Bearer {token}"}
        assert client.post("/api/spending/budget", headers=headers, json={"amount": "5", "currency": "USD"}).status_code == 200
        with Project.open(root) as writer:
            writer.db.execute("BEGIN IMMEDIATE")
            try:
                result = client.get("/api/spending", headers=headers)
                assert result.status_code == 200 and result.json()["remaining"] == "5"
            finally:
                writer.db.rollback()
        assert client.delete("/api/spending/budget", headers=headers).status_code == 204
        assert client.get("/api/spending", headers=headers).json()["remaining"] is None


def test_password_account_budgets_are_private_and_changes_require_same_origin(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DISCOURSELENS_WEB_AUTH_MODE", "password")
    monkeypatch.setenv("DISCOURSELENS_WEB_SESSION_SECRET", secrets.token_urlsafe(40))
    monkeypatch.delenv("DISCOURSELENS_WEB_PUBLIC_ORIGIN", raising=False)
    monkeypatch.setenv("DISCOURSELENS_WEB_COOKIE_SECURE", "false")
    app = create_app(tmp_path / "web")
    origin = {"Origin": "http://testserver"}
    with TestClient(app) as client:
        assert client.post("/auth/setup", headers=origin, json={"email": "first@example.test", "password": secrets.token_urlsafe(20)}).status_code == 201
        assert client.post("/api/spending/budget", headers={"Origin": "https://elsewhere.test"}, json={"amount": "7", "currency": "USD"}).status_code == 403
        assert client.post("/api/spending/budget", headers=origin, json={"amount": "7", "currency": "USD"}).status_code == 200
        assert client.get("/api/spending").json()["remaining"] == "7"
        assert client.post("/auth/logout", headers=origin).status_code == 200
        assert client.post("/auth/register", headers=origin, json={"email": "second@example.test", "password": secrets.token_urlsafe(20)}).status_code == 201
        assert client.get("/api/spending").json()["status"] == "unconfigured"
