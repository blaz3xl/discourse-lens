import json
from decimal import Decimal, localcontext

import pytest

from discourselens import Project
from discourselens.errors import ModelProviderError
from discourselens.execution import execute_model_preset, snapshot_model_preset
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
    ModelPrice,
    TokenUsage,
    estimate_model_cost,
)


def _price(
    input_rate: str,
    output_rate: str,
    *,
    version: str = "v1",
    rate_date: str = "2026-01-01",
    reasoning_rate: str | None = None,
    source_reference: str = "https://prices.example/models/fixture-model",
) -> ModelPrice:
    return ModelPrice(
        provider_id="openai-compatible",
        model="fixture-model",
        input_per_million_tokens=input_rate,
        output_per_million_tokens=output_rate,
        reasoning_per_million_tokens=reasoning_rate,
        currency="USD",
        source_reference=source_reference,
        rate_version=version,
        rate_date=rate_date,
    )


def test_cost_uses_exact_decimal_arithmetic_and_prices_total_output_once():
    price = _price("2.00", "3.000")

    estimate = estimate_model_cost(
        price,
        TokenUsage(input_tokens=1_000, output_tokens=500, reasoning_tokens=200),
    )

    assert estimate == Decimal("0.0035")


def test_cost_preserves_rates_beyond_the_ambient_decimal_precision():
    price = _price("123456789012345678901234567890", "0")

    with localcontext() as context:
        context.prec = 2
        estimate = estimate_model_cost(
            price, TokenUsage(input_tokens=1, output_tokens=0)
        )

    assert estimate == Decimal("123456789012345678901234.56789")


def test_cost_is_not_underflowed_by_the_ambient_decimal_exponent_range():
    price = _price("1", "0")

    with localcontext() as context:
        context.prec = 2
        context.Emin = -2
        context.Emax = 2
        estimate = estimate_model_cost(price, TokenUsage(input_tokens=1, output_tokens=0))

    assert estimate == Decimal("0.000001")


def test_no_rate_keeps_execution_estimate_null(tmp_path):
    project = Project.init(tmp_path / "no-rate")
    try:
        execution_id = project.record_model_execution(
            "openai-compatible",
            "fixture",
            "fixture-model",
            None,
            "request-hash",
            "response-hash",
            input_tokens=1_000,
            output_tokens=500,
            reasoning_tokens=200,
        )

        row = project.db.execute(
            "SELECT model_price_json,estimated_cost,estimated_cost_currency "
            "FROM model_executions WHERE id=?",
            (execution_id,),
        ).fetchone()
        assert row["model_price_json"] is None
        assert row["estimated_cost"] is None
        assert row["estimated_cost_currency"] is None
    finally:
        project.close()


def test_missing_usage_does_not_produce_an_estimate():
    price = _price("2", "3")

    assert estimate_model_cost(price, None) is None
    assert estimate_model_cost(price, TokenUsage(input_tokens=1_000)) is None
    assert estimate_model_cost(price, TokenUsage(output_tokens=500)) is None


def test_explicit_reasoning_rate_reprices_the_reasoning_subset(tmp_path):
    project = Project.init(tmp_path / "reasoning-rate")
    try:
        price = _price("2", "3", reasoning_rate="7")
        project.save_model_price(price)

        execution_id = project.record_model_execution(
            "openai-compatible",
            "fixture",
            "fixture-model",
            None,
            "request-hash",
            "response-hash",
            input_tokens=1_000,
            output_tokens=500,
            reasoning_tokens=200,
            model_price=price,
        )

        row = project.db.execute(
            "SELECT model_price_json,estimated_cost,estimated_cost_currency "
            "FROM model_executions WHERE id=?",
            (execution_id,),
        ).fetchone()
        assert json.loads(row["model_price_json"]) == price.to_snapshot()
        assert row["estimated_cost"] == "0.0043"
        assert row["estimated_cost_currency"] == "USD"
    finally:
        project.close()


def test_reasoning_rate_requires_a_known_consistent_breakdown():
    price = _price("2", "3", reasoning_rate="7")

    assert estimate_model_cost(
        price, TokenUsage(input_tokens=1_000, output_tokens=500)
    ) is None
    assert estimate_model_cost(
        price,
        TokenUsage(input_tokens=1_000, output_tokens=500, reasoning_tokens=501),
    ) is None


def test_inconsistent_usage_does_not_produce_an_estimate_without_reasoning_rate():
    price = _price("2", "3")

    assert estimate_model_cost(
        price,
        TokenUsage(input_tokens=1_000, output_tokens=500, reasoning_tokens=501),
    ) is None


@pytest.mark.parametrize(
    "source_reference",
    [
        "https://prices.example/models/fixture?api_key=do-not-persist",
        "https://user:password@prices.example/models/fixture",
        "price source bearer private-token",
        "price source access_token=private-token",
    ],
)
def test_model_price_rejects_secret_bearing_source_references(source_reference):
    with pytest.raises(ModelProviderError, match="secret-like values"):
        _price("1", "2", source_reference=source_reference)


@pytest.mark.parametrize(
    "source_reference",
    [
        "https://prices.example/docs?sig=private-signature",
        "https://prices.example/docs?credential=private-credential",
        "https://prices.example/docs?X-Amz-Security-Token=private-token",
        "https://prices.example/docs?X-Amz-Credential=private-credential",
        "https://prices.example/docs?auth=private-auth",
    ],
)
def test_model_price_rejects_signed_url_query_parameters(source_reference):
    with pytest.raises(ModelProviderError, match="secret-like values"):
        _price("1", "2", source_reference=source_reference)


@pytest.mark.parametrize(
    "source_reference",
    [
        "https://prices.example/docs?tab=pricing",
        "https://prices.example/docs?lang=en",
        "https://prices.example/docs?model=fixture&tab=pricing",
    ],
)
def test_model_price_accepts_ordinary_documentation_reference_with_query(source_reference):
    price = _price("1", "2", source_reference=source_reference)

    assert price.source_reference == source_reference


class _UsageProvider:
    provider_id = "openai-compatible"

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            "fixture response",
            self.provider_id,
            request.model,
            TokenUsage(input_tokens=1_000, output_tokens=500),
        )


def _usage_provider_factory(_endpoint, _secret_reference, _provider_id):
    return _UsageProvider()


def test_execution_keeps_its_price_snapshot_when_new_rate_version_is_added(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "historical-price")
    monkeypatch.setenv("DL_MODEL_PRICE_TEST_KEY", "unused-test-key")
    try:
        project.save_credential_reference(
            CredentialReference(
                "openai-compatible",
                "fixture",
                "DL_MODEL_PRICE_TEST_KEY",
                "https://fixture.example/v1",
            )
        )
        project.save_model_preset(
            ModelPreset("review", "openai-compatible", "fixture", "fixture-model")
        )
        first_price = _price("1", "2", version="v1", rate_date="2026-01-01")
        project.save_model_price(first_price)
        first_snapshot = snapshot_model_preset(project, "review")

        project.save_model_price(
            _price("10", "20", version="v2", rate_date="2026-02-01")
        )
        first_execution = execute_model_preset(
            project,
            "review",
            "fixture prompt",
            provider_factory=_usage_provider_factory,
            preset_snapshot=first_snapshot,
        )
        second_execution = execute_model_preset(
            project,
            "review",
            "fixture prompt",
            provider_factory=_usage_provider_factory,
        )

        first_row = project.db.execute(
            "SELECT model_price_json,estimated_cost FROM model_executions WHERE id=?",
            (first_execution.execution_id,),
        ).fetchone()
        second_row = project.db.execute(
            "SELECT model_price_json,estimated_cost FROM model_executions WHERE id=?",
            (second_execution.execution_id,),
        ).fetchone()
        assert json.loads(first_row["model_price_json"])["rate_version"] == "v1"
        assert first_row["estimated_cost"] == "0.002"
        assert json.loads(second_row["model_price_json"])["rate_version"] == "v2"
        assert second_row["estimated_cost"] == "0.02"
    finally:
        project.close()
