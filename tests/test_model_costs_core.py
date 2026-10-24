from __future__ import annotations

import json

import pytest

from discourselens import Project
from discourselens.errors import DocumentError, ProjectError
from discourselens.models import ModelPrice
from discourselens.project import _validated_model_execution_rows


def _price(version: str, input_rate: str) -> ModelPrice:
    return ModelPrice(
        provider_id="openai-compatible",
        model="fixture-model",
        input_per_million_tokens=input_rate,
        output_per_million_tokens="2",
        currency="USD",
        source_reference=f"https://provider.example/pricing-{version}",
        rate_version=version,
        rate_date=f"2026-01-0{1 if version == 'v1' else 2}",
    )


def test_model_cost_page_is_bounded_and_uses_immutable_execution_snapshots(tmp_path):
    project = Project.init(tmp_path / "model-cost-page")
    first_price = _price("v1", "1")
    second_price = _price("v2", "100")
    first_id = project.record_model_execution(
        "openai-compatible",
        "university",
        "fixture-model",
        None,
        "request-one",
        "response-one",
        input_tokens=1_000,
        output_tokens=500,
        model_price=first_price,
        endpoint="https://provider.example/v1",
    )
    second_id = project.record_model_execution(
        "openai-compatible",
        "lab",
        "fixture-model",
        None,
        "request-two",
        "response-two",
        input_tokens=1_000,
        output_tokens=500,
    )
    project.save_model_price(first_price)
    project.save_model_price(second_price)
    project.db.execute(
        "UPDATE model_executions SET started_at=? WHERE id=?",
        ("2026-01-01T00:00:01+00:00", first_id),
    )
    project.db.execute(
        "UPDATE model_executions SET started_at=? WHERE id=?",
        ("2026-01-01T00:00:02+00:00", second_id),
    )
    project.db.commit()

    before_changes = project.db.total_changes
    first_page = project.model_costs_page(offset=0, limit=1)
    second_page = project.model_costs_page(offset=1, limit=1)

    assert first_page["total_count"] == second_page["total_count"] == 2
    assert first_page["has_more"] is True
    assert second_page["has_more"] is False
    first_rows = first_page["rows"]
    second_rows = second_page["rows"]
    assert isinstance(first_rows, list) and isinstance(second_rows, list)
    assert first_rows[0]["execution_id"] == first_id
    assert first_rows[0]["estimated_cost"] == "0.002"
    assert first_rows[0]["model_price_snapshot"] == first_price.to_snapshot()
    assert "endpoint" not in first_rows[0]
    assert second_rows[0]["execution_id"] == second_id
    assert second_rows[0]["estimated_cost"] is None
    assert second_rows[0]["model_price_snapshot"] is None
    payload = json.dumps(first_page, sort_keys=True)
    for forbidden in (
        "credential_alias",
        "request_hash",
        "response_hash",
        "parameters_json",
        "university",
    ):
        assert forbidden not in payload
    assert project.db.total_changes == before_changes
    project.close()


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:password@provider.example/v1",
        "https://provider.example/v1?token=secret",
        "https://provider.example/v1#fragment",
        "https://provider.example/v1/abcdefghijklmnopqrstuvwxyz0123456789",
    ],
)
def test_model_execution_rejects_credential_material_in_endpoint(tmp_path, endpoint):
    project = Project.init(tmp_path / "unsafe-endpoint")
    with pytest.raises(DocumentError, match="endpoint must not contain credential material"):
        project.record_model_execution(
            "openai-compatible",
            "fixture",
            "fixture-model",
            None,
            "request-hash",
            "response-hash",
            endpoint=endpoint,
        )
    project.close()


def test_model_execution_summary_validation_rejects_unsafe_stored_endpoint(tmp_path):
    project = Project.init(tmp_path / "unsafe-stored-endpoint")
    execution_id = project.record_model_execution(
        "openai-compatible",
        "fixture",
        "fixture-model",
        None,
        "request-hash",
        "response-hash",
        endpoint="https://provider.example/v1",
    )
    project.db.execute(
        "UPDATE model_executions SET endpoint=? WHERE id=?",
        ("https://provider.example/v1?token=secret", execution_id),
    )
    project.db.commit()
    row = project.db.execute(
        "SELECT * FROM model_executions WHERE id=?", (execution_id,)
    ).fetchall()
    with pytest.raises(DocumentError, match="stored model execution metadata is unsafe"):
        _validated_model_execution_rows(row)
    project.close()


def test_model_cost_page_normalizes_unrecognized_snapshot_keys(tmp_path):
    project = Project.init(tmp_path / "model-cost-snapshot")
    price = _price("v1", "1")
    execution_id = project.record_model_execution(
        "openai-compatible",
        "fixture",
        "fixture-model",
        None,
        "request-hash",
        "response-hash",
        model_price=price,
    )
    project.db.execute(
        "UPDATE model_executions SET model_price_json=? WHERE id=?",
        (json.dumps({**price.to_snapshot(), "api_key": "do-not-return"}), execution_id),
    )
    project.db.commit()

    page = project.model_costs_page()
    rows = page["rows"]
    assert isinstance(rows, list)
    assert rows[0]["model_price_snapshot"] == price.to_snapshot()
    assert "do-not-return" not in json.dumps(page)
    project.close()


def test_model_cost_page_rejects_corrupt_snapshot_without_echoing_contents(tmp_path):
    project = Project.init(tmp_path / "bad-model-cost-snapshot")
    execution_id = project.record_model_execution(
        "openai-compatible",
        "fixture",
        "fixture-model",
        None,
        "request-hash",
        "response-hash",
    )
    secret = "not-for-output"
    project.db.execute(
        "UPDATE model_executions SET model_price_json=? WHERE id=?",
        (json.dumps({"rate_unit": "unknown", "api_key": secret}), execution_id),
    )
    project.db.commit()

    with pytest.raises(ProjectError, match="stored model price snapshot is invalid") as caught:
        project.model_costs_page()
    assert secret not in str(caught.value)
    project.close()


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(-1, 100), (True, 10), (0, 0), (0, 1_001), (0, True)],
)
def test_model_cost_page_rejects_invalid_bounds(tmp_path, offset, limit):
    project = Project.init(tmp_path / f"invalid-model-cost-page-{offset}-{limit}")

    with pytest.raises(ProjectError, match="model cost page"):
        project.model_costs_page(offset=offset, limit=limit)
    project.close()
