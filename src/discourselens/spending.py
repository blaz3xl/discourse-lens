"""A local, explicit budget backed by immutable model execution cost snapshots.

This is neither a provider wallet nor a payment system. Missing usage, prices or
currencies make the remaining estimate unavailable; they never become free calls.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING
from uuid import uuid4

from .errors import DocumentError
from .models import ModelPrice, decimal_text

if TYPE_CHECKING:
    from .project import Project

_BUDGET_KEY = "chat_spending_budget_v1"
_SEARCH_RATE = "0.01"
_SEARCH_SOURCE = "https://developers.openai.com/api/docs/pricing"
_RATE_DATE = "2026-10-25"
_MODEL_SOURCE = "https://developers.openai.com/api/docs/models/gpt-6-luna"


def _amount(value: object, label: str) -> Decimal:
    # Bound both magnitude and precision before Decimal or fixed-point rendering.
    if not isinstance(value, str) or not re.fullmatch(r"\d{1,9}(?:\.\d{1,8})?", value):
        raise DocumentError(f"{label}: usa un importe positivo con hasta ocho decimales")
    return Decimal(value)


def _currency(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z]{3}", value):
        raise DocumentError("Indica una moneda de tres letras, por ejemplo USD o EUR")
    return value.upper()


def set_budget(project: Project, amount: object, currency: object) -> dict[str, object]:
    """Take an atomic starting point; an entered amount is available FROM now."""
    project._require_writable()
    available = _amount(amount, "Saldo")
    unit = _currency(currency)
    project.db.execute("BEGIN IMMEDIATE")
    try:
        last = project.db.execute(
            "SELECT id FROM model_executions ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        budget = {
            "amount": decimal_text(available), "currency": unit,
            "baseline_execution_id": str(last[0]) if last else None,
            "created_at": datetime.now(UTC).isoformat(),
            "web_search_usd_per_call": _SEARCH_RATE,
            "web_search_rate_date": _RATE_DATE, "web_search_source": _SEARCH_SOURCE,
        }
        project.db.execute(
            "INSERT INTO project_meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (_BUDGET_KEY, json.dumps(budget, separators=(",", ":"))),
        )
        project.db.commit()
    except BaseException:
        project.db.rollback()
        raise
    return spending_summary(project)


def clear_budget(project: Project) -> None:
    project._require_writable()
    project.db.execute("DELETE FROM project_meta WHERE key=?", (_BUDGET_KEY,))
    project.db.commit()


def spending_summary(project: Project) -> dict[str, object]:
    """Read confirmed executions only. Polling cannot reserve or debit money."""
    result: dict[str, object] = {
        "kind": "local_estimate", "status": "unconfigured", "budget": None,
        "remaining": None, "spent": "0", "execution_count": 0,
        "unpriced_count": 0, "currency_mismatch_count": 0,
        "web_search_calls": 0, "web_search_cost": "0",
        "scope": "recorded_text_model_calls_and_native_web_search",
    }
    row = project.db.execute("SELECT value FROM project_meta WHERE key=?", (_BUDGET_KEY,)).fetchone()
    if row is None:
        return result
    budget = json.loads(row[0])
    result["budget"] = budget
    baseline = budget["baseline_execution_id"]
    baseline_row = project.db.execute(
        "SELECT rowid FROM model_executions WHERE id=?", (baseline,)
    ).fetchone() if baseline else None
    # Resolve the ID every time: VACUUM may renumber implicit rowids.
    if baseline and baseline_row is None:
        result["status"] = "baseline_missing"
        return result
    rows = project.db.execute(
        "SELECT estimated_cost,estimated_cost_currency,parameters_json "
        "FROM model_executions WHERE rowid>? ORDER BY rowid",
        (baseline_row[0] if baseline_row else 0,),
    )
    spent, search_cost = Decimal(0), Decimal(0)
    count = unpriced = mismatched = searches = 0
    for execution in rows:
        count += 1
        cost = execution["estimated_cost"]
        if cost is None:
            unpriced += 1
        elif execution["estimated_cost_currency"] != budget["currency"]:
            mismatched += 1
        else:
            try:
                parsed = Decimal(cost)
                if not parsed.is_finite() or parsed < 0:
                    raise InvalidOperation
                spent += parsed
            except (InvalidOperation, ValueError):
                unpriced += 1
        usage = json.loads(execution["parameters_json"]).get("_provider_usage", {})
        calls = usage.get("web_search_calls", 0)
        if type(calls) is not int or not 0 <= calls <= 1000:
            unpriced += 1
            continue
        searches += calls
        if calls:
            if budget["currency"] != "USD":
                mismatched += 1
            else:
                search_cost += Decimal(budget["web_search_usd_per_call"]) * calls
    spent += search_cost
    complete = not unpriced and not mismatched
    result.update(
        status="ready" if complete else "incomplete",
        remaining=decimal_text(Decimal(budget["amount"]) - spent) if complete else None,
        spent=decimal_text(spent), execution_count=count, unpriced_count=unpriced,
        currency_mismatch_count=mismatched, web_search_calls=searches,
        web_search_cost=decimal_text(search_cost),
    )
    return result


def price_for_preset(project: Project, preset_name: str | None) -> dict[str, object] | None:
    name = preset_name or project.default_chat_model_preset()
    if not name:
        return None
    preset = project.model_preset(name)
    provider, model = str(preset["provider_id"]), str(preset["model"])
    price = project.model_price(provider, model)
    result: dict[str, object] = {
        "preset_name": name, "provider_id": provider, "model": model,
        "saved": price.to_snapshot() if price else None, "suggestion": None,
    }
    if provider == "openai" and model == "gpt-6-luna":
        result["suggestion"] = {
            "input_per_million_tokens": "0.10", "output_per_million_tokens": "0.50",
            "currency": "USD", "source_reference": _MODEL_SOURCE, "rate_date": _RATE_DATE,
        }
    return result


def save_preset_price(project: Project, payload: dict[str, object]) -> str:
    name = payload.get("preset_name")
    if not isinstance(name, str) or not name or len(name) > 256:
        raise DocumentError("Selecciona un modelo guardado para configurar su tarifa")
    preset = project.model_preset(name)
    input_rate = _amount(payload.get("input_per_million_tokens"), "Tarifa de entrada")
    output_rate = _amount(payload.get("output_per_million_tokens"), "Tarifa de salida")
    source = payload.get("source_reference", "Configuración manual del usuario")
    if not isinstance(source, str) or not source.strip() or len(source) > 2048:
        raise DocumentError("Indica una referencia de tarifa de hasta 2048 caracteres")
    price = ModelPrice(
        provider_id=str(preset["provider_id"]), model=str(preset["model"]),
        input_per_million_tokens=input_rate, output_per_million_tokens=output_rate,
        currency=_currency(payload.get("currency")), source_reference=source,
        rate_date=datetime.now(UTC).date().isoformat(), rate_version=f"web-{uuid4().hex}",
    )
    return project.save_model_price(price)
