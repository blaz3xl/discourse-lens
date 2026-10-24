"""Provider-specific reasoning constraints shared across save, preview and execution."""

from __future__ import annotations

from collections.abc import Mapping

from .errors import ModelProviderError

COHERE_MINIMUM_FINAL_RESPONSE_TOKENS = 1_000


def validate_cohere_manual_budget(
    budget_tokens: int | None,
    parameters: Mapping[str, object],
) -> None:
    """Validate Cohere's budgeted-thinking request and preserve response headroom.

    Cohere documents small thinking budgets (including 500 tokens) and recommends
    leaving at least 1K tokens for the final response. Requiring ``max_tokens`` to
    be explicit avoids relying on model/API defaults when checking that headroom.
    """
    if (
        isinstance(budget_tokens, bool)
        or not isinstance(budget_tokens, int)
        or budget_tokens < 1
    ):
        raise ModelProviderError(
            "Cohere manual reasoning requires a token budget that is a positive integer"
        )

    max_tokens = parameters.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ModelProviderError(
            "Cohere manual reasoning requires an explicit positive max_tokens value"
        )
    if budget_tokens + COHERE_MINIMUM_FINAL_RESPONSE_TOKENS > max_tokens:
        raise ModelProviderError(
            "Cohere manual reasoning must leave at least 1000 max_tokens for the final response"
        )
