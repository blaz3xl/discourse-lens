import json
from typing import cast

import pytest

from benchmarks.corpus_workload import (
    WorkloadOptions,
    _synthetic_document_text,
    _validate_options,
    run_benchmark,
)


def test_workload_records_import_scope_frequency_kwic_and_fts_measurements():
    result = run_benchmark(
        WorkloadOptions(documents=2, tokens_per_document=35, seed=123, fts_search=True)
    )

    assert result["benchmark"] == "synthetic-local-corpus-workload"
    notice = result["notice"]
    assert isinstance(notice, str) and "not scientific performance claims" in notice
    json.dumps(result)

    dimensions = cast(dict[str, object], result["corpus_dimensions"])
    assert dimensions["documents"] == 2
    assert dimensions["scope_revisions"] == 2
    assert dimensions["requested_tokens"] == 70
    assert dimensions["observed_tokens"] == 70
    assert dimensions["search_term_frequency"] == 6
    assert dimensions["kwic_rows"] == 6
    assert dimensions["fts_matches"] == 2

    elapsed = cast(dict[str, float | None], result["elapsed_seconds"])
    assert set(elapsed) == {
        "text_generation",
        "project_initialization",
        "document_import",
        "scope_creation",
        "frequency",
        "kwic",
        "fts_search",
    }
    assert all(value is not None and value >= 0 for value in elapsed.values())

    size = cast(dict[str, int], result["project_size_bytes"])
    assert size["project_total_bytes"] >= size["database_bytes"] > 0
    assert size["managed_source_bytes"] > 0
    assert size["artifact_bytes"] == 0


def test_generated_document_text_is_deterministic_and_has_requested_tokens():
    first = _synthetic_document_text(document_index=4, token_count=31, seed=987)
    repeated = _synthetic_document_text(document_index=4, token_count=31, seed=987)
    other_seed = _synthetic_document_text(document_index=4, token_count=31, seed=988)

    assert first == repeated
    assert first != other_seed
    assert len(first.split()) == 31


def test_workload_limits_require_explicit_opt_in_and_remain_bounded():
    with pytest.raises(ValueError, match="require --allow-large"):
        _validate_options(WorkloadOptions(documents=11, tokens_per_document=120))

    _validate_options(
        WorkloadOptions(documents=11, tokens_per_document=1_001, allow_large=True)
    )
    with pytest.raises(ValueError, match="total generated tokens"):
        _validate_options(
            WorkloadOptions(documents=11, tokens_per_document=100_000, allow_large=True)
        )
