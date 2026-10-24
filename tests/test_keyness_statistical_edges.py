from math import log
from typing import cast

import pytest

from discourselens import Project
from discourselens.corpus import keyness_log_likelihood
from discourselens.errors import ScopeViolationError
from discourselens.validation import validate_finding


def test_keyness_matches_hand_calculated_g2_for_explicit_populations(tmp_path):
    project = Project.init(tmp_path / "keyness-populations")
    _, left_first = project.import_text("left-first.txt", "alpha alpha alpha beta")
    _, left_second = project.import_text("left-second.txt", "beta")
    _, right_first = project.import_text("right-first.txt", "alpha gamma gamma")
    _, right_second = project.import_text("right-second.txt", "gamma")
    project.import_text("outside-left.txt", "gamma gamma gamma")
    project.import_text("outside-right.txt", "beta beta beta beta outsideonly")

    left_scope = project.snapshot([left_first, left_second])
    right_scope = project.snapshot([right_first, right_second])
    rows = {
        str(row["token"]): row for row in keyness_log_likelihood(project, left_scope, right_scope)
    }

    # Scope totals are L=5 and R=4. The union vocabulary is exactly alpha,
    # beta, gamma; unselected revisions and absent-on-both terms contribute none.
    assert set(rows) == {"alpha", "beta", "gamma"}
    assert (rows["alpha"]["left_count"], rows["alpha"]["right_count"]) == (3, 1)
    assert (rows["beta"]["left_count"], rows["beta"]["right_count"]) == (2, 0)
    assert (rows["gamma"]["left_count"], rows["gamma"]["right_count"]) == (0, 3)

    # G² is computed from the full 2×2 table: focal-token counts and their
    # non-focal complements. A zero observed cell contributes the limiting
    # value 0, but the complementary observed cells still contribute.
    alpha_g2 = 2 * (3 * log(27 / 20) + log(9 / 16) + 2 * log(18 / 25) + 3 * log(27 / 20))
    beta_g2 = 2 * (2 * log(9 / 5) + 3 * log(27 / 35) + 4 * log(9 / 7))
    gamma_g2 = 2 * (3 * log(9 / 4) + 5 * log(3 / 2) + log(3 / 8))
    assert rows["alpha"]["log_likelihood"] == pytest.approx(alpha_g2)
    assert rows["beta"]["log_likelihood"] == pytest.approx(beta_g2)
    assert rows["gamma"]["log_likelihood"] == pytest.approx(-gamma_g2)
    assert float(rows["alpha"]["log_likelihood"]) > 0
    assert float(rows["beta"]["log_likelihood"]) > 0
    assert float(rows["gamma"]["log_likelihood"]) < 0


def test_keyness_rejects_a_scope_with_zero_token_population(tmp_path):
    project = Project.init(tmp_path / "keyness-empty-population")
    _, empty_revision = project.import_text("empty.txt", "!!!")
    _, populated_revision = project.import_text("populated.txt", "one two")

    with pytest.raises(ScopeViolationError, match="two non-empty explicit scopes"):
        keyness_log_likelihood(
            project,
            project.snapshot([empty_revision]),
            project.snapshot([populated_revision]),
        )


def test_legacy_v1_keyness_comparison_remains_readable_and_validatable(tmp_path):
    project = Project.init(tmp_path / "legacy-keyness")
    _, left = project.import_text("left.txt", "alpha alpha beta")
    _, right = project.import_text("right.txt", "alpha gamma")
    left_scope = project.snapshot([left])
    right_scope = project.snapshot([right])

    # Preserve a result created by the former focal-cell-only v1 calculation.
    # Historical rows are integrity-checked as recorded; opening them must not
    # recalculate them under the corrected v2 statistic.
    legacy_rows = [
        {
            "token": "alpha",
            "left_count": 2,
            "right_count": 1,
            "log_likelihood": 2 * (2 * log(2 / (3 * 3 / 5)) + log(1 / (2 * 3 / 5))),
        },
        {"token": "beta", "left_count": 1, "right_count": 0, "log_likelihood": 2 * log(5 / 3)},
        {"token": "gamma", "left_count": 0, "right_count": 1, "log_likelihood": -2 * log(5 / 3)},
    ]
    comparison_id = project.create_comparison_run_stream(
        left_scope,
        right_scope,
        metric="keyness",
        calculation_version="surface-comparison-v1",
        config={"legacy_fixture": True},
        rows=legacy_rows,
    )

    persisted = project.comparison_run(comparison_id)
    page = project.comparison_rows_page(comparison_id, limit=10)
    page_rows = cast(list[dict[str, object]], page["rows"])
    finding = project.create_comparative_finding(
        comparison_id,
        "alpha",
        comparability_basis="Generated legacy fixture with explicit populations.",
    )

    assert persisted["calculation_version"] == "surface-comparison-v1"
    assert page["result_hash_verified"] is True
    assert any(row["token"] == "alpha" for row in page_rows)
    validation = cast(dict[str, object], finding["validation"])
    assert validation["valid"] is True
    assert validate_finding(project, str(finding["finding_id"])).valid
