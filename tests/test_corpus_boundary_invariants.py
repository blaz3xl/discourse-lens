"""Boundary invariants for deterministic scope-bound corpus operations."""

from typing import cast

import pytest
from pytest import approx

from discourselens import Project
from discourselens.corpus import (
    collocation_analysis,
    collocation_association_evidence,
    collocations,
    compare_documents,
    compare_frequency,
    compare_frequency_page,
    frequency_page,
    kwic,
    kwic_evidence,
    kwic_page,
    lexical_bundles,
    ngrams,
)
from discourselens.errors import EvidenceValidationError, ScopeViolationError


def test_ngrams_and_collocations_do_not_cross_scoped_revision_boundaries(tmp_path):
    project = Project.init(tmp_path / "corpus-boundaries")
    _, first = project.import_text("first.txt", "red target")
    _, second = project.import_text("second.txt", "boundary blue")
    project.import_text("outside.txt", "target leaked")
    scope = project.snapshot([first, second])

    bigrams = ngrams(project, scope, 2)
    neighbors = collocations(project, scope, "target", window=1)

    assert bigrams == {("red", "target"): 1, ("boundary", "blue"): 1}
    assert ("target", "boundary") not in bigrams
    assert neighbors == {"red": 1}
    assert "leaked" not in neighbors


def test_normalized_frequency_comparison_uses_only_each_snapshot(tmp_path):
    project = Project.init(tmp_path / "normalized-frequency-boundaries")
    _, left_one = project.import_text("left-one.txt", "apple pear")
    _, left_two = project.import_text("left-two.txt", "pear")
    project.import_text("left-outside.txt", "apple apple apple")
    _, right_one = project.import_text("right-one.txt", "apple apple pear")
    _, right_two = project.import_text("right-two.txt", "pear")
    left_scope = project.snapshot([left_one, left_two])
    right_scope = project.snapshot([right_one, right_two])

    row = next(
        item
        for item in compare_frequency(project, left_scope, right_scope)
        if item["token"] == "apple"
    )

    assert row["left_count"] == 1
    assert row["right_count"] == 2
    assert row["left_rate"] == approx(1 / 3)
    assert row["right_rate"] == approx(1 / 2)
    assert row["rate_difference"] == approx(-1 / 6)


def test_kwic_offsets_are_unicode_codepoint_half_open_spans_inside_snapshot(tmp_path):
    project = Project.init(tmp_path / "kwic-unicode-boundary")
    _, selected = project.import_text("selected.txt", "🧭 Café migration!")
    project.import_text("outside.txt", "migration")
    scope = project.snapshot([selected])

    rows = kwic(project, scope, "migration", window=0)

    assert len(rows) == 1
    row = rows[0]
    assert row["revision_id"] == selected
    assert (row["start_offset"], row["end_offset"]) == (7, 16)
    assert "🧭 Café migration!"[row["start_offset"] : row["end_offset"]] == row["keyword"]
    assert row["left"] == row["right"] == ""


@pytest.mark.parametrize(
    ("offset", "limit"),
    [
        (True, 1),
        (float("nan"), 1),
        (float("inf"), 1),
        (0, True),
        (0, float("inf")),
    ],
)
def test_frequency_page_rejects_non_integer_pagination(tmp_path, offset, limit):
    project = Project.init(tmp_path / "frequency-numeric-boundary")
    _, revision = project.import_text("sample.txt", "alpha beta alpha")
    scope = project.snapshot([revision])

    with pytest.raises(ScopeViolationError, match="frequency offset"):
        frequency_page(project, scope, offset=offset, limit=limit)


@pytest.mark.parametrize(
    "window",
    [True, 1.5, float("nan"), float("inf"), 10**100],
)
def test_kwic_rejects_non_integer_and_overflow_windows(tmp_path, window):
    project = Project.init(tmp_path / "kwic-numeric-boundary")
    _, revision = project.import_text("sample.txt", "alpha beta alpha")
    scope = project.snapshot([revision])

    with pytest.raises(ValueError, match="window"):
        kwic(project, scope, "alpha", window=window)


def test_kwic_page_rejects_non_integer_pagination(tmp_path):
    project = Project.init(tmp_path / "kwic-page-numeric-boundary")
    _, revision = project.import_text("sample.txt", "alpha beta alpha")
    scope = project.snapshot([revision])

    with pytest.raises(ScopeViolationError, match="KWIC offset"):
        kwic_page(project, scope, "alpha", offset=cast(int, float("nan")))


def test_boolean_kwic_offsets_cannot_create_citable_evidence(tmp_path):
    project = Project.init(tmp_path / "kwic-evidence-numeric-boundary")
    _, revision = project.import_text("sample.txt", "A beta")
    scope = project.snapshot([revision])

    with pytest.raises(EvidenceValidationError, match="offsets are not a KWIC hit"):
        kwic_evidence(project, scope, revision, "A", False, True)


def test_boolean_ngram_size_is_not_coerced_to_unigrams(tmp_path):
    project = Project.init(tmp_path / "ngrams-numeric-boundary")
    _, revision = project.import_text("sample.txt", "alpha beta")
    scope = project.snapshot([revision])

    with pytest.raises(ValueError, match="n-gram size"):
        ngrams(project, scope, True)


def test_nan_threshold_cannot_silently_change_statistical_results(tmp_path):
    project = Project.init(tmp_path / "statistics-numeric-boundary")
    _, revision = project.import_text("sample.txt", "red target blue red target blue")
    scope = project.snapshot([revision])

    with pytest.raises(ScopeViolationError, match="minimum bundle frequency"):
        lexical_bundles(project, scope, size=2, min_frequency=cast(int, float("nan")))

    with pytest.raises(ScopeViolationError, match="min_frequency"):
        collocation_analysis(
            project, scope, "target", window=1, min_frequency=cast(int, float("nan"))
        )

    with pytest.raises(ScopeViolationError, match="minimum frequency"):
        collocation_association_evidence(
            project,
            scope,
            "target",
            "red",
            window=1,
            min_frequency=cast(int, float("inf")),
        )


def test_boolean_collocation_window_is_not_coerced_to_one(tmp_path):
    project = Project.init(tmp_path / "collocations-numeric-boundary")
    _, revision = project.import_text("sample.txt", "left target middle right")
    scope = project.snapshot([revision])

    with pytest.raises(ScopeViolationError, match="window"):
        collocations(project, scope, "target", window=True)


def test_comparison_pages_reject_non_integer_offsets(tmp_path):
    project = Project.init(tmp_path / "comparison-numeric-boundary")
    _, left = project.import_text("left.txt", "alpha beta")
    _, right = project.import_text("right.txt", "beta gamma")
    left_scope = project.snapshot([left])
    right_scope = project.snapshot([right])

    with pytest.raises(EvidenceValidationError, match="comparison page"):
        compare_frequency_page(
            project, left_scope, right_scope, offset=cast(int, float("nan"))
        )


def test_persisted_comparison_rejects_bad_pagination_before_snapshot_creation(tmp_path):
    project = Project.init(tmp_path / "persisted-comparison-numeric-boundary")
    _, left = project.import_text("left.txt", "alpha beta")
    _, right = project.import_text("right.txt", "beta gamma")

    with pytest.raises(ScopeViolationError, match="comparison page"):
        compare_documents(project, left, right, offset=cast(int, float("nan")))

    assert project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == 0
