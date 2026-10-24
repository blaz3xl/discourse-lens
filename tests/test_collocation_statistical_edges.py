import sqlite3
from math import log2

import pytest

from discourselens import Project
from discourselens.corpus import collocation_analysis, collocation_association_evidence
from discourselens.errors import ScopeViolationError


def test_collocation_scores_match_hand_counts_at_window_and_revision_edges(tmp_path):
    project = Project.init(tmp_path / "collocation-edges")
    _, first = project.import_text("first.txt", "a focus b c focus d")
    _, second = project.import_text("second.txt", "focus focus d")
    project.import_text("outside.txt", "focus outsider")
    scope = project.snapshot([first, second])

    result = collocation_analysis(
        project, scope, "focus", window=2, min_frequency=1
    )

    # N=9 and F=4, so there are 5 non-focus tokens. M=8: six valid
    # neighbors in the first revision and one `d` neighbor for each edge focus
    # in the second. The adjacent focus token is not a non-focus slot, and the
    # revisions cannot contribute a cross-boundary pair. `b` and `c` are each
    # one corpus token but two observed pairs because the two focus windows overlap.
    assert result["scope_id"] == scope
    assert result["population_tokens"] == 9
    assert result["focus_frequency"] == 4
    assert result["pair_opportunities"] == 8

    rows = {row["term"]: row for row in result["rows"]}
    assert set(rows) == {"a", "b", "c", "d"}
    assert (rows["a"]["token_frequency"], rows["a"]["pair_frequency"]) == (1, 1)
    assert (rows["b"]["token_frequency"], rows["b"]["pair_frequency"]) == (1, 2)
    assert (rows["c"]["token_frequency"], rows["c"]["pair_frequency"]) == (1, 2)
    assert (rows["d"]["token_frequency"], rows["d"]["pair_frequency"]) == (2, 3)

    # E=M*f/(N-F): 8/5 for singletons and 16/5 for `d`.
    assert rows["a"]["expected_frequency"] == pytest.approx(8 / 5)
    assert rows["b"]["expected_frequency"] == pytest.approx(8 / 5)
    assert rows["c"]["expected_frequency"] == pytest.approx(8 / 5)
    assert rows["d"]["expected_frequency"] == pytest.approx(16 / 5)
    assert rows["a"]["mutual_information"] == pytest.approx(log2(5 / 8))
    assert rows["b"]["mutual_information"] == pytest.approx(log2(5 / 4))
    assert rows["c"]["mutual_information"] == pytest.approx(log2(5 / 4))
    assert rows["d"]["mutual_information"] == pytest.approx(log2(15 / 16))
    assert rows["a"]["t_score"] == pytest.approx(-3 / 5)
    assert rows["b"]["t_score"] == pytest.approx((2 / 5) / 2**0.5)
    assert rows["c"]["t_score"] == pytest.approx((2 / 5) / 2**0.5)
    assert rows["d"]["t_score"] == pytest.approx((-1 / 5) / 3**0.5)
    assert "outsider" not in rows

    filtered = collocation_analysis(
        project, scope, "focus", window=2, min_frequency=2, limit=2
    )
    assert [row["term"] for row in filtered["rows"]] == ["b", "c"]
    assert filtered["truncated"] is True


def test_collocation_all_focus_population_has_no_undefined_association_scores(tmp_path):
    project = Project.init(tmp_path / "association-all-focus")
    _, revision = project.import_text("focus-only.txt", "focus focus focus")
    scope = project.snapshot([revision])

    result = collocation_analysis(project, scope, "focus", window=2, min_frequency=1)

    assert result["population_tokens"] == 3
    assert result["focus_frequency"] == 3
    assert result["pair_opportunities"] == 0
    assert result["rows"] == []
    with pytest.raises(ScopeViolationError, match="pair-frequency threshold"):
        collocation_association_evidence(
            project, scope, "focus", "other", window=2, min_frequency=1
        )


def test_collocation_evidence_rolls_back_coverage_when_evidence_insert_fails(tmp_path):
    project = Project.init(tmp_path / "association-evidence-atomicity")
    _, revision = project.import_text("sample.txt", "apple red apple red apple blue")
    scope = project.snapshot([revision])
    counts_before = (
        project.db.execute("SELECT COUNT(*) FROM coverage_descriptors").fetchone()[0],
        project.db.execute("SELECT COUNT(*) FROM evidence_artifacts").fetchone()[0],
    )
    project.db.execute(
        "CREATE TEMP TRIGGER reject_corpus_statistic BEFORE INSERT ON main.evidence_artifacts "
        "BEGIN SELECT RAISE(ABORT, 'injected evidence write failure'); END"
    )

    with pytest.raises(sqlite3.IntegrityError, match="injected evidence write failure"):
        collocation_association_evidence(
            project, scope, "apple", "red", window=1, min_frequency=1
        )

    counts_after = (
        project.db.execute("SELECT COUNT(*) FROM coverage_descriptors").fetchone()[0],
        project.db.execute("SELECT COUNT(*) FROM evidence_artifacts").fetchone()[0],
    )
    assert counts_after == counts_before
