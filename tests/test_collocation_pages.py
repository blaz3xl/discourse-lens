import pytest

from discourselens import Project
from discourselens.corpus import collocations_page
from discourselens.errors import ScopeViolationError


def test_collocations_page_is_stable_bounded_and_explicitly_scoped(tmp_path):
    project = Project.init(tmp_path / "collocation-pages")
    _, included_revision = project.import_text(
        "included.txt", "target zulu target alpha target beta target alpha"
    )
    project.import_text("outside.txt", "target outsider target")
    scope_id = project.snapshot([included_revision])

    first = project.corpus_collocations_page(scope_id, "target", window=1, limit=2)
    second = collocations_page(
        project, scope_id, "target", window=1, offset=2, limit=1
    )
    beyond = project.corpus_collocations_page(
        scope_id, "target", window=1, offset=10, limit=2
    )

    assert first == {
        "scope_id": scope_id,
        "query": "target",
        "window": 1,
        "offset": 0,
        "limit": 2,
        "rows": [{"term": "alpha", "count": 3}, {"term": "beta", "count": 2}],
        "total_values": 3,
        "has_more": True,
    }
    assert second["rows"] == [{"term": "zulu", "count": 2}]
    assert second["has_more"] is False
    assert beyond["rows"] == []
    assert beyond["has_more"] is False
    assert all(row["term"] != "outsider" for row in first["rows"])

    unbounded = project.corpus_collocations(scope_id, "target", window=1)
    expected_order = sorted(unbounded.items(), key=lambda item: (-item[1], item[0]))
    assert first["rows"] + second["rows"] == [
        {"term": term, "count": count} for term, count in expected_order
    ]
    project.close()


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(-1, 1), (0, 0), (0, 1_001), (True, 1), (0, True)],
)
def test_collocations_page_rejects_invalid_bounds(tmp_path, offset, limit):
    project = Project.init(tmp_path / f"collocation-page-invalid-{offset}-{limit}")
    _, revision = project.import_text("text.txt", "target alpha")
    scope_id = project.snapshot([revision])

    with pytest.raises(ScopeViolationError, match="collocation offset"):
        project.corpus_collocations_page(
            scope_id, "target", offset=offset, limit=limit
        )

    project.close()


def test_collocations_page_retains_core_scope_and_query_validation(tmp_path):
    project = Project.init(tmp_path / "collocation-page-core-validation")
    with pytest.raises(ScopeViolationError, match="scope"):
        project.corpus_collocations_page("missing-scope", "target")

    _, revision = project.import_text("text.txt", "target alpha")
    scope_id = project.snapshot([revision])
    with pytest.raises(ScopeViolationError, match="window"):
        project.corpus_collocations_page(scope_id, "target", window=0)
    with pytest.raises(ScopeViolationError):
        project.corpus_collocations_page(scope_id, "two tokens")
    project.close()


@pytest.mark.parametrize("window", [0, 101, 10**100, True])
def test_collocations_api_rejects_windows_outside_shared_contract(tmp_path, window):
    project = Project.init(tmp_path / f"collocation-window-invalid-{window}")
    _, revision = project.import_text("text.txt", "target alpha beta")
    scope_id = project.snapshot([revision])

    with pytest.raises(ScopeViolationError, match="collocation window must be between 1 and 100"):
        project.corpus_collocations(scope_id, "target", window=window)
    with pytest.raises(ScopeViolationError, match="collocation window must be between 1 and 100"):
        project.corpus_collocations_page(scope_id, "target", window=window)

    project.close()


@pytest.mark.parametrize("window", [1, 100])
def test_collocations_api_accepts_window_contract_boundaries(tmp_path, window):
    project = Project.init(tmp_path / f"collocation-window-valid-{window}")
    _, revision = project.import_text("text.txt", "alpha target beta")
    scope_id = project.snapshot([revision])

    assert project.corpus_collocations(scope_id, "target", window=window) == {
        "alpha": 1,
        "beta": 1,
    }
    page = project.corpus_collocations_page(scope_id, "target", window=window)
    assert page["window"] == window
    assert page["total_values"] == 2

    project.close()
