import pytest

from discourselens import Project
from discourselens.corpus import document_dispersion_page
from discourselens.errors import ScopeViolationError


def test_dispersion_pages_preserve_scope_order_and_exhaustive_totals(tmp_path):
    project = Project.init(tmp_path / "dispersion-pages")
    _, first_revision = project.import_text("first.txt", "migration policy")
    _, second_revision = project.import_text("second.txt", "public policy")
    _, third_revision = project.import_text("third.txt", "migration migration")
    _, outside_revision = project.import_text("outside.txt", "migration outside the selected scope")
    scope_id = project.snapshot([first_revision, second_revision, third_revision])

    full = project.corpus_dispersion(scope_id, "migration")
    first = project.corpus_dispersion_page(scope_id, "migration", limit=2)
    second = document_dispersion_page(
        project, scope_id, "migration", offset=2, limit=2
    )

    summary_fields = (
        "term",
        "scope_id",
        "calculation_version",
        "document_count",
        "document_frequency",
        "document_range",
        "total_occurrences",
        "population_tokens",
    )
    assert {key: first[key] for key in summary_fields} == {
        key: full[key] for key in summary_fields
    }
    assert first["documents"] == full["documents"][:2]
    assert second["documents"] == full["documents"][2:]
    assert first["total_rows"] == second["total_rows"] == len(full["documents"]) == 3
    assert first["has_more"] is True
    assert second["has_more"] is False
    assert [row["revision_id"] for row in first["documents"] + second["documents"]] == [
        row["revision_id"] for row in full["documents"]
    ]
    assert all(
        row["revision_id"] != outside_revision
        for row in first["documents"] + second["documents"]
    )
    project.close()


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(-1, 1), (0, 0), (0, 1_001), (True, 1), (0, True)],
)
def test_dispersion_page_rejects_invalid_bounds(tmp_path, offset, limit):
    project = Project.init(tmp_path / f"dispersion-page-invalid-{offset}-{limit}")
    _, revision = project.import_text("text.txt", "migration policy")
    scope_id = project.snapshot([revision])

    with pytest.raises(ScopeViolationError, match="dispersion offset"):
        project.corpus_dispersion_page(scope_id, "migration", offset=offset, limit=limit)

    project.close()


def test_dispersion_page_keeps_nonempty_scope_and_population_validation(tmp_path):
    project = Project.init(tmp_path / "dispersion-page-validation")
    with pytest.raises(ScopeViolationError, match="scope"):
        project.corpus_dispersion_page("missing-scope", "migration")

    with pytest.raises(ScopeViolationError, match="scope is required"):
        project.snapshot([])

    _, empty_revision = project.import_text("empty.txt", "  \n")
    empty_population_scope = project.snapshot([empty_revision])
    with pytest.raises(ScopeViolationError, match="non-empty token population"):
        project.corpus_dispersion_page(empty_population_scope, "migration")
    project.close()
