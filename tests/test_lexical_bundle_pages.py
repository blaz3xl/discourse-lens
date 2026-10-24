import pytest

from discourselens import Project
from discourselens.corpus import lexical_bundles
from discourselens.errors import ScopeViolationError


def test_lexical_bundle_pages_are_stable_scoped_and_empty_past_end(tmp_path):
    project = Project.init(tmp_path / "lexical-bundle-pages")
    _, first = project.import_text("first.txt", "zeta beta alpha")
    _, second = project.import_text("second.txt", "gamma delta epsilon")
    project.import_text("outside.txt", "outside scope bundle")
    scope = project.snapshot([first, second])

    first_page = project.corpus_lexical_bundles(
        scope, size=2, min_frequency=1, offset=0, limit=2
    )
    continuation = lexical_bundles(
        project, scope, size=2, min_frequency=1, offset=2, limit=2
    )
    empty_page = lexical_bundles(
        project, scope, size=2, min_frequency=1, offset=4, limit=2
    )

    assert [row["bundle"] for row in first_page["rows"]] == [
        "beta alpha",
        "delta epsilon",
    ]
    assert [row["bundle"] for row in continuation["rows"]] == [
        "gamma delta",
        "zeta beta",
    ]
    assert first_page["has_more"] is True
    assert first_page["truncated"] is True
    assert continuation["has_more"] is False
    assert continuation["truncated"] is False
    assert empty_page["rows"] == []
    assert empty_page["has_more"] is False
    assert empty_page["truncated"] is False
    for page, offset in ((first_page, 0), (continuation, 2), (empty_page, 4)):
        assert page["offset"] == offset
        assert page["limit"] == 2
        assert page["candidate_count"] == 4
        assert page["scope_id"] == scope
        assert all("outside" not in row["bundle"] for row in page["rows"])


def test_lexical_bundle_v1_keeps_its_explicit_legacy_tokenizer(tmp_path):
    project = Project.init(tmp_path / "lexical-bundle-v1-tokenizer")
    _, revision = project.import_text("unicode.txt", "cafe\u0301 one")
    scope = project.snapshot([revision])

    result = lexical_bundles(project, scope, size=2, min_frequency=1)

    assert result["calculation_version"] == "surface-lexical-bundles-v1"
    assert [row["bundle"] for row in result["rows"]] == ["cafe one"]


@pytest.mark.parametrize("offset", [True, -1, 1.5])
def test_lexical_bundle_pages_reject_invalid_offsets(tmp_path, offset):
    project = Project.init(tmp_path / "invalid-lexical-bundle-offset")
    _, revision = project.import_text("sample.txt", "one two three")
    scope = project.snapshot([revision])

    with pytest.raises(ScopeViolationError, match="offset must be non-negative"):
        lexical_bundles(project, scope, size=2, min_frequency=1, offset=offset)
