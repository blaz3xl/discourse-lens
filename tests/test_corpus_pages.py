import pytest

from discourselens import Project
from discourselens.corpus import ngram_page
from discourselens.errors import ScopeViolationError


def test_ngram_page_is_stable_bounded_and_explicitly_scoped(tmp_path):
    project = Project.init(tmp_path / "ngram-pages")
    _, included_revision = project.import_text(
        "included.txt", "zeta alpha beta alpha beta"
    )
    project.import_text("outside.txt", "outside alpha beta")
    scope_id = project.snapshot([included_revision])

    first = project.corpus_ngram_page(scope_id, 2, limit=2)
    second = ngram_page(project, scope_id, 2, offset=1, limit=1)
    beyond = project.corpus_ngram_page(scope_id, 2, offset=10, limit=2)

    assert first == {
        "scope_id": scope_id,
        "size": 2,
        "offset": 0,
        "limit": 2,
        "rows": [
            {"ngram": "alpha beta", "count": 2},
            {"ngram": "beta alpha", "count": 1},
        ],
        "total_values": 3,
        "has_more": True,
    }
    assert second["rows"] == [{"ngram": "beta alpha", "count": 1}]
    assert second["has_more"] is True
    assert beyond["rows"] == []
    assert beyond["has_more"] is False
    assert first["total_values"] == second["total_values"] == beyond["total_values"]
    assert all("outside" not in row["ngram"] for row in first["rows"])

    unbounded = project.corpus_ngrams(scope_id, 2)
    expected_order = sorted(unbounded.items(), key=lambda item: (-item[1], item[0]))
    assert first["rows"] == [
        {"ngram": " ".join(terms), "count": count}
        for terms, count in expected_order[:2]
    ]
    project.close()


def test_ngram_v2_normalizes_decomposed_and_composed_unicode_like_frequency(tmp_path):
    project = Project.init(tmp_path / "ngram-nfc")
    _, decomposed = project.import_text("nfd.txt", "cafe\u0301 policy")
    _, composed = project.import_text("nfc.txt", "café policy")
    scope_id = project.snapshot([decomposed, composed])

    counts = project.corpus_ngrams(scope_id, 2)
    page = project.corpus_ngram_page(scope_id, 2)
    frequencies = project.corpus_frequency(scope_id)

    assert counts == {("café", "policy"): 2}
    assert page["rows"] == [{"ngram": "café policy", "count": 2}]
    assert frequencies["café"] == frequencies["policy"] == 2
    project.close()


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(-1, 1), (0, 0), (0, 1_001), (True, 1), (0, True)],
)
def test_ngram_page_rejects_invalid_bounds(tmp_path, offset, limit):
    project = Project.init(tmp_path / f"ngram-page-invalid-{offset}-{limit}")
    _, revision = project.import_text("text.txt", "alpha beta")
    scope_id = project.snapshot([revision])

    with pytest.raises(ScopeViolationError, match="n-gram offset"):
        project.corpus_ngram_page(scope_id, 2, offset=offset, limit=limit)

    project.close()
