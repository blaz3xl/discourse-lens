from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.mcp_server import corpus_ngrams


def test_mcp_ngram_pages_match_sdk_order_and_remain_revision_scoped(tmp_path, monkeypatch):
    root = tmp_path / "mcp-ngram-pages"
    project = Project.init(root)
    _, revision = project.import_text("included.txt", "zeta alpha beta alpha beta")
    project.import_text("outside.txt", "outside gamma delta")
    scope_id = project.snapshot([revision])
    sdk_first = project.corpus_ngram_page(scope_id, 2, limit=2)
    sdk_next = project.corpus_ngram_page(scope_id, 2, offset=2, limit=2)
    project.close()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    mcp_first = corpus_ngrams(revision, 2, limit=2)
    mcp_next = corpus_ngrams(revision, 2, offset=2, limit=2)

    assert {key: value for key, value in mcp_first.items() if key != "truncated"} == {
        key: value for key, value in sdk_first.items() if key != "scope_id"
    }
    assert {key: value for key, value in mcp_next.items() if key != "truncated"} == {
        key: value for key, value in sdk_next.items() if key != "scope_id"
    }
    assert mcp_first["truncated"] is mcp_first["has_more"] is True
    assert mcp_next["truncated"] is mcp_next["has_more"] is False
    mcp_rows = cast(list[dict[str, object]], mcp_first["rows"])
    assert all("outside" not in str(row["ngram"]) for row in mcp_rows)
    assert mcp_first["total_values"] == 3


def test_mcp_ngram_page_uses_nfc_normalized_surface_tokens(tmp_path, monkeypatch):
    root = tmp_path / "mcp-ngram-nfc"
    project = Project.init(root)
    _, revision = project.import_text("nfd.txt", "cafe\u0301 policy")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    result = corpus_ngrams(revision, 2, limit=5)

    rows = cast(list[dict[str, object]], result["rows"])
    assert [(row["ngram"], row["count"]) for row in rows] == [("café policy", 1)]


@pytest.mark.parametrize("offset,limit", [(-1, 1), (0, 0), (0, 1_001)])
def test_mcp_ngram_pages_reject_invalid_bounds(tmp_path, monkeypatch, offset, limit):
    root = tmp_path / f"mcp-ngram-invalid-{offset}-{limit}"
    project = Project.init(root)
    _, revision = project.import_text("text.txt", "alpha beta")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ProjectError):
        corpus_ngrams(revision, 2, offset=offset, limit=limit)
