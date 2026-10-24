from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.mcp_server import corpus_frequency_for_scope, corpus_kwic_for_scope


def test_mcp_scope_corpus_pages_match_sdk_and_read_only_scope_membership(
    tmp_path, monkeypatch
):
    root = tmp_path / "scope-corpus-readers"
    project = Project.init(root)
    _, selected_revision = project.import_text(
        "selected.txt", "Policy policy civic policy action."
    )
    scope_id = project.snapshot([selected_revision])
    _, outside_revision = project.import_text("later.txt", "policy lateronly.")
    project.close()

    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    with Project.open(root, read_only=True) as core:
        frequency_page_one = core.corpus_frequency_page(scope_id, offset=0, limit=2)
        frequency_page_two = core.corpus_frequency_page(scope_id, offset=2, limit=2)
        frequency_full = core.corpus_frequency_page(scope_id, offset=0, limit=100)
        kwic_page_one = core.corpus_kwic_page(
            scope_id, "policy", window=1, offset=0, limit=2
        )
        kwic_page_two = core.corpus_kwic_page(
            scope_id, "policy", window=1, offset=2, limit=2
        )
    assert database.read_bytes() == before

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)

    mcp_frequency_one = corpus_frequency_for_scope(scope_id, offset=0, limit=2)
    mcp_frequency_two = corpus_frequency_for_scope(scope_id, offset=2, limit=2)
    mcp_kwic_one = corpus_kwic_for_scope(
        scope_id, "policy", window=1, offset=0, limit=2
    )
    mcp_kwic_two = corpus_kwic_for_scope(
        scope_id, "policy", window=1, offset=2, limit=2
    )

    assert mcp_frequency_one == frequency_page_one
    assert mcp_frequency_two == frequency_page_two
    assert mcp_frequency_one["scope_id"] == scope_id
    assert mcp_frequency_one["has_more"] is True
    assert mcp_frequency_two["has_more"] is False
    frequency_rows = cast(list[dict[str, object]], frequency_full["rows"])
    mcp_frequency_rows = cast(list[dict[str, object]], mcp_frequency_one["rows"])
    mcp_frequency_rows += cast(list[dict[str, object]], mcp_frequency_two["rows"])
    assert mcp_frequency_rows == frequency_rows
    assert "lateronly" not in {str(row["token"]) for row in mcp_frequency_rows}

    def without_mcp_truncation_fields(page: Mapping[str, object]) -> dict[str, object]:
        result = {key: value for key, value in page.items() if key != "text_truncated"}
        rows = cast(list[dict[str, object]], result["rows"])
        result["rows"] = [
            {key: value for key, value in row.items() if key != "text_truncated"}
            for row in rows
        ]
        return result

    assert without_mcp_truncation_fields(mcp_kwic_one) == kwic_page_one
    assert without_mcp_truncation_fields(mcp_kwic_two) == kwic_page_two
    assert mcp_kwic_one["scope_id"] == scope_id
    assert mcp_kwic_one["total_count"] == 3
    assert mcp_kwic_one["has_more"] is True
    assert mcp_kwic_two["has_more"] is False
    assert mcp_kwic_one["text_truncated"] is False
    kwic_rows = cast(list[dict[str, object]], mcp_kwic_one["rows"])
    kwic_rows += cast(list[dict[str, object]], mcp_kwic_two["rows"])
    assert outside_revision not in {str(row["revision_id"]) for row in kwic_rows}
    assert database.read_bytes() == before


@pytest.mark.parametrize(
    ("scope_id", "offset", "limit"),
    [
        ("", 0, 100),
        (" padded ", 0, 100),
        ("s" * 129, 0, 100),
        ("scope", True, 100),
        ("scope", -1, 100),
        ("scope", 0x8000_0000_0000_0000, 100),
        ("scope", 0, True),
        ("scope", 0, 0),
        ("scope", 0, 1_001),
    ],
)
def test_mcp_scope_frequency_rejects_invalid_input_before_open(
    monkeypatch, scope_id, offset, limit
):
    def reject_open():
        pytest.fail("invalid scope frequency input opened the project")

    monkeypatch.setattr(mcp_module, "_open_project", reject_open)
    with pytest.raises(ProjectError):
        corpus_frequency_for_scope(scope_id, offset=offset, limit=limit)


@pytest.mark.parametrize(
    ("scope_id", "query", "window", "offset", "limit"),
    [
        ("", "policy", 5, 0, 100),
        (" padded ", "policy", 5, 0, 100),
        ("scope", "", 5, 0, 100),
        ("scope", "   ", 5, 0, 100),
        ("scope", "two words", 5, 0, 100),
        ("scope", "x" * (mcp_module.MCP_KWIC_MAX_QUERY_CHARS + 1), 5, 0, 100),
        ("scope", "policy", True, 0, 100),
        ("scope", "policy", -1, 0, 100),
        ("scope", "policy", mcp_module.MCP_SCOPE_CORPUS_MAX_WINDOW + 1, 0, 100),
        ("scope", "policy", 5, True, 100),
        ("scope", "policy", 5, -1, 100),
        ("scope", "policy", 5, 0x8000_0000_0000_0000, 100),
        ("scope", "policy", 5, 0, True),
        ("scope", "policy", 5, 0, 0),
        ("scope", "policy", 5, 0, 1_001),
    ],
)
def test_mcp_scope_kwic_rejects_invalid_input_before_open(
    monkeypatch, scope_id, query, window, offset, limit
):
    def reject_open():
        pytest.fail("invalid scope KWIC input opened the project")

    monkeypatch.setattr(mcp_module, "_open_project", reject_open)
    with pytest.raises(ProjectError):
        corpus_kwic_for_scope(
            scope_id,
            query,
            window=window,
            offset=offset,
            limit=limit,
        )
