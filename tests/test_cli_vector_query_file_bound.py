from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app
from discourselens.vectors import MAX_PRESET_QUERY_CHARS, RetrievalSearchResult


def _project_with_revision(root):
    project = Project.init(root)
    try:
        _, revision_id = project.import_text("source.txt", "A local query fixture.")
        return revision_id
    finally:
        project.close()


def _invoke_search(project_root, revision_id, query_file, search_kind):
    command = "search" if search_kind == "semantic" else "hybrid-search-preset"
    return CliRunner().invoke(
        app,
        [
            "vectors",
            command,
            str(project_root),
            "fixture-preset",
            revision_id,
            "fixture-artifact",
            "--query-file",
            str(query_file),
        ],
    )


@pytest.mark.parametrize("search_kind", ["semantic", "hybrid"])
@pytest.mark.parametrize(
    ("input_kind", "expected_error", "marker"),
    [
        ("bytes", "byte input limit", "private-byte-marker"),
        ("characters", "character limit", "private-character-marker"),
        ("utf8", "must be valid UTF-8", "private-invalid-marker"),
    ],
)
def test_cli_rejects_oversized_or_malformed_preset_query_before_core(
    tmp_path, monkeypatch, search_kind, input_kind, expected_error, marker
):
    project_root = tmp_path / "project"
    revision_id = _project_with_revision(project_root)
    query_file = tmp_path / "query.txt"
    max_bytes = MAX_PRESET_QUERY_CHARS * 4
    if input_kind == "bytes":
        prefix = marker.encode()
        query_file.write_bytes(prefix + b"x" * (max_bytes + 1 - len(prefix)))
    elif input_kind == "characters":
        prefix = marker
        query_file.write_text(
            prefix + "x" * (MAX_PRESET_QUERY_CHARS + 1 - len(prefix)), encoding="utf-8"
        )
    else:
        query_file.write_bytes(marker.encode() + b"\xff")

    core_calls: list[object] = []

    def unexpected_core_call(*args, **kwargs):
        core_calls.append((args, kwargs))
        raise AssertionError("invalid query files must be rejected before Core")

    monkeypatch.setattr(cli_module, "semantic_search_with_preset", unexpected_core_call)
    monkeypatch.setattr(cli_module, "hybrid_search_with_preset", unexpected_core_call)

    result = _invoke_search(project_root, revision_id, query_file, search_kind)

    assert result.exit_code != 0
    assert expected_error in result.output
    assert marker not in result.output
    assert core_calls == []


@pytest.mark.parametrize("search_kind", ["semantic", "hybrid"])
def test_cli_accepts_query_file_at_exact_unicode_character_limit(
    tmp_path, monkeypatch, search_kind
):
    project_root = tmp_path / "project"
    revision_id = _project_with_revision(project_root)
    query_text = "🧪" * MAX_PRESET_QUERY_CHARS
    query_file = tmp_path / "query.txt"
    query_file.write_text(query_text, encoding="utf-8")
    assert query_file.stat().st_size == MAX_PRESET_QUERY_CHARS * 4
    captured_queries: list[str] = []
    captured_scopes: list[str] = []

    def capture_core_query(project, preset, scope_id, artifact_ids, query, **kwargs):
        del project, preset, artifact_ids, kwargs
        captured_queries.append(query)
        captured_scopes.append(scope_id)
        return RetrievalSearchResult(scope_id, "fixture-run-id", [])

    monkeypatch.setattr(cli_module, "semantic_search_with_preset", capture_core_query)
    monkeypatch.setattr(cli_module, "hybrid_search_with_preset", capture_core_query)

    result = _invoke_search(project_root, revision_id, query_file, search_kind)

    assert result.exit_code == 0, result.output
    assert captured_queries == [query_text]
    payload = json.loads(result.output)
    assert payload == {
        "scope_id": captured_scopes[0],
        "retrieval_run_id": "fixture-run-id",
        "hits": [],
    }
    with Project.open(project_root, read_only=True) as reopened:
        assert reopened.scope_revisions(payload["scope_id"]) == [revision_id]
