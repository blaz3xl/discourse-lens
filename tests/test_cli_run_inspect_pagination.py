from __future__ import annotations

import json
from typing import cast

import pytest
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def _create_run_with_executions(
    root,
) -> tuple[str, dict[str, object], dict[str, object]]:
    project = Project.init(root)
    _, first_revision = project.import_text("speech.txt", "Evidence remains local.")
    _, second_revision = project.import_text("notes.txt", "Methods remain explicit.")
    scope_id = project.snapshot([first_revision, second_revision])
    run_id = project.create_run(scope_id, {"profile": "fixture"})
    for model in ("fixture-a", "fixture-b", "fixture-c"):
        project.record_model_execution(
            "fixture",
            "research",
            model,
            None,
            "a" * 64,
            "b" * 64,
            analysis_run_id=run_id,
        )
    full_summary = project.analysis_run_summary(run_id)
    paged_summary = project.analysis_run_summary_page(run_id, offset=1, limit=1)
    project.close()
    return run_id, full_summary, paged_summary


def test_run_inspect_without_paging_preserves_full_summary_shape(tmp_path):
    root = tmp_path / "full-run-summary"
    run_id, full_summary, _ = _create_run_with_executions(root)

    result = CliRunner().invoke(app, ["run", "inspect", str(root), run_id])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == full_summary
    assert "pagination" not in payload
    assert len(payload["revision_ids"]) == 2
    assert len(payload["model_executions"]) == 3


def test_run_inspect_paging_bounds_revision_and_execution_rows(tmp_path):
    root = tmp_path / "paged-run-summary"
    run_id, _, expected_page = _create_run_with_executions(root)

    result = CliRunner().invoke(
        app,
        ["run", "inspect", str(root), run_id, "--offset", "1", "--limit", "1"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == expected_page
    assert len(payload["revision_ids"]) == 1
    assert len(payload["model_executions"]) == 1
    pagination = cast(dict[str, dict[str, object]], payload["pagination"])
    assert pagination["revision_ids"] == {
        "offset": 1,
        "limit": 1,
        "total_count": 2,
        "has_more": False,
    }
    assert pagination["model_executions"] == {
        "offset": 1,
        "limit": 1,
        "total_count": 3,
        "has_more": True,
    }
    for key in (
        "skill_runs",
        "model_executions",
        "retrieval_runs",
        "retrieval_hits",
        "reranking_runs",
        "reranking_hits",
    ):
        assert len(cast(list[object], payload[key])) <= 1


@pytest.mark.parametrize(
    ("option", "value"),
    [("--offset", "-1"), ("--limit", "0"), ("--limit", "101")],
)
def test_run_inspect_rejects_invalid_page_bounds(tmp_path, option, value):
    root = tmp_path / f"invalid-{option[2:]}-{value}"
    run_id, _, _ = _create_run_with_executions(root)

    result = CliRunner().invoke(
        app, ["run", "inspect", str(root), run_id, option, value]
    )

    assert result.exit_code == 2
    assert option in result.output


def test_run_inspect_keeps_unknown_run_error_clear(tmp_path):
    root = tmp_path / "unknown-run"
    Project.init(root).close()

    result = CliRunner().invoke(app, ["run", "inspect", str(root), "missing-analysis-run"])

    assert result.exit_code == 1
    assert "unknown analysis run" in result.output.lower()
