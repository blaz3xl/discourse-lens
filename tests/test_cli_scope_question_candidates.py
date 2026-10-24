from __future__ import annotations

import csv
import io
import json

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app
from discourselens.errors import ProjectError
from discourselens.types import CoverageDescriptor, CoverageMode, NegativeResultStatus


def _result(*, candidate_count: int = 1) -> dict[str, object]:
    evidence_ids = [f"evidence-{index}" for index in range(candidate_count)]
    return {
        "analysis_run_id": "analysis-run",
        "scope_snapshot_id": "scope-snapshot",
        "skill_run_id": "skill-run",
        "retrieval_run_id": "retrieval-run",
        "finding_id": "finding" if candidate_count else None,
        "evidence_ids": evidence_ids,
        "coverage_id": "coverage",
        "candidate_count": candidate_count,
        "coverage": CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id="scope-snapshot",
            population_count=4,
            selection_policy="researcher question lexical terms",
            inspection_count=4,
            termination_reason="retrieval_completed",
            negative_result_status=(
                NegativeResultStatus.NOT_APPLICABLE
                if candidate_count
                else NegativeResultStatus.NO_MATCHES_UNDER_PLAN
            ),
        ),
    }


class _ProjectStub:
    def __init__(self, result: object | None = None, error: Exception | None = None) -> None:
        self.result = result if result is not None else _result()
        self.error = error
        self.calls: list[tuple[list[str], str, str, int]] = []
        self.closed = False

    def run_scope_question_candidates(
        self, revision_ids: list[str], framework_id: str, question: str, *, limit: int = 12
    ) -> object:
        self.calls.append((revision_ids, framework_id, question, limit))
        if self.error is not None:
            raise self.error
        return self.result

    def close(self) -> None:
        self.closed = True


def _install_project_stub(monkeypatch: pytest.MonkeyPatch, stub: _ProjectStub) -> list[str]:
    opened: list[str] = []

    def open_project(path) -> _ProjectStub:
        opened.append(str(path))
        return stub

    monkeypatch.setattr(cli_module.Project, "open", staticmethod(open_project))
    return opened


def _arguments(*extra: str) -> list[str]:
    return [
        "skills",
        "scope-question-candidates",
        "research",
        "--revision",
        "revision-a",
        *extra,
    ]


def test_scope_question_candidates_json_calls_project_api_with_explicit_scope(monkeypatch):
    stub = _ProjectStub()
    _install_project_stub(monkeypatch, stub)

    result = CliRunner().invoke(
        app,
        _arguments(
            "--revision",
            "revision-b",
            "--framework",
            "fairclough-text",
            "--question",
            "How are workers represented?",
            "--format",
            "json",
        ),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["candidate_count"] == 1
    assert payload["coverage"]["coverage_mode"] == "TARGETED"
    assert stub.calls == [
        (
            ["revision-a", "revision-b"],
            "fairclough-text",
            "How are workers represented?",
            12,
        )
    ]
    assert stub.closed


@pytest.mark.parametrize("output_format", ["jsonl", "csv", "table"])
def test_scope_question_candidates_row_formats_emit_only_provenance_summary(
    monkeypatch, output_format
):
    stub = _ProjectStub(_result(candidate_count=2))
    _install_project_stub(monkeypatch, stub)

    result = CliRunner().invoke(
        app,
        _arguments(
            "--framework",
            "fairclough-text",
            "--question",
            "How are workers represented?",
            "--format",
            output_format,
        ),
    )

    assert result.exit_code == 0, result.output
    assert "scope_snapshot_id" in result.stdout
    assert "retrieval_run_id" in result.stdout
    assert "candidate_count" in result.stdout
    assert "TARGETED" in result.stdout
    assert "text" not in result.stdout
    assert "targeted examples, not an exhaustive answer" in result.stderr
    assert stub.closed

    if output_format == "jsonl":
        row = json.loads(result.stdout)
        assert row["candidate_count"] == 2
        assert "evidence_ids" not in row
    elif output_format == "csv":
        row = next(csv.DictReader(io.StringIO(result.stdout)))
        assert row["candidate_count"] == "2"
        assert row["negative_result_status"] == "NOT_APPLICABLE"
    else:
        assert "finding_id" in result.stdout


def test_scope_question_zero_results_disclose_targeted_limitations(monkeypatch):
    stub = _ProjectStub(_result(candidate_count=0))
    _install_project_stub(monkeypatch, stub)

    result = CliRunner().invoke(
        app,
        _arguments(
            "--framework",
            "fairclough-text",
            "--question",
            "Where are the examples?",
            "--format",
            "csv",
        ),
    )

    assert result.exit_code == 0, result.output
    assert "no examples were found under this targeted lexical plan" in result.stderr
    assert "zero results do not establish absence" in result.stderr
    row = next(csv.DictReader(io.StringIO(result.stdout)))
    assert row["candidate_count"] == "0"
    assert row["coverage_mode"] == "TARGETED"
    assert row["negative_result_status"] == "NO_MATCHES_UNDER_PLAN"


def test_scope_question_validates_before_opening_project(monkeypatch):
    stub = _ProjectStub()
    opened = _install_project_stub(monkeypatch, stub)
    runner = CliRunner()
    duplicate_revision = runner.invoke(
        app,
        _arguments(
            "--revision",
            "revision-a",
            "--framework",
            "fairclough-text",
            "--question",
            "Question?",
        ),
    )
    invalid_framework = runner.invoke(
        app,
        _arguments("--framework", "critical-reviewer", "--question", "Question?"),
    )
    empty_question = runner.invoke(
        app,
        _arguments("--framework", "fairclough-text", "--question", "   "),
    )
    excessive_limit = runner.invoke(
        app,
        [
            *_arguments("--framework", "fairclough-text", "--question", "Question?"),
            "--limit",
            "13",
        ],
    )

    assert duplicate_revision.exit_code == 2
    assert "must not be duplicated" in duplicate_revision.output
    assert invalid_framework.exit_code == 2
    assert "Invalid value for --framework" in invalid_framework.output
    assert empty_question.exit_code == 2
    assert "question must not be empty" in empty_question.output
    assert excessive_limit.exit_code == 2
    assert "12" in excessive_limit.output
    assert opened == []


def test_scope_question_closes_project_and_redacts_api_errors(monkeypatch, tmp_path):
    sensitive_path = tmp_path / "private-research-root"
    stub = _ProjectStub(error=ProjectError(f"scope operation failed for {sensitive_path}"))
    _install_project_stub(monkeypatch, stub)

    result = CliRunner().invoke(
        app,
        _arguments("--framework", "fairclough-text", "--question", "Question?"),
    )

    assert result.exit_code == 1
    assert "scope operation failed for <path>" in result.output
    assert str(sensitive_path) not in result.output
    assert "Traceback" not in result.output
    assert stub.closed


def test_scope_question_candidates_runs_against_real_project_and_keeps_scope(tmp_path):
    project_path = tmp_path / "real-project"
    project = Project.init(project_path)
    _, selected_revision = project.import_text(
        "selected.txt",
        "The procurement committee selected a contractor after reviewing the bids.\n\n"
        "The committee described the contractor selection as necessary.",
    )
    _, outside_revision = project.import_text(
        "outside.txt",
        "The procurement committee selected a contractor after reviewing the bids.",
    )
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "skills",
            "scope-question-candidates",
            str(project_path),
            "--revision",
            selected_revision,
            "--framework",
            "fairclough-text",
            "--question",
            "How did the committee select the contractor?",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["candidate_count"] >= 1
    assert payload["coverage"]["coverage_mode"] == "TARGETED"
    assert len(payload["evidence_ids"]) == payload["candidate_count"]

    reopened = Project.open(project_path)
    try:
        assert reopened.scope_revisions(payload["scope_snapshot_id"]) == [selected_revision]
        retrieval_summary = reopened.analysis_run_summary(payload["analysis_run_id"])
        retrieved_revision_ids = {
            row["revision_id"]
            for row in reopened.db.execute(
                "SELECT s.revision_id FROM retrieval_hits h "
                "JOIN source_spans s ON s.id=h.source_span_id "
                "WHERE h.retrieval_run_id=?",
                (payload["retrieval_run_id"],),
            ).fetchall()
        }
        assert retrieved_revision_ids == {selected_revision}
        assert outside_revision not in retrieved_revision_ids
        analysis_run = retrieval_summary["analysis_run"]
        assert isinstance(analysis_run, dict)
        assert analysis_run["id"] == payload["analysis_run_id"]
    finally:
        reopened.close()
