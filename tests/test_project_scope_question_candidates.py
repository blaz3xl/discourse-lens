from collections.abc import Mapping

import pytest

from discourselens import Project
from discourselens.errors import ProjectError, ScopeViolationError
from discourselens.types import CoverageMode, NegativeResultStatus


def test_project_scope_question_candidates_preserve_scope_and_lineage(tmp_path):
    project = Project.init(tmp_path / "project")
    _, selected = project.import_text(
        "selected.txt", "Refugees are described as a burden in the public debate."
    )
    _, excluded = project.import_text(
        "excluded.txt", "Refugees are described as a burden in the public debate."
    )

    result = project.run_scope_question_candidates(
        [selected],
        "fairclough-text",
        "How are refugees represented?",
    )

    assert result["candidate_count"] == 1
    assert len(result["evidence_ids"]) == 1
    assert result["finding_id"] is not None
    assert result["coverage"].coverage_mode is CoverageMode.TARGETED
    assert result["coverage"].scope_snapshot_id == result["scope_snapshot_id"]
    assert project.scope_revisions(result["scope_snapshot_id"]) == [selected]

    metadata = project.analysis_run_metadata(result["analysis_run_id"])
    run = metadata["analysis_run"]
    assert isinstance(run, Mapping)
    assert metadata["revision_ids"] == [selected]
    assert run["scope_id"] == result["scope_snapshot_id"]
    assert run["status"] == "COMPLETED"

    skill_run = project.db.execute(
        "SELECT analysis_run_id,scope_id,status FROM skill_runs WHERE id=?",
        (result["skill_run_id"],),
    ).fetchone()
    assert skill_run["analysis_run_id"] == result["analysis_run_id"]
    assert skill_run["scope_id"] == result["scope_snapshot_id"]
    assert skill_run["status"] == "COMPLETED"

    retrieval = project.db.execute(
        "SELECT analysis_run_id,skill_run_id,scope_id,coverage_id,returned_count "
        "FROM retrieval_runs WHERE id=?",
        (result["retrieval_run_id"],),
    ).fetchone()
    assert retrieval["analysis_run_id"] == result["analysis_run_id"]
    assert retrieval["skill_run_id"] == result["skill_run_id"]
    assert retrieval["scope_id"] == result["scope_snapshot_id"]
    assert retrieval["coverage_id"] == result["coverage_id"]
    assert retrieval["returned_count"] == result["candidate_count"]
    candidate_revision = project.db.execute(
        "SELECT s.revision_id FROM retrieval_hits h "
        "JOIN source_spans s ON s.id=h.source_span_id WHERE h.retrieval_run_id=?",
        (result["retrieval_run_id"],),
    ).fetchone()["revision_id"]
    assert candidate_revision == selected
    assert candidate_revision != excluded
    assert project.get_finding(result["finding_id"])["analysis_run_id"] == result[
        "analysis_run_id"
    ]


def test_project_scope_question_candidates_empty_result_is_targeted_not_absence(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("report.txt", "The committee discussed transit funding.")

    result = project.run_scope_question_candidates(
        [revision],
        "fairclough-text",
        "Where is the xenolexicon moon observatory?",
    )

    assert result["candidate_count"] == 0
    assert result["finding_id"] is None
    assert result["evidence_ids"] == []
    assert result["coverage"].coverage_mode is CoverageMode.TARGETED
    assert result["coverage"].negative_result_status is NegativeResultStatus.NO_MATCHES_UNDER_PLAN
    assert result["coverage"].negative_result_status is not NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION
    run = project.db.execute(
        "SELECT status,scope_id FROM analysis_runs WHERE id=?", (result["analysis_run_id"],)
    ).fetchone()
    assert run["status"] == "COMPLETED"
    assert run["scope_id"] == result["scope_snapshot_id"]


@pytest.mark.parametrize(
    "scope,framework_id,question,limit",
    [
        ([], "fairclough-text", "How is this represented?", 12),
        (["  "], "fairclough-text", "How is this represented?", 12),
        (["revision-a", "revision-a"], "fairclough-text", "How is this represented?", 12),
        (["revision-a"], "unknown-framework", "How is this represented?", 12),
        (["revision-a"], "critical-reviewer", "How is this represented?", 12),
        (["revision-a"], "fairclough-text", " \t ", 12),
        (["revision-a"], "fairclough-text", "How is this represented?", 0),
        (["revision-a"], "fairclough-text", "How is this represented?", 13),
        (["revision-a"], "fairclough-text", "How is this represented?", True),
    ],
)
def test_project_scope_question_candidates_reject_invalid_inputs_before_run(
    tmp_path, scope, framework_id, question, limit
):
    project = Project.init(tmp_path / "project")
    project.import_text("report.txt", "A report about the public debate.")

    with pytest.raises(ProjectError):
        project.run_scope_question_candidates(
            scope,
            framework_id,
            question,
            limit=limit,
        )

    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == 0


def test_project_scope_question_candidates_reject_unknown_revision_before_run(tmp_path):
    project = Project.init(tmp_path / "project")

    with pytest.raises(ScopeViolationError):
        project.run_scope_question_candidates(
            ["unknown-revision"],
            "fairclough-text",
            "How is this represented?",
        )

    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == 0
